"""Windows-aware environment discovery and fail-closed ML doctor probes.

The control plane must remain importable on a small CPU-only installation, but
Phase 0 must never claim a usable ML environment from metadata alone.  The
helpers in this module therefore separate cheap discovery (``collect_environment``)
from the required executable checks (``run_environment_doctor``).  Missing
optional packages and unavailable CUDA are represented as named probe failures,
not import-time crashes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .command import run_guarded

ENVIRONMENT_DOCTOR_SCHEMA_VERSION = 2
RUNTIME_LOCK_SCHEMA_VERSION = 1
RUNTIME_LOCK_FILENAME = "windows-runtime-lock.json"
WINDOWS_RUNTIME_DRIFT = "WINDOWS_RUNTIME_DRIFT"

# This is the last known successful native-Windows run.  It is deliberately
# treated as a recovery candidate, not as proof that the current interpreter
# has the same environment.  ``load_recovery_pin`` also reads the adjacent
# receipt when available so the observed GPU names and versions remain
# content-addressed by the checked-in file.
DEFAULT_RECOVERY_PIN = Path(__file__).resolve().parents[2] / "runs" / "windows-cuda-v3.json"
DEFAULT_RUNTIME_LOCK = Path(__file__).resolve().parents[2] / "runs" / RUNTIME_LOCK_FILENAME
KNOWN_WINDOWS_RECOVERY = {
    "platform": "Windows-11-10.0.26200-SP0",
    "python_version": "3.12.6",
    "torch_version": "2.13.0+cu130",
    "cuda_runtime": "13.0",
    "cuda_index_url": "https://download.pytorch.org/whl/cu130",
    "expected_gpu_count": 2,
    "expected_gpu_names": ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 5060 Ti"],
}

REQUIRED_ENVIRONMENT_PROBES = (
    "native_windows",
    "python",
    "torch_import",
    "torch_version",
    "cuda_runtime",
    "cuda_available",
    "expected_gpu",
    "bf16_tensor",
    "small_cuda_gemm",
    "safetensors_load",
    "d2m_checkpoint_load",
    "p16_forward",
    "dense_teacher_ffn_forward",
    "oracle_module_import",
    "source_checkpoint_readable",
)


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / name,
            Path(r"C:\Windows\System32") / f"{name}.exe",
            Path(r"C:\Program Files\NVIDIA Corporation\NVSMI") / f"{name}.exe",
            Path(r"C:\Program Files\Git\cmd") / f"{name}.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


def run_probe(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = run_guarded(
            command,
            name=f"probe-{Path(str(command[0])).name}",
            category="FAST",
            timeout=30.0,
            emit=lambda _line: None,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "status": completed.status,
            "elapsed_seconds": completed.elapsed_seconds,
            "unavailable": completed.status != "DONE",
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {"command": list(command), "returncode": None, "stdout": "", "stderr": str(exc), "unavailable": True}


def _probe_tool(name: str, args: Sequence[str]) -> dict[str, Any]:
    executable = _which(name)
    if not executable:
        return {"command": [name, *args], "unavailable": True, "returncode": None, "stdout": "", "stderr": "not found"}
    return run_probe([executable, *args])


def detect_wsl() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except (FileNotFoundError, OSError):
        return False


def _python_version_from_text(value: str) -> str | None:
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", value)
    return match.group(1) if match else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _project_python(repo_root: Path) -> Path:
    """Return the only interpreter allowed to run D2M on Windows."""

    return repo_root / ".venv" / "Scripts" / "python.exe"


def _normalise_path(value: str | os.PathLike[str] | None) -> str:
    if value is None:
        return ""
    raw = str(value).replace("/", os.sep).replace("\\", os.sep)
    return os.path.normcase(os.path.abspath(os.path.normpath(raw)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _driver_versions(environment: dict[str, Any]) -> list[str]:
    query = environment.get("tools", {}).get("nvidia_smi_query", {})
    stdout = str(query.get("stdout", "")) if isinstance(query, dict) else ""
    versions: set[str] = set()
    for line in stdout.splitlines()[1:]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4 and fields[3]:
            versions.add(fields[3])
    return sorted(versions)


def _selected_device_indices(environment: dict[str, Any], devices: list[dict[str, Any]]) -> list[int]:
    raw = environment.get("selected_training_gpus")
    if raw is None:
        raw = os.environ.get("D2M_SELECTED_GPUS", "")
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        selected: list[int] = []
        for value in raw:
            try:
                selected.append(int(value))
            except (TypeError, ValueError):
                continue
        if selected:
            return sorted(set(selected))
    return sorted(int(item.get("index", index)) for index, item in enumerate(devices) if isinstance(item, dict))


def _runtime_fingerprint(environment: dict[str, Any]) -> dict[str, Any]:
    platform_info = environment.get("platform", {}) if isinstance(environment.get("platform"), dict) else {}
    torch_info = environment.get("torch", {}) if isinstance(environment.get("torch"), dict) else {}
    devices = torch_info.get("devices", []) if isinstance(torch_info.get("devices"), list) else []
    device_inventory = [
        {
            "index": int(item.get("index", index)),
            "name": str(item.get("name", "")),
            "compute_capability": list(item.get("compute_capability", [])),
            "total_memory": int(item.get("total_memory", 0)),
            "bf16_supported": bool(item.get("bf16_supported", item.get("bf16_tensor_operation", False))),
        }
        for index, item in enumerate(devices)
        if isinstance(item, dict)
    ]
    device_inventory.sort(key=lambda item: item["index"])
    python_executable = str(environment.get("python_executable") or platform_info.get("python_executable") or sys.executable)
    python_version = _python_version_from_text(str(environment.get("python_version") or platform_info.get("python", "")))
    selected = _selected_device_indices(environment, device_inventory)
    packages = environment.get("packages", {}) if isinstance(environment.get("packages"), dict) else {}
    package_versions = {
        str(name): str(value.get("version")) if isinstance(value, dict) and value.get("version") is not None else None
        for name, value in packages.items()
    }
    package_versions["torch"] = str(torch_info.get("version")) if torch_info.get("version") else None
    package_versions["torch_compiled_cuda"] = str(torch_info.get("compiled_cuda")) if torch_info.get("compiled_cuda") else None
    return {
        "platform": str(platform_info.get("system", "")),
        "windows_release": str(platform_info.get("release", "")),
        "windows_version": str(platform_info.get("version", "")),
        "python_executable": _normalise_path(python_executable),
        "python_version": python_version,
        "torch_version": str(torch_info.get("version", "")),
        "compiled_cuda": str(torch_info.get("compiled_cuda", "")),
        "driver_versions": _driver_versions(environment),
        "gpu_inventory": device_inventory,
        "selected_training_gpus": selected,
        "bf16_capability": bool(torch_info.get("bf16_tensor")),
        "package_versions": package_versions,
    }


def load_runtime_lock(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the current approved lock, if one exists."""

    target = Path(path) if path is not None else DEFAULT_RUNTIME_LOCK
    if not target.exists():
        return {"status": "MISSING", "ok": True, "path": str(target)}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("runtime lock must be a JSON object")
        if int(payload.get("schema_version", 0)) != RUNTIME_LOCK_SCHEMA_VERSION:
            raise ValueError("runtime lock schema version is unsupported")
        recorded = str(payload.get("lock_sha256", ""))
        unsigned = dict(payload)
        unsigned.pop("lock_sha256", None)
        if not recorded or recorded != _canonical_sha256(unsigned):
            raise ValueError("runtime lock SHA256 does not match its content")
        return {"status": "LOCKED", "ok": True, "path": str(target), "payload": payload}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "INVALID", "ok": False, "path": str(target), "error": str(exc)}


def check_runtime_lock(environment: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """Compare current runtime facts to the approved lock, not recovery history."""

    loaded = load_runtime_lock(path)
    if loaded.get("status") == "MISSING":
        return loaded | {"drift": []}
    if not loaded.get("ok"):
        return loaded | {"status": WINDOWS_RUNTIME_DRIFT, "drift": [str(loaded.get("error", "invalid lock"))]}
    expected = loaded["payload"].get("runtime_fingerprint", {})
    if not isinstance(expected, dict):
        return loaded | {
            "status": WINDOWS_RUNTIME_DRIFT,
            "ok": False,
            "drift": ["runtime_fingerprint"],
            "expected_fingerprint": expected,
            "actual_fingerprint": _runtime_fingerprint(environment),
        }
    actual = _runtime_fingerprint(environment)
    drift = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
    return loaded | {
        "status": "LOCKED" if not drift else WINDOWS_RUNTIME_DRIFT,
        "ok": not drift,
        "drift": drift,
        "expected_fingerprint": expected,
        "actual_fingerprint": actual,
    }


def write_runtime_lock(
    environment: dict[str, Any],
    doctor: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    path: str | Path | None = None,
    environment_receipts: Sequence[str | Path] = (),
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Approve the first green capability result as the current runtime lock."""

    if not doctor.get("ok"):
        raise ValueError("cannot approve a runtime lock while the capability gate is blocked")
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    target = Path(path) if path is not None else root / "runs" / RUNTIME_LOCK_FILENAME
    receipt_hashes = {
        str(Path(receipt)): _sha256_file(Path(receipt))
        for receipt in environment_receipts
        if Path(receipt).is_file()
    }
    embedded_path = environment.get("environment_receipt_path")
    if embedded_path and Path(str(embedded_path)).is_file():
        receipt_hashes.setdefault(str(Path(str(embedded_path))), _sha256_file(Path(str(embedded_path))))
    if not receipt_hashes:
        embedded = environment.get("environment_receipt_sha256")
        if embedded:
            receipt_hashes["embedded"] = str(embedded)
    try:
        from .provenance import current_git_commit

        resolved_commit = code_commit or current_git_commit()
    except (RuntimeError, OSError):
        resolved_commit = code_commit or "unknown"
    fingerprint = _runtime_fingerprint(environment)
    platform_info = environment.get("platform", {}) if isinstance(environment.get("platform"), dict) else {}
    payload: dict[str, Any] = {
        "schema_version": RUNTIME_LOCK_SCHEMA_VERSION,
        "status": "APPROVED",
        "approved_at": _utc_now(),
        "platform": fingerprint["platform"],
        "windows_version": fingerprint["windows_version"],
        "python_executable": str(environment.get("python_executable") or platform_info.get("python_executable") or sys.executable),
        "python_version": fingerprint["python_version"],
        "torch_version": fingerprint["torch_version"],
        "compiled_cuda": fingerprint["compiled_cuda"],
        "driver": fingerprint["driver_versions"],
        "gpu_inventory": fingerprint["gpu_inventory"],
        "selected_training_gpus": fingerprint["selected_training_gpus"],
        "bf16_capability": fingerprint["bf16_capability"],
        "package_versions": fingerprint["package_versions"],
        "environment_receipt_hashes": receipt_hashes,
        "doctor_schema_version": doctor.get("schema_version", ENVIRONMENT_DOCTOR_SCHEMA_VERSION),
        "code_commit": resolved_commit,
        "runtime_fingerprint": fingerprint,
    }
    payload["lock_sha256"] = _canonical_sha256(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f"{target.stem}-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload | {"path": str(target)}


def load_recovery_pin(path: str | Path | None = None) -> dict[str, Any]:
    """Return the last known-good Windows runtime as a recovery candidate."""

    target = Path(path) if path is not None else DEFAULT_RECOVERY_PIN
    payload: dict[str, Any] = {
        **KNOWN_WINDOWS_RECOVERY,
        "reference": str(target),
        "reference_exists": target.exists(),
        "status": "fallback",
    }
    if not target.exists():
        payload["message"] = "historical native-Windows recovery receipt is unavailable"
        return payload
    try:
        observed = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(observed, dict):
            raise TypeError("recovery receipt must be a JSON object")
        payload["status"] = "available"
        payload["observed"] = observed
        payload["platform"] = observed.get("platform", payload["platform"])
        payload["python_version"] = _python_version_from_text(str(observed.get("python", ""))) or payload["python_version"]
        payload["torch_version"] = observed.get("torch", payload["torch_version"])
        payload["cuda_runtime"] = observed.get("torch_cuda", payload["cuda_runtime"])
        devices = observed.get("devices", [])
        if isinstance(devices, list):
            names = sorted(str(item.get("name")) for item in devices if isinstance(item, dict) and item.get("name"))
            if names:
                payload["expected_gpu_names"] = names
            payload["expected_gpu_count"] = int(observed.get("device_count", len(devices)))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        payload["status"] = "invalid"
        payload["error"] = str(exc)
    return payload


def _optional_package_versions() -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    for name in ("numpy", "psutil", "safetensors", "transformers", "accelerate"):
        try:
            packages[name] = {"installed": True, "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"installed": False, "version": None}
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
            packages[name] = {"installed": False, "version": None, "error": str(exc)}
    return packages


def probe_torch() -> dict[str, Any]:
    """Probe torch and CUDA while remaining safe on CPU-only installs."""

    result: dict[str, Any] = {
        "installed": False,
        "import_ok": False,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cuda_available": False,
        "devices": [],
    }
    try:
        import torch  # type: ignore
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
        result.update({"reason": "PyTorch is not installed or cannot be imported", "error": str(exc)})
        return result
    try:
        result.update({"installed": True, "import_ok": True, "version": str(torch.__version__), "compiled_cuda": getattr(torch.version, "cuda", None)})
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            result["reason"] = "PyTorch has no CUDA namespace"
            return result
        try:
            result["cuda_available"] = bool(cuda.is_available())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            result["cuda_error"] = str(exc)
            return result
        if not result["cuda_available"]:
            result["reason"] = "torch.cuda.is_available() is false"
            return result
        try:
            device_count = int(cuda.device_count())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            result["cuda_error"] = str(exc)
            return result
        devices: list[dict[str, Any]] = []
        for index in range(device_count):
            try:
                with cuda.device(index):
                    props = cuda.get_device_properties(index)
                    try:
                        bf16_supported = bool(cuda.is_bf16_supported())
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        bf16_supported = False
                    item: dict[str, Any] = {
                        "index": index,
                        "name": str(getattr(props, "name", "unknown")),
                        "compute_capability": [int(getattr(props, "major", 0)), int(getattr(props, "minor", 0))],
                        "total_memory": int(getattr(props, "total_memory", 0)),
                        "bf16_supported": bf16_supported,
                        "allocated": int(cuda.memory_allocated(index)),
                        "reserved": int(cuda.memory_reserved(index)),
                    }
                    for dtype_name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
                        try:
                            left = torch.ones((16, 16), device=f"cuda:{index}", dtype=dtype)
                            output = left @ left
                            cuda.synchronize(index)
                            finite = bool(torch.isfinite(output).all().item())
                            item[f"{dtype_name}_matmul"] = finite
                            item[f"{dtype_name}_tensor"] = finite
                        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                            item[f"{dtype_name}_matmul"] = False
                            item[f"{dtype_name}_tensor"] = False
                            item[f"{dtype_name}_error"] = str(exc)
                    item["small_cuda_gemm"] = bool(item.get("float16_matmul"))
                    item["bf16_tensor_operation"] = bool(item.get("bfloat16_matmul"))
                    devices.append(item)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                devices.append({"index": index, "status": "ERROR", "error": str(exc)})
        result["devices"] = devices
        result["device_count"] = len(devices)
        result["small_cuda_gemm"] = bool(devices) and all(item.get("small_cuda_gemm", False) for item in devices)
        result["bf16_tensor"] = bool(devices) and all(item.get("bf16_tensor_operation", False) for item in devices)
        try:
            result["peer_access"] = [[bool(cuda.can_device_access_peer(i, j)) for j in range(len(devices))] for i in range(len(devices))]
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            result["peer_access_error"] = str(exc)
    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        result["error"] = str(exc)
    return result


def _probe_status(ok: bool, *, detail: str | None = None, **values: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "PASS" if ok else "BLOCKED", "ok": bool(ok)}
    if detail:
        payload["detail"] = detail
    payload.update(values)
    return payload


def _default_checkpoint_path(repo_root: Path) -> Path | None:
    candidate = repo_root / "runs" / "20260815-162258-windows-real-d2m-v2" / "artifacts" / "hf-moe" / "full-model-spike" / "model.safetensors"
    return candidate if candidate.exists() else None


def _default_source_checkpoint_path(repo_root: Path) -> Path | None:
    candidate = repo_root / "runs" / "20260815-030931-windows" / "source" / "model-00001-of-00018.safetensors"
    return candidate if candidate.exists() else None


def _probe_safetensors(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _probe_status(False, detail="no existing D2M checkpoint path was found")
    if not path.exists():
        return _probe_status(False, detail="checkpoint path does not exist", path=str(path))
    try:
        from safetensors import safe_open  # type: ignore
        with safe_open(str(path), framework="np") as handle:
            keys = sorted(handle.keys())
            if not keys:
                return _probe_status(False, detail="checkpoint contains no tensors", path=str(path))
            sample = handle.get_tensor(keys[0])
            shape = list(getattr(sample, "shape", ()))
        return _probe_status(True, path=str(path), tensor_count=len(keys), sample_tensor=keys[0], sample_shape=shape)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        return _probe_status(False, detail="safetensors checkpoint load failed", path=str(path), error=str(exc))


def _probe_d2m_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _probe_status(False, detail="no existing D2M checkpoint path was found")
    if path.suffix.lower() == ".json":
        try:
            from .checkpoint.layer import load_layer_checkpoint
            checkpoint = load_layer_checkpoint(path)
            tensor_path = Path(checkpoint.tensor_file) if checkpoint.tensor_file else None
            if tensor_path is not None and not tensor_path.is_absolute():
                tensor_path = path.parent / tensor_path
            result = _probe_safetensors(tensor_path)
            result.update({"metadata_path": str(path), "profile": checkpoint.profile, "layer": checkpoint.layer})
            return result
        except (ImportError, ModuleNotFoundError, OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return _probe_status(False, detail="D2M layer checkpoint load failed", path=str(path), error=str(exc))
    result = _probe_safetensors(path)
    result["checkpoint_type"] = "safetensors"
    return result


def _probe_torch_models(torch: Any, *, device: str) -> dict[str, dict[str, Any]]:
    if not bool(torch.cuda.is_available()):
        detail = "CUDA is unavailable; CUDA model forwards were not attempted"
        return {name: _probe_status(False, detail=detail, device=device) for name in ("p16_forward", "dense_teacher_ffn_forward")}
    try:
        from .models.torch_moe import TorchQwen35SwiGLUMoE
        model = TorchQwen35SwiGLUMoE(hidden_size=8, intermediate_size=40, routed_experts=16, expert_intermediate_size=2, shared_intermediate_size=8, top_k=4, dtype=torch.float32, device=device).eval()
        values = torch.randn((2, 8), dtype=torch.float32, device=device)
        with torch.inference_mode():
            output = model(values)
        finite = bool(torch.isfinite(output).all().item())
        p16 = _probe_status(finite, detail=None if finite else "p16 output is non-finite", device=device, shape=list(output.shape), topology="p16/top4")
    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        p16 = _probe_status(False, detail="p16 forward failed", device=device, error=str(exc))
    try:
        import torch.nn.functional as F
        values = torch.randn((2, 8), dtype=torch.float32, device=device)
        gate = torch.nn.Linear(8, 16, bias=False, device=device)
        up = torch.nn.Linear(8, 16, bias=False, device=device)
        down = torch.nn.Linear(16, 8, bias=False, device=device)
        with torch.inference_mode():
            output = down(F.silu(gate(values)) * up(values))
        finite = bool(torch.isfinite(output).all().item())
        dense = _probe_status(finite, detail=None if finite else "dense teacher FFN output is non-finite", device=device, shape=list(output.shape))
    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        dense = _probe_status(False, detail="dense teacher FFN forward failed", device=device, error=str(exc))
    return {"p16_forward": p16, "dense_teacher_ffn_forward": dense}


def run_environment_doctor(
    *,
    environment: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    source_checkpoint_path: str | Path | None = None,
    expected_gpu_names: Sequence[str] | None = None,
    expected_gpu_count: int | None = None,
    recovery_pin_path: str | Path | None = None,
    runtime_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the capability gate and compare only with the current runtime lock.

    ``runs/windows-cuda-v3.json`` is intentionally returned as advisory
    recovery metadata. It never creates a blocker merely because a current
    capable Windows runtime uses a different patch version, Torch/CUDA build,
    or GPU inventory.
    """

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    observed = environment if environment is not None else collect_environment(repo_root=root)
    pin = load_recovery_pin(recovery_pin_path)
    torch_info = observed.get("torch", {}) if isinstance(observed.get("torch"), dict) else {}
    platform_info = observed.get("platform", {}) if isinstance(observed.get("platform"), dict) else {}
    devices = torch_info.get("devices", []) if isinstance(torch_info.get("devices"), list) else []
    device_dicts = [item for item in devices if isinstance(item, dict)]
    selected_indices = _selected_device_indices(observed, device_dicts)
    selected_devices = [item for item in device_dicts if int(item.get("index", -1)) in selected_indices]
    names_actual = sorted(str(item.get("name")) for item in selected_devices if item.get("name"))
    names_expected = sorted(str(item) for item in expected_gpu_names) if expected_gpu_names is not None else names_actual
    count_expected = int(expected_gpu_count) if expected_gpu_count is not None else len(selected_indices)
    requirements: dict[str, dict[str, Any]] = {}
    native = platform_info.get("system") == "Windows" and not bool(observed.get("wsl"))
    requirements["native_windows"] = _probe_status(
        native,
        detail=None if native else "PLATFORM_POLICY_VIOLATION: doctor must run in native Windows, not WSL/Linux",
        system=platform_info.get("system"),
        wsl=bool(observed.get("wsl")),
    )
    python_executable = str(observed.get("python_executable") or platform_info.get("python_executable") or sys.executable)
    python_version = _python_version_from_text(str(observed.get("python_version") or platform_info.get("python", "")))
    expected_python_path = _project_python(root)
    python_match = bool(python_version) and _normalise_path(python_executable) == _normalise_path(expected_python_path)
    requirements["python"] = _probe_status(
        python_match,
        detail=None if python_match else f"interpreter must be the project Windows venv: {expected_python_path}",
        executable=python_executable,
        version=python_version,
        expected_executable=str(expected_python_path),
    )
    imported = bool(torch_info.get("installed")) and bool(torch_info.get("import_ok"))
    requirements["torch_import"] = _probe_status(
        imported,
        detail=None if imported else str(torch_info.get("reason", "PyTorch import failed")),
        version=torch_info.get("version"),
    )
    actual_torch = str(torch_info.get("version", ""))
    requirements["torch_version"] = _probe_status(
        imported and bool(actual_torch),
        detail=None if imported and actual_torch else "PyTorch version is unavailable after import",
        observed=actual_torch or None,
        recovery_candidate=pin.get("torch_version"),
    )
    actual_cuda = str(torch_info.get("compiled_cuda", ""))
    requirements["cuda_runtime"] = _probe_status(
        imported and bool(actual_cuda),
        detail=None if imported and actual_cuda else "PyTorch was not built with a discoverable CUDA runtime",
        observed=actual_cuda or None,
        recovery_candidate=pin.get("cuda_runtime"),
    )
    available = bool(torch_info.get("cuda_available"))
    requirements["cuda_available"] = _probe_status(available, detail=None if available else str(torch_info.get("reason", "torch.cuda.is_available() is false")))
    gpu_match = bool(selected_devices) and len(selected_indices) == count_expected and (not names_expected or names_actual == names_expected)
    requirements["expected_gpu"] = _probe_status(
        gpu_match,
        detail=None if gpu_match else "selected training GPU(s) are not visible in the current native runtime",
        observed=names_actual,
        selected_indices=selected_indices,
        expected=names_expected,
        expected_count=count_expected,
        recovery_candidate=pin.get("expected_gpu_names", []),
    )
    selected_ok = bool(selected_devices) and all(bool(item.get("bf16_supported", item.get("bf16_tensor_operation", False))) for item in selected_devices)
    requirements["bf16_tensor"] = _probe_status(
        available and bool(torch_info.get("bf16_tensor")) and selected_ok,
        detail=None if available and torch_info.get("bf16_tensor") and selected_ok else "BF16 tensor operation failed on one or more selected GPUs",
        selected_indices=selected_indices,
    )
    gemm_ok = bool(torch_info.get("small_cuda_gemm")) or all(bool(item.get("small_cuda_gemm")) for item in selected_devices)
    requirements["small_cuda_gemm"] = _probe_status(
        available and bool(selected_devices) and gemm_ok,
        detail=None if available and selected_devices and gemm_ok else "CUDA GEMM failed on one or more selected GPUs",
        selected_indices=selected_indices,
    )
    candidate = Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint_path(root)
    source_candidate = Path(source_checkpoint_path) if source_checkpoint_path is not None else _default_source_checkpoint_path(root)
    requirements["safetensors_load"] = _probe_safetensors(candidate)
    requirements["d2m_checkpoint_load"] = _probe_d2m_checkpoint(candidate)
    requirements["source_checkpoint_readable"] = _probe_safetensors(source_candidate)
    if imported and available:
        try:
            import torch  # type: ignore
            requirements.update(_probe_torch_models(torch, device=f"cuda:{selected_indices[0] if selected_indices else 0}"))
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            requirements.update({name: _probe_status(False, detail="PyTorch model probes could not import", error=str(exc)) for name in ("p16_forward", "dense_teacher_ffn_forward")})
    else:
        detail = "CUDA is unavailable; CUDA model forwards were not attempted"
        requirements.update({name: _probe_status(False, detail=detail, device="cuda:0") for name in ("p16_forward", "dense_teacher_ffn_forward")})
    try:
        from .partition import oracle as _oracle  # noqa: F401
        requirements["oracle_module_import"] = _probe_status(True, module="dense2moe.partition.oracle")
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        requirements["oracle_module_import"] = _probe_status(False, detail="oracle module import failed", error=str(exc))
    blockers = [f"{name}: {probe.get('detail', 'probe failed')}" for name, probe in requirements.items() if name in REQUIRED_ENVIRONMENT_PROBES and probe.get("status") != "PASS"]
    lock_path = Path(runtime_lock_path) if runtime_lock_path is not None else root / "runs" / RUNTIME_LOCK_FILENAME
    runtime_lock = check_runtime_lock(observed, lock_path)
    if runtime_lock.get("status") in {WINDOWS_RUNTIME_DRIFT, "INVALID"}:
        drift = ", ".join(runtime_lock.get("drift", [])) or str(runtime_lock.get("error", "invalid runtime lock"))
        blockers.append(f"runtime_lock: {WINDOWS_RUNTIME_DRIFT}: {drift}")
    status = "GREEN" if not blockers else "BLOCKED"
    return {
        "schema_version": ENVIRONMENT_DOCTOR_SCHEMA_VERSION,
        "status": status,
        "ok": not blockers,
        "requirements": requirements,
        "checks": requirements,
        "required_probes": list(REQUIRED_ENVIRONMENT_PROBES),
        "blockers": blockers,
        "checkpoint_path": str(candidate) if candidate is not None else None,
        "source_checkpoint_path": str(source_candidate) if source_candidate is not None else None,
        "recovery_pin": pin,
        "recovery_profile_advisory": True,
        "expected_gpu_names": names_expected,
        "expected_gpu_count": count_expected,
        "selected_training_gpus": selected_indices,
        "runtime_lock": runtime_lock,
        "runtime_lock_path": str(lock_path),
    }


def collect_environment(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    system = platform.system()
    memory = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        memory = {"total": vm.total, "available": vm.available, "free": vm.free, "swap_total": swap.total, "swap_free": swap.free}
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        pass
    tools = {
        "python": (None, ["--version"]),
        "git": ("git", ["--version"]),
        "cmake": ("cmake", ["--version"]),
        "ninja": ("ninja", ["--version"]),
        "nvcc": ("nvcc", ["--version"]),
        "nvidia_smi_list": ("nvidia-smi", ["-L"]),
        "nvidia_smi_query": ("nvidia-smi", ["--query-gpu=index,name,memory.total,driver_version,pci.bus_id,temperature.gpu,power.limit", "--format=csv"]),
    }
    probes = {name: (run_probe([sys.executable, *args]) if executable is None else _probe_tool(executable, args)) for name, (executable, args) in tools.items()}
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return {
        "platform": {"system": system, "release": platform.release(), "version": platform.version(), "machine": platform.machine(), "python": sys.version, "python_executable": sys.executable},
        "python_executable": sys.executable,
        "python_version": sys.version,
        "windows": system == "Windows",
        "wsl": detect_wsl(),
        "memory": memory,
        "working_directory": str(Path.cwd()),
        "tools": probes,
        "torch": probe_torch(),
        "packages": _optional_package_versions(),
        "recovery_pin": load_recovery_pin(root / "runs" / "windows-cuda-v3.json"),
        "runtime_lock_path": str(root / "runs" / RUNTIME_LOCK_FILENAME),
        "environment_variables": {key: os.environ.get(key) for key in ("CUDA_PATH", "CUDA_HOME", "WSL_DISTRO_NAME") if key in os.environ},
    }
