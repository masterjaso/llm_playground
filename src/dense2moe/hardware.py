"""Windows-aware environment discovery and fail-closed ML doctor probes.

The control plane must remain importable on a small CPU-only installation, but
Phase 0 must never claim a usable ML environment from metadata alone.  The
helpers in this module therefore separate cheap discovery (``collect_environment``)
from the required executable checks (``run_environment_doctor``).  Missing
optional packages and unavailable CUDA are represented as named probe failures,
not import-time crashes.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .command import run_guarded

ENVIRONMENT_DOCTOR_SCHEMA_VERSION = 1

# This is the last known successful native-Windows run.  It is deliberately
# treated as a recovery candidate, not as proof that the current interpreter
# has the same environment.  ``load_recovery_pin`` also reads the adjacent
# receipt when available so the observed GPU names and versions remain
# content-addressed by the checked-in file.
DEFAULT_RECOVERY_PIN = Path(__file__).resolve().parents[2] / "runs" / "windows-cuda-v3.json"
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
    import re

    match = re.search(r"\b(\d+\.\d+\.\d+)\b", value)
    return match.group(1) if match else None


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


def run_environment_doctor(*, environment: dict[str, Any] | None = None, repo_root: str | Path | None = None, checkpoint_path: str | Path | None = None, expected_gpu_names: Sequence[str] | None = None, expected_gpu_count: int | None = None, recovery_pin_path: str | Path | None = None) -> dict[str, Any]:
    """Run the Phase 0 contract and fail closed when a required probe misses."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    observed = environment if environment is not None else collect_environment(repo_root=root)
    pin = load_recovery_pin(recovery_pin_path)
    torch_info = observed.get("torch", {}) if isinstance(observed.get("torch"), dict) else {}
    platform_info = observed.get("platform", {}) if isinstance(observed.get("platform"), dict) else {}
    names_expected = sorted(str(item) for item in (expected_gpu_names if expected_gpu_names is not None else pin.get("expected_gpu_names", [])))
    count_expected = int(expected_gpu_count if expected_gpu_count is not None else pin.get("expected_gpu_count", len(names_expected)))
    devices = torch_info.get("devices", []) if isinstance(torch_info.get("devices"), list) else []
    names_actual = sorted(str(item.get("name")) for item in devices if isinstance(item, dict) and item.get("name"))
    requirements: dict[str, dict[str, Any]] = {}
    native = platform_info.get("system") == "Windows" and not bool(observed.get("wsl"))
    requirements["native_windows"] = _probe_status(native, detail=None if native else "doctor must run in native Windows, not WSL/Linux", system=platform_info.get("system"), wsl=bool(observed.get("wsl")))
    python_version = _python_version_from_text(str(platform_info.get("python", "")))
    expected_python = str(pin.get("python_version", ""))
    python_match = bool(python_version) and (not expected_python or python_version == expected_python)
    requirements["python"] = _probe_status(python_match, detail=None if python_match else f"expected pinned Python {expected_python!r}", executable=sys.executable, version=python_version, expected_version=expected_python)
    imported = bool(torch_info.get("installed")) and bool(torch_info.get("import_ok"))
    requirements["torch_import"] = _probe_status(imported, detail=None if imported else str(torch_info.get("reason", "PyTorch import failed")), version=torch_info.get("version"))
    expected_torch = str(pin.get("torch_version", ""))
    actual_torch = str(torch_info.get("version", ""))
    requirements["torch_version"] = _probe_status(imported and actual_torch == expected_torch, detail=None if actual_torch == expected_torch else f"expected pinned torch {expected_torch!r}", observed=actual_torch or None, expected=expected_torch)
    expected_cuda = str(pin.get("cuda_runtime", ""))
    actual_cuda = str(torch_info.get("compiled_cuda", ""))
    requirements["cuda_runtime"] = _probe_status(imported and actual_cuda == expected_cuda, detail=None if actual_cuda == expected_cuda else f"expected pinned CUDA runtime {expected_cuda!r}", observed=actual_cuda or None, expected=expected_cuda)
    available = bool(torch_info.get("cuda_available"))
    requirements["cuda_available"] = _probe_status(available, detail=None if available else str(torch_info.get("reason", "torch.cuda.is_available() is false")))
    gpu_match = bool(names_actual) and len(names_actual) == count_expected and (not names_expected or names_actual == names_expected)
    requirements["expected_gpu"] = _probe_status(gpu_match, detail=None if gpu_match else "visible GPU set does not match the pinned recovery environment", observed=names_actual, expected=names_expected, expected_count=count_expected)
    requirements["bf16_tensor"] = _probe_status(available and bool(torch_info.get("bf16_tensor")), detail=None if torch_info.get("bf16_tensor") else "BF16 tensor operation failed on one or more GPUs")
    requirements["small_cuda_gemm"] = _probe_status(available and bool(torch_info.get("small_cuda_gemm")), detail=None if torch_info.get("small_cuda_gemm") else "small CUDA GEMM failed on one or more GPUs")
    candidate = Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint_path(root)
    requirements["safetensors_load"] = _probe_safetensors(candidate)
    requirements["d2m_checkpoint_load"] = _probe_d2m_checkpoint(candidate)
    if imported and available:
        try:
            import torch  # type: ignore
            requirements.update(_probe_torch_models(torch, device="cuda:0"))
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
        "recovery_pin": pin,
        "expected_gpu_names": names_expected,
        "expected_gpu_count": count_expected,
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
        "platform": {"system": system, "release": platform.release(), "version": platform.version(), "machine": platform.machine(), "python": sys.version},
        "windows": system == "Windows",
        "wsl": detect_wsl(),
        "memory": memory,
        "working_directory": str(Path.cwd()),
        "tools": probes,
        "torch": probe_torch(),
        "packages": _optional_package_versions(),
        "recovery_pin": load_recovery_pin(root / "runs" / "windows-cuda-v3.json"),
        "environment_variables": {key: os.environ.get(key) for key in ("CUDA_PATH", "CUDA_HOME", "WSL_DISTRO_NAME") if key in os.environ},
    }
