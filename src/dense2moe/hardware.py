"""Windows-aware environment and accelerator discovery."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .command import run_guarded


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


def probe_torch() -> dict[str, Any]:
    result: dict[str, Any] = {"installed": bool(importlib.util.find_spec("torch"))}
    if not result["installed"]:
        result["reason"] = "PyTorch is not installed in the active Windows interpreter"
        return result
    try:
        import torch  # type: ignore

        result.update({"version": torch.__version__, "compiled_cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())})
        devices: list[dict[str, Any]] = []
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                props = torch.cuda.get_device_properties(index)
                try:
                    bf16_supported = bool(torch.cuda.is_bf16_supported())
                except TypeError:  # older torch versions do not accept keyword flags
                    bf16_supported = False
                except (OSError, RuntimeError, ValueError):
                    bf16_supported = False
                device_record: dict[str, Any] = {
                    "index": index,
                    "name": props.name,
                    "compute_capability": [props.major, props.minor],
                    "total_memory": props.total_memory,
                    "bf16_supported": bf16_supported,
                    "allocated": int(torch.cuda.memory_allocated(index)),
                    "reserved": int(torch.cuda.memory_reserved(index)),
                }
                for dtype_name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
                    try:
                        x = torch.ones((16, 16), device=f"cuda:{index}", dtype=dtype)
                        _ = x @ x
                        torch.cuda.synchronize(index)
                        device_record[f"{dtype_name}_matmul"] = True
                    except (OSError, RuntimeError, ValueError, TypeError) as exc:
                        device_record[f"{dtype_name}_matmul"] = False
                        device_record[f"{dtype_name}_error"] = str(exc)
                devices.append(device_record)
        result["devices"] = devices
        if result["cuda_available"]:
            result["peer_access"] = [[bool(torch.cuda.can_device_access_peer(i, j)) for j in range(torch.cuda.device_count())] for i in range(torch.cuda.device_count())]
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def collect_environment() -> dict[str, Any]:
    system = platform.system()
    memory = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        memory = {"total": vm.total, "available": vm.available, "free": vm.free, "swap_total": swap.total, "swap_free": swap.free}
    except ImportError:
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
    return {
        "platform": {"system": system, "release": platform.release(), "version": platform.version(), "machine": platform.machine(), "python": sys.version},
        "windows": system == "Windows",
        "wsl": detect_wsl(),
        "memory": memory,
        "working_directory": str(Path.cwd()),
        "tools": probes,
        "torch": probe_torch(),
        "environment_variables": {key: os.environ.get(key) for key in ("CUDA_PATH", "CUDA_HOME", "WSL_DISTRO_NAME") if key in os.environ},
    }
