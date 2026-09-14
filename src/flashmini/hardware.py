"""Hardware doctor — report GPU/CPU/RAM/CUDA/torch environment.

Emitted as machine-readable JSON for every run's environment.json.
"""

from __future__ import annotations

import json
import platform
import sys
from typing import Any


def _torch_info() -> dict[str, Any]:
    try:
        import torch

        info: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            info["devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "total_memory_gib": round(props.total_memory / 2**30, 2),
                        "capability": [props.major, props.minor],
                    }
                )
        return info
    except Exception as e:  # pragma: no cover
        return {"torch_error": str(e)}


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_version": platform.python_version(),
        "cpu_count": None,
        "ram_bytes": None,
        "ram_gib": None,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_bytes"] = int(vm.total)
        info["ram_gib"] = round(vm.total / 2**30, 2)
    except Exception:
        pass
    return info


def collect() -> dict[str, Any]:
    return {"system": _system_info(), "torch": _torch_info()}


def main() -> None:
    print(json.dumps(collect(), indent=2))


if __name__ == "__main__":
    main()