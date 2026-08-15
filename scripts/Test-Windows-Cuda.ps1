param(
    [string]$Receipt = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Missing .venv\Scripts\python.exe. Run scripts\Setup-Windows.ps1 first."
}
if (-not $Receipt) {
    $Receipt = Join-Path $projectRoot "runs\windows-cuda-receipt.json"
}

Push-Location $projectRoot
try {
    $script = @'
import json
import platform
import sys
import time
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise SystemExit("WINDOWS_CUDA_NOT_READY: torch.cuda.is_available() is false")
if torch.cuda.device_count() < 2:
    raise SystemExit(f"WINDOWS_CUDA_NOT_READY: expected >=2 GPUs, found {torch.cuda.device_count()}")

devices = []
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        props = torch.cuda.get_device_properties(index)
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        started = time.perf_counter()
        left = torch.randn((2048, 2048), device=f"cuda:{index}", dtype=torch.bfloat16)
        right = torch.randn((2048, 2048), device=f"cuda:{index}", dtype=torch.bfloat16)
        result = left @ right
        torch.cuda.synchronize(index)
        elapsed = time.perf_counter() - started
        devices.append({
            "index": index,
            "name": props.name,
            "total_memory": int(props.total_memory),
            "compute_capability": [int(props.major), int(props.minor)],
            "bf16_supported": bf16_supported,
            "bf16_matmul": {"ok": bool(torch.isfinite(result).all().item()), "seconds": elapsed},
            "free_memory": int(torch.cuda.mem_get_info(index)[0]),
        })
        del left, right, result
        torch.cuda.empty_cache()

payload = {
    "status": "WINDOWS_CUDA_READY",
    "platform": platform.platform(),
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": torch.cuda.device_count(),
    "devices": devices,
}
target = Path(r'''__RECEIPT__''')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
'@
    $script = $script.Replace("__RECEIPT__", $Receipt.Replace("'", "''"))
    $temporaryScript = Join-Path $env:TEMP ("d2m-cuda-gate-" + [guid]::NewGuid().ToString("N") + ".py")
    Set-Content -LiteralPath $temporaryScript -Value $script -Encoding UTF8
    try {
        $process = Start-Process -FilePath $venvPython -ArgumentList @($temporaryScript) -Wait -PassThru -NoNewWindow
        if ($process.ExitCode -ne 0) { throw "CUDA gate failed" }
    }
    finally {
        Remove-Item -LiteralPath $temporaryScript -Force -ErrorAction SilentlyContinue
    }
    Write-Host "WINDOWS_CUDA_READY"
}
finally {
    Pop-Location
}
