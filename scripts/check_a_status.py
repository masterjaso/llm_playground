#!/usr/bin/env python3
"""Quick check of POC_A Kaggle run status."""
import subprocess, os, json

env = os.environ.copy()
key_line = open(".env").read().strip().split("\n")[0]
api_key = key_line.split("=")[1] if "=" in key_line else ""
env["KAGGLE_API_KEY"] = api_key

# Kernel status
r = subprocess.run(
    ["kaggle", "kernels", "status", "masterjaso/flashmini-v3-kaggle-continuation-worker"],
    capture_output=True, text=True, timeout=30, env=env,
)
print("Kernel status:", r.stdout.strip() or r.stderr.strip())

# Check A_w5 output
import os.path as osp
if osp.isdir("runs/flashmini/kaggle_continuation/kernel_output/A_w5"):
    print("\nA_w5 output exists!")
    for root, dirs, files in os.walk("runs/flashmini/kaggle_continuation/kernel_output/A_w5"):
        for f in files:
            fp = osp.join(root, f)
            sz = osp.getsize(fp)
            print(f"  {fp}: {sz:,} bytes")
else:
    print("\nA_w5 output not yet downloaded.")

# Controller state
if osp.isfile("runs/flashmini/kaggle_continuation/controller_state.json"):
    with open("runs/flashmini/kaggle_continuation/controller_state.json") as f:
        state = json.load(f)
    print("\nController state A:")
    for k, v in state["A"].items():
        print(f"  {k}: {v}")

# Kernel info via API
import urllib.request
url = "https://www.kaggle.com/api/v1/kernels/masterjaso/flashmini-v3-kaggle-continuation-worker/latest"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
try:
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    print(f"\nKernel info:")
    print(f"  status: {data.get('status')}")
    print(f"  lastRunTime: {data.get('lastRunTime')}")
    print(f"  kernelVersionId: {data.get('kernelVersionId')}")
except Exception as e:
    print(f"\nAPI error: {e}")
