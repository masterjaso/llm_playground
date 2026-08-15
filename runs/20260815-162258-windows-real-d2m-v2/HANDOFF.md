# Handoff for `20260815-162258-windows-real-d2m-v2`

- Current status: `BLOCKED`
- Current phase: `training`
- Last completed gate: `full-model-spike`
- Active blocker: `TEACHER_RESOURCE_INSUFFICIENT: install CUDA-enabled PyTorch or provide a multi-device/offload runtime, then resume capture`
- Exact next command: `d2m capture --run-dir runs/20260815-162258-windows-real-d2m-v2 --layers 0,16,32,48,63 --dataset-manifest runs/20260815-162258-windows-real-d2m-v2/capture/data-plan.json --source-dir runs/20260815-030931-windows/source --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --split both --device-map auto --resume`
- Expected output: native teacher activation shards and split manifests after a CUDA/offload-capable runtime is installed
- Code commit: `d2e612c44728335c636ec4bbb537f55752b7909e`
- Relevant log: `runs/20260815-162258-windows-real-d2m-v2/events.jsonl`
- Resume command: `d2m run --run-dir runs/20260815-162258-windows-real-d2m-v2 --resume`
