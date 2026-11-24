#!/usr/bin/env bash
# Runs ON the Spot VM (as root, started by gcp_spot_smoke.sh). Not for local use.
# Stages the locked corpus from GCS, builds the env, then runs a short batch-size
# sweep of one treatment and writes /tmp/smoke/summary.md. Marks DONE or FAILED.
set -euo pipefail
OUTD=/tmp/smoke
mkdir -p "${OUTD}"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then touch "${OUTD}/FAILED"; fi' EXIT
t() { date -u +%T; }
step() { echo "[$(t)] $*" | tee -a "${OUTD}/steps.log"; }
T0=$(date +%s)

: "${BUCKET:?}"
SWEEP_BATCHES="${SWEEP_BATCHES:-16 64 128}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-180}"
TREATMENT="${TREATMENT:-c}"
EXPECTED_MANIFEST_SHA=b06f559f65e6969a9dae36d392873610330b8a2425fe27f47518c4d200d42dab

step "gpu"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "${OUTD}/gpu.txt"
nproc | tee "${OUTD}/nproc.txt"; free -g | tee "${OUTD}/mem.txt"

step "stage corpus from gs://${BUCKET}"
mkdir -p /data/fineweb_v3_2b
S1=$(date +%s)
gcloud storage cp -r "gs://${BUCKET}/data/fineweb_v3_2b/*" /data/fineweb_v3_2b/
echo "corpus_copy_seconds=$(( $(date +%s) - S1 ))" | tee "${OUTD}/timing_corpus.txt"
GOT=$(sha256sum /data/fineweb_v3_2b/data_manifest.json | cut -c1-64)
echo "manifest_sha256=${GOT}" | tee -a "${OUTD}/timing_corpus.txt"
[[ "${GOT}" == "${EXPECTED_MANIFEST_SHA}" ]] || { echo "MANIFEST MISMATCH"; exit 10; }

step "unpack source"
mkdir -p /opt && tar -C /opt -xzf /tmp/src.tgz
cd /opt/flashmini

step "python env (uv)"
S2=$(date +%s)
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
uv venv /opt/venv --python 3.13 || uv venv /opt/venv --python 3.12
PY=/opt/venv/bin/python
if ! uv pip install --python "${PY}" torch --index-url https://download.pytorch.org/whl/cu130; then
  step "cu130 torch failed, trying cu128"
  uv pip install --python "${PY}" torch --index-url https://download.pytorch.org/whl/cu128
fi
uv pip install --python "${PY}" -e /opt/flashmini
echo "env_seconds=$(( $(date +%s) - S2 ))" | tee "${OUTD}/timing_env.txt"
"${PY}" - <<'PYEOF' | tee "${OUTD}/torch.txt"
import torch, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item(); print("bf16 matmul ok")
PYEOF

step "verify corpus integrity"
S3=$(date +%s)
"${PY}" -c 'from pathlib import Path; from flashmini.data import verify_dataset_integrity; print(verify_dataset_integrity(Path("/data/fineweb_v3_2b")))' 2>&1 | tail -5 | tee "${OUTD}/integrity.txt"
echo "integrity_seconds=$(( $(date +%s) - S3 ))" | tee "${OUTD}/timing_integrity.txt"

nvidia-smi dmon -s um -d 5 > "${OUTD}/gpu_dmon.txt" 2>&1 &
DMON=$!

for B in ${SWEEP_BATCHES}; do
  step "train treatment=${TREATMENT} batch=${B} for ${SEGMENT_SECONDS}s"
  RUN="/data/runs/smoke_${TREATMENT}_b${B}"
  set +e
  timeout --signal=INT "${SEGMENT_SECONDS}" \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "${PY}" -m flashmini.cli train \
      --config "configs/flashmini/poc_${TREATMENT}_v3.yaml" \
      --data-dir /data/fineweb_v3_2b --run-dir "${RUN}" \
      --tokens 250000000 --batch-size "${B}" --grad-accum 1 --seed 17 \
      --lr 3e-4 --ple-lr-multiplier 5 --warmup-tokens 524288 \
      --cosine-decay --min-lr-ratio 0.1 --no-checkpoints --log-every 10 \
      > "${OUTD}/train_b${B}.log" 2>&1
  echo "exit=$?" | tee "${OUTD}/train_b${B}.exit"
  set -e
  mkdir -p "${OUTD}/runs_b${B}"; cp "${RUN}/metrics.jsonl" "${OUTD}/runs_b${B}/" 2>/dev/null || true
  tail -n 5 "${OUTD}/train_b${B}.log" | tee -a "${OUTD}/steps.log"
done
kill "${DMON}" 2>/dev/null || true

step "summarize"
"${PY}" - "${OUTD}" "${SWEEP_BATCHES}" <<'PYEOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); batches = sys.argv[2].split()
lines = ["# FlashMini Spot smoke summary", "",
         "| batch | records | tokens_seen | first loss | last loss | steady tok/s (2nd half) | peak GPU GiB | exit |",
         "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
for b in batches:
    p = out / f"runs_b{b}" / "metrics.jsonl"
    recs = []
    if p.exists():
        for ln in p.read_text().splitlines():
            try: recs.append(json.loads(ln))
            except Exception: pass
    recs = [r for r in recs if r.get("event") == "train" and "tokens_seen" in r and "wall_clock" in r]
    ex = (out / f"train_b{b}.exit").read_text().strip() if (out / f"train_b{b}.exit").exists() else "?"
    if len(recs) < 4:
        lines.append(f"| {b} | {len(recs)} | | | | | | {ex} |"); continue
    mid = recs[len(recs)//2]; last = recs[-1]
    dt = last["wall_clock"] - mid["wall_clock"]; dtok = last["tokens_seen"] - mid["tokens_seen"]
    rate = dtok / dt if dt > 0 else float("nan")
    loss = lambda r: r.get("loss", r.get("train_loss", r.get("ema_loss")))
    mem = last.get("gpu_peak_memory") or ""
    if isinstance(mem, dict) and mem: mem = f"{max(mem.values()):.1f}"
    lines.append(f"| {b} | {len(recs)} | {last['tokens_seen']} | {loss(recs[0])} | {loss(last)} | {rate:,.0f} | {mem} | {ex} |")
lines += ["", "Raw logs: train_b*.log, runs_b*/metrics.jsonl, gpu_dmon.txt, timing_*.txt"]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF
echo "total_seconds=$(( $(date +%s) - T0 ))" | tee "${OUTD}/timing_total.txt"
touch "${OUTD}/DONE"
