#!/usr/bin/env bash
# FlashMini Spot smoke/throughput test on one GCP G4 (RTX PRO 6000 Blackwell 96 GB).
#
# Modeled on arc_modelling research/recipes/qwen36-engine-perf-ladder/run_spot_perf.sh:
# CONFIRM_CREATE gate, Spot only, zone fallback, IAP SSH, nohup job + DONE/FAILED
# markers, results pulled back, EXIT trap that deletes the VM and disks and lists
# residuals. Adds --max-run-duration as an out-of-band delete in case this shell dies.
#
# Usage (creates and pays for a VM):
#   CONFIRM_CREATE=yes scripts/cloud/gcp_spot_smoke.sh
# Dry run (prints the plan only):
#   scripts/cloud/gcp_spot_smoke.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"

GCP_PROJECT="${GCP_PROJECT:-melodyarc-demo-01}"
MACHINE_TYPE="${MACHINE_TYPE:-g4-standard-48}"
ZONE_CANDIDATES="${ZONE_CANDIDATES:-us-central1-b,us-central1-c,us-central1-f,us-east1-b,us-east1-d,us-east4-b,us-east4-c}"
ALLOC_ROUNDS="${ALLOC_ROUNDS:-8}"            # G4 Spot stock churns between zones; retry whole list
ALLOC_SLEEP="${ALLOC_SLEEP:-20}"
BOOT_DISK_GB="${BOOT_DISK_GB:-200}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-55m}"   # GCE deletes the VM after this, regardless of this shell
POLL_MAX_MIN="${POLL_MAX_MIN:-40}"
SWEEP_BATCHES="${SWEEP_BATCHES:-16 64 128}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-180}"
TREATMENT="${TREATMENT:-c}"
ART_ROOT="${ART_ROOT:-/home/jasonneal/workspace/.artifacts/flashmini-spot}"
FLASHMINI_BUCKET="${FLASHMINI_BUCKET:-$(cat "${ART_ROOT}/bucket.txt" 2>/dev/null || true)}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
INSTANCE_NAME="${INSTANCE_NAME:-flashmini-smoke-${TS,,}}"
NET_TAG="flashmini-smoke-${TS,,}"
FW_RULE="${NET_TAG}-iap-ssh"
OUT="${ART_ROOT}/run-${TS}"

export CLOUDSDK_CORE_PROJECT="${GCP_PROJECT}"
export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1

[[ -n "${FLASHMINI_BUCKET}" ]] || { echo "FLASHMINI_BUCKET not set and ${ART_ROOT}/bucket.txt missing" >&2; exit 2; }

echo "plan:"
echo "  project=${GCP_PROJECT} machine=${MACHINE_TYPE} provisioning=SPOT zones=${ZONE_CANDIDATES}"
echo "  max_run_duration=${MAX_RUN_DURATION} (auto-delete) poll_max=${POLL_MAX_MIN}min"
echo "  data=gs://${FLASHMINI_BUCKET}/data/fineweb_v3_2b  treatment=${TREATMENT}"
echo "  sweep batches='${SWEEP_BATCHES}' segment=${SEGMENT_SECONDS}s each"
echo "  out=${OUT}"
if [[ "${CONFIRM_CREATE:-}" != "yes" ]]; then
  echo "Dry run only. Re-run with CONFIRM_CREATE=yes to create the VM." >&2
  exit 2
fi

mkdir -p "${OUT}"
log() { printf '[%s] %s\n' "$(date -u +%T)" "$*" | tee -a "${OUT}/run.log"; }

WINNING_ZONE=""
cleanup() {
  local ec=$?
  log "=== teardown (exit=${ec}) ==="
  if [[ -n "${WINNING_ZONE}" ]]; then
    gcloud compute instances delete "${INSTANCE_NAME}" --zone="${WINNING_ZONE}" \
      --delete-disks=all --quiet 2>&1 | tee -a "${OUT}/teardown.log" || true
  else
    IFS=',' read -r -a zs <<<"${ZONE_CANDIDATES}"
    for z in "${zs[@]}"; do
      gcloud compute instances delete "${INSTANCE_NAME}" --zone="${z}" \
        --delete-disks=all --quiet >/dev/null 2>&1 || true
    done
  fi
  for _ in 1 2 3; do
    gcloud compute firewall-rules delete "${FW_RULE}" --quiet >/dev/null 2>&1 && break
    gcloud compute firewall-rules describe "${FW_RULE}" >/dev/null 2>&1 || break
    sleep 10
  done
  log "residual instances named flashmini-smoke*:"
  gcloud compute instances list --filter="name~^flashmini-smoke" \
    --format='table(name,zone.basename(),status)' 2>&1 | tee -a "${OUT}/residuals.txt" || true
  log "residual disks named flashmini-smoke*:"
  gcloud compute disks list --filter="name~^flashmini-smoke" \
    --format='table(name,zone.basename(),sizeGb)' 2>&1 | tee -a "${OUT}/residuals.txt" || true
  log "firewall rules named flashmini-smoke*:"
  gcloud compute firewall-rules list --filter="name~^flashmini-smoke" \
    --format='table(name)' 2>&1 | tee -a "${OUT}/residuals.txt" || true
  log "teardown attempted"
}
trap cleanup EXIT

gcloud auth print-access-token >/dev/null
log "project=$(gcloud config get-value project 2>/dev/null) account=$(gcloud config get-value account 2>/dev/null)"
gcloud storage ls "gs://${FLASHMINI_BUCKET}/data/fineweb_v3_2b/data_manifest.json" | tee "${OUT}/gcs-check.txt"

log "=== firewall (IAP SSH only, tag ${NET_TAG}) ==="
gcloud compute firewall-rules create "${FW_RULE}" --network=default --direction=INGRESS \
  --action=ALLOW --rules=tcp:22 --source-ranges=35.235.240.0/20 --target-tags="${NET_TAG}" \
  2>&1 | tee -a "${OUT}/create.log"

log "=== create SPOT ${MACHINE_TYPE} ==="
IFS=',' read -r -a ZONES <<<"${ZONE_CANDIDATES}"
for round in $(seq 1 "${ALLOC_ROUNDS}"); do
  for z in "${ZONES[@]}"; do
    log "round ${round}/${ALLOC_ROUNDS}: trying zone ${z} ..."
    if gcloud compute instances create "${INSTANCE_NAME}" \
        --zone="${z}" --machine-type="${MACHINE_TYPE}" \
        --provisioning-model=SPOT --instance-termination-action=DELETE \
        --max-run-duration="${MAX_RUN_DURATION}" \
        --maintenance-policy=TERMINATE \
        --image-family=common-cu129-ubuntu-2204-nvidia-580 \
        --image-project=deeplearning-platform-release \
        --boot-disk-type="${BOOT_DISK_TYPE}" --boot-disk-size="${BOOT_DISK_GB}GB" \
        --boot-disk-auto-delete --tags="${NET_TAG}" --scopes=cloud-platform \
        --labels=app=flashmini,purpose=smoke-test,managed-by=manual \
        >>"${OUT}/create.log" 2>&1; then
      WINNING_ZONE="${z}"
      break 2
    fi
    log "  miss: $(grep -Eo 'ZONE_RESOURCE_POOL_EXHAUSTED[A-Z_]*|QUOTA_EXCEEDED|Quota [^.]*exceeded|not found|Permission[^.]*' "${OUT}/create.log" | tail -1)"
  done
  [[ "${round}" -lt "${ALLOC_ROUNDS}" ]] && sleep "${ALLOC_SLEEP}"
done
[[ -n "${WINNING_ZONE}" ]] || { log "allocation_failed in all zones"; exit 3; }
log "created in ${WINNING_ZONE}"

ssh_cmd() {
  gcloud compute ssh "${INSTANCE_NAME}" --zone="${WINNING_ZONE}" --tunnel-through-iap \
    --quiet --command="$1"
}

log "=== wait for SSH ==="
for i in $(seq 1 40); do
  if ssh_cmd 'echo ssh-ok' >/dev/null 2>&1; then log "ssh ready (attempt ${i})"; break; fi
  sleep 15
  [[ "${i}" -eq 40 ]] && { log "ssh timeout"; exit 4; }
done

log "=== upload source (git archive HEAD) ==="
git -C "${REPO}" archive --format=tar.gz --prefix=flashmini/ -o "${OUT}/src.tgz" HEAD
cp "${HERE}/remote_smoke.sh" "${OUT}/remote_smoke.sh"
gcloud compute scp --zone="${WINNING_ZONE}" --tunnel-through-iap --quiet \
  "${OUT}/src.tgz" "${OUT}/remote_smoke.sh" "${INSTANCE_NAME}:/tmp/"

log "=== start remote job ==="
ssh_cmd "rm -rf /tmp/smoke && mkdir -p /tmp/smoke && \
  BUCKET='${FLASHMINI_BUCKET}' SWEEP_BATCHES='${SWEEP_BATCHES}' SEGMENT_SECONDS='${SEGMENT_SECONDS}' TREATMENT='${TREATMENT}' \
  setsid nohup sudo -E bash /tmp/remote_smoke.sh >/tmp/smoke/remote.out 2>&1 </dev/null & disown; echo started"

log "=== poll (up to ${POLL_MAX_MIN} min) ==="
STATE=""
for i in $(seq 1 $((POLL_MAX_MIN * 2))); do
  if ssh_cmd 'test -f /tmp/smoke/DONE' 2>/dev/null; then STATE=DONE; break; fi
  if ssh_cmd 'test -f /tmp/smoke/FAILED' 2>/dev/null; then STATE=FAILED; break; fi
  if (( i % 4 == 0 )); then
    ssh_cmd 'tail -n 6 /tmp/smoke/remote.out' 2>/dev/null | tee -a "${OUT}/poll.log" || true
  fi
  sleep 30
done
log "poll_end state=${STATE:-timeout}"

log "=== pull results ==="
gcloud compute scp --zone="${WINNING_ZONE}" --tunnel-through-iap --quiet --recurse \
  "${INSTANCE_NAME}:/tmp/smoke" "${OUT}/" 2>&1 | tail -3 || true
[[ -f "${OUT}/smoke/summary.md" ]] && cat "${OUT}/smoke/summary.md"
log "complete; artifacts in ${OUT} (state=${STATE:-timeout})"
