# FlashMini GCP Spot helpers

Ephemeral Spot G4 smoke / throughput tests, modeled on the arc_modelling Spot
playbook (`CONFIRM_CREATE` gate, zone fallback, IAP SSH, EXIT teardown).

```bash
# Dry run (no VM)
scripts/cloud/gcp_spot_smoke.sh

# Create Spot g4-standard-48, train short segments, tear down
CONFIRM_CREATE=yes scripts/cloud/gcp_spot_smoke.sh
```

Requires MelodyArc `gcloud` auth to `melodyarc-demo-01`, and the locked PoC
corpus staged under a GCS path named by `.artifacts/flashmini-spot/bucket.txt`
or `FLASHMINI_BUCKET` when running a data-backed smoke. Do not leave VMs or
temp buckets behind; delete residuals after the run.
