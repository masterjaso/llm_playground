# Handoff for `20260815-145505-windows-real-d2m`

- Current status: `BLOCKED` (resumable external corpus dependency)
- Parent run: `20260815-030931-windows` (immutable historical evidence)
- Current phase: `calibration`
- Last completed gate: `oracle-study`
- Active blocker: approved representative calibration corpus and tokenizer/split manifest are not available in scope; no empty plan was accepted. The p8 oracle ceiling is red and needs a bounded capacity follow-up after data is available.
- Exact next command: `d2m prepare-data --run-dir runs/20260815-145505-windows-real-d2m --corpus-manifest <approved-jsonl>`
- Expected output: `capture/data-plan.json` with disjoint fixed train/holdout IDs and dataset hash
- Resume command: `d2m run --run-dir runs/20260815-145505-windows-real-d2m --resume`
- Evidence: `reports/final-report.json`, `reports/quality-report.json`, `metrics/oracle-ablation.json`
