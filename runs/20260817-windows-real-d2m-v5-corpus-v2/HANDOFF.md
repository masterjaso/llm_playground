# Handoff for `20260817-windows-real-d2m-v5-corpus-v2`

- Current status: `BLOCKED`
- Current phase: `oracle-routed-basis-smoke`
- Last completed gate: `freeze-corpus-v2-complete-visible-trajectories-and-balanced-plan` (complete visible trajectories and final artifact hashes reconciled)
- Active blocker: `BASIS_QUALITY`
- Exact next command: `PYTHONPATH=src python scripts/run_oracle_routed_basis_refinement.py --corpus-receipt data/public_v2/corpus-v2-receipt.json --topology p16/top4 --rows 2 --epochs 1`
- Expected output: Oracle-routed smoke; current environment lacks usable PyTorch, so no serious optimization or replay
- Frozen corpus: 676 records; 20/20 visible Open-SWE trajectories complete (not prefix/suffix truncated); FIT/A/B/C repository/task groups are disjoint
- Final on-disk freeze: 676 records; manifest `b4e758527717c55127ea04fe3ee4b94087a7dcfd9deea39800c55657c00be35b`, receipt `a62de56317146b5a8d2eb828f95aa65a4561e59ae918a9dea1b509feccc8ff6b`, splits `29a5f3b1dbcea6ea6df5ad3a6c27fad35c7d587bff0d738f9a28e63b5e7a8646`
- Source-mixture gate: `READY_FOR_BALANCED_CAPTURE` (complete trajectories are retained; activation plan caps each trajectory at 12,288 tokens and targets a 750,000-token production budget)
- Code commit: `b750733f3c2c337410f466c9bfc5c7db3042f941`
- Relevant log: `runs/20260817-windows-real-d2m-v5-corpus-v2/events.jsonl`
- Resume command: `d2m run --run-dir runs/20260817-windows-real-d2m-v5-corpus-v2 --resume`
