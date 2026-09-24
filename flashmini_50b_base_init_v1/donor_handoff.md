# Donor handoff

**BASE PARAMETER COUNT:** 50,276,673,408
**MTP PARAMETER COUNT:** 685,511,168
**TOTAL CHECKPOINT PARAMS:** 50,962,184,576

Source commit: `5387a53a6cbe81ca82339360fb4adcedfec1cb6f` (tree `5ed8ad6e83c155bbd773034150e2a9e6347e5fe03b461d01320e5b2598dadb66`).
Tokenizer fingerprint: `ad7e623aca9d08891d9db1d270c7d6a8c612780890e01511255e0bbeb597e56e`.

1. Check out `main` at or after the source commit; outside `flashmini_50b_base_init_v1/` it must equal the source commit.
2. Create the environment from `environment.lock` (Python 3.13).
3. Materialize all 32 shards (101,924,369,152 bytes): see `materialization_command.txt`.
   The first run uses `--no-expected-hashes` and then `record-hashes`. After
   `shard_hashes.json` exists, rematerialize once so every shard is checked.
4. Run preflight: see `preflight_command.txt`. It must end with `FLASHMINI V4 TRAINING PREFLIGHT: PASS`.
5. Fill every null in a copy of `train_example.yaml`, then launch with `training_launch_command.txt`.
6. Resume: rerun the same launch command; the runner restores `checkpoint.dir/latest`.

No training has been run. No optimizer state is included.
