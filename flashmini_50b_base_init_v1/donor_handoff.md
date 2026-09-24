# Donor handoff

**BASE PARAMETER COUNT:** 50,276,673,408
**MTP PARAMETER COUNT:** 685,511,168
**TOTAL CHECKPOINT PARAMS:** 50,962,184,576

Source commit: `6b07972b2d0f0d82cd1e256b2ed076ba9b02caf2` (tree `078bde3c71449536deb42ddb224bf897ed9625a89f5dd4129d5fad56dc316cc4`).
Tokenizer fingerprint: `ad7e623aca9d08891d9db1d270c7d6a8c612780890e01511255e0bbeb597e56e`.

1. Check out `main` at or after the source commit; outside `flashmini_50b_base_init_v1/` it must equal the source commit.
2. Create the environment from `environment.lock` (Python 3.13).
3. Materialize all 32 shards (101,924,369,152 bytes): see `materialization_command.txt`.
   Every shard is checked against `shard_hashes.json`.
4. Run preflight: see `preflight_command.txt`. It must end with `FLASHMINI V4 TRAINING PREFLIGHT: PASS`.
5. Fill every null in a copy of `train_example.yaml`, then launch with `training_launch_command.txt`.
6. Resume: rerun the same launch command; the runner restores `checkpoint.dir/latest`.

No training has been run. No optimizer state is included.
