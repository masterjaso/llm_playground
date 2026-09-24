# Donor handoff

**BASE PARAMETER COUNT:** 50276673408  
**MTP PARAMETER COUNT:** 685511168  
**TOTAL CHECKPOINT PARAMS:** 50962184576  

Run the materializer first: `python -m flashmini.base_init_bundle materialize --bundle . --output checkpoint`.

Then run the donor smoke test: `pytest -q tests/flashmini/test_flashmini_50b_base_init.py`.

The bundle contains a deterministic InitSpec and planned safetensors shards. No full 50B tensor set was materialized on the workstation. No optimizer state or training checkpoint is included. The tokenizer artifact is pending and must be frozen before the first optimizer step.
