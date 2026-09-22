# FlashMini sources (v4)

Candidate families and roles; verified revision/license/gating come from
`registry/source_snapshot.lock.json` after `source-lock`. Gated NVIDIA
sources stay recipe-only unless terms explicitly permit redistribution.

## Enabled sources (corpus-v1)

| Source | Dataset | Revision | Domain | Access | Class | Selected docs (corpus-v1) |
|---|---|---|---|---|---|---|
| `fineweb_edu` | HuggingFaceFW/fineweb-edu | `87f09149ef47` | general_web | streams | mirror_allowed | ~500/shard across 84 shards |
| `fineweb` | HuggingFaceFW/fineweb | `9bb295ddab0e` | general_web | streams | mirror_allowed | ~500/shard |
| `common_pile_books` | common-pile/pre_1929_books | `a158135b4765` | books_reference | streams | mirror_allowed | dominant (pre-1929, public domain) |
| `cosmopedia` | HuggingFaceTB/cosmopedia-v2 | `3ba9d6057741` | synthetic_edu | streams | mirror_allowed | minority |
| `finemath` | HuggingFaceTB/finemath | `e92b25a61673` | math_stem | streams | mirror_allowed | in lost 12-shard gap |
| `open_web_math` | open-web-math/open-web-math | `fde8ef8de230` | math_stem | streams | mirror_allowed | in lost 12-shard gap |
| `stackv2_edu` | common-pile/stackv2_edu_filtered | `c354dbe88469` | code | streams | review_required | in lost 12-shard gap; recipe-only content |
| `stack_edu` | HuggingFaceTB/stack-edu | `eeec5caac5cc` | code | streams (config Python) | mirror_allowed | probe only; rows lack content column in first rows |
| `openthoughts` | open-thoughts/OpenThoughts3-1.2M | `61bcf9d4eb38` | reasoning | streams (conversations adapter) | review_required | in lost 12-shard gap; recipe-only content |
| `nemotron_cc_v2` | nvidia/Nemotron-CC-v2 | `2669787c66d1` | general_web | `SOURCE_BLOCKED` | gated_recipe_only | terms not accepted |
| `nemotron_code` | nvidia/Nemotron-Pretraining-Code-v1 | `01393d3bd890` | code | `SOURCE_BLOCKED` | gated_recipe_only | terms not accepted |
| `nemotron_math` | nvidia/Nemotron-CC-Math-v1 | `397a2502f202` | math_stem | `SOURCE_BLOCKED` | gated_recipe_only | terms not accepted |

## Corpus-v1 composition

| Domain | Estimated tokens | Share |
|---|---|---|
| books_reference | ~612M | ~35% |
| general_web | ~581M | ~33% |
| code | ~277M | ~16% |
| synthetic_edu | ~230M | ~13% |
| math_stem | ~35M | ~2% |
| reasoning | ~30M | ~2% |

math_stem and reasoning are underrepresented because the shards containing
those domains were among the 12 lost shards (documented in
`manifests/README.md`). Their source pools remain pinned and available for
a follow-up build that recreates those domains.

## Post-training pools

Post-training pools (separate from base): code/engineering, reasoning,
instruction-following (verifiable constraints), tool use, operational safety
(read-only < reversible < recoverable < irreversible/external/credential),
concise neutral style (minimum words to fully satisfy; no canned
friendliness), general assistant, preference pairs.

