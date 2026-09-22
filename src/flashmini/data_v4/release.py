"""Immutable release manifests and readiness gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import dedupe, filters, recipes
from .contamination import CONTAMINATION_VERSION


@dataclass(frozen=True)
class Readiness:
    one_b_ready: bool
    fifty_b_pipeline_ready: bool
    fifty_b_materialized_ready: bool
    blockers: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "FLASHMINI_1B_DATA_READY": self.one_b_ready,
            "FLASHMINI_50B_DATA_PIPELINE_READY": self.fifty_b_pipeline_ready,
            "FLASHMINI_50B_DATA_MATERIALIZED_READY": self.fifty_b_materialized_ready,
            "blockers": list(self.blockers),
        }


def _totals(shard_rows: list[dict]) -> dict:
    domain = Counter(); source = Counter(); split = Counter(); exact_split = Counter()
    exact_domain = Counter(); exact_source = Counter()
    for row in shard_rows:
        for field, target in (("domain_distribution", domain), ("source_distribution", source),
                              ("split_distribution", split)):
            for key, value in (row.get(field) or {}).items():
                target[key] += int(value)
        for key, value in (row.get("exact_tokens_by_split") or {}).items():
            exact_split[key] += int(value)
        for key, value in (row.get("exact_tokens_by_domain") or {}).items():
            exact_domain[key] += int(value)
        for key, value in (row.get("exact_tokens_by_source") or {}).items():
            exact_source[key] += int(value)
    return {"domain": dict(domain), "source": dict(source),
            "exact_domain": dict(exact_domain), "exact_source": dict(exact_source),
            "split": dict(split),
            "exact_tokens_by_split": dict(exact_split)}


def readiness_from_manifest(manifest: dict, *, target_tokens: int,
                            source_lock_ok: bool = True,
                            source_capacity_ready: bool = False,
                            tokenizer_frozen: bool = False,
                            representation_ready: bool = False,
                            sampler_ready: bool = False,
                            cache_ready: bool = False,
                            decontamination_ready: bool = False) -> Readiness:
    rows = list(manifest.get("shards", []))
    totals = _totals(rows)
    train_exact = int(totals["exact_tokens_by_split"].get("train", 0))
    all_remote = bool(rows) and all(row.get("published") and row.get("sha256") for row in rows)
    one_flags = {
        "source_lock": source_lock_ok,
        "source_capacity": source_capacity_ready,
        "tokenizer": tokenizer_frozen,
        "representation": representation_ready,
        "sampler": sampler_ready,
        "cache": cache_ready,
        "decontamination": decontamination_ready,
        "exact_tokens": bool(manifest.get("exact_token_ready")) and train_exact == int(target_tokens),
        "remote_artifacts": all_remote,
    }
    blockers = tuple(f"1B gate: {key}" for key, ok in one_flags.items() if not ok)
    one_ready = not blockers
    # Pipeline readiness is intentionally distinct from materialization.  It
    # requires the same mechanisms but does not claim an 8T corpus exists.
    pipeline_ready = all((source_lock_ok, tokenizer_frozen, representation_ready,
                          sampler_ready, cache_ready, decontamination_ready,
                          source_capacity_ready))
    materialized = pipeline_ready and train_exact >= 8_000_000_000_000
    return Readiness(one_ready, pipeline_ready, materialized, blockers)


def build_release_manifest(*, release_id: str, view_id: str, recipe: dict,
                           canonical_fingerprint: str, training_view_fingerprint: str,
                           tokenizer: dict, source_lock_sha256: str,
                           shard_rows: list[dict], validation_tokens: int,
                           status: str = "estimate-only") -> dict:
    totals = _totals(shard_rows)
    exact_train = int(totals["exact_tokens_by_split"].get("train", 0))
    exact_val = int(totals["exact_tokens_by_split"].get("val", 0))
    return {
        "release_id": release_id,
        "view_id": view_id,
        "recipe_name": recipe["name"],
        "recipe_hash": recipes.recipe_hash(recipe),
        "canonical_corpus_fingerprint": canonical_fingerprint,
        "training_view_fingerprint": training_view_fingerprint,
        "tokenizer": tokenizer,
        "exact_train_tokens": exact_train,
        "target_train_tokens": int(recipe["target_tokens"]),
        "exact_validation_tokens": exact_val,
        "validation_tokens_target": int(validation_tokens),
        "domain_exact_tokens": totals["exact_domain"],
        "source_exact_tokens": totals["exact_source"],
        "source_lock_sha256": source_lock_sha256,
        "packing_version": "flashmini-v4-pack-contiguous-v2",
        "dedupe_version": dedupe.DEDUPE_VERSION,
        "filter_version": filters.FILTER_VERSION,
        "benchmark_exclusion_version": CONTAMINATION_VERSION,
        "shards": shard_rows,
        "status": status,
    }


def storage_projection(*, tokens: int, bytes_per_token: float,
                       canonical_bytes_per_token: float = 0.0,
                       index_bytes: int = 0, temporary_multiplier: float = 1.0) -> dict:
    """Report physical storage components without assuming pilot ratios."""
    token_bytes = round(int(tokens) * float(bytes_per_token))
    canonical_bytes = round(int(tokens) * float(canonical_bytes_per_token))
    return {
        "tokens": int(tokens), "tokenized_bytes": token_bytes,
        "canonical_bytes": canonical_bytes, "index_bytes": int(index_bytes),
        "temporary_working_set_bytes": round((token_bytes + canonical_bytes + index_bytes)
                                              * float(temporary_multiplier)),
        "total_bytes": token_bytes + canonical_bytes + int(index_bytes),
    }


def write_release_manifest(manifest: dict, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    target.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
