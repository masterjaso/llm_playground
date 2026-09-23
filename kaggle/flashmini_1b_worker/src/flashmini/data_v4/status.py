"""Concise corpus-build metrics dashboard (v4)."""

from __future__ import annotations


def render_dashboard(state: dict, *, cache_used_bytes: int = 0,
                     hf_bytes: int = 0, hf_revision: str = "") -> str:
    tok_by_dom = state.get("estimated_tokens_by_domain", {})
    total = sum(tok_by_dom.values()) if tok_by_dom else 0
    target = int(state.get("training_target_tokens", 0) or 0)
    published_exact = int(state.get("published_train_tokens", 0) or 0)
    scheduler = state.get("scheduler") or {}
    deficits = scheduler.get("targets", {})
    actual = scheduler.get("actual", {})
    remaining = {
        domain: max(0, int(value) - int(actual.get(domain, 0)))
        for domain, value in deficits.items()
    }
    telemetry = state.get("telemetry") or {}
    stream_seconds = float(telemetry.get("source_stream_seconds", 0.0) or 0.0)
    accepted = int(state.get("training_tokens", 0) or 0)
    accepted_rate = accepted / stream_seconds if stream_seconds else 0.0
    published_ratio = published_exact / target if target else 0.0
    source_mb_rate = (float(telemetry.get("source_bytes", 0) or 0) / 1024**2 /
                      stream_seconds if stream_seconds else 0.0)
    upload_seconds = float(telemetry.get("upload_and_verify_seconds", 0.0) or 0.0)
    upload_mb_rate = (float(state.get("published_bytes", 0) or 0) / 1024**2 /
                      upload_seconds if upload_seconds else 0.0)
    lines = [
        f"recipe={state.get('recipe_name','')} hash={str(state.get('recipe_hash',''))[:12]}",
        (f"docs selected={state.get('selected_documents',0)} "
         f"exact_dupes={state.get('exact_duplicates',0)} "
         f"near_dupes={state.get('near_duplicates',0)}"),
        f"rejected={state.get('rejected_documents',{})}",
        f"est. tokens total~{total} by_domain={tok_by_dom}",
        (f"published shards={len(state.get('published_shards',[]))} "
         f"bytes={state.get('published_bytes',0)} rev="
         f"{state.get('hf_progress_revision') or hf_revision or state.get('hf_revision','')}"),
        (f"exact published={published_exact}/{target} "
         f"({published_ratio:.6%}) "
         f"accepted_exact_tokens_per_source_second={accepted_rate:.2f}"),
        (f"throughput source={source_mb_rate:.2f} MB/s "
         f"upload_verified={upload_mb_rate:.2f} MB/s "
         f"records={telemetry.get('source_records', 0)}"),
        f"domain_deficits={remaining}",
        (f"benchmark_exclusion={state.get('benchmark_exclusion_status','unknown')} "
         f"private_eval_pending={state.get('benchmark_exclusion_status') != 'complete'}"),
        f"cache used={cache_used_bytes / 1024**3:.2f} GiB hf_bytes={hf_bytes}",
        (f"last_op={state.get('last_successful_operation','')} "
         f"fingerprint={str(state.get('corpus_fingerprint',''))[:16]}"),
    ]
    return "\n".join(lines)
