"""Concise corpus-build metrics dashboard (v4)."""

from __future__ import annotations


def render_dashboard(state: dict, *, cache_used_bytes: int = 0,
                     hf_bytes: int = 0, hf_revision: str = "") -> str:
    tok_by_dom = state.get("estimated_tokens_by_domain", {})
    total = sum(tok_by_dom.values()) if tok_by_dom else 0
    lines = [
        f"recipe={state.get('recipe_name','')} hash={str(state.get('recipe_hash',''))[:12]}",
        f"docs selected={state.get('selected_documents',0)} "
        f"exact_dupes={state.get('exact_duplicates',0)} "
        f"near_dupes={state.get('near_duplicates',0)}",
        f"rejected={state.get('rejected_documents',{})}",
        f"est. tokens total~{total} by_domain={tok_by_dom}",
        f"published shards={len(state.get('published_shards',[]))} "
        f"bytes={state.get('published_bytes',0)} rev={hf_revision or state.get('hf_revision','')}",
        f"cache used={cache_used_bytes / 1024**3:.2f} GiB hf_bytes={hf_bytes}",
        f"last_op={state.get('last_successful_operation','')} "
        f"fingerprint={str(state.get('corpus_fingerprint',''))[:16]}",
    ]
    return "\n".join(lines)
