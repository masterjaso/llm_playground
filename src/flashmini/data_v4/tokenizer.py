"""Tokenizer helpers: corpus stays tokenizer-independent (v4)."""

from __future__ import annotations


def tokenizer_identity(tokenizer_id: str, revision: str) -> str:
    return f"{tokenizer_id}@{revision}"


def load_tokenizer(tokenizer_id: str = "gpt2", revision: str | None = None):
    from transformers import AutoTokenizer
    kwargs = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
    if tok.eos_token_id is None:
        raise ValueError("tokenizer requires an EOS token")
    return tok
