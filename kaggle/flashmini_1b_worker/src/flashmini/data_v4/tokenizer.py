"""Tokenizer freeze and provenance helpers for production views."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

TOKENIZER_FORMAT_VERSION = "flashmini-token-format-v1"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


def tokenizer_identity(tokenizer_id: str, revision: str) -> str:
    return f"{tokenizer_id}@{revision}"


def token_dtype_for_vocab(vocab_size: int) -> str:
    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    return "uint16" if int(vocab_size) <= 65536 else "uint32"


@dataclass(frozen=True)
class TokenizerSpec:
    tokenizer_id: str
    revision: str
    vocab_size: int
    eos_token_id: int
    pad_policy: str = "eos"
    fingerprint: str = ""
    token_format_version: str = TOKENIZER_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.tokenizer_id or not self.revision:
            raise ValueError("tokenizer_id and immutable revision are required")
        if not _IMMUTABLE_REVISION.fullmatch(self.revision):
            raise ValueError("production tokenizer revision must be an immutable Hub commit")
        if int(self.vocab_size) <= 0 or not 0 <= int(self.eos_token_id) < int(self.vocab_size):
            raise ValueError("invalid tokenizer vocabulary or EOS ID")
        if self.pad_policy not in {"eos", "none", "id"}:
            raise ValueError("pad_policy must be eos, none, or id")

    @property
    def identity(self) -> str:
        return tokenizer_identity(self.tokenizer_id, self.revision)

    @property
    def dtype(self) -> str:
        return token_dtype_for_vocab(self.vocab_size)

    def as_dict(self) -> dict:
        return {
            "tokenizer_id": self.tokenizer_id,
            "revision": self.revision,
            "vocab_size": int(self.vocab_size),
            "eos_token_id": int(self.eos_token_id),
            "pad_policy": self.pad_policy,
            "fingerprint": self.fingerprint,
            "token_format_version": self.token_format_version,
            "token_dtype": self.dtype,
        }


def spec_fingerprint(spec: TokenizerSpec, *, vocab_sha256: str = "") -> str:
    payload = spec.as_dict().copy()
    payload["fingerprint"] = ""
    payload["vocab_sha256"] = vocab_sha256
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_spec(path: str | Path) -> TokenizerSpec:
    import yaml
    raw = yaml.safe_load(Path(path).read_text()) or {}
    spec = TokenizerSpec(
        tokenizer_id=str(raw.get("tokenizer_id", "")),
        revision=str(raw.get("revision", raw.get("tokenizer_revision", ""))),
        vocab_size=int(raw.get("vocab_size", 0)),
        eos_token_id=int(raw.get("eos_token_id", -1)),
        pad_policy=str(raw.get("pad_policy", "eos")),
        fingerprint=str(raw.get("fingerprint", "")),
        token_format_version=str(raw.get("token_format_version", TOKENIZER_FORMAT_VERSION)),
    )
    if spec.fingerprint and len(spec.fingerprint) != 64:
        raise ValueError("tokenizer fingerprint must be SHA256 hex")
    return spec


def load_frozen_spec(path: str | Path = "training_data/tokenizer/production.yaml") -> TokenizerSpec:
    return load_spec(path)


def load_tokenizer(tokenizer_id: str = "gpt2", revision: str | None = None,
                   *, production: bool = False):
    if production and (revision is None or not _IMMUTABLE_REVISION.fullmatch(revision)):
        raise ValueError("production tokenization requires an immutable tokenizer revision")
    from transformers import AutoTokenizer
    kwargs = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
    if tok.eos_token_id is None:
        raise ValueError("tokenizer requires an EOS token")
    return tok
