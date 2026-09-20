"""Deterministic tokenization + packing cache for canonical shards (v4).

The canonical corpus is tokenizer-independent: shards store text plus stable
document identity.  Training consumes *tokens*, generated on the fly from a
declared tokenizer identity so a tokenizer freeze decision can still be made
later without rebuilding the corpus.

Packing contract (document-granular, version ``PACKING_VERSION``):

* one training sequence corresponds to one canonical document;
* the sequence is a fixed ``seq_len`` window over that document's token ids;
* if the document is longer than ``seq_len`` the window start is staggered
  deterministically by ``hash(document_id, epoch)`` so long documents are seen
  in full across epochs instead of always being truncated at the same offset;
* the window is terminated with EOS when there is room, then padded.

Every packed array is reproducible from
``(shard sha256, tokenizer identity, seq_len, epoch, PACKING_VERSION)``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PACKING_VERSION = "flashmini-v4-pack-doc-v1"
TOKEN_DTYPE = np.int32


def tokenizer_identity(tokenizer_id: str, revision: str | None) -> str:
    """Stable identity string for a tokenizer revision."""
    return f"{tokenizer_id}@{revision or 'main'}"


def tokenizer_key(tokenizer_id: str, revision: str | None) -> str:
    """Short filesystem-safe key derived from the tokenizer identity."""
    raw = tokenizer_identity(tokenizer_id, revision).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def packed_dir(cache_root: Path) -> Path:
    path = Path(cache_root) / "packed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def packed_path(cache_root: Path, shard_sha256: str, tok_key: str, *,
                seq_len: int, epoch: int) -> Path:
    name = (f"{shard_sha256[:32]}-{tok_key}-s{int(seq_len)}"
            f"-e{int(epoch)}-{PACKING_VERSION}.npy")
    return packed_dir(cache_root) / name


def stagger_offset(document_id: str, token_count: int, seq_len: int,
                   epoch: int) -> int:
    """Deterministic window start inside a long document."""
    span = token_count - seq_len
    if span <= 0:
        return 0
    digest = hashlib.sha256(
        f"{PACKING_VERSION}\0{document_id}\0{int(epoch)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (span + 1)


def pack_document(token_ids: list[int], *, seq_len: int, document_id: str,
                  epoch: int, eos_token_id: int,
                  pad_token_id: int) -> np.ndarray:
    """Window one document into a single ``seq_len`` training sequence."""
    ids = list(token_ids)
    if not ids:
        ids = [eos_token_id]
    if len(ids) > seq_len:
        start = stagger_offset(document_id, len(ids), seq_len, epoch)
        ids = ids[start:start + seq_len]
    if len(ids) < seq_len and ids[-1] != eos_token_id:
        ids = ids + [eos_token_id]
    if len(ids) < seq_len:
        ids = ids + [pad_token_id] * (seq_len - len(ids))
    return np.asarray(ids[:seq_len], dtype=TOKEN_DTYPE)


def encode_text(text: str, tokenizer) -> list[int]:
    """Encode without adding tokenizer-specific special tokens.

    EOS is added by :func:`pack_document` so document boundaries stay under
    FlashMini's control rather than the upstream tokenizer's.
    """
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if ids is None:
        ids = getattr(encoded, "input_ids", None)
    if ids is None:
        ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids)


def pack_shard(texts: list[str], document_ids: list[str], *, tokenizer,
               seq_len: int, epoch: int) -> np.ndarray:
    """Pack every document in a shard into a ``(n_docs, seq_len)`` int32 array."""
    eos = int(tokenizer.eos_token_id)
    pad = tokenizer.pad_token_id
    pad = int(pad) if pad is not None else eos
    out = np.empty((len(texts), int(seq_len)), dtype=TOKEN_DTYPE)
    for row, (text, did) in enumerate(zip(texts, document_ids)):
        out[row] = pack_document(
            encode_text(text, tokenizer), seq_len=int(seq_len), document_id=did,
            epoch=int(epoch), eos_token_id=eos, pad_token_id=pad)
    return out


def load_or_build_packed(*, cache_root: Path, shard_sha256: str,
                         texts: list[str], document_ids: list[str],
                         tokenizer, tokenizer_id: str,
                         tokenizer_revision: str | None, seq_len: int,
                         epoch: int) -> np.ndarray:
    """Return the cached packed array for a shard, building it once if absent."""
    key = tokenizer_key(tokenizer_id, tokenizer_revision)
    target = packed_path(cache_root, shard_sha256, key, seq_len=seq_len,
                         epoch=epoch)
    if target.is_file():
        try:
            arr = np.load(target, mmap_mode="r")
            if arr.shape == (len(texts), int(seq_len)):
                return arr
        except (OSError, ValueError):
            pass  # corrupt/incomplete cache entry: rebuild below
    arr = pack_shard(texts, document_ids, tokenizer=tokenizer,
                     seq_len=int(seq_len), epoch=int(epoch))
    # np.save would append ".npy" to a .tmp name, so write through a handle and
    # rename atomically: a killed process never leaves a half-written cache file
    # that later looks valid.
    tmp = target.with_name(target.name + ".part")
    with open(tmp, "wb") as handle:
        np.save(handle, arr, allow_pickle=False)
    tmp.replace(target)
    return arr


def tokenizer_fingerprint(tokenizer) -> str:
    """Best-effort local fingerprint of a loaded tokenizer (for provenance)."""
    parts: dict[str, object] = {
        "class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    vocab = getattr(tokenizer, "get_vocab", None)
    if callable(vocab):
        digest = hashlib.sha256()
        for token, idx in sorted(vocab().items()):
            digest.update(f"{token}\0{idx}\0".encode("utf-8"))
        parts["vocab_sha256"] = digest.hexdigest()
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
