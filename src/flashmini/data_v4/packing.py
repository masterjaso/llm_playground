"""Tokenizer-specific representations and deterministic sequence packing.

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
import os
from pathlib import Path

import numpy as np

PACKING_VERSION = "flashmini-v4-pack-doc-v1"
PRODUCTION_PACKING_VERSION = "flashmini-v4-pack-contiguous-v2"
TOKEN_FORMAT_VERSION = "flashmini-token-format-v1"
TOKEN_DTYPE = np.uint16


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
        f"{PACKING_VERSION}\0{document_id}\0{int(epoch)}".encode()).digest()
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
    """Legacy document-window packing retained for v4 compatibility tests."""
    eos = int(tokenizer.eos_token_id)
    pad = tokenizer.pad_token_id
    pad = int(pad) if pad is not None else eos
    dtype = np.uint16 if int(getattr(tokenizer, "vocab_size", 65536)) <= 65536 else np.uint32
    out = np.empty((len(texts), int(seq_len)), dtype=dtype)
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
            digest.update(f"{token}\0{idx}\0".encode())
        parts["vocab_sha256"] = digest.hexdigest()
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def token_dtype_for_tokenizer(tokenizer):
    """Return the smallest safe NumPy unsigned dtype for a tokenizer."""
    return np.uint16 if int(getattr(tokenizer, "vocab_size", 0)) <= 65536 else np.uint32


def tokenize_documents(texts: list[str], document_ids: list[str], *, tokenizer,
                       append_eos: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize all documents into one contiguous array plus offset metadata.

    Returns ``(tokens, offsets, lengths)`` where ``offsets`` has ``N+1`` entries.
    No sequence-length decision is made here, so the same representation can
    serve 2K, 8K, or long-context views.
    """
    if len(texts) != len(document_ids):
        raise ValueError("texts and document_ids must have the same length")
    eos = int(getattr(tokenizer, "eos_token_id", -1))
    if append_eos and eos < 0:
        raise ValueError("tokenizer must define eos_token_id")
    dtype = token_dtype_for_tokenizer(tokenizer)
    arrays: list[np.ndarray] = []
    offsets = [0]
    lengths: list[int] = []
    for text in texts:
        ids = encode_text(text, tokenizer)
        if append_eos:
            ids.append(eos)
        arr = np.asarray(ids, dtype=dtype)
        arrays.append(arr)
        lengths.append(int(arr.size))
        offsets.append(offsets[-1] + int(arr.size))
    tokens = np.concatenate(arrays) if arrays else np.empty((0,), dtype=dtype)
    return tokens, np.asarray(offsets, dtype=np.int64), np.asarray(lengths, dtype=np.int64)


def pack_token_stream(tokens: np.ndarray, offsets: np.ndarray, *, seq_len: int,
                      eos_token_id: int, pad_token_id: int,
                      policy: str = "document_mix") -> tuple[np.ndarray, list[dict]]:
    """Pack every token from a contiguous document store into fixed sequences.

    Short documents are combined with EOS boundaries.  Long documents are
    split into contiguous windows, so no accepted token is discarded merely
    because it exceeds the training sequence length.  The returned metadata
    records document IDs only at the caller boundary; ``doc_index`` and token
    spans are enough to reconstruct provenance.
    """
    if int(seq_len) <= 0:
        raise ValueError("seq_len must be positive")
    if policy not in {"document_mix", "coherent"}:
        raise ValueError("unknown packing policy")
    tokens = np.asarray(tokens)
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0:
        raise ValueError("offsets must be a one-dimensional array beginning at zero")
    if int(offsets[-1]) != int(tokens.size):
        raise ValueError("offsets[-1] must equal the token count")
    dtype = tokens.dtype if tokens.dtype in (np.dtype(np.uint16), np.dtype(np.uint32)) else np.dtype(np.uint32)
    rows: list[np.ndarray] = []
    metadata: list[dict] = []
    current: list[int] = []
    current_docs: list[int] = []

    def flush() -> None:
        nonlocal current, current_docs
        if not current:
            return
        used = len(current)
        row = current[:int(seq_len)] + [int(pad_token_id)] * (int(seq_len) - used)
        rows.append(np.asarray(row, dtype=dtype))
        metadata.append({"document_indices": list(current_docs), "content_tokens": used})
        current, current_docs = [], []

    for doc_index in range(len(offsets) - 1):
        start, end = int(offsets[doc_index]), int(offsets[doc_index + 1])
        doc = [int(x) for x in tokens[start:end]]
        if not doc:
            continue
        if len(doc) > int(seq_len):
            flush()
            for pos in range(0, len(doc), int(seq_len)):
                chunk = doc[pos:pos + int(seq_len)]
                # EOS is already part of the contiguous representation; only
                # add one when a chunk has room for a boundary marker.
                if len(chunk) < int(seq_len) and chunk[-1] != int(eos_token_id):
                    chunk.append(int(eos_token_id))
                used = len(chunk)
                row = chunk + [int(pad_token_id)] * (int(seq_len) - used)
                rows.append(np.asarray(row, dtype=dtype))
                metadata.append({"document_indices": [doc_index], "content_tokens": used})
            continue
        if policy == "coherent" and current:
            flush()
        boundary = 1 if current and current[-1] != int(eos_token_id) else 0
        needed = len(doc) + boundary
        if current and len(current) + needed > int(seq_len):
            flush()
        if current and current[-1] != int(eos_token_id):
            current.append(int(eos_token_id))
        current.extend(doc)
        current_docs.append(doc_index)
        if policy == "coherent" or len(current) == int(seq_len):
            flush()
    flush()
    packed = np.stack(rows) if rows else np.empty((0, int(seq_len)), dtype=dtype)
    return packed, metadata


def pack_documents(texts: list[str], document_ids: list[str], *, tokenizer,
                   seq_len: int, policy: str = "document_mix",
                   return_metadata: bool = False):
    """Tokenize and pack documents while consuming the full token stream."""
    tokens, offsets, _lengths = tokenize_documents(
        texts, document_ids, tokenizer=tokenizer, append_eos=True)
    eos = int(tokenizer.eos_token_id)
    pad = tokenizer.pad_token_id
    pad = eos if pad is None else int(pad)
    packed, metadata = pack_token_stream(
        tokens, offsets, seq_len=seq_len, eos_token_id=eos,
        pad_token_id=pad, policy=policy)
    return (packed, metadata) if return_metadata else packed


def write_token_store(tokens: np.ndarray, offsets: np.ndarray, prefix: Path, *,
                      tokenizer_id: str, tokenizer_revision: str,
                      tokenizer_fingerprint: str, document_ids: list[str] | None = None) -> dict:
    """Atomically write a mmap-friendly flat token store and sidecar index."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(tokens)
    if tokens.dtype not in (np.dtype(np.uint16), np.dtype(np.uint32)):
        raise ValueError("token store dtype must be uint16 or uint32")
    bin_path = prefix.with_suffix(".bin")
    idx_path = prefix.with_suffix(".idx.json")
    bin_tmp = bin_path.with_suffix(bin_path.suffix + ".part")
    idx_tmp = idx_path.with_suffix(idx_path.suffix + ".part")
    tokens.tofile(bin_tmp)
    os.replace(bin_tmp, bin_path)
    payload = {
        "format": TOKEN_FORMAT_VERSION,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "dtype": str(tokens.dtype),
        "token_count": int(tokens.size),
        "document_count": max(0, len(offsets) - 1),
        "offsets": [int(x) for x in np.asarray(offsets, dtype=np.int64)],
        "document_ids": list(document_ids or []),
        "sha256": hashlib.sha256(bin_path.read_bytes()).hexdigest(),
    }
    idx_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(idx_tmp, idx_path)
    return payload | {"bin_path": str(bin_path), "index_path": str(idx_path)}


def load_token_store(prefix: Path):
    """Open a previously written store as a read-only memory map."""
    prefix = Path(prefix)
    meta = json.loads(prefix.with_suffix(".idx.json").read_text())
    dtype = np.dtype(meta["dtype"])
    if dtype not in (np.dtype(np.uint16), np.dtype(np.uint32)):
        raise ValueError("unsupported token store dtype")
    tokens = np.memmap(prefix.with_suffix(".bin"), dtype=dtype, mode="r",
                       shape=(int(meta["token_count"]),))
    offsets = np.asarray(meta["offsets"], dtype=np.int64)
    return tokens, offsets, meta
