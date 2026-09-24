"""v4 packed-token data: build from canonical text, verify identity, stream deterministically.

Format ``flashmini-v4-packed-uint32-v1``: each shard is raw little-endian
``uint32`` token IDs (documents followed by EOS, concatenated).  The manifest
binds the shards to the tokenizer fingerprint; a stream refuses any manifest
whose fingerprint differs from the frozen tokenizer.

Stream order: the shards form one token sequence.  Window ``j`` is
``tokens[j*L : j*L + L + 1]`` (inputs ``[:-1]``, labels ``[1:]``).  Optimizer step
``s`` consumes the contiguous windows ``[c, c + S)`` where ``c`` is the cursor and
``S = global_batch_sequences``; window ``c + (m * world + r) * B + b`` is row ``b``
of microbatch ``m`` on rank ``r``.  The logical batch is therefore independent of
how it is split across ranks and microbatches, and the cursor fully determines
resume position.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

FORMAT = "flashmini-v4-packed-uint32-v1"


class DataExhausted(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_packed(token_documents: Iterable[list[int]], output_dir: Path | str, *, tokenizer_fingerprint: str,
                 vocab_size: int, eos_id: int, shard_tokens: int = 1 << 28, source: dict[str, Any] | None = None,
                 name: str = "tokens") -> dict[str, Any]:
    """Write already-tokenized documents (EOS appended here) as packed shards."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    buffer: list[np.ndarray] = []
    buffered = documents = 0

    def flush():
        nonlocal buffer, buffered
        if not buffered:
            return
        path = output_dir / f"{name}-{len(shards):05d}.bin"
        np.concatenate(buffer).astype("<u4").tofile(path)
        shards.append({"path": path.name, "tokens": int(buffered), "sha256": _sha256(path)})
        buffer, buffered = [], 0

    for ids in token_documents:
        array = np.asarray(list(ids) + [eos_id], dtype=np.int64)
        if array.size and (array.min() < 0 or array.max() >= vocab_size):
            raise ValueError("token id outside vocabulary")
        buffer.append(array.astype(np.uint32))
        buffered += array.size
        documents += 1
        if buffered >= shard_tokens:
            flush()
    flush()
    manifest = {
        "schema_version": 1, "format": FORMAT, "tokenizer_fingerprint": tokenizer_fingerprint,
        "vocab_size": int(vocab_size), "eos_id": int(eos_id), "documents": documents,
        "total_tokens": int(sum(item["tokens"] for item in shards)), "shards": shards, "source": source or {},
    }
    (output_dir / "packed_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_from_parquet(parquet_paths: list[Path | str], output_dir: Path | str, tokenizer, *, text_column: str = "text",
                       shard_tokens: int = 1 << 28) -> dict[str, Any]:
    """Tokenize canonical parquet documents with the frozen tokenizer into packed shards."""
    import pyarrow.parquet as pq

    sources = []

    def documents():
        for path in parquet_paths:
            path = Path(path)
            sources.append({"path": str(path), "sha256": _sha256(path)})
            texts = pq.read_table(path, columns=[text_column])[text_column].to_pylist()
            for start in range(0, len(texts), 256):
                yield from tokenizer.encode_batch([text for text in texts[start:start + 256] if text])

    manifest = write_packed(documents(), output_dir, tokenizer_fingerprint=tokenizer.fingerprint,
                            vocab_size=tokenizer.vocab_size, eos_id=tokenizer.eos_id, shard_tokens=shard_tokens)
    manifest["source"] = {"parquet": sources, "text_column": text_column}
    (Path(output_dir) / "packed_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_manifest(path: Path | str, *, expected_fingerprint: str, vocab_size: int, verify_hashes: bool = False) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported packed format {manifest.get('format')!r}")
    if manifest["tokenizer_fingerprint"] != expected_fingerprint:
        raise ValueError(f"{path}: tokenizer fingerprint {manifest['tokenizer_fingerprint']} != frozen {expected_fingerprint}")
    if manifest["vocab_size"] != vocab_size:
        raise ValueError(f"{path}: vocab size {manifest['vocab_size']} != model {vocab_size}")
    for shard in manifest["shards"]:
        shard_path = path.parent / shard["path"]
        if not shard_path.exists() or shard_path.stat().st_size != shard["tokens"] * 4:
            raise ValueError(f"{shard_path}: missing or wrong size")
        if verify_hashes and _sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"{shard_path}: sha256 mismatch")
    manifest["_directory"] = str(path.parent)
    return manifest


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    window_indices: list[int]


class PackedTokenStream:
    def __init__(self, manifest: dict[str, Any], seq_len: int):
        self.manifest, self.seq_len = manifest, int(seq_len)
        directory = Path(manifest["_directory"])
        self._arrays = [np.memmap(directory / shard["path"], dtype="<u4", mode="r") for shard in manifest["shards"]]
        self._starts = np.cumsum([0] + [shard["tokens"] for shard in manifest["shards"]])
        self.total_tokens = int(self._starts[-1])
        self.num_windows = max(0, (self.total_tokens - 1) // self.seq_len)

    def _slice(self, start: int, stop: int) -> np.ndarray:
        parts = []
        shard = int(np.searchsorted(self._starts, start, side="right") - 1)
        while start < stop:
            local = start - self._starts[shard]
            take = min(stop - start, self._starts[shard + 1] - start)
            parts.append(np.asarray(self._arrays[shard][local:local + take]))
            start += take
            shard += 1
        return np.concatenate(parts).astype(np.int64)

    def window(self, index: int) -> np.ndarray:
        if index >= self.num_windows:
            raise DataExhausted(f"window {index} beyond {self.num_windows} available")
        start = index * self.seq_len
        return self._slice(start, start + self.seq_len + 1)

    def microbatch(self, cursor: int, *, micro_index: int, rank: int, world: int, micro_batch: int) -> Batch:
        first = cursor + (micro_index * world + rank) * micro_batch
        indices = list(range(first, first + micro_batch))
        windows = torch.from_numpy(np.stack([self.window(index) for index in indices]))
        return Batch(windows[:, :-1].contiguous(), windows[:, 1:].contiguous(), indices)


__all__ = ["Batch", "DataExhausted", "FORMAT", "PackedTokenStream", "build_from_parquet", "load_manifest", "write_packed"]
