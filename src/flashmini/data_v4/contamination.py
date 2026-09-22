"""Deterministic benchmark exclusion for production pretraining views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .canonical import canonicalize_text, content_hash
from .dedupe import lsh_buckets, minhash_signature

CONTAMINATION_VERSION = "flashmini-v4-benchmark-exclusion-v2"


@dataclass(frozen=True)
class ContaminationResult:
    excluded: bool
    benchmark: str = ""
    method: str = "none"
    threshold: float = 1.0
    score: float = 0.0


class BenchmarkExcluder:
    """Exact and LSH-candidate near-match filter across benchmark sources."""

    def __init__(self, rules: dict | list[str] | None = None, *, threshold: float = 0.9) -> None:
        if isinstance(rules, dict):
            names = rules.get("benchmarks", [])
            configured = rules.get("thresholds", {})
            default_method = rules.get("method", "exact-and-minhash-lsh-v1")
        else:
            names = rules or []
            configured = {}
            default_method = "exact-and-minhash-lsh-v1"
        self.threshold = float(threshold)
        self.default_method = default_method
        self.names = [str(name) for name in names]
        self.thresholds = {str(k): float(v) for k, v in configured.items()}
        self._exact: dict[str, set[str]] = {name: set() for name in self.names}
        self._near: dict[str, dict[str, list[int]]] = {name: {} for name in self.names}
        self._buckets: dict[tuple[str, str], list[str]] = {}
        self.loaded_documents = 0
        self.configured_corpus_files: dict[str, str] = {}
        self.loaded_benchmarks: set[str] = set()
        self.stats: dict[str, dict[str, int]] = {
            name: {"documents_excluded": 0, "tokens_excluded": 0} for name in self.names
        }

    @classmethod
    def from_path(cls, path: str | Path) -> BenchmarkExcluder:
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text()) or {}
        excluder = cls(raw)
        base = config_path.parent
        for name, source in (raw.get("corpus_files") or {}).items():
            excluder.configured_corpus_files[str(name)] = str(source)
            corpus_path = Path(source)
            if not corpus_path.is_absolute():
                corpus_path = base / corpus_path
            if corpus_path.is_file():
                documents = _read_documents(corpus_path)
                excluder.add_benchmark(name, documents)
                excluder.loaded_documents += len(documents)
                if documents:
                    excluder.loaded_benchmarks.add(str(name))
        return excluder

    def add_benchmark(self, name: str, documents) -> None:
        name = str(name)
        if name not in self._exact:
            self.names.append(name)
            self._exact[name] = set()
            self._near[name] = {}
            self.stats[name] = {"documents_excluded": 0, "tokens_excluded": 0}
        for text in documents:
            normalized = canonicalize_text(str(text))
            digest = content_hash(normalized)
            self._exact[name].add(digest)
            signature = minhash_signature(normalized)
            for bucket in lsh_buckets(signature):
                self._near[name].setdefault(bucket, signature)

    def check(self, text: str, *, token_count: int = 0) -> ContaminationResult:
        normalized = canonicalize_text(text)
        digest = content_hash(normalized)
        signature = minhash_signature(normalized)
        for name in self.names:
            if digest in self._exact.get(name, set()):
                self.stats[name]["documents_excluded"] += 1
                self.stats[name]["tokens_excluded"] += int(token_count)
                return ContaminationResult(True, name, "exact_hash", 1.0, 1.0)
            threshold = self.thresholds.get(name, self.threshold)
            for bucket in lsh_buckets(signature):
                other = self._near.get(name, {}).get(bucket)
                if other is None:
                    continue
                score = sum(a == b for a, b in zip(signature, other)) / len(signature)
                if score >= threshold:
                    self.stats[name]["documents_excluded"] += 1
                    self.stats[name]["tokens_excluded"] += int(token_count)
                    return ContaminationResult(True, name, "minhash_lsh", threshold, score)
        return ContaminationResult(False)

    def is_excluded(self, text: str) -> bool:
        return self.check(text).excluded

    def report(self) -> dict:
        missing = sorted(set(self.names) - self.loaded_benchmarks)
        return {
            "version": CONTAMINATION_VERSION,
            "method": self.default_method,
            "corpus_ready": self.loaded_documents > 0,
            "complete": not missing,
            "configured_corpus_files": dict(sorted(self.configured_corpus_files.items())),
            "loaded_benchmarks": sorted(self.loaded_benchmarks),
            "missing_benchmarks": missing,
            "loaded_documents": self.loaded_documents,
            "threshold": self.threshold,
            "benchmarks": json.loads(json.dumps(self.stats, sort_keys=True)),
            "documents_excluded": sum(v["documents_excluded"] for v in self.stats.values()),
            "tokens_excluded": sum(v["tokens_excluded"] for v in self.stats.values()),
        }


def _read_documents(path: Path) -> list[str]:
    """Read a small frozen benchmark export without imposing a benchmark schema."""
    if path.suffix.lower() in {".txt", ".text"}:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        values = json.loads(path.read_text())
    if isinstance(values, dict):
        values = values.get("documents", values.get("data", []))
    out = []
    for value in values or []:
        if isinstance(value, str):
            out.append(value)
            continue
        if isinstance(value, dict):
            fields = ("text", "content", "prompt", "question", "problem", "input")
            text = "\n".join(str(value[field]) for field in fields if value.get(field))
            if text:
                out.append(text)
    return out
