"""FlashMini-50B v4 tokenizer: deterministic build, freeze, and identity checks.

The v4 tokenizer is a byte-level BPE with exactly 131072 slots:

* IDs ``0..255`` are special tokens (named control tokens followed by reserved
  slots, so post-training can add roles without changing the frozen vocabulary);
* the next 256 IDs are the byte alphabet (complete byte coverage, no UNK);
* the remaining 130560 IDs are learned merges.

The training corpus is the approved FlashMini corpus-v1 pilot release pinned by
``CORPUS_REPOSITORY``/``CORPUS_REVISION``.  Only shards whose bytes match the
release manifest SHA256 are used; the sample policy (domain weights, per-doc
character cap, bounded repetition) is recorded in the tokenizer manifest.  The
fingerprint is the SHA256 of the canonical JSON of ``tokenizer.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

VOCAB_SIZE = 131_072
SPECIAL_SLOT_COUNT = 256
BYTE_ALPHABET_SIZE = 256
TOKENIZER_FAMILY = "custom_byte_level_bpe"
TOKENIZER_NAME = "flashmini-50b-v4-bpe131072"
FINGERPRINT_ALGORITHM = "sha256_canonical_tokenizer_json_v1"
DEFAULT_TOKENIZER_DIR = Path("training_data/tokenizer/flashmini_50b_v4")
CORPUS_REPOSITORY = "mjaso/flashmini-data-v1"
CORPUS_REVISION = "e83398462169164d9e4127627ad4f72d95b05a41"
RECIPE_PATH = Path("training_data/recipes/flashmini_50b_full_v1.yaml")

NAMED_SPECIAL_TOKENS = (
    "<|endoftext|>", "<|pad|>", "<|bos|>", "<|im_start|>", "<|im_end|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<|think_start|>", "<|think_end|>",
    "<|tool_call_start|>", "<|tool_call_end|>", "<|tool_response_start|>",
    "<|tool_response_end|>",
)
SPECIAL_TOKEN_ROLES = {
    "eos": "<|endoftext|>", "pad": "<|pad|>", "bos": "<|bos|>",
    "im_start": "<|im_start|>", "im_end": "<|im_end|>",
    "fim_prefix": "<|fim_prefix|>", "fim_middle": "<|fim_middle|>",
    "fim_suffix": "<|fim_suffix|>", "fim_pad": "<|fim_pad|>",
    "repo_name": "<|repo_name|>", "file_sep": "<|file_sep|>",
    "think_start": "<|think_start|>", "think_end": "<|think_end|>",
    "tool_call_start": "<|tool_call_start|>", "tool_call_end": "<|tool_call_end|>",
    "tool_response_start": "<|tool_response_start|>", "tool_response_end": "<|tool_response_end|>",
}
# cl100k-style split: contractions, letter runs, 1-3 digit groups, punctuation
# runs, and whitespace handling that keeps code indentation as stable tokens.
PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SAMPLE_POLICY = {
    "version": "flashmini-v4-tokenizer-sample-v1",
    "target_total_characters": 500_000_000,
    "max_characters_per_document": 20_000,
    "max_repeats_per_domain": 3,
    "document_order": "manifest_shard_order_then_row_order",
    "domain_merge": {"multilingual": "general_web", "tech_docs": "books_reference"},
    "shard_filter": "split_train_and_local_sha256_equals_release_manifest",
}


def special_tokens() -> list[str]:
    reserved = [f"<|reserved_special_{index:03d}|>" for index in range(SPECIAL_SLOT_COUNT - len(NAMED_SPECIAL_TOKENS))]
    return list(NAMED_SPECIAL_TOKENS) + reserved


def special_token_ids() -> dict[str, int]:
    ordered = special_tokens()
    return {role: ordered.index(token) for role, token in SPECIAL_TOKEN_ROLES.items()}


def canonical_tokenizer_json(raw: str | bytes) -> bytes:
    return json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def tokenizer_fingerprint(tokenizer_json_path: Path | str) -> str:
    return hashlib.sha256(canonical_tokenizer_json(Path(tokenizer_json_path).read_bytes())).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recipe_domain_weights(recipe_path: Path) -> dict[str, float]:
    import yaml

    recipe = yaml.safe_load(recipe_path.read_text())
    weights: dict[str, float] = {}
    for domain, spec in recipe["domains"].items():
        target = SAMPLE_POLICY["domain_merge"].get(domain, domain)
        weights[target] = weights.get(target, 0.0) + float(spec["weight"])
    return weights


def verified_shards(corpus_dir: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (verified, rejected) shard records in release-manifest order."""
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / "manifests" / "corpus_manifest.json").read_text())
    verified, rejected = [], []
    for shard in manifest["shards"]:
        path = corpus_dir / "shards" / shard["path"]
        record = {"path": shard["path"], "expected_sha256": shard["sha256"], "split": shard["split"]}
        if shard["split"] != "train":
            rejected.append({**record, "reason": "not_train_split"})
        elif not path.exists():
            rejected.append({**record, "reason": "missing_from_release_revision"})
        elif _sha256(path) != shard["sha256"]:
            rejected.append({**record, "reason": "sha256_mismatch"})
        else:
            verified.append(record)
    return verified, rejected


def sample_documents(corpus_dir: Path | str, *, recipe_path: Path | str = RECIPE_PATH) -> tuple[list[str], dict[str, Any]]:
    """Deterministically select training text by recipe domain budgets."""
    import pyarrow.parquet as pq

    corpus_dir = Path(corpus_dir)
    verified, rejected = verified_shards(corpus_dir)
    weights = _recipe_domain_weights(Path(recipe_path))
    total = float(sum(weights.values()))
    budgets = {domain: int(SAMPLE_POLICY["target_total_characters"] * weight / total) for domain, weight in weights.items()}
    cap = SAMPLE_POLICY["max_characters_per_document"]
    pools: dict[str, list[str]] = {domain: [] for domain in budgets}
    for shard in verified:
        table = pq.read_table(corpus_dir / "shards" / shard["path"], columns=["text", "domain"])
        for text, domain in zip(table["text"].to_pylist(), table["domain"].to_pylist()):
            if domain in pools and text:
                pools[domain].append(text[:cap])
    selected: list[str] = []
    usage: dict[str, Any] = {}
    for domain in sorted(budgets):
        budget, used, repeats, documents = budgets[domain], 0, 0, 0
        while used < budget and pools[domain] and repeats < SAMPLE_POLICY["max_repeats_per_domain"]:
            repeats += 1
            for text in pools[domain]:
                if used >= budget:
                    break
                selected.append(text)
                used += len(text)
                documents += 1
        usage[domain] = {"budget_characters": budget, "selected_characters": used, "selected_documents": documents,
                         "passes": repeats, "unique_documents_available": len(pools[domain])}
    provenance = {
        "corpus_repository": CORPUS_REPOSITORY,
        "corpus_revision": CORPUS_REVISION,
        "recipe": str(recipe_path),
        "recipe_domain_weights": weights,
        "sample_policy": SAMPLE_POLICY,
        "verified_shards": [{"path": item["path"], "sha256": item["expected_sha256"]} for item in verified],
        "rejected_shards": rejected,
        "domain_usage": usage,
        "selected_documents": len(selected),
        "selected_characters": sum(len(text) for text in selected),
        "selected_text_sha256": _text_digest(selected),
    }
    return selected, provenance


def _text_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def build_tokenizer(texts: list[str]):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(iter(texts), trainer=trainer, length=len(texts))
    return tokenizer


def train(corpus_dir: Path | str, output_dir: Path | str = DEFAULT_TOKENIZER_DIR) -> dict[str, Any]:
    import tokenizers

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts, provenance = sample_documents(corpus_dir)
    tokenizer = build_tokenizer(texts)
    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB_SIZE:
        raise RuntimeError(f"trained vocabulary has {tokenizer.get_vocab_size()} slots, expected {VOCAB_SIZE}")
    path = output_dir / "tokenizer.json"
    path.write_bytes(canonical_tokenizer_json(tokenizer.to_str()))
    manifest = {
        "schema_version": 1,
        "name": TOKENIZER_NAME,
        "family": TOKENIZER_FAMILY,
        "vocab_size": VOCAB_SIZE,
        "special_token_count": SPECIAL_SLOT_COUNT,
        "special_token_ids": special_token_ids(),
        "special_tokens_range": [0, SPECIAL_SLOT_COUNT - 1],
        "byte_alphabet_size": BYTE_ALPHABET_SIZE,
        "pretokenize_pattern": PRETOKENIZE_PATTERN,
        "normalization": "none",
        "artifact": "tokenizer.json",
        "artifact_sha256": _sha256(path),
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "fingerprint": tokenizer_fingerprint(path),
        "token_dtype": "uint32",
        "identity_status": "frozen",
        "immutable_after_optimizer_step": 1,
        "build": {
            "tokenizers_version": tokenizers.__version__,
            "trainer": {"model": "BPE", "min_frequency": 2, "initial_alphabet": "ByteLevel.alphabet()"},
            "training_data": provenance,
        },
    }
    (output_dir / "tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class FrozenTokenizer:
    """Loaded v4 tokenizer whose identity has been verified against its manifest."""

    def __init__(self, manifest_path: Path | str = DEFAULT_TOKENIZER_DIR / "tokenizer_manifest.json",
                 *, expected_fingerprint: str | None = None):
        from tokenizers import Tokenizer

        manifest_path = Path(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        artifact = manifest_path.parent / self.manifest["artifact"]
        fingerprint = tokenizer_fingerprint(artifact)
        if fingerprint != self.manifest["fingerprint"]:
            raise ValueError(f"tokenizer artifact fingerprint {fingerprint} != manifest {self.manifest['fingerprint']}")
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(f"tokenizer fingerprint {fingerprint} != required {expected_fingerprint}")
        self.fingerprint = fingerprint
        self._tokenizer = Tokenizer.from_file(str(artifact))
        if self._tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB_SIZE:
            raise ValueError("tokenizer vocabulary is not exactly 131072 slots")
        self.special_token_ids = dict(self.manifest["special_token_ids"])
        for role, token in SPECIAL_TOKEN_ROLES.items():
            if self._tokenizer.token_to_id(token) != self.special_token_ids.get(role):
                raise ValueError(f"special token {role} id mismatch")
        self.eos_id = self.special_token_ids["eos"]
        self.vocab_size = VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [item.ids for item in self._tokenizer.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, index: int) -> str | None:
        return self._tokenizer.id_to_token(index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flashmini.v4_tokenizer")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("train", help="train the frozen v4 tokenizer from the pinned corpus release")
    build.add_argument("--corpus-dir", required=True)
    build.add_argument("--output-dir", default=str(DEFAULT_TOKENIZER_DIR))
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", default=str(DEFAULT_TOKENIZER_DIR / "tokenizer_manifest.json"))
    args = parser.parse_args(argv)
    if args.command == "train":
        manifest = train(args.corpus_dir, args.output_dir)
        print(json.dumps({"fingerprint": manifest["fingerprint"], "vocab_size": manifest["vocab_size"]}))
    else:
        tokenizer = FrozenTokenizer(args.manifest)
        print(json.dumps({"fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size, "eos_id": tokenizer.eos_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
