"""Frozen FlashMini-50B v4 tokenizer identity, coverage, and PLE token stability."""

from __future__ import annotations

from pathlib import Path

import pytest

from flashmini.base_init_config import load_config
from flashmini.v4_tokenizer import (
    SPECIAL_TOKEN_ROLES,
    VOCAB_SIZE,
    FrozenTokenizer,
    special_token_ids,
    tokenizer_fingerprint,
)

MANIFEST = Path("training_data/tokenizer/flashmini_50b_v4/tokenizer_manifest.json")


@pytest.fixture(scope="module")
def tokenizer():
    return FrozenTokenizer(MANIFEST)


def test_vocabulary_and_special_ids_match_frozen_config(tokenizer):
    config = load_config()
    section = config.section("tokenizer")
    assert tokenizer.vocab_size == VOCAB_SIZE == 131_072 == config.vocab_size
    assert tokenizer.special_token_ids == special_token_ids() == section["special_token_ids"]
    assert set(SPECIAL_TOKEN_ROLES) <= set(tokenizer.special_token_ids)
    assert tokenizer.fingerprint == section["fingerprint"] == tokenizer_fingerprint(MANIFEST.parent / "tokenizer.json")
    for role, token in SPECIAL_TOKEN_ROLES.items():
        assert tokenizer.token_to_id(token) == tokenizer.special_token_ids[role]
        assert tokenizer.id_to_token(tokenizer.special_token_ids[role]) == token


def test_fingerprint_is_deterministic_and_bound_to_canonical_json(tokenizer):
    first = tokenizer_fingerprint(MANIFEST.parent / "tokenizer.json")
    second = tokenizer_fingerprint(MANIFEST.parent / "tokenizer.json")
    assert first == second == tokenizer.fingerprint
    assert len(first) == 64 and set(first) <= set("0123456789abcdef")


def test_round_trip_and_byte_coverage(tokenizer):
    from tokenizers.pre_tokenizers import ByteLevel

    texts = [
        "Hello, FlashMini.",
        "def add(a, b):\n    return a + b\n",
        "日本語とemoji 😀",
        "\tindented\n\nblank",
        "".join(chr(i) for i in range(1, 128)),
    ]
    for text in texts:
        ids = tokenizer.encode(text)
        assert ids and all(0 <= i < VOCAB_SIZE for i in ids)
        assert tokenizer.decode(ids) == text
    alphabet = ByteLevel.alphabet()
    assert len(alphabet) == 256
    missing = [ch for ch in alphabet if tokenizer.token_to_id(ch) is None]
    assert missing == []


def test_tokenizer_ids_are_stable_for_ple_hashing(tokenizer):
    text = "alpha beta gamma"
    first, second = tokenizer.encode(text), tokenizer.encode(text)
    assert first == second
    assert 0 not in first  # EOS is never inserted by encode()
    assert tokenizer.eos_id == 0
    other = tokenizer.encode("alpha beta delta")
    assert first != other


def test_manifest_matches_architecture_and_training_pipeline_identity():
    import json

    manifest = json.loads(MANIFEST.read_text())
    config = load_config()
    section = config.section("tokenizer")
    assert manifest["family"] == section["family"] == "custom_byte_level_bpe"
    assert manifest["vocab_size"] == section["vocab_size"]
    assert manifest["fingerprint"] == section["fingerprint"]
    assert manifest["identity_status"] == "frozen"
    assert manifest["build"]["training_data"]["recipe"] == "training_data/recipes/flashmini_50b_full_v1.yaml"
    assert manifest["build"]["training_data"]["corpus_repository"] == "mjaso/flashmini-data-v1"
    assert manifest["build"]["training_data"]["verified_shards"]
