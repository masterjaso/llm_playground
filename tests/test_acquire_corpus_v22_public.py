from scripts.acquire_corpus_v22_public import _bucket, _codeagent_tier, _oasst_domain, _oasst_tier


def test_group_tiers_are_deterministic_and_supported() -> None:
    assert _codeagent_tier("task-1") == _codeagent_tier("task-1")
    assert _codeagent_tier("task-1") in {"FIT-TRAIN", "FIT-DEV"}
    assert _oasst_tier("tree-1") in {"FIT-TRAIN", "FIT-DEV", "GATE-A", "SHADOW-B", "SHADOW-C"}


def test_bucket_requires_complete_weights() -> None:
    assert _bucket("group", (("A", 10_000),)) == "A"


def test_oasst_domains_are_content_based() -> None:
    assert _oasst_domain("Return this object as JSON")[0] == "structured"
    assert _oasst_domain("Write a Python function")[0] == "code"
    assert _oasst_domain("Explain this software API")[0] == "software-engineering-natural-language"
    assert _oasst_domain("Tell me about Roman history")[0] == "general"
