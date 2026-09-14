"""Behavioral boundaries for fresh-data accounting and treatment comparisons."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from flashmini.comparison import validate_ple_pair
from flashmini.config import FlashMiniConfig
from flashmini.experiment import EpochSampler, validate_data_contract
from flashmini.gates import final_go_readiness


class ArrayData:
    seq_len = 4

    def __len__(self):
        return 10

    def get_batch(self, indices):
        a = np.zeros((len(indices), self.seq_len), dtype=np.int64)
        return a, a.copy()


def test_sequence_mismatch_and_budget_guard():
    cfg = SimpleNamespace(architecture_version=3, max_seq_len=4, experiment_mode="screening")
    assert validate_data_contract(ArrayData(), cfg, 4, 40)["actual_seq_len"] == 4
    with pytest.raises(ValueError, match="sequence length"):
        validate_data_contract(ArrayData(), cfg, 2048, 40)
    with pytest.raises(ValueError, match="repeats"):
        validate_data_contract(ArrayData(), cfg, 4, 80)
    result = validate_data_contract(ArrayData(), cfg, 4, 80, allow_repeated=True)
    assert result["non_decisive_repeated_corpus"] and result["implied_corpus_passes"] == 2


def test_sampler_never_repeats_within_epoch_and_resumes_exactly():
    sampler = EpochSampler(103, 17)
    first = np.concatenate([sampler.take(20) for _ in range(6)])
    assert len(first) == len(set(first)) == 103
    continued = sampler.take(20)
    assert np.array_equal(continued, EpochSampler(103, 17, 103).take(20))
    assert not np.array_equal(first, EpochSampler(103, 23).take(103))
    assert np.array_equal(first[40:60], EpochSampler(103, 17, 40).take(20))


def pair():
    cfg = FlashMiniConfig(max_seq_len=4).to_dict()
    cfg["use_ple"] = False
    b = {"architecture_version": 2, "config": cfg, "extra": {
        "tokens_seen": 40, "real_tokens_seen": 40, "data_manifest_sha256": "abc",
        "training": {"seed": 17, "batch_size": 2, "seq_len": 4, "grad_accum": 1,
                     "schedule": {"cosine_decay": True},
                     "dataset": {"tokenizer": "gpt2", "tokenizer_revision": "tok",
                                 "dataset_revision": "data"},
                     "run_metadata": {"source_sha256": "source", "shared_optimizer": {"lr": 0.001},
                                      "data_contract": {"sampling": "permutation"}}}}}
    c = copy.deepcopy(b)
    c["config"]["use_ple"] = True
    return b, c


def test_matched_pair_and_ablation_labels():
    b, c = pair()
    result = validate_ple_pair(b, c, "abc")
    assert result["comparison_type"] == "B_vs_C_PLE_treatment"
    assert "not_B_baseline" in result["ablation_type"]
    assert not result["final_go_eligible"]


@pytest.mark.parametrize("mutation", ["baseline", "version", "backbone", "dataset", "sequence", "tokens", "seed", "schedule", "optimizer", "source"])
def test_invalid_comparison_rejected(mutation):
    b, c = pair()
    if mutation == "baseline": b["config"]["use_ple"] = True
    if mutation == "version": c["architecture_version"] = 3
    if mutation == "backbone": c["config"]["d_model"] += 1
    if mutation == "dataset": c["extra"]["data_manifest_sha256"] = "different"
    if mutation == "sequence": c["extra"]["training"]["seq_len"] = 2048
    if mutation == "tokens": c["extra"]["tokens_seen"] = 48
    if mutation == "seed": c["extra"]["training"]["seed"] = 23
    if mutation == "schedule": c["extra"]["training"]["schedule"] = {}
    if mutation == "optimizer": c["extra"]["training"]["run_metadata"]["shared_optimizer"] = {}
    if mutation == "source": c["extra"]["training"]["run_metadata"]["source_sha256"] = "changed"
    with pytest.raises(ValueError):
        validate_ple_pair(b, c, "abc")


def test_final_go_requires_long_context_and_seed_confirmation():
    result = final_go_readiness(matched_a_control=True, fresh_corpus=True, scale_confirmation=True)
    assert result["status"] == "BLOCKED"
    assert "genuine_long_context_quality_evaluation" in result["missing_evidence"]
    assert "three_matched_seeds_for_small_effect" in result["missing_evidence"]
