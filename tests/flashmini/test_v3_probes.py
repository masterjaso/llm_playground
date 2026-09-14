"""Structural context probe does not confer a long-context capability claim."""

import torch

from flashmini.config import FlashMiniConfig, GatedDeltaNetConfig, MoEConfig, PLEConfig
from flashmini.context_probe import run_context_probe


def test_longer_context_structural_probe():
    config = FlashMiniConfig(architecture_version=3, vocab_size=32, d_model=16,
        num_layers=4, num_heads=2, head_dim=8, max_seq_len=256, use_ple=True,
        moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=16),
        gdn=GatedDeltaNetConfig(d_state=4, chunk_size=16),
        ple=PLEConfig(ngram_vocab_size_base=37, heads_per_ngram=2, embed_dim=16, eos_id=31))
    report = run_context_probe(config, seq_len=1024, device="cpu", seed=17)
    assert report["status"] == "PASS"
    assert report["requested_seq_len"] == 1024
    assert report["structural_only"] and not report["long_context_capability_claim"]


def test_eval_restores_rng_and_is_deterministic():
    import numpy as np

    from flashmini.eval import compute_validation_nll
    from flashmini.models import FlashMiniModel
    config = FlashMiniConfig(vocab_size=16, d_model=8, num_heads=1, head_dim=8,
        num_layers=1, max_seq_len=4, moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=8))
    class Data:
        def __len__(self): return 1
        def get_batch(self, indices):
            return np.array([[1, 2, 3, 4]]), np.array([[2, 3, 4, 5]])
    model = FlashMiniModel(config)
    before = torch.get_rng_state().clone()
    first = compute_validation_nll(model, Data(), torch.device("cpu"))
    assert torch.equal(before, torch.get_rng_state())
    assert first == compute_validation_nll(model, Data(), torch.device("cpu"))


def test_evaluation_selects_exact_holdout_suffix():
    import numpy as np

    from flashmini.compare_ple import _Slice
    from flashmini.eval import compute_validation_nll
    from flashmini.models import FlashMiniModel
    config = FlashMiniConfig(vocab_size=16, d_model=8, num_heads=1, head_dim=8,
        num_layers=1, max_seq_len=4, moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=8))
    class Data:
        def __len__(self): return 5
        def get_batch(self, indices):
            x = np.array([[int(i), 2, 3, 4] for i in indices])
            return x, (x + 1) % 16
    model = FlashMiniModel(config)
    expected = compute_validation_nll(model, _Slice(Data(), 2, 4), torch.device("cpu"))
    actual = compute_validation_nll(model, Data(), torch.device("cpu"),
                                    start_sequence=2, max_batches=2)
    assert actual == expected
