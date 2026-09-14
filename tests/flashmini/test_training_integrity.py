"""Focused integrity checks for deterministic FlashMini training/evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from flashmini.checkpoint import load_checkpoint, save_checkpoint
from flashmini.config import FlashMiniConfig
from flashmini.eval import compute_validation_nll
from flashmini.training import train, train_step


class _ArrayDataset:
    def __init__(self, *, rows: int = 8, seq_len: int = 4, vocab_size: int = 7):
        self.inputs = np.arange(rows * seq_len, dtype=np.int64).reshape(rows, seq_len) % vocab_size
        self.labels = np.roll(self.inputs, -1, axis=1)
        self.labels[:, -1] = -100

    def __len__(self):
        return len(self.inputs)

    def get_batch(self, indices):
        return self.inputs[indices], self.labels[indices]


class _ToyLM(nn.Module):
    def __init__(self, config: FlashMiniConfig, *, with_ple: bool = False):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        # The training loop only needs this marker to request the same-state
        # ablation; the toy model keeps its forward contract explicit below.
        self.ple = object() if with_ple else None

    def forward(self, input_ids, labels=None, ple_enabled=None):
        logits = self.head(self.embedding(input_ids))
        if self.ple is not None and ple_enabled is False:
            logits = logits * 0.5
        result = {"logits": logits, "stats": {}}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return result


class _FixedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Linear(1, 1)
        self.register_buffer(
            "fixed_logits",
            torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                    [4.0, 0.0, 0.0],
                ]
            ),
        )

    def forward(self, input_ids, labels=None, ple_enabled=None):
        del labels, ple_enabled
        logits = self.fixed_logits.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        return {"logits": logits, "stats": {}}


def _config() -> FlashMiniConfig:
    return FlashMiniConfig(
        vocab_size=7,
        d_model=8,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        max_seq_len=4,
        use_ple=False,
    )


def _model_and_optimizer() -> tuple[FlashMiniConfig, _ToyLM, torch.optim.Optimizer]:
    config = _config()
    model = _ToyLM(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    return config, model, optimizer


class TrainingIntegrityTests(unittest.TestCase):
    def test_cpu_resume_matches_uninterrupted_sampling_and_weights(self):
        dataset = _ArrayDataset()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(123)
            config_full, full_model, full_optimizer = _model_and_optimizer()
            full = train(
                full_model,
                full_optimizer,
                dataset,
                config_full,
                root / "full",
                total_tokens=32,
                seq_len=4,
                device=torch.device("cpu"),
                batch_size=2,
                log_every=1,
                ckpt_every_tokens=16,
                seed=19,
                use_amp=False,
            )

            torch.manual_seed(123)
            config_part, part_model, part_optimizer = _model_and_optimizer()
            train(
                part_model,
                part_optimizer,
                dataset,
                config_part,
                root / "resumed",
                total_tokens=16,
                seq_len=4,
                device=torch.device("cpu"),
                batch_size=2,
                log_every=1,
                ckpt_every_tokens=16,
                seed=19,
                use_amp=False,
            )

            config_resume, resume_model, resume_optimizer = _model_and_optimizer()
            resumed = train(
                resume_model,
                resume_optimizer,
                dataset,
                config_resume,
                root / "resumed",
                total_tokens=32,
                seq_len=4,
                device=torch.device("cpu"),
                batch_size=2,
                log_every=1,
                ckpt_every_tokens=16,
                resume_from=root / "resumed" / "checkpoints" / "step_2.pt",
                seed=19,
                use_amp=False,
            )

            self.assertEqual(full["steps"], resumed["steps"])
            self.assertEqual(full["tokens_seen"], resumed["tokens_seen"])
            for expected, actual in zip(full_model.parameters(), resume_model.parameters()):
                self.assertTrue(torch.equal(expected, actual))

            checkpoint = torch.load(
                root / "resumed" / "checkpoints" / "step_4.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertIn("rng_state", checkpoint["extra"])
            self.assertEqual(checkpoint["extra"]["training"]["seed"], 19)

    def test_validation_masks_ignore_index_and_restores_mixed_modes(self):
        class EvalDataset:
            def __len__(self):
                return 1

            def get_batch(self, indices):
                del indices
                return np.zeros((1, 4), dtype=np.int64), np.array([[0, -100, 2, 0]], dtype=np.int64)

        model = _FixedLM()
        model.train()
        model.child.eval()
        result = compute_validation_nll(model, EvalDataset(), torch.device("cpu"))

        expected_nll = F.cross_entropy(
            model.fixed_logits[[0, 2, 3]],
            torch.tensor([0, 2, 0]),
            reduction="mean",
        ).item()
        self.assertAlmostEqual(result["nll"], expected_nll)
        self.assertEqual(result["tokens"], 3)
        self.assertEqual(result["correct_tokens"], 3)
        self.assertEqual(result["top1_accuracy"], 1.0)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)

        model.eval()
        compute_validation_nll(model, EvalDataset(), torch.device("cpu"))
        self.assertFalse(model.training)

    def test_empty_validation_is_rejected(self):
        class EmptyDataset:
            def __len__(self):
                return 0

        with self.assertRaisesRegex(ValueError, "empty"):
            compute_validation_nll(_FixedLM(), EmptyDataset(), torch.device("cpu"))

    def test_checkpoint_architecture_mismatch_is_rejected(self):
        config, model, optimizer = _model_and_optimizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, step=1, config=config)
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["architecture_version"] = 1
            mismatch = Path(tmp) / "mismatch.pt"
            torch.save(state, mismatch)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_checkpoint(mismatch, _model_and_optimizer()[1])

            del state["architecture_version"]
            legacy = Path(tmp) / "legacy.pt"
            torch.save(state, legacy)
            with self.assertRaisesRegex(ValueError, "architecture_version"):
                load_checkpoint(legacy, _model_and_optimizer()[1])

    def test_accumulation_and_nonfinite_loss_are_explicit(self):
        dataset = _ArrayDataset()
        config, model, optimizer = _model_and_optimizer()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "grad_accum"):
            train(
                model,
                optimizer,
                dataset,
                config,
                Path(tmp) / "run",
                total_tokens=8,
                seq_len=4,
                device=torch.device("cpu"),
                batch_size=1,
                grad_accum=2,
            )

        class NaNModel(_ToyLM):
            def forward(self, input_ids, labels=None, ple_enabled=None):
                del input_ids, labels, ple_enabled
                return {"loss": torch.tensor(float("nan"), requires_grad=True), "stats": {}}

        nan_config, _, nan_optimizer = _model_and_optimizer()
        del nan_config
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            train_step(
                NaNModel(_config()),
                nan_optimizer,
                torch.zeros((1, 4), dtype=torch.long),
                torch.zeros((1, 4), dtype=torch.long),
                use_amp=False,
            )

    def test_periodic_validation_summary_and_lr_groups(self):
        dataset = _ArrayDataset()
        config = _config()
        model = _ToyLM(config, with_ple=True)
        optimizer = torch.optim.SGD(
            [
                {"params": model.embedding.parameters(), "lr": 0.05, "name": "backbone"},
                {"params": model.head.parameters(), "lr": 0.1, "name": "head"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary = train(
                model,
                optimizer,
                dataset,
                config,
                run_dir,
                total_tokens=16,
                seq_len=4,
                device=torch.device("cpu"),
                batch_size=2,
                log_every=1,
                ckpt_every_tokens=8,
                val_dataset=dataset,
                eval_every_tokens=8,
                val_max_batches=1,
                warmup_tokens=4,
                cosine_decay=True,
                min_lr_ratio=0.2,
                seed=3,
                use_amp=False,
            )
            self.assertEqual(len(summary["evaluations"]), 2)
            persisted = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(len(persisted["evaluations"]), 2)
            rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            validation_rows = [row for row in rows if row.get("event") == "validation"]
            self.assertEqual(len(validation_rows), 2)
            self.assertIn("ple_off", validation_rows[0]["validation"])
            train_rows = [row for row in rows if row.get("event") == "train"]
            self.assertIn("backbone", train_rows[0]["lr_groups"])
            self.assertIn("head", train_rows[0]["lr_groups"])
            self.assertEqual(summary["real_tokens_seen"], 12)


if __name__ == "__main__":
    unittest.main()
