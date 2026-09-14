"""A failed replacement must never remove the last recovery checkpoint."""

import pytest
import torch

from flashmini.checkpoint import save_checkpoint


def save(path, step):
    save_checkpoint(path, torch.nn.Linear(2, 2), None, step,
                    {"architecture_version": 2}, keep_latest_only=True)


def test_retention_preserves_unrelated_and_newer_files(tmp_path):
    save(tmp_path / "step_1.pt", 1)
    (tmp_path / "milestone.pt").write_bytes(b"pinned")
    (tmp_path / "step_99.pt").write_bytes(b"newer")
    save(tmp_path / "step_2.pt", 2)
    assert not (tmp_path / "step_1.pt").exists()
    assert torch.load(tmp_path / "step_2.pt", weights_only=False)["step"] == 2
    assert (tmp_path / "milestone.pt").exists()
    assert (tmp_path / "step_99.pt").exists()


@pytest.mark.parametrize("failure", ["save", "load", "replace", "fsync"])
def test_failed_replacement_keeps_previous(tmp_path, monkeypatch, failure):
    import flashmini.checkpoint as checkpoints

    save(tmp_path / "step_1.pt", 1)
    original = (tmp_path / "step_1.pt").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    target = checkpoints.torch if failure in ("save", "load") else checkpoints.os
    monkeypatch.setattr(target, failure, fail)
    with pytest.raises(OSError, match="injected"):
        save(tmp_path / "step_2.pt", 2)
    assert (tmp_path / "step_1.pt").read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
