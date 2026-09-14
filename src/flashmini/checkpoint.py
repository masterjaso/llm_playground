"""Checkpoint / resume — atomic save/load of model + optimizer + step + config.

Preserves reproducibility metadata (config, hashes) alongside weights.
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch

from .config import FlashMiniConfig

# Checkpoints written by the current FlashMini model must identify the model
# architecture they belong to.  Keeping this value in the checkpoint envelope
# (as well as in the serialized config) lets us reject old or mismatched files
# before touching model or optimizer state.
ARCHITECTURE_VERSION = 2


def _config_dict(config: FlashMiniConfig | dict) -> dict:
    """Return a plain config mapping without requiring a specific dataclass."""
    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if to_dict is not None:
        return dict(to_dict())
    values = getattr(config, "__dict__", None)
    if isinstance(values, dict):
        return dict(values)
    raise TypeError("config must be a mapping or expose to_dict()")


def _architecture_version(config: FlashMiniConfig | dict) -> int:
    values = _config_dict(config)
    value = values.get("architecture_version", ARCHITECTURE_VERSION)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid architecture_version in config: {value!r}") from exc


def _model_architecture_version(model: torch.nn.Module) -> int:
    model_config = getattr(model, "config", None)
    if model_config is None:
        return ARCHITECTURE_VERSION
    return _architecture_version(model_config)


def _config_mismatch(saved: Any, current: Any, path: tuple[str, ...] = ()) -> str | None:
    """Find the first semantic config difference, allowing storage offload only."""
    if path == ("ple", "offload"):
        return None
    if isinstance(saved, dict) and isinstance(current, dict):
        if set(saved) != set(current):
            missing = sorted(set(saved) ^ set(current))
            return ".".join(path + (missing[0],)) if missing else ".".join(path)
        for key in sorted(saved):
            mismatch = _config_mismatch(saved[key], current[key], path + (str(key),))
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(saved) != len(current):
            return ".".join(path)
        for index, (saved_value, current_value) in enumerate(zip(saved, current)):
            mismatch = _config_mismatch(saved_value, current_value, path + (str(index),))
            if mismatch is not None:
                return mismatch
        return None
    return None if saved == current else ".".join(path)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: FlashMiniConfig,
    extra: dict | None = None,
    *,
    keep_latest_only: bool = False,
) -> None:
    """Atomically save a checkpoint (write temp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_values = _config_dict(config)
    architecture_version = _architecture_version(config_values)
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config_values,
        "architecture_version": architecture_version,
        "extra": extra or {},
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, tmp)
        # Read back before retiring the only recovery point. Do not reduce
        # optimizer or RNG precision: these are required for exact continuation.
        verified = torch.load(tmp, map_location="cpu", weights_only=False)
        if verified["step"] != step or verified["architecture_version"] != architecture_version:
            raise ValueError("checkpoint read-back metadata mismatch")
        del verified
        with open(tmp, "rb") as saved:
            os.fsync(saved.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if keep_latest_only:
            for previous in path.parent.iterdir():
                match = re.fullmatch(r"step_(\d+)\.pt", previous.name)
                if (match and int(match[1]) < step and previous != path
                        and previous.is_file() and not previous.is_symlink()):
                    try:
                        previous.unlink()
                    except OSError as error:
                        warnings.warn(f"checkpoint saved, but could not remove {previous}: {error}",
                                      RuntimeWarning, stacklevel=2)
    finally:
        # A failed write must not leave an apparently usable temporary
        # checkpoint in the run directory.
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into model (and optionally optimizer). Returns metadata."""
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved_version = state.get("architecture_version")
    if saved_version is None:
        raise ValueError(
            f"checkpoint {path} has no architecture_version; refusing legacy checkpoint"
        )
    try:
        saved_version = int(saved_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint {path} has invalid architecture_version: {saved_version!r}"
        ) from exc

    expected_version = _model_architecture_version(model)
    if saved_version != expected_version:
        raise ValueError(
            "checkpoint architecture mismatch: "
            f"checkpoint={saved_version}, model={expected_version}"
        )

    saved_config = state.get("config")
    if isinstance(saved_config, dict) and "architecture_version" in saved_config:
        try:
            config_version = int(saved_config["architecture_version"])
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint config has invalid architecture_version") from exc
        if config_version != saved_version:
            raise ValueError(
                "checkpoint config architecture mismatch: "
                f"envelope={saved_version}, config={config_version}"
            )

    model_config = getattr(model, "config", None)
    if isinstance(saved_config, dict) and model_config is not None:
        mismatch = _config_mismatch(saved_config, _config_dict(model_config))
        if mismatch is not None:
            raise ValueError(
                f"checkpoint config mismatch at {mismatch}; only ple.offload may differ"
            )

    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return {
        "step": state["step"],
        "config": state["config"],
        "architecture_version": saved_version,
        "extra": state.get("extra", {}),
    }


def verify_resume_continuity(
    ckpt1: Path,
    ckpt2: Path,
    model: torch.nn.Module,
) -> bool:
    """Verify that ckpt2 continues ckpt1 (step increases, weights differ but sane)."""
    s1 = torch.load(ckpt1, map_location="cpu", weights_only=False)
    s2 = torch.load(ckpt2, map_location="cpu", weights_only=False)
    if s2["step"] <= s1["step"]:
        return False
    # Weights should have changed (training progressed)
    w1 = s1["model_state_dict"]
    w2 = s2["model_state_dict"]
    changed = False
    for k in w1:
        if k in w2 and not torch.equal(w1[k], w2[k]):
            changed = True
            break
    return changed
