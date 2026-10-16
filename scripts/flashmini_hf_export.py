"""Export a native FlashMini config/checkpoint as a Hugging Face folder."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from flashmini.config import FlashMiniConfig as NativeFlashMiniConfig
from flashmini.hf_adapter import (
    FlashMiniConfig,
    FlashMiniForCausalLM,
    register_flashmini_auto_classes,
)


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) if path.suffix.lower() in {".yaml", ".yml"} else json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"config file {path} must contain a mapping")
    return value


def _load_tokenizer(identifier: str):
    tokenizer = AutoTokenizer.from_pretrained(identifier)
    tokenizer.padding_side = "right"
    if tokenizer.padding_side != "right":
        raise ValueError("FlashMini HF exports require a right-padding tokenizer")
    return tokenizer


def _prepare_desktop_pad_token(tokenizer, source_values: dict[str, Any]) -> dict[str, Any]:
    """Add a dedicated pad row while leaving the source config untouched."""
    source_vocab_size = source_values.get("vocab_size")
    if source_vocab_size is None or len(tokenizer) != int(source_vocab_size):
        raise ValueError(
            "--desktop-pad-token requires tokenizer size to match source config vocab_size "
            f"before adding the pad token (tokenizer={len(tokenizer)}, source={source_vocab_size})"
        )
    original_vocab_size = len(tokenizer)
    added = tokenizer.add_special_tokens({"pad_token": "<|flashmini_pad|>"})
    tokenizer.padding_side = "right"
    if tokenizer.padding_side != "right":
        raise ValueError("FlashMini HF exports require a right-padding tokenizer")
    if len(tokenizer) not in (original_vocab_size, original_vocab_size + 1) or len(tokenizer) != original_vocab_size + added:
        raise ValueError("--desktop-pad-token may add at most one tokenizer vocabulary entry")
    if (
        tokenizer.pad_token_id is None
        or not 0 <= tokenizer.pad_token_id < len(tokenizer)
        or tokenizer.pad_token_id == tokenizer.eos_token_id
    ):
        raise ValueError("--desktop-pad-token requires a pad token distinct from eos_token")
    values = copy.deepcopy(source_values)
    values["vocab_size"] = len(tokenizer)
    ple_values = values.get("ple")
    if isinstance(ple_values, dict) and tokenizer.eos_token_id is not None:
        ple_values["eos_id"] = int(tokenizer.eos_token_id)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="destination Hugging Face model folder")
    parser.add_argument("--config", type=Path, help="native FlashMini YAML/JSON config")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="native checkpoint file or production checkpoint directory (optional for a fresh model)",
    )
    parser.add_argument("--tokenizer", help="tokenizer folder or Hub id to copy into the export")
    parser.add_argument("--max-shard-size", default="50GB", help="Transformers shard size")
    parser.add_argument("--step", type=int, default=0, help="metadata step for no-checkpoint exports")
    parser.add_argument("--seed", type=int, default=17, help="torch seed for fresh model initialization")
    parser.add_argument(
        "--desktop-pad-token",
        action="store_true",
        help="fresh export only: add a dedicated right-padding token for Desktop/Unsloth",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="dtype used when writing fresh or imported model weights",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config is None and args.checkpoint is None:
        raise SystemExit("one of --config or --checkpoint is required")
    if args.desktop_pad_token and args.checkpoint is not None:
        raise SystemExit("--desktop-pad-token is supported only for fresh exports, not checkpoints")
    if args.desktop_pad_token and args.tokenizer is None:
        raise SystemExit("--desktop-pad-token requires --tokenizer")
    if args.output.exists():
        if not args.output.is_dir():
            raise SystemExit(f"export output is not a directory: {args.output}")
        if any(args.output.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty export directory: {args.output}")

    register_flashmini_auto_classes()
    torch.manual_seed(args.seed)
    hf_config = None
    tokenizer = None
    if args.config is not None:
        source_values = _read_config(args.config)
        if args.desktop_pad_token:
            assert args.tokenizer is not None
            tokenizer = _load_tokenizer(args.tokenizer)
            source_values = _prepare_desktop_pad_token(tokenizer, source_values)
        native_config = NativeFlashMiniConfig.from_dict(source_values)
        hf_config = FlashMiniConfig.from_native_config(native_config)

    if args.checkpoint is not None:
        model = FlashMiniForCausalLM.from_native_checkpoint(args.checkpoint, config=hf_config)
    else:
        assert hf_config is not None
        model = FlashMiniForCausalLM(hf_config)

    dtype = getattr(torch, args.dtype)
    model.to(dtype=dtype)

    if args.tokenizer is not None and tokenizer is None:
        tokenizer = _load_tokenizer(args.tokenizer)
    if tokenizer is not None:
        if len(tokenizer) != model.config.vocab_size:
            raise ValueError(
                f"tokenizer size {len(tokenizer)} does not match model vocab_size {model.config.vocab_size}"
            )
        for field in ("bos_token_id", "eos_token_id", "pad_token_id", "sep_token_id", "cls_token_id"):
            token_id = getattr(tokenizer, field, None)
            if token_id is not None:
                setattr(model.config, field, token_id)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, max_shard_size=args.max_shard_size)

    if tokenizer is not None:
        tokenizer.save_pretrained(args.output)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_type": model.config.model_type,
                "architecture_version": model.config.architecture_version,
                "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
                "tokenizer": args.tokenizer,
                "desktop_pad_token": bool(args.desktop_pad_token),
                "dtype": args.dtype,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
