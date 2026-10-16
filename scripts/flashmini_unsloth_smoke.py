"""Bounded integration probe using the installed Unsloth Desktop training backend.

Run with Desktop's Python after installing this repository with --no-deps.
This invokes the same loader and trainer as Desktop, without starting its UI.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, help="Local FlashMini HF export; default: tiny offline fixture"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New probe directory inside Desktop's outputs directory",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10 or not 2 <= args.sequence_length <= 128:
        parser.error("Probe bounds: steps 1..10 and sequence length 2..128")
    if args.gpu < 0:
        parser.error("GPU must be a nonnegative index")
    if args.output.exists():
        parser.error("Use a new output directory; existing results are preserved")
    if args.model is not None and not (args.model / "config.json").is_file():
        parser.error("--model must contain a Hugging Face config.json")
    if (
        args.model is not None
        and json.loads((args.model / "config.json").read_text(encoding="utf-8")).get("model_type")
        != "flashmini"
    ):
        parser.error("Only a locally exported FlashMini model is supported")

    # Set before importing CUDA/Unsloth. The busy second card is never selected implicitly.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    args.output.mkdir(parents=True)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str((args.output / "compile-cache").resolve())
    studio = importlib.util.find_spec("studio")
    if studio is None or not studio.submodule_search_locations:
        raise RuntimeError("Use Unsloth Desktop's Python environment")
    backend = Path(next(iter(studio.submodule_search_locations))) / "backend"
    sys.path.insert(0, str(backend))

    import torch
    from core.training.trainer import UnslothTrainer
    from core.training.worker import _model_load_security_error
    from datasets import Dataset
    from utils.paths import resolve_output_dir

    resolve_output_dir(str(args.output.resolve()))
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires a CUDA GPU")
    if args.model is None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        from flashmini.config import FlashMiniConfig as NativeConfig
        from flashmini.config import GatedDeltaNetConfig, KVConfig, MoEConfig, PLEConfig
        from flashmini.hf_adapter import FlashMiniConfig, FlashMiniForCausalLM

        torch.manual_seed(17)
        native = NativeConfig(
            architecture_version=3,
            vocab_size=64,
            d_model=32,
            num_layers=8,
            num_heads=2,
            head_dim=16,
            hc_lowrank=8,
            max_seq_len=128,
            use_ple=True,
            moe=MoEConfig(num_experts=4, top_k=2, expert_intermediate=48),
            gdn=GatedDeltaNetConfig(d_state=8, chunk_size=4),
            ple=PLEConfig(
                sparse=False, heads_per_ngram=2, embed_dim=32, ngram_vocab_size_base=31, eos_id=1
            ),
            kvc=KVConfig(enabled=True),
        )
        args.model = args.output / "base-model"
        model = FlashMiniForCausalLM(FlashMiniConfig.from_native_config(native))
        model.save_pretrained(args.model)
        vocabulary = [
            "[UNK]",
            "[EOS]",
            "[PAD]",
            "A",
            "a",
            "small",
            "language",
            "model",
            "learns",
            "from",
            "short",
            "sequence",
            "of",
            "words",
            ".",
            "Training",
            "updates",
            "weights",
            "using",
            "examples",
            "and",
            "measured",
            "loss",
            "FlashMini",
            "uses",
            "experts",
            "recurrent",
            "layers",
            "to",
            "predict",
            "the",
            "next",
            "token",
            "This",
            "run",
            "checks",
            "loading",
            "gradients",
            "saving",
        ]
        vocabulary += [f"unused{index}" for index in range(64 - len(vocabulary))]
        tokenizer = Tokenizer(WordLevel(dict(zip(vocabulary, range(64))), unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            model_max_length=128,
            padding_side="right",
        ).save_pretrained(args.model)
        del model
    security_error = _model_load_security_error(
        {
            "model_name": str(args.model.resolve()),
            "trust_remote_code": True,
            "local_files_only": True,
            "load_in_4bit": False,
        },
        str(args.model.resolve()),
        None,
    )
    if security_error:
        raise RuntimeError(f"Desktop security check refused this export: {security_error}")
    torch.cuda.reset_peak_memory_stats()
    trainer = UnslothTrainer()
    loaded = trainer.load_model(
        model_name=str(args.model.resolve()),
        max_seq_length=args.sequence_length,
        load_in_4bit=False,
        full_finetuning=True,
        trust_remote_code=True,  # Only the explicitly selected local repository export.
        local_files_only=True,
        gpu_ids=[0],
        use_gradient_checkpointing=False,
    )
    if not loaded:
        raise RuntimeError(f"Desktop loader failed: {trainer.model_load_error}")
    if not trainer.prepare_model_for_training(use_lora=False, use_gradient_checkpointing=False):
        raise RuntimeError(f"Desktop preparation failed: {trainer.training_progress.error}")
    parameter_count = sum(p.numel() for p in trainer.model.parameters())
    before = trainer.model.get_input_embeddings().weight.detach().float().cpu().clone()
    dataset = Dataset.from_dict(
        {
            "text": [
                text * 16
                for text in [
                    "A small language model learns from a short sequence of words.",
                    "Training updates model weights using examples and a measured loss.",
                    "FlashMini uses experts and recurrent layers to predict the next token.",
                    "This short run checks loading, gradients, updates and saving.",
                ]
            ]
        }
    )
    started = trainer.start_training(
        dataset={"dataset": dataset},
        output_dir=str((args.output / "training").resolve()),
        max_steps=args.steps,
        save_steps=args.steps,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        random_seed=17,
        packing=False,
        train_on_completions=False,
        dataset_final_format="raw_text",
        max_seq_length=args.sequence_length,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )
    if not started:
        raise RuntimeError(f"Desktop refused training: {trainer.training_progress.error}")
    # Training runs in Desktop's own thread; errors are returned through its progress payload.
    trainer.training_thread.join()
    progress = dataclasses.asdict(trainer.training_progress)
    state = trainer.trainer.state if trainer.trainer is not None else None
    history = state.log_history if state is not None else []
    loss_values = [row["loss"] for row in history if "loss" in row]
    changed = False
    if not progress["error"]:
        after = trainer.model.get_input_embeddings().weight.detach().float().cpu()
        changed = not torch.equal(before, after)
    report = {
        "model": str(args.model.resolve()),
        "parameters": parameter_count,
        "physical_gpu": args.gpu,
        "sequence_length": args.sequence_length,
        "steps": state.global_step if state is not None else 0,
        "weights_changed": changed,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "progress": progress,
        "log_history": history,
        "scope": "Installed Desktop backend; frontend and long-run quality not tested",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if progress["error"] or report["steps"] != args.steps or not changed:
        raise RuntimeError(f"Integration probe failed; see {args.output / 'report.json'}")
    if not loss_values or not all(math.isfinite(value) for value in loss_values):
        raise RuntimeError("Integration probe did not record finite training losses")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "parameters",
                    "steps",
                    "weights_changed",
                    "peak_cuda_allocated_gib",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
