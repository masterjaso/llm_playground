"""FlashMini-50B v4 training runner.

``torchrun ... -m flashmini.v4_train --config train.yaml``

Per optimizer step the runner consumes one logical batch of
``world * gradient_accumulation * micro_batch_sequences`` windows, and optimizes

``total = 1.0 * main + mtp_coefficient * mean_k(mtp_k) + router_aux_coefficient * router_aux``

where ``main`` and every ``mtp_k`` are token means over the whole logical batch
(sum-reduced cross entropy divided by the global label count, gradients summed
across ranks and microbatches), ``mtp_coefficient`` is 0.30 before 70% of the
planned tokens and 0.10 after, and ``router_aux`` is the global/logical-batch
Qwen balancing loss from :mod:`flashmini.v4_balance`.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import faulthandler
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.distributed.tensor import DTensor

from . import v4_checkpoint
from .base_init_config import load_config
from .base_init_model import FlashMini50BBaseInit, MTP_MAX_WINDOW, mtp_recursive_steps, mtp_teacher_and_targets, surrogate_config
from .base_init_optimizer import OptimizerTaxonomy
from .v4_balance import MODES as BALANCE_MODES
from .v4_balance import RouterBalance
from .v4_data import DataExhausted, PackedTokenStream, load_manifest
from .v4_optim import OptimizerSettings, OptimizerStack
from .v4_ple_store import PLETableStore

MAIN_LOSS_WEIGHT = 1.0
MTP_WINDOW = MTP_MAX_WINDOW
MTP_COEFFICIENT_EARLY = 0.30
MTP_COEFFICIENT_LATE = 0.10
MTP_SWITCH_FRACTION = 0.70
STRATEGIES = ("single", "ddp", "fsdp2")
LR_SCHEDULES = ("constant", "cosine", "linear", "wsd")
PURPOSES = ("production", "engineering_test")

# Every path must be present and non-null; startup lists all unresolved ones.
REQUIRED = (
    "run.name", "run.purpose", "run.output_dir", "run.seed", "run.deterministic",
    "model.config", "model.gdn_kernel", "model.activation_checkpointing", "model.compute_dtype",
    "tokenizer.fingerprint",
    "distributed.strategy", "distributed.nodes", "distributed.gpus_per_node", "distributed.ple_backing",
    "schedule.planned_total_tokens", "schedule.lr.kind", "schedule.lr.warmup_tokens", "schedule.lr.min_lr_ratio",
    "optimizer.muon.lr", "optimizer.muon.weight_decay", "optimizer.muon.ns_dtype",
    "optimizer.adamw.lr", "optimizer.adamw.betas", "optimizer.adamw.eps",
    "optimizer.adamw.weight_decay.embedding", "optimizer.adamw.weight_decay.control_matrix",
    "optimizer.adamw.weight_decay.no_decay",
    "optimizer.ple.lr", "optimizer.ple.betas", "optimizer.ple.eps", "optimizer.grad_clip",
    "loss.router_aux_coefficient", "loss.router_balance_mode",
    "data.phases", "data.teacher_mixture",
    "checkpoint.dir", "checkpoint.interval_optimizer_steps", "checkpoint.keep_last",
    "observability.metrics_path", "watchdog.step_timeout_seconds",
)
OPERATIONAL_KEYS = (
    ("run", "name"), ("run", "output_dir"), ("run", "stop_after_optimizer_steps"), ("run", "source_commit"),
    ("checkpoint", None), ("observability", None), ("watchdog", None),
)
PHASE_REQUIRED = ("name", "manifest", "sequence_length", "micro_batch_sequences", "gradient_accumulation", "until_tokens")


def _get(raw: dict[str, Any], dotted: str) -> Any:
    value: Any = raw
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


@dataclass(frozen=True)
class Phase:
    name: str
    manifest: str
    sequence_length: int
    micro_batch_sequences: int
    gradient_accumulation: int
    until_tokens: int


class TrainConfig:
    def __init__(self, raw: dict[str, Any], path: Path | None = None):
        self.raw, self.path = raw, path
        self.validate()

    @classmethod
    def load(cls, path: Path | str) -> "TrainConfig":
        path = Path(path)
        return cls(yaml.safe_load(path.read_text()), path)

    def get(self, dotted: str) -> Any:
        return _get(self.raw, dotted)

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() or self.path is None else (Path.cwd() / path)

    @property
    def sha256(self) -> str:
        """Fingerprint of the training semantics; operational locations and stop points are excluded."""
        semantic = copy.deepcopy(self.raw)
        for section, key in OPERATIONAL_KEYS:
            if key is None:
                semantic.pop(section, None)
            elif isinstance(semantic.get(section), dict):
                semantic[section].pop(key, None)
        return hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def unresolved(self) -> list[str]:
        missing = [path for path in REQUIRED if self.get(path) is None]
        for index, phase in enumerate(self.get("data.phases") or []):
            missing += [f"data.phases[{index}].{key}" for key in PHASE_REQUIRED if phase.get(key) is None]
        return missing

    def validate(self) -> None:
        missing = self.unresolved()
        if missing:
            raise ValueError("unresolved mandatory training settings: " + ", ".join(missing))
        purpose = self.get("run.purpose")
        if purpose not in PURPOSES:
            raise ValueError(f"run.purpose must be one of {PURPOSES}")
        if purpose == "production" and self.get("model.surrogate"):
            raise ValueError("production runs cannot use the surrogate model")
        if purpose == "production":
            for path in ("model.bundle", "model.init_checkpoint", "tokenizer.manifest"):
                if self.get(path) is None:
                    raise ValueError(f"unresolved mandatory training settings: {path}")
        if self.get("distributed.strategy") not in STRATEGIES:
            raise ValueError(f"distributed.strategy must be one of {STRATEGIES}")
        if self.get("distributed.ple_backing") not in ("process", "shm"):
            raise ValueError("distributed.ple_backing must be 'process' or 'shm'")
        if self.get("distributed.ple_backing") == "shm" and not self.get("distributed.ple_shm_dir"):
            raise ValueError("unresolved mandatory training settings: distributed.ple_shm_dir")
        if self.get("schedule.lr.kind") not in LR_SCHEDULES:
            raise ValueError(f"schedule.lr.kind must be one of {LR_SCHEDULES}")
        if self.get("schedule.lr.kind") == "wsd" and self.get("schedule.lr.decay_start_fraction") is None:
            raise ValueError("unresolved mandatory training settings: schedule.lr.decay_start_fraction")
        if self.get("loss.router_balance_mode") not in BALANCE_MODES:
            raise ValueError(f"loss.router_balance_mode must be one of {BALANCE_MODES}")
        coefficient = self.get("loss.router_aux_coefficient")
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or coefficient < 0 or not math.isfinite(coefficient):
            raise ValueError("loss.router_aux_coefficient must be an explicit finite non-negative number")
        for forbidden in ("main_weight", "mtp_coefficient", "mtp_window"):
            if self.get(f"loss.{forbidden}") is not None:
                raise ValueError(f"loss.{forbidden} is frozen by the v4 contract and cannot be overridden")
        mixture = self.get("data.teacher_mixture")
        if not isinstance(mixture, dict) or not mixture or abs(sum(float(v) for v in mixture.values()) - 1.0) > 1e-9:
            raise ValueError("data.teacher_mixture must be a non-empty mapping whose fractions sum to 1")
        phases = self.phases
        if [p.until_tokens for p in phases] != sorted({p.until_tokens for p in phases}):
            raise ValueError("data.phases until_tokens must be strictly increasing")
        if phases[-1].until_tokens != int(self.get("schedule.planned_total_tokens")):
            raise ValueError("final data phase until_tokens must equal schedule.planned_total_tokens")
        if self.get("model.compute_dtype") not in ("bfloat16", "float32"):
            raise ValueError("model.compute_dtype must be bfloat16 or float32")
        if self.get("model.gdn_kernel") not in ("chunked", "recurrent"):
            raise ValueError("model.gdn_kernel must be chunked or recurrent")

    @property
    def phases(self) -> list[Phase]:
        return [Phase(str(p["name"]), str(p["manifest"]), int(p["sequence_length"]), int(p["micro_batch_sequences"]),
                      int(p["gradient_accumulation"]), int(p["until_tokens"])) for p in self.get("data.phases")]


def lr_multiplier(schedule: dict[str, Any], tokens: int, planned: int) -> float:
    """LR multiplier after ``tokens`` student tokens (including the current step)."""
    warmup = int(schedule["warmup_tokens"])
    floor = float(schedule["min_lr_ratio"])
    if warmup > 0 and tokens < warmup:
        return tokens / warmup
    kind = schedule["kind"]
    progress = min(1.0, max(0.0, (tokens - warmup) / max(planned - warmup, 1)))
    if kind == "constant":
        return 1.0
    if kind == "cosine":
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == "linear":
        return 1.0 - (1.0 - floor) * progress
    start = float(schedule["decay_start_fraction"]) * planned
    if tokens <= start:
        return 1.0
    return 1.0 - (1.0 - floor) * min(1.0, (tokens - start) / max(planned - start, 1))


def mtp_coefficient(tokens_before_step: int, planned: int) -> tuple[float, str]:
    if tokens_before_step < MTP_SWITCH_FRACTION * planned:
        return MTP_COEFFICIENT_EARLY, "early"
    return MTP_COEFFICIENT_LATE, "late"


def load_init_checkpoint(model: FlashMini50BBaseInit, checkpoint_dir: Path) -> None:
    """Load materialized BF16 init shards into (possibly FSDP2-sharded) parameters and PLE tables."""
    from safetensors import safe_open

    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, Any] = {}

    def handle(name: str):
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(checkpoint_dir / shard), framework="pt")
        return handles[shard]

    with torch.no_grad():
        for name, target in model.state_dict().items():
            if name not in index:
                raise KeyError(f"init checkpoint missing {name}")
            if isinstance(target, DTensor):
                local = target.to_local()
                full_shape = target.shape
                rows = full_shape[0]
                rank = target.device_mesh.get_local_rank()
                world = target.device_mesh.size()
                chunk = -(-rows // world)
                start, stop = min(rank * chunk, rows), min((rank + 1) * chunk, rows)
                local.copy_(handle(name).get_slice(name)[start:stop])
            else:
                target.copy_(handle(name).get_tensor(name))
        for name, table in model.ple.store.named_tables():
            if getattr(model.ple.store, "writer", True):
                table.copy_(handle(name).get_tensor(name))


class Runner:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.metrics_file = None

    # -- setup -----------------------------------------------------------------------
    def _init_distributed(self) -> None:
        strategy = self.cfg.get("distributed.strategy")
        world_env = int(os.environ.get("WORLD_SIZE", "1"))
        expected = int(self.cfg.get("distributed.nodes")) * int(self.cfg.get("distributed.gpus_per_node"))
        if world_env != expected:
            raise RuntimeError(f"WORLD_SIZE {world_env} != distributed.nodes * gpus_per_node = {expected}")
        if strategy == "single" and world_env != 1:
            raise RuntimeError("strategy 'single' requires WORLD_SIZE 1")
        use_cuda = torch.cuda.is_available() and not self.cfg.get("distributed.force_cpu")
        if strategy != "single" and not dist.is_initialized():
            dist.init_process_group("nccl" if use_cuda else "gloo", timeout=datetime.timedelta(seconds=int(self.cfg.get("watchdog.step_timeout_seconds"))))
        self.rank, self.world = _rank_world()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if use_cuda:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        self.cpu_group = dist.new_group(backend="gloo") if dist.is_initialized() else None
        self.gpus_per_node = int(self.cfg.get("distributed.gpus_per_node"))

    def _seed(self) -> None:
        seed = int(self.cfg.get("run.seed"))
        torch.manual_seed(seed)
        if self.cfg.get("run.deterministic"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False

    def _identity(self) -> None:
        cfg = self.cfg
        if cfg.get("model.surrogate"):
            self.arch = surrogate_config(**(cfg.get("model.surrogate_geometry") or {}))
        else:
            self.arch = load_config(cfg.resolve(cfg.get("model.config")))
        self.tokenizer_fingerprint = cfg.get("tokenizer.fingerprint")
        if not cfg.get("model.surrogate"):
            from .v4_tokenizer import FrozenTokenizer

            tokenizer = FrozenTokenizer(cfg.resolve(cfg.get("tokenizer.manifest")), expected_fingerprint=self.arch.section("tokenizer")["fingerprint"])
            if tokenizer.fingerprint != self.tokenizer_fingerprint:
                raise ValueError("train config tokenizer.fingerprint differs from the frozen tokenizer")
            if tokenizer.special_token_ids != self.arch.section("tokenizer")["special_token_ids"]:
                raise ValueError("tokenizer special IDs differ from the architecture config")
        if cfg.get("run.purpose") == "production":
            from . import v4_preflight

            problems = v4_preflight.check_source_and_identity(cfg.resolve(cfg.get("model.bundle")), self.arch)
            if problems:
                raise RuntimeError("startup identity checks failed: " + "; ".join(problems))
        self.source_commit = cfg.get("run.source_commit") or _current_commit()

    def _build_model(self) -> None:
        cfg = self.cfg
        strategy = cfg.get("distributed.strategy")
        store = PLETableStore(self.arch.section("ple")["hash_head_rows"], self.arch.section("ple")["head_dim"], device="meta")
        with torch.device("meta"):
            model = FlashMini50BBaseInit(self.arch, gdn_kernel=cfg.get("model.gdn_kernel"), ple_store=store)
        model.activation_checkpointing = bool(cfg.get("model.activation_checkpointing"))
        if strategy == "fsdp2":
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if cfg.get("model.compute_dtype") == "bfloat16" else torch.float32,
                                          reduce_dtype=torch.float32)
            for block in model.blocks:
                fully_shard(block, mp_policy=policy)
            fully_shard(model.mtp, mp_policy=policy)
            fully_shard(model, mp_policy=policy)
            for module in model.modules():
                if hasattr(module, "set_gradient_divide_factor"):
                    module.set_gradient_divide_factor(1.0)
        model.to_empty(device=self.device)
        backing = cfg.get("distributed.ple_backing")
        writer = backing == "process" or self.local_rank == 0
        store.allocate(device="cpu", backing=backing, shm_dir=cfg.get("distributed.ple_shm_dir"), writer=writer)
        if cfg.get("model.surrogate"):
            with torch.no_grad():
                for name, tensor in model.named_logical_tensors():
                    if isinstance(tensor, DTensor):
                        from . import v4_init

                        full = v4_init.materialize(name, tuple(tensor.shape))
                        local = tensor.to_local()
                        rows, world, rank = tensor.shape[0], tensor.device_mesh.size(), tensor.device_mesh.get_local_rank()
                        chunk = -(-rows // world)
                        local.copy_(full[min(rank * chunk, rows):min((rank + 1) * chunk, rows)])
                    elif writer or not name.startswith("ple.tables."):
                        from . import v4_init

                        v4_init.fill_(tensor, name)
        else:
            load_init_checkpoint(model, cfg.resolve(cfg.get("model.init_checkpoint")))
        if dist.is_initialized() and backing == "shm":
            dist.barrier(group=self.cpu_group)
        model.train()
        self.model = model
        self.sharded = strategy == "fsdp2"

    def _build_training(self) -> None:
        cfg = self.cfg
        settings = OptimizerSettings.from_mapping(cfg.get("optimizer"))
        self.stack = OptimizerStack(self.model, OptimizerTaxonomy(self.arch), settings,
                                    group=None, ple_group=self.cpu_group if dist.is_initialized() else None)
        moe = self.arch.section("moe")
        self.balance = RouterBalance(moe["routed_experts"], moe["top_k"], mode=cfg.get("loss.router_balance_mode"),
                                     group=None, device=self.device)
        self.router_coefficient = float(cfg.get("loss.router_aux_coefficient"))
        self.planned_tokens = int(cfg.get("schedule.planned_total_tokens"))
        vocab = self.arch.vocab_size
        self.streams = {}
        for phase in cfg.phases:
            manifest = load_manifest(cfg.resolve(phase.manifest), expected_fingerprint=self.tokenizer_fingerprint, vocab_size=vocab)
            self.streams[phase.name] = PackedTokenStream(manifest, phase.sequence_length)
        self.state = {
            "tokens_consumed": 0, "label_tokens_consumed": 0, "optimizer_step": 0, "micro_step": 0,
            "phase_index": 0, "cursors": {phase.name: 0 for phase in cfg.phases}, "mtp_phase": "early",
            "lr_multiplier": 0.0,
        }

    def identity_record(self) -> dict[str, Any]:
        return {
            "architecture_sha256": self.arch.architecture_sha256,
            "config_sha256": self.arch.config_sha256,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "train_config_sha256": self.cfg.sha256,
            "source_commit": self.source_commit,
            "topology": {"strategy": self.cfg.get("distributed.strategy"), "world_size": self.world,
                         "nodes": int(self.cfg.get("distributed.nodes")), "gpus_per_node": self.gpus_per_node,
                         "ple_backing": self.cfg.get("distributed.ple_backing")},
        }

    def _maybe_resume(self) -> None:
        root = self.cfg.resolve(self.cfg.get("checkpoint.dir"))
        found = v4_checkpoint.latest(root)
        if found is None:
            return
        identity = self.identity_record()
        expect = {key: identity[key] for key in ("architecture_sha256", "config_sha256", "tokenizer_fingerprint", "train_config_sha256", "source_commit")}
        restored = v4_checkpoint.load(found, model=self.model, stack=self.stack, expect=expect)
        if restored["topology"] != identity["topology"]:
            raise ValueError(f"checkpoint topology {restored['topology']} != current {identity['topology']}")
        self.state = restored["trainer"]
        self._log({"event": "resumed", "checkpoint": str(found), "optimizer_step": self.state["optimizer_step"],
                   "tokens_consumed": self.state["tokens_consumed"]})

    def setup(self) -> None:
        self._init_distributed()
        self._seed()
        self._identity()
        self._build_model()
        self._build_training()
        if self.rank == 0:
            path = self.cfg.resolve(self.cfg.get("observability.metrics_path"))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_file = path.open("a")
            self._log({"event": "start", **self.identity_record(), "train_config": self.cfg.raw,
                       "loss_contract": {"main_weight": MAIN_LOSS_WEIGHT, "mtp_window": MTP_WINDOW,
                                         "mtp_coefficient_early": MTP_COEFFICIENT_EARLY, "mtp_coefficient_late": MTP_COEFFICIENT_LATE,
                                         "mtp_switch_fraction": MTP_SWITCH_FRACTION, "router_aux_coefficient": self.router_coefficient,
                                         "router_balance_mode": self.cfg.get("loss.router_balance_mode")}})
        self._maybe_resume()

    # -- step ------------------------------------------------------------------------
    def _log(self, record: dict[str, Any]) -> None:
        if self.metrics_file is not None:
            self.metrics_file.write(json.dumps(record, sort_keys=True, default=float) + "\n")
            self.metrics_file.flush()

    def _all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(tensor)
        return tensor

    def _autocast(self):
        if self.device.type == "cuda" and self.cfg.get("model.compute_dtype") == "bfloat16" and not self.sharded:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _phase(self) -> Phase:
        phases = self.cfg.phases
        while self.state["phase_index"] < len(phases) - 1 and self.state["tokens_consumed"] >= phases[self.state["phase_index"]].until_tokens:
            self.state["phase_index"] += 1
        return phases[self.state["phase_index"]]

    def _require_finite(self, values: dict[str, torch.Tensor], where: str) -> None:
        flag = torch.tensor([0.0 if all(bool(torch.isfinite(v.detach()).all()) for v in values.values()) else 1.0], device=self.device)
        if float(self._all_reduce(flag)) > 0:
            bad = [key for key, value in values.items() if not bool(torch.isfinite(value.detach()).all())]
            self.stack.zero_grad()
            raise FloatingPointError(f"non-finite {where} {bad or '(on another rank)'}; optimizer step refused")

    def _sync_grads(self) -> None:
        if self.cfg.get("distributed.strategy") != "ddp" or self.world == 1:
            return
        for param in self.model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad)

    def train_step(self) -> dict[str, Any]:
        phase = self._phase()
        stream = self.streams[phase.name]
        cursor = self.state["cursors"][phase.name]
        micro, accumulation = phase.micro_batch_sequences, phase.gradient_accumulation
        steps = mtp_recursive_steps(MTP_WINDOW)
        tokens_before = self.state["tokens_consumed"]
        step_tokens = self.world * accumulation * micro * phase.sequence_length
        coefficient, mtp_phase = mtp_coefficient(tokens_before, self.planned_tokens)
        multiplier = lr_multiplier(self.cfg.get("schedule.lr"), tokens_before + step_tokens, self.planned_tokens)
        self.stack.set_lr_multiplier(multiplier)
        started = time.perf_counter()
        batches = [stream.microbatch(cursor, micro_index=m, rank=self.rank, world=self.world, micro_batch=micro) for m in range(accumulation)]
        counts = torch.zeros(1 + steps, dtype=torch.float64, device=self.device)
        for batch in batches:
            counts[0] += (batch.labels != -100).sum()
            for depth, (_, target) in enumerate(mtp_teacher_and_targets(batch.labels, steps, self.model.pad_id), start=1):
                counts[depth] += (target != -100).sum()
        counts = self._all_reduce(counts)
        self.balance.begin_step()
        self.balance.set_logical_tokens(accumulation * micro * phase.sequence_length)
        if self.balance.mode == "exact_prepass":
            with torch.no_grad(), self._autocast():
                for batch in batches:
                    out = self.model(batch.input_ids.to(self.device), labels=batch.labels.to(self.device),
                                     mtp_window=MTP_WINDOW, routing_only=True)
                    self.balance.add_prepass(out["stats"])
        sums = torch.zeros(1 + steps + 1, dtype=torch.float64, device=self.device)
        for index, batch in enumerate(batches):
            ids, labels = batch.input_ids.to(self.device), batch.labels.to(self.device)
            last = index == len(batches) - 1
            if self.sharded:
                self.model.set_requires_gradient_sync(last)
            with self._autocast():
                out = self.model(ids, labels=labels, mtp_window=MTP_WINDOW)
                aux = self.balance.microbatch_loss(out["stats"], ids.numel())
            main = out["loss_sum"] / counts[0].clamp_min(1).float()
            mtp_terms = [total / counts[depth].clamp_min(1).float() for depth, total in enumerate(out["mtp_loss_sums"], start=1)]
            mtp = torch.stack(mtp_terms).mean()
            loss = MAIN_LOSS_WEIGHT * main + coefficient * mtp + self.router_coefficient * aux
            self._require_finite({"main": main, "mtp": mtp, "router_aux": aux, "total": loss}, "forward/loss")
            loss.backward()
            self.model.ple.store.collect_gradients()
            sums[0] += out["loss_sum"].detach().double()
            for depth, total in enumerate(out["mtp_loss_sums"], start=1):
                sums[depth] += total.detach().double()
            sums[-1] += aux.detach().double()
            self.state["micro_step"] += 1
        self._sync_grads()
        optim = self.stack.step()
        sums = self._all_reduce(sums)
        balance = self.balance.finalize()
        main_value = float(sums[0] / counts[0].clamp_min(1))
        mtp_values = [float(sums[depth] / counts[depth].clamp_min(1)) for depth in range(1, steps + 1)]
        mtp_value = sum(mtp_values) / len(mtp_values)
        aux_value = float(sums[-1])
        self.state["cursors"][phase.name] = cursor + self.world * accumulation * micro
        self.state["tokens_consumed"] += step_tokens
        self.state["label_tokens_consumed"] += int(counts[0])
        self.state["optimizer_step"] += 1
        self.state["mtp_phase"] = mtp_phase
        self.state["lr_multiplier"] = multiplier
        elapsed = time.perf_counter() - started
        layers = balance["layers"].values()
        record = {
            "event": "step", "optimizer_step": self.state["optimizer_step"], "global_step": self.state["optimizer_step"],
            "micro_step": self.state["micro_step"], "tokens_consumed": self.state["tokens_consumed"],
            "label_tokens_consumed": self.state["label_tokens_consumed"], "phase": phase.name,
            "loss_main": main_value, "loss_mtp_by_depth": {f"t+{depth + 1}": value for depth, value in enumerate(mtp_values, start=1)},
            "loss_mtp": mtp_value, "mtp_coefficient": coefficient, "mtp_phase": mtp_phase,
            "router_aux_objective": aux_value, "router_aux_logical": balance["router_aux_logical"],
            "router_aux_coefficient": self.router_coefficient,
            "loss_total": MAIN_LOSS_WEIGHT * main_value + coefficient * mtp_value + self.router_coefficient * aux_value,
            "expert_load_entropy_normalized_min": min(item["load_entropy_normalized"] for item in layers),
            "expert_load_entropy_normalized_mean": sum(item["load_entropy_normalized"] for item in layers) / len(balance["layers"]),
            "expert_max_over_mean_load": max(item["max_over_mean_load"] for item in layers),
            "expert_min_over_mean_load": min(item["min_over_mean_load"] for item in layers),
            "expert_load_fraction": {key: item["load_fraction"] for key, item in balance["layers"].items()} if self.cfg.get("observability.log_expert_distribution") else None,
            "grad_norm_total": optim["grad_norm_total"], "grad_norm_by_family": optim["grad_norm_by_family"],
            "clip_coefficient": optim["clip_coefficient"], "lr": self.stack.current_lrs(), "lr_multiplier": multiplier,
            "ple": {**self.model.ple.store.stats.snapshot_and_reset(), "rows_updated_owned": optim["ple_rows_updated"]},
            "tokens_per_second": step_tokens / max(elapsed, 1e-9), "step_seconds": elapsed,
            "hardware": _hardware(self.device),
        }
        return record

    def checkpoint(self) -> Path:
        root = self.cfg.resolve(self.cfg.get("checkpoint.dir"))
        trainer_state = {**self.identity_record(), "trainer": copy.deepcopy(self.state),
                         "scheduler": {"kind": self.cfg.get("schedule.lr.kind"), "lr_multiplier": self.state["lr_multiplier"]}}
        path = v4_checkpoint.save(root, self.state["optimizer_step"], model=self.model, stack=self.stack, trainer_state=trainer_state)
        if self.rank == 0:
            keep = int(self.cfg.get("checkpoint.keep_last"))
            existing = sorted(item for item in root.glob("step_*") if item.is_dir() and not item.name.endswith(".tmp"))
            for old in existing[:-keep] if keep > 0 else []:
                shutil.rmtree(old)
        return path

    def run(self) -> int:
        interval = int(self.cfg.get("checkpoint.interval_optimizer_steps"))
        timeout = int(self.cfg.get("watchdog.step_timeout_seconds"))
        stop_after = self.cfg.get("run.stop_after_optimizer_steps")
        status = "completed"
        try:
            while self.state["tokens_consumed"] < self.planned_tokens:
                if stop_after is not None and self.state["optimizer_step"] >= int(stop_after):
                    status = "stopped_at_requested_step"
                    break
                faulthandler.dump_traceback_later(timeout, exit=True)
                try:
                    record = self.train_step()
                except DataExhausted as exc:
                    status = f"data_exhausted: {exc}"
                    break
                finally:
                    faulthandler.cancel_dump_traceback_later()
                record["checkpoint"] = None
                if self.state["optimizer_step"] % interval == 0:
                    record["checkpoint"] = str(self.checkpoint())
                self._log(record)
        except FloatingPointError as exc:
            self._log({"event": "failed", "reason": str(exc), "optimizer_step": self.state["optimizer_step"]})
            raise
        if self.state["optimizer_step"] % interval != 0:
            self._log({"event": "checkpoint", "path": str(self.checkpoint())})
        self._log({"event": "end", "status": status, **{key: self.state[key] for key in ("optimizer_step", "tokens_consumed")}})
        return 0

    def close(self) -> None:
        if self.metrics_file is not None:
            self.metrics_file.close()


def _hardware(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device)}
    info = {"device": torch.cuda.get_device_name(device), "memory_allocated": torch.cuda.memory_allocated(device),
            "max_memory_allocated": torch.cuda.max_memory_allocated(device)}
    try:
        info["utilization_percent"] = torch.cuda.utilization(device)
    except Exception:
        info["utilization_percent"] = None
    return info


def _current_commit() -> str | None:
    import subprocess

    result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flashmini.v4_train")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    runner = Runner(TrainConfig.load(args.config))
    try:
        runner.setup()
        return runner.run()
    finally:
        runner.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["MTP_COEFFICIENT_EARLY", "MTP_COEFFICIENT_LATE", "MTP_SWITCH_FRACTION", "Phase", "REQUIRED", "Runner",
           "TrainConfig", "load_init_checkpoint", "lr_multiplier", "mtp_coefficient"]
