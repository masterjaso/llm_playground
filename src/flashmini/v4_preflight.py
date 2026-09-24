"""Donor preflight for FlashMini-50B v4: fail closed before any training step.

``python -m flashmini.v4_preflight --bundle flashmini_50b_base_init_v1 \
    --checkpoint <materialized-dir> --train-config <train.yaml>``

Scopes: ``full`` (default) runs artifact and runtime checks and is the only scope
that prints ``FLASHMINI V4 TRAINING PREFLIGHT: PASS``.  ``artifacts`` skips the
host/accelerator checks (useful off-cluster) and prints a distinct line.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from . import base_init_bundle as bundle_mod
from .base_init_accounting import parameter_report
from .base_init_config import FlashMini50BConfig, load_config
from .base_init_optimizer import OptimizerTaxonomy

PASS_LINE = "FLASHMINI V4 TRAINING PREFLIGHT: PASS"
ARTIFACT_PASS_LINE = "FLASHMINI V4 ARTIFACT PREFLIGHT: PASS (runtime host/accelerator checks not run)"
FAIL_LINE = "FLASHMINI V4 TRAINING PREFLIGHT: FAIL"
FROZEN_PARAMETER_COUNTS = {"base": 50_276_673_408, "mtp": 685_511_168, "total": 50_962_184_576}
REPO_ROOT = Path(__file__).resolve().parents[2]


class Check:
    def __init__(self, name: str, problems: list[str], detail: dict[str, Any] | None = None):
        self.name, self.problems, self.detail = name, problems, detail or {}

    @property
    def ok(self) -> bool:
        return not self.problems


def _git(*args: str, repo: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


def check_source(bundle: Path, repo: Path = REPO_ROOT) -> list[str]:
    provenance = json.loads((bundle / "source_provenance.json").read_text())
    source = provenance["source_commit"]
    problems = []
    if (bundle / "source_git_sha.txt").read_text().strip() != source:
        problems.append("source_git_sha.txt differs from source_provenance.json")
    spec = json.loads((bundle / "init_spec.json").read_text())
    if spec.get("source_git_sha") != source:
        problems.append("init_spec.json source_git_sha differs from source_provenance.json")
    if _git("cat-file", "-e", f"{source}^{{commit}}", repo=repo).returncode != 0:
        return problems + [f"source commit {source} not present in this checkout"]
    head = _git("rev-parse", "HEAD", repo=repo).stdout.strip()
    if _git("merge-base", "--is-ancestor", source, head, repo=repo).returncode != 0:
        problems.append(f"source commit {source} is not an ancestor of HEAD {head}")
    diff = _git("diff", "--name-only", source, head, "--", ".", f":(exclude){bundle_mod.BUNDLE_DIRNAME}", repo=repo).stdout.split()
    if diff:
        problems.append(f"HEAD differs from source commit outside the bundle: {diff[:5]}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", repo=repo).stdout.strip()
    if dirty:
        problems.append(f"working tree has uncommitted tracked changes: {dirty.splitlines()[:5]}")
    if bundle_mod.source_tree_sha256(repo, head) != provenance["source_tree_sha256"]:
        problems.append("source_tree_sha256 of HEAD differs from the bundle provenance")
    return problems


def check_identity(bundle: Path, config: FlashMini50BConfig, repo: Path = REPO_ROOT) -> list[str]:
    problems = []
    spec = json.loads((bundle / "init_spec.json").read_text())
    manifest = json.loads((bundle / "tensor_manifest.json").read_text())
    bundled = load_config(bundle / "flashmini_50b_base_init_v1.yaml")
    for label, value in (("repository config", config.config_sha256), ("bundled config", bundled.config_sha256), ("tensor manifest", manifest["config_sha256"])):
        if value != spec["config_sha256"]:
            problems.append(f"config SHA mismatch: {label} {value} != init_spec {spec['config_sha256']}")
    for label, value in (("repository config", config.architecture_sha256), ("bundled config", bundled.architecture_sha256), ("tensor manifest", manifest["architecture_sha256"])):
        if value != spec["architecture_sha256"]:
            problems.append(f"architecture SHA mismatch: {label} {value} != init_spec {spec['architecture_sha256']}")
    return problems


def check_source_and_identity(bundle: Path, config: FlashMini50BConfig) -> list[str]:
    return check_source(Path(bundle)) + check_identity(Path(bundle), config)


def check_tokenizer(config: FlashMini50BConfig, repo: Path = REPO_ROOT) -> list[str]:
    from .v4_tokenizer import SPECIAL_TOKEN_ROLES, VOCAB_SIZE, FrozenTokenizer

    section = config.section("tokenizer")
    manifest = repo / section["manifest"]
    if not manifest.exists():
        return [f"missing tokenizer manifest {manifest}"]
    try:
        tokenizer = FrozenTokenizer(manifest, expected_fingerprint=section["fingerprint"])
    except (FileNotFoundError, ValueError) as exc:
        return [f"tokenizer: {exc}"]
    problems = []
    if tokenizer.vocab_size != config.vocab_size or VOCAB_SIZE != config.vocab_size:
        problems.append(f"tokenizer vocab {tokenizer.vocab_size} != model {config.vocab_size}")
    ids = tokenizer.special_token_ids
    missing = [role for role in SPECIAL_TOKEN_ROLES if role not in ids]
    if missing:
        problems.append(f"incomplete special IDs: missing {missing}")
    if ids != section["special_token_ids"]:
        problems.append("tokenizer special IDs differ from the architecture config")
    for name, index in ids.items():
        if tokenizer.id_to_token(index) is None:
            problems.append(f"special token {name} id {index} has no vocabulary entry")
    return problems


def check_parameters(bundle: Path, config: FlashMini50BConfig) -> list[str]:
    report = parameter_report(config)
    manifest = json.loads((bundle / "tensor_manifest.json").read_text())
    actual = {"base": report["base_total_learned_parameters"], "mtp": report["mtp_total_learned_parameters"],
              "total": report["checkpoint_total_learned_parameters"]}
    problems = []
    if actual != FROZEN_PARAMETER_COUNTS:
        problems.append(f"parameter count {actual} != frozen {FROZEN_PARAMETER_COUNTS}")
    if {key: manifest["parameter_counts"][key] for key in actual} != actual:
        problems.append("tensor manifest parameter counts differ from the model")
    shapes = {item["name"]: item["shape"] for item in report["tensors"]}
    if shapes != {item["name"]: item["shape"] for item in manifest["tensors"]}:
        problems.append("tensor manifest names/shapes differ from the model")
    return problems


def check_optimizer_classification(bundle: Path, config: FlashMini50BConfig) -> list[str]:
    taxonomy = OptimizerTaxonomy(config)
    manifest = json.loads((bundle / "tensor_manifest.json").read_text())
    problems = []
    for item in manifest["tensors"]:
        try:
            cls = taxonomy.classify(item["name"], tuple(item["shape"]))
        except (KeyError, ValueError) as exc:
            problems.append(str(exc))
            continue
        recorded = (item["optimizer"], item["logical_operator"], item["decay_class"], item["logical_slices"])
        if recorded != (cls.family, cls.logical_operator, cls.decay_class, [s.as_dict() for s in cls.slices]):
            problems.append(f"{item['name']}: manifest optimizer classification differs from the taxonomy")
    return problems[:20]


def check_environment(bundle: Path) -> list[str]:
    reference = json.loads((bundle / "rng_reference.json").read_text())
    pinned = reference["environment"]
    current = bundle_mod.environment_record()
    problems = []
    if platform.python_version() != pinned["python"]:
        problems.append(f"python {platform.python_version()} != pinned {pinned['python']}")
    for name, version in pinned["packages"].items():
        if current["packages"].get(name) != version:
            problems.append(f"{name} {current['packages'].get(name)} != pinned {version}")
    problems += bundle_mod.check_rng_reference(reference["vectors"])
    return problems


def check_train_config(path: Path | None, config: FlashMini50BConfig) -> tuple[list[str], Any]:
    from .v4_train import TrainConfig

    if path is None:
        return ["no --train-config given; mandatory training hyperparameters unresolved"], None
    try:
        train = TrainConfig.load(path)
    except (ValueError, FileNotFoundError) as exc:
        return [str(exc)], None
    problems = []
    if train.get("run.purpose") != "production":
        problems.append("train config run.purpose must be 'production' for donor training")
    if train.get("tokenizer.fingerprint") != config.section("tokenizer")["fingerprint"]:
        problems.append("train config tokenizer.fingerprint differs from the frozen tokenizer")
    from .v4_data import load_manifest

    for phase in train.phases:
        try:
            load_manifest(train.resolve(phase.manifest), expected_fingerprint=config.section("tokenizer")["fingerprint"], vocab_size=config.vocab_size)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            problems.append(f"data phase {phase.name}: {exc}")
    return problems, train


def check_ple_storage(train, config: FlashMini50BConfig, *, available_bytes: int | None = None,
                      shm_free_bytes: int | None = None) -> list[str]:
    from .v4_ple_store import available_host_bytes, host_memory_requirement

    ple = config.section("ple")
    world = int(train.get("distributed.nodes")) * int(train.get("distributed.gpus_per_node"))
    backing = train.get("distributed.ple_backing")
    need = host_memory_requirement(ple["hash_head_rows"], ple["head_dim"], world_size=world,
                                   ranks_per_node=int(train.get("distributed.gpus_per_node")), backing=backing)
    available = available_host_bytes() if available_bytes is None else available_bytes
    problems = []
    if need["node_total_bytes"] > available:
        problems.append(f"PLE storage unavailable: node needs {need['node_total_bytes']:,} host bytes, {available:,} available")
    if backing == "shm":
        directory = Path(train.get("distributed.ple_shm_dir"))
        free = shutil.disk_usage(directory if directory.exists() else directory.parent).free if shm_free_bytes is None else shm_free_bytes
        if need["table_bytes"] > free:
            problems.append(f"PLE storage unavailable: {directory} has {free:,} bytes free, tables need {need['table_bytes']:,}")
    return problems


def check_topology(train, config: FlashMini50BConfig, *, cuda_devices: int | None = None,
                   device_memory_bytes: int | None = None, world_env: int | None = None) -> list[str]:
    problems = []
    nodes, per_node = int(train.get("distributed.nodes")), int(train.get("distributed.gpus_per_node"))
    world = nodes * per_node
    strategy = train.get("distributed.strategy")
    devices = torch.cuda.device_count() if cuda_devices is None else cuda_devices
    if devices == 0:
        problems.append("unsupported environment: no CUDA devices visible")
    elif devices < per_node:
        problems.append(f"incompatible topology: {devices} CUDA devices visible, gpus_per_node={per_node}")
    env_world = world_env if world_env is not None else (int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else None)
    if env_world is not None and env_world != world:
        problems.append(f"incompatible topology: WORLD_SIZE {env_world} != nodes*gpus_per_node {world}")
    if strategy != "fsdp2":
        problems.append(f"incompatible topology: strategy {strategy!r} cannot hold 50B dense state; use fsdp2")
    report = parameter_report(config)
    dense = report["checkpoint_total_learned_parameters"] - sum(config.section("ple")["hash_head_rows"]) * config.section("ple")["head_dim"]
    # fp32 master + fp32 grad + two fp32 moments (upper bound; Muon keeps one).
    sharded_state = dense * 16 // world
    largest_layer = max(sum(item["numel"] for item in report["tensors"] if item["name"].startswith(f"blocks.{i}.")) for i in range(config.num_layers))
    needed = sharded_state + 2 * largest_layer * 2
    memory = device_memory_bytes
    if memory is None and devices:
        memory = torch.cuda.get_device_properties(0).total_memory
    if memory is not None and needed > 0.8 * memory:
        problems.append(f"incompatible topology: ~{needed:,} bytes/GPU of sharded state exceeds 80% of {memory:,}")
    return problems


def run(bundle: Path, checkpoint: Path | None, train_config: Path | None, *, scope: str = "full",
        hash_shards: bool = True, config_path: Path | None = None) -> tuple[bool, list[Check]]:
    config = load_config(config_path or (REPO_ROOT / "configs/flashmini/flashmini_50b_base_init_v1.yaml"))
    checks: list[Check] = []

    def add(name: str, fn: Callable[[], list[str]]) -> None:
        try:
            checks.append(Check(name, fn()))
        except Exception as exc:  # any crash is a failed check, never a pass
            checks.append(Check(name, [f"{type(exc).__name__}: {exc}"]))

    add("source_provenance", lambda: check_source(bundle))
    add("config_and_architecture_sha", lambda: check_identity(bundle, config))
    add("tokenizer", lambda: check_tokenizer(config))
    add("parameter_count", lambda: check_parameters(bundle, config))
    add("optimizer_classification", lambda: check_optimizer_classification(bundle, config))
    add("environment", lambda: check_environment(bundle))

    def shards() -> list[str]:
        if checkpoint is None:
            return ["no --checkpoint given; checkpoint shards missing"]
        if not (bundle / bundle_mod.EXPECTED_HASHES).exists():
            return [f"bundle has no {bundle_mod.EXPECTED_HASHES}"]
        result = bundle_mod.verify_checkpoint(bundle, checkpoint, hash_shards=hash_shards,
                                              recompute=("blocks.0.mixer.A_log", "ple.conv1d.weight", "mtp.fusion.fc_hidden.weight"))
        return result["problems"]

    add("checkpoint_shards", shards)
    train_problems, train = check_train_config(train_config, config)
    checks.append(Check("training_hyperparameters", train_problems))
    if scope == "full":
        if train is None:
            checks.append(Check("ple_storage", ["train config unavailable"]))
            checks.append(Check("distributed_topology", ["train config unavailable"]))
        else:
            add("ple_storage", lambda: check_ple_storage(train, config))
            add("distributed_topology", lambda: check_topology(train, config))
    return all(check.ok for check in checks), checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flashmini.v4_preflight")
    parser.add_argument("--bundle", default=str(bundle_mod.DEFAULT_BUNDLE))
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-config")
    parser.add_argument("--scope", choices=("full", "artifacts"), default="full")
    parser.add_argument("--skip-shard-hashes", action="store_true", help="shape/key checks only (not a PASS for donor use)")
    parser.add_argument("--json", help="write the check report here")
    args = parser.parse_args(argv)
    ok, checks = run(Path(args.bundle), Path(args.checkpoint) if args.checkpoint else None,
                     Path(args.train_config) if args.train_config else None, scope=args.scope,
                     hash_shards=not args.skip_shard_hashes)
    if args.skip_shard_hashes:
        checks.append(Check("shard_hash_policy", ["shard hashes were skipped"]))
        ok = False
    for check in checks:
        print(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}")
        for problem in check.problems:
            print(f"    - {problem}")
    if args.json:
        Path(args.json).write_text(json.dumps({"ok": ok, "scope": args.scope, "checks": [{"name": c.name, "ok": c.ok, "problems": c.problems} for c in checks]}, indent=2) + "\n")
    if ok:
        print(PASS_LINE if args.scope == "full" else ARTIFACT_PASS_LINE)
        return 0
    print(FAIL_LINE + ": " + ", ".join(check.name for check in checks if not check.ok))
    return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ARTIFACT_PASS_LINE", "FROZEN_PARAMETER_COUNTS", "PASS_LINE", "check_environment", "check_identity",
           "check_optimizer_classification", "check_parameters", "check_ple_storage", "check_source",
           "check_source_and_identity", "check_tokenizer", "check_topology", "check_train_config", "run"]
