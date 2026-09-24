"""Deterministic v4 InitSpec, tensor manifest, shard plan, materializer, and donor bundle.

Provenance: ``source_commit`` is the commit containing the implementation.  The
bundle directory is generated from that commit and committed afterwards, so the
bundle can never name the commit that contains it.  Identity is instead proven by
``source_tree_sha256`` - a SHA-256 over the ``git ls-tree -r`` listing (mode, blob
SHA, path) of every tracked file outside ``flashmini_50b_base_init_v1/`` - which
preflight recomputes for the checked-out ``HEAD`` and compares, and by requiring
that ``HEAD`` differs from ``source_commit`` only inside the bundle directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

from . import v4_init
from .base_init_accounting import build_meta_model, parameter_report
from .base_init_config import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_ID,
    DEFAULT_CONFIG_PATH,
    MODEL_ID,
    FlashMini50BConfig,
    load_config,
)
from .base_init_optimizer import OptimizerTaxonomy, optimizer_contract

SCHEMA_VERSION = 2
BUNDLE_DIRNAME = "flashmini_50b_base_init_v1"
DEFAULT_BUNDLE = Path(BUNDLE_DIRNAME)
DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"
SAFETENSORS_DTYPE = {"bfloat16": "BF16"}
EXPECTED_HASHES = "shard_hashes.json"

# Exact tested environment.  Initialization identity depends on torch's CPU
# Philox/Box-Muller output; preflight re-derives the RNG reference vectors.
PINNED_PACKAGES = ("torch", "safetensors", "PyYAML", "tokenizers", "numpy")
PINNED_PYTHON = "3.13"

RNG_REFERENCE_TENSORS = (
    "blocks.0.mixer.A_log",
    "blocks.0.mixer.conv1d.weight",
    "blocks.3.mixer.q_proj.weight",
    "blocks.0.moe.router.weight",
    "ple.tables.0.weight",
    "mtp.fusion.fc_hidden.weight",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def git_sha(repo_root: Path, ref: str = "HEAD") -> str:
    return _git(repo_root, "rev-parse", ref).strip()


def source_tree_sha256(repo_root: Path, commit: str) -> str:
    """SHA-256 of the tracked tree at ``commit`` excluding the generated bundle."""
    listing = _git(repo_root, "ls-tree", "-r", "--full-tree", commit)
    kept = [line for line in listing.splitlines() if not line.split("\t", 1)[1].startswith(BUNDLE_DIRNAME + "/")]
    return hashlib.sha256(("\n".join(kept) + "\n").encode()).hexdigest()


def source_provenance(repo_root: Path, commit: str) -> dict[str, Any]:
    commit = git_sha(repo_root, commit)
    return {
        "source_commit": commit,
        "source_tree_sha256": source_tree_sha256(repo_root, commit),
        "source_tree_algorithm": "sha256(git ls-tree -r --full-tree <commit>, excluding flashmini_50b_base_init_v1/)",
        "bundle_directory": BUNDLE_DIRNAME,
        "rule": "HEAD must equal source_commit outside flashmini_50b_base_init_v1/ and reproduce source_tree_sha256",
    }


def environment_record() -> dict[str, Any]:
    packages = {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}
    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "materialization_device": "cpu",
        "materialization_depends_on_cuda": False,
        "external_muon_package": None,
    }


def environment_lock_text(record: dict[str, Any]) -> str:
    packages = record["packages"]
    torch_version = packages["torch"]
    cuda_tag = torch_version.split("+", 1)[1] if "+" in torch_version else None
    lines = [
        "# FlashMini-50B v4 exact tested environment (pip requirements format).",
        f"# python=={record['python']}",
        f"# torch CUDA runtime: {record['torch_cuda']}; initialization runs on CPU and does not depend on CUDA.",
        "# Muon is implemented in-repo (flashmini.v4_optim); no external Muon package.",
    ]
    if cuda_tag:
        lines.append(f"--extra-index-url https://download.pytorch.org/whl/{cuda_tag}")
    lines += [f"{name}=={version}" for name, version in packages.items()]
    return "\n".join(lines) + "\n"


def build_manifest(config: FlashMini50BConfig) -> dict[str, Any]:
    """Complete tensor manifest from the meta model, without allocating storage."""
    model = build_meta_model(config)
    report = parameter_report(config, model)
    taxonomy = OptimizerTaxonomy(config)
    global_seed = int(config.section("initialization")["global_seed"])
    if global_seed != v4_init.GLOBAL_SEED:
        raise ValueError(f"config global seed {global_seed} != InitSpec seed {v4_init.GLOBAL_SEED}")
    tensors = []
    for item in report["tensors"]:
        name = item["name"]
        cls = taxonomy.classify(name, tuple(item["shape"]))
        tensors.append({
            "name": name,
            "shape": item["shape"],
            "dtype": DTYPE_NAME,
            "numel": item["numel"],
            "category": item["category"],
            "optimizer": cls.family,
            "logical_operator": cls.logical_operator,
            "decay_class": cls.decay_class,
            "logical_slices": [slice_.as_dict() for slice_ in cls.slices],
            "init_law": v4_init.init_law(name),
            "init_name_seed": {
                "algorithm": v4_init.SEED_ALGORITHM,
                "global_seed": global_seed,
                "chunk_elements": v4_init.CHUNK_ELEMENTS,
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "architecture_version": ARCHITECTURE_VERSION,
        "config_sha256": config.config_sha256,
        "architecture_sha256": config.architecture_sha256,
        "tensors": tensors,
        "tensor_count": len(tensors),
        "base_tensor_count": sum(not item["name"].startswith("mtp.") for item in tensors),
        "mtp_tensor_count": sum(item["name"].startswith("mtp.") for item in tensors),
        "parameter_counts": {
            "base": report["base_total_learned_parameters"],
            "mtp": report["mtp_total_learned_parameters"],
            "total": report["checkpoint_total_learned_parameters"],
            "base_active_per_token": report["base_active_parameters_per_token"],
        },
    }


def plan_shards(manifest: dict[str, Any], *, target_shard_bytes: int = 4 * 1024**3) -> dict[str, Any]:
    """Deterministic shard plan by sorted name; each PLE hash head gets its own shard."""
    target = int(target_shard_bytes)
    if target <= 0:
        raise ValueError("target_shard_bytes must be positive")
    shards: list[dict[str, Any]] = []
    current: list[str] = []
    current_bytes = 0

    def emit(names: list[str], size: int, kind: str) -> None:
        shards.append({"shard": f"model-{len(shards) + 1:05d}.safetensors", "tensors": names, "bytes": size, "kind": kind})

    for item in sorted(manifest["tensors"], key=lambda value: value["name"]):
        size = item["numel"] * 2
        if item["name"].startswith("ple.tables."):
            emit([item["name"]], size, "ple_hash_head")
            continue
        if size > target:
            raise ValueError(f"tensor {item['name']} ({size} bytes) exceeds target shard size")
        if current and current_bytes + size > target:
            emit(current, current_bytes, "dense_or_expert")
            current, current_bytes = [], 0
        current.append(item["name"])
        current_bytes += size
    if current:
        emit(current, current_bytes, "dense_or_expert")
    for shard in shards:
        shard["contains_mtp"] = any(name.startswith("mtp.") for name in shard["tensors"])
        rules = sorted({v4_init.init_law(name)["rule"] for name in shard["tensors"]})
        shard["init_rules"] = rules
    return {"schema_version": SCHEMA_VERSION, "target_shard_bytes": target, "shards": shards,
            "shard_count": len(shards), "planned_bytes": sum(item["bytes"] for item in shards)}


def materialize_tensor(item: dict[str, Any]) -> torch.Tensor:
    """One tensor, bit-identical to ``FlashMini50BBaseInit.initialize`` for the same name."""
    law = v4_init.init_law(item["name"])
    if item["init_law"] != law:
        raise ValueError(f"{item['name']}: manifest init law {item['init_law']} != InitSpec {law}")
    if item["dtype"] != DTYPE_NAME:
        raise ValueError(f"{item['name']}: unsupported dtype {item['dtype']}")
    return v4_init.materialize(item["name"], tuple(item["shape"]), law)


def _safetensors_header(items: list[dict[str, Any]], metadata: dict[str, str]) -> bytes:
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for item in items:
        size = item["numel"] * 2
        header[item["name"]] = {"dtype": SAFETENSORS_DTYPE[item["dtype"]], "shape": item["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    raw += b" " * ((8 - (8 + len(raw)) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw


def write_shard(path: Path, items: list[dict[str, Any]], *, metadata: dict[str, str]) -> str:
    """Stream a safetensors file one tensor at a time; returns its SHA-256."""
    digest = hashlib.sha256()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        header = _safetensors_header(items, metadata)
        handle.write(header)
        digest.update(header)
        for item in items:
            data = materialize_tensor(item).contiguous().view(torch.int16).numpy().tobytes()
            handle.write(data)
            digest.update(data)
            del data
    tmp.replace(path)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def materialize(bundle: Path | str, output: Path | str, *, only_shards: Iterable[int] | None = None,
                require_expected: bool = True) -> dict[str, Any]:
    """Materialize planned shards one at a time and check them against expected hashes."""
    bundle, output = Path(bundle), Path(output)
    manifest = _load_json(bundle / "tensor_manifest.json")
    plan = _load_json(bundle / "shard_plan.json")
    expected_path = bundle / EXPECTED_HASHES
    expected = _load_json(expected_path)["shards"] if expected_path.exists() else None
    if require_expected and expected is None:
        raise FileNotFoundError(f"{expected_path} missing; pass require_expected=False only when recording hashes")
    by_name = {item["name"]: item for item in manifest["tensors"]}
    output.mkdir(parents=True, exist_ok=True)
    selected = set(only_shards) if only_shards is not None else None
    emitted, mismatches = [], []
    metadata = {"format": "pt", "model_id": MODEL_ID, "checkpoint_id": CHECKPOINT_ID,
                "architecture_sha256": manifest["architecture_sha256"]}
    for index, shard in enumerate(plan["shards"]):
        if selected is not None and index not in selected:
            continue
        path = output / shard["shard"]
        digest = write_shard(path, [by_name[name] for name in shard["tensors"]], metadata=metadata)
        record = {"index": index, "shard": shard["shard"], "sha256": digest, "bytes": path.stat().st_size}
        if expected is not None and expected[shard["shard"]] != digest:
            mismatches.append(record)
        emitted.append(record)
    index_payload = {
        "metadata": {"total_size": plan["planned_bytes"], "total_parameters": manifest["parameter_counts"]["total"],
                     "architecture_sha256": manifest["architecture_sha256"]},
        "weight_map": {name: shard["shard"] for shard in plan["shards"] for name in shard["tensors"]},
    }
    (output / "model.safetensors.index.json").write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
    receipt = {"emitted": emitted, "all_shards_emitted": len(emitted) == plan["shard_count"],
               "expected_hashes_checked": expected is not None, "mismatches": mismatches}
    (output / "materialization_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if mismatches:
        raise ValueError(f"materialized shard hash mismatch: {[item['shard'] for item in mismatches]}")
    return receipt


def verify_checkpoint(bundle: Path | str, checkpoint: Path | str, *, recompute: Iterable[str] = (),
                      hash_shards: bool = True) -> dict[str, Any]:
    """Fail-closed integrity check of a materialized checkpoint directory."""
    from safetensors import safe_open

    bundle, checkpoint = Path(bundle), Path(checkpoint)
    manifest = _load_json(bundle / "tensor_manifest.json")
    plan = _load_json(bundle / "shard_plan.json")
    expected = _load_json(bundle / EXPECTED_HASHES)["shards"]
    by_name = {item["name"]: item for item in manifest["tensors"]}
    seen: set[str] = set()
    problems: list[str] = []
    for shard in plan["shards"]:
        path = checkpoint / shard["shard"]
        if not path.exists():
            problems.append(f"missing shard {shard['shard']}")
            continue
        if hash_shards and sha256_file(path) != expected[shard["shard"]]:
            problems.append(f"sha256 mismatch {shard['shard']}")
        with safe_open(str(path), framework="pt") as handle:
            keys = list(handle.keys())
            if sorted(keys) != sorted(shard["tensors"]) or len(set(keys)) != len(keys):
                problems.append(f"{shard['shard']}: key set differs from plan")
            for key in keys:
                if key in seen:
                    problems.append(f"duplicate tensor {key}")
                seen.add(key)
                sliced = handle.get_slice(key)
                if list(sliced.get_shape()) != by_name[key]["shape"] or sliced.get_dtype() != "BF16":
                    problems.append(f"{key}: shape/dtype {sliced.get_shape()}/{sliced.get_dtype()} != manifest")
    if seen != set(by_name):
        problems.append(f"tensor coverage differs: missing={sorted(set(by_name) - seen)[:5]}")
    weight_map = {name: shard["shard"] for shard in plan["shards"] for name in shard["tensors"]}
    for name in recompute:
        with safe_open(str(checkpoint / weight_map[name]), framework="pt") as handle:
            if not torch.equal(handle.get_tensor(name).view(torch.int16), materialize_tensor(by_name[name]).view(torch.int16)):
                problems.append(f"{name}: stored tensor differs from InitSpec re-derivation")
    total = sum(by_name[name]["numel"] for name in seen & set(by_name))
    if total != manifest["parameter_counts"]["total"]:
        problems.append(f"parameter count {total} != {manifest['parameter_counts']['total']}")
    return {"ok": not problems, "problems": problems, "tensors": len(seen), "parameters": total,
            "shards_checked": plan["shard_count"], "hashes_checked": hash_shards, "recomputed": list(recompute)}


def rng_reference_vectors(config: FlashMini50BConfig, count: int = 16) -> list[dict[str, Any]]:
    report = {item["name"]: item for item in parameter_report(config)["tensors"]}
    vectors = []
    for name in RNG_REFERENCE_TENSORS:
        shape = report[name]["shape"]
        law = v4_init.init_law(name)
        first = v4_init.materialize(name, (min(count, report[name]["numel"]),), law)
        vectors.append({
            "name": name, "shape": shape, "rule": law["rule"],
            "chunk0_seed": v4_init.name_seed(name, 0),
            "first_bf16_bits": [int(v) & 0xFFFF for v in first.view(torch.int16).tolist()],
        })
    small = v4_init.materialize("blocks.0.mixer.A_log", tuple(report["blocks.0.mixer.A_log"]["shape"]), v4_init.init_law("blocks.0.mixer.A_log"))
    vectors.append({"name": "blocks.0.mixer.A_log", "full_tensor_sha256": hashlib.sha256(small.view(torch.int16).numpy().tobytes()).hexdigest()})
    return vectors


def check_rng_reference(vectors: list[dict[str, Any]]) -> list[str]:
    problems = []
    for item in vectors:
        name = item["name"]
        law = v4_init.init_law(name)
        if "first_bf16_bits" in item:
            first = v4_init.materialize(name, (len(item["first_bf16_bits"]),), law)
            if [int(v) & 0xFFFF for v in first.view(torch.int16).tolist()] != item["first_bf16_bits"]:
                problems.append(f"RNG reference drift for {name}")
        else:
            full = v4_init.materialize(name, tuple(_shape_of(name, vectors)), law)
            if hashlib.sha256(full.view(torch.int16).numpy().tobytes()).hexdigest() != item["full_tensor_sha256"]:
                problems.append(f"RNG full-tensor drift for {name}")
    return problems


def _shape_of(name: str, vectors: list[dict[str, Any]]) -> list[int]:
    return next(item["shape"] for item in vectors if item["name"] == name and "shape" in item)


def init_spec(config: FlashMini50BConfig, manifest: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "architecture_version": ARCHITECTURE_VERSION,
        "source_git_sha": provenance["source_commit"],
        "source_provenance": provenance,
        "config_sha256": config.config_sha256,
        "architecture_sha256": config.architecture_sha256,
        "tokenizer": config.section("tokenizer"),
        "global_initialization_seed": v4_init.GLOBAL_SEED,
        "seed_algorithm": v4_init.SEED_ALGORITHM,
        "chunk_elements": v4_init.CHUNK_ELEMENTS,
        "generator": "torch.Generator(device='cpu') per (name, chunk); float32 draw then round to bfloat16",
        "dtype": DTYPE_NAME,
        "init_rules": [{"rule": rule, "pattern": pattern.pattern, **law} for rule, pattern, law in v4_init.INIT_RULES],
        "optimizer_contract": optimizer_contract(),
        "parameter_counts": manifest["parameter_counts"],
        "shard_mapping": "model.safetensors.index.json",
        "tensor_manifest": "tensor_manifest.json",
        "expected_shard_hashes": EXPECTED_HASHES,
        "materialization": "python -m flashmini.base_init_bundle materialize --bundle flashmini_50b_base_init_v1 --output <checkpoint-dir>",
    }


def _docs(manifest: dict[str, Any], plan: dict[str, Any], provenance: dict[str, Any], config: FlashMini50BConfig) -> dict[str, str]:
    counts = manifest["parameter_counts"]
    tokenizer = config.section("tokenizer")
    return {
        "architecture.md": f"""# FlashMini-50B-Base architecture v4

`FlashMini-50B-Base` / `FlashMini-50B-Base-Init-v1`. The canonical YAML
(`flashmini_50b_base_init_v1.yaml`) is the source of truth; `architecture_sha256`
({config.architecture_sha256}) covers every geometry field and excludes
training, inference, storage and provenance metadata.

- vocab {config.vocab_size}, d_model {config.d_model}, {config.num_layers} layers (38 Gated DeltaNet, 10 attention at 1-based 4,8,12,16,20,24,29,34,39,44)
- attention: 16 Q / 2 KV heads, head_dim 256, partial RoPE 64/256, context 262144, theta 1e7; five KV-cache pairs (reuse layers have no K/V tensors)
- MoE: 80 routed + 1 sigmoid-gated shared SwiGLU-1280 expert, top-6; balancing is global/logical-batch
- hyper-connections: 4 streams, low-rank 256
- PLE: injection at 0-based layer 1; 8 bigram + 8 trigram hash heads over segment-local `(x_t, x_(t-1), x_(t-2))`, EOS reset; tables on host
- MTP: one layer, recursive depths 1..3 (window base + 3), shares embedding and LM head
- untied input embedding and LM head; base count excludes MTP

Parameters: base {counts['base']:,}; MTP {counts['mtp']:,}; total {counts['total']:,}; active base per token {counts['base_active_per_token']:,}.

GDN short convolution is additive (`x + conv(x)`) as frozen in the v4 design, not Qwen's `silu(conv(x))`.
""",
        "training_recipe.md": f"""# v4 training recipe (implemented by `python -m flashmini.v4_train`)

Loss: `total = 1.0 * main + mtp_coefficient * mean(MTP t+2, t+3, t+4) + router_aux_coefficient * router_aux`.
`mtp_coefficient` is 0.30 while fewer than 70% of the planned tokens have been
consumed and 0.10 afterwards. Every term is logged separately with the total.

MTP uses ground-truth teacher forcing: depth k at position t embeds `x_(t+k)` and
predicts `x_(t+k+1)`; no sampled rollout.

Router balancing: Qwen global-batch load-balancing `E * sum_i f_i P_i` per logical
MoE layer, averaged over layers; expert counts are all-reduced across data-parallel
ranks and accumulated over gradient-accumulation microbatches
(`exact_prepass` gives the exact logical-batch gradient; `ga_buffer` is Qwen's buffer).
The coefficient has no default; startup fails until it is set.

Optimizers: Muon (Newton-Schulz, 8 steps, per logical operator slice) for hidden
matrices; AdamW for embeddings, LM head, router/gates/HC control matrices (with
explicit weight decay) and norms/GDN scalars/convs (no decay); row-sparse Adam
(weight decay 0) for the host-resident PLE tables. Hyperparameters are donor inputs.

Donor inputs that must be set explicitly (startup fails on null): nodes/GPUs,
micro-batch, gradient accumulation, curriculum phases (sequence length, token
budget, data manifest), teacher mixture, planned total tokens, learning rates and
schedule, router coefficient and balance mode. See `train_example.yaml`.
""",
        "donor_handoff.md": f"""# Donor handoff

**BASE PARAMETER COUNT:** {counts['base']:,}
**MTP PARAMETER COUNT:** {counts['mtp']:,}
**TOTAL CHECKPOINT PARAMS:** {counts['total']:,}

Source commit: `{provenance['source_commit']}` (tree `{provenance['source_tree_sha256']}`).
Tokenizer fingerprint: `{tokenizer['fingerprint']}`.

1. Check out `main` at or after the source commit; outside `{BUNDLE_DIRNAME}/` it must equal the source commit.
2. Create the environment from `environment.lock` (Python {PINNED_PYTHON}).
3. Materialize all {plan['shard_count']} shards ({plan['planned_bytes']:,} bytes): see `materialization_command.txt`.
   The first run uses `--no-expected-hashes` and then `record-hashes`. After
   `shard_hashes.json` exists, rematerialize once so every shard is checked.
4. Run preflight: see `preflight_command.txt`. It must end with `FLASHMINI V4 TRAINING PREFLIGHT: PASS`.
5. Fill every null in a copy of `train_example.yaml`, then launch with `training_launch_command.txt`.
6. Resume: rerun the same launch command; the runner restores `checkpoint.dir/latest`.

No training has been run. No optimizer state is included.
""",
    }


def write_bundle(destination: Path | str = DEFAULT_BUNDLE, *, config_path: Path | str = DEFAULT_CONFIG_PATH,
                 repo_root: Path | str | None = None, source_commit: str = "HEAD",
                 validation_report: Path | str | None = None) -> dict[str, Any]:
    destination = Path(destination)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    provenance = source_provenance(repo_root, source_commit)
    manifest = build_manifest(config)
    plan = plan_shards(manifest, target_shard_bytes=config.section("storage")["target_shard_bytes"])
    environment = environment_record()
    shutil.copy2(config_path, destination / "flashmini_50b_base_init_v1.yaml")
    tokenizer_manifest = repo_root / config.section("tokenizer")["manifest"]
    shutil.copy2(tokenizer_manifest, destination / "tokenizer_manifest.json")
    payloads = {
        "init_spec.json": init_spec(config, manifest, provenance),
        "tensor_manifest.json": manifest,
        "parameter_report.json": parameter_report(config),
        "shard_plan.json": plan,
        "source_provenance.json": provenance,
        "rng_reference.json": {"environment": environment, "vectors": rng_reference_vectors(config)},
    }
    for name, payload in payloads.items():
        (destination / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for name, text in _docs(manifest, plan, provenance, config).items():
        (destination / name).write_text(text)
    (destination / "source_git_sha.txt").write_text(provenance["source_commit"] + "\n")
    (destination / "environment.lock").write_text(environment_lock_text(environment))
    (destination / "materialization_command.txt").write_text(
        "# First run on a host that does not yet have shard_hashes.json:\n"
        "python -m flashmini.base_init_bundle materialize --bundle flashmini_50b_base_init_v1 --output /path/to/flashmini_50b_init_checkpoint --no-expected-hashes\n"
        "python -m flashmini.base_init_bundle record-hashes --bundle flashmini_50b_base_init_v1 --receipt /path/to/flashmini_50b_init_checkpoint/materialization_receipt.json\n"
        "# Subsequent runs check every shard against shard_hashes.json:\n"
        "python -m flashmini.base_init_bundle materialize --bundle flashmini_50b_base_init_v1 --output /path/to/flashmini_50b_init_checkpoint\n")
    (destination / "preflight_command.txt").write_text(
        "python -m flashmini.v4_preflight --bundle flashmini_50b_base_init_v1 --checkpoint /path/to/flashmini_50b_init_checkpoint --train-config /path/to/train.yaml\n")
    (destination / "training_launch_command.txt").write_text(
        "torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node-rank $NODE_RANK --rdzv-backend c10d --rdzv-endpoint $MASTER_ADDR:29500 -m flashmini.v4_train --config /path/to/train.yaml\n")
    shutil.copy2(repo_root / "configs/flashmini/v4_train_example.yaml", destination / "train_example.yaml")
    (destination / "donor_smoke_test_command.txt").unlink(missing_ok=True)
    if validation_report is not None:
        shutil.copy2(validation_report, destination / "validation_report.json")
    return {"bundle": str(destination), "config_sha256": config.config_sha256, "architecture_sha256": config.architecture_sha256,
            "shard_count": plan["shard_count"], "planned_bytes": plan["planned_bytes"], **provenance}


def record_expected_hashes(bundle: Path | str, receipt_path: Path | str) -> dict[str, Any]:
    """Record per-shard expected SHA-256 from a complete materialization receipt."""
    bundle = Path(bundle)
    receipt = _load_json(Path(receipt_path))
    plan = _load_json(bundle / "shard_plan.json")
    if not receipt["all_shards_emitted"]:
        raise ValueError("expected hashes can only be recorded from a complete materialization")
    shards = {item["shard"]: item["sha256"] for item in receipt["emitted"]}
    if set(shards) != {shard["shard"] for shard in plan["shards"]}:
        raise ValueError("receipt does not cover the shard plan")
    manifest = _load_json(bundle / "tensor_manifest.json")
    payload = {"architecture_sha256": manifest["architecture_sha256"], "shards": shards,
               "total_bytes": sum(item["bytes"] for item in receipt["emitted"])}
    (bundle / EXPECTED_HASHES).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flashmini.base_init_bundle")
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--destination", default=str(DEFAULT_BUNDLE))
    write.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    write.add_argument("--source-commit", default="HEAD")
    write.add_argument("--validation-report")
    mat = sub.add_parser("materialize")
    mat.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    mat.add_argument("--output", required=True)
    mat.add_argument("--shards", help="comma-separated 0-based shard indices")
    mat.add_argument("--no-expected-hashes", action="store_true", help="only for the initial hash recording run")
    rec = sub.add_parser("record-hashes")
    rec.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    rec.add_argument("--receipt", required=True)
    ver = sub.add_parser("verify")
    ver.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ver.add_argument("--checkpoint", required=True)
    ver.add_argument("--recompute", default="", help="comma-separated tensor names to re-derive bitwise")
    args = parser.parse_args(argv)
    if args.command == "write":
        result = write_bundle(args.destination, config_path=args.config, source_commit=args.source_commit,
                              validation_report=args.validation_report)
    elif args.command == "materialize":
        shards = [int(value) for value in args.shards.split(",")] if args.shards else None
        result = materialize(args.bundle, args.output, only_shards=shards, require_expected=not args.no_expected_hashes)
    elif args.command == "record-hashes":
        result = record_expected_hashes(args.bundle, args.receipt)
    else:
        names = [value for value in args.recompute.split(",") if value]
        result = verify_checkpoint(args.bundle, args.checkpoint, recompute=names)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "build_manifest", "check_rng_reference", "environment_lock_text", "environment_record", "init_spec", "materialize",
    "materialize_tensor", "plan_shards", "record_expected_hashes", "rng_reference_vectors", "source_provenance",
    "source_tree_sha256", "verify_checkpoint", "write_bundle", "write_shard",
]
