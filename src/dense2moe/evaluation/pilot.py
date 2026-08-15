"""One-real-layer partition pilot against immutable source weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import MoEProfile
from ..partition import partition_indices
from ..partition.oracle import frozen_slice_simplex_oracle
from ..provenance import current_git_commit


def run_real_layer_pilot(source_dir: str | Path, profiles: list[MoEProfile], *, layer: int = 0, batch_size: int = 2, seed: int = 17) -> dict[str, Any]:
    """Measure dense-vs-initial-sparse error for one actual MLP layer.

    The pilot loads only the three layer-0 MLP tensors and never writes into
    the source directory.  It intentionally does not label the untrained
    sparse output as a quality result.
    """

    artifact_commit = current_git_commit()
    try:
        import torch  # type: ignore
    except ImportError:
        return _run_numpy_pilot(source_dir, profiles, layer=layer, batch_size=batch_size, seed=seed)
    try:
        import torch.nn.functional as F  # type: ignore
        from safetensors import safe_open  # type: ignore
    except ImportError as exc:
        return {"status": "BLOCKED", "reason": f"optional ML reader unavailable: {exc}", "code_commit": artifact_commit}
    source = Path(source_dir)
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        return {"status": "BLOCKED", "reason": "safetensors index is missing", "code_commit": artifact_commit}
    import json

    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key: value for key, value in index.get("weight_map", {}).items() if key.startswith(prefix)}
    required = {"down_proj.weight", "gate_proj.weight", "up_proj.weight"}
    if {key[len(prefix) :] for key in names} != required:
        return {"status": "BLOCKED", "reason": "layer MLP tensor inventory is incomplete", "found": sorted(names), "code_commit": artifact_commit}
    tensors: dict[str, torch.Tensor] = {}
    for short_name, shard in ((key[len(prefix) :], value) for key, value in names.items()):
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            tensors[short_name] = handle.get_tensor(prefix + short_name).float()
    hidden_size = int(tensors["gate_proj.weight"].shape[1])
    dense_intermediate = int(tensors["gate_proj.weight"].shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn((batch_size, hidden_size), generator=generator)
    dense_hidden = F.silu(inputs @ tensors["gate_proj.weight"].T) * (inputs @ tensors["up_proj.weight"].T)
    dense_output = dense_hidden @ tensors["down_proj.weight"].T
    dense_norm = float(torch.mean(dense_output.square()).item()) + 1e-12
    results: list[dict[str, Any]] = []
    for profile in profiles:
        plan = partition_indices(profile.dense_intermediate_size, profile.routed_experts, profile.expert_intermediate_size, profile.shared_intermediate_size)
        if profile.dense_intermediate_size != dense_intermediate:
            results.append({"profile": profile.name, "status": "incompatible", "reason": "source MLP width differs"})
            continue
        shared = plan.shared_indices
        routed = plan.expert_indices
        shared_output = dense_hidden[:, list(shared)] @ tensors["down_proj.weight"][:, list(shared)].T
        all_output = shared_output.clone()
        for group in routed:
            all_output = all_output + dense_hidden[:, list(group)] @ tensors["down_proj.weight"][:, list(group)].T
        exact_error = float(torch.mean((all_output - dense_output).square()).item())
        routed_outputs: list[torch.Tensor] = []
        for group in routed:
            routed_outputs.append(dense_hidden[:, list(group)] @ tensors["down_proj.weight"][:, list(group)].T)
        sparse_output = shared_output.clone()
        for group in routed[: profile.top_k]:
            sparse_output = sparse_output + dense_hidden[:, list(group)] @ tensors["down_proj.weight"][:, list(group)].T / profile.top_k
        relative_mse = float(torch.mean((sparse_output - dense_output).square()).item() / dense_norm)
        oracle = frozen_slice_simplex_oracle(shared_output.detach().cpu().numpy(), torch.stack(routed_outputs, dim=1).detach().cpu().numpy(), dense_output.detach().cpu().numpy(), top_k=profile.top_k)
        results.append({"profile": profile.name, "status": "measured", "classification": "RANDOM_INPUT_FROZEN_SLICE_DIAGNOSTIC", "layer": layer, "batch_size": batch_size, "dense_mse": 0.0, "all_expert_reconstruction_mse": exact_error, "initial_sparse_relative_mse": relative_mse, "oracle_sparse_normalized_mse": float(oracle["normalized_mse"]), "oracle_cosine": float(oracle["cosine"]), "source_tensor_shard": next(iter(names.values()))})
    return {
        "status": "PILOT_COMPLETE",
        "classification": "RANDOM_INPUT_FROZEN_SLICE_DIAGNOSTIC",
        "quality_gate_eligible": False,
        "source_dir": str(source),
        "layer": layer,
        "results": results,
        "code_commit": artifact_commit,
        "note": "historical random-input frozen-slice diagnostic; not a retained-quality claim or trainable-MoE ceiling",
    }


def _run_numpy_pilot(source_dir: str | Path, profiles: list[MoEProfile], *, layer: int, batch_size: int, seed: int) -> dict[str, Any]:
    """Real-weight pilot fallback for CPU environments without PyTorch."""

    import json

    artifact_commit = current_git_commit()

    import numpy as np  # type: ignore
    from safetensors import safe_open  # type: ignore

    source = Path(source_dir)
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        return {"status": "BLOCKED", "reason": "safetensors index is missing", "code_commit": artifact_commit}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: value for key, value in index.get("weight_map", {}).items() if key.startswith(prefix)}
    required = {"down_proj.weight", "gate_proj.weight", "up_proj.weight"}
    if set(names) != required:
        return {"status": "BLOCKED", "reason": "layer MLP tensor inventory is incomplete", "found": sorted(names), "code_commit": artifact_commit}
    tensors: dict[str, Any] = {}
    for short_name, shard in names.items():
        with safe_open(str(source / shard), framework="numpy") as handle:
            tensors[short_name] = np.asarray(handle.get_tensor(prefix + short_name), dtype=np.float32)
    hidden_size = int(tensors["gate_proj.weight"].shape[1])
    dense_intermediate = int(tensors["gate_proj.weight"].shape[0])
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, 0.02, size=(batch_size, hidden_size)).astype(np.float32)
    gate_values = inputs @ tensors["gate_proj.weight"].T
    dense_hidden = (gate_values / (1.0 + np.exp(-gate_values))) * (inputs @ tensors["up_proj.weight"].T)
    dense_output = dense_hidden @ tensors["down_proj.weight"].T
    dense_norm = float(np.mean(dense_output**2)) + 1e-12
    from ..partition import frozen_slice_simplex_oracle, partition_indices, swiglu_contributions

    results: list[dict[str, Any]] = []
    for profile in profiles:
        if profile.dense_intermediate_size != dense_intermediate:
            results.append({"profile": profile.name, "status": "incompatible", "reason": "source MLP width differs"})
            continue
        plan = partition_indices(profile.dense_intermediate_size, profile.routed_experts, profile.expert_intermediate_size, profile.shared_intermediate_size)
        shared, routed = swiglu_contributions(inputs, tensors["gate_proj.weight"], tensors["up_proj.weight"], tensors["down_proj.weight"], plan)
        all_output = shared + routed.sum(axis=1)
        oracle = frozen_slice_simplex_oracle(shared, routed, dense_output, top_k=profile.top_k)
        sparse_ids = np.tile(np.arange(profile.top_k), (batch_size, 1))
        sparse_output = shared.copy()
        for token in range(batch_size):
            for slot in range(profile.top_k):
                sparse_output[token] += routed[token, sparse_ids[token, slot]] / profile.top_k
        results.append({
            "profile": profile.name,
            "status": "measured",
            "classification": "RANDOM_INPUT_FROZEN_SLICE_DIAGNOSTIC",
            "layer": layer,
            "batch_size": batch_size,
            "dense_mse": 0.0,
            "all_expert_reconstruction_mse": float(np.mean((all_output - dense_output) ** 2)),
            "initial_sparse_relative_mse": float(np.mean((sparse_output - dense_output) ** 2) / dense_norm),
            "oracle_sparse_normalized_mse": float(oracle["normalized_mse"]),
            "oracle_cosine": float(oracle["cosine"]),
            "source_tensor_shard": next(iter(names.values())),
            "backend": "numpy-safetensors",
        })
    return {
        "status": "PILOT_COMPLETE",
        "classification": "RANDOM_INPUT_FROZEN_SLICE_DIAGNOSTIC",
        "quality_gate_eligible": False,
        "source_dir": str(source),
        "layer": layer,
        "results": results,
        "code_commit": artifact_commit,
        "note": "historical random-input frozen-slice diagnostic; not a retained-quality claim or trainable-MoE ceiling",
    }


def run_oracle_ablation(source_dir: str | Path, *, layer: int = 0, batch_size: int = 8, seed: int = 17) -> dict[str, Any]:
    """Run the bounded p8 oracle fallback matrix on one immutable real layer."""

    import json

    artifact_commit = current_git_commit()

    import numpy as np  # type: ignore
    from safetensors import safe_open  # type: ignore

    source = Path(source_dir)
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    values: dict[str, Any] = {}
    for name, shard in names.items():
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            values[name] = handle.get_tensor(prefix + name).float().numpy()
    hidden_size = int(values["gate_proj.weight"].shape[1])
    dense_size = int(values["gate_proj.weight"].shape[0])
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, 0.02, size=(batch_size, hidden_size)).astype(np.float32)
    gate_values = inputs @ values["gate_proj.weight"].T
    dense_hidden = (gate_values / (1.0 + np.exp(-gate_values))) * (inputs @ values["up_proj.weight"].T)
    target = dense_hidden @ values["down_proj.weight"].T
    from ..partition import frozen_slice_simplex_oracle, partition_indices, swiglu_contributions

    def evaluate(name: str, plan: Any, top_k: int) -> dict[str, Any]:
        shared, routed = swiglu_contributions(inputs, values["gate_proj.weight"], values["up_proj.weight"], values["down_proj.weight"], plan)
        oracle = frozen_slice_simplex_oracle(shared, routed, target, top_k=top_k)
        return {"name": name, "top_k": top_k, "classification": "RANDOM_INPUT_FROZEN_SLICE_DIAGNOSTIC", "shared_width": plan.shared_intermediate_size, "expert_width": plan.expert_intermediate_size, "oracle_normalized_mse": float(oracle["normalized_mse"]), "oracle_cosine": float(oracle["cosine"]), "all_expert_reconstruction_mse": float(np.mean((shared + routed.sum(axis=1) - target) ** 2))}

    base = partition_indices(dense_size, 8, 2048, 1024)
    activation_scores = np.mean(np.abs(gate_values), axis=0)
    contribution_scores = np.mean(np.linalg.norm(dense_hidden[:, :, None] * values["down_proj.weight"].T[None, :, :], axis=-1), axis=0)
    variants = [
        evaluate("contiguous_top2", base, 2),
        evaluate("contiguous_top3_diagnostic", base, 3),
        evaluate("interleave_top2", partition_indices(dense_size, 8, 2048, 1024, strategy="interleave"), 2),
        evaluate("activation_magnitude_top2", partition_indices(dense_size, 8, 2048, 1024, strategy="activation_magnitude", scores=activation_scores), 2),
        evaluate("contribution_signature_top2", partition_indices(dense_size, 8, 2048, 1024, strategy="output_contribution", scores=contribution_scores), 2),
        evaluate("wider_shared_top2", partition_indices(dense_size, 8, 1920, 2048), 2),
    ]
    best = min(variants, key=lambda item: item["oracle_normalized_mse"])
    return {
        "status": "ORACLE_ABLATION_COMPLETE",
        "classification": "RANDOM_INPUT_FROZEN_SLICE_DIAGNOSTIC",
        "quality_gate_eligible": False,
        "source_dir": str(source),
        "layer": layer,
        "variants": variants,
        "best_variant": best["name"],
        "gate": {
            "green": best["oracle_normalized_mse"] <= 0.05,
            "yellow": best["oracle_normalized_mse"] <= 0.10,
            "applies_to": "historical diagnostic only",
        },
        "code_commit": artifact_commit,
        "note": "historical random-input frozen-slice diagnostic; do not treat as a trainable-MoE ceiling or architecture gate",
    }
