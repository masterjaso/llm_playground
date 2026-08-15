"""One-real-layer partition pilot against immutable source weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..capture import iter_activation_shards
from ..config import MoEProfile
from ..partition import partition_indices, swiglu_contributions
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


def _vectorized_top2_oracle(shared: Any, routed: Any, target: Any, *, positive: bool = False) -> dict[str, Any]:
    """Fast exact top-2 frozen oracle for a bounded real-activation array."""

    import numpy as np  # type: ignore

    shared_values = np.asarray(shared, dtype=np.float64)
    routed_values = np.asarray(routed, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    goal = target_values - shared_values
    tokens, experts, _width = routed_values.shape
    best_error = np.full(tokens, np.inf, dtype=np.float64)
    best_ids = np.zeros((tokens, 2), dtype=np.int64)
    best_weights = np.zeros((tokens, 2), dtype=np.float64)
    for first in range(experts):
        a = routed_values[:, first]
        for second in range(first + 1, experts):
            b = routed_values[:, second]
            if positive:
                aa = np.sum(a * a, axis=1)
                bb = np.sum(b * b, axis=1)
                ab = np.sum(a * b, axis=1)
                ag = np.sum(a * goal, axis=1)
                bg = np.sum(b * goal, axis=1)
                det = aa * bb - ab * ab
                wa = np.divide(ag * bb - bg * ab, det, out=np.zeros_like(det), where=np.abs(det) > 1e-20)
                wb = np.divide(bg * aa - ag * ab, det, out=np.zeros_like(det), where=np.abs(det) > 1e-20)
                candidates = [
                    (np.maximum(wa, 0.0), np.maximum(wb, 0.0), (wa >= 0.0) & (wb >= 0.0)),
                    (np.maximum(ag / np.maximum(aa, 1e-20), 0.0), np.zeros(tokens), np.ones(tokens, dtype=bool)),
                    (np.zeros(tokens), np.maximum(bg / np.maximum(bb, 1e-20), 0.0), np.ones(tokens, dtype=bool)),
                    (np.zeros(tokens), np.zeros(tokens), np.ones(tokens, dtype=bool)),
                ]
            else:
                direction = a - b
                denominator = np.sum(direction * direction, axis=1)
                alpha = np.divide(
                    np.sum(direction * (goal - b), axis=1),
                    denominator,
                    out=np.full(tokens, 0.5, dtype=np.float64),
                    where=denominator > 1e-20,
                )
                alpha = np.clip(alpha, 0.0, 1.0)
                candidates = [(alpha, 1.0 - alpha, np.ones(tokens, dtype=bool))]
            for weight_a, weight_b, valid in candidates:
                prediction = weight_a[:, None] * a + weight_b[:, None] * b
                error = np.mean((prediction - goal) ** 2, axis=1)
                error = np.where(valid, error, np.inf)
                better = error < best_error
                best_error[better] = error[better]
                best_ids[better, 0] = first
                best_ids[better, 1] = second
                best_weights[better, 0] = weight_a[better]
                best_weights[better, 1] = weight_b[better]
    reconstruction = shared_values + np.take_along_axis(routed_values, best_ids[:, :, None], axis=1)[:, 0] * best_weights[:, 0, None] + np.take_along_axis(routed_values, best_ids[:, :, None], axis=1)[:, 1] * best_weights[:, 1, None]
    mse = float(np.mean((reconstruction - target_values) ** 2))
    norm = float(np.mean(target_values**2)) + 1e-12
    cosine = float(np.mean(np.sum(reconstruction * target_values, axis=1) / (np.linalg.norm(reconstruction, axis=1) * np.linalg.norm(target_values, axis=1) + 1e-12)))
    return {
        "indices": best_ids,
        "weights": best_weights,
        "reconstruction": reconstruction,
        "mse": mse,
        "normalized_mse": mse / norm,
        "cosine": cosine,
        "selection_error": float(np.mean(best_error)),
        "coefficient_error": float(np.mean((best_weights.sum(axis=1) - 1.0) ** 2)) if positive else 0.0,
    }


def _run_real_activation_oracle(
    source_dir: Path,
    activation_manifest: Path,
    *,
    layer: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate p8 alternatives on actual fixed holdout MLP inputs."""

    import numpy as np  # type: ignore
    from safetensors import safe_open  # type: ignore
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ValueError("real activation oracle requires PyTorch to decode the pinned bfloat16 teacher weights") from exc

    manifest_payload = json.loads(activation_manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("split") not in {"holdout", "both"}:
        reference = manifest_payload.get("holdout_manifest")
        if not reference or reference == "pending":
            raise ValueError("real oracle study requires an explicit holdout activation manifest")
        holdout_manifest = activation_manifest.parent / str(reference)
    else:
        holdout_manifest = activation_manifest
    inputs = np.concatenate(list(iter_activation_shards(holdout_manifest, expected_split="holdout")), axis=0)
    index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    required = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    if set(names) != required:
        raise ValueError(f"layer {layer} source MLP inventory is incomplete")
    values: dict[str, Any] = {}
    for name, shard in names.items():
        with safe_open(str(source_dir / shard), framework="pt", device="cpu") as handle:
            values[name] = torch.as_tensor(handle.get_tensor(prefix + name)).float().numpy()
    dense_hidden = (inputs @ values["gate_proj.weight"].T)
    dense_hidden = (dense_hidden / (1.0 + np.exp(-dense_hidden))) * (inputs @ values["up_proj.weight"].T)
    target = dense_hidden @ values["down_proj.weight"].T
    dense_size = int(values["gate_proj.weight"].shape[0])
    variants: list[dict[str, Any]] = []
    for strategy, kwargs in (
        ("contiguous", {}),
        ("interleave", {}),
        ("activation_magnitude", {"scores": np.mean(np.abs(dense_hidden), axis=0)}),
        # |h[t,n] * W[o,n]| separates over tokens and output dimensions;
        # compute the same mean without materializing a 1176 x 17408 x 5120
        # temporary (which would exceed 390 GiB for this layer).
        (
            "output_contribution",
            {
                "scores": np.mean(np.abs(dense_hidden), axis=0)
                * np.mean(np.abs(values["down_proj.weight"]), axis=0)
            },
        ),
    ):
        plan = partition_indices(dense_size, 8, 2048, 1024, strategy=strategy, **kwargs)
        shared, routed = swiglu_contributions(
            inputs,
            values["gate_proj.weight"],
            values["up_proj.weight"],
            values["down_proj.weight"],
            plan,
        )
        simplex = _vectorized_top2_oracle(shared, routed, target)
        positive = _vectorized_top2_oracle(shared, routed, target, positive=True)
        variants.append({
            "name": f"{strategy}_top2",
            "partition_strategy": strategy,
            "shared_width": plan.shared_intermediate_size,
            "expert_width": plan.expert_intermediate_size,
            "simplex": {"normalized_mse": simplex["normalized_mse"], "cosine": simplex["cosine"]},
            "positive": {"normalized_mse": positive["normalized_mse"], "cosine": positive["cosine"]},
            "selection_error": simplex["selection_error"],
            "coefficient_error": positive["coefficient_error"],
            "partition_error": float(np.mean((shared + routed.sum(axis=1) - target) ** 2)),
            "active_capacity_limitation": float(simplex["normalized_mse"]),
        })
    best = min(variants, key=lambda item: item["positive"]["normalized_mse"])
    return {
        "status": "REAL_ACTIVATION_ORACLE_COMPLETE",
        "classification": "REAL_TEACHER_ACTIVATION_HOLDOUT",
        "quality_gate_eligible": True,
        "source_dir": str(source_dir),
        "activation_manifest": str(holdout_manifest),
        "dataset_hash": manifest_payload.get("dataset_hash", ""),
        "layer": layer,
        "holdout_tokens": int(inputs.shape[0]),
        "variants": variants,
        "best_variant": best["name"],
        "gate": {
            "green": best["positive"]["normalized_mse"] <= 0.05,
            "yellow": best["positive"]["normalized_mse"] <= 0.10,
            "applies_to": "real teacher activation holdout",
        },
        "seed": seed,
        "code_commit": current_git_commit(),
    }


def run_oracle_ablation(
    source_dir: str | Path,
    *,
    layer: int = 0,
    batch_size: int = 8,
    seed: int = 17,
    activation_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run real-activation p8 study, falling back only to historical diagnostics."""

    if activation_manifest is not None:
        return _run_real_activation_oracle(Path(source_dir), Path(activation_manifest), layer=layer, seed=seed)

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
