"""One-real-layer partition pilot against immutable source weights."""

from __future__ import annotations

import json
import os
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


def _gpu_vectorized_top2_oracle(
    shared: Any,
    routed: Any,
    target: Any,
    *,
    positive: bool,
    materialize_reconstruction: bool = True,
) -> dict[str, Any]:
    """Evaluate the exact p8 pair oracle on CUDA without materializing pairs.

    The old NumPy implementation is mathematically correct but performs
    dozens of full ``tokens x hidden`` reductions on the host.  A complete
    16k-token holdout therefore spends tens of minutes in one oracle call.
    Keeping the same closed-form pair equations on the calibration GPU makes
    the real study bounded while preserving every token and partition.
    """

    import numpy as np  # type: ignore
    import torch  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    # ``swiglu_contributions`` currently returns NumPy arrays, while the
    # streamed/experimental callers may already have their bounded tensors on
    # CUDA.  Avoid the invalid ``np.asarray(cuda_tensor)`` path and keep the
    # exact same float32 equations in either case.
    def _cuda_float32(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)

    shared_t = _cuda_float32(shared)
    routed_t = _cuda_float32(routed)
    target_t = _cuda_float32(target)
    goal_t = target_t - shared_t
    token_count, expert_count, _width = routed_t.shape
    best_error = torch.full((token_count,), float("inf"), dtype=torch.float32, device=device)
    best_ids = torch.zeros((token_count, 2), dtype=torch.int64, device=device)
    best_weights = torch.zeros((token_count, 2), dtype=torch.float32, device=device)
    for first in range(expert_count):
        first_values = routed_t[:, first]
        for second in range(first + 1, expert_count):
            second_values = routed_t[:, second]
            candidates: tuple[tuple[Any, Any, Any], ...]
            if positive:
                aa = torch.sum(first_values * first_values, dim=1)
                bb = torch.sum(second_values * second_values, dim=1)
                ab = torch.sum(first_values * second_values, dim=1)
                ag = torch.sum(first_values * goal_t, dim=1)
                bg = torch.sum(second_values * goal_t, dim=1)
                determinant = aa * bb - ab * ab
                interior_a = torch.where(
                    determinant.abs() > 1e-20,
                    (ag * bb - bg * ab) / determinant,
                    torch.zeros_like(determinant),
                )
                interior_b = torch.where(
                    determinant.abs() > 1e-20,
                    (bg * aa - ag * ab) / determinant,
                    torch.zeros_like(determinant),
                )
                candidates = (
                    (torch.clamp(interior_a, min=0.0), torch.clamp(interior_b, min=0.0), (interior_a >= 0.0) & (interior_b >= 0.0)),
                    (torch.clamp(torch.where(aa > 1e-20, ag / aa, torch.zeros_like(aa)), min=0.0), torch.zeros_like(aa), torch.ones_like(aa, dtype=torch.bool)),
                    (torch.zeros_like(bb), torch.clamp(torch.where(bb > 1e-20, bg / bb, torch.zeros_like(bb)), min=0.0), torch.ones_like(bb, dtype=torch.bool)),
                    (torch.zeros_like(aa), torch.zeros_like(aa), torch.ones_like(aa, dtype=torch.bool)),
                )
            else:
                direction = first_values - second_values
                denominator = torch.sum(direction * direction, dim=1)
                alpha = torch.where(
                    denominator > 1e-20,
                    torch.sum(direction * (goal_t - second_values), dim=1) / denominator,
                    torch.full_like(denominator, 0.5),
                ).clamp(0.0, 1.0)
                candidates = ((alpha, 1.0 - alpha, torch.ones_like(alpha, dtype=torch.bool)),)
            for weight_first, weight_second, valid in candidates:
                prediction = weight_first[:, None] * first_values + weight_second[:, None] * second_values
                error = torch.mean((prediction - goal_t) ** 2, dim=1)
                error = torch.where(valid, error, torch.full_like(error, float("inf")))
                better = error < best_error
                best_error = torch.where(better, error, best_error)
                best_ids[:, 0] = torch.where(better, torch.tensor(first, device=device), best_ids[:, 0])
                best_ids[:, 1] = torch.where(better, torch.tensor(second, device=device), best_ids[:, 1])
                best_weights[:, 0] = torch.where(better, weight_first, best_weights[:, 0])
                best_weights[:, 1] = torch.where(better, weight_second, best_weights[:, 1])
    selected = torch.gather(routed_t, 1, best_ids[:, :, None].expand(-1, -1, routed_t.shape[2]))
    reconstruction = shared_t + (selected * best_weights[:, :, None]).sum(dim=1)
    error = reconstruction - target_t
    mse = torch.mean(error * error)
    norm = torch.mean(target_t * target_t) + 1e-12
    cosine = torch.mean(torch.sum(reconstruction * target_t, dim=1) / (torch.linalg.vector_norm(reconstruction, dim=1) * torch.linalg.vector_norm(target_t, dim=1) + 1e-12))
    result = {
        "indices": best_ids.cpu().numpy(),
        "weights": best_weights.cpu().numpy().astype(np.float64),
        "mse": float(mse.item()),
        "normalized_mse": float((mse / norm).item()),
        "cosine": float(cosine.item()),
        "selection_error": float(best_error.mean().item()),
        "coefficient_error": float(torch.mean((best_weights.sum(dim=1) - 1.0) ** 2).item()) if positive else 0.0,
        "backend": "torch-cuda-exact-pair-formula",
    }
    if materialize_reconstruction:
        result["reconstruction"] = reconstruction.cpu().numpy().astype(np.float64)
    del shared_t, routed_t, target_t, goal_t, best_error, best_ids, best_weights, selected, reconstruction
    torch.cuda.empty_cache()
    return result


def _vectorized_top2_oracle(
    shared: Any,
    routed: Any,
    target: Any,
    *,
    positive: bool = False,
    materialize_reconstruction: bool = True,
) -> dict[str, Any]:
    """Fast exact top-2 frozen oracle for a bounded real-activation array."""

    import numpy as np  # type: ignore

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return _gpu_vectorized_top2_oracle(
                shared,
                routed,
                target,
                positive=positive,
                materialize_reconstruction=materialize_reconstruction,
            )
    except ImportError:
        pass

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
    train_activation_manifest: Path | None = None,
) -> dict[str, Any]:
    """Evaluate p8 alternatives on actual fixed holdout MLP inputs.

    The holdout path is deliberately materialized only once: it is the fixed
    scientific evaluation split and is small enough for the bounded oracle
    study.  Each partition is evaluated independently and released before the
    next one is built.  A learned-scale result is emitted only when the caller
    supplies an explicit, disjoint train activation manifest; this prevents a
    target-assisted holdout fit from being misreported as generalization.
    """

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
        open_kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source_dir / shard), framework="pt", device="cpu", **open_kwargs) as handle:
            values[name] = torch.as_tensor(handle.get_tensor(prefix + name)).float().numpy()
    dense_hidden = (inputs @ values["gate_proj.weight"].T)
    dense_hidden = (dense_hidden / (1.0 + np.exp(-dense_hidden))) * (inputs @ values["up_proj.weight"].T)
    target = dense_hidden @ values["down_proj.weight"].T
    dense_size = int(values["gate_proj.weight"].shape[0])
    target_norm = float(np.mean(target**2)) + 1e-12
    activation_scores = np.mean(np.abs(dense_hidden), axis=0)
    contribution_scores = activation_scores * np.mean(np.abs(values["down_proj.weight"]), axis=0)
    # These retain a sample axis.  The partitioner hashes the complete
    # per-neuron signature, so neither is a scalar-only sorting shortcut.
    signature_sample_count = min(2048, int(dense_hidden.shape[0]))
    activation_signature = np.asarray(dense_hidden[:signature_sample_count], dtype=np.float32)
    contribution_signature = np.asarray(
        dense_hidden[:signature_sample_count]
        * np.asarray(np.mean(np.abs(values["down_proj.weight"]), axis=0), dtype=np.float32)[None, :],
        dtype=np.float32,
    )

    variants: list[dict[str, Any]] = []
    partition_specs: tuple[tuple[str, int, int, str, dict[str, Any]], ...] = (
        ("contiguous", 2048, 1024, "contiguous", {}),
        ("interleave", 2048, 1024, "interleave", {}),
        ("activation_magnitude", 2048, 1024, "activation_magnitude", {"scores": activation_scores}),
        ("output_contribution", 2048, 1024, "output_contribution", {"scores": contribution_scores}),
        ("balanced_activation_signature", 2048, 1024, "balanced_signature", {"scores": activation_signature}),
        ("balanced_contribution_signature", 2048, 1024, "balanced_signature", {"scores": contribution_signature}),
        # Preserve the wider-shared diagnostic from the original planning
        # study; it has the same total dense capacity and remains p8/top-2.
        ("wider_shared", 1920, 2048, "contiguous", {}),
    )
    for name, expert_width, shared_width, strategy, kwargs in partition_specs:
        plan = partition_indices(dense_size, 8, expert_width, shared_width, strategy=strategy, **kwargs)
        shared, routed = swiglu_contributions(
            inputs,
            values["gate_proj.weight"],
            values["up_proj.weight"],
            values["down_proj.weight"],
            plan,
        )
        simplex = _vectorized_top2_oracle(shared, routed, target)
        positive = _vectorized_top2_oracle(shared, routed, target, positive=True)
        partition_mse = float(np.mean((shared + routed.sum(axis=1) - target) ** 2))
        partition_payload = plan.as_dict()
        variants.append({
            "name": f"{name}_top2",
            "partition_strategy": strategy,
            "signature_definition": "activation_per_neuron" if name == "balanced_activation_signature" else "contribution_magnitude_per_neuron" if name == "balanced_contribution_signature" else None,
            "shared_width": plan.shared_intermediate_size,
            "expert_width": plan.expert_intermediate_size,
            "top_k": 2,
            "simplex": {"normalized_mse": simplex["normalized_mse"], "cosine": simplex["cosine"]},
            "positive": {"normalized_mse": positive["normalized_mse"], "cosine": positive["cosine"]},
            # Keep the error taxonomy explicit.  ``selection_error`` is the
            # residual after choosing the best pair; ``scaling_error`` is the
            # improvement from relaxing the simplex coefficient constraint;
            # ``capacity_error`` is the remaining non-negative sparse error.
            "selection_error": float(simplex["selection_error"] / target_norm),
            "scaling_error": float(max(0.0, simplex["normalized_mse"] - positive["normalized_mse"])),
            "capacity_error": float(positive["normalized_mse"]),
            "coefficient_error": positive["coefficient_error"],
            "partition_error": partition_mse,
            "partition_normalized_mse": partition_mse / target_norm,
            "partition": partition_payload,
            "active_capacity_limitation": float(simplex["normalized_mse"]),
            "oracle_backend": simplex.get("backend", "numpy"),
        })
        del shared, routed, simplex, positive

    # Top-3 is retained as a bounded capacity diagnostic.  The p8/top-2
    # decision remains based on the exact pair oracle above; this deterministic
    # norm-ranked top-3 baseline avoids silently invoking a much more expensive
    # high-dimensional active-face solve.
    contiguous = partition_indices(dense_size, 8, 2048, 1024)
    shared3, routed3 = swiglu_contributions(
        inputs,
        values["gate_proj.weight"],
        values["up_proj.weight"],
        values["down_proj.weight"],
        contiguous,
    )
    top3_ids = np.argpartition(-np.linalg.norm(routed3, axis=-1), 2, axis=1)[:, :3]
    top3_values = np.take_along_axis(routed3, top3_ids[:, :, None], axis=1)
    top3_prediction = shared3 + top3_values.mean(axis=1)
    top3_mse = float(np.mean((top3_prediction - target) ** 2))
    variants.append({
        "name": "contiguous_top3_diagnostic",
        "partition_strategy": "contiguous",
        "shared_width": contiguous.shared_intermediate_size,
        "expert_width": contiguous.expert_intermediate_size,
        "top_k": 3,
        "oracle_method": "norm_ranked_equal_weight_diagnostic",
        "simplex": {
            "normalized_mse": top3_mse / target_norm,
            "cosine": float(np.mean(np.sum(top3_prediction * target, axis=1) / (np.linalg.norm(top3_prediction, axis=1) * np.linalg.norm(target, axis=1) + 1e-12))),
        },
        "positive": None,
        "selection_error": None,
        "scaling_error": None,
        "capacity_error": top3_mse / target_norm,
        "partition_error": float(np.mean((shared3 + routed3.sum(axis=1) - target) ** 2)),
        "partition_normalized_mse": float(np.mean((shared3 + routed3.sum(axis=1) - target) ** 2)) / target_norm,
    })
    del shared3, routed3, top3_values, top3_prediction

    decision_variants = [item for item in variants if item.get("top_k") == 2]
    best = min(decision_variants, key=lambda item: item["positive"]["normalized_mse"])
    learned_scales: dict[str, Any]
    if train_activation_manifest is None:
        learned_scales = {
            "status": "NOT_RUN_NO_EXPLICIT_TRAIN_MANIFEST",
            "fit_scope": "train_only_required",
            "reason": "holdout target-assisted scale fitting is not a valid generalization result",
        }
    else:
        best_spec = next(spec for spec in partition_specs if spec[0] + "_top2" == best["name"])
        best_plan = partition_indices(
            dense_size,
            8,
            best_spec[1],
            best_spec[2],
            strategy=best_spec[3],
            **best_spec[4],
        )
        learned_scales = _fit_streaming_expert_scales(
            source_dir,
            train_activation_manifest,
            holdout_manifest,
            layer=layer,
            plan=best_plan,
        )
    return {
        "status": "REAL_ACTIVATION_ORACLE_COMPLETE",
        "classification": "REAL_TEACHER_ACTIVATION_HOLDOUT",
        "quality_gate_eligible": True,
        "source_dir": str(source_dir),
        "activation_manifest": str(holdout_manifest),
        "dataset_hash": manifest_payload.get("dataset_hash", ""),
        "layer": layer,
        "holdout_tokens": int(inputs.shape[0]),
        "partition_signature_sample_count": signature_sample_count,
        "variants": variants,
        "best_variant": best["name"],
        "best_partition": best["partition"],
        "learned_expert_scale_oracle": learned_scales,
        "gate": {
            "green": best["positive"]["normalized_mse"] <= 0.05,
            "yellow": best["positive"]["normalized_mse"] <= 0.10,
            "applies_to": "real teacher activation holdout",
        },
        "seed": seed,
        "code_commit": current_git_commit(),
    }


def _split_manifest_path(path: Path, split: str) -> Path:
    """Resolve a split-labelled activation manifest without positional splits."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") == split:
        return path
    reference = payload.get(f"{split}_manifest")
    if not reference and isinstance(payload.get("splits"), dict):
        reference = payload["splits"].get(split)
    if not reference or reference == "pending":
        raise ValueError(f"explicit {split} activation manifest is required: {path}")
    candidate = Path(str(reference))
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"activation manifest does not exist: {candidate}")
    return candidate


def _fit_streaming_expert_scales(
    source_dir: Path,
    train_activation_manifest: Path,
    holdout_activation_manifest: Path,
    *,
    layer: int,
    plan: Any,
) -> dict[str, Any]:
    """Fit global expert scales from train shards and score fixed holdout.

    Routing is still the target-assisted frozen-slice oracle, but the scale
    coefficients are learned from train only.  Normal equations are reduced
    shard by shard, so the full 131k-token train corpus is never concatenated.
    """

    import numpy as np  # type: ignore
    import torch  # type: ignore
    from safetensors import safe_open  # type: ignore

    train_path = _split_manifest_path(train_activation_manifest, "train")
    holdout_path = _split_manifest_path(holdout_activation_manifest, "holdout")
    index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    values: dict[str, Any] = {}
    for name, shard in names.items():
        open_kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source_dir / shard), framework="pt", device="cpu", **open_kwargs) as handle:
            values[name] = torch.as_tensor(handle.get_tensor(prefix + name)).float().numpy()

    def shard_arrays(values_array: Any) -> tuple[Any, Any, Any]:
        inputs = np.asarray(values_array, dtype=np.float32)
        gate_values = inputs @ values["gate_proj.weight"].T
        hidden = (gate_values / (1.0 + np.exp(-gate_values))) * (inputs @ values["up_proj.weight"].T)
        target_values = hidden @ values["down_proj.weight"].T
        shared_values, routed_values = swiglu_contributions(
            inputs,
            values["gate_proj.weight"],
            values["up_proj.weight"],
            values["down_proj.weight"],
            plan,
        )
        return shared_values, routed_values, target_values

    expert_count = int(plan.routed_experts)
    gram = np.zeros((expert_count, expert_count), dtype=np.float64)
    rhs = np.zeros(expert_count, dtype=np.float64)
    train_tokens = 0
    train_error = 0.0
    for shard in iter_activation_shards(train_path, expected_split="train"):
        shared_values, routed_values, target_values = shard_arrays(shard)
        routing = _vectorized_top2_oracle(shared_values, routed_values, target_values, materialize_reconstruction=False)
        features = np.zeros((routed_values.shape[0], expert_count, routed_values.shape[2]), dtype=np.float32)
        rows = np.arange(routed_values.shape[0])
        for slot in range(routing["indices"].shape[1]):
            ids = routing["indices"][:, slot]
            features[rows, ids] += routing["weights"][:, slot, None].astype(np.float32) * routed_values[rows, ids]
        residual = target_values - shared_values
        gram += np.einsum("tio,tjo->ij", features, features, optimize=True)
        rhs += np.einsum("tio,to->i", features, residual, optimize=True)
        train_error += float(np.sum((features.sum(axis=1) - residual) ** 2))
        train_tokens += int(routed_values.shape[0])
        del shared_values, routed_values, target_values, routing, features, residual
    try:
        scales = np.linalg.solve(gram + np.eye(expert_count, dtype=np.float64) * 1e-8, rhs)
    except np.linalg.LinAlgError:
        scales = np.linalg.lstsq(gram, rhs, rcond=None)[0]

    def evaluate(path: Path) -> dict[str, Any]:
        squared_error = 0.0
        target_norm = 0.0
        cosine_sum = 0.0
        token_count = 0
        for shard in iter_activation_shards(path, expected_split=path.stem.endswith("-train") and "train" or "holdout"):
            shared_values, routed_values, target_values = shard_arrays(shard)
            routing = _vectorized_top2_oracle(shared_values, routed_values, target_values, materialize_reconstruction=False)
            features = np.zeros((routed_values.shape[0], expert_count, routed_values.shape[2]), dtype=np.float32)
            rows = np.arange(routed_values.shape[0])
            for slot in range(routing["indices"].shape[1]):
                ids = routing["indices"][:, slot]
                features[rows, ids] += routing["weights"][:, slot, None].astype(np.float32) * routed_values[rows, ids]
            prediction = shared_values + np.sum(features * scales[None, :, None], axis=1)
            squared_error += float(np.sum((prediction - target_values) ** 2))
            target_norm += float(np.sum(target_values**2))
            cosine_sum += float(np.sum(np.sum(prediction * target_values, axis=1) / (np.linalg.norm(prediction, axis=1) * np.linalg.norm(target_values, axis=1) + 1e-12)))
            token_count += int(target_values.shape[0])
            del shared_values, routed_values, target_values, routing, features, prediction
        mse = squared_error / max(token_count * int(values["down_proj.weight"].shape[0]), 1)
        norm = target_norm / max(token_count * int(values["down_proj.weight"].shape[0]), 1)
        return {"tokens": token_count, "mse": mse, "normalized_mse": mse / (norm + 1e-12), "cosine": cosine_sum / max(token_count, 1)}

    holdout_metrics = evaluate(holdout_path)
    return {
        "status": "LEARNED_SCALE_ORACLE_COMPLETE",
        "fit_scope": "train_only",
        "routing_target_assisted": True,
        "layer": layer,
        "partition": plan.as_dict(),
        "train_tokens": train_tokens,
        "train_fit_residual_mse": train_error / max(train_tokens * int(values["down_proj.weight"].shape[0]), 1),
        "holdout": holdout_metrics,
        "scales": [float(value) for value in scales],
        "train_manifest": str(train_path),
        "holdout_manifest": str(holdout_path),
    }


def run_oracle_ablation(
    source_dir: str | Path,
    *,
    layer: int = 0,
    batch_size: int = 8,
    seed: int = 17,
    activation_manifest: str | Path | None = None,
    train_activation_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run real-activation p8 study, falling back only to historical diagnostics."""

    if activation_manifest is not None:
        return _run_real_activation_oracle(
            Path(source_dir),
            Path(activation_manifest),
            layer=layer,
            seed=seed,
            train_activation_manifest=Path(train_activation_manifest) if train_activation_manifest is not None else None,
        )

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
