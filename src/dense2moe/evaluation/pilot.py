"""One-real-layer partition pilot against immutable source weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import MoEProfile
from ..partition import partition_indices


def run_real_layer_pilot(source_dir: str | Path, profiles: list[MoEProfile], *, layer: int = 0, batch_size: int = 2, seed: int = 17) -> dict[str, Any]:
    """Measure dense-vs-initial-sparse error for one actual MLP layer.

    The pilot loads only the three layer-0 MLP tensors and never writes into
    the source directory.  It intentionally does not label the untrained
    sparse output as a quality result.
    """

    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from safetensors import safe_open  # type: ignore
    except ImportError as exc:
        return {"status": "BLOCKED", "reason": f"optional ML reader unavailable: {exc}"}
    source = Path(source_dir)
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        return {"status": "BLOCKED", "reason": "safetensors index is missing"}
    import json

    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key: value for key, value in index.get("weight_map", {}).items() if key.startswith(prefix)}
    required = {"down_proj.weight", "gate_proj.weight", "up_proj.weight"}
    if {key[len(prefix) :] for key in names} != required:
        return {"status": "BLOCKED", "reason": "layer MLP tensor inventory is incomplete", "found": sorted(names)}
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
        sparse_output = shared_output.clone()
        for group in routed[: profile.top_k]:
            sparse_output = sparse_output + dense_hidden[:, list(group)] @ tensors["down_proj.weight"][:, list(group)].T / profile.top_k
        relative_mse = float(torch.mean((sparse_output - dense_output).square()).item() / dense_norm)
        results.append({"profile": profile.name, "status": "measured", "layer": layer, "batch_size": batch_size, "dense_mse": 0.0, "all_expert_reconstruction_mse": exact_error, "initial_sparse_relative_mse": relative_mse, "source_tensor_shard": next(iter(names.values()))})
    return {"status": "PILOT_COMPLETE", "source_dir": str(source), "layer": layer, "results": results, "note": "untrained sparse error is a structural pilot, not a retained-quality claim"}

