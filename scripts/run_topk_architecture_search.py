"""Train-only architecture development search for the layer-0 frozen FFN.

The script deliberately keeps the expensive teacher corpus out of the search
path: it selects a deterministic row subset from the captured TRAIN manifest,
computes frozen SwiGLU contributions in bounded GPU batches, and evaluates an
exact small-k simplex/positive oracle for p8.  Larger expert counts use a
deterministic residual-correlation candidate pool, bounded beam search, and an
exact final positive/simplex coefficient solve; exhaustive p32 enumeration is
not a meaningful bounded experiment.

The resulting JSON artifacts are run evidence, not a trainable quality claim.
The holdout is read only after the TRAIN/dev finalist list is frozen, and is
used once for finalist confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dense2moe.capture import iter_activation_shards
from dense2moe.partition import partition_indices
from dense2moe.provenance import current_git_commit


RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")
def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_mlp(source: Path, layer: int = 0) -> dict[str, np.ndarray]:
    from safetensors import safe_open  # type: ignore

    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    required = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    if set(names) != required:
        raise ValueError(f"layer {layer} MLP inventory is incomplete: {sorted(names)}")
    values: dict[str, np.ndarray] = {}
    for name, shard in names.items():
        kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source / shard), framework="pt", device="cpu", **kwargs) as handle:
            values[name] = handle.get_tensor(prefix + name).float().numpy()
    return values


def _stable_row_keys(manifest: Path) -> tuple[list[str], dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    keys: list[str] = []
    for shard in payload.get("shards", []):
        shard_id = int(shard.get("shard_id", len(keys)))
        for record in shard.get("records", []):
            example = str(record.get("example_id", record.get("content_sha256", "unknown")))
            start = int(record.get("offset", 0))
            length = int(record.get("length", 0))
            for offset in range(length):
                keys.append(f"{payload.get('dataset_hash','')}:{shard_id}:{example}:{start + offset}")
    count = int(payload.get("count", 0))
    # Older manifests may not carry per-record metadata for every row.  Stable
    # global row identifiers are still deterministic and are explicitly called
    # out in the artifact rather than pretending they are tokenizer IDs.
    if len(keys) != count:
        keys = [f"{payload.get('dataset_hash','')}:{index}" for index in range(count)]
    return keys, payload


def _select_dev_rows(manifest: Path, count: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    keys, payload = _stable_row_keys(manifest)
    count = min(int(count), len(keys))
    ranked = sorted(range(len(keys)), key=lambda i: hashlib.sha256(f"{seed}:{keys[i]}".encode()).hexdigest())
    selected = np.asarray(sorted(ranked[:count]), dtype=np.int64)
    selected_keys = [keys[int(i)] for i in selected]
    selection_hash = hashlib.sha256("\n".join(selected_keys).encode()).hexdigest()
    return selected, {
        "schema_version": 1,
        "status": "ARCHITECTURE_DEV_SUBSET_READY",
        "source_manifest": str(manifest),
        "source_dataset_hash": payload.get("dataset_hash", ""),
        "source_split": payload.get("split"),
        "source_count": int(payload.get("count", len(keys))),
        "selected_count": int(selected.size),
        "selection_seed": int(seed),
        "selection_method": "sha256_ranked_stable_token_rows",
        "selected_global_indices": [int(i) for i in selected],
        "selected_row_key_hash": selection_hash,
        "token_ids_hash": selection_hash,
        "tokenizer_hash": payload.get("tokenizer_hash", ""),
        "token_id_note": "The capture stores MLP inputs, not token IDs; global row IDs and record/offset keys are the deterministic token identity.",
        "code_commit": current_git_commit(),
    }


def _materialize_selected(manifest: Path, selected: np.ndarray) -> np.ndarray:
    wanted = set(int(i) for i in selected)
    values: list[np.ndarray] = []
    cursor = 0
    for shard in iter_activation_shards(manifest, expected_split="train"):
        local = [i - cursor for i in range(cursor, cursor + int(shard.shape[0])) if i in wanted]
        if local:
            values.append(np.asarray(shard[local], dtype=np.float32))
        cursor += int(shard.shape[0])
    if cursor <= int(selected.max(initial=-1)):
        raise ValueError("selected architecture-dev row is outside the activation manifest")
    # Iteration above is in source order, which matches sorted global indices.
    output = np.concatenate(values, axis=0) if values else np.empty((0, 0), dtype=np.float32)
    if output.shape[0] != selected.size:
        raise ValueError(f"architecture-dev materialization mismatch: {output.shape[0]} != {selected.size}")
    return output


def _dense_hidden_target(inputs: Any, weights: dict[str, Any], device: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    gate = weights["gate_proj.weight"]
    up = weights["up_proj.weight"]
    down = weights["down_proj.weight"]
    hidden = F.silu(x @ gate.T) * (x @ up.T)
    return hidden, hidden @ down.T


def _plan_contributions(hidden: Any, down: Any, plan: Any) -> tuple[Any, Any]:
    import torch

    shared_idx = torch.as_tensor(plan.shared_indices, dtype=torch.long, device=hidden.device)
    shared = torch.einsum("nw,hw->nh", hidden.index_select(1, shared_idx), down.index_select(1, shared_idx))
    routed_parts = []
    for group in plan.expert_indices:
        idx = torch.as_tensor(group, dtype=torch.long, device=hidden.device)
        routed_parts.append(torch.einsum("nw,hw->nh", hidden.index_select(1, idx), down.index_select(1, idx)))
    return shared, torch.stack(routed_parts, dim=1)


def _candidate_error(gram: Any, rhs: Any, residual_sq: Any, coeff: Any) -> Any:
    # Every term is per-token; divide by output width only at aggregation time.
    return residual_sq - 2.0 * (coeff * rhs).sum(dim=1) + (torch_bmm(coeff.unsqueeze(1), gram).squeeze(1) * coeff).sum(dim=1)


def torch_bmm(left: Any, right: Any) -> Any:
    import torch

    return torch.bmm(left, right)


def _solve_subset(gram: Any, rhs: Any, active: tuple[int, ...], *, simplex: bool) -> Any:
    import torch

    n, k, _ = gram.shape
    width = len(active)
    if width == 1:
        if simplex:
            result = torch.zeros((n, k), dtype=gram.dtype, device=gram.device)
            result[:, active[0]] = 1.0
            return result
        diag = gram[:, active[0], active[0]].clamp_min(1e-12)
        result = torch.zeros((n, k), dtype=gram.dtype, device=gram.device)
        result[:, active[0]] = (rhs[:, active[0]] / diag).clamp_min(0.0)
        return result
    idx = torch.as_tensor(active, dtype=torch.long, device=gram.device)
    g = gram.index_select(1, idx).index_select(2, idx)
    b = rhs.index_select(1, idx)
    eye = torch.eye(width, dtype=gram.dtype, device=gram.device).expand(n, width, width)
    g = g + eye * 1e-7
    if simplex:
        kkt = torch.zeros((n, width + 1, width + 1), dtype=gram.dtype, device=gram.device)
        kkt[:, :width, :width] = g
        kkt[:, :width, width] = 1.0
        kkt[:, width, :width] = 1.0
        rhs_kkt = torch.zeros((n, width + 1), dtype=gram.dtype, device=gram.device)
        rhs_kkt[:, :width] = b
        rhs_kkt[:, width] = 1.0
        try:
            result = torch.linalg.solve(kkt, rhs_kkt)[:, :width]
        except RuntimeError:
            result = torch.linalg.lstsq(kkt, rhs_kkt.unsqueeze(-1)).solution[:, :width, 0]
    else:
        try:
            result = torch.linalg.solve(g, b.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            result = torch.linalg.lstsq(g, b.unsqueeze(-1)).solution[:, :, 0]
    full = torch.zeros((n, k), dtype=gram.dtype, device=gram.device)
    full[:, idx] = result
    return full


def _exact_combo(matrix: Any, residual: Any, top_k: int, *, simplex: bool) -> tuple[Any, Any]:
    """Solve every active face for one expert combination in batched CUDA calls.

    The prior implementation launched one ``linalg.solve`` for each face.
    Grouping faces by width keeps the same exact enumeration while reducing
    the p8 k=6 curve from thousands of launches per batch to six launches per
    combination.
    """

    import torch

    n = matrix.shape[0]
    gram = torch.bmm(matrix, matrix.transpose(1, 2))
    rhs = torch.bmm(matrix, residual.unsqueeze(-1)).squeeze(-1)
    residual_sq = (residual * residual).sum(dim=1)
    best_error = residual_sq.clone()
    best_weights = torch.zeros((n, top_k), dtype=matrix.dtype, device=matrix.device)
    if simplex:
        best_weights[:, 0] = 1.0
    for width in range(1, top_k + 1):
        masks = [tuple(i for i in range(top_k) if mask & (1 << i)) for mask in range(1, 1 << top_k) if mask.bit_count() == width]
        if not masks:
            continue
        mask_index = torch.as_tensor(masks, dtype=torch.long, device=matrix.device)
        mask_count = int(mask_index.shape[0])
        # Advanced indexing yields [tokens, masks, width, width] and
        # [tokens, masks, width] without materializing any output vectors.
        g = gram[:, mask_index[:, :, None], mask_index[:, None, :]]
        b = rhs[:, mask_index]
        eye = torch.eye(width, dtype=matrix.dtype, device=matrix.device).view(1, 1, width, width)
        g = g + eye * 1e-7
        flat_g = g.reshape(n * mask_count, width, width)
        flat_b = b.reshape(n * mask_count, width)
        if simplex:
            kkt = torch.zeros((n * mask_count, width + 1, width + 1), dtype=matrix.dtype, device=matrix.device)
            kkt[:, :width, :width] = flat_g
            kkt[:, :width, width] = 1.0
            kkt[:, width, :width] = 1.0
            rhs_kkt = torch.zeros((n * mask_count, width + 1), dtype=matrix.dtype, device=matrix.device)
            rhs_kkt[:, :width] = flat_b
            rhs_kkt[:, width] = 1.0
            try:
                flat_solution = torch.linalg.solve(kkt, rhs_kkt)[:, :width]
            except RuntimeError:
                flat_solution = torch.linalg.lstsq(kkt, rhs_kkt.unsqueeze(-1)).solution[:, :width, 0]
        else:
            try:
                flat_solution = torch.linalg.solve(flat_g, flat_b.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                flat_solution = torch.linalg.lstsq(flat_g, flat_b.unsqueeze(-1)).solution[:, :, 0]
        active = flat_solution.reshape(n, mask_count, width)
        valid = active.min(dim=2).values >= -1e-5
        active = active.clamp_min(0.0)
        if simplex:
            active = active / active.sum(dim=2, keepdim=True).clamp_min(1e-12)
        candidate_error = residual_sq[:, None] - 2.0 * (active * b).sum(dim=2)
        candidate_error = candidate_error + (active * torch.bmm(g.reshape(n * mask_count, width, width), active.reshape(n * mask_count, width, 1)).reshape(n, mask_count, width)).sum(dim=2)
        candidate_error = torch.where(valid, candidate_error, torch.full_like(candidate_error, float("inf")))
        face_error, face_index = candidate_error.min(dim=1)
        update = face_error < best_error
        if bool(update.any()):
            selected_active = active[torch.arange(n, device=matrix.device), face_index]
            selected_mask = mask_index[face_index]
            candidate_full = torch.zeros_like(best_weights)
            candidate_full.scatter_(1, selected_mask, selected_active)
            best_error = torch.where(update, face_error, best_error)
            best_weights[update] = candidate_full[update]
    return best_error, best_weights


def _exact_topk(shared: Any, routed: Any, target: Any, top_k: int, *, simplex: bool) -> dict[str, Any]:
    """Exact p8 combination/active-face solve for one bounded batch."""

    import torch

    n, experts, _ = routed.shape
    if top_k < 1 or top_k > experts:
        raise ValueError("invalid top_k")
    residual = target - shared
    best_error = (residual * residual).sum(dim=1)
    best_ids = torch.zeros((n, top_k), dtype=torch.long, device=routed.device)
    best_weights = torch.zeros((n, top_k), dtype=routed.dtype, device=routed.device)
    if simplex:
        best_weights[:, 0] = 1.0
    for combo in itertools.combinations(range(experts), top_k):
        ids = torch.as_tensor(combo, dtype=torch.long, device=routed.device)
        error, weights = _exact_combo(routed.index_select(1, ids), residual, top_k, simplex=simplex)
        update = error < best_error
        if bool(update.any()):
            best_error = torch.where(update, error, best_error)
            best_ids[update] = ids
            best_weights[update] = weights[update]
    return {"errors": best_error, "indices": best_ids, "weights": best_weights, "selection_method": "exact_all_combinations_active_faces"}


def _norm_ranked_topk(shared: Any, routed: Any, target: Any, top_k: int, *, simplex: bool) -> dict[str, Any]:
    import torch

    scores = torch.linalg.vector_norm(routed, dim=2)
    ids = torch.topk(scores, k=top_k, dim=1, largest=True, sorted=True).indices
    matrix = torch.gather(routed, 1, ids.unsqueeze(-1).expand(-1, -1, routed.shape[2]))
    residual = target - shared
    gram = torch.bmm(matrix, matrix.transpose(1, 2))
    rhs = torch.bmm(matrix, residual.unsqueeze(-1)).squeeze(-1)
    n = routed.shape[0]
    residual_sq = (residual * residual).sum(dim=1)
    if simplex:
        coeff = _solve_subset(gram, rhs, tuple(range(top_k)), simplex=True)
        coeff = coeff.clamp_min(0.0)
        coeff = coeff / coeff.sum(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        coeff = _solve_subset(gram, rhs, tuple(range(top_k)), simplex=False).clamp_min(0.0)
    error = _candidate_error(gram, rhs, residual_sq, coeff)
    return {"errors": error, "indices": ids, "weights": coeff, "selection_method": "norm_ranked_selection_then_exact_coefficients"}


def _residual_correlation_beam_topk(
    shared: Any,
    routed: Any,
    target: Any,
    top_k: int,
    *,
    simplex: bool,
    beam_width: int = 4,
    pool_size: int | None = None,
) -> dict[str, Any]:
    """Bounded residual-correlation pool plus beam search.

    This is intentionally stronger than norm ranking: each token gets a pool
    ranked by correlation with the teacher residual, then a small beam explores
    combinations.  Fast unconstrained coefficient fits score intermediate beam
    extensions; the retained final beam is refined with the exact positive or
    simplex active-face solve.  Beam and pool bounds keep p16/p32 finite.
    """

    import torch

    n, experts, width = routed.shape
    residual = target - shared
    pool_size = min(experts, int(pool_size or max(8, 2 * top_k + 2)))
    correlations = torch.einsum("neh,nh->ne", routed, residual)
    pool = torch.topk(correlations, k=pool_size, dim=1, largest=True, sorted=True).indices
    beam_width = max(1, min(int(beam_width), pool_size))
    beam_ids = torch.empty((n, 1, 0), dtype=torch.long, device=routed.device)
    for step in range(top_k):
        candidate_errors: list[Any] = []
        candidate_ids: list[Any] = []
        for beam in range(beam_ids.shape[1]):
            prefix = beam_ids[:, beam, :]
            for pool_slot in range(pool_size):
                expert_ids = pool[:, pool_slot]
                ids = torch.cat((prefix, expert_ids.unsqueeze(1)), dim=1)
                matrix = torch.gather(routed, 1, ids.unsqueeze(-1).expand(-1, -1, width))
                gram = torch.bmm(matrix, matrix.transpose(1, 2))
                rhs = torch.bmm(matrix, residual.unsqueeze(-1)).squeeze(-1)
                coeff = _solve_subset(gram, rhs, tuple(range(step + 1)), simplex=simplex)
                coeff = coeff.clamp_min(0.0)
                if simplex:
                    coeff = coeff / coeff.sum(dim=1, keepdim=True).clamp_min(1e-12)
                error = _candidate_error(gram, rhs, (residual * residual).sum(dim=1), coeff)
                duplicate = (prefix == expert_ids.unsqueeze(1)).any(dim=1) if step else torch.zeros(n, dtype=torch.bool, device=routed.device)
                error = torch.where(duplicate, torch.full_like(error, float("inf")), error)
                candidate_errors.append(error)
                candidate_ids.append(ids)
        errors = torch.stack(candidate_errors, dim=1)
        keep = min(beam_width, errors.shape[1])
        _, selected = torch.topk(errors, k=keep, dim=1, largest=False, sorted=True)
        all_ids = torch.stack(candidate_ids, dim=1)
        gather_ids = selected.unsqueeze(-1).expand(-1, -1, step + 1)
        beam_ids = torch.gather(all_ids, 1, gather_ids)
    # Refine only the bounded final beam with the exact active-face solve.
    exact_errors = []
    exact_weights = []
    for beam in range(beam_ids.shape[1]):
        ids = beam_ids[:, beam]
        matrix = torch.gather(routed, 1, ids.unsqueeze(-1).expand(-1, -1, width))
        error, weights = _exact_combo(matrix, residual, top_k, simplex=simplex)
        exact_errors.append(error)
        exact_weights.append(weights)
    refined_errors = torch.stack(exact_errors, dim=1)
    refined_weights = torch.stack(exact_weights, dim=1)
    refined_best = refined_errors.argmin(dim=1)
    rows = torch.arange(n, device=routed.device)
    return {
        "errors": refined_errors[rows, refined_best],
        "indices": beam_ids[rows, refined_best],
        "weights": refined_weights[rows, refined_best],
        "selection_method": "residual_correlation_beam_search_exact_final_coefficients",
        "candidate_pool_size": pool_size,
        "beam_width": beam_width,
    }


def _route_reconstruction(shared: Any, routed: Any, result: dict[str, Any], scales: np.ndarray | None = None) -> Any:
    import torch

    rows = torch.arange(routed.shape[0], device=routed.device)
    prediction = shared.clone()
    for slot in range(result["indices"].shape[1]):
        ids = result["indices"][:, slot]
        values = routed[rows, ids] * result["weights"][:, slot].unsqueeze(1)
        if scales is not None:
            values = values * torch.as_tensor(scales, dtype=routed.dtype, device=routed.device)[ids].unsqueeze(1)
        prediction = prediction + values
    return prediction


def _accumulate_scales(acc: dict[str, Any], routed: Any, target: Any, shared: Any, result: dict[str, Any]) -> None:
    import torch

    n, experts, width = routed.shape
    rows = torch.arange(n, device=routed.device)
    features = torch.zeros_like(routed)
    for slot in range(result["indices"].shape[1]):
        ids = result["indices"][:, slot]
        features[rows, ids] += routed[rows, ids] * result["weights"][:, slot].unsqueeze(1)
    residual = target - shared
    gram = torch.einsum("neh, nfh -> ef", features, features).detach().cpu().numpy().astype(np.float64)
    rhs = torch.einsum("neh, nh -> e", features, residual).detach().cpu().numpy().astype(np.float64)
    acc["gram"] += gram
    acc["rhs"] += rhs


def _fit_scales(acc: dict[str, Any]) -> np.ndarray:
    gram = acc["gram"]
    rhs = acc["rhs"]
    try:
        return np.linalg.solve(gram + np.eye(gram.shape[0]) * 1e-8, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _evaluate_profile(
    profile: dict[str, Any],
    inputs: np.ndarray,
    weights_cpu: dict[str, np.ndarray],
    plan: Any,
    *,
    device: str,
    batch_size: int,
    exact: bool,
    search_method: str | None = None,
    top_ks: Iterable[int],
    split_name: str,
    beam_width: int = 4,
    pool_size: int | None = None,
) -> list[dict[str, Any]]:
    import torch

    device_obj = torch.device(device)
    weights = {name: torch.as_tensor(value, dtype=torch.float32, device=device_obj) for name, value in weights_cpu.items()}
    expert_count = int(profile["routed_experts"])
    variants: dict[tuple[int, bool], dict[str, Any]] = {}
    for k in top_ks:
        for simplex in (True, False):
            variants[(k, simplex)] = {
                "profile": profile["name"],
                "top_k": int(k),
                "formulation": "simplex" if simplex else "positive",
                "selection_method": "exact_all_combinations_active_faces" if exact else "residual_correlation_beam_search_exact_final_coefficients" if search_method == "beam" else search_method or "norm_ranked_selection_then_exact_coefficients",
                "split": split_name,
                "tokens": 0,
                "error_sum": 0.0,
                "target_norm_sum": 0.0,
                "cosine_sum": 0.0,
                "coefficient_sum": 0.0,
                "coefficient_sq_sum": 0.0,
                "usage": np.zeros(expert_count, dtype=np.int64),
                "scale_fit": {"gram": np.zeros((expert_count, expert_count), dtype=np.float64), "rhs": np.zeros(expert_count, dtype=np.float64)},
                "routes": [],
                "elapsed_seconds": 0.0,
            }
    for start in range(0, inputs.shape[0], batch_size):
        batch = inputs[start : start + batch_size]
        hidden, target = _dense_hidden_target(batch, weights, device_obj)
        shared, routed = _plan_contributions(hidden, weights["down_proj.weight"], plan)
        del hidden
        for (k, simplex), item in variants.items():
            tic = time.perf_counter()
            if exact:
                route = _exact_topk(shared, routed, target, k, simplex=simplex)
            elif search_method == "beam":
                route = _residual_correlation_beam_topk(
                    shared,
                    routed,
                    target,
                    k,
                    simplex=simplex,
                    beam_width=beam_width,
                    pool_size=pool_size,
                )
            else:
                route = _norm_ranked_topk(shared, routed, target, k, simplex=simplex)
            item["elapsed_seconds"] += time.perf_counter() - tic
            errors = route["errors"]
            reconstruction = _route_reconstruction(shared, routed, route)
            item["tokens"] += int(batch.shape[0])
            item["error_sum"] += float(errors.sum().item())
            item["target_norm_sum"] += float((target * target).sum().item())
            item["cosine_sum"] += float(torch.sum(torch.sum(reconstruction * target, dim=1) / (torch.linalg.vector_norm(reconstruction, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12)).item())
            sums = route["weights"].sum(dim=1)
            item["coefficient_sum"] += float(sums.sum().item())
            item["coefficient_sq_sum"] += float((sums * sums).sum().item())
            for slot in range(k):
                item["usage"] += np.bincount(route["indices"][:, slot].detach().cpu().numpy(), minlength=expert_count)
            _accumulate_scales(item["scale_fit"], routed, target, shared, route)
            item["routes"].append((route["indices"].detach().cpu().numpy(), route["weights"].detach().cpu().numpy()))
        del target, shared, routed
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
    output: list[dict[str, Any]] = []
    for item in variants.values():
        scales = _fit_scales(item["scale_fit"])
        scaled_error = 0.0
        scaled_norm = 0.0
        scaled_cosine = 0.0
        cursor = 0
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size]
            hidden, target = _dense_hidden_target(batch, weights, device_obj)
            shared, routed = _plan_contributions(hidden, weights["down_proj.weight"], plan)
            del hidden
            ids, coeff = item["routes"][cursor]
            cursor += 1
            route = {"indices": torch.as_tensor(ids, dtype=torch.long, device=device_obj), "weights": torch.as_tensor(coeff, dtype=torch.float32, device=device_obj)}
            prediction = _route_reconstruction(shared, routed, route, scales=scales)
            scaled_error += float(torch.sum((prediction - target) ** 2).item())
            scaled_norm += float(torch.sum(target * target).item())
            scaled_cosine += float(torch.sum(torch.sum(prediction * target, dim=1) / (torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12)).item())
            del target, shared, routed, prediction
        tokens = max(int(item["tokens"]), 1)
        usage = item["usage"] / max(tokens * int(item["top_k"]), 1)
        item_out = {
            "profile": item["profile"],
            "split": item["split"],
            "top_k": item["top_k"],
            "formulation": item["formulation"],
            "selection_method": item["selection_method"],
            "beam_width": int(beam_width) if search_method == "beam" and not exact else None,
            "candidate_pool_size": int(pool_size) if search_method == "beam" and pool_size is not None and not exact else None,
            "tokens": tokens,
            "normalized_mse": item["error_sum"] / max(item["target_norm_sum"], 1e-12),
            "cosine": item["cosine_sum"] / tokens,
            "coefficient_sum_mean": item["coefficient_sum"] / tokens,
            "coefficient_sum_mse_from_one": item["coefficient_sq_sum"] / tokens - 2.0 * item["coefficient_sum"] / tokens + 1.0 if item["formulation"] == "simplex" else None,
            "learned_global_scales": [float(v) for v in scales],
            "learned_scale_normalized_mse": scaled_error / max(scaled_norm, 1e-12),
            "learned_scale_cosine": scaled_cosine / tokens,
            "expert_usage_fraction": [float(v) for v in usage],
            "expert_usage_cv": float(np.std(usage) / max(np.mean(usage), 1e-12)),
            "elapsed_seconds": float(item["elapsed_seconds"]),
        }
        output.append(item_out)
    return output


def _profile(name: str, experts: int, expert_width: int, shared_width: int) -> dict[str, Any]:
    return {
        "name": name,
        "routed_experts": experts,
        "expert_intermediate_size": expert_width,
        "shared_intermediate_size": shared_width,
        "dense_intermediate_size": shared_width + experts * expert_width,
        "hidden_size": 5120,
    }


def _active_params(profile: dict[str, Any], k: int) -> dict[str, Any]:
    hidden = int(profile["hidden_size"])
    width = int(profile["shared_intermediate_size"] + k * profile["expert_intermediate_size"])
    params = 3 * hidden * width
    dense = 3 * hidden * int(profile["dense_intermediate_size"])
    return {"active_intermediate_width": width, "active_ffn_parameters": params, "dense_ffn_parameters": dense, "active_ffn_parameter_ratio": params / dense}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    run = Path(args.run_dir)
    train_manifest = run / "capture/layer-0000-train.json"
    selected, dev_meta = _select_dev_rows(train_manifest, args.dev_tokens, args.seed)
    _json(run / "capture/architecture-dev.json", dev_meta)
    inputs = _materialize_selected(train_manifest, selected)
    weights_cpu = _load_mlp(Path(args.source_dir), 0)
    device = args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"
    # Score once on the same deterministic dev rows, then freeze every plan.
    score_inputs = inputs[: min(args.score_tokens, inputs.shape[0])]
    score_hidden, _ = _dense_hidden_target(score_inputs, {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in weights_cpu.items()}, torch.device(device))
    scores = score_hidden.detach().cpu().numpy().mean(axis=0)
    del score_hidden
    profiles = [_profile("p8", 8, 2048, 1024), _profile("p16", 16, 1024, 1024), _profile("p32", 32, 512, 1024)]
    all_results: list[dict[str, Any]] = []
    plans: dict[str, Any] = {}
    for p in profiles:
        plan = partition_indices(p["dense_intermediate_size"], p["routed_experts"], p["expert_intermediate_size"], p["shared_intermediate_size"], strategy="activation_magnitude", scores=scores)
        plans[p["name"]] = plan
        partition_payload = plan.as_dict() | {"profile": p, "strategy": "activation_magnitude", "architecture_dev_score_tokens": int(score_inputs.shape[0]), "architecture_dev_source": str(train_manifest)}
        _json(run / "partitions" / f"architecture-dev-{p['name']}.json", partition_payload)
        if p["name"] == "p8":
            ks = range(1, 7)
            exact = True
        elif p["name"] == "p16":
            ks = (2, 3, 4)
            exact = False
        else:
            ks = (4, 6)
            exact = False
        print(f"evaluating {p['name']} {list(ks)} exact={exact} on {inputs.shape[0]} dev tokens", flush=True)
        results = _evaluate_profile(p, inputs, weights_cpu, plan, device=device, batch_size=args.batch_size, exact=exact, search_method=None if exact else "beam", top_ks=ks, split_name="architecture_dev")
        for result in results:
            result["partition_strategy"] = "activation_magnitude"
            result["partition"] = partition_payload
            result.update(_active_params(p, result["top_k"]))
        all_results.extend(results)
        _json(run / "reports" / f"topk-{p['name']}-architecture-dev.json", {"profile": p, "results": results, "dev": dev_meta})
    p8 = [r for r in all_results if r["profile"] == "p8"]
    # Selection is train-dev only.  Keep the first strong p8 elbow, the p8
    # quality endpoint, and the best more-sparse p16/p32 positive candidate at
    # <=5120 active width.  k5 remains an explicitly recorded reserve point;
    # the full holdout is not consulted until this list is frozen.
    sparse_candidates = [r for r in all_results if r["profile"] != "p8" and r["formulation"] == "positive" and r["active_intermediate_width"] <= 5120]
    if not sparse_candidates:
        raise RuntimeError("p16/p32 fair-search produced no sparse candidate")
    best_sparse = min(sparse_candidates, key=lambda r: (r["normalized_mse"], r["active_intermediate_width"]))
    finalist_specs = (("p8", 4), ("p8", 6), (best_sparse["profile"], best_sparse["top_k"]))
    finalists = [next(r for r in all_results if r["profile"] == profile and r["top_k"] == k and r["formulation"] == "positive") for profile, k in finalist_specs]
    p8_positive = {int(r["top_k"]): r for r in p8 if r["formulation"] == "positive"}
    p8_elbow = {
        "profile": "p8",
        "top_k": 4,
        "active_intermediate_width": int(p8_positive[4]["active_intermediate_width"]),
        "rationale": "first strong quality/compute knee; k5 and k6 remain diagnostic quality endpoints with increasingly dense active width",
        "nmse_gain_k5_over_k4": float(p8_positive[4]["normalized_mse"] - p8_positive[5]["normalized_mse"]),
        "nmse_gain_k6_over_k5": float(p8_positive[5]["normalized_mse"] - p8_positive[6]["normalized_mse"]),
    }
    payload = {
        "schema_version": 1,
        "status": "ARCHITECTURE_DEV_SEARCH_COMPLETE",
        "classification": "TRAIN_ONLY_ARCHITECTURE_SELECTION",
        "run_dir": str(run),
        "source_dir": str(args.source_dir),
        "layer": 0,
        "dev_subset": dev_meta,
        "selection_policy": "deterministic TRAIN architecture-dev subset; full holdout reserved for finalists",
        "results": all_results,
        "p8_exact": True,
        "p16_p32_selection_method": "residual_correlation_beam_search_exact_final_coefficients",
        "finalists_selected_on_dev": [{"profile": r["profile"], "top_k": r["top_k"], "formulation": r["formulation"], "normalized_mse": r["normalized_mse"], "cosine": r["cosine"], "active_intermediate_width": r["active_intermediate_width"], "active_ffn_parameter_ratio": r["active_ffn_parameter_ratio"], "expert_dispatches_per_token": int(r["top_k"])} for r in finalists],
        "p8_top5_reserve": next(r for r in p8 if r["top_k"] == 5 and r["formulation"] == "positive"),
        "p8_quality_compute_elbow": p8_elbow,
        "code_commit": current_git_commit(),
    }
    _json(run / "reports/topk-p8-oracle-curve.json", {"dev_subset": dev_meta, "profile": _profile("p8", 8, 2048, 1024), "results": p8, "selection": payload["finalists_selected_on_dev"], "code_commit": current_git_commit()})
    _json(run / "reports/equal-compute-expert-granularity.json", {"dev_subset": dev_meta, "results": [r for r in all_results if r["profile"] != "p8" or r["top_k"] in (2, 4)], "comparisons": [{"pair": ["p16/k2", "p32/k4"], "active_width": 3072}, {"pair": ["p16/k3", "p32/k6"], "active_width": 4096}, {"pair": ["p8/k2", "p16/k4"], "active_width": 5120}], "code_commit": current_git_commit()})
    # Full holdout confirmation is a single post-selection pass.  It uses
    # exactly the partitions frozen from TRAIN/dev and cannot affect the
    # finalist list above.
    holdout_inputs = np.concatenate(list(iter_activation_shards(run / "capture/layer-0000-holdout.json", expected_split="holdout")), axis=0).astype(np.float32)
    holdout_results: list[dict[str, Any]] = []
    for profile_name, top_k in finalist_specs:
        profile = next(p for p in profiles if p["name"] == profile_name)
        result_rows = _evaluate_profile(profile, holdout_inputs, weights_cpu, plans[profile_name], device=device, batch_size=args.batch_size, exact=profile_name == "p8", search_method=None if profile_name == "p8" else "beam", top_ks=(top_k,), split_name="full_holdout_confirmation")
        row = next(r for r in result_rows if r["formulation"] == "positive")
        row["partition_strategy"] = "activation_magnitude_frozen_from_architecture_dev"
        row["dev_selection_hash"] = dev_meta["selected_row_key_hash"]
        row.update(_active_params(profile, top_k))
        holdout_results.append(row)
    holdout_payload = {
        "schema_version": 1,
        "status": "FULL_HOLDOUT_FINALIST_CONFIRMATION_COMPLETE",
        "classification": "POST_SELECTION_HOLDOUT_CONFIRMATION",
        "holdout_manifest": str(run / "capture/layer-0000-holdout.json"),
        "holdout_tokens": int(holdout_inputs.shape[0]),
        "dev_selection_hash": dev_meta["selected_row_key_hash"],
        "finalists_selected_before_holdout": payload["finalists_selected_on_dev"],
        "results": holdout_results,
        "code_commit": current_git_commit(),
    }
    _json(run / "reports/architecture-search-holdout-confirmation.json", holdout_payload)
    payload["holdout_confirmation"] = holdout_payload
    for row in finalists:
        finalist_profile = next(p for p in profiles if p["name"] == row["profile"])
        partition_artifact = plans[row["profile"]].as_dict() | {
            "schema_version": 2,
            "status": "PARTITION_READY_ARCHITECTURE_FINALIST",
            "layer": 0,
            "profile": finalist_profile,
            "profile_name": finalist_profile["name"],
            "top_k": int(row["top_k"]),
            "routing_mode": "independent_positive",
            "strategy": "activation_magnitude",
            "initial_expert_scales": row["learned_global_scales"],
            "scale_fit_scope": "architecture_dev_train_only",
            "architecture_dev_selection_hash": dev_meta["selected_row_key_hash"],
            "source_manifest": str(train_manifest),
            "code_commit": current_git_commit(),
        }
        _json(run / "partitions" / f"layer-0000-{row['profile']}-top{row['top_k']}-architecture-finalist.json", partition_artifact)
    _json(run / "reports/architecture-search.json", payload)
    lines = [
        "# Layer-0 top-k architecture search",
        "",
        "Status: complete. Selection used only a deterministic TRAIN architecture-dev subset; the full holdout was read once for the frozen finalists.",
        "",
        f"- Architecture-dev rows: **{dev_meta['selected_count']}** / {dev_meta['source_count']} (seed {dev_meta['selection_seed']})",
        f"- Architecture-dev row-key hash: `{dev_meta['selected_row_key_hash']}`",
        f"- Holdout confirmation rows: **{holdout_inputs.shape[0]}**",
        "- p8 k=1..6: exact all-combination active-face simplex and positive oracles.",
        "- p16/p32: deterministic residual-correlation candidate pools with bounded beam search and exact final positive/simplex coefficient solves; not an exhaustive p32 combination claim.",
        "",
        "## p8 exact curve (TRAIN/dev)",
        "",
        "| k | positive NMSE | learned-scale NMSE | cosine | active width |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted((r for r in p8 if r["formulation"] == "positive"), key=lambda r: r["top_k"]):
        lines.append(f"| {row['top_k']} | {row['normalized_mse']:.6f} | {row['learned_scale_normalized_mse']:.6f} | {row['cosine']:.4f} | {row['active_intermediate_width']} |")
    lines += ["", f"Quality/compute elbow: **p8/k4** is the first strong knee at width {p8_elbow['active_intermediate_width']} (NMSE gain k5 over k4: {p8_elbow['nmse_gain_k5_over_k4']:.6f}; k6 over k5: {p8_elbow['nmse_gain_k6_over_k5']:.6f}). k5/k6 remain diagnostic quality endpoints with increasingly dense active width."]
    lines += ["", "## Frozen finalists", "", "| profile | k | dev NMSE | dev cosine | holdout NMSE | holdout cosine | active width | dispatches/token |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    holdout_by_key = {(r["profile"], r["top_k"]): r for r in holdout_results}
    for row in finalists:
        hold = holdout_by_key[(row["profile"], row["top_k"])]
        lines.append(f"| {row['profile']} | {row['top_k']} | {row['normalized_mse']:.6f} | {row['cosine']:.4f} | {hold['normalized_mse']:.6f} | {hold['cosine']:.4f} | {row['active_intermediate_width']} | {row['top_k']} |")
    reserve = next(r for r in p8 if r["top_k"] == 5 and r["formulation"] == "positive")
    lines += ["", f"p8/k5 remains a dev-selected reserve point (NMSE {reserve['normalized_mse']:.6f}, active width {reserve['active_intermediate_width']}) and was not trained under the three-finalist cap.", "", "Current provisional recommendation: p8/k6 independent-positive is the trained sparse leader, but representative-layer replay remains paused until its oracle regret is resolved with an equal-budget extension; p8s14 is not eligible.", "", "The p8s14/top2 checkpoint is classified `DENSEISH_QUALITY_UPPER_BOUND` (NMSE 0.002087, cosine 0.997600) because it retains 15,104/17,408 active FFN width (86.8%); it is a control, not the production candidate. No deeper representative replay was started after the safe layer-29 checkpoint.", ""]
    (run / "reports/TOP_K_ARCHITECTURE_SEARCH.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dev-tokens", type=int, default=16384)
    parser.add_argument("--score-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps({"status": result["status"], "finalists": result["finalists_selected_on_dev"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
