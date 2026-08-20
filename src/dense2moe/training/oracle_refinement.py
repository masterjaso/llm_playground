"""Bounded oracle-routed basis refinement primitives.

The ordinary distillation path calls the learned selector during every basis
update.  That is useful for selector training, but it makes a weak selector
part of the capacity experiment.  This module separates the two steps used by
the corpus-v2 refinement protocol:

``oracle_assignments`` (E step)
    Score a bounded set of sparse routes against the current basis and dense
    target, fitting non-negative route coefficients for every candidate.

``oracle_routed_forward`` / ``train_oracle_routed_basis`` (M step)
    Reconstruct with the frozen assignments and update only the basis tensors.
    The learned selector is never called and is explicitly kept frozen.

The implementation is intentionally bounded.  p16/top4 can enumerate all
``C(16, 4)`` sets, while larger topologies use a deterministic
correlation-ranked candidate pool.  Candidate tensors are scored in chunks so
the E step does not materialize a token-by-candidate-by-hidden cube.
"""

from __future__ import annotations

import itertools
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OracleAssignments:
    """One bounded E-step result for a batch of activation rows."""

    indices: Any
    coefficients: Any
    residual_mse: Any
    method: str
    candidate_count: int
    effective_candidate_pool_size: int
    coefficient_constraint: str

    def as_dict(self) -> dict[str, Any]:
        """Return receipt-safe metadata without serializing tensor payloads."""

        return {
            "method": self.method,
            "candidate_count": int(self.candidate_count),
            "effective_candidate_pool_size": int(self.effective_candidate_pool_size),
            "coefficient_constraint": self.coefficient_constraint,
            "batch_size": int(self.indices.shape[0]),
            "top_k": int(self.indices.shape[1]),
            "mean_residual_mse": float(self.residual_mse.detach().mean().cpu().item()),
        }

    def save(self, path: str | Path) -> Path:
        """Persist the E-step payload atomically for a later frozen M step."""

        return save_oracle_assignments(self, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> OracleAssignments:
        """Reload a saved E-step payload without consulting a selector."""

        return load_oracle_assignments(path, device=device)


def _torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError("PyTorch is required for oracle-routed basis refinement") from exc


def _validate_batch_inputs(model: Any, inputs: Any, target: Any) -> tuple[Any, Any]:
    torch = _torch()
    values = (
        inputs if isinstance(inputs, torch.Tensor) else torch.as_tensor(inputs, dtype=torch.float32)
    )
    teacher = (
        target if isinstance(target, torch.Tensor) else torch.as_tensor(target, dtype=torch.float32)
    )
    if values.ndim < 2 or values.shape[-1] != int(model.hidden_size):
        raise ValueError("inputs must have a final dimension matching model.hidden_size")
    if teacher.ndim != values.ndim or tuple(teacher.shape) != tuple(values.shape):
        raise ValueError("target must have the same shape as inputs")
    return values, teacher


def _basis_outputs(model: Any, inputs: Any) -> tuple[Any, Any]:
    """Evaluate shared and every scaled routed contribution with gradients."""

    torch = _torch()
    import torch.nn.functional as F

    values = inputs.reshape(-1, int(model.hidden_size))
    shared = model.shared_down_proj(
        F.silu(model.shared_gate_proj(values)) * model.shared_up_proj(values)
    )
    routed = torch.stack(
        [
            model._expert_output(values, expert) * model.expert_scales[expert]
            for expert in range(model.routed_experts)
        ],
        dim=1,
    )
    return shared, routed


def _candidate_templates(
    experts: int,
    top_k: int,
    *,
    candidate_pool_size: int | None,
    max_combinations: int,
) -> tuple[Any, int, int, str]:
    """Build global or local route templates with explicit assurance metadata."""

    torch = _torch()
    if experts <= 0 or top_k <= 0 or top_k > experts:
        raise ValueError("experts/top_k must satisfy 0 < top_k <= experts")
    if max_combinations <= 0:
        raise ValueError("max_combinations must be positive")
    if candidate_pool_size is not None and candidate_pool_size < top_k:
        raise ValueError("candidate_pool_size must be at least top_k")
    configured_pool = (
        experts if candidate_pool_size is None else min(experts, int(candidate_pool_size))
    )
    full_count = math.comb(configured_pool, top_k)
    if configured_pool == experts and full_count <= max_combinations:
        combinations = list(itertools.combinations(range(experts), top_k))
        return (
            torch.as_tensor(combinations, dtype=torch.long),
            experts,
            len(combinations),
            "exact_all_combinations",
        )

    # The default p32/top5 pool is 2*top_k+4 = 14, yielding 2,002 sets.  If a
    # caller requests a larger pool whose combinations exceed the budget,
    # retain the deterministic base route and one-slot replacements instead
    # of silently claiming exhaustive coverage.
    effective_pool = min(
        experts, max(top_k, configured_pool if candidate_pool_size is not None else 2 * top_k + 4)
    )
    pool_count = math.comb(effective_pool, top_k)
    if pool_count <= max_combinations:
        combinations = list(itertools.combinations(range(effective_pool), top_k))
    else:
        combinations_set: set[tuple[int, ...]] = {tuple(range(top_k))}
        for replacement in range(effective_pool):
            for slot in range(top_k):
                candidate = list(range(top_k))
                candidate[slot] = replacement
                if len(set(candidate)) == top_k:
                    combinations_set.add(tuple(sorted(candidate)))
        combinations = sorted(combinations_set)[:max_combinations]
    return (
        torch.as_tensor(combinations, dtype=torch.long),
        effective_pool,
        len(combinations),
        "bounded_correlation_candidate_pool",
    )


def _positive_or_simplex_fit(selected: Any, residual: Any, *, simplex: bool) -> tuple[Any, Any]:
    """Fit non-negative (or simplex) coefficients for candidate routes.

    ``selected`` is ``[batch, candidates, top_k, hidden]``.  Active-face
    enumeration is bounded by ``2**top_k-1`` and is exact for the small k
    values used by the product profiles.  The caller chunks the candidate axis
    before invoking this function.
    """

    torch = _torch()
    batch, candidates, top_k, hidden = selected.shape
    gram = torch.einsum("bckh,bclh->bckl", selected, selected)
    rhs = torch.einsum("bckh,bh->bck", selected, residual)
    residual_squared = torch.sum(residual * residual, dim=-1, keepdim=True)
    best_error = (
        torch.full((batch, candidates), float("inf"), dtype=selected.dtype, device=selected.device)
        if simplex
        else residual_squared.expand(batch, candidates).clone()
    )
    best_weights = torch.zeros(
        (batch, candidates, top_k), dtype=selected.dtype, device=selected.device
    )

    for mask in range(1, 1 << top_k):
        active = tuple(index for index in range(top_k) if mask & (1 << index))
        width = len(active)
        active_index = torch.as_tensor(active, dtype=torch.long, device=selected.device)
        active_gram = gram.index_select(2, active_index).index_select(3, active_index)
        active_rhs = rhs.index_select(2, active_index)
        flat_gram = active_gram.reshape(batch * candidates, width, width)
        flat_rhs = active_rhs.reshape(batch * candidates, width)
        # A tiny ridge keeps duplicate deterministic fixture vectors finite;
        # the unregularized normal equations are still used for scoring.
        solve_gram = (
            flat_gram
            + torch.eye(width, dtype=flat_gram.dtype, device=flat_gram.device).unsqueeze(0) * 1e-7
        )
        if simplex:
            kkt = torch.zeros(
                (batch * candidates, width + 1, width + 1),
                dtype=flat_gram.dtype,
                device=flat_gram.device,
            )
            kkt[:, :width, :width] = solve_gram
            kkt[:, :width, width] = 1.0
            kkt[:, width, :width] = 1.0
            kkt_rhs = torch.zeros(
                (batch * candidates, width + 1), dtype=flat_gram.dtype, device=flat_gram.device
            )
            kkt_rhs[:, :width] = flat_rhs
            kkt_rhs[:, width] = 1.0
            try:
                solution = torch.linalg.solve(kkt, kkt_rhs.unsqueeze(-1)).squeeze(-1)[:, :width]
            except RuntimeError:
                solution = torch.linalg.lstsq(kkt, kkt_rhs.unsqueeze(-1)).solution.squeeze(-1)[
                    :, :width
                ]
        else:
            try:
                solution = torch.linalg.solve(solve_gram, flat_rhs.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                solution = torch.linalg.lstsq(solve_gram, flat_rhs.unsqueeze(-1)).solution.squeeze(
                    -1
                )
        solution = solution.reshape(batch, candidates, width)
        valid = torch.isfinite(solution).all(dim=-1) & (solution.min(dim=-1).values >= -1e-6)
        solution = solution.clamp_min(0.0)
        if simplex:
            total = solution.sum(dim=-1, keepdim=True)
            valid = valid & (total[..., 0] > 1e-8)
            solution = solution / total.clamp_min(1e-8)
        quadratic = torch.einsum("bcw,bcwv,bcv->bc", solution, active_gram, solution)
        candidate_error = (
            residual_squared - 2.0 * torch.sum(solution * active_rhs, dim=-1) + quadratic
        )
        candidate_error = torch.where(
            valid, candidate_error, torch.full_like(candidate_error, float("inf"))
        )
        update = candidate_error < best_error
        full = torch.zeros_like(best_weights)
        full.index_copy_(2, active_index, solution)
        best_weights = torch.where(update.unsqueeze(-1), full, best_weights)
        best_error = torch.where(update, candidate_error, best_error)

    if simplex:
        # Every simplex has a feasible vertex.  Active-face enumeration above
        # normally records it; this fallback handles pathological NaN systems.
        missing = ~torch.isfinite(best_error)
        if bool(missing.any()):
            best_error = torch.where(missing, residual_squared.expand_as(best_error), best_error)
            fallback = torch.zeros_like(best_weights)
            fallback[..., 0] = 1.0
            best_weights = torch.where(missing.unsqueeze(-1), fallback, best_weights)
    return best_error / max(hidden, 1), best_weights


def oracle_assignments(
    model: Any,
    inputs: Any,
    target: Any,
    *,
    top_k: int | None = None,
    simplex: bool | None = None,
    candidate_pool_size: int | None = None,
    max_combinations: int = 4096,
    max_candidate_bytes: int = 64 * 1024 * 1024,
) -> OracleAssignments:
    """Compute a strong bounded sparse assignment for each input row.

    The model's selector is not consulted.  The current shared/expert basis
    supplies candidate vectors and the dense target supplies the residual.
    For ``independent_positive`` routing, coefficients are non-negative and
    need not sum to one.  ``simplex=True`` gives the normalized alternative.
    """

    torch = _torch()
    values, teacher = _validate_batch_inputs(model, inputs, target)
    values = values.to(next(model.parameters()).device)
    teacher = teacher.to(values.device)
    k = int(top_k if top_k is not None else model.top_k)
    if k <= 0 or k > int(model.routed_experts):
        raise ValueError("top_k must be in [1, routed_experts]")
    use_simplex = (
        bool(simplex) if simplex is not None else str(model.routing_mode) == "normalized_softmax"
    )
    if max_candidate_bytes <= 0:
        raise ValueError("max_candidate_bytes must be positive")
    with torch.no_grad():
        shared, routed = _basis_outputs(model, values)
        residual = teacher.reshape(-1, teacher.shape[-1]) - shared
        templates, effective_pool, candidate_count, method = _candidate_templates(
            int(model.routed_experts),
            k,
            candidate_pool_size=candidate_pool_size,
            max_combinations=max_combinations,
        )
        templates = templates.to(values.device)
        batch, _experts, hidden = routed.shape
        if method == "exact_all_combinations":
            candidate_ids = templates.unsqueeze(0).expand(batch, -1, -1)
        else:
            correlations = torch.einsum("beh,bh->be", routed, residual)
            order = torch.argsort(correlations, dim=-1, descending=True, stable=True)[
                :, :effective_pool
            ]
            candidate_ids = torch.gather(
                order.unsqueeze(1).expand(-1, candidate_count, -1),
                2,
                templates.unsqueeze(0).expand(batch, -1, -1),
            )
        bytes_per_candidate = max(batch * k * hidden * int(values.element_size()), 1)
        if max_candidate_bytes < bytes_per_candidate:
            raise ValueError(
                "max_candidate_bytes is smaller than one candidate block: "
                f"need at least {bytes_per_candidate} bytes"
            )
        chunk = max(1, min(candidate_count, int(max_candidate_bytes) // bytes_per_candidate))
        best_error = torch.full((batch,), float("inf"), dtype=routed.dtype, device=routed.device)
        best_ids = torch.zeros((batch, k), dtype=torch.long, device=routed.device)
        best_weights = torch.zeros((batch, k), dtype=routed.dtype, device=routed.device)
        expanded_routed = routed.unsqueeze(1)
        for start in range(0, candidate_count, chunk):
            stop = min(candidate_count, start + chunk)
            ids_block = candidate_ids[:, start:stop]
            selected = torch.gather(
                expanded_routed.expand(-1, stop - start, -1, -1),
                2,
                ids_block.unsqueeze(-1).expand(-1, -1, -1, hidden),
            )
            errors, weights = _positive_or_simplex_fit(selected, residual, simplex=use_simplex)
            block_error, block_choice = errors.min(dim=1)
            rows = torch.arange(batch, device=routed.device)
            update = block_error < best_error
            chosen_ids = ids_block[rows, block_choice]
            chosen_weights = weights[rows, block_choice]
            best_error = torch.where(update, block_error, best_error)
            best_ids = torch.where(update.unsqueeze(-1), chosen_ids, best_ids)
            best_weights = torch.where(update.unsqueeze(-1), chosen_weights, best_weights)
        # Return detached payloads so callers can safely retain assignments
        # through the M step without creating an accidental E->M graph.
        return OracleAssignments(
            indices=best_ids.detach(),
            coefficients=best_weights.detach(),
            residual_mse=best_error.detach(),
            method=method,
            candidate_count=int(candidate_count),
            effective_candidate_pool_size=int(effective_pool),
            coefficient_constraint="simplex" if use_simplex else "nonnegative",
        )


def oracle_routed_forward(
    model: Any, inputs: Any, assignments: OracleAssignments | Mapping[str, Any]
) -> Any:
    """Reconstruct with frozen assignments while bypassing the learned router."""

    torch = _torch()
    values = (
        inputs if isinstance(inputs, torch.Tensor) else torch.as_tensor(inputs, dtype=torch.float32)
    )
    ids = (
        assignments.indices
        if isinstance(assignments, OracleAssignments)
        else assignments["indices"]
    )
    coefficients = (
        assignments.coefficients
        if isinstance(assignments, OracleAssignments)
        else assignments["coefficients"]
    )
    ids = ids.to(device=values.device, dtype=torch.long)
    coefficients = coefficients.to(device=values.device, dtype=values.dtype)
    if ids.ndim != 2 or coefficients.shape != ids.shape:
        raise ValueError("oracle indices and coefficients must both be [batch, top_k]")
    flat = values.reshape(-1, int(model.hidden_size))
    if ids.shape[0] != flat.shape[0]:
        raise ValueError("oracle assignment row count does not match inputs")
    if bool((ids < 0).any()) or bool((ids >= int(model.routed_experts)).any()):
        raise ValueError("oracle assignment contains an expert outside the model")
    shared, routed = _basis_outputs(model, flat)
    selected = torch.gather(routed, 1, ids.unsqueeze(-1).expand(-1, -1, routed.shape[-1]))
    prediction = shared + torch.sum(selected * coefficients.unsqueeze(-1), dim=1)
    return prediction.reshape(*values.shape[:-1], int(model.hidden_size))


def save_oracle_assignments(assignments: OracleAssignments, path: str | Path) -> Path:
    """Write assignment tensors and solver metadata with an atomic replace.

    The saved object contains only detached CPU tensors and primitive metadata;
    it never serializes the model or its learned selector. This makes an
    assignment receipt safe to use as the frozen input to a later M step.
    """

    torch = _torch()
    if not isinstance(assignments, OracleAssignments):
        raise TypeError("assignments must be an OracleAssignments instance")
    if assignments.indices.ndim != 2 or assignments.coefficients.shape != assignments.indices.shape:
        raise ValueError("oracle assignment tensors must both be [batch, top_k]")
    if assignments.residual_mse.ndim != 1 or assignments.residual_mse.shape[0] != assignments.indices.shape[0]:
        raise ValueError("residual_mse must be [batch]")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "indices": assignments.indices.detach().cpu().to(dtype=torch.long),
        "coefficients": assignments.coefficients.detach().cpu(),
        "residual_mse": assignments.residual_mse.detach().cpu(),
        "method": str(assignments.method),
        "candidate_count": int(assignments.candidate_count),
        "effective_candidate_pool_size": int(assignments.effective_candidate_pool_size),
        "coefficient_constraint": str(assignments.coefficient_constraint),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def load_oracle_assignments(path: str | Path, *, device: str | None = None) -> OracleAssignments:
    """Load and validate a frozen assignment payload."""

    torch = _torch()
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    payload = torch.load(str(target), map_location=device or "cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported oracle assignment payload")
    required = ("indices", "coefficients", "residual_mse", "method", "candidate_count", "effective_candidate_pool_size", "coefficient_constraint")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"oracle assignment payload is missing: {', '.join(missing)}")
    indices = torch.as_tensor(payload["indices"], device=device or "cpu", dtype=torch.long)
    coefficients = torch.as_tensor(payload["coefficients"], device=device or "cpu")
    residual_mse = torch.as_tensor(payload["residual_mse"], device=device or "cpu")
    if indices.ndim != 2 or coefficients.shape != indices.shape:
        raise ValueError("oracle assignment tensors must both be [batch, top_k]")
    if residual_mse.ndim != 1 or residual_mse.shape[0] != indices.shape[0]:
        raise ValueError("residual_mse must be [batch]")
    if not torch.isfinite(coefficients).all() or not torch.isfinite(residual_mse).all():
        raise ValueError("oracle assignment payload contains non-finite values")
    return OracleAssignments(
        indices=indices,
        coefficients=coefficients,
        residual_mse=residual_mse,
        method=str(payload["method"]),
        candidate_count=int(payload["candidate_count"]),
        effective_candidate_pool_size=int(payload["effective_candidate_pool_size"]),
        coefficient_constraint=str(payload["coefficient_constraint"]),
    )


def _iter_batches(source: Iterable[Any] | Callable[[], Iterable[Any]]) -> Iterator[Any]:
    values = source() if callable(source) else source
    return iter(values)


def train_oracle_routed_basis(
    model: Any,
    batches: Iterable[tuple[Any, Any]] | Callable[[], Iterable[tuple[Any, Any]]],
    *,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    assignment_refresh_steps: int = 1,
    m_step_repeats: int = 1,
    train_shared: bool = True,
    train_experts: bool = True,
    train_scales: bool = True,
    simplex: bool | None = None,
    candidate_pool_size: int | None = None,
    max_combinations: int = 4096,
    max_candidate_bytes: int = 64 * 1024 * 1024,
    loss_coefficients: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run a bounded EM-like basis refinement over ``(inputs, target)`` batches.

    ``assignment_refresh_steps`` controls the frozen-interval length in
    batches.  Each interval performs one E step per batch, followed by
    ``m_step_repeats`` passes over those same assignments.  A callable
    ``batches`` source is re-opened for each epoch; a one-shot iterator is
    accepted for a single epoch only.
    """

    torch = _torch()
    if epochs < 0 or learning_rate <= 0:
        raise ValueError("epochs must be non-negative and learning_rate must be positive")
    if assignment_refresh_steps <= 0 or m_step_repeats <= 0:
        raise ValueError("assignment_refresh_steps and m_step_repeats must be positive")
    coefficients = {"mse": 1.0, "cosine": 0.05}
    coefficients.update(
        {str(name): float(value) for name, value in (loss_coefficients or {}).items()}
    )
    if any(value < 0 for value in coefficients.values()):
        raise ValueError("loss coefficients must be non-negative")

    for parameter in model.parameters():
        parameter.requires_grad = False
    if train_shared:
        for module in (model.shared_gate_proj, model.shared_up_proj, model.shared_down_proj):
            for parameter in module.parameters():
                parameter.requires_grad = True
    if train_experts:
        for module in (*model.expert_gate_proj, *model.expert_up_proj, *model.expert_down_proj):
            for parameter in module.parameters():
                parameter.requires_grad = True
    if train_scales:
        if not isinstance(model.expert_scales, torch.nn.Parameter):
            raise ValueError("train_scales requires model.expert_scales to be learnable")
        model.expert_scales.requires_grad = True
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable and epochs > 0:
        raise ValueError("oracle basis refinement has no trainable parameters")
    router_trainable = [
        parameter for parameter in model.router.parameters() if parameter.requires_grad
    ]
    if getattr(model, "routing_mode", None) == "independent_positive":
        router_trainable.extend(
            parameter
            for parameter in model.amplitude_router.parameters()
            if parameter.requires_grad
        )
    if router_trainable:
        raise AssertionError("oracle-routed basis refinement must keep selector parameters frozen")
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate)) if trainable else None
    updates = 0
    refreshes = 0
    losses: list[float] = []
    assignment_methods: Counter[str] = Counter()
    last_assignment: dict[str, Any] | None = None

    for _epoch in range(int(epochs)):
        iterator = _iter_batches(batches)
        while True:
            interval: list[tuple[Any, Any, OracleAssignments]] = []
            for _ in range(int(assignment_refresh_steps)):
                try:
                    raw_inputs, raw_target = next(iterator)
                except StopIteration:
                    break
                inputs, target = _validate_batch_inputs(model, raw_inputs, raw_target)
                inputs = inputs.to(device)
                target = target.to(device)
                assignment = oracle_assignments(
                    model,
                    inputs,
                    target,
                    simplex=simplex,
                    candidate_pool_size=candidate_pool_size,
                    max_combinations=max_combinations,
                    max_candidate_bytes=max_candidate_bytes,
                )
                interval.append((inputs, target, assignment))
                assignment_methods[assignment.method] += 1
                last_assignment = assignment.as_dict()
            if not interval:
                break
            refreshes += 1
            for _repeat in range(int(m_step_repeats)):
                for inputs, target, assignment in interval:
                    prediction = oracle_routed_forward(model, inputs, assignment)
                    prediction_flat = prediction.reshape(-1, prediction.shape[-1])
                    target_flat = target.reshape(-1, target.shape[-1])
                    mse = torch.mean((prediction_flat - target_flat).square())
                    cosine = 1.0 - torch.mean(
                        torch.sum(prediction_flat * target_flat, dim=-1)
                        / (
                            torch.linalg.vector_norm(prediction_flat, dim=-1)
                            * torch.linalg.vector_norm(target_flat, dim=-1)
                            + 1e-12
                        )
                    )
                    loss = coefficients["mse"] * mse + coefficients["cosine"] * cosine
                    if optimizer is None:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    updates += 1
                    losses.append(float(loss.detach().cpu().item()))

    return {
        "status": "ORACLE_ROUTED_BASIS_REFINEMENT_COMPLETE",
        "epochs": int(epochs),
        "updates": int(updates),
        "assignment_refreshes": int(refreshes),
        "assignment_refresh_steps": int(assignment_refresh_steps),
        "m_step_repeats": int(m_step_repeats),
        "loss_mean": float(sum(losses) / len(losses)) if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_coefficients": coefficients,
        "selector_frozen": True,
        "selector_trainable_parameters": 0,
        "basis_trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "assignment_methods": dict(assignment_methods),
        "last_assignment": last_assignment,
        "candidate_pool_size": candidate_pool_size,
        "max_combinations": int(max_combinations),
        "max_candidate_bytes": int(max_candidate_bytes),
        "coefficient_constraint": "simplex"
        if (
            simplex is True
            or (simplex is None and getattr(model, "routing_mode", None) == "normalized_softmax")
        )
        else "nonnegative",
    }


__all__ = [
    "OracleAssignments",
    "load_oracle_assignments",
    "oracle_assignments",
    "oracle_routed_forward",
    "save_oracle_assignments",
    "train_oracle_routed_basis",
]
