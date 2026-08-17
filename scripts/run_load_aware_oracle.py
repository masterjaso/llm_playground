"""Run bounded load-aware oracle diagnostics from contribution arrays.

The expensive teacher/model stage is intentionally kept separate.  A caller
materializes a FIT/validation-only contribution store containing
``shared.npy``, ``routed.npy``, ``target.npy`` and ``manifest.json``.  The
arrays are opened with NumPy read-only memory mapping, so the loader never
eagerly decompresses a multi-gigabyte contribution cube.  A small ``.npz``
fixture remains available only through the explicit ``--allow-eager-npz``
test escape hatch.  This separation makes it impossible for the diagnostic to
silently open the full holdout while still producing a concise quality/load
Pareto report.

Example::

    python scripts/run_load_aware_oracle.py \
      --input validation/p16-top4-contributions --topology p16/top4 \
      --input validation/p32-top5-contributions --topology p32/top5 \
      --output reports/load-aware-oracle.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Allow checkout-local execution before ``pip install -e .`` (the same path
# bootstrap used by the guarded-command entry point).
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.partition import frozen_slice_load_aware_oracle
from dense2moe.provenance import current_git_commit

_ARRAY_NAMES = ("shared", "routed", "target")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_store_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    basis_source = manifest.get("basis_source")
    # Keep the low-level loader backward compatible for old test fixtures and
    # historical artifacts, but label them as unusable rather than guessing a
    # basis.  ``run`` below refuses this label for an oracle report.
    if basis_source is None:
        basis_source = "legacy_unlabelled"
        manifest["basis_source"] = basis_source
    if basis_source not in {"raw_dense_partition", "trained_checkpoint", "legacy_unlabelled"}:
        raise ValueError(f"{path / 'manifest.json'} has an unsupported basis_source: {basis_source!r}")
    if basis_source == "trained_checkpoint" and (
        not manifest.get("checkpoint_path") or not manifest.get("checkpoint_tensor_sha256")
    ):
        raise ValueError(
            f"{path / 'manifest.json'} trained_checkpoint stores require checkpoint_path and checkpoint_tensor_sha256"
        )
    topology = manifest.get("topology_manifest", manifest.get("topology"))
    if topology is None and basis_source == "legacy_unlabelled":
        topology = {}
    if not isinstance(topology, dict):
        raise TypeError(f"{path / 'manifest.json'} must record topology_manifest")
    for field in ("expert_count", "expert_width", "shared_width", "top_k"):
        if field not in topology and basis_source != "legacy_unlabelled":
            raise ValueError(f"{path / 'manifest.json'} topology is missing {field}")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise TypeError(f"{path / 'manifest.json'} must contain an 'arrays' mapping")
    for name in _ARRAY_NAMES:
        entry = arrays.get(name)
        if not isinstance(entry, dict):
            raise TypeError(f"{path / 'manifest.json'} is missing arrays.{name}")
        relative = Path(str(entry.get("path", f"{name}.npy")))
        candidate = (path / relative).resolve()
        if candidate.parent != path.resolve():
            raise ValueError(f"contribution array path escapes store: {relative}")
        if candidate.suffix != ".npy":
            raise ValueError(f"contribution array must be a .npy file: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        entry["path"] = relative.as_posix()
    return manifest


def _load_arrays(path: Path, *, allow_eager_npz: bool = False) -> tuple[tuple[Any, Any, Any], dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("numpy is required for load-aware oracle diagnostics") from exc
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"memory-mapped contribution stores require {manifest_path}")
        manifest = _validate_store_manifest(path, json.loads(manifest_path.read_text(encoding="utf-8")))
        arrays = tuple(
            np.load(path / str(manifest["arrays"][name]["path"]), mmap_mode="r", allow_pickle=False)
            for name in _ARRAY_NAMES
        )
        for name, values in zip(_ARRAY_NAMES, arrays):
            if not isinstance(values, np.memmap):
                raise TypeError(f"{path / name}.npy did not open as a read-only memmap")
            entry = manifest["arrays"][name]
            expected_shape = entry.get("shape")
            if expected_shape is not None and list(values.shape) != list(expected_shape):
                raise ValueError(f"{name} shape {values.shape} disagrees with manifest {expected_shape}")
            expected_dtype = entry.get("dtype")
            if expected_dtype is not None and str(values.dtype) != str(expected_dtype):
                raise ValueError(f"{name} dtype {values.dtype} disagrees with manifest {expected_dtype}")
        return arrays, {
            "format": "npy_memmap",
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "array_paths": {name: str(path / str(manifest["arrays"][name]["path"])) for name in _ARRAY_NAMES},
            "basis_source": manifest["basis_source"],
            "checkpoint_path": manifest.get("checkpoint_path"),
            "checkpoint_tensor_sha256": manifest.get("checkpoint_tensor_sha256"),
            "partition_path": manifest.get("partition_path"),
            "partition_sha256": manifest.get("partition_sha256"),
            "source_revision": manifest.get("source_revision"),
            "code_commit": manifest.get("code_commit"),
            "dataset_hash": manifest.get("dataset_hash"),
            "split": manifest.get("split"),
            "row_count": manifest.get("row_count"),
            "topology_manifest": manifest.get("topology_manifest", manifest.get("topology")),
        }
    if path.suffix.lower() != ".npz":
        raise ValueError(f"input must be a contribution-store directory or .npz fixture: {path}")
    if not allow_eager_npz:
        raise ValueError(
            f"refusing eager NPZ input {path}; use shared.npy/routed.npy/target.npy + manifest.json "
            "or pass --allow-eager-npz for a small test fixture"
        )
    with np.load(path, allow_pickle=False) as values:
        required = set(_ARRAY_NAMES)
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(f"{path} is missing contribution arrays: {missing}")
        arrays = tuple(np.asarray(values[name]) for name in _ARRAY_NAMES)
    return arrays, {
        "format": "eager_npz",
        "manifest": None,
        "manifest_sha256": None,
        "array_paths": {},
        "basis_source": "fixture_unlabelled",
        "checkpoint_path": None,
        "checkpoint_tensor_sha256": None,
        "partition_path": None,
        "partition_sha256": None,
        "source_revision": None,
        "code_commit": None,
        "dataset_hash": None,
        "split": None,
        "row_count": None,
        "topology_manifest": None,
    }


def run(
    inputs: list[Path],
    topologies: list[str],
    *,
    target_load_cv: float,
    candidate_pool_size: int | None,
    max_combinations: int,
    iterations: int,
    batch_size: int,
    max_in_memory_bytes: int,
    storage_dir: Path | None,
    allow_eager_npz: bool = False,
) -> dict[str, Any]:
    if len(inputs) != len(topologies):
        raise ValueError("each --input requires a matching --topology")
    rows: list[dict[str, Any]] = []
    for row_index, (path, topology) in enumerate(zip(inputs, topologies)):
        if "/" not in topology:
            raise ValueError(f"topology must look like p16/top4: {topology!r}")
        profile, raw_top_k = topology.split("/", 1)
        top_k = int(raw_top_k.removeprefix("top"))
        (shared, routed, target), input_metadata = _load_arrays(path, allow_eager_npz=allow_eager_npz)
        if input_metadata["basis_source"] not in {"raw_dense_partition", "trained_checkpoint"}:
            raise ValueError(
                f"refusing oracle input without explicit basis_source: {path}; "
                "rematerialize it as raw_dense_partition or trained_checkpoint"
            )
        expected_experts = {"p16": 16, "p32": 32}.get(profile)
        if expected_experts is None:
            raise ValueError(f"unsupported topology profile: {profile!r}")
        if top_k not in ({4} if profile == "p16" else {4, 5}):
            raise ValueError(f"unsupported product topology: {topology!r}")
        if int(routed.shape[1]) != expected_experts:
            raise ValueError(f"{topology} expects {expected_experts} routed experts, received {routed.shape[1]}")
        input_storage = (
            storage_dir / f"{row_index:02d}-{profile}-top{top_k}"
            if storage_dir is not None
            else None
        )
        effective_candidate_pool = candidate_pool_size
        if effective_candidate_pool is None and profile == "p32" and top_k == 5:
            # C(15,5)=3003 is the default strong practical p32/top5 pool;
            # callers may still request a smaller bounded screen explicitly.
            effective_candidate_pool = 15
        result = frozen_slice_load_aware_oracle(
            shared,
            routed,
            target,
            top_k=top_k,
            target_load_cv=target_load_cv,
            candidate_pool_size=effective_candidate_pool,
            max_combinations=max_combinations,
            iterations=iterations,
            batch_size=batch_size,
            max_in_memory_bytes=max_in_memory_bytes,
            storage_dir=input_storage,
            materialize_outputs=False,
        )
        rows.append(
            {
                "topology": topology,
                "profile": profile,
                "top_k": top_k,
                "active_intermediate_width": (1024 + top_k * (1024 if profile == "p16" else 512)),
                "ffn_reduction": 1.0 - (1024 + top_k * (1024 if profile == "p16" else 512)) / 17408.0,
                "input": str(path),
                "input_format": input_metadata["format"],
                "input_manifest": input_metadata["manifest"],
                "input_manifest_sha256": input_metadata["manifest_sha256"],
                "input_array_paths": input_metadata["array_paths"],
                "basis_source": input_metadata["basis_source"],
                "checkpoint_path": input_metadata["checkpoint_path"],
                "checkpoint_tensor_sha256": input_metadata["checkpoint_tensor_sha256"],
                "partition_path": input_metadata["partition_path"],
                "partition_sha256": input_metadata["partition_sha256"],
                "source_revision": input_metadata["source_revision"],
                "input_code_commit": input_metadata["code_commit"],
                "dataset_hash": input_metadata["dataset_hash"],
                "split": input_metadata["split"],
                "row_count": input_metadata["row_count"],
                "topology_manifest": input_metadata["topology_manifest"],
                "tokens": int(result["indices"].shape[0]),
                "exact_or_bounded": result["assurance"],
                "candidate_pool_size": result["candidate_pool_size"],
                "effective_candidate_pool_size": result["effective_candidate_pool_size"],
                "combinations_considered_per_token": result["combinations_considered_per_token"],
                "max_combinations": result["max_combinations"],
                "assurance": result["assurance"],
                "candidate_error_storage": result["candidate_error_storage"],
                "candidate_id_storage": result["candidate_id_storage"],
                "candidate_storage_ephemeral": result["candidate_storage_ephemeral"],
                "coefficient_solver": result["coefficient_solver"],
                "candidate_fit_exact": result["candidate_fit_exact"],
                "candidate_chunk_size": result["candidate_chunk_size"],
                "candidate_batch_size": result["candidate_batch_size"],
                "requested_batch_size": result["requested_batch_size"],
                "input_bytes_per_token": result["input_bytes_per_token"],
                "input_batch_bytes": result["input_batch_bytes"],
                "exact_oracle_global_nmse": result["unconstrained"]["global_nmse"],
                "exact_oracle_cosine": result["unconstrained"]["cosine"],
                "oracle_hard_quartile_cosine": result["unconstrained"]["hard_quartile_cosine"],
                "oracle_load_cv": result["unconstrained"]["load_cv"],
                "global_nmse": result["global_nmse"],
                "cosine": result["cosine"],
                "hard_quartile_cosine": result["hard_quartile_cosine"],
                "load_constrained_oracle_cosine": result["cosine"],
                "load_constrained_oracle_cv": result["load_cv"],
                "load_cv": result["load_cv"],
                "expert_usage_counts": result["expert_usage_counts"],
                "dead_experts": result["dead_experts"],
                "feasible_load_target": result["feasible_load_target"],
                "green_gate": result["green_gate"],
                "hard_feasible": result["hard_feasible"],
                "selected_penalty": result["selected_penalty"],
                "pareto": result["pareto"],
            }
        )
    return {
        "schema_version": 1,
        "status": "LOAD_AWARE_ORACLE_COMPLETE",
        "classification": "FIT_OR_VALIDATION_ARRAYS_ONLY_HOLDOUT_NOT_OPENED",
        "holdout_opened": False,
        "hypothesis": (
            "A bounded load-aware candidate pool can expose whether each frozen "
            "product partition has enough route capacity to clear cosine >= 0.98 "
            "and load CV <= 0.50 without claiming exhaustive optimality."
        ),
        "falsifier": (
            "The bounded oracle remains below cosine 0.98 or above load CV 0.50 "
            "with dead experts, or input provenance is not FIT-only."
        ),
        "decision_enabled": (
            "retain near-target p32 topologies for topology-specific refinement "
            "when bounded evidence is close; never authorize holdout or replay "
            "from a bounded oracle result alone."
        ),
        "target_load_cv": float(target_load_cv),
        "candidate_pool_size": candidate_pool_size,
        "max_combinations": int(max_combinations),
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "max_in_memory_bytes": int(max_in_memory_bytes),
        "storage_dir": str(storage_dir) if storage_dir is not None else None,
        "allow_eager_npz": bool(allow_eager_npz),
        "budget": {
            "target_load_cv": float(target_load_cv),
            "candidate_pool_size": candidate_pool_size,
            "max_combinations": int(max_combinations),
            "iterations": int(iterations),
            "batch_size": int(batch_size),
            "max_in_memory_bytes": int(max_in_memory_bytes),
            "holdout_opened": False,
        },
        "results": rows,
        "code_commit": current_git_commit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--topology", action="append", required=True, help="p16/top4, p32/top5, or p32/top4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-load-cv", type=float, default=0.50)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-in-memory-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--storage-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-eager-npz",
        action="store_true",
        help="allow a small NPZ fixture; real validation runs must use an mmap contribution directory",
    )
    args = parser.parse_args()
    payload = run(
        args.input,
        args.topology,
        target_load_cv=args.target_load_cv,
        candidate_pool_size=args.candidate_pool_size,
        max_combinations=args.max_combinations,
        iterations=args.iterations,
        batch_size=args.batch_size,
        max_in_memory_bytes=args.max_in_memory_bytes,
        storage_dir=args.storage_dir,
        allow_eager_npz=args.allow_eager_npz,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "results": payload["results"], "output": str(args.output)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
