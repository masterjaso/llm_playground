from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.run_load_aware_oracle import _load_arrays


def _write_store(root) -> None:
    shapes = {
        "shared": (5, 3),
        "routed": (5, 4, 3),
        "target": (5, 3),
    }
    arrays = {}
    for name, shape in shapes.items():
        path = root / f"{name}.npy"
        values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        values[:] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values.flush()
        del values
        arrays[name] = {"path": path.name, "shape": list(shape), "dtype": "float32"}
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "storage_mode": "npy_memmap", "arrays": arrays}) + "\n",
        encoding="utf-8",
    )


def test_load_arrays_uses_read_only_npy_memmaps(tmp_path) -> None:
    _write_store(tmp_path)

    (shared, routed, target), metadata = _load_arrays(tmp_path)

    assert metadata["format"] == "npy_memmap"
    assert metadata["manifest_sha256"]
    assert all(isinstance(values, np.memmap) for values in (shared, routed, target))
    assert all(not values.flags.writeable for values in (shared, routed, target))
    assert routed.shape == (5, 4, 3)


def test_npz_is_rejected_without_explicit_fixture_escape_hatch(tmp_path) -> None:
    fixture = tmp_path / "fixture.npz"
    np.savez(fixture, shared=np.zeros((1, 1)), routed=np.zeros((1, 2, 1)), target=np.zeros((1, 1)))

    with pytest.raises(ValueError, match="refusing eager NPZ"):
        _load_arrays(fixture)


def test_small_npz_fixture_requires_and_accepts_explicit_opt_in(tmp_path) -> None:
    fixture = tmp_path / "fixture.npz"
    np.savez(fixture, shared=np.zeros((1, 1)), routed=np.zeros((1, 2, 1)), target=np.zeros((1, 1)))

    (shared, routed, target), metadata = _load_arrays(fixture, allow_eager_npz=True)

    assert metadata["format"] == "eager_npz"
    assert not isinstance(shared, np.memmap)
    assert routed.shape == (1, 2, 1)
    assert target.shape == (1, 1)
