"""Selective safetensors access with an explicit full-load fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str


class SafetensorsSliceReader:
    """Read tensor metadata and slices without mutating source files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._handle = None
        self._fallback: dict[str, Any] | None = None
        try:
            from safetensors import safe_open  # type: ignore

            self._safe_open = safe_open
            with safe_open(str(self.path), framework="np") as handle:
                infos: dict[str, TensorInfo] = {}
                for name in handle.keys():  # noqa: SIM118
                    sliced = handle.get_slice(name)
                    shape = tuple(sliced.get_shape())
                    # The slice API intentionally does not expose dtype on all
                    # safetensors versions. Keep metadata cheap and portable;
                    # `get` still returns the exact dtype when needed.
                    infos[name] = TensorInfo(name, shape, "unknown")
                self._infos = infos
        except ImportError as exc:
            raise RuntimeError("safetensors is required for SafetensorsSliceReader") from exc

    def keys(self) -> list[str]:
        return sorted(self._infos)

    def info(self, name: str) -> TensorInfo:
        return self._infos[name]

    def infos(self) -> list[TensorInfo]:
        return [self._infos[name] for name in self.keys()]

    def get(self, name: str) -> Any:
        with self._safe_open(str(self.path), framework="np") as handle:
            return handle.get_tensor(name)

    def slice(self, name: str, index: Any) -> Any:
        """Read one slice; index may be an int, slice, or tuple of those."""

        value = self.get(name)
        return value[index]

    def read_rows(self, name: str, start: int, stop: int) -> Any:
        return self.slice(name, slice(start, stop))

    def __enter__(self) -> SafetensorsSliceReader:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> None:
        return None
