"""Conservative artifact and memory estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import MoEProfile


@dataclass(frozen=True)
class ResourceEstimate:
    source_checkpoint_bytes: int
    extracted_checkpoint_bytes: int
    activation_bytes: int
    layer_checkpoint_bytes: int
    assembled_checkpoint_bytes: int
    gguf_bytes: int
    quantized_bytes: int
    imatrix_bytes: int
    build_bytes: int
    logs_bytes: int
    safety_margin_bytes: int
    total_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_resources(
    profile: MoEProfile,
    *,
    source_bytes: int | None = None,
    calibration_tokens: int = 100_000,
    activation_dtype_bytes: int = 2,
) -> ResourceEstimate:
    # The default source estimate is intentionally a bound, not a claim about
    # Hub metadata. The CLI replaces it when actual metadata is available.
    source = int(source_bytes or 56 * 1024**3)
    extracted = int(source * 0.82)
    activation = calibration_tokens * profile.num_hidden_layers * profile.hidden_size * activation_dtype_bytes
    layer = int(extracted * 0.16)
    assembled = extracted
    gguf = int(assembled * 1.02)
    quantized = int(assembled * 0.30)
    imatrix = min(activation, 4 * 1024**3)
    build = 2 * 1024**3
    logs = 512 * 1024**2
    subtotal = source + extracted + activation + layer + assembled + gguf + quantized + imatrix + build + logs
    margin = int(subtotal * 0.15)
    return ResourceEstimate(source, extracted, activation, layer, assembled, gguf, quantized, imatrix, build, logs, margin, subtotal + margin)

