"""FlashMini data format v4 — remote-sharded corpus contract.

v3 (``flashmini.data``) remains frozen for PoC reproduction. v4 adds a
remote-manifest, bounded-cache, deterministically resumable corpus backed by
Hugging Face rather than monolithic local memmaps.
"""

from .canonical import canonicalize_text, content_hash, document_id
from .splits import VAL_FRACTION_DEFAULT, SPLIT_SALT_DEFAULT, assign_split

__all__ = [
    "canonicalize_text",
    "content_hash",
    "document_id",
    "assign_split",
    "SPLIT_SALT_DEFAULT",
    "VAL_FRACTION_DEFAULT",
]

DATA_V4_VERSION = 4
