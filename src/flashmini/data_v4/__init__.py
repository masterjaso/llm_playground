"""FlashMini data format v4 — remote-sharded corpus contract.

v3 (``flashmini.data``) remains frozen for PoC reproduction. v4 adds a
remote-manifest, bounded-cache, deterministically resumable corpus backed by
Hugging Face rather than monolithic local memmaps.
"""

from .canonical import canonicalize_text, content_hash, document_id
from .packing import PACKING_VERSION, pack_document, tokenizer_identity
from .sampler import HierarchicalSampler, RemoteShardDataset
from .scheduler import DeficitScheduler, DeficitTokenScheduler
from .splits import SPLIT_SALT_DEFAULT, VAL_FRACTION_DEFAULT, assign_split
from .tokenizer import TokenizerSpec

__all__ = [
    "PACKING_VERSION",
    "SPLIT_SALT_DEFAULT",
    "VAL_FRACTION_DEFAULT",
    "DeficitScheduler",
    "DeficitTokenScheduler",
    "HierarchicalSampler",
    "RemoteShardDataset",
    "TokenizerSpec",
    "assign_split",
    "canonicalize_text",
    "content_hash",
    "document_id",
    "pack_document",
    "tokenizer_identity",
]

DATA_V4_VERSION = 4
