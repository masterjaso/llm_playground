"""Compatibility shim for the authoritative :mod:`dense2moe.assembly` code."""

from dense2moe.assembly.checkpoint import assemble_checkpoint, target_state_dict_inventory

__all__ = ["assemble_checkpoint", "target_state_dict_inventory"]
