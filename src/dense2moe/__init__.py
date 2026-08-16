"""Durable dense-to-MoE experiment primitives."""

from .config import MoEProfile, load_config
from .command import CommandResult, GuardedCommandError, guarded_command, run_guarded
from .provenance import current_git_commit

__all__ = [
    "CommandResult",
    "GuardedCommandError",
    "MoEProfile",
    "current_git_commit",
    "guarded_command",
    "load_config",
    "run_guarded",
]
__version__ = "0.1.0"
