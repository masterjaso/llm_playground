"""Portable durability primitives for atomic file replacement."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def fsync_directory(path: Path | str) -> None:
    """Flush a directory after an atomic replacement.

    POSIX exposes directories as file descriptors, while Windows requires a
    directory handle opened with ``FILE_FLAG_BACKUP_SEMANTICS``.  Keep the
    platform-specific handle setup here so callers retain the same durable
    replacement sequence on both platforms.
    """
    path = Path(path)
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    flush_buffers = kernel32.FlushFileBuffers
    flush_buffers.argtypes = [ctypes.c_void_p]
    flush_buffers.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int

    generic_read = 0x80000000
    generic_write = 0x40000000
    share_read = 0x00000001
    share_write = 0x00000002
    share_delete = 0x00000004
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        str(path),
        generic_read | generic_write,
        share_read | share_write | share_delete,
        None,
        open_existing,
        file_flag_backup_semantics,
        None,
    )
    if handle in (None, invalid_handle):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not flush_buffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close_handle(handle)
