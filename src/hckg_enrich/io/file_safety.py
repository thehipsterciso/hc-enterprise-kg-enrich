"""File safety primitives for graph persistence.

Provides advisory file locking, atomic writes, and backup rotation so that
concurrent enrichment runs and crashes cannot corrupt graph.json.

Usage::

    from hckg_enrich.io.file_safety import atomic_write_json, GraphFileLock

    # Safe write with backup rotation
    atomic_write_json(path, graph_dict)

    # Manual exclusive lock
    with GraphFileLock(path, exclusive=True):
        path.write_text(content)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class LockTimeoutError(OSError):
    """Raised when a file lock cannot be acquired within the timeout."""


class GraphFileLock:
    """Advisory file lock (shared or exclusive) with timeout.

    Uses a companion ``.lock`` file to avoid interfering with the data file.
    POSIX: ``fcntl.flock()``. Windows: ``msvcrt.locking()``.

    Parameters
    ----------
    path:
        Path to the file to lock.
    exclusive:
        ``True`` for write lock, ``False`` for read (shared) lock.
    timeout:
        Maximum seconds to wait. ``0`` = non-blocking.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        exclusive: bool = True,
        timeout: float = 10.0,
    ) -> None:
        self._path = Path(path)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._exclusive = exclusive
        self._timeout = timeout
        self._fd: int | None = None

    def acquire(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        self._fd = os.open(str(self._lock_path), flags, 0o666)
        deadline = time.monotonic() + self._timeout
        poll = 0.05
        while True:
            try:
                self._try_lock()
                return
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise LockTimeoutError(
                        f"Could not acquire {'exclusive' if self._exclusive else 'shared'} "
                        f"lock on {self._path} within {self._timeout}s"
                    ) from None
                time.sleep(poll)

    def release(self) -> None:
        if self._fd is not None:
            try:
                self._try_unlock()
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> GraphFileLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _try_lock(self) -> None:
        assert self._fd is not None
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl
            flag = fcntl.LOCK_NB | (fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH)
            fcntl.flock(self._fd, flag)

    def _try_unlock(self) -> None:
        assert self._fd is not None
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_UN)


def _rotate_backups(path: Path, keep: int = 3) -> None:
    """Rotate .1 / .2 / .3 backup files before overwriting path."""
    for i in range(keep, 0, -1):
        src = path.with_suffix(path.suffix + f".{i}")
        dst = path.with_suffix(path.suffix + f".{i + 1}")
        if src.exists():
            src.rename(dst)
    if path.exists():
        path.rename(path.with_suffix(path.suffix + ".1"))


def atomic_write_json(path: Path, data: Any, *, backup: bool = True, indent: int = 2) -> None:
    """Write *data* as JSON to *path* atomically with optional backup rotation.

    Uses ``tempfile.mkstemp`` + ``os.replace`` so the file is never
    partially written. An exclusive lock is held for the duration.

    Parameters
    ----------
    path:
        Destination path for the JSON file.
    data:
        JSON-serialisable object.
    backup:
        If ``True``, rotate existing backups before writing.
    indent:
        JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=indent) + "\n"

    with GraphFileLock(path, exclusive=True, timeout=15.0):
        if backup and path.exists():
            _rotate_backups(path)

        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
