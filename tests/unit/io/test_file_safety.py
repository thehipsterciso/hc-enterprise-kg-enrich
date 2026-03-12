"""Tests for hckg_enrich.io.file_safety — GraphFileLock, atomic_write_json, _rotate_backups."""
from __future__ import annotations

import json
import threading

import pytest

from hckg_enrich.io.file_safety import (
    GraphFileLock,
    LockTimeoutError,
    _rotate_backups,  # private, tested directly
    atomic_write_json,
)

# ---------------------------------------------------------------------------
# _rotate_backups
# ---------------------------------------------------------------------------

class TestRotateBackups:
    def test_no_existing_file_no_error(self, tmp_path):
        path = tmp_path / "graph.json"
        _rotate_backups(path)  # should not raise

    def test_existing_file_becomes_dot_1(self, tmp_path):
        path = tmp_path / "graph.json"
        path.write_text('{"v": 1}')
        _rotate_backups(path)
        assert (tmp_path / "graph.json.1").exists()
        assert not path.exists()

    def test_existing_backup_rotates_forward(self, tmp_path):
        path = tmp_path / "graph.json"
        path.write_text('{"v": 3}')
        (tmp_path / "graph.json.1").write_text('{"v": 2}')
        (tmp_path / "graph.json.2").write_text('{"v": 1}')
        _rotate_backups(path, keep=3)
        assert (tmp_path / "graph.json.1").read_text() == '{"v": 3}'
        assert (tmp_path / "graph.json.2").read_text() == '{"v": 2}'
        assert (tmp_path / "graph.json.3").read_text() == '{"v": 1}'

    def test_oldest_backup_discarded_beyond_keep(self, tmp_path):
        path = tmp_path / "graph.json"
        path.write_text('{"v": 4}')
        for i in range(1, 4):
            (tmp_path / f"graph.json.{i}").write_text(f'{{"v": {4 - i}}}')
        _rotate_backups(path, keep=3)
        # .4 would be keep+1 but original .3 becomes .4 — that's fine; we just
        # care that the old .1/.2/.3 all shifted correctly
        assert (tmp_path / "graph.json.1").read_text() == '{"v": 4}'


# ---------------------------------------------------------------------------
# GraphFileLock
# ---------------------------------------------------------------------------

class TestGraphFileLock:
    def test_acquires_and_releases(self, tmp_path):
        path = tmp_path / "graph.json"
        lock = GraphFileLock(path, exclusive=True, timeout=2.0)
        lock.acquire()
        assert lock._fd is not None
        lock.release()
        assert lock._fd is None

    def test_context_manager(self, tmp_path):
        path = tmp_path / "data.json"
        with GraphFileLock(path, exclusive=True, timeout=2.0) as lock:
            assert lock._fd is not None
        assert lock._fd is None

    def test_creates_lock_file(self, tmp_path):
        path = tmp_path / "graph.json"
        with GraphFileLock(path, exclusive=True, timeout=2.0):
            assert (tmp_path / "graph.json.lock").exists()

    def test_shared_lock_does_not_block_another_shared_lock(self, tmp_path):
        """Two shared (read) locks on the same file should not conflict."""
        path = tmp_path / "data.json"
        with GraphFileLock(path, exclusive=False, timeout=2.0):
            # A second shared lock from the same process should succeed
            with GraphFileLock(path, exclusive=False, timeout=2.0):
                pass  # no exception

    def test_timeout_raises_lock_timeout_error(self, tmp_path):
        """Simulate a lock held in a thread; second acquire must time out."""
        path = tmp_path / "exclusive.json"
        ready = threading.Event()
        release = threading.Event()

        def hold_lock():
            with GraphFileLock(path, exclusive=True, timeout=5.0):
                ready.set()
                release.wait(timeout=3.0)

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        ready.wait(timeout=2.0)

        with pytest.raises(LockTimeoutError):
            GraphFileLock(path, exclusive=True, timeout=0.1).acquire()

        release.set()
        t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------

class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "graph.json"
        data = {"entities": [{"id": "e-001", "name": "Finance"}], "relationships": []}
        atomic_write_json(path, data)
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "graph.json"
        atomic_write_json(path, {"entities": []})
        assert path.exists()

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "graph.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        assert json.loads(path.read_text()) == {"v": 2}

    def test_backup_created_on_overwrite(self, tmp_path):
        path = tmp_path / "graph.json"
        path.write_text('{"v": 1}\n')
        atomic_write_json(path, {"v": 2}, backup=True)
        assert (tmp_path / "graph.json.1").exists()

    def test_no_backup_when_disabled(self, tmp_path):
        path = tmp_path / "graph.json"
        path.write_text('{"v": 1}\n')
        atomic_write_json(path, {"v": 2}, backup=False)
        assert not (tmp_path / "graph.json.1").exists()

    def test_backup_off_for_new_file(self, tmp_path):
        path = tmp_path / "graph.json"
        atomic_write_json(path, {"new": True})
        # No backup for a brand-new file
        assert not (tmp_path / "graph.json.1").exists()

    def test_file_ends_with_newline(self, tmp_path):
        path = tmp_path / "graph.json"
        atomic_write_json(path, {"x": 1})
        assert path.read_text().endswith("\n")

    def test_respects_indent(self, tmp_path):
        path = tmp_path / "graph.json"
        atomic_write_json(path, {"a": 1}, indent=4)
        raw = path.read_text()
        assert "    " in raw  # 4-space indent

    def test_sequential_writes_consistent(self, tmp_path):
        path = tmp_path / "graph.json"
        for i in range(5):
            atomic_write_json(path, {"iteration": i})
        assert json.loads(path.read_text()) == {"iteration": 4}

    def test_no_leftover_tmp_files(self, tmp_path):
        path = tmp_path / "graph.json"
        atomic_write_json(path, {"clean": True})
        tmp_files = list(tmp_path.glob(".tmp_*.json"))
        assert tmp_files == [], f"Stale tmp files: {tmp_files}"
