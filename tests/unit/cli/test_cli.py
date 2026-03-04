"""Tests for CLI entry points (no real API calls)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the hckg-enrich CLI via the installed script and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "hckg_enrich.cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_run_command_help():
    result = _run_cli("run", "--help")
    assert result.returncode == 0
    assert "--graph" in result.stdout


def test_demo_command_help():
    result = _run_cli("demo", "--help")
    assert result.returncode == 0
    assert "--out" in result.stdout


def test_run_command_missing_graph():
    result = _run_cli(
        "run",
        "--graph", "/nonexistent/path/graph.json",
        "--out", "/tmp/out.json",
    )
    assert result.returncode == 1
    assert "not found" in result.stderr


def test_run_command_with_valid_graph_exits_without_api_key():
    """With a valid graph but no API key, file-not-found path must NOT be taken."""
    graph_data = {"entities": [], "relationships": []}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(graph_data, f)
        graph_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
        out_path = out_f.name

    # Pass no API key env — should fail at provider init, not at file-not-found
    import os
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY",)}
    result = _run_cli(
        "run",
        "--graph", graph_path,
        "--out", out_path,
        "--no-search",
        env=env,
    )
    # The graph file exists, so the "not found" path must NOT be taken.
    # It will either succeed (0 entities) or fail with a missing key error.
    assert "not found" not in result.stderr
    Path(graph_path).unlink(missing_ok=True)
    Path(out_path).unlink(missing_ok=True)
