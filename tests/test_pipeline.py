"""Tests for the top-level pipeline orchestrator."""

from __future__ import annotations

import argparse

import pytest

import pipeline


def test_main_runs_fixed_python_command_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline options are passed as arguments, not shell input."""
    calls: list[tuple[list[str], bool, bool]] = []

    def fake_run(command: list[str], *, check: bool, shell: bool) -> None:
        calls.append((command, check, shell))

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline.main(["--years", "2025", "--wetbulb-source", "isd"])

    assert calls == [
        (
            [
                pipeline.sys.executable,
                "isd.py",
                "--start-year",
                "2025",
                "--end-year",
                "2025",
                "--city-shard-count",
                "1",
                "--concurrency",
                "4",
                "--out-dir",
                ".",
            ],
            True,
            False,
        )
    ]


def test_main_rejects_an_output_directory_that_looks_like_an_option() -> None:
    """A child command cannot receive a user-supplied option as its path."""
    with pytest.raises(argparse.ArgumentTypeError, match="--out-dir"):
        pipeline.main(["--out-dir=-unsafe"])
