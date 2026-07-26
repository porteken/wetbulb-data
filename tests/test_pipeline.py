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


def test_eu_region_appends_eu_crosswalk_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--region eu` points the ISD child command at the EU crosswalk CSVs."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool, shell: bool) -> None:
        calls.append(command)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline.main(
        [
            "--years",
            "2025",
            "--wetbulb-source",
            "isd",
            "--region",
            "eu",
            "--out-dir",
            "eu",
        ]
    )

    assert calls[0][-4:] == [
        "--cities-csv",
        pipeline.EU_CITIES_CSV,
        "--station-map-csv",
        pipeline.EU_STATION_MAP_CSV,
    ]


def test_eu_region_rejects_non_isd_sources() -> None:
    """`--region eu` only supports the ISD worker."""
    with pytest.raises(argparse.ArgumentTypeError, match="--region eu"):
        pipeline.main(["--wetbulb-source", "nldas", "--region", "eu"])


def test_eu_region_with_default_out_dir_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A shared default out-dir would collide with any US run's parquet filenames."""
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *_a, **_k: None)
    with caplog.at_level("WARNING"):
        pipeline.main(["--region", "eu"])
    assert any("--out-dir" in message for message in caplog.messages)


def test_us_region_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default region behaves exactly as before -- no EU flags appended."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda command, **_k: calls.append(command)
    )
    pipeline.main(["--years", "2025", "--wetbulb-source", "isd"])
    assert "--cities-csv" not in calls[0]
