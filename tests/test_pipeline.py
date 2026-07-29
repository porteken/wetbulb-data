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
                "--cities-csv",
                pipeline.NA_CITIES_CSV,
                "--station-map-csv",
                pipeline.NA_STATION_MAP_CSV,
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
    """`--region eu` rejects sources without European coverage."""
    with pytest.raises(argparse.ArgumentTypeError, match="--region eu"):
        pipeline.main(["--wetbulb-source", "nldas", "--region", "eu"])


@pytest.mark.parametrize(
    ("region", "cities", "station_map"),
    [
        (
            "na",
            pipeline.NA_CITIES_CSV,
            pipeline.NA_GHCNH_STATION_MAP_CSV,
        ),
        (
            "eu",
            pipeline.EU_CITIES_CSV,
            pipeline.EU_GHCNH_STATION_MAP_CSV,
        ),
    ],
)
def test_auto_source_uses_ghcnh_for_2026_and_later(
    monkeypatch: pytest.MonkeyPatch,
    region: str,
    cities: str,
    station_map: str,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    pipeline.main(
        [
            "--years",
            "2026",
            "--region",
            region,
            "--out-dir",
            region,
        ],
    )

    assert calls[0][1] == "ghcnh.py"
    assert calls[0][-4:] == [
        "--cities-csv",
        cities,
        "--station-map-csv",
        station_map,
    ]


def test_auto_source_keeps_isd_for_pre_2026_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    pipeline.main(["--years", "2025"])

    assert calls[0][1] == "isd.py"


def test_eu_region_with_default_out_dir_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A shared default out-dir would collide with any US run's parquet filenames."""
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *_a, **_k: None)
    with caplog.at_level("WARNING"):
        pipeline.main(["--region", "eu"])
    assert any("--out-dir" in message for message in caplog.messages)


def test_na_region_uses_combined_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default region passes the combined catalog and station crosswalk."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda command, **_k: calls.append(command)
    )
    pipeline.main(["--years", "2025", "--wetbulb-source", "isd"])
    assert calls[0][-4:] == [
        "--cities-csv",
        pipeline.NA_CITIES_CSV,
        "--station-map-csv",
        pipeline.NA_STATION_MAP_CSV,
    ]


def test_old_region_names_are_rejected() -> None:
    with pytest.raises(SystemExit):
        pipeline.main(["--region", "us"])
