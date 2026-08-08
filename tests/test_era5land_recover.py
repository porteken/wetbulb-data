# Copyright (C) 2026 Kenneth Porter

"""Tests for rebuilding the EU ERA5-Land gap-fill from a download snapshot."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import era5land_recover as recover_mod
import gapfill


def _raw_span_frame(hours: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid_time": hours,
            "t2m": [293.15] * len(hours),
            "d2m": [283.15] * len(hours),
            "sp": [101325.0] * len(hours),
        }
    )


def _city_row(location_id: int = 1000, lat: float = 51.5, lng: float = -0.1) -> Any:
    rows = pd.DataFrame(
        {
            "location_id": [location_id],
            "lat": [lat],
            "lng": [lng],
            "utc_offset_hours": [0.0],
        }
    ).itertuples()
    return next(iter(rows))


def _write_span(snapshot: Path, lat: float, lng: float, start: int, end: int) -> Path:
    path = recover_mod._span_csv(snapshot, lat, lng, start, end)
    _raw_span_frame([f"{start}-01-01T00:00:00"]).to_csv(path, index=False)
    return path


class TestLiveRunGuard:
    def test_current_process_is_active(self) -> None:
        import os

        assert recover_mod._live_run_is_active(os.getpid())

    def test_an_unused_pid_is_inactive(self) -> None:
        assert not recover_mod._live_run_is_active(999999999)


class TestSpanCsv:
    def test_name_matches_the_downloader(self, tmp_path: Path) -> None:
        import era5land

        path = recover_mod._span_csv(tmp_path, 51.5, -0.1, 2000, 2001)
        assert path.name == era5land.span_filename(51.5, -0.1, 2000, 2001)


class TestReadSpan:
    def test_reads_a_plain_csv(self, tmp_path: Path) -> None:
        path = _write_span(tmp_path, 51.5, -0.1, 2000, 2000)
        result = recover_mod._read_span(path)
        assert result is not None
        assert list(result["t2m"]) == [293.15]

    def test_truncated_file_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "truncated.csv"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("readme.txt", "no data here")

        with caplog.at_level("WARNING"):
            result = recover_mod._read_span(path)

        assert result is None
        assert any("Unreadable or truncated" in m for m in caplog.messages)


class TestCityFrame:
    def test_present_spans_are_concatenated(self, tmp_path: Path) -> None:
        _write_span(tmp_path, 51.5, -0.1, 2000, 2001)
        _write_span(tmp_path, 51.5, -0.1, 2010, 2010)

        frame, missing = recover_mod._city_frame(
            tmp_path, _city_row(), [2000, 2001, 2010]
        )

        assert missing == []
        assert frame is not None
        assert len(frame) == 2

    def test_absent_span_is_reported_as_missing(self, tmp_path: Path) -> None:
        _write_span(tmp_path, 51.5, -0.1, 2000, 2000)

        frame, missing = recover_mod._city_frame(tmp_path, _city_row(), [2000, 2010])

        assert missing == [(2010, 2010)]
        assert frame is not None
        assert len(frame) == 1

    def test_nothing_readable_yields_no_frame(self, tmp_path: Path) -> None:
        frame, missing = recover_mod._city_frame(tmp_path, _city_row(), [2000])
        assert frame is None
        assert missing == [(2000, 2000)]


class TestFilledFromFrames:
    @staticmethod
    def _hourly() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": [1000],
                "time": [pd.Timestamp("2020-01-01")],
                "Tair": [290.0],
                "Qair": [0.01],
                "PSurf": [100000.0],
            }
        )

    @staticmethod
    def _missing_cells() -> pd.DataFrame:
        return pd.DataFrame(
            {"location_id": [1000], "date": pd.to_datetime(["2020-01-01"])}
        )

    def test_no_frames_returns_none(self) -> None:
        assert recover_mod._filled_from_frames([], self._missing_cells()) is None

    def test_empty_daily_aggregate_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            recover_mod.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(columns=pd.Index(["location_id", "date"])),
        )
        result = recover_mod._filled_from_frames(
            [self._hourly()], self._missing_cells()
        )
        assert result is None

    def test_data_matching_no_gap_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            recover_mod.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1000],
                    "date": [pd.Timestamp("2021-06-01").date()],
                    "wetbulb": [20.0],
                    "wetbulb_avg": [19.0],
                }
            ),
        )
        result = recover_mod._filled_from_frames(
            [self._hourly()], self._missing_cells()
        )
        assert result is None

    def test_matching_rows_are_tagged_with_the_era5land_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            recover_mod.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1000, 1000],
                    "date": [
                        pd.Timestamp("2020-01-01").date(),
                        pd.Timestamp("2020-01-02").date(),
                    ],
                    "wetbulb": [20.0, 21.0],
                    "wetbulb_avg": [19.0, 20.0],
                }
            ),
        )

        result = recover_mod._filled_from_frames(
            [self._hourly()], self._missing_cells()
        )

        assert result is not None
        assert len(result) == 1
        assert result["date"].iloc[0] == pd.Timestamp("2020-01-01").date()
        assert (result["source"] == recover_mod.ERA5LAND_SOURCE).all()


class _RecoverHarness:
    """Stubs the shard resolution so `recover` runs against a fake snapshot."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            recover_mod.nldas,
            "load_nldas_city_shard",
            lambda *_a: pd.DataFrame(
                {"location_id": [1000], "lat": [51.5], "lng": [-0.1]}
            ),
        )
        monkeypatch.setattr(gapfill, "pending_years", lambda *_a, **_k: [2020])
        monkeypatch.setattr(
            gapfill,
            "find_missing_cells",
            lambda *_a, **_k: pd.DataFrame(
                {"location_id": [1000], "date": pd.to_datetime(["2020-01-01"])}
            ),
        )
        monkeypatch.setattr(gapfill, "_filter_material_gaps", lambda cells, _n: cells)
        monkeypatch.setattr(
            recover_mod,
            "_load_utc_offsets",
            lambda _path: pd.DataFrame(
                {"location_id": [1000], "utc_offset_hours": [0.0]}
            ),
        )
        monkeypatch.setattr(
            recover_mod.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1000],
                    "date": [pd.Timestamp("2020-01-01").date()],
                    "wetbulb": [20.0],
                    "wetbulb_avg": [19.0],
                }
            ),
        )
        self.writes: list[Any] = []
        monkeypatch.setattr(
            recover_mod,
            "write_pending_year_batches",
            lambda *args, **_k: self.writes.append(args),
        )


def _recover(snapshot: Path, out_dir: Path, **overrides: Any) -> int:
    kwargs: dict[str, Any] = {
        "allow_partial": False,
        "dry_run": False,
        **overrides,
    }
    return recover_mod.recover(
        str(snapshot),
        2020,
        2020,
        str(out_dir),
        1,
        "cities_eu.csv",
        "cities_eu_isd_stations.csv",
        **kwargs,
    )


class TestRecover:
    def test_missing_snapshot_directory_fails(self, tmp_path: Path) -> None:
        assert _recover(tmp_path / "absent", tmp_path) == 1

    def test_nothing_pending_is_a_clean_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _RecoverHarness(monkeypatch)
        monkeypatch.setattr(gapfill, "pending_years", lambda *_a, **_k: [])
        assert _recover(tmp_path, tmp_path) == 0

    def test_no_material_gaps_is_a_clean_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _RecoverHarness(monkeypatch)
        monkeypatch.setattr(
            gapfill,
            "find_missing_cells",
            lambda *_a, **_k: pd.DataFrame(
                {
                    "location_id": pd.Series(dtype="int64"),
                    "date": pd.Series(dtype="datetime64[ns]"),
                }
            ),
        )
        assert _recover(tmp_path, tmp_path) == 0

    def test_partial_snapshot_refuses_to_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        harness = _RecoverHarness(monkeypatch)
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()

        with caplog.at_level("WARNING"):
            code = _recover(snapshot, tmp_path)

        assert code == 2
        assert harness.writes == []
        assert any("Refusing to write a partial" in m for m in caplog.messages)

    def test_allow_partial_writes_what_is_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = _RecoverHarness(monkeypatch)
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        _write_span(snapshot, 51.5, -0.1, 2020, 2020)

        code = _recover(snapshot, tmp_path, allow_partial=True)

        assert code == 0
        assert len(harness.writes) == 1

    def test_unusable_snapshot_returns_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = _RecoverHarness(monkeypatch)
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()

        code = _recover(snapshot, tmp_path, allow_partial=True)

        assert code == 1
        assert harness.writes == []

    def test_complete_snapshot_writes_the_gapfill_partitions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = _RecoverHarness(monkeypatch)
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        _write_span(snapshot, 51.5, -0.1, 2020, 2020)

        code = _recover(snapshot, tmp_path)

        assert code == 0
        assert len(harness.writes) == 1
        written = harness.writes[0][0]
        assert (written["source"] == recover_mod.ERA5LAND_SOURCE).all()

    def test_dry_run_reports_without_writing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        harness = _RecoverHarness(monkeypatch)
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        _write_span(snapshot, 51.5, -0.1, 2020, 2020)

        with caplog.at_level("INFO"):
            code = _recover(snapshot, tmp_path, dry_run=True)

        assert code == 0
        assert harness.writes == []
        assert any("not writing" in m for m in caplog.messages)


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(recover_mod.sys, "argv", ["era5land_recover.py"])
        args = recover_mod._parse_args()
        assert args.snapshot_dir == recover_mod.DEFAULT_SNAPSHOT
        assert args.cities_csv == recover_mod.EU_CITIES_CSV
        assert args.station_map_csv == recover_mod.EU_STATION_MAP_CSV
        assert not args.allow_partial
        assert not args.dry_run


class TestMain:
    @staticmethod
    def _args(**overrides: Any) -> Any:
        import argparse

        defaults = {
            "snapshot_dir": "snapshot",
            "cities_csv": "cities_eu.csv",
            "station_map_csv": "cities_eu_isd_stations.csv",
            "start_year": 2020,
            "end_year": 2020,
            "out_dir": "eu",
            "min_missing_days": 19,
            "allow_partial": False,
            "dry_run": False,
            "ignore_live_run": False,
        }
        return argparse.Namespace(**{**defaults, **overrides})

    def test_a_live_run_blocks_recovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(recover_mod, "_parse_args", self._args)
        monkeypatch.setattr(recover_mod, "_live_run_is_active", lambda _pid: True)

        with pytest.raises(SystemExit) as exc_info:
            recover_mod.main()

        assert exc_info.value.code == 3

    def test_ignore_live_run_proceeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            recover_mod, "_parse_args", lambda: self._args(ignore_live_run=True)
        )
        monkeypatch.setattr(recover_mod, "_live_run_is_active", lambda _pid: True)
        monkeypatch.setattr(recover_mod, "recover", lambda *_a, **_k: 0)

        with pytest.raises(SystemExit) as exc_info:
            recover_mod.main()

        assert exc_info.value.code == 0

    def test_exits_with_the_recover_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(recover_mod, "_parse_args", self._args)
        monkeypatch.setattr(recover_mod, "_live_run_is_active", lambda _pid: False)
        monkeypatch.setattr(recover_mod, "recover", lambda *_a, **_k: 2)

        with pytest.raises(SystemExit) as exc_info:
            recover_mod.main()

        assert exc_info.value.code == 2
