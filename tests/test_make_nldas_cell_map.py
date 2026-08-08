# Copyright (C) 2026 Kenneth Porter

"""Tests for make_nldas_cell_map.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import make_nldas_cell_map as cellmap


class TestBuildCellMap:
    def test_builds_map_and_flags_missing_and_snapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cities_csv = tmp_path / "cities.csv"
        cities_csv.write_text(
            "location_id,lat,lng\n2,41.0,-75.0\n1,40.0,-74.0\n3,42.0,-76.0\n"
        )
        monkeypatch.setattr(
            cellmap.nldas,
            "nearest_grid_indices",
            lambda _lats, _lons: (np.array([10, 11, 12]), np.array([20, 21, 22])),
        )
        monkeypatch.setattr(
            cellmap.nldas,
            "resolve_location_indices",
            lambda _shard_df, _candidate_hours: (
                np.array([10, -1, 13]),
                np.array([20, -1, 22]),
            ),
        )
        with caplog.at_level("ERROR"):
            result = cellmap.build_cell_map(str(cities_csv))

        assert list(result["location_id"]) == [1, 2, 3]
        assert any("no valid land cell" in m for m in caplog.messages)

        assert bool(result.loc[0, "snapped"]) is False
        assert result.loc[0, "cell_lat"] == pytest.approx(
            cellmap.nldas.NLDAS_GRID_LAT0 + 10 * cellmap.nldas.NLDAS_GRID_STEP
        )

        assert bool(result.loc[1, "snapped"]) is True
        assert np.isnan(result.loc[1, "cell_lat"])
        assert np.isnan(result.loc[1, "cell_lon"])

        assert bool(result.loc[2, "snapped"]) is True
        assert result.loc[2, "cell_lat"] == pytest.approx(
            cellmap.nldas.NLDAS_GRID_LAT0 + 13 * cellmap.nldas.NLDAS_GRID_STEP
        )

    def test_no_missing_cells_skips_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cities_csv = tmp_path / "cities.csv"
        cities_csv.write_text("location_id,lat,lng\n1,40.0,-74.0\n")
        monkeypatch.setattr(
            cellmap.nldas,
            "nearest_grid_indices",
            lambda _lats, _lons: (np.array([10]), np.array([20])),
        )
        monkeypatch.setattr(
            cellmap.nldas,
            "resolve_location_indices",
            lambda _shard_df, _candidate_hours: (np.array([10]), np.array([20])),
        )
        with caplog.at_level("ERROR"):
            result = cellmap.build_cell_map(str(cities_csv))
        assert not any("no valid land cell" in m for m in caplog.messages)
        assert bool(result.loc[0, "snapped"]) is False


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["make_nldas_cell_map.py"])
        args = cellmap._parse_args()
        assert args.cities_csv == "cities_na.csv"
        assert args.out == "cities_nldas_cells.csv"

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["make_nldas_cell_map.py", "--cities-csv", "other.csv", "--out", "out.csv"],
        )
        args = cellmap._parse_args()
        assert args.cities_csv == "other.csv"
        assert args.out == "out.csv"


class TestMain:
    def test_writes_output_and_logs_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cities_csv = tmp_path / "cities.csv"
        cities_csv.write_text("location_id,lat,lng\n1,40.0,-74.0\n")
        out_csv = tmp_path / "out.csv"

        monkeypatch.setattr(
            cellmap,
            "_parse_args",
            lambda: cast_namespace(cities_csv=str(cities_csv), out=str(out_csv)),
        )
        monkeypatch.setattr(
            cellmap.nldas,
            "nearest_grid_indices",
            lambda _lats, _lons: (np.array([10]), np.array([20])),
        )
        monkeypatch.setattr(
            cellmap.nldas,
            "resolve_location_indices",
            lambda _shard_df, _candidate_hours: (np.array([11]), np.array([20])),
        )
        with caplog.at_level("INFO"):
            cellmap.main()

        assert out_csv.exists()
        written = pd.read_csv(out_csv)
        assert len(written) == 1
        assert bool(written.loc[0, "snapped"]) is True
        assert any("Wrote 1 row" in m for m in caplog.messages)


def cast_namespace(**kwargs: Any) -> Any:
    from argparse import Namespace

    return Namespace(**kwargs)
