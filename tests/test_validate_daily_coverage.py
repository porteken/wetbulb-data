# Copyright (C) 2026 Kenneth Porter

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import validate_daily_coverage as coverage
from partition_io import write_batch_partition


def _write_cities(path: Path) -> None:
    pd.DataFrame({"location_id": [1, 2]}).to_csv(path, index=False)


def _write_days(root: Path, dates: pd.DatetimeIndex) -> None:
    frame = pd.DataFrame(
        {
            "location_id": [location_id for location_id in (1, 2) for _ in dates],
            "date": list(dates) * 2,
            "wetbulb": 20.0,
            "wetbulb_avg": 18.0,
            "source": "era5land",
        }
    )
    write_batch_partition(str(root), 2020, 0, frame, 0, file_prefix="wetbulb_fill")


def test_complete_calendar_passes(monkeypatch: Any, tmp_path: Path) -> None:
    cities = tmp_path / "cities.csv"
    root = tmp_path / "data"
    _write_cities(cities)
    _write_days(root, pd.date_range("2020-01-01", "2020-01-03"))
    monkeypatch.setattr(
        coverage, "expected_end", lambda *_a: pd.Timestamp("2020-01-03")
    )

    coverage.validate_coverage(str(cities), str(root), 2020, 8)


def test_missing_city_day_fails(monkeypatch: Any, tmp_path: Path) -> None:
    cities = tmp_path / "cities.csv"
    root = tmp_path / "data"
    _write_cities(cities)
    _write_days(root, pd.date_range("2020-01-01", "2020-01-02"))
    monkeypatch.setattr(
        coverage, "expected_end", lambda *_a: pd.Timestamp("2020-01-03")
    )

    with pytest.raises(RuntimeError, match="2 missing city-days"):
        coverage.validate_coverage(str(cities), str(root), 2020, 8)


def test_incremental_range_ignores_earlier_gaps(tmp_path: Path) -> None:
    cities = tmp_path / "cities.csv"
    root = tmp_path / "data"
    _write_cities(cities)
    _write_days(root, pd.date_range("2020-06-10", "2020-06-12"))

    coverage.validate_coverage(
        str(cities),
        str(root),
        2020,
        8,
        "2020-06-10",
        "2020-06-12",
    )
