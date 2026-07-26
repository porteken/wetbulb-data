"""Tests for the direct North America global-ISD station mapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import make_isd_station_map_na as na_stationmap

if TYPE_CHECKING:
    import pytest


def test_build_station_map_na_uses_north_america_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        na_stationmap,
        "build_station_map_region",
        lambda cities_csv, **kwargs: calls.append((cities_csv, kwargs)) or "map",
    )

    result = na_stationmap.build_station_map_na(
        "cities.csv", start_year=2000, end_year=2025
    )

    assert result == "map"
    assert calls == [
        (
            "cities.csv",
            {
                "min_lat": na_stationmap.NORTH_AMERICA_HISTORY_MIN_LAT,
                "max_lat": na_stationmap.NORTH_AMERICA_HISTORY_MAX_LAT,
                "min_lon": na_stationmap.NORTH_AMERICA_HISTORY_MIN_LON,
                "max_lon": na_stationmap.NORTH_AMERICA_HISTORY_MAX_LON,
                "start_year": 2000,
                "end_year": 2025,
            },
        )
    ]
