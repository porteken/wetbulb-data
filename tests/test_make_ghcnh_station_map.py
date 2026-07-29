"""Tests for the current GHCNh city crosswalk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import make_ghcnh_station_map as station_map


def test_active_station_ids_requires_temperature_dewpoint_and_pressure(
    tmp_path: Path,
) -> None:
    variables = [
        {"id": "temperature", "coverage": 100},
        {"id": "dew_point_temperature", "coverage": 100},
        {"id": "altimeter", "coverage": 50},
    ]
    (tmp_path / "GHCNh_TEST0000001_2026.json").write_text(
        json.dumps({"stations": [{"dataTypes": variables}]}),
        encoding="utf-8",
    )
    (tmp_path / "GHCNh_TEST0000002_2026.json").write_text(
        json.dumps({"stations": [{"dataTypes": variables[:2]}]}),
        encoding="utf-8",
    )

    assert station_map.active_station_ids(tmp_path) == {"TEST0000001"}


def test_build_station_map_obeys_distance_and_elevation_limits() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [1, 2],
            "lat": [40.0, 50.0],
            "lng": [-74.0, 10.0],
            "dem_m": [10.0, 10.0],
            "utc_offset_hours": [-5.0, 1.0],
        },
    )
    stations = pd.DataFrame(
        {
            "GHCN_ID": ["NEAR", "HIGH", "FAR"],
            "LATITUDE": [40.05, 40.01, 52.0],
            "LONGITUDE": [-74.0, -74.0, 10.0],
            "ELEVATION": [20.0, 500.0, 20.0],
        },
    )

    result = station_map.build_station_map(
        cities,
        stations,
        {"NEAR", "HIGH", "FAR"},
    )

    assert list(result["location_id"]) == [1]
    assert result.iloc[0]["ghcn_id"] == "NEAR"


def test_build_station_map_excludes_usl_marine_style_stations() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [1],
            "lat": [40.0],
            "lng": [-74.0],
            "dem_m": [10.0],
            "utc_offset_hours": [-5.0],
        },
    )
    stations = pd.DataFrame(
        {
            "GHCN_ID": ["USL000MARINE", "USW000LAND01"],
            "LATITUDE": [40.0, 40.05],
            "LONGITUDE": [-74.0, -74.0],
            "ELEVATION": [10.0, 20.0],
        },
    )

    result = station_map.build_station_map(
        cities,
        stations,
        {"USL000MARINE", "USW000LAND01"},
    )

    assert result.iloc[0]["ghcn_id"] == "USW000LAND01"
