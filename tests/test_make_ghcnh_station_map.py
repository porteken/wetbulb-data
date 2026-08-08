# Copyright (C) 2026 Kenneth Porter

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


def test_build_station_map_excludes_known_humidity_outlier() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [24],
            "lat": [39.76],
            "lng": [-104.88],
            "dem_m": [1651.0],
            "utc_offset_hours": [-7.0],
        },
    )
    stations = pd.DataFrame(
        {
            "GHCN_ID": ["USW00023012", "USW00023036"],
            "LATITUDE": [39.78, 39.83],
            "LONGITUDE": [-104.90, -104.75],
            "ELEVATION": [1645.0, 1726.1],
        },
    )

    result = station_map.build_station_map(
        cities,
        stations,
        {"USW00023012", "USW00023036"},
    )

    assert result["ghcn_id"].tolist() == ["USW00023036"]


def test_build_station_map_returns_multiple_candidates_ranked_by_coverage() -> None:
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
            "GHCN_ID": ["NEAREST_SPARSE", "FARTHER_COMPLETE"],
            "LATITUDE": [40.01, 40.05],
            "LONGITUDE": [-74.0, -74.0],
            "ELEVATION": [10.0, 10.0],
        },
    )

    result = station_map.build_station_map(
        cities,
        stations,
        {"NEAREST_SPARSE": 1.0, "FARTHER_COMPLETE": 99.0},
        max_candidates_per_city=2,
    )

    assert list(result["ghcn_id"]) == ["FARTHER_COMPLETE", "NEAREST_SPARSE"]
    assert list(result["candidate_rank"]) == [1, 2]


def test_year_specific_map_does_not_mix_station_inventories() -> None:
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
            "GHCN_ID": ["OLD", "NEW"],
            "LATITUDE": [40.01, 40.01],
            "LONGITUDE": [-74.0, -74.0],
            "ELEVATION": [10.0, 10.0],
        },
    )

    result = station_map.build_year_specific_station_map(
        cities,
        stations,
        {2000: {"OLD": 100.0}, 2025: {"NEW": 100.0}},
    )

    assert result[["year", "ghcn_id"]].to_dict("records") == [
        {"year": 2000, "ghcn_id": "OLD"},
        {"year": 2025, "ghcn_id": "NEW"},
    ]
