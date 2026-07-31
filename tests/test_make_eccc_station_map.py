from __future__ import annotations

import pandas as pd

import make_eccc_station_map as station_map


def test_build_station_map_keeps_canada_and_ranks_near_recent_records() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [1, 2],
            "country": ["CA", "US"],
            "lat": [49.28, 40.0],
            "lng": [-123.12, -74.0],
            "dem_m": [5.0, 10.0],
            "utc_offset_hours": [-8.0, -5.0],
        },
    )
    stations = pd.DataFrame(
        {
            "eccc_station_id": [10, 20],
            "station_name": ["Near short", "Far long"],
            "lat": [49.29, 49.35],
            "lon": [-123.12, -123.12],
            "elev_m": [5.0, 5.0],
            "start_year": [2010, 1953],
            "end_year": [2026, 2026],
        },
    )

    result = station_map.build_station_map(cities, stations)

    assert result["location_id"].unique().tolist() == [1]
    assert result.iloc[0]["eccc_station_id"] == 10
    assert result.iloc[0]["candidate_rank"] == 1


def test_build_station_map_prefers_recent_station_over_retired_long_record() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [1],
            "country": ["CA"],
            "lat": [49.28],
            "lng": [-123.12],
            "dem_m": [5.0],
            "utc_offset_hours": [-8.0],
        },
    )
    stations = pd.DataFrame(
        {
            "eccc_station_id": [10, 20],
            "station_name": ["Retired long", "Active short"],
            "lat": [49.29, 49.30],
            "lon": [-123.12, -123.12],
            "elev_m": [5.0, 5.0],
            "start_year": [1953, 2010],
            "end_year": [2013, 2026],
        },
    )

    result = station_map.build_station_map(cities, stations, max_candidates_per_city=2)

    assert result.iloc[0]["eccc_station_id"] == 20
    assert result.iloc[1]["eccc_station_id"] == 10
