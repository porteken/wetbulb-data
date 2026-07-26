from __future__ import annotations

import pandas as pd

from make_eccc_station_map import rank_stations


def test_rank_stations_prefers_full_record_before_distance() -> None:
    inventory = pd.DataFrame(
        {
            "station_id": [1, 2],
            "lat": [45.01, 45.1],
            "lng": [-75.0, -75.0],
            "elevation_m": [10, 10],
            "hourly_first_year": [2000, 1990],
            "hourly_last_year": [2025, 2025],
        }
    )

    result = rank_stations(
        inventory,
        city_lat=45,
        city_lng=-75,
        start_year=1991,
        end_year=2025,
    )

    assert result.iloc[0]["station_id"] == 2
