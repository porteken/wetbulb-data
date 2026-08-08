# Copyright (C) 2026 Kenneth Porter

"""Map Canadian cities to ranked ECCC hourly climate-station candidates."""

from __future__ import annotations

import argparse
import logging
import math
from typing import Any

import pandas as pd
import requests

ECCC_STATIONS_URL = "https://api.weather.gc.ca/collections/climate-stations/items"
MAX_DISTANCE_KM = 60.0
MAX_ELEVATION_DIFFERENCE_M = 300.0
MAX_CANDIDATES_PER_CITY = 5
EARTH_RADIUS_KM = 6371.0
LOGGER = logging.getLogger(__name__)


def fetch_hourly_stations(*, session: requests.Session | None = None) -> pd.DataFrame:
    """Return ECCC stations with hourly period metadata."""
    http = session or requests.Session()
    response = http.get(
        ECCC_STATIONS_URL,
        params={"f": "json", "limit": 10_000},
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        if properties.get("HAS_HOURLY_DATA") != "Y":
            continue
        lon, lat = feature["geometry"]["coordinates"][:2]
        first = pd.to_datetime(properties.get("HLY_FIRST_DATE"), errors="coerce")
        last = pd.to_datetime(properties.get("HLY_LAST_DATE"), errors="coerce")
        elevation = pd.to_numeric(properties.get("ELEVATION"), errors="coerce")
        if pd.isna(first) or pd.isna(last) or pd.isna(elevation):
            continue
        rows.append(
            {
                "eccc_station_id": int(properties["STN_ID"]),
                "station_name": properties["STATION_NAME"],
                "lat": float(lat),
                "lon": float(lon),
                "elev_m": float(elevation),
                "start_year": int(first.year),
                "end_year": int(last.year),
            },
        )
    return pd.DataFrame(rows)


def _haversine_km(
    lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    phi1 = math.radians(lat1)
    phi2 = lat2.map(math.radians)
    dphi = phi2 - phi1
    dlambda = lon2.map(math.radians) - math.radians(lon1)
    a = (dphi / 2).map(math.sin) ** 2 + (
        math.cos(phi1) * phi2.map(math.cos) * (dlambda / 2).map(math.sin) ** 2
    )
    return 2 * EARTH_RADIUS_KM * a.map(math.sqrt).map(math.asin)


def build_station_map(
    cities: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    max_candidates_per_city: int = MAX_CANDIDATES_PER_CITY,
) -> pd.DataFrame:
    """Return long-form, period-aware candidates for Canadian cities."""
    rows: list[dict[str, Any]] = []
    canadian = cities[cities["country"] == "CA"] if "country" in cities else cities
    for city in canadian.to_dict("records"):
        nearby = stations.copy()
        nearby["dist_km"] = _haversine_km(
            float(city["lat"]),
            float(city["lng"]),
            nearby["lat"],
            nearby["lon"],
        )
        nearby["elevation_difference_m"] = (
            nearby["elev_m"] - float(city["dem_m"])
        ).abs()
        nearby = nearby[
            (nearby["dist_km"] <= MAX_DISTANCE_KM)
            & (nearby["elevation_difference_m"] <= MAX_ELEVATION_DIFFERENCE_M)
        ].copy()
        if nearby.empty:
            continue
        nearby["span_years"] = nearby["end_year"] - nearby["start_year"] + 1
        recent = nearby.sort_values(
            [
                "end_year",
                "dist_km",
                "elevation_difference_m",
                "span_years",
                "eccc_station_id",
            ],
            ascending=[False, True, True, False, True],
        )
        recent_slots = max(1, (max_candidates_per_city + 1) // 2)
        selected = recent.head(recent_slots)
        historical = nearby.sort_values(
            [
                "span_years",
                "end_year",
                "dist_km",
                "elevation_difference_m",
                "eccc_station_id",
            ],
            ascending=[False, False, True, True, True],
        )
        historical = historical[~historical.index.isin(selected.index)]
        nearby = pd.concat(
            [selected, historical.head(max_candidates_per_city - len(selected))]
        )
        for rank, station in enumerate(nearby.to_dict("records"), start=1):
            rows.append(
                {
                    "location_id": int(city["location_id"]),
                    "eccc_station_id": station["eccc_station_id"],
                    "station_name": station["station_name"],
                    "candidate_rank": rank,
                    "lon": round(float(station["lon"]), 5),
                    "dist_km": round(float(station["dist_km"]), 2),
                    "elev_m": round(float(station["elev_m"]), 1),
                    "elevation_difference_m": round(
                        float(station["elevation_difference_m"]),
                        1,
                    ),
                    "start_year": int(station["start_year"]),
                    "end_year": int(station["end_year"]),
                    "utc_offset_hours": float(city["utc_offset_hours"]),
                },
            )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", default="cities_na.csv")
    parser.add_argument("--out", default="cities_na_eccc_stations.csv")
    parser.add_argument(
        "--max-candidates-per-city",
        type=int,
        default=MAX_CANDIDATES_PER_CITY,
    )
    return parser.parse_args()


def main() -> None:
    """Generate the committed Canadian ECCC station candidate map."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    cities = pd.read_csv(args.cities)
    station_map = build_station_map(
        cities,
        fetch_hourly_stations(),
        max_candidates_per_city=args.max_candidates_per_city,
    )
    station_map.to_csv(args.out, index=False)
    LOGGER.info(
        "Wrote %d ECCC candidates for %d Canadian cities to %s.",
        len(station_map),
        station_map["location_id"].nunique(),
        args.out,
    )


if __name__ == "__main__":
    main()
