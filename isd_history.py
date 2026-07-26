"""Shared helpers for matching cities to NCEI ISD stations."""

from __future__ import annotations

import importlib
from typing import Any, cast

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))

type DataFrame = Any

EARTH_RADIUS_KM = 6371.0088
MAX_STATION_DISTANCE_KM = 60.0
MAX_ELEV_DELTA_M = 300.0

PLACEHOLDER_USAF = "999999"
PLACEHOLDER_WBAN = "99999"


def haversine_km(
    lat1: DataFrame | float,
    lon1: DataFrame | float,
    lat2: DataFrame,
    lon2: DataFrame,
) -> DataFrame:
    """Return the great-circle distance in km between one point and a series of points."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat_rad = np.radians(lat2 - lat1)
    dlon_rad = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat_rad / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad / 2.0) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def prepare_history(
    history: DataFrame,
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> DataFrame:
    """Numeric-parse the global inventory and pre-filter it to a bounding box."""
    history = history.copy()
    history["LAT_NUM"] = pd.to_numeric(history["LAT"], errors="coerce")
    history["LON_NUM"] = pd.to_numeric(history["LON"], errors="coerce")
    history["ELEV_NUM"] = pd.to_numeric(history["ELEV(M)"], errors="coerce")
    history = history.dropna(subset=["LAT_NUM", "LON_NUM", "ELEV_NUM"])
    return history[
        (history["LAT_NUM"] >= min_lat)
        & (history["LAT_NUM"] <= max_lat)
        & (history["LON_NUM"] >= min_lon)
        & (history["LON_NUM"] <= max_lon)
    ]


def rank_station_rows(
    history: DataFrame,
    city_lat: float,
    city_lng: float,
    city_dem_m: float,
    *,
    start_year: int,
    end_year: int,
) -> DataFrame:
    """Return candidate station rows within range, ranked by (coverage, distance)."""
    dist_km = haversine_km(city_lat, city_lng, history["LAT_NUM"], history["LON_NUM"])
    elev_delta = (history["ELEV_NUM"] - city_dem_m).abs()
    within_range = (dist_km <= MAX_STATION_DISTANCE_KM) & (
        elev_delta <= MAX_ELEV_DELTA_M
    )
    candidates = history[within_range].copy()
    if candidates.empty:
        return candidates
    candidates["dist_km"] = dist_km[within_range]

    begin_year = pd.to_numeric(candidates["BEGIN"].str[:4], errors="coerce")
    end_year_col = pd.to_numeric(candidates["END"].str[:4], errors="coerce")
    full_coverage = (begin_year <= start_year) & (end_year_col >= end_year - 1)
    recent_only = end_year_col >= end_year - 1
    candidates["coverage_bucket"] = np.where(
        full_coverage, 0, np.where(recent_only, 1, 2)
    )

    return candidates.sort_values(["coverage_bucket", "dist_km"])


def candidate_ids_for_station(usaf: str, wban: str) -> list[str]:
    """Return the ordered id fallthrough for one physical station."""
    ids: list[str] = []
    if usaf != PLACEHOLDER_USAF and wban != PLACEHOLDER_WBAN:
        ids.append(usaf + wban)
    if usaf != PLACEHOLDER_USAF:
        ids.append(usaf + PLACEHOLDER_WBAN)
    if wban != PLACEHOLDER_WBAN:
        ids.append(PLACEHOLDER_USAF + wban)

    seen: set[str] = set()
    ordered: list[str] = []
    for station_id in ids:
        if station_id not in seen:
            seen.add(station_id)
            ordered.append(station_id)
    return ordered
