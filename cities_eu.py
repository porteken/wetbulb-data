# Copyright (C) 2026 Kenneth Porter

"""Build the 500 largest eligible European cities for the wet-bulb pipeline.

GeoNames administrative features are excluded: only populated places (feature class
``P``) enter the population ranking.  Selection accepts a city only when it can claim
both an unused ERA5-Land grid point and an unused, verified ISD weather station.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, cast
from zoneinfo import ZoneInfo

import requests

import nldas
from cities_na import CITY_COORD_DECIMALS, ERA5_LAND_GRID_DEG
from isd_history import candidate_ids_for_station, prepare_history, rank_station_rows
from make_isd_station_map import _candidate_verified_at, fetch_isd_history

type DataFrame = Any
type StationVerifier = Callable[[list[str], int], bool]
pd: Any = cast("Any", import_module("pandas"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

GEONAMES_CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
GEONAMES_COUNTRY_INFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"

EU_LOCATION_ID_OFFSET = 1000
MAX_CITIES = 500
GRID_DEG = ERA5_LAND_GRID_DEG
EU_START_YEAR = 1991

_GEONAMES_COLUMNS = {
    1: "city",
    4: "lat",
    5: "lng",
    6: "feature_class",
    8: "country_code",
    14: "population",
    16: "dem_m",
    17: "timezone",
}

EU_COUNTRY_CODES = frozenset(
    {
        "AT",
        "BE",
        "BG",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "HU",
        "IE",
        "IT",
        "LT",
        "LU",
        "LV",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
        "GB",
        "CH",
    },
)

EUROPE_MIN_LAT = 34.0
EUROPE_MAX_LAT = 72.0
EUROPE_MIN_LNG = -25.0
EUROPE_MAX_LNG = 40.0

_STANDARD_OFFSET_REFERENCE = datetime(2025, 1, 15, 12, tzinfo=UTC)


def load_geonames_cities(url: str = GEONAMES_CITIES_URL) -> DataFrame:
    """Load and normalize the GeoNames `cities15000` dump."""
    df = pd.read_csv(
        url,
        sep="\t",
        header=None,
        compression="zip",
        quoting=csv.QUOTE_NONE,
        usecols=list(_GEONAMES_COLUMNS),
        names=None,
        dtype=str,
    )
    return df.rename(columns=_GEONAMES_COLUMNS)


def load_country_names(url: str = GEONAMES_COUNTRY_INFO_URL) -> dict[str, str]:
    """Return a mapping of ISO-3166 alpha-2 country code -> English country name."""
    df = pd.read_csv(url, sep="\t", header=None, comment="#", dtype=str, usecols=[0, 4])
    return dict(zip(df[0], df[4], strict=True))


def filter_europe(df: DataFrame) -> DataFrame:
    """Keep populated places in the EU-27, UK, and Switzerland, within the bbox."""
    df = df[df["feature_class"] == "P"]
    df = df[df["country_code"].isin(EU_COUNTRY_CODES)]
    lat = pd.to_numeric(df["lat"])
    lng = pd.to_numeric(df["lng"])
    return df[
        (lat >= EUROPE_MIN_LAT)
        & (lat <= EUROPE_MAX_LAT)
        & (lng >= EUROPE_MIN_LNG)
        & (lng <= EUROPE_MAX_LNG)
    ]


def standard_utc_offset_hours(tz_name: str) -> int:
    """Return a city's fixed standard (winter, non-DST) UTC offset in whole hours."""
    offset = _STANDARD_OFFSET_REFERENCE.astimezone(ZoneInfo(tz_name)).utcoffset()
    if offset is None:
        msg = f"{tz_name} has no UTC offset"
        raise ValueError(msg)
    hours = offset.total_seconds() / 3600.0
    if hours != int(hours):
        msg = f"{tz_name} standard UTC offset {hours} is not a whole hour"
        raise ValueError(msg)
    return int(hours)


def process_cities_eu(df: DataFrame, country_names: dict[str, str]) -> DataFrame:
    """Normalize the population-ranked city candidates.

    This helper retains grid-only selection for compatibility with callers that do not
    have an ISD inventory. The production path uses :func:`select_cities_eu`, which
    jointly claims grid cells and stations.
    """
    df = df.copy()
    df["population"] = pd.to_numeric(df["population"])
    df = df.sort_values("population", ascending=False)

    df["_snap_lat"] = (pd.to_numeric(df["lat"]) / GRID_DEG).round() * GRID_DEG
    df["_snap_lng"] = (pd.to_numeric(df["lng"]) / GRID_DEG).round() * GRID_DEG
    df = df.drop_duplicates(subset=["_snap_lat", "_snap_lng"], keep="first")
    df = df.drop(columns=["_snap_lat", "_snap_lng"])

    df = df.head(MAX_CITIES).reset_index(drop=True)
    df["location_id"] = EU_LOCATION_ID_OFFSET + df.index
    df["lat"] = pd.to_numeric(df["lat"]).round(CITY_COORD_DECIMALS)
    df["lng"] = pd.to_numeric(df["lng"]).round(CITY_COORD_DECIMALS)
    df["dem_m"] = pd.to_numeric(df["dem_m"])
    df["state"] = df["country_code"].map(country_names).fillna(df["country_code"])
    df["utc_offset_hours"] = df["timezone"].apply(standard_utc_offset_hours)

    return df[
        [
            "location_id",
            "city",
            "state",
            "lat",
            "lng",
            "dem_m",
            "timezone",
            "utc_offset_hours",
        ]
    ]


def prepare_eu_history(history: DataFrame) -> DataFrame:
    """Numeric-parse and pre-filter the ISD inventory to a loose Europe bbox."""
    return prepare_history(
        history,
        min_lat=30.0,
        max_lat=75.0,
        min_lon=-30.0,
        max_lon=65.0,
    )


def _grid_cell(lat: float, lng: float) -> tuple[int, int]:
    """Return the ERA5-Land grid point claimed by a coordinate."""
    return (round(lat / GRID_DEG), round(lng / GRID_DEG))


def _select_station(
    station_candidates: DataFrame,
    claimed_stations: set[tuple[str, str]],
    *,
    verify: StationVerifier,
    start_year: int,
    end_year: int,
) -> tuple[Any, tuple[str, str], list[str], bool] | None:
    """Return the first unclaimed station with data for the required end year."""
    for station in station_candidates.itertuples():
        station_key = (str(station.USAF), str(station.WBAN))
        if station_key in claimed_stations:
            continue
        candidate_ids = candidate_ids_for_station(*station_key)
        if not candidate_ids or not verify(candidate_ids, end_year):
            continue
        return (
            station,
            station_key,
            candidate_ids,
            bool(verify(candidate_ids, start_year)),
        )
    return None


def select_cities_eu(
    candidates: DataFrame,
    country_names: dict[str, str],
    history: DataFrame,
    *,
    verify: StationVerifier,
    start_year: int = EU_START_YEAR,
    end_year: int = nldas.NLDAS_END_YEAR,
    max_cities: int = MAX_CITIES,
) -> tuple[DataFrame, DataFrame]:
    """Select cities with unique ERA5-Land grid points and unique ISD stations."""
    ranked_cities = candidates.copy()
    ranked_cities["population"] = pd.to_numeric(ranked_cities["population"])
    ranked_cities = ranked_cities.sort_values(
        ["population", "city"], ascending=[False, True], kind="stable"
    )

    claimed_cells: set[tuple[int, int]] = set()
    claimed_stations: set[tuple[str, str]] = set()
    city_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []

    for city in ranked_cities.itertuples():
        if len(city_rows) >= max_cities:
            break
        cell = _grid_cell(float(city.lat), float(city.lng))
        if cell in claimed_cells:
            continue

        station_candidates = rank_station_rows(
            history,
            float(city.lat),
            float(city.lng),
            float(city.dem_m),
            start_year=start_year,
            end_year=end_year,
        )
        selected = _select_station(
            station_candidates,
            claimed_stations,
            verify=verify,
            start_year=start_year,
            end_year=end_year,
        )
        if selected is None:
            continue

        station, station_key, candidate_ids, begin_verified = selected
        location_id = EU_LOCATION_ID_OFFSET + len(city_rows)
        offset_hours = standard_utc_offset_hours(str(city.timezone))
        claimed_cells.add(cell)
        claimed_stations.add(station_key)
        city_rows.append(
            {
                "location_id": location_id,
                "city": str(city.city),
                "state": country_names.get(
                    str(city.country_code), str(city.country_code)
                ),
                "lat": round(float(city.lat), CITY_COORD_DECIMALS),
                "lng": round(float(city.lng), CITY_COORD_DECIMALS),
                "dem_m": float(city.dem_m),
                "timezone": str(city.timezone),
                "utc_offset_hours": offset_hours,
            }
        )
        station_rows.append(
            {
                "location_id": location_id,
                "usaf": station.USAF,
                "wban": station.WBAN,
                "isd_ids": "|".join(candidate_ids),
                "lon": float(station.LON_NUM),
                "dist_km": round(float(station.dist_km), 2),
                "elev_m": float(station.ELEV_NUM),
                "utc_offset_hours": offset_hours,
                "begin_verified": begin_verified,
            }
        )

    if len(city_rows) < max_cities:
        message = (
            f"only {len(city_rows)} of {max_cities} cities could claim a unique "
            "ERA5-Land grid point and ISD station"
        )
        raise ValueError(message)
    return pd.DataFrame(city_rows), pd.DataFrame(station_rows)


def main() -> None:
    """Orchestrate EU city processing."""
    logger.info("Generating EU cities dataset...")
    candidates = filter_europe(load_geonames_cities())
    country_names = load_country_names()
    session = requests.Session()
    history = prepare_eu_history(fetch_isd_history(session=session))
    verified_cache: dict[tuple[str, int], bool] = {}

    def verify(candidate_ids: list[str], year: int) -> bool:
        return _candidate_verified_at(
            candidate_ids, year, session=session, cache=verified_cache
        )

    cities, stations = select_cities_eu(
        candidates, country_names, history, verify=verify
    )

    cities.to_csv(
        "cities_eu.csv", index=False, float_format=f"%.{CITY_COORD_DECIMALS}f"
    )
    stations.to_csv("cities_eu_isd_stations.csv", index=False)
    logger.info(
        "Saved %d cities and unique stations to cities_eu.csv and "
        "cities_eu_isd_stations.csv",
        len(cities),
    )


if __name__ == "__main__":
    main()
