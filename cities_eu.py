"""Prepare EU city data for the wetbulb pipeline (mirrors `cities.py` for the US)."""

from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, cast
from zoneinfo import ZoneInfo

from cities import CITY_COORD_DECIMALS, GRID_DEG

type DataFrame = Any
pd: Any = cast("Any", import_module("pandas"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

GEONAMES_CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
GEONAMES_COUNTRY_INFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"

EU_LOCATION_ID_OFFSET = 1000
MAX_CITIES = 500

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
        # EU-27
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
    """Deduplicate by grid cell, cap at 500 cities, and assign location IDs."""
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


def main() -> None:
    """Orchestrate EU city processing."""
    logger.info("Generating EU cities dataset...")
    df = filter_europe(load_geonames_cities())
    country_names = load_country_names()
    df = process_cities_eu(df, country_names)

    df.to_csv("cities_eu.csv", index=False, float_format=f"%.{CITY_COORD_DECIMALS}f")
    logger.info("Successfully saved %d cities to cities_eu.csv", len(df))


if __name__ == "__main__":
    main()
