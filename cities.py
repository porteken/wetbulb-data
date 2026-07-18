"""Prepare city data for the application."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, cast

GRID_DEG: float = 0.25

type DataFrame = Any
np: Any = cast("Any", import_module("numpy"))
pd: Any = cast("Any", import_module("pandas"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
CITY_COORD_DECIMALS = 3

CITIES_SOURCE_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "6601958c1afba7bc40bd1f0497c4fc69931878f0/us-cities-top-1k.csv"
)


def load_data(url: str) -> DataFrame:
    """Load and normalize data from Plotly."""
    df = pd.read_csv(url)
    rename_map = {
        "City": "city",
        "name": "city",
        "State": "state",
        "Population": "population",
        "pop": "population",
        "lon": "lng",
    }
    df = df.rename(columns=rename_map)
    df.columns = df.columns.str.lower()
    return df


def filter_bounding_box(df: DataFrame) -> DataFrame:
    """Filter cities within the continental US."""
    min_lat, max_lat = 24.25, 49.25
    min_lng, max_lng = -124.5, -66.5
    return df[
        (df["lat"] >= min_lat)
        & (df["lat"] <= max_lat)
        & (df["lng"] >= min_lng)
        & (df["lng"] <= max_lng)
    ]


MAX_CITIES = 500


def process_cities(df: DataFrame) -> DataFrame:
    """Deduplicate by ERA5 grid cell, cap at 500 cities, and assign location IDs."""
    df = df.copy()

    df = df.sort_values("population", ascending=False)

    df["_snap_lat"] = (pd.to_numeric(df["lat"]) / GRID_DEG).round() * GRID_DEG
    df["_snap_lng"] = (pd.to_numeric(df["lng"]) / GRID_DEG).round() * GRID_DEG
    df = df.drop_duplicates(subset=["_snap_lat", "_snap_lng"], keep="first")
    df = df.drop(columns=["_snap_lat", "_snap_lng"])

    df = df.head(MAX_CITIES).reset_index(drop=True).reset_index(names="location_id")
    df["lat"] = pd.to_numeric(df["lat"]).round(CITY_COORD_DECIMALS)
    df["lng"] = pd.to_numeric(df["lng"]).round(CITY_COORD_DECIMALS)

    return df[["location_id", "city", "state", "lat", "lng"]]


def main() -> None:
    """Orchestrate city processing."""
    logger.info("Generating locations dataset...")
    df = load_data(CITIES_SOURCE_URL)
    df = filter_bounding_box(df)
    df = process_cities(df)

    df.to_csv("cities.csv", index=False, float_format=f"%.{CITY_COORD_DECIMALS}f")
    logger.info("Successfully saved %d cities to cities.csv", len(df))


if __name__ == "__main__":
    main()
