"""Validate committed North America catalog and ISD crosswalk invariants."""
# ruff: noqa: EM101, TRY003
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cities_na import ERA5_LAND_GRID_DEG, MAX_CITIES
from isd_history import MAX_ELEV_DELTA_M, MAX_STATION_DISTANCE_KM


def _grid_cell(value: pd.Series) -> pd.Series:
    return (value / ERA5_LAND_GRID_DEG).round().astype(int)


def validate_catalog(base: Path = Path()) -> None:
    """Validate catalog identity, completeness, and crosswalk constraints."""
    catalog_path = base / "cities_na.csv"
    station_path = base / "cities_na_isd_stations.csv"
    manifest_path = base / "cities_na.catalog.json"
    cities = pd.read_csv(catalog_path, dtype={"place_id": str})
    expected_ids = list(range(MAX_CITIES))
    if cities["location_id"].tolist() != expected_ids:
        raise ValueError("cities_na.csv must contain ordered location IDs 0..499")
    if cities["place_id"].nunique() != MAX_CITIES or cities.isna().any().any():
        raise ValueError("cities_na.csv requires 500 unique, fully resolved places")
    if set(cities["country"]) - {"US", "CA"}:
        raise ValueError("cities_na.csv contains an unsupported country")
    cells = pd.DataFrame(
        {"lat": _grid_cell(cities["lat"]), "lng": _grid_cell(cities["lng"])}
    )
    if cells.duplicated().any():
        raise ValueError("cities_na.csv contains duplicate ERA5-Land grid cells")

    manifest = json.loads(manifest_path.read_text())
    if (
        hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        != manifest["catalog_sha256"]
    ):
        raise ValueError("cities_na.csv does not match its catalog manifest")

    stations = pd.read_csv(station_path, dtype={"usaf": str, "wban": str})
    if stations["location_id"].tolist() != expected_ids:
        raise ValueError("ISD crosswalk must contain one ordered row per city")
    if stations[["usaf", "wban"]].duplicated().any():
        raise ValueError("ISD crosswalk assigns a station more than once")
    if stations.isna().any().any():
        raise ValueError("ISD crosswalk requires fully resolved stations")
    if (stations["dist_km"] > MAX_STATION_DISTANCE_KM).any():
        raise ValueError("ISD station exceeds 60 km")
    elevations = cities[["location_id", "dem_m"]].merge(
        stations[["location_id", "elev_m"]], on="location_id", validate="one_to_one"
    )
    if ((elevations["dem_m"] - elevations["elev_m"]).abs() > MAX_ELEV_DELTA_M).any():
        raise ValueError("ISD station exceeds 300 m elevation difference")


def main() -> None:
    """Run validation against the committed inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    validate_catalog(Path(args.root))


if __name__ == "__main__":
    main()
