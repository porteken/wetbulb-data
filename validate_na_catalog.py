"""Validate committed North America catalog and ISD crosswalk invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cities_na import ERA5_LAND_GRID_DEG, MAX_CITIES
from isd_history import MAX_ELEV_DELTA_M, MAX_STATION_DISTANCE_KM, haversine_km
from make_city_center_map_na import MAX_CENTER_OFFSET_KM


def _grid_cell(value: pd.Series) -> pd.Series:
    return (value / ERA5_LAND_GRID_DEG).round().astype(int)


def validate_city_centers(cities: pd.DataFrame, centers_path: Path) -> None:
    """Check the display-coordinate map still lines up with the catalog it was built from."""
    if not centers_path.exists():
        return
    centers = pd.read_csv(centers_path)
    if centers["location_id"].tolist() != cities["location_id"].tolist():
        message = "center map must contain one ordered row per city"
        raise ValueError(message)
    if centers.isna().any().any():
        message = "center map requires fully resolved coordinates"
        raise ValueError(message)
    drift = haversine_km(
        cities["lat"].to_numpy(),
        cities["lng"].to_numpy(),
        centers["center_lat"].to_numpy(),
        centers["center_lng"].to_numpy(),
    )
    if (drift > MAX_CENTER_OFFSET_KM).any():
        message = "center map is stale: rerun make_city_center_map_na.py"
        raise ValueError(message)


def validate_catalog(base: Path = Path()) -> None:
    """Validate catalog identity, completeness, and crosswalk constraints."""
    catalog_path = base / "cities_na.csv"
    station_path = base / "cities_na_isd_stations.csv"
    manifest_path = base / "cities_na.catalog.json"
    cities = pd.read_csv(catalog_path, dtype={"place_id": str})
    expected_ids = list(range(MAX_CITIES))
    if cities["location_id"].tolist() != expected_ids:
        message = "cities_na.csv must contain ordered location IDs 0..499"
        raise ValueError(message)
    if cities["place_id"].nunique() != MAX_CITIES or cities.isna().any().any():
        message = "cities_na.csv requires 500 unique, fully resolved places"
        raise ValueError(message)
    if set(cities["country"]) - {"US", "CA"}:
        message = "cities_na.csv contains an unsupported country"
        raise ValueError(message)
    cells = pd.DataFrame(
        {"lat": _grid_cell(cities["lat"]), "lng": _grid_cell(cities["lng"])}
    )
    if cells.duplicated().any():
        message = "cities_na.csv contains duplicate ERA5-Land grid cells"
        raise ValueError(message)

    manifest = json.loads(manifest_path.read_text())
    if (
        hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        != manifest["catalog_sha256"]
    ):
        message = "cities_na.csv does not match its catalog manifest"
        raise ValueError(message)

    stations = pd.read_csv(station_path, dtype={"usaf": str, "wban": str})
    if stations["location_id"].tolist() != expected_ids:
        message = "ISD crosswalk must contain one ordered row per city"
        raise ValueError(message)
    if stations[["usaf", "wban"]].duplicated().any():
        message = "ISD crosswalk assigns a station more than once"
        raise ValueError(message)
    if stations.isna().any().any():
        message = "ISD crosswalk requires fully resolved stations"
        raise ValueError(message)
    if (stations["dist_km"] > MAX_STATION_DISTANCE_KM).any():
        message = "ISD station exceeds 60 km"
        raise ValueError(message)
    elevations = cities[["location_id", "dem_m"]].merge(
        stations[["location_id", "elev_m"]], on="location_id", validate="one_to_one"
    )
    if ((elevations["dem_m"] - elevations["elev_m"]).abs() > MAX_ELEV_DELTA_M).any():
        message = "ISD station exceeds 300 m elevation difference"
        raise ValueError(message)

    validate_city_centers(cities, base / "cities_na_centers.csv")


def main() -> None:
    """Run validation against the committed inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    validate_catalog(Path(args.root))


if __name__ == "__main__":
    main()
