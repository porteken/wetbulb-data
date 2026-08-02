"""Validate committed North America catalog and GHCNh crosswalk invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cities_na import ERA5_LAND_GRID_DEG
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
    station_path = base / "cities_na_ghcnh_stations.csv"
    manifest_path = base / "cities_na.catalog.json"
    cities = pd.read_csv(catalog_path, dtype={"place_id": str})
    expected_ids = list(range(len(cities)))
    if cities["location_id"].tolist() != expected_ids:
        message = "cities_na.csv must contain contiguous ordered location IDs"
        raise ValueError(message)
    if cities["place_id"].nunique() != len(cities) or cities.isna().any().any():
        message = "cities_na.csv requires unique, fully resolved places"
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

    stations = pd.read_csv(station_path, dtype={"ghcn_id": str})
    if not set(stations["location_id"]).issubset(expected_ids):
        message = "GHCNh crosswalk contains an unknown location ID"
        raise ValueError(message)
    if stations[["location_id", "year", "candidate_rank"]].duplicated().any():
        message = "GHCNh crosswalk contains a duplicate candidate rank"
        raise ValueError(message)
    if stations.isna().any().any():
        message = "GHCNh crosswalk requires fully resolved candidates"
        raise ValueError(message)
    if (stations["dist_km"] > MAX_STATION_DISTANCE_KM).any():
        message = "GHCNh station exceeds 60 km"
        raise ValueError(message)
    if (stations["elevation_difference_m"].abs() > MAX_ELEV_DELTA_M).any():
        message = "GHCNh station exceeds 300 m elevation difference"
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
