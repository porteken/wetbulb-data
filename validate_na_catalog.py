# Copyright (C) 2026 Kenneth Porter

"""Validate committed North America catalog and GHCNh crosswalk invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from cities_na import ERA5_LAND_GRID_DEG
from isd_history import MAX_ELEV_DELTA_M, MAX_STATION_DISTANCE_KM, haversine_km
from make_city_center_map_na import MAX_CENTER_OFFSET_KM
from make_ghcnh_station_map import MAX_ELEVATION_DIFFERENCE_M

_KM_PER_DEGREE = 111.32
_GHCNH_DESYNC_LON_KM = 150.0
_GHCNH_DESYNC_MESSAGE = (
    "GHCNh station map location_id desync; regenerate via "
    "make_ghcnh_station_map.py --year-specific"
)


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


def validate_ghcnh_map_alignment(cities: pd.DataFrame, stations: pd.DataFrame) -> None:
    """Fail when current city coordinates no longer match the GHCNh station map."""
    catalog_max = int(cities["location_id"].max())
    map_max = int(stations["location_id"].max()) if not stations.empty else -1
    if catalog_max > map_max:
        raise ValueError(_GHCNH_DESYNC_MESSAGE)

    if "lon" not in stations.columns:
        raise ValueError(_GHCNH_DESYNC_MESSAGE)

    mapped = stations.merge(
        cities[["location_id", "lat", "lng", "dem_m"]],
        on="location_id",
        how="inner",
    )
    if mapped.empty:
        return

    lon_km = (
        mapped["lat"].map(math.radians).map(math.cos).abs()
        * _KM_PER_DEGREE
        * (mapped["lng"] - mapped["lon"]).abs()
    )
    if (lon_km > _GHCNH_DESYNC_LON_KM).any():
        raise ValueError(_GHCNH_DESYNC_MESSAGE)

    if "elev_m" in mapped.columns:
        elev_delta = (mapped["dem_m"] - mapped["elev_m"]).abs()
        if (elev_delta > MAX_ELEVATION_DIFFERENCE_M).any():
            raise ValueError(_GHCNH_DESYNC_MESSAGE)


def validate_ghcnh_map_files(cities_csv: Path, station_map_csv: Path) -> None:
    """Fail when a city catalog and GHCNh station map are out of alignment."""
    cities = pd.read_csv(cities_csv)
    stations = pd.read_csv(station_map_csv, dtype={"ghcn_id": str})
    validate_ghcnh_map_alignment(cities, stations)


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

    validate_ghcnh_map_files(catalog_path, station_path)
    eu_cities_path = base / "cities_eu.csv"
    eu_station_path = base / "cities_eu_ghcnh_stations.csv"
    if eu_cities_path.exists() and eu_station_path.exists():
        validate_ghcnh_map_files(eu_cities_path, eu_station_path)
    validate_city_centers(cities, base / "cities_na_centers.csv")


def main() -> None:
    """Run validation against the committed inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--cities", type=Path)
    parser.add_argument("--station-map", type=Path)
    args = parser.parse_args()
    if args.cities is not None or args.station_map is not None:
        if args.cities is None or args.station_map is None:
            parser.error("--cities and --station-map are required together")
        validate_ghcnh_map_files(args.cities, args.station_map)
        return
    validate_catalog(Path(args.root))


if __name__ == "__main__":
    main()
