"""Validate committed U.S. catalog/crosswalk invariants before loading."""
# ruff: noqa: EM101, PLR2004, TRY003
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

MAX_STATION_DISTANCE_KM = 60
MAX_STATION_ELEVATION_DIFFERENCE_M = 300


def validate_catalog(base: Path = Path()) -> None:
    """Validate catalog identity, completeness, and crosswalk constraints."""
    cities = pd.read_csv(base / "cities.csv", dtype={"census_geoid": str})
    expected_ids = list(range(500))
    if cities["location_id"].tolist() != expected_ids:
        raise ValueError("cities.csv must contain ordered location IDs 0..499")
    if cities["census_geoid"].nunique() != 500 or cities.isna().any().any():
        raise ValueError("cities.csv requires 500 unique, fully resolved GEOIDs")
    manifest = json.loads((base / "cities.catalog.json").read_text())
    if (
        hashlib.sha256((base / "cities.csv").read_bytes()).hexdigest()
        != manifest["catalog_sha256"]
    ):
        raise ValueError("cities.csv does not match its catalog manifest")
    cells = pd.read_csv(base / "cities_nldas_cells.csv")
    if set(cells["location_id"]) != set(expected_ids):
        raise ValueError("NLDAS cell map must contain one row per city")
    if cells[["cell_lat", "cell_lon"]].isna().any().any():
        raise ValueError("every city requires a valid NLDAS land cell")
    stations = pd.read_csv(base / "cities_isd_stations.csv")
    if set(stations["location_id"]) != set(expected_ids):
        raise ValueError("ISD crosswalk must contain one row per city")
    matched = stations[stations["isd_ids"].notna()]
    if (matched["dist_km"] > MAX_STATION_DISTANCE_KM).any():
        raise ValueError("ISD station exceeds 60 km")
    elevation_column = (
        "elev_diff_m" if "elev_diff_m" in matched else "elevation_difference_m"
    )
    if (
        elevation_column in matched
        and (matched[elevation_column].abs() > MAX_STATION_ELEVATION_DIFFERENCE_M).any()
    ):
        raise ValueError("ISD station exceeds 300 m elevation difference")


def main() -> None:
    """Run validation against the committed inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.parse_args()
    # Catalog inputs remain committed at repository root; --root identifies the
    # output whose hash has already been checked by pull_wetbulb.sh.
    validate_catalog(Path(__file__).parent)


if __name__ == "__main__":
    main()
