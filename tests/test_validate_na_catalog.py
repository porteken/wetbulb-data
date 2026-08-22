# Copyright (C) 2026 Kenneth Porter

"""Tests for North America catalog invariant validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

import validate_na_catalog as catalog


def test_catalog_and_crosswalk_with_valid_constraints_pass(
    tmp_path: Path,
) -> None:
    """The validator accepts a catalog, manifest, and GHCNh crosswalk."""
    cities = pd.DataFrame(
        {
            "location_id": [0, 1],
            "place_id": ["first", "second"],
            "country": ["US", "CA"],
            "lat": [40.0, 40.2],
            "lng": [-74.0, -74.2],
            "dem_m": [10.0, 20.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0, 1],
            "ghcn_id": ["STATION1", "STATION2"],
            "year": [2025, 2025],
            "candidate_rank": [1, 1],
            "dist_km": [1.0, 2.0],
            "lon": [-74.0, -74.2],
            "elev_m": [10.0, 20.0],
            "elevation_difference_m": [0.0, 0.0],
        }
    )
    cities_path = tmp_path / "cities_na.csv"
    cities.to_csv(cities_path, index=False)
    stations.to_csv(tmp_path / "cities_na_ghcnh_stations.csv", index=False)
    (tmp_path / "cities_na.catalog.json").write_text(
        json.dumps(
            {"catalog_sha256": hashlib.sha256(cities_path.read_bytes()).hexdigest()}
        ),
        encoding="utf-8",
    )

    catalog.validate_catalog(tmp_path)


@pytest.mark.parametrize(
    ("centers", "error"),
    [
        (
            pd.DataFrame({"location_id": [2]}),
            "one ordered row per city",
        ),
        (
            pd.DataFrame(
                {
                    "location_id": [1],
                    "center_lat": [float("nan")],
                    "center_lng": [-74.0],
                }
            ),
            "fully resolved coordinates",
        ),
    ],
)
def test_city_center_validation_rejects_invalid_maps(
    tmp_path: Path,
    centers: pd.DataFrame,
    error: str,
) -> None:
    cities = pd.DataFrame({"location_id": [1], "lat": [40.0], "lng": [-74.0]})
    centers_path = tmp_path / "centers.csv"
    centers.to_csv(centers_path, index=False)

    with pytest.raises(ValueError, match=error):
        catalog.validate_city_centers(cities, centers_path)


def test_city_center_validation_allows_an_absent_optional_map(tmp_path: Path) -> None:
    cities = pd.DataFrame({"location_id": [1], "lat": [40.0], "lng": [-74.0]})

    catalog.validate_city_centers(cities, tmp_path / "does-not-exist.csv")


def test_city_center_validation_rejects_a_stale_map(tmp_path: Path) -> None:
    cities = pd.DataFrame({"location_id": [1], "lat": [40.0], "lng": [-74.0]})
    centers_path = tmp_path / "centers.csv"
    pd.DataFrame(
        {"location_id": [1], "center_lat": [41.0], "center_lng": [-74.0]}
    ).to_csv(centers_path, index=False)

    with pytest.raises(ValueError, match="center map is stale"):
        catalog.validate_city_centers(cities, centers_path)


def test_ghcnh_map_aligned_with_cities_passes() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [0, 1],
            "lat": [40.0, 40.2],
            "lng": [-74.0, -74.2],
            "dem_m": [10.0, 20.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0, 1],
            "candidate_rank": [1, 1],
            "lon": [-74.0, -74.2],
            "elev_m": [10.0, 20.0],
        }
    )

    catalog.validate_ghcnh_map_alignment(cities, stations)


def test_ghcnh_map_aligned_near_mapping_limit_passes() -> None:
    """An east-west offset near the 60 km mapping limit is not a desync."""
    lat = 40.0
    city_lng = -74.0
    lon_delta = 59.9 / (math.cos(math.radians(lat)) * catalog._KM_PER_DEGREE)
    cities = pd.DataFrame(
        {
            "location_id": [0],
            "lat": [lat],
            "lng": [city_lng],
            "dem_m": [10.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0],
            "candidate_rank": [1],
            "lon": [city_lng + lon_delta],
            "elev_m": [10.0],
        }
    )

    catalog.validate_ghcnh_map_alignment(cities, stations)


def test_ghcnh_map_shifted_location_ids_fail() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [0, 1],
            "lat": [40.0, 40.0],
            "lng": [-74.0, -82.0],
            "dem_m": [10.0, 10.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [1, 2],
            "candidate_rank": [1, 1],
            "lon": [-74.0, -82.0],
            "elev_m": [10.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="location_id desync"):
        catalog.validate_ghcnh_map_alignment(cities, stations)


def test_ghcnh_map_station_longitude_drift_fails() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [0],
            "lat": [40.0],
            "lng": [-74.0],
            "dem_m": [10.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0],
            "candidate_rank": [1],
            "lon": [-82.0],
            "elev_m": [10.0],
        }
    )

    with pytest.raises(ValueError, match="location_id desync"):
        catalog.validate_ghcnh_map_alignment(cities, stations)


def test_ghcnh_map_elevation_desync_fails() -> None:
    """A station far above the current city DEM is a location_id desync."""
    cities = pd.DataFrame(
        {
            "location_id": [0],
            "lat": [40.0],
            "lng": [-74.0],
            "dem_m": [10.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0],
            "candidate_rank": [1],
            "lon": [-74.0],
            "elev_m": [400.0],
        }
    )

    with pytest.raises(ValueError, match="location_id desync"):
        catalog.validate_ghcnh_map_alignment(cities, stations)


def test_ghcnh_map_missing_lon_fails() -> None:
    """A station map expected to carry longitude must fail closed without it."""
    cities = pd.DataFrame(
        {
            "location_id": [0],
            "lat": [40.0],
            "lng": [-74.0],
            "dem_m": [10.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0],
            "candidate_rank": [1],
            "elev_m": [10.0],
        }
    )

    with pytest.raises(ValueError, match="location_id desync"):
        catalog.validate_ghcnh_map_alignment(cities, stations)


def test_ghcnh_map_rejects_catalog_ids_the_map_never_saw() -> None:
    cities = pd.DataFrame(
        {
            "location_id": [0, 1, 2],
            "lat": [40.0, 40.1, 40.2],
            "lng": [-74.0, -74.1, -74.2],
            "dem_m": [10.0, 10.0, 10.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0, 1],
            "candidate_rank": [1, 1],
            "lon": [-74.0, -74.1],
            "elev_m": [10.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="location_id desync"):
        catalog.validate_ghcnh_map_alignment(cities, stations)


def test_ghcnh_map_files_validates_csv_paths(tmp_path: Path) -> None:
    """Path-based alignment loads the catalog and station map CSVs."""
    cities = pd.DataFrame(
        {
            "location_id": [0],
            "lat": [40.0],
            "lng": [-74.0],
            "dem_m": [10.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0],
            "ghcn_id": ["STATION1"],
            "candidate_rank": [1],
            "lon": [-82.0],
            "elev_m": [10.0],
        }
    )
    cities_csv = tmp_path / "cities.csv"
    station_map_csv = tmp_path / "stations.csv"
    cities.to_csv(cities_csv, index=False)
    stations.to_csv(station_map_csv, index=False)

    with pytest.raises(ValueError, match="location_id desync"):
        catalog.validate_ghcnh_map_files(cities_csv, station_map_csv)


def test_catalog_validation_rejects_eu_ghcnh_desync(tmp_path: Path) -> None:
    """The root validator also checks the Europe catalog against its GHCNh map."""
    cities = pd.DataFrame(
        {
            "location_id": [0, 1],
            "place_id": ["first", "second"],
            "country": ["US", "CA"],
            "lat": [40.0, 40.2],
            "lng": [-74.0, -74.2],
            "dem_m": [10.0, 20.0],
        }
    )
    stations = pd.DataFrame(
        {
            "location_id": [0, 1],
            "ghcn_id": ["STATION1", "STATION2"],
            "year": [2025, 2025],
            "candidate_rank": [1, 1],
            "dist_km": [1.0, 2.0],
            "lon": [-74.0, -74.2],
            "elev_m": [10.0, 20.0],
            "elevation_difference_m": [0.0, 0.0],
        }
    )
    cities_path = tmp_path / "cities_na.csv"
    cities.to_csv(cities_path, index=False)
    stations.to_csv(tmp_path / "cities_na_ghcnh_stations.csv", index=False)
    (tmp_path / "cities_na.catalog.json").write_text(
        json.dumps(
            {"catalog_sha256": hashlib.sha256(cities_path.read_bytes()).hexdigest()}
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "location_id": [0],
            "lat": [48.0],
            "lng": [2.0],
            "dem_m": [30.0],
        }
    ).to_csv(tmp_path / "cities_eu.csv", index=False)
    pd.DataFrame(
        {
            "location_id": [0],
            "ghcn_id": ["EUSTATION"],
            "lon": [10.0],
            "elev_m": [30.0],
        }
    ).to_csv(tmp_path / "cities_eu_ghcnh_stations.csv", index=False)

    with pytest.raises(ValueError, match="location_id desync"):
        catalog.validate_catalog(tmp_path)
