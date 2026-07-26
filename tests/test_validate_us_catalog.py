from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import validate_us_catalog

CITY_COUNT = 500


def _write_catalog(base: Path, *, cities: pd.DataFrame | None = None) -> None:
    frame = (
        cities
        if cities is not None
        else pd.DataFrame(
            {
                "location_id": range(CITY_COUNT),
                "census_geoid": [f"{index:07d}" for index in range(CITY_COUNT)],
                "city": [f"City{index}" for index in range(CITY_COUNT)],
            }
        )
    )
    cities_path = base / "cities.csv"
    frame.to_csv(cities_path, index=False)
    (base / "cities.catalog.json").write_text(
        json.dumps(
            {"catalog_sha256": hashlib.sha256(cities_path.read_bytes()).hexdigest()}
        )
    )


def _write_cells(base: Path, **overrides: object) -> None:
    frame = pd.DataFrame(
        {
            "location_id": range(CITY_COUNT),
            "cell_lat": [40.0] * CITY_COUNT,
            "cell_lon": [-90.0] * CITY_COUNT,
        }
    )
    for column, value in overrides.items():
        frame.loc[0, column] = value
    frame.to_csv(base / "cities_nldas_cells.csv", index=False)


def _write_stations(base: Path, **overrides: object) -> None:
    frame = pd.DataFrame(
        {
            "location_id": range(CITY_COUNT),
            "isd_ids": ["720000-99999"] * CITY_COUNT,
            "dist_km": [10.0] * CITY_COUNT,
            "elev_diff_m": [20.0] * CITY_COUNT,
        }
    )
    for column, value in overrides.items():
        frame.loc[0, column] = value
    frame.to_csv(base / "cities_isd_stations.csv", index=False)


@pytest.fixture
def valid_base(tmp_path: Path) -> Path:
    _write_catalog(tmp_path)
    _write_cells(tmp_path)
    _write_stations(tmp_path)
    return tmp_path


def test_accepts_a_consistent_catalog(valid_base: Path) -> None:
    validate_us_catalog.validate_catalog(valid_base)


def test_rejects_unordered_location_ids(valid_base: Path) -> None:
    cities = pd.read_csv(valid_base / "cities.csv", dtype={"census_geoid": str})
    cities.loc[0, "location_id"] = 999
    _write_catalog(valid_base, cities=cities)

    with pytest.raises(ValueError, match="ordered location IDs"):
        validate_us_catalog.validate_catalog(valid_base)


def test_rejects_duplicate_geoids(valid_base: Path) -> None:
    cities = pd.read_csv(valid_base / "cities.csv", dtype={"census_geoid": str})
    cities.loc[1, "census_geoid"] = cities.loc[0, "census_geoid"]
    _write_catalog(valid_base, cities=cities)

    with pytest.raises(ValueError, match="unique, fully resolved GEOIDs"):
        validate_us_catalog.validate_catalog(valid_base)


def test_rejects_manifest_hash_mismatch(valid_base: Path) -> None:
    (valid_base / "cities.catalog.json").write_text(
        json.dumps({"catalog_sha256": "0" * 64})
    )

    with pytest.raises(ValueError, match="does not match its catalog manifest"):
        validate_us_catalog.validate_catalog(valid_base)


def test_rejects_incomplete_cell_map(valid_base: Path) -> None:
    cells = pd.read_csv(valid_base / "cities_nldas_cells.csv")
    cells = cells.iloc[:-1]
    cells.to_csv(valid_base / "cities_nldas_cells.csv", index=False)

    with pytest.raises(ValueError, match="one row per city"):
        validate_us_catalog.validate_catalog(valid_base)


def test_rejects_missing_cell_coordinates(valid_base: Path) -> None:
    _write_cells(valid_base, cell_lat=None)

    with pytest.raises(ValueError, match="valid NLDAS land cell"):
        validate_us_catalog.validate_catalog(valid_base)


def test_rejects_incomplete_station_crosswalk(valid_base: Path) -> None:
    stations = pd.read_csv(valid_base / "cities_isd_stations.csv")
    stations = stations.iloc[:-1]
    stations.to_csv(valid_base / "cities_isd_stations.csv", index=False)

    with pytest.raises(ValueError, match="ISD crosswalk must contain one row"):
        validate_us_catalog.validate_catalog(valid_base)


def test_rejects_distant_station(valid_base: Path) -> None:
    _write_stations(valid_base, dist_km=61.0)

    with pytest.raises(ValueError, match="exceeds 60 km"):
        validate_us_catalog.validate_catalog(valid_base)


def test_rejects_large_elevation_difference(valid_base: Path) -> None:
    _write_stations(valid_base, elev_diff_m=-301.0)

    with pytest.raises(ValueError, match="300 m elevation difference"):
        validate_us_catalog.validate_catalog(valid_base)


def test_ignores_unmatched_stations(valid_base: Path) -> None:
    _write_stations(valid_base, isd_ids=None, dist_km=999.0, elev_diff_m=9999.0)

    validate_us_catalog.validate_catalog(valid_base)


def test_main_validates_the_committed_inputs(
    monkeypatch: pytest.MonkeyPatch, valid_base: Path
) -> None:
    seen: list[Path] = []
    monkeypatch.setattr(validate_us_catalog, "validate_catalog", seen.append)
    monkeypatch.setattr(sys, "argv", ["validate_us_catalog.py", "--root", "us/x"])

    validate_us_catalog.main()

    assert seen == [Path(validate_us_catalog.__file__).parent]
