from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import cities_na


def test_clean_place_name_removes_legal_suffix_and_uses_override() -> None:
    assert cities_na.clean_place_name("Springfield city", "0000000") == "Springfield"
    assert cities_na.clean_place_name("Macon-Bibb County", "1349008") == "Macon"


def test_clean_place_name_strips_only_the_trailing_legal_suffix() -> None:
    assert cities_na.clean_place_name("Kansas City city", "0000000") == "Kansas City"
    assert cities_na.clean_place_name("Jersey City city", "0000000") == "Jersey City"
    assert cities_na.clean_place_name("Carson City city", "0000000") == "Carson City"


def test_load_census_places_keeps_incorporated_conus_places() -> None:
    payload = (
        b"SUMLEV,STATE,PLACE,NAME,POPESTIMATE2025\n"
        b"162,06,44000,Example city,1000\n"
        b"160,06,99999,Example CDP,2000\n"
        b"162,02,03000,Alaska city,3000\n"
    )
    result = cities_na.load_census_places(payload)
    assert result[["census_geoid", "city", "state", "population"]].to_dict(
        "records"
    ) == [
        {
            "census_geoid": "0644000",
            "city": "Example",
            "state": "CA",
            "population": 1000,
        }
    ]


def test_load_census_places_rejects_unknown_state() -> None:
    payload = (
        b"SUMLEV,STATE,PLACE,NAME,POPESTIMATE2025\n162,99,44000,Example city,1000\n"
    )
    with pytest.raises(ValueError, match="unknown state"):
        cities_na.load_census_places(payload)


def test_build_ca_places_filters_nonmunicipal_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = pd.DataFrame(
        {
            "CSDuid": ["1", "2"],
            "CSDname": ["Toronto", "Unorganized"],
            "CSDtype": ["C", "NO"],
            "CSDpop_2021": ["100000", "60000"],
            "PRuid": ["35", "35"],
            "CSDrplat": ["43.6", "44.0"],
            "CSDrplong": ["-79.4", "-80.0"],
        }
    )
    with caplog.at_level("WARNING"):
        result = cities_na.build_ca_places(source)
    assert result["place_id"].tolist() == ["CA1"]
    assert result["country"].tolist() == ["CA"]
    assert "non-municipal" in caplog.text


def test_build_ca_places_requires_schema() -> None:
    source = pd.DataFrame({"CSDuid": []})
    with pytest.raises(ValueError, match="missing column"):
        cities_na.build_ca_places(source)


def test_attach_terrain_attributes_uses_nearest_place_and_drops_distant() -> None:
    places = pd.DataFrame(
        {
            "place_id": ["US1", "US2"],
            "city": ["Near", "Far"],
            "state": ["NY", "NY"],
            "country": ["US", "US"],
            "population": [2, 1],
            "lat": [40.0, 10.0],
            "lng": [-74.0, 10.0],
        }
    )
    reference = pd.DataFrame(
        {
            "lat": [40.01],
            "lng": [-74.01],
            "dem_m": [12],
            "timezone": ["America/New_York"],
        }
    )
    result = cities_na.attach_terrain_attributes(places, reference)
    assert result["city"].tolist() == ["Near"]
    assert result.iloc[0]["dem_m"] == 12


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "USAF": ["000001", "000002"],
            "WBAN": ["00001", "00002"],
            "BEGIN": ["19900101", "19900101"],
            "END": ["20251231", "20251231"],
            "LAT_NUM": [40.0, 41.0],
            "LON_NUM": [-74.0, -75.0],
            "ELEV_NUM": [10.0, 20.0],
        }
    )


def test_select_cities_claims_unique_cells_and_stations() -> None:
    candidates = pd.DataFrame(
        {
            "place_id": ["US1", "US2", "US3"],
            "city": ["One", "Same cell", "Two"],
            "state": ["NY", "NY", "PA"],
            "country": ["US", "US", "US"],
            "population": [30, 20, 10],
            "lat": [40.0, 40.01, 41.0],
            "lng": [-74.0, -74.01, -75.0],
            "dem_m": [10, 10, 20],
            "timezone": ["America/New_York"] * 3,
        }
    )
    calls: list[tuple[tuple[str, ...], int]] = []

    def verify(ids: list[str], year: int) -> bool:
        calls.append((tuple(ids), year))
        return True

    catalog, stations = cities_na.select_cities(
        candidates, _history(), verify=verify, max_cities=2
    )
    assert catalog["city"].tolist() == ["One", "Two"]
    assert stations[["usaf", "wban"]].duplicated().sum() == 0
    assert {year for _, year in calls} == {2000, 2025}


def test_select_cities_fails_when_constraints_cannot_be_filled() -> None:
    candidate = pd.DataFrame(
        {
            "place_id": ["US1"],
            "city": ["One"],
            "state": ["NY"],
            "country": ["US"],
            "population": [1],
            "lat": [40.0],
            "lng": [-74.0],
            "dem_m": [10],
            "timezone": ["America/New_York"],
        }
    )
    history = _history()

    def verify(_ids: list[str], _year: int) -> bool:
        return False

    with pytest.raises(ValueError, match="only 0 of 1"):
        cities_na.select_cities(candidate, history, verify=verify, max_cities=1)


def test_write_catalog_writes_matching_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cities_na, "MAX_CITIES", 2)
    catalog = pd.DataFrame(
        [
            [0, "US1", "One", "NY", "US", 2, 40.0, -74.0, 10, "America/New_York", -5],
            [1, "CA1", "Two", "ON", "CA", 1, 41.0, -75.0, 20, "America/Toronto", -5],
        ],
        columns=cities_na.CATALOG_COLUMNS,
    )
    stations = pd.DataFrame(
        [
            [0, "000001", "00001", "00000100001", -74.0, 0.0, 10.0, -5, True],
            [1, "000002", "00002", "00000200002", -75.0, 0.0, 20.0, -5, True],
        ],
        columns=cities_na.STATION_COLUMNS,
    )
    digest = cities_na.write_catalog(
        catalog,
        stations,
        out="cities_na.csv",
        station_out="cities_na_isd_stations.csv",
        manifest_out="cities_na.catalog.json",
        source_urls={"source": "https://example.test"},
    )
    manifest = json.loads((tmp_path / "cities_na.catalog.json").read_text())
    assert (
        digest == hashlib.sha256((tmp_path / "cities_na.csv").read_bytes()).hexdigest()
    )
    assert manifest["catalog_sha256"] == digest
