"""Tests for the North America city-center display coordinate mapper."""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING

import pandas as pd

import make_city_center_map_na as centermap

if TYPE_CHECKING:
    import pathlib

    import pytest

_GEONAMES_ROWS = [
    (
        "5128581",
        "New York City",
        "New York City",
        "New York,NYC,Nueva York",
        "40.7143",
        "-74.0060",
        "P",
        "US",
        "NY",
        "8804190",
    ),
    (
        "5110302",
        "Brooklyn",
        "Brooklyn",
        "Bruklin",
        "40.6501",
        "-73.9496",
        "P",
        "US",
        "NY",
        "2300664",
    ),
    (
        "6167865",
        "Toronto",
        "Toronto",
        "Toronto,Torontas",
        "43.7064",
        "-79.3986",
        "P",
        "CA",
        "08",
        "2600000",
    ),
    (
        "6077243",
        "Montréal",
        "Montreal",
        "Montreal,Ville-Marie",
        "45.5088",
        "-73.5878",
        "P",
        "CA",
        "10",
        "1600000",
    ),
    (
        "4407066",
        "St. Louis",
        "St. Louis",
        "Saint Louis",
        "38.6270",
        "-90.1994",
        "P",
        "US",
        "MO",
        "300000",
    ),
    (
        "0000001",
        "Duplicate Springs",
        "Duplicate Springs",
        "",
        "35.0000",
        "-95.0000",
        "A",
        "US",
        "TX",
        "500000",
    ),
]


def _geonames_zip() -> bytes:
    lines = []
    for row in _GEONAMES_ROWS:
        fields = [""] * 19
        (
            fields[0],
            fields[1],
            fields[2],
            fields[3],
            fields[4],
            fields[5],
            fields[6],
            fields[8],
            fields[10],
            fields[14],
        ) = row
        lines.append("\t".join(fields))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("cities15000.txt", "\n".join(lines) + "\n")
    return buffer.getvalue()


def _places() -> centermap.DataFrame:
    return centermap.load_geonames_populated_places(_geonames_zip())


def test_normalize_place_name_folds_accents_punctuation_and_saint() -> None:
    assert centermap.normalize_place_name("Montréal") == "montreal"
    assert centermap.normalize_place_name("St. Louis") == "saint louis"
    assert centermap.normalize_place_name("Chatham-Kent") == "chatham kent"


def test_name_variants_splits_bilingual_names() -> None:
    assert "greater sudbury" in centermap.name_variants(
        "Greater Sudbury / Grand Sudbury"
    )


def test_geonames_admin1_code_maps_provinces_but_not_states() -> None:
    assert centermap.geonames_admin1_code("CA", "ON") == "08"
    assert centermap.geonames_admin1_code("US", "NY") == "NY"


def test_load_geonames_populated_places_drops_non_populated_features() -> None:
    assert "Duplicate Springs" not in set(_places()["name"])


def test_resolve_city_center_prefers_the_named_place_over_a_closer_one() -> None:
    catalog_point = pd.Series(
        {
            "place_id": "US3651000",
            "city": "New York",
            "state": "NY",
            "country": "US",
            "lat": 40.6627,
            "lng": -73.9387,
        }
    )

    match = centermap.resolve_city_center(_places(), catalog_point)

    assert match is not None
    place, offset_km = match
    assert place["name"] == "New York City"
    assert offset_km > 0


def test_resolve_city_center_matches_accented_and_provincial_names() -> None:
    catalog_point = pd.Series(
        {
            "place_id": "CA2466023",
            "city": "Montréal",
            "state": "QC",
            "country": "CA",
            "lat": 45.5076,
            "lng": -73.7231,
        }
    )

    match = centermap.resolve_city_center(_places(), catalog_point)

    assert match is not None
    assert match[0]["name"] == "Montréal"


def test_resolve_city_center_rejects_a_match_beyond_the_offset_cap() -> None:
    catalog_point = pd.Series(
        {
            "place_id": "CA3520005",
            "city": "Toronto",
            "state": "ON",
            "country": "CA",
            "lat": 49.0,
            "lng": -100.0,
        }
    )

    assert centermap.resolve_city_center(_places(), catalog_point) is None


def test_build_center_map_falls_back_to_catalog_points(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(
        centermap,
        "_download",
        lambda _url, *, session: _geonames_zip(),  # noqa: ARG005
    )
    cities_csv = tmp_path / "cities_na.csv"
    cities_csv.write_text(
        "location_id,place_id,city,state,country,lat,lng\n"
        "0,US3651000,New York,NY,US,40.6627,-73.9387\n"
        "1,US4899999,Nowhere,TX,US,31.0,-99.0\n",
        encoding="utf-8",
    )

    result = centermap.build_center_map(cities_csv)

    assert result.loc[0, "matched"]
    assert result.loc[0, "center_lat"] == 40.7143
    assert not result.loc[1, "matched"]
    assert result.loc[1, "center_lat"] == 31.0
    assert result.loc[1, "offset_km"] == 0.0


def test_name_variants_redirects_an_amalgamated_municipality_to_its_seat() -> None:
    assert centermap.name_variants("Chatham-Kent", "CA3536020") == {"chatham"}
    assert centermap.name_variants("Chatham-Kent") == {"chatham kent"}
