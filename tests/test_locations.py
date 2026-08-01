"""Tests for database-ready locations CSV generation."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import pytest

import locations

CITIES_CSV_CONTENT = "location_id,city,state,lat,lng\n0,Test ville,TS,30.123,-90.456\n"


def test_main_derives_locations_from_cities_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        locations,
        "_parse_args",
        lambda: argparse.Namespace(
            cities_csv="cities_na.csv",
            centers_csv="cities_na_centers.csv",
            out="locations.csv",
        ),
    )
    (tmp_path / "cities_na.csv").write_text(CITIES_CSV_CONTENT, encoding="utf-8")
    (tmp_path / "cities_na_centers.csv").write_text(
        CENTERS_CSV_CONTENT, encoding="utf-8"
    )
    output_file = tmp_path / "locations.csv"

    locations.main()

    assert output_file.exists()
    frame = pd.read_csv(output_file)
    assert list(frame.columns) == ["id", "city", "state", "lat", "lng"]
    assert frame.iloc[0]["id"] == 0


def test_main_generates_cities_csv_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        locations,
        "_parse_args",
        lambda: argparse.Namespace(
            cities_csv="cities_na.csv",
            centers_csv="cities_na_centers.csv",
            out="locations.csv",
        ),
    )

    def fake_generate() -> None:
        (tmp_path / "cities_na.csv").write_text(CITIES_CSV_CONTENT, encoding="utf-8")
        (tmp_path / "cities_na_centers.csv").write_text(
            CENTERS_CSV_CONTENT, encoding="utf-8"
        )

    monkeypatch.setattr(locations, "generate_cities_csv", fake_generate)

    locations.main()

    assert (tmp_path / "locations.csv").exists()


def test_main_uses_custom_cities_csv_and_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cities_eu.csv").write_text(CITIES_CSV_CONTENT, encoding="utf-8")
    monkeypatch.setattr(
        locations,
        "_parse_args",
        lambda: argparse.Namespace(
            cities_csv="cities_eu.csv",
            centers_csv="cities_na_centers.csv",
            out="locations_eu.csv",
        ),
    )

    locations.main()

    output_file = tmp_path / "locations_eu.csv"
    assert output_file.exists()
    assert not (tmp_path / "locations.csv").exists()


def test_locations_frame_from_cities_csv_renames_location_id_to_id(
    tmp_path: pathlib.Path,
) -> None:
    csv_path = tmp_path / "cities.csv"
    csv_path.write_text(CITIES_CSV_CONTENT, encoding="utf-8")

    result = locations.locations_frame_from_cities_csv(csv_path)

    assert list(result.columns) == ["id", "city", "state", "lat", "lng"]
    assert result.iloc[0]["id"] == 0


CENTERS_CSV_CONTENT = (
    "location_id,center_lat,center_lng,offset_km,matched\n0,30.5,-90.9,5.2,True\n"
)


def test_apply_city_centers_replaces_catalog_coordinates(
    tmp_path: pathlib.Path,
) -> None:
    centers_path = tmp_path / "centers.csv"
    centers_path.write_text(CENTERS_CSV_CONTENT, encoding="utf-8")
    frame = pd.DataFrame(
        {
            "id": [0],
            "city": ["Test ville"],
            "state": ["TS"],
            "lat": [30.123],
            "lng": [-90.456],
        }
    )

    result = locations.apply_city_centers(frame, centers_path)

    assert list(result.columns) == ["id", "city", "state", "lat", "lng"]
    assert result.iloc[0]["lat"] == 30.5
    assert result.iloc[0]["lng"] == -90.9


def test_apply_city_centers_keeps_coordinates_without_a_center_row(
    tmp_path: pathlib.Path,
) -> None:
    centers_path = tmp_path / "centers.csv"
    centers_path.write_text(CENTERS_CSV_CONTENT, encoding="utf-8")
    frame = pd.DataFrame(
        {
            "id": [1000],
            "city": ["Berlin"],
            "state": ["Germany"],
            "lat": [52.5244],
            "lng": [13.4105],
        }
    )

    result = locations.apply_city_centers(frame, centers_path)

    assert result.iloc[0]["lat"] == 52.5244
    assert result.iloc[0]["lng"] == 13.4105


def test_apply_city_centers_without_a_center_file_is_a_passthrough(
    tmp_path: pathlib.Path,
) -> None:
    frame = pd.DataFrame(
        {
            "id": [0],
            "city": ["Test ville"],
            "state": ["TS"],
            "lat": [30.123],
            "lng": [-90.456],
        }
    )

    result = locations.apply_city_centers(frame, tmp_path / "missing.csv")

    assert result.iloc[0]["lat"] == 30.123


def test_apply_city_centers_requires_center_file_when_requested(
    tmp_path: pathlib.Path,
) -> None:
    frame = pd.DataFrame(
        {
            "id": [9, 21],
            "city": ["San Diego", "San Francisco"],
            "state": ["CA", "CA"],
            "lat": [32.8304, 37.7272],
            "lng": [-117.1209, -123.0322],
        }
    )

    with pytest.raises(FileNotFoundError, match=r"make_city_center_map_na\.py"):
        locations.apply_city_centers(
            frame,
            tmp_path / "missing.csv",
            required=True,
        )
