"""Tests for database-ready locations CSV generation."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pandas as pd

import locations

if TYPE_CHECKING:
    import pytest

CITIES_CSV_CONTENT = "location_id,city,state,lat,lng\n0,Test ville,TS,30.123,-90.456\n"


def test_main_derives_locations_from_cities_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cities.csv").write_text(CITIES_CSV_CONTENT, encoding="utf-8")
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

    def fake_generate() -> None:
        (tmp_path / "cities.csv").write_text(CITIES_CSV_CONTENT, encoding="utf-8")

    monkeypatch.setattr(locations, "generate_cities_csv", fake_generate)

    locations.main()

    assert (tmp_path / "locations.csv").exists()


def test_locations_frame_from_cities_csv_renames_location_id_to_id(
    tmp_path: pathlib.Path,
) -> None:
    csv_path = tmp_path / "cities.csv"
    csv_path.write_text(CITIES_CSV_CONTENT, encoding="utf-8")

    result = locations.locations_frame_from_cities_csv(csv_path)

    assert list(result.columns) == ["id", "city", "state", "lat", "lng"]
    assert result.iloc[0]["id"] == 0


def test_build_locations_frame_renames_location_id_to_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = pd.DataFrame(
        {
            "location_id": [0],
            "city": ["Test ville"],
            "state": ["TS"],
            "lat": [30.123],
            "lng": [-90.456],
        }
    )

    monkeypatch.setattr(locations, "load_data", lambda _url: pd.DataFrame())
    monkeypatch.setattr(locations, "filter_bounding_box", lambda df: df)
    monkeypatch.setattr(locations, "process_cities", lambda _df: sample)

    result = locations.build_locations_frame("https://example.test/cities.csv")

    assert list(result.columns) == ["id", "city", "state", "lat", "lng"]
    assert result.iloc[0]["id"] == 0
