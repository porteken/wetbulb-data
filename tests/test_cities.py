"""Tests for city data processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cities
from cities import CITY_COORD_DECIMALS, filter_bounding_box, process_cities


class TestFilterBoundingBox:
    def test_keeps_continental_us(self) -> None:
        df = pd.DataFrame(
            {
                "lat": [40.0, 50.0, 20.0, 35.0],
                "lng": [-90.0, -80.0, -90.0, -100.0],
            }
        )
        result = filter_bounding_box(df)
        assert len(result) == 2

    def test_excludes_outside_bounds(self) -> None:
        df = pd.DataFrame({"lat": [10.0], "lng": [-130.0]})
        result = filter_bounding_box(df)
        assert len(result) == 0


class TestLoadData:
    def test_normalizes_column_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df = pd.DataFrame(
            {
                "City": ["A"],
                "State": ["X"],
                "Population": [1000],
                "lat": [30.0],
                "lon": [-90.0],
            }
        )
        monkeypatch.setattr(pd, "read_csv", lambda _url: df)
        result = cities.load_data("https://example.com")
        assert list(result.columns) == ["city", "state", "population", "lat", "lng"]

    def test_normalizes_alternative_column_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = pd.DataFrame(
            {
                "name": ["A"],
                "State": ["X"],
                "pop": [1000],
                "lat": [30.0],
                "lon": [-90.0],
            }
        )
        monkeypatch.setattr(pd, "read_csv", lambda _url: df)
        result = cities.load_data("https://example.com")
        assert list(result.columns) == ["city", "state", "population", "lat", "lng"]


class TestMain:
    def test_main_saves_csv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import cities

        df = pd.DataFrame(
            {
                "city": ["A"],
                "state": ["X"],
                "lat": [30.0],
                "lng": [-90.0],
                "population": [1000],
            }
        )
        monkeypatch.setattr(cities, "load_data", lambda _url: df)
        monkeypatch.setattr(cities, "filter_bounding_box", lambda d: d)
        monkeypatch.setattr(cities, "process_cities", lambda d: d)

        output_file = tmp_path / "cities.csv"
        monkeypatch.chdir(tmp_path)

        cities.main()

        assert output_file.exists()


class TestProcessCities:
    def test_returns_at_most_500_cities(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "city": [f"City{i}" for i in range(600)],
                "state": ["ST"] * 600,
                "lat": rng.uniform(25, 49, 600),
                "lng": rng.uniform(-124, -67, 600),
                "population": rng.integers(1000, 1_000_000, 600),
            }
        )
        result = process_cities(df)
        assert len(result) <= 500

    def test_columns_present(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["A", "B"],
                "state": ["X", "Y"],
                "lat": [30.0, 31.0],
                "lng": [-90.0, -91.0],
                "population": [100000, 200000],
            }
        )
        result = process_cities(df)
        assert set(result.columns) == {"location_id", "city", "state", "lat", "lng"}

    def test_keeps_highest_population_per_grid_cell(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["Small", "Big"],
                "state": ["ST", "ST"],
                "lat": [30.01, 30.02],
                "lng": [-90.01, -90.02],
                "population": [1000, 5000],
            }
        )
        result = process_cities(df)
        assert len(result) == 1
        assert result.iloc[0]["city"] == "Big"

    def test_location_id_starts_at_zero(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["A"],
                "state": ["X"],
                "lat": [30.0],
                "lng": [-90.0],
                "population": [100000],
            }
        )
        result = process_cities(df)
        assert result.iloc[0]["location_id"] == 0

    def test_rounds_coordinates_to_configured_precision(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["A"],
                "state": ["X"],
                "lat": [30.12341],
                "lng": [-90.98761],
                "population": [100000],
            }
        )

        result = process_cities(df)

        assert result.iloc[0]["lat"] == pytest.approx(
            round(30.12341, CITY_COORD_DECIMALS)
        )
        assert result.iloc[0]["lng"] == pytest.approx(
            round(-90.98761, CITY_COORD_DECIMALS)
        )
