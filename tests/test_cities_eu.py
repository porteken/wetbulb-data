"""Tests for EU city data processing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import cities_eu
from cities_eu import (
    CITY_COORD_DECIMALS,
    EU_LOCATION_ID_OFFSET,
    filter_europe,
    process_cities_eu,
    standard_utc_offset_hours,
)


def _row(
    city: str,
    country_code: str,
    lat: float,
    lng: float,
    *,
    population: int = 100_000,
    feature_class: str = "P",
    timezone: str = "Europe/London",
    dem_m: float = 10.0,
) -> dict[str, object]:
    return {
        "city": city,
        "lat": lat,
        "lng": lng,
        "feature_class": feature_class,
        "country_code": country_code,
        "population": population,
        "dem_m": dem_m,
        "timezone": timezone,
    }


class TestFilterEurope:
    def test_keeps_allowlisted_country_within_bbox(self) -> None:
        df = pd.DataFrame([_row("Paris", "FR", 48.85, 2.35)])
        result = filter_europe(df)
        assert len(result) == 1

    def test_keeps_uk_and_switzerland(self) -> None:
        df = pd.DataFrame(
            [
                _row("London", "GB", 51.51, -0.13),
                _row("Zurich", "CH", 47.37, 8.54),
            ]
        )
        result = filter_europe(df)
        assert set(result["city"]) == {"London", "Zurich"}

    def test_keeps_cyprus_despite_being_geographically_asian(self) -> None:
        df = pd.DataFrame([_row("Limassol", "CY", 34.68, 33.04)])
        result = filter_europe(df)
        assert set(result["city"]) == {"Limassol"}

    def test_excludes_non_eu_european_countries(self) -> None:
        """Norway/Iceland/Balkans/Ukraine/Russia/Turkey are out of scope."""
        df = pd.DataFrame(
            [
                _row("Oslo", "NO", 59.91, 10.75),
                _row("Reykjavik", "IS", 64.14, -21.90),
                _row("Belgrade", "RS", 44.80, 20.47),
                _row("Kyiv", "UA", 50.45, 30.52),
                _row("Moscow", "RU", 55.75, 37.62),
                _row("Istanbul", "TR", 41.01, 28.98),
            ]
        )
        result = filter_europe(df)
        assert result.empty

    def test_excludes_european_microstates(self) -> None:
        df = pd.DataFrame(
            [
                _row("Andorra la Vella", "AD", 42.51, 1.52),
                _row("Monaco", "MC", 43.73, 7.42),
                _row("Gibraltar", "GI", 36.14, -5.35),
            ]
        )
        result = filter_europe(df)
        assert result.empty

    def test_excludes_non_allowlisted_country(self) -> None:
        df = pd.DataFrame([_row("Yerevan", "AM", 40.18, 44.51)])
        result = filter_europe(df)
        assert result.empty

    def test_excludes_non_populated_place(self) -> None:
        df = pd.DataFrame([_row("Somewhere", "FR", 48.85, 2.35, feature_class="A")])
        result = filter_europe(df)
        assert result.empty

    def test_bbox_drops_atlantic_island_territories(self) -> None:
        """Canaries/Madeira/Azores carry mainland ES/PT codes; the bbox drops them."""
        df = pd.DataFrame(
            [
                _row("Las Palmas", "ES", 28.10, -15.42),
                _row("Funchal", "PT", 32.65, -16.91),
                _row("Ponta Delgada", "PT", 37.74, -25.67),
                _row("Lisbon", "PT", 38.72, -9.14),
            ]
        )
        result = filter_europe(df)
        assert set(result["city"]) == {"Lisbon"}


class TestStandardUtcOffsetHours:
    @pytest.mark.parametrize(
        ("tz_name", "expected_hours"),
        [
            ("Europe/London", 0),
            ("Europe/Dublin", 0),
            ("Europe/Lisbon", 0),
            ("Europe/Paris", 1),
            ("Europe/Zurich", 1),
            ("Europe/Madrid", 1),
            ("Europe/Helsinki", 2),
            ("Europe/Athens", 2),
            ("Asia/Nicosia", 2),
        ],
    )
    def test_standard_offset(self, tz_name: str, expected_hours: int) -> None:
        assert standard_utc_offset_hours(tz_name) == expected_hours


class TestProcessCitiesEu:
    def test_returns_at_most_500_cities(self) -> None:
        rows = [
            _row(f"City{i}", "FR", 45.0 + (i * 0.001), 2.0 + (i * 0.001))
            for i in range(600)
        ]
        result = process_cities_eu(pd.DataFrame(rows), {"FR": "France"})
        assert len(result) <= 500

    def test_columns_present(self) -> None:
        df = pd.DataFrame(
            [_row("Paris", "FR", 48.85, 2.35), _row("Berlin", "DE", 52.52, 13.4)]
        )
        result = process_cities_eu(df, {"FR": "France", "DE": "Germany"})
        assert set(result.columns) == {
            "location_id",
            "city",
            "state",
            "lat",
            "lng",
            "dem_m",
            "timezone",
            "utc_offset_hours",
        }

    def test_keeps_highest_population_per_grid_cell(self) -> None:
        df = pd.DataFrame(
            [
                _row("Small", "FR", 48.851, 2.351, population=1000),
                _row("Big", "FR", 48.852, 2.352, population=5000),
            ]
        )
        result = process_cities_eu(df, {"FR": "France"})
        assert len(result) == 1
        assert result.iloc[0]["city"] == "Big"

    def test_location_id_starts_at_offset(self) -> None:
        df = pd.DataFrame([_row("Paris", "FR", 48.85, 2.35)])
        result = process_cities_eu(df, {"FR": "France"})
        assert result.iloc[0]["location_id"] == EU_LOCATION_ID_OFFSET

    def test_state_uses_country_name(self) -> None:
        df = pd.DataFrame([_row("Paris", "FR", 48.85, 2.35)])
        result = process_cities_eu(df, {"FR": "France"})
        assert result.iloc[0]["state"] == "France"

    def test_state_falls_back_to_country_code_when_unmapped(self) -> None:
        df = pd.DataFrame([_row("Paris", "FR", 48.85, 2.35)])
        result = process_cities_eu(df, {})
        assert result.iloc[0]["state"] == "FR"

    def test_rounds_coordinates_to_configured_precision(self) -> None:
        df = pd.DataFrame([_row("Paris", "FR", 48.123412, 2.987612)])
        result = process_cities_eu(df, {"FR": "France"})
        assert result.iloc[0]["lat"] == pytest.approx(
            round(48.123412, CITY_COORD_DECIMALS)
        )
        assert result.iloc[0]["lng"] == pytest.approx(
            round(2.987612, CITY_COORD_DECIMALS)
        )

    def test_utc_offset_hours_computed_from_timezone(self) -> None:
        df = pd.DataFrame([_row("Paris", "FR", 48.85, 2.35, timezone="Europe/Paris")])
        result = process_cities_eu(df, {"FR": "France"})
        assert result.iloc[0]["utc_offset_hours"] == 1


class TestMain:
    def test_main_saves_csv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        df = pd.DataFrame([_row("Paris", "FR", 48.85, 2.35)])
        monkeypatch.setattr(cities_eu, "load_geonames_cities", lambda: df)
        monkeypatch.setattr(cities_eu, "filter_europe", lambda d: d)
        monkeypatch.setattr(cities_eu, "load_country_names", lambda: {"FR": "France"})
        monkeypatch.setattr(cities_eu, "process_cities_eu", lambda d, _names: d)

        output_file = tmp_path / "cities_eu.csv"
        monkeypatch.chdir(tmp_path)

        cities_eu.main()

        assert output_file.exists()
