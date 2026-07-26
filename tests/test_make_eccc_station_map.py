from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import make_eccc_station_map as stationmap
from make_eccc_station_map import rank_stations

INVENTORY_HEADER = "banner\nbanner\nbanner\n"
INVENTORY_COLUMNS = (
    "Station ID,Name,Latitude (Decimal Degrees),Longitude (Decimal Degrees),"
    "Elevation (m),HLY First Year,HLY Last Year"
)


def _inventory_csv(rows: str) -> str:
    return f"{INVENTORY_HEADER}{INVENTORY_COLUMNS}\n{rows}"


def _inventory_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "station_name": ["Near", "Mid", "Far"],
            "lat": [45.05, 45.3, 45.7],
            "lng": [-75.0, -75.0, -75.0],
            "elevation_m": [70.0, 80.0, 90.0],
            "hourly_first_year": [2000, 1990, 1990],
            "hourly_last_year": [2025, 2025, 2025],
        }
    )


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return


class _FakeSession:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse(self.content)


class TestLoadStationInventory:
    def test_reads_a_local_csv_and_normalizes_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "inventory.csv"
        path.write_text(_inventory_csv("1,Ottawa,45.3,-75.7,70,2000,2025\n"))

        frame = stationmap.load_station_inventory(str(path))

        assert frame["station_id"].tolist() == [1]
        assert frame.iloc[0]["station_name"] == "Ottawa"
        assert frame.iloc[0]["hourly_last_year"] == 2025

    def test_downloads_over_https(self) -> None:
        session = _FakeSession(
            _inventory_csv("2,Toronto,43.7,-79.4,80,1995,2020\n").encode("latin1")
        )

        frame = stationmap.load_station_inventory(
            "https://example.test/inventory.csv", session=cast("Any", session)
        )

        assert session.calls == ["https://example.test/inventory.csv"]
        assert frame.iloc[0]["station_id"] == 2

    def test_drops_rows_with_unusable_numbers(self, tmp_path: Path) -> None:
        path = tmp_path / "inventory.csv"
        path.write_text(
            _inventory_csv(
                "1,Ottawa,45.3,-75.7,70,2000,2025\n3,Broken,,,,,\n",
            )
        )

        frame = stationmap.load_station_inventory(str(path))

        assert frame["station_id"].tolist() == [1]

    def test_rejects_an_inventory_missing_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "inventory.csv"
        path.write_text(f"{INVENTORY_HEADER}Station ID,Name\n1,Ottawa\n")
        source = str(path)

        with pytest.raises(ValueError, match="missing columns"):
            stationmap.load_station_inventory(source)


class TestHaversineKm:
    def test_zero_distance_to_itself(self) -> None:
        distance = stationmap.haversine_km(
            45.0, -75.0, pd.Series([45.0]), pd.Series([-75.0])
        )
        assert distance.iloc[0] == pytest.approx(0.0)

    def test_one_degree_of_latitude_is_about_111_km(self) -> None:
        distance = stationmap.haversine_km(
            45.0, -75.0, pd.Series([46.0]), pd.Series([-75.0])
        )
        assert distance.iloc[0] == pytest.approx(111.2, abs=0.5)


class TestRankStations:
    def test_prefers_full_record_before_distance(self) -> None:
        inventory = pd.DataFrame(
            {
                "station_id": [1, 2],
                "lat": [45.01, 45.1],
                "lng": [-75.0, -75.0],
                "elevation_m": [10, 10],
                "hourly_first_year": [2000, 1990],
                "hourly_last_year": [2025, 2025],
            }
        )

        result = rank_stations(
            inventory,
            city_lat=45,
            city_lng=-75,
            start_year=1991,
            end_year=2025,
        )

        assert result.iloc[0]["station_id"] == 2

    def test_returns_empty_when_nothing_is_within_range(self) -> None:
        result = rank_stations(
            _inventory_frame(),
            city_lat=0.0,
            city_lng=0.0,
            start_year=2000,
            end_year=2025,
        )

        assert result.empty

    def test_prefers_the_nearest_radius_tier(self) -> None:
        result = rank_stations(
            _inventory_frame(),
            city_lat=45.0,
            city_lng=-75.0,
            start_year=2000,
            end_year=2025,
        )

        assert result["radius_tier"].tolist() == [0, 1, 2]
        assert result.iloc[0]["station_id"] == 1


class TestBuildStationMap:
    def _cities_csv(self, tmp_path: Path, *, lat: float, lng: float) -> str:
        path = tmp_path / "cities_ca.csv"
        path.write_text(f"location_id,city,lat,lng\n500,Ottawa,{lat},{lng}\n")
        return str(path)

    def test_maps_up_to_three_candidates(self, tmp_path: Path) -> None:
        result = stationmap.build_station_map(
            self._cities_csv(tmp_path, lat=45.0, lng=-75.0),
            _inventory_frame(),
            start_year=2000,
            end_year=2025,
        )

        row = result.iloc[0]
        assert row["location_id"] == 500
        assert row["eccc_station_ids"] == "1|2|3"
        assert row["primary_station_name"] == "Near"
        assert row["elevation_m"] == pytest.approx(70.0)
        assert row["utc_offset_hours"] == -5

    def test_records_an_empty_row_when_no_station_is_near(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            result = stationmap.build_station_map(
                self._cities_csv(tmp_path, lat=0.0, lng=0.0),
                _inventory_frame(),
            )

        row = result.iloc[0]
        assert row["eccc_station_ids"] is None
        assert row["distance_km"] is None
        assert any("No ECCC hourly station" in message for message in caplog.messages)


class TestMain:
    def test_writes_the_crosswalk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cities = tmp_path / "cities_ca.csv"
        cities.write_text("location_id,city,lat,lng\n500,Ottawa,45.0,-75.0\n")
        inventory = tmp_path / "inventory.csv"
        inventory.write_text(_inventory_csv("1,Ottawa,45.05,-75.0,70,2000,2025\n"))
        out = tmp_path / "cities_ca_eccc_stations.csv"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "make_eccc_station_map.py",
                "--cities-csv",
                str(cities),
                "--inventory",
                str(inventory),
                "--out",
                str(out),
            ],
        )

        stationmap.main()

        written = pd.read_csv(out)
        assert written["location_id"].tolist() == [500]
        assert written.iloc[0]["eccc_station_ids"] == 1
