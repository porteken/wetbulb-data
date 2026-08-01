from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pandas as pd

import earth_engine_era5land as gee


class _StubEe:
    def __init__(self) -> None:
        self.initialized: tuple[Any, Any] | None = None

        def service_account_credentials(email: str, *, key_data: str) -> Any:
            return (email, json.loads(key_data))

        def initialize(credentials: Any = None, *, project: str | None = None) -> None:
            self.initialized = (credentials, project)

        self.ServiceAccountCredentials = service_account_credentials
        self.Initialize = initialize


def test_initialize_accepts_json_secret(monkeypatch: Any) -> None:
    stub = _StubEe()
    monkeypatch.setattr(gee, "_ee", lambda: stub)
    details = {
        "client_email": "weather@example.invalid",
        "project_id": "weather-project",
        "private_key": "secret",
    }

    gee.initialize_earth_engine(json.dumps(details))

    assert stub.initialized is not None
    credentials, project = stub.initialized
    assert credentials[0] == "weather@example.invalid"
    assert project == "weather-project"


def test_hourly_frame_converts_earth_engine_schema() -> None:
    raw = pd.DataFrame(
        {
            "time": [1_577_836_800_000],
            "temperature_2m": [293.15],
            "dewpoint_temperature_2m": [283.15],
            "surface_pressure": [101_325.0],
        }
    )

    result = gee._hourly_frame(5, raw)

    assert list(result.columns) == ["location_id", "time", "Tair", "Qair", "PSurf"]
    assert result.iloc[0]["location_id"] == 5
    assert result.iloc[0]["Tair"] == 293.15
    assert pd.notna(result.iloc[0]["Qair"])


def test_filled_rows_requires_24_hours(monkeypatch: Any) -> None:
    hours = pd.date_range("2020-01-01", periods=23, freq="h")
    hourly = pd.DataFrame(
        {
            "location_id": 5,
            "time": hours,
            "Tair": 293.15,
            "Qair": 0.008,
            "PSurf": 101_325.0,
            "utc_offset_hours": 0.0,
        }
    )
    monkeypatch.setattr(gee, "_collection", lambda *_a: object())
    monkeypatch.setattr(gee, "_collection_chunks", lambda collection, *_a: [collection])
    monkeypatch.setattr(gee, "_fetch_batch", lambda *_a: {5: (hourly, False)})
    missing = pd.DataFrame({"location_id": [5], "date": pd.to_datetime(["2020-01-01"])})

    result = gee._filled_rows(
        [SimpleNamespace(location_id=5)], missing, 2020, 2020, 1, 0
    )

    assert result is None


def test_collection_chunks_bounds_each_earth_engine_request() -> None:
    class Collection:
        def __init__(self) -> None:
            self.ranges: list[tuple[str, str]] = []

        def filterDate(self, start: str, stop: str) -> object:  # noqa: N802
            self.ranges.append((start, stop))
            return object()

    collection = Collection()

    chunks = gee._collection_chunks(collection, 2020, 2020)

    assert len(chunks) == 12
    assert collection.ranges[0] == ("2019-12-30", "2020-01-30")
    assert collection.ranges[-1] == ("2020-12-05", "2021-01-03")


def test_fetch_city_retries_earth_engine_exception(monkeypatch: Any) -> None:
    class EarthEngineError(Exception):
        pass

    class Result:
        def __init__(self) -> None:
            self.calls = 0

        def getInfo(self) -> list[list[Any]]:  # noqa: N802
            self.calls += 1
            if self.calls == 1:
                raise EarthEngineError
            return [
                [
                    "time",
                    "temperature_2m",
                    "dewpoint_temperature_2m",
                    "surface_pressure",
                ],
                [1_577_836_800_000, 293.15, 283.15, 101_325.0],
            ]

    result = Result()
    collection = SimpleNamespace(getRegion=lambda *_a: result)
    ee = SimpleNamespace(
        Geometry=SimpleNamespace(Point=lambda coordinates: coordinates),
        ee_exception=SimpleNamespace(EEException=EarthEngineError),
    )
    monkeypatch.setattr(gee, "_ee", lambda: ee)
    monkeypatch.setattr(gee, "EE_RETRY_DELAY_SECONDS", 0)
    row = SimpleNamespace(location_id=5, lat=40.0, lng=-75.0, utc_offset_hours=-5)

    frame, had_gap = gee._fetch_city([collection], row)

    assert result.calls == 2
    assert not had_gap
    assert len(frame) == 1
