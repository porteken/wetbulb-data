# Copyright (C) 2026 Kenneth Porter

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

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


def test_initialize_uses_ambient_credentials_and_validates_secret(
    monkeypatch: Any,
) -> None:
    stub = _StubEe()
    monkeypatch.setattr(gee, "_ee", lambda: stub)
    monkeypatch.delenv(gee.EE_CREDENTIALS_ENV, raising=False)

    gee.initialize_earth_engine(project="ambient-project")
    assert stub.initialized == (None, "ambient-project")
    with pytest.raises(ValueError, match="client_email"):
        gee.initialize_earth_engine('{"project_id": "only-project"}')


def test_collection_and_empty_frame_helpers(monkeypatch: Any) -> None:
    selected = SimpleNamespace(
        filterDate=lambda *_args: selected, select=lambda _bands: selected
    )
    monkeypatch.setattr(
        gee, "_ee", lambda: SimpleNamespace(ImageCollection=lambda _name: selected)
    )

    assert gee._collection(2020, 2020) is selected
    assert gee._raw_region_frame([]).empty
    assert gee._hourly_frame(1, pd.DataFrame()).empty


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

        def filter_date(self, start: str, stop: str) -> object:
            self.ranges.append((start, stop))
            return object()

    collection = Collection()
    Collection.filterDate = Collection.filter_date

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

        def get_info(self) -> list[list[Any]]:
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
    Result.getInfo = Result.get_info
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


def test_fetch_city_returns_gap_after_all_attempts(monkeypatch: Any) -> None:
    class EarthEngineError(Exception):
        pass

    collection = SimpleNamespace(
        getRegion=lambda *_args: SimpleNamespace(
            getInfo=lambda: (_ for _ in ()).throw(EarthEngineError())
        )
    )
    ee = SimpleNamespace(
        Geometry=SimpleNamespace(Point=lambda coordinates: coordinates),
        ee_exception=SimpleNamespace(EEException=EarthEngineError),
    )
    monkeypatch.setattr(gee, "_ee", lambda: ee)
    monkeypatch.setattr(gee, "EE_MAX_RETRIES", 1)
    row = SimpleNamespace(location_id=5, lat=40.0, lng=-75.0, utc_offset_hours=-5)

    frame, had_gap = gee._fetch_city([collection], row)

    assert frame.empty
    assert had_gap


def test_fetch_city_marks_empty_earth_engine_results_as_gap(monkeypatch: Any) -> None:
    collection = SimpleNamespace(
        getRegion=lambda *_args: SimpleNamespace(getInfo=lambda: [["time"]])
    )
    ee = SimpleNamespace(
        Geometry=SimpleNamespace(Point=lambda coordinates: coordinates)
    )
    monkeypatch.setattr(gee, "_ee", lambda: ee)
    row = SimpleNamespace(location_id=5, lat=40.0, lng=-75.0, utc_offset_hours=-5)

    frame, had_gap = gee._fetch_city([collection], row)

    assert frame.empty
    assert had_gap


def test_fetch_batch_and_filled_rows_writeable_result(monkeypatch: Any) -> None:
    row = SimpleNamespace(location_id=5, lat=40.0, lng=-75.0, utc_offset_hours=-5)
    frame = pd.DataFrame({"location_id": [5]})
    monkeypatch.setattr(gee, "_fetch_city", lambda *_args: (frame, False))
    assert gee._fetch_batch([], [row], 1, 0) == {5: (frame, False)}

    hourly = pd.DataFrame({"location_id": [5]})
    daily = pd.DataFrame({"location_id": [5], "date": pd.to_datetime(["2020-01-01"])})
    missing = pd.DataFrame({"location_id": [5], "date": pd.to_datetime(["2020-01-01"])})
    monkeypatch.setattr(gee, "_collection", lambda *_args: object())
    monkeypatch.setattr(gee, "_collection_chunks", lambda *_args: [object()])
    monkeypatch.setattr(gee, "_fetch_batch", lambda *_args: {5: (hourly, False)})
    monkeypatch.setattr(gee.lcd, "concat_frames", lambda _frames: hourly)
    monkeypatch.setattr(
        gee.nldas, "compute_daily_wetbulb", lambda *_args, **_kwargs: daily
    )

    result = gee._filled_rows([row], missing, 2020, 2020, 1, 0)

    assert result is not None
    assert result.loc[0, "source"] == gee.era5land.ERA5LAND_SOURCE


@pytest.mark.parametrize(
    ("result", "daily"),
    [
        ({5: (pd.DataFrame(), True)}, pd.DataFrame()),
        ({5: (pd.DataFrame(), False)}, pd.DataFrame()),
    ],
)
def test_filled_rows_returns_none_for_incomplete_or_empty_daily_data(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[int, tuple[pd.DataFrame, bool]],
    daily: pd.DataFrame,
) -> None:
    missing = pd.DataFrame({"location_id": [5], "date": pd.to_datetime(["2020-01-01"])})
    monkeypatch.setattr(gee, "_collection", lambda *_args: object())
    monkeypatch.setattr(gee, "_collection_chunks", lambda *_args: [object()])
    monkeypatch.setattr(gee, "_fetch_batch", lambda *_args: result)
    monkeypatch.setattr(gee.lcd, "concat_frames", lambda _frames: pd.DataFrame())
    monkeypatch.setattr(
        gee.nldas, "compute_daily_wetbulb", lambda *_args, **_kwargs: daily
    )

    assert gee._filled_rows([], missing, 2020, 2020, 1, 0) is None


def test_filled_rows_returns_none_when_no_days_match_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily = pd.DataFrame({"location_id": [5], "date": pd.to_datetime(["2020-01-02"])})
    missing = pd.DataFrame({"location_id": [5], "date": pd.to_datetime(["2020-01-01"])})
    monkeypatch.setattr(gee, "_collection", lambda *_args: object())
    monkeypatch.setattr(gee, "_collection_chunks", lambda *_args: [object()])
    monkeypatch.setattr(
        gee, "_fetch_batch", lambda *_args: {5: (pd.DataFrame(), False)}
    )
    monkeypatch.setattr(gee.lcd, "concat_frames", lambda _frames: pd.DataFrame())
    monkeypatch.setattr(
        gee.nldas, "compute_daily_wetbulb", lambda *_args, **_kwargs: daily
    )

    assert gee._filled_rows([], missing, 2020, 2020, 1, 0) is None


def test_process_and_main_forward_gapfill_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = pd.DataFrame({"location_id": [5], "lng": [-75.0]})
    missing = pd.DataFrame({"location_id": [5], "date": pd.to_datetime(["2020-01-01"])})
    monkeypatch.setattr(
        gee,
        "resolve_gapfill_targets",
        lambda *_args, **_kwargs: (shard, "fs", "base", [2020], None, missing),
    )
    monkeypatch.setattr(gee.nldas, "load_nldas_city_shard", lambda *_args: shard)
    monkeypatch.setattr(
        gee.era5land,
        "load_utc_offsets",
        lambda _path: pd.DataFrame(columns=["location_id", "utc_offset_hours"]),
    )
    monkeypatch.setattr(
        gee.era5land, "apply_cell_overrides", lambda frame, _path: frame
    )
    monkeypatch.setattr(gee, "_gap_years_by_location", lambda _missing: {5: [2020]})
    filled = pd.DataFrame({"location_id": [5], "date": ["2020-01-01"]})
    monkeypatch.setattr(gee, "_filled_rows", lambda *_args: filled)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        gee, "write_pending_year_batches", lambda *args, **_kwargs: calls.append(args)
    )
    gee.process_earth_engine_gapfill(2020, 2020, "out", 0, 1, 1)
    assert calls

    process = []
    monkeypatch.setattr(sys, "argv", ["earth_engine_era5land.py"])
    monkeypatch.setattr(gee, "initialize_earth_engine", lambda *_args: None)
    monkeypatch.setattr(
        gee,
        "process_earth_engine_gapfill",
        lambda *args, **_kwargs: process.append(args),
    )
    gee.main()
    assert process


@pytest.mark.parametrize(
    "resolved",
    [None, (pd.DataFrame(), "fs", "base", [2020], None, pd.DataFrame())],
)
def test_process_returns_when_no_gapfill_targets(
    monkeypatch: pytest.MonkeyPatch, resolved: Any
) -> None:
    monkeypatch.setattr(
        gee, "resolve_gapfill_targets", lambda *_args, **_kwargs: resolved
    )
    monkeypatch.setattr(
        gee.nldas, "load_nldas_city_shard", lambda *_args: pd.DataFrame()
    )

    gee.process_earth_engine_gapfill(2020, 2020, "out", 0, 1, 1)


def test_process_returns_when_gapfill_fetch_produces_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = pd.DataFrame({"location_id": [5], "lng": [-75.0]})
    missing = pd.DataFrame({"location_id": [5], "date": pd.to_datetime(["2020-01-01"])})
    monkeypatch.setattr(
        gee,
        "resolve_gapfill_targets",
        lambda *_args, **_kwargs: (shard, "fs", "base", [2020], None, missing),
    )
    monkeypatch.setattr(gee.nldas, "load_nldas_city_shard", lambda *_args: shard)
    monkeypatch.setattr(
        gee.era5land,
        "load_utc_offsets",
        lambda _path: pd.DataFrame(columns=["location_id", "utc_offset_hours"]),
    )
    monkeypatch.setattr(
        gee.era5land, "apply_cell_overrides", lambda frame, _path: frame
    )
    monkeypatch.setattr(gee, "_gap_years_by_location", lambda _missing: {5: [2020]})
    monkeypatch.setattr(gee, "_filled_rows", lambda *_args: None)

    gee.process_earth_engine_gapfill(2020, 2020, "out", 0, 1, 1)
