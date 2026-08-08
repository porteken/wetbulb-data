# Copyright (C) 2026 Kenneth Porter

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

import eccc


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def get(self, *_args: Any, **_kwargs: Any) -> _Response:
        return _Response(self.payload)


class _FailingSession:
    def get(self, *_args: Any, **_kwargs: Any) -> _Response:
        raise eccc.requests.RequestException("unavailable")


def test_fetch_station_year_parses_complete_eccc_observations() -> None:
    features = []
    for hour in range(24):
        features.append(
            {
                "properties": {
                    "UTC_DATE": f"2000-01-01T{hour:02d}:00:00",
                    "TEMP": 10.0,
                    "DEW_POINT_TEMP": 5.0,
                    "STATION_PRESSURE": 100.0,
                },
            },
        )
    payload = {
        "features": features,
        "numberMatched": 24,
        "numberReturned": 24,
    }

    frame, gap = eccc.fetch_station_year(
        889,
        2000,
        utc_offset_hours=-8,
        session=cast("Any", _Session(payload)),
    )

    assert gap is False
    assert len(frame) == 24
    assert frame.iloc[0]["pressure_hpa"] == pytest.approx(1000.0)
    assert pd.Timestamp(frame.iloc[0]["time"]).hour == 16


def test_fetch_station_year_returns_empty_for_no_features() -> None:
    frame, gap = eccc.fetch_station_year(
        889,
        2000,
        utc_offset_hours=-8,
        session=cast(
            "Any",
            _Session({"features": [], "numberMatched": 0, "numberReturned": 0}),
        ),
    )

    assert frame.empty
    assert gap is False


def test_fetch_station_year_rejects_flagged_values() -> None:
    payload = {
        "features": [
            {
                "properties": {
                    "UTC_DATE": "2000-01-01T00:00:00",
                    "TEMP": 10.0,
                    "TEMP_FLAG": "E",
                    "DEW_POINT_TEMP": 5.0,
                    "DEW_POINT_TEMP_FLAG": None,
                    "STATION_PRESSURE": 100.0,
                    "STATION_PRESSURE_FLAG": None,
                },
            },
        ],
        "numberMatched": 1,
        "numberReturned": 1,
    }

    frame, gap = eccc.fetch_station_year(
        889,
        2000,
        utc_offset_hours=-8,
        session=cast("Any", _Session(payload)),
    )

    assert frame.empty
    assert gap is False


def test_get_json_retries_and_returns_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eccc, "ECCC_MAX_RETRIES", 1)

    assert (
        eccc._get_json_with_retries(
            cast("Any", _FailingSession()), {}, station_id=1, year=2000
        )
        is None
    )


def test_get_json_retries_before_a_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, *_args: Any, **_kwargs: Any) -> _Response:
            self.calls += 1
            if self.calls == 1:
                raise eccc.requests.RequestException("temporary")
            return _Response({"ok": True})

    monkeypatch.setattr(
        eccc, "time", type("Clock", (), {"sleep": lambda _seconds: None})
    )
    assert eccc._get_json_with_retries(Session(), {}, station_id=1, year=2000) == {
        "ok": True
    }


def test_load_station_map_validates_and_adds_optional_columns(tmp_path: Path) -> None:
    missing = eccc._load_station_map(str(tmp_path / "missing.csv"))
    assert missing.empty

    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text("location_id\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        eccc._load_station_map(str(incomplete))

    stations = tmp_path / "stations.csv"
    stations.write_text(
        "location_id,eccc_station_id,utc_offset_hours\n1,10,-5\n",
        encoding="utf-8",
    )
    result = eccc._load_station_map(str(stations))

    assert result.loc[0, "candidate_rank"] == 1
    assert result.loc[0, "dist_km"] is None


def test_fetch_candidates_and_candidate_daily_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame(
        {
            "time": pd.date_range("2000-01-01", periods=24, freq="h"),
            "tair_c": [10.0] * 24,
            "dewpoint_c": [5.0] * 24,
            "pressure_hpa": [1000.0] * 24,
        }
    )
    monkeypatch.setattr(
        eccc,
        "fetch_station_year",
        lambda *_args, **_kwargs: (source, False),
    )
    results = eccc._fetch_candidates([(10, -5.0)], [2000], concurrency=1)
    assert len(results[(10, -5.0)][0]) == 24

    row = pd.Series(
        {
            "location_id": 1,
            "eccc_station_id": 10,
            "start_year": 2000,
            "end_year": 2000,
            "dist_km": 2.0,
            "elevation_difference_m": 3.0,
            "candidate_rank": 1,
        }
    )
    monkeypatch.setattr(
        eccc,
        "_station_to_hourly",
        lambda *_args: source,
    )
    monkeypatch.setattr(
        eccc.nldas,
        "compute_daily_wetbulb",
        lambda *_args, **_kwargs: pd.DataFrame({"date": ["2000-01-01"]}),
    )

    daily = eccc._candidate_daily_frame(row, source)

    assert daily.loc[0, "source"] == "eccc"
    assert daily.loc[0, "station_id"] == "10"


def test_candidate_helpers_handle_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eccc,
        "fetch_station_year",
        lambda *_args, **_kwargs: (pd.DataFrame(), True),
    )
    result = eccc._fetch_candidates([(10, -5.0)], [2000], concurrency=1)
    assert result[(10, -5.0)][0].empty
    row = pd.Series({"location_id": 1, "eccc_station_id": 10})
    monkeypatch.setattr(eccc, "_station_to_hourly", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(
        eccc.nldas, "compute_daily_wetbulb", lambda *_args, **_kwargs: pd.DataFrame()
    )
    assert eccc._candidate_daily_frame(row, pd.DataFrame()).empty


def test_candidate_daily_frame_filters_year_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = pd.Series(
        {
            "location_id": 1,
            "eccc_station_id": 10,
            "start_year": 2001,
            "end_year": 2002,
        }
    )
    monkeypatch.setattr(eccc, "_station_to_hourly", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(
        eccc.nldas,
        "compute_daily_wetbulb",
        lambda *_args, **_kwargs: pd.DataFrame({"date": ["2000-01-01"]}),
    )

    assert eccc._candidate_daily_frame(row, pd.DataFrame()).empty


def test_process_eccc_selects_and_writes_station_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = pd.DataFrame({"location_id": [1]})
    station_map = pd.DataFrame(
        {
            "location_id": [1],
            "eccc_station_id": [10],
            "utc_offset_hours": [-5.0],
            "candidate_rank": [1],
        }
    )
    daily = pd.DataFrame({"location_id": [1], "date": ["2000-01-01"]})
    monkeypatch.setattr(
        eccc,
        "_load_pending_shard",
        lambda *_args, **_kwargs: (shard, [2000], "fs", "base", "root"),
    )
    monkeypatch.setattr(eccc, "_load_station_map", lambda _path: station_map)
    monkeypatch.setattr(
        eccc.ghcnh, "station_map_for_years", lambda frame, _years: frame
    )
    monkeypatch.setattr(
        eccc,
        "_fetch_candidates",
        lambda *_args: {(10, -5.0): (pd.DataFrame(), set())},
    )
    monkeypatch.setattr(eccc, "_candidate_daily_frame", lambda *_args: daily)
    selected = MagicMock(return_value=daily)
    write = MagicMock()
    monkeypatch.setattr(eccc.ghcnh, "select_best_station_days", selected)
    monkeypatch.setattr(eccc, "write_pending_year_batches", write)

    eccc.process_eccc(2000, 2000, "out", 0, 1, 1)

    selected.assert_called_once()
    write.assert_called_once()


@pytest.mark.parametrize(
    ("loaded", "station_map"),
    [
        (None, pd.DataFrame()),
        (
            (
                pd.DataFrame({"location_id": [1]}),
                [2000],
                "fs",
                "base",
                "root",
            ),
            pd.DataFrame(),
        ),
        (
            (
                pd.DataFrame({"location_id": [1]}),
                [2000],
                "fs",
                "base",
                "root",
            ),
            pd.DataFrame(
                {
                    "location_id": [2],
                    "eccc_station_id": [10],
                    "utc_offset_hours": [-5.0],
                }
            ),
        ),
    ],
)
def test_process_eccc_returns_for_unusable_inputs(
    monkeypatch: pytest.MonkeyPatch, loaded: Any, station_map: pd.DataFrame
) -> None:
    monkeypatch.setattr(eccc, "_load_pending_shard", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(eccc, "_load_station_map", lambda _path: station_map)
    monkeypatch.setattr(
        eccc.ghcnh, "station_map_for_years", lambda frame, _years: frame
    )

    eccc.process_eccc(2000, 2000, "out", 0, 1, 1)


def test_process_eccc_returns_when_candidate_days_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = pd.DataFrame({"location_id": [1]})
    station_map = pd.DataFrame(
        {"location_id": [1], "eccc_station_id": [10], "utc_offset_hours": [-5.0]}
    )
    monkeypatch.setattr(
        eccc,
        "_load_pending_shard",
        lambda *_args, **_kwargs: (shard, [2000], "fs", "base", "root"),
    )
    monkeypatch.setattr(eccc, "_load_station_map", lambda _path: station_map)
    monkeypatch.setattr(
        eccc.ghcnh, "station_map_for_years", lambda frame, _years: frame
    )
    monkeypatch.setattr(
        eccc,
        "_fetch_candidates",
        lambda *_args: {(10, -5.0): (pd.DataFrame(), set())},
    )
    monkeypatch.setattr(eccc, "_candidate_daily_frame", lambda *_args: pd.DataFrame())

    eccc.process_eccc(2000, 2000, "out", 0, 1, 1)


def test_main_forwards_arguments_and_handles_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock()
    monkeypatch.setattr(sys, "argv", ["eccc.py"])
    monkeypatch.setattr(eccc, "process_eccc", process)
    eccc.main()
    process.assert_called_once()

    monkeypatch.setattr(
        eccc,
        "process_eccc",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(SystemExit, match="130"):
        eccc.main()
