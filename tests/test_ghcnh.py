"""Tests for the NOAA GHCNh wet-bulb worker."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ghcnh


def _parquet_payload(rows: list[dict[str, Any]]) -> bytes:
    defaults: dict[str, Any] = {
        "DATE": "2026-01-01T00:00:00",
        "ELEVATION": 10.0,
        "temperature": 20.0,
        "temperature_Quality_Code": "1",
        "temperature_Report_Type": "FM15",
        "dew_point_temperature": 15.0,
        "dew_point_temperature_Quality_Code": "1",
        "station_level_pressure": 1000.0,
        "station_level_pressure_Quality_Code": "1",
        "sea_level_pressure": None,
        "sea_level_pressure_Quality_Code": None,
        "altimeter": None,
        "altimeter_Quality_Code": None,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    target = io.BytesIO()
    arrow = cast("Any", pa)
    pq.write_table(arrow.Table.from_pandas(frame, preserve_index=False), target)
    return target.getvalue()


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            message = f"HTTP {self.status_code}"
            raise ghcnh.requests.HTTPError(message)


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self.responses = list(responses)

    def get(self, _url: str, **_kwargs: Any) -> _FakeResponse:
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_parse_filters_reports_quality_and_prefers_fm15() -> None:
    payload = _parquet_payload(
        [
            {
                "DATE": "2026-01-01T01:00:00",
                "temperature": 30.0,
                "temperature_Report_Type": "FM16",
            },
            {
                "DATE": "2026-01-01T01:00:00",
                "temperature": 20.0,
                "temperature_Report_Type": "FM15",
            },
            {
                "DATE": "2026-01-01T02:00:00",
                "temperature_Quality_Code": "6",
            },
            {
                "DATE": "2026-01-01T03:00:00",
                "temperature_Report_Type": "FM94_1",
            },
        ],
    )

    result = ghcnh._parse_ghcnh_parquet(
        payload,
        lon=0,
        utc_offset_hours=0,
        drop_incomplete_latest_day=False,
    )

    assert len(result) == 1
    assert result.iloc[0]["tair_c"] == pytest.approx(20.0)


def test_parse_derives_station_pressure_from_sea_level_pressure() -> None:
    payload = _parquet_payload(
        [
            {
                "station_level_pressure": None,
                "station_level_pressure_Quality_Code": None,
                "sea_level_pressure": 1013.25,
                "sea_level_pressure_Quality_Code": "1",
            },
        ],
    )

    result = ghcnh._parse_ghcnh_parquet(
        payload,
        lon=0,
        utc_offset_hours=0,
        drop_incomplete_latest_day=False,
    )

    assert 1000 < result.iloc[0]["pressure_hpa"] < 1013.25


def test_parse_accepts_environment_canada_reports() -> None:
    payload = _parquet_payload(
        [{"temperature_Report_Type": "EnvCan"}],
    )

    result = ghcnh._parse_ghcnh_parquet(
        payload,
        lon=-104.67,
        utc_offset_hours=-6,
        drop_incomplete_latest_day=False,
    )

    assert len(result) == 1


def test_current_local_day_is_removed_but_prior_synoptic_day_is_retained() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01T18:00:00",
                    "2026-01-02T00:00:00",
                    "2026-01-02T20:00:00",
                ],
            ),
            "tair_c": [1.0, 2.0, 3.0],
        },
    )

    result = ghcnh._drop_current_local_day(
        frame,
        utc_offset_hours=0,
        now_utc=datetime(2026, 1, 2, 21, tzinfo=UTC),
    )

    assert list(pd.to_datetime(result["time"]).dt.day) == [1]


def test_previous_local_day_is_retained_without_hour_23() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T18:00:00"]),
        },
    )

    result = ghcnh._drop_current_local_day(
        frame,
        utc_offset_hours=0,
        now_utc=datetime(2026, 1, 2, 12, tzinfo=UTC),
    )

    assert len(result) == 2


def test_fetch_station_year_returns_not_found_as_nontransient_gap() -> None:
    session = cast("Any", _FakeSession([_FakeResponse(404)]))

    frame, gap = ghcnh.fetch_station_year(
        "USW00014732",
        2026,
        lon=-73.88,
        utc_offset_hours=-5,
        drop_incomplete_latest_day=True,
        session=session,
    )

    assert frame.empty
    assert gap is False


def test_fetch_station_year_marks_exhausted_request_as_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ghcnh.time, "sleep", lambda _seconds: None)
    session = cast(
        "Any",
        _FakeSession(
            [ghcnh.requests.RequestException("down")] * ghcnh.GHCNH_MAX_RETRIES,
        ),
    )

    frame, gap = ghcnh.fetch_station_year(
        "USW00014732",
        2026,
        lon=-73.88,
        utc_offset_hours=-5,
        drop_incomplete_latest_day=True,
        session=session,
    )

    assert frame.empty
    assert gap is True
