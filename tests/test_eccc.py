# Copyright (C) 2026 Kenneth Porter

from __future__ import annotations

from typing import Any, cast

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
