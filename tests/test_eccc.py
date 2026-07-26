from __future__ import annotations

import pandas as pd
import pytest

import eccc


def test_parse_eccc_hourly_normalizes_units_flags_and_lst() -> None:
    text = """Date/Time (LST),Temp (°C),Temp Flag,Dew Point Temp (°C),Dew Point Temp Flag,Stn Press (kPa),Stn Press Flag
2025-07-01 00:00,25.0,,20.0,,99.5,
2025-07-01 01:00,99.0,X,19.0,,99.4,
2025-07-01 02:00,24.0,,18.0,,99.3,M
"""
    result = eccc.parse_eccc_hourly(text)

    assert len(result) == 1
    assert result.iloc[0]["tair_c"] == pytest.approx(25.0)
    assert result.iloc[0]["pressure_hpa"] == pytest.approx(995.0)
    assert result.iloc[0]["time"] == pd.Timestamp("2025-07-01 00:00")


def test_parse_eccc_hourly_derives_station_pressure_from_sea_level() -> None:
    text = """Date/Time (LST),Temp (°C),Dew Point Temp (°C),Stn Press (kPa),Sea Level Press (kPa)
2025-01-01 00:00,10.0,5.0,,101.3
"""
    result = eccc.parse_eccc_hourly(text, elevation_m=100.0)

    assert len(result) == 1
    assert 990.0 < result.iloc[0]["pressure_hpa"] < 1013.0


def test_fetch_station_year_falls_through_empty_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(
        station_id: int, _year: int, month: int, **_kwargs: object
    ) -> tuple[str, bool]:
        if station_id == 1:
            return "", False
        return (
            "Date/Time (LST),Temp (°C),Dew Point Temp (°C),Stn Press (kPa)\n"
            f"2025-{month:02d}-01 00:00,20,10,100\n",
            False,
        )

    monkeypatch.setattr(eccc, "download_eccc_month", fake_download)
    frame, gap = eccc.fetch_station_year([1, 2], 2025)

    assert not gap
    assert len(frame) == 12
