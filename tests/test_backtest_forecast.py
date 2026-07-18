"""Tests for leakage-safe backtest interval calibration."""

from __future__ import annotations

import pandas as pd
import pytest

import backtest_forecast


def test_calibration_uses_only_observable_group_balanced_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backtest_forecast, "MIN_CALIBRATION_CASES", 1)
    common = {
        "metric": "avg_wetbulb",
        "gmst": 10.0,
        "legacy": 10.0,
        "lower": 9.0,
        "upper": 11.0,
        "oracle": False,
        "horizon": 2,
        "season": "Annual",
    }
    cases = pd.DataFrame(
        [
            {
                **common,
                "location_id": 1,
                "station_group": "shared",
                "origin": 2009,
                "year": 2011,
                "actual": 12.0,
            },
            {
                **common,
                "location_id": 2,
                "station_group": "shared",
                "origin": 2009,
                "year": 2011,
                "actual": 20.0,
            },
            {
                **common,
                "location_id": 3,
                "station_group": "future",
                "origin": 2010,
                "year": 2013,
                "actual": 30.0,
            },
            {
                **common,
                "location_id": 4,
                "station_group": "gate",
                "origin": 2012,
                "year": 2014,
                "actual": 10.0,
            },
        ]
    )

    calibrated = backtest_forecast.calibrate_intervals(cases)
    gate = calibrated.loc[calibrated["origin"] == 2012].iloc[0]

    assert gate["calibration_factor"] == pytest.approx(2.0)
    assert gate["lower"] == pytest.approx(8.0)
    assert gate["upper"] == pytest.approx(12.0)
