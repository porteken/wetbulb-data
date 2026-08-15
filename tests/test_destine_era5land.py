"""Copyright (C) 2026 Kenneth Porter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import destine_era5land as destine


def _dataset() -> xr.Dataset:
    times = pd.date_range("2019-12-30", "2021-01-02", freq="h")
    coordinates = {
        "valid_time": times,
        "latitude": [50.0],
        "longitude": [359.0],
    }
    shape = (len(times), 1, 1)
    return xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), np.full(shape, 280.0)),
            "d2m": (("valid_time", "latitude", "longitude"), np.full(shape, 275.0)),
            "sp": (
                ("valid_time", "latitude", "longitude"),
                np.full(shape, 100_000.0),
            ),
        },
        coords=coordinates,
    )


def test_requires_api_key() -> None:
    with pytest.raises(ValueError, match=destine.EDH_API_KEY_ENV):
        destine.DestineEra5LandClient("")


def test_uses_current_hourly_dataset() -> None:
    assert destine.EDH_DATASET_URL.endswith(
        "/era5/reanalysis-era5-land-no-antartica-v0.zarr"
    )


def test_retrieve_uses_basic_auth_and_western_longitude(
    monkeypatch: Any, tmp_path: Any
) -> None:
    captured: dict[str, Any] = {}

    def filesystem(_protocol: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(get_mapper=lambda url: url)

    monkeypatch.setattr(destine.fsspec, "filesystem", filesystem)
    monkeypatch.setattr(
        destine.xr, "open_dataset", lambda *_args, **_kwargs: _dataset()
    )
    client = destine.DestineEra5LandClient("secret")
    target = tmp_path / "span.csv"

    client.retrieve(
        "unused",
        {
            "location": {"latitude": 50.0, "longitude": -1.0},
            "date": ["2020-01-01/2020-12-31"],
        },
        str(target),
    )

    authorization = captured["client_kwargs"]["headers"]["Authorization"]
    assert authorization == destine.aiohttp.encode_basic_auth("edh", "secret")
    frame = pd.read_csv(target)
    assert list(frame.columns) == ["valid_time", "t2m", "d2m", "sp"]
    assert len(frame) > 8_760


def test_retrieve_limits_data_to_incremental_date_range(
    monkeypatch: Any, tmp_path: Any
) -> None:
    monkeypatch.setattr(
        destine.fsspec,
        "filesystem",
        lambda *_args, **_kwargs: SimpleNamespace(get_mapper=lambda url: url),
    )
    monkeypatch.setattr(
        destine.xr, "open_dataset", lambda *_args, **_kwargs: _dataset()
    )
    client = destine.DestineEra5LandClient("secret")
    client.start_date = "2020-06-10"
    client.end_date = "2020-06-12"
    target = tmp_path / "span.csv"

    client.retrieve(
        "unused",
        {
            "location": {"latitude": 50.0, "longitude": -1.0},
            "date": ["2020-01-01/2020-12-31"],
        },
        str(target),
    )

    frame = pd.read_csv(target)
    assert frame["valid_time"].min().startswith("2020-06-08")
    assert frame["valid_time"].max().startswith("2020-06-14")
