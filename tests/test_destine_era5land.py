"""Copyright (C) 2026 Kenneth Porter."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
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
    assert destine.EDH_DATASET_URL.endswith("/era5/era5-land-v0.zarr")


def test_retries_transient_dataset_access(monkeypatch: Any) -> None:
    attempts = 0
    delays: list[int] = []

    def open_dataset(*_args: Any, **_kwargs: Any) -> xr.Dataset:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise destine.aiohttp.ClientResponseError(
                request_info=None,
                history=(),
                status=403,
            )
        return _dataset()

    monkeypatch.setattr(destine.xr, "open_dataset", open_dataset)
    monkeypatch.setattr(destine.time, "sleep", delays.append)

    client = destine.DestineEra5LandClient("secret")
    client._open_dataset()

    assert attempts == 3
    assert delays == [300, 300]


def test_does_not_open_dataset_until_retrieval(monkeypatch: Any) -> None:
    attempts = 0

    def open_dataset(*_args: Any, **_kwargs: Any) -> xr.Dataset:
        nonlocal attempts
        attempts += 1
        return _dataset()

    monkeypatch.setattr(destine.xr, "open_dataset", open_dataset)

    destine.DestineEra5LandClient("secret")

    assert attempts == 0


def test_concurrent_retrievals_share_one_dataset_open(monkeypatch: Any) -> None:
    attempts = 0

    def open_dataset(*_args: Any, **_kwargs: Any) -> xr.Dataset:
        nonlocal attempts
        attempts += 1
        time.sleep(0.05)
        return _dataset()

    monkeypatch.setattr(destine.xr, "open_dataset", open_dataset)
    client = destine.DestineEra5LandClient("secret")

    with ThreadPoolExecutor(max_workers=8) as executor:
        datasets = list(executor.map(lambda _index: client._open_dataset(), range(8)))

    assert attempts == 1
    assert all(dataset is datasets[0] for dataset in datasets)


def test_retrieve_uses_basic_auth_and_western_longitude(
    monkeypatch: Any, tmp_path: Any
) -> None:
    captured: dict[str, str] = {}

    def open_dataset(url: str, **_kwargs: Any) -> xr.Dataset:
        captured["url"] = url
        return _dataset()

    monkeypatch.setattr(destine.xr, "open_dataset", open_dataset)
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

    assert captured["url"].startswith("https://edh:secret@data.earthdatahub")
    frame = pd.read_csv(target)
    assert list(frame.columns) == ["valid_time", "t2m", "d2m", "sp"]
    assert len(frame) > 8_760


def test_retrieve_limits_data_to_incremental_date_range(
    monkeypatch: Any, tmp_path: Any
) -> None:
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
