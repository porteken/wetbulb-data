"""Tests for the NLDAS-2 wet-bulb worker helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

import nldas
from nldas import (
    EarthdataSession,
    _compute_daily_wetbulb,
    _extract_point_values,
    _iter_time_batches,
    _nearest_grid_indices,
    _resolve_valid_indices,
    _resolve_variable_name,
    _select_time_shard_batches,
    granule_url,
)


class TestGranuleUrl:
    def test_url_shape(self) -> None:
        url = granule_url(pd.Timestamp("2024-07-04 13:00"))
        assert url == (
            "https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/"
            "NLDAS_FORA0125_H.2.0/2024/186/"
            "NLDAS_FORA0125_H.A20240704.1300.020.nc"
        )

    def test_day_of_year_zero_padded(self) -> None:
        url = granule_url(pd.Timestamp("2024-01-05 00:00"))
        assert "/2024/005/" in url

    def test_leap_day(self) -> None:
        url = granule_url(pd.Timestamp("2000-02-29 06:00"))
        assert "/2000/060/" in url
        assert "A20000229.0600" in url

    def test_hour_zero_padded(self) -> None:
        url = granule_url(pd.Timestamp("2024-03-01 03:00"))
        assert "A20240301.0300" in url


class TestNearestGridIndices:
    def test_snaps_to_grid_origin(self) -> None:
        iy, ix = _nearest_grid_indices(
            np.array([nldas.NLDAS_GRID_LAT0]),
            np.array([nldas.NLDAS_GRID_LON0]),
        )
        assert (iy[0], ix[0]) == (0, 0)

    def test_clips_out_of_range_coordinates(self) -> None:
        iy, ix = _nearest_grid_indices(np.array([90.0]), np.array([-200.0]))
        assert iy[0] == nldas.NLDAS_GRID_NLAT - 1
        assert ix[0] == 0

    def test_nearest_rounding(self) -> None:
        lat = nldas.NLDAS_GRID_LAT0 + 2 * nldas.NLDAS_GRID_STEP + 0.01
        iy, _ix = _nearest_grid_indices(
            np.array([lat]), np.array([nldas.NLDAS_GRID_LON0])
        )
        assert iy[0] == 2


class TestResolveValidIndices:
    def test_leaves_land_cells_unchanged(self) -> None:
        grid = np.ones((5, 5))
        iy, ix = _resolve_valid_indices(
            np.array([2]),
            np.array([2]),
            np.array([0.0]),
            np.array([0.0]),
            grid,
        )
        assert (iy[0], ix[0]) == (2, 2)

    def test_finds_nearest_land_cell_for_fill_cell(self) -> None:
        grid = np.ones((5, 5))
        grid[2, 2] = nldas.NLDAS_FILL_THRESHOLD - 1.0  # fill/water
        lats = np.array([nldas.NLDAS_GRID_LAT0 + 2 * nldas.NLDAS_GRID_STEP])
        lons = np.array([nldas.NLDAS_GRID_LON0 + 2 * nldas.NLDAS_GRID_STEP])
        iy, ix = _resolve_valid_indices(np.array([2]), np.array([2]), lats, lons, grid)
        assert (iy[0], ix[0]) != (2, 2)
        assert grid[iy[0], ix[0]] > nldas.NLDAS_FILL_THRESHOLD

    def test_sentinel_when_no_land_within_radius(self) -> None:
        grid = np.full((5, 5), nldas.NLDAS_FILL_THRESHOLD - 1.0)
        iy, ix = _resolve_valid_indices(
            np.array([2]),
            np.array([2]),
            np.array([0.0]),
            np.array([0.0]),
            grid,
            max_radius=1,
        )
        assert (iy[0], ix[0]) == (-1, -1)

    def test_picks_closest_of_multiple_candidates(self) -> None:
        grid = np.ones((7, 7))
        grid[3, 3] = nldas.NLDAS_FILL_THRESHOLD - 1.0
        lat0 = nldas.NLDAS_GRID_LAT0
        lon0 = nldas.NLDAS_GRID_LON0
        step = nldas.NLDAS_GRID_STEP
        lats = np.array([lat0 + 3 * step + 0.01])  # nudged toward (4, 3)
        lons = np.array([lon0 + 3 * step])
        iy, ix = _resolve_valid_indices(np.array([3]), np.array([3]), lats, lons, grid)
        assert (iy[0], ix[0]) == (4, 3)


class _FakeDataset:
    """Minimal stand-in for an xarray.Dataset: attribute + item access."""

    def __init__(self, variables: dict[str, SimpleNamespace]) -> None:
        self.data_vars = variables
        self._variables = variables

    def __getitem__(self, key: str) -> SimpleNamespace:
        return self._variables[key]


class TestExtractPointValues:
    def test_sentinel_index_yields_nan(self) -> None:
        fake_ds = _FakeDataset(
            {
                "Tair": SimpleNamespace(values=np.ones((3, 3)) * 300.0),
                "Qair": SimpleNamespace(values=np.ones((3, 3))),
                "PSurf": SimpleNamespace(values=np.ones((3, 3))),
            },
        )
        values = _extract_point_values(
            cast("nldas.Dataset", fake_ds),
            np.array([1, -1]),
            np.array([1, -1]),
        )
        assert values["Tair"][0] == pytest.approx(300.0)
        assert np.isnan(values["Tair"][1])

    def test_fill_value_becomes_nan(self) -> None:
        grid = np.ones((3, 3)) * 300.0
        grid[0, 0] = nldas.NLDAS_FILL_THRESHOLD - 1.0
        fake_ds = _FakeDataset(
            {
                "Tair": SimpleNamespace(values=grid),
                "Qair": SimpleNamespace(values=np.ones((3, 3))),
                "PSurf": SimpleNamespace(values=np.ones((3, 3))),
            },
        )
        values = _extract_point_values(
            cast("nldas.Dataset", fake_ds),
            np.array([0]),
            np.array([0]),
        )
        assert np.isnan(values["Tair"][0])


class TestResolveVariableName:
    def test_finds_primary_name(self) -> None:
        fake_ds = SimpleNamespace(data_vars={"Tair": None, "Qair": None, "PSurf": None})
        assert _resolve_variable_name(cast("nldas.Dataset", fake_ds), "Tair") == "Tair"

    def test_falls_back_to_candidate(self) -> None:
        fake_ds = SimpleNamespace(data_vars={"TMP": None})
        assert _resolve_variable_name(cast("nldas.Dataset", fake_ds), "Tair") == "TMP"

    def test_raises_when_no_candidate_present(self) -> None:
        fake_ds = SimpleNamespace(data_vars={"unrelated": None})
        with pytest.raises(KeyError):
            _resolve_variable_name(cast("nldas.Dataset", fake_ds), "Tair")


class TestComputeDailyWetbulb:
    def _hourly_frame(self, n_hours: int, *, location_id: int = 1) -> pd.DataFrame:
        times = pd.date_range("2024-07-01", periods=n_hours, freq="h")
        return pd.DataFrame(
            {
                "location_id": location_id,
                "time": times,
                "Tair": np.full(n_hours, 303.15),
                "Qair": np.full(n_hours, 0.015),
                "PSurf": np.full(n_hours, 101325.0),
            },
        )

    def test_computes_max_and_avg(self) -> None:
        df = self._hourly_frame(24)
        daily = _compute_daily_wetbulb(df)
        assert len(daily) == 1
        assert daily.loc[0, "wetbulb"] >= daily.loc[0, "wetbulb_avg"] - 1e-6

    def test_drops_incomplete_days(self) -> None:
        df = self._hourly_frame(19)
        daily = _compute_daily_wetbulb(df)
        assert daily.empty

    def test_keeps_day_with_minimum_hours(self) -> None:
        df = self._hourly_frame(nldas.MIN_DAILY_HOURS)
        daily = _compute_daily_wetbulb(df)
        assert len(daily) == 1

    def test_empty_input_returns_empty_with_columns(self) -> None:
        empty = pd.DataFrame(
            columns=pd.Index(["location_id", "time", "Tair", "Qair", "PSurf"]),
        )
        daily = _compute_daily_wetbulb(empty)
        assert list(daily.columns) == ["location_id", "date", "wetbulb", "wetbulb_avg"]
        assert daily.empty

    def test_drops_rows_with_missing_inputs(self) -> None:
        df = self._hourly_frame(24)
        df.loc[0, "Tair"] = np.nan
        daily = _compute_daily_wetbulb(df)
        assert len(daily) == 1  # 23 remaining hours still clears MIN_DAILY_HOURS


class TestIterAndSelectTimeShardBatches:
    def test_covers_full_range(self) -> None:
        hours = list(pd.date_range("2024-01-01", periods=48, freq="h"))
        batches = _iter_time_batches(hours, 24)
        assert len(batches) == 2
        assert sum(len(h) for _idx, h in batches) == 48

    def test_raises_on_zero_batch_hours(self) -> None:
        with pytest.raises(ValueError, match="batch_hours"):
            _iter_time_batches([pd.Timestamp("2024-01-01")], 0)

    def test_shard_selection_partitions_batches(self) -> None:
        hours = list(pd.date_range("2024-01-01", periods=96, freq="h"))
        batches = _iter_time_batches(hours, 24)
        shard0 = _select_time_shard_batches(batches, 0, 2)
        shard1 = _select_time_shard_batches(batches, 1, 2)
        assert {b[0] for b in shard0} | {b[0] for b in shard1} == {
            b[0] for b in batches
        }
        assert not ({b[0] for b in shard0} & {b[0] for b in shard1})

    def test_raises_on_invalid_shard_index(self) -> None:
        with pytest.raises(ValueError, match="time_shard_index"):
            _select_time_shard_batches([(0, [])], 2, 2)


class TestEarthdataSession:
    def _prepared_request(
        self, url: str, *, authorization: str | None
    ) -> SimpleNamespace:
        headers: dict[str, str] = {}
        if authorization is not None:
            headers["Authorization"] = authorization
        return SimpleNamespace(url=url, headers=headers)

    def _response(self, request_url: str) -> SimpleNamespace:
        return SimpleNamespace(request=SimpleNamespace(url=request_url))

    def test_keeps_auth_when_redirected_to_urs(self) -> None:
        session = EarthdataSession("user", "pass")
        prepared = self._prepared_request(
            "https://urs.earthdata.nasa.gov/oauth/authorize",
            authorization="Basic xyz",
        )
        response = self._response("https://hydro1.gesdisc.eosdis.nasa.gov/data/file.nc")
        session.rebuild_auth(prepared, response)
        assert "Authorization" in prepared.headers

    def test_keeps_auth_when_redirected_from_urs(self) -> None:
        session = EarthdataSession("user", "pass")
        prepared = self._prepared_request(
            "https://hydro1.gesdisc.eosdis.nasa.gov/data/file.nc",
            authorization="Basic xyz",
        )
        response = self._response("https://urs.earthdata.nasa.gov/oauth/authorize")
        session.rebuild_auth(prepared, response)
        assert "Authorization" in prepared.headers

    def test_strips_auth_for_unrelated_host_redirect(self) -> None:
        session = EarthdataSession("user", "pass")
        prepared = self._prepared_request(
            "https://evil.example.com/steal",
            authorization="Basic xyz",
        )
        response = self._response("https://hydro1.gesdisc.eosdis.nasa.gov/data/file.nc")
        session.rebuild_auth(prepared, response)
        assert "Authorization" not in prepared.headers

    def test_keeps_auth_when_host_unchanged(self) -> None:
        session = EarthdataSession("user", "pass")
        prepared = self._prepared_request(
            "https://hydro1.gesdisc.eosdis.nasa.gov/data/other.nc",
            authorization="Basic xyz",
        )
        response = self._response("https://hydro1.gesdisc.eosdis.nasa.gov/data/file.nc")
        session.rebuild_auth(prepared, response)
        assert "Authorization" in prepared.headers

    def test_no_auth_header_is_a_noop(self) -> None:
        session = EarthdataSession("user", "pass")
        prepared = self._prepared_request(
            "https://evil.example.com/steal",
            authorization=None,
        )
        response = self._response("https://hydro1.gesdisc.eosdis.nasa.gov/data/file.nc")
        session.rebuild_auth(prepared, response)
        assert "Authorization" not in prepared.headers
