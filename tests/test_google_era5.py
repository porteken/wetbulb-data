"""Tests for ERA5 batch/shard helpers."""

from __future__ import annotations

import argparse
import pathlib
from datetime import UTC
from types import SimpleNamespace
from typing import Self, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import google_era5
from google_era5 import (
    DEFAULT_BATCH_HOURS,
    ERA5_TIME_ORIGIN,
    _approximate_dsrp,
    _BatchWriteTargets,
    _compute_location_frame,
    _compute_pet_chunk,
    _compute_wetbulb_series,
    _hourly_flux_from_accumulation,
    _iter_time_batches,
    _normalize_longitudes_for_solar_geometry,
    _process_era5_batch_job,
    _ProductPlan,
    _resolve_era5_max_workers,
    _resolve_products,
    _run_era5_batch_jobs,
    _select_time_shard_batches,
    _ThermofeelModule,
    _wrap_longitudes_for_arco_selection,
    _year_time_slice,
)
from partition_io import PartitionTarget as _PartitionTarget


class TestArcoStableEndYear:
    def test_returns_previous_year_early_in_year(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import datetime

        fake_now = datetime(2024, 3, 1, tzinfo=UTC)

        class FakeDatetime:
            @classmethod
            def now(cls, tz: object = None) -> datetime:
                _ = (cls, tz)
                return fake_now

        monkeypatch.setattr(google_era5, "datetime", FakeDatetime)
        assert google_era5._arco_stable_end_year() == 2022

    def test_returns_last_year_later_in_year(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import datetime

        fake_now = datetime(2024, 5, 1, tzinfo=UTC)

        class FakeDatetime:
            @classmethod
            def now(cls, tz: object = None) -> datetime:
                _ = (cls, tz)
                return fake_now

        monkeypatch.setattr(google_era5, "datetime", FakeDatetime)
        assert google_era5._arco_stable_end_year() == 2023


class TestResolveStoreVariables:
    def test_raises_on_missing_variables(self) -> None:
        from google_era5 import _resolve_store_variables

        with pytest.raises(KeyError, match="Required ARCO variables missing"):
            _resolve_store_variables({"10u"})


class TestDefaultBatchHours:
    def test_is_monthly(self) -> None:
        assert DEFAULT_BATCH_HOURS == 24 * 30


class TestYearTimeSlice:
    def test_returns_tuple(self) -> None:
        start_h, end_h = _year_time_slice(2024)
        assert isinstance(start_h, int)
        assert isinstance(end_h, int)
        assert end_h > start_h

    def test_year_span_is_correct(self) -> None:
        start_h, end_h = _year_time_slice(2024)

        assert end_h - start_h + 1 == 8784

    def test_starts_at_beginning_of_year(self) -> None:
        start_h, _ = _year_time_slice(2024)
        epoch = pd.Timestamp(google_era5.ERA5_TIME_ORIGIN)
        assert epoch + pd.Timedelta(hours=start_h) == pd.Timestamp(
            "2024-01-01 00:00:00"
        )

    def test_non_leap_year(self) -> None:
        start_h, end_h = _year_time_slice(2023)
        assert end_h - start_h + 1 == 8760


class TestIterTimeBatches:
    def test_covers_full_year(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        start_h, end_h = _year_time_slice(2024)
        assert batches[0][1] == start_h
        assert batches[-1][2] == end_h

    def test_batch_count_monthly(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=720)

        assert len(batches) == 13

    def test_no_gaps_between_batches(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        for i in range(1, len(batches)):
            prev_end = batches[i - 1][2]
            curr_start = batches[i][1]
            assert curr_start == prev_end + 1

    def test_batch_indices_sequential(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        assert [b[0] for b in batches] == list(range(len(batches)))

    def test_raises_on_zero_batch_hours(self) -> None:
        with pytest.raises(ValueError, match="batch_hours"):
            _iter_time_batches(2024, batch_hours=0)


class TestSelectTimeShardBatches:
    def test_all_batches_assigned(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        all_assigned = []
        for shard_index in range(4):
            selected = _select_time_shard_batches(
                batches, time_shard_index=shard_index, time_shard_count=4
            )
            all_assigned.extend(selected)

        assert sorted(b[0] for b in all_assigned) == sorted(b[0] for b in batches)

    def test_single_shard_returns_all(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        selected = _select_time_shard_batches(
            batches, time_shard_index=0, time_shard_count=1
        )
        assert len(selected) == len(batches)

    def test_four_shards_balanced(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        shard_sizes = []
        for shard_index in range(4):
            selected = _select_time_shard_batches(
                batches, time_shard_index=shard_index, time_shard_count=4
            )
            shard_sizes.append(len(selected))

        assert max(shard_sizes) - min(shard_sizes) <= 1

    def test_raises_on_invalid_index(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        with pytest.raises(ValueError, match="time_shard_index"):
            _select_time_shard_batches(batches, time_shard_index=4, time_shard_count=4)


class TestResolveEra5MaxWorkers:
    def test_minus_one_returns_cpu_count(self) -> None:
        result = _resolve_era5_max_workers(-1)
        assert result >= 1

    def test_explicit_value_passes_through(self) -> None:
        assert _resolve_era5_max_workers(8) == 8

    def test_raises_on_zero(self) -> None:
        with pytest.raises(ValueError, match="max_workers"):
            _resolve_era5_max_workers(0)


class TestHourlyFluxFromAccumulation:
    def test_converts_joules_per_square_meter_to_watts_per_square_meter(self) -> None:
        converted = _hourly_flux_from_accumulation(np.array([3600.0, 7200.0, 1800.0]))

        np.testing.assert_allclose(converted, np.array([1.0, 2.0, 0.5]))


class TestApproximateDsrp:
    def test_does_not_cap_direct_normal_radiation_to_ssrd(self) -> None:
        thermofeel = MagicMock()
        thermofeel.approximate_dsrp.return_value = np.array([500.0, 300.0, 100.0])

        result = _approximate_dsrp(
            cast("_ThermofeelModule", cast("object", thermofeel)),
            fdir_flux=np.array([50.0, 60.0, 70.0]),
            cossza=np.array([0.5, 0.2, 0.05]),
            _ssrd_flux=np.array([100.0, 80.0, 90.0]),
        )

        np.testing.assert_allclose(result, np.array([500.0, 300.0, 0.0]))


class TestLongitudeNormalization:
    def test_wrap_longitudes_for_arco_selection_maps_negative_us_longitudes(
        self,
    ) -> None:
        wrapped = _wrap_longitudes_for_arco_selection(
            np.array([-104.75, -80.5, -74.0, 0.0, 179.75, 360.0])
        )

        np.testing.assert_allclose(
            wrapped,
            np.array([255.25, 279.5, 286.0, 0.0, 179.75, 0.0]),
        )

    def test_normalize_longitudes_for_solar_geometry_restores_western_hemisphere(
        self,
    ) -> None:
        normalized = _normalize_longitudes_for_solar_geometry(
            np.array([255.25, 279.5, 286.0, 0.0, 179.75, 181.0])
        )

        np.testing.assert_allclose(
            normalized,
            np.array([-104.75, -80.5, -74.0, 0.0, 179.75, -179.0]),
        )


class TestComputeLocationFrameErrors:
    def test_warns_on_extreme_mrt(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Trigger the MRT sanity range warning."""
        from unittest.mock import MagicMock

        # Set up mock dataset
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = mock_ds
        mock_ds.sel.return_value = mock_ds
        mock_ds.compute.return_value = mock_ds
        mock_ds.assign_coords.return_value = mock_ds

        fake_time = MagicMock()
        fake_time.values = np.array([1000000], dtype="int64")
        mock_ds.time = fake_time

        # Each var access should return something with .values as a 2D array (n_locs, n_times)
        mock_ds.values = np.array([[300.0], [300.0]])

        # Setup thermofeel mock
        tf = MagicMock()
        # MRT in Kelvin. 400K is ~126.85C (>120C), 300K is ~26.85C (valid)
        tf.calculate_mean_radiant_temperature.return_value = np.array([400.0, 300.0])

        def fake_import(name: str) -> object:
            if name == "thermofeel":
                return tf
            return MagicMock()

        monkeypatch.setattr(google_era5, "_import_optional_module", fake_import)

        cities = pd.DataFrame(
            {"location_id": [0, 1], "lat": [40.0, 41.0], "lng": [-75.0, -76.0]}
        )

        import dask

        monkeypatch.setattr(dask, "config", MagicMock())

        from google_era5 import _compute_location_frame

        _compute_location_frame(mock_ds, cities, compute_workers=1)

        assert "Discarding 1 MRT value(s) outside sanity range" in caplog.text


class TestComputePetChunk:
    def test_discards_pet_values_above_sixty_c(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = pd.DataFrame(
            {
                "t": [25.0, 26.0, 27.0],
                "mrt": [30.0, 31.0, 32.0],
                "v": [1.0, 1.0, 1.0],
                "rh": [50.0, 50.0, 50.0],
            }
        )

        monkeypatch.setattr(
            google_era5,
            "pet_corrected",
            lambda *_args, **_kwargs: np.array([22.0, 61.0, -55.0]),
        )

        result = _compute_pet_chunk(df)

        assert result["pet"].tolist() == pytest.approx([22.0])


class TestComputeWetbulbSeries:
    def test_discards_wetbulb_values_outside_sanity_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            google_era5,
            "wetbulb_stull",
            lambda *_args, **_kwargs: np.array([20.0, 999.0, -999.0]),
        )

        result = _compute_wetbulb_series(
            np.array([25.0, 25.0, 25.0]),
            np.array([50.0, 50.0, 50.0]),
        )

        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])


class TestProcessEra5BatchJob:
    def test_writes_pet_and_wetbulb_partitions(self, tmp_path: pathlib.Path) -> None:
        frame = pd.DataFrame(
            {
                "location_id": [1, 1],
                "date": [
                    pd.Timestamp("2020-01-01").date(),
                    pd.Timestamp("2020-01-02").date(),
                ],
                "pet": [20.0, 21.0],
                "wetbulb": [15.0, 16.0],
            },
        )

        def fake_compute_location_frame(
            _ds: object,
            _cities: object,
            _compute_workers: int,
        ) -> pd.DataFrame:
            return frame

        original_compute_location_frame = google_era5._compute_location_frame
        google_era5._compute_location_frame = fake_compute_location_frame
        try:
            pet_root = tmp_path / "pet"
            wetbulb_root = tmp_path / "wetbulb"

            _process_era5_batch_job(
                ds=MagicMock(),
                targets=_BatchWriteTargets(
                    pet_target=_PartitionTarget(str(pet_root)),
                    wetbulb_target=_PartitionTarget(str(wetbulb_root)),
                ),
                year=2020,
                city_shard_index=0,
                batch_index=0,
                start_h=0,
                end_h=23,
                pending_batch_df=pd.DataFrame(
                    {"location_id": [1], "lat": [40.0], "lng": [-75.0]},
                ),
                compute_workers=1,
            )
        finally:
            google_era5._compute_location_frame = original_compute_location_frame

        pet_files = sorted((pet_root / "year=2020").glob("pet_batch_*.parquet"))
        wetbulb_files = sorted(
            (wetbulb_root / "year=2020").glob("wetbulb_batch_*.parquet"),
        )
        assert len(pet_files) == 1
        assert len(wetbulb_files) == 1

        pet_df = pd.read_parquet(pet_files[0])
        wetbulb_df = pd.read_parquet(wetbulb_files[0])
        assert list(pet_df.columns) == ["location_id", "date", "pet"]
        assert list(wetbulb_df.columns) == ["location_id", "date", "wetbulb"]

    def test_skips_wetbulb_partition_when_all_nan(self, tmp_path: pathlib.Path) -> None:
        frame = pd.DataFrame(
            {
                "location_id": [1],
                "date": [pd.Timestamp("2020-01-01").date()],
                "pet": [20.0],
                "wetbulb": [np.nan],
            },
        )

        def fake_compute_location_frame(
            _ds: object,
            _cities: object,
            _compute_workers: int,
        ) -> pd.DataFrame:
            return frame

        original_compute_location_frame = google_era5._compute_location_frame
        google_era5._compute_location_frame = fake_compute_location_frame
        try:
            pet_root = tmp_path / "pet"
            wetbulb_root = tmp_path / "wetbulb"

            _process_era5_batch_job(
                ds=MagicMock(),
                targets=_BatchWriteTargets(
                    pet_target=_PartitionTarget(str(pet_root)),
                    wetbulb_target=_PartitionTarget(str(wetbulb_root)),
                ),
                year=2020,
                city_shard_index=0,
                batch_index=0,
                start_h=0,
                end_h=23,
                pending_batch_df=pd.DataFrame(
                    {"location_id": [1], "lat": [40.0], "lng": [-75.0]},
                ),
                compute_workers=1,
            )
        finally:
            google_era5._compute_location_frame = original_compute_location_frame

        pet_files = sorted((pet_root / "year=2020").glob("pet_batch_*.parquet"))
        assert len(pet_files) == 1
        assert not wetbulb_root.exists() or not any(
            (wetbulb_root / "year=2020").glob("wetbulb_batch_*.parquet"),
        )

    def _run_with_products(
        self,
        tmp_path: pathlib.Path,
        *,
        write_pet: bool,
        write_wetbulb: bool,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        frame = pd.DataFrame(
            {
                "location_id": [1],
                "date": [pd.Timestamp("2020-01-01").date()],
                "pet": [20.0],
                "wetbulb": [15.0],
            },
        )

        def fake_compute_location_frame(
            _ds: object,
            _cities: object,
            _compute_workers: int,
        ) -> pd.DataFrame:
            return frame

        original = google_era5._compute_location_frame
        google_era5._compute_location_frame = fake_compute_location_frame
        pet_root = tmp_path / "pet"
        wetbulb_root = tmp_path / "wetbulb"
        try:
            _process_era5_batch_job(
                ds=MagicMock(),
                targets=_BatchWriteTargets(
                    pet_target=_PartitionTarget(str(pet_root)),
                    wetbulb_target=_PartitionTarget(str(wetbulb_root)),
                    write_pet=write_pet,
                    write_wetbulb=write_wetbulb,
                ),
                year=2020,
                city_shard_index=0,
                batch_index=0,
                start_h=0,
                end_h=23,
                pending_batch_df=pd.DataFrame(
                    {"location_id": [1], "lat": [40.0], "lng": [-75.0]},
                ),
                compute_workers=1,
            )
        finally:
            google_era5._compute_location_frame = original
        return pet_root, wetbulb_root

    def test_wetbulb_only_skips_pet_partition(self, tmp_path: pathlib.Path) -> None:
        pet_root, wetbulb_root = self._run_with_products(
            tmp_path, write_pet=False, write_wetbulb=True
        )
        assert not pet_root.exists()
        assert (
            len(list((wetbulb_root / "year=2020").glob("wetbulb_batch_*.parquet"))) == 1
        )

    def test_pet_only_skips_wetbulb_partition(self, tmp_path: pathlib.Path) -> None:
        pet_root, wetbulb_root = self._run_with_products(
            tmp_path, write_pet=True, write_wetbulb=False
        )
        assert len(list((pet_root / "year=2020").glob("pet_batch_*.parquet"))) == 1
        assert not wetbulb_root.exists()


class TestResolveProducts:
    def test_both_enables_both(self) -> None:
        assert _resolve_products("both") == (True, True)

    def test_pet_only(self) -> None:
        assert _resolve_products("pet") == (True, False)

    def test_wetbulb_only(self) -> None:
        assert _resolve_products("wetbulb") == (False, True)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="products must be one of"):
            _resolve_products("humidex")


class TestRunEra5BatchJobs:
    def test_parallel_branch_uses_threads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        shard_df = pd.DataFrame(
            {
                "location_id": [1],
                "lat": [35.0],
                "lng": [-80.0],
            }
        )
        completed_batches: list[int] = []
        pending_frames: list[pd.DataFrame] = []

        def fake_pet_batch_exists(*_args: object, **_kwargs: object) -> bool:
            return False

        def fake_process_batch(**kwargs: object) -> int:
            batch_index = cast("int", kwargs["batch_index"])
            completed_batches.append(batch_index)
            pending_frames.append(cast("pd.DataFrame", kwargs["pending_batch_df"]))
            return batch_index

        monkeypatch.setattr(
            google_era5,
            "batch_exists",
            fake_pet_batch_exists,
        )
        monkeypatch.setattr(
            google_era5,
            "_process_era5_batch_with_thread_dataset",
            fake_process_batch,
        )
        monkeypatch.setenv("ERA5_CONCURRENCY_PROFILE", "aggressive")

        wrote_batches = _run_era5_batch_jobs(
            selected_batches=[(0, 0, 23), (1, 24, 47)],
            shard_df=shard_df,
            plan=_ProductPlan(
                str(tmp_path / "era5"),
                str(tmp_path / "wetbulb"),
            ),
            year=2001,
            city_shard_index=0,
            city_shard_count=1,
            time_shard_index=0,
            time_shard_count=1,
            max_workers=2,
            force=False,
        )

        assert wrote_batches is True
        assert sorted(completed_batches) == [0, 1]
        assert all(frame is shard_df for frame in pending_frames)


class TestMainExitCodes:
    @staticmethod
    def _make_args() -> argparse.Namespace:
        return argparse.Namespace(
            year=2024,
            months=None,
            out_dir=".",
            city_shard_index=0,
            city_shard_count=1,
            time_shard_index=0,
            time_shard_count=1,
            max_workers=1,
            batch_hours=DEFAULT_BATCH_HOURS,
            concurrency_profile="balanced",
            products="both",
        )

    def test_successful_main_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(google_era5, "_parse_args", self._make_args)
        monkeypatch.setattr(google_era5, "process_era5", lambda **_kwargs: None)

        with pytest.raises(SystemExit) as exc:
            google_era5.main()

        assert exc.value.code == 0

    def test_failed_main_exits_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = MagicMock()
        message = "boom"

        def fail_process_era5(**_kwargs: object) -> None:
            raise RuntimeError(message)

        monkeypatch.setattr(google_era5, "_parse_args", self._make_args)
        monkeypatch.setattr(google_era5, "process_era5", fail_process_era5)
        monkeypatch.setattr(google_era5, "LOGGER", logger)

        with pytest.raises(SystemExit) as exc:
            google_era5.main()

        assert exc.value.code == 1
        logger.exception.assert_called_once_with("ERA5 processing failed.")


class _FakeVar:
    def __init__(self, arr: np.ndarray | None = None) -> None:
        self.values = arr if arr is not None else np.array([])


class _FakeSel:
    def sel(self, *_args: object, **_kw: object) -> _FakeSel:
        return self


class _FakeDataset:
    def __init__(
        self,
        raw: dict[str, np.ndarray] | None = None,
        time_vals: np.ndarray | None = None,
    ) -> None:
        self._raw = raw if raw is not None else {}
        self.time = _FakeVar(time_vals if time_vals is not None else np.array([]))

    def __getitem__(self, key: object) -> _FakeVar | _FakeDataset:
        if isinstance(key, (list, tuple, set)):
            return self
        return _FakeVar(self._raw[cast("str", key)])

    def sel(self, *_args: object, **_kw: object) -> _FakeDataset:
        return self

    def compute(self, *_args: object, **_kw: object) -> _FakeDataset:
        return self

    def assign_coords(self, **kwargs: object) -> _FakeDataset:
        new_time_vals = self.time.values
        if "time" in kwargs:
            from typing import Any

            val = cast("Any", kwargs["time"])
            new_time_vals = val.values if hasattr(val, "values") else np.asarray(val)
        return _FakeDataset(self._raw, new_time_vals)


fake_ds = _FakeDataset()


def assign_coords(**kwargs: object) -> _FakeDataset:
    copy = _FakeDataset()
    if "time" in kwargs:
        copy.time = _FakeVar(np.asarray(kwargs["time"]))
    return copy


def compute() -> object:
    return fake_ds


def sel(*_args: object, **_kw: object) -> _FakeSel:
    return _FakeSel()


def _make_fake_dataset(
    n_times: int,
    n_locs: int,
    hot_temp_k: float,
    cold_temp_k: float,
) -> MagicMock:
    """Return a minimal xarray-like Dataset mock with two distinct temperature profiles."""
    temps = np.empty((n_times, n_locs), dtype=float)
    temps[:, 0] = hot_temp_k
    temps[:, 1] = cold_temp_k
    dew = temps - 10.0

    np.full((n_times, n_locs), np.sqrt(2.0))

    rad_down = np.full((n_times, n_locs), 300.0 * 3600.0)
    rad_net = np.zeros((n_times, n_locs))

    raw = {
        "10u": temps * 0 + np.sqrt(2.0),
        "10v": temps * 0 + np.sqrt(2.0),
        "2t": temps,
        "2d": dew,
        "ssrd": rad_net.copy(),
        "strd": rad_down.copy(),
        "ssr": rad_net.copy(),
        "str": rad_net.copy(),
        "fdir": rad_net.copy(),
        "msdrswrf": rad_net.copy(),
    }

    epoch = pd.Timestamp(ERA5_TIME_ORIGIN)
    start_h = int((pd.Timestamp("2010-01-01") - epoch).total_seconds() // 3600)
    time_vals = np.arange(start_h, start_h + n_times, dtype=int)

    return _FakeDataset(raw, time_vals)  # type: ignore[return-value]


class TestComputeLocationFrameAlignment:
    """Verify weather value alignment to location and time."""

    def test_hot_city_has_higher_pet_than_cold_city(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """City 0 (hot) should consistently produce a higher daily max PET than city 1 (cold)."""
        import dask as _dask

        class _FakeDaskCtx:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_: object) -> None:
                # Fake context manager has no teardown; the test only needs enter/exit hooks.
                pass

        monkeypatch.setattr(
            _dask, "config", MagicMock(set=lambda **_kw: _FakeDaskCtx())
        )
        original_import_optional_module = google_era5._import_optional_module

        def fake_data_array(data: object, dims: str) -> object:
            _ = dims
            return data

        n_times = 24
        n_locs = 2

        def fake_approximate_dsrp(
            fdir: object, cossza: object, threshold: float = 0.1
        ) -> object:
            _ = (cossza, threshold)
            return np.asarray(fdir)

        def fake_calculate_mean_radiant_temperature(**_kwargs: object) -> object:
            return np.full(n_times * n_locs, 300.0)

        fake_xarray = SimpleNamespace(DataArray=fake_data_array)
        fake_thermofeel = SimpleNamespace(
            approximate_dsrp=fake_approximate_dsrp,
            calculate_mean_radiant_temperature=fake_calculate_mean_radiant_temperature,
        )

        def fake_import_optional_module(module_name: str) -> object:
            if module_name == "xarray":
                return fake_xarray
            if module_name == "thermofeel":
                return fake_thermofeel
            return original_import_optional_module(module_name)

        monkeypatch.setattr(
            google_era5, "_import_optional_module", fake_import_optional_module
        )

        fake_ds_with_slice = _make_fake_dataset(
            n_times=n_times,
            n_locs=n_locs,
            hot_temp_k=300.0,
            cold_temp_k=280.0,
        )

        cities = pd.DataFrame(
            {
                "location_id": [0, 1],
                "lat": [40.0, 40.0],
                "lng": [-75.0, -75.0],
            }
        )

        result = _compute_location_frame(
            fake_ds_with_slice,  # type: ignore[arg-type]
            cities,
            compute_workers=1,
        )

        assert not result.empty, "Expected non-empty PET result"

        pet_hot = result.loc[result["location_id"] == 0, "pet"].max()
        pet_cold = result.loc[result["location_id"] == 1, "pet"].max()
        assert pet_hot > pet_cold, (
            f"Hot city PET ({pet_hot}) should exceed cold city PET ({pet_cold}). "
            "This failure typically means the (n_times, n_locs) arrays are being "
            "raveled without transposition, scrambling loc/time alignment."
        )

        assert "wetbulb" in result.columns
        wetbulb_hot = result.loc[result["location_id"] == 0, "wetbulb"].max()
        wetbulb_cold = result.loc[result["location_id"] == 1, "wetbulb"].max()
        assert wetbulb_hot > wetbulb_cold, (
            f"Hot city wetbulb ({wetbulb_hot}) should exceed cold city wetbulb "
            f"({wetbulb_cold})."
        )
