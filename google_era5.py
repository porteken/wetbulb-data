"""Download ARCO ERA5 data and compute daily maximum PET."""

from __future__ import annotations

import argparse
import importlib
import logging
import multiprocessing
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import cache
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from tqdm.auto import tqdm

from partition_io import PartitionTarget, batch_exists, write_batch_partition
from shards import resolve_filesystem

pa = cast("Any", importlib.import_module("pyarrow"))
pd = cast("Any", importlib.import_module("pandas"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))
np = cast("Any", importlib.import_module("numpy"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

type DataFrame = Any
type Dataset = Any
type ArrayLike = Any

GRID_DEG: float = 0.25
DEGREES_PER_CIRCLE: float = 360.0
HALF_CIRCLE_DEGREES: float = 180.0
SAFE_STABLE_DATA_MONTH = 4
PET_ROUNDING_FACTOR = 2.0
CHUNK_SIZE = 50000
ERA5_TIME_ORIGIN = "1900-01-01"
DEFAULT_BATCH_HOURS = 24 * 30
ERA5_THREAD_LOCAL = threading.local()
SECONDS_PER_HOUR = 3600.0
PET_INPUT_COLUMNS = ["v", "t", "rh", "mrt"]

GCS_BATCH_MAX_RETRIES = 3
GCS_BATCH_RETRY_DELAY_SECONDS = 10

MIN_RH = 1.0
MAX_RH = 100.0
SUNLIT_COSSZA_THRESHOLD = 0.10
MAX_REASONABLE_MRT_C = 120.0
MIN_REASONABLE_MRT_C = -80.0
MIN_REASONABLE_PET_C = -50.0
MAX_REASONABLE_PET_C = 60.0
MIN_REASONABLE_WETBULB_C = -50.0
MAX_REASONABLE_WETBULB_C = 40.0

_B = 17.625
_C = 243.04

ERA5_ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ERA5_VARIABLE_CANDIDATES: dict[str, list[str]] = {
    "10u": ["10u", "10m_u_component_of_wind"],
    "10v": ["10v", "10m_v_component_of_wind"],
    "2t": ["2t", "2m_temperature"],
    "2d": ["2d", "2m_dewpoint_temperature"],
    "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "strd": ["strd", "surface_thermal_radiation_downwards"],
    "ssr": ["ssr", "surface_net_solar_radiation"],
    "str": ["str", "surface_net_thermal_radiation"],
    "fdir": [
        "fdir",
        "total_sky_direct_solar_radiation_at_surface",
    ],
}

ERA5_WEATHER_VARIABLES = ["10u", "10v", "2t", "2d"]
ERA5_RADIATION_VARIABLES = ["ssrd", "strd", "ssr", "str", "fdir"]
ERA5_ALL_ARCO_VARIABLES = [*ERA5_WEATHER_VARIABLES, *ERA5_RADIATION_VARIABLES]


class _PetCorrectedCallable(Protocol):
    def __call__(
        self,
        tair: object,
        t_mrt: object,
        v_air: object,
        rh: object,
        *,
        icl: float,
    ) -> ArrayLike: ...


class _ThermofeelModule(Protocol):
    def approximate_dsrp(
        self,
        fdir: ArrayLike,
        cossza: ArrayLike,
        threshold: float = 0.1,
    ) -> ArrayLike: ...

    def calculate_mean_radiant_temperature(
        self,
        ssrd: ArrayLike,
        ssr: ArrayLike,
        dsrp: ArrayLike,
        strd: ArrayLike,
        fdir: ArrayLike,
        strr: ArrayLike,
        cossza: ArrayLike,
    ) -> ArrayLike: ...


try:
    pet_module: Any = importlib.import_module("pet_corrected")
    pet_corrected: _PetCorrectedCallable = cast(
        "_PetCorrectedCallable",
        pet_module.pet_corrected,
    )
except (ImportError, AttributeError) as error:
    LOGGER.exception("Could not import pet_corrected from pet_corrected.py.")
    raise SystemExit(1) from error

try:
    wetbulb_module: Any = importlib.import_module("wetbulb")
    wetbulb_stull: Callable[..., ArrayLike] = cast(
        "Callable[..., ArrayLike]",
        wetbulb_module.wetbulb_stull,
    )
except (ImportError, AttributeError) as error:
    LOGGER.exception("Could not import wetbulb_stull from wetbulb.py.")
    raise SystemExit(1) from error


def _arco_stable_end_year() -> int:
    now = datetime.now(tz=UTC)
    return now.year - 2 if now.month < SAFE_STABLE_DATA_MONTH else now.year - 1


ERA5_START_YEAR = 2000
ERA5_END_YEAR = _arco_stable_end_year()


def _resolve_store_variables(array_names: set[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical_name, candidates in ERA5_VARIABLE_CANDIDATES.items():
        actual_name = next((name for name in candidates if name in array_names), None)
        if actual_name is None:
            missing.append(canonical_name)
            continue
        resolved[canonical_name] = actual_name
    if missing:
        msg = f"Required ARCO variables missing: {missing}"
        raise KeyError(msg)
    return resolved


@cache
def _import_optional_module(module_name: str) -> object:
    """Import an optional dependency at once and reuse the module object."""
    return importlib.import_module(module_name)


def _open_zarr_store() -> Dataset:
    gcsfs = cast("Any", _import_optional_module("gcsfs"))
    xr = cast("Any", _import_optional_module("xarray"))
    dask_array = cast("Any", _import_optional_module("dask.array"))
    zarr = cast("Any", _import_optional_module("zarr"))

    fs = gcsfs.GCSFileSystem(
        token="anon",  # noqa: S106
        default_block_size=8 * 1024 * 1024,
        skip_instance_cache=True,
    )
    root = zarr.open_group(fs.get_mapper(ERA5_ARCO_STORE), mode="r")
    resolved_names = _resolve_store_variables(set(root.array_keys()))

    return xr.Dataset(
        data_vars={
            canonical: (
                ("time", "latitude", "longitude"),
                dask_array.from_zarr(root[actual]),
            )
            for canonical, actual in resolved_names.items()
        },
        coords={
            "time": root["time"][:],
            "latitude": root["latitude"][:],
            "longitude": root["longitude"][:],
        },
    )


def _configure_concurrency(profile: str) -> None:
    zarr = cast("Any", _import_optional_module("zarr"))
    dask = cast("Any", _import_optional_module("dask"))
    configs = {
        "conservative": {"zarr_concurrency": 16},
        "balanced": {"zarr_concurrency": 64},
        "aggressive": {"zarr_concurrency": 128},
    }
    profile_cfg = configs.get(profile, configs["balanced"])
    if hasattr(zarr, "config"):
        zarr.config.set({"async.concurrency": profile_cfg["zarr_concurrency"]})
    dask.config.set({"array.slicing.split_large_chunks": False})


def _resolve_era5_max_workers(requested_max_workers: int) -> int:
    if requested_max_workers == 0:
        msg = "max_workers must be positive or -1 for auto"
        raise ValueError(msg)
    if requested_max_workers == -1:
        return max(1, os.cpu_count() or 1)
    return requested_max_workers


def _year_time_slice(year: int) -> tuple[int, int]:
    epoch = pd.Timestamp(ERA5_TIME_ORIGIN)
    start_h = int((pd.Timestamp(f"{year}-01-01") - epoch).total_seconds() // 3600)
    end_h = int((pd.Timestamp(f"{year + 1}-01-01") - epoch).total_seconds() // 3600) - 1
    return start_h, end_h


def _iter_time_batches(
    year: int,
    batch_hours: int,
    allowed_months: list[int] | None = None,
) -> list[tuple[int, int, int]]:
    if batch_hours <= 0:
        msg = f"batch_hours must be positive, got {batch_hours}"
        raise ValueError(msg)

    start_h, end_h = _year_time_slice(year)
    batches: list[tuple[int, int, int]] = []
    batch_index = 0
    batch_start = start_h

    while batch_start <= end_h:
        batch_end = min(batch_start + batch_hours - 1, end_h)
        batch_start_ts = pd.to_datetime(batch_start, unit="h", origin=ERA5_TIME_ORIGIN)
        batch_end_ts = pd.to_datetime(batch_end, unit="h", origin=ERA5_TIME_ORIGIN)

        if (
            not allowed_months
            or batch_start_ts.month in allowed_months
            or batch_end_ts.month in allowed_months
        ):
            batches.append((batch_index, batch_start, batch_end))

        batch_index += 1
        batch_start = batch_end + 1

    return batches


def _select_time_shard_batches(
    batches: list[tuple[int, int, int]],
    time_shard_index: int,
    time_shard_count: int,
) -> list[tuple[int, int, int]]:
    if time_shard_index >= time_shard_count:
        msg = f"time_shard_index {time_shard_index} must be less than time_shard_count {time_shard_count}"
        raise ValueError(
            msg,
        )
    return [b for b in batches if b[0] % time_shard_count == time_shard_index]


def _load_era5_city_shard(city_shard_index: int, city_shard_count: int) -> DataFrame:
    cities_df = pd.read_csv("cities.csv", usecols=["location_id", "lat", "lng"])
    cities_df["lat"] = (pd.to_numeric(cities_df["lat"]) / GRID_DEG).round() * GRID_DEG
    cities_df["lng"] = (pd.to_numeric(cities_df["lng"]) / GRID_DEG).round() * GRID_DEG
    cities_df = cities_df.sort_values("location_id").reset_index(drop=True)

    shard_size = (len(cities_df) + city_shard_count - 1) // city_shard_count
    start = city_shard_index * shard_size
    end = min(start + shard_size, len(cities_df))
    return cities_df.iloc[start:end].copy()


def _wrap_longitudes_for_arco_selection(longitudes: ArrayLike) -> ArrayLike:
    return np.mod(np.asarray(longitudes, dtype="float64"), DEGREES_PER_CIRCLE)


def _normalize_longitudes_for_solar_geometry(longitudes: ArrayLike) -> ArrayLike:
    normalized = np.asarray(longitudes, dtype="float64")
    return np.where(
        normalized > HALF_CIRCLE_DEGREES,
        normalized - DEGREES_PER_CIRCLE,
        normalized,
    )


def _calc_cossza(lats: ArrayLike, lons: ArrayLike, times: ArrayLike) -> ArrayLike:
    """Vectorized solar zenith angle for all (locations x times) in one broadcast."""
    lats_2d = np.asarray(lats, dtype="float64").reshape(-1, 1)
    lons_2d = np.asarray(lons, dtype="float64").reshape(-1, 1)

    timestamps = pd.to_datetime(times)
    day_of_year = timestamps.dayofyear.to_numpy(dtype="float64").reshape(1, -1)
    hour = (
        timestamps.hour.to_numpy(dtype="float64")
        + timestamps.minute.to_numpy(dtype="float64") / 60.0
    ).reshape(1, -1)

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    eq_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )

    tst_min = hour * 60.0 + eq_time + 4.0 * lons_2d
    ha = np.deg2rad(tst_min / 4.0 - 180.0)
    lat_r = np.deg2rad(lats_2d)

    cossza = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    return np.clip(cossza, 0.0, 1.0)


def _hourly_flux_from_accumulation(accumulated_radiation: ArrayLike) -> ArrayLike:
    return np.asarray(accumulated_radiation, dtype="float64") / SECONDS_PER_HOUR


def _approximate_dsrp(
    thermofeel_module: _ThermofeelModule,
    fdir_flux: ArrayLike,
    cossza: ArrayLike,
    _ssrd_flux: ArrayLike,
) -> ArrayLike:
    dsrp = np.asarray(
        thermofeel_module.approximate_dsrp(
            fdir_flux,
            cossza,
            threshold=SUNLIT_COSSZA_THRESHOLD,
        ),
        dtype="float64",
    )
    dsrp = np.where(cossza >= SUNLIT_COSSZA_THRESHOLD, dsrp, 0.0)

    return np.clip(dsrp, 0.0, None)


def _compute_pet_chunk(df_chunk: DataFrame) -> DataFrame:
    t_vals = df_chunk["t"].to_numpy(dtype="float32")
    mrt_vals = df_chunk["mrt"].to_numpy(dtype="float32")
    v_vals = df_chunk["v"].to_numpy(dtype="float32")
    rh_vals = df_chunk["rh"].to_numpy(dtype="float32")

    pet_results = np.asarray(
        pet_corrected(t_vals, mrt_vals, v_vals, rh_vals, icl=0.5),
        dtype="float32",
    )
    invalid_pet = (
        ~np.isfinite(pet_results)
        | (pet_results < MIN_REASONABLE_PET_C)
        | (pet_results > MAX_REASONABLE_PET_C)
    )
    if np.any(invalid_pet):
        LOGGER.warning(
            "Discarding %s PET value(s) outside sanity range [%s, %s] C.",
            int(np.count_nonzero(invalid_pet)),
            MIN_REASONABLE_PET_C,
            MAX_REASONABLE_PET_C,
        )
        pet_results[invalid_pet] = np.nan

    df_result = df_chunk.copy()
    df_result["pet"] = (pet_results * PET_ROUNDING_FACTOR).round() / PET_ROUNDING_FACTOR
    return df_result.dropna(subset=["pet"])


def _compute_wetbulb_series(temp_c: ArrayLike, rh: ArrayLike) -> ArrayLike:
    wetbulb_results = np.asarray(
        wetbulb_stull(
            np.asarray(temp_c, dtype="float32"), np.asarray(rh, dtype="float32")
        ),
        dtype="float32",
    )
    invalid_wetbulb = (
        ~np.isfinite(wetbulb_results)
        | (wetbulb_results < MIN_REASONABLE_WETBULB_C)
        | (wetbulb_results > MAX_REASONABLE_WETBULB_C)
    )
    if np.any(invalid_wetbulb):
        LOGGER.warning(
            "Discarding %s wetbulb value(s) outside sanity range [%s, %s] C.",
            int(np.count_nonzero(invalid_wetbulb)),
            MIN_REASONABLE_WETBULB_C,
            MAX_REASONABLE_WETBULB_C,
        )
        wetbulb_results[invalid_wetbulb] = np.nan

    return (wetbulb_results * PET_ROUNDING_FACTOR).round() / PET_ROUNDING_FACTOR


def _filter_frame_to_months(
    frame: DataFrame,
    allowed_months: list[int] | None,
) -> DataFrame:
    if not allowed_months or frame.empty:
        return frame
    allowed_month_set = set(allowed_months)
    return frame[pd.to_datetime(frame["date"]).dt.month.isin(allowed_month_set)].copy()


def _compute_location_frame(
    ds: Dataset,
    selected_cities: DataFrame,
    compute_workers: int,
) -> DataFrame:
    dask = cast("Any", _import_optional_module("dask"))
    xr = cast("Any", _import_optional_module("xarray"))
    tf = cast("_ThermofeelModule", _import_optional_module("thermofeel"))

    lats = selected_cities["lat"].values.astype(float)
    selection_lons = np.asarray(
        _wrap_longitudes_for_arco_selection(selected_cities["lng"].values),
        dtype="float64",
    )
    solar_lons = np.asarray(
        _normalize_longitudes_for_solar_geometry(selection_lons),
        dtype="float64",
    )
    loc_ids = selected_cities["location_id"].values

    city_selection = ds[ERA5_ALL_ARCO_VARIABLES].sel(
        latitude=xr.DataArray(lats, dims="location"),
        longitude=xr.DataArray(selection_lons, dims="location"),
        method="nearest",
    )

    # Materialize every variable in one parallel pass: computing here (instead
    # of at each later `.values` access) downloads all nine variables from GCS
    # concurrently with the requested worker count.
    with dask.config.set(scheduler="threads", num_workers=compute_workers):
        city_data = city_selection.compute()

    city_data = city_data.assign_coords(
        time=pd.to_datetime(city_data.time.values, unit="h", origin=ERA5_TIME_ORIGIN),
    )

    u10 = city_data["10u"].values.T
    v10 = city_data["10v"].values.T
    t2m = city_data["2t"].values.T
    d2m = city_data["2d"].values.T

    ssrd = _hourly_flux_from_accumulation(city_data["ssrd"].values.T)
    strd = _hourly_flux_from_accumulation(city_data["strd"].values.T)
    ssr = _hourly_flux_from_accumulation(city_data["ssr"].values.T)
    strr = _hourly_flux_from_accumulation(city_data["str"].values.T)
    fdir = _hourly_flux_from_accumulation(city_data["fdir"].values.T)

    times = city_data.time.values
    n_locs, n_times = len(lats), len(times)

    cossza_times = pd.to_datetime(times) - pd.Timedelta(minutes=30)
    cossza = _calc_cossza(lats, solar_lons, cossza_times.values)

    temp_c = t2m - 273.15
    dew_c = d2m - 273.15
    wind_speed = np.sqrt(u10**2 + v10**2)

    gamma_t = _B * temp_c / (_C + temp_c)
    gamma_td = _B * dew_c / (_C + dew_c)
    rh = np.exp(gamma_td - gamma_t) * 100.0
    rh = np.clip(rh, 0.0, 100.0)

    dsrp = _approximate_dsrp(tf, fdir, cossza, ssrd)

    mrt_k = tf.calculate_mean_radiant_temperature(
        ssrd=ssrd.ravel(),
        ssr=ssr.ravel(),
        dsrp=dsrp.ravel(),
        strd=strd.ravel(),
        fdir=fdir.ravel(),
        strr=strr.ravel(),
        cossza=cossza.ravel(),
    )
    mrt_c = (mrt_k - 273.15).reshape(n_locs, n_times)
    invalid_mrt = (
        (~np.isfinite(mrt_c))
        | (mrt_c < MIN_REASONABLE_MRT_C)
        | (mrt_c > MAX_REASONABLE_MRT_C)
    )
    if np.any(invalid_mrt):
        LOGGER.warning(
            "Discarding %s MRT value(s) outside sanity range [%s, %s] C.",
            int(np.count_nonzero(invalid_mrt)),
            MIN_REASONABLE_MRT_C,
            MAX_REASONABLE_MRT_C,
        )
        mrt_c = mrt_c.astype("float64", copy=False)
        mrt_c[invalid_mrt] = np.nan

    frame = pd.DataFrame(
        {
            "location_id": np.repeat(loc_ids, n_times),
            "time": np.tile(times, n_locs),
            "t": temp_c.ravel(),
            "v": wind_speed.ravel(),
            "rh": rh.ravel(),
            "mrt": mrt_c.ravel(),
        },
    ).dropna()

    df = frame[
        (frame["rh"] >= MIN_RH) & (frame["rh"] <= MAX_RH) & (frame["v"] > 0.0)
    ].copy()

    df.loc[:, PET_INPUT_COLUMNS] = (
        df[PET_INPUT_COLUMNS] * PET_ROUNDING_FACTOR
    ).round() / PET_ROUNDING_FACTOR
    df["wetbulb"] = _compute_wetbulb_series(df["t"].to_numpy(), df["rh"].to_numpy())
    df["_pet_key"] = pd.factorize(
        pd.MultiIndex.from_frame(df[PET_INPUT_COLUMNS]),
        sort=False,
    )[0]

    df_distinct = df.loc[
        ~df["_pet_key"].duplicated(),
        ["_pet_key", *PET_INPUT_COLUMNS],
    ].reset_index(drop=True)
    chunks = [
        df_distinct.iloc[i : i + CHUNK_SIZE]
        for i in range(0, len(df_distinct), CHUNK_SIZE)
    ]

    if not chunks:
        return pd.DataFrame(columns=["location_id", "date", "pet", "wetbulb"])

    results: list[DataFrame] = []
    n_chunk_workers = max(1, min(compute_workers, len(chunks)))
    if n_chunk_workers == 1:
        results = [_compute_pet_chunk(chunk) for chunk in chunks]
    else:
        with ThreadPoolExecutor(max_workers=n_chunk_workers) as chunk_executor:
            results = list(chunk_executor.map(_compute_pet_chunk, chunks))

    if not results:
        return pd.DataFrame(columns=["location_id", "date", "pet", "wetbulb"])

    df_pet_unique = pd.concat(results, ignore_index=True)
    df = df.merge(df_pet_unique[["_pet_key", "pet"]], on="_pet_key", how="inner")
    df = df.drop(columns="_pet_key")
    df["date"] = pd.to_datetime(df["time"]).dt.date

    return df.groupby(["location_id", "date"], as_index=False).agg(
        pet=("pet", "max"),
        wetbulb=("wetbulb", "max"),
    )


class _ProductPlan(NamedTuple):
    """Which data products to write and where their parquet trees live."""

    pet_root: str
    wetbulb_root: str
    write_pet: bool = True
    write_wetbulb: bool = True


class _BatchWriteTargets(NamedTuple):
    """Where to write each product and whether it should be written."""

    pet_target: PartitionTarget
    wetbulb_target: PartitionTarget
    write_pet: bool = True
    write_wetbulb: bool = True


def _write_pet_partition(
    pet_root: str,
    year: int,
    city_shard_index: int,
    frame: DataFrame,
    batch_index: int,
    *,
    filesystem: object | None = None,
    base_path: str | None = None,
) -> None:
    write_batch_partition(
        pet_root,
        year,
        city_shard_index,
        frame,
        batch_index,
        file_prefix="pet",
        filesystem=filesystem,
        base_path=base_path,
    )


def _write_wetbulb_partition(
    wetbulb_root: str,
    year: int,
    city_shard_index: int,
    frame: DataFrame,
    batch_index: int,
    *,
    filesystem: object | None = None,
    base_path: str | None = None,
) -> None:
    write_batch_partition(
        wetbulb_root,
        year,
        city_shard_index,
        frame,
        batch_index,
        file_prefix="wetbulb",
        filesystem=filesystem,
        base_path=base_path,
    )


def _process_era5_batch_job(
    *,
    ds: Dataset,
    targets: _BatchWriteTargets,
    year: int,
    city_shard_index: int,
    batch_index: int,
    start_h: int,
    end_h: int,
    pending_batch_df: DataFrame,
    compute_workers: int,
    allowed_months: list[int] | None = None,
) -> int:
    LOGGER.info(
        "Computing ERA5->PET for year %s batch %s over %s cities.",
        year,
        batch_index,
        len(pending_batch_df),
    )

    frame = _compute_location_frame(
        ds.sel(time=slice(start_h, end_h)),
        pending_batch_df,
        compute_workers,
    )
    frame = _filter_frame_to_months(frame, allowed_months)

    if frame.empty:
        return batch_index

    if targets.write_pet:
        _write_pet_partition(
            targets.pet_target.root,
            year,
            city_shard_index,
            frame[["location_id", "date", "pet"]].copy(),
            batch_index,
            filesystem=targets.pet_target.filesystem,
            base_path=targets.pet_target.base_path,
        )
    if targets.write_wetbulb:
        wetbulb_frame = (
            frame[["location_id", "date", "wetbulb"]]
            .dropna(
                subset=["wetbulb"],
            )
            .copy()
        )
        if not wetbulb_frame.empty:
            _write_wetbulb_partition(
                targets.wetbulb_target.root,
                year,
                city_shard_index,
                wetbulb_frame,
                batch_index,
                filesystem=targets.wetbulb_target.filesystem,
                base_path=targets.wetbulb_target.base_path,
            )
    return batch_index


def _process_era5_batch_with_thread_dataset(
    *,
    targets: _BatchWriteTargets,
    year: int,
    city_shard_index: int,
    batch_index: int,
    start_h: int,
    end_h: int,
    pending_batch_df: DataFrame,
    compute_workers: int,
    allowed_months: list[int] | None = None,
) -> int:
    for attempt in range(1, GCS_BATCH_MAX_RETRIES + 1):
        try:
            ds = getattr(ERA5_THREAD_LOCAL, "ds", None)
            if ds is None:
                _configure_concurrency(
                    os.environ.get("ERA5_CONCURRENCY_PROFILE", "balanced"),
                )
                ds = ERA5_THREAD_LOCAL.ds = _open_zarr_store()

            return _process_era5_batch_job(
                ds=ds,
                targets=targets,
                year=year,
                city_shard_index=city_shard_index,
                batch_index=batch_index,
                start_h=start_h,
                end_h=end_h,
                pending_batch_df=pending_batch_df,
                compute_workers=compute_workers,
                allowed_months=allowed_months,
            )
        except OSError:
            if attempt == GCS_BATCH_MAX_RETRIES:
                raise

            jitter = random.uniform(0.0, 2.0)  # noqa: S311
            time.sleep(GCS_BATCH_RETRY_DELAY_SECONDS * attempt + jitter)

    return batch_index


PRODUCT_CHOICES = ("pet", "wetbulb", "both")


def _resolve_products(products: str) -> tuple[bool, bool]:
    """Map a --products choice to (write_pet, write_wetbulb) flags."""
    if products not in PRODUCT_CHOICES:
        msg = f"products must be one of {PRODUCT_CHOICES}, got {products!r}"
        raise ValueError(msg)
    return products in ("pet", "both"), products in ("wetbulb", "both")


def _collect_pending_batch_jobs(
    *,
    selected_batches: list[tuple[int, int, int]],
    shard_df: DataFrame,
    targets: _BatchWriteTargets,
    year: int,
    city_shard_index: int,
    force: bool,
) -> list[tuple[int, int, int, DataFrame]]:
    batch_jobs: list[tuple[int, int, int, DataFrame]] = []
    for batch_index, start_h, end_h in selected_batches:
        required_exist = []
        if targets.write_pet:
            required_exist.append(
                batch_exists(
                    targets.pet_target.root,
                    year,
                    city_shard_index,
                    batch_index,
                    filesystem=targets.pet_target.filesystem,
                    base_path=targets.pet_target.base_path,
                )
            )
        if targets.write_wetbulb:
            required_exist.append(
                batch_exists(
                    targets.wetbulb_target.root,
                    year,
                    city_shard_index,
                    batch_index,
                    file_prefix="wetbulb",
                    filesystem=targets.wetbulb_target.filesystem,
                    base_path=targets.wetbulb_target.base_path,
                )
            )
        if force or not all(required_exist):
            batch_jobs.append((batch_index, start_h, end_h, shard_df))
    return batch_jobs


def _run_era5_batch_jobs(
    *,
    selected_batches: list[tuple[int, int, int]],
    shard_df: DataFrame,
    plan: _ProductPlan,
    year: int,
    city_shard_index: int,
    city_shard_count: int,
    time_shard_index: int,
    time_shard_count: int,
    max_workers: int,
    allowed_months: list[int] | None = None,
    force: bool = False,
) -> bool:
    era5_root = plan.pet_root
    wetbulb_root = plan.wetbulb_root
    write_pet = plan.write_pet
    write_wetbulb = plan.write_wetbulb
    concurrency_profile = os.environ.get("ERA5_CONCURRENCY_PROFILE", "balanced")
    if concurrency_profile == "aggressive":
        dask_workers = 8
    elif concurrency_profile == "conservative":
        dask_workers = 2
    else:
        dask_workers = 4

    filesystem, base_path = resolve_filesystem(era5_root)
    wetbulb_filesystem, wetbulb_base_path = resolve_filesystem(wetbulb_root)
    pet_target = PartitionTarget(era5_root, filesystem, base_path)
    wetbulb_target = PartitionTarget(
        wetbulb_root, wetbulb_filesystem, wetbulb_base_path
    )
    targets = _BatchWriteTargets(pet_target, wetbulb_target, write_pet, write_wetbulb)
    batch_jobs = _collect_pending_batch_jobs(
        selected_batches=selected_batches,
        shard_df=shard_df,
        targets=targets,
        year=year,
        city_shard_index=city_shard_index,
        force=force,
    )

    if not batch_jobs:
        return False

    LOGGER.info(
        "ERA5->PET year=%d city_shard=%d/%d time_shard=%d/%d: %d batch(es)",
        year,
        city_shard_index,
        city_shard_count,
        time_shard_index,
        time_shard_count,
        len(batch_jobs),
    )

    worker_count = max(1, min(_resolve_era5_max_workers(max_workers), len(batch_jobs)))

    if worker_count == 1:
        ds = _open_zarr_store()
        for batch_index, _start_h, _end_h, pending_df in tqdm(
            batch_jobs,
            desc=f"ERA5->PET {year}",
        ):
            _process_era5_batch_job(
                ds=ds,
                targets=targets,
                year=year,
                city_shard_index=city_shard_index,
                batch_index=batch_index,
                start_h=_start_h,
                end_h=_end_h,
                pending_batch_df=pending_df,
                compute_workers=dask_workers,
                allowed_months=allowed_months,
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_era5_batch_with_thread_dataset,
                    targets=targets,
                    year=year,
                    city_shard_index=city_shard_index,
                    batch_index=batch_index,
                    start_h=start_h,
                    end_h=end_h,
                    pending_batch_df=pending_df,
                    compute_workers=dask_workers,
                    allowed_months=allowed_months,
                )
                for batch_index, start_h, end_h, pending_df in batch_jobs
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"ERA5->PET {year}",
            ):
                future.result()

    return True


def process_era5(
    year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    time_shard_index: int,
    time_shard_count: int,
    max_workers: int,
    batch_hours: int,
    concurrency_profile: str,
    months: list[int] | None = None,
    products: str = "both",
) -> None:
    """Download ERA5 data from ARCO, compute PET and wetbulb, and save as parquet shards."""
    write_pet, write_wetbulb = _resolve_products(products)
    _configure_concurrency(concurrency_profile)
    os.environ["ERA5_CONCURRENCY_PROFILE"] = concurrency_profile
    pet_root = f"{out_dir}/pet_data_csv"
    wetbulb_root = f"{out_dir}/wetbulb_data_csv"

    shard_df = _load_era5_city_shard(city_shard_index, city_shard_count)
    if shard_df.empty:
        LOGGER.info(
            "No cities found for shard %s/%s.",
            city_shard_index,
            city_shard_count,
        )
        return

    selected_batches = _select_time_shard_batches(
        _iter_time_batches(year, batch_hours, months),
        time_shard_index,
        time_shard_count,
    )

    _run_era5_batch_jobs(
        selected_batches=selected_batches,
        shard_df=shard_df,
        plan=_ProductPlan(pet_root, wetbulb_root, write_pet, write_wetbulb),
        year=year,
        city_shard_index=city_shard_index,
        city_shard_count=city_shard_count,
        time_shard_index=time_shard_index,
        time_shard_count=time_shard_count,
        max_workers=max_workers,
        allowed_months=months,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        help="Only process specific months (1-12)",
    )
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--city-shard-index", type=int, default=0)
    parser.add_argument("--city-shard-count", type=int, default=1)
    parser.add_argument(
        "--time-shard-index",
        type=int,
        default=None,
        help=(
            "Time shard to process. Defaults to the CLOUD_RUN_TASK_INDEX "
            "environment variable (0 outside Cloud Run), so a single Cloud "
            "Run execution with --tasks=N fans out across time shards."
        ),
    )
    parser.add_argument("--time-shard-count", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=-1)
    parser.add_argument("--batch-hours", type=int, default=DEFAULT_BATCH_HOURS)
    parser.add_argument(
        "--concurrency-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="aggressive",
    )
    parser.add_argument(
        "--products",
        choices=list(PRODUCT_CHOICES),
        default="both",
        help="Which parquet trees to write: pet, wetbulb, or both (default).",
    )
    args = parser.parse_args()
    if args.time_shard_index is None:
        args.time_shard_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    return args


def main() -> None:
    """Execute the ERA5 processing pipeline."""
    exit_code = 0
    try:
        args = _parse_args()
        process_era5(
            year=args.year,
            out_dir=args.out_dir,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
            time_shard_index=args.time_shard_index,
            time_shard_count=args.time_shard_count,
            max_workers=args.max_workers,
            batch_hours=args.batch_hours,
            concurrency_profile=args.concurrency_profile,
            months=args.months,
            products=args.products,
        )
    except KeyboardInterrupt:
        exit_code = 130
        LOGGER.warning("ERA5 processing interrupted by user.")
    except SystemExit:
        raise
    except (
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
        ImportError,
        AttributeError,
        TypeError,
        IndexError,
    ):
        exit_code = 1
        LOGGER.exception("ERA5 processing failed.")

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
