# Copyright (C) 2026 Kenneth Porter

"""Fetch NOAA GHCNh station observations and compute daily wet-bulb temperature."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

import nldas
from lcd import (
    _load_pending_shard,
    _station_to_hourly,
    add_common_shard_args,
    concat_frames,
    derive_station_pressure,
)
from partition_io import pending_years, write_pending_year_batches
from shards import resolve_filesystem
from station_exclusions import DISALLOWED_GHCNH_STATION_IDS

if TYPE_CHECKING:
    from collections.abc import Sequence

pd = cast("Any", importlib.import_module("pandas"))
pa = cast("Any", importlib.import_module("pyarrow"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))
_PARQUET_PARSE_ERRORS = (OSError, pa.ArrowException)

type DataFrame = Any
type CandidateRow = Any
type CandidateKey = tuple[str, float | None, float | None]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
_RNG = random.SystemRandom()

GHCNH_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
    "hourly/access/by-year/{year}/parquet/GHCNh_{station_id}_{year}.parquet"
)
GHCNH_REQUEST_TIMEOUT_SECONDS = 90
GHCNH_MAX_RETRIES = 3
GHCNH_RETRY_DELAY_SECONDS = 5
GHCNH_DEFAULT_CONCURRENCY = 8
GHCNH_CACHE_DIRECTORY = "ghcnh_cache"
GHCNH_MIN_DAILY_HOURS = 20
GHCNH_RELIABLE_DAILY_HOURS = 20
GHCNH_REFERENCE_LOOKBACK_YEARS = 10
GHCNH_MIN_MONTHLY_OVERLAP_DAYS = 20
GHCNH_MIN_GLOBAL_OVERLAP_DAYS = 60
STATION_MAP_PATH = "cities_na_ghcnh_stations.csv"

_HOURLY_FRAME_COLUMNS = ("time", "tair_c", "dewpoint_c", "pressure_hpa")
_PARQUET_COLUMNS = (
    "DATE",
    "ELEVATION",
    "temperature",
    "temperature_Quality_Code",
    "temperature_Report_Type",
    "dew_point_temperature",
    "dew_point_temperature_Quality_Code",
    "station_level_pressure",
    "station_level_pressure_Quality_Code",
    "sea_level_pressure",
    "sea_level_pressure_Quality_Code",
    "altimeter",
    "altimeter_Quality_Code",
)
_REPORT_PRIORITY = {
    "FM15": 0,
    "FM16": 1,
    "SAO": 2,
    "SAOSP": 3,
    "SSA": 4,
    "SAAU": 5,
    "SYMT": 6,
    "SYSA": 7,
    "SYAU": 8,
    "SYAE": 9,
    "FM12": 10,
    "FM94_1": 11,
    "AUTO": 12,
    "MESOH": 13,
    "MESOS": 14,
    "AUST": 15,
    "BRAZ": 16,
    "GREEN": 17,
    "MEXIC": 18,
    "SMARS": 19,
    "WBO": 20,
    "WNO": 21,
    "EnvCan": 22,
}
_REPORT_TYPES = frozenset(_REPORT_PRIORITY)
_REJECT_QC_CODES = frozenset({"2", "3", "6", "7"})
_MIN_AIR_TEMPERATURE_C = -80.0
_MAX_AIR_TEMPERATURE_C = 60.0
_MIN_DEWPOINT_C = -100.0
_MAX_DEWPOINT_C = 40.0
_MAX_DEWPOINT_ABOVE_AIR_C = 1.0
_DAILY_SPIKE_NEIGHBOR_DAYS = 3
_DAILY_SPIKE_MIN_NEIGHBORS = 2
_DAILY_SPIKE_NEIGHBOR_DELTA_C = 8.0
_DAILY_SPIKE_MAX_MEAN_GAP_C = 10.0
_DAILY_SPIKE_EVENT_SUPPORT_C = 3.0


def _empty_hourly_frame() -> DataFrame:
    return pd.DataFrame(columns=list(_HOURLY_FRAME_COLUMNS))


def _quality_filtered(raw: DataFrame, value: str) -> DataFrame:
    values = pd.to_numeric(raw[value], errors="coerce")
    quality = raw[f"{value}_Quality_Code"].astype("string").str.strip()
    return values.where(~quality.isin(_REJECT_QC_CODES))


def _physical_quality_filtered(hourly: DataFrame) -> DataFrame:
    """Discard impossible thermodynamic inputs missed by upstream QC."""
    valid = (
        hourly["tair_c"].between(
            _MIN_AIR_TEMPERATURE_C,
            _MAX_AIR_TEMPERATURE_C,
        )
        & hourly["dewpoint_c"].between(_MIN_DEWPOINT_C, _MAX_DEWPOINT_C)
        & (hourly["dewpoint_c"] <= hourly["tair_c"] + _MAX_DEWPOINT_ABOVE_AIR_C)
    )
    return hourly[valid].copy()


def filter_daily_wetbulb_spikes(daily: DataFrame) -> DataFrame:
    """Remove isolated daily maxima unsupported by hourly or nearby-day data."""
    if daily.empty:
        return daily.copy()
    result = daily.copy()
    result["_date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["location_id", "_date"])
    keep = pd.Series(data=True, index=result.index)

    for _location_id, group in result.groupby("location_id", sort=False):
        dates = group["_date"]
        maxima = pd.to_numeric(group["wetbulb"], errors="coerce")
        means = pd.to_numeric(group["wetbulb_avg"], errors="coerce")
        neighbors: list[DataFrame] = []
        adjacent: list[DataFrame] = []
        for offset in range(1, _DAILY_SPIKE_NEIGHBOR_DAYS + 1):
            previous = maxima.shift(offset).where(
                (dates - dates.shift(offset)).dt.days <= _DAILY_SPIKE_NEIGHBOR_DAYS,
            )
            following = maxima.shift(-offset).where(
                (dates.shift(-offset) - dates).dt.days <= _DAILY_SPIKE_NEIGHBOR_DAYS,
            )
            neighbors.extend([previous, following])
            if offset == 1:
                adjacent.extend(
                    [
                        previous.where((dates - dates.shift()).dt.days == 1),
                        following.where((dates.shift(-1) - dates).dt.days == 1),
                    ],
                )

        neighbor_frame = pd.concat(neighbors, axis="columns")
        adjacent_frame = pd.concat(adjacent, axis="columns")
        enough_neighbors = (
            neighbor_frame.notna().sum(axis="columns") >= _DAILY_SPIKE_MIN_NEIGHBORS
        )
        neighbor_median = neighbor_frame.median(axis="columns")
        isolated_jump = maxima - neighbor_median >= _DAILY_SPIKE_NEIGHBOR_DELTA_C
        unsupported_maximum = maxima - means >= _DAILY_SPIKE_MAX_MEAN_GAP_C
        event_supported = adjacent_frame.ge(
            maxima - _DAILY_SPIKE_EVENT_SUPPORT_C,
            axis="index",
        ).any(axis="columns")
        keep.loc[group.index] = ~(
            enough_neighbors & isolated_jump & unsupported_maximum & ~event_supported
        )

    return result[keep].drop(columns="_date").sort_index()


def _drop_current_local_day(
    frame: DataFrame,
    *,
    utc_offset_hours: float,
    now_utc: datetime | None = None,
) -> DataFrame:
    """Exclude the current local day while retaining valid synoptic-only days."""
    if frame.empty:
        return frame
    timestamps = pd.to_datetime(frame["time"])
    reference = now_utc or datetime.now(tz=UTC)
    local_today = (
        pd.Timestamp(reference) + pd.to_timedelta(utc_offset_hours, unit="h")
    ).date()
    return frame[timestamps.dt.date < local_today].reset_index(drop=True)


def _parse_ghcnh_parquet(
    payload: bytes,
    *,
    lon: float | None,
    utc_offset_hours: float | None,
    drop_incomplete_latest_day: bool,
) -> DataFrame:
    try:
        raw = pq.read_table(
            pa.BufferReader(payload),
            columns=list(_PARQUET_COLUMNS),
        ).to_pandas()
    except _PARQUET_PARSE_ERRORS:
        return _empty_hourly_frame()

    report_type = raw["temperature_Report_Type"].astype("string").str.strip()
    hourly = raw[report_type.isin(_REPORT_TYPES)].copy()
    if hourly.empty:
        return _empty_hourly_frame()
    hourly["REPORT_TYPE"] = report_type.loc[hourly.index]
    hourly["tair_c"] = _quality_filtered(hourly, "temperature")
    hourly["dewpoint_c"] = _quality_filtered(hourly, "dew_point_temperature")
    hourly["station_pressure_hpa"] = _quality_filtered(
        hourly,
        "station_level_pressure",
    )
    hourly["slp_hpa"] = _quality_filtered(hourly, "sea_level_pressure")
    hourly["altimeter_hpa"] = _quality_filtered(hourly, "altimeter")
    hourly["time_utc"] = pd.to_datetime(
        hourly["DATE"],
        format="ISO8601",
        errors="coerce",
    )
    hourly["ELEVATION"] = pd.to_numeric(hourly["ELEVATION"], errors="coerce")
    hourly = hourly.dropna(subset=["time_utc", "tair_c", "dewpoint_c"])
    hourly = _physical_quality_filtered(hourly)
    if hourly.empty:
        return _empty_hourly_frame()

    hourly["_priority"] = hourly["REPORT_TYPE"].map(_REPORT_PRIORITY)
    hourly = hourly.sort_values(["time_utc", "_priority"]).drop_duplicates(
        subset="time_utc",
        keep="first",
    )
    pressure_hpa = derive_station_pressure(hourly)

    if utc_offset_hours is None or pd.isna(utc_offset_hours):
        utc_offset_hours = round(float(lon or 0.0) / 15.0)
    local_time = hourly["time_utc"] + pd.to_timedelta(
        float(utc_offset_hours),
        unit="h",
    )
    parsed = pd.DataFrame(
        {
            "time": local_time.to_numpy(),
            "tair_c": hourly["tair_c"].to_numpy(dtype="float64"),
            "dewpoint_c": hourly["dewpoint_c"].to_numpy(dtype="float64"),
            "pressure_hpa": pressure_hpa.to_numpy(dtype="float64"),
        },
    ).dropna(subset=["pressure_hpa"])
    if drop_incomplete_latest_day:
        parsed = _drop_current_local_day(
            parsed,
            utc_offset_hours=float(utc_offset_hours),
        )
    return parsed.reset_index(drop=True)


def _get_with_retries(
    session: requests.Session,
    url: str,
    *,
    station_id: str,
    year: int,
) -> requests.Response | None:
    for attempt in range(1, GHCNH_MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=GHCNH_REQUEST_TIMEOUT_SECONDS)
            if response.status_code == requests.codes.not_found:
                return response
            response.raise_for_status()
        except requests.RequestException:
            if attempt == GHCNH_MAX_RETRIES:
                LOGGER.warning(
                    "Giving up on station=%s year=%d after %d attempt(s).",
                    station_id,
                    year,
                    attempt,
                )
                return None
            time.sleep(GHCNH_RETRY_DELAY_SECONDS * attempt + _RNG.uniform(0.0, 2.0))
        else:
            return response
    return None


def fetch_station_year(
    station_id: str,
    year: int,
    *,
    lon: float | None,
    utc_offset_hours: float | None,
    drop_incomplete_latest_day: bool,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
) -> tuple[DataFrame, bool]:
    """Return one station-year frame and whether a transient fetch gap occurred."""
    cache_path = None
    missing_path = None
    if cache_dir is not None and year < datetime.now(tz=UTC).year:
        cache_path = cache_dir / str(year) / f"{station_id}.parquet"
        missing_path = cache_path.with_suffix(".missing")
        if cache_path.exists():
            return (
                _parse_ghcnh_parquet(
                    cache_path.read_bytes(),
                    lon=lon,
                    utc_offset_hours=utc_offset_hours,
                    drop_incomplete_latest_day=drop_incomplete_latest_day,
                ),
                False,
            )
        if missing_path.exists():
            return _empty_hourly_frame(), False

    http = session or requests.Session()
    url = GHCNH_URL_TEMPLATE.format(year=year, station_id=station_id)
    response = _get_with_retries(http, url, station_id=station_id, year=year)
    if response is None:
        return _empty_hourly_frame(), True
    if response.status_code == requests.codes.not_found:
        if missing_path is not None:
            _write_cache_file(missing_path, b"")
        return _empty_hourly_frame(), False
    if cache_path is not None:
        _write_cache_file(cache_path, response.content)
    return (
        _parse_ghcnh_parquet(
            response.content,
            lon=lon,
            utc_offset_hours=utc_offset_hours,
            drop_incomplete_latest_day=drop_incomplete_latest_day,
        ),
        False,
    )


def _write_cache_file(path: Path, payload: bytes) -> None:
    """Atomically publish one cache entry, tolerating concurrent workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temp_path.write_bytes(payload)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_station_map(path: str) -> DataFrame:
    map_path = Path(path)
    if not map_path.exists():
        LOGGER.warning("%s not found; no cities can be mapped to GHCNh.", path)
        return pd.DataFrame(
            columns=["location_id", "ghcn_id", "lon", "utc_offset_hours"],
        )
    available = pd.read_csv(map_path, nrows=0).columns
    required = ["location_id", "ghcn_id", "lon", "utc_offset_hours"]
    optional = [
        "candidate_rank",
        "variable_coverage",
        "dist_km",
        "elevation_difference_m",
        "year",
        "start_year",
        "end_year",
    ]
    station_map = pd.read_csv(
        map_path,
        usecols=[column for column in [*required, *optional] if column in available],
    )
    station_map = station_map[
        ~station_map["ghcn_id"].isin(DISALLOWED_GHCNH_STATION_IDS)
    ].copy()
    if "candidate_rank" not in station_map:
        station_map["candidate_rank"] = (
            station_map.groupby("location_id").cumcount() + 1
        )
    for column in ("variable_coverage", "dist_km", "elevation_difference_m"):
        if column not in station_map:
            station_map[column] = None
    return station_map


def station_map_for_years(station_map: DataFrame, years: list[int]) -> DataFrame:
    """Discard candidate rows whose declared period cannot serve this run."""
    if station_map.empty:
        return station_map
    wanted = set(years)
    if "year" in station_map:
        declared_year = pd.to_numeric(station_map["year"], errors="coerce")
        return station_map[declared_year.isna() | declared_year.isin(wanted)].copy()
    mask = pd.Series(data=True, index=station_map.index)
    if "start_year" in station_map:
        start = pd.to_numeric(station_map["start_year"], errors="coerce")
        mask &= start.isna() | (start <= max(wanted))
    if "end_year" in station_map:
        end = pd.to_numeric(station_map["end_year"], errors="coerce")
        mask &= end.isna() | (end >= min(wanted))
    return station_map[mask].copy()


def assign_reference_stations(station_map: DataFrame) -> DataFrame:
    """Attach one stable reference station to every mapped location.

    Year-specific inventory coverage is useful for finding candidates, but it
    must not redefine a city's climate baseline every year.  Prefer stations
    that rank first most often in the latest catalog decade, then use physical
    proximity as a deterministic tie breaker.  The rule automatically applies
    to newly generated city catalogs.
    """
    if station_map.empty:
        result = station_map.copy()
        result["reference_station_id"] = pd.Series(dtype="string")
        return result
    frame = station_map.copy()
    frame["candidate_rank"] = pd.to_numeric(
        frame["candidate_rank"], errors="coerce"
    ).fillna(999)
    if "year" in frame:
        years = pd.to_numeric(frame["year"], errors="coerce")
        latest_year = years.max()
        if pd.notna(latest_year):
            recent = frame[
                years >= int(latest_year) - GHCNH_REFERENCE_LOOKBACK_YEARS + 1
            ].copy()
        else:
            recent = frame.copy()
    else:
        recent = frame.copy()
    recent["_rank_one"] = (recent["candidate_rank"] == 1).astype(int)
    for column in ("variable_coverage", "dist_km", "elevation_difference_m"):
        recent[column] = pd.to_numeric(recent[column], errors="coerce")
    scores = (
        recent.groupby(["location_id", "ghcn_id"], as_index=False)
        .agg(
            rank_one_years=("_rank_one", "sum"),
            available_years=("ghcn_id", "size"),
            mean_coverage=("variable_coverage", "mean"),
            distance_km=("dist_km", "median"),
            elevation_difference_m=("elevation_difference_m", "median"),
        )
        .sort_values(
            [
                "location_id",
                "rank_one_years",
                "available_years",
                "mean_coverage",
                "distance_km",
                "elevation_difference_m",
                "ghcn_id",
            ],
            ascending=[True, False, False, False, True, True, True],
            na_position="last",
            kind="stable",
        )
    )
    references = scores.drop_duplicates("location_id").rename(
        columns={"ghcn_id": "reference_station_id"}
    )[["location_id", "reference_station_id"]]
    return frame.merge(references, on="location_id", how="left")


def _homogenize_secondary(
    secondary: DataFrame,
    reference_values: DataFrame,
    location_id: object,
    station_id: object,
    reference_id: object,
) -> DataFrame | None:
    paired = secondary.merge(reference_values, on="date", how="inner")
    if paired.empty:
        return None
    paired["_month"] = paired["date"].dt.month
    paired["_wetbulb_delta"] = paired["_reference_wetbulb"] - paired["wetbulb"]
    paired["_wetbulb_avg_delta"] = (
        paired["_reference_wetbulb_avg"] - paired["wetbulb_avg"]
    )
    global_count = len(paired)
    monthly = paired.groupby("_month").agg(
        overlap_days=("date", "size"),
        wetbulb_adjustment=("_wetbulb_delta", "median"),
        wetbulb_avg_adjustment=("_wetbulb_avg_delta", "median"),
    )
    secondary["_month"] = secondary["date"].dt.month
    merged_secondary = secondary.merge(
        monthly, left_on="_month", right_index=True, how="left"
    )
    monthly_ok = merged_secondary["overlap_days"] >= GHCNH_MIN_MONTHLY_OVERLAP_DAYS
    global_ok = global_count >= GHCNH_MIN_GLOBAL_OVERLAP_DAYS
    if global_ok:
        merged_secondary.loc[~monthly_ok, "overlap_days"] = global_count
        merged_secondary.loc[~monthly_ok, "wetbulb_adjustment"] = paired[
            "_wetbulb_delta"
        ].median()
        merged_secondary.loc[~monthly_ok, "wetbulb_avg_adjustment"] = paired[
            "_wetbulb_avg_delta"
        ].median()
    calibrated = monthly_ok | global_ok
    if not calibrated.any():
        LOGGER.warning(
            "location_id=%s station %s has only %d overlap days with "
            "reference %s; rejecting it as an uncalibrated fallback",
            location_id,
            station_id,
            global_count,
            reference_id,
        )
        return None
    result = merged_secondary[calibrated].copy()
    result["homogenization_method"] = "monthly_overlap"
    result.loc[~monthly_ok[calibrated], "homogenization_method"] = "global_overlap"
    result["homogenization_overlap_days"] = result["overlap_days"].astype(int)
    result["wetbulb"] += result["wetbulb_adjustment"]
    result["wetbulb_avg"] += result["wetbulb_avg_adjustment"]
    return result.drop(columns=["_month", "overlap_days"])


def _homogenize_city(
    city_group: DataFrame,
    location_id: object,
    reference_id: object,
) -> list[DataFrame]:
    if pd.isna(reference_id):
        return []
    city = city_group.copy()
    city["date"] = pd.to_datetime(city["date"])
    reference = city[city["station_id"] == reference_id].copy()
    if reference.empty:
        LOGGER.warning(
            "location_id=%s reference station %s has no usable days; "
            "rejecting uncalibrated secondary stations",
            location_id,
            reference_id,
        )
        return []
    reference["homogenization_method"] = "reference"
    reference["homogenization_overlap_days"] = 0
    reference["wetbulb_adjustment"] = 0.0
    reference["wetbulb_avg_adjustment"] = 0.0
    output = [reference]
    reference_values = reference[["date", "wetbulb", "wetbulb_avg"]].rename(
        columns={
            "wetbulb": "_reference_wetbulb",
            "wetbulb_avg": "_reference_wetbulb_avg",
        }
    )
    secondary_groups = city[city["station_id"] != reference_id].groupby(
        "station_id", sort=False
    )
    for station_id, secondary_group in secondary_groups:
        secondary = _homogenize_secondary(
            secondary_group.copy(),
            reference_values,
            location_id,
            station_id,
            reference_id,
        )
        if secondary is not None:
            output.append(secondary)
    return output


def homogenize_station_candidates(candidates: DataFrame) -> DataFrame:
    """Put secondary-station daily values onto each city's reference baseline."""
    if candidates.empty:
        return candidates.copy()
    required = {"reference_station_id", "station_id", "date", "location_id"}
    if not required.issubset(candidates.columns):
        missing = sorted(required - set(candidates.columns))
        msg = f"station homogenization is missing columns: {missing}"
        raise ValueError(msg)
    output: list[DataFrame] = []
    city_groups = candidates.groupby(
        ["location_id", "reference_station_id"], sort=False, dropna=False
    )
    for (location_id, reference_id), city_group in city_groups:
        output.extend(_homogenize_city(city_group, location_id, reference_id))
    if not output:
        return candidates.iloc[0:0].copy()
    return concat_frames(output)


def _candidate_key(row: CandidateRow) -> CandidateKey:
    return row.ghcn_id, row.lon, row.utc_offset_hours


def _fetch_stations_batch(
    candidates: Sequence[CandidateKey],
    years: list[int],
    worker_count: int,
    city_shard_index: int,
    cache_dir: Path | None = None,
) -> dict[CandidateKey, tuple[DataFrame, set[int]]]:
    frames: dict[CandidateKey, list[DataFrame]] = {
        candidate: [] for candidate in candidates
    }
    gaps: dict[CandidateKey, set[int]] = {candidate: set() for candidate in candidates}
    thread_state = threading.local()

    def fetch(candidate: CandidateKey, year: int) -> tuple[DataFrame, bool]:
        session = getattr(thread_state, "session", None)
        if session is None:
            session = requests.Session()
            thread_state.session = session
        return fetch_station_year(
            candidate[0],
            year,
            lon=candidate[1],
            utc_offset_hours=candidate[2],
            drop_incomplete_latest_day=year == datetime.now(tz=UTC).year,
            session=session,
            cache_dir=cache_dir,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(fetch, candidate, year): (candidate, year)
            for candidate in candidates
            for year in years
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"GHCNh station-years city_shard {city_shard_index}",
        ):
            candidate, year = futures[future]
            frame, gap = future.result()
            if gap:
                gaps[candidate].add(year)
            elif not frame.empty:
                frames[candidate].append(frame)
    return {
        candidate: (
            concat_frames(candidate_frames)
            if candidate_frames
            else _empty_hourly_frame(),
            gaps[candidate],
        )
        for candidate, candidate_frames in frames.items()
    }


def _candidate_daily_frame(row: CandidateRow, station_df: DataFrame) -> DataFrame:
    """Aggregate one physical station independently and attach provenance."""
    hourly = _station_to_hourly(int(row.location_id), station_df)
    daily = nldas.compute_daily_wetbulb(
        hourly,
        min_daily_hours=GHCNH_MIN_DAILY_HOURS,
        include_observation_count=True,
    )
    daily = filter_daily_wetbulb_spikes(daily)
    if daily.empty:
        return daily
    dates = pd.to_datetime(daily["date"])
    if hasattr(row, "year") and not pd.isna(row.year):
        daily = daily[dates.dt.year == int(row.year)]
    else:
        if hasattr(row, "start_year") and not pd.isna(row.start_year):
            daily = daily[dates.dt.year >= int(row.start_year)]
            dates = pd.to_datetime(daily["date"])
        if hasattr(row, "end_year") and not pd.isna(row.end_year):
            daily = daily[dates.dt.year <= int(row.end_year)]
    if daily.empty:
        return daily
    daily["source"] = "ghcnh"
    daily["station_id"] = str(row.ghcn_id)
    daily["reference_station_id"] = str(row.reference_station_id)
    daily["station_distance_km"] = row.dist_km
    daily["station_elevation_difference_m"] = row.elevation_difference_m
    daily["station_quality"] = daily["observed_hours"].map(
        lambda count: "complete" if count >= GHCNH_RELIABLE_DAILY_HOURS else "sparse",
    )
    daily["_candidate_rank"] = int(row.candidate_rank)
    daily["_variable_coverage"] = row.variable_coverage
    return daily


def select_best_station_days(candidates: DataFrame) -> DataFrame:
    """Prefer the stable reference, then choose a calibrated fallback day."""
    if candidates.empty:
        return candidates
    ranked = candidates.copy()
    ranked["_variable_coverage"] = pd.to_numeric(
        ranked["_variable_coverage"],
        errors="coerce",
    ).fillna(0.0)
    ranked["station_distance_km"] = pd.to_numeric(
        ranked["station_distance_km"],
        errors="coerce",
    )
    if "reference_station_id" in ranked.columns:
        ranked["_is_reference"] = ranked["station_id"] == ranked["reference_station_id"]
    else:
        ranked["_is_reference"] = False
    ranked = ranked.sort_values(
        [
            "location_id",
            "date",
            "_is_reference",
            "observed_hours",
            "_variable_coverage",
            "station_distance_km",
            "station_id",
        ],
        ascending=[True, True, False, False, False, True, True],
        kind="stable",
    )
    return (
        ranked.drop_duplicates(["location_id", "date"], keep="first")
        .drop(columns=["_candidate_rank", "_variable_coverage", "_is_reference"])
        .reset_index(drop=True)
    )


def process_ghcnh(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    force: bool = False,
    cities_csv: str = "cities_na.csv",
    station_map_csv: str = STATION_MAP_PATH,
) -> None:
    """Fetch mapped stations, compute daily values, and write parquet shards."""
    loaded = _load_pending_shard(
        city_shard_index,
        city_shard_count,
        start_year,
        end_year,
        out_dir,
        force=force,
        logger=LOGGER,
        resolve_fs=resolve_filesystem,
        compute_pending_years=pending_years,
        cities_csv=cities_csv,
    )
    if loaded is None:
        return
    shard_df, years, filesystem, base_path, wetbulb_root = loaded
    station_map = station_map_for_years(
        assign_reference_stations(_load_station_map(station_map_csv)), years
    )
    shard_df = shard_df.merge(station_map, on="location_id", how="left")
    unmapped_ids = set(shard_df.loc[shard_df["ghcn_id"].isna(), "location_id"])
    if unmapped_ids:
        LOGGER.warning(
            "city_shard=%d/%d: %d city(ies) have no GHCNh mapping and require gap-fill.",
            city_shard_index,
            city_shard_count,
            len(unmapped_ids),
        )
    shard_df = shard_df.dropna(subset=["ghcn_id"])
    if shard_df.empty:
        return

    candidate_rows: dict[CandidateKey, list[Any]] = {}
    for row in shard_df.itertuples():
        candidate_rows.setdefault(_candidate_key(row), []).append(row)
    worker_count = max(1, min(concurrency, len(candidate_rows)))
    station_results = _fetch_stations_batch(
        list(candidate_rows),
        years,
        worker_count,
        city_shard_index,
        Path(out_dir) / GHCNH_CACHE_DIRECTORY,
    )

    daily_frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for candidate, (station_df, gaps) in station_results.items():
        gapped_years |= gaps
        daily_frames.extend(
            _candidate_daily_frame(row, station_df) for row in candidate_rows[candidate]
        )
    usable = [frame for frame in daily_frames if not frame.empty]
    if not usable:
        return
    homogenized = homogenize_station_candidates(concat_frames(usable))
    if homogenized.empty:
        return
    selected = select_best_station_days(homogenized)
    writable_years = [year for year in years if year not in gapped_years]
    if gapped_years:
        LOGGER.warning(
            "city_shard=%d/%d: transient GHCNh failures affected year(s) %s; "
            "skipping those writes so they remain retryable.",
            city_shard_index,
            city_shard_count,
            sorted(gapped_years),
        )
    if writable_years:
        write_pending_year_batches(
            selected,
            writable_years,
            wetbulb_root,
            city_shard_index,
            filesystem,
            base_path,
            file_prefix="wetbulb",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_shard_args(parser)
    parser.add_argument("--concurrency", type=int, default=GHCNH_DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--station-map-csv", default=STATION_MAP_PATH)
    return parser.parse_args()


def main() -> None:
    """Execute the GHCNh wet-bulb processing pipeline."""
    load_dotenv(override=False)
    args = _parse_args()
    try:
        process_ghcnh(
            args.start_year,
            args.end_year,
            args.out_dir,
            args.city_shard_index,
            args.city_shard_count,
            args.concurrency,
            force=args.force,
            cities_csv=args.cities_csv,
            station_map_csv=args.station_map_csv,
        )
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
