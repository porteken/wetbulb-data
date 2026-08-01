"""Fetch NOAA GHCNh station observations and compute daily wet-bulb temperature."""

from __future__ import annotations

import argparse
import importlib
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

import nldas
from lcd import (
    _HYPSOMETRIC_SCALE_M_PER_K,
    _ICAO_EXPONENT,
    _ICAO_LAPSE_K_PER_M,
    _ICAO_SEA_LEVEL_T_K,
    _KELVIN_OFFSET,
    _load_pending_shard,
    _station_to_hourly,
    add_common_shard_args,
    concat_frames,
)
from partition_io import pending_years, write_pending_year_batches
from shards import resolve_filesystem
from station_exclusions import DISALLOWED_GHCNH_STATION_IDS

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))
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
GHCNH_MIN_DAILY_HOURS = 20
GHCNH_RELIABLE_DAILY_HOURS = 20
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
_REPORT_TYPES = frozenset({"EnvCan", "FM12", "FM15", "FM16"})
_REPORT_PRIORITY = {"FM15": 0, "FM16": 1, "FM12": 2, "EnvCan": 3}
_REJECT_QC_CODES = frozenset({"2", "3", "6", "7"})


def _empty_hourly_frame() -> DataFrame:
    return pd.DataFrame(columns=list(_HOURLY_FRAME_COLUMNS))


def _quality_filtered(raw: DataFrame, value: str) -> DataFrame:
    values = pd.to_numeric(raw[value], errors="coerce")
    quality = raw[f"{value}_Quality_Code"].astype("string")
    return values.where(~quality.isin(_REJECT_QC_CODES))


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
    if hourly.empty:
        return _empty_hourly_frame()

    hourly["_priority"] = hourly["REPORT_TYPE"].map(_REPORT_PRIORITY)
    hourly = hourly.sort_values(["time_utc", "_priority"]).drop_duplicates(
        subset="time_utc",
        keep="first",
    )
    pressure_hpa = hourly["station_pressure_hpa"]
    sea_level_derived = hourly["slp_hpa"] * np.exp(
        -hourly["ELEVATION"]
        / (_HYPSOMETRIC_SCALE_M_PER_K * (hourly["tair_c"] + _KELVIN_OFFSET)),
    )
    altimeter_derived = (
        hourly["altimeter_hpa"]
        * (
            (_ICAO_SEA_LEVEL_T_K - _ICAO_LAPSE_K_PER_M * hourly["ELEVATION"])
            / _ICAO_SEA_LEVEL_T_K
        )
        ** _ICAO_EXPONENT
    )
    pressure_hpa = pressure_hpa.fillna(sea_level_derived).fillna(altimeter_derived)

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
) -> tuple[DataFrame, bool]:
    """Return one station-year frame and whether a transient fetch gap occurred."""
    http = session or requests.Session()
    url = GHCNH_URL_TEMPLATE.format(year=year, station_id=station_id)
    response = _get_with_retries(http, url, station_id=station_id, year=year)
    if response is None:
        return _empty_hourly_frame(), True
    if response.status_code == requests.codes.not_found:
        return _empty_hourly_frame(), False
    return (
        _parse_ghcnh_parquet(
            response.content,
            lon=lon,
            utc_offset_hours=utc_offset_hours,
            drop_incomplete_latest_day=drop_incomplete_latest_day,
        ),
        False,
    )


def _fetch_station_series(
    session: requests.Session,
    station_id: str,
    years: list[int],
    lon: float | None,
    utc_offset_hours: float | None,
) -> tuple[DataFrame, set[int]]:
    frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    current_year = datetime.now(tz=UTC).year
    for year in years:
        frame, gap = fetch_station_year(
            station_id,
            year,
            lon=lon,
            utc_offset_hours=utc_offset_hours,
            drop_incomplete_latest_day=year == current_year,
            session=session,
        )
        if gap:
            gapped_years.add(year)
        elif not frame.empty:
            frames.append(frame)
    if not frames:
        return _empty_hourly_frame(), gapped_years
    return concat_frames(frames), gapped_years


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


def _candidate_key(row: CandidateRow) -> CandidateKey:
    return row.ghcn_id, row.lon, row.utc_offset_hours


def _fetch_stations_batch(
    candidates: list[CandidateKey],
    years: list[int],
    worker_count: int,
    city_shard_index: int,
) -> dict[CandidateKey, tuple[DataFrame, set[int]]]:
    results: dict[CandidateKey, tuple[DataFrame, set[int]]] = {}
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fetch_station_series,
                session,
                candidate[0],
                years,
                candidate[1],
                candidate[2],
            ): candidate
            for candidate in candidates
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"GHCNh->wetbulb city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()
    return results


def _candidate_daily_frame(row: CandidateRow, station_df: DataFrame) -> DataFrame:
    """Aggregate one physical station independently and attach provenance."""
    hourly = _station_to_hourly(int(row.location_id), station_df)
    daily = nldas.compute_daily_wetbulb(
        hourly,
        min_daily_hours=GHCNH_MIN_DAILY_HOURS,
        include_observation_count=True,
    )
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
    daily["station_distance_km"] = row.dist_km
    daily["station_elevation_difference_m"] = row.elevation_difference_m
    daily["station_quality"] = daily["observed_hours"].map(
        lambda count: "complete" if count >= GHCNH_RELIABLE_DAILY_HOURS else "sparse",
    )
    daily["_candidate_rank"] = int(row.candidate_rank)
    daily["_variable_coverage"] = row.variable_coverage
    return daily


def select_best_station_days(candidates: DataFrame) -> DataFrame:
    """Choose one physical station per city-day without mixing hourly rows."""
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
    ranked = ranked.sort_values(
        [
            "location_id",
            "date",
            "_candidate_rank",
            "observed_hours",
            "_variable_coverage",
            "station_distance_km",
            "station_id",
        ],
        ascending=[True, True, True, False, False, True, True],
        kind="stable",
    )
    return (
        ranked.drop_duplicates(["location_id", "date"], keep="first")
        .drop(columns=["_candidate_rank", "_variable_coverage"])
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
    station_map = station_map_for_years(_load_station_map(station_map_csv), years)
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
    selected = select_best_station_days(concat_frames(usable))
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
