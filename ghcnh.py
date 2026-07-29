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

from lcd import (
    _HYPSOMETRIC_SCALE_M_PER_K,
    _ICAO_EXPONENT,
    _ICAO_LAPSE_K_PER_M,
    _ICAO_SEA_LEVEL_T_K,
    _KELVIN_OFFSET,
    _load_pending_shard,
    _station_to_hourly,
    _write_daily_shard,
    add_common_shard_args,
    concat_frames,
)
from partition_io import pending_years, write_pending_year_batches
from shards import resolve_filesystem

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))
pa = cast("Any", importlib.import_module("pyarrow"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))

type DataFrame = Any

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
_REPORT_TYPES = frozenset({"FM12", "FM15", "FM16"})
_REPORT_PRIORITY = {"FM15": 0, "FM16": 1, "FM12": 2}
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
    except OSError, pa.ArrowException:
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
    return pd.read_csv(
        map_path,
        usecols=["location_id", "ghcn_id", "lon", "utc_offset_hours"],
    )


def _fetch_stations_batch(
    station_ids: list[str],
    lon_by_station: dict[str, float | None],
    offset_by_station: dict[str, float | None],
    years: list[int],
    worker_count: int,
    city_shard_index: int,
) -> dict[str, tuple[DataFrame, set[int]]]:
    results: dict[str, tuple[DataFrame, set[int]]] = {}
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fetch_station_series,
                session,
                station_id,
                years,
                lon_by_station[station_id],
                offset_by_station[station_id],
            ): station_id
            for station_id in station_ids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"GHCNh->wetbulb city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()
    return results


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
    station_map = _load_station_map(station_map_csv)
    shard_df = shard_df.merge(station_map, on="location_id", how="left")
    unmapped = shard_df[shard_df["ghcn_id"].isna()]
    if not unmapped.empty:
        LOGGER.warning(
            "city_shard=%d/%d: %d city(ies) have no GHCNh mapping and require gap-fill.",
            city_shard_index,
            city_shard_count,
            len(unmapped),
        )
    shard_df = shard_df.dropna(subset=["ghcn_id"])
    if shard_df.empty:
        return

    station_to_locations: dict[str, list[int]] = {}
    lon_by_station: dict[str, float | None] = {}
    offset_by_station: dict[str, float | None] = {}
    for row in shard_df.itertuples():
        station_to_locations.setdefault(row.ghcn_id, []).append(row.location_id)
        lon_by_station[row.ghcn_id] = row.lon
        offset_by_station[row.ghcn_id] = row.utc_offset_hours
    worker_count = max(1, min(concurrency, len(station_to_locations)))
    station_results = _fetch_stations_batch(
        list(station_to_locations),
        lon_by_station,
        offset_by_station,
        years,
        worker_count,
        city_shard_index,
    )

    hourly_frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for station_id, (station_df, gaps) in station_results.items():
        gapped_years |= gaps
        hourly_frames.extend(
            _station_to_hourly(location_id, station_df)
            for location_id in station_to_locations[station_id]
        )
    _write_daily_shard(
        hourly_frames,
        city_shard_index,
        city_shard_count,
        years,
        gapped_years,
        wetbulb_root,
        filesystem,
        base_path,
        logger=LOGGER,
        write_batches=write_pending_year_batches,
        source="ghcnh",
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
