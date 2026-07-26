"""Fetch NOAA ISD Global Hourly station observations and compute daily wet-bulb temperature."""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

type DataFrame = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
_RNG = random.SystemRandom()

ISD_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"
)
ISD_REQUEST_TIMEOUT_SECONDS = 60
ISD_MAX_RETRIES = 3
ISD_RETRY_DELAY_SECONDS = 5
ISD_DEFAULT_CONCURRENCY = 8

STATION_MAP_PATH = "cities_na_isd_stations.csv"
_STATION_MAP_WARNED = [False]

_HOURLY_FRAME_COLUMNS = ("time", "tair_c", "dewpoint_c", "pressure_hpa")

HOURLY_REPORT_TYPES: tuple[str, ...] = ("FM-15", "FM-16", "FM-12")

_ISD_REJECT_QC_CODES = frozenset({"2", "3", "6", "7"})
_ISD_MISSING_TMP_DEW = "9999"
_ISD_MISSING_PRESSURE = "99999"


def _empty_hourly_frame() -> DataFrame:
    return pd.DataFrame(columns=list(_HOURLY_FRAME_COLUMNS))


def _column_or_missing(df: DataFrame, name: str) -> DataFrame:
    """Return `df[name]`, or an all-missing string Series if the column is absent."""
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _parse_isd_field(raw: DataFrame, *, missing: str) -> DataFrame:
    """Split a comma-packed `"value,quality_code"` ISD element into a scaled float."""
    parts = raw.str.split(",")
    value_str = parts.str[0]
    qc = parts.str[1]
    value_raw = pd.to_numeric(value_str, errors="coerce")
    is_bad = (
        value_raw.isna()
        | (value_raw.abs() == float(missing))
        | qc.isin(_ISD_REJECT_QC_CODES)
    )
    return (value_raw / 10.0).where(~is_bad)


def _get_with_retries(
    session: requests.Session,
    url: str,
    *,
    station_id: str,
    year: int,
) -> requests.Response | None:
    """GET with retry/backoff; a 404 is returned as-is (not retried)."""
    for attempt in range(1, ISD_MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=ISD_REQUEST_TIMEOUT_SECONDS)
            if response.status_code == requests.codes.not_found:
                return response
            response.raise_for_status()
        except requests.RequestException:
            if attempt == ISD_MAX_RETRIES:
                LOGGER.warning(
                    "Giving up on station=%s year=%d after %d attempt(s).",
                    station_id,
                    year,
                    attempt,
                )
                return None
            jitter = _RNG.uniform(0.0, 2.0)
            time.sleep(ISD_RETRY_DELAY_SECONDS * attempt + jitter)
        else:
            return response
    return None


def _parse_isd_response(
    response_text: str,
    *,
    lon: float | None,
    utc_offset_hours: float | None = None,
) -> DataFrame:
    """Parse one candidate id's ISD station-year CSV into the hourly frame."""
    raw = pd.read_csv(io.StringIO(response_text), dtype=str, low_memory=False)
    raw["REPORT_TYPE"] = _column_or_missing(raw, "REPORT_TYPE").str.strip()
    hourly = raw[raw["REPORT_TYPE"].isin(HOURLY_REPORT_TYPES)].copy()
    if hourly.empty:
        return _empty_hourly_frame()

    hourly["ELEVATION"] = pd.to_numeric(
        _column_or_missing(hourly, "ELEVATION"), errors="coerce"
    )
    hourly["tair_c"] = _parse_isd_field(
        _column_or_missing(hourly, "TMP"), missing=_ISD_MISSING_TMP_DEW
    )
    hourly["dewpoint_c"] = _parse_isd_field(
        _column_or_missing(hourly, "DEW"), missing=_ISD_MISSING_TMP_DEW
    )
    hourly["slp_hpa"] = _parse_isd_field(
        _column_or_missing(hourly, "SLP"), missing=_ISD_MISSING_PRESSURE
    )

    ma1 = _column_or_missing(hourly, "MA1").str.split(",")
    altimeter_value_str = ma1.str[0]
    altimeter_qc = ma1.str[1]
    station_pressure_str = ma1.str[2]
    station_pressure_qc = ma1.str[3]
    altimeter_value = pd.to_numeric(altimeter_value_str, errors="coerce") / 10.0
    altimeter_bad = (
        altimeter_value_str.isna()
        | (altimeter_value_str == _ISD_MISSING_PRESSURE)
        | altimeter_qc.isin(_ISD_REJECT_QC_CODES)
    )
    hourly["altimeter_hpa"] = altimeter_value.where(~altimeter_bad)
    station_pressure_value = pd.to_numeric(station_pressure_str, errors="coerce") / 10.0
    station_pressure_bad = (
        station_pressure_str.isna()
        | (station_pressure_str == _ISD_MISSING_PRESSURE)
        | station_pressure_qc.isin(_ISD_REJECT_QC_CODES)
    )
    hourly["station_pressure_hpa"] = station_pressure_value.where(~station_pressure_bad)

    hourly["time_utc"] = pd.to_datetime(hourly["DATE"], errors="coerce")
    hourly = hourly.dropna(subset=["time_utc", "tair_c", "dewpoint_c"])
    if hourly.empty:
        return _empty_hourly_frame()

    report_priority = {"FM-15": 0, "FM-16": 1, "FM-12": 2}
    hourly["_priority"] = hourly["REPORT_TYPE"].map(report_priority)
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
        station_lon = float(lon) if lon is not None else None
        if station_lon is None:
            observed_lon = pd.to_numeric(
                _column_or_missing(hourly, "LONGITUDE"), errors="coerce"
            ).dropna()
            station_lon = float(observed_lon.iloc[0]) if not observed_lon.empty else 0.0
        utc_offset_hours = round(station_lon / 15.0)
    local_time = hourly["time_utc"] + pd.to_timedelta(float(utc_offset_hours), unit="h")

    return pd.DataFrame(
        {
            "time": local_time.to_numpy(),
            "tair_c": hourly["tair_c"].to_numpy(dtype="float64"),
            "dewpoint_c": hourly["dewpoint_c"].to_numpy(dtype="float64"),
            "pressure_hpa": pressure_hpa.to_numpy(dtype="float64"),
        },
    ).reset_index(drop=True)


def fetch_station_year(
    candidate_ids: list[str],
    year: int,
    *,
    lon: float | None = None,
    utc_offset_hours: float | None = None,
    session: requests.Session | None = None,
) -> tuple[DataFrame, bool]:
    """Return (hourly dry-bulb/dewpoint/pressure frame, gap) for one station-year."""
    http = session or requests.Session()
    for station_id in candidate_ids:
        url = ISD_URL_TEMPLATE.format(year=year, station_id=station_id)
        response = _get_with_retries(http, url, station_id=station_id, year=year)
        if response is None:
            return _empty_hourly_frame(), True
        if response.status_code == requests.codes.not_found:
            continue
        frame = _parse_isd_response(
            response.text, lon=lon, utc_offset_hours=utc_offset_hours
        )
        if not frame.empty:
            return frame, False
        LOGGER.info(
            "Candidate %s year=%d exists but has no hourly rows; trying "
            "the next candidate.",
            station_id,
            year,
        )

    LOGGER.info(
        "No ISD file with hourly rows among %d candidate(s) for year=%d.",
        len(candidate_ids),
        year,
    )
    return _empty_hourly_frame(), False


def _fetch_station_series(
    session: requests.Session,
    candidate_ids: list[str],
    years: list[int],
    lon: float | None,
    utc_offset_hours: float | None = None,
) -> tuple[DataFrame, set[int]]:
    """Fetch every pending year for one station; return (frame, gapped years)."""
    year_frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for year in years:
        frame, gap = fetch_station_year(
            candidate_ids,
            year,
            lon=lon,
            utc_offset_hours=utc_offset_hours,
            session=session,
        )
        if gap:
            gapped_years.add(year)
        elif not frame.empty:
            year_frames.append(frame)
    if not year_frames:
        return _empty_hourly_frame(), gapped_years
    return concat_frames(year_frames), gapped_years


def _load_station_map(path: str | None = None) -> DataFrame:
    """Load the city -> ISD candidate-id crosswalk, if available."""
    map_path = Path(path if path is not None else STATION_MAP_PATH)
    if not map_path.exists():
        if not _STATION_MAP_WARNED[0]:
            LOGGER.warning(
                "%s not found; no cities can be mapped to an ISD station. "
                "Run make_isd_station_map.py to generate it.",
                path,
            )
            _STATION_MAP_WARNED[0] = True
        return pd.DataFrame(
            {
                "location_id": pd.Series(dtype="int64"),
                "isd_ids": pd.Series(dtype="object"),
                "lon": pd.Series(dtype="float64"),
                "utc_offset_hours": pd.Series(dtype="float64"),
            },
        )
    available_columns = pd.read_csv(map_path, nrows=0).columns
    usecols = ["location_id", "isd_ids", "lon"]
    if "utc_offset_hours" in available_columns:
        usecols.append("utc_offset_hours")
    station_map = pd.read_csv(map_path, usecols=usecols)
    if "utc_offset_hours" not in station_map.columns:
        station_map["utc_offset_hours"] = pd.Series(
            [None] * len(station_map), dtype="float64"
        )
    return station_map


def _fetch_stations_batch(
    station_keys: list[str],
    lon_by_key: dict[str, float | None],
    offset_by_key: dict[str, float | None],
    years: list[int],
    session: requests.Session,
    worker_count: int,
    city_shard_index: int,
) -> dict[str, tuple[DataFrame, set[int]]]:
    """Fetch a batch of (deduplicated) stations concurrently, `worker_count` at a time."""
    results: dict[str, tuple[DataFrame, set[int]]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fetch_station_series,
                session,
                station_key.split("|"),
                years,
                lon_by_key[station_key],
                offset_by_key.get(station_key),
            ): station_key
            for station_key in station_keys
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"ISD->wetbulb city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()
    return results


def process_isd(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    force: bool = False,
    cities_csv: str = "cities_na.csv",
    station_map_csv: str | None = None,
) -> None:
    """Fetch NOAA ISD station data, compute daily wet-bulb, and save as parquet shards."""
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
    shard_df, pending_year_list, filesystem, base_path, wetbulb_root = loaded

    station_map = _load_station_map(station_map_csv)
    shard_df = shard_df.merge(station_map, on="location_id", how="left")
    unmapped = shard_df[shard_df["isd_ids"].isna()]
    if not unmapped.empty:
        LOGGER.warning(
            "city_shard=%d/%d: %d cit(ies) have no ISD station mapped and will "
            "be skipped: %s",
            city_shard_index,
            city_shard_count,
            len(unmapped),
            ", ".join(str(x) for x in unmapped["location_id"].tolist()),
        )
    shard_df = shard_df.dropna(subset=["isd_ids"])
    if shard_df.empty:
        return

    station_to_locations: dict[str, list[int]] = {}
    lon_by_key: dict[str, float | None] = {}
    offset_by_key: dict[str, float | None] = {}
    for row in shard_df.itertuples():
        station_to_locations.setdefault(row.isd_ids, []).append(row.location_id)
        lon_by_key[row.isd_ids] = row.lon
        offset_by_key[row.isd_ids] = row.utc_offset_hours

    LOGGER.info(
        "ISD->wetbulb city_shard=%d/%d: %d city(ies) across %d station(s), "
        "%d pending year(s).",
        city_shard_index,
        city_shard_count,
        len(shard_df),
        len(station_to_locations),
        len(pending_year_list),
    )

    session = requests.Session()
    worker_count = max(1, min(concurrency, len(station_to_locations)))
    station_results = _fetch_stations_batch(
        list(station_to_locations),
        lon_by_key,
        offset_by_key,
        pending_year_list,
        session,
        worker_count,
        city_shard_index,
    )

    hourly_frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for station_key, (station_df, gaps) in station_results.items():
        gapped_years |= gaps
        hourly_frames.extend(
            _station_to_hourly(location_id, station_df)
            for location_id in station_to_locations[station_key]
        )

    _write_daily_shard(
        hourly_frames,
        city_shard_index,
        city_shard_count,
        pending_year_list,
        gapped_years,
        wetbulb_root,
        filesystem,
        base_path,
        logger=LOGGER,
        write_batches=write_pending_year_batches,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_shard_args(parser)
    parser.add_argument("--concurrency", type=int, default=ISD_DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--station-map-csv", type=str, default=STATION_MAP_PATH)
    return parser.parse_args()


def main() -> None:
    """Execute the ISD wet-bulb processing pipeline."""
    load_dotenv(override=False)
    exit_code = 0
    try:
        args = _parse_args()
        process_isd(
            start_year=args.start_year,
            end_year=args.end_year,
            out_dir=args.out_dir,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
            concurrency=args.concurrency,
            force=args.force,
            cities_csv=args.cities_csv,
            station_map_csv=args.station_map_csv,
        )
    except KeyboardInterrupt:
        exit_code = 130
        LOGGER.warning("ISD processing interrupted by user.")
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
        LOGGER.exception("ISD processing failed.")

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
