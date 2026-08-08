# Copyright (C) 2026 Kenneth Porter

"""Fill Canadian station gaps from ECCC's official hourly climate archive."""

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

import ghcnh
import nldas
from lcd import (
    _load_pending_shard,
    _station_to_hourly,
    add_common_shard_args,
)
from partition_io import pending_years, write_pending_year_batches
from shards import resolve_filesystem

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any
type CandidateRow = Any
type CandidateKey = tuple[int, float]

LOGGER = logging.getLogger(__name__)
_RNG = random.SystemRandom()

ECCC_HOURLY_URL = "https://api.weather.gc.ca/collections/climate-hourly/items"
ECCC_FILE_PREFIX = "wetbulb_eccc"
ECCC_SOURCE = "eccc"
ECCC_STATION_MAP_PATH = "cities_na_eccc_stations.csv"
ECCC_REQUEST_TIMEOUT_SECONDS = 90
ECCC_MAX_RETRIES = 3
ECCC_RETRY_DELAY_SECONDS = 3
ECCC_PAGE_LIMIT = 10_000
ECCC_MIN_DAILY_HOURS = 20
ECCC_DEFAULT_CONCURRENCY = 6
_REQUEST_ERRORS = (requests.RequestException, ValueError)


def _empty_hourly_frame() -> DataFrame:
    return pd.DataFrame(columns=["time", "tair_c", "dewpoint_c", "pressure_hpa"])


def _get_json_with_retries(
    session: requests.Session,
    params: dict[str, Any],
    *,
    station_id: int,
    year: int,
) -> dict[str, Any] | None:
    for attempt in range(1, ECCC_MAX_RETRIES + 1):
        try:
            response = session.get(
                ECCC_HOURLY_URL,
                params=params,
                timeout=ECCC_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return cast("dict[str, Any]", response.json())
        except _REQUEST_ERRORS:
            if attempt == ECCC_MAX_RETRIES:
                LOGGER.warning(
                    "Giving up on ECCC station=%s year=%d after %d attempts.",
                    station_id,
                    year,
                    attempt,
                )
                return None
            time.sleep(ECCC_RETRY_DELAY_SECONDS * attempt + _RNG.uniform(0.0, 1.0))
    return None


def fetch_station_year(
    station_id: int,
    year: int,
    *,
    utc_offset_hours: float,
    session: requests.Session | None = None,
) -> tuple[DataFrame, bool]:
    """Return one ECCC station-year and whether a transient fetch gap occurred."""
    http = session or requests.Session()
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T23:59:59Z"
    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "f": "json",
            "STN_ID": station_id,
            "datetime": f"{start}/{end}",
            "limit": ECCC_PAGE_LIMIT,
            "offset": offset,
        }
        payload = _get_json_with_retries(
            http,
            params,
            station_id=station_id,
            year=year,
        )
        if payload is None:
            return _empty_hourly_frame(), True
        page = payload.get("features", [])
        features.extend(page)
        returned = int(payload.get("numberReturned", len(page)))
        matched = int(payload.get("numberMatched", len(features)))
        offset += returned
        if returned == 0 or offset >= matched:
            break

    if not features:
        return _empty_hourly_frame(), False
    rows = [feature.get("properties", {}) for feature in features]
    raw = pd.DataFrame(rows)
    for column in ("TEMP", "DEW_POINT_TEMP", "STATION_PRESSURE"):
        raw[column] = pd.to_numeric(raw.get(column), errors="coerce")
        flag_column = f"{column}_FLAG"
        if flag_column in raw:
            flags = raw[flag_column].astype("string").str.strip()
            raw[column] = raw[column].where(flags.isna() | flags.eq(""))
    time_utc = pd.to_datetime(raw.get("UTC_DATE"), format="ISO8601", errors="coerce")
    frame = pd.DataFrame(
        {
            "time": time_utc + pd.to_timedelta(utc_offset_hours, unit="h"),
            "tair_c": raw["TEMP"],
            "dewpoint_c": raw["DEW_POINT_TEMP"],
            # The ECCC archive publishes station pressure in kPa.
            "pressure_hpa": raw["STATION_PRESSURE"] * 10.0,
        },
    ).dropna()
    if year == datetime.now(tz=UTC).year:
        local_today = (
            pd.Timestamp(datetime.now(tz=UTC))
            + pd.to_timedelta(utc_offset_hours, unit="h")
        ).date()
        frame = frame[pd.to_datetime(frame["time"]).dt.date < local_today]
    return frame.drop_duplicates("time").sort_values("time").reset_index(
        drop=True
    ), False


def _load_station_map(path: str) -> DataFrame:
    map_path = Path(path)
    if not map_path.exists():
        LOGGER.warning("%s not found; no Canadian ECCC gaps can be filled.", path)
        return pd.DataFrame()
    station_map = pd.read_csv(map_path)
    required = {"location_id", "eccc_station_id", "utc_offset_hours"}
    missing = required - set(station_map.columns)
    if missing:
        msg = f"ECCC station map missing columns: {sorted(missing)}"
        raise ValueError(msg)
    if "candidate_rank" not in station_map:
        station_map["candidate_rank"] = (
            station_map.groupby("location_id").cumcount() + 1
        )
    for column in ("dist_km", "elevation_difference_m", "start_year", "end_year"):
        if column not in station_map:
            station_map[column] = None
    return station_map


def _candidate_key(row: CandidateRow) -> CandidateKey:
    return int(row.eccc_station_id), float(row.utc_offset_hours)


def _fetch_candidates(
    candidates: list[CandidateKey],
    years: list[int],
    concurrency: int,
) -> dict[CandidateKey, tuple[DataFrame, set[int]]]:
    def fetch(candidate: CandidateKey) -> tuple[DataFrame, set[int]]:
        station_id, utc_offset_hours = candidate
        frames: list[DataFrame] = []
        gaps: set[int] = set()
        session = requests.Session()
        for year in years:
            frame, gap = fetch_station_year(
                station_id,
                year,
                utc_offset_hours=utc_offset_hours,
                session=session,
            )
            if gap:
                gaps.add(year)
            elif not frame.empty:
                frames.append(frame)
        if not frames:
            return _empty_hourly_frame(), gaps
        return pd.concat(frames, ignore_index=True), gaps

    results: dict[CandidateKey, tuple[DataFrame, set[int]]] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(concurrency, len(candidates)))
    ) as pool:
        futures = {pool.submit(fetch, candidate): candidate for candidate in candidates}
        for future in tqdm(as_completed(futures), total=len(futures), desc="ECCC gaps"):
            results[futures[future]] = future.result()
    return results


def _candidate_daily_frame(row: CandidateRow, station_df: DataFrame) -> DataFrame:
    hourly = _station_to_hourly(int(row.location_id), station_df)
    daily = nldas.compute_daily_wetbulb(
        hourly,
        min_daily_hours=ECCC_MIN_DAILY_HOURS,
        include_observation_count=True,
    )
    if daily.empty:
        return daily
    years = pd.to_datetime(daily["date"]).dt.year
    if not pd.isna(row.start_year):
        daily = daily[years >= int(row.start_year)]
        years = pd.to_datetime(daily["date"]).dt.year
    if not pd.isna(row.end_year):
        daily = daily[years <= int(row.end_year)]
    if daily.empty:
        return daily
    daily["source"] = ECCC_SOURCE
    daily["station_id"] = str(int(row.eccc_station_id))
    daily["station_distance_km"] = row.dist_km
    daily["station_elevation_difference_m"] = row.elevation_difference_m
    daily["station_quality"] = "complete"
    daily["_candidate_rank"] = int(row.candidate_rank)
    daily["_variable_coverage"] = 0.0
    return daily


def process_eccc(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    force: bool = False,
    cities_csv: str = "cities_na.csv",
    station_map_csv: str = ECCC_STATION_MAP_PATH,
) -> None:
    """Write complete ECCC station-days as a primary supplement to GHCNh."""
    loaded = _load_pending_shard(
        city_shard_index,
        city_shard_count,
        start_year,
        end_year,
        out_dir,
        force=force,
        logger=LOGGER,
        resolve_fs=resolve_filesystem,
        compute_pending_years=lambda years, root, shard, fs, base, **kwargs: (
            pending_years(
                years,
                root,
                shard,
                fs,
                base,
                file_prefix=ECCC_FILE_PREFIX,
                force=kwargs["force"],
            )
        ),
        cities_csv=cities_csv,
    )
    if loaded is None:
        return
    shard_df, years, filesystem, base_path, wetbulb_root = loaded
    station_map = ghcnh.station_map_for_years(_load_station_map(station_map_csv), years)
    if station_map.empty:
        return
    candidates = shard_df.merge(station_map, on="location_id", how="inner")
    if candidates.empty:
        return
    candidate_rows: dict[CandidateKey, list[Any]] = {}
    for row in candidates.itertuples():
        candidate_rows.setdefault(_candidate_key(row), []).append(row)
    results = _fetch_candidates(list(candidate_rows), years, concurrency)
    frames: list[DataFrame] = []
    gapped_years: set[int] = set()
    for candidate, (station_df, gaps) in results.items():
        gapped_years |= gaps
        frames.extend(
            _candidate_daily_frame(row, station_df) for row in candidate_rows[candidate]
        )
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return
    selected = ghcnh.select_best_station_days(pd.concat(usable, ignore_index=True))
    writable_years = [year for year in years if year not in gapped_years]
    if writable_years:
        write_pending_year_batches(
            selected,
            writable_years,
            wetbulb_root,
            city_shard_index,
            filesystem,
            base_path,
            file_prefix=ECCC_FILE_PREFIX,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_shard_args(parser)
    parser.add_argument("--concurrency", type=int, default=ECCC_DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--station-map-csv", default=ECCC_STATION_MAP_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the ECCC primary-station supplement."""
    load_dotenv(override=False)
    args = _parse_args()
    try:
        process_eccc(
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
