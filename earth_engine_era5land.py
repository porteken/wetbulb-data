# Copyright (C) 2026 Kenneth Porter

"""Fill station gaps with Google Earth Engine's ERA5-Land hourly collection."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from tqdm.auto import tqdm

import era5land
import lcd
import nldas
from gapfill import (
    GAPFILL_FILE_PREFIX,
    MIN_MISSING_DAYS_DEFAULT,
    _gap_years_by_location,
    resolve_gapfill_targets,
)
from partition_io import write_pending_year_batches

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any
type CityRow = Any
type EarthEngine = Any
type ImageCollection = Any

LOGGER = logging.getLogger(__name__)

EE_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
EE_BANDS = ("temperature_2m", "dewpoint_temperature_2m", "surface_pressure")
EE_SCALE_METERS = 11_132
EE_CHUNK_DAYS = 31
EE_DEFAULT_CONCURRENCY = 8
EE_MAX_RETRIES = 4
EE_RETRY_DELAY_SECONDS = 2
EE_CREDENTIALS_ENV = "GOOGLE_EARTH_ENGINE_CREDENTIALS"
GOOGLE_APPLICATION_CREDENTIALS_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
EE_PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"


def _ee() -> EarthEngine:
    return importlib.import_module("ee")


def initialize_earth_engine(
    credentials_json: str | None = None,
    project: str | None = None,
) -> None:
    """Initialize EE from JSON content, a JSON path, or ambient credentials."""
    ee = _ee()
    credential_value = (
        credentials_json
        or os.getenv(EE_CREDENTIALS_ENV)
        or os.getenv(GOOGLE_APPLICATION_CREDENTIALS_ENV)
    )
    project_id = project or os.getenv(EE_PROJECT_ENV)
    if not credential_value:
        ee.Initialize(project=project_id)
        return

    path = Path(credential_value).expanduser()
    details = json.loads(path.read_text() if path.is_file() else credential_value)
    project_id = project_id or details.get("project_id")
    email = details.get("client_email")
    if not email or not project_id:
        msg = "Earth Engine credentials require client_email and project_id"
        raise ValueError(msg)
    credentials = ee.ServiceAccountCredentials(email, key_data=json.dumps(details))
    ee.Initialize(credentials, project=project_id)


def _collection(start_year: int, end_year: int) -> ImageCollection:
    """Include boundary UTC hours needed to form local-standard calendar days."""
    ee = _ee()
    return (
        ee.ImageCollection(EE_COLLECTION)
        .filterDate(f"{start_year - 1}-12-30", f"{end_year + 1}-01-03")
        .select(list(EE_BANDS))
    )


def _collection_chunks(
    collection: ImageCollection, start_year: int, end_year: int
) -> list[ImageCollection]:
    """Split a collection so each EE reduction has a bounded memory footprint."""
    start = date(start_year - 1, 12, 30)
    stop = date(end_year + 1, 1, 3)
    chunks = []
    while start < stop:
        chunk_stop = min(start + timedelta(days=EE_CHUNK_DAYS), stop)
        chunks.append(collection.filterDate(start.isoformat(), chunk_stop.isoformat()))
        start = chunk_stop
    return chunks


def _fetch_errors(ee: EarthEngine) -> tuple[type[BaseException], ...]:
    """Return local and Earth Engine failures that are safe to retry."""
    errors: tuple[type[BaseException], ...] = (
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
    )
    ee_exception = getattr(getattr(ee, "ee_exception", None), "EEException", None)
    return (*errors, ee_exception) if isinstance(ee_exception, type) else errors


def _raw_region_frame(values: list[list[Any]]) -> DataFrame:
    if not values:
        return pd.DataFrame()
    return pd.DataFrame(values[1:], columns=values[0])


def _hourly_frame(location_id: int, raw: DataFrame) -> DataFrame:
    """Convert Earth Engine getRegion output into the shared hourly schema."""
    if raw.empty:
        return era5land.empty_hourly_frame()
    renamed = raw.rename(
        columns={
            "temperature_2m": "Tair",
            "dewpoint_temperature_2m": "dewpoint",
            "surface_pressure": "PSurf",
        }
    )
    for column in ("Tair", "dewpoint", "PSurf"):
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed["Qair"] = era5land.dewpoint_to_specific_humidity(
        renamed["dewpoint"] - era5land.KELVIN_OFFSET,
        renamed["PSurf"] / 100.0,
    )
    renamed["location_id"] = location_id
    renamed["time"] = pd.to_datetime(renamed["time"], unit="ms")
    return renamed[["location_id", "time", "Tair", "Qair", "PSurf"]].dropna(
        subset=["Tair", "Qair", "PSurf"]
    )


def _fetch_city(
    collections: list[ImageCollection], row: CityRow
) -> tuple[DataFrame, bool]:
    """Fetch one mapped land point, retrying transient Earth Engine failures."""
    ee = _ee()
    frames = []
    point = ee.Geometry.Point([float(row.lng), float(row.lat)])
    for chunk_index, collection in enumerate(collections, start=1):
        last_error: BaseException | None = None
        for attempt in range(1, EE_MAX_RETRIES + 1):
            try:
                values = collection.getRegion(point, EE_SCALE_METERS).getInfo()
                frame = _hourly_frame(int(row.location_id), _raw_region_frame(values))
                if frame.empty:
                    break
                frame["utc_offset_hours"] = float(row.utc_offset_hours)
                frames.append(frame)
                break
            except _fetch_errors(ee) as exc:
                last_error = exc
                if attempt < EE_MAX_RETRIES:
                    time.sleep(EE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)))
        else:
            LOGGER.warning(
                "Giving up on location_id=%d chunk=%d after %d attempt(s): %s",
                row.location_id,
                chunk_index,
                EE_MAX_RETRIES,
                last_error,
            )
            return era5land.empty_hourly_frame(), True
    if not frames:
        LOGGER.warning(
            "location_id=%d returned no usable ERA5-Land values at %.4f, %.4f",
            row.location_id,
            row.lat,
            row.lng,
        )
        return era5land.empty_hourly_frame(), True
    return lcd.concat_frames(frames), False


def _fetch_batch(
    collections: list[ImageCollection],
    rows: list[CityRow],
    concurrency: int,
    city_shard_index: int,
) -> dict[int, tuple[DataFrame, bool]]:
    results: dict[int, tuple[DataFrame, bool]] = {}
    worker_count = max(1, min(concurrency, len(rows)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_fetch_city, collections, row): row for row in rows}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Earth Engine ERA5-Land shard {city_shard_index}",
        ):
            row = futures[future]
            results[int(row.location_id)] = future.result()
    return results


def _filled_rows(
    rows: list[CityRow],
    missing_cells: DataFrame,
    start_year: int,
    end_year: int,
    concurrency: int,
    city_shard_index: int,
) -> DataFrame | None:
    collection = _collection(start_year, end_year)
    results = _fetch_batch(
        _collection_chunks(collection, start_year, end_year),
        rows,
        concurrency,
        city_shard_index,
    )
    if not results or any(had_gap for _, had_gap in results.values()):
        LOGGER.warning("Earth Engine fetch was incomplete; refusing a partial write.")
        return None
    hourly = lcd.concat_frames([frame for frame, _ in results.values()])
    # Daily maximum/mean are only reliable when the local-standard day has all hours.
    daily = nldas.compute_daily_wetbulb(hourly, min_daily_hours=24)
    if daily.empty:
        return None
    daily["date"] = pd.to_datetime(daily["date"])
    filled = daily.merge(missing_cells, on=["location_id", "date"], how="inner")
    if filled.empty:
        return None
    filled["source"] = era5land.ERA5LAND_SOURCE
    filled["date"] = filled["date"].dt.date
    return filled


def process_earth_engine_gapfill(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    cities_csv: str = era5land.EU_CITIES_CSV,
    station_map_csv: str = era5land.EU_STATION_MAP_CSV,
    location_ids: list[int] | None = None,
    min_missing_days: int = MIN_MISSING_DAYS_DEFAULT,
    force: bool = False,
    cell_map_csv: str | None = None,
) -> None:
    """Fill station gaps while preserving station-source precedence at database load."""
    wetbulb_root = f"{out_dir}/wetbulb_data_csv"
    resolved = resolve_gapfill_targets(
        nldas.load_nldas_city_shard(city_shard_index, city_shard_count, cities_csv),
        wetbulb_root,
        start_year,
        end_year,
        city_shard_index,
        city_shard_count,
        location_ids=location_ids,
        min_missing_days=min_missing_days,
        force=force,
        logger=LOGGER,
    )
    if resolved is None:
        return
    shard_df, filesystem, base_path, pending_year_list, _, missing_cells = resolved
    if missing_cells.empty:
        LOGGER.info("No station gaps to fill.")
        return

    offsets = era5land.load_utc_offsets(station_map_csv)
    shard_df = shard_df.merge(offsets, on="location_id", how="left")
    shard_df["utc_offset_hours"] = shard_df["utc_offset_hours"].fillna(
        (shard_df["lng"] / 15.0).round()
    )
    shard_df = era5land.apply_cell_overrides(shard_df, cell_map_csv)
    gapped_ids = set(_gap_years_by_location(missing_cells))
    rows = list(shard_df[shard_df["location_id"].isin(gapped_ids)].itertuples())
    filled = _filled_rows(
        rows,
        missing_cells,
        start_year,
        end_year,
        concurrency,
        city_shard_index,
    )
    if filled is None:
        return
    write_pending_year_batches(
        filled,
        pending_year_list,
        wetbulb_root,
        city_shard_index,
        filesystem,
        base_path,
        file_prefix=GAPFILL_FILE_PREFIX,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    lcd.add_common_shard_args(parser)
    parser.set_defaults(cities_csv=era5land.EU_CITIES_CSV)
    parser.add_argument("--station-map-csv", default=era5land.EU_STATION_MAP_CSV)
    parser.add_argument("--cell-map-csv", default=None)
    parser.add_argument("--credentials", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--concurrency", type=int, default=EE_DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--location-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--min-missing-days", type=int, default=MIN_MISSING_DAYS_DEFAULT
    )
    return parser.parse_args()


def main() -> None:
    """Initialize Earth Engine and execute the requested city shard."""
    load_dotenv(override=False)
    args = _parse_args()
    initialize_earth_engine(args.credentials, args.project)
    process_earth_engine_gapfill(
        args.start_year,
        args.end_year,
        args.out_dir,
        args.city_shard_index,
        args.city_shard_count,
        args.concurrency,
        cities_csv=args.cities_csv,
        station_map_csv=args.station_map_csv,
        location_ids=args.location_ids,
        min_missing_days=args.min_missing_days,
        force=args.force,
        cell_map_csv=args.cell_map_csv,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
