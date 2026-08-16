# Copyright (C) 2026 Kenneth Porter

"""Fill (location_id, date) gaps the EU ISD station pipeline could not produce."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from dotenv import load_dotenv
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Generator

import giovanni
import lcd
import nldas
from gapfill import (
    GAPFILL_FILE_PREFIX,
    MIN_MISSING_DAYS_DEFAULT,
    _gap_years_by_location,
    resolve_gapfill_targets,
)
from lcd import (
    _KELVIN_OFFSET as KELVIN_OFFSET,
)
from lcd import (
    _dewpoint_to_specific_humidity as dewpoint_to_specific_humidity,
)
from partition_io import write_pending_year_batches

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any
type CdsClient = Any
type CityRow = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

ERA5LAND_DATASET = "reanalysis-era5-land-timeseries"
ERA5LAND_VARIABLES = ("2m_temperature", "2m_dewpoint_temperature", "surface_pressure")
ERA5LAND_SOURCE = "era5land"
ERA5LAND_DEFAULT_CONCURRENCY = 2
ERA5LAND_MAX_RETRIES = 3
ERA5LAND_RETRY_DELAY_SECONDS = 30

EU_CITIES_CSV = "cities_eu.csv"
EU_STATION_MAP_CSV = "cities_eu_isd_stations.csv"

_HOURLY_FRAME_COLUMNS = ("location_id", "time", "Tair", "Qair", "PSurf")
_REQUIRED_DOWNLOAD_COLUMNS = ("valid_time", "t2m", "d2m", "sp")
_ARCHIVE_READ_ERRORS = (OSError, ValueError, KeyError, zipfile.BadZipFile)
_SPAN_LOCKS: dict[str, threading.Lock] = {}
_SPAN_LOCKS_GUARD = threading.Lock()


class Era5LandSources(NamedTuple):
    """Optional on-disk inputs: a reusable download cache and a cell override map."""

    cache_dir: str | None = None
    cell_map_csv: str | None = None
    client: CdsClient | None = None
    start_date: str | None = None
    end_date: str | None = None


NO_EXTRA_SOURCES = Era5LandSources()

RETRYABLE_FETCH_ERRORS = (
    RuntimeError,
    ValueError,
    KeyError,
    OSError,
    AssertionError,
)
"""`AssertionError` is included because multiurl validates a completed download
with a bare `assert` on its byte count, so a truncated response from CDS
surfaces as one. Left uncaught it kills the whole run mid-backfill."""


def empty_hourly_frame() -> DataFrame:
    """Return an empty frame with the canonical hourly columns."""
    return pd.DataFrame(columns=list(_HOURLY_FRAME_COLUMNS))


def cds_client() -> CdsClient:
    """Build a CDS API client from the ambient CDSAPI_URL/CDSAPI_KEY settings."""
    cdsapi = importlib.import_module("cdsapi")
    return cdsapi.Client(retry_max=3, sleep_max=60, timeout=60)


class _BoundedCdsClient:
    def __init__(
        self,
        client: CdsClient,
        start_date: str | None,
        end_date: str | None,
    ) -> None:
        self.client = client
        self.start_date = start_date
        self.end_date = end_date

    def retrieve(self, dataset: str, request: dict[str, Any], target: str) -> None:
        bounded_request = request.copy()
        request_start, request_end = request["date"][0].split("/")
        if self.start_date is not None:
            request_start = max(request_start, self.start_date)
        if self.end_date is not None:
            request_end = min(request_end, self.end_date)
        request_start = (pd.Timestamp(request_start) - pd.Timedelta(days=2)).strftime(
            "%Y-%m-%d"
        )
        request_end = (pd.Timestamp(request_end) + pd.Timedelta(days=2)).strftime(
            "%Y-%m-%d"
        )
        bounded_request["date"] = [f"{request_start}/{request_end}"]
        self.client.retrieve(dataset, bounded_request, target)


def read_era5land_download(target: str) -> DataFrame:
    """Read a CDS timeseries download, which is a zip of one CSV per variable group.

    The API returns a zip archive even when the request asks for `data_format:
    csv`, so the members are merged on `valid_time` back into the single wide
    frame the rest of this module expects.
    """
    if not zipfile.is_zipfile(target):
        return pd.read_csv(target)

    with zipfile.ZipFile(target) as archive:
        frames = [
            pd.read_csv(archive.open(name))
            for name in sorted(archive.namelist())
            if name.endswith(".csv")
        ]
    if not frames:
        message = f"ERA5-Land download contained no CSV member: {target}"
        raise ValueError(message)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame, on="valid_time", how="inner", suffixes=("", "_duplicate")
        )
    return merged.loc[:, ~merged.columns.str.endswith("_duplicate")]


def span_filename(lat: float, lng: float, start_year: int, end_year: int) -> str:
    """Name the download that holds one city's span; also the cache key."""
    return f"era5land_{lat}_{lng}_{start_year}_{end_year}.csv"


def _cached_span(target: str) -> DataFrame | None:
    """Return an already-downloaded span, or `None` when absent or unusable.

    A download interrupted mid-write is not a valid zip and would otherwise be
    read as a plain CSV of garbage, so the columns are checked before the
    cached copy is trusted.
    """
    if not Path(target).exists():
        return None
    try:
        frame = read_era5land_download(target)
    except _ARCHIVE_READ_ERRORS:
        LOGGER.warning("Re-fetching unreadable cached download: %s", Path(target).name)
        return None
    if not set(_REQUIRED_DOWNLOAD_COLUMNS).issubset(frame.columns):
        LOGGER.warning("Re-fetching incomplete cached download: %s", Path(target).name)
        return None
    return frame


@contextlib.contextmanager
def download_cache(path: str | None) -> Generator[str]:
    """Yield a persistent download cache when configured, else a scratch dir."""
    if path is None:
        with tempfile.TemporaryDirectory() as scratch:
            yield scratch
        return
    Path(path).mkdir(parents=True, exist_ok=True)
    yield path


def fetch_city_span(
    client: CdsClient,
    *,
    lat: float,
    lng: float,
    start_year: int,
    end_year: int,
    download_dir: str,
) -> DataFrame:
    """Retrieve one city's ERA5-Land hourly point time-series for a year span."""
    target = str(Path(download_dir) / span_filename(lat, lng, start_year, end_year))
    with _SPAN_LOCKS_GUARD:
        span_lock = _SPAN_LOCKS.setdefault(target, threading.Lock())
    with span_lock:
        cached = _cached_span(target)
        if cached is not None:
            return cached

        request = {
            "variable": list(ERA5LAND_VARIABLES),
            "location": {"latitude": lat, "longitude": lng},
            "date": [f"{start_year}-01-01/{end_year}-12-31"],
            "data_format": "csv",
        }
        with tempfile.NamedTemporaryFile(
            dir=download_dir,
            prefix=f".{Path(target).name}.",
            suffix=".csv",
            delete=False,
        ) as temporary:
            temporary_target = temporary.name
        try:
            client.retrieve(ERA5LAND_DATASET, request, temporary_target)
            frame = read_era5land_download(temporary_target)
            if not set(_REQUIRED_DOWNLOAD_COLUMNS).issubset(frame.columns):
                message = f"ERA5-Land download missing required columns: {target}"
                raise ValueError(message)
            Path(temporary_target).replace(target)
            return frame
        finally:
            Path(temporary_target).unlink(missing_ok=True)


def _hourly_frame_from_era5land(
    location_id: int,
    raw: DataFrame,
    utc_offset_hours: float,
) -> DataFrame:
    """Convert one city's raw ERA5-Land CSV rows to `location_id,time,Tair,Qair,PSurf`."""
    if raw.empty:
        return empty_hourly_frame()

    time_utc = pd.to_datetime(raw["valid_time"], utc=True).dt.tz_localize(None)
    tair_k = pd.to_numeric(raw["t2m"], errors="coerce")
    dewpoint_k = pd.to_numeric(raw["d2m"], errors="coerce")
    psurf_pa = pd.to_numeric(raw["sp"], errors="coerce")
    pressure_hpa = psurf_pa / 100.0
    qair = dewpoint_to_specific_humidity(dewpoint_k - KELVIN_OFFSET, pressure_hpa)

    local_time = time_utc + pd.to_timedelta(int(utc_offset_hours), unit="h")
    return pd.DataFrame(
        {
            "location_id": location_id,
            "time": local_time.to_numpy(),
            "Tair": tair_k.to_numpy(dtype="float64"),
            "Qair": qair.to_numpy(dtype="float64"),
            "PSurf": psurf_pa.to_numpy(dtype="float64"),
        },
    ).dropna(subset=["Tair", "Qair", "PSurf"])


def _fetch_span_with_retries(
    client: CdsClient,
    row: CityRow,
    start_year: int,
    end_year: int,
    download_dir: str,
) -> DataFrame | None:
    """Fetch one span, retrying transient CDS failures; `None` once retries run out."""
    last_error: Exception | None = None
    for attempt in range(1, ERA5LAND_MAX_RETRIES + 1):
        try:
            return fetch_city_span(
                client,
                lat=row.lat,
                lng=row.lng,
                start_year=start_year,
                end_year=end_year,
                download_dir=download_dir,
            )
        except RETRYABLE_FETCH_ERRORS as exc:
            last_error = exc
            if attempt < ERA5LAND_MAX_RETRIES:
                time.sleep(ERA5LAND_RETRY_DELAY_SECONDS * attempt)
    LOGGER.warning(
        "Giving up on location_id=%d %d-%d after %d attempt(s): %s",
        row.location_id,
        start_year,
        end_year,
        ERA5LAND_MAX_RETRIES,
        last_error,
    )
    return None


def _span_frame(
    row: CityRow, raw: DataFrame, start_year: int, end_year: int
) -> DataFrame:
    """Convert one fetched span, warning when a land-only cell yields nothing usable."""
    frame = _hourly_frame_from_era5land(row.location_id, raw, row.utc_offset_hours)
    if frame.empty and not raw.empty:
        LOGGER.warning(
            "location_id=%d %d-%d returned %d row(s) but none usable. "
            "ERA5-Land is land-only, so this cell is probably all sea; "
            "map the city to a nearby land cell.",
            row.location_id,
            start_year,
            end_year,
            len(raw),
        )
    return frame


def _fetch_city_gaps(
    client: CdsClient,
    row: CityRow,
    gap_years: list[int],
    download_dir: str,
) -> tuple[DataFrame, bool]:
    """Fetch every contiguous gap-year range for one city; return (frame, had_gap)."""
    year_frames: list[DataFrame] = []
    had_gap = False
    for start_year, end_year in giovanni.contiguous_year_ranges(gap_years):
        raw = _fetch_span_with_retries(client, row, start_year, end_year, download_dir)
        if raw is None:
            had_gap = True
            continue
        frame = _span_frame(row, raw, start_year, end_year)
        if not frame.empty:
            year_frames.append(frame)

    if not year_frames:
        return empty_hourly_frame(), had_gap
    return lcd.concat_frames(year_frames), had_gap


def _fetch_gaps_batch(
    rows: list[CityRow],
    gap_years_by_location: dict[int, list[int]],
    client: CdsClient,
    download_dir: str,
    worker_count: int,
    city_shard_index: int,
) -> dict[int, tuple[DataFrame, bool]]:
    """Fetch each gapped city's ERA5-Land data concurrently, `worker_count` at a time."""
    results: dict[int, tuple[DataFrame, bool]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fetch_city_gaps,
                client,
                row,
                gap_years_by_location[row.location_id],
                download_dir,
            ): row.location_id
            for row in rows
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"ERA5-Land gap-fill city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()
    return results


def _log_cache_reuse(
    gapped_rows: list[CityRow],
    gap_years_by_location: dict[int, list[int]],
    download_dir: str,
) -> None:
    """Report how much of this run's fetch is already sitting in the cache."""
    total = cached = 0
    for row in gapped_rows:
        for start_year, end_year in giovanni.contiguous_year_ranges(
            gap_years_by_location[row.location_id]
        ):
            total += 1
            name = span_filename(row.lat, row.lng, start_year, end_year)
            cached += (Path(download_dir) / name).exists()
    LOGGER.info(
        "Download cache holds %d/%d span(s); %d still to fetch from CDS.",
        cached,
        total,
        total - cached,
    )


def _fetch_filled_rows(
    gapped_rows: list[CityRow],
    gap_years_by_location: dict[int, list[int]],
    missing_cells: DataFrame,
    concurrency: int,
    city_shard_index: int,
    city_shard_count: int,
    start_year: int,
    end_year: int,
    cache_dir: str | None,
    client: CdsClient | None = None,
) -> DataFrame | None:
    """Fetch ERA5-Land data for every gapped city and merge it against `missing_cells`."""
    source_client = client or cds_client()
    worker_count = max(1, min(concurrency, len(gapped_rows)))
    with download_cache(cache_dir) as download_dir:
        _log_cache_reuse(gapped_rows, gap_years_by_location, download_dir)
        city_results = _fetch_gaps_batch(
            gapped_rows,
            gap_years_by_location,
            source_client,
            download_dir,
            worker_count,
            city_shard_index,
        )

    hourly_frames = [df for df, _ in city_results.values()]
    shard_had_gap = any(gap for _, gap in city_results.values())
    if not hourly_frames:
        return None
    hourly_df = lcd.concat_frames(hourly_frames)
    daily_df = nldas.compute_daily_wetbulb(hourly_df)
    if daily_df.empty:
        return None

    if shard_had_gap:
        LOGGER.warning(
            "city_shard=%d/%d: one or more cities had a CDS fetch gap while "
            "gap-filling %d-%d; skipping the parquet write for these "
            "pending year(s) so a future run retries them instead of "
            "treating incomplete data as done.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return None

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    filled = daily_df.merge(missing_cells, on=["location_id", "date"], how="inner")
    if filled.empty:
        LOGGER.info(
            "city_shard=%d/%d: fetched ERA5-Land data for %d-%d produced no "
            "rows matching a known gap; nothing to write.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return None
    filled["source"] = ERA5LAND_SOURCE
    filled["date"] = filled["date"].dt.date
    return filled


def apply_cell_overrides(shard_df: DataFrame, cell_map_csv: str | None) -> DataFrame:
    """Point selected cities at a different ERA5-Land cell than their own centre.

    Coastal cities can sit in a cell ERA5-Land treats as sea, which yields an
    all-NaN series. The override map moves just the fetch coordinate; the city's
    real lat/lng is untouched everywhere else.
    """
    if not cell_map_csv:
        return shard_df
    path = Path(cell_map_csv)
    if not path.exists():
        LOGGER.warning(
            "Cell map %s does not exist; falling back to city centres.", cell_map_csv
        )
        return shard_df

    overrides = pd.read_csv(path, usecols=["location_id", "lat", "lng"])
    merged = shard_df.merge(
        overrides, on="location_id", how="left", suffixes=("", "_cell")
    )
    replaced = merged["lat_cell"].notna() & merged["lng_cell"].notna()
    merged.loc[replaced, "lat"] = merged.loc[replaced, "lat_cell"]
    merged.loc[replaced, "lng"] = merged.loc[replaced, "lng_cell"]
    LOGGER.info(
        "Applied ERA5-Land cell override for %d/%d city(ies).",
        int(replaced.sum()),
        len(overrides),
    )
    return merged.drop(columns=["lat_cell", "lng_cell"])


def load_utc_offsets(station_map_csv: str) -> DataFrame:
    """Load each city's standard UTC offset from the EU ISD station crosswalk."""
    path = Path(station_map_csv)
    if not path.exists():
        return pd.DataFrame(
            {
                "location_id": pd.Series(dtype="int64"),
                "utc_offset_hours": pd.Series(dtype="float64"),
            },
        )
    return pd.read_csv(
        path, usecols=["location_id", "utc_offset_hours"]
    ).drop_duplicates(subset="location_id")


def process_era5land_gapfill(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    cities_csv: str = EU_CITIES_CSV,
    station_map_csv: str = EU_STATION_MAP_CSV,
    location_ids: list[int] | None = None,
    min_missing_days: int = MIN_MISSING_DAYS_DEFAULT,
    force: bool = False,
    sources: Era5LandSources = NO_EXTRA_SOURCES,
) -> None:
    """Fill (location_id, date) cells the ISD pipeline could not produce, via ERA5-Land."""
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
        start_date=sources.start_date,
        end_date=sources.end_date,
    )
    if resolved is None:
        return
    shard_df, filesystem, base_path, pending_year_list, _, missing_cells = resolved
    if missing_cells.empty:
        LOGGER.info(
            "city_shard=%d/%d: no material gaps found in ISD output for "
            "%d-%d; nothing to fill.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return

    gap_years_by_location = _gap_years_by_location(missing_cells)
    gapped_ids = set(gap_years_by_location)
    LOGGER.info(
        "ERA5-Land gap-fill city_shard=%d/%d: %d/%d city(ies) have a gap "
        "across %d cell(s).",
        city_shard_index,
        city_shard_count,
        len(gapped_ids),
        len(shard_df),
        len(missing_cells),
    )

    offsets = load_utc_offsets(station_map_csv)
    shard_df = shard_df.merge(offsets, on="location_id", how="left")
    shard_df["utc_offset_hours"] = shard_df["utc_offset_hours"].fillna(
        (shard_df["lng"] / 15.0).round()
    )
    shard_df = apply_cell_overrides(shard_df, sources.cell_map_csv)
    gapped_rows = list(shard_df[shard_df["location_id"].isin(gapped_ids)].itertuples())

    filled = _fetch_filled_rows(
        gapped_rows,
        gap_years_by_location,
        missing_cells,
        concurrency,
        city_shard_index,
        city_shard_count,
        start_year,
        end_year,
        sources.cache_dir,
        sources.client,
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
        merge_existing=sources.start_date is not None or sources.end_date is not None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    lcd.add_common_shard_args(parser)
    parser.set_defaults(cities_csv=EU_CITIES_CSV)
    parser.add_argument("--station-map-csv", type=str, default=EU_STATION_MAP_CSV)
    parser.add_argument("--concurrency", type=int, default=ERA5LAND_DEFAULT_CONCURRENCY)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cell-map-csv",
        type=str,
        default=None,
        help=(
            "CSV of location_id,lat,lng overriding the ERA5-Land fetch "
            "coordinate for cities whose own cell has no land."
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help=(
            "Persist CDS downloads here and reuse any span already present, so "
            "an interrupted run resumes instead of re-fetching. Defaults to a "
            "scratch directory that is discarded on exit."
        ),
    )
    parser.add_argument(
        "--location-ids",
        type=int,
        nargs="+",
        default=None,
        help="Restrict gap-filling to these location_id(s), for a targeted run.",
    )
    parser.add_argument(
        "--min-missing-days",
        type=int,
        default=MIN_MISSING_DAYS_DEFAULT,
        help=(
            "Only fill a city-year missing at least this many days. Pass 1 "
            "to fill every gap."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Execute the ERA5-Land gap-fill pipeline."""
    load_dotenv(override=False)
    exit_code = 0
    try:
        args = _parse_args()
        client = _BoundedCdsClient(
            cds_client(),
            args.start_date,
            args.end_date,
        )
        process_era5land_gapfill(
            start_year=args.start_year,
            end_year=args.end_year,
            out_dir=args.out_dir,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
            concurrency=args.concurrency,
            cities_csv=args.cities_csv,
            station_map_csv=args.station_map_csv,
            location_ids=args.location_ids,
            min_missing_days=args.min_missing_days,
            force=args.force,
            sources=Era5LandSources(
                cache_dir=args.download_dir,
                cell_map_csv=args.cell_map_csv,
                client=client,
                start_date=args.start_date,
                end_date=args.end_date,
            ),
        )
    except KeyboardInterrupt:
        exit_code = 130
        LOGGER.warning("ERA5-Land gap-fill interrupted by user.")
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
        AssertionError,
    ):
        exit_code = 1
        LOGGER.exception("ERA5-Land gap-fill failed.")

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
