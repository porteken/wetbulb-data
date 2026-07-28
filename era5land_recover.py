"""Rebuild the EU ERA5-Land gap-fill parquet from snapshotted CDS downloads.

The live `era5land.py` run holds every city in memory and writes only at the
very end, so a crash discards the whole fetch. This reconstructs that final
write from the CSVs mirrored out of the run's temp directory.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any, cast

import giovanni
import lcd
import nldas
from era5land import (
    ERA5LAND_SOURCE,
    EU_CITIES_CSV,
    EU_STATION_MAP_CSV,
    _hourly_frame_from_era5land,
    _load_utc_offsets,
    read_era5land_download,
)
from gapfill import GAPFILL_FILE_PREFIX, _gap_years_by_location, resolve_gapfill_targets
from partition_io import write_pending_year_batches

pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any
type CityRow = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

DEFAULT_SNAPSHOT = "/home/kenneth-porter/era5land_gapfill_snapshot"
LIVE_RUN_PID = 2853596


def _live_run_is_active(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _span_csv(snapshot: Path, lat: float, lng: float, start: int, end: int) -> Path:
    return snapshot / f"era5land_{lat}_{lng}_{start}_{end}.csv"


def _read_span(path: Path) -> DataFrame | None:
    try:
        return read_era5land_download(str(path))
    except OSError, ValueError, KeyError, zipfile.BadZipFile:
        LOGGER.warning("Unreadable or truncated snapshot file: %s", path.name)
        return None


def _city_frame(
    snapshot: Path,
    row: CityRow,
    gap_years: list[int],
) -> tuple[DataFrame | None, list[tuple[int, int]]]:
    """Return one city's hourly frame plus the spans absent from the snapshot."""
    frames: list[DataFrame] = []
    missing_spans: list[tuple[int, int]] = []
    for start_year, end_year in giovanni.contiguous_year_ranges(gap_years):
        path = _span_csv(snapshot, row.lat, row.lng, start_year, end_year)
        raw = _read_span(path) if path.exists() else None
        if raw is None:
            missing_spans.append((start_year, end_year))
            continue
        frame = _hourly_frame_from_era5land(row.location_id, raw, row.utc_offset_hours)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return None, missing_spans
    return lcd.concat_frames(frames), missing_spans


def _resolve_gapped_rows(
    wetbulb_root: str,
    start_year: int,
    end_year: int,
    min_missing_days: int,
    cities_csv: str,
    station_map_csv: str,
) -> tuple[list[CityRow], dict[int, list[int]], Any, str, list[int], DataFrame] | None:
    """Resolve this shard's gapped cities, or `None` when there is nothing to fill."""
    resolved = resolve_gapfill_targets(
        nldas.load_nldas_city_shard(0, 1, cities_csv),
        wetbulb_root,
        start_year,
        end_year,
        0,
        1,
        location_ids=None,
        min_missing_days=min_missing_days,
        force=False,
        logger=LOGGER,
    )
    if resolved is None:
        LOGGER.info("Nothing pending; the gap-fill output is already in place.")
        return None

    shard_df, filesystem, base_path, pending_year_list, _, missing_cells = resolved
    if missing_cells.empty:
        LOGGER.info("No material gaps to fill.")
        return None

    gap_years_by_location = _gap_years_by_location(missing_cells)
    offsets = _load_utc_offsets(station_map_csv)
    shard_df = shard_df.merge(offsets, on="location_id", how="left")
    shard_df["utc_offset_hours"] = shard_df["utc_offset_hours"].fillna(
        (shard_df["lng"] / 15.0).round()
    )
    gapped_rows = list(
        shard_df[shard_df["location_id"].isin(set(gap_years_by_location))].itertuples()
    )
    return (
        gapped_rows,
        gap_years_by_location,
        filesystem,
        base_path,
        pending_year_list,
        missing_cells,
    )


def _filled_from_frames(
    hourly_frames: list[DataFrame],
    missing_cells: DataFrame,
) -> DataFrame | None:
    """Aggregate recovered hourly rows to the daily gap cells they fill."""
    if not hourly_frames:
        LOGGER.error("Snapshot yielded no usable rows.")
        return None

    daily_df = nldas.compute_daily_wetbulb(lcd.concat_frames(hourly_frames))
    if daily_df.empty:
        LOGGER.error("Daily wet-bulb aggregation produced no rows.")
        return None

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    filled = daily_df.merge(missing_cells, on=["location_id", "date"], how="inner")
    if filled.empty:
        LOGGER.error("Recovered data matched no known gap cell; nothing to write.")
        return None
    filled["source"] = ERA5LAND_SOURCE
    filled["date"] = filled["date"].dt.date
    return filled


def recover(
    snapshot_dir: str,
    start_year: int,
    end_year: int,
    out_dir: str,
    min_missing_days: int,
    cities_csv: str,
    station_map_csv: str,
    *,
    allow_partial: bool,
    dry_run: bool,
) -> int:
    """Rebuild and write the gap-fill parquet; return a process exit code."""
    snapshot = Path(snapshot_dir)
    if not snapshot.is_dir():
        LOGGER.error("Snapshot directory does not exist: %s", snapshot)
        return 1

    wetbulb_root = f"{out_dir}/wetbulb_data_csv"
    targets = _resolve_gapped_rows(
        wetbulb_root,
        start_year,
        end_year,
        min_missing_days,
        cities_csv,
        station_map_csv,
    )
    if targets is None:
        return 0
    (
        gapped_rows,
        gap_years_by_location,
        filesystem,
        base_path,
        pending_year_list,
        missing_cells,
    ) = targets

    hourly_frames: list[DataFrame] = []
    incomplete: dict[int, list[tuple[int, int]]] = {}
    for row in gapped_rows:
        frame, missing_spans = _city_frame(
            snapshot, row, gap_years_by_location[row.location_id]
        )
        if frame is not None:
            hourly_frames.append(frame)
        if missing_spans:
            incomplete[row.location_id] = missing_spans

    expected_spans = sum(
        len(giovanni.contiguous_year_ranges(y)) for y in gap_years_by_location.values()
    )
    recovered_spans = expected_spans - sum(len(v) for v in incomplete.values())
    LOGGER.info(
        "Snapshot covers %d/%d span(s) across %d/%d gapped city(ies).",
        recovered_spans,
        expected_spans,
        len(gapped_rows) - len(incomplete),
        len(gapped_rows),
    )

    if incomplete:
        LOGGER.warning(
            "%d city(ies) are missing at least one span. Re-fetch them with:\n"
            "  uv run python era5land.py --cities-csv %s --station-map-csv %s "
            "--start-year %d --end-year %d --out-dir %s --city-shard-index 0 "
            "--city-shard-count 1 --concurrency 2 --min-missing-days %d "
            "--location-ids %s",
            len(incomplete),
            cities_csv,
            station_map_csv,
            start_year,
            end_year,
            out_dir,
            min_missing_days,
            " ".join(str(i) for i in sorted(incomplete)),
        )
        if not allow_partial:
            LOGGER.error(
                "Refusing to write a partial gap-fill. Re-fetch the cities above, "
                "then re-run; pass --allow-partial to write what is available."
            )
            return 2

    filled = _filled_from_frames(hourly_frames, missing_cells)
    if filled is None:
        return 1

    LOGGER.info(
        "Rebuilt %d gap-fill row(s) across %d location(s) and %d pending year(s).",
        len(filled),
        filled["location_id"].nunique(),
        len(pending_year_list),
    )
    if dry_run:
        LOGGER.info("--dry-run: not writing.")
        return 0

    write_pending_year_batches(
        filled,
        pending_year_list,
        wetbulb_root,
        0,
        filesystem,
        base_path,
        file_prefix=GAPFILL_FILE_PREFIX,
    )
    LOGGER.info("Wrote gap-fill partitions under %s", wetbulb_root)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--cities-csv", default=EU_CITIES_CSV)
    parser.add_argument("--station-map-csv", default=EU_STATION_MAP_CSV)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--out-dir", default="eu")
    parser.add_argument("--min-missing-days", type=int, default=19)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--ignore-live-run",
        action="store_true",
        help=f"Proceed even while pid {LIVE_RUN_PID} is still running.",
    )
    return parser.parse_args()


def main() -> None:
    """Rebuild the ERA5-Land gap-fill output from the download snapshot."""
    args = _parse_args()
    if (
        _live_run_is_active(LIVE_RUN_PID)
        and not args.ignore_live_run
        and not args.dry_run
    ):
        LOGGER.error(
            "The live gap-fill run (pid %d) is still active; it will write its own "
            "output on completion. Pass --ignore-live-run to override.",
            LIVE_RUN_PID,
        )
        sys.exit(3)
    sys.exit(
        recover(
            args.snapshot_dir,
            args.start_year,
            args.end_year,
            args.out_dir,
            args.min_missing_days,
            args.cities_csv,
            args.station_map_csv,
            allow_partial=args.allow_partial,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
