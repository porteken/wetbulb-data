"""Fill (location_id, date) gaps the ISD station pipeline could not produce.

ISD (`isd.py`) stays the primary daily wet-bulb source: real station
observations beat a gridded model on the extreme-value days this dataset
exists for (a 15-city LCD-vs-NLDAS pilot showed NLDAS's p95 daily-max error
is ~4.4 C -- an 8 F swing -- concentrated on exactly the high-humidity days
this dataset cares about). But some station-years are thin: the 20/24-hour
daily coverage gate in `nldas.compute_daily_wetbulb` drops a day whenever
the assigned station didn't report enough hours, which happens for ~77
cities in 2000-2010 (pre-ASOS/AWOS automation, no better station existed
then either) and a handful of cities more recently.

This module detects those holes by diffing each pending year's calendar
against the `wetbulb_batch_*` parquet ISD already wrote, keeps only the
city-years whose gap is material (>= `--min-missing-days`; see
`MIN_MISSING_DAYS_DEFAULT` -- without that floor the near-universal 1-3
missing days per station-year degenerate the fetch set into a full
multi-decade backfill), and fetches those years from NLDAS-2 via the
Giovanni API (reusing
`giovanni.py`'s fetch/retry machinery and its NLDAS_FORA0125_H_2_0
variables, so fill values are numerically consistent with the rest of the
pipeline), and writes the result to a *separate* `wetbulb_fill_batch_*`
parquet file with `source='nldas'` stamped on every row. Gap-fill rows never
share a batch-0 file with ISD's own `wetbulb_batch_*` output, so resume
tracking (`partition_io.pending_years`) stays independent per source and a
later ISD re-run can never be shadowed by a stale fill file.

ISD-produced rows always win when both exist for the same cell -- enforced
here by only ever writing rows that inner-join against a detected gap, and
again at load time (`load.py`'s upsert prefers `source='isd'`; see its
`_upsert_from_staging`).

Falls back to `nldas.py`'s raw hourly-granule downloads
(`pipeline.py --wetbulb-source granules --months ...`) if the Giovanni API
ever regresses; that path is slower (whole-year hourly downloads instead of
one multi-year point request) but requires no dependency on this module.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from tqdm.auto import tqdm

import giovanni
import lcd
import nldas
from partition_io import pending_years, write_pending_year_batches
from shards import resolve_filesystem

pd = cast("Any", importlib.import_module("pandas"))
pa = cast("Any", importlib.import_module("pyarrow"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))
fs_module = cast("Any", importlib.import_module("pyarrow.fs"))
requests = cast("Any", importlib.import_module("requests"))

type DataFrame = Any
type Session = Any
type CityRow = Any
type Filesystem = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

ISD_FILE_PREFIX = "wetbulb"
GAPFILL_FILE_PREFIX = "wetbulb_fill"
GAPFILL_SOURCE = "nldas"
DEFAULT_CONCURRENCY = 8
MIN_MISSING_DAYS_DEFAULT = 19


def _empty_cells_frame() -> DataFrame:
    return pd.DataFrame(
        {
            "location_id": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
        },
    )


def _existing_cells(
    filesystem: Filesystem,
    base_path: str,
    year: int,
    location_ids: set[int],
) -> DataFrame:
    """Return the (location_id, date) cells the ISD pipeline already wrote for `year`."""
    partition_dir = f"{base_path}/year={year}"
    try:
        file_infos = filesystem.get_file_info(fs_module.FileSelector(partition_dir))
    except OSError, pa.ArrowException:
        return _empty_cells_frame()

    paths = [
        file_info.path
        for file_info in file_infos
        if file_info.type == fs_module.FileType.File
        and Path(file_info.path).name.startswith(f"{ISD_FILE_PREFIX}_batch_")
    ]
    if not paths:
        return _empty_cells_frame()

    frames = [
        pq.read_table(
            path, columns=["location_id", "date"], filesystem=filesystem
        ).to_pandas()
        for path in paths
    ]
    existing = pd.concat(frames, ignore_index=True)
    existing["date"] = pd.to_datetime(existing["date"])
    return existing[existing["location_id"].isin(location_ids)]


def find_missing_cells(
    location_ids: list[int],
    years: list[int],
    filesystem: Filesystem,
    base_path: str,
) -> DataFrame:
    """Return every (location_id, date) cell in `years` the ISD pipeline hasn't written.

    Builds the full calendar for `location_ids` x `years` and diffs it
    against whatever `wetbulb_batch_*` parquet already exists under each
    `year=YYYY` partition -- never against this module's own
    `wetbulb_fill_batch_*` output, so a gap is always defined relative to
    ISD, not to a previous gap-fill run.
    """
    if not location_ids or not years:
        return _empty_cells_frame()

    location_id_set = set(location_ids)
    missing_frames: list[DataFrame] = []
    for year in years:
        calendar = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        expected = pd.MultiIndex.from_product(
            [location_ids, calendar],
            names=["location_id", "date"],
        ).to_frame(index=False)
        existing = _existing_cells(filesystem, base_path, year, location_id_set)
        if existing.empty:
            missing_frames.append(expected)
            continue
        merged = expected.merge(
            existing.drop_duplicates(),
            on=["location_id", "date"],
            how="left",
            indicator=True,
        )
        missing_frames.append(
            merged.loc[merged["_merge"] == "left_only", ["location_id", "date"]],
        )

    return pd.concat(missing_frames, ignore_index=True)


def _gap_years_by_location(missing_cells: DataFrame) -> dict[int, list[int]]:
    """Return each gapped city's sorted list of years that have >=1 missing cell."""
    years = missing_cells.assign(year=missing_cells["date"].dt.year)
    return (
        years.groupby("location_id")["year"]
        .apply(lambda values: sorted(values.unique().tolist()))
        .to_dict()
    )


def _filter_material_gaps(missing_cells: DataFrame, min_missing_days: int) -> DataFrame:
    """Keep only cells in (city, year) pairs missing >= `min_missing_days` days.

    See `MIN_MISSING_DAYS_DEFAULT` for why small gaps aren't worth filling.
    A kept city-year keeps ALL of its missing cells (including the handful of
    scattered days), since the fetch covers the whole year anyway.
    """
    if missing_cells.empty or min_missing_days <= 1:
        return missing_cells
    cells = missing_cells.assign(year=missing_cells["date"].dt.year)
    gap_sizes = cells.groupby(["location_id", "year"])["date"].transform("size")
    return (
        cells[gap_sizes >= min_missing_days].drop(columns="year").reset_index(drop=True)
    )


def _fetch_gapfill_batch(
    rows: list[CityRow],
    gap_years_by_location: dict[int, list[int]],
    session: Session,
    token_manager: giovanni.TokenManagerProtocol,
    worker_count: int,
    city_shard_index: int,
) -> dict[int, tuple[DataFrame, bool]]:
    """Fetch each gapped city's own contiguous gap-year ranges concurrently.

    Unlike `giovanni._fetch_cities_batch` (every city in a shard fetches the
    same pending years), each city here has its own gap-year set, so
    `contiguous_year_ranges` is computed per row rather than once for the
    whole batch.
    """
    results: dict[int, tuple[DataFrame, bool]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                giovanni.fetch_city_series,
                session,
                token_manager,
                row,
                giovanni.contiguous_year_ranges(
                    gap_years_by_location[row.location_id],
                ),
            ): row.location_id
            for row in rows
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"NLDAS gap-fill city_shard {city_shard_index}",
        ):
            results[futures[future]] = future.result()
    return results


def _fetch_gaps_with_retries(
    rows_by_location: dict[int, CityRow],
    gap_years_by_location: dict[int, list[int]],
    session: Session,
    token_manager: giovanni.TokenManagerProtocol,
    worker_count: int,
    city_shard_index: int,
    city_shard_count: int,
) -> dict[int, tuple[DataFrame, bool]]:
    """Fetch every gapped city, retrying just the ones with a fetch gap."""
    pending_rows = list(rows_by_location.values())
    city_results: dict[int, tuple[DataFrame, bool]] = {}
    for attempt in range(1, giovanni.GIOVANNI_SHARD_RETRY_ATTEMPTS + 1):
        city_results.update(
            _fetch_gapfill_batch(
                pending_rows,
                gap_years_by_location,
                session,
                token_manager,
                worker_count,
                city_shard_index,
            ),
        )
        pending_rows = [
            rows_by_location[location_id]
            for location_id, (_, gap) in city_results.items()
            if gap
        ]
        if not pending_rows or attempt == giovanni.GIOVANNI_SHARD_RETRY_ATTEMPTS:
            break
        LOGGER.warning(
            "city_shard=%d/%d: %d/%d gapped city(ies) still had a fetch gap "
            "after attempt %d/%d; retrying just those after a %ds pause.",
            city_shard_index,
            city_shard_count,
            len(pending_rows),
            len(rows_by_location),
            attempt,
            giovanni.GIOVANNI_SHARD_RETRY_ATTEMPTS,
            giovanni.GIOVANNI_SHARD_RETRY_DELAY_SECONDS,
        )
        time.sleep(giovanni.GIOVANNI_SHARD_RETRY_DELAY_SECONDS)
    return city_results


def _fetch_filled_rows(
    gapped_rows: DataFrame,
    missing_cells: DataFrame,
    gap_years_by_location: dict[int, list[int]],
    concurrency: int,
    city_shard_index: int,
    city_shard_count: int,
    start_year: int,
    end_year: int,
) -> DataFrame | None:
    """Fetch NLDAS-2 data for every gapped city and merge it against `missing_cells`.

    Returns None if any city had a fetch gap (the whole write is skipped so
    a future run retries them, mirroring `giovanni.process_giovanni`'s
    identical rationale) or if there was nothing to write.
    """
    token_manager = giovanni.TokenManager.from_env()
    session = requests.Session()
    worker_count = max(1, min(concurrency, len(gapped_rows)))
    rows_by_location = {row.location_id: row for row in gapped_rows.itertuples()}
    city_results = _fetch_gaps_with_retries(
        rows_by_location,
        gap_years_by_location,
        session,
        token_manager,
        worker_count,
        city_shard_index,
        city_shard_count,
    )

    hourly_frames = [df for df, _ in city_results.values()]
    shard_had_gap = any(gap for _, gap in city_results.values())
    if not hourly_frames:
        return None
    hourly_df = pd.concat(hourly_frames, ignore_index=True)
    if "utc_offset_hours" in gapped_rows.columns and not hourly_df.empty:
        hourly_df = hourly_df.merge(
            gapped_rows[["location_id", "utc_offset_hours"]],
            on="location_id",
            how="left",
            validate="many_to_one",
        )
    daily_df = nldas.compute_daily_wetbulb(hourly_df)
    if daily_df.empty:
        return None

    if shard_had_gap:
        LOGGER.warning(
            "city_shard=%d/%d: one or more cities had a Giovanni fetch gap "
            "while gap-filling %d-%d; skipping the parquet write for these "
            "pending year(s) so a future run (once the API recovers) retries "
            "them instead of treating incomplete data as done.",
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
            "city_shard=%d/%d: fetched NLDAS-2 data for %d-%d produced no rows "
            "matching a known gap; nothing to write.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return None
    filled["source"] = GAPFILL_SOURCE
    filled["date"] = filled["date"].dt.date
    return filled


def resolve_gapfill_targets(
    shard_df: DataFrame,
    wetbulb_root: str,
    start_year: int,
    end_year: int,
    city_shard_index: int,
    city_shard_count: int,
    *,
    location_ids: list[int] | None,
    min_missing_days: int,
    force: bool,
    logger: logging.Logger,
) -> tuple[DataFrame, Any, str, list[int], DataFrame, DataFrame] | None:
    """Narrow a shard to its pending years and gap cells; `None` if there's no work.

    Shared by `process_gapfill` and `era5land.process_era5land_gapfill`, which
    diverge only in how they report the cells filtered out as immaterial.
    """
    if location_ids is not None:
        shard_df = shard_df[shard_df["location_id"].isin(location_ids)]
    if shard_df.empty:
        logger.info(
            "No cities found for shard %s/%s.", city_shard_index, city_shard_count
        )
        return None

    filesystem, base_path = resolve_filesystem(wetbulb_root)
    pending_year_list = pending_years(
        range(start_year, end_year + 1),
        wetbulb_root,
        city_shard_index,
        filesystem,
        base_path,
        file_prefix=GAPFILL_FILE_PREFIX,
        force=force,
    )
    if not pending_year_list:
        logger.info(
            "city_shard=%d/%d: gap-fill years %d-%d already present.",
            city_shard_index,
            city_shard_count,
            start_year,
            end_year,
        )
        return None

    all_missing_cells = find_missing_cells(
        shard_df["location_id"].tolist(), pending_year_list, filesystem, base_path
    )
    return (
        shard_df,
        filesystem,
        base_path,
        pending_year_list,
        all_missing_cells,
        _filter_material_gaps(all_missing_cells, min_missing_days),
    )


def process_gapfill(
    start_year: int,
    end_year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    concurrency: int,
    *,
    location_ids: list[int] | None = None,
    min_missing_days: int = MIN_MISSING_DAYS_DEFAULT,
    force: bool = False,
) -> None:
    """Fill (location_id, date) cells the ISD pipeline could not produce, via NLDAS-2."""
    wetbulb_root = f"{out_dir}/wetbulb_data_csv"

    resolved = resolve_gapfill_targets(
        nldas.load_nldas_city_shard(city_shard_index, city_shard_count),
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
    (
        shard_df,
        filesystem,
        base_path,
        pending_year_list,
        all_missing_cells,
        missing_cells,
    ) = resolved
    if len(missing_cells) < len(all_missing_cells):
        LOGGER.info(
            "city_shard=%d/%d: ignoring %d cell(s) in city-years missing "
            "fewer than %d day(s) (already above the views' coverage gate).",
            city_shard_index,
            city_shard_count,
            len(all_missing_cells) - len(missing_cells),
            min_missing_days,
        )
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
        "NLDAS gap-fill city_shard=%d/%d: %d/%d city(ies) have a gap across "
        "%d cell(s).",
        city_shard_index,
        city_shard_count,
        len(gapped_ids),
        len(shard_df),
        len(missing_cells),
    )

    cell_map = giovanni.load_cell_map()
    shard_df = shard_df.merge(cell_map, on="location_id", how="left")
    shard_df["fetch_lat"] = shard_df["cell_lat"].fillna(shard_df["lat"])
    shard_df["fetch_lon"] = shard_df["cell_lon"].fillna(shard_df["lng"])
    gapped_rows = shard_df[shard_df["location_id"].isin(gapped_ids)]

    filled = _fetch_filled_rows(
        gapped_rows,
        missing_cells,
        gap_years_by_location,
        concurrency,
        city_shard_index,
        city_shard_count,
        start_year,
        end_year,
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
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
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
            "Only fill a city-year missing at least this many days. Years "
            "missing fewer already pass the views' >=95%% coverage gate, and "
            "fetching them degenerates into a full backfill. Pass 1 to fill "
            "every gap."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Execute the NLDAS-2 gap-fill pipeline."""
    load_dotenv(override=False)
    exit_code = 0
    try:
        args = _parse_args()
        process_gapfill(
            start_year=args.start_year,
            end_year=args.end_year,
            out_dir=args.out_dir,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
            concurrency=args.concurrency,
            location_ids=args.location_ids,
            min_missing_days=args.min_missing_days,
            force=args.force,
        )
    except KeyboardInterrupt:
        exit_code = 130
        LOGGER.warning("NLDAS gap-fill interrupted by user.")
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
        LOGGER.exception("NLDAS gap-fill failed.")

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
