# Copyright (C) 2026 Kenneth Porter

"""Load wetbulb parquet/CSV data into the database without touching other tables."""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg import sql

from load import (
    DEFAULT_LOAD_WORKERS,
    _add_wetbulb_load_args,
    _discover_wetbulb_csv_paths,
    _load_table_files,
    _validate_load_shard_args,
    execute_sql_files_with_retries,
    refresh_query_planner_statistics_with_retries,
)
from shared_config import DATABASE_CONFIG_HINT, resolve_database_uri

if TYPE_CHECKING:
    from psycopg import Connection

LOGGER = logging.getLogger(__name__)
TABLE_NAME = "wetbulb"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load wetbulb parquet shards (or a single CSV) into public.wetbulb. "
            "Views and other tables are left untouched."
        ),
    )
    _add_wetbulb_load_args(parser)
    parser.add_argument(
        "--load-workers",
        type=int,
        default=DEFAULT_LOAD_WORKERS,
        help=(
            "Parallel database connections used to load files "
            "(append/upsert loads only; truncating loads stay serial)."
        ),
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate public.wetbulb before loading. Defaults to appending.",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Do not refresh planner statistics after loading.",
    )
    parser.add_argument("--replace-location-ids", type=int, nargs="+")
    return parser.parse_args(argv)


def _delete_replaced_locations(
    conn: Connection[Any],
    location_ids: list[int],
    start_year: int,
    end_year: int,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "DELETE FROM public.wetbulb WHERE location_id = ANY(%s) "
                "AND date >= make_date(%s, 1, 1) "
                "AND date < make_date(%s, 1, 1)"
            ),
            (location_ids, start_year, end_year + 1),
        )
        return max(cur.rowcount, 0)


def _wetbulb_table_exists(conn: Connection[Any]) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"public.{TABLE_NAME}",))
        row = cur.fetchone()
    return row is not None and row[0] is not None


def main() -> None:
    """Load wetbulb data into public.wetbulb."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    _validate_load_shard_args(args.load_shard_index, args.load_shard_count)
    if args.replace_location_ids and (
        args.wetbulb_start_year is None or args.wetbulb_end_year is None
    ):
        msg = "--replace-location-ids requires both wetbulb year bounds"
        raise SystemExit(msg)

    db_uri = resolve_database_uri()
    if not db_uri:
        LOGGER.warning(
            "Postgres database credentials are not configured. %s Skipping load.",
            DATABASE_CONFIG_HINT,
        )
        return

    csv_paths = _discover_wetbulb_csv_paths(args)
    if not csv_paths:
        LOGGER.warning(
            "No wetbulb inputs found under %s (or at %s). Nothing to load.",
            args.wetbulb_root,
            args.wetbulb_csv,
        )
        return

    LOGGER.info("Connecting to the database...")
    conn: Connection[Any] = psycopg.connect(db_uri)
    conn.autocommit = True
    try:
        if not _wetbulb_table_exists(conn):
            msg = (
                "Table public.wetbulb does not exist. "
                "Run create_tables.sql (or load.py) first."
            )
            raise SystemExit(msg)
        execute_sql_files_with_retries(
            db_uri,
            ("migrate_wetbulb_station_provenance.sql",),
        )

        if args.replace_location_ids:
            deleted = _delete_replaced_locations(
                conn,
                sorted(set(args.replace_location_ids)),
                args.wetbulb_start_year,
                args.wetbulb_end_year,
            )
            LOGGER.info("Deleted %d replaced wetbulb row(s).", deleted)

        _load_table_files(
            conn,
            db_uri,
            csv_paths,
            TABLE_NAME,
            batch_size=args.copy_batch_size,
            truncate=args.truncate,
            workers=args.load_workers,
        )

        if not args.skip_analyze:
            refresh_query_planner_statistics_with_retries(
                db_uri,
                table_names=(TABLE_NAME,),
            )
    finally:
        conn.close()
        LOGGER.info("Database connection closed.")


if __name__ == "__main__":
    main()
