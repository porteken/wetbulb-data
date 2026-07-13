"""Load wetbulb parquet/CSV data into the database without touching other tables."""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

import psycopg

from load import (
    COPY_BATCH_SIZE,
    DEFAULT_LOAD_WORKERS,
    _discover_wetbulb_csv_paths,
    _load_table_files,
    _validate_load_shard_args,
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
    parser.add_argument("--wetbulb-csv", default="wetbulb.csv")
    parser.add_argument("--wetbulb-root", default="wetbulb_data_csv")
    parser.add_argument(
        "--prefer-wetbulb-csv",
        action="store_true",
        help=(
            "Load the explicit --wetbulb-csv input even when wetbulb parquet "
            "shards are also present under --wetbulb-root."
        ),
    )
    parser.add_argument("--load-shard-index", type=int, default=0)
    parser.add_argument("--load-shard-count", type=int, default=1)
    parser.add_argument("--copy-batch-size", type=int, default=COPY_BATCH_SIZE)
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
    return parser.parse_args(argv)


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
            LOGGER.info("Refreshing query planner statistics...")
            with conn.cursor() as cur:
                cur.execute("ANALYZE public.wetbulb")
    finally:
        conn.close()
        LOGGER.info("Database connection closed.")


if __name__ == "__main__":
    main()
