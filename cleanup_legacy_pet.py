# Copyright (C) 2026 Kenneth Porter

"""Explicitly remove the obsolete PET schema to reclaim database storage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from load import execute_sql_files_with_retries
from shared_config import DATABASE_CONFIG_HINT, resolve_database_uri

LOGGER = logging.getLogger(__name__)
MIGRATION_PATH = Path(__file__).with_name("drop_legacy_pet.sql")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-db-write",
        action="store_true",
        help="Required confirmation for permanently deleting legacy PET data.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Apply the opt-in, idempotent PET cleanup migration."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    if not args.confirm_db_write:
        message = (
            "Legacy PET cleanup permanently deletes obsolete data; "
            "re-run with --confirm-db-write."
        )
        raise SystemExit(message)

    db_uri = resolve_database_uri()
    if not db_uri:
        message = (
            f"Postgres database credentials are not configured. {DATABASE_CONFIG_HINT}"
        )
        raise SystemExit(message)

    LOGGER.info("Removing obsolete PET database objects...")
    execute_sql_files_with_retries(db_uri, (MIGRATION_PATH,))
    LOGGER.info("Legacy PET cleanup complete.")


if __name__ == "__main__":
    main()
