# Copyright (C) 2026 Kenneth Porter

"""Refuse to reuse populated database location IDs for different cities."""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

import pandas as pd
import psycopg

from shared_config import DATABASE_CONFIG_HINT, resolve_database_uri

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


def identity_conflicts(incoming: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Return populated IDs whose incoming city/state identity has changed."""
    merged = incoming.merge(existing, on="id", how="inner", suffixes=("_new", "_old"))
    changed = (merged["city_new"] != merged["city_old"]) | (
        merged["state_new"] != merged["state_old"]
    )
    return merged.loc[changed].reset_index(drop=True)


def validate_database_catalog(csv_path: str | Path) -> None:
    """Raise when an incoming catalog reassigns an ID that has wet-bulb rows."""
    incoming = pd.read_csv(csv_path, usecols=["id", "city", "state"])
    db_uri = resolve_database_uri()
    if not db_uri:
        LOGGER.warning("%s Skipping database catalog validation.", DATABASE_CONFIG_HINT)
        return

    ids = [int(value) for value in incoming["id"]]
    with psycopg.connect(db_uri) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT l.id, l.city, l.state
            FROM public.locations AS l
            WHERE l.id = ANY(%s)
              AND EXISTS (
                  SELECT 1 FROM public.wetbulb AS w WHERE w.location_id = l.id
              )
            """,
            (ids,),
        )
        existing = pd.DataFrame(cur.fetchall(), columns=["id", "city", "state"])

    conflicts = identity_conflicts(incoming, existing)
    if conflicts.empty:
        return
    examples = "; ".join(
        f"{row.id}: {row.city_old}, {row.state_old} -> {row.city_new}, {row.state_new}"
        for row in conflicts.head(10).itertuples()
    )
    message = (
        f"Incoming catalog reassigns {len(conflicts)} populated location ID(s): "
        f"{examples}. Replace the affected wetbulb rows before updating locations."
    )
    raise RuntimeError(message)


def main() -> None:
    """Validate an incoming locations catalog against populated database IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations-csv", default="locations.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    validate_database_catalog(args.locations_csv)


if __name__ == "__main__":
    main()
