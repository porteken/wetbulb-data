"""Refresh materialized views in dependency order without dropping them."""

from __future__ import annotations

import argparse
import logging
import time
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, Any, LiteralString, cast

import psycopg

from load import refresh_query_planner_statistics
from shared_config import DATABASE_CONFIG_HINT, resolve_database_uri

if TYPE_CHECKING:
    from psycopg import Connection

LOGGER = logging.getLogger(__name__)
REFRESH_MAX_ATTEMPTS = 4
REFRESH_RETRY_DELAY_SECONDS = 30

_MATVIEWS_QUERY: LiteralString = """
SELECT matviewname, ispopulated
FROM pg_matviews
WHERE schemaname = 'public'
"""

_MATVIEW_DEPENDENCIES_QUERY: LiteralString = """
SELECT DISTINCT dependent.relname AS view_name, source.relname AS depends_on
FROM pg_depend AS d
JOIN pg_rewrite AS r ON r.oid = d.objid
JOIN pg_class AS dependent ON dependent.oid = r.ev_class
JOIN pg_class AS source ON source.oid = d.refobjid
JOIN pg_namespace AS n ON n.oid = dependent.relnamespace
WHERE dependent.relkind = 'm'
AND source.relkind = 'm'
AND n.nspname = 'public'
AND dependent.oid != source.oid
"""

_UNIQUE_INDEX_QUERY: LiteralString = """
SELECT 1
FROM pg_index AS ix
JOIN pg_class AS t ON t.oid = ix.indrelid
JOIN pg_namespace AS n ON n.oid = t.relnamespace
WHERE n.nspname = 'public'
AND t.relname = %s
AND ix.indisunique
LIMIT 1
"""


def _discover_matviews(conn: Connection[Any]) -> dict[str, bool]:
    """Return public materialized views mapped to their populated state."""
    with conn.cursor() as cur:
        cur.execute(_MATVIEWS_QUERY)
        return {row[0]: bool(row[1]) for row in cur.fetchall()}


def _matview_refresh_order(conn: Connection[Any], matviews: set[str]) -> list[str]:
    """Topologically sort matviews so dependencies refresh first."""
    dependencies: dict[str, set[str]] = {name: set() for name in matviews}
    with conn.cursor() as cur:
        cur.execute(_MATVIEW_DEPENDENCIES_QUERY)
        for view_name, depends_on in cur.fetchall():
            if view_name in dependencies and depends_on in matviews:
                dependencies[view_name].add(depends_on)
    return list(TopologicalSorter(dependencies).static_order())


def _has_unique_index(conn: Connection[Any], matview_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(_UNIQUE_INDEX_QUERY, (matview_name,))
        return cur.fetchone() is not None


def refresh_materialized_views(
    conn: Connection[Any],
    *,
    allow_concurrent: bool = True,
) -> list[str]:
    """Refresh every public matview in dependency order; return the order."""
    matviews = _discover_matviews(conn)
    if not matviews:
        LOGGER.warning("No materialized views found in schema public.")
        return []

    refresh_order = _matview_refresh_order(conn, set(matviews))
    for matview_name in refresh_order:
        concurrently = (
            allow_concurrent
            and matviews[matview_name]
            and _has_unique_index(conn, matview_name)
        )
        statement = "REFRESH MATERIALIZED VIEW {}public.{}".format(
            "CONCURRENTLY " if concurrently else "",
            matview_name,
        )
        LOGGER.info("%s...", statement)
        with conn.cursor() as cur:
            cur.execute(cast("LiteralString", statement))
    return refresh_order


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh public materialized views in dependency order and "
            "update planner statistics, without dropping or recreating views."
        ),
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Do not refresh planner statistics after refreshing views.",
    )
    parser.add_argument(
        "--non-concurrent",
        action="store_true",
        help="Use lower-overhead blocking refreshes for maintenance runs.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Refresh materialized views and planner statistics."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    db_uri = resolve_database_uri()
    if not db_uri:
        LOGGER.warning(
            "Postgres database credentials are not configured. %s Skipping refresh.",
            DATABASE_CONFIG_HINT,
        )
        return

    for attempt in range(1, REFRESH_MAX_ATTEMPTS + 1):
        LOGGER.info("Connecting to the database...")
        conn: Connection[Any] | None = None
        try:
            conn = psycopg.connect(db_uri)
            conn.autocommit = True
            refreshed = refresh_materialized_views(
                conn,
                allow_concurrent=not args.non_concurrent,
            )
            LOGGER.info("Refreshed %d materialized view(s).", len(refreshed))
            if not args.skip_analyze:
                refresh_query_planner_statistics(conn)
        except psycopg.OperationalError:
            if attempt == REFRESH_MAX_ATTEMPTS:
                raise
            LOGGER.warning(
                "Database connection failed during view refresh; retrying "
                "with a fresh connection (%d/%d).",
                attempt + 1,
                REFRESH_MAX_ATTEMPTS,
            )
            time.sleep(REFRESH_RETRY_DELAY_SECONDS * attempt)
        else:
            return
        finally:
            if conn is not None:
                try:
                    conn.close()
                except psycopg.Error:
                    LOGGER.warning("Database connection was already unusable.")
                LOGGER.info("Database connection closed.")


if __name__ == "__main__":
    main()
