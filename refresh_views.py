# Copyright (C) 2026 Kenneth Porter

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
KEEPALIVES_IDLE_SECONDS = 30
KEEPALIVES_INTERVAL_SECONDS = 10
KEEPALIVES_COUNT = 6
DEFAULT_MATVIEW_PREFIX = "wetbulb_"
OPTIONAL_MATVIEWS = frozenset({"wetbulb_forecast_scenarios"})

_REFRESH_SESSION_STATEMENTS: tuple[LiteralString, ...] = (
    "SET max_parallel_workers_per_gather = 0",
    "SET max_parallel_maintenance_workers = 0",
    "SET work_mem = '64MB'",
    "SET maintenance_work_mem = '128MB'",
)

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


def _discover_matviews(
    conn: Connection[Any],
    *,
    name_prefix: str | None = None,
) -> dict[str, bool]:
    """Return public materialized views mapped to their populated state."""
    with conn.cursor() as cur:
        cur.execute(_MATVIEWS_QUERY)
        matviews = {row[0]: bool(row[1]) for row in cur.fetchall()}
    if name_prefix is None:
        return matviews
    return {
        name: is_populated
        for name, is_populated in matviews.items()
        if name.startswith(name_prefix)
    }


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
    name_prefix: str | None = None,
    include_optional: bool = False,
    already_refreshed: set[str] | None = None,
    names: set[str] | None = None,
) -> list[str]:
    """Refresh matching public matviews in dependency order; return the order."""
    matviews = _discover_matviews(conn, name_prefix=name_prefix)
    if not include_optional:
        matviews = {
            name: is_populated
            for name, is_populated in matviews.items()
            if name not in OPTIONAL_MATVIEWS
        }
    if names is not None:
        matviews = {
            name: is_populated
            for name, is_populated in matviews.items()
            if name in names
        }
    if not matviews:
        LOGGER.warning("No materialized views found in schema public.")
        return []

    refresh_order = _matview_refresh_order(conn, set(matviews))
    completed = already_refreshed if already_refreshed is not None else set()
    for matview_name in refresh_order:
        if matview_name in completed:
            LOGGER.info(
                "Skipping previously refreshed materialized view %s.", matview_name
            )
            continue
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
        completed.add(matview_name)
    return refresh_order


def configure_refresh_session(conn: Connection[Any]) -> None:
    """Use a bounded, serial plan to avoid external-sort tape failures."""
    with conn.cursor() as cur:
        for statement in _REFRESH_SESSION_STATEMENTS:
            cur.execute(statement)


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
    parser.add_argument(
        "--matview-prefix",
        default=DEFAULT_MATVIEW_PREFIX,
        help=(
            "Only refresh materialized views whose names begin with this value "
            f"(default: {DEFAULT_MATVIEW_PREFIX!r})."
        ),
    )
    parser.add_argument(
        "--include-forecast-scenarios",
        action="store_true",
        help="Also refresh the optional wetbulb_forecast_scenarios view.",
    )
    parser.add_argument(
        "--matview-name",
        action="append",
        default=None,
        help="Refresh only this exact materialized-view name; may be repeated.",
    )
    return parser.parse_args(argv)


def _is_storage_refresh_error(exc: psycopg.InternalError) -> bool:
    """Identify server errors caused by exhausted or corrupt temporary storage."""
    message = str(exc).lower()
    return "no space left on device" in message or "unexpected end of tape" in message


def _refresh_once(
    db_uri: str,
    args: argparse.Namespace,
    already_refreshed: set[str] | None = None,
) -> None:
    """Refresh views once using a single database connection."""
    conn: Connection[Any] | None = None
    try:
        conn = psycopg.connect(
            db_uri,
            keepalives=1,
            keepalives_idle=KEEPALIVES_IDLE_SECONDS,
            keepalives_interval=KEEPALIVES_INTERVAL_SECONDS,
            keepalives_count=KEEPALIVES_COUNT,
        )
        conn.autocommit = True
        configure_refresh_session(conn)
        refreshed = refresh_materialized_views(
            conn,
            allow_concurrent=not args.non_concurrent,
            name_prefix=args.matview_prefix,
            include_optional=args.include_forecast_scenarios,
            already_refreshed=already_refreshed,
            names=set(args.matview_name) if args.matview_name else None,
        )
        LOGGER.info("Refreshed %d materialized view(s).", len(refreshed))
        if not args.skip_analyze:
            refresh_query_planner_statistics(conn)
    except psycopg.InternalError as exc:
        if _is_storage_refresh_error(exc):
            LOGGER.exception(
                "PostgreSQL ran out of usable temporary storage while refreshing the "
                "views. If legacy PET objects are present, the explicit 'cleanup' step "
                "can reclaim about 1.1 GB before retrying 'views'."
            )
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except psycopg.Error:
                LOGGER.warning("Database connection was already unusable.")
            LOGGER.info("Database connection closed.")


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

    already_refreshed: set[str] = set()
    for attempt in range(1, REFRESH_MAX_ATTEMPTS + 1):
        LOGGER.info("Connecting to the database...")
        try:
            _refresh_once(db_uri, args, already_refreshed)
        except psycopg.OperationalError:
            if attempt == REFRESH_MAX_ATTEMPTS:
                raise
            LOGGER.warning(
                "Database connection failed during view refresh; retrying with a fresh "
                "connection (%d/%d).",
                attempt + 1,
                REFRESH_MAX_ATTEMPTS,
            )
            time.sleep(REFRESH_RETRY_DELAY_SECONDS * attempt)
        else:
            return


if __name__ == "__main__":
    main()
