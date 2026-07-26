"""Disposable-PostgreSQL integration tests for `load._upsert_from_staging` precedence."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, LiteralString, cast

import psycopg
import pytest

from load import _create_staging_table, _upsert_from_staging, execute_sql_file

if TYPE_CHECKING:
    from collections.abc import Iterator

PROJECT_ROOT_CREATE_TABLES = "create_tables.sql"
_COLUMNS = ["location_id", "date", "wetbulb", "wetbulb_avg", "source"]


@pytest.fixture
def conn() -> Iterator[psycopg.Connection[Any]]:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    connection: psycopg.Connection[Any] = psycopg.connect(database_url)
    connection.autocommit = True
    with connection.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")
    execute_sql_file(connection, PROJECT_ROOT_CREATE_TABLES)
    with connection.cursor() as cur:
        cur.execute(
            "INSERT INTO public.locations (id, city, state, lat, lng) "
            "VALUES (1, 'Alpha', 'Test', 0, 0)",
        )
    try:
        yield connection
    finally:
        connection.close()


def _upsert_rows(
    connection: psycopg.Connection[Any], rows: list[tuple[Any, ...]]
) -> int:
    with connection.transaction():
        staging_name = _create_staging_table(connection, "wetbulb", _COLUMNS)
        with connection.cursor() as cur:
            cur.executemany(
                cast(
                    "LiteralString",
                    f"INSERT INTO {staging_name} "
                    "(location_id, date, wetbulb, wetbulb_avg, source) "
                    "VALUES (%s, %s, %s, %s, %s)",
                ),
                rows,
            )
        return _upsert_from_staging(
            connection, "wetbulb", staging_name, _COLUMNS, ("location_id", "date")
        )


def _wetbulb_row(connection: psycopg.Connection[Any]) -> tuple[Any, ...] | None:
    with connection.cursor() as cur:
        cur.execute(
            "SELECT wetbulb, source FROM public.wetbulb "
            "WHERE location_id = 1 AND date = '2020-06-01'",
        )
        return cur.fetchone()


@pytest.mark.db
def test_fill_row_cannot_overwrite_existing_primary_row(
    conn: psycopg.Connection[Any],
) -> None:
    _upsert_rows(conn, [(1, "2020-06-01", 20.0, 19.0, "isd")])
    _upsert_rows(conn, [(1, "2020-06-01", 25.0, 24.0, "era5land")])
    assert _wetbulb_row(conn) == (20.0, "isd")


@pytest.mark.db
def test_fill_row_lands_in_an_empty_cell(conn: psycopg.Connection[Any]) -> None:
    _upsert_rows(conn, [(1, "2020-06-01", 25.0, 24.0, "era5land")])
    assert _wetbulb_row(conn) == (25.0, "era5land")


@pytest.mark.db
def test_primary_row_overwrites_an_existing_fill_row(
    conn: psycopg.Connection[Any],
) -> None:
    _upsert_rows(conn, [(1, "2020-06-01", 25.0, 24.0, "era5land")])
    _upsert_rows(conn, [(1, "2020-06-01", 20.0, 19.0, "isd")])
    assert _wetbulb_row(conn) == (20.0, "isd")


@pytest.mark.db
def test_in_batch_dedup_prefers_primary_over_any_fill_source(
    conn: psycopg.Connection[Any],
) -> None:
    """`'era5land'` sorts before `'isd'` alphabetically; rank must win instead."""
    _upsert_rows(
        conn,
        [
            (1, "2020-06-01", 25.0, 24.0, "era5land"),
            (1, "2020-06-01", 20.0, 19.0, "isd"),
        ],
    )
    assert _wetbulb_row(conn) == (20.0, "isd")


@pytest.mark.db
def test_one_fill_source_can_overwrite_another(
    conn: psycopg.Connection[Any],
) -> None:
    """Only `isd` is protected; fill sources may still overwrite each other."""
    _upsert_rows(conn, [(1, "2020-06-01", 22.0, 21.0, "nldas")])
    _upsert_rows(conn, [(1, "2020-06-01", 25.0, 24.0, "era5land")])
    assert _wetbulb_row(conn) == (25.0, "era5land")
