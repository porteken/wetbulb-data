"""Disposable-PostgreSQL integration test for the US/EU rankings-view split."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
import pytest

from load import execute_sql_file, execute_sql_files_atomically

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.db
def test_rankings_views_split_at_location_id_1000() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    conn: psycopg.Connection[Any] = psycopg.connect(database_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA public CASCADE")
            cur.execute("CREATE SCHEMA public")
        execute_sql_file(conn, PROJECT_ROOT / "create_tables.sql")
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.locations (id, city, state, lat, lng)
                VALUES (1, 'New York', 'New York', 40.7, -74.0),
                       (1000, 'Paris', 'France', 48.85, 2.35)
                """
            )
            cur.execute(
                """
                INSERT INTO public.wetbulb
                (location_id, date, wetbulb, wetbulb_avg, source)
                SELECT location_id, day::date, 20.0, 18.0, 'isd'
                FROM (VALUES (1), (1000)) AS locations (location_id)
                CROSS JOIN generate_series(
                    '2000-01-01'::date, '2000-12-31'::date, '1 day'::interval
                ) AS day
                """
            )

        execute_sql_files_atomically(
            conn,
            (
                PROJECT_ROOT / "drop_views.sql",
                PROJECT_ROOT / "create_views.sql",
                PROJECT_ROOT / "create_gmst_views.sql",
                PROJECT_ROOT / "create_eu_views.sql",
            ),
        )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT location_id FROM public.wetbulb_city_rankings_view "
                "ORDER BY location_id",
            )
            us_ids = {row[0] for row in cur.fetchall()}
            cur.execute(
                "SELECT location_id FROM public.wetbulb_eu_city_rankings_view "
                "ORDER BY location_id",
            )
            eu_ids = {row[0] for row in cur.fetchall()}

        assert us_ids == {1}
        assert eu_ids == {1000}
    finally:
        conn.close()
