# Copyright (C) 2026 Kenneth Porter

"""Disposable-PostgreSQL integration test for the complete forecast DDL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
import pytest

from load import execute_sql_file, execute_sql_files_atomically

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.db
def test_complete_gmst_ddl_matches_reference_forecast() -> None:
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
                VALUES (1, 'Alpha', 'Test', 0, 0), (2, 'Beta', 'Test', 1, 1)
                """
            )
            cur.execute(
                """
                INSERT INTO public.gmst_observations
                (year, anomaly, anomaly_lo, anomaly_hi, source_version)
                SELECT year, (year - 2000) * 0.04, (year - 2000) * 0.04 - 0.1,
                       (year - 2000) * 0.04 + 0.1, 'synthetic'
                FROM generate_series(2000, 2025) AS year
                """
            )
            cur.execute(
                """
                INSERT INTO public.gmst_scenarios
                (scenario, year, anomaly, anomaly_lo, anomaly_hi, sigma_g)
                SELECT scenario, year, 1.0 + ((year - 2025) * rate),
                       0.9 + ((year - 2025) * rate),
                       1.1 + ((year - 2025) * rate), 0.0
                FROM (VALUES ('ssp126', 0.01), ('ssp245', 0.02), ('ssp370', 0.03))
                     AS scenarios (scenario, rate)
                CROSS JOIN generate_series(2026, 2100) AS year
                """
            )
            cur.execute(
                """
                INSERT INTO public.forecast_station_groups (location_id, station_group)
                VALUES (1, 'group-a'), (2, 'group-b')
                """
            )
            cur.execute(
                """
                INSERT INTO public.forecast_interval_calibration
                (metric, calibration_factor)
                VALUES ('avg_wetbulb', 1), ('max_wetbulb', 1),
                       ('avg_wetbulb_avg', 1), ('max_wetbulb_avg', 1)
                """
            )
            cur.execute(
                """
                INSERT INTO public.wetbulb
                (location_id, date, wetbulb, wetbulb_avg, source)
                SELECT location_id, day::date,
                       10.0 + (2.0 * ((EXTRACT(YEAR FROM day) - 2000) * 0.04)),
                       8.0 + (1.5 * ((EXTRACT(YEAR FROM day) - 2000) * 0.04)),
                       'isd'
                FROM generate_series(1, 2) AS location_id
                CROSS JOIN generate_series(
                    '2000-01-01'::date, '2025-12-31'::date, '1 day'::interval
                ) AS day
                """
            )

        execute_sql_files_atomically(
            conn,
            (
                PROJECT_ROOT / "drop_views.sql",
                PROJECT_ROOT / "create_views.sql",
                PROJECT_ROOT / "create_gmst_views.sql",
            ),
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT scenario, wetbulb, lower, upper, model_type
                FROM public.wetbulb_forecast_scenarios
                WHERE location_id = 1 AND year = 2026 AND season = 'Annual'
                ORDER BY scenario
                """
            )
            rows = cur.fetchall()
            assert [row[0] for row in rows] == ["ssp126", "ssp245", "ssp370"]
            for scenario, point, lower, upper, model_type in rows:
                rate = {"ssp126": 0.01, "ssp245": 0.02, "ssp370": 0.03}[scenario]
                expected = round(10.0 + (2.0 * (1.0 + rate)), 1)
                assert float(point) == pytest.approx(expected, abs=0.05)
                assert float(lower) <= float(point) <= float(upper)
                assert model_type == "gmst_linear"

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM pg_attribute AS a
                    JOIN pg_class AS c ON c.oid = a.attrelid
                    JOIN pg_namespace AS n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public' AND c.relname = 'wetbulb_forecast'
                    AND a.attnum > 0 AND NOT a.attisdropped
                    """
                )
            assert cur.fetchone() == (13,)
            cur.execute(
                """
                SELECT COUNT(*) FROM pg_indexes
                WHERE schemaname = 'public'
                AND indexname IN (
                    'wetbulb_forecast_scenarios_uidx',
                    'wetbulb_forecast_max_scenarios_uidx',
                    'wetbulb_forecast_location_year_season_uidx',
                    'wetbulb_forecast_max_location_year_season_uidx'
                )
                """
            )
            assert cur.fetchone() == (4,)
    finally:
        conn.close()
