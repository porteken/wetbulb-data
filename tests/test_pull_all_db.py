"""Integration tests for the pull_all pipeline database load."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, LiteralString, cast

import pytest

from shared_config import resolve_database_uri

if TYPE_CHECKING:
    from collections.abc import Generator

    from psycopg import Connection

pytestmark = pytest.mark.db

REQUIRED_TABLES = [
    "locations",
    "pet",
]

REQUIRED_MATERIALIZED_VIEWS = [
    "pet_forecast",
    "pet_forecast_max",
    "pet_year_stats",
]

REQUIRED_VIEWS = [
    "city_rankings_view",
]


@pytest.fixture(scope="module")
def db_conn() -> Generator[Connection[Any]]:
    """Yield a psycopg connection, skip if unavailable."""
    db_uri = resolve_database_uri()
    if not db_uri:
        pytest.skip("Postgres database credentials are not set")

    try:
        import psycopg
    except ImportError:
        psycopg = None
        pytest.skip("psycopg not installed")

    try:
        conn = psycopg.connect(db_uri)
        conn.autocommit = True
    except psycopg.OperationalError as exc:
        pytest.skip(f"Cannot connect to database: {exc}")

    yield conn
    conn.close()


def _row_count(conn: Connection[Any], relation: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            cast("LiteralString", "SELECT COUNT(*) FROM pg_class WHERE relname = %s"),
            (relation,),
        )
        row = cast("tuple[Any, ...]", cur.fetchone())
        assert row is not None
        if row[0] == 0:
            pytest.fail(f"Relation {relation!r} does not exist")
        # noinspection PyTypeChecker
        cur.execute(cast("LiteralString", f"SELECT COUNT(*) FROM {relation}"))
        count_row = cast("tuple[Any, ...]", cur.fetchone())
        assert count_row is not None
        return count_row[0]


def _get_relation_kind(conn: Connection[Any], relation: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "SELECT c.relkind "
                "FROM pg_catalog.pg_class AS c "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' "
                "AND c.relname = %s",
            ),
            (relation,),
        )
        row = cast("tuple[Any, ...]", cur.fetchone())
        assert row is not None, f"Relation {relation!r} does not exist"
        return row[0]


@pytest.mark.parametrize("table", REQUIRED_TABLES)
def test_table_has_rows(db_conn: Connection[Any], table: str) -> None:
    count = _row_count(db_conn, table)
    assert count > 0, f"Table {table!r} is empty (0 rows)"


@pytest.mark.parametrize("view", REQUIRED_MATERIALIZED_VIEWS)
def test_materialized_view_has_rows(db_conn: Connection[Any], view: str) -> None:
    assert _get_relation_kind(db_conn, view) == "m"
    count = _row_count(db_conn, view)
    assert count > 0, f"Materialized view {view!r} is empty (0 rows)"


@pytest.mark.parametrize("view", REQUIRED_VIEWS)
def test_view_has_rows(db_conn: Connection[Any], view: str) -> None:
    assert _get_relation_kind(db_conn, view) == "v"
    count = _row_count(db_conn, view)
    assert count > 0, f"View {view!r} is empty (0 rows)"


def test_pet_covers_year_range(db_conn: Connection[Any]) -> None:
    """The pet table should have data for the years it contains."""
    with db_conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "SELECT MIN(EXTRACT(YEAR FROM date))::int, "
                "MAX(EXTRACT(YEAR FROM date))::int FROM pet",
            )
        )
        result = cast("tuple[Any, ...]", cur.fetchone())
        assert result is not None
        min_year, max_year = result
    assert min_year is not None, "pet table has no date data"

    assert min_year >= 2000, f"Earliest year {min_year} is before supported start 2000"
    assert max_year <= 2026, f"Latest year {max_year} is beyond expected 2026"


def test_locations_not_empty(db_conn: Connection[Any]) -> None:
    count = _row_count(db_conn, "locations")
    assert count > 0, "locations table is empty"


def test_compact_schema_types(db_conn: Connection[Any]) -> None:
    """Core tables and analytics views should use the compact types defined in SQL."""
    assert _get_relation_column_types(db_conn, "locations") == {
        "id": "smallint",
        "city": "text",
        "state": "text",
        "lat": "real",
        "lng": "real",
    }
    assert _get_relation_column_types(db_conn, "pet") == {
        "id": "smallint",
        "location_id": "smallint",
        "date": "date",
        "pet": "real",
    }
    assert _get_relation_column_types(db_conn, "pet_year") == {
        "location_id": "integer",
        "date": "date",
        "year": "smallint",
        "pet": "real",
    }
    assert _get_relation_column_types(db_conn, "pet_year_avg") == {
        "location_id": "integer",
        "year": "smallint",
        "season": "text",
        "pet": "numeric(5,1)",
    }
    assert _get_relation_column_types(db_conn, "pet_year_max") == {
        "location_id": "integer",
        "year": "smallint",
        "season": "text",
        "pet": "numeric(5,1)",
    }
    assert _get_relation_column_types(db_conn, "pet_percentiles") == {
        "year": "smallint",
        "location_id": "integer",
        "season": "text",
        "p10": "numeric(5,1)",
        "p90": "numeric(5,1)",
    }
    assert _get_relation_column_types(db_conn, "pet_forecast") == {
        "location_id": "integer",
        "year": "smallint",
        "season": "text",
        "pet": "numeric(5,1)",
        "lower": "numeric(5,1)",
        "upper": "numeric(5,1)",
        "model_type": "text",
        "full_years_used": "smallint",
        "warming_rate": "numeric(5,2)",
        "acceleration": "numeric(6,3)",
    }
    assert _get_relation_column_types(db_conn, "city_rankings_view") == {
        "location_id": "integer",
        "year": "smallint",
        "season": "text",
        "avg_pet": "numeric(5,1)",
        "max_pet": "numeric(5,1)",
        "city": "text",
        "state": "text",
        "p10": "numeric(5,1)",
        "p90": "numeric(5,1)",
        "future_lower": "numeric(5,1)",
        "future_upper": "numeric(5,1)",
        "change_from_2000": "numeric(5,2)",
    }


def test_pet_has_id_index(db_conn: Connection[Any]) -> None:
    """Pet should expose a covering location/date index for reference graph queries."""
    with db_conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "SELECT pg_get_indexdef(i.oid) "
                "FROM pg_catalog.pg_class AS i "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = i.relnamespace "
                "WHERE n.nspname = 'public' "
                "AND i.relname = 'pet_location_date_covering_idx'",
            )
        )
        row = cast("tuple[Any, ...]", cur.fetchone())

    assert row is not None
    index_definition = row[0]
    assert index_definition is not None
    assert (
        "ON public.pet USING btree (location_id, date) INCLUDE (pet)"
        in index_definition
    )


def _get_relation_columns(conn: Connection[Any], relation: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "SELECT a.attname "
                "FROM pg_catalog.pg_attribute AS a "
                "JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' "
                "AND c.relname = %s "
                "AND a.attnum > 0 "
                "AND NOT a.attisdropped "
                "ORDER BY a.attnum",
            ),
            (relation,),
        )
        return [row[0] for row in cur.fetchall()]


def _get_relation_column_types(conn: Connection[Any], relation: str) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) "
                "FROM pg_catalog.pg_attribute AS a "
                "JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' "
                "AND c.relname = %s "
                "AND a.attnum > 0 "
                "AND NOT a.attisdropped "
                "ORDER BY a.attnum",
            ),
            (relation,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def _get_index_columns(conn: Connection[Any], index_name: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "SELECT a.attname "
                "FROM pg_catalog.pg_class AS i "
                "JOIN pg_catalog.pg_index AS ix ON ix.indexrelid = i.oid "
                "JOIN pg_catalog.pg_class AS t ON t.oid = ix.indrelid "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = i.relnamespace "
                "JOIN pg_catalog.pg_attribute AS a ON a.attrelid = t.oid "
                "AND a.attnum = ANY(ix.indkey) "
                "WHERE n.nspname = 'public' "
                "AND i.relname = %s "
                "AND t.relname = 'pet' "
                "ORDER BY array_position(ix.indkey, a.attnum)",
            ),
            (index_name,),
        )
        return [row[0] for row in cur.fetchall()]


def _assert_year_season_pet_view(
    db_conn: Connection[Any], view: str, aggregate: str
) -> None:
    with db_conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "SELECT COUNT(*) FROM ("
                "SELECT location_id, year, season, COUNT(*) AS row_count "
                f"FROM {view} "
                "GROUP BY location_id, year, season "
                "HAVING COUNT(*) > 1"
                ") AS duplicates",
            )
        )
        duplicate_count_row = cast("tuple[Any, ...]", cur.fetchone())
        assert duplicate_count_row is not None

        cur.execute(
            cast(
                "LiteralString",
                "WITH pet_with_seasons AS ("
                "SELECT location_id, EXTRACT(YEAR FROM date)::int AS year, 'Annual'::text AS season, pet "
                "FROM pet "
                "UNION ALL "
                "SELECT "
                "location_id, "
                "EXTRACT(YEAR FROM date)::int AS year, "
                "CASE "
                "WHEN EXTRACT(MONTH FROM date)::int IN (12, 1, 2) THEN 'Winter' "
                "WHEN EXTRACT(MONTH FROM date)::int IN (3, 4, 5) THEN 'Spring' "
                "WHEN EXTRACT(MONTH FROM date)::int IN (6, 7, 8) THEN 'Summer' "
                "ELSE 'Fall' "
                "END AS season, "
                "pet "
                "FROM pet"
                "), expected AS ("
                "SELECT location_id, year, season, "
                f"ROUND({aggregate}(pet)::numeric, 1) AS pet "
                "FROM pet_with_seasons "
                "GROUP BY location_id, year, season"
                "), actual AS ("
                f"SELECT location_id, year, season, pet FROM {view}"
                "), missing AS ("
                "SELECT * FROM expected "
                "EXCEPT "
                "SELECT * FROM actual"
                "), extra AS ("
                "SELECT * FROM actual "
                "EXCEPT "
                "SELECT * FROM expected"
                ") "
                "SELECT "
                "(SELECT COUNT(*) FROM missing), "
                "(SELECT COUNT(*) FROM extra)",
            )
        )
        diff_count_row = cast("tuple[Any, ...]", cur.fetchone())
        assert diff_count_row is not None

        cur.execute(
            cast("LiteralString", f"SELECT DISTINCT season FROM {view} ORDER BY season")
        )
        seasons = {row[0] for row in cur.fetchall()}

    columns = _get_relation_columns(db_conn, view)
    assert columns == ["location_id", "year", "season", "pet"]

    duplicate_count = duplicate_count_row[0]
    missing_count, extra_count = diff_count_row
    assert duplicate_count == 0
    assert seasons
    assert "Annual" in seasons
    assert seasons.issubset({"Annual", "Winter", "Spring", "Summer", "Fall"})
    assert missing_count == 0
    assert extra_count == 0


def test_pet_year_avg_has_annual_and_seasons(db_conn: Connection[Any]) -> None:
    """pet_year_avg should expose Annual plus Winter/Spring/Summer/Fall rows."""
    _assert_year_season_pet_view(db_conn, "pet_year_avg", "AVG")


def test_pet_year_max_has_annual_and_seasons(db_conn: Connection[Any]) -> None:
    """Verify pet_year_max contains annual and seasonal rows."""
    _assert_year_season_pet_view(db_conn, "pet_year_max", "MAX")


def test_pet_percentiles_match_pet_quantiles(db_conn: Connection[Any]) -> None:
    """pet_percentiles should yearly match season-specific 10th/90th quantiles from pet."""
    with db_conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "WITH pet_with_seasons AS ("
                "SELECT "
                "EXTRACT(YEAR FROM date)::int AS year, "
                "location_id, "
                "'Annual'::text AS season, "
                "pet "
                "FROM pet "
                "UNION ALL "
                "SELECT "
                "EXTRACT(YEAR FROM date)::int AS year, "
                "location_id, "
                "CASE "
                "WHEN EXTRACT(MONTH FROM date)::int IN (12, 1, 2) THEN 'Winter' "
                "WHEN EXTRACT(MONTH FROM date)::int IN (3, 4, 5) THEN 'Spring' "
                "WHEN EXTRACT(MONTH FROM date)::int IN (6, 7, 8) THEN 'Summer' "
                "ELSE 'Fall' "
                "END AS season, "
                "pet "
                "FROM pet"
                "), expected AS ("
                "SELECT "
                "year, "
                "location_id, "
                "season, "
                "ROUND((PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY pet))::numeric, 1) AS p10, "
                "ROUND((PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY pet))::numeric, 1) AS p90 "
                "FROM pet_with_seasons "
                "GROUP BY year, location_id, season"
                "), actual AS ("
                "SELECT year, location_id, season, p10, p90 FROM pet_percentiles"
                "), missing AS ("
                "SELECT * FROM expected "
                "EXCEPT "
                "SELECT * FROM actual"
                "), extra AS ("
                "SELECT * FROM actual "
                "EXCEPT "
                "SELECT * FROM expected"
                ") "
                "SELECT "
                "(SELECT COUNT(*) FROM missing), "
                "(SELECT COUNT(*) FROM extra), "
                "(SELECT COUNT(*) FROM ("
                "SELECT year, location_id, season, COUNT(*) AS row_count "
                "FROM pet_percentiles "
                "GROUP BY year, location_id, season "
                "HAVING COUNT(*) > 1"
                ") AS duplicates)",
            )
        )
        diff_count_row = cast("tuple[Any, ...]", cur.fetchone())
        assert diff_count_row is not None

    columns = _get_relation_columns(db_conn, "pet_percentiles")
    assert columns == ["year", "location_id", "season", "p10", "p90"]

    missing_count, extra_count, duplicate_count = diff_count_row
    assert missing_count == 0
    assert extra_count == 0
    assert duplicate_count == 0


def test_city_rankings_view_matches_expected_projection(
    db_conn: Connection[Any],
) -> None:
    """city_rankings_view should inline annual/seasonal stats, seasonal percentile bands, forecast bounds, and decade deltas."""
    with db_conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "WITH pet_with_seasons AS ("
                "SELECT "
                "EXTRACT(YEAR FROM date)::int AS year, "
                "location_id, "
                "'Annual'::text AS season, "
                "pet "
                "FROM pet "
                "UNION ALL "
                "SELECT "
                "EXTRACT(YEAR FROM date)::int AS year, "
                "location_id, "
                "CASE "
                "WHEN EXTRACT(MONTH FROM date)::int IN (12, 1, 2) THEN 'Winter' "
                "WHEN EXTRACT(MONTH FROM date)::int IN (3, 4, 5) THEN 'Spring' "
                "WHEN EXTRACT(MONTH FROM date)::int IN (6, 7, 8) THEN 'Summer' "
                "ELSE 'Fall' "
                "END AS season, "
                "pet "
                "FROM pet"
                "), expected_percentiles AS ("
                "SELECT "
                "year, "
                "location_id, "
                "season, "
                "ROUND((PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY pet))::numeric, 1) AS p10, "
                "ROUND((PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY pet))::numeric, 1) AS p90 "
                "FROM pet_with_seasons "
                "GROUP BY year, location_id, season"
                "), combined_yearly_avg AS ("
                "SELECT "
                "location_id, "
                "year::int AS year, "
                "season, "
                "AVG(pet) AS pet, "
                "0 AS source_order "
                "FROM pet_year_avg "
                "GROUP BY location_id, year, season "
                "UNION ALL "
                "SELECT location_id, year::int AS year, season, pet, 1 AS source_order "
                "FROM pet_forecast"
                "), deduplicated_yearly_avg AS ("
                "SELECT DISTINCT ON (location_id, year, season) location_id, year, season, pet "
                "FROM combined_yearly_avg "
                "ORDER BY location_id, year, season, source_order"
                "), year_2000_value AS ("
                "SELECT location_id, season, pet "
                "FROM deduplicated_yearly_avg "
                "WHERE year = 2000"
                "), projected AS ("
                "SELECT "
                "a.location_id, "
                "a.year, "
                "a.season, "
                "a.pet AS avg_pet, "
                "m.pet AS max_pet, "
                "l.city, "
                "l.state, "
                "p.p10, "
                "p.p90, "
                "f.lower AS future_lower, "
                "f.upper AS future_upper, "
                "ROUND((a.pet - y2k.pet)::numeric, 2)::numeric(5,2) AS change_from_2000 "
                "FROM pet_year_avg AS a "
                "JOIN locations AS l "
                "ON l.id = a.location_id "
                "AND l.id >= 0 "
                "JOIN pet_year_max AS m "
                "ON m.location_id = a.location_id "
                "AND m.year = a.year "
                "AND m.season = a.season "
                "JOIN expected_percentiles AS p "
                "ON p.location_id = a.location_id "
                "AND p.year = a.year "
                "AND p.season = a.season "
                "LEFT JOIN pet_forecast AS f "
                "ON f.location_id = a.location_id "
                "AND f.year = 2100::smallint "
                "AND f.season = a.season "
                "LEFT JOIN year_2000_value AS y2k "
                "ON y2k.location_id = a.location_id "
                "AND y2k.season = a.season "
                "WHERE a.location_id >= 0"
                "), actual AS ("
                "SELECT "
                "location_id, "
                "year, "
                "season, "
                "avg_pet, "
                "max_pet, "
                "city, "
                "state, "
                "p10, "
                "p90, "
                "future_lower, "
                "future_upper, "
                "change_from_2000 "
                "FROM city_rankings_view"
                "), missing AS ("
                "SELECT * FROM projected "
                "EXCEPT "
                "SELECT * FROM actual"
                "), extra AS ("
                "SELECT * FROM actual "
                "EXCEPT "
                "SELECT * FROM projected"
                ") "
                "SELECT "
                "(SELECT COUNT(*) FROM missing), "
                "(SELECT COUNT(*) FROM extra), "
                "(SELECT COUNT(*) FROM ("
                "SELECT location_id, year, season, COUNT(*) AS row_count "
                "FROM city_rankings_view "
                "GROUP BY location_id, year, season "
                "HAVING COUNT(*) > 1"
                ") AS duplicates)",
            )
        )
        diff_count_row = cast("tuple[Any, ...]", cur.fetchone())
        assert diff_count_row is not None

        cur.execute(
            cast(
                "LiteralString",
                "SELECT DISTINCT season FROM city_rankings_view ORDER BY season",
            )
        )
        seasons = {row[0] for row in cur.fetchall()}

    columns = _get_relation_columns(db_conn, "city_rankings_view")
    assert columns == [
        "location_id",
        "year",
        "season",
        "avg_pet",
        "max_pet",
        "city",
        "state",
        "p10",
        "p90",
        "future_lower",
        "future_upper",
        "change_from_2000",
    ]

    missing_count, extra_count, duplicate_count = diff_count_row
    assert seasons
    assert "Annual" in seasons
    assert seasons.issubset({"Annual", "Winter", "Spring", "Summer", "Fall"})
    assert missing_count == 0
    assert extra_count == 0
    assert duplicate_count == 0


def test_pet_forecast_has_annual_and_seasons(db_conn: Connection[Any]) -> None:
    """pet_forecast should expose annual plus seasonal forecasts with no duplicates."""
    with db_conn.cursor() as cur:
        cur.execute(
            cast(
                "LiteralString",
                "WITH pet_with_seasons AS ("
                "SELECT location_id, date, 'Annual'::text AS season, pet "
                "FROM pet "
                "UNION ALL "
                "SELECT "
                "location_id, "
                "date, "
                "CASE "
                "WHEN EXTRACT(MONTH FROM date)::int IN (12, 1, 2) THEN 'Winter' "
                "WHEN EXTRACT(MONTH FROM date)::int IN (3, 4, 5) THEN 'Spring' "
                "WHEN EXTRACT(MONTH FROM date)::int IN (6, 7, 8) THEN 'Summer' "
                "ELSE 'Fall' "
                "END AS season, "
                "pet "
                "FROM pet"
                "), daily_pet AS ("
                "SELECT location_id, date, season, AVG(pet) AS pet "
                "FROM pet_with_seasons "
                "GROUP BY location_id, date, season"
                "), yearly_pet AS ("
                "SELECT "
                "location_id, "
                "EXTRACT(YEAR FROM date)::int AS year, "
                "season, "
                "AVG(pet) AS pet, "
                "COUNT(*)::int AS days_present "
                "FROM daily_pet "
                "GROUP BY location_id, EXTRACT(YEAR FROM date)::int, season"
                "), complete_yearly_pet AS ("
                "SELECT location_id, year, season, pet "
                "FROM yearly_pet "
                "WHERE days_present = CASE "
                "WHEN season = 'Annual' THEN CASE "
                "WHEN MOD(year, 4) = 0 AND (MOD(year, 100) <> 0 OR MOD(year, 400) = 0) THEN 366 "
                "ELSE 365 "
                "END "
                "WHEN season = 'Winter' THEN CASE "
                "WHEN MOD(year, 4) = 0 AND (MOD(year, 100) <> 0 OR MOD(year, 400) = 0) THEN 91 "
                "ELSE 90 "
                "END "
                "WHEN season IN ('Spring', 'Summer') THEN 92 "
                "ELSE 91 "
                "END"
                "), forecast_inputs AS ("
                "SELECT "
                "location_id, "
                "season, "
                "array_agg(year ORDER BY year) AS years, "
                "array_agg(pet ORDER BY year) AS pet_values "
                "FROM complete_yearly_pet "
                "GROUP BY location_id, season"
                "), expected AS ("
                "SELECT "
                "forecast.location_id, "
                "forecast.year, "
                "inputs.season, "
                "forecast.pet, "
                "forecast.lower, "
                "forecast.upper, "
                "forecast.model_type, "
                "forecast.full_years_used, "
                "forecast.warming_rate, "
                "forecast.acceleration "
                "FROM forecast_inputs AS inputs "
                "CROSS JOIN LATERAL pet_forecast_for_location(inputs.location_id, inputs.years, inputs.pet_values) AS forecast"
                "), actual AS ("
                "SELECT "
                "location_id, "
                "year, "
                "season, "
                "pet, "
                "lower, "
                "upper, "
                "model_type, "
                "full_years_used, "
                "warming_rate, "
                "acceleration "
                "FROM pet_forecast"
                "), missing AS ("
                "SELECT * FROM expected "
                "EXCEPT "
                "SELECT * FROM actual"
                "), extra AS ("
                "SELECT * FROM actual "
                "EXCEPT "
                "SELECT * FROM expected"
                ") "
                "SELECT "
                "(SELECT COUNT(*) FROM missing), "
                "(SELECT COUNT(*) FROM extra), "
                "(SELECT COUNT(*) FROM ("
                "SELECT location_id, year, season, COUNT(*) AS row_count "
                "FROM pet_forecast "
                "GROUP BY location_id, year, season "
                "HAVING COUNT(*) > 1"
                ") AS duplicates)",
            )
        )
        diff_count_row = cast("tuple[Any, ...]", cur.fetchone())
        assert diff_count_row is not None

        cur.execute(
            cast(
                "LiteralString",
                "SELECT DISTINCT season FROM pet_forecast ORDER BY season",
            )
        )
        seasons = {row[0] for row in cur.fetchall()}

    columns = _get_relation_columns(db_conn, "pet_forecast")
    assert columns == [
        "location_id",
        "year",
        "season",
        "pet",
        "lower",
        "upper",
        "model_type",
        "full_years_used",
        "warming_rate",
        "acceleration",
    ]

    missing_count, extra_count, duplicate_count = diff_count_row
    assert seasons
    assert "Annual" in seasons
    assert seasons.issubset({"Annual", "Winter", "Spring", "Summer", "Fall"})
    assert missing_count == 0
    assert extra_count == 0
    assert duplicate_count == 0


def test_pet_forecast_matches_legacy_table_when_present(
    db_conn: Connection[Any],
) -> None:
    """Compare Annual derived pet_forecast rows to any preserved legacy table rows."""
    with db_conn.cursor() as cur:
        cur.execute(
            cast("LiteralString", "SELECT to_regclass('public.pet_forecast_legacy')")
        )
        legacy_relation_row = cast("tuple[Any, ...]", cur.fetchone())
        assert legacy_relation_row is not None
        if legacy_relation_row[0] is None:
            pytest.skip("pet_forecast_legacy not present for migration comparison")

        cur.execute(
            cast(
                "LiteralString",
                "WITH actual AS ("
                "SELECT "
                "location_id, "
                "year, "
                "ROUND(pet::numeric, 1) AS pet, "
                "ROUND(lower::numeric, 1) AS lower, "
                "ROUND(upper::numeric, 1) AS upper, "
                "model_type, "
                "full_years_used, "
                "ROUND(warming_rate::numeric, 2) AS warming_rate, "
                "ROUND(acceleration::numeric, 3) AS acceleration "
                "FROM pet_forecast "
                "WHERE season = 'Annual'"
                "), legacy AS ("
                "SELECT "
                "location_id, "
                "year, "
                "ROUND(pet::numeric, 1) AS pet, "
                "ROUND(lower::numeric, 1) AS lower, "
                "ROUND(upper::numeric, 1) AS upper, "
                "model_type, "
                "full_years_used, "
                "ROUND(warming_rate::numeric, 2) AS warming_rate, "
                "ROUND(acceleration::numeric, 3) AS acceleration "
                "FROM pet_forecast_legacy"
                "), missing AS ("
                "SELECT * FROM legacy "
                "EXCEPT "
                "SELECT * FROM actual"
                "), extra AS ("
                "SELECT * FROM actual "
                "EXCEPT "
                "SELECT * FROM legacy"
                ") "
                "SELECT "
                "(SELECT COUNT(*) FROM missing), "
                "(SELECT COUNT(*) FROM extra)",
            )
        )
        diff_count_row = cast("tuple[Any, ...]", cur.fetchone())
        assert diff_count_row is not None

    missing_count, extra_count = diff_count_row
    assert missing_count == 0
    assert extra_count == 0
