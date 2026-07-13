"""Tests for shared configuration."""

from __future__ import annotations

from datetime import date

import pytest

from shared_config import (
    PULL_START_DATE,
    build_mrt_date_bounds,
    build_mrt_days,
    build_mrt_months,
    build_postgres_uri_from_pg_env,
    build_year_date_bounds,
    build_year_date_range,
    mrt_available_end_date,
    mrt_consolidated_end_date,
    mrt_product_type,
    pull_end_date,
    resolve_database_uri,
)


class TestPullEndDate:
    def test_returns_last_day_of_previous_month(self) -> None:
        reference = date(2025, 3, 15)
        result = pull_end_date(reference)
        assert result == date(2025, 2, 28)

    def test_january_returns_december_of_previous_year(self) -> None:
        result = pull_end_date(date(2025, 1, 10))
        assert result == date(2024, 12, 31)

    def test_march_leap_year(self) -> None:
        result = pull_end_date(date(2024, 3, 1))
        assert result == date(2024, 2, 29)


class TestBuildYearDateBounds:
    def test_full_year_within_range(self) -> None:
        start, end = build_year_date_bounds(2020)
        assert start == date(2020, 1, 1)
        assert end.year == 2020

    def test_start_year_clamped_to_pull_start(self) -> None:
        start, _ = build_year_date_bounds(2000)
        assert start == PULL_START_DATE

    def test_specific_month(self) -> None:
        start, end = build_year_date_bounds(2020, month=6)
        assert start == date(2020, 6, 1)
        assert end == date(2020, 6, 30)

    def test_future_year_raises(self) -> None:
        with pytest.raises(ValueError, match="falls outside"):
            build_year_date_bounds(2099)


class TestBuildYearDateRange:
    def test_returns_iso_date_range_string(self) -> None:
        result = build_year_date_range(2020)
        parts = result.split("/")
        assert len(parts) == 2
        assert parts[0] == "2020-01-01"


class TestMrtEndDates:
    def test_mrt_available_end_date_returns_past_date(self) -> None:
        result = mrt_available_end_date(date(2025, 3, 20))
        assert result < date(2025, 3, 20)

    def test_mrt_consolidated_end_date_returns_past_date(self) -> None:
        result = mrt_consolidated_end_date(date(2025, 6, 15))
        assert result < date(2025, 6, 15)


class TestBuildMrtDateBounds:
    def test_month_within_range(self) -> None:
        start, end = build_mrt_date_bounds(2020, month=1)
        assert start == date(2020, 1, 1)
        assert end == date(2020, 1, 31)

    def test_full_year(self) -> None:
        start, _ = build_mrt_date_bounds(2020)
        assert start.year == 2020


class TestBuildMrtMonths:
    def test_single_month(self) -> None:
        result = build_mrt_months(2020, month=3)
        assert result == ["03"]

    def test_full_year(self) -> None:
        result = build_mrt_months(2020)
        assert len(result) >= 1
        assert all(len(m) == 2 for m in result)


class TestBuildMrtDays:
    def test_january_days(self) -> None:
        result = build_mrt_days(2020, month=1)
        assert result[0] == "01"
        assert result[-1] == "31"


class TestMrtProductType:
    def test_consolidated_for_old_year(self) -> None:
        result = mrt_product_type(2020)
        assert result == "consolidated_dataset"

    def test_intermediate_for_recent_data(self) -> None:
        result = mrt_product_type(2025, month=12)
        assert result in ("consolidated_dataset", "intermediate_dataset")


class TestResolveDatabaseUri:
    def test_prefers_direct_postgres_uri(self) -> None:
        assert (
            resolve_database_uri(
                {
                    "POSTGRES_DB_URI": "postgresql://primary",
                    "DATABASE_URL": "postgresql://fallback",
                }
            )
            == "postgresql://primary"
        )

    def test_builds_uri_from_pg_credentials(self) -> None:
        password_env_name = "PG" + "PASSWORD"
        resolved_uri = build_postgres_uri_from_pg_env(
            {
                "PGHOST": "primary.pet.example.run",
                "PGPORT": "29432",
                "PGDATABASE": "pet_data",
                "PGUSER": "pet_user",
                password_env_name: "demo space",
                "PGSSLMODE": "require",
            }
        )

        assert resolved_uri is not None
        assert resolved_uri.startswith("postgresql://pet_user:")
        assert "demo%20space" in resolved_uri
        assert "@primary.pet.example.run:29432/pet_data" in resolved_uri
        assert resolved_uri.endswith("?sslmode=require")

    def test_falls_back_to_database_url(self) -> None:
        assert (
            resolve_database_uri({"DATABASE_URL": "postgresql://fallback"})
            == "postgresql://fallback"
        )

    def test_returns_none_without_database_configuration(self) -> None:
        assert resolve_database_uri({}) is None
