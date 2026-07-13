"""Shared configuration for weather, MRT, and database scripts."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

if TYPE_CHECKING:
    from collections.abc import Mapping

PULL_START_DATE = date(2000, 1, 1)
SHARED_AREA: tuple[float, float, float, float] = (49.25, -124.5, 24.25, -66.5)
DECEMBER = 12
MRT_INTERMEDIATE_LAG_DAYS = 5
MRT_CONSOLIDATED_LAG_MONTHS = 3
PRIMARY_DATABASE_URI_ENV_VARS: tuple[str, ...] = (
    "POSTGRES_DB_URI",
    "DATABASE_URL",
)
POSTGRES_COMPONENT_ENV_VARS: tuple[str, ...] = (
    "PGHOST",
    "PGDATABASE",
    "PGUSER",
    "PGPASSWORD",
)
DATABASE_CONFIG_HINT = (
    "Set POSTGRES_DB_URI, DATABASE_URL, or PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."
)


def _resolve_environ(environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Return the supplied environment mapping or the current process environment."""
    # noinspection PyTypeChecker
    return os.environ if environ is None else environ


def build_postgres_uri_from_pg_env(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Build a Postgres connection URI from standard libpq environment variables."""
    resolved_environ = _resolve_environ(environ)
    if any(
        not resolved_environ.get(env_name) for env_name in POSTGRES_COMPONENT_ENV_VARS
    ):
        return None

    username = quote(resolved_environ["PGUSER"], safe="")
    password = quote(resolved_environ["PGPASSWORD"], safe="")
    database = quote(resolved_environ["PGDATABASE"], safe="")
    netloc = f"{username}:{password}@{resolved_environ['PGHOST']}"
    if port := resolved_environ.get("PGPORT"):
        netloc = f"{netloc}:{port}"
    query = ""
    if sslmode := resolved_environ.get("PGSSLMODE"):
        query = urlencode({"sslmode": sslmode})
    return f"postgresql://{netloc}/{database}{f'?{query}' if query else ''}"


def resolve_database_uri(
    environ: Mapping[str, str] | None = None,
    *,
    direct_env_vars: tuple[str, ...] = PRIMARY_DATABASE_URI_ENV_VARS,
) -> str | None:
    """Resolve the active Postgres database URI from direct or component env vars."""
    resolved_environ = _resolve_environ(environ)
    for env_name in direct_env_vars:
        if env_value := resolved_environ.get(env_name):
            return env_value
    return build_postgres_uri_from_pg_env(resolved_environ)


def _resolve_current_date(current_date: date | None = None) -> date:
    """Return the current UTC date or the provided override."""
    return datetime.now(tz=UTC).date() if current_date is None else current_date


def _month_start(current_date: date) -> date:
    """Return the first day of the month for a date."""
    return current_date.replace(day=1)


def _shift_month_start(current_date: date, months: int) -> date:
    """Shift a month-start date by a signed month delta."""
    month_index = current_date.year * 12 + (current_date.month - 1) + months
    shifted_year, shifted_month_index = divmod(month_index, 12)
    return date(shifted_year, shifted_month_index + 1, 1)


def pull_end_date(current_date: date | None = None) -> date:
    """Return the last day of the month before the current date."""
    reference_date = _resolve_current_date(current_date)
    return reference_date.replace(day=1) - timedelta(days=1)


def _month_end(year: int, month: int) -> date:
    """Return the last day for the requested month."""
    if month == DECEMBER:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def _build_bounds(
    year: int,
    end_date: date,
    *,
    month: int | None = None,
    name: str = "Requested window",
) -> tuple[date, date]:
    """Build bounds and validate ranges."""
    range_start = max(PULL_START_DATE, date(year, 1, 1))
    range_end = min(end_date, date(year, 12, 31))

    if month is not None:
        month_start = date(year, month, 1)
        range_start = max(range_start, month_start)
        range_end = min(range_end, _month_end(year, month))

    if range_start > range_end:
        msg = (
            f"{name} for year {year}"
            f"{'' if month is None else f', month {month:02d}'} "
            f"falls outside the supported pull window "
            f"{PULL_START_DATE.isoformat()} to {end_date.isoformat()}."
        )
        raise ValueError(msg)

    return range_start, range_end


def build_year_date_bounds(year: int, *, month: int | None = None) -> tuple[date, date]:
    """Return the supported inclusive start/end dates for a year or month."""
    return _build_bounds(year, pull_end_date(), month=month)


def mrt_available_end_date(current_date: date | None = None) -> date:
    """Return the latest month-end safely available from the MRT dataset."""
    reference_date = _resolve_current_date(current_date) - timedelta(
        days=MRT_INTERMEDIATE_LAG_DAYS,
    )
    return _month_start(reference_date) - timedelta(days=1)


def mrt_consolidated_end_date(current_date: date | None = None) -> date:
    """Return the latest month-end safely available in consolidated MRT data."""
    reference_date = _resolve_current_date(current_date)
    consolidated_month_start = _shift_month_start(
        _month_start(reference_date),
        -MRT_CONSOLIDATED_LAG_MONTHS,
    )
    return _month_end(
        consolidated_month_start.year,
        consolidated_month_start.month,
    )


def build_mrt_date_bounds(year: int, *, month: int | None = None) -> tuple[date, date]:
    """Return the supported inclusive MRT start/end dates for a year or month."""
    return _build_bounds(
        year,
        mrt_available_end_date(),
        month=month,
        name="Requested MRT window",
    )


def build_mrt_months(year: int, *, month: int | None = None) -> list[str]:
    """Return the supported MRT month list for a specific year or month."""
    range_start, range_end = build_mrt_date_bounds(year, month=month)
    return [
        f"{month_value:02d}"
        for month_value in range(range_start.month, range_end.month + 1)
    ]


def build_mrt_days(year: int, *, month: int) -> list[str]:
    """Return the supported MRT day list for a specific month."""
    range_start, range_end = build_mrt_date_bounds(year, month=month)
    return [
        f"{day_value:02d}" for day_value in range(range_start.day, range_end.day + 1)
    ]


def mrt_product_type(year: int, *, month: int | None = None) -> str:
    """Return the MRT dataset variant to request for a year or month."""
    _, range_end = build_mrt_date_bounds(year, month=month)
    if range_end <= mrt_consolidated_end_date():
        return "consolidated_dataset"
    return "intermediate_dataset"


def build_year_date_range(year: int, *, month: int | None = None) -> str:
    """Return the supported pull window for a specific year or month."""
    range_start, range_end = build_year_date_bounds(year, month=month)
    return f"{range_start.isoformat()}/{range_end.isoformat()}"


def build_year_months(year: int, *, month: int | None = None) -> list[str]:
    """Return the supported month list for a specific year or month."""
    range_start, range_end = build_year_date_bounds(year, month=month)
    return [
        f"{month_value:02d}"
        for month_value in range(range_start.month, range_end.month + 1)
    ]


def build_full_date_range(*, start_year: int | None = None) -> str:
    """Return the full supported pull window, optionally clamped to a start year."""
    end_date = pull_end_date()
    effective_start = PULL_START_DATE
    if start_year is not None:
        effective_start = max(PULL_START_DATE, date(start_year, 1, 1))
    return f"{effective_start.isoformat()}/{end_date.isoformat()}"


def shared_area() -> list[float]:
    """Return geographic bounding box for defaults."""
    return list(SHARED_AREA)
