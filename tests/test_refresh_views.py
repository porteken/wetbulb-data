"""Tests for dependency-ordered materialized view refresh."""

from __future__ import annotations

import logging
import sys
from typing import Any, cast
from unittest.mock import MagicMock

import psycopg
import pytest

import refresh_views


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        """Route queries to canned results on the parent connection."""
        self._connection = connection
        self._results: list[tuple[Any, ...]] = []

    def execute(self, statement: str, params: object = None) -> None:
        self._connection.executed_statements.append(statement)
        _ = params
        if "pg_matviews" in statement:
            self._results = list(self._connection.matviews)
        elif "pg_rewrite" in statement:
            self._results = list(self._connection.dependencies)
        elif "indisunique" in statement:
            table = cast("tuple[str]", params)[0]
            self._results = [(1,)] if table in self._connection.unique_indexed else []
        else:
            self._results = []

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._results

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._results[0] if self._results else None

    def __enter__(self) -> FakeCursor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)


class FakeConnection:
    def __init__(
        self,
        matviews: list[tuple[str, bool]],
        dependencies: list[tuple[str, str]] | None = None,
        unique_indexed: set[str] | None = None,
    ) -> None:
        """Record executed statements and serve canned catalog results."""
        self.matviews = matviews
        self.dependencies = dependencies or []
        self.unique_indexed = unique_indexed or set()
        self.executed_statements: list[str] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


def _refresh_statements(conn: FakeConnection) -> list[str]:
    return [
        statement
        for statement in conn.executed_statements
        if statement.startswith("REFRESH")
    ]


class TestRefreshMaterializedViews:
    def test_refreshes_dependencies_first(self) -> None:
        conn = FakeConnection(
            matviews=[("mv_b", True), ("mv_a", True)],
            dependencies=[("mv_b", "mv_a")],
            unique_indexed={"mv_a", "mv_b"},
        )

        order = refresh_views.refresh_materialized_views(cast("Any", conn))

        assert order.index("mv_a") < order.index("mv_b")
        assert _refresh_statements(conn) == [
            "REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_a",
            "REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_b",
        ]

    def test_unpopulated_or_unindexed_matviews_refresh_non_concurrently(
        self,
    ) -> None:
        conn = FakeConnection(
            matviews=[("mv_plain", True), ("mv_empty", False)],
            unique_indexed={"mv_empty"},
        )

        refresh_views.refresh_materialized_views(cast("Any", conn))

        statements = _refresh_statements(conn)
        assert "REFRESH MATERIALIZED VIEW public.mv_plain" in statements
        assert "REFRESH MATERIALIZED VIEW public.mv_empty" in statements

    def test_can_force_lower_overhead_nonconcurrent_refresh(self) -> None:
        conn = FakeConnection(
            matviews=[("mv_indexed", True)],
            unique_indexed={"mv_indexed"},
        )

        refresh_views.refresh_materialized_views(
            cast("Any", conn),
            allow_concurrent=False,
        )

        assert _refresh_statements(conn) == [
            "REFRESH MATERIALIZED VIEW public.mv_indexed"
        ]

    def test_no_matviews_warns_and_returns_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = FakeConnection(matviews=[])

        with caplog.at_level(logging.WARNING):
            order = refresh_views.refresh_materialized_views(cast("Any", conn))

        assert order == []
        assert "No materialized views" in caplog.text

    def test_can_filter_matviews_by_prefix(self) -> None:
        conn = FakeConnection(
            matviews=[("wetbulb_stats", True), ("pet_stats", True)],
            unique_indexed={"wetbulb_stats", "pet_stats"},
        )

        order = refresh_views.refresh_materialized_views(
            cast("Any", conn),
            name_prefix="wetbulb_",
        )

        assert order == ["wetbulb_stats"]
        assert _refresh_statements(conn) == [
            "REFRESH MATERIALIZED VIEW CONCURRENTLY public.wetbulb_stats"
        ]


def test_configure_refresh_session_disables_parallel_sort_workers() -> None:
    conn = FakeConnection(matviews=[])

    refresh_views.configure_refresh_session(cast("Any", conn))

    assert "SET max_parallel_workers_per_gather = 0" in conn.executed_statements
    assert "SET max_parallel_maintenance_workers = 0" in conn.executed_statements


class TestMain:
    def test_skips_without_database_credentials(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["refresh_views.py"])
        monkeypatch.setattr(refresh_views, "resolve_database_uri", lambda: None)
        connect = MagicMock()
        monkeypatch.setattr(refresh_views.psycopg, "connect", connect)

        with caplog.at_level(logging.WARNING):
            refresh_views.main()

        assert not connect.called
        assert "not configured" in caplog.text

    def test_refreshes_and_analyzes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = MagicMock()
        monkeypatch.setattr(
            refresh_views, "resolve_database_uri", lambda: "postgresql://x"
        )
        monkeypatch.setattr(refresh_views.psycopg, "connect", lambda _uri: conn)
        refresh = MagicMock(return_value=["mv_a"])
        analyze = MagicMock()
        monkeypatch.setattr(refresh_views, "refresh_materialized_views", refresh)
        monkeypatch.setattr(refresh_views, "refresh_query_planner_statistics", analyze)
        monkeypatch.setattr(sys, "argv", ["refresh_views.py"])

        refresh_views.main()

        assert refresh.called
        assert analyze.called
        assert conn.close.called

    def test_skip_analyze_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = MagicMock()
        monkeypatch.setattr(
            refresh_views, "resolve_database_uri", lambda: "postgresql://x"
        )
        monkeypatch.setattr(refresh_views.psycopg, "connect", lambda _uri: conn)
        monkeypatch.setattr(
            refresh_views, "refresh_materialized_views", MagicMock(return_value=[])
        )
        analyze = MagicMock()
        monkeypatch.setattr(refresh_views, "refresh_query_planner_statistics", analyze)
        monkeypatch.setattr(sys, "argv", ["refresh_views.py", "--skip-analyze"])

        refresh_views.main()

        assert not analyze.called

    def test_reconnects_after_operational_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        connections = [MagicMock(), MagicMock()]
        connect = MagicMock(side_effect=connections)
        refresh = MagicMock(side_effect=[psycopg.OperationalError("SSL EOF"), []])
        monkeypatch.setattr(
            refresh_views,
            "resolve_database_uri",
            lambda: "postgresql://x",
        )
        monkeypatch.setattr(refresh_views.psycopg, "connect", connect)
        monkeypatch.setattr(refresh_views, "refresh_materialized_views", refresh)
        monkeypatch.setattr(
            refresh_views,
            "refresh_query_planner_statistics",
            MagicMock(),
        )
        monkeypatch.setattr(refresh_views.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(sys, "argv", ["refresh_views.py"])

        refresh_views.main()

        assert refresh.call_count == 2
        assert all(connection.close.called for connection in connections)

    def test_does_not_retry_storage_internal_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = MagicMock()
        connect = MagicMock(return_value=conn)
        refresh = MagicMock(side_effect=psycopg.InternalError("unexpected end of tape"))
        monkeypatch.setattr(
            refresh_views,
            "resolve_database_uri",
            lambda: "postgresql://x",
        )
        monkeypatch.setattr(refresh_views.psycopg, "connect", connect)
        monkeypatch.setattr(refresh_views, "refresh_materialized_views", refresh)
        monkeypatch.setattr(sys, "argv", ["refresh_views.py"])

        with caplog.at_level(logging.ERROR), pytest.raises(psycopg.InternalError):
            refresh_views.main()

        assert refresh.call_count == 1
        assert "temporary storage" in caplog.text
