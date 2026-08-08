# Copyright (C) 2026 Kenneth Porter

"""Tests for the standalone wetbulb loader."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import psycopg
import pytest

import load
import load_wetbulb
from load import execute_sql_files_atomically


class FakeCursor:
    def __init__(self, parent: FakeConnection) -> None:
        """Initialize the fake cursor."""
        self.parent = parent

    def execute(self, statement: object, params: object | None = None) -> None:
        self.parent.executed_statements.append((statement, params))

    def fetchone(self) -> tuple[str | None] | None:
        return self.parent.regclass_value

    def __enter__(self) -> FakeCursor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)


class FakeConnection:
    def __init__(self, regclass_value: tuple[str | None] | None = None) -> None:
        """Initialize the fake connection."""
        self.executed_statements: list[tuple[object, object | None]] = []
        self.regclass_value = regclass_value or ("public.wetbulb",)
        self.autocommit = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = True


class TransactionConnection(FakeConnection):
    def transaction(self) -> TransactionConnection:
        return self

    def __enter__(self) -> TransactionConnection:
        """Enter the fake transaction context."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit the fake transaction context."""
        _ = (exc_type, exc, tb)


def _make_shard(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "wetbulb_data_csv" / "year=2024"
    shard_dir.mkdir(parents=True)
    batch_path = shard_dir / "wetbulb_batch_0000_00.parquet"
    batch_path.touch()
    return batch_path


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--wetbulb-root",
        str(tmp_path / "wetbulb_data_csv"),
        "--wetbulb-csv",
        str(tmp_path / "wetbulb.csv"),
        *extra,
    ]


def test_discovery_includes_eccc_station_batches(tmp_path: Path) -> None:
    root = tmp_path / "wetbulb_data_csv" / "year=2025"
    root.mkdir(parents=True)
    ghcnh_path = root / "wetbulb_batch_0000_00.parquet"
    eccc_path = root / "wetbulb_eccc_batch_0000_00.parquet"
    fill_path = root / "wetbulb_fill_batch_0000_00.parquet"
    for path in (ghcnh_path, eccc_path, fill_path):
        path.touch()
    args = load._parse_args(
        [
            "--wetbulb-root",
            str(tmp_path / "wetbulb_data_csv"),
            "--wetbulb-csv",
            str(tmp_path / "wetbulb.csv"),
        ]
    )

    discovered = load._discover_wetbulb_csv_paths(args)

    assert discovered == [ghcnh_path, eccc_path, fill_path]


def test_discovery_does_not_repeat_direct_csv_for_absent_optional_batches(
    tmp_path: Path,
) -> None:
    primary_path = _make_shard(tmp_path)
    direct_path = tmp_path / "wetbulb.csv"
    direct_path.touch()
    args = load._parse_args(
        [
            "--wetbulb-root",
            str(tmp_path / "wetbulb_data_csv"),
            "--wetbulb-csv",
            str(direct_path),
        ]
    )

    assert load._discover_wetbulb_csv_paths(args) == [primary_path]


def test_file_group_retries_each_file_with_a_fresh_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [Path("first.parquet"), Path("second.parquet")]
    connections = [FakeConnection(), FakeConnection(), FakeConnection()]
    connect = MagicMock(side_effect=connections)
    bulk_insert = MagicMock(
        side_effect=[psycopg.OperationalError("SSL EOF"), None, None]
    )
    monkeypatch.setattr(load.psycopg, "connect", connect)
    monkeypatch.setattr(load, "bulk_insert_csv_files", bulk_insert)
    monkeypatch.setattr(load.time, "sleep", lambda _seconds: None)

    load._load_file_group(
        "postgresql://test",
        paths,
        "wetbulb",
        batch_size=100,
    )

    assert [call.args[1] for call in bulk_insert.call_args_list] == [
        [paths[0]],
        [paths[0]],
        [paths[1]],
    ]
    assert all(connection.closed for connection in connections)


def test_load_file_reraises_the_final_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection()
    monkeypatch.setattr(load, "LOAD_FILE_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(load.psycopg, "connect", lambda _uri: connection)
    monkeypatch.setattr(
        load,
        "bulk_insert_csv_files",
        MagicMock(side_effect=psycopg.OperationalError("SSL EOF")),
    )

    with pytest.raises(psycopg.OperationalError, match="SSL EOF"):
        load._load_file_with_retries(
            "postgresql://test",
            Path("first.parquet"),
            "wetbulb",
            batch_size=100,
        )

    assert connection.closed is True


class TestParseArgs:
    def test_defaults_append_and_analyze(self) -> None:
        args = load_wetbulb._parse_args([])

        assert args.wetbulb_root == "wetbulb_data_csv"
        assert args.wetbulb_csv == "wetbulb.csv"
        assert args.truncate is False
        assert args.skip_analyze is False
        assert args.load_shard_count == 1


def test_execute_sql_files_atomically_uses_one_transaction(tmp_path: Path) -> None:
    first = tmp_path / "first.sql"
    second = tmp_path / "second.sql"
    first.write_text("SELECT 1;", encoding="utf-8")
    second.write_text("SELECT 2;", encoding="utf-8")
    conn = TransactionConnection()

    execute_sql_files_atomically(cast("Any", conn), (first, second))

    assert [str(statement) for statement, _ in conn.executed_statements] == [
        "SELECT 1;",
        "SELECT 2;",
    ]


def test_atomic_sql_reconnects_after_operational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections = [TransactionConnection(), TransactionConnection()]
    execute = MagicMock(side_effect=[psycopg.OperationalError("SSL EOF"), None])
    monkeypatch.setattr(load.psycopg, "connect", MagicMock(side_effect=connections))
    monkeypatch.setattr(load, "execute_sql_files_atomically", execute)
    monkeypatch.setattr(load.time, "sleep", lambda _seconds: None)

    load.execute_sql_files_with_retries(
        "postgresql://test",
        ("create_views.sql",),
    )

    assert execute.call_count == 2
    assert all(connection.closed for connection in connections)


class TestMain:
    def test_loads_discovered_shard_and_analyzes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        batch_path = _make_shard(tmp_path)
        conn = TransactionConnection()
        bulk_insert = MagicMock()

        monkeypatch.setattr(sys, "argv", ["load_wetbulb.py", *_argv(tmp_path)])
        monkeypatch.setattr(
            load_wetbulb, "resolve_database_uri", lambda: "postgresql://x"
        )
        monkeypatch.setattr(load_wetbulb.psycopg, "connect", lambda _uri: conn)
        monkeypatch.setattr(load_wetbulb, "_load_table_files", bulk_insert)

        load_wetbulb.main()

        assert bulk_insert.call_args.args[2] == [batch_path]
        assert bulk_insert.call_args.args[3] == "wetbulb"
        assert bulk_insert.call_args.kwargs["truncate"] is False
        assert ("ANALYZE public.wetbulb", None) in conn.executed_statements
        assert conn.closed is True

    def test_truncate_flag_is_forwarded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _make_shard(tmp_path)
        conn = TransactionConnection()
        bulk_insert = MagicMock()

        monkeypatch.setattr(
            sys, "argv", ["load_wetbulb.py", *_argv(tmp_path, "--truncate")]
        )
        monkeypatch.setattr(
            load_wetbulb, "resolve_database_uri", lambda: "postgresql://x"
        )
        monkeypatch.setattr(load_wetbulb.psycopg, "connect", lambda _uri: conn)
        monkeypatch.setattr(load_wetbulb, "_load_table_files", bulk_insert)

        load_wetbulb.main()

        assert bulk_insert.call_args.kwargs["truncate"] is True

    def test_skip_analyze_omits_analyze(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _make_shard(tmp_path)
        conn = TransactionConnection()

        monkeypatch.setattr(
            sys, "argv", ["load_wetbulb.py", *_argv(tmp_path, "--skip-analyze")]
        )
        monkeypatch.setattr(
            load_wetbulb, "resolve_database_uri", lambda: "postgresql://x"
        )
        monkeypatch.setattr(load_wetbulb.psycopg, "connect", lambda _uri: conn)
        monkeypatch.setattr(load_wetbulb, "_load_table_files", MagicMock())

        load_wetbulb.main()

        statements = [stmt for stmt, _ in conn.executed_statements]
        assert "ANALYZE public.wetbulb" not in statements

    def test_raises_when_wetbulb_table_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _make_shard(tmp_path)
        conn = FakeConnection(regclass_value=(None,))

        monkeypatch.setattr(sys, "argv", ["load_wetbulb.py", *_argv(tmp_path)])
        monkeypatch.setattr(
            load_wetbulb, "resolve_database_uri", lambda: "postgresql://x"
        )
        monkeypatch.setattr(load_wetbulb.psycopg, "connect", lambda _uri: conn)
        monkeypatch.setattr(load_wetbulb, "_load_table_files", MagicMock())

        with pytest.raises(SystemExit, match="does not exist"):
            load_wetbulb.main()

        assert conn.closed is True

    def test_skips_when_no_inputs_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        connect = MagicMock()
        monkeypatch.setattr(sys, "argv", ["load_wetbulb.py", *_argv(tmp_path)])
        monkeypatch.setattr(
            load_wetbulb, "resolve_database_uri", lambda: "postgresql://x"
        )
        monkeypatch.setattr(load_wetbulb.psycopg, "connect", connect)

        load_wetbulb.main()

        assert not connect.called

    def test_skips_without_database_uri(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _make_shard(tmp_path)
        connect = MagicMock()
        monkeypatch.setattr(sys, "argv", ["load_wetbulb.py", *_argv(tmp_path)])
        monkeypatch.setattr(load_wetbulb, "resolve_database_uri", lambda: "")
        monkeypatch.setattr(load_wetbulb.psycopg, "connect", connect)

        load_wetbulb.main()

        assert not connect.called
