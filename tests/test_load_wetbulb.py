"""Tests for the standalone wetbulb loader."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import load_wetbulb


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


class TestParseArgs:
    def test_defaults_append_and_analyze(self) -> None:
        args = load_wetbulb._parse_args([])

        assert args.wetbulb_root == "wetbulb_data_csv"
        assert args.wetbulb_csv == "wetbulb.csv"
        assert args.truncate is False
        assert args.skip_analyze is False
        assert args.load_shard_count == 1


class TestMain:
    def test_loads_discovered_shard_and_analyzes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        batch_path = _make_shard(tmp_path)
        conn = FakeConnection()
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
        conn = FakeConnection()
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
        conn = FakeConnection()

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
