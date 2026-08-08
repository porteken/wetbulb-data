# Copyright (C) 2026 Kenneth Porter

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import psycopg
import pytest

import load


class _Copy:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def __enter__(self) -> _Copy:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)


class _Cursor:
    def __init__(self, copy: _Copy | None = None, rowcount: int = 0) -> None:
        self.copy_result = copy or _Copy()
        self.rowcount = rowcount
        self.statements: list[object] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def copy(self, _statement: object) -> _Copy:
        return self.copy_result

    def execute(self, statement: object) -> None:
        self.statements.append(statement)


class _Connection:
    def __init__(self, cursor: _Cursor | None = None) -> None:
        self.cursor_result = cursor or _Cursor()

    def cursor(self) -> _Cursor:
        return self.cursor_result


def test_iter_sql_statements_preserves_quoted_and_commented_semicolons() -> None:
    statements = list(
        load._iter_sql_statements(
            "SELECT ';'; -- ;\n/* nested /* ; */ ; */ SELECT $$a;b$$; SELECT 3",
        )
    )

    assert statements == [
        "SELECT ';';",
        "-- ;\n/* nested /* ; */ ; */ SELECT $$a;b$$;",
        "SELECT 3",
    ]


def test_discover_shards_selects_one_group_and_rejects_multiple(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    selected = root / "shard_count=00002" / "part=one" / "input.csv"
    selected.parent.mkdir(parents=True)
    selected.touch()

    assert load._discover_shard_csv_inputs(
        shard_root=root,
        shard_file_name="input.csv",
        shard_count=None,
        shard_partition_key="shard_count",
    ) == [selected]

    other = root / "shard_count=00003" / "part=one" / "input.csv"
    other.parent.mkdir(parents=True)
    other.touch()
    with pytest.raises(RuntimeError, match="multiple analytics shard groups"):
        load._discover_shard_csv_inputs(
            shard_root=root,
            shard_file_name="input.csv",
            shard_count=None,
            shard_partition_key="shard_count",
        )


def test_discover_and_select_partitioned_paths(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    paths = []
    for year in ("2023", "2024", "alpha"):
        path = root / f"year={year}" / "data.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths.append(path)

    assert (
        load._discover_csv_inputs(
            "missing.csv", shard_root=root, shard_file_name="data.csv"
        )
        == paths
    )
    assert load._select_partition_shard_paths(
        paths,
        root_path=root,
        partition_key="year",
        shard_index=1,
        shard_count=2,
    ) == [paths[1]]
    with pytest.raises(ValueError, match="load_shard_index"):
        load._select_partition_shard_paths(
            paths,
            root_path=root,
            partition_key="year",
            shard_index=2,
            shard_count=2,
        )


def test_filter_paths_by_year_rejects_invalid_and_unpartitioned_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inputs"
    valid = root / "year=2024" / "data.csv"
    invalid = root / "year=unknown" / "data.csv"
    for path in (valid, invalid):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert load._filter_paths_by_year([valid], str(root), 2024, 2024) == [valid]
    with pytest.raises(ValueError, match="Invalid wetbulb year partition"):
        load._filter_paths_by_year([invalid], str(root), None, None)
    with pytest.raises(ValueError, match="unpartitioned"):
        load._filter_paths_by_year([tmp_path / "data.csv"], str(root), 2024, None)


def test_file_column_discovery_and_csv_copy(tmp_path: Path) -> None:
    csv_path = tmp_path / "locations.csv"
    csv_path.write_text("location_id, city\n1,Alpha\n", encoding="utf-8")
    empty_path = tmp_path / "empty.csv"
    empty_path.touch()
    cursor = _Cursor(rowcount=1)

    assert load._file_copy_column_names("locations", csv_path) == ["id", "city"]
    assert load._file_copy_column_names("locations", empty_path) == []
    assert load._union_copy_column_names("locations", [csv_path]) == ["id", "city"]
    assert (
        load._copy_csv_file_in_batches(
            _Connection(cursor), "locations", csv_path, batch_size=1
        )
        == 1
    )
    assert cursor.copy_result.chunks == ["1,Alpha\n"]


def test_validated_path_and_bulk_insert_control_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("id\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load._validated_load_path(tmp_path.parent, base_dir=tmp_path)

    transaction = MagicMock()
    transaction.__enter__.return_value = transaction
    transaction.__exit__.return_value = None
    connection = MagicMock()
    connection.transaction.return_value = transaction
    monkeypatch.setattr(load, "_validated_load_path", lambda path: path)
    monkeypatch.setattr(load, "_union_copy_column_names", lambda *_args: ["id"])
    monkeypatch.setattr(load, "_create_staging_table", lambda *_args: "staging")
    monkeypatch.setattr(load, "_copy_file", lambda *_args, **_kwargs: 1)
    upsert = MagicMock(return_value=1)
    monkeypatch.setattr(load, "_upsert_from_staging", upsert)

    load.bulk_insert_csv_files(
        connection, [source], "locations", batch_size=10, truncate=True
    )

    upsert.assert_called_once()


def test_execute_sql_and_planner_helpers(tmp_path: Path) -> None:
    script = tmp_path / "schema.sql"
    script.write_text("SELECT 1; SELECT 2;", encoding="utf-8")
    cursor = _Cursor()
    connection = MagicMock()
    connection.cursor.return_value = cursor
    connection.transaction.return_value.__enter__.return_value = connection
    connection.transaction.return_value.__exit__.return_value = None

    load.execute_sql_file(connection, script)
    load.execute_sql_files_atomically(connection, (script,))
    load.refresh_query_planner_statistics(connection)

    assert len(cursor.statements) == 2 + 2 + len(load.TABLE_NAMES)


def test_sql_retry_helpers_reconnect_after_operational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections = [MagicMock(), MagicMock()]
    monkeypatch.setattr(load.psycopg, "connect", MagicMock(side_effect=connections))
    monkeypatch.setattr(
        load,
        "execute_sql_files_atomically",
        MagicMock(side_effect=[psycopg.OperationalError("lost"), None]),
    )
    monkeypatch.setattr(load.time, "sleep", lambda _seconds: None)

    load.execute_sql_files_with_retries("postgresql://test", ("schema.sql",))

    assert all(connection.close.called for connection in connections)


def test_copy_file_and_upsert_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id\n1\n", encoding="utf-8")
    cursor = _Cursor(rowcount=3)
    connection = _Connection(cursor)

    assert load._copy_file(connection, "locations", csv_path, batch_size=2) == 3
    assert (
        load._create_staging_table(connection, "locations", ["id"])
        == "locations_load_staging"
    )
    assert (
        load._upsert_from_staging(
            connection, "locations", "staging", ["id", "city"], ("id",)
        )
        == 3
    )
    assert (
        load._upsert_from_staging(connection, "locations", "staging", ["id"], ()) == 3
    )

    parquet = tmp_path / "data.parquet"
    parquet.touch()
    batch = MagicMock(num_rows=2)
    parquet_file = MagicMock()
    parquet_file.schema_arrow.names = ["id"]
    parquet_file.iter_batches.return_value = [batch]
    monkeypatch.setattr(load.pq, "ParquetFile", lambda _path: parquet_file)
    monkeypatch.setattr(load.pacsv, "write_csv", lambda *_args, **_kwargs: None)
    assert load._copy_file(connection, "locations", parquet, batch_size=2) == 2


def test_load_requested_tables_and_parallel_file_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = load._parse_args(["--append-only", "--load-workers", "2"])
    actual_load_table_files = load._load_table_files
    loaded = MagicMock()
    monkeypatch.setattr(load, "_load_table_files", loaded)
    monkeypatch.setattr(
        load, "_discover_locations_csv_paths", lambda _args: [Path("a")]
    )
    monkeypatch.setattr(load, "_discover_wetbulb_csv_paths", lambda _args: [])
    monkeypatch.setattr(load, "_discover_gmst_observations_csv_paths", lambda _args: [])
    monkeypatch.setattr(load, "_discover_gmst_scenarios_csv_paths", lambda _args: [])
    monkeypatch.setattr(
        load, "_discover_forecast_station_groups_csv_paths", lambda _args: []
    )
    monkeypatch.setattr(
        load, "_discover_forecast_interval_calibration_csv_paths", lambda _args: []
    )

    load._load_requested_tables(
        MagicMock(),
        args,
        db_uri="postgresql://test",
        truncate_tables=set(),
        skip_tables=set(),
    )

    assert loaded.call_count == 1
    group = MagicMock()
    monkeypatch.setattr(load, "_load_file_group", group)
    actual_load_table_files(
        MagicMock(),
        "postgresql://test",
        [Path("one"), Path("two")],
        "wetbulb",
        batch_size=10,
        truncate=False,
        workers=2,
    )
    assert group.call_count == 2


def test_main_skips_without_database_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["load.py"])
    monkeypatch.setattr(load, "resolve_database_uri", lambda: "")
    load.main()

    connection = MagicMock()
    monkeypatch.setattr(load, "resolve_database_uri", lambda: "postgresql://test")
    monkeypatch.setattr(load.psycopg, "connect", lambda _uri: connection)
    monkeypatch.setattr(
        load, "_refresh_schema_before_load", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(load, "_load_requested_tables", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(load, "_refresh_views", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        load, "refresh_query_planner_statistics_with_retries", lambda _uri: None
    )
    load.main()

    assert connection.close.called
