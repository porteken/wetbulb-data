"""Tests for CSV discovery and DB loading."""

from __future__ import annotations

import argparse
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

import load
from load import (
    TABLE_NAMES,
    _discover_csv_inputs,
    _discover_locations_csv_paths,
    _discover_pet_csv_paths,
    _discover_wetbulb_csv_paths,
    _extract_partition_marker,
    _iter_sql_statements,
    _normalize_copy_column_names,
    _select_partition_shard_paths,
    _validate_load_shard_args,
    _validated_load_path,
    execute_sql_file,
    refresh_query_planner_statistics,
)


class FakeCursor:
    def __init__(self, executed_statements: list[str]) -> None:
        """Initialize the fake cursor."""
        self.executed_statements = executed_statements

    def execute(self, statement: str) -> None:
        self.executed_statements.append(statement)

    def __enter__(self) -> FakeCursor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)


class FakeConnection:
    def __init__(self) -> None:
        """Initialize the fake connection."""
        self.executed_statements: list[str] = []
        self.transaction_count = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.executed_statements)

    @contextmanager
    def transaction(self) -> Generator[None]:
        self.transaction_count += 1
        yield


class TestParseArgs:
    def test_default_args(self) -> None:
        args = load._parse_args([])
        assert args.locations_csv == "locations.csv"
        assert args.pet_csv == "pet.csv"
        assert args.load_shard_index == 0
        assert args.load_shard_count == 1

    def test_custom_args(self) -> None:
        args = load._parse_args(
            [
                "--pet-csv",
                "other.csv",
                "--load-shard-index",
                "1",
                "--load-shard-count",
                "2",
            ]
        )
        assert args.pet_csv == "other.csv"
        assert args.load_shard_index == 1
        assert args.load_shard_count == 2


class TestDiscoverShardCsvInputs:
    def test_returns_empty_when_root_none(self) -> None:
        assert (
            load._discover_shard_csv_inputs(
                shard_root=None,
                shard_file_name="f.csv",
                shard_count=1,
                shard_partition_key="k",
            )
            == []
        )

    def test_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        assert (
            load._discover_shard_csv_inputs(
                shard_root=tmp_path / "missing",
                shard_file_name="f.csv",
                shard_count=1,
                shard_partition_key="k",
            )
            == []
        )

    def test_raises_on_multiple_groups(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        (root / "k=1").mkdir(parents=True)
        (root / "k=1" / "f.csv").touch()
        (root / "k=2").mkdir(parents=True)
        (root / "k=2" / "f.csv").touch()

        with pytest.raises(RuntimeError, match="multiple analytics shard groups"):
            load._discover_shard_csv_inputs(
                shard_root=root,
                shard_file_name="f.csv",
                shard_count=None,
                shard_partition_key="k",
            )


class TestExecuteSqlFileErrors:
    def test_logs_warning_on_missing_file(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = FakeConnection()
        with caplog.at_level(logging.WARNING):
            execute_sql_file(cast("Any", conn), "nonexistent.sql")
        assert "SQL file nonexistent.sql not found" in caplog.text

    def test_skips_empty_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sql_path = tmp_path / "empty.sql"
        sql_path.touch()
        conn = FakeConnection()
        with caplog.at_level(logging.INFO):
            execute_sql_file(cast("Any", conn), sql_path)
        assert "is empty" in caplog.text


class TestValidatedLoadPath:
    def test_accepts_path_inside_base_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "data.csv"
        nested.parent.mkdir()
        nested.write_text("id\n1\n")

        assert _validated_load_path(nested, base_dir=tmp_path) == nested.resolve()

    def test_accepts_base_dir_itself(self, tmp_path: Path) -> None:
        assert _validated_load_path(tmp_path, base_dir=tmp_path) == tmp_path.resolve()

    def test_rejects_path_outside_base_dir(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.csv"

        with pytest.raises(ValueError, match="outside the allowed directory"):
            _validated_load_path(outside, base_dir=tmp_path)

    def test_rejects_traversal_escaping_base_dir(self, tmp_path: Path) -> None:
        traversal = tmp_path / "sub" / ".." / ".." / "secret.csv"

        with pytest.raises(ValueError, match="outside the allowed directory"):
            _validated_load_path(traversal, base_dir=tmp_path)

    def test_rejects_sibling_directory_with_shared_prefix(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "data"
        base_dir.mkdir()
        sibling = tmp_path / "data-secret" / "leak.csv"

        with pytest.raises(ValueError, match="outside the allowed directory"):
            _validated_load_path(sibling, base_dir=base_dir)


class TestValidateLoadShardArgs:
    def test_valid_args(self) -> None:
        _validate_load_shard_args(0, 1)
        _validate_load_shard_args(0, 5)
        _validate_load_shard_args(4, 5)

    def test_invalid_shard_count(self) -> None:
        with pytest.raises(ValueError, match="shard_count"):
            _validate_load_shard_args(0, 0)

    def test_invalid_shard_index(self) -> None:
        with pytest.raises(ValueError, match="shard_index"):
            _validate_load_shard_args(5, 5)

        with pytest.raises(ValueError, match="shard_index"):
            _validate_load_shard_args(-1, 5)


class TestExtractPartitionMarker:
    def test_extracts_value(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "shard_count=00020" / "shard_index=00003" / "data.csv.gz"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        result = _extract_partition_marker(csv_path, tmp_path, "shard_count")
        assert result == "00020"

    def test_returns_none_for_missing_key(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "other=5" / "data.csv.gz"
        result = _extract_partition_marker(csv_path, tmp_path, "shard_count")
        assert result is None


class TestDiscoverCsvInputs:
    def test_direct_path_fallback(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "pet.csv.gz"
        csv_file.write_text("location_id,date,pet\n1,2020-01-01,10\n")
        result = _discover_csv_inputs(csv_file)
        assert len(result) == 1
        assert result[0] == csv_file

    def test_shard_root_discovery(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "analytics" / "shard_count=00020" / "shard_index=00000"
        shard_dir.mkdir(parents=True)
        from typing import cast

        import pandas as pd

        pd.DataFrame(
            columns=cast("Any", ["year", "location_id", "p10", "p90"])
        ).to_parquet(shard_dir / "percentiles.parquet", index=False)

        result = _discover_csv_inputs(
            "percentiles.parquet",
            shard_root=str(tmp_path / "analytics"),
            shard_file_name="percentiles.parquet",
            shard_count=20,
            shard_partition_key="shard_count",
        )
        assert len(result) == 1

    def test_no_files_found(self, tmp_path: Path) -> None:
        result = _discover_csv_inputs(tmp_path / "nonexistent.parquet")
        assert result == []


class TestDiscoverPetCsvPaths:
    def test_prefers_direct_csv_when_requested(self, tmp_path: Path) -> None:
        pet_root = tmp_path / "pet_data_csv"
        shard_dir = pet_root / "year=2024"
        shard_dir.mkdir(parents=True)
        (shard_dir / "pet_batch_0000_00.parquet").touch()

        pet_csv = tmp_path / "pet_full.csv"
        pet_csv.write_text("location_id,date,pet\n1,2024-05-01,25.0\n")

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv=str(pet_csv),
            load_shard_index=0,
            load_shard_count=1,
            prefer_pet_csv=True,
        )

        assert _discover_pet_csv_paths(args) == [pet_csv]


class TestDiscoverWetbulbCsvPaths:
    def test_prefers_direct_csv_when_requested(self, tmp_path: Path) -> None:
        wetbulb_root = tmp_path / "wetbulb_data_csv"
        shard_dir = wetbulb_root / "year=2024"
        shard_dir.mkdir(parents=True)
        (shard_dir / "wetbulb_batch_0000_00.parquet").touch()

        wetbulb_csv = tmp_path / "wetbulb_full.csv"
        wetbulb_csv.write_text("location_id,date,wetbulb\n1,2024-05-01,18.0\n")

        args = argparse.Namespace(
            wetbulb_root=str(wetbulb_root),
            wetbulb_csv=str(wetbulb_csv),
            load_shard_index=0,
            load_shard_count=1,
            prefer_wetbulb_csv=True,
        )

        assert _discover_wetbulb_csv_paths(args) == [wetbulb_csv]

    def test_discovers_batch_parquet_shards(self, tmp_path: Path) -> None:
        wetbulb_root = tmp_path / "wetbulb_data_csv"
        shard_dir = wetbulb_root / "year=2024"
        shard_dir.mkdir(parents=True)
        batch_path = shard_dir / "wetbulb_batch_0000_00.parquet"
        batch_path.touch()

        args = argparse.Namespace(
            wetbulb_root=str(wetbulb_root),
            wetbulb_csv=str(tmp_path / "missing_wetbulb.csv"),
            load_shard_index=0,
            load_shard_count=1,
            prefer_wetbulb_csv=False,
        )

        assert _discover_wetbulb_csv_paths(args) == [batch_path]


class TestIterSqlStatements:
    def test_keeps_dollar_quoted_function_body_intact(self) -> None:
        sql_text = (
            "CREATE FUNCTION public.test_fn()\n"
            "RETURNS void\n"
            "LANGUAGE plpgsql\n"
            "AS $$\n"
            "BEGIN\n"
            "PERFORM 1;\n"
            "END\n"
            "$$ ;\n"
            "CREATE TABLE public.example (id integer);\n"
        )

        statements = list(_iter_sql_statements(sql_text))

        assert len(statements) == 2
        assert "PERFORM 1;" in statements[0]
        assert statements[1] == "CREATE TABLE public.example (id integer);"

    def test_ignores_semicolons_in_comments_and_strings(self) -> None:
        sql_text = "-- comment;\nSELECT 'a; b';\n/* block; */\nSELECT 2;"

        assert list(_iter_sql_statements(sql_text)) == [
            "-- comment;\nSELECT 'a; b';",
            "/* block; */\nSELECT 2;",
        ]


class TestExecuteSqlFile:
    def test_executes_each_statement_separately(self, tmp_path: Path) -> None:
        sql_path = tmp_path / "schema.sql"
        sql_path.write_text("SELECT 1;\nSELECT 2;\n", encoding="utf-8")
        conn = FakeConnection()

        execute_sql_file(cast("Any", conn), sql_path)

        assert conn.executed_statements == ["SELECT 1;", "SELECT 2;"]
        assert conn.transaction_count == 1


class TestRefreshQueryPlannerStatistics:
    def test_analyzes_core_runtime_tables(self) -> None:
        conn = FakeConnection()

        refresh_query_planner_statistics(cast("Any", conn))

        assert conn.executed_statements == [
            "ANALYZE public.locations",
            "ANALYZE public.pet",
            "ANALYZE public.wetbulb",
        ]


class TestMain:
    def test_main_runs_with_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import load

        # load.py confines file loads to the current working directory, so
        # run from tmp_path like the CLI is run from the repo checkout root.
        monkeypatch.chdir(tmp_path)

        # Mock dependencies to avoid actual DB and file ops
        monkeypatch.setattr(
            load,
            "_parse_args",
            lambda: argparse.Namespace(
                locations_csv=str(tmp_path / "loc.csv"),
                pet_csv=str(tmp_path / "pet.csv"),
                pet_root=str(tmp_path / "pet_root"),
                prefer_pet_csv=True,
                wetbulb_csv=str(tmp_path / "wetbulb.csv"),
                wetbulb_root=str(tmp_path / "wetbulb_root"),
                prefer_wetbulb_csv=False,
                load_shard_index=0,
                load_shard_count=1,
                copy_batch_size=10,
                load_workers=1,
                append_only=False,
                skip_drop_views=True,
                skip_create_views=True,
                ensure_schema=False,
                truncate_tables=None,
                skip_tables=["pet"],
            ),
        )

        monkeypatch.setattr(
            load, "resolve_database_uri", lambda: "postgresql://user:pass@host/db"
        )

        mock_conn = MagicMock()
        monkeypatch.setattr(load.psycopg, "connect", lambda _uri: mock_conn)

        # Mock table loading
        monkeypatch.setattr(load, "bulk_insert_csv_files", MagicMock())
        monkeypatch.setattr(load, "execute_sql_file", MagicMock())
        monkeypatch.setattr(load, "refresh_query_planner_statistics", MagicMock())

        # Create dummy locations file
        loc_file = tmp_path / "loc.csv"
        loc_file.write_text("id,city,state,lat,lng\n0,A,B,1,2\n")

        load.main()

        assert mock_conn.close.called


class TestDiscoverLocationsCsvPaths:
    def test_uses_locations_csv_argument(self, tmp_path: Path) -> None:
        locations_csv = tmp_path / "locations.csv"
        args = argparse.Namespace(locations_csv=str(locations_csv))

        assert _discover_locations_csv_paths(args) == [locations_csv]


class TestNormalizeCopyColumnNames:
    def test_locations_header_maps_location_id_to_id(self) -> None:
        assert _normalize_copy_column_names(
            "locations",
            ["location_id", "city", "state", "lat", "lng"],
        ) == ["id", "city", "state", "lat", "lng"]

    def test_non_locations_headers_remain_unchanged(self) -> None:
        assert _normalize_copy_column_names(
            "pet",
            ["location_id", "date", "pet"],
        ) == ["location_id", "date", "pet"]


class TestSelectPartitionShardPaths:
    def test_shards_partitioned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "pet_data"
        p1 = root / "year=2020" / "data.csv"
        p2 = root / "year=2021" / "data.csv"
        p1.parent.mkdir(parents=True)
        p2.parent.mkdir(parents=True)
        p1.touch()
        p2.touch()

        shard0 = _select_partition_shard_paths(
            [p1, p2], root_path=root, partition_key="year", shard_index=0, shard_count=2
        )
        assert shard0 == [p1]

        shard1 = _select_partition_shard_paths(
            [p1, p2], root_path=root, partition_key="year", shard_index=1, shard_count=2
        )
        assert shard1 == [p2]

    def test_handles_unpartitioned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "pet_data"
        root.mkdir()
        p1 = tmp_path / "pet.csv"
        p1.touch()

        shard0 = _select_partition_shard_paths(
            [p1], root_path=root, partition_key="year", shard_index=0, shard_count=2
        )
        assert shard0 == [p1]

        shard1 = _select_partition_shard_paths(
            [p1], root_path=root, partition_key="year", shard_index=1, shard_count=2
        )
        assert shard1 == []

    def test_distributes_multiple_unpartitioned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "pet_data"
        root.mkdir()
        p1 = tmp_path / "pet_1.csv"
        p2 = tmp_path / "pet_2.csv"
        p1.touch()
        p2.touch()

        shard0 = _select_partition_shard_paths(
            [p1, p2], root_path=root, partition_key="year", shard_index=0, shard_count=2
        )
        shard1 = _select_partition_shard_paths(
            [p1, p2], root_path=root, partition_key="year", shard_index=1, shard_count=2
        )

        assert len(shard0) == 1
        assert len(shard1) == 1
        assert shard0 != shard1


class TestTableNames:
    def test_expected_tables(self) -> None:
        assert "locations" in TABLE_NAMES
        assert "pet" in TABLE_NAMES
        assert "wetbulb" in TABLE_NAMES
        assert "pet_forecast" not in TABLE_NAMES
        assert "pet_percentiles" not in TABLE_NAMES
        assert "pet_change" not in TABLE_NAMES


class _CopyContext:
    def __init__(self, payloads: list[bytes]) -> None:
        self._payloads = payloads

    def write(self, data: bytes | str) -> None:
        self._payloads.append(data.encode() if isinstance(data, str) else bytes(data))

    def __enter__(self) -> _CopyContext:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = (exc_type, exc, tb)


class CopyRecordingCursor:
    def __init__(self, connection: CopyRecordingConnection) -> None:
        """Record statements against the parent connection."""
        self._connection = connection
        self.rowcount = connection.rowcount

    def execute(self, statement: object, params: object = None) -> None:
        _ = params
        rendered = (
            statement if isinstance(statement, str) else statement.as_string(None)  # type: ignore[union-attr]
        )
        self._connection.statements.append(rendered)

    def copy(self, statement: object) -> _CopyContext:
        rendered = (
            statement if isinstance(statement, str) else statement.as_string(None)  # type: ignore[union-attr]
        )
        self._connection.statements.append(rendered)
        return _CopyContext(self._connection.copy_payloads)

    def __enter__(self) -> CopyRecordingCursor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)


class CopyRecordingConnection:
    def __init__(self, rowcount: int = 1) -> None:
        """Record statements, COPY payloads, and transaction usage."""
        self.statements: list[str] = []
        self.copy_payloads: list[bytes] = []
        self.rowcount = rowcount
        self.transaction_count = 0

    def cursor(self) -> CopyRecordingCursor:
        return CopyRecordingCursor(self)

    @contextmanager
    def transaction(self) -> Generator[None]:
        self.transaction_count += 1
        yield


class TestFileCopyColumnNames:
    def test_reads_csv_header(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pet.csv"
        csv_path.write_text("location_id,date,pet\n1,2020-01-01,10\n")

        assert load._file_copy_column_names("pet", csv_path) == [
            "location_id",
            "date",
            "pet",
        ]

    def test_reads_parquet_schema_and_normalizes(self, tmp_path: Path) -> None:
        import pandas as pd

        parquet_path = tmp_path / "locations.parquet"
        pd.DataFrame({"location_id": [1], "city": ["A"]}).to_parquet(
            parquet_path, index=False
        )

        assert load._file_copy_column_names("locations", parquet_path) == [
            "id",
            "city",
        ]

    def test_empty_csv_returns_no_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        csv_path.touch()

        assert load._file_copy_column_names("pet", csv_path) == []


class TestBulkInsertStaging:
    def _write_parquet(self, path: Path) -> None:
        import pandas as pd

        pd.DataFrame(
            {
                "location_id": [1, 2],
                "date": ["2024-01-01", "2024-01-02"],
                "pet": [10.0, 11.0],
            }
        ).to_parquet(path, index=False)

    def test_stages_and_upserts_in_one_transaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        parquet_path = tmp_path / "pet_batch_0000_00.parquet"
        self._write_parquet(parquet_path)
        conn = CopyRecordingConnection(rowcount=2)

        load.bulk_insert_csv_files(
            cast("Any", conn),
            [parquet_path],
            "pet",
            batch_size=1000,
            truncate=False,
        )

        assert conn.transaction_count == 1
        create = next(s for s in conn.statements if s.startswith("CREATE TEMP TABLE"))
        assert '"pet_load_staging"' in create
        assert "WITH NO DATA" in create
        copy = next(s for s in conn.statements if s.startswith("COPY"))
        assert '"pet_load_staging"' in copy
        upsert = next(s for s in conn.statements if s.startswith("INSERT INTO"))
        assert 'ON CONFLICT ("location_id", "date")' in upsert
        assert 'DO UPDATE SET "pet" = EXCLUDED."pet"' in upsert
        assert 'SELECT DISTINCT ON ("location_id", "date")' in upsert
        assert b'1,"2024-01-01",10' in b"".join(conn.copy_payloads)

    def test_truncate_runs_inside_the_load_transaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        parquet_path = tmp_path / "pet_batch_0000_00.parquet"
        self._write_parquet(parquet_path)
        conn = CopyRecordingConnection(rowcount=2)

        load.bulk_insert_csv_files(
            cast("Any", conn),
            [parquet_path],
            "pet",
            batch_size=1000,
            truncate=True,
        )

        assert conn.statements[0] == 'TRUNCATE TABLE "pet" CASCADE'

    def test_no_paths_warns_and_skips(self, caplog: pytest.LogCaptureFixture) -> None:
        conn = CopyRecordingConnection()

        with caplog.at_level(logging.WARNING):
            load.bulk_insert_csv_files(
                cast("Any", conn), [], "pet", batch_size=10, truncate=False
            )

        assert conn.statements == []
        assert "No data inputs" in caplog.text


class TestUpsertFromStaging:
    def test_plain_insert_without_key_columns(self) -> None:
        conn = CopyRecordingConnection(rowcount=3)

        affected = load._upsert_from_staging(
            cast("Any", conn), "pet", "pet_load_staging", ["location_id"], ()
        )

        assert affected == 3
        assert "ON CONFLICT" not in conn.statements[0]

    def test_do_nothing_when_all_columns_are_keys(self) -> None:
        conn = CopyRecordingConnection(rowcount=0)

        load._upsert_from_staging(
            cast("Any", conn),
            "pet",
            "pet_load_staging",
            ["location_id", "date"],
            ("location_id", "date"),
        )

        assert "DO NOTHING" in conn.statements[0]


class TestLoadTableFiles:
    def test_truncate_stays_single_connection(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bulk = MagicMock()
        monkeypatch.setattr(load, "bulk_insert_csv_files", bulk)
        group_loader = MagicMock()
        monkeypatch.setattr(load, "_load_file_group", group_loader)
        paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]

        load._load_table_files(
            cast("Any", MagicMock()),
            "postgresql://x",
            paths,
            "pet",
            batch_size=10,
            truncate=True,
            workers=4,
        )

        assert bulk.called
        assert not group_loader.called

    def test_parallel_workers_split_files_round_robin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(load, "bulk_insert_csv_files", MagicMock())
        groups: list[list[Path]] = []

        def record_group(
            _db_uri: str, csv_paths: list[Path], _table: str, **_kwargs: object
        ) -> None:
            groups.append(csv_paths)

        monkeypatch.setattr(load, "_load_file_group", record_group)
        paths = [tmp_path / f"{i}.parquet" for i in range(5)]

        load._load_table_files(
            cast("Any", MagicMock()),
            "postgresql://x",
            paths,
            "pet",
            batch_size=10,
            truncate=False,
            workers=2,
        )

        assert sorted(p for group in groups for p in group) == sorted(paths)
        assert len(groups) == 2
