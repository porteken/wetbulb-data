"""Tests for historical PET export and merge helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import historical_pet_update as hpu

FetchoneValue = tuple[str | None] | None


DEFAULT_COPY_CHUNKS = (b"location_id,date,pet\n", b"1,2000-05-01,20.0\n")


class FakeCopy:
    def __init__(self, cursor: FakeCursor, chunks: tuple[bytes, ...]) -> None:
        """Initialize the fake COPY context manager."""
        self.cursor = cursor
        self.chunks = chunks

    def __enter__(self) -> FakeCopy:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)

    def __iter__(self) -> Iterator[memoryview]:
        """Yield exported CSV chunks the way psycopg streams COPY output."""
        return iter([memoryview(chunk) for chunk in self.chunks])


class FakeCursor:
    def __init__(
        self,
        fetchone_values: list[FetchoneValue] | None = None,
        copy_chunks: tuple[bytes, ...] = DEFAULT_COPY_CHUNKS,
    ) -> None:
        """Initialize the fake cursor."""
        self.copy_query = ""
        self.copy_params: tuple[str, str] | None = None
        self.execute_calls: list[tuple[object, object | None]] = []
        self._fetchone_values = list(fetchone_values or [("public.pet",)])
        self._copy_chunks = copy_chunks

    def execute(self, query: object, params: object | None = None) -> None:
        self.execute_calls.append((query, params))

    def fetchone(self) -> FetchoneValue:
        if not self._fetchone_values:
            return None
        return self._fetchone_values.pop(0)

    def copy(self, query: str, params: tuple[str, str] | None = None) -> FakeCopy:
        self.copy_query = query
        self.copy_params = params
        return FakeCopy(self, self._copy_chunks)

    def __enter__(self) -> FakeCursor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)


class FakeConnection:
    def __init__(
        self,
        fetchone_values: list[FetchoneValue] | None = None,
        copy_chunks: tuple[bytes, ...] = DEFAULT_COPY_CHUNKS,
    ) -> None:
        """Initialize fake connection."""
        self.cursor_instance = FakeCursor(fetchone_values, copy_chunks)
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


class RecordingCursor:
    def __init__(
        self,
        execute_calls: list[tuple[object, object | None]],
        fetchone_values: list[FetchoneValue] | None = None,
    ) -> None:
        """Initialize a cursor that records execute calls."""
        self.execute_calls = execute_calls
        self._fetchone_values = list(fetchone_values or [])

    def execute(self, query: object, params: object | None = None) -> None:
        self.execute_calls.append((query, params))

    def fetchone(self) -> FetchoneValue:
        if not self._fetchone_values:
            return None
        return self._fetchone_values.pop(0)

    def __enter__(self) -> RecordingCursor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)


class RecordingConnection:
    def __init__(self, cursor_fetchone_values: list[list[FetchoneValue]]) -> None:
        """Initialize connection with deterministic cursor responses."""
        self.execute_calls: list[tuple[object, object | None]] = []
        self.cursors = [
            RecordingCursor(self.execute_calls, fetchone_values)
            for fetchone_values in cursor_fetchone_values
        ]
        self._cursor_index = 0
        self.autocommit = False
        self.closed = False

    def cursor(self) -> RecordingCursor:
        cursor = self.cursors[self._cursor_index]
        self._cursor_index += 1
        return cursor

    def close(self) -> None:
        self.closed = True


def test_export_all_writes_pet_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_conn = FakeConnection()
    monkeypatch.setattr(hpu.psycopg, "connect", lambda _: fake_conn)
    monkeypatch.setenv("POSTGRES_DB_URI", "postgresql://example")

    output_path = tmp_path / "existing_pet.csv"
    hpu.export_pet(None, None, str(output_path))

    assert output_path.read_text() == "location_id,date,pet\n1,2000-05-01,20.0\n"
    assert "FROM public.pet" in fake_conn.cursor_instance.copy_query
    assert "ORDER BY location_id, date" in fake_conn.cursor_instance.copy_query
    assert fake_conn.cursor_instance.copy_params is None


def test_export_all_preserves_multibyte_characters_split_across_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A UTF-8 character straddling two COPY chunks must survive the write."""
    payload = "location_id,date,pet\n1,2000-05-01,20.0,Cañón\n".encode()
    split_at = payload.index(b"\xc3\xb1") + 1
    fake_conn = FakeConnection(copy_chunks=(payload[:split_at], payload[split_at:]))
    monkeypatch.setattr(hpu.psycopg, "connect", lambda _: fake_conn)
    monkeypatch.setenv("POSTGRES_DB_URI", "postgresql://example")

    output_path = tmp_path / "existing_pet.csv"
    hpu.export_pet(None, None, str(output_path))

    assert output_path.read_text(encoding="utf-8") == payload.decode("utf-8")


def test_merge_csvs_prefers_later_sources_for_duplicates(tmp_path: Path) -> None:
    existing_pet = tmp_path / "existing_pet.csv"
    existing_pet.write_text(
        "location_id,date,pet\n1,2024-05-01,20.0\n2,2023-05-01,21.0\n",
    )

    pet_root = tmp_path / "pet_data_csv" / "year=2024"
    pet_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"location_id": 1, "date": "2024-05-01", "pet": 25.0},
            {"location_id": 3, "date": "2024-05-02", "pet": 30.0},
        ],
    ).to_parquet(pet_root / "pet_batch_0000_00.parquet", index=False)

    output_path = tmp_path / "pet_full.csv"
    hpu.merge_csvs(
        [str(tmp_path / "pet_data_csv")], [str(existing_pet)], str(output_path)
    )

    merged = pd.read_csv(output_path)
    assert len(merged) == 3
    assert float(
        merged.loc[
            (merged["location_id"] == 1) & (merged["date"] == "2024-05-01"),
            "pet",
        ].iloc[0],
    ) == pytest.approx(25.0)


def test_build_export_copy_query_with_window_filters_out_target_range() -> None:
    query, params = hpu._build_export_copy_query(
        window_start="2024-01-01",
        window_end="2024-01-31",
    )

    assert "WHERE date < %s::date OR date > %s::date" in query
    assert params == ("2024-01-01", "2024-01-31")


def _clear_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all postgres database connection env vars."""
    for var in (
        "POSTGRES_DB_URI",
        "DATABASE_URL",
        "PGHOST",
        "PGPORT",
        "PGDATABASE",
        "PGUSER",
        "PGPASSWORD",
        "PGSSLMODE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_export_pet_without_database_uri_writes_empty_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_db_env(monkeypatch)

    output_path = tmp_path / "existing_pet.csv"
    hpu.export_pet(None, None, str(output_path))

    assert output_path.read_text(encoding="utf-8") == hpu.PET_CSV_HEADER
    assert "Postgres database credentials are not configured" in capsys.readouterr().out


def test_export_pet_writes_empty_csv_when_pet_table_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_conn = FakeConnection(fetchone_values=[(None,)])
    monkeypatch.setattr(hpu.psycopg, "connect", lambda _: fake_conn)
    monkeypatch.setenv("POSTGRES_DB_URI", "postgresql://example")

    output_path = tmp_path / "existing_pet.csv"
    hpu.export_pet(None, None, str(output_path))

    assert output_path.read_text(encoding="utf-8") == hpu.PET_CSV_HEADER
    assert fake_conn.cursor_instance.copy_query == ""
    assert "Table 'pet' does not exist" in capsys.readouterr().out


def test_merge_csvs_writes_empty_csv_when_sources_are_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "pet_full.csv"

    hpu.merge_csvs([], [str(tmp_path / "missing.csv")], str(output_path))

    assert output_path.read_text(encoding="utf-8") == hpu.PET_CSV_HEADER
    captured = capsys.readouterr().out
    assert "Merging PET data into" in captured
    assert "No source data found; wrote empty CSV." in captured


def test_delete_window_without_database_uri_skips_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_db_env(monkeypatch)

    hpu.delete_window("2024-01-01", "2024-01-31")

    assert "Postgres database credentials are not configured" in capsys.readouterr().out


def test_delete_window_deletes_rows_and_truncates_analytics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    conn = RecordingConnection([[("public.pet",)], []])
    monkeypatch.setattr(hpu.psycopg, "connect", lambda _: conn)
    monkeypatch.setattr(hpu, "_existing_public_tables", lambda *_args: ["pet_forecast"])
    monkeypatch.setenv("POSTGRES_DB_URI", "postgresql://example")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "drop_views.sql").write_text(
        "DROP VIEW IF EXISTS pet_year;", encoding="utf-8"
    )
    (tmp_path / "create_tables.sql").write_text(
        "CREATE TABLE IF NOT EXISTS pet(id int);", encoding="utf-8"
    )

    hpu.delete_window("2024-01-01", "2024-01-31")

    assert conn.autocommit is True
    assert conn.closed is True
    # Views must be left untouched: the DDL files exist but are not executed.
    assert not any(
        call[0] == "DROP VIEW IF EXISTS pet_year;" for call in conn.execute_calls
    )
    assert not any(
        call[0] == "CREATE TABLE IF NOT EXISTS pet(id int);"
        for call in conn.execute_calls
    )
    assert any(
        call[0] == "DELETE FROM public.pet WHERE date BETWEEN %s::date AND %s::date"
        and call[1] == ("2024-01-01", "2024-01-31")
        for call in conn.execute_calls
    )
    assert any(call[0].__class__.__name__ == "Composed" for call in conn.execute_calls)
    captured = capsys.readouterr().out
    assert "Deleting PET data in window [2024-01-01, 2024-01-31]..." in captured
    assert "Truncating analytics tables..." in captured


def test_delete_window_deletes_both_pet_and_wetbulb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    conn = RecordingConnection(
        [[("public.pet",), ("public.wetbulb",)], []],
    )
    monkeypatch.setattr(hpu.psycopg, "connect", lambda _: conn)
    monkeypatch.setattr(hpu, "_existing_public_tables", lambda *_args: [])
    monkeypatch.setenv("POSTGRES_DB_URI", "postgresql://example")
    monkeypatch.chdir(tmp_path)

    hpu.delete_window("2024-01-01", "2024-01-31")

    assert any(
        call[0] == "DELETE FROM public.pet WHERE date BETWEEN %s::date AND %s::date"
        and call[1] == ("2024-01-01", "2024-01-31")
        for call in conn.execute_calls
    )
    assert any(
        call[0] == "DELETE FROM public.wetbulb WHERE date BETWEEN %s::date AND %s::date"
        and call[1] == ("2024-01-01", "2024-01-31")
        for call in conn.execute_calls
    )
    captured = capsys.readouterr().out
    assert "Deleting PET data in window [2024-01-01, 2024-01-31]..." in captured
    assert "Deleting wetbulb data in window [2024-01-01, 2024-01-31]..." in captured


def test_export_pet_with_table_wetbulb_queries_wetbulb_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_conn = FakeConnection()
    monkeypatch.setattr(hpu.psycopg, "connect", lambda _: fake_conn)
    monkeypatch.setenv("POSTGRES_DB_URI", "postgresql://example")

    output_path = tmp_path / "existing_wetbulb.csv"
    hpu.export_pet(None, None, str(output_path), product="wetbulb")

    assert "FROM public.wetbulb" in fake_conn.cursor_instance.copy_query
    assert "SELECT location_id, date, wetbulb" in fake_conn.cursor_instance.copy_query


@pytest.mark.parametrize(
    ("argv", "expected_call"),
    [
        (
            [
                "historical_pet_update.py",
                "export",
                "2024-01-01",
                "2024-01-31",
                "out.csv",
            ],
            ("export", ("2024-01-01", "2024-01-31", "out.csv")),
        ),
        (
            ["historical_pet_update.py", "export-all", "out.csv"],
            ("export", (None, None, "out.csv")),
        ),
        (
            [
                "historical_pet_update.py",
                "merge",
                "pet_full.csv",
                "existing.csv",
                "incoming.csv",
                "--dirs",
                "pet_data_csv",
                "more_data",
            ],
            (
                "merge",
                (
                    ["pet_data_csv", "more_data"],
                    ["existing.csv", "incoming.csv"],
                    "pet_full.csv",
                ),
            ),
        ),
        (
            ["historical_pet_update.py", "delete-window", "2024-01-01", "2024-01-31"],
            ("delete", ("2024-01-01", "2024-01-31")),
        ),
    ],
)
def test_main_dispatches_supported_commands(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_call: tuple[str, tuple[Any, ...]],
) -> None:
    recorded_calls: list[tuple[str, tuple[Any, ...]]] = []

    monkeypatch.setattr(
        hpu,
        "export_pet",
        lambda *args, **_kwargs: recorded_calls.append(("export", args)),
    )
    monkeypatch.setattr(
        hpu,
        "merge_csvs",
        lambda *args, **_kwargs: recorded_calls.append(("merge", args)),
    )
    monkeypatch.setattr(
        hpu,
        "delete_window",
        lambda *args, **_kwargs: recorded_calls.append(("delete", args)),
    )
    monkeypatch.setattr(hpu.sys, "argv", argv)

    hpu.main()

    assert recorded_calls == [expected_call]


def test_main_rejects_unknown_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hpu.sys, "argv", ["historical_pet_update.py", "mystery"])

    with pytest.raises(SystemExit, match="Unknown command: mystery"):
        hpu.main()


def test_main_dispatches_export_all_with_table_wetbulb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    monkeypatch.setattr(
        hpu,
        "export_pet",
        lambda *args, **kwargs: recorded_calls.append(("export", args, kwargs)),
    )
    monkeypatch.setattr(
        hpu.sys,
        "argv",
        [
            "historical_pet_update.py",
            "export-all",
            "wetbulb_full.csv",
            "--table",
            "wetbulb",
        ],
    )

    hpu.main()

    assert recorded_calls == [
        ("export", (None, None, "wetbulb_full.csv"), {"product": "wetbulb"})
    ]


def test_main_rejects_unknown_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hpu.sys,
        "argv",
        ["historical_pet_update.py", "export-all", "out.csv", "--table", "bogus"],
    )

    with pytest.raises(SystemExit, match="Unknown --table value: bogus"):
        hpu.main()
