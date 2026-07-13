"""Utility for PET data export, merging, and incremental updates."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString, cast

import psycopg
from psycopg import sql

from shared_config import DATABASE_CONFIG_HINT, resolve_database_uri

if TYPE_CHECKING:
    from psycopg import Connection

_TO_REGCLASS_QUERY: LiteralString = "SELECT to_regclass(%s)"

DATA_PRODUCTS: dict[str, dict[str, str]] = {
    "pet": {
        "table": "pet",
        "column": "pet",
        "batch_glob": "pet_batch_*.parquet",
        "csv_name": "pet.csv",
        "csv_header": "location_id,date,pet\n",
        "label": "PET",
    },
    "wetbulb": {
        "table": "wetbulb",
        "column": "wetbulb",
        "batch_glob": "wetbulb_batch_*.parquet",
        "csv_name": "wetbulb.csv",
        "csv_header": "location_id,date,wetbulb\n",
        "label": "wetbulb",
    },
}
WINDOWED_PRODUCTS: tuple[str, ...] = ("pet", "wetbulb")
PET_CSV_HEADER = DATA_PRODUCTS["pet"]["csv_header"]
ANALYTICS_TABLES: tuple[str, ...] = ()


def _write_empty_csv(output_path: str | Path, product: str = "pet") -> None:
    header = DATA_PRODUCTS[product]["csv_header"]
    Path(output_path).write_text(header, encoding="utf-8")  # NOSONAR


def _build_export_copy_query(
    *,
    window_start: str | None,
    window_end: str | None,
    product: str = "pet",
) -> tuple[LiteralString, tuple[str, str] | None]:
    table = DATA_PRODUCTS[product]["table"]
    column = DATA_PRODUCTS[product]["column"]
    if window_start is None or window_end is None:
        return cast(
            "LiteralString",
            "COPY ("
            f"SELECT location_id, date, {column} "
            f"FROM public.{table} "
            "ORDER BY location_id, date"
            ") TO STDOUT WITH CSV HEADER",
        ), None

    return cast(
        "LiteralString",
        "COPY ("
        f"SELECT location_id, date, {column} "
        f"FROM public.{table} "
        "WHERE date < %s::date OR date > %s::date "
        "ORDER BY location_id, date"
        ") TO STDOUT WITH CSV HEADER",
    ), (window_start, window_end)


def export_pet(
    window_start: str | None,
    window_end: str | None,
    output_path: str,
    *,
    product: str = "pet",
) -> None:
    """Export PET/wetbulb data, optionally excluding a target date window."""
    table = DATA_PRODUCTS[product]["table"]
    db_uri = resolve_database_uri()
    if not db_uri:
        print(
            "Postgres database credentials are not configured. "
            f"{DATABASE_CONFIG_HINT} Skipping database export.",
        )
        _write_empty_csv(output_path, product)
        return

    conn = psycopg.connect(db_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(_TO_REGCLASS_QUERY, (f"public.{table}",))
            regclass = cur.fetchone()
            if regclass is None or regclass[0] is None:  # type: ignore[index]
                print(f"Table '{table}' does not exist. Creating empty export.")
                _write_empty_csv(output_path, product)
                return

            copy_query, copy_params = _build_export_copy_query(
                window_start=window_start,
                window_end=window_end,
                product=product,
            )

            # psycopg streams COPY output as memoryview chunks, which may split a
            # multi-byte character, so write bytes straight through without decoding.
            with Path(output_path).open("wb") as csv_file:  # NOSONAR
                copy_context = (
                    cur.copy(copy_query)
                    if copy_params is None
                    else cur.copy(copy_query, copy_params)
                )
                with copy_context as copy:
                    csv_file.writelines(cast("Any", copy))
    finally:
        conn.close()


def _iter_source_paths(
    source_dirs: list[str],
    source_files: list[str],
    *,
    product: str = "pet",
) -> list[Path]:
    source_paths = [Path(path_str) for path_str in source_files]
    batch_glob = DATA_PRODUCTS[product]["batch_glob"]
    csv_name = DATA_PRODUCTS[product]["csv_name"]

    for directory in source_dirs:
        directory_path = Path(directory)
        parquet_paths = sorted(directory_path.rglob(batch_glob))
        if parquet_paths:
            source_paths.extend(parquet_paths)
            continue

        source_paths.extend(sorted(directory_path.rglob(csv_name)))

    return source_paths


def _load_data_frame_from_path(
    source_path: Path,
    pd: Any,
    *,
    product: str = "pet",
) -> Any:
    columns = ["location_id", "date", DATA_PRODUCTS[product]["column"]]
    if source_path.suffix == ".parquet":
        return pd.read_parquet(source_path, columns=columns)
    return pd.read_csv(source_path, usecols=columns)


def merge_csvs(
    source_dirs: list[str],
    source_files: list[str],
    output_file: str,
    *,
    product: str = "pet",
) -> None:
    """Merge multiple PET/wetbulb CSV/parquet sources into one deduplicated CSV."""
    pd = importlib.import_module("pandas")
    column = DATA_PRODUCTS[product]["column"]
    label = DATA_PRODUCTS[product]["label"]

    print(f"Merging {label} data into {output_file}...")
    frames = []

    for source_order, source_path in enumerate(
        _iter_source_paths(source_dirs, source_files, product=product),
    ):
        if not source_path.exists():
            continue

        try:
            frame = _load_data_frame_from_path(source_path, pd, product=product)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {source_path}: {exc}")
            continue

        frame = frame[["location_id", "date", column]].copy()
        frame["_source_order"] = source_order
        frames.append(frame)

    if not frames:
        _write_empty_csv(output_file, product)
        print("No source data found; wrote empty CSV.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["location_id", "date", column])
    combined = combined.sort_values(
        ["location_id", "date", "_source_order"],
        kind="stable",
    )
    combined = combined.drop_duplicates(
        subset=["location_id", "date"],
        keep="last",
    )
    combined = combined.sort_values(["location_id", "date"], kind="stable")
    combined = combined.drop(columns=["_source_order"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(output_file, index=False)
    print(f"Wrote {len(combined)} rows to {output_file}.")


def _existing_public_tables(
    conn: Connection[Any],
    table_names: tuple[str, ...],
) -> list[str]:
    existing_tables: list[str] = []
    with conn.cursor() as cur:
        for table_name in table_names:
            cur.execute(_TO_REGCLASS_QUERY, (f"public.{table_name}",))
            row = cur.fetchone()
            if row is not None and row[0] is not None:  # type: ignore[index]
                existing_tables.append(table_name)
    return existing_tables


def delete_window(window_start: str, window_end: str) -> None:
    """Delete PET/wetbulb rows within a specific date window and reset analytics tables.

    Views are left untouched: materialized views do not block row deletion and
    are refreshed (or recreated) by the load pipeline afterwards.
    """
    db_uri = resolve_database_uri()
    if not db_uri:
        print(
            "Postgres database credentials are not configured. "
            f"{DATABASE_CONFIG_HINT} Skipping database cleanup.",
        )
        return

    conn = psycopg.connect(db_uri)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for product in WINDOWED_PRODUCTS:
                table = DATA_PRODUCTS[product]["table"]
                label = DATA_PRODUCTS[product]["label"]
                cur.execute(_TO_REGCLASS_QUERY, (f"public.{table}",))
                regclass = cur.fetchone()
                if regclass is None or regclass[0] is None:  # type: ignore[index]
                    continue
                print(
                    f"Deleting {label} data in window [{window_start}, {window_end}]..."
                )
                cur.execute(
                    cast(
                        "LiteralString",
                        f"DELETE FROM public.{table} "
                        "WHERE date BETWEEN %s::date AND %s::date",
                    ),
                    (window_start, window_end),
                )

        analytics_tables = _existing_public_tables(conn, ANALYTICS_TABLES)
        if analytics_tables:
            print("Truncating analytics tables...")
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("TRUNCATE TABLE {}").format(
                        sql.SQL(", ").join(
                            sql.Identifier("public", table_name)
                            for table_name in analytics_tables
                        ),
                    ),
                )
    finally:
        conn.close()


def _extract_table_option(argv: list[str]) -> tuple[list[str], str]:
    """Strip an optional --table <name> pair from argv, returning (remaining, product)."""
    remaining: list[str] = []
    product = "pet"
    index = 0
    while index < len(argv):
        if argv[index] == "--table" and index + 1 < len(argv):
            product = argv[index + 1]
            index += 2
            continue
        remaining.append(argv[index])
        index += 1

    if product not in DATA_PRODUCTS:
        msg = f"Unknown --table value: {product}"
        raise SystemExit(msg)

    return remaining, product


def main() -> None:
    """Execute historical PET/wetbulb update commands."""
    argv, product = _extract_table_option(sys.argv)

    min_args = 2
    if len(argv) < min_args:
        msg = (
            "Usage: historical_pet_update.py "
            "[export|export-all|merge|delete-window] ... [--table pet|wetbulb]"
        )
        raise SystemExit(msg)

    command = argv[1]
    if command == "export":
        export_pet(argv[2], argv[3], argv[4], product=product)
    elif command == "export-all":
        export_pet(None, None, argv[2], product=product)
    elif command == "merge":
        output_file = argv[2]
        source_files = []
        source_dirs = []
        current_list = source_files
        for arg in argv[3:]:
            if arg == "--dirs":
                current_list = source_dirs
            else:
                current_list.append(arg)
        merge_csvs(source_dirs, source_files, output_file, product=product)
    elif command == "delete-window":
        delete_window(argv[2], argv[3])
    else:
        msg = f"Unknown command: {command}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
