"""Load parquet data into the database and manage views."""

from __future__ import annotations

import argparse
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString, cast

import psycopg
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from psycopg import sql

from shared_config import DATABASE_CONFIG_HINT, resolve_database_uri

if TYPE_CHECKING:
    from collections.abc import Iterator

    from psycopg import Connection

LOGGER = logging.getLogger(__name__)
COPY_BATCH_SIZE = 50_000
CSV_COPY_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_LOAD_WORKERS = 4
DOLLAR_QUOTE_RE = re.compile(r"\$(?:[A-Za-z_]\w*)?\$")
TABLE_NAMES = [
    "locations",
    "pet",
    "wetbulb",
]
# Conflict targets that make loads idempotent: rows are COPYed into a temp
# staging table and upserted, so re-running a load (or retrying a failed one)
# can never produce duplicates.
TABLE_UNIQUE_KEYS: dict[str, tuple[str, ...]] = {
    "locations": ("id",),
    "pet": ("location_id", "date"),
    "wetbulb": ("location_id", "date"),
}


def _consume_line_comment(sql_text: str, index: int) -> int:
    """Skip to the first character after a line comment."""
    newline_index = sql_text.find("\n", index)
    if newline_index == -1:
        return len(sql_text)
    return newline_index + 1


def _consume_block_comment(sql_text: str, index: int) -> int:
    """Skip over a possibly nested block comment."""
    block_comment_depth = 1

    while index < len(sql_text) and block_comment_depth > 0:
        next_two = sql_text[index : index + 2]
        if next_two == "/*":
            block_comment_depth += 1
            index += 2
            continue
        if next_two == "*/":
            block_comment_depth -= 1
            index += 2
            continue
        index += 1

    return index


def _consume_quoted_string(sql_text: str, index: int, quote_char: str) -> int:
    """Skip over a single- or double-quoted string."""
    index += 1

    while index < len(sql_text):
        if sql_text[index] != quote_char:
            index += 1
            continue
        if index + 1 < len(sql_text) and sql_text[index + 1] == quote_char:
            index += 2
            continue
        return index + 1

    return index


def _match_dollar_quote(sql_text: str, index: int) -> str | None:
    """Return the opening dollar-quote tag at the current position, if any."""
    match = DOLLAR_QUOTE_RE.match(sql_text, index)
    if match is None:
        return None
    return match.group(0)


def _consume_dollar_quote(sql_text: str, index: int, tag: str) -> int:
    """Skip over a dollar-quoted PostgreSQL string body."""
    closing_index = sql_text.find(tag, index + len(tag))
    if closing_index == -1:
        return len(sql_text)
    return closing_index + len(tag)


def _consume_special_region(sql_text: str, index: int) -> int | None:
    """Consume any non-statement-delimiting SQL region at the current index."""
    next_two = sql_text[index : index + 2]
    if next_two == "--":
        return _consume_line_comment(sql_text, index + 2)
    if next_two == "/*":
        return _consume_block_comment(sql_text, index + 2)

    current_char = sql_text[index]
    if current_char in {"'", '"'}:
        return _consume_quoted_string(sql_text, index, current_char)
    if current_char != "$":
        return None

    dollar_quote = _match_dollar_quote(sql_text, index)
    if dollar_quote is None:
        return None
    return _consume_dollar_quote(sql_text, index, dollar_quote)


def _iter_sql_statements(sql_text: str) -> Iterator[str]:
    """Yield SQL statements while preserving dollar-quoted function bodies."""
    start_idx = 0
    index = 0

    while index < len(sql_text):
        special_region_end = _consume_special_region(sql_text, index)
        if special_region_end is not None:
            index = special_region_end
            continue

        current_char = sql_text[index]
        if current_char == ";":
            statement = sql_text[start_idx:index].strip()
            if statement:
                yield f"{statement};"
            start_idx = index + 1

        index += 1

    statement = sql_text[start_idx:].strip()
    if statement:
        yield statement


def execute_sql_file(conn: Connection[Any], file_path: str | Path) -> None:
    """Execute a raw SQL file atomically in a single transaction.

    Running inside one transaction keeps SET LOCAL timeouts in the SQL files
    scoped to the file itself instead of leaking into later COPY/DELETE
    statements on the same connection.
    """
    path = Path(file_path)
    if not path.exists():
        LOGGER.warning("SQL file %s not found. Skipping.", file_path)
        return

    LOGGER.info("Executing SQL file: %s...", file_path)
    statements = list(_iter_sql_statements(path.read_text(encoding="utf-8")))
    if not statements:
        LOGGER.info("SQL file %s is empty. Skipping.", file_path)
        return

    with conn.transaction(), conn.cursor() as cur:
        for statement in statements:
            cur.execute(cast("LiteralString", statement))
    LOGGER.info("Successfully executed %s (%d statements).", file_path, len(statements))


def refresh_query_planner_statistics(conn: Connection[Any]) -> None:
    """Refresh planner statistics for the core runtime tables."""
    LOGGER.info("Refreshing query planner statistics...")
    with conn.cursor() as cur:
        for table_name in TABLE_NAMES:
            cur.execute(cast("LiteralString", f"ANALYZE public.{table_name}"))
    LOGGER.info("Finished refreshing query planner statistics.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load shard-aware PET CSV outputs into the database and recreate views."
        ),
    )
    parser.add_argument(
        "--locations-csv",
        "--cities-csv",
        dest="locations_csv",
        default="locations.csv",
    )
    parser.add_argument("--pet-csv", default="pet.csv")
    parser.add_argument("--pet-root", default="pet_data_csv")
    parser.add_argument(
        "--prefer-pet-csv",
        action="store_true",
        help=(
            "Load the explicit --pet-csv input even when PET parquet shards are "
            "also present under --pet-root."
        ),
    )
    parser.add_argument("--wetbulb-csv", default="wetbulb.csv")
    parser.add_argument("--wetbulb-root", default="wetbulb_data_csv")
    parser.add_argument(
        "--prefer-wetbulb-csv",
        action="store_true",
        help=(
            "Load the explicit --wetbulb-csv input even when wetbulb parquet "
            "shards are also present under --wetbulb-root."
        ),
    )
    parser.add_argument("--load-shard-index", type=int, default=0)
    parser.add_argument("--load-shard-count", type=int, default=1)
    parser.add_argument("--copy-batch-size", type=int, default=COPY_BATCH_SIZE)
    parser.add_argument(
        "--load-workers",
        type=int,
        default=DEFAULT_LOAD_WORKERS,
        help=(
            "Parallel database connections used to load files into a table "
            "(append/upsert loads only; truncating loads stay serial)."
        ),
    )
    parser.add_argument(
        "--append-only",
        action="store_true",
        help="Append rows without truncating destination tables first.",
    )
    parser.add_argument(
        "--skip-drop-views",
        action="store_true",
        help="Leave existing views in place before loading data.",
    )
    parser.add_argument(
        "--skip-create-views",
        action="store_true",
        help="Do not recreate SQL views after loading data.",
    )
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        help=(
            "Execute create_tables.sql before loading even when views are "
            "skipped, so appends run against a migrated schema."
        ),
    )
    parser.add_argument(
        "--truncate-table",
        dest="truncate_tables",
        action="append",
        choices=TABLE_NAMES,
        help=(
            "Table to truncate before loading. Defaults to truncating all tables "
            "to preserve full rebuild behavior."
        ),
    )
    parser.add_argument(
        "--skip-table",
        dest="skip_tables",
        action="append",
        choices=TABLE_NAMES,
        help="Table to leave untouched during this run.",
    )
    return parser.parse_args(argv)


def _discover_csv_inputs(
    direct_path: str | Path,
    *,
    shard_root: str | Path | None = None,
    shard_file_name: str | None = None,
    shard_count: int | None = None,
    shard_partition_key: str | None = None,
) -> list[Path]:
    shard_paths = _discover_shard_csv_inputs(
        shard_root=shard_root,
        shard_file_name=shard_file_name,
        shard_count=shard_count,
        shard_partition_key=shard_partition_key,
    )
    if shard_paths:
        return shard_paths

    direct_csv_path = Path(direct_path)
    if direct_csv_path.exists():
        return [direct_csv_path]

    return []


def _discover_shard_csv_inputs(
    *,
    shard_root: str | Path | None,
    shard_file_name: str | None,
    shard_count: int | None,
    shard_partition_key: str | None,
) -> list[Path]:
    if shard_root is None or shard_file_name is None:
        return []

    root_path = Path(shard_root)
    if not root_path.exists():
        return []

    if shard_partition_key is None:
        return sorted(root_path.rglob(shard_file_name))

    if shard_count is not None:
        selected_root = root_path / f"shard_count={shard_count:05d}"
        return sorted(selected_root.rglob(shard_file_name))

    shard_groups: dict[str, list[Path]] = {}
    for csv_path in sorted(root_path.rglob(shard_file_name)):
        shard_group = _extract_partition_marker(
            csv_path,
            root_path,
            shard_partition_key,
        )
        if shard_group is None:
            continue
        shard_groups.setdefault(shard_group, []).append(csv_path)

    if not shard_groups:
        return []
    if len(shard_groups) > 1:
        group_labels = ", ".join(sorted(shard_groups))
        msg = (
            f"Found multiple analytics shard groups for {shard_file_name}: "
            f"{group_labels}. Pass --analytics-shard-count to select one."
        )
        raise RuntimeError(msg)

    return next(iter(shard_groups.values()))


def _extract_partition_marker(
    csv_path: Path,
    root_path: Path,
    partition_key: str,
) -> str | None:
    try:
        relative_parts = csv_path.relative_to(root_path).parts
    except ValueError:
        return None

    prefix = f"{partition_key}="
    for part in relative_parts:
        if part.startswith(prefix):
            return part.removeprefix(prefix)
    return None


def _validate_load_shard_args(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        msg = "load_shard_count must be >= 1"
        raise ValueError(msg)
    if shard_index < 0 or shard_index >= shard_count:
        msg = f"load_shard_index must be between 0 and {shard_count - 1}"
        raise ValueError(msg)


def _partition_sort_key(partition_value: str) -> tuple[int, object]:
    try:
        return 0, int(partition_value)
    except ValueError:
        return 1, partition_value


def _select_partition_shard_paths(
    csv_paths: list[Path],
    *,
    root_path: Path,
    partition_key: str,
    shard_index: int,
    shard_count: int,
) -> list[Path]:
    _validate_load_shard_args(shard_index, shard_count)
    if shard_count == 1:
        return csv_paths

    grouped_paths: dict[str, list[Path]] = {}
    for csv_path in csv_paths:
        partition_value = _extract_partition_marker(csv_path, root_path, partition_key)
        if partition_value is None:
            partition_value = f"__unpartitioned__:{csv_path.name}"
        grouped_paths.setdefault(partition_value, []).append(csv_path)

    selected_partition_values = {
        partition_value
        for position, partition_value in enumerate(
            sorted(grouped_paths, key=_partition_sort_key),
        )
        if position % shard_count == shard_index
    }
    return [
        csv_path
        for partition_value in sorted(grouped_paths, key=_partition_sort_key)
        if partition_value in selected_partition_values
        for csv_path in grouped_paths[partition_value]
    ]


def _file_copy_column_names(table_name: str, file_path: Path) -> list[str]:
    """Return the destination column names for a parquet or CSV input file."""
    if file_path.suffix == ".parquet":
        raw_names = pq.ParquetFile(str(file_path)).schema_arrow.names
    else:
        with file_path.open("r", encoding="utf-8", newline="") as csv_file:
            header = next(csv_file, None)
        if header is None:
            return []
        raw_names = [col.strip() for col in header.split(",")]
    return _normalize_copy_column_names(table_name, raw_names)


def _copy_parquet_file_in_batches(
    conn: Connection[Any],
    table_name: str,
    parquet_path: Path,
    *,
    batch_size: int,
    destination: str | None = None,
) -> int:
    """Stream a parquet file into a table with a single COPY statement.

    Record batches are serialized with pyarrow's native CSV writer, avoiding
    the parquet -> pandas -> to_csv round trip. `destination` lets callers
    COPY into a staging table while normalizing columns for `table_name`.
    """
    parquet_file = pq.ParquetFile(str(parquet_path))
    column_names = _normalize_copy_column_names(
        table_name,
        parquet_file.schema_arrow.names,
    )
    copy_statement = sql.SQL("COPY {} ({}) FROM STDIN WITH CSV").format(
        sql.Identifier(destination or table_name),
        sql.SQL(", ").join(sql.Identifier(col) for col in column_names),
    )

    total_rows = 0
    write_options = pacsv.WriteOptions(include_header=False)
    with conn.cursor() as cur, cur.copy(copy_statement) as copy:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            sink = io.BytesIO()
            pacsv.write_csv(batch, sink, write_options=write_options)
            copy.write(sink.getvalue())
            total_rows += batch.num_rows
    return total_rows


def _copy_csv_file_in_batches(
    conn: Connection[Any],
    table_name: str,
    csv_path: Path,
    *,
    batch_size: int,
    destination: str | None = None,
) -> int:
    """Stream a plain CSV file into a table with a single COPY statement."""
    _ = batch_size  # kept for signature parity with the parquet copy helper
    with conn.cursor() as cur, csv_path.open("r", encoding="utf-8", newline="") as f:
        header = next(f, None)
        if header is None:
            LOGGER.warning("CSV file %s is empty. Skipping.", csv_path)
            return 0
        column_names = _normalize_copy_column_names(
            table_name,
            [col.strip() for col in header.split(",")],
        )
        copy_statement = sql.SQL("COPY {} ({}) FROM STDIN WITH CSV").format(
            sql.Identifier(destination or table_name),
            sql.SQL(", ").join(sql.Identifier(col) for col in column_names),
        )
        with cur.copy(copy_statement) as copy:
            while chunk := f.read(CSV_COPY_CHUNK_BYTES):
                copy.write(chunk)
        return max(cur.rowcount, 0)


def _validated_load_path(path: Path, *, base_dir: Path | None = None) -> Path:
    """Resolve `path` and reject it if it escapes `base_dir` (defaults to CWD).

    CLI arguments may be supplied by an agent acting on untrusted input, so
    file paths derived from them must not be allowed to traverse outside the
    directory the tool was invoked from before being opened.
    """
    base = (base_dir or Path.cwd()).resolve()
    resolved = path.resolve()
    if resolved != base and base not in resolved.parents:
        msg = f"Path {path} resolves outside the allowed directory {base}"
        raise ValueError(msg)
    return resolved


def _create_staging_table(
    conn: Connection[Any],
    table_name: str,
    column_names: list[str],
) -> str:
    """Create a temp staging table mirroring the columns being loaded."""
    staging_name = f"{table_name}_load_staging"
    create_statement = sql.SQL(
        "CREATE TEMP TABLE {} ON COMMIT DROP AS SELECT {} FROM {} WITH NO DATA",
    ).format(
        sql.Identifier(staging_name),
        sql.SQL(", ").join(sql.Identifier(col) for col in column_names),
        sql.Identifier("public", table_name),
    )
    with conn.cursor() as cur:
        cur.execute(create_statement)
    return staging_name


def _upsert_from_staging(
    conn: Connection[Any],
    table_name: str,
    staging_name: str,
    column_names: list[str],
    key_columns: tuple[str, ...],
) -> int:
    """Upsert staged rows into the destination table; return affected rows."""
    columns_sql = sql.SQL(", ").join(sql.Identifier(col) for col in column_names)
    if not key_columns:
        insert_statement = sql.SQL(
            "INSERT INTO {table} ({columns}) SELECT {columns} FROM {staging}",
        ).format(
            table=sql.Identifier("public", table_name),
            columns=columns_sql,
            staging=sql.Identifier(staging_name),
        )
        with conn.cursor() as cur:
            cur.execute(insert_statement)
            return max(cur.rowcount, 0)

    keys_sql = sql.SQL(", ").join(sql.Identifier(col) for col in key_columns)
    update_columns = [col for col in column_names if col not in key_columns]
    if update_columns:
        conflict_action = sql.SQL("DO UPDATE SET {}").format(
            sql.SQL(", ").join(
                sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
                for col in update_columns
            ),
        )
    else:
        conflict_action = sql.SQL("DO NOTHING")

    # DISTINCT ON guards against duplicate keys inside a single load batch,
    # which would otherwise abort the INSERT ("cannot affect row a second
    # time"). Ordering by the key keeps b-tree insertion mostly sequential.
    upsert_statement = sql.SQL(
        "INSERT INTO {table} ({columns}) "
        "SELECT DISTINCT ON ({keys}) {columns} FROM {staging} ORDER BY {keys} "
        "ON CONFLICT ({keys}) {conflict_action}",
    ).format(
        table=sql.Identifier("public", table_name),
        columns=columns_sql,
        keys=keys_sql,
        staging=sql.Identifier(staging_name),
        conflict_action=conflict_action,
    )
    with conn.cursor() as cur:
        cur.execute(upsert_statement)
        return max(cur.rowcount, 0)


def _copy_file(
    conn: Connection[Any],
    table_name: str,
    file_path: Path,
    *,
    batch_size: int,
    destination: str | None = None,
) -> int:
    """COPY one parquet or CSV file into `destination` (default: the table)."""
    if file_path.suffix == ".parquet":
        return _copy_parquet_file_in_batches(
            conn,
            table_name,
            file_path,
            batch_size=batch_size,
            destination=destination,
        )
    return _copy_csv_file_in_batches(
        conn,
        table_name,
        file_path,
        batch_size=batch_size,
        destination=destination,
    )


def bulk_insert_csv_files(
    conn: Connection[Any],
    csv_paths: list[Path],
    table_name: str,
    *,
    batch_size: int,
    truncate: bool,
) -> None:
    """Load one or more parquet (or CSV) files into a destination table.

    Rows are COPYed into a temp staging table and upserted with ON CONFLICT
    inside a single transaction, so a load is atomic and re-running it never
    creates duplicate (location_id, date) rows.
    """
    if not csv_paths:
        LOGGER.warning("No data inputs found for %s. Skipping.", table_name)
        return

    key_columns = TABLE_UNIQUE_KEYS.get(table_name, ())
    validated_paths = [_validated_load_path(path) for path in csv_paths]
    column_names = _file_copy_column_names(table_name, validated_paths[0])
    if not column_names:
        LOGGER.warning("First input for %s has no columns. Skipping.", table_name)
        return

    LOGGER.info(
        "%s %s file(s) into %s...",
        "Truncating and loading" if truncate else "Upserting",
        len(validated_paths),
        table_name,
    )

    with conn.transaction():
        if truncate:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("TRUNCATE TABLE {} CASCADE").format(
                        sql.Identifier(table_name),
                    ),
                )

        staging_name = _create_staging_table(conn, table_name, column_names)
        staged_rows = 0
        for file_path in validated_paths:
            LOGGER.info("Staging %s for %s...", file_path, table_name)
            staged_rows += _copy_file(
                conn,
                table_name,
                file_path,
                batch_size=batch_size,
                destination=staging_name,
            )

        if staged_rows == 0:
            msg = f"Zero rows were staged for table {table_name}. Continuing load."
            LOGGER.warning(msg)
            return

        affected_rows = _upsert_from_staging(
            conn,
            table_name,
            staging_name,
            column_names,
            key_columns,
        )

    LOGGER.info(
        "Successfully loaded %d staged rows into %s (%d inserted or updated).",
        int(staged_rows),
        str(table_name),
        int(affected_rows),
    )


def _normalize_copy_column_names(
    table_name: str,
    column_names: list[str],
) -> list[str]:
    if table_name != "locations":
        return column_names

    return [
        "id" if column_name == "location_id" else column_name
        for column_name in column_names
    ]


def _discover_locations_csv_paths(args: argparse.Namespace) -> list[Path]:
    return [Path(args.locations_csv)]


def _discover_batch_parquet_paths(
    args: argparse.Namespace,
    *,
    direct_csv: str,
    root: str,
    file_glob: str,
    prefer_direct: bool,
) -> list[Path]:
    direct_csv_path = Path(direct_csv)
    if prefer_direct and direct_csv_path.exists():
        return _select_partition_shard_paths(
            [direct_csv_path],
            root_path=Path(root),
            partition_key="year",
            shard_index=args.load_shard_index,
            shard_count=args.load_shard_count,
        )

    csv_paths = _discover_csv_inputs(
        direct_csv,
        shard_root=root,
        shard_file_name=file_glob,
        shard_partition_key=None,
    )
    return _select_partition_shard_paths(
        csv_paths,
        root_path=Path(root),
        partition_key="year",
        shard_index=args.load_shard_index,
        shard_count=args.load_shard_count,
    )


def _discover_pet_csv_paths(args: argparse.Namespace) -> list[Path]:
    return _discover_batch_parquet_paths(
        args,
        direct_csv=args.pet_csv,
        root=args.pet_root,
        file_glob="pet_batch_*.parquet",
        prefer_direct=args.prefer_pet_csv,
    )


def _discover_wetbulb_csv_paths(args: argparse.Namespace) -> list[Path]:
    return _discover_batch_parquet_paths(
        args,
        direct_csv=args.wetbulb_csv,
        root=args.wetbulb_root,
        file_glob="wetbulb_batch_*.parquet",
        prefer_direct=args.prefer_wetbulb_csv,
    )


def _load_file_group(
    db_uri: str,
    csv_paths: list[Path],
    table_name: str,
    *,
    batch_size: int,
) -> None:
    """Load a group of files on a dedicated connection (parallel worker)."""
    conn: Connection[Any] = psycopg.connect(db_uri)
    conn.autocommit = True
    try:
        bulk_insert_csv_files(
            conn,
            csv_paths,
            table_name,
            batch_size=batch_size,
            truncate=False,
        )
    finally:
        conn.close()


def _load_table_files(
    conn: Connection[Any],
    db_uri: str,
    csv_paths: list[Path],
    table_name: str,
    *,
    batch_size: int,
    truncate: bool,
    workers: int,
) -> None:
    """Load files into a table, fanning out across connections when possible.

    Upserts on disjoint files touch disjoint (location_id, date) ranges, so
    parallel workers do not contend. Truncating loads keep the truncate and
    the load in one transaction and therefore stay single-connection.
    """
    effective_workers = max(1, min(workers, len(csv_paths)))
    if truncate or effective_workers == 1:
        bulk_insert_csv_files(
            conn,
            csv_paths,
            table_name,
            batch_size=batch_size,
            truncate=truncate,
        )
        return

    file_groups = [
        csv_paths[index::effective_workers] for index in range(effective_workers)
    ]
    LOGGER.info(
        "Loading %d file(s) into %s across %d parallel connections...",
        len(csv_paths),
        table_name,
        effective_workers,
    )
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(
                _load_file_group,
                db_uri,
                group,
                table_name,
                batch_size=batch_size,
            )
            for group in file_groups
            if group
        ]
        for future in futures:
            future.result()


def _load_requested_tables(
    conn: Connection[Any],
    args: argparse.Namespace,
    *,
    db_uri: str,
    truncate_tables: set[str],
    skip_tables: set[str],
) -> None:
    table_csv_resolvers = (
        ("locations", _discover_locations_csv_paths),
        ("pet", _discover_pet_csv_paths),
        ("wetbulb", _discover_wetbulb_csv_paths),
    )

    for table_name, csv_resolver in table_csv_resolvers:
        if table_name in skip_tables:
            continue

        csv_paths = csv_resolver(args)
        if not csv_paths:
            LOGGER.warning("No data inputs found for %s. Skipping.", table_name)
            continue

        _load_table_files(
            conn,
            db_uri,
            csv_paths,
            table_name,
            batch_size=args.copy_batch_size,
            truncate=table_name in truncate_tables
            and not (table_name == "locations" and args.skip_drop_views),
            workers=args.load_workers,
        )


def main() -> None:
    """Load datasets and recreate views."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    _validate_load_shard_args(args.load_shard_index, args.load_shard_count)
    db_uri = resolve_database_uri()

    if args.append_only:
        truncate_table_names: list[str] = []
    elif args.truncate_tables is None:
        truncate_table_names = TABLE_NAMES
    else:
        truncate_table_names = args.truncate_tables

    truncate_tables: set[str] = set(truncate_table_names)
    skip_tables: set[str] = set(args.skip_tables or [])

    if not db_uri:
        LOGGER.warning(
            "Postgres database credentials are not configured. %s Skipping database operations.",
            DATABASE_CONFIG_HINT,
        )
        return

    LOGGER.info("Connecting to the database...")
    conn: Connection[Any] = psycopg.connect(db_uri)
    conn.autocommit = True
    should_refresh_schema = not args.skip_drop_views or not args.skip_create_views

    try:
        if not args.skip_drop_views:
            execute_sql_file(conn, "drop_views.sql")

        if should_refresh_schema or args.ensure_schema:
            execute_sql_file(conn, "create_tables.sql")

        _load_requested_tables(
            conn,
            args,
            db_uri=db_uri,
            truncate_tables=truncate_tables,
            skip_tables=skip_tables,
        )

        if not args.skip_create_views:
            execute_sql_file(conn, "create_views.sql")

        if should_refresh_schema:
            refresh_query_planner_statistics(conn)

    finally:
        conn.close()
        LOGGER.info("Database connection closed.")


if __name__ == "__main__":
    main()
