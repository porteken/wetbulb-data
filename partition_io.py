# Copyright (C) 2026 Kenneth Porter

"""Shared parquet partition write/resume helpers for pipeline workers."""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple, cast

from shards import resolve_filesystem

pa = cast("Any", importlib.import_module("pyarrow"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))
pd = cast("Any", importlib.import_module("pandas"))

type DataFrame = Any
_PARQUET_METADATA_ERRORS = (OSError, pa.ArrowException)


class PartitionTarget(NamedTuple):
    """Where a product's parquet tree lives, with an optional pre-resolved filesystem."""

    root: str
    filesystem: object | None = None
    base_path: str | None = None


def write_batch_partition(
    root: str,
    year: int,
    city_shard_index: int,
    frame: DataFrame,
    batch_index: int,
    *,
    file_prefix: str,
    filesystem: object | None = None,
    base_path: str | None = None,
) -> None:
    """Write one batch of rows to `root/year=YYYY/{file_prefix}_batch_NNNN_SS.parquet`."""
    if filesystem is None or base_path is None:
        filesystem, base_path = resolve_filesystem(root)
    fs: Any = filesystem
    partition_dir = f"{base_path}/year={year}"
    fs.create_dir(partition_dir, recursive=True)

    output_path = f"{partition_dir}/{file_prefix}_batch_{batch_index:04d}_{city_shard_index:02d}.parquet"

    float_cols = frame.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        frame[float_cols] = frame[float_cols].astype("float32")

    table = pa.Table.from_pandas(frame, preserve_index=False)
    if getattr(fs, "type_name", "") == "local":
        temp_path = f"{output_path}.tmp"
        with fs.open_output_stream(temp_path) as out_stream:
            pq.write_table(table, out_stream, compression="snappy")
        fs.move(temp_path, output_path)
    else:
        with fs.open_output_stream(output_path) as out_stream:
            pq.write_table(table, out_stream, compression="snappy")


def batch_exists(
    root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
    *,
    file_prefix: str = "wetbulb",
    filesystem: object | None = None,
    base_path: str | None = None,
) -> bool:
    """Return True if a complete (footer-readable) batch partition exists."""
    if filesystem is None or base_path is None:
        filesystem, base_path = resolve_filesystem(root)
    fs: Any = filesystem
    path = f"{base_path}/year={year}/{file_prefix}_batch_{batch_index:04d}_{city_shard_index:02d}.parquet"
    try:
        if fs.get_file_info(path).type == 0:
            return False
        pq.read_metadata(path, filesystem=fs)
    except _PARQUET_METADATA_ERRORS:
        return False
    return True


def pending_years(
    years: range,
    root: str,
    city_shard_index: int,
    filesystem: object | None,
    base_path: str | None,
    *,
    file_prefix: str,
    force: bool,
) -> list[int]:
    """Return the years in `years` whose shard batch isn't already written."""
    return [
        year
        for year in years
        if force
        or not batch_exists(
            root,
            year,
            city_shard_index,
            0,
            file_prefix=file_prefix,
            filesystem=filesystem,
            base_path=base_path,
        )
    ]


def write_pending_year_batches(
    daily_df: DataFrame,
    pending_year_list: list[int],
    root: str,
    city_shard_index: int,
    filesystem: object | None,
    base_path: str | None,
    *,
    file_prefix: str,
) -> None:
    """Write each pending year's rows as its own shard partition."""
    daily_df["year"] = pd.to_datetime(daily_df["date"]).dt.year
    pending_year_set = set(pending_year_list)
    for year, year_df in daily_df.groupby("year"):
        if year not in pending_year_set:
            continue
        write_batch_partition(
            root,
            int(year),
            city_shard_index,
            year_df.drop(columns="year"),
            0,
            file_prefix=file_prefix,
            filesystem=filesystem,
            base_path=base_path,
        )
