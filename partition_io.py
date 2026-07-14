"""Shared parquet partition write/resume helpers for pipeline workers.

Used by both `google_era5.py` (PET) and `nldas.py` (wet-bulb) so batch
outputs land in the same `year=YYYY/{prefix}_batch_NNNN_SS.parquet` tree
regardless of which worker produced them.
"""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple, cast

from shards import resolve_filesystem

pa = cast("Any", importlib.import_module("pyarrow"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))

type DataFrame = Any


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
        # Write-then-rename so an interrupted run never leaves a partial file
        # at the final path (which the resume check would treat as complete).
        # S3 writes need no equivalent: an incomplete multipart upload never
        # materializes the object.
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
    file_prefix: str = "pet",
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
        # Validate the footer (a cheap ranged read) so a truncated or corrupt
        # file from an interrupted run is recomputed instead of skipped.
        pq.read_metadata(path, filesystem=fs)
    except OSError, pa.ArrowException:
        return False
    return True
