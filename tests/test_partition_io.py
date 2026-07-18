"""Tests for shared parquet partition write/resume helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

import partition_io
from shards import resolve_filesystem


class TestPendingYears:
    def test_force_returns_all_years_without_checking(self, tmp_path: Any) -> None:
        filesystem, base_path = resolve_filesystem(str(tmp_path))
        result = partition_io.pending_years(
            range(2020, 2023),
            str(tmp_path),
            0,
            filesystem,
            base_path,
            file_prefix="wetbulb",
            force=True,
        )
        assert result == [2020, 2021, 2022]

    def test_filters_years_with_existing_batches(self, tmp_path: Any) -> None:
        wetbulb_root = str(tmp_path)
        filesystem, base_path = resolve_filesystem(wetbulb_root)
        df = pd.DataFrame(
            {
                "location_id": [1],
                "date": [pd.Timestamp("2020-06-01")],
                "wetbulb": [20.0],
                "wetbulb_avg": [19.0],
            }
        )
        partition_io.write_batch_partition(
            wetbulb_root,
            2020,
            0,
            df.copy(),
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )
        result = partition_io.pending_years(
            range(2020, 2022),
            wetbulb_root,
            0,
            filesystem,
            base_path,
            file_prefix="wetbulb",
            force=False,
        )
        assert result == [2021]

    def test_file_prefix_isolates_products(self, tmp_path: Any) -> None:
        """A batch written under one file_prefix doesn't satisfy another's resume check."""
        root = str(tmp_path)
        filesystem, base_path = resolve_filesystem(root)
        df = pd.DataFrame(
            {
                "location_id": [1],
                "date": [pd.Timestamp("2020-06-01")],
                "wetbulb": [20.0],
                "wetbulb_avg": [19.0],
            }
        )
        partition_io.write_batch_partition(
            root,
            2020,
            0,
            df.copy(),
            0,
            file_prefix="archive",
            filesystem=filesystem,
            base_path=base_path,
        )
        result = partition_io.pending_years(
            range(2020, 2021),
            root,
            0,
            filesystem,
            base_path,
            file_prefix="wetbulb",
            force=False,
        )
        assert result == [2020]


class TestWritePendingYearBatches:
    def test_writes_only_pending_years(self, tmp_path: Any) -> None:
        wetbulb_root = str(tmp_path)
        filesystem, base_path = resolve_filesystem(wetbulb_root)
        daily_df = pd.DataFrame(
            {
                "location_id": [1, 1],
                "date": [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")],
                "wetbulb": [20.0, 21.0],
                "wetbulb_avg": [19.0, 20.0],
            }
        )
        partition_io.write_pending_year_batches(
            daily_df,
            [2021],
            wetbulb_root,
            0,
            filesystem,
            base_path,
            file_prefix="wetbulb",
        )
        assert partition_io.batch_exists(
            wetbulb_root,
            2021,
            0,
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )
        assert not partition_io.batch_exists(
            wetbulb_root,
            2020,
            0,
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )

    def test_respects_file_prefix(self, tmp_path: Any) -> None:
        root = str(tmp_path)
        filesystem, base_path = resolve_filesystem(root)
        daily_df = pd.DataFrame(
            {
                "location_id": [1],
                "date": [pd.Timestamp("2020-06-01")],
                "wetbulb": [20.0],
                "wetbulb_avg": [19.0],
            }
        )
        partition_io.write_pending_year_batches(
            daily_df, [2020], root, 0, filesystem, base_path, file_prefix="wetbulb"
        )
        assert partition_io.batch_exists(
            root,
            2020,
            0,
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )
        assert not partition_io.batch_exists(
            root,
            2020,
            0,
            0,
            file_prefix="archive",
            filesystem=filesystem,
            base_path=base_path,
        )
