"""Tests for shard management."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shards import (
    ShardKey,
    discover_common_shards,
    discover_parquet_shards,
    read_parquet_files,
    resolve_filesystem,
    select_shards,
)


class TestShardKey:
    def test_partition_path_no_month(self) -> None:
        key = ShardKey(year=2020, tile_id=5)
        assert key.partition_path == "year=2020/tile_id=5"

    def test_partition_path_with_month(self) -> None:
        key = ShardKey(year=2020, month=3, tile_id=5)
        assert key.partition_path == "year=2020/month=03/tile_id=5"

    def test_label_no_month(self) -> None:
        key = ShardKey(year=2020, tile_id=5)
        assert key.label == "2020-tile-005"

    def test_label_with_month(self) -> None:
        key = ShardKey(year=2020, month=3, tile_id=5)
        assert key.label == "2020-03-tile-005"

    def test_equality(self) -> None:
        k1 = ShardKey(year=2020, tile_id=1)
        k2 = ShardKey(year=2020, tile_id=1)
        k3 = ShardKey(year=2021, tile_id=1)
        assert k1 == k2
        assert k1 != k3

    def test_sorting(self) -> None:
        k1 = ShardKey(year=2020, tile_id=2)
        k2 = ShardKey(year=2020, tile_id=1)
        k3 = ShardKey(year=2021, tile_id=1)
        assert sorted([k1, k2, k3]) == [k2, k1, k3]


class TestResolveFilesystem:
    def test_resolves_local_path(self, tmp_path: Path) -> None:
        fs, root = resolve_filesystem(tmp_path)
        from pyarrow.fs import LocalFileSystem

        assert isinstance(fs, LocalFileSystem)
        assert root == str(tmp_path.resolve())

    def test_expands_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", "/fake/home")
        _, root = resolve_filesystem("~/data")
        assert "/fake/home/data" in root


class TestDiscoverParquetShards:
    def test_returns_empty_on_missing_dir(self, tmp_path: Path) -> None:
        assert discover_parquet_shards(tmp_path / "missing") == {}

    def test_discovers_year_tile_shards(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        pd.DataFrame({"x": [1, 2, 3]}).to_parquet(shard_dir / "data.parquet")

        result = discover_parquet_shards(tmp_path)
        assert ShardKey(year=2020, tile_id=1) in result
        assert len(result[ShardKey(year=2020, tile_id=1)]) == 1

    def test_discovers_year_month_tile_shards(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "month=03" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_parquet(shard_dir / "data.parquet")

        result = discover_parquet_shards(tmp_path)
        assert ShardKey(year=2020, tile_id=1, month=3) in result

    def test_skips_non_parquet_files(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        (shard_dir / "not_parquet.txt").touch()

        result = discover_parquet_shards(tmp_path)
        assert result == {}


class TestDiscoverCommonShards:
    def test_returns_empty_on_no_roots(self) -> None:
        assert discover_common_shards() == []

    def test_returns_intersection(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        for root in [root_a, root_b]:
            shard_dir = root / "year=2020" / "tile_id=1"
            shard_dir.mkdir(parents=True)
            pd.DataFrame({"x": [1]}).to_parquet(shard_dir / "data.parquet")

        shard_dir_a_only = root_a / "year=2020" / "tile_id=2"
        shard_dir_a_only.mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_parquet(shard_dir_a_only / "data.parquet")

        common = discover_common_shards(root_a, root_b)
        assert len(common) == 1
        assert common[0] == ShardKey(year=2020, tile_id=1)


class TestSelectShards:
    def test_filters_by_year(self) -> None:
        keys = [ShardKey(2020, 1), ShardKey(2021, 1)]
        selected = select_shards(keys, year=2020)
        assert selected == [ShardKey(2020, 1)]

    def test_filters_by_tile_ids(self) -> None:
        keys = [ShardKey(2020, 1), ShardKey(2020, 2)]
        selected = select_shards(keys, tile_ids=[1])
        assert selected == [ShardKey(2020, 1)]

    def test_shards_evenly(self) -> None:
        keys = [ShardKey(2020, i) for i in range(10)]
        s0 = select_shards(keys, shard_index=0, shard_count=2)
        s1 = select_shards(keys, shard_index=1, shard_count=2)
        assert len(s0) == 5
        assert len(s1) == 5
        assert set(s0) & set(s1) == set()

    def test_raises_on_invalid_shard_index(self) -> None:
        with pytest.raises(ValueError, match="shard_index"):
            select_shards([], shard_index=2, shard_count=2)

    def test_raises_on_zero_shard_count(self) -> None:
        with pytest.raises(ValueError, match="shard_count"):
            select_shards([], shard_count=0)


class TestReadParquetFiles:
    def test_reads_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "year=2020" / "tile_id=1"
        path.mkdir(parents=True)
        file_path = path / "data.parquet"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(file_path)

        df = read_parquet_files(tmp_path, [str(file_path)], columns=["a"])
        assert list(df.columns) == ["a"]
        assert len(df) == 2
