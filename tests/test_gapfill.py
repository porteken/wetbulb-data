# Copyright (C) 2026 Kenneth Porter

"""Tests for the NLDAS-2 gap-filler."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd

import gapfill
from partition_io import write_batch_partition
from shards import resolve_filesystem

if TYPE_CHECKING:
    import pytest


class _StubTokenManager:
    def get(self) -> str:
        return "tok"

    def refresh(self) -> str:
        return "tok"


class TestFindMissingCells:
    def test_thin_year_yields_calendar_minus_present(self, tmp_path: Any) -> None:
        root = str(tmp_path / "wetbulb_data_csv")
        filesystem, base_path = resolve_filesystem(root)

        thin_dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")[:100]
        df = pd.DataFrame(
            {
                "location_id": 2,
                "date": thin_dates,
                "wetbulb": 20.0,
                "wetbulb_avg": 18.0,
                "source": "isd",
            }
        )
        write_batch_partition(
            root,
            2020,
            0,
            df,
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )

        missing = gapfill.find_missing_cells([2], [2020], filesystem, base_path)

        assert len(missing) == 366 - 100
        assert set(missing["location_id"]) == {2}

    def test_complete_year_yields_no_gaps(self, tmp_path: Any) -> None:
        root = str(tmp_path / "wetbulb_data_csv")
        filesystem, base_path = resolve_filesystem(root)

        full_dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        df = pd.DataFrame(
            {
                "location_id": 1,
                "date": full_dates,
                "wetbulb": 20.0,
                "wetbulb_avg": 18.0,
                "source": "isd",
            }
        )
        write_batch_partition(
            root,
            2020,
            0,
            df,
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )

        missing = gapfill.find_missing_cells([1], [2020], filesystem, base_path)

        assert missing.empty

    def test_no_isd_output_at_all_yields_full_calendar(self, tmp_path: Any) -> None:
        root = str(tmp_path / "wetbulb_data_csv")
        filesystem, base_path = resolve_filesystem(root)

        missing = gapfill.find_missing_cells([1], [2021], filesystem, base_path)

        assert len(missing) == 365

    def test_eccc_output_counts_as_primary_station_coverage(
        self, tmp_path: Any
    ) -> None:
        root = str(tmp_path / "wetbulb_data_csv")
        filesystem, base_path = resolve_filesystem(root)
        station_df = pd.DataFrame(
            {
                "location_id": [1],
                "date": [pd.Timestamp("2021-01-01").date()],
                "wetbulb": [20.0],
                "wetbulb_avg": [18.0],
                "source": ["eccc"],
            },
        )
        write_batch_partition(
            root,
            2021,
            0,
            station_df,
            0,
            file_prefix="wetbulb_eccc",
            filesystem=filesystem,
            base_path=base_path,
        )

        missing = gapfill.find_missing_cells([1], [2021], filesystem, base_path)

        assert len(missing) == 364

    def test_ignores_fill_files_when_detecting_gaps(self, tmp_path: Any) -> None:
        """A gap is always relative to ISD output, never to a prior gap-fill run."""
        root = str(tmp_path / "wetbulb_data_csv")
        filesystem, base_path = resolve_filesystem(root)

        fill_df = pd.DataFrame(
            {
                "location_id": [1],
                "date": [pd.Timestamp("2020-01-01").date()],
                "wetbulb": [20.0],
                "wetbulb_avg": [18.0],
                "source": ["nldas"],
            }
        )
        write_batch_partition(
            root,
            2020,
            0,
            fill_df,
            0,
            file_prefix="wetbulb_fill",
            filesystem=filesystem,
            base_path=base_path,
        )

        missing = gapfill.find_missing_cells([1], [2020], filesystem, base_path)

        assert len(missing) == 366

    def test_current_year_stops_at_era5land_availability_window(
        self,
        tmp_path: Any,
    ) -> None:
        root = str(tmp_path / "wetbulb_data_csv")
        filesystem, base_path = resolve_filesystem(root)
        current_year = pd.Timestamp.now(tz="UTC").year

        missing = gapfill.find_missing_cells(
            [1],
            [current_year],
            filesystem,
            base_path,
        )

        expected_end = (
            pd.Timestamp.now(tz="UTC").normalize()
            - timedelta(days=gapfill.CURRENT_YEAR_LAG_DAYS)
        ).date()
        assert missing["date"].max().date() == expected_end


class TestGapYearsByLocation:
    def test_groups_and_sorts_years_per_location(self) -> None:
        missing = pd.DataFrame(
            {
                "location_id": [1, 1, 2],
                "date": pd.to_datetime(["2021-01-01", "2000-06-01", "2005-01-01"]),
            }
        )

        result = gapfill._gap_years_by_location(missing)

        assert result == {1: [2000, 2021], 2: [2005]}


class TestFilterMaterialGaps:
    def test_drops_city_years_below_threshold(self) -> None:
        missing = pd.DataFrame(
            {
                "location_id": [1] * 3 + [1] * 20 + [2] * 2,
                "date": pd.to_datetime(
                    [f"2020-01-{d:02d}" for d in range(1, 4)]
                    + [f"2021-03-{d:02d}" for d in range(1, 21)]
                    + ["2021-06-01", "2021-06-02"]
                ),
            }
        )

        kept = gapfill._filter_material_gaps(missing, 19)

        assert len(kept) == 20
        assert set(kept["location_id"]) == {1}
        assert (kept["date"].dt.year == 2021).all()

    def test_threshold_of_one_keeps_everything(self) -> None:
        missing = pd.DataFrame(
            {
                "location_id": [1],
                "date": pd.to_datetime(["2020-01-01"]),
            }
        )

        assert len(gapfill._filter_material_gaps(missing, 1)) == 1

    def test_empty_input_passes_through(self) -> None:
        empty = gapfill._empty_cells_frame()

        assert gapfill._filter_material_gaps(empty, 19).empty


class TestResolveGapfillTargets:
    @staticmethod
    def _shard_df() -> pd.DataFrame:
        return pd.DataFrame(
            {"location_id": [1, 2], "lat": [40.0, 41.0], "lng": [-74.0, -75.0]}
        )

    def _resolve(self, tmp_path: Any, *, location_ids: list[int] | None) -> Any:
        return gapfill.resolve_gapfill_targets(
            self._shard_df(),
            f"{tmp_path}/wetbulb_data_csv",
            2020,
            2020,
            0,
            1,
            location_ids=location_ids,
            min_missing_days=1,
            force=False,
            logger=gapfill.LOGGER,
        )

    def test_narrows_the_shard_to_the_requested_locations(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            gapfill,
            "find_missing_cells",
            lambda ids, *_a, **_k: pd.DataFrame(
                {"location_id": ids, "date": pd.to_datetime(["2020-01-01"] * len(ids))}
            ),
        )

        resolved = self._resolve(tmp_path, location_ids=[2])

        assert resolved is not None
        assert resolved[0]["location_id"].tolist() == [2]
        assert resolved[4]["location_id"].tolist() == [2]

    def test_returns_none_when_the_filter_empties_the_shard(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO"):
            resolved = self._resolve(tmp_path, location_ids=[999])

        assert resolved is None
        assert any("No cities found" in message for message in caplog.messages)

    def test_limits_missing_cells_to_incremental_dates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            gapfill,
            "find_missing_cells",
            lambda *_args, **_kwargs: pd.DataFrame(
                {
                    "location_id": [1, 1, 1],
                    "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                }
            ),
        )

        resolved = gapfill.resolve_gapfill_targets(
            self._shard_df(),
            f"{tmp_path}/wetbulb_data_csv",
            2020,
            2020,
            0,
            1,
            location_ids=None,
            min_missing_days=1,
            force=False,
            logger=gapfill.LOGGER,
            start_date="2020-01-02",
            end_date="2020-01-02",
        )

        assert resolved is not None
        assert resolved[4]["date"].dt.strftime("%F").tolist() == ["2020-01-02"]


class TestProcessGapfill:
    @staticmethod
    def _shard_df() -> pd.DataFrame:
        return pd.DataFrame({"location_id": [1], "lat": [40.0], "lng": [-74.0]})

    @staticmethod
    def _empty_cell_map() -> pd.DataFrame:
        return pd.DataFrame(columns=pd.Index(["location_id", "cell_lat", "cell_lon"]))

    def test_no_cities_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            gapfill.nldas,
            "load_nldas_city_shard",
            lambda *_a: pd.DataFrame(columns=pd.Index(["location_id", "lat", "lng"])),
        )
        with caplog.at_level("INFO"):
            gapfill.process_gapfill(2020, 2020, str(tmp_path), 0, 1, 4)
        assert any("No cities found" in m for m in caplog.messages)

    def test_no_pending_years_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            gapfill.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(gapfill, "pending_years", lambda *_a, **_k: [])
        with caplog.at_level("INFO"):
            gapfill.process_gapfill(2020, 2020, str(tmp_path), 0, 1, 4)
        assert any("already present" in m for m in caplog.messages)

    def test_no_gaps_found_returns_early_without_fetching(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            gapfill.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(gapfill, "pending_years", lambda *_a, **_k: [2020])
        monkeypatch.setattr(
            gapfill,
            "find_missing_cells",
            lambda *_a, **_k: gapfill._empty_cells_frame(),
        )
        called: list[int] = []
        monkeypatch.setattr(
            gapfill, "_fetch_gaps_with_retries", lambda *_a, **_k: called.append(1)
        )
        with caplog.at_level("INFO"):
            gapfill.process_gapfill(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not called
        assert any("nothing to fill" in m for m in caplog.messages)

    def _missing_cells(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": [1],
                "date": pd.to_datetime(["2020-01-01"]),
            }
        )

    def _stub_fetch_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            gapfill.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(gapfill, "pending_years", lambda *_a, **_k: [2020])
        monkeypatch.setattr(
            gapfill, "find_missing_cells", lambda *_a, **_k: self._missing_cells()
        )
        monkeypatch.setattr(gapfill, "_filter_material_gaps", lambda cells, _n: cells)
        monkeypatch.setattr(gapfill.giovanni, "load_cell_map", self._empty_cell_map)
        monkeypatch.setattr(
            gapfill.giovanni.TokenManager, "from_env", _StubTokenManager
        )
        monkeypatch.setattr(gapfill.requests, "Session", SimpleNamespace)

    def test_fetch_gap_skips_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._stub_fetch_setup(monkeypatch)
        hourly = pd.DataFrame(
            {
                "location_id": [1],
                "time": [pd.Timestamp("2020-01-01")],
                "Tair": [290.0],
                "Qair": [0.01],
                "PSurf": [100000.0],
            }
        )
        monkeypatch.setattr(
            gapfill, "_fetch_gaps_with_retries", lambda *_a, **_k: {1: (hourly, True)}
        )
        monkeypatch.setattr(
            gapfill.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1],
                    "date": [pd.Timestamp("2020-01-01").date()],
                    "wetbulb": [20.0],
                    "wetbulb_avg": [19.0],
                }
            ),
        )
        write_called: list[int] = []
        monkeypatch.setattr(
            gapfill,
            "write_pending_year_batches",
            lambda *_a, **_k: write_called.append(1),
        )
        with caplog.at_level("WARNING"):
            gapfill.process_gapfill(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not write_called
        assert any("fetch gap" in m for m in caplog.messages)

    def test_empty_daily_df_skips_write(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        self._stub_fetch_setup(monkeypatch)
        hourly = pd.DataFrame(
            {
                "location_id": [1],
                "time": [pd.Timestamp("2020-01-01")],
                "Tair": [290.0],
                "Qair": [0.01],
                "PSurf": [100000.0],
            }
        )
        monkeypatch.setattr(
            gapfill, "_fetch_gaps_with_retries", lambda *_a, **_k: {1: (hourly, False)}
        )
        monkeypatch.setattr(
            gapfill.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                columns=pd.Index(["location_id", "date", "wetbulb", "wetbulb_avg"])
            ),
        )
        write_called: list[int] = []
        monkeypatch.setattr(
            gapfill,
            "write_pending_year_batches",
            lambda *_a, **_k: write_called.append(1),
        )
        gapfill.process_gapfill(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not write_called

    def test_fetched_data_not_matching_a_gap_skips_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The fetch succeeded but landed on a date outside the known gap set."""
        self._stub_fetch_setup(monkeypatch)
        hourly = pd.DataFrame(
            {
                "location_id": [1],
                "time": [pd.Timestamp("2020-06-01")],
                "Tair": [290.0],
                "Qair": [0.01],
                "PSurf": [100000.0],
            }
        )
        monkeypatch.setattr(
            gapfill, "_fetch_gaps_with_retries", lambda *_a, **_k: {1: (hourly, False)}
        )
        monkeypatch.setattr(
            gapfill.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1],
                    "date": [pd.Timestamp("2020-06-01").date()],
                    "wetbulb": [20.0],
                    "wetbulb_avg": [19.0],
                }
            ),
        )
        write_called: list[int] = []
        monkeypatch.setattr(
            gapfill,
            "write_pending_year_batches",
            lambda *_a, **_k: write_called.append(1),
        )
        with caplog.at_level("INFO"):
            gapfill.process_gapfill(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not write_called
        assert any("no rows matching a known gap" in m for m in caplog.messages)

    def test_successful_run_writes_only_gapped_cells_with_nldas_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        self._stub_fetch_setup(monkeypatch)
        hourly = pd.DataFrame(
            {
                "location_id": [1],
                "time": [pd.Timestamp("2020-01-01")],
                "Tair": [290.0],
                "Qair": [0.01],
                "PSurf": [100000.0],
            }
        )
        monkeypatch.setattr(
            gapfill, "_fetch_gaps_with_retries", lambda *_a, **_k: {1: (hourly, False)}
        )
        monkeypatch.setattr(
            gapfill.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1, 1],
                    "date": [
                        pd.Timestamp("2020-01-01").date(),
                        pd.Timestamp("2020-01-02").date(),
                    ],
                    "wetbulb": [20.0, 21.0],
                    "wetbulb_avg": [19.0, 20.0],
                }
            ),
        )
        write_calls: list[Any] = []
        monkeypatch.setattr(
            gapfill,
            "write_pending_year_batches",
            lambda *args, **_k: write_calls.append(args),
        )
        gapfill.process_gapfill(2020, 2020, str(tmp_path), 0, 1, 4)

        assert len(write_calls) == 1
        written_frame = write_calls[0][0]
        assert len(written_frame) == 1
        assert written_frame["date"].iloc[0] == pd.Timestamp("2020-01-01").date()
        assert (written_frame["source"] == "nldas").all()

    def test_resume_skips_already_written_gap_fill_year(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = str(tmp_path / "wetbulb_data_csv")
        filesystem, base_path = resolve_filesystem(root)
        fill_df = pd.DataFrame(
            {
                "location_id": [1],
                "date": [pd.Timestamp("2020-01-01").date()],
                "wetbulb": [20.0],
                "wetbulb_avg": [19.0],
                "source": ["nldas"],
            }
        )
        write_batch_partition(
            root,
            2020,
            0,
            fill_df,
            0,
            file_prefix="wetbulb_fill",
            filesystem=filesystem,
            base_path=base_path,
        )

        monkeypatch.setattr(
            gapfill.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        called: list[int] = []
        monkeypatch.setattr(
            gapfill, "find_missing_cells", lambda *_a, **_k: called.append(1)
        )

        gapfill.process_gapfill(2020, 2020, str(tmp_path), 0, 1, 4)

        assert not called


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gapfill.sys, "argv", ["gapfill.py"])
        args = gapfill._parse_args()
        assert args.start_year == gapfill.nldas.NLDAS_START_YEAR
        assert args.end_year == gapfill.nldas.NLDAS_END_YEAR
        assert args.out_dir == "."
        assert args.city_shard_index == 0
        assert args.city_shard_count == 1
        assert args.concurrency == gapfill.DEFAULT_CONCURRENCY
        assert args.force is False
        assert args.location_ids is None
        assert args.min_missing_days == gapfill.MIN_MISSING_DAYS_DEFAULT

    def test_location_ids_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            gapfill.sys,
            "argv",
            ["gapfill.py", "--location-ids", "1", "2", "3"],
        )
        args = gapfill._parse_args()
        assert args.location_ids == [1, 2, 3]
