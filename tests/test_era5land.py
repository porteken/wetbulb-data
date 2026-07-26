"""Tests for the EU ERA5-Land gap-filler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import era5land


class _StubCdsClient:
    """Writes a queued CSV (or raises) each time `retrieve` is called."""

    def __init__(self, outcomes: list[pd.DataFrame | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    def retrieve(self, _dataset: str, request: dict[str, Any], target: str) -> None:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        outcome.to_csv(target, index=False)


def _raw_era5land_frame(hours: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid_time": hours,
            "t2m": [293.15] * len(hours),
            "d2m": [283.15] * len(hours),
            "sp": [101325.0] * len(hours),
        }
    )


class TestFetchCitySpan:
    def test_writes_request_and_returns_parsed_frame(self, tmp_path: Path) -> None:
        client = _StubCdsClient([_raw_era5land_frame(["2020-01-01T00:00:00"])])
        result = era5land.fetch_city_span(
            client,
            lat=51.5,
            lng=-0.1,
            start_year=2020,
            end_year=2020,
            download_dir=str(tmp_path),
        )
        assert list(result["t2m"]) == [293.15]
        assert client.requests[0]["location"] == {"latitude": 51.5, "longitude": -0.1}
        assert client.requests[0]["date"] == ["2020-01-01/2020-12-31"]


class TestHourlyFrameFromEra5land:
    def test_converts_units_and_shifts_local_time(self) -> None:
        raw = _raw_era5land_frame(["2020-06-01T12:00:00"])
        result = era5land._hourly_frame_from_era5land(1000, raw, utc_offset_hours=1)
        assert list(result.columns) == ["location_id", "time", "Tair", "Qair", "PSurf"]
        assert result["Tair"].iloc[0] == pytest.approx(293.15)
        assert result["PSurf"].iloc[0] == pytest.approx(101325.0)
        assert 0 < result["Qair"].iloc[0] < 0.02
        assert result["time"].iloc[0] == pd.Timestamp("2020-06-01T13:00:00")

    def test_empty_input_returns_empty_typed_frame(self) -> None:
        result = era5land._hourly_frame_from_era5land(
            1000, pd.DataFrame(), utc_offset_hours=0
        )
        assert result.empty
        assert list(result.columns) == ["location_id", "time", "Tair", "Qair", "PSurf"]


class TestFetchCityGaps:
    @staticmethod
    def _row(location_id: int = 1000, lat: float = 51.5, lng: float = -0.1) -> Any:
        rows = pd.DataFrame(
            {
                "location_id": [location_id],
                "lat": [lat],
                "lng": [lng],
                "utc_offset_hours": [0],
            }
        ).itertuples()
        return next(iter(rows))

    def test_fetches_each_contiguous_range_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[tuple[int, int]] = []

        def fake_fetch(
            _client: Any,
            *,
            lat: float,
            lng: float,
            start_year: int,
            end_year: int,
            download_dir: str,
        ) -> pd.DataFrame:
            calls.append((start_year, end_year))
            return _raw_era5land_frame([f"{start_year}-01-01T00:00:00"])

        monkeypatch.setattr(era5land, "fetch_city_span", fake_fetch)
        frame, had_gap = era5land._fetch_city_gaps(
            None, self._row(), [2000, 2001, 2010], str(tmp_path)
        )
        assert calls == [(2000, 2001), (2010, 2010)]
        assert not had_gap
        assert len(frame) == 2

    def test_retries_then_gives_up_marks_gap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(era5land, "ERA5LAND_RETRY_DELAY_SECONDS", 0)

        def always_fails(*_a: Any, **_k: Any) -> pd.DataFrame:
            msg = "boom"
            raise RuntimeError(msg)

        monkeypatch.setattr(era5land, "fetch_city_span", always_fails)
        frame, had_gap = era5land._fetch_city_gaps(
            None, self._row(), [2000], str(tmp_path)
        )
        assert had_gap
        assert frame.empty

    def test_succeeds_on_a_later_retry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(era5land, "ERA5LAND_RETRY_DELAY_SECONDS", 0)
        attempts: list[int] = []

        def flaky(*_a: Any, **_k: Any) -> pd.DataFrame:
            attempts.append(1)
            if len(attempts) < 2:
                msg = "transient"
                raise OSError(msg)
            return _raw_era5land_frame(["2000-01-01T00:00:00"])

        monkeypatch.setattr(era5land, "fetch_city_span", flaky)
        frame, had_gap = era5land._fetch_city_gaps(
            None, self._row(), [2000], str(tmp_path)
        )
        assert not had_gap
        assert len(frame) == 1


class TestFetchGapsBatch:
    def test_fetches_each_row_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            TestFetchCityGaps._row(location_id=1000),
            TestFetchCityGaps._row(location_id=1001, lat=48.85, lng=2.35),
        ]
        gap_years_by_location = {1000: [2020], 1001: [2020]}

        def fake_fetch(_client: Any, row: Any, _years: Any, _dir: Any) -> Any:
            return _raw_era5land_frame(["2020-01-01T00:00:00"]), False

        monkeypatch.setattr(era5land, "_fetch_city_gaps", fake_fetch)
        results = era5land._fetch_gaps_batch(
            rows, gap_years_by_location, None, str(tmp_path), 2, 0
        )
        assert set(results) == {1000, 1001}
        assert all(not gap for _frame, gap in results.values())


class TestLoadUtcOffsets:
    def test_missing_file_returns_empty_typed_frame(self, tmp_path: Path) -> None:
        result = era5land._load_utc_offsets(str(tmp_path / "missing.csv"))
        assert result.empty
        assert list(result.columns) == ["location_id", "utc_offset_hours"]

    def test_loads_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cities_eu_isd_stations.csv"
        path.write_text("location_id,utc_offset_hours\n1000,1\n")
        result = era5land._load_utc_offsets(str(path))
        assert list(result["location_id"]) == [1000]
        assert list(result["utc_offset_hours"]) == [1]


class TestProcessEra5landGapfill:
    @staticmethod
    def _shard_df() -> pd.DataFrame:
        return pd.DataFrame({"location_id": [1000], "lat": [51.5], "lng": [-0.1]})

    def test_no_cities_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            era5land.nldas,
            "load_nldas_city_shard",
            lambda *_a: pd.DataFrame(columns=pd.Index(["location_id", "lat", "lng"])),
        )
        with caplog.at_level("INFO"):
            era5land.process_era5land_gapfill(2020, 2020, str(tmp_path), 0, 1, 2)
        assert any("No cities found" in m for m in caplog.messages)

    def test_no_pending_years_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            era5land.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(era5land, "pending_years", lambda *_a, **_k: [])
        with caplog.at_level("INFO"):
            era5land.process_era5land_gapfill(2020, 2020, str(tmp_path), 0, 1, 2)
        assert any("already present" in m for m in caplog.messages)

    def test_no_gaps_found_returns_early_without_fetching(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            era5land.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(era5land, "pending_years", lambda *_a, **_k: [2020])
        monkeypatch.setattr(
            era5land,
            "find_missing_cells",
            lambda *_a, **_k: pd.DataFrame(
                {
                    "location_id": pd.Series(dtype="int64"),
                    "date": pd.Series(dtype="datetime64[ns]"),
                }
            ),
        )
        called: list[int] = []
        monkeypatch.setattr(era5land, "_cds_client", lambda: called.append(1))
        with caplog.at_level("INFO"):
            era5land.process_era5land_gapfill(2020, 2020, str(tmp_path), 0, 1, 2)
        assert not called
        assert any("nothing to fill" in m for m in caplog.messages)

    def _missing_cells(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": [1000],
                "date": pd.to_datetime(["2020-01-01"]),
            }
        )

    def _stub_fetch_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            era5land.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(era5land, "pending_years", lambda *_a, **_k: [2020])
        monkeypatch.setattr(
            era5land, "find_missing_cells", lambda *_a, **_k: self._missing_cells()
        )
        monkeypatch.setattr(era5land, "_filter_material_gaps", lambda cells, _n: cells)
        monkeypatch.setattr(era5land, "_cds_client", object)
        monkeypatch.setattr(
            era5land,
            "_load_utc_offsets",
            lambda _path: pd.DataFrame(
                {"location_id": [1000], "utc_offset_hours": [0]}
            ),
        )

    def test_fetch_gap_skips_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._stub_fetch_setup(monkeypatch)
        hourly = pd.DataFrame(
            {
                "location_id": [1000],
                "time": [pd.Timestamp("2020-01-01")],
                "Tair": [290.0],
                "Qair": [0.01],
                "PSurf": [100000.0],
            }
        )
        monkeypatch.setattr(
            era5land, "_fetch_gaps_batch", lambda *_a, **_k: {1000: (hourly, True)}
        )
        monkeypatch.setattr(
            era5land.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1000],
                    "date": [pd.Timestamp("2020-01-01").date()],
                    "wetbulb": [20.0],
                    "wetbulb_avg": [19.0],
                }
            ),
        )
        write_called: list[int] = []
        monkeypatch.setattr(
            era5land,
            "write_pending_year_batches",
            lambda *_a, **_k: write_called.append(1),
        )
        with caplog.at_level("WARNING"):
            era5land.process_era5land_gapfill(2020, 2020, str(tmp_path), 0, 1, 2)
        assert not write_called
        assert any("fetch gap" in m for m in caplog.messages)

    def test_successful_run_writes_only_gapped_cells_with_era5land_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        self._stub_fetch_setup(monkeypatch)
        hourly = pd.DataFrame(
            {
                "location_id": [1000],
                "time": [pd.Timestamp("2020-01-01")],
                "Tair": [290.0],
                "Qair": [0.01],
                "PSurf": [100000.0],
            }
        )
        monkeypatch.setattr(
            era5land, "_fetch_gaps_batch", lambda *_a, **_k: {1000: (hourly, False)}
        )
        monkeypatch.setattr(
            era5land.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1000, 1000],
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
            era5land,
            "write_pending_year_batches",
            lambda *args, **_k: write_calls.append(args),
        )
        era5land.process_era5land_gapfill(2020, 2020, str(tmp_path), 0, 1, 2)

        assert len(write_calls) == 1
        written_frame = write_calls[0][0]
        assert len(written_frame) == 1
        assert written_frame["date"].iloc[0] == pd.Timestamp("2020-01-01").date()
        assert (written_frame["source"] == "era5land").all()


class TestParseArgs:
    def test_defaults_use_eu_cities_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(era5land.sys, "argv", ["era5land.py"])
        args = era5land._parse_args()
        assert args.cities_csv == era5land.EU_CITIES_CSV
        assert args.station_map_csv == era5land.EU_STATION_MAP_CSV
        assert args.concurrency == era5land.ERA5LAND_DEFAULT_CONCURRENCY
        assert args.min_missing_days == era5land.MIN_MISSING_DAYS_DEFAULT
        assert args.location_ids is None


class TestMain:
    def test_keyboard_interrupt_exits_130(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_interrupt() -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(era5land, "load_dotenv", lambda **_k: None)
        monkeypatch.setattr(era5land, "_parse_args", raise_interrupt)
        with pytest.raises(SystemExit) as exc_info:
            era5land.main()
        assert exc_info.value.code == 130

    def test_handled_exception_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_value_error() -> Any:
            msg = "boom"
            raise ValueError(msg)

        monkeypatch.setattr(era5land, "load_dotenv", lambda **_k: None)
        monkeypatch.setattr(era5land, "_parse_args", raise_value_error)
        with pytest.raises(SystemExit) as exc_info:
            era5land.main()
        assert exc_info.value.code == 1
