"""Tests for the NOAA LCD v2 wet-bulb worker (default wetbulb pipeline source)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import lcd

SAMPLE_LCD_CSV = """"STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","NAME","REPORT_TYPE","SOURCE","HourlyAltimeterSetting","HourlyDewPointTemperature","HourlyDryBulbTemperature","HourlyStationPressure","HourlySeaLevelPressure","HourlyWetBulbTemperature"
"72503014732","2024-01-01T00:51:00","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-15","7","1017.6","2.0","5.0","1016.0","1017.5","3.5"
"72503014732","2024-01-01T01:51:00","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-15","7","1017.8","2.5","5.5","1016.2","1017.7","4.0"
"72503014732","2024-01-01T01:53:00","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-16","7","1017.8","2.4","5.4","1016.1","1017.6","3.9"
"72503014732","2024-01-01T06:00:00","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","SOD","7","","","","","",""
"""

NO_HOURLY_CSV = """"STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","NAME","REPORT_TYPE","SOURCE","HourlyAltimeterSetting","HourlyDewPointTemperature","HourlyDryBulbTemperature","HourlyStationPressure","HourlySeaLevelPressure","HourlyWetBulbTemperature"
"72503014732","2024-01-01T06:00:00","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","SOD","7","","","","","",""
"72503014732","2024-01-31T23:59:00","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","SOM","7","","","","","",""
"""

NO_STATION_PRESSURE_CSV = """"STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","NAME","REPORT_TYPE","SOURCE","HourlyAltimeterSetting","HourlyDewPointTemperature","HourlyDryBulbTemperature","HourlyStationPressure","HourlySeaLevelPressure","HourlyWetBulbTemperature"
"72503014732","2000-01-01T00:51:00","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-15","7","1017.6","2.0","5.0","","1017.5",""
"""


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = f"HTTP {self.status_code}"
            raise lcd.requests.HTTPError(msg)


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> _FakeResponse:
        self.calls.append(url)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TestGetWithRetries:
    def test_returns_404_without_retrying(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(404)]))
        response = lcd._get_with_retries(
            session, "http://x", station_id="AAA", year=2020
        )
        assert response is not None
        assert response.status_code == 404

    def test_returns_none_after_exhausting_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lcd.time, "sleep", lambda _s: None)
        session = cast(
            "Any",
            _FakeSession([lcd.requests.RequestException("boom")] * lcd.LCD_MAX_RETRIES),
        )
        response = lcd._get_with_retries(
            session, "http://x", station_id="AAA", year=2020
        )
        assert response is None

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(lcd.time, "sleep", lambda _s: None)
        session = cast(
            "Any",
            _FakeSession(
                [lcd.requests.RequestException("boom"), _FakeResponse(200, "ok")]
            ),
        )
        response = lcd._get_with_retries(
            session, "http://x", station_id="AAA", year=2020
        )
        assert response is not None
        assert response.status_code == 200
        assert response.text == "ok"


class TestFetchStationYear:
    def test_parses_hourly_rows_and_dedupes_by_priority(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(200, SAMPLE_LCD_CSV)]))
        frame, gap = lcd.fetch_station_year("72503014732", 2024, session=session)
        assert gap is False
        assert len(frame) == 3
        assert set(frame["tair_c"]) == {5.0, 5.5, 5.4}

    def test_returns_empty_not_gap_on_404(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(404)]))
        frame, gap = lcd.fetch_station_year("72503014732", 1990, session=session)
        assert frame.empty
        assert gap is False

    def test_returns_empty_not_gap_when_only_daily_summaries(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(200, NO_HOURLY_CSV)]))
        frame, gap = lcd.fetch_station_year("72503014732", 2024, session=session)
        assert frame.empty
        assert gap is False

    def test_returns_gap_true_on_exhausted_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lcd.time, "sleep", lambda _s: None)
        session = cast(
            "Any",
            _FakeSession([lcd.requests.RequestException("boom")] * lcd.LCD_MAX_RETRIES),
        )
        frame, gap = lcd.fetch_station_year("72503014732", 2024, session=session)
        assert frame.empty
        assert gap is True

    def test_pressure_fallback_uses_sea_level_when_station_pressure_missing(
        self,
    ) -> None:
        session = cast(
            "Any", _FakeSession([_FakeResponse(200, NO_STATION_PRESSURE_CSV)])
        )
        frame, gap = lcd.fetch_station_year("72503014732", 2000, session=session)
        assert gap is False
        assert len(frame) == 1
        assert frame["pressure_hpa"].iloc[0] < 1017.5
        assert frame["pressure_hpa"].iloc[0] > 1010.0


class TestDewpointToSpecificHumidity:
    def test_round_trips_through_wetbulb_vapor_pressure(self) -> None:
        import wetbulb

        dewpoint_c = pd.Series([10.0, 20.0])
        pressure_hpa = pd.Series([1000.0, 1000.0])
        qair = lcd._dewpoint_to_specific_humidity(dewpoint_c, pressure_hpa)

        epsilon = wetbulb.EPSILON
        reconstructed_e = qair * pressure_hpa / (epsilon + (1 - epsilon) * qair)
        expected_e = wetbulb.saturation_vapor_pressure_hpa(dewpoint_c)
        assert reconstructed_e.to_numpy() == pytest.approx(
            expected_e.to_numpy(), rel=1e-9
        )


class TestFetchStationSeries:
    def test_aggregates_gapped_years_and_concatenates_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_fetch_station_year(
            station_id: str, year: int, *, session: Any
        ) -> tuple[pd.DataFrame, bool]:
            if year == 2021:
                return lcd._empty_hourly_frame(), True
            return (
                pd.DataFrame(
                    {
                        "time": [pd.Timestamp(f"{year}-01-01")],
                        "tair_c": [10.0],
                        "dewpoint_c": [5.0],
                        "pressure_hpa": [1000.0],
                    }
                ),
                False,
            )

        monkeypatch.setattr(lcd, "fetch_station_year", fake_fetch_station_year)
        frame, gaps = lcd._fetch_station_series(
            cast("Any", None), "AAA", [2020, 2021, 2022]
        )
        assert gaps == {2021}
        assert len(frame) == 2

    def test_all_years_gapped_returns_empty_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            lcd,
            "fetch_station_year",
            lambda *_a, **_k: (lcd._empty_hourly_frame(), True),
        )
        frame, gaps = lcd._fetch_station_series(cast("Any", None), "AAA", [2020])
        assert frame.empty
        assert gaps == {2020}


class TestStationToHourly:
    def test_converts_units_and_computes_qair(self) -> None:
        station_df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2020-01-01")],
                "tair_c": [10.0],
                "dewpoint_c": [5.0],
                "pressure_hpa": [1000.0],
            }
        )
        result = lcd._station_to_hourly(7, station_df)
        assert list(result.columns) == ["location_id", "time", "Tair", "Qair", "PSurf"]
        assert (result["location_id"] == 7).all()
        assert result["Tair"].iloc[0] == pytest.approx(283.15)
        assert result["PSurf"].iloc[0] == pytest.approx(100000.0)
        assert 0 < result["Qair"].iloc[0] < 0.02

    def test_empty_input_returns_empty_typed_frame(self) -> None:
        result = lcd._station_to_hourly(7, lcd._empty_hourly_frame())
        assert result.empty
        assert list(result.columns) == ["location_id", "time", "Tair", "Qair", "PSurf"]

    def test_drops_rows_missing_pressure(self) -> None:
        station_df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2020-01-01")],
                "tair_c": [10.0],
                "dewpoint_c": [5.0],
                "pressure_hpa": [float("nan")],
            }
        )
        result = lcd._station_to_hourly(7, station_df)
        assert result.empty


class TestLoadStationMap:
    def test_missing_file_warns_once_and_returns_empty_typed_frame(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(lcd, "STATION_MAP_PATH", str(tmp_path / "missing.csv"))
        monkeypatch.setattr(lcd, "_STATION_MAP_WARNED", [False])
        with caplog.at_level("WARNING"):
            result = lcd._load_station_map()
        assert result.empty
        assert list(result.columns) == ["location_id", "lcd_id"]
        assert any("not found" in m for m in caplog.messages)

    def test_loads_existing_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "cities_lcd_stations.csv"
        path.write_text(
            "location_id,icao,lcd_id,dist_km,elev_m,archive_begin,tier\n"
            "0,NYC,USW00094728,8.0,27.0,1990-01-01,1\n"
        )
        monkeypatch.setattr(lcd, "STATION_MAP_PATH", str(path))
        result = lcd._load_station_map()
        assert list(result["location_id"]) == [0]
        assert list(result["lcd_id"]) == ["USW00094728"]


class TestFetchStationsBatch:
    def test_fetches_each_station_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_series(
            _session: Any, station_id: str, _years: Any
        ) -> tuple[pd.DataFrame, set[int]]:
            calls.append(station_id)
            return pd.DataFrame({"time": [pd.Timestamp("2020-01-01")]}), set()

        monkeypatch.setattr(lcd, "_fetch_station_series", fake_series)
        results = lcd._fetch_stations_batch(
            ["AAA", "BBB"],
            [2020],
            cast("Any", None),
            worker_count=2,
            city_shard_index=0,
        )
        assert set(results) == {"AAA", "BBB"}
        assert sorted(calls) == ["AAA", "BBB"]


class TestProcessLcd:
    @staticmethod
    def _shard_df() -> pd.DataFrame:
        return pd.DataFrame({"location_id": [1], "lat": [40.0], "lng": [-74.0]})

    @staticmethod
    def _station_map_one_station() -> pd.DataFrame:
        return pd.DataFrame({"location_id": [1], "lcd_id": ["AAA"]})

    @staticmethod
    def _empty_station_map() -> pd.DataFrame:
        return pd.DataFrame(columns=pd.Index(["location_id", "lcd_id"]))

    def test_no_cities_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas,
            "load_nldas_city_shard",
            lambda *_a: pd.DataFrame(columns=pd.Index(["location_id", "lat", "lng"])),
        )
        called: list[int] = []
        monkeypatch.setattr(
            lcd, "_fetch_stations_batch", lambda *_a, **_k: called.append(1)
        )
        with caplog.at_level("INFO"):
            lcd.process_lcd(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not called
        assert any("No cities found" in m for m in caplog.messages)

    def test_no_pending_years_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(lcd, "pending_years", lambda *_a, **_k: [])
        called: list[int] = []
        monkeypatch.setattr(
            lcd, "_fetch_stations_batch", lambda *_a, **_k: called.append(1)
        )
        with caplog.at_level("INFO"):
            lcd.process_lcd(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not called
        assert any("already present" in m for m in caplog.messages)

    def test_unmapped_city_is_warned_and_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(lcd, "_load_station_map", self._empty_station_map)
        called: list[int] = []
        monkeypatch.setattr(
            lcd, "_fetch_stations_batch", lambda *_a, **_k: called.append(1)
        )
        with caplog.at_level("WARNING"):
            lcd.process_lcd(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not called
        assert any("no LCD station mapped" in m for m in caplog.messages)

    def test_gapped_year_excluded_while_clean_year_still_written(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A gap in one year only blocks that year's write, not the whole shard.

        This is the key semantic difference from giovanni's whole-shard skip.
        """
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(lcd, "_load_station_map", self._station_map_one_station)

        station_df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")],
                "tair_c": [10.0, 11.0],
                "dewpoint_c": [5.0, 6.0],
                "pressure_hpa": [1000.0, 1000.0],
            }
        )
        monkeypatch.setattr(
            lcd,
            "_fetch_stations_batch",
            lambda *_a, **_k: {"AAA": (station_df, {2021})},
        )
        monkeypatch.setattr(
            lcd.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1, 1],
                    "date": [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")],
                    "wetbulb": [20.0, 21.0],
                    "wetbulb_avg": [19.0, 20.0],
                }
            ),
        )
        with caplog.at_level("WARNING"):
            lcd.process_lcd(2020, 2021, str(tmp_path), 0, 1, 4)

        import partition_io

        filesystem, base_path = lcd.resolve_filesystem(f"{tmp_path}/wetbulb_data_csv")
        assert partition_io.batch_exists(
            f"{tmp_path}/wetbulb_data_csv",
            2020,
            0,
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )
        assert not partition_io.batch_exists(
            f"{tmp_path}/wetbulb_data_csv",
            2021,
            0,
            0,
            file_prefix="wetbulb",
            filesystem=filesystem,
            base_path=base_path,
        )
        assert any("transient fetch failure" in m for m in caplog.messages)

    def test_shared_station_fetched_once_emitted_per_city(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        shard_df = pd.DataFrame(
            {
                "location_id": [1, 2],
                "lat": [40.0, 40.1],
                "lng": [-74.0, -74.1],
            }
        )
        monkeypatch.setattr(lcd.nldas, "load_nldas_city_shard", lambda *_a: shard_df)
        monkeypatch.setattr(
            lcd,
            "_load_station_map",
            lambda: pd.DataFrame(
                {"location_id": [1, 2], "lcd_id": ["SHARED", "SHARED"]}
            ),
        )

        batch_calls: list[list[str]] = []

        def fake_batch(
            station_ids: list[str],
            _years: Any,
            _session: Any,
            _workers: int,
            _shard: int,
        ) -> dict[str, tuple[pd.DataFrame, set[int]]]:
            batch_calls.append(list(station_ids))
            station_df = pd.DataFrame(
                {
                    "time": [pd.Timestamp("2020-01-01")],
                    "tair_c": [10.0],
                    "dewpoint_c": [5.0],
                    "pressure_hpa": [1000.0],
                }
            )
            return {"SHARED": (station_df, set())}

        monkeypatch.setattr(lcd, "_fetch_stations_batch", fake_batch)

        seen_location_ids: list[int] = []

        def fake_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
            seen_location_ids.extend(hourly_df["location_id"].tolist())
            return pd.DataFrame(
                {
                    "location_id": [1, 2],
                    "date": [pd.Timestamp("2020-01-01")] * 2,
                    "wetbulb": [20.0, 20.0],
                    "wetbulb_avg": [19.0, 19.0],
                }
            )

        monkeypatch.setattr(lcd.nldas, "compute_daily_wetbulb", fake_daily)

        lcd.process_lcd(2020, 2020, str(tmp_path), 0, 1, 4)

        assert batch_calls == [["SHARED"]]
        assert sorted(seen_location_ids) == [1, 2]

    def test_successful_run_writes_batches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(lcd, "_load_station_map", self._station_map_one_station)
        station_df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2020-01-01")],
                "tair_c": [10.0],
                "dewpoint_c": [5.0],
                "pressure_hpa": [1000.0],
            }
        )
        monkeypatch.setattr(
            lcd, "_fetch_stations_batch", lambda *_a, **_k: {"AAA": (station_df, set())}
        )
        monkeypatch.setattr(
            lcd.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                {
                    "location_id": [1],
                    "date": [pd.Timestamp("2020-01-01")],
                    "wetbulb": [20.0],
                    "wetbulb_avg": [19.0],
                }
            ),
        )
        write_calls: list[Any] = []
        monkeypatch.setattr(
            lcd,
            "write_pending_year_batches",
            lambda *args, **_k: write_calls.append(args),
        )
        lcd.process_lcd(2020, 2020, str(tmp_path), 0, 1, 4)
        assert len(write_calls) == 1
        written_frame = write_calls[0][0]
        assert (written_frame["source"] == "isd").all()

    def test_empty_daily_df_returns_early(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(lcd, "_load_station_map", self._station_map_one_station)
        station_df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2020-01-01")],
                "tair_c": [10.0],
                "dewpoint_c": [5.0],
                "pressure_hpa": [1000.0],
            }
        )
        monkeypatch.setattr(
            lcd, "_fetch_stations_batch", lambda *_a, **_k: {"AAA": (station_df, set())}
        )
        monkeypatch.setattr(
            lcd.nldas,
            "compute_daily_wetbulb",
            lambda _df: pd.DataFrame(
                columns=pd.Index(["location_id", "date", "wetbulb", "wetbulb_avg"])
            ),
        )
        write_calls: list[Any] = []
        monkeypatch.setattr(
            lcd,
            "write_pending_year_batches",
            lambda *args, **_k: write_calls.append(args),
        )
        lcd.process_lcd(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not write_calls


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(lcd.sys, "argv", ["lcd.py"])
        args = lcd._parse_args()
        assert args.start_year == lcd.nldas.NLDAS_START_YEAR
        assert args.end_year == lcd.nldas.NLDAS_END_YEAR
        assert args.out_dir == "."
        assert args.city_shard_index == 0
        assert args.city_shard_count == 1
        assert args.concurrency == lcd.LCD_DEFAULT_CONCURRENCY
        assert args.force is False

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            lcd.sys,
            "argv",
            ["lcd.py", "--city-shard-index", "2", "--city-shard-count", "5", "--force"],
        )
        args = lcd._parse_args()
        assert args.city_shard_index == 2
        assert args.city_shard_count == 5
        assert args.force is True


class TestMain:
    def test_normal_flow_calls_process_lcd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = argparse.Namespace(
            start_year=2000,
            end_year=2001,
            out_dir=".",
            city_shard_index=0,
            city_shard_count=1,
            concurrency=8,
            force=False,
            cities_csv="cities.csv",
        )
        monkeypatch.setattr(lcd, "_parse_args", lambda: args)
        monkeypatch.setattr(lcd, "load_dotenv", lambda **_k: None)
        called: list[Any] = []
        monkeypatch.setattr(lcd, "process_lcd", lambda **kwargs: called.append(kwargs))
        with pytest.raises(SystemExit) as exc_info:
            lcd.main()
        assert exc_info.value.code == 0
        assert called[0]["start_year"] == 2000

    def test_keyboard_interrupt_exits_130(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_interrupt() -> argparse.Namespace:
            raise KeyboardInterrupt

        monkeypatch.setattr(lcd, "load_dotenv", lambda **_k: None)
        monkeypatch.setattr(lcd, "_parse_args", raise_interrupt)
        with pytest.raises(SystemExit) as exc_info:
            lcd.main()
        assert exc_info.value.code == 130

    def test_handled_exception_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_value_error() -> argparse.Namespace:
            msg = "boom"
            raise ValueError(msg)

        monkeypatch.setattr(lcd, "load_dotenv", lambda **_k: None)
        monkeypatch.setattr(lcd, "_parse_args", raise_value_error)
        with pytest.raises(SystemExit) as exc_info:
            lcd.main()
        assert exc_info.value.code == 1
