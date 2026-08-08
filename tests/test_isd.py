# Copyright (C) 2026 Kenneth Porter

"""Tests for the NOAA ISD Global Hourly wet-bulb worker (default wetbulb pipeline source)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import isd
import lcd

_HEADER = (
    '"STATION","DATE","SOURCE","LATITUDE","LONGITUDE","ELEVATION","NAME",'
    '"REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","TMP","DEW","SLP","MA1"'
)
SAMPLE_ISD_CSV = f"""{_HEADER}
"72503014732","2024-01-01T00:53:00","4","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-15","KNYC ","V020","+0050,5","+0020,5","10175,5","10176,5,10160,5"
"72503014732","2024-01-01T01:53:00","4","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-16","KNYC ","V020","+0054,5","+0024,5","10176,5","10178,5,10161,5"
"72503014732","2024-01-01T01:53:00","4","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-15","KNYC ","V020","+0055,5","+0025,5","10177,5","10178,5,10162,5"
"72503014732","2024-01-01T06:00:00","4","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","SOD  ","KNYC ","V020","+9999,9","+9999,9","99999,9",""
"""

NO_HOURLY_CSV = f"""{_HEADER}
"72503014732","2024-01-01T06:00:00","4","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","SOD  ","KNYC ","V020","+9999,9","+9999,9","99999,9",""
"72503014732","2024-01-31T23:59:00","4","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","SOM  ","KNYC ","V020","+9999,9","+9999,9","99999,9",""
"""

NO_STATION_PRESSURE_CSV = f"""{_HEADER}
"72503014732","2000-01-01T00:51:00","4","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-15","KNYC ","V020","+0050,5","+0020,5","10175,5","10176,5,99999,9"
"""

NO_MA1_COLUMN_CSV = (
    '"STATION","DATE","SOURCE","LATITUDE","LONGITUDE","ELEVATION","NAME",'
    '"REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","TMP","DEW","SLP"\n'
    '"72503014732","2000-01-01T00:51:00","4","40.7789","-73.9692","39.6",'
    '"NEW YORK CENTRAL PARK, NY US","FM-15","KNYC ","V020","+0050,5",'
    '"+0020,5","10175,5"\n'
)

REJECTED_QC_CSV = f"""{_HEADER}
"72503014732","2024-01-01T00:53:00","4","40.7789","-73.9692","39.6","NEW YORK CENTRAL PARK, NY US","FM-15","KNYC ","V020","+0500,6","+0020,5","10175,5","10176,5,10160,5"
"""


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = f"HTTP {self.status_code}"
            raise isd.requests.HTTPError(msg)


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


class TestColumnOrMissing:
    def test_returns_existing_column(self) -> None:
        df = pd.DataFrame({"A": [1, 2]})
        result = isd._column_or_missing(df, "A")
        assert list(result) == [1, 2]

    def test_returns_all_missing_series_when_absent(self) -> None:
        df = pd.DataFrame({"A": [1, 2]})
        result = isd._column_or_missing(df, "MA1")
        assert len(result) == 2
        assert result.isna().all()


class TestParseIsdField:
    def test_scales_value_by_tenth(self) -> None:
        result = isd._parse_isd_field(pd.Series(["+0050,5"]), missing="9999")
        assert result.iloc[0] == pytest.approx(5.0)

    def test_sign_prefixed_sentinel_is_rejected(self) -> None:
        """The TMP/DEW sentinel carries a '+' sign that a literal string match would miss."""
        result = isd._parse_isd_field(pd.Series(["+9999,9"]), missing="9999")
        assert pd.isna(result.iloc[0])

    def test_unsigned_pressure_sentinel_is_rejected(self) -> None:
        result = isd._parse_isd_field(pd.Series(["99999,9"]), missing="99999")
        assert pd.isna(result.iloc[0])

    def test_reject_qc_codes_drop_the_value(self) -> None:
        for code in ("2", "3", "6", "7"):
            result = isd._parse_isd_field(pd.Series([f"+0500,{code}"]), missing="9999")
            assert pd.isna(result.iloc[0]), f"qc={code} should be rejected"

    def test_pass_qc_codes_keep_the_value(self) -> None:
        for code in ("1", "5", "9"):
            result = isd._parse_isd_field(pd.Series([f"+0050,{code}"]), missing="9999")
            assert result.iloc[0] == pytest.approx(5.0), f"qc={code} should pass"


class TestGetWithRetries:
    def test_returns_404_without_retrying(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(404)]))
        response = isd._get_with_retries(
            session, "http://x", station_id="AAA", year=2020
        )
        assert response is not None
        assert response.status_code == 404

    def test_returns_none_after_exhausting_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(isd.time, "sleep", lambda _s: None)
        session = cast(
            "Any",
            _FakeSession([isd.requests.RequestException("boom")] * isd.ISD_MAX_RETRIES),
        )
        response = isd._get_with_retries(
            session, "http://x", station_id="AAA", year=2020
        )
        assert response is None

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(isd.time, "sleep", lambda _s: None)
        session = cast(
            "Any",
            _FakeSession(
                [isd.requests.RequestException("boom"), _FakeResponse(200, "ok")]
            ),
        )
        response = isd._get_with_retries(
            session, "http://x", station_id="AAA", year=2020
        )
        assert response is not None
        assert response.status_code == 200
        assert response.text == "ok"


class TestFetchStationYear:
    def test_parses_hourly_rows_and_fm15_wins_dedup(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(200, SAMPLE_ISD_CSV)]))
        frame, gap = isd.fetch_station_year(["72503014732"], 2024, session=session)
        assert gap is False
        assert len(frame) == 2
        assert set(frame["tair_c"]) == {5.0, 5.5}
        assert 5.4 not in set(frame["tair_c"])

    def test_returns_empty_not_gap_when_every_candidate_404s(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(404), _FakeResponse(404)]))
        frame, gap = isd.fetch_station_year(
            ["72503014732", "72503099999"], 1990, session=session
        )
        assert frame.empty
        assert gap is False
        assert len(cast("_FakeSession", session).calls) == 2

    def test_falls_through_404_to_second_candidate(self) -> None:
        session = cast(
            "Any",
            _FakeSession([_FakeResponse(404), _FakeResponse(200, SAMPLE_ISD_CSV)]),
        )
        frame, gap = isd.fetch_station_year(
            ["72503099999", "72503014732"], 2024, session=session
        )
        assert gap is False
        assert len(frame) == 2

    def test_falls_through_empty_hourly_candidate_to_next(self) -> None:
        """The Orlando bug: a candidate's file exists but has zero hourly rows."""
        session = cast(
            "Any",
            _FakeSession(
                [_FakeResponse(200, NO_HOURLY_CSV), _FakeResponse(200, SAMPLE_ISD_CSV)]
            ),
        )
        frame, gap = isd.fetch_station_year(
            ["72503099999", "72503014732"], 2024, session=session
        )
        assert gap is False
        assert len(frame) == 2

    def test_returns_gap_true_on_exhausted_retries_without_trying_next_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(isd.time, "sleep", lambda _s: None)
        session = cast(
            "Any",
            _FakeSession([isd.requests.RequestException("boom")] * isd.ISD_MAX_RETRIES),
        )
        frame, gap = isd.fetch_station_year(
            ["72503014732", "72503099999"], 2024, session=session
        )
        assert frame.empty
        assert gap is True
        assert len(cast("_FakeSession", session).calls) == isd.ISD_MAX_RETRIES

    def test_pressure_fallback_uses_sea_level_when_station_pressure_missing(
        self,
    ) -> None:
        session = cast(
            "Any", _FakeSession([_FakeResponse(200, NO_STATION_PRESSURE_CSV)])
        )
        frame, gap = isd.fetch_station_year(
            ["72503014732"], 2000, lon=-73.9692, session=session
        )
        assert gap is False
        assert len(frame) == 1
        assert frame["pressure_hpa"].iloc[0] < 1017.5
        assert frame["pressure_hpa"].iloc[0] > 1010.0

    def test_missing_ma1_column_falls_back_to_sea_level_pressure(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(200, NO_MA1_COLUMN_CSV)]))
        frame, gap = isd.fetch_station_year(
            ["72503014732"], 2000, lon=-73.9692, session=session
        )
        assert gap is False
        assert len(frame) == 1
        assert frame["pressure_hpa"].iloc[0] < 1017.5
        assert frame["pressure_hpa"].iloc[0] > 1010.0

    def test_rejected_qc_code_drops_the_row(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(200, REJECTED_QC_CSV)]))
        frame, gap = isd.fetch_station_year(
            ["72503014732"], 2024, lon=-73.9692, session=session
        )
        assert gap is False
        assert frame.empty

    def test_local_time_shift_uses_explicit_lon(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(200, SAMPLE_ISD_CSV)]))
        frame, _gap = isd.fetch_station_year(
            ["72503014732"], 2024, lon=-75.0, session=session
        )
        first = pd.Timestamp(frame.sort_values("time")["time"].iloc[0])
        assert first == pd.Timestamp("2023-12-31T19:53:00")

    def test_local_time_shift_falls_back_to_file_longitude(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(200, SAMPLE_ISD_CSV)]))
        frame, _gap = isd.fetch_station_year(["72503014732"], 2024, session=session)
        first = pd.Timestamp(frame.sort_values("time")["time"].iloc[0])
        assert first == pd.Timestamp("2023-12-31T19:53:00")


class TestFetchStationSeries:
    def test_aggregates_gapped_years_and_concatenates_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_fetch_station_year(
            candidate_ids: list[str],
            year: int,
            *,
            lon: Any,
            utc_offset_hours: Any = None,
            session: Any,
        ) -> tuple[pd.DataFrame, bool]:
            if year == 2021:
                return isd._empty_hourly_frame(), True
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

        monkeypatch.setattr(isd, "fetch_station_year", fake_fetch_station_year)
        frame, gaps = isd._fetch_station_series(
            cast("Any", None), ["AAA"], [2020, 2021, 2022], None
        )
        assert gaps == {2021}
        assert len(frame) == 2

    def test_all_years_gapped_returns_empty_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            isd,
            "fetch_station_year",
            lambda *_a, **_k: (isd._empty_hourly_frame(), True),
        )
        frame, gaps = isd._fetch_station_series(
            cast("Any", None), ["AAA"], [2020], None
        )
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
        result = isd._station_to_hourly(7, station_df)
        assert list(result.columns) == ["location_id", "time", "Tair", "Qair", "PSurf"]
        assert (result["location_id"] == 7).all()
        assert result["Tair"].iloc[0] == pytest.approx(283.15)
        assert result["PSurf"].iloc[0] == pytest.approx(100000.0)
        assert 0 < result["Qair"].iloc[0] < 0.02

    def test_empty_input_returns_empty_typed_frame(self) -> None:
        result = isd._station_to_hourly(7, isd._empty_hourly_frame())
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
        result = isd._station_to_hourly(7, station_df)
        assert result.empty


class TestLoadStationMap:
    def test_missing_file_warns_once_and_returns_empty_typed_frame(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(isd, "STATION_MAP_PATH", str(tmp_path / "missing.csv"))
        monkeypatch.setattr(isd, "_STATION_MAP_WARNED", [False])
        with caplog.at_level("WARNING"):
            result = isd._load_station_map()
        assert result.empty
        assert list(result.columns) == [
            "location_id",
            "isd_ids",
            "lon",
            "utc_offset_hours",
        ]
        assert any("not found" in m for m in caplog.messages)

    def test_loads_existing_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "cities_isd_stations.csv"
        path.write_text(
            "location_id,wban,isd_ids,lon,dist_km,elev_m\n"
            "0,94728,72503014732|72505399999,-73.9692,8.0,27.0\n"
        )
        monkeypatch.setattr(isd, "STATION_MAP_PATH", str(path))
        result = isd._load_station_map()
        assert list(result["location_id"]) == [0]
        assert list(result["isd_ids"]) == ["72503014732|72505399999"]
        assert result["utc_offset_hours"].isna().all()

    def test_loads_utc_offset_hours_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "cities_eu_isd_stations.csv"
        path.write_text(
            "location_id,usaf,wban,isd_ids,lon,dist_km,elev_m,utc_offset_hours\n"
            "1000,03772,99999,0377299999,-0.4614,5.0,25.0,0\n"
        )
        monkeypatch.setattr(isd, "STATION_MAP_PATH", str(path))
        result = isd._load_station_map()
        assert list(result["utc_offset_hours"]) == [0.0]


class TestFetchStationsBatch:
    def test_fetches_each_station_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def fake_series(
            _session: Any,
            candidate_ids: list[str],
            _years: Any,
            _lon: Any,
            _utc_offset_hours: Any = None,
        ) -> tuple[pd.DataFrame, set[int]]:
            calls.append(candidate_ids)
            return pd.DataFrame({"time": [pd.Timestamp("2020-01-01")]}), set()

        monkeypatch.setattr(isd, "_fetch_station_series", fake_series)
        results = isd._fetch_stations_batch(
            ["AAA|BBB", "CCC"],
            {"AAA|BBB": -74.0, "CCC": -75.0},
            {"AAA|BBB": None, "CCC": None},
            [2020],
            cast("Any", None),
            worker_count=2,
            city_shard_index=0,
        )
        assert set(results) == {"AAA|BBB", "CCC"}
        assert sorted(calls) == [["AAA", "BBB"], ["CCC"]]


class TestProcessIsd:
    @staticmethod
    def _shard_df() -> pd.DataFrame:
        return pd.DataFrame({"location_id": [1], "lat": [40.0], "lng": [-74.0]})

    @staticmethod
    def _station_map_one_station(*_args: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": [1],
                "isd_ids": ["AAA"],
                "lon": [-74.0],
                "utc_offset_hours": [None],
            }
        )

    @staticmethod
    def _empty_station_map(*_args: Any) -> pd.DataFrame:
        return pd.DataFrame(
            columns=pd.Index(["location_id", "isd_ids", "lon", "utc_offset_hours"])
        )

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
            isd, "_fetch_stations_batch", lambda *_a, **_k: called.append(1)
        )
        with caplog.at_level("INFO"):
            isd.process_isd(2020, 2020, str(tmp_path), 0, 1, 4)
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
        monkeypatch.setattr(isd, "pending_years", lambda *_a, **_k: [])
        called: list[int] = []
        monkeypatch.setattr(
            isd, "_fetch_stations_batch", lambda *_a, **_k: called.append(1)
        )
        with caplog.at_level("INFO"):
            isd.process_isd(2020, 2020, str(tmp_path), 0, 1, 4)
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
        monkeypatch.setattr(isd, "_load_station_map", self._empty_station_map)
        called: list[int] = []
        monkeypatch.setattr(
            isd, "_fetch_stations_batch", lambda *_a, **_k: called.append(1)
        )
        with caplog.at_level("WARNING"):
            isd.process_isd(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not called
        assert any("no ISD station mapped" in m for m in caplog.messages)

    def test_gapped_year_excluded_while_clean_year_still_written(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A gap in one year only blocks that year's write, not the whole shard."""
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(isd, "_load_station_map", self._station_map_one_station)

        station_df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")],
                "tair_c": [10.0, 11.0],
                "dewpoint_c": [5.0, 6.0],
                "pressure_hpa": [1000.0, 1000.0],
            }
        )
        monkeypatch.setattr(
            isd,
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
            isd.process_isd(2020, 2021, str(tmp_path), 0, 1, 4)

        import partition_io

        filesystem, base_path = isd.resolve_filesystem(f"{tmp_path}/wetbulb_data_csv")
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
            isd,
            "_load_station_map",
            lambda *_args: pd.DataFrame(
                {
                    "location_id": [1, 2],
                    "isd_ids": ["SHARED", "SHARED"],
                    "lon": [-74.0, -74.0],
                    "utc_offset_hours": [None, None],
                }
            ),
        )

        batch_calls: list[list[str]] = []

        def fake_batch(
            station_keys: list[str],
            _lon_by_key: Any,
            _offset_by_key: Any,
            _years: Any,
            _session: Any,
            _workers: int,
            _shard: int,
        ) -> dict[str, tuple[pd.DataFrame, set[int]]]:
            batch_calls.append(list(station_keys))
            station_df = pd.DataFrame(
                {
                    "time": [pd.Timestamp("2020-01-01")],
                    "tair_c": [10.0],
                    "dewpoint_c": [5.0],
                    "pressure_hpa": [1000.0],
                }
            )
            return {"SHARED": (station_df, set())}

        monkeypatch.setattr(isd, "_fetch_stations_batch", fake_batch)

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

        isd.process_isd(2020, 2020, str(tmp_path), 0, 1, 4)

        assert batch_calls == [["SHARED"]]
        assert sorted(seen_location_ids) == [1, 2]

    def test_successful_run_writes_batches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(isd, "_load_station_map", self._station_map_one_station)
        station_df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2020-01-01")],
                "tair_c": [10.0],
                "dewpoint_c": [5.0],
                "pressure_hpa": [1000.0],
            }
        )
        monkeypatch.setattr(
            isd, "_fetch_stations_batch", lambda *_a, **_k: {"AAA": (station_df, set())}
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
            isd,
            "write_pending_year_batches",
            lambda *args, **_k: write_calls.append(args),
        )
        isd.process_isd(2020, 2020, str(tmp_path), 0, 1, 4)
        assert len(write_calls) == 1
        written_frame = write_calls[0][0]
        assert (written_frame["source"] == "isd").all()

    def test_empty_daily_df_returns_early(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(isd, "_load_station_map", self._station_map_one_station)
        station_df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2020-01-01")],
                "tair_c": [10.0],
                "dewpoint_c": [5.0],
                "pressure_hpa": [1000.0],
            }
        )
        monkeypatch.setattr(
            isd, "_fetch_stations_batch", lambda *_a, **_k: {"AAA": (station_df, set())}
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
            isd,
            "write_pending_year_batches",
            lambda *args, **_k: write_calls.append(args),
        )
        isd.process_isd(2020, 2020, str(tmp_path), 0, 1, 4)
        assert not write_calls


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(isd.sys, "argv", ["isd.py"])
        args = isd._parse_args()
        assert args.start_year == lcd.nldas.NLDAS_START_YEAR
        assert args.end_year == lcd.nldas.NLDAS_END_YEAR
        assert args.out_dir == "."
        assert args.city_shard_index == 0
        assert args.city_shard_count == 1
        assert args.concurrency == isd.ISD_DEFAULT_CONCURRENCY
        assert args.force is False

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            isd.sys,
            "argv",
            ["isd.py", "--city-shard-index", "2", "--city-shard-count", "5", "--force"],
        )
        args = isd._parse_args()
        assert args.city_shard_index == 2
        assert args.city_shard_count == 5
        assert args.force is True


class TestMain:
    def test_normal_flow_calls_process_isd(
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
            station_map_csv="cities_isd_stations.csv",
        )
        monkeypatch.setattr(isd, "_parse_args", lambda: args)
        monkeypatch.setattr(isd, "load_dotenv", lambda **_k: None)
        called: list[Any] = []
        monkeypatch.setattr(isd, "process_isd", lambda **kwargs: called.append(kwargs))
        with pytest.raises(SystemExit) as exc_info:
            isd.main()
        assert exc_info.value.code == 0
        assert called[0]["start_year"] == 2000

    def test_keyboard_interrupt_exits_130(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_interrupt() -> argparse.Namespace:
            raise KeyboardInterrupt

        monkeypatch.setattr(isd, "load_dotenv", lambda **_k: None)
        monkeypatch.setattr(isd, "_parse_args", raise_interrupt)
        with pytest.raises(SystemExit) as exc_info:
            isd.main()
        assert exc_info.value.code == 130

    def test_handled_exception_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_value_error() -> argparse.Namespace:
            msg = "boom"
            raise ValueError(msg)

        monkeypatch.setattr(isd, "load_dotenv", lambda **_k: None)
        monkeypatch.setattr(isd, "_parse_args", raise_value_error)
        with pytest.raises(SystemExit) as exc_info:
            isd.main()
        assert exc_info.value.code == 1
