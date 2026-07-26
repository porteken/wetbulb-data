from __future__ import annotations

import io
import sys
from typing import Any

import pandas as pd
import pytest
import requests

import eccc
import lcd

HOURLY_HEADER = "Date/Time (LST),Temp (°C),Dew Point Temp (°C),Stn Press (kPa)"


def _hourly_csv(month: int) -> str:
    return f"{HOURLY_HEADER}\n2025-{month:02d}-01 00:00,20,10,100\n"


class _FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= requests.codes.bad_request:
            raise requests.HTTPError(str(self.status_code))


class TestParseEcccHourly:
    def test_normalizes_units_flags_and_lst(self) -> None:
        text = """Date/Time (LST),Temp (°C),Temp Flag,Dew Point Temp (°C),Dew Point Temp Flag,Stn Press (kPa),Stn Press Flag
2025-07-01 00:00,25.0,,20.0,,99.5,
2025-07-01 01:00,99.0,X,19.0,,99.4,
2025-07-01 02:00,24.0,,18.0,,99.3,M
"""
        result = eccc.parse_eccc_hourly(text)

        assert len(result) == 1
        assert result.iloc[0]["tair_c"] == pytest.approx(25.0)
        assert result.iloc[0]["pressure_hpa"] == pytest.approx(995.0)
        assert result.iloc[0]["time"] == pd.Timestamp("2025-07-01 00:00")

    def test_derives_station_pressure_from_sea_level(self) -> None:
        text = """Date/Time (LST),Temp (°C),Dew Point Temp (°C),Stn Press (kPa),Sea Level Press (kPa)
2025-01-01 00:00,10.0,5.0,,101.3
"""
        result = eccc.parse_eccc_hourly(text, elevation_m=100.0)

        assert len(result) == 1
        assert 990.0 < result.iloc[0]["pressure_hpa"] < 1013.0

    def test_infers_elevation_from_the_csv_when_not_supplied(self) -> None:
        text = """Date/Time (LST),Elevation (m),Temp (°C),Dew Point Temp (°C),Stn Press (kPa),Sea Level Press (kPa)
2025-01-01 00:00,100,10.0,5.0,,101.3
"""
        result = eccc.parse_eccc_hourly(text)

        assert len(result) == 1
        assert result.iloc[0]["pressure_hpa"] == pytest.approx(
            eccc.parse_eccc_hourly(text, elevation_m=100.0).iloc[0]["pressure_hpa"]
        )

    def test_returns_empty_for_blank_text(self) -> None:
        assert eccc.parse_eccc_hourly("   \n").empty

    def test_returns_empty_when_required_columns_are_absent(self) -> None:
        assert eccc.parse_eccc_hourly("Date/Time (LST),Wind Spd\n2025-01-01,4\n").empty

    def test_drops_duplicate_timestamps_and_sorts(self) -> None:
        text = (
            f"{HOURLY_HEADER}\n"
            "2025-07-01 01:00,21,11,100\n"
            "2025-07-01 00:00,20,10,100\n"
            "2025-07-01 00:00,99,10,100\n"
        )
        result = eccc.parse_eccc_hourly(text)

        assert len(result) == 2
        assert result["time"].is_monotonic_increasing
        assert result.iloc[0]["tair_c"] == pytest.approx(20.0)


class TestDownloadEcccMonth:
    def test_returns_decoded_text_on_success(self) -> None:
        session = _make_session([_FakeResponse(200, b"\xef\xbb\xbfok")])

        text, gap = eccc.download_eccc_month(1, 2025, 1, session=session)

        assert text == "ok"
        assert not gap

    def test_treats_404_as_a_permanent_absence(self) -> None:
        session = _make_session([_FakeResponse(404)])

        text, gap = eccc.download_eccc_month(1, 2025, 1, session=session)

        assert text == ""
        assert not gap

    def test_reports_a_gap_after_exhausting_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(eccc.time, "sleep", lambda _seconds: None)
        session = _make_session(
            [requests.ConnectionError("boom")] * eccc.ECCC_MAX_RETRIES
        )

        text, gap = eccc.download_eccc_month(1, 2025, 1, session=session)

        assert text is None
        assert gap
        assert len(session.calls) == eccc.ECCC_MAX_RETRIES

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eccc.time, "sleep", lambda _seconds: None)
        session = _make_session(
            [requests.ConnectionError("boom"), _FakeResponse(200, b"fine")]
        )

        text, gap = eccc.download_eccc_month(1, 2025, 1, session=session)

        assert text == "fine"
        assert not gap


def _make_session(outcomes: list[Any]) -> Any:
    class _Session:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._outcomes = list(outcomes)

        def get(self, url: str, **kwargs: Any) -> _FakeResponse:
            self.calls.append({"url": url, **kwargs})
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    return _Session()


class TestFetchStationYear:
    def test_falls_through_empty_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(
            station_id: int, _year: int, month: int, **_kwargs: object
        ) -> tuple[str, bool]:
            if station_id == 1:
                return "", False
            return _hourly_csv(month), False

        monkeypatch.setattr(eccc, "download_eccc_month", fake_download)
        frame, gap = eccc.fetch_station_year([1, 2], 2025)

        assert not gap
        assert len(frame) == 12

    def test_propagates_a_transient_gap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eccc, "download_eccc_month", lambda *_a, **_k: (None, True))
        frame, gap = eccc.fetch_station_year([1, 2], 2025)

        assert gap
        assert frame.empty

    def test_returns_empty_when_no_candidate_has_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(eccc, "download_eccc_month", lambda *_a, **_k: ("", False))
        frame, gap = eccc.fetch_station_year([1, 2], 2025)

        assert not gap
        assert frame.empty


class TestFetchStationSeries:
    def test_collects_years_and_records_gaps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_year(
            _ids: list[int], year: int, **_kwargs: object
        ) -> tuple[pd.DataFrame, bool]:
            if year == 2021:
                return eccc._empty_hourly(), True
            return pd.read_csv(io.StringIO(_hourly_csv(1))), False

        monkeypatch.setattr(eccc, "fetch_station_year", fake_year)
        frame, gaps = eccc._fetch_station_series(
            [1], [2020, 2021, 2022], None, requests.Session()
        )

        assert gaps == {2021}
        assert len(frame) == 2

    def test_returns_empty_frame_when_every_year_is_gapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            eccc, "fetch_station_year", lambda *_a, **_k: (eccc._empty_hourly(), True)
        )
        frame, gaps = eccc._fetch_station_series(
            [1], [2020, 2021], None, requests.Session()
        )

        assert gaps == {2020, 2021}
        assert frame.empty


class TestLoadStationMap:
    def test_reads_the_expected_columns(self, tmp_path: Any) -> None:
        path = tmp_path / "map.csv"
        path.write_text(
            "location_id,eccc_station_ids,elevation_m,extra\n500,1|2,70.0,ignored\n"
        )

        frame = eccc._load_station_map(str(path))

        assert list(frame.columns) == [
            "location_id",
            "eccc_station_ids",
            "elevation_m",
        ]

    def test_raises_a_helpful_error_when_absent(self, tmp_path: Any) -> None:
        missing = str(tmp_path / "missing.csv")

        with pytest.raises(FileNotFoundError, match=r"make_eccc_station_map\.py"):
            eccc._load_station_map(missing)


class TestProcessEccc:
    def _shard_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"location_id": [500], "lat": [45.0], "lng": [-75.0]},
        )

    def _station_map(self, *_args: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": [500],
                "eccc_station_ids": ["1|2"],
                "elevation_m": [70.0],
            }
        )

    def _station_frame(self) -> pd.DataFrame:
        hours = pd.date_range("2020-06-01", periods=24, freq="h")
        return pd.DataFrame(
            {
                "time": hours,
                "tair_c": [20.0] * len(hours),
                "dewpoint_c": [10.0] * len(hours),
                "pressure_hpa": [1000.0] * len(hours),
            }
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
            eccc, "_fetch_station_series", lambda *_a, **_k: called.append(1)
        )

        with caplog.at_level("INFO"):
            eccc.process_eccc(2020, 2020, str(tmp_path), 0, 1, 2)

        assert not called

    def test_unmapped_city_returns_before_fetching(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(
            eccc,
            "_load_station_map",
            lambda _path: pd.DataFrame(
                {
                    "location_id": [500],
                    "eccc_station_ids": [None],
                    "elevation_m": [None],
                }
            ),
        )
        called: list[int] = []
        monkeypatch.setattr(
            eccc, "_fetch_station_series", lambda *_a, **_k: called.append(1)
        )

        eccc.process_eccc(2020, 2020, str(tmp_path), 0, 1, 2)

        assert not called

    def test_no_pending_years_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(eccc, "pending_years", lambda *_a, **_k: [])
        called: list[int] = []
        monkeypatch.setattr(
            eccc, "_fetch_station_series", lambda *_a, **_k: called.append(1)
        )

        with caplog.at_level("INFO"):
            eccc.process_eccc(2020, 2020, str(tmp_path), 0, 1, 2)

        assert not called
        assert any("already present" in message for message in caplog.messages)

    def test_writes_a_daily_shard_stamped_as_eccc(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(eccc, "_load_station_map", self._station_map)
        monkeypatch.setattr(
            eccc,
            "_fetch_station_series",
            lambda *_a, **_k: (self._station_frame(), set()),
        )
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(
            eccc,
            "write_pending_year_batches",
            lambda daily, *_a, **_k: written.append({"daily": daily}),
        )

        eccc.process_eccc(2020, 2020, str(tmp_path), 0, 1, 2)

        assert len(written) == 1
        assert set(written[0]["daily"]["source"]) == {"eccc"}
        assert written[0]["daily"]["location_id"].tolist() == [500]

    def test_gapped_year_is_not_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(
            lcd.nldas, "load_nldas_city_shard", lambda *_a: self._shard_df()
        )
        monkeypatch.setattr(eccc, "_load_station_map", self._station_map)
        monkeypatch.setattr(
            eccc,
            "_fetch_station_series",
            lambda *_a, **_k: (self._station_frame(), {2020}),
        )
        written: list[Any] = []
        monkeypatch.setattr(
            eccc,
            "write_pending_year_batches",
            lambda *args, **_k: written.append(args),
        )

        eccc.process_eccc(2020, 2020, str(tmp_path), 0, 1, 2)

        assert not written


class TestMain:
    def test_forwards_parsed_arguments(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        captured: list[tuple[Any, ...]] = []
        monkeypatch.setattr(
            eccc,
            "process_eccc",
            lambda *args, **kwargs: captured.append((args, kwargs)),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eccc.py",
                "--start-year",
                "2001",
                "--end-year",
                "2002",
                "--out-dir",
                str(tmp_path),
                "--city-shard-index",
                "1",
                "--city-shard-count",
                "3",
                "--concurrency",
                "7",
                "--force",
            ],
        )

        eccc.main()

        args, kwargs = captured[0]
        assert args[:2] == (2001, 2002)
        assert args[3:] == (1, 3, 7)
        assert kwargs["force"] is True
        assert kwargs["station_map_csv"] == eccc.ECCC_STATION_MAP

    def test_keyboard_interrupt_exits_130(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_interrupt(*_args: Any, **_kwargs: Any) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(eccc, "process_eccc", raise_interrupt)
        monkeypatch.setattr(sys, "argv", ["eccc.py"])

        with pytest.raises(SystemExit) as excinfo:
            eccc.main()

        assert excinfo.value.code == 130
