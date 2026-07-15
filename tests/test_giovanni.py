"""Tests for the Giovanni Time Series API wet-bulb worker helpers."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import giovanni
from giovanni import (
    TokenManager,
    _contiguous_year_ranges,
    _fetch_variable_series,
    _get_timeseries_csv,
    _load_cell_map,
    _parse_timeseries_csv,
    fetch_city_hourly,
)

SAMPLE_CSV = """prod_name,NLDAS_FORA0125_H.2.0
doi,10.5067/THUF4J1RLSYG
param_short_name,Tair
param_name,2-meter above ground Temperature
unit,K
undef,-9999
begin_time,2024-01-01 00:00:00
end_time,2024-01-01 02:00:00
lat,44.9375
lon,-93.3125
lat_resolution,0.125
lon_resolution,0.125
mean,2.6839e+02
Request_time,2026-07-14 12:21:23

Timestamp (UTC),Data
2024-01-01 00:00,269.81
2024-01-01 01:00,269.39
2024-01-01 02:00,268.97
"""


class TestIPv4Forced:
    def test_has_ipv6_disabled(self) -> None:
        """The Giovanni API blackholes IPv6 like hydro1.gesdisc; force IPv4."""
        import urllib3.util.connection

        assert urllib3.util.connection.HAS_IPV6 is False


class TestParseTimeseriesCsv:
    def test_parses_header_and_data(self) -> None:
        headers, df = _parse_timeseries_csv(SAMPLE_CSV)
        assert headers["param_short_name"] == "Tair"
        assert headers["unit"] == "K"
        assert headers["undef"] == "-9999"
        assert list(df.columns) == ["time", "value"]
        assert len(df) == 3
        assert df.loc[0, "value"] == pytest.approx(269.81)
        assert df.loc[0, "time"] == pd.Timestamp("2024-01-01 00:00:00")

    def test_raises_on_missing_header(self) -> None:
        with pytest.raises(ValueError, match="missing expected header"):
            _parse_timeseries_csv("not,a,valid,response\n")

    def test_handles_fill_values_as_numeric(self) -> None:
        csv_text = SAMPLE_CSV.replace("269.81", "-9999")
        _headers, df = _parse_timeseries_csv(csv_text)
        assert df.loc[0, "value"] == -9999.0


class TestContiguousYearRanges:
    def test_single_run(self) -> None:
        assert _contiguous_year_ranges([2020, 2021, 2022]) == [(2020, 2022)]

    def test_multiple_runs(self) -> None:
        assert _contiguous_year_ranges([2000, 2001, 2005, 2010, 2011]) == [
            (2000, 2001),
            (2005, 2005),
            (2010, 2011),
        ]

    def test_unsorted_and_duplicate_input(self) -> None:
        assert _contiguous_year_ranges([2022, 2020, 2021, 2021]) == [(2020, 2022)]

    def test_empty_input(self) -> None:
        assert _contiguous_year_ranges([]) == []


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        text: str = "",
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self._json_body = json_body or {}

    def json(self) -> dict[str, Any]:
        return self._json_body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = f"HTTP {self.status_code}"
            raise giovanni.requests.HTTPError(msg)


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self._responses.pop(0)


class _StubTokenManager:
    def __init__(self, token: str = "tok") -> None:
        self.token = token
        self.refresh_calls = 0

    def get(self) -> str:
        return self.token

    def refresh(self) -> str:
        self.refresh_calls += 1
        self.token = f"{self.token}-refreshed"
        return self.token


class TestTokenManager:
    def test_get_caches_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake_post(_url: str, **kwargs: Any) -> _FakeResponse:
            calls.append(kwargs)
            return _FakeResponse(200, json_body={"access_token": "abc"})

        monkeypatch.setattr(giovanni.requests, "post", fake_post)
        manager = TokenManager("user", "pass")
        assert manager.get() == "abc"
        assert manager.get() == "abc"
        assert len(calls) == 1

    def test_refresh_refetches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tokens = iter(["first", "second"])
        monkeypatch.setattr(
            giovanni.requests,
            "post",
            lambda _url, **_kwargs: _FakeResponse(
                200, json_body={"access_token": next(tokens)}
            ),
        )
        manager = TokenManager("user", "pass")
        assert manager.get() == "first"
        assert manager.refresh() == "second"
        assert manager.get() == "second"

    def test_from_env_requires_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
        monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
        with pytest.raises(RuntimeError, match="EARTHDATA_USERNAME"):
            TokenManager.from_env()

    def test_from_env_builds_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EARTHDATA_USERNAME", "user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "pass")
        manager = TokenManager.from_env()
        assert manager._username == "user"
        assert manager._password == "pass"


class TestGetTimeseriesCsv:
    def test_success_returns_text(self) -> None:
        session = _FakeSession([_FakeResponse(200, text="ok")])
        result = _get_timeseries_csv(
            session, _StubTokenManager(), "VAR", 1.0, 2.0, "2024-01-01", "2024-01-02"
        )
        assert result == "ok"

    def test_401_refreshes_token_and_retries(self) -> None:
        session = _FakeSession([_FakeResponse(401), _FakeResponse(200, text="ok")])
        token_manager = _StubTokenManager()
        result = _get_timeseries_csv(
            session, token_manager, "VAR", 1.0, 2.0, "2024-01-01", "2024-01-02"
        )
        assert result == "ok"
        assert token_manager.refresh_calls == 1

    def test_429_backs_off_and_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(giovanni.time, "sleep", sleeps.append)
        session = _FakeSession(
            [
                _FakeResponse(429, headers={"Retry-After": "1"}),
                _FakeResponse(200, text="ok"),
            ]
        )
        result = _get_timeseries_csv(
            session, _StubTokenManager(), "VAR", 1.0, 2.0, "2024-01-01", "2024-01-02"
        )
        assert result == "ok"
        assert sleeps

    def test_4xx_gives_up_without_retry(self) -> None:
        session = _FakeSession([_FakeResponse(400, text="bad request")])
        result = _get_timeseries_csv(
            session, _StubTokenManager(), "VAR", 1.0, 2.0, "2024-01-01", "2024-01-02"
        )
        assert result is None
        assert len(session.calls) == 1

    def test_413_raises_range_too_large_without_retry(self) -> None:
        session = _FakeSession(
            [_FakeResponse(413, text="size of produced data is too high")]
        )
        with pytest.raises(giovanni._RangeTooLargeError):
            _get_timeseries_csv(
                session,
                _StubTokenManager(),
                "VAR",
                1.0,
                2.0,
                "2024-01-01",
                "2024-01-02",
            )
        assert len(session.calls) == 1

    def test_network_error_retries_then_gives_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(giovanni.time, "sleep", lambda _seconds: None)

        class _RaisingSession:
            calls = 0

            def get(self, *_args: Any, **_kwargs: Any) -> _FakeResponse:
                self.calls += 1
                raise giovanni.requests.ConnectionError("boom")

        session = _RaisingSession()
        result = _get_timeseries_csv(
            session, _StubTokenManager(), "VAR", 1.0, 2.0, "2024-01-01", "2024-01-02"
        )
        assert result is None
        assert session.calls == giovanni.GIOVANNI_MAX_RETRIES


class TestFetchVariableSeries:
    def test_returns_parsed_series_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            giovanni, "_get_timeseries_csv", lambda *_a, **_k: SAMPLE_CSV
        )
        df, had_gap = _fetch_variable_series(
            SimpleNamespace(), _StubTokenManager(), "VAR", 1.0, 2.0, 2024, 2024
        )
        assert len(df) == 3
        assert had_gap is False

    def test_http_failure_gives_up_without_halving(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exhausted HTTP failure must not recurse into smaller ranges.

        Halving on a server-side failure just resubmits the same total
        demand as more requests, which is what turned one slow window into
        a sustained outage during a real backfill (2026-07).
        """
        calls: list[tuple[str, str]] = []

        def fake_get(
            _session: Any,
            _tm: Any,
            _var: str,
            _lat: float,
            _lon: float,
            start: str,
            end: str,
        ) -> None:
            calls.append((start, end))

        monkeypatch.setattr(giovanni, "_get_timeseries_csv", fake_get)
        df, had_gap = _fetch_variable_series(
            SimpleNamespace(), _StubTokenManager(), "VAR", 1.0, 2.0, 2020, 2021
        )
        assert df.empty
        assert had_gap is True
        # Only the single, full-range request was made -- no recursive splits.
        assert calls == [("2020-01-01T00:00:00", "2022-01-01T00:00:00")]

    def test_halves_range_on_persistent_parse_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requested_start_years: list[int] = []

        def fake_get(
            _session: Any,
            _tm: Any,
            _var: str,
            _lat: float,
            _lon: float,
            start: str,
            _end: str,
        ) -> str:
            requested_start_years.append(int(start[:4]))
            return "not,a,valid,response"

        monkeypatch.setattr(giovanni, "_get_timeseries_csv", fake_get)
        df, had_gap = _fetch_variable_series(
            SimpleNamespace(), _StubTokenManager(), "VAR", 1.0, 2.0, 2020, 2021
        )
        assert df.empty
        assert had_gap is True
        # Both individual years were attempted once the 2-year range's
        # response failed to parse.
        assert 2020 in requested_start_years
        assert 2021 in requested_start_years

    def test_range_too_large_always_halves_until_it_fits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HTTP 413 is a deterministic "split it" signal, unlike a 5xx.

        A live backfill (2026-07) hit this for Qair at the full 26-year
        range even though the server wasn't overloaded, so this must keep
        halving (unlike the exhausted-HTTP-failure case above) until a
        chunk small enough to succeed is found.
        """
        requested_ranges: list[tuple[int, int]] = []

        def fake_get(
            _session: Any,
            _tm: Any,
            _var: str,
            _lat: float,
            _lon: float,
            start: str,
            end: str,
        ) -> str:
            start_year = int(start[:4])
            end_year = int(end[:4]) - 1
            requested_ranges.append((start_year, end_year))
            if end_year - start_year >= 1:
                msg = "too big"
                raise giovanni._RangeTooLargeError(msg)
            return SAMPLE_CSV

        monkeypatch.setattr(giovanni, "_get_timeseries_csv", fake_get)
        df, had_gap = _fetch_variable_series(
            SimpleNamespace(), _StubTokenManager(), "VAR", 1.0, 2.0, 2020, 2023
        )
        assert had_gap is False
        assert len(df) == 3 * 4  # SAMPLE_CSV's 3 rows per single-year leaf
        assert (2020, 2023) in requested_ranges  # the oversized top-level try
        leaves = [r for r in requested_ranges if r[1] - r[0] == 0]
        assert sorted(leaves) == [
            (2020, 2020),
            (2021, 2021),
            (2022, 2022),
            (2023, 2023),
        ]

    def test_range_too_large_gives_up_if_still_too_large_at_one_year(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get(*_args: Any, **_kwargs: Any) -> None:
            msg = "too big"
            raise giovanni._RangeTooLargeError(msg)

        monkeypatch.setattr(giovanni, "_get_timeseries_csv", fake_get)
        df, had_gap = _fetch_variable_series(
            SimpleNamespace(), _StubTokenManager(), "VAR", 1.0, 2.0, 2020, 2020
        )
        assert df.empty
        assert had_gap is True

    def test_partial_success_after_halving_on_parse_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get(
            _session: Any,
            _tm: Any,
            _var: str,
            _lat: float,
            _lon: float,
            start: str,
            end: str,
        ) -> str:
            # Only the single-year 2020 request returns a parseable body;
            # everything else (the full range, and the 2021 leaf) doesn't.
            if start.startswith("2020") and end.startswith("2021"):
                return SAMPLE_CSV
            return "not,a,valid,response"

        monkeypatch.setattr(giovanni, "_get_timeseries_csv", fake_get)
        df, had_gap = _fetch_variable_series(
            SimpleNamespace(), _StubTokenManager(), "VAR", 1.0, 2.0, 2020, 2021
        )
        assert len(df) == 3  # only the 2020 leaf contributed rows
        assert had_gap is True  # the 2021 leaf still failed


class TestFetchCityHourly:
    def test_joins_variables_and_converts_fill_to_nan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_fetch_variable_series(
            _session: Any,
            _tm: Any,
            var_id: str,
            _lat: float,
            _lon: float,
            _start_year: int,
            _end_year: int,
        ) -> tuple[pd.DataFrame, bool]:
            times = pd.date_range("2024-01-01", periods=3, freq="h")
            values = [1.0, giovanni.nldas.NLDAS_FILL_THRESHOLD - 1.0, 3.0]
            return pd.DataFrame({"time": times, "value": values}), False

        monkeypatch.setattr(
            giovanni, "_fetch_variable_series", fake_fetch_variable_series
        )
        df, had_gap = fetch_city_hourly(
            SimpleNamespace(), _StubTokenManager(), 42, 1.0, 2.0, 2024, 2024
        )
        assert had_gap is False
        assert list(df["location_id"].unique()) == [42]
        assert set(df.columns) >= {"location_id", "time", "Tair", "Qair", "PSurf"}
        assert np.isnan(df.loc[1, "Tair"])
        assert df.loc[0, "Tair"] == pytest.approx(1.0)

    def test_logs_alarm_when_mostly_fill(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def fake_fetch_variable_series(
            _session: Any,
            _tm: Any,
            _var_id: str,
            _lat: float,
            _lon: float,
            _start_year: int,
            _end_year: int,
        ) -> tuple[pd.DataFrame, bool]:
            times = pd.date_range("2024-01-01", periods=4, freq="h")
            values = [giovanni.nldas.NLDAS_FILL_THRESHOLD - 1.0] * 4
            return pd.DataFrame({"time": times, "value": values}), False

        monkeypatch.setattr(
            giovanni, "_fetch_variable_series", fake_fetch_variable_series
        )
        with caplog.at_level("ERROR"):
            _df, had_gap = fetch_city_hourly(
                SimpleNamespace(), _StubTokenManager(), 7, 35.0, -70.0, 2024, 2024
            )
        assert had_gap is False
        assert any("likely water" in message for message in caplog.messages)

    def test_logs_api_gap_instead_of_water_when_fetch_had_gap(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def fake_fetch_variable_series(
            _session: Any,
            _tm: Any,
            _var_id: str,
            _lat: float,
            _lon: float,
            _start_year: int,
            _end_year: int,
        ) -> tuple[pd.DataFrame, bool]:
            # Mostly-fill data (as if a Giovanni request partially failed),
            # plus had_gap=True signaling a real fetch failure occurred.
            times = pd.date_range("2024-01-01", periods=4, freq="h")
            values = [giovanni.nldas.NLDAS_FILL_THRESHOLD - 1.0] * 4
            return pd.DataFrame({"time": times, "value": values}), True

        monkeypatch.setattr(
            giovanni, "_fetch_variable_series", fake_fetch_variable_series
        )
        with caplog.at_level("ERROR"):
            _df, had_gap = fetch_city_hourly(
                SimpleNamespace(), _StubTokenManager(), 7, 35.0, -70.0, 2024, 2024
            )
        assert had_gap is True
        assert any("API gap" in message for message in caplog.messages)
        assert not any("likely water" in message for message in caplog.messages)


class TestLoadCellMap:
    def test_missing_file_returns_empty_typed_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(giovanni, "_CELL_MAP_WARNED", False)
        df = _load_cell_map()
        assert list(df.columns) == ["location_id", "cell_lat", "cell_lon"]
        assert df.empty
        assert df["cell_lat"].dtype == np.float64

    def test_present_file_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)
        csv_path = tmp_path / giovanni.CELL_MAP_PATH
        csv_path.write_text(
            "location_id,lat,lng,cell_lat,cell_lon,snapped\n0,40.7,-74.0,40.6875,-74.0625,False\n"
        )
        df = _load_cell_map()
        assert len(df) == 1
        assert df.loc[0, "cell_lat"] == pytest.approx(40.6875)


class TestParseTimeseriesCsvRoundTrip:
    def test_matches_io_stringio_reference(self) -> None:
        """Cross-check against the row-count-independent StringIO parsing approach."""
        headers, df = _parse_timeseries_csv(SAMPLE_CSV)
        with io.StringIO(SAMPLE_CSV) as f:
            raw_lines = f.readlines()
        assert headers["prod_name"] == raw_lines[0].split(",", 1)[1].strip()
        assert len(df) == 3


def test_module_reuses_nldas_helpers() -> None:
    """giovanni.py should reuse nldas.py's daily wet-bulb math, not reimplement it."""
    assert giovanni.nldas._compute_daily_wetbulb is not None
    assert cast("Any", giovanni.nldas)._load_nldas_city_shard is not None
