"""Tests for make_isd_station_map.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import make_isd_station_map as stationmap


class _FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeSession:
    """Returns queued responses (or raises) keyed by call order."""

    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self._responses = list(responses)
        self.requested_urls: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> _FakeResponse:
        self.requested_urls.append(url)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


HOURLY_ROW = '"72308513750","2005-01-01T00:53:00","4","FM-15",...'
DAILY_ONLY_ROW = '"72308513750","2005-01-01T00:00:00","4","SOD ",...'


def _history_row(
    usaf: str,
    wban: str,
    begin: str,
    end: str,
    lon: str = "-76.289",
) -> dict[str, str]:
    return {"USAF": usaf, "WBAN": wban, "BEGIN": begin, "END": end, "LON": lon}


class TestIsdHourlyDataPresent:
    def test_true_when_hourly_marker_present(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(206, HOURLY_ROW)]))
        assert stationmap._isd_hourly_data_present("72308513750", 2005, session=session)

    def test_false_when_only_daily_summaries(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(206, DAILY_ONLY_ROW)]))
        assert not stationmap._isd_hourly_data_present(
            "72308513750", 2005, session=session
        )

    def test_false_on_404(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(404, "")]))
        assert not stationmap._isd_hourly_data_present(
            "72308513750", 2005, session=session
        )

    def test_false_on_request_exception(self) -> None:
        import requests

        session = cast("Any", _FakeSession([requests.RequestException("boom")]))
        assert not stationmap._isd_hourly_data_present(
            "72308513750", 2005, session=session
        )


class TestCandidateIdsForWban:
    def test_orders_real_usaf_by_end_descending(self) -> None:
        history = pd.DataFrame(
            [
                _history_row("722051", "12841", "19410616", "19900206"),
                _history_row("722053", "12841", "19900208", "20250827"),
            ]
        )
        ids, lon = stationmap._candidate_ids_for_wban(history, "12841")
        assert ids == ["72205312841", "72205112841"]
        assert lon == pytest.approx(-76.289)

    def test_includes_same_usaf_placeholder_wban_variant(self) -> None:
        """Orlando case: 72205399999 carries real hourly data missing from 72205312841."""
        history = pd.DataFrame(
            [
                _history_row("722053", "12841", "19900208", "20250827"),
                _history_row("722053", "99999", "20000101", "20031231"),
            ]
        )
        ids, _lon = stationmap._candidate_ids_for_wban(history, "12841")
        assert ids == ["72205312841", "72205399999"]

    def test_appends_usaf_placeholder_last(self) -> None:
        history = pd.DataFrame(
            [
                _history_row("722053", "12841", "19900208", "20250827"),
                _history_row("999999", "12841", "19500101", "19721231"),
            ]
        )
        ids, _lon = stationmap._candidate_ids_for_wban(history, "12841")
        assert ids == ["72205312841", "99999912841"]

    def test_deduplicates_repeated_usaf_rows(self) -> None:
        history = pd.DataFrame(
            [
                _history_row("722053", "12841", "19900208", "20031231"),
                _history_row("722053", "12841", "20040101", "20250827"),
            ]
        )
        ids, _lon = stationmap._candidate_ids_for_wban(history, "12841")
        assert ids == ["72205312841"]

    def test_returns_empty_for_unknown_wban(self) -> None:
        history = pd.DataFrame(
            [_history_row("722051", "12841", "19410616", "19900206")]
        )
        ids, lon = stationmap._candidate_ids_for_wban(history, "00000")
        assert ids == []
        assert lon is None


class TestCandidateVerifiedAt:
    def test_short_circuits_on_first_success(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(206, HOURLY_ROW)]))
        cache: dict[tuple[str, int], bool] = {}
        assert stationmap._candidate_verified_at(
            ["AAA", "BBB"], 2005, session=session, cache=cache
        )
        assert cast("_FakeSession", session).requested_urls == [
            stationmap.ISD_URL_TEMPLATE.format(year=2005, station_id="AAA")
        ]

    def test_falls_through_to_second_candidate(self) -> None:
        session = cast(
            "Any",
            _FakeSession([_FakeResponse(404, ""), _FakeResponse(206, HOURLY_ROW)]),
        )
        cache: dict[tuple[str, int], bool] = {}
        assert stationmap._candidate_verified_at(
            ["AAA", "BBB"], 2005, session=session, cache=cache
        )

    def test_false_when_all_candidates_fail(self) -> None:
        session = cast(
            "Any", _FakeSession([_FakeResponse(404, ""), _FakeResponse(404, "")])
        )
        cache: dict[tuple[str, int], bool] = {}
        assert not stationmap._candidate_verified_at(
            ["AAA", "BBB"], 2005, session=session, cache=cache
        )

    def test_caches_by_station_and_year(self) -> None:
        calls: list[tuple[str, int]] = []

        def fake_present(station_id: str, year: int, *, session: Any) -> bool:
            calls.append((station_id, year))
            return True

        original = stationmap._isd_hourly_data_present
        stationmap._isd_hourly_data_present = fake_present  # type: ignore[assignment]
        try:
            cache: dict[tuple[str, int], bool] = {}
            stationmap._candidate_verified_at(
                ["AAA"], 2005, session=cast("Any", None), cache=cache
            )
            stationmap._candidate_verified_at(
                ["AAA"], 2005, session=cast("Any", None), cache=cache
            )
            assert calls == [("AAA", 2005)]
        finally:
            stationmap._isd_hourly_data_present = original


class TestBuildStationMap:
    def test_end_to_end(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        lcd_csv = tmp_path / "cities_lcd_stations.csv"
        lcd_csv.write_text(
            "location_id,lcd_id,dist_km,elev_m\n"
            "0,USW00013750,9.6,5.0\n"
            "1,USW00099999,10.0,20.0\n"
            "2,,,\n"
        )

        history = pd.DataFrame(
            [
                _history_row("722051", "13750", "19730101", "20250827"),
            ]
        )
        monkeypatch.setattr(stationmap, "fetch_isd_history", lambda **_k: history)

        def fake_verified(
            candidate_ids: list[str], _year: int, *, session: Any, cache: Any
        ) -> bool:
            return bool(candidate_ids) and candidate_ids[0].endswith("13750")

        monkeypatch.setattr(stationmap, "_candidate_verified_at", fake_verified)

        with caplog.at_level("WARNING"):
            result = stationmap.build_station_map(
                str(lcd_csv), session=cast("Any", None), start_year=2000, end_year=2025
            )

        assert len(result) == 3
        by_id = result.set_index("location_id")

        assert by_id.loc[0, "isd_ids"] == "72205113750"
        assert by_id.loc[0, "wban"] == "13750"

        assert pd.isna(by_id.loc[1, "isd_ids"])  # wban 99999 -> no history row
        assert pd.isna(by_id.loc[2, "isd_ids"])  # no lcd_id at all

        assert any(
            "2 city/cities have no verified ISD station" in m for m in caplog.messages
        )


class TestMain:
    def test_exits_nonzero_when_not_fully_matched(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        out_csv = tmp_path / "out.csv"
        partial_map = pd.DataFrame(
            {
                "location_id": [0, 1],
                "wban": ["13750", None],
                "isd_ids": ["72205113750", None],
                "lon": [-76.289, None],
                "dist_km": [9.6, None],
                "elev_m": [5.0, None],
            }
        )
        args = argparse.Namespace(
            lcd_stations_csv="cities_lcd_stations.csv",
            out=str(out_csv),
            start_year=2000,
            end_year=2025,
        )
        monkeypatch.setattr(
            stationmap, "build_station_map", lambda *_a, **_k: partial_map
        )
        monkeypatch.setattr(stationmap, "_parse_args", lambda: args)
        with pytest.raises(SystemExit) as excinfo:
            stationmap.main()
        assert excinfo.value.code == 1

    def test_exits_zero_when_fully_matched(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        out_csv = tmp_path / "out.csv"
        full_map = pd.DataFrame(
            {
                "location_id": [0],
                "wban": ["13750"],
                "isd_ids": ["72205113750"],
                "lon": [-76.289],
                "dist_km": [9.6],
                "elev_m": [5.0],
            }
        )
        args = argparse.Namespace(
            lcd_stations_csv="cities_lcd_stations.csv",
            out=str(out_csv),
            start_year=2000,
            end_year=2025,
        )
        monkeypatch.setattr(stationmap, "build_station_map", lambda *_a, **_k: full_map)
        monkeypatch.setattr(stationmap, "_parse_args", lambda: args)
        stationmap.main()
        assert out_csv.exists()
