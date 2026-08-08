"""Tests for make_lcd_station_map.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import make_lcd_station_map as stationmap


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


HOURLY_ROW = '"FM-15","2024-01-01T00:53:00"'
DAILY_ONLY_ROW = '"SOD","2024-01-01T00:00:00"'


class TestLcdHourlyDataPresent:
    def test_true_when_hourly_marker_present(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(200, HOURLY_ROW)]))
        assert stationmap._lcd_hourly_data_present("USW00023062", 2024, session=session)

    def test_false_when_only_daily_summaries(self) -> None:
        """The Buckley SFB case: file exists but has no hourly rows."""
        session = cast("Any", _FakeSession([_FakeResponse(200, DAILY_ONLY_ROW)]))
        assert not stationmap._lcd_hourly_data_present(
            "USW00023062", 2024, session=session
        )

    def test_false_on_404(self) -> None:
        session = cast("Any", _FakeSession([_FakeResponse(404, "")]))
        assert not stationmap._lcd_hourly_data_present(
            "USW00023062", 2024, session=session
        )

    def test_false_on_request_exception(self) -> None:
        import requests

        session = cast("Any", _FakeSession([requests.RequestException("boom")]))
        assert not stationmap._lcd_hourly_data_present(
            "USW00023062", 2024, session=session
        )


class TestFirstFullArchiveYear:
    def test_clamps_to_start_year_when_archive_already_covers_it(self) -> None:
        assert stationmap._first_full_archive_year("1990-05-01", 2000) == 2000

    def test_uses_year_after_archive_begin_for_short_archives(self) -> None:
        assert stationmap._first_full_archive_year("2015-08-01", 2000) == 2016


class TestStationVerifiedAt:
    def test_caches_by_station_and_year(self) -> None:
        calls: list[tuple[str, int]] = []

        def fake_present(station_id: str, year: int, *, session: Any) -> bool:
            calls.append((station_id, year))
            return True

        import make_lcd_station_map as mod

        original = mod._lcd_hourly_data_present
        mod._lcd_hourly_data_present = fake_present
        try:
            cache: dict[tuple[str, int], bool] = {}
            assert stationmap._station_verified_at(
                "AAA",
                2020,
                session=cast("Any", None),
                cache=cache,
            )
            assert stationmap._station_verified_at(
                "AAA",
                2020,
                session=cast("Any", None),
                cache=cache,
            )
            assert calls == [("AAA", 2020)]
        finally:
            mod._lcd_hourly_data_present = original


def _station(lcd_id: str, lat: float, lng: float, archive_begin: str) -> dict[str, Any]:
    return {
        "icao": lcd_id[-3:],
        "lat": lat,
        "lng": lng,
        "name": lcd_id,
        "lcd_id": lcd_id,
        "elev_m": 100.0,
        "archive_begin": archive_begin,
    }


class TestPickStation:
    def test_picks_nearest_tier1_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        near = _station("NEAR", 40.0, -74.0, "1990-01-01")
        far = _station("FAR", 41.0, -75.0, "1990-01-01")
        monkeypatch.setattr(
            stationmap, "_lcd_hourly_data_present", lambda *_a, **_k: True
        )
        result = stationmap._pick_station(
            [near, far],
            start_year=2000,
            end_year=2025,
            session=cast("Any", None),
            cache={},
        )
        assert result is not None
        picked, tier = result
        assert picked["lcd_id"] == "NEAR"
        assert tier == stationmap.TIER_FULL_WINDOW

    def test_falls_back_to_tier2_when_no_full_window_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        short_archive = _station("SHORT", 40.0, -74.0, "2015-01-01")
        monkeypatch.setattr(
            stationmap, "_lcd_hourly_data_present", lambda *_a, **_k: True
        )
        result = stationmap._pick_station(
            [short_archive],
            start_year=2000,
            end_year=2025,
            session=cast("Any", None),
            cache={},
        )
        assert result is not None
        picked, tier = result
        assert picked["lcd_id"] == "SHORT"
        assert tier == stationmap.TIER_SHORT_ARCHIVE

    def test_returns_none_when_no_candidate_verifies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        station = _station("DEAD", 40.0, -74.0, "1990-01-01")
        monkeypatch.setattr(
            stationmap, "_lcd_hourly_data_present", lambda *_a, **_k: False
        )
        result = stationmap._pick_station(
            [station],
            start_year=2000,
            end_year=2025,
            session=cast("Any", None),
            cache={},
        )
        assert result is None

    def test_skips_tier2_candidate_whose_archive_starts_after_end_year(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        too_new = _station("TOONEW", 40.0, -74.0, "2026-01-01")
        monkeypatch.setattr(
            stationmap, "_lcd_hourly_data_present", lambda *_a, **_k: True
        )
        result = stationmap._pick_station(
            [too_new],
            start_year=2000,
            end_year=2025,
            session=cast("Any", None),
            cache={},
        )
        assert result is None

    def test_prefers_tier1_over_a_closer_tier2_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closer_short_archive = _station("CLOSE_SHORT", 40.0, -74.0, "2015-01-01")
        farther_full_window = _station("FAR_FULL", 41.0, -75.0, "1990-01-01")
        monkeypatch.setattr(
            stationmap, "_lcd_hourly_data_present", lambda *_a, **_k: True
        )
        result = stationmap._pick_station(
            [closer_short_archive, farther_full_window],
            start_year=2000,
            end_year=2025,
            session=cast("Any", None),
            cache={},
        )
        assert result is not None
        picked, tier = result
        assert picked["lcd_id"] == "FAR_FULL"
        assert tier == stationmap.TIER_FULL_WINDOW


class TestBuildStationMap:
    def test_end_to_end_with_tier1_tier2_and_unmatched(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cities_csv = tmp_path / "cities.csv"
        cities_csv.write_text(
            "location_id,city,state,lat,lng\n"
            "0,Full City,ST,40.0,-74.0\n"
            "1,Short City,ST,41.0,-75.0\n"
            "2,No Station City,ST,50.0,-100.0\n"
        )
        monkeypatch.setattr(stationmap, "LCD_MAX_CANDIDATES_PER_CITY", 1)

        stations = [
            _station("FULL", 40.0, -74.0, "1990-01-01"),
            _station("SHORT", 41.0, -75.0, "2015-01-01"),
            _station("DEAD", 50.0, -100.0, "1990-01-01"),
        ]
        monkeypatch.setattr(stationmap, "fetch_asos_stations", lambda **_k: stations)

        def fake_present(station_id: str, _year: int, *, session: Any) -> bool:
            return station_id in ("FULL", "SHORT")

        monkeypatch.setattr(stationmap, "_lcd_hourly_data_present", fake_present)

        with caplog.at_level("WARNING"):
            result = stationmap.build_station_map(
                str(cities_csv), start_year=2000, end_year=2025
            )

        assert len(result) == 3
        by_id = result.set_index("location_id")

        assert by_id.loc[0, "lcd_id"] == "FULL"
        assert by_id.loc[0, "tier"] == stationmap.TIER_FULL_WINDOW

        assert by_id.loc[1, "lcd_id"] == "SHORT"
        assert by_id.loc[1, "tier"] == stationmap.TIER_SHORT_ARCHIVE

        assert pd.isna(by_id.loc[2, "lcd_id"])
        assert any("No Station City" in m for m in caplog.messages)
        assert any("tier-2" in m for m in caplog.messages)


class TestMain:
    def test_exits_nonzero_when_not_fully_matched(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import pandas as pd

        out_csv = tmp_path / "out.csv"
        partial_map = pd.DataFrame(
            {
                "location_id": [0, 1],
                "icao": ["AAA", None],
                "lcd_id": ["AAA", None],
                "dist_km": [1.0, None],
                "elev_m": [10.0, None],
                "archive_begin": ["1990-01-01", None],
                "tier": [1, None],
            }
        )
        args = argparse.Namespace(
            cities_csv="cities.csv", out=str(out_csv), start_year=2000, end_year=2025
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
        import pandas as pd

        out_csv = tmp_path / "out.csv"
        full_map = pd.DataFrame(
            {
                "location_id": [0],
                "icao": ["AAA"],
                "lcd_id": ["AAA"],
                "dist_km": [1.0],
                "elev_m": [10.0],
                "archive_begin": ["1990-01-01"],
                "tier": [1],
            }
        )
        args = argparse.Namespace(
            cities_csv="cities.csv", out=str(out_csv), start_year=2000, end_year=2025
        )
        monkeypatch.setattr(stationmap, "build_station_map", lambda *_a, **_k: full_map)
        monkeypatch.setattr(stationmap, "_parse_args", lambda: args)
        stationmap.main()
        assert out_csv.exists()
