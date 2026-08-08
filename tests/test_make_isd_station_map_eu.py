# Copyright (C) 2026 Kenneth Porter

"""Tests for make_isd_station_map_eu.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import make_isd_station_map_eu as eu_stationmap
from make_isd_station_map import PLACEHOLDER_USAF, PLACEHOLDER_WBAN


def _history_row(
    usaf: str,
    wban: str,
    begin: str,
    end: str,
    lat: str = "51.4700",
    lon: str = "-0.4543",
    elev: str = "25.0",
) -> dict[str, str]:
    return {
        "USAF": usaf,
        "WBAN": wban,
        "LAT": lat,
        "LON": lon,
        "ELEV(M)": elev,
        "BEGIN": begin,
        "END": end,
    }


class TestHaversineKm:
    def test_zero_distance_for_same_point(self) -> None:
        assert eu_stationmap.haversine_km(
            51.47, -0.4543, 51.47, -0.4543
        ) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_london_to_paris(self) -> None:
        distance = eu_stationmap.haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        assert distance == pytest.approx(343.0, rel=0.02)


class TestCandidateIdsForStation:
    def test_normal_station_returns_three_ordered_candidates(self) -> None:
        ids = eu_stationmap.candidate_ids_for_station("722051", "13750")
        assert ids == ["72205113750", "72205199999", "99999913750"]

    def test_placeholder_usaf_returns_placeholder_usaf_variant_only(self) -> None:
        ids = eu_stationmap.candidate_ids_for_station(PLACEHOLDER_USAF, "13750")
        assert ids == ["99999913750"]

    def test_placeholder_wban_returns_placeholder_wban_variant_only(self) -> None:
        ids = eu_stationmap.candidate_ids_for_station("722051", PLACEHOLDER_WBAN)
        assert ids == ["72205199999"]

    def test_both_placeholders_returns_empty(self) -> None:
        ids = eu_stationmap.candidate_ids_for_station(
            PLACEHOLDER_USAF, PLACEHOLDER_WBAN
        )
        assert ids == []


class TestRankStationRows:
    @staticmethod
    def _history(rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_excludes_stations_beyond_max_distance(self) -> None:
        history = self._history(
            [
                {
                    "USAF": "722051",
                    "WBAN": "13750",
                    "LAT_NUM": 60.0,
                    "LON_NUM": 10.0,
                    "ELEV_NUM": 25.0,
                    "BEGIN": "19730101",
                    "END": "20250827",
                },
            ]
        )
        ranked = eu_stationmap.rank_station_rows(
            history, 51.47, -0.4543, 25.0, start_year=1991, end_year=2025
        )
        assert ranked.empty

    def test_excludes_stations_beyond_max_elevation_delta(self) -> None:
        history = self._history(
            [
                {
                    "USAF": "722051",
                    "WBAN": "13750",
                    "LAT_NUM": 51.471,
                    "LON_NUM": -0.4543,
                    "ELEV_NUM": 1000.0,
                    "BEGIN": "19730101",
                    "END": "20250827",
                },
            ]
        )
        ranked = eu_stationmap.rank_station_rows(
            history, 51.47, -0.4543, 25.0, start_year=1991, end_year=2025
        )
        assert ranked.empty

    def test_full_coverage_station_ranks_before_farther_recent_only_station(
        self,
    ) -> None:
        history = self._history(
            [
                {
                    "USAF": "111111",
                    "WBAN": "11111",
                    "LAT_NUM": 51.50,
                    "LON_NUM": -0.40,
                    "ELEV_NUM": 25.0,
                    "BEGIN": "19850101",
                    "END": "20250827",
                },
                {
                    "USAF": "222222",
                    "WBAN": "22222",
                    "LAT_NUM": 51.471,
                    "LON_NUM": -0.4543,
                    "ELEV_NUM": 25.0,
                    "BEGIN": "20100101",
                    "END": "20250827",
                },
            ]
        )
        ranked = eu_stationmap.rank_station_rows(
            history, 51.47, -0.4543, 25.0, start_year=1991, end_year=2025
        )
        assert ranked.iloc[0]["USAF"] == "111111"
        assert ranked.iloc[0]["coverage_bucket"] == 0
        assert ranked.iloc[1]["USAF"] == "222222"
        assert ranked.iloc[1]["coverage_bucket"] == 1

    def test_closer_station_ranks_first_within_the_same_bucket(self) -> None:
        history = self._history(
            [
                {
                    "USAF": "111111",
                    "WBAN": "11111",
                    "LAT_NUM": 51.50,
                    "LON_NUM": -0.40,
                    "ELEV_NUM": 25.0,
                    "BEGIN": "19850101",
                    "END": "20250827",
                },
                {
                    "USAF": "222222",
                    "WBAN": "22222",
                    "LAT_NUM": 51.471,
                    "LON_NUM": -0.4543,
                    "ELEV_NUM": 25.0,
                    "BEGIN": "19850101",
                    "END": "20250827",
                },
            ]
        )
        ranked = eu_stationmap.rank_station_rows(
            history, 51.47, -0.4543, 25.0, start_year=1991, end_year=2025
        )
        assert ranked.iloc[0]["USAF"] == "222222"


class TestPrepareHistory:
    def test_parses_numeric_columns_and_filters_to_loose_bbox(self) -> None:
        history = pd.DataFrame(
            [
                _history_row("722051", "13750", "19730101", "20250827"),
                _history_row(
                    "999001", "88888", "19730101", "20250827", lat="10.0", lon="10.0"
                ),
            ]
        )
        result = eu_stationmap._prepare_history(history)
        assert list(result["USAF"]) == ["722051"]


class TestBuildStationMapEu:
    def test_end_to_end(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cities_csv = tmp_path / "cities_eu.csv"
        cities_csv.write_text(
            "location_id,lat,lng,dem_m,utc_offset_hours\n"
            "1000,51.47,-0.4543,25.0,0\n"
            "1001,90.0,90.0,0.0,5\n"
        )

        history = pd.DataFrame(
            [_history_row("722051", "13750", "19730101", "20250827")]
        )
        monkeypatch.setattr(eu_stationmap, "fetch_isd_history", lambda **_k: history)

        def fake_verified(
            candidate_ids: list[str], _year: int, *, session: Any, cache: Any
        ) -> bool:
            return bool(candidate_ids) and candidate_ids[0].endswith("13750")

        monkeypatch.setattr(eu_stationmap, "_candidate_verified_at", fake_verified)

        with caplog.at_level("WARNING"):
            result = eu_stationmap.build_station_map_eu(
                str(cities_csv),
                session=cast("Any", None),
                start_year=1991,
                end_year=2025,
            )

        assert len(result) == 2
        by_id = result.set_index("location_id")

        assert by_id.loc[1000, "isd_ids"] == "72205113750|72205199999|99999913750"
        assert by_id.loc[1000, "utc_offset_hours"] == 0
        assert by_id.loc[1000, "begin_verified"]

        assert pd.isna(by_id.loc[1001, "isd_ids"])
        assert any(
            "1 city/cities have no verified ISD station" in m for m in caplog.messages
        )


class TestMain:
    def test_exits_nonzero_when_not_fully_matched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out_csv = tmp_path / "out.csv"
        partial_map = pd.DataFrame(
            {
                "location_id": [1000, 1001],
                "usaf": ["722051", None],
                "wban": ["13750", None],
                "isd_ids": ["72205113750", None],
                "lon": [-0.4543, None],
                "dist_km": [1.0, None],
                "elev_m": [25.0, None],
                "utc_offset_hours": [0, 5],
                "begin_verified": [True, False],
            }
        )
        args = argparse.Namespace(
            cities_csv="cities_eu.csv",
            out=str(out_csv),
            start_year=1991,
            end_year=2025,
        )
        monkeypatch.setattr(
            eu_stationmap, "build_station_map_eu", lambda *_a, **_k: partial_map
        )
        monkeypatch.setattr(eu_stationmap, "_parse_args", lambda: args)
        with pytest.raises(SystemExit) as excinfo:
            eu_stationmap.main()
        assert excinfo.value.code == 1

    def test_exits_zero_when_fully_matched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out_csv = tmp_path / "out.csv"
        full_map = pd.DataFrame(
            {
                "location_id": [1000],
                "usaf": ["722051"],
                "wban": ["13750"],
                "isd_ids": ["72205113750"],
                "lon": [-0.4543],
                "dist_km": [1.0],
                "elev_m": [25.0],
                "utc_offset_hours": [0],
                "begin_verified": [True],
            }
        )
        args = argparse.Namespace(
            cities_csv="cities_eu.csv",
            out=str(out_csv),
            start_year=1991,
            end_year=2025,
        )
        monkeypatch.setattr(
            eu_stationmap, "build_station_map_eu", lambda *_a, **_k: full_map
        )
        monkeypatch.setattr(eu_stationmap, "_parse_args", lambda: args)
        eu_stationmap.main()
        assert out_csv.exists()
