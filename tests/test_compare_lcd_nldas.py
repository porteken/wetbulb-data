"""Tests for the LCD-vs-NLDAS wet-bulb comparison report."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import compare_lcd_nldas


def test_build_comparison_joins_on_location_and_date_and_computes_diffs() -> None:
    lcd_df = pd.DataFrame(
        {
            "location_id": [1, 1, 2],
            "date": ["2010-01-01", "2010-01-02", "2020-06-15"],
            "wetbulb": [20.0, 21.0, 22.0],
            "wetbulb_avg": [15.0, 16.0, 17.0],
        }
    )
    nldas_df = pd.DataFrame(
        {
            "location_id": [1, 1, 2, 3],
            "date": ["2010-01-01", "2010-01-02", "2020-06-15", "2010-01-01"],
            "wetbulb": [19.0, 20.0, 25.0, 5.0],
            "wetbulb_avg": [14.0, 15.5, 16.0, 4.0],
        }
    )

    merged = compare_lcd_nldas.build_comparison(lcd_df, nldas_df)

    assert len(merged) == 3
    assert merged["diff"].tolist() == pytest.approx([1.0, 1.0, -3.0])
    assert merged["diff_avg"].tolist() == pytest.approx([1.0, 0.5, 1.0])
    assert merged["decade"].tolist() == [2010, 2010, 2020]


def _merged_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": [1, 1, 2, 2],
            "city": ["Austin", "Austin", "Miami", "Miami"],
            "state": ["TX", "TX", "FL", "FL"],
            "date": ["2010-01-01", "2010-01-02", "2020-06-15", "2020-06-16"],
            "diff": [1.0, -1.0, 2.0, 0.0],
            "diff_avg": [0.5, -0.5, 1.0, 0.0],
            "decade": [2010, 2010, 2020, 2020],
        }
    )


class TestSummarize:
    def test_reports_bias_mae_and_p95_absolute_error(self) -> None:
        merged = _merged_frame()

        summary = compare_lcd_nldas._summarize(merged)

        assert summary["n_days"] == 4
        assert summary["bias_max"] == pytest.approx(0.5)
        assert summary["mae_max"] == pytest.approx(1.0)
        assert summary["bias_avg"] == pytest.approx(0.25)
        assert summary["mae_avg"] == pytest.approx(0.5)


class TestSummarizeComparison:
    def test_returns_overall_by_city_and_by_decade_tables(self) -> None:
        merged = _merged_frame()

        summaries = compare_lcd_nldas.summarize_comparison(merged)

        assert set(summaries) == {"overall", "by_city", "by_decade"}
        assert len(summaries["overall"]) == 1
        assert set(summaries["by_city"]["city"]) == {"Austin", "Miami"}
        assert set(summaries["by_decade"]["decade"]) == {2010, 2020}


class TestCoverageReport:
    def test_reports_day_counts_and_gaps_per_location(self) -> None:
        lcd_df = pd.DataFrame({"location_id": [1, 1, 2]})
        nldas_df = pd.DataFrame({"location_id": [1, 1, 1, 3]})

        coverage = compare_lcd_nldas._coverage_report(lcd_df, nldas_df)

        by_location = coverage.set_index("location_id")
        assert by_location.loc[1, "lcd_days"] == 2
        assert by_location.loc[1, "nldas_days"] == 3
        assert by_location.loc[1, "missing_from_lcd"] == 1
        assert by_location.loc[2, "nldas_days"] == 0
        assert by_location.loc[3, "lcd_days"] == 0
        assert by_location.loc[3, "missing_from_lcd"] == 1


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["compare_lcd_nldas.py"])

    args = compare_lcd_nldas._parse_args()

    assert args.lcd_csv == "lcd_pilot_wetbulb.csv"
    assert args.out_prefix == "lcd_vs_nldas"
