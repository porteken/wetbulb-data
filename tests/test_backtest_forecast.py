"""Tests for leakage-safe backtest interval calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import backtest_forecast
from forecast_model import LinearFit, PooledPrior, dersimonian_laird, fit_linear


def test_safe_cli_path_rejects_paths_outside_the_working_directory() -> None:
    """CLI output paths cannot traverse out of the repository."""
    with pytest.raises(argparse.ArgumentTypeError):
        backtest_forecast._safe_cli_path("../backtest_report.json")


def test_calibration_uses_only_observable_group_balanced_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backtest_forecast, "MIN_CALIBRATION_CASES", 1)
    common = {
        "metric": "avg_wetbulb",
        "gmst": 10.0,
        "legacy": 10.0,
        "lower": 9.0,
        "upper": 11.0,
        "oracle": False,
        "horizon": 2,
        "season": "Annual",
    }
    cases = pd.DataFrame(
        [
            {
                **common,
                "location_id": 1,
                "station_group": "shared",
                "origin": 2009,
                "year": 2011,
                "actual": 12.0,
            },
            {
                **common,
                "location_id": 2,
                "station_group": "shared",
                "origin": 2009,
                "year": 2011,
                "actual": 20.0,
            },
            {
                **common,
                "location_id": 3,
                "station_group": "future",
                "origin": 2010,
                "year": 2013,
                "actual": 30.0,
            },
            {
                **common,
                "location_id": 4,
                "station_group": "gate",
                "origin": 2012,
                "year": 2014,
                "actual": 10.0,
            },
        ]
    )

    calibrated = backtest_forecast.calibrate_intervals(cases)
    gate = calibrated.loc[calibrated["origin"] == 2012].iloc[0]

    assert gate["calibration_factor"] == pytest.approx(2.0)
    assert gate["lower"] == pytest.approx(8.0)
    assert gate["upper"] == pytest.approx(12.0)


def _write_batch(
    data_root: Path,
    year: int,
    frame: pd.DataFrame,
    *,
    name: str = "wetbulb_batch_0.parquet",
) -> None:
    directory = data_root / f"year={year}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / name, index=False)


class TestAnnualMetrics:
    def test_uses_cache_when_present_and_seasoned(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.csv"
        cached = pd.DataFrame(
            {
                "location_id": [1],
                "year": [2001],
                "season": ["Annual"],
                "avg_wetbulb": [20.0],
            }
        )
        cached.to_csv(cache_path, index=False)

        result = backtest_forecast.annual_metrics(
            tmp_path / "does-not-exist", cache_path
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), cached.reset_index(drop=True)
        )

    def test_raises_when_no_parquet_batches_are_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            backtest_forecast.annual_metrics(tmp_path, tmp_path / "cache.csv")

    def test_prefers_isd_rows_and_masks_incomplete_locations(
        self, tmp_path: Path
    ) -> None:
        dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")
        full_year = pd.DataFrame(
            {
                "location_id": 1,
                "date": dates.strftime("%Y-%m-%d"),
                "wetbulb": 20.0,
                "wetbulb_avg": 15.0,
                "source": "isd",
            }
        )
        duplicate_nldas_row = pd.DataFrame(
            {
                "location_id": [1],
                "date": ["2001-04-10"],
                "wetbulb": [99.0],
                "wetbulb_avg": [99.0],
                "source": ["nldas"],
            }
        )
        sparse_location = pd.DataFrame(
            {
                "location_id": 2,
                "date": ["2001-01-01", "2001-01-02", "2001-01-03"],
                "wetbulb": 20.0,
                "wetbulb_avg": 15.0,
                "source": "isd",
            }
        )
        data_root = tmp_path / "data"
        _write_batch(
            data_root,
            2001,
            pd.concat([full_year, duplicate_nldas_row], ignore_index=True),
            name="wetbulb_batch_0.parquet",
        )
        _write_batch(data_root, 2001, sparse_location, name="wetbulb_batch_1.parquet")
        cache_path = tmp_path / "cache" / "annual_metrics.csv"

        result = backtest_forecast.annual_metrics(data_root, cache_path)

        assert cache_path.exists()
        annual_loc1 = result.loc[
            (result["location_id"] == 1) & (result["season"] == "Annual")
        ].iloc[0]
        assert annual_loc1["days_present"] == 365
        assert annual_loc1["avg_wetbulb"] == pytest.approx(20.0)

        annual_loc2 = result.loc[
            (result["location_id"] == 2) & (result["season"] == "Annual")
        ].iloc[0]
        assert pd.isna(annual_loc2["avg_wetbulb"])


class TestRepresentativeFits:
    def _fit(self, *, n: int, slope_variance: float) -> LinearFit:
        return LinearFit(
            intercept_mean=0.0,
            predictor_mean=0.0,
            slope=1.0,
            slope_variance=slope_variance,
            residual_variance=0.1,
            n=n,
        )

    def test_picks_highest_n_then_lowest_variance_per_group(self) -> None:
        fits = {
            1: self._fit(n=10, slope_variance=0.5),
            2: self._fit(n=15, slope_variance=0.3),
            3: self._fit(n=10, slope_variance=0.1),
        }
        station_groups = {1: "A", 2: "A", 3: "B"}

        representatives = backtest_forecast._representative_fits(fits, station_groups)

        chosen_n = sorted(fit.n for fit in representatives)
        assert chosen_n == [10, 15]
        assert len(representatives) == 2

    def test_ignores_locations_without_a_station_group(self) -> None:
        fits = {1: self._fit(n=10, slope_variance=0.5)}

        representatives = backtest_forecast._representative_fits(fits, {})

        assert representatives == []


class TestGmstExtrapolation:
    def test_extrapolates_a_linear_trend(self) -> None:
        years = [2010, 2011, 2012, 2013]
        values = [0.0, 0.1, 0.2, 0.3]
        training_gmst = pd.Series(values, index=years)

        predicted, variance = backtest_forecast._gmst_extrapolation(training_gmst, 2015)

        assert predicted == pytest.approx(0.5, abs=1e-6)
        assert variance >= 0.0

    def test_raises_with_too_few_observations(self) -> None:
        training_gmst = pd.Series([0.0, 0.1], index=[2010, 2011])
        with pytest.raises(ValueError, match="three observations"):
            backtest_forecast._gmst_extrapolation(training_gmst, 2015)


class TestLegacyPrediction:
    def test_predicts_from_a_linear_series(self) -> None:
        years = np.array([2010, 2011, 2012, 2013])
        values = np.array([0.0, 1.0, 2.0, 3.0])

        prediction = backtest_forecast._legacy_prediction(years, values, 2015)

        assert prediction == pytest.approx(5.0)

    def test_returns_none_with_too_few_observations(self) -> None:
        years = np.array([2010, 2011])
        values = np.array([0.0, 1.0])

        assert backtest_forecast._legacy_prediction(years, values, 2015) is None


class TestLocationFits:
    def test_fits_locations_with_enough_training_years(self) -> None:
        years = list(range(2000, 2012))
        gmst = pd.Series([0.05 * (year - 2000) for year in years], index=years)
        training = pd.DataFrame(
            {
                "location_id": [1] * len(years) + [2] * 5,
                "year": years + years[:5],
                "avg_wetbulb": [10.0 + 2.0 * value for value in gmst.to_numpy()]
                + [10.0] * 5,
            }
        )

        fits, metric_values = backtest_forecast._location_fits(
            training, "avg_wetbulb", gmst
        )

        assert set(fits) == {1}
        assert fits[1].n == len(years)
        assert set(metric_values) == {1}

    def test_raises_when_location_id_is_not_integer(self) -> None:
        training = pd.DataFrame(
            {
                "location_id": ["a", "a", "a"],
                "year": [2000, 2001, 2002],
                "avg_wetbulb": [1.0, 2.0, 3.0],
            }
        )
        gmst = pd.Series([0.0, 0.1, 0.2], index=[2000, 2001, 2002])

        with pytest.raises(TypeError, match="must be an integer"):
            backtest_forecast._location_fits(training, "avg_wetbulb", gmst)


class TestMetricForecastCases:
    def _fit_and_prior(self) -> tuple[LinearFit, PooledPrior]:
        gmst = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        values = 10.0 + 5.0 * gmst
        fit = fit_linear(gmst, values)
        assert fit is not None
        prior = dersimonian_laird(np.array([fit.slope]), np.array([fit.slope_variance]))
        return fit, prior

    def test_builds_a_row_per_location_with_a_fit(self) -> None:
        fit, prior = self._fit_and_prior()
        years = np.array([2000, 2001, 2002, 2003, 2004])
        gmst_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        metric_values_arr = 10.0 + 5.0 * gmst_values
        observations = pd.Series(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], index=range(2000, 2006)
        )
        training_gmst = observations.loc[:2004]
        future = pd.DataFrame(
            {"location_id": [1, 2], "year": [2005, 2005], "avg_wetbulb": [15.0, 15.0]}
        )

        rows = backtest_forecast._metric_forecast_cases(
            future,
            {1: fit},
            {1: (years, metric_values_arr, gmst_values)},
            prior,
            observations,
            training_gmst,
            {1: "group-a"},
            metric="avg_wetbulb",
            season="Annual",
            origin=2004,
            oracle=False,
        )

        assert len(rows) == 1
        assert rows[0]["location_id"] == 1
        assert rows[0]["horizon"] == 1
        assert rows[0]["oracle"] is False

    def test_oracle_uses_observed_future_gmst(self) -> None:
        fit, prior = self._fit_and_prior()
        years = np.array([2000, 2001, 2002, 2003, 2004])
        gmst_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        metric_values_arr = 10.0 + 5.0 * gmst_values
        observations = pd.Series(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], index=range(2000, 2006)
        )
        future = pd.DataFrame(
            {"location_id": [1], "year": [2005], "avg_wetbulb": [15.0]}
        )

        rows = backtest_forecast._metric_forecast_cases(
            future,
            {1: fit},
            {1: (years, metric_values_arr, gmst_values)},
            prior,
            observations,
            observations.loc[:2004],
            {1: "group-a"},
            metric="avg_wetbulb",
            season="Annual",
            origin=2004,
            oracle=True,
        )

        assert len(rows) == 1
        assert rows[0]["oracle"] is True


class TestForecastCases:
    def _annual_frame(self) -> pd.DataFrame:
        years = list(range(2000, 2026))
        location_1 = pd.DataFrame(
            {
                "location_id": 1,
                "year": years,
                "season": "Annual",
                "avg_wetbulb": [10.0 + 0.1 * (year - 2000) for year in years],
                "max_wetbulb": [20.0 + 0.1 * (year - 2000) for year in years],
                "avg_wetbulb_avg": [10.0 + 0.1 * (year - 2000) for year in years],
                "max_wetbulb_avg": [20.0 + 0.1 * (year - 2000) for year in years],
            }
        )
        location_2 = location_1.copy()
        location_2["location_id"] = 2
        return pd.concat([location_1, location_2], ignore_index=True)

    def _observations(self) -> pd.Series:
        years = list(range(2000, 2026))
        return pd.Series([0.05 * (year - 2000) for year in years], index=years)

    def test_produces_annual_cases_for_representative_groups(self) -> None:
        annual = self._annual_frame()
        observations = self._observations()
        station_groups = {1: "group-a", 2: "group-b"}

        cases = backtest_forecast._forecast_cases(
            annual,
            observations,
            station_groups,
            oracle=False,
            origins=range(2020, 2022),
        )

        assert not cases.empty
        assert set(cases["season"]) == {"Annual"}
        assert set(cases["oracle"]) == {False}
        assert cases["horizon"].between(1, 10).all()

    def test_oracle_flag_is_threaded_through(self) -> None:
        annual = self._annual_frame()
        observations = self._observations()
        station_groups = {1: "group-a", 2: "group-b"}

        cases = backtest_forecast._forecast_cases(
            annual,
            observations,
            station_groups,
            oracle=True,
            origins=range(2020, 2022),
        )

        assert not cases.empty
        assert set(cases["oracle"]) == {True}


class TestCalibrationFactors:
    def test_defaults_to_one_below_the_minimum_case_count(self) -> None:
        cases = pd.DataFrame(
            {
                "actual": [10.0, 11.0],
                "gmst": [10.0, 10.0],
                "lower": [9.0, 9.0],
                "upper": [11.0, 11.0],
                "metric": ["avg_wetbulb", "avg_wetbulb"],
                "origin": [2010, 2010],
                "year": [2011, 2011],
                "season": ["Annual", "Annual"],
                "location_id": [1, 2],
                "station_group": ["a", "b"],
            }
        )

        factors = backtest_forecast.calibration_factors(cases, 2020)

        assert all(factor == pytest.approx(1.0) for factor in factors.values())

    def test_computes_a_quantile_factor_with_enough_known_cases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backtest_forecast, "MIN_CALIBRATION_CASES", 5)
        n = 10
        cases = pd.DataFrame(
            {
                "actual": [10.0 + i for i in range(n)],
                "gmst": [10.0] * n,
                "lower": [9.0] * n,
                "upper": [11.0] * n,
                "metric": ["avg_wetbulb"] * n,
                "origin": [2010] * n,
                "year": [2011] * n,
                "season": ["Annual"] * n,
                "location_id": list(range(n)),
                "station_group": [f"group-{i}" for i in range(n)],
            }
        )

        factors = backtest_forecast.calibration_factors(cases, 2020)

        assert factors["avg_wetbulb"] >= 1.0

    def test_deduplicates_by_station_group_per_origin_year_metric(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backtest_forecast, "MIN_CALIBRATION_CASES", 1)
        cases = pd.DataFrame(
            {
                "actual": [10.0, 999.0],
                "gmst": [10.0, 10.0],
                "lower": [9.0, 9.0],
                "upper": [11.0, 11.0],
                "metric": ["avg_wetbulb", "avg_wetbulb"],
                "origin": [2010, 2010],
                "year": [2011, 2011],
                "season": ["Annual", "Annual"],
                "location_id": [1, 2],
                "station_group": ["shared", "shared"],
            }
        )

        factors = backtest_forecast.calibration_factors(cases, 2020)

        assert factors["avg_wetbulb"] == pytest.approx(1.0)


class TestSummary:
    def test_computes_errors_and_coverage(self) -> None:
        cases = pd.DataFrame(
            {
                "gmst": [10.0, 12.0],
                "actual": [11.0, 13.0],
                "legacy": [9.0, 15.0],
                "lower": [9.0, 9.0],
                "upper": [12.0, 12.0],
            }
        )

        summary = backtest_forecast._summary(cases)

        assert summary["gmst_abs_error"].tolist() == pytest.approx([1.0, 1.0])
        assert summary["legacy_abs_error"].tolist() == pytest.approx([2.0, 2.0])
        assert summary["covered"].tolist() == [True, False]


def _gate_cases(*, coverage_fraction: float) -> pd.DataFrame:
    n = 20
    covered_count = round(n * coverage_fraction)
    actual = [10.0] * n
    lower = [9.0] * covered_count + [10.5] * (n - covered_count)
    upper = [11.0] * n
    return pd.DataFrame(
        {
            "horizon": [7] * n,
            "season": ["Annual"] * n,
            "metric": ["avg_wetbulb"] * n,
            "gmst": [10.0] * n,
            "actual": actual,
            "legacy": [10.5] * n,
            "lower": lower,
            "upper": upper,
            "calibration_factor": [1.0] * n,
        }
    )


class TestEvaluateGate:
    def test_passes_when_errors_and_coverage_are_in_bounds(self) -> None:
        cases = _gate_cases(coverage_fraction=0.8)

        passed, gate = backtest_forecast.evaluate_gate(cases)

        overall = cast("dict[str, float]", gate["overall"])
        by_metric = cast("dict[str, dict[str, float]]", gate["by_metric"])
        assert passed is True
        assert overall["mae_ratio"] < 1.0
        assert by_metric["avg_wetbulb"]["n"] == 20

    def test_fails_when_coverage_is_out_of_bounds(self) -> None:
        cases = _gate_cases(coverage_fraction=1.0)

        passed, _gate = backtest_forecast.evaluate_gate(cases)

        assert passed is False


class TestDiagnosticSummary:
    def test_groups_by_season_and_metric(self) -> None:
        cases = _gate_cases(coverage_fraction=0.8)

        diagnostics = backtest_forecast.diagnostic_summary(cases)

        assert "Annual" in diagnostics
        assert "avg_wetbulb" in diagnostics["Annual"]
        assert diagnostics["Annual"]["avg_wetbulb"]["n"] == 20


class TestBootstrapDiagnostics:
    def test_returns_overall_and_per_metric_intervals(self) -> None:
        n = 20
        cases = pd.DataFrame(
            {
                "horizon": [7] * n,
                "season": ["Annual"] * n,
                "metric": ["avg_wetbulb"] * n,
                "gmst": [10.0 + (i % 3) for i in range(n)],
                "actual": [10.0] * n,
                "legacy": [10.5] * n,
                "lower": [9.0] * n,
                "upper": [11.0] * n,
                "station_group": [f"group-{i % 4}" for i in range(n)],
            }
        )

        diagnostics = backtest_forecast.bootstrap_diagnostics(cases)

        assert set(diagnostics) == {"overall", "by_metric"}
        overall = cast("dict[str, list[float]]", diagnostics["overall"])
        by_metric = cast("dict[str, object]", diagnostics["by_metric"])
        overall_mae = overall["mae_ratio_95"]
        assert len(overall_mae) == 2
        assert overall_mae[0] <= overall_mae[1]
        assert "avg_wetbulb" in by_metric


def test_parse_args_defaults_resolve_within_the_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["backtest_forecast.py"])

    args = backtest_forecast._parse_args()

    assert args.data_root == backtest_forecast._safe_cli_path("wetbulb_data_csv")
    assert args.annual_cache == backtest_forecast._safe_cli_path(
        "forecast_inputs/annual_metrics.csv"
    )
    assert args.output == backtest_forecast._safe_cli_path(
        "forecast_inputs/backtest_report.json"
    )
