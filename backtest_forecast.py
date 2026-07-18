"""Rolling-origin ship gate for climate-informed wet-bulb forecasts.

The operational model extrapolates GMST with information available at each
origin.  An observed-future-GMST run is emitted separately as an oracle
diagnostic and never participates in the ship decision.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from forecast_model import (
    LinearFit,
    PooledPrior,
    dersimonian_laird,
    fit_linear,
    forecast,
)

FIRST_YEAR = 2000
ORIGINS = range(2012, 2025)
CALIBRATION_ORIGINS = range(2009, 2012)
HORIZONS = range(1, 11)
METRICS = ("avg_wetbulb", "max_wetbulb", "avg_wetbulb_avg", "max_wetbulb_avg")
SEASONS = ("Annual", "Winter", "Spring", "Summer", "Fall")
MIN_TRAINING_YEARS = 10
GATE_MIN_HORIZON = 5
GATE_MAX_HORIZON = 10
OVERALL_ERROR_RATIO_LIMIT = 1.02
METRIC_ERROR_RATIO_LIMIT = 1.05
MIN_COVERAGE = 0.75
MAX_COVERAGE = 0.85
CALIBRATION_QUANTILE = 0.7
MIN_CALIBRATION_CASES = 100
BOOTSTRAP_SAMPLES = 1000
LEAP_YEAR_DAYS = 366
LOGGER = logging.getLogger(__name__)


def annual_metrics(data_root: Path, cache_path: Path) -> pd.DataFrame:
    """Build annual metrics after preferring ISD rows to NLDAS fills per day."""
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if "season" in cached:
            return cached
    files = sorted(data_root.glob("year=*/wetbulb*_batch_*.parquet"))
    if not files:
        message = f"No wet-bulb parquet batches below {data_root}"
        raise FileNotFoundError(message)
    daily = pd.concat(
        [
            pd.read_parquet(
                str(file),
                columns=["location_id", "date", "wetbulb", "wetbulb_avg", "source"],
            )
            for file in files
        ],
        ignore_index=True,
    )
    daily["source_rank"] = daily["source"].ne("isd").astype(int)
    daily = daily.sort_values(["location_id", "date", "source_rank"])
    daily = daily.drop_duplicates(["location_id", "date"], keep="first")
    dates = pd.to_datetime(daily["date"])
    daily["year"] = dates.dt.year
    months = dates.dt.month
    daily["season"] = np.select(
        [months.isin([12, 1, 2]), months.isin([3, 4, 5]), months.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Fall",
    )

    def aggregate(group_columns: list[str]) -> pd.DataFrame:
        return cast(
            "pd.DataFrame",
            daily.groupby(group_columns, as_index=False).agg(
                days_present=("wetbulb", "count"),
                avg_wetbulb=("wetbulb", "mean"),
                max_wetbulb=("wetbulb", "max"),
                days_present_avg=("wetbulb_avg", "count"),
                avg_wetbulb_avg=("wetbulb_avg", "mean"),
                max_wetbulb_avg=("wetbulb_avg", "max"),
            ),
        )

    annual = aggregate(["location_id", "year"])
    annual["season"] = "Annual"
    seasonal = aggregate(["location_id", "year", "season"])
    grouped = pd.concat([annual, seasonal], ignore_index=True)
    annual_days = np.where(
        (grouped["year"] % 4 == 0)
        & ((grouped["year"] % 100 != 0) | (grouped["year"] % 400 == 0)),
        366,
        365,
    )
    required_days = np.select(
        [
            grouped["season"] == "Annual",
            grouped["season"] == "Winter",
            grouped["season"].isin(["Spring", "Summer"]),
        ],
        [annual_days, np.where(annual_days == LEAP_YEAR_DAYS, 91, 90), 92],
        default=91,
    )
    grouped.loc[
        grouped["days_present"] < 0.95 * required_days, ["avg_wetbulb", "max_wetbulb"]
    ] = np.nan
    grouped.loc[
        grouped["days_present_avg"] < 0.95 * required_days,
        ["avg_wetbulb_avg", "max_wetbulb_avg"],
    ] = np.nan
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(cache_path, index=False)
    return cast("pd.DataFrame", grouped)


def _representative_fits(
    fits: dict[int, LinearFit], station_groups: dict[int, str]
) -> list[LinearFit]:
    """Choose one fit per exact ISD group by n, variance, then location id."""
    by_group: dict[str, list[tuple[int, LinearFit]]] = {}
    for location_id, fit in fits.items():
        group = station_groups.get(location_id)
        if group:
            by_group.setdefault(group, []).append((location_id, fit))
    return [
        min(candidates, key=lambda item: (-item[1].n, item[1].slope_variance, item[0]))[
            1
        ]
        for candidates in by_group.values()
    ]


def _gmst_extrapolation(
    training_gmst: pd.Series, target_year: int
) -> tuple[float, float]:
    years = training_gmst.index.to_numpy(dtype=float)
    values = training_gmst.to_numpy(dtype=float)
    fit = fit_linear(years, values)
    if fit is None:
        message = "GMST extrapolation requires three observations"
        raise ValueError(message)
    predicted = fit.intercept_mean + fit.slope * (target_year - fit.predictor_mean)
    # Prediction variance of a centered linear extrapolation.
    sxx = float(np.square(years - fit.predictor_mean).sum())
    variance = fit.residual_variance * (
        1 + 1 / fit.n + (target_year - fit.predictor_mean) ** 2 / sxx
    )
    return predicted, max(variance, 0.0)


def _legacy_prediction(
    years: np.ndarray, values: np.ndarray, target_year: int
) -> float | None:
    fit = fit_linear(years.astype(float), values)
    if fit is None:
        return None
    return fit.intercept_mean + fit.slope * (target_year - fit.predictor_mean)


def _location_fits(
    training: pd.DataFrame, metric: str, observations: pd.Series
) -> tuple[dict[int, LinearFit], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Fit locations with enough complete observations for one metric."""
    fits: dict[int, LinearFit] = {}
    metric_values: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for location_id, group in training.groupby("location_id"):
        if not isinstance(location_id, (int, np.integer)):
            message = "location_id must be an integer"
            raise TypeError(message)
        valid = group.dropna(subset=[metric])
        years = valid["year"].to_numpy(dtype=int)
        values = valid[metric].to_numpy(dtype=float)
        gmst = observations.reindex(years).to_numpy(dtype=float)
        fit = fit_linear(gmst, values)
        if fit is not None and fit.n >= MIN_TRAINING_YEARS:
            fits[int(location_id)] = fit
            metric_values[int(location_id)] = (years, values, gmst)
    return fits, metric_values


def _metric_forecast_cases(
    future: pd.DataFrame,
    fits: dict[int, LinearFit],
    metric_values: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    prior: PooledPrior,
    observations: pd.Series,
    training_gmst: pd.Series,
    station_groups: dict[int, str],
    *,
    metric: str,
    season: str,
    origin: int,
    oracle: bool,
) -> list[dict[str, float | int | str | bool]]:
    """Produce backtest rows for one origin, season, and metric."""
    rows: list[dict[str, float | int | str | bool]] = []
    for location_id_value, target_year_value, actual_value in future[
        ["location_id", "year", metric]
    ].itertuples(index=False, name=None):
        location_id = int(location_id_value)
        target_year = int(target_year_value)
        fit = fits.get(location_id)
        if fit is None:
            continue
        if oracle:
            predictor, predictor_variance = float(observations[target_year]), 0.0
        else:
            predictor, predictor_variance = _gmst_extrapolation(
                training_gmst, target_year
            )
        model = forecast(fit, prior, predictor, predictor_variance=predictor_variance)
        years, values, _ = metric_values[location_id]
        legacy = _legacy_prediction(years, values, target_year)
        if legacy is not None:
            rows.append(
                {
                    "location_id": location_id,
                    "station_group": station_groups[location_id],
                    "origin": origin,
                    "year": target_year,
                    "horizon": target_year - origin,
                    "metric": metric,
                    "season": season,
                    "actual": float(actual_value),
                    "gmst": model.point,
                    "legacy": legacy,
                    "lower": model.lower,
                    "upper": model.upper,
                    "oracle": oracle,
                }
            )
    return rows


def _forecast_cases(
    annual: pd.DataFrame,
    observations: pd.Series,
    station_groups: dict[int, str],
    *,
    oracle: bool,
    origins: range = ORIGINS,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for origin in origins:
        training_gmst = observations.loc[
            (observations.index >= FIRST_YEAR) & (observations.index <= origin)
        ]
        for season in SEASONS:
            season_rows = annual.loc[annual["season"] == season]
            for metric in METRICS:
                training = season_rows.loc[
                    (season_rows["year"] >= FIRST_YEAR)
                    & (season_rows["year"] <= origin),
                    ["location_id", "year", metric],
                ]
                fits, metric_values = _location_fits(training, metric, observations)
                representatives = _representative_fits(fits, station_groups)
                if not representatives:
                    continue
                prior = dersimonian_laird(
                    np.array([item.slope for item in representatives]),
                    np.array([item.slope_variance for item in representatives]),
                )
                future = season_rows.loc[
                    season_rows["year"].between(origin + 1, origin + 10),
                    ["location_id", "year", metric],
                ].dropna()
                rows.extend(
                    _metric_forecast_cases(
                        future,
                        fits,
                        metric_values,
                        prior,
                        observations,
                        training_gmst,
                        station_groups,
                        metric=metric,
                        season=season,
                        origin=origin,
                        oracle=oracle,
                    )
                )
    return pd.DataFrame(rows)


def calibrate_intervals(cases: pd.DataFrame) -> pd.DataFrame:
    """Calibrate margins from errors observable before each gate origin.

    Each exact ISD station group contributes at most one score for an
    origin/target/metric combination. This keeps duplicated city mappings from
    dominating the empirical 80th percentile.
    """
    calibrated = cases.copy()
    calibrated["raw_margin"] = (calibrated["upper"] - calibrated["lower"]) / 2.0
    calibrated["standardized_error"] = (
        calibrated["actual"] - calibrated["gmst"]
    ).abs() / calibrated["raw_margin"].clip(lower=np.finfo(float).eps)
    calibrated["calibration_factor"] = 1.0

    for origin in ORIGINS:
        factors = calibration_factors(calibrated, origin)
        gate_rows = calibrated["origin"] == origin
        for metric, factor in factors.items():
            calibrated.loc[
                gate_rows & (calibrated["metric"] == metric),
                "calibration_factor",
            ] = max(factor, 1.0)

    calibrated["lower"] = calibrated["gmst"] - (
        calibrated["raw_margin"] * calibrated["calibration_factor"]
    )
    calibrated["upper"] = calibrated["gmst"] + (
        calibrated["raw_margin"] * calibrated["calibration_factor"]
    )
    return calibrated.loc[calibrated["origin"].isin(ORIGINS)].copy()


def calibration_factors(cases: pd.DataFrame, origin: int) -> dict[str, float]:
    """Return station-group-balanced factors using only data known at origin."""
    values = cases.copy()
    if "standardized_error" not in values:
        margin = (values["upper"] - values["lower"]) / 2.0
        values["standardized_error"] = (
            values["actual"] - values["gmst"]
        ).abs() / margin.clip(lower=np.finfo(float).eps)
    known = values.loc[
        (values["origin"] < origin)
        & (values["year"] <= origin)
        & (values["season"] == "Annual")
    ].sort_values("location_id")
    known = known.drop_duplicates(
        ["station_group", "origin", "year", "metric"], keep="first"
    )
    factors: dict[str, float] = {}
    for metric in METRICS:
        scores = known.loc[known["metric"] == metric, "standardized_error"].to_numpy(
            dtype=float
        )
        factor = 1.0
        if len(scores) >= MIN_CALIBRATION_CASES:
            factor = float(np.quantile(scores, CALIBRATION_QUANTILE, method="higher"))
        factors[metric] = max(factor, 1.0)
    return factors


def _summary(cases: pd.DataFrame) -> pd.DataFrame:
    values = cases.copy()
    values["gmst_abs_error"] = (values["gmst"] - values["actual"]).abs()
    values["legacy_abs_error"] = (values["legacy"] - values["actual"]).abs()
    values["gmst_sq_error"] = (values["gmst"] - values["actual"]) ** 2
    values["legacy_sq_error"] = (values["legacy"] - values["actual"]) ** 2
    values["covered"] = values["actual"].between(values["lower"], values["upper"])
    return values


def evaluate_gate(cases: pd.DataFrame) -> tuple[bool, dict[str, object]]:
    """Evaluate the specified horizons 5-10 gate on paired operational cases."""
    values = _summary(
        cases.loc[
            (cases["horizon"] >= GATE_MIN_HORIZON)
            & (cases["horizon"] <= GATE_MAX_HORIZON)
            & (cases["season"] == "Annual")
        ]
    )
    overall = {
        "mae_ratio": float(
            values["gmst_abs_error"].mean() / values["legacy_abs_error"].mean()
        ),
        "rmse_ratio": float(
            np.sqrt(values["gmst_sq_error"].mean() / values["legacy_sq_error"].mean())
        ),
    }
    by_metric: dict[str, dict[str, float]] = {}
    for metric, group in values.groupby("metric"):
        by_metric[str(metric)] = {
            "mae_ratio": float(
                group["gmst_abs_error"].mean() / group["legacy_abs_error"].mean()
            ),
            "rmse_ratio": float(
                np.sqrt(group["gmst_sq_error"].mean() / group["legacy_sq_error"].mean())
            ),
            "coverage_80": float(group["covered"].mean()),
            "calibration_factor_min": float(group["calibration_factor"].min()),
            "calibration_factor_max": float(group["calibration_factor"].max()),
            "n": len(group),
        }
    passed = (
        overall["mae_ratio"] <= OVERALL_ERROR_RATIO_LIMIT
        and overall["rmse_ratio"] <= OVERALL_ERROR_RATIO_LIMIT
        and all(
            result["mae_ratio"] <= METRIC_ERROR_RATIO_LIMIT
            and result["rmse_ratio"] <= METRIC_ERROR_RATIO_LIMIT
            and MIN_COVERAGE <= result["coverage_80"] <= MAX_COVERAGE
            for result in by_metric.values()
        )
    )
    return passed, {"overall": overall, "by_metric": by_metric, "n": len(values)}


def diagnostic_summary(cases: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Return non-gating seasonal accuracy and coverage diagnostics."""
    values = _summary(
        cases.loc[cases["horizon"].between(GATE_MIN_HORIZON, GATE_MAX_HORIZON)]
    )
    result: dict[str, dict[str, dict[str, float]]] = {}
    for key, group in values.groupby(["season", "metric"]):
        season, metric = cast("tuple[object, object]", key)
        result.setdefault(str(season), {})[str(metric)] = {
            "mae_ratio": float(
                group["gmst_abs_error"].mean() / group["legacy_abs_error"].mean()
            ),
            "rmse_ratio": float(
                np.sqrt(group["gmst_sq_error"].mean() / group["legacy_sq_error"].mean())
            ),
            "coverage_80": float(group["covered"].mean()),
            "n": float(len(group)),
        }
    return result


def bootstrap_diagnostics(cases: pd.DataFrame) -> dict[str, object]:
    """Bootstrap paired error-ratio uncertainty by exact station group."""
    values = _summary(
        cases.loc[
            cases["horizon"].between(GATE_MIN_HORIZON, GATE_MAX_HORIZON)
            & (cases["season"] == "Annual")
        ]
    )
    rng = np.random.default_rng(20260718)

    def intervals(frame: pd.DataFrame) -> dict[str, list[float]]:
        grouped = frame.groupby("station_group").agg(
            gmst_abs=("gmst_abs_error", "sum"),
            legacy_abs=("legacy_abs_error", "sum"),
            gmst_sq=("gmst_sq_error", "sum"),
            legacy_sq=("legacy_sq_error", "sum"),
        )
        matrix = grouped.to_numpy(dtype=float)
        group_count = len(matrix)
        mae_ratios = np.empty(BOOTSTRAP_SAMPLES)
        rmse_ratios = np.empty(BOOTSTRAP_SAMPLES)
        for index in range(BOOTSTRAP_SAMPLES):
            sample = matrix[rng.integers(0, group_count, size=group_count)].sum(axis=0)
            mae_ratios[index] = sample[0] / sample[1]
            rmse_ratios[index] = np.sqrt(sample[2] / sample[3])
        return {
            "mae_ratio_95": np.quantile(mae_ratios, [0.025, 0.975]).tolist(),
            "rmse_ratio_95": np.quantile(rmse_ratios, [0.025, 0.975]).tolist(),
        }

    return {
        "overall": intervals(values),
        "by_metric": {
            str(metric): intervals(group) for metric, group in values.groupby("metric")
        },
    }


def _safe_cli_path(value: str) -> Path:
    """Resolve a CLI path only when it remains within the working directory."""
    resolved = os.path.realpath(value)
    base_dir = os.path.realpath(os.getcwd())  # noqa: PTH109
    if resolved != base_dir and not resolved.startswith(base_dir + os.sep):
        msg = "paths must not escape the working directory"
        raise argparse.ArgumentTypeError(msg)
    return Path(resolved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=_safe_cli_path, default=_safe_cli_path("wetbulb_data_csv")
    )
    parser.add_argument(
        "--annual-cache",
        type=_safe_cli_path,
        default=_safe_cli_path("forecast_inputs/annual_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=_safe_cli_path,
        default=_safe_cli_path("forecast_inputs/backtest_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the operational and oracle diagnostics and write a machine-readable report."""
    args = _parse_args()
    annual = annual_metrics(
        _safe_cli_path(args.data_root), _safe_cli_path(args.annual_cache)
    )
    observations = cast(
        "pd.Series",
        pd.read_csv("forecast_inputs/gmst_observations.csv").set_index("year")[
            "anomaly"
        ],
    )
    groups = pd.read_csv("forecast_inputs/forecast_station_groups.csv")
    station_groups = dict(
        zip(groups["location_id"], groups["station_group"], strict=True)
    )
    all_origins = range(CALIBRATION_ORIGINS.start, ORIGINS.stop)
    operational_raw = _forecast_cases(
        annual,
        observations,
        station_groups,
        oracle=False,
        origins=all_origins,
    )
    operational = calibrate_intervals(operational_raw)
    oracle = calibrate_intervals(
        _forecast_cases(
            annual,
            observations,
            station_groups,
            oracle=True,
            origins=all_origins,
        )
    )
    passed, gate = evaluate_gate(operational)
    oracle_passed, oracle_gate = evaluate_gate(oracle)
    production_factors = calibration_factors(operational_raw, 2025)
    calibration_rows = [
        {"metric": metric, "calibration_factor": factor}
        for metric, factor in production_factors.items()
    ]
    pd.DataFrame(calibration_rows).to_csv(
        "forecast_inputs/forecast_interval_calibration.csv", index=False
    )
    report = {
        "passed": passed,
        "operational": gate,
        "bootstrap_station_group": bootstrap_diagnostics(operational),
        "seasonal_diagnostics": diagnostic_summary(operational),
        "production_calibration_factors": production_factors,
        "oracle_diagnostic": {"passed": oracle_passed, **oracle_gate},
    }
    _safe_cli_path(args.output).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
