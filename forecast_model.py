"""Reference implementation of the climate-informed wet-bulb forecast model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLOPE_VARIANCE_FLOOR = 1e-8
P80_Z = 1.2816
MIN_LINEAR_OBSERVATIONS = 3


@dataclass(frozen=True)
class LinearFit:
    """Sufficient statistics for a centered ordinary least-squares fit."""

    intercept_mean: float
    predictor_mean: float
    slope: float
    slope_variance: float
    residual_variance: float
    n: int


@dataclass(frozen=True)
class PooledPrior:
    """DerSimonian-Laird random-effects slope prior."""

    mean: float
    variance: float
    tau_squared: float
    groups: int
    fallback: str | None = None


@dataclass(frozen=True)
class Forecast:
    """One unrounded predictive forecast."""

    point: float
    lower: float
    upper: float
    slope: float
    slope_variance: float


def fit_linear(predictor: np.ndarray, response: np.ndarray) -> LinearFit | None:
    """Fit a centered line, returning ``None`` for unusable samples."""
    x = np.asarray(predictor, dtype=float)
    y = np.asarray(response, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)
    if n < MIN_LINEAR_OBSERVATIONS:
        return None

    xbar = float(x.mean())
    ybar = float(y.mean())
    centered_x = x - xbar
    sxx = float(centered_x @ centered_x)
    if sxx <= 0:
        return None

    slope = float(centered_x @ (y - ybar) / sxx)
    residuals = y - (ybar + slope * centered_x)
    residual_variance = float(residuals @ residuals / (n - 2))
    slope_variance = max(residual_variance / sxx, SLOPE_VARIANCE_FLOOR)
    return LinearFit(
        intercept_mean=ybar,
        predictor_mean=xbar,
        slope=slope,
        slope_variance=slope_variance,
        residual_variance=max(residual_variance, 0.0),
        n=n,
    )


def dersimonian_laird(slopes: np.ndarray, variances: np.ndarray) -> PooledPrior:
    """Pool independent group representatives with safe fixed-effect fallbacks."""
    effects = np.asarray(slopes, dtype=float)
    variances = np.maximum(np.asarray(variances, dtype=float), SLOPE_VARIANCE_FLOOR)
    valid = np.isfinite(effects) & np.isfinite(variances)
    effects = effects[valid]
    variances = variances[valid]
    k = len(effects)
    if k == 0:
        message = "At least one usable station group is required"
        raise ValueError(message)

    weights = 1.0 / variances
    weight_sum = float(weights.sum())
    fixed_mean = float(weights @ effects / weight_sum)
    fixed_variance = max(1.0 / weight_sum, SLOPE_VARIANCE_FLOOR)
    if k == 1:
        return PooledPrior(fixed_mean, fixed_variance, 0.0, k, "one_group_full_pooling")

    q = float(weights @ np.square(effects - fixed_mean))
    denominator = weight_sum - float(weights @ weights) / weight_sum
    if denominator <= 0:
        return PooledPrior(
            fixed_mean, fixed_variance, 0.0, k, "nonpositive_denominator"
        )

    tau_squared = max(0.0, (q - (k - 1)) / denominator)
    random_weights = 1.0 / (variances + tau_squared)
    random_sum = float(random_weights.sum())
    return PooledPrior(
        mean=float(random_weights @ effects / random_sum),
        variance=max(1.0 / random_sum, SLOPE_VARIANCE_FLOOR),
        tau_squared=tau_squared,
        groups=k,
        fallback=None if tau_squared > 0 else "fixed_effect",
    )


def shrink_slope(fit: LinearFit, prior: PooledPrior) -> tuple[float, float]:
    """Combine a location slope with the pooled prior."""
    if prior.tau_squared <= 0:
        return prior.mean, prior.variance

    local_precision = 1.0 / max(fit.slope_variance, SLOPE_VARIANCE_FLOOR)
    prior_precision = 1.0 / max(prior.tau_squared, SLOPE_VARIANCE_FLOOR)
    variance = 1.0 / (local_precision + prior_precision)
    slope = variance * ((fit.slope * local_precision) + (prior.mean * prior_precision))
    return slope, max(variance, SLOPE_VARIANCE_FLOOR)


def forecast(
    fit: LinearFit,
    prior: PooledPrior,
    predictor: float,
    *,
    predictor_variance: float = 0.0,
) -> Forecast:
    """Create the nominal 10th/90th predictive interval from the model formula."""
    slope, slope_variance = shrink_slope(fit, prior)
    delta = float(predictor) - fit.predictor_mean
    point = fit.intercept_mean + (slope * delta)
    variance = (
        fit.residual_variance * (1.0 + (1.0 / fit.n))
        + (slope_variance * delta * delta)
        + (slope * slope * max(float(predictor_variance), 0.0))
    )
    margin = P80_Z * np.sqrt(max(variance, 0.0))
    return Forecast(point, point - margin, point + margin, slope, slope_variance)
