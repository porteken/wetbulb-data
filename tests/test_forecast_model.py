"""Tests for the numeric GMST forecast reference model."""

from __future__ import annotations

import numpy as np
import pytest

from forecast_model import (
    P80_Z,
    SLOPE_VARIANCE_FLOOR,
    dersimonian_laird,
    fit_linear,
    forecast,
)


def test_linear_fit_and_noiseless_forecast_are_exact() -> None:
    predictor = np.array([0.0, 1.0, 2.0, 3.0])
    response = 10.0 + (2.0 * predictor)
    fit = fit_linear(predictor, response)

    assert fit is not None
    assert fit.slope == pytest.approx(2.0)
    prior = dersimonian_laird(np.array([2.0]), np.array([1e-12]))
    result = forecast(fit, prior, 4.0)

    assert result.point == pytest.approx(18.0)
    assert result.lower == pytest.approx(18.0, abs=0.001)
    assert result.upper == pytest.approx(18.0, abs=0.001)


def test_pooling_floors_variances_and_handles_one_group() -> None:
    pooled = dersimonian_laird(np.array([1.5]), np.array([0.0]))

    assert pooled.mean == pytest.approx(1.5)
    assert pooled.variance >= SLOPE_VARIANCE_FLOOR
    assert pooled.fallback == "one_group_full_pooling"


def test_prediction_interval_uses_nominal_tenth_ninetieth_z_score() -> None:
    fit = fit_linear(np.arange(5, dtype=float), np.array([1.0, 2.0, 1.0, 2.0, 1.0]))
    assert fit is not None
    prior = dersimonian_laird(np.array([fit.slope]), np.array([fit.slope_variance]))

    result = forecast(fit, prior, 6.0, predictor_variance=0.25)

    assert result.upper - result.point == pytest.approx(result.point - result.lower)
    assert result.upper - result.point > 0
    assert pytest.approx(1.2816) == P80_Z
