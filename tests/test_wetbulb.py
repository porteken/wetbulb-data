"""Tests for the Stull and Davies-Jones wet-bulb temperature approximations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from wetbulb import wetbulb_davies_jones, wetbulb_stull

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SEA_LEVEL_PA = 101325.0
_KELVIN_OFFSET = 273.15


def _specific_humidity_from_rh(
    temp_c: float, rh_pct: float, pressure_pa: float
) -> float:
    """Invert the Bolton (1980) saturation curve to get q from T, RH, p."""
    p_hpa = pressure_pa / 100.0
    es_hpa = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    e_hpa = (rh_pct / 100.0) * es_hpa
    epsilon = 0.622
    return float(epsilon * e_hpa / (p_hpa - (1 - epsilon) * e_hpa))


def _as_float(value: object) -> float:
    return float(np.asarray(value, dtype=float).item())


def _as_float_array(value: object) -> NDArray[np.float64]:
    return np.asarray(value, dtype=float)


class TestWetbulbStull:
    def test_scalar_output(self) -> None:
        result = wetbulb_stull(20.0, 50.0)
        assert isinstance(result, float)

    def test_known_value_20_50(self) -> None:
        assert _as_float(wetbulb_stull(20.0, 50.0)) == pytest.approx(13.699, abs=0.01)

    def test_known_value_25_50(self) -> None:
        assert _as_float(wetbulb_stull(25.0, 50.0)) == pytest.approx(17.998, abs=0.01)

    def test_known_value_0_50(self) -> None:
        assert _as_float(wetbulb_stull(0.0, 50.0)) == pytest.approx(-3.498, abs=0.01)

    def test_wetbulb_below_air_temp_when_not_saturated(self) -> None:
        assert _as_float(wetbulb_stull(30.0, 70.0)) < 30.0

    def test_wetbulb_near_air_temp_at_saturation(self) -> None:
        result = _as_float(wetbulb_stull(25.0, 100.0))
        assert result == pytest.approx(25.0, abs=0.5)

    def test_monotonic_with_relative_humidity(self) -> None:
        rh = np.array([20.0, 50.0, 80.0])
        tair = np.full_like(rh, 25.0)
        arr = _as_float_array(wetbulb_stull(tair, rh))
        assert arr[0] < arr[1] < arr[2]

    def test_vector_input(self) -> None:
        tair = np.array([20.0, 25.0, 30.0])
        rh = np.array([50.0, 50.0, 50.0])
        result = _as_float_array(wetbulb_stull(tair, rh))
        assert len(result) == 3

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="Length"):
            wetbulb_stull(np.array([20.0, 25.0]), np.array([50.0, 60.0, 70.0]))

    def test_scalar_broadcast_with_vector(self) -> None:
        tair = np.array([20.0, 25.0, 30.0])
        result = _as_float_array(wetbulb_stull(tair, 50.0))
        assert len(result) == 3

    def test_nan_propagates(self) -> None:
        result = _as_float(wetbulb_stull(float("nan"), 50.0))
        assert np.isnan(result)


class TestWetbulbDaviesJones:
    def test_scalar_output(self) -> None:
        q = _specific_humidity_from_rh(20.0, 50.0, _SEA_LEVEL_PA)
        result = wetbulb_davies_jones(20.0 + _KELVIN_OFFSET, q, _SEA_LEVEL_PA)
        assert isinstance(result, float)

    def test_known_value_20_50(self) -> None:
        q = _specific_humidity_from_rh(20.0, 50.0, _SEA_LEVEL_PA)
        result = _as_float(
            wetbulb_davies_jones(20.0 + _KELVIN_OFFSET, q, _SEA_LEVEL_PA),
        )
        assert result == pytest.approx(13.7, abs=0.5)

    def test_known_value_25_50(self) -> None:
        q = _specific_humidity_from_rh(25.0, 50.0, _SEA_LEVEL_PA)
        result = _as_float(
            wetbulb_davies_jones(25.0 + _KELVIN_OFFSET, q, _SEA_LEVEL_PA),
        )
        assert result == pytest.approx(18.0, abs=0.5)

    def test_known_value_0_50(self) -> None:
        q = _specific_humidity_from_rh(0.0, 50.0, _SEA_LEVEL_PA)
        result = _as_float(
            wetbulb_davies_jones(0.0 + _KELVIN_OFFSET, q, _SEA_LEVEL_PA),
        )
        assert result == pytest.approx(-3.5, abs=1.0)

    def test_dewpoint_le_wetbulb_le_airtemp(self) -> None:
        rng = np.random.default_rng(42)
        temps_c = rng.uniform(-10, 45, size=50)
        rhs = rng.uniform(5, 99, size=50)
        pressures_pa = rng.uniform(70000, 103000, size=50)
        q = np.array(
            [
                _specific_humidity_from_rh(t, rh, p)
                for t, rh, p in zip(temps_c, rhs, pressures_pa, strict=True)
            ],
        )
        tw = _as_float_array(
            wetbulb_davies_jones(temps_c + _KELVIN_OFFSET, q, pressures_pa),
        )
        assert np.all(tw <= temps_c + 1e-6)

    def test_wetbulb_near_air_temp_at_saturation(self) -> None:
        q = _specific_humidity_from_rh(25.0, 100.0, _SEA_LEVEL_PA)
        result = _as_float(
            wetbulb_davies_jones(25.0 + _KELVIN_OFFSET, q, _SEA_LEVEL_PA),
        )
        assert result == pytest.approx(25.0, abs=0.2)

    def test_monotonic_with_relative_humidity(self) -> None:
        rhs = np.array([20.0, 50.0, 80.0])
        q = np.array(
            [_specific_humidity_from_rh(25.0, rh, _SEA_LEVEL_PA) for rh in rhs]
        )
        tair = np.full_like(rhs, 25.0 + _KELVIN_OFFSET)
        pressure = np.full_like(rhs, _SEA_LEVEL_PA)
        arr = _as_float_array(wetbulb_davies_jones(tair, q, pressure))
        assert arr[0] < arr[1] < arr[2]

    def test_lower_pressure_increases_wetbulb_depression(self) -> None:
        """Lower pressure means a larger wet-bulb depression at fixed T/RH.

        Evaporative cooling is more efficient at lower surface pressure, so
        the depression (T - Tw) should be larger than at sea level for the
        same temperature and relative humidity.
        """
        q_sea_level = _specific_humidity_from_rh(25.0, 50.0, _SEA_LEVEL_PA)
        denver_pa = 85000.0
        q_denver = _specific_humidity_from_rh(25.0, 50.0, denver_pa)
        tw_sea_level = _as_float(
            wetbulb_davies_jones(25.0 + _KELVIN_OFFSET, q_sea_level, _SEA_LEVEL_PA),
        )
        tw_denver = _as_float(
            wetbulb_davies_jones(25.0 + _KELVIN_OFFSET, q_denver, denver_pa),
        )
        assert (25.0 - tw_denver) > (25.0 - tw_sea_level)

    def test_agrees_with_stull_within_tolerance(self) -> None:
        temps_c = np.array([0.0, 10.0, 20.0, 25.0, 30.0, 35.0])
        rhs = np.full_like(temps_c, 50.0)
        pressure = np.full_like(temps_c, _SEA_LEVEL_PA)
        q = np.array(
            [
                _specific_humidity_from_rh(t, rh, _SEA_LEVEL_PA)
                for t, rh in zip(temps_c, rhs, strict=True)
            ],
        )
        stull_result = _as_float_array(wetbulb_stull(temps_c, rhs))
        dj_result = _as_float_array(
            wetbulb_davies_jones(temps_c + _KELVIN_OFFSET, q, pressure),
        )
        assert np.all(np.abs(stull_result - dj_result) < 1.5)

    def test_vector_input(self) -> None:
        temps_c = np.array([20.0, 25.0, 30.0])
        rhs = np.array([50.0, 50.0, 50.0])
        pressure = np.full_like(temps_c, _SEA_LEVEL_PA)
        q = np.array(
            [
                _specific_humidity_from_rh(t, rh, _SEA_LEVEL_PA)
                for t, rh in zip(temps_c, rhs, strict=True)
            ],
        )
        result = _as_float_array(
            wetbulb_davies_jones(temps_c + _KELVIN_OFFSET, q, pressure),
        )
        assert len(result) == 3

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="Length"):
            wetbulb_davies_jones(
                np.array([293.15, 298.15]),
                np.array([0.007, 0.008, 0.009]),
                _SEA_LEVEL_PA,
            )

    def test_nan_propagates(self) -> None:
        result = _as_float(wetbulb_davies_jones(float("nan"), 0.007, _SEA_LEVEL_PA))
        assert np.isnan(result)
