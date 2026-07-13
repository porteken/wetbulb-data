"""Compute wet-bulb temperature via the Stull (2011) and Davies-Jones (2008) methods."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

np: Any = cast("Any", import_module("numpy"))
type Dyn = Any

_STULL_C1 = 0.151977
_STULL_C2 = 8.313659
_STULL_C3 = 1.676331
_STULL_C4 = 0.00391838
_STULL_C5 = 1.5
_STULL_C6 = 0.023101
_STULL_OFFSET = 4.686035

# Bolton (1980) constants for saturation vapor pressure, the LCL temperature
# (eq. 15), and equivalent potential temperature (eq. 39). These feed the
# Davies-Jones (2008) pseudoadiabatic wet-bulb calculation below.
_EPSILON = 0.622  # ratio of gas constants, dry air / water vapor
_BOLTON_ES_A = 6.112  # hPa
_BOLTON_ES_B = 17.67
_BOLTON_ES_C = 243.5  # deg C
_KELVIN_OFFSET = 273.15
_P0_HPA = 1000.0
_LCL_A = 56.0
_LCL_B = 800.0
_THETA_E_KAPPA = 0.2854
_THETA_E_R_COEF = 0.28
_THETA_E_EXP_A = 3.376
_THETA_E_EXP_B = 0.00254
_THETA_E_EXP_C = 0.81
_DAVIES_JONES_BISECTION_ITERS = 30


def wetbulb_stull(tair: object = 20.0, rh: object = 50.0) -> object:
    """Compute wet-bulb temperature (deg C) from air temperature (deg C) and RH (%)."""
    inputs = [np.atleast_1d(x).astype(float) for x in (tair, rh)]
    is_scalar = bool(len(inputs[0]) == 1)

    max_len = max(len(a) for a in inputs)
    for a in inputs:
        if len(a) != 1 and len(a) != max_len:
            msg = (
                "Length of inputs differ! Single value or vectors of same "
                "length required."
            )
            raise ValueError(msg)

    t = np.broadcast_to(inputs[0], max_len)
    rh_pct = np.broadcast_to(inputs[1], max_len)

    tw = (
        t * np.arctan(_STULL_C1 * np.sqrt(rh_pct + _STULL_C2))
        + np.arctan(t + rh_pct)
        - np.arctan(rh_pct - _STULL_C3)
        + _STULL_C4 * rh_pct**_STULL_C5 * np.arctan(_STULL_C6 * rh_pct)
        - _STULL_OFFSET
    )

    if is_scalar:
        return float(np.asarray(tw, dtype="float64").reshape(-1)[0])
    return tw


def _broadcast_inputs(values: list[object]) -> tuple[list[Dyn], bool]:
    arrays = [np.atleast_1d(x).astype(float) for x in values]
    is_scalar = bool(len(arrays[0]) == 1) and all(len(a) == 1 for a in arrays)

    max_len = max(len(a) for a in arrays)
    for a in arrays:
        if len(a) != 1 and len(a) != max_len:
            msg = (
                "Length of inputs differ! Single value or vectors of same "
                "length required."
            )
            raise ValueError(msg)

    return [np.broadcast_to(a, max_len).astype(float) for a in arrays], is_scalar


def _saturation_vapor_pressure_hpa(temp_c: Dyn) -> Dyn:
    """Bolton (1980) eq. 10 saturation vapor pressure over liquid water."""
    return _BOLTON_ES_A * np.exp(
        _BOLTON_ES_B * temp_c / (temp_c + _BOLTON_ES_C),
    )


def _equivalent_potential_temperature_k(
    temp_k: Dyn,
    lcl_temp_k: Dyn,
    pressure_hpa: Dyn,
    mixing_ratio: Dyn,
) -> Dyn:
    """Bolton (1980) eq. 39 equivalent potential temperature (deg K)."""
    return (
        temp_k
        * (_P0_HPA / pressure_hpa)
        ** (_THETA_E_KAPPA * (1 - _THETA_E_R_COEF * mixing_ratio))
        * np.exp(
            (_THETA_E_EXP_A / lcl_temp_k - _THETA_E_EXP_B)
            * mixing_ratio
            * 1000.0
            * (1 + _THETA_E_EXP_C * mixing_ratio),
        )
    )


def wetbulb_davies_jones(
    tair_k: object = 293.15,
    qair: object = 0.01,
    psurf_pa: object = 101325.0,
) -> object:
    """Compute wet-bulb temperature (deg C) from 2m air temperature (K), specific humidity (kg/kg), and surface pressure (Pa).

    Follows the pseudoadiabatic equivalent-potential-temperature approach of
    Davies-Jones (2008): equivalent potential temperature is conserved along
    a pseudoadiabat, so the wet-bulb temperature Tw at pressure p is the
    temperature at which a saturated parcel at p has the same equivalent
    potential temperature as the actual (unsaturated) parcel. Bolton (1980)
    supplies the saturation vapor pressure, LCL temperature, and equivalent
    potential temperature formulas; the resulting single-variable equation is
    solved by bisection (bracketed between dewpoint and air temperature,
    where the target is guaranteed to lie) rather than reproducing
    Davies-Jones's fast empirical polynomial fit, since bisection converges
    to the same physical solution to well under 0.01 K without depending on
    unverified fit coefficients.
    """
    (temp_k, q, pressure_pa), is_scalar = _broadcast_inputs(
        [tair_k, qair, psurf_pa],
    )

    pressure_hpa = pressure_pa / 100.0

    with np.errstate(invalid="ignore", divide="ignore"):
        vapor_pressure_hpa = q * pressure_hpa / (_EPSILON + (1 - _EPSILON) * q)
        invalid = (
            ~np.isfinite(temp_k)
            | ~np.isfinite(vapor_pressure_hpa)
            | (vapor_pressure_hpa <= 0)
            | (vapor_pressure_hpa >= pressure_hpa)
        )
        vapor_pressure_hpa = np.where(invalid, np.nan, vapor_pressure_hpa)

        mixing_ratio = (
            _EPSILON * vapor_pressure_hpa / (pressure_hpa - vapor_pressure_hpa)
        )

        log_e_ratio = np.log(vapor_pressure_hpa / _BOLTON_ES_A)
        dewpoint_c = _BOLTON_ES_C * log_e_ratio / (_BOLTON_ES_B - log_e_ratio)
        dewpoint_k = dewpoint_c + _KELVIN_OFFSET

        lcl_temp_k = (
            1.0 / (1.0 / (dewpoint_k - _LCL_A) + np.log(temp_k / dewpoint_k) / _LCL_B)
            + _LCL_A
        )
        theta_e_target = _equivalent_potential_temperature_k(
            temp_k,
            lcl_temp_k,
            pressure_hpa,
            mixing_ratio,
        )

        lo = np.where(invalid, np.nan, dewpoint_k)
        hi = np.where(invalid, np.nan, temp_k)
        for _ in range(_DAVIES_JONES_BISECTION_ITERS):
            mid = 0.5 * (lo + hi)
            mid_c = mid - _KELVIN_OFFSET
            es_mid = _saturation_vapor_pressure_hpa(mid_c)
            rs_mid = _EPSILON * es_mid / (pressure_hpa - es_mid)
            # A saturated parcel's LCL is itself, so eq. 39 is evaluated
            # with lcl_temp_k = mid.
            theta_e_mid = _equivalent_potential_temperature_k(
                mid,
                mid,
                pressure_hpa,
                rs_mid,
            )
            below_target = theta_e_mid < theta_e_target
            lo = np.where(below_target, mid, lo)
            hi = np.where(below_target, hi, mid)

        wetbulb_k = 0.5 * (lo + hi)
        result = wetbulb_k - _KELVIN_OFFSET

    if is_scalar:
        return float(np.asarray(result, dtype="float64").reshape(-1)[0])
    return result
