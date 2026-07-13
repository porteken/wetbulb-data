"""Tests for the PET thermodynamic model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pet_corrected import PetConstants, c2k, pet_corrected, svp_murray

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _as_float(value: object) -> float:
    return float(np.asarray(value, dtype=float).item())


def _as_float_array(value: object) -> NDArray[np.float64]:
    return np.asarray(value, dtype=float)


class TestC2K:
    def test_freezing_point(self) -> None:
        assert np.isclose(_as_float(c2k(0.0)), 273.15, rtol=1e-09, atol=1e-09)

    def test_boiling_point(self) -> None:
        assert np.isclose(_as_float(c2k(100.0)), 373.15, rtol=1e-09, atol=1e-09)


class TestSvpMurray:
    def test_at_20c(self) -> None:
        result = svp_murray(c2k(20.0))
        assert 20 < _as_float(result) < 30


class TestPetConstants:
    def test_standard_atmosphere(self) -> None:
        c = PetConstants()
        assert np.isclose(c.po, 1013.25, rtol=1e-09, atol=1e-09)
        assert np.isclose(c.p, 1013.25, rtol=1e-09, atol=1e-09)


class TestPetCorrected:
    def test_scalar_output(self) -> None:
        result = pet_corrected(25.0, 30.0, 1.0, 50.0, icl=0.5)
        assert isinstance(result, float)

    def test_comfortable_conditions(self) -> None:
        result = pet_corrected(22.0, 22.0, 0.5, 50.0, icl=0.5)

        assert 10 < _as_float(result) < 40

    def test_hot_conditions(self) -> None:
        result = pet_corrected(40.0, 60.0, 0.5, 70.0, icl=0.5)
        assert _as_float(result) > 30

    def test_cold_conditions(self) -> None:
        result = pet_corrected(0.0, -5.0, 3.0, 60.0, icl=0.9)
        assert _as_float(result) < 10

    def test_vector_input(self) -> None:
        tair = np.array([20.0, 25.0, 30.0])
        t_mrt = np.array([25.0, 30.0, 35.0])
        v_air = np.array([1.0, 1.0, 1.0])
        rh = np.array([50.0, 50.0, 50.0])
        result = _as_float_array(pet_corrected(tair, t_mrt, v_air, rh, icl=0.5))
        assert len(result) == 3

    def test_vector_output_monotonic_with_temperature(self) -> None:
        """Verify PET increases with temperature and MRT."""
        tair = np.array([15.0, 25.0, 35.0])
        t_mrt = np.array([20.0, 30.0, 40.0])
        v_air = np.array([1.0, 1.0, 1.0])
        rh = np.array([50.0, 50.0, 50.0])
        arr = np.asarray(pet_corrected(tair, t_mrt, v_air, rh, icl=0.5), dtype=float)
        assert arr[0] < arr[1] < arr[2]

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="Length"):
            pet_corrected(
                np.array([20.0, 25.0]),
                np.array([25.0]),
                np.array([1.0, 1.0, 1.0]),
                np.array([50.0]),
                icl=0.5,
            )
