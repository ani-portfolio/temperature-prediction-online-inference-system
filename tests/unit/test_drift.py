"""PSI, with the same bins and clipping as the offline evaluation, so its 0.25 threshold means the same live."""

import numpy as np

from src.drift import psi

RNG = np.random.default_rng(0)


def test_the_same_distribution_has_a_psi_near_0():
    assert psi(RNG.normal(size=5000), RNG.normal(size=5000)) < 0.02


def test_a_clear_shift_is_above_the_alert_threshold():
    assert psi(RNG.normal(size=5000), RNG.normal(1.0, size=5000)) > 0.25


def test_constant_reference_data_has_a_psi_of_0():
    assert psi(np.ones(100), RNG.normal(size=100)) == 0.0


def test_missing_values_are_left_out():
    a = np.r_[RNG.normal(size=2000), [np.nan] * 50]
    assert psi(a, a) < 1e-9
