# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The vendored q-Mexican hat's L2 constant A_q (Borges, Tsallis, Miranda & Andrade 2004,
J. Phys. A 37 9125, eqs. 17-18) through q = 1.

Eqs. 17-18 are ratios of Gamma functions whose arguments grow like 2/|q - 1|. Double-precision
Gamma overflows past an argument of ~171.6, so the literal ratio is NaN for 0.988 < q < 1.012
although the ratio itself is finite and smooth there. The constant is checked against the same
formula evaluated at 50 digits, and against its defining property ||psi_q||_2 = 1.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from dynamix._vendor.wtmm.wavelets import _borges_normalization, _q_mexican_hat

mpmath = pytest.importorskip("mpmath")

_ACROSS_ONE = (-0.9, -0.5, 0.0, 0.5, 0.9, 0.98, 0.99, 0.995, 0.999, 1.0 - 1e-8,
               1.0 + 1e-8, 1.001, 1.005, 1.01, 1.02, 1.5, 2.0, 2.5, 2.9)


def _eqs_17_18(q: float, beta: float) -> float:
    with mpmath.workdps(50):
        q, beta = mpmath.mpf(q), mpmath.mpf(beta)
        c = beta ** 0.25 / (mpmath.pi ** 0.25 * mpmath.sqrt(3))
        if q > 1:
            a = 2 * q / (q - 1)
            return float(c * mpmath.sqrt((q - 1) ** 2.5 * mpmath.gamma(a)
                                         / mpmath.gamma(a - 2.5)))
        a = 2 * q / (1 - q)
        return float(c * mpmath.sqrt(5 - q) * mpmath.sqrt(3 + q) / 2
                     * mpmath.sqrt(mpmath.sqrt(1 - q) * mpmath.gamma(a + 1.5)
                                   / mpmath.gamma(a + 1)))


@pytest.mark.parametrize("beta", (0.5, 1.0))
@pytest.mark.parametrize("q", _ACROSS_ONE)
def test_a_q_matches_eqs_17_18_at_50_digits(q, beta):
    assert _borges_normalization(q, beta) == pytest.approx(_eqs_17_18(q, beta), rel=1e-11)


def test_a_q_is_continuous_through_q_equal_one():
    a1 = _borges_normalization(1.0)
    for d in (1e-3, 1e-5, 1e-8, 1e-11):
        for q in (1.0 - d, 1.0 + d):
            assert abs(_borges_normalization(q) - a1) < d          # |dA_q/dq| ~ 0.14 at q = 1


@pytest.mark.parametrize("q", (0.99, 0.995, 1.005, 1.01))
def test_the_q_mexican_hat_has_unit_l2_norm_inside_the_old_nan_band(q):
    norm2 = 2.0 * quad(lambda x: _q_mexican_hat(np.array([x]), q)[0] ** 2, 0.0, np.inf,
                       limit=400)[0]
    assert norm2 == pytest.approx(1.0, rel=1e-8)
