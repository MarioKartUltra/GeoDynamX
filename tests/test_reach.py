# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The reach of a scale (spec 22 §3.1): the |ψ| mass radius to eps of the analysing wavelet as the
transform samples it, converged when the 1024 and 2048 grids agree within 25%."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import fft_policy
from dynamix.core import reach as R
from dynamix.core.smoothers import smoother
from dynamix.core.wtmm_backend import compute_scales2d


def _finest(a_min):
    return compute_scales2d(1, 1, a_min=a_min)[0]


def _brute(name, params, s, n, eps=1e-3):
    """Every pixel sorted by radius, the tail summed from outside (the measurement script's way)."""
    k = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    p = np.fft.fftshift(fft_policy.double().ifft2(
        1j * (kx * s) * smoother(name, params).theta_hat(kx * s, ky * s)).real)
    y, x = np.indices(p.shape) - n // 2
    r = np.hypot(x, y).ravel()
    o = np.argsort(r)
    v = np.abs(p.ravel()[o])
    t = np.cumsum(v[::-1])[::-1] / v.sum()
    return float(r[o][np.flatnonzero(t > eps)[-1]])


CASES = [("mexican", None), ("gaussian", None), ("q_gaussian", {"q": 1.5}),
         ("frac_bspline", {"alpha": 3.0}), ("q_meyer", {"q": 1.5})]


@pytest.mark.parametrize("name,params", CASES)
@pytest.mark.parametrize("a_min", [0.5, 1.0, 2.0])
def test_measured_radius_matches_the_brute_force_mass_integral_within_a_pixel(name, params, a_min):
    s = _finest(a_min)
    items = tuple(sorted((params or {}).items()))
    got = R.measured_radius(name, items, s, 1e-3, 1024)
    assert abs(got - _brute(name, params, s, 1024)) <= 1.0


@pytest.mark.parametrize("name,params,a_min,converged", [
    ("mexican", None, 0.9, False), ("mexican", None, 1.0, True), ("mexican", None, 1.5, True),
    ("gaussian", None, 0.8, False), ("gaussian", None, 0.9, True),
    ("q_gaussian", {"q": 1.5}, 1.0, False), ("q_gaussian", {"q": 1.5}, 1.5, True),
    ("frac_bspline", {"alpha": 3.0}, 2.0, False), ("q_meyer", {"q": 1.5}, 0.5, True)])
def test_convergence_follows_the_smoother_not_eta(name, params, a_min, converged):
    got = R.reach(name, params, _finest(a_min))
    assert got.converged is converged
    assert (got.radius is not None) is converged
    assert got.grids == (R.GRID, 2 * R.GRID) and got.ratio is not None


def test_pinned_radii():
    assert R.reach("mexican", None, _finest(1.5)).radius == pytest.approx(10.77, abs=0.3)
    assert R.reach("gaussian", None, _finest(1.0)).radius == pytest.approx(6.40, abs=0.3)
    assert R.reach("q_meyer", {"q": 1.5}, _finest(1.0)).radius == pytest.approx(60.3, abs=0.5)


def test_scales_above_s_ref_scale_the_reference_radius():
    ref = R.reach("mexican", None, R.S_REF)
    big = R.reach("mexican", None, 4 * R.S_REF)
    assert big.radius == pytest.approx(4 * ref.radius) and big.ratio == ref.ratio


def test_a_refused_smoother_is_not_measured():
    got = R.reach("q_gaussian", {"q": 2.5}, _finest(1.0))
    assert got.radius is None and got.measured is None and not got.converged


def test_a_repeated_question_hits_the_cache():
    R.reach("mexican", None, _finest(1.25))
    before = R.measured_radius.cache_info().hits
    R.reach("mexican", None, _finest(1.25))
    assert R.measured_radius.cache_info().hits >= before + 2


# ------------------------------- the check at the run's own grid (the author, 2026-10-16 night)

def test_a_run_grid_beyond_2048_adds_a_third_grid():
    """The Mexican hat at a_min 1.0 converges on 1024/2048 (8.0/9.2) and not at 4096 (21)."""
    plain = R.reach("mexican", None, _finest(1.0))
    run = R.reach("mexican", None, _finest(1.0), run_grid=4000)
    assert plain.converged and plain.grids == (1024, 2048) and plain.ratio_run is None
    assert run.grids == (1024, 2048, 4096) and run.ratio_run == pytest.approx(2.3, abs=0.2)
    assert not run.converged and run.radius is None and run.measured > plain.measured


def test_a_run_grid_within_2048_adds_nothing():
    run = R.reach("mexican", None, _finest(1.0), run_grid=1500)
    assert run.grids == (1024, 2048) and run.ratio_run is None and run.converged


def test_a_reach_that_holds_at_the_run_grid_takes_its_radius_there():
    run = R.reach("mexican", None, _finest(1.5), run_grid=3000)
    assert run.converged and run.grids[-1] == 4096 and run.ratio_run == pytest.approx(1.0, abs=0.05)
    assert run.radius == pytest.approx(10.77, abs=0.3)


def test_the_run_grid_is_capped(monkeypatch):
    monkeypatch.setattr(R, "RUN_GRID_CAP", 4096)
    assert R.reach("mexican", None, _finest(1.5), run_grid=20000).grids[-1] == 4096


def test_a_scale_that_failed_on_1024_and_2048_is_not_measured_again():
    run = R.reach("mexican", None, _finest(0.5), run_grid=4000)
    assert run.grids == (1024, 2048) and not run.converged
