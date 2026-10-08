# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The two loops on the mirror-extended working image."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("numba")
pywt = pytest.importorskip("pywt")
pytest.importorskip("scipy")

from dynamix.core import dwt  # noqa: E402
from dynamix.core.inpaint import correct as co  # noqa: E402
from dynamix.core.inpaint import forward as fw  # noqa: E402
from dynamix.core.inpaint import priors as pr  # noqa: E402
from dynamix.core.inpaint import readouts as ro  # noqa: E402
from dynamix.core.inpaint import solver as so  # noqa: E402
from dynamix.core.wtmm_backend import ComputeCancelled  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_microcanonical import _cascade2d  # noqa: E402


def _problem(y, gap, J, *, heldout=None):
    """The target term alone on the working image of ``y`` (NaN inside ``gap``)."""
    ny, nx = y.shape
    shape = dwt.working_shape(y.shape, J)
    known_g = ~gap
    x0 = co.prefill(np.where(gap, np.nan, y), gap)
    held_g = np.zeros(gap.shape, dtype=bool) if heldout is None else heldout
    y_w = dwt.mirror(np.where(gap & ~held_g, x0, y), shape)   # held-out pixels keep their data
    known = dwt.mirror(known_g.astype(np.float64), shape) > 0.5
    region = ~known
    ring = co.boundary_ring(region)
    w = known.astype(np.float64)
    terms = [fw.Term(fw.Identity(), y_w, w)]
    held = None if heldout is None else dwt.mirror(heldout.astype(np.float64), shape) > 0.5
    L = fw.lipschitz(terms, shape, np.random.default_rng(0))
    return so.Problem(terms, y_w, known, ring, region, L, heldout=held), dwt.mirror(x0, shape)


def _gap(shape=(128, 128), box=(48, 80, 48, 80)):
    m = np.zeros(shape, dtype=bool)
    m[box[0]:box[1], box[2]:box[3]] = True
    return m


def test_fista_lowers_the_cost_keeps_the_best_iterate_and_reports_its_stop():
    y = np.random.default_rng(1).standard_normal((96, 96)).cumsum(0).cumsum(1)
    gap = _gap((96, 96), (30, 60, 30, 60))
    problem, x0 = _problem(y, gap, 3)
    clear1 = np.ones(tuple(n // 2 for n in problem.y.shape), np.uint8)       # level 1 of the 192^2 working image
    prior = pr.Sparsity("sym7", 3, lam=pr.auto_lambda(problem.y, clear1, "sym7", 0.2),
                        rng=np.random.default_rng(0))
    x, hist = so.fista(problem, prior, x0, so.Controls(n_iter=60, tol=1e-6))
    assert hist.iterations == 60 and hist.stop == "cap"
    assert len(hist.cost) == 6 and hist.cost[-1] <= hist.cost[0]
    assert hist.returned == int(np.argmin(hist.score)) + 1
    assert np.isfinite(x).all() and x.shape == problem.y.shape
    x2, hist2 = so.fista(problem, prior, x0, so.Controls(n_iter=500, tol=5e-2))
    assert hist2.stop == "converged" and hist2.iterations < 500


def test_fista_scores_by_the_held_out_blocks_when_there_are_any():
    y = np.random.default_rng(2).standard_normal((96, 96)).cumsum(0).cumsum(1)
    gap = _gap((96, 96), (30, 60, 30, 60))
    held = ro.heldout(~gap, 4, 8, 8, np.random.default_rng(3))
    problem, x0 = _problem(y, gap | held, 3, heldout=held)
    prior = pr.Sparsity("sym7", 3, 1.0, np.random.default_rng(0))
    x, hist = so.fista(problem, prior, x0, so.Controls(n_iter=20, tol=1e-9, correct=False))
    assert len(hist.score) == 20 and all(np.isfinite(hist.score))
    assert problem.score(x) == pytest.approx(ro.errors(x, problem.y, problem.heldout)["rms_rel"])
    assert problem.score(x) == min(hist.score)              # the best iterate by held-out error


def test_fista_cancels_through_the_callback():
    y = np.random.default_rng(1).standard_normal((64, 64)).cumsum(0)
    problem, x0 = _problem(y, _gap((64, 64), (20, 40, 20, 40)), 3)
    prior = pr.Sparsity("sym7", 3, 1.0, np.random.default_rng(0))
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(ComputeCancelled):
        so.fista(problem, prior, x0, so.Controls(n_iter=50), cancel=cancel)


def test_the_prox_commutes_with_a_shift_by_a_multiple_of_the_dyadic_cell():
    """The spec's shift test, stated where it holds exactly: cycle spinning draws the same
    shifts for the same seed and the periodized pyramid is invariant under a shift by 2**J,
    so the prox of a rolled working image is the rolled prox. (The full loop's covariance is
    not pinned: at the cap the two runs return different iterates; measured ratios 0.26-0.6.)"""
    x = dwt.mirror(np.random.default_rng(4).standard_normal((96, 96)).cumsum(0).cumsum(1))
    a = pr.Sparsity("sym7", 3, 1.0, np.random.default_rng(0)).prox(x, 0.5)
    b = pr.Sparsity("sym7", 3, 1.0, np.random.default_rng(0)).prox(np.roll(x, (8, 16), axis=(0, 1)), 0.5)
    np.testing.assert_allclose(np.roll(a, (8, 16), axis=(0, 1)), b, atol=1e-9)


def test_the_plain_cascade_loop_does_not_inflate_the_region():
    """Finding 19: with a parent whose eq. 15 interval is empty left alone, three passes of
    inference and correction keep the region's variance error near the pre-fill's (measured
    0.92, 0.90, 0.91 against 1.006); the spec's hull scan gave 2.5, 226, 482."""
    gap = _gap()
    dens, prior = _haar_cascade(gap, 6)
    problem, x0 = _problem(dens, gap, 6)
    truth = dwt.mirror(dens)
    start = ro.errors(x0, truth, problem.region)["var_err"]
    x = x0.copy()
    for _ in range(3):
        x = prior.infer(x)
        x = x - co.correct(x, problem.y, problem.known, problem.ring, pad=0)
        assert ro.errors(x, truth, problem.region)["var_err"] < 1.5 * start


def _haar_cascade(gap, J, synthesis=False):
    dens, _ = _cascade2d()
    coeffs = pywt.wavedec2(dwt.mirror(dens), "haar", mode="periodization", level=J)
    gap_w = dwt.mirror(gap.astype(np.float64)) > 0.5
    clear, rho = [], []
    a = gap_w.astype(np.float64)
    for j in range(1, J + 1):
        s = 2 ** j
        hit = gap_w.reshape(256 // s, s, 256 // s, s).any(axis=(1, 3))
        clear.append(np.ascontiguousarray((~hit).astype(np.uint8)))
        a, _d = pywt.dwt2(a, "haar", mode="periodization")
        rho.append(np.clip(a / 2.0 ** j, 0.0, 1.0))
    ab = pr.Abacus.from_pyramid(coeffs, clear, "haar")
    prior = pr.Cascade("haar", J, ab, rho, synthesis=synthesis, rng=np.random.default_rng(0),
                       pyramid=lambda x: pywt.wavedec2(x, "haar", mode="periodization", level=J),
                       inverse=lambda c: pywt.waverec2(c, "haar", mode="periodization"))
    return dens, prior


def test_the_cascade_loop_meets_the_papers_stop_rule_within_the_cap():
    gap = _gap()
    dens, prior = _haar_cascade(gap, 6)
    problem, x0 = _problem(dens, gap, 6)
    x, hist = so.cascade_loop(problem, prior, x0, so.Controls(n_iter=20, stop_var_err=0.01))
    assert hist.stop in ("converged", "cap")
    assert ro.errors(x, problem.y, problem.known)["var_err"] < 0.01
    assert np.isfinite(x).all()
    assert hist.returned >= 1 and len(hist.score) == hist.iterations


def test_the_cascade_loop_takes_a_gradient_step_on_extra_terms_first():
    gap = _gap()
    dens, prior = _haar_cascade(gap, 6)
    problem, x0 = _problem(dens, gap, 6)
    target = dwt.mirror(dens)
    extra = [fw.Term(fw.Identity(), target, 0.3 * problem.region.astype(np.float64))]
    plain, _ = so.cascade_loop(problem, prior, x0, so.Controls(n_iter=3, stop_var_err=0.0))
    guided, _ = so.cascade_loop(problem, prior, x0, so.Controls(n_iter=3, stop_var_err=0.0),
                                extra_terms=extra)
    assert ro.errors(guided, target, problem.region)["var_err"] < \
        ro.errors(plain, target, problem.region)["var_err"]
