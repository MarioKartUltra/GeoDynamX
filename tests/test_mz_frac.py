# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The fractional-order Mallat-Zhong forward (core/mz_edges.py, §6.4 of the 2026-09-19
split plan): sign-preserving fractional refinement order over the UNTOUCHED mzlib oracle.

The oracle chain: cascade <-> analytic |psi_hat| = |w||sinc(w/4)|^(alpha+1) (tight, all
orders) <-> frac_bspline Part A2 real-space forms (exact at the integer anchors, where the
causal and symmetric variants coincide; truncation-bounded at fractional orders). Lambda:
the Table-II recipe -- discrete step-edge peak over theta_alpha(0) -- must REPRODUCE the
paper's own table at alpha=3 before it is trusted anywhere else."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import fbm2d
from dynamix.core import frac_bspline as fb
from dynamix.core import mz_edges, mzlib


def test_lambda_recipe_reproduces_table_ii_at_alpha_3():
    """The pin: lambda_j = (discrete dyadic step-edge modulus peak at level j) / theta(0).
    At alpha=3 this must give the paper's Table II to its own 2-decimal precision --
    measured unrounded: [1.5000, 1.1250, 1.0313, 1.0078, 1.0020]."""
    lam = [mz_edges.lam_frac(j, 3.0) for j in range(1, 6)]
    np.testing.assert_allclose(lam, [1.5000, 1.1250, 1.03125, 1.0078, 1.0020], atol=2e-4)
    for got, table in zip(lam, mzlib.LAMBDA_TABLE):
        assert abs(got - table) <= 0.0051, (got, table)   # the paper printed 2 decimals


def test_theta0_at_alpha_3_is_the_papers_4_3():
    assert abs(mz_edges._theta0_frac(3.0) - mzlib.THETA0) < 1e-4


def test_alpha3_forward_is_the_mzlib_filter_bank_exactly():
    """At alpha=3 the sign-preserving fractional filter IS cos^3, so the whole transform
    (S and every W pair) must match mzlib's -- compared at use_lambda=False because the
    frac path's numeric lambda deliberately keeps Table II's unrounded values (1.1250, not
    the paper's printed 1.12)."""
    f = fbm2d(64, 0.5, seed=9)
    S_ref, W_ref = mzlib.atrous2d_forward(f, 4, use_lambda=False)
    S_got, W_got = mz_edges.atrous2d_forward_frac(f, 4, 3.0, use_lambda=False)
    np.testing.assert_allclose(S_got, S_ref, rtol=1e-12, atol=1e-12)
    for (a1, a2), (b1, b2) in zip(W_got, W_ref):
        np.testing.assert_allclose(a1, b1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a2, b2, rtol=1e-12, atol=1e-12)


def test_cascade_transfer_is_the_finite_viete_product_exactly():
    """The level-j x-transfer of the cascade has the CLOSED trigonometric form
    ``|X_j(w)| = 4 A (A / (2^(j-1) |sin(w/2)|))^alpha`` with ``A = |sin(2^(j-1) w/2)|``
    (finite Viete product of the sign-preserving filters) -- exact at every order, the
    implementation pin. With increasing j it converges to the continuum
    ``|w| |sinc(w/4)|^(alpha+1)`` -- the symmetric fractional-spline derivative wavelet
    (measured: rel err 0.03 at j=2 falling monotonically to <1e-3 by j=5, every alpha)."""
    N = 2 ** 13
    w = 2 * np.pi * np.fft.fftfreq(N)
    s2 = np.abs(np.sin(w / 2)); s2[0] = 1.0
    for alpha in (1.5, 2.5, 3.0, 4.0):
        cont_errs = []
        for j in (2, 3, 4, 5):
            X = mzlib.Gf((2 ** (j - 1)) * w).astype(complex)
            for pp in range(j - 1):
                X = X * mz_edges._hf_frac((2 ** pp) * w, alpha)
            got = np.abs(X)
            A = np.abs(np.sin(2 ** (j - 1) * w / 2))
            exact = 4 * A * (A / (2 ** (j - 1) * s2)) ** alpha
            exact[0] = 0.0
            assert np.abs(got - exact).max() / got.max() < 1e-10, (alpha, j)
            wj = (2 ** j) * w
            cont = np.abs(wj) * np.abs(np.sinc(wj / 4 / np.pi)) ** (alpha + 1.0)
            m = np.abs(wj) < np.pi
            scale = got[m].max() / cont[m].max()
            cont_errs.append(np.abs(got[m] - scale * cont[m]).max() / got[m].max())
        assert all(a > b for a, b in zip(cont_errs, cont_errs[1:])), (alpha, cont_errs)
        assert cont_errs[0] < 0.06 and cont_errs[-1] < 0.005, (alpha, cont_errs)


def test_analytic_form_matches_part_a2_at_the_integer_anchors():
    """Closing the oracle chain: |FFT| of the SAMPLED Part A2 wavelet (4*beta'_alpha(2x),
    the M-Z theta = 2*beta(2x) identity) matches the analytic |w||sinc(w/4)|^(alpha+1)
    tightly at integer alpha (exact truncation), and to the documented ~10% truncation
    bound at a fractional order."""
    x = np.arange(-64, 64, 0.015625)
    w = 2 * np.pi * np.fft.fftfreq(x.size, d=0.015625)
    for alpha, tol in ((3.0, 1e-6), (5.0, 1e-9), (2.5, 0.1)):
        # The copied evaluator is only meant NEAR its support (cell 70 plots u in [-4,4]):
        # at fractional alpha the truncated series GROWS polynomially far outside it
        # (integer orders telescope to exact 0), so evaluate in a support window and zero
        # beyond -- the ~10% truncation bound is the fractional tolerance, same cause as
        # the mass deficit pinned in test_frac_bspline_port.
        u = 2.0 * x
        ucut = (alpha + 1.0) / 2.0 + 2.0    # measured: rel 0.033 here, 0.165 at +4 --
                                            # the series garbage starts right past support
        psi = np.zeros_like(u)
        win = np.abs(u) <= ucut
        psi[win] = fb._frac_bspline_derivative(u[win], alpha, 1.0)
        got = np.abs(np.fft.fft(psi))
        target = np.abs(w) * np.abs(np.sinc(w / 4 / np.pi)) ** (alpha + 1.0)
        m = np.abs(w) < 2 * np.pi
        scale = got[m].max() / target[m].max()
        rel = np.abs(got[m] - scale * target[m]).max() / got[m].max()
        assert rel < tol, (alpha, rel)


def test_frac_forward_kills_constants_and_returns_the_coarse():
    const = np.full((32, 32), 5.5)
    S, W = mz_edges.atrous2d_forward_frac(const, 3, 2.5)
    for W1, W2 in W:
        assert np.abs(W1).max() < 1e-10 and np.abs(W2).max() < 1e-10
    np.testing.assert_allclose(S, 5.5, rtol=1e-10)


def test_lambda_is_cached_and_orders_are_guarded():
    a = mz_edges.lam_frac(2, 2.5)
    assert mz_edges.lam_frac(2, 2.5) == a                 # cached, deterministic
    assert mz_edges.lam_frac(8, 2.5) > 0                  # deep levels available
    with pytest.raises(ValueError, match="alpha"):
        mz_edges.lam_frac(1, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        mz_edges.atrous2d_forward_frac(np.zeros((16, 16)), 2, -1.0)


# ---------------------------------------------------------------- analyze() / device wiring

def test_analyze_frac_wavelet_builds_the_bundle_from_the_frac_forward():
    """wavelet="frac_bspline" routes analyze() through atrous2d_forward_frac; the bundle
    records the wavelet and order, and the maxima genuinely differ from the mz_spline
    default on the same field."""
    f = fbm2d(64, 0.5, seed=11)
    b_std = mz_edges.analyze(f, 3)
    b_frac = mz_edges.analyze(f, 3, wavelet="frac_bspline", alpha=2.0)
    assert b_std.get("wavelet", "mz_spline") == "mz_spline"
    assert b_frac["wavelet"] == "frac_bspline" and b_frac["alpha"] == 2.0
    assert len(b_frac["extrema"]) == 3
    std_n = [e["x"].size for e in b_std["extrema"]]
    frac_n = [e["x"].size for e in b_frac["extrema"]]
    assert std_n != frac_n or any(
        not np.array_equal(a["x"], b["x"])
        for a, b in zip(b_std["extrema"], b_frac["extrema"]))


def test_analyze_default_is_the_untouched_mz_spline_path():
    """No wavelet kwarg == the pre-split behavior, byte for byte."""
    f = fbm2d(48, 0.5, seed=12)
    a = mz_edges.analyze(f, 3)
    b = mz_edges.analyze(f, 3, wavelet="mz_spline")
    for ea, eb in zip(a["extrema"], b["extrema"]):
        np.testing.assert_array_equal(ea["x"], eb["x"])
        np.testing.assert_array_equal(ea["mod"], eb["mod"])
    with pytest.raises(ValueError, match="wavelet"):
        mz_edges.analyze(f, 3, wavelet="haar")


def test_preview_refuses_fractional_bundles_with_a_pointed_message():
    """Fractional POCS is a follow-up (pocs2d's projections use the standard filter bank
    internally): preview says so instead of reconstructing with mismatched filters. Bundles
    from before the wavelet key existed still preview (the .get default)."""
    f = fbm2d(32, 0.5, seed=13)
    frac = mz_edges.analyze(f, 2, wavelet="frac_bspline", alpha=2.5)
    with pytest.raises(ValueError, match="mz_spline"):
        mz_edges.preview(f, frac, n_iter=1)
    legacy = mz_edges.analyze(f, 2)
    legacy.pop("wavelet", None)                      # a cached pre-split bundle
    img, diag = mz_edges.preview(f, legacy, n_iter=1)
    assert img.shape == f.shape


def test_device_exposes_the_wavelet_and_order_knobs(clean_registry):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices.mz_edges import MZEdges
    from dynamix.model.device import defaults_for

    params = {p.name: p for p in MZEdges.params}
    assert params["wavelet"].choices == ("mz_spline", "frac_bspline")
    assert params["wavelet"].default == "mz_spline"
    assert params["alpha"].default == 3.0

    f = fbm2d(64, 0.5, seed=11)
    field = RasterField(name="f", values=f, frame=LocalFrame(),
                        x_axis=np.arange(64, dtype=np.float64),
                        y_axis=np.arange(64, dtype=np.float64))
    dev = MZEdges()
    base = dict(defaults_for(dev), n_levels=3)
    res_std = dev.compute(field, base)
    assert res_std["params"]["wavelet"] == "mz_spline"   # the pre-split stamp, now a knob
    res_frac = dev.compute(field, dict(base, wavelet="frac_bspline", alpha=2.0))
    assert res_frac["params"]["wavelet"] == "frac_bspline"
    assert dev.cache_key("s", base) != dev.cache_key(
        "s", dict(base, wavelet="frac_bspline"))
    assert dev.cache_key("s", dict(base, wavelet="frac_bspline")) != dev.cache_key(
        "s", dict(base, wavelet="frac_bspline", alpha=2.0))
