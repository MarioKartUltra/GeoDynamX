# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Reconstruction from multiscale edges: the vectorized POCS against the mzlib oracle."""
from __future__ import annotations

import functools
import pathlib
import time

import numpy as np
import pytest

from dynamix.core import mz_edges as mz, mzlib

DEM = pathlib.Path(__file__).resolve().parents[1] / "docs" / "demo" / "dem_crop.npz"


def _dem():
    if not DEM.is_file():
        pytest.skip("the demo raster is not in this checkout")
    return np.load(DEM)["values"].astype(np.float64)


def _synthetic(n=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    return np.sin(6 * x) + (x > 0.5) + 0.3 * np.cos(9 * y) + 0.05 * rng.standard_normal((n, n))


def _fields(n=64):
    """The synthetic field, plus the demo raster when the checkout has it."""
    return [_synthetic(n)] + ([_dem()] if DEM.is_file() else [])


FIELDS = ("synthetic", "dem")


def _field(name):
    """A test field by name: the 64 x 64 synthetic (every checkout), or the demo raster (skipped
    where the checkout lacks it). 64 is divisible by 2**4, so the thumbnail coarse applies."""
    return _synthetic() if name == "synthetic" else _dem()


@functools.lru_cache(maxsize=None)
def _pocs2d_oracle(name, constraint_mode):
    """``mzlib.pocs2d`` at J = 4 and 10 iterations on a named field. Its pure-Python P_Gamma is
    slow on the demo raster, so each (field, constraint mode) runs once per session."""
    v = _field(name)
    b = mz.analyze(v, 4)
    S = mz._coarse_for(v, b, 4)
    ref, ref_resid, _ = mzlib.pocs2d(b["mz_maxima"], S, v.shape, 4, n_iter=10,
                                     constraint_mode=constraint_mode)
    return ref, ref_resid


@pytest.mark.parametrize("n,s", [(64, 2.0), (64, 16.0), (96, 8.0)])
def test_vectorized_pgamma_equals_mzlib_row_by_row(n, s):
    rng = np.random.default_rng(3)
    g = rng.standard_normal((n, n))
    mask = rng.random((n, n)) < 0.05
    mask[5] = False                                  # a row with no maxima: unchanged
    mask[6] = False; mask[6, 17] = True              # one maximum: exp of the torus distance
    mask[7] = False; mask[7, 0] = True; mask[7, n - 1] = True      # the wrap segment
    vals = rng.standard_normal(mask.sum())
    fast = mz._pgamma_apply(g, vals, mz._pgamma_operator(mask, s))
    ref = np.empty_like(g); k = 0
    for r in range(n):
        idx = np.flatnonzero(mask[r])
        ref[r] = mzlib.p_gamma(g[r], idx, vals[k:k + len(idx)], s); k += len(idx)
    np.testing.assert_allclose(fast, ref, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fast[5], g[5])


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("mode", mz.RECON_MODES)
def test_every_mode_matches_pocs2d_on_the_spline(mode, field):
    v = _field(field)
    b = mz.analyze(v, 4)
    oracle_mode = "set_points" if mode == "set_points" else "separable"
    ref, ref_resid = _pocs2d_oracle(field, oracle_mode)
    img, diag = mz.reconstruct(v, b, n_iter=10, mode=mode)
    scale = np.abs(v).max()
    np.testing.assert_allclose(img, ref, rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(diag["resid"], ref_resid, rtol=1e-9)
    assert diag["n_iter"] == 10 and diag["mode"] == mode


def test_separable_rebuilds_its_operators_per_iteration_above_the_budget(monkeypatch):
    """Above the memory budget the P_Gamma operators are rebuilt every iteration instead of
    cached: the same arithmetic, so the same image and residual trajectory."""
    v = _synthetic(); b = mz.analyze(v, 4)
    cached, dc = mz.reconstruct(v, b, n_iter=5)
    monkeypatch.setattr(mz, "_PGAMMA_BUDGET", 0)
    rebuilt, dr = mz.reconstruct(v, b, n_iter=5)
    np.testing.assert_array_equal(rebuilt, cached)
    assert dr["resid"] == dc["resid"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("coarse,policy", [("full", "full"), ("thumbnail", "thumbnail")])
def test_reconstruct_matches_preview(coarse, policy, field):
    v = _field(field)
    ref, _ = mz.preview(v, mz.analyze(v, 4, coarse=policy), n_iter=10)
    img, diag = mz.reconstruct(v, mz.analyze(v, 4), n_iter=10, coarse=coarse)
    np.testing.assert_allclose(img, ref, rtol=0, atol=1e-9 * np.abs(v).max())
    assert diag["coarse"] == coarse


@pytest.mark.parametrize("alpha", [0.5, 1.5, 2.5, 3.0, 4.2])
def test_fractional_bank_round_trips(alpha):
    img = np.random.default_rng(1).standard_normal((64, 48))
    bank = mz._Bank("frac_bspline", alpha)
    S, pairs = bank.forward_full(mz._mirror2d(img), 4)
    rec = bank.inverse(S, pairs)[:64, :48]
    assert np.linalg.norm(rec - img) / np.linalg.norm(img) < 1e-12


@pytest.mark.parametrize("alpha", [2.0, 3.0, 3.7])
def test_fractional_bank_forward_is_the_analysis_forward(alpha):
    """The bank POCS constrains with must be the transform that produced the maxima values."""
    img = np.random.default_rng(2).standard_normal((64, 48))
    S1, P1 = mz._Bank("frac_bspline", alpha).forward_full(mz._mirror2d(img), 4)
    S2, P2 = mz.atrous2d_forward_frac(img, 4, alpha)
    np.testing.assert_allclose(S1, S2, rtol=0, atol=1e-12)
    for (a1, b1), (a2, b2) in zip(P1, P2, strict=True):
        np.testing.assert_allclose(a1, a2, rtol=0, atol=1e-12)
        np.testing.assert_allclose(b1, b2, rtol=0, atol=1e-12)


@pytest.mark.parametrize("wavelet,alpha", [("mz_spline", 3.0), ("frac_bspline", 2.5)])
def test_edges_only_snr_is_scored_against_the_synthesised_detail(wavelet, alpha):
    """Edges only pins S_J to zero, so the best it can reach is the synthesis of the field's own
    details: the field minus S_J passed through the low-pass synthesis chain."""
    v = _synthetic(); ny, nx = v.shape
    bank = mz._Bank(wavelet, alpha)
    S, pairs = bank.forward_full(mz._mirror2d(v), 4)
    detail = bank.inverse(np.zeros_like(S), pairs)[:ny, :nx]
    np.testing.assert_allclose(mz._detail_target(v, bank, 4), detail,
                               rtol=0, atol=1e-9 * np.abs(v).max())
    img, d = mz.reconstruct(v, mz.analyze(v, 4, wavelet=wavelet, alpha=alpha), n_iter=5,
                            coarse="none")
    want = 10 * np.log10(np.sum((detail - detail.mean()) ** 2) / np.sum((detail - img) ** 2))
    assert d["snr_db"] == pytest.approx(want, abs=1e-6)


def test_fractional_quality_gate(frac_gate_db):
    """Fractional orders reconstruct about as well as the paper's spline order: at 10 iterations
    no order diverges and each is within the gate of alpha = 3 on both fields."""
    for v in _fields(128):
        ref = mz.reconstruct(v, mz.analyze(v, 4, wavelet="frac_bspline", alpha=3.0))[1]
        for alpha in (2.0, 2.5, 3.5, 4.0):
            d = mz.reconstruct(v, mz.analyze(v, 4, wavelet="frac_bspline", alpha=alpha))[1]
            assert d["status"] != "diverging", (alpha, d["status"])
            assert d["snr_db"] >= ref["snr_db"] - frac_gate_db, (alpha, d["snr_db"], ref["snr_db"])
            assert d["wavelet"] == "frac_bspline" and d["alpha"] == alpha


def test_separable_costs_at_most_twice_set_points_on_the_demo():
    v = _dem(); b = mz.analyze(v, 4)

    def best(mode):
        out = []
        for _ in range(3):
            t = time.perf_counter(); mz.reconstruct(v, b, n_iter=10, mode=mode)
            out.append(time.perf_counter() - t)
        return min(out)
    assert best("separable") <= 2.0 * best("set_points")


def test_thumbnail_coarse_refuses_a_grid_not_divisible_by_2_to_the_J():
    v = _synthetic(60)                                           # 60 is not divisible by 16
    b = mz.analyze(v, 4)
    with pytest.raises(ValueError, match="divisible"):
        mz.reconstruct(v, b, coarse="thumbnail")
    thumb = mz.coarse_thumbnail(v, b)                            # the thumbnail itself still works
    assert thumb.shape == (4, 4) and thumb.dtype == np.float32   # ceil(60/16) = 4


def test_unknown_mode_or_coarse_is_refused():
    v = _synthetic(); b = mz.analyze(v, 2)
    with pytest.raises(ValueError, match="mode"):
        mz.reconstruct(v, b, mode="svd")
    with pytest.raises(ValueError, match="coarse"):
        mz.reconstruct(v, b, coarse="half")


@pytest.mark.parametrize("field", FIELDS)
def test_edges_only_and_status_and_cancel(field):
    from dynamix.core.wtmm_backend import ComputeCancelled
    v = _field(field); b = mz.analyze(v, 4)
    img, d = mz.reconstruct(v, b, n_iter=10, coarse="none")
    assert d["coarse"] == "none" and d["status"] in ("converged", "still improving", "rising",
                                                     "diverging")
    assert img.shape == v.shape and np.isfinite(d["snr_db"])
    with pytest.raises(ComputeCancelled):
        mz.reconstruct(v, b, n_iter=10, cancel=lambda: True)
    seen = []
    mz.reconstruct(v, b, n_iter=3, progress=lambda s, f: seen.append(f))
    assert seen[-1] == pytest.approx(1.0) and len(seen) == 3


@pytest.mark.parametrize("resid,status", [
    ([4.0, 2.0, 1.0], "still improving"),
    ([4.0, 2.0, 2.0005], "converged"),                   # |change| under 1e-3 relative
    ([4.0, 1.0, 1.5], "rising"),                         # grew, but under 5x its minimum
    ([4.0, 1.0, 5.5], "diverging"),                      # past 5x its minimum
])
def test_status_names_where_the_residual_stands(resid, status):
    assert mz._recon_status(resid) == status


@pytest.mark.parametrize("field", FIELDS)
def test_coarse_image_is_the_pinned_coarse_channel(field):
    v = _field(field); b = mz.analyze(v, 4)
    S = mz._coarse_for(v, b, 4)
    np.testing.assert_allclose(mz.coarse_image(v, b), S[:v.shape[0], :v.shape[1]], atol=1e-9)
    thumb = mz.coarse_thumbnail(v, b)
    assert thumb.dtype == np.float32
    np.testing.assert_array_equal(thumb, S[:v.shape[0]:16, :v.shape[1]:16].astype(np.float32))


def test_coarse_image_reproduces_the_dither():
    v = np.round(_synthetic(64) * 4) / 4                         # quantised: a measurable LSB
    b = mz.analyze(v, 3, dither=True)
    assert b["lsb"] is not None
    np.testing.assert_allclose(mz.coarse_image(v, b), mz._coarse_for(v, b, 3)[:64, :64],
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("engine", ["fftw", "mlx"])
def test_fft_engines_agree_to_float32(engine, field):
    from dynamix.core import fft_policy
    if engine == "mlx":
        pytest.importorskip("mlx.core")
    v = _field(field); b = mz.analyze(v, 4)
    ref = mz.reconstruct(v, b, n_iter=5)[0]
    try:
        fft_policy.configure(engine, 32)
        got = mz.reconstruct(v, b, n_iter=5)[0]
    finally:
        fft_policy.reset()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4 * np.abs(v).max())
