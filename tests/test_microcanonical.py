# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.microcanonical -- the Turiel microcanonical formalism core
(fresh reimplementation of the author's microcanonical_wtmm.ipynb notebook, with the
prototype's defects corrected -- see the module docstring).

Two synthetic grounds, chosen deliberately:

* A CONSERVATIVE 2D multiplicative cascade -- the measure functional's native habitat, with
  EXACT per-pixel ground truth: each dyadic block's mean density is exactly the product of the
  weights above it (per-split weights sum to 4), so the expected OLS slope over any dyadic
  window is computable per pixel. Remaining estimator error is kernel localization blur, the
  trade-off Turiel 2008 SS4.4/appendix A describes -- pinned at its measured level, not wished
  away.
* fBm -- deliberately kept as the PATHOLOGY, not the null: fBm's discrete ||grad s|| is a
  stationary mean-dominated density, so gradient-measure projections converge to the ensemble
  mean and the h-map degenerates to ~0 REGARDLESS of H (Pont et al. 2006's "ill-behaved
  measure" case, which prescribes multiaffine/increment functionals for such signals). The
  prototype notebook's fBm cell expected h = H-1 from this pipeline -- a wrong premise, pinned
  here as such so nobody "fixes" the estimator against it.

The eq-21 resolution convention (THE prototype bug) is guarded in both places that take r0:
a pixel-valued r0 (>= 1) raises instead of silently flipping the sign of the correction.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import fbm2d
from dynamix.core import microcanonical as mc

WEIGHTS = (0.4, 0.8, 1.2, 1.6)          # sum 4 -> exactly conservative per 2x2 split
N_LEVELS = 7                              # 128 x 128
THEORY_MODE = float(-np.log2(WEIGHTS).mean())   # mode of -mean(log2 W): +0.176


def _cascade2d(n=N_LEVELS, weights=WEIGHTS, seed=0):
    """Conservative multiplicative cascade + the per-level log2-weight maps (coarsest first).

    Each 2x2 split places a random permutation of ``weights`` (sum 4), so every dyadic
    block's mean density is EXACTLY the product of the weights of the levels above it."""
    rng = np.random.default_rng(seed)
    dens = np.ones((1, 1))
    levels: list = []
    w = np.asarray(weights, dtype=np.float64)
    for _ in range(n):
        ny, nx = dens.shape
        p = rng.permuted(np.tile(w, (ny * nx, 1)), axis=1).reshape(ny, nx, 2, 2)
        W = p.transpose(0, 2, 1, 3).reshape(2 * ny, 2 * nx)
        dens = np.repeat(np.repeat(dens, 2, 0), 2, 1) * W
        levels = [np.repeat(np.repeat(L, 2, 0), 2, 1) for L in levels]
        levels.append(np.log2(W))
    return dens, np.stack(levels)


@pytest.fixture(scope="module")
def cascade():
    dens, LW = _cascade2d()
    ms = np.arange(1, 6)                         # dyadic block sides 2..32 px
    scales = 2.0 ** ms
    # exact block-mean log2 density per level, and the exact per-pixel OLS slope over the
    # window (OLS on the true values -- the only mismatch left to the estimator is the kernel)
    logT = np.stack([LW[: N_LEVELS - m].sum(0) for m in ms])
    x = ms * np.log(2.0)
    xc = x - x.mean()
    h_true = (xc[:, None, None] * (logT * np.log(2.0))).sum(0) / (xc ** 2).sum()
    T = mc.measure_projections(dens, scales)
    return {"dens": dens, "scales": scales, "T": T, "h_true": h_true}


# ------------------------------------------------------------------- eq-21 convention guard

def test_dh_histogram_refuses_pixel_r0():
    """THE prototype bug: r0 in pixels (>= 1) flips the sign of the eq-21 correction. Refused
    loudly, never silently computed."""
    h = np.random.default_rng(0).normal(0.3, 0.1, (64, 64))
    with pytest.raises(ValueError, match="RELATIVE"):
        mc.dh_histogram(h, r0_rel=2.0)
    with pytest.raises(ValueError, match="RELATIVE"):
        mc.singularity_map_point(np.abs(h), 2.0)


def test_dh_histogram_is_bounded_by_d_with_equality_at_the_mode():
    """With the CORRECT convention (log r0 < 0): D <= d everywhere, D = d exactly at the modal
    bin -- the two properties the pixel-r0 bug destroyed."""
    h = np.random.default_rng(1).normal(0.3, 0.1, (128, 128))
    hc, D = mc.dh_histogram(h, d=2.0)
    finite = np.isfinite(D)
    assert finite.any()
    assert np.nanmax(D) == pytest.approx(2.0)
    assert (D[finite] <= 2.0 + 1e-12).all()


def test_dh_histogram_default_r0_is_the_papers_resolution():
    """Default r0_rel = 1/sqrt(nx*ny) (Turiel 2008 SS4.4)."""
    h = np.random.default_rng(2).normal(0.3, 0.1, (64, 64))
    hc0, D0 = mc.dh_histogram(h)
    hc1, D1 = mc.dh_histogram(h, r0_rel=1.0 / 64.0)
    np.testing.assert_allclose(D0, D1, equal_nan=True)


# ------------------------------------------------------------------- cascade ground truth

def test_regression_recovers_cascade_exponents_to_kernel_blur(cascade):
    """Against the EXACT per-pixel OLS truth: small bias, pointwise correlation at the level
    kernel localization allows, and clearly better after blur-matching the truth to the
    kernel's neighborhood (the estimator measures a neighborhood, not a pixel). r2_min=0:
    on a cascade the log T wiggle is the random walk of the weights -- physics, not noise --
    and gating on R^2 is a selection bias (measured: it biased the median +0.1)."""
    h_map, _r2 = mc.singularity_map_regression(cascade["T"], cascade["scales"], r2_min=0.0)
    h_true = cascade["h_true"]
    err = h_map - h_true
    assert abs(np.nanmedian(err)) < 0.1
    assert np.nanmedian(np.abs(err)) < 0.3
    assert np.corrcoef(h_true.ravel(), h_map.ravel())[0, 1] > 0.45
    blurred = gaussian_filter(h_true, 4.0)
    assert np.corrcoef(blurred.ravel(), h_map.ravel())[0, 1] > 0.65


def test_point_estimate_agrees_with_regression_on_the_cascade(cascade):
    """Pont 2006's PUNCTUAL (single-finest-scale) estimator sees the same kernel-blurred
    field as the multiscale regression -- strong mutual agreement is the cross-validation
    contract. (Their tails differ -- punctual loses the right tail, SS V.B.2 -- but the
    point-to-point h values track on this cascade.)"""
    r0_rel = mc.relative_scale(cascade["scales"][0], cascade["dens"].shape)
    h_pt = mc.singularity_map_point(cascade["T"][0], r0_rel)
    h_reg, _ = mc.singularity_map_regression(cascade["T"], cascade["scales"], r2_min=0.0)
    both = np.isfinite(h_pt) & np.isfinite(h_reg)
    assert np.corrcoef(h_pt[both], h_reg[both])[0, 1] > 0.9


def test_dh_histogram_peaks_near_the_cascade_theory_mode(cascade):
    """D(h) peak sits at d, at an h within kernel-blur distance of the multinomial mode
    -mean(log2 w) = +0.176 (measured offset ~0.07)."""
    h_map, _ = mc.singularity_map_regression(cascade["T"], cascade["scales"], r2_min=0.0)
    hc, D = mc.dh_histogram(h_map)
    i = int(np.nanargmax(D))
    assert np.nanmax(D) == pytest.approx(2.0)
    assert abs(hc[i] - THEORY_MODE) < 0.15


# ------------------------------------------------------------------- the fBm pathology, pinned

def test_fbm_gradient_measure_is_mean_dominated_pont_caveat():
    """fBm's discrete ||grad s|| is stationary with a nonzero mean, so block averages converge
    to <||grad s||> and the measure-functional h-map sits at ~0 REGARDLESS of H -- Pont et al.
    2006's ill-behaved case (their prescription: multiaffine/increment functionals for such
    signals). The prototype notebook's fBm cell expected h = H-1 from this pipeline; that
    premise was wrong, and this test exists so the ~0 result is never 'fixed' against it."""
    scales = np.geomspace(2, 16, 10)
    for H in (0.3, 0.8):
        f = fbm2d(128, H, seed=5)
        T = mc.measure_projections(mc.gradient_measure(f), scales)
        h_map, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
        assert abs(np.nanmedian(h_map)) < 0.15, f"H={H}"


# ------------------------------------------------------------------- fracint operator

def test_fracint_zero_is_identity_and_returns_a_copy():
    f = fbm2d(64, 0.5, seed=3)
    out = mc.fracint2d_fourier(f, 0.0)
    np.testing.assert_array_equal(out, f)
    assert out is not f


def test_fracint_is_a_semigroup():
    """alpha composes additively (F/K^a then F/K^b == F/K^(a+b)) up to pad/detrend edges."""
    f = fbm2d(64, 0.5, seed=3)
    ab = mc.fracint2d_fourier(mc.fracint2d_fourier(f, 0.4), 0.6)
    apb = mc.fracint2d_fourier(f, 1.0)
    rel = np.abs(ab - apb).mean() / np.abs(apb - apb.mean()).mean()
    assert rel < 0.01


def test_fracint_attenuates_a_single_mode_by_one_over_k():
    """A pure harmonic at frequency index f0 is an (approximate) eigenfunction: amplitude
    ratio 1/f0 at alpha=1 (pad=0 keeps it exactly periodic; detrend of a zero-mean harmonic
    is ~none)."""
    xx = np.arange(128)[None, :].repeat(128, 0)
    mode = np.cos(2 * np.pi * 8 * xx / 128.0)
    out = mc.fracint2d_fourier(mode, 1.0, pad=0)
    assert np.std(out) / np.std(mode) == pytest.approx(1.0 / 8.0, rel=0.01)


# ------------------------------------------------------------------- MSC + reconstruction

def test_extract_msc_density_is_the_msc_fraction_of_finite_pixels():
    h = np.array([[np.nan, -1.0], [0.5, -0.2]])
    mask, density = mc.extract_msc(h, h_theta=0.0)
    np.testing.assert_array_equal(mask, [[False, True], [False, True]])
    assert density == pytest.approx(2.0 / 3.0)


def test_full_mask_reconstruction_inverts_the_gradient(cascade):
    """With the whole domain as 'MSC' the kernel inverts np.gradient up to the central-
    difference vs continuous-derivative mismatch: >40 dB and correlation > 0.999 on fBm."""
    f = fbm2d(128, 0.8, seed=2)
    recon, psnr, rel_err = mc.reconstruct_from_msc(f, np.ones(f.shape, dtype=bool))
    assert psnr > 40.0
    assert np.corrcoef(f.ravel(), recon.ravel())[0, 1] > 0.999
    assert rel_err < 0.02


def test_reconstruction_quality_grows_with_msc_density():
    """Sparser essential gradients reconstruct worse -- the MSC sweep's monotone backbone
    (full > 30% > 5%)."""
    scales = np.geomspace(2, 16, 10)
    f = fbm2d(128, 0.8, seed=2)
    h_map, _ = mc.singularity_map_regression(
        mc.measure_projections(mc.gradient_measure(f), scales), scales, r2_min=0.0)
    psnrs = []
    for pct in (100, 30, 5):
        if pct == 100:
            mask = np.ones(f.shape, dtype=bool)
        else:
            mask, _ = mc.extract_msc(h_map, float(np.nanpercentile(h_map, pct)))
        psnrs.append(mc.reconstruct_from_msc(f, mask)[1])
    assert psnrs[0] > psnrs[1] > psnrs[2]


# ------------------------------------------------------------------- multiaffine (Ricker)

def test_ricker_projections_kill_constants():
    """Zero-mean on the discrete grid, exactly: a constant field projects to ~0 at every
    scale (the property the measure route lacks -- its mean-domination pathology)."""
    T = mc.ricker_projections(np.full((64, 64), 7.3), [1.0, 2.0, 4.0])
    assert np.abs(T).max() < 1e-10


def test_multiaffine_gamma_tracks_fbm_hurst():
    """The estimator for smooth/function-class fields: on fBm the per-pixel log-log slope of
    |Ricker projections| recovers gamma ~ H (measured: slope 0.98, offset -0.06 across
    H = 0.3/0.5/0.8 at 256^2) -- the validation the gradient-measure route cannot give
    (Pont pathology). Fit range = the paper's own r_1 = 1 zero-crossing radius, kappa = 10
    (Turiel 2008 fig 2 + SS4.4). r2_min = 0: gating on R^2 is a strong selection bias here
    too (|T| dips at response zero crossings are geometry, not noise; the paper's own
    multiaffine test reports < 50% of points above R^2 = 0.8 on real fields)."""
    scales = np.geomspace(1.0, 10.0, 10)
    for H in (0.3, 0.5, 0.8):
        f = fbm2d(256, H, seed=5)
        T = mc.ricker_projections(f, scales)
        g, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
        assert abs(np.nanmedian(g) - H) < 0.1, f"H={H}"


def test_scale_grid_count_is_nearly_irrelevant_but_range_dominates():
    """Turiel 2009: regressions run over "a range of scales typically going from 1 to 8 pixels
    non uniformly sampled" -- a FEW scales over ~3 octaves. Pinned finding: within
    the SAME range the sample count barely changes the map (dyadic 4-pt vs geometric 6-pt over
    kappa=8: r > 0.9), while widening the RANGE changes the physics (kappa=32: r < 0.7 against
    the Turiel-range map -- coarse-scale blur homogenizes exactly what the microcanonical
    formalism exists to localize). The prototype notebook's 2..50 px grid measured r = -0.06."""
    f = fbm2d(256, 0.5, seed=5)

    def gmap(scales):
        T = mc.ricker_projections(f, scales)
        g, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
        return g.ravel()

    dyadic = gmap(np.array([1.0, 2.0, 4.0, 8.0]))
    geom_same = gmap(np.geomspace(1, 8, 6))
    geom_wide = gmap(np.geomspace(1, 32, 10))
    # jointly-finite pixels: at 32-bit FFT precision the few |T|~0 zero
    # crossings of the multiaffine projection are honestly NaN
    ok = np.isfinite(dyadic) & np.isfinite(geom_same) & np.isfinite(geom_wide)
    assert ok.mean() > 0.99
    assert np.corrcoef(dyadic[ok], geom_same[ok])[0, 1] > 0.9
    assert np.corrcoef(dyadic[ok], geom_wide[ok])[0, 1] < 0.7


def test_ricker_wavelet_family_identities_and_ordering():
    """The generalized Mexican hats: q -> 1 recovers the Ricker EXACTLY, and q = 1.5 coincides
    with the Lorentzian-Marr at beta = 2 (the q-Gaussian family IS the generalized-Lorentzian /
    Student-t family -- same kernel, two parameterizations). All three kill constants (exact
    discrete zero-mean), all recover gamma increasing with H on fBm; heavy tails COMPRESS gamma
    (tail-weighted mixing, the Turiel 2008 SS4.2.2 localization/calibration trade -- measured:
    at H=0.8, gaussian +0.71, q=1.5 +0.43, L1 +0.29), so gamma values are NOT cross-comparable
    between wavelets."""
    scales = np.geomspace(1.0, 8.0, 6)
    f = fbm2d(128, 0.5, seed=1)
    a = mc.ricker_projections(f, scales, wavelet="gaussian")
    b = mc.ricker_projections(f, scales, wavelet="q_gaussian", q_tsallis=1.0)
    np.testing.assert_array_equal(a, b)
    c = mc.ricker_projections(f, scales, wavelet="q_gaussian", q_tsallis=1.5)
    d = mc.ricker_projections(f, scales, wavelet="lorentzian", beta=2.0)
    np.testing.assert_allclose(c, d, rtol=1e-12, atol=1e-15)

    const = np.full((64, 64), 3.1)
    for wav in ("gaussian", "q_gaussian", "lorentzian"):
        assert np.abs(mc.ricker_projections(const, [1.0, 2.0], wavelet=wav)).max() < 1e-10

    for wav, kw in (("q_gaussian", {"q_tsallis": 1.5}), ("lorentzian", {"beta": 1.0})):
        meds = []
        for H in (0.3, 0.8):
            g, _ = mc.singularity_map_regression(
                mc.ricker_projections(fbm2d(128, H, seed=5), scales, wavelet=wav, **kw),
                scales, r2_min=0.0)
            meds.append(np.nanmedian(g))
        assert meds[1] > meds[0], (wav, meds)

    with pytest.raises(ValueError, match="wavelet"):
        mc.ricker_projections(f, scales, wavelet="haar")


# ------------------------------------------------------------------- band reconstruction

def test_band_mask_is_the_generalized_msc():
    h = np.array([[np.nan, -1.0], [0.2, 0.9]])
    np.testing.assert_array_equal(mc.band_mask(h, 0.0, 0.5), [[False, False], [True, False]])
    # extract_msc is the (-inf, h_theta) special case
    np.testing.assert_array_equal(mc.band_mask(h, -np.inf, 0.0),
                                  mc.extract_msc(h, 0.0)[0])


def test_cumulative_band_reconstruction_is_monotone_and_ends_at_full(cascade):
    """The fig-13 study contract: adding bands from most singular upward never hurts, and the
    final stage (every finite-h pixel) matches the full-mask reconstruction."""
    f = fbm2d(128, 0.8, seed=2)
    scales = np.geomspace(1, 8, 6)
    T = mc.ricker_projections(f, scales)
    h, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
    edges = np.nanquantile(h, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf          # full coverage by construction
    rows = mc.reconstruction_by_bands(f, h, edges, cumulative=True)
    psnrs = [r["psnr_db"] for r in rows]
    assert all(b >= a - 1e-6 for a, b in zip(psnrs, psnrs[1:]))
    assert rows[-1]["density"] > 0.999
    _, full_psnr, _ = mc.reconstruct_from_msc(f, np.isfinite(h))
    assert abs(psnrs[-1] - full_psnr) < 1e-9


def test_single_band_reconstruction_answers_which_band_carries_the_features():
    """Non-cumulative rows reconstruct each band ALONE -- densities partition the finite
    pixels and every stage returns finite quality numbers."""
    f = fbm2d(128, 0.8, seed=2)
    scales = np.geomspace(1, 8, 6)
    T = mc.ricker_projections(f, scales)
    h, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
    edges = np.nanquantile(h, [0.0, 1 / 3, 2 / 3, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    rows = mc.reconstruction_by_bands(f, h, edges, cumulative=False)
    assert abs(sum(r["density"] for r in rows) - 1.0) < 1e-6
    assert all(np.isfinite(r["psnr_db"]) for r in rows)


def test_reconstruction_border_fix_on_a_nonperiodic_field():
    """The Fourier kernel assumes periodicity; a plain TREND (maximally non-periodic) is the
    acid test. Default symmetric extension reconstructs it well from the full mask; the raw
    periodic inversion (pad=0) demonstrably cannot."""
    yy, xx = np.mgrid[:128, :128].astype(np.float64)
    s = xx + 0.5 * yy
    full = np.ones(s.shape, dtype=bool)
    _, psnr_padded, _ = mc.reconstruct_from_msc(s, full)
    _, psnr_raw, _ = mc.reconstruct_from_msc(s, full, pad=0)
    assert psnr_padded > 30.0
    assert psnr_padded > psnr_raw + 10.0


def test_band_reconstructor_matches_the_exact_inversion_and_is_fast_per_tick():
    """The live-drag engine: at full resolution and the same pad it matches
    reconstruct_from_msc on the band mask (tolerance covers the half-spectrum rFFT's
    Nyquist-bin handling -- preview-grade, the commit path uses the exact function), the
    stride follows max_dim, and NaN h pixels never enter a band."""
    f = fbm2d(128, 0.8, seed=2)
    scales = np.geomspace(1, 8, 6)
    h, _ = mc.singularity_map_regression(mc.ricker_projections(f, scales), scales, r2_min=0.0)
    h[0, :3] = np.nan
    br = mc.BandReconstructor(f, h, max_dim=4096, pad=64)
    assert br.stride == 1
    a = br.reconstruct(-1.0, 0.5)
    b, _, _ = mc.reconstruct_from_msc(f, mc.band_mask(h, -1.0, 0.5), pad=64)
    span = float(f.max() - f.min())
    assert np.abs(a.astype(np.float64) - b.astype(np.float64)).max() < 0.005 * span

    br2 = mc.BandReconstructor(np.tile(f, (2, 2)), np.tile(h, (2, 2)), max_dim=128)
    assert br2.stride == 2
    assert br2.reconstruct(-1.0, 0.5).shape == (128, 128)

    with pytest.raises(ValueError, match="share a grid"):
        mc.BandReconstructor(f, h[:64])


def test_flat_patches_get_nan_not_absurd_slopes():
    """The +-200 artifact: a field with a large FLAT patch (nodata fill / quantized plain) has
    zero measure there; the projection hits the log floor at fine scales and the regression
    would manufacture slopes of +-hundreds. Floored pixels are NaN, and every finite h stays
    physically plausible."""
    rng = np.random.default_rng(0)
    f = rng.normal(0.0, 1.0, (128, 128)).cumsum(axis=1)     # textured half
    f[:, :64] = 3.7                                          # exactly flat half
    scales = np.geomspace(1, 8, 6)
    T = mc.measure_projections(mc.gradient_measure(f), scales)
    h, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
    assert np.isnan(h[:, :48]).all()                         # deep in the flat: no exponent
    finite = h[np.isfinite(h)]
    assert finite.size > 0
    assert np.abs(finite).max() < 50.0                       # no floor-jump slopes survive


# ------------------------------------------------------------------- fractional Gaussian family
#

def test_marr_family_names_alias_their_kernels():
    """The method-true names (the wavelet family is a property of the METHOD, not an
    orthogonal knob): g2 / q_mexican / lorentzian_marr are the multiaffine route's own names
    for the kernels the conflated device called gaussian / q_gaussian / lorentzian -- same
    arrays, exactly, so the split devices change no physics."""
    f = fbm2d(64, 0.5, seed=3)
    scales = [1.0, 2.0, 4.0]
    np.testing.assert_array_equal(mc.ricker_projections(f, scales, wavelet="g2"),
                                  mc.ricker_projections(f, scales, wavelet="gaussian"))
    # q_mexican is Borges et al. 2004's parameterization of the SAME family (see
    # test_q_mexican_is_borges_2004_in_2d): their q maps onto q_gaussian's q' = 1/(2 - q).
    np.testing.assert_array_equal(
        mc.ricker_projections(f, scales, wavelet="q_mexican", q_tsallis=1.7),
        mc.ricker_projections(f, scales, wavelet="q_gaussian", q_tsallis=1.0 / (2.0 - 1.7)))
    np.testing.assert_array_equal(
        mc.ricker_projections(f, scales, wavelet="lorentzian_marr", beta=1.5),
        mc.ricker_projections(f, scales, wavelet="lorentzian", beta=1.5))


def test_frac_gaussian_n2_reproduces_the_ricker():
    """The convention pin: the fractional family's real-space closed form is
    psi(rho) ~ 1F1((n+2)/2; 1; -rho^2/2sigma^2) (the exact 2-D isotropic inverse FT of
    ||k||^n exp(-sigma^2 k^2/2)), and at n=2 that IS (1-u)e^-u -- the Ricker formula
    ricker_projections already uses -- so frac_n=2 must reproduce wavelet="gaussian" to
    hyp1f1-evaluation tolerance, same zero-crossing-radius scale convention (sigma^2=r^2/2)."""
    f = fbm2d(64, 0.5, seed=3)
    scales = [1.0, 2.0, 4.0]
    a = mc.ricker_projections(f, scales, wavelet="gaussian")
    b = mc.ricker_projections(f, scales, wavelet="frac_gaussian", frac_n=2.0)
    np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-8)


def test_g1_g3_are_the_integer_orders_of_the_fractional_family():
    """g1/g2/g3 generalize to a REAL-valued vanishing-moment order n (the derivative-order
    axis); the named integer orders are literally frac_gaussian at n=1/3."""
    f = fbm2d(64, 0.5, seed=3)
    scales = [2.0, 4.0]
    np.testing.assert_array_equal(
        mc.ricker_projections(f, scales, wavelet="g1"),
        mc.ricker_projections(f, scales, wavelet="frac_gaussian", frac_n=1.0))
    np.testing.assert_array_equal(
        mc.ricker_projections(f, scales, wavelet="g3"),
        mc.ricker_projections(f, scales, wavelet="frac_gaussian", frac_n=3.0))


def test_frac_gaussian_kills_constants_and_frac_n_binds():
    """Exact discrete zero mean at every order (the k -= k.mean() guarantee the whole
    multiaffine route shares), and the frac_n knob actually changes the kernel."""
    const = np.full((48, 48), 2.2)
    for n in (1.0, 1.7, 3.0):
        assert np.abs(mc.ricker_projections(const, [1.0, 2.0], wavelet="frac_gaussian",
                                            frac_n=n)).max() < 1e-10, n
    f = fbm2d(64, 0.5, seed=3)
    a = mc.ricker_projections(f, [2.0], wavelet="frac_gaussian", frac_n=1.0)
    b = mc.ricker_projections(f, [2.0], wavelet="frac_gaussian", frac_n=2.5)
    assert not np.array_equal(a, b)


def test_measure_frac_gaussian_is_positive_unit_mass():
    """The measure route's fractional Gaussian (the corrected taxonomy)
    DROPS the ||k||^n -- a kernel whose FT vanishes at k=0 has zero mean and cannot be the
    positive kernel the log T contract requires -- and fractionalizes the ENVELOPE instead:
    exp(-rho^n/2), positive for every n > 0, unit discrete mass. A constant measure therefore
    projects to ITSELF at every scale (the property the multiaffine route deliberately
    lacks); n=2 is byte-for-byte the existing gaussian kernel."""
    from dynamix.core import fft_policy

    const = np.full((48, 48), 1.6)
    # the identity to float64 on 64-bit FFTW3, to float32 on the 32-bit engines
    for engine, precision, rtol in (("fftw", 64, 1e-9), ("fftw", 32, 1e-6), ("mlx", 32, 1e-6)):
        if engine == "mlx":
            try:
                import mlx.core  # noqa: F401
            except ImportError:
                continue
        fft_policy.configure(engine, precision)
        for n in (1.2, 2.0, 3.0):
            T = mc.measure_projections(const, [2.0, 4.0], wavelet="frac_gaussian", frac_n=n)
            np.testing.assert_allclose(T, 1.6, rtol=rtol)


def test_measure_frac_gaussian_n2_is_exactly_the_gaussian():
    mu = np.abs(fbm2d(64, 0.5, seed=7)) + 0.1
    a = mc.measure_projections(mu, [1.5, 3.0], wavelet="gaussian")
    b = mc.measure_projections(mu, [1.5, 3.0], wavelet="frac_gaussian", frac_n=2.0)
    np.testing.assert_array_equal(b, a)


def test_vanishing_moment_order_is_the_smooth_field_saturation_slope():
    """The derivative-order axis: order n measures only gamma < n, so on a C-infinity field
    the log-log slope SATURATES near n -- the property that makes g1/g3/frac_n a physics
    knob rather than a style choice. Measured at the center of a wide Gaussian bump
    (scales 4..16): g1 0.91, frac_n=1.5 1.37, g2 1.83, g3 2.73 -- monotone in n, each a
    shade under its order (finite scale range + discrete L1 normalization)."""
    yy, xx = np.mgrid[:128, :128].astype(np.float64)
    bump = np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2.0 * 30.0 ** 2))
    scales = np.geomspace(4.0, 16.0, 6)
    meds = {}
    for wav, kw, n in (("g1", {}, 1.0), ("frac_gaussian", {"frac_n": 1.5}, 1.5),
                       ("g2", {}, 2.0), ("g3", {}, 3.0)):
        T = mc.ricker_projections(bump, scales, wavelet=wav, **kw)
        g, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
        meds[n] = float(np.nanmedian(g[56:72, 56:72]))
        assert abs(meds[n] - n) < 0.35, (wav, n, meds[n])
    orders = sorted(meds)
    assert all(meds[a] < meds[b] for a, b in zip(orders, orders[1:])), meds


def test_g1_g3_track_fbm_hurst_with_the_tail_compression_caveat():
    """Vanishing-moment recovery on fBm: every order n > H tracks H, so the
    order knob is safe to scrub -- but g1's algebraic rho^-3 tail (Riesz nonlocality) is the
    heaviest in the family and COMPRESSES gamma exactly as the q/Lorentzian heavy tails do
    (measured, 256^2 scales 1..10: at H=0.8 g1 +0.59 vs g2 +0.72 / g3 +0.75; at H=0.5 all
    three within 0.13 of H). Absolute values are NOT cross-comparable between orders --
    the family test's standing rule."""
    scales = np.geomspace(1.0, 10.0, 8)
    for wav in ("g1", "g3"):
        meds = []
        for H in (0.3, 0.5, 0.8):
            T = mc.ricker_projections(fbm2d(256, H, seed=5), scales, wavelet=wav)
            g, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
            meds.append(float(np.nanmedian(g)))
        assert meds[0] < meds[1] < meds[2], (wav, meds)
        assert abs(meds[1] - 0.5) < 0.15, (wav, meds[1])


def test_flat_patches_get_nan_in_the_punctual_estimate_too():
    """The +-200 artifact class, punctual edition: FFT leakage dust (~1e-16 of the signal)
    keeps a no-support pixel 'positive', so a plain T0 > 0 gate fabricates a plausible-looking
    finite h across a nodata flat -- determined by rounding noise, not data. The regression
    path's relative floor applies at the single finest scale too. (This function-level floor
    catches DUST; the multiaffine route's scale-1 alias coupling is above it and is masked at
    the device layer instead -- the estimator-orthogonal cross-scale mask in
    ``holder_methods.method_arrays``, whose test lives with the devices.)"""
    rng = np.random.default_rng(0)
    f = rng.normal(0.0, 1.0, (128, 128)).cumsum(axis=1)     # textured half
    f[:, :64] = 3.7                                          # exactly flat half
    scales = np.geomspace(1, 8, 6)
    T = mc.measure_projections(mc.gradient_measure(f), scales)
    h = mc.singularity_map_point(T[0], mc.relative_scale(scales[0], f.shape))
    assert np.isnan(h[:, :48]).all()                         # deep in the flat: no exponent
    assert np.isfinite(h[:, 80:]).any()                      # the textured half still reads


# ---------------------------------------------------------------- q below 1: compact support

def _q_kernel(q, r=4.0, n=64):
    """The measure route's positive q-Gaussian, centred."""
    return np.fft.fftshift(mc._radial_kernel((n, n), r, "q_gaussian", 1.0, q))


@pytest.mark.parametrize("q", [-1.0, 0.0, 0.5, 0.9])
def test_q_gaussian_below_one_is_a_compact_positive_kernel(q):
    """Below q = 1 the Tsallis q-Gaussian [1 - (1-q) rho^2 / 2r^2]_+^(1/(1-q)) has COMPACT
    support, radius r sqrt(2/(1-q)): finite, non-negative, unit mass, exactly zero outside."""
    r, n = 4.0, 64
    k = _q_kernel(q, r, n)
    assert np.isfinite(k).all() and (k >= 0).all()
    assert k.sum() == pytest.approx(1.0)
    y, x = np.indices(k.shape) - n // 2
    rho = np.hypot(x, y)
    edge = r * np.sqrt(2.0 / (1.0 - q))
    assert (k[rho > edge * (1.0 + 1e-9)] == 0).all()     # the edge itself rounds to ~0
    assert (k[rho < edge - 1.0] > 0).all()


def test_q_gaussian_named_members_below_one():
    """q = 0 is the Epanechnikov paraboloid (1 - rho^2/2r^2)_+ and q = -1 the half-dome
    sqrt(1 - rho^2/r^2)_+ -- neither is a delta (the delta is the q -> -inf limit)."""
    r, n = 4.0, 64
    y, x = np.indices((n, n)) - n // 2
    rho2 = (x ** 2 + y ** 2) / r ** 2
    for q, shape in ((0.0, np.clip(1.0 - rho2 / 2.0, 0.0, None)),
                     (-1.0, np.sqrt(np.clip(1.0 - rho2, 0.0, None)))):
        np.testing.assert_allclose(_q_kernel(q, r, n), shape / shape.sum(), atol=1e-15)


def test_q_gaussian_is_continuous_through_q_one():
    np.testing.assert_allclose(_q_kernel(1.0 - 1e-6), _q_kernel(1.0), atol=1e-7)


@pytest.mark.parametrize("q", [0.5, 0.7, 0.9])
def test_q_mexican_hat_below_one_is_bounded_and_crosses_at_r(q):
    """The q-Mexican hat (the q-Gaussian's Laplacian) below q = 1: bounded for q >= 1/2, zero
    outside the support r / sqrt(1-q), zero crossing still at rho = r, constants still killed."""
    r = 4.0
    rho = np.linspace(0.0, 3.0 * r, 3001)
    k = mc._marr_kernel(rho ** 2, r, "q_gaussian", 1.0, q, 2.0, None)
    edge = r / np.sqrt(1.0 - q)
    assert np.isfinite(k).all()
    assert (k[rho < 0.99 * r] > 0).all()
    assert (k[(rho > 1.01 * r) & (rho < 0.99 * edge)] < 0).all()
    assert (k[rho >= edge] == 0).all()
    f = fbm2d(128, 0.5, seed=1)
    assert np.isfinite(mc.ricker_projections(f, [1.0, 2.0, 4.0], wavelet="q_gaussian",
                                             q_tsallis=q)).all()
    const = np.full((64, 64), 3.1)
    assert np.abs(mc.ricker_projections(const, [1.0, 2.0], wavelet="q_gaussian",
                                        q_tsallis=q)).max() < 1e-10


def test_q_gaussian_laplacian_refuses_q_at_or_below_zero():
    """At q = 0 the compact q-Gaussian's Laplacian degenerates (a flat disk plus a ring ON the
    support edge, which point samples miss); below 0 it is not integrable. 0 < q < 1/2 is
    integrable but unbounded at the edge -- allowed (Borges's q < 0 lands there)."""
    with pytest.raises(ValueError, match="q > 0"):
        mc.ricker_projections(np.zeros((32, 32)), [2.0], wavelet="q_gaussian", q_tsallis=0.0)
    assert np.isfinite(mc.ricker_projections(fbm2d(64, 0.5, seed=2), [2.0, 4.0],
                                             wavelet="q_gaussian", q_tsallis=0.4)).all()


def _kernel_through(wavelet, r, **kw):
    """The route's own sampled kernel: a delta through ricker_projections (|.| of it)."""
    n = 129
    d = np.zeros((n, n))
    d[n // 2, n // 2] = 1.0
    return mc.ricker_projections(d, [r], wavelet=wavelet, pad=0, **kw)[0]


@pytest.mark.parametrize("q", [-1.0, -0.5, 0.0, 0.5, 1.5, 1.9])
def test_q_mexican_is_borges_2004_in_2d(q):
    """Borges, Tsallis, Miranda & Andrade 2004 (J. Phys. A 37 9125) eq. 15: the q-Mexican hat
    is the second derivative of the q-Gaussian raised to the power 2 - q. In 2-D (the
    Laplacian) that is  psi_q ~ [1 - (2-q) b rho^2] [e_q^(-b rho^2)]^q ; with the zero
    crossing pinned at r (b = 1/((2-q) r^2), u = rho^2/r^2):  (1-u) [1-(1-q)u/(2-q)]_+^(q/(1-q)).
    Exactly zero-mean and L1-normalized on the grid, like every multiaffine kernel."""
    r, n = 6.0, 129
    y, x = np.indices((n, n)) - n // 2
    u = (x ** 2 + y ** 2) / r ** 2
    if q == 1.0:
        shape = (1.0 - u) * np.exp(-u)
    else:
        base = 1.0 - (1.0 - q) * u / (2.0 - q)
        shape = (1.0 - u) * np.where(base > 0, np.clip(base, 0, None) ** (q / (1.0 - q)), 0.0)
    shape = shape - shape.mean()
    shape /= np.abs(shape).sum()
    got = _kernel_through("q_mexican", r, q_tsallis=q)
    live = got > 0                      # beyond the kernel's reach a delta has no support
    assert live[u <= 1.0].all()         # ...but the whole positive core is measured
    want = np.abs(shape)                # FFT convolution at the app's float32 default:
    np.testing.assert_allclose(got[live], want[live], rtol=1e-5, atol=1e-6 * want.max())


def test_q_mexican_recovers_the_ricker_at_q_one_and_refuses_q_two():
    """q -> 1 is the usual Mexican hat (g2). At q = 2 the 2-D Laplacian of [e_q]^(2-q) = 1
    vanishes identically, and above 2 the profile never changes sign: no 2-D wavelet exists
    there (Borges's q < 3 bound is the 1-D one)."""
    f = fbm2d(64, 0.5, seed=4)
    np.testing.assert_array_equal(mc.ricker_projections(f, [1.0, 3.0], wavelet="q_mexican",
                                                        q_tsallis=1.0),
                                  mc.ricker_projections(f, [1.0, 3.0], wavelet="g2"))
    for q in (2.0, 2.5):
        with pytest.raises(ValueError, match="q < 2"):
            mc.ricker_projections(f, [2.0], wavelet="q_mexican", q_tsallis=q)


def test_q_mexican_meets_lorentzian_marr_where_the_reference_says():
    """devices_reference.tex: q_mexican at q equals lorentzian_marr at beta = (2-q)/(q-1) for
    1 < q < 2 (the relabelled Laplacian-of-q-Gaussian family) -- e.g. q = 1.5 == beta = 1."""
    f = fbm2d(64, 0.5, seed=6)
    for q in (1.25, 1.5, 1.8):
        np.testing.assert_allclose(
            mc.ricker_projections(f, [1.0, 3.0], wavelet="q_mexican", q_tsallis=q),
            mc.ricker_projections(f, [1.0, 3.0], wavelet="lorentzian_marr",
                                  beta=(2.0 - q) / (q - 1.0)), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("q", [0.5, 1.0])
@pytest.mark.parametrize("h", [0.3, 0.6])
def test_multiaffine_route_is_l1_normalized_at_a_cusp(q, h):
    """The multiaffine route is L1 too (Turiel's Psi_r = r^-d Psi(x/r), realised on the grid:
    exact zero mean, then unit L1 norm at every scale), so at a cusp |x - x0|^h,
    |T(x0, r)| ~ r^h. Read over r >= 12 px: below ~10 px the discretization of kernel and cusp
    biases the local slope (sharp-cutoff kernels, q < 1, the most)."""
    n = 385
    c = n // 2
    y, x = np.indices((n, n)) - c
    scales = np.geomspace(12.0, 40.0, 6)
    T = mc.ricker_projections(np.hypot(x, y) ** h, scales, wavelet="q_mexican", q_tsallis=q)
    slope = np.polyfit(np.log(scales), np.log(T[:, c, c]), 1)[0]
    assert slope == pytest.approx(h, abs=0.03)
