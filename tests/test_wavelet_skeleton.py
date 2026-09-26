# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Tang-You wavelet skeleton core (dynamix/core/wavelet_skeleton.py) -- the constructed
compact-support wavelet (Tang & You 2003, eq 4-5) and the modulus-minima thinning
(You/Chen/Fang/Tang 2006, Algorithm 1), reimplemented against the corpus papers
(tang_you_2003_ribbon_skeleton_wavelet, you_etal_2006_thinning_modulus_minima).

The acceptance tests ARE the papers' own theorems, measured on synthetic ribbons: maxima
separation == the SCALE (width-invariant -- positions exactly equal across stroke widths
4/8/12 at fixed s), modulus minimum on the centerline within 1 px at every (d, s) tested."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import wavelet_skeleton as wsk


def _stripe(n=129, d=8, axis="v"):
    f = np.zeros((n, n))
    c = n // 2
    if axis == "v":
        f[:, c - d // 2: c + d // 2] = 1.0
    else:
        f[c - d // 2: c + d // 2, :] = 1.0
    return f


# ------------------------------------------------------------------- the constructed wavelet

def test_psi_is_bounded_continuous_and_compactly_supported():
    """Eq 4-5's internal consistency, measured: the three 1/x singularities cancel exactly
    at 0 (coefficients -1 + 9 - 8), each piece vanishes at the breakpoint where its root
    would go imaginary (machine-exact continuity at 1/4, 3/4), and psi dies at 1."""
    x = np.linspace(1e-9, 1.5, 100001)
    p = wsk.psi_radial(x)
    assert np.all(np.isfinite(p))
    assert abs(wsk.psi_radial(np.array([1e-7]))[0]) < 1e-4          # bounded at 0+
    assert np.abs(p).max() < 1.0                                     # measured range [-0.97, 0.33]
    for b in (0.25, 0.75, 1.0):
        lo = wsk.psi_radial(np.array([b - 1e-9]))[0]
        hi = wsk.psi_radial(np.array([b + 1e-9]))[0]
        assert abs(lo - hi) < 1e-6, b
    assert np.all(p[x >= 1.0] == 0.0)


def test_phi_is_the_compact_support_primitive():
    """The paper's phi needs its integration constant anchored at the support edge
    (phi(1) = 0; the literal int_0^x reading gives phi(1) = -0.383 and no compact support)
    -- so phi(rho) = -int_rho^1 psi: a positive bump, phi(0) = +0.383, zero beyond 1."""
    rho = np.linspace(0.0, 1.2, 1001)
    ph = wsk.phi_radial(rho)
    assert ph[0] == pytest.approx(0.3829, abs=2e-3)
    assert np.all(ph[rho >= 1.0] == 0.0)
    assert np.all(ph >= 0.0)
    assert abs(wsk.phi_radial(np.array([0.999]))[0]) < 5e-3          # continuous to 0 at the edge


def test_theta_kernel_is_unit_sum_with_support_radius_s():
    k = wsk.theta_kernel((64, 64), 9.0)
    assert k.sum() == pytest.approx(1.0)
    kc = np.fft.fftshift(k)
    yy, xx = np.mgrid[:64, :64]
    rho = np.hypot(xx - 32, yy - 32)
    assert np.all(kc[rho > 9.5] == 0.0)                              # compact: dead past s
    assert kc[32, 32] > 0.0


# ------------------------------------------------------------------- the papers' theorems

def test_ribbon_maxima_separation_is_the_scale_and_width_invariant():
    """Tang-You 2003 property 3, measured: the two maxima contours sit ~s apart, and their
    POSITIONS are exactly equal across stroke widths 4/8/12 at fixed s -- the property that
    neither the Gaussian nor the quadratic spline delivers."""
    positions = {}
    for d in (4, 8, 12):
        W1, W2 = wsk.gradient_wt(_stripe(d=d), 16.0)
        row = np.hypot(W1, W2)[64]
        left = int(np.argmax(row[:64])); right = 64 + int(np.argmax(row[64:]))
        assert abs((right - left) - 16.0) <= 2.0, (d, right - left)
        positions[d] = (left, right)
    assert positions[4] == positions[8] == positions[12], positions


def test_modulus_minimum_sits_on_the_centerline_scale_independently():
    """You et al. 2006: the modulus minimum is the medial axis, at EVERY scale."""
    f = _stripe(d=8)
    for s in (12.0, 20.0):
        W1, W2 = wsk.gradient_wt(f, s)
        row = np.hypot(W1, W2)[64]
        assert abs(int(np.argmin(row[58:71])) + 58 - 64) <= 1, s


def test_modulus_minima_mask_finds_the_centerline_and_ignores_flats():
    f = _stripe(d=8)
    W1, W2 = wsk.gradient_wt(f, 12.0)
    m = wsk.modulus_minima(W1, W2)
    cols = np.where(m[64])[0]
    assert len(cols) > 0
    assert np.all(np.abs(cols - 64) <= 1), cols                      # only the axis, no ghosts
    flatW = np.zeros((64, 64))
    assert not wsk.modulus_minima(flatW, flatW).any()                # no support -> nothing


# ------------------------------------------------------------------- Algorithm 1

def test_skeletonize_a_stripe_to_its_axis():
    """Scale >= stroke width (the paper's property-2 hypothesis: the two edge responses
    must overlap for a medial signal to exist)."""
    res = wsk.skeletonize(_stripe(d=8), s1=8.0, s2=6.0)
    ys, xs_ = np.where(res["skeleton"])
    assert len(ys) > 0
    assert np.all(np.abs(xs_ - 64) <= 1)                             # thin, on the axis
    assert len(np.unique(ys)) >= 90                                  # spans most of the stripe


def test_skeletonize_is_width_and_gray_level_robust():
    """The 2006 abstract's claim: works across stroke widths and on gray-level data. The
    ridge-pair gate even RELAXES the paper's scale-covers-width hypothesis: at s=6 on a
    width-14 stripe (edges at +-7, ridge tails reaching the center within reach) the true
    axis is still found exactly (measured: cols 63-64, nothing else) -- detection needs
    only that the flanking tails within reach clear the significance floor."""
    for d, s1 in ((4, 6.0), (14, 16.0), (14, 6.0)):
        res = wsk.skeletonize(_stripe(d=d), s1=s1, s2=6.0)
        _ys, xs_ = np.where(res["skeleton"])
        assert len(xs_) > 0 and np.all(np.abs(xs_ - 64) <= 1), (d, s1)
    res = wsk.skeletonize(0.3 * _stripe(d=8) + 0.1, s1=8.0, s2=6.0)  # gray levels
    _ys, xs_ = np.where(res["skeleton"])
    assert len(xs_) > 0 and np.all(np.abs(xs_ - 64) <= 1)


def test_skeletonize_flat_field_is_empty_and_background_stays_clean():
    res = wsk.skeletonize(np.full((64, 64), 2.5), s1=6.0, s2=6.0)
    assert not res["skeleton"].any()
    res = wsk.skeletonize(_stripe(d=8), s1=8.0, s2=6.0)
    assert not res["skeleton"][:, :40].any()                         # far background: nothing
    assert not res["skeleton"][:, 90:].any()


def test_skeletonize_result_carries_the_stage_record():
    res = wsk.skeletonize(_stripe(d=8), s1=6.0, s2=4.0, n_stages=3)
    assert res["params"]["s1"] == 6.0 and res["params"]["s2"] == 4.0
    assert res["params"]["n_stages"] == 3 and res["params"]["t_frac"] == 0.5
    assert res["skeleton"].dtype == bool
    assert res["mod"].shape == res["skeleton"].shape                 # stage-1 field modulus


# --------------------------------------------------- the adversarial-geometry suite (pinned)

def test_adversarial_geometries_are_ghost_free():
    """Adversarial geometries, pinned against the ridge-pair test: one-sided gates gave 68%
    ghost pixels on a 45-deg stripe, 95% with 1% noise, and support-boundary bands at
    ~s+d/2; the pair test with pair DOMINANCE (a valley whose flanking pair is under half the
    strongest reachable ridge is on someone else's slope -- the ring's inner-tail chord
    self-pairing) measures 0% >2px on all four geometries, max deviation <= 1 px."""
    n = 129
    yy, xx = np.mgrid[:n, :n].astype(float)
    rr = np.hypot(xx - 64, yy - 64)
    f = np.zeros((n, n)); f[:, 60:68] = 1.0
    rng = np.random.default_rng(0)
    cases = [
        (f, lambda y, x: np.abs(x - 63.5)),
        ((np.abs((xx - yy) / np.sqrt(2)) <= 4.0).astype(float),
         lambda y, x: np.abs((x - y) / np.sqrt(2))),
        (((rr >= 36) & (rr <= 44)).astype(float),
         lambda y, x: np.abs(np.hypot(x - 64, y - 64) - 40)),
        (f + 0.01 * rng.normal(size=f.shape), lambda y, x: np.abs(x - 63.5)),
    ]
    for i, (fld, dist) in enumerate(cases):
        res = wsk.skeletonize(fld, s1=8.0, s2=6.0)
        ys, xs_ = np.where(res["skeleton"])
        assert len(ys) > 50, i
        assert dist(ys, xs_).max() <= 2.0, (i, dist(ys, xs_).max())


def test_nan_regions_grow_a_distrust_band_not_phantoms():
    """Mean-filling a NaN disk on a structureless ramp fabricated a 149-px phantom ribbon, 42 px
    of it INSIDE the data-free region. The wtmm extrema2d NaN-dilation doctrine applies -- one
    kernel support around every non-finite pixel is distrusted, same policy as the raster
    border."""
    n = 129
    yy, xx = np.mgrid[:n, :n].astype(float)
    rr = np.hypot(xx - 64, yy - 64)
    ramp = (xx / n).copy()
    ramp[rr < 10] = np.nan
    assert not wsk.skeletonize(ramp, s1=8.0, s2=6.0)["skeleton"].any()
    f = np.zeros((n, n)); f[:, 60:68] = 1.0
    f[np.hypot(xx - 30, yy - 30) < 8] = np.nan
    res = wsk.skeletonize(f, s1=8.0, s2=6.0)
    ys, xs_ = np.where(res["skeleton"])
    assert len(ys) > 50
    assert np.all(np.abs(xs_ - 63.5) <= 2)                     # the axis, unpolluted
    assert not (np.hypot(xs_ - 30, ys - 30) < 20).any()        # nothing near the disk


def test_gap_corridor_is_the_polarity_blind_reading():
    """Documented deviation, pinned: a stroke GAP does not bridge -- the gap corridor is
    itself a dark ribbon between bright strokes and draws its own perpendicular medial
    bar. For figure/ground data that is an artifact; for a general raster it is the
    honest sign-blind reading (the polarity knob is the reserved refinement)."""
    n = 129
    f = np.zeros((n, n)); f[:, 60:68] = 1.0
    f[58:70, :] = 0.0
    res = wsk.skeletonize(f, s1=8.0, s2=6.0)
    ys, xs_ = np.where(res["skeleton"])
    bar = (np.abs(ys - 63.5) <= 2) & (np.abs(xs_ - 63.5) > 2)
    on_axis = np.abs(xs_ - 63.5) <= 2
    assert bar.sum() > 0                                       # the corridor's own axis
    assert (on_axis & (np.abs(ys - 63.5) > 8)).sum() > 80      # the stroke axes, intact


def test_theta_kernel_menu_is_the_papers_comparison_set():
    """Tang-You 2003 benchmarks kappa against "Gaussian function and quadratic spline" --
    the menu is exactly that set. Both compact kernels die at radius s; the Gaussian
    (their critique: no compact support) at sigma = s/2 carries ~86% of its mass inside s (the 2-D Rayleigh tail e^-2).
    bspline2 evaluates through the VERBATIM Creep copy (drift-guarded provenance)."""
    for kernel in wsk.THETA_KERNELS:
        k = wsk.theta_kernel((64, 64), 9.0, kernel)
        assert k.sum() == pytest.approx(1.0), kernel
        kc = np.fft.fftshift(k)
        yy, xx = np.mgrid[:64, :64]
        rho = np.hypot(xx - 32, yy - 32)
        if kernel in ("tang_you", "bspline2"):
            assert np.all(kc[rho > 9.5] == 0.0), kernel      # compact: dead past s
        else:
            assert 0.08 < kc[rho > 9.5].sum() < 0.16, kernel # gaussian: the e^-2 tail
    with pytest.raises(ValueError, match="kernel"):
        wsk.theta_kernel((32, 32), 4.0, "haar")


def test_every_kernel_finds_the_stripe_axis():
    """The skeleton machinery is kernel-agnostic: all three smoothing profiles recover the
    axis on the clean stripe (the theorems' EXACTNESS is kappa's alone -- separation = s,
    width-invariance -- but the medial minima exist for any radial smoother)."""
    f = _stripe(d=8)
    for kernel in wsk.THETA_KERNELS:
        res = wsk.skeletonize(f, s1=8.0, s2=6.0, kernel=kernel)
        ys, xs_ = np.where(res["skeleton"])
        assert len(ys) > 50, kernel
        assert np.all(np.abs(xs_ - 63.5) <= 2), kernel
        assert res["params"]["kernel"] == kernel


def test_pair_dom_knob_binds():
    """pair_dom=0 disables the dominance gate (the ring's inner-tail circle returns);
    the default 0.5 keeps the ring ghost-free -- the knob genuinely gates."""
    n = 129
    yy, xx = np.mgrid[:n, :n].astype(float)
    rr = np.hypot(xx - 64, yy - 64)
    ring = ((rr >= 36) & (rr <= 44)).astype(float)
    on = wsk.skeletonize(ring, s1=8.0, s2=6.0, pair_dom=0.5)
    off = wsk.skeletonize(ring, s1=8.0, s2=6.0, pair_dom=0.0)
    d_on = np.abs(np.hypot(*(np.where(on["skeleton"])[::-1] - np.array([[64], [64]]))) - 40)
    assert d_on.max() <= 2.0
    assert off["skeleton"].sum() > on["skeleton"].sum()      # gate off: extra structure back
