# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Spec 15's nonconcave-spectrum options (Touchette & Beck 2005), off by default.

The Legendre route returns the concave envelope of D(h); the kink check flags the jump in h(q) and
the straight D(h) segment it leaves, and the Gaussian (generalised canonical) ensemble,
W ∝ |T|^q exp(γ (ln|T|)² / ln(a/L)), reaches points inside a nonconcave dip. Fixtures are exact:
at every scale, n(h) = N0 (a/L)^-f(h) points of modulus (a/L)^h, so f is the true spectrum.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.nonconcave import gaussian_ensemble, kink_flags

SCALES = 2.0 ** np.arange(3, 9)                        # 8 .. 256 px
#: Far out, so ln(a/L) ≈ -50: the canonical fit's finite-size term, about 1/(2 ln(a/L)) on the
#: slopes (the weights' Gaussian spread around the saddle), drops to about 0.01.
L = 256.0 * np.exp(50.0)


def _f_dip(h):
    """A concave cap with a dip at h = 0.5, about 0.064 below the envelope (f'' = 10 there)."""
    return 0.9 - 1.0 * (h - 0.5) ** 2 - 0.12 * np.exp(-((h - 0.5) ** 2) / (2 * 0.1 ** 2))


def _f_cap(h):
    return 0.9 - 2.2 * (h - 0.5) ** 2


def _sample(f, h=np.linspace(0.15, 0.85, 141), n0=1e3):
    mods = [(a / L) ** h for a in SCALES]
    counts = [n0 * (a / L) ** (-f(h)) for a in SCALES]
    return mods, counts, h


def _upper_hull(x, y):
    """The concave envelope of the points (x, y), evaluated at x."""
    hull = []
    for p in zip(x, y):
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(p)
    hx, hy = zip(*hull)
    return np.interp(x, hx, hy)


def test_gamma_zero_is_the_canonical_direct_fit():
    mods, counts, _h = _sample(_f_cap)
    q = np.linspace(-1.2, 1.2, 13)              # saddles h = 0.5 - q/4.4 stay inside the h grid
    g = gaussian_ensemble(mods, SCALES, q, [0.0], 3.0, 8.0, counts_per_scale=counts, length_px=L)
    # the canonical direct fit by hand: W ∝ n |T|^q, h = slope Σ W ln|T|, D = slope Σ W ln(W/n)
    la = np.log(SCALES)
    for i, qq in enumerate(q):
        sh, sd = [], []
        for m, n in zip(mods, counts):
            w = n * m ** qq
            w = w / w.sum()
            sh.append(np.sum(w * np.log(m)))
            sd.append(np.sum(w * np.log(w / n)))
        assert g["h"][i, 0] == pytest.approx(np.polyfit(la, sh, 1)[0], abs=1e-9)
        assert g["D"][i, 0] == pytest.approx(np.polyfit(la, sd, 1)[0], abs=1e-9)
    # an exact concave spectrum is read back exactly: D(h) on f
    ok = np.isfinite(g["h"][:, 0])
    assert np.allclose(g["D"][ok, 0], _f_cap(g["h"][ok, 0]), atol=0.015)


def test_the_consistency_identity_holds_at_every_q_and_gamma():
    mods, counts, _h = _sample(_f_dip)
    g = gaussian_ensemble(mods, SCALES, np.linspace(-2, 2, 9), [0.0, 2.0, 8.0, 20.0], 3.0, 8.0,
                          counts_per_scale=counts, length_px=L)
    assert np.nanmax(np.abs(g["residual"])) < 1e-9


def test_legendre_keeps_the_envelope_and_the_gaussian_ensemble_enters_the_dip():
    mods, counts, h = _sample(_f_dip)
    # The saddle f'(h) = q + 2γh: a point at h needs q ≈ f'(h) - 2γh, so q spans what γ asks.
    q = np.linspace(-20.0, 20.0, 161)
    gammas = np.linspace(0.0, 10.0, 21)
    g = gaussian_ensemble(mods, SCALES, q, gammas, 3.0, 8.0, counts_per_scale=counts, length_px=L)
    hull = lambda x: np.interp(x, h, _upper_hull(h, _f_dip(h)))          # noqa: E731
    std_h, std_d = g["h"][:, 0], g["D"][:, 0]
    ok = np.isfinite(std_h) & (np.abs(std_h - 0.5) < 0.3)              # saddles inside the grid
    assert np.all(std_d[ok] >= hull(std_h[ok]) - 0.015)                # γ = 0: the envelope
    depth = float(np.max(_upper_hull(h, _f_dip(h)) - _f_dip(h)))
    assert depth > 0.05
    inside = (np.abs(g["h"] - 0.5) < 0.06) & np.isfinite(g["D"])
    gap = hull(g["h"][inside]) - g["D"][inside]
    assert gap.max() > 0.6 * depth                                   # well inside the dip
    best = np.argmax(gap)
    assert g["D"][inside][best] == pytest.approx(_f_dip(g["h"][inside][best]), abs=0.015)


def test_the_kink_check_fires_on_a_first_order_transition_only():
    q = np.linspace(-3, 3, 61)
    lines = np.minimum(0.3 * q - 0.2, 0.7 * q - 0.2)                 # τ(q) with a kink at q = 0
    flags = kink_flags(q, lines)
    assert len(flags) == 1
    k = flags[0]
    assert k["q"] == pytest.approx(0.0, abs=0.1)
    assert (k["h_lo"], k["h_hi"]) == pytest.approx((0.3, 0.7), abs=1e-6)
    smooth = 0.5 * q - 0.05 * q ** 2                                  # a log-normal τ(q)
    assert kink_flags(q, smooth) == []


def test_the_nonconcave_fixture_shows_its_kink_at_gamma_zero():
    mods, counts, _h = _sample(_f_dip)
    q = np.linspace(-1.2, 1.2, 49)
    g = gaussian_ensemble(mods, SCALES, q, [0.0], 3.0, 8.0, counts_per_scale=counts, length_px=L)
    flags = kink_flags(q, g["tau"][:, 0])
    assert flags and abs(flags[0]["q"]) < 0.3 and flags[0]["h_hi"] - flags[0]["h_lo"] > 0.15
