# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The directional thermodynamic formalism on the hyperbolic coefficients: the index families,
the structure functions of a product field, and the fractional Brownian sheet's closed forms
(Ben Slimane et al. 2019, Theorem 7): η(p, e_i) = p H_i, η(p) = p min(H₁, H₂), the anisotropy
profile on the smoother axis's side; an isotropic field reads equal exponents and α₁* = 1."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pywt")

from dynamix.core import directional as dr                                 # noqa: E402
from dynamix.core import dwt                                                # noqa: E402

P = np.array([0.5, 1.0, 2.0, 3.0])


def test_the_anisotropy_tie_break_is_symmetric_between_axes():
    a = np.array([.5, 1., 1.5])
    result = dr.directional_scaling({"alpha": a, "eta_alpha": np.array([a, 2 - a])})
    assert result["alpha_star_e1"][0] == 1.5
    assert result["alpha_star_e2"][1] == 1.5


def fbm_sheet(n, h1, h2, seed=0):
    """Fourier synthesis of a fractional Brownian sheet: the spectral density is the product
    of the two 1-D fBm densities, |k_x|^{−(2H₁+1)} |k_y|^{−(2H₂+1)}, so increments along x
    scale with H₁ and along y with H₂; periodic on the torus, unit variance."""
    from dynamix.core.fft_policy import double

    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(n)[None, :]
    ky = np.fft.fftfreq(n)[:, None]
    with np.errstate(divide="ignore"):
        ax = np.where(kx == 0, 0.0, np.abs(kx) ** (-(h1 + 0.5)))
        ay = np.where(ky == 0, 0.0, np.abs(ky) ** (-(h2 + 0.5)))
    amp = ax * ay
    amp[0, :] = 0.0
    amp[:, 0] = 0.0
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    f = np.asarray(double().ifft2(amp * phase)).real
    return (f - f.mean()) / f.std()


def test_the_families_follow_the_anisotropy_and_reach_the_coarsest_level():
    around, edge_x, edge_y = dr.families(4, 0.5, 6)                  # [jα₁] = 2, [jα₂] = 6
    assert set(around) == {(x, y) for x in (1, 2, 3) for y in (5, 6)}
    assert edge_x == [(2, 6)] and edge_y == [(2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
    around, edge_x, edge_y = dr.families(3, 1.0, 6)                  # the diagonal: (3, 3)
    assert (3, 3) in around and len(around) == 9
    assert edge_x == [(3, y) for y in range(3, 7)] and edge_y == [(x, 3) for x in range(3, 7)]
    assert list(dr.alpha_grid()) == pytest.approx(list(np.arange(1, 16) / 8.0))


def test_a_product_field_has_separable_structure_functions():
    """u(x) v(y): log2 S_{j₁j₂}(p) splits into a term in j₁ and a term in j₂, so its second
    mixed difference vanishes."""
    rng = np.random.default_rng(1)
    u = np.cumsum(rng.standard_normal(256))
    v = np.cumsum(rng.standard_normal(256))
    coef = dr.coefficients(np.outer(v - v.mean(), u - u.mean()), "sym4", 4)
    assert coef["levels"] == 4 and coef["border_cells"] == dwt.border_cells("sym4") - 1
    S = dr.structure_functions(coef, P)
    assert S.shape == (4, 4, 4) and np.isfinite(S).all()
    mixed = S[:, 1:, 1:] - S[:, :-1, 1:] - S[:, 1:, :-1] + S[:, :-1, :-1]
    assert np.abs(mixed).max() < 0.15                                 # the border trims each axis a little


def _knee(leg, d=2.0):
    """The smallest h at which the bound reaches d."""
    L = leg["L"]
    return float(leg["h"][np.flatnonzero(np.isfinite(L) & (L >= d - 0.02))[0]])


def test_the_fractional_brownian_sheet_reads_its_two_exponents_and_their_minimum():
    """Haar (no border) to level 7 on 1024², fitted from level 2: the anisotropies α₁ ∈ [½, 1½]
    keep three unclipped levels; η_α(p) follows p·min(α₁H₁, α₂H₂) there, the sup over α reads
    each axis's exponent (the sup of noisy estimates leans high), the diagonal their minimum.
    At the finest levels [jα] cannot resolve a fractional α (it sticks at 1), so the fit
    starts at level 2 and anisotropy needs depth."""
    h1, h2 = 0.3, 0.7
    p_list = np.concatenate([P, [4.0, 6.0, 8.0]])        # the bound needs p past η_α(p) = d
    r = dr.analyse(fbm_sheet(1024, h1, h2, seed=2), "haar", 7, j_min=2, p_list=p_list)
    assert r["eta_alpha"].shape == (7, 15) and r["log2_S"].shape == (7, 7, 7)
    mid = (r["alpha"] >= 0.5) & (r["alpha"] <= 1.5)
    assert np.all(r["n_levels"][mid] >= 3) and np.all(np.isnan(r["eta_alpha"][:, r["n_levels"] < 3]))
    theory = np.minimum(r["alpha"] * h1, (2.0 - r["alpha"]) * h2)
    for i, p in enumerate(P):
        assert np.allclose(r["eta_alpha"][i, mid] / p, theory[mid], atol=0.12), p
        assert abs(r["eta_e1"][i] / p - h1) < 0.12, (p, r["eta_e1"][i] / p)
        assert abs(r["eta_e2"][i] / p - h2) < 0.1, (p, r["eta_e2"][i] / p)
        assert abs(r["eta_iso"][i] / p - min(h1, h2)) < 0.1, (p, r["eta_iso"][i] / p)
    low = slice(0, P.size)                               # the profile is read at moderate p
    assert np.all(r["alpha_star_e1"][low] >= 1.0)        # the smoother axis is y: α₁* ≥ 1 …
    assert np.all(r["alpha_star_e2"][low] <= 1.0)        # … and α₂* ≤ 1 (2H₁/(H₁+H₂) = 0.6)
    L1, L2 = r["L_e1"], r["L_e2"]
    assert L1["h"].shape == L1["L"].shape and np.nanmax(L1["L"]) <= 2.0 + 1e-9
    # The one-sided bound's knee: exact at H₁ along the rough axis (the diagonal α, α_min = 1,
    # falls below d at once); along the smooth axis the anisotropies that bind carry α_min < 1,
    # whose flank falls only once p exceeds d/(α₂(H₂ − H)) — past this grid's p = 8 near H₂ — so
    # the knee lands between the two exponents.
    assert abs(_knee(L1) - h1) < 0.05 and h1 - 0.05 <= _knee(L2) <= h2 + 0.05
    assert r["n_levels"][np.argmin(np.abs(r["alpha"] - 1.0))] == 5           # j = 2..6 unclipped at α = 1


def test_an_isotropic_field_reads_equal_exponents_and_no_anisotropy():
    import sys
    sys.path.insert(0, "tests")
    from conftest import fbm2d

    r = dr.analyse(fbm2d(1024, 0.5, seed=3), "sym4", 6, j_min=1, p_list=P)
    for i, p in enumerate(P):
        for key in ("eta_e1", "eta_e2", "eta_iso"):
            assert abs(r[key][i] / p - 0.5) < 0.1, (key, p, r[key][i] / p)
        assert r["eta_e1"][i] == pytest.approx(r["eta_e2"][i], abs=0.05 * p)
    assert np.all(np.abs(r["alpha_star_e1"] - 1.0) <= 0.125 + 1e-9)       # within one grid step
    assert np.all(np.abs(r["alpha_star_e2"] - 1.0) <= 0.125 + 1e-9)
    assert np.all(r["valid_cells"] > 0)


def test_the_legendre_bound_of_a_closed_form_eta():
    """η_α(p) = p(α₁H₁ ∧ α₂H₂) exactly (the sheet's limit): the bound L(H, e₁) reaches d = 2 at
    H = H₁ and falls below it to the left; an α whose η_α never reaches d on the p grid
    contributes nothing."""
    alphas = dr.alpha_grid()
    p = np.linspace(0.5, 8.0, 31)
    eta = p[:, None] * np.minimum(alphas * 0.3, (2.0 - alphas) * 0.7)[None, :]
    leg = dr.directional_legendre({"alpha": alphas, "eta_alpha": eta}, p, direction=1)
    assert abs(_knee(leg) - 0.3) < 0.03 and np.nanmax(leg["L"]) == pytest.approx(2.0)
    below = leg["L"][leg["h"] < 0.2]
    assert np.all(np.isnan(below) | (below < 1.5))                       # the left flank falls
    none = dr.directional_legendre({"alpha": alphas, "eta_alpha": eta * 0.0 + 0.1}, p)
    assert np.all(np.isnan(none["L"]))
