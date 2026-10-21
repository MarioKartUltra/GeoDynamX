# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Wavelet leaders, p-leaders and (θ, ω)-leaders on the dyadic pyramid: the registration and
border of the bands, the three quantities against brute-force references and closed forms, the
readouts on fBm, a binomial cascade, cusps and chirps."""
from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("pywt")

from dynamix.core import dwt, leaders as L                               # noqa: E402
from dynamix.core.spectra import fit_spectra                            # noqa: E402

sys.path.insert(0, "tests")
from conftest import fbm2d                                              # noqa: E402

Q = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0])


def _tables(r):
    return L.tables(r["samples"], Q)


def _c(r, method, gamma=0.0):
    """The method's log-cumulants over the fitted levels, original frame."""
    return L.cumulants(_tables(r)[method], r["j_min"], r["levels"], gamma)["original"]


def _qi(table, q):
    return int(np.argmin(np.abs(table["q_list"] - q)))


# ------------------------------------------------------------------ registration and border

@pytest.mark.parametrize("name", dwt.BANKS)
def test_registration_settles_to_the_child_shift_along_the_low_pass_axis(name):
    """From level 4 up the horizontal band's column shift and the vertical band's row shift are
    the bank's child shift, and the band-pass axes and the diagonal band sit on their own cells."""
    reg = dwt.registration(name, 6)
    cs = dwt.child_shift(name)
    for j in (4, 5, 6):
        assert reg[(j, "horizontal")][1] == cs and reg[(j, "vertical")][0] == cs
        assert reg[(j, "horizontal")][0] == 0 and reg[(j, "vertical")][1] == 0
        assert reg[(j, "diagonal")][:2] == (0, 0)
        assert reg[(j, "detail")][0] == 0
    for key, value in reg.items():
        residual = value[2:] if key[1] != "detail" else value[1:]
        assert all(abs(r) <= 2 ** key[0] / 2 + 1e-9 for r in residual)      # at most half a cell


def test_the_two_dimensional_atom_centroid_is_the_pair_of_one_dimensional_ones():
    """The horizontal band's atom is ψ along rows times φ along columns: its centroid, measured
    on a 2-D zero pyramid as test_dwt does, equals the two 1-D centroids."""
    n, j, k = 512, 3, 32
    zero = dwt.pyramid(np.zeros((n, n)), "sym7", 4)
    c = [np.zeros_like(zero[0])] + [tuple(np.zeros_like(d) for d in lev) for lev in zero[1:]]
    c[-j][0][k, k] = 1.0
    atom = np.abs(dwt.inverse(c, "sym7"))
    w = atom / atom.sum()
    ii, jj = np.indices(atom.shape)
    nominal = 2 ** j * (k + 0.5) - 0.5
    assert abs((w * ii).sum() - nominal - dwt.atom_centroid("sym7", j, "psi")) < 1e-8
    assert abs((w * jj).sum() - nominal - dwt.atom_centroid("sym7", j, "phi")) < 1e-8


def test_border_cells_pinned_per_bank():
    assert [dwt.border_cells(b) for b in ("sym4", "sym7", "db4", "bl3", "haar")] == [5, 6, 5, 13, 1]


@pytest.mark.parametrize("name", ["sym4", "sym8", "db6", "coif5", "bl3"])
def test_cells_beyond_the_border_match_the_periodic_parent(name):
    """A field cut from a periodic parent and mirror-extended: on the cells the border keeps, its
    leaders agree with the parent's to 1e-3 of the level mean (the border's criterion) and the
    bare coefficients, one cell closer to the edge, to 3e-3 (the coiflets' long tails)."""
    parent = fbm2d(1024, 0.5, seed=11)
    crop = parent[256:768, 256:768]
    e = dwt.border_cells(name)
    cp, cc = L.coefficients(parent, name, 5), L.coefficients(crop, name, 5)
    lp, lc = L.leaders(cp), L.leaders(cc)
    for j in range(1, 6):
        n, off = 1024 // 2 ** j, 256 // 2 ** j
        vc = L.valid_cells(cc, e - 1)[j - 1][:n, :n]
        vl = L.valid_cells(cc, e)[j - 1][:n, :n]
        P, C = cp["mods"][j - 1][:, off:off + n, off:off + n], cc["mods"][j - 1][:, :n, :n]
        if vc.any():
            assert np.abs(P - C)[:, vc].max() <= 3e-3 * P.mean()
        if vl.any():
            Pl, Cl = lp[j - 1][off:off + n, off:off + n], lc[j - 1][:n, :n]
            assert np.abs(Pl - Cl)[vl].max() <= 1e-3 * Pl.mean()


def test_mirror_extends_a_signal_like_an_image():
    x = np.arange(5.0)
    m = dwt.mirror(x, (16,))
    assert m.shape == (16,) and np.allclose(m[:5], x) and np.allclose(m[8:], m[:8][::-1])
    assert np.allclose(dwt.mirror(x), np.concatenate([x, x[::-1]]))


# ------------------------------------------------------------- the quantities, brute force

def _coef(mods):
    """A coefficient dict over given per-level stacks (no bad pixels, the whole grid the field)."""
    return {"mods": mods, "bad": [np.zeros(m.shape[1:], bool) for m in mods],
            "quarter": [m.shape[1:] for m in mods], "levels": len(mods), "ndim": mods[0].ndim - 1,
            "gamma": 0.0}


def test_leaders_are_the_sup_over_the_subtree_of_the_three_by_three_block():
    rng = np.random.default_rng(0)
    mods = [rng.random((3, 32 // 2 ** (j - 1), 32 // 2 ** (j - 1))) for j in range(1, 4)]
    lead = L.leaders(_coef(mods))
    J = len(mods)
    for j in range(1, J + 1):
        n = mods[j - 1].shape[1]
        for r in range(n):
            for c in range(n):
                best = 0.0
                for jj in range(1, j + 1):                       # λ itself and its descendants
                    f = 2 ** (j - jj)
                    rows = [(x % (n * f)) for x in range((r - 1) * f, (r + 2) * f)]
                    cols = [(x % (n * f)) for x in range((c - 1) * f, (c + 2) * f)]
                    best = max(best, mods[jj - 1][:, rows][:, :, cols].max())
                assert abs(lead[j - 1][r, c] - best) < 1e-12


def test_p_leaders_weight_the_finer_levels_by_a_quarter_per_level_and_tend_to_the_leaders():
    rng = np.random.default_rng(1)
    mods = [rng.random((3, 16 // 2 ** (j - 1), 16 // 2 ** (j - 1))) for j in range(1, 3)]
    coef = _coef(mods)
    p = 2.0
    lp = L.pleaders(coef, p)
    # level 2, cell (1, 1): its own 3 × 3 block at level 2 plus the 6 × 6 block of level 1 at 1/4
    own = (mods[1] ** p)[:, 0:3, 0:3].sum()
    fine = 0.25 * (mods[0] ** p)[:, 0:6, 0:6].sum()
    assert abs(lp[1][1, 1] - (own + fine) ** (1 / p)) < 1e-12
    lead = L.leaders(coef)
    big = L.pleaders(coef, 64.0)
    for j in range(2):
        assert np.all(np.abs(np.log2(big[j] / lead[j])) < 0.1)


def test_wse_window_is_the_sup_within_the_half_width_of_the_coarse_centre():
    rng = np.random.default_rng(2)
    fine = rng.random((16, 16))
    coarse = L._window_max(fine, 1, 1.0)                 # 2 × finer; within one coarse cell
    assert coarse.shape == (8, 8)
    for r in range(8):
        for c in range(8):
            rows = [x % 16 for x in range(2 * r - 1, 2 * r + 3)]     # centres within 2 fine cells
            cols = [x % 16 for x in range(2 * c - 1, 2 * c + 3)]
            assert coarse[r, c] == fine[rows][:, cols].max()
    same = L._window_max(fine, 0, 1.0)                   # the level itself: the 3 × 3 block
    assert np.allclose(same, L._box(fine, 1, np.maximum))
    assert np.allclose(L._window_max(fine, 0, 0.0), fine)


def test_the_fixed_window_lies_inside_three_lambda_so_the_oscillation_ratio_is_nonnegative():
    rng = np.random.default_rng(3)
    mods = [rng.random((3, 32 // 2 ** (j - 1), 32 // 2 ** (j - 1))) for j in range(1, 4)]
    coef = _coef(mods)
    lead, weak = L.leaders(coef), L.wse(coef, L.wse_schedule(3, rule="fixed", depth=1, width=1.0))
    for a, b, o in zip(lead, weak, L.oscillation(lead, weak)):
        assert np.all(b <= a + 1e-12) and np.all(o >= -1e-12)


def test_the_dumeur_schedule_grows_with_the_resolution_index():
    sched = L.wse_schedule(6, rule="dumeur", log2_n=10.0)
    assert sched[1] == [(0, 8.0), (1, 9.0)]                    # floor(8^¼) = 1
    assert sched[5] == [(0, 4.0), (1, 5.0)]                      # j = 6: j_r = 4, depth 1
    assert L.wse_schedule(2, rule="fixed", depth=2, width=1.5) == [[(0, 1.5), (1, 1.5), (2, 1.5)]] * 2
    with pytest.raises(ValueError):
        L.wse_schedule(2, rule="other")


def test_dumeur_origin_neighborhood_matches_definition_2_2():
    fine = np.arange(32.)
    actual = L._window_max(fine, 1, 1., anchor="origin")
    for k, value in enumerate(actual):
        assert value == max(fine[(2*k+t) % 32] for t in range(-2, 3))


def test_wse_fitting_samples_use_spatial_decimation_and_omega_normalization():
    values = [np.arange(64.).reshape(8, 8)+1, np.ones((4, 4))]
    valid = {name: [np.ones_like(x, bool) for x in values] for name in L.METHODS+("coefficients",)}
    quantity = dict.fromkeys(L.METHODS, values)
    samples = L.sample_sets(quantity, values, valid, [2., 4.], (16, 16),
                            wse_schedule=[[(0, 2.)], [(0, 1.)]])
    s = samples["sets"]["wse"]
    assert s["decimation"] == [4, 2]
    assert s["valid"][0].sum() == 4
    t = L.tables(samples, [1.])["wse"]
    # omega^d times the selected sum, without border loss.
    assert 2**t["tau_qa"][0, 0] == pytest.approx(4*values[0][::4, ::4].sum())


def test_an_impulse_coefficient_enters_exactly_the_leaders_whose_block_contains_it():
    J = 4
    mods = [np.zeros((3, 64 // 2 ** (j - 1), 64 // 2 ** (j - 1))) for j in range(1, J + 1)]
    mods[1][2, 20, 9] = 1.0                                    # level 2, diagonal band, cell (20, 9)
    lead = L.leaders(_coef(mods))
    assert not lead[0].any()
    for j in range(2, J + 1):
        f = 2 ** (j - 2)
        hit = np.zeros(lead[j - 1].shape, bool)
        hit[20 // f - 1:20 // f + 2, 9 // f - 1:9 // f + 2] = True
        assert np.array_equal(lead[j - 1] > 0, hit)


def test_valid_cells_keep_the_interior_and_shun_non_finite_pixels():
    f = fbm2d(128, 0.5, seed=4)
    f[40, 70] = np.nan
    coef = L.coefficients(f, "sym4", 3)
    v = L.valid_cells(coef, 2)
    for j in range(1, 4):
        n = 128 // 2 ** j
        assert not v[j - 1][:2].any() and not v[j - 1][:, n - 2:n].any() and not v[j - 1][:, n:].any()
        r, c = 40 // 2 ** j, 70 // 2 ** j
        assert not v[j - 1][r - 2:r + 3, c - 2:c + 3].any()
        assert v[j - 1][r + 4, c + 4]


def test_coefficients_integrate_by_two_to_the_gamma_j():
    f = fbm2d(64, 0.5, seed=5)
    a, b = L.coefficients(f, "sym4", 3), L.coefficients(f, "sym4", 3, gamma=0.5)
    for j in range(1, 4):
        assert np.allclose(b["mods"][j - 1], a["mods"][j - 1] * 2 ** (0.5 * j))


def test_the_ribbon_identity_ties_the_two_dimensional_kernels_to_the_one_dimensional_ones():
    """A field constant along rows: the vertical band holds the row's 1-D L¹ coefficients, the
    2-D leaders are the 1-D leaders and the 2-D p-leaders are 3^{1/p} times the 1-D ones."""
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.standard_normal(1024))
    walk -= walk.mean()
    c1, c2 = L.coefficients(walk, "sym4", 5), L.coefficients(np.tile(walk, (256, 1)), "sym4", 5)
    l1, l2 = L.leaders(c1), L.leaders(c2)
    p1, p2 = L.pleaders(c1, 2.0), L.pleaders(c2, 2.0)
    e = dwt.border_cells("sym4")
    for j in range(1, 6):
        assert np.abs(c2["mods"][j - 1][1][0] - c1["mods"][j - 1][0]).max() < 1e-10
        assert c2["mods"][j - 1][[0, 2]].max() < 1e-9
        assert np.abs(l2[j - 1][e:-e] - l1[j - 1][None]).max() < 1e-10
        assert np.abs(p2[j - 1][e:-e] / 3 ** 0.5 - p1[j - 1][None]).max() < 1e-10


# ----------------------------------------------------------------------------- readouts

def test_local_exponent_reads_an_exact_pyramid_and_paint_tiles_the_cells():
    per_level = [np.full((16 // 2 ** (j - 1),) * 2, 2.0 ** (0.7 * j)) for j in range(1, 4)]
    valid = [np.ones(a.shape, bool) for a in per_level]
    h, r2 = L.local_exponent(per_level, valid, 1)
    assert h.shape == (16, 16) and np.allclose(h, 0.7) and np.allclose(r2, 1.0)
    valid[2][:] = False
    assert np.all(np.isnan(L.local_exponent(per_level, valid, 1)[0]))
    painted = L.paint(np.arange(4.0).reshape(2, 2), 2, (7, 8))
    assert painted.shape == (7, 8) and painted[3, 3] == 0 and painted[3, 4] == 1 and painted[4, 0] == 2


def test_the_structure_functions_count_cells_as_the_dyadic_grid_does_whatever_the_border_takes():
    """τ(0) = −d exactly: the table's count is the field's cell count per level, not the valid
    count the border thins faster at coarse levels."""
    r = L.run_leaders(fbm2d(512, 0.5, seed=6), scale_min=4, scale_max=32, wse_rule="fixed")
    samples = r["samples"]
    assert set(samples["sets"]) == {"leaders", "p-leaders", "wse", "coefficients"}
    assert list(samples["scales"]) == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert samples["cells"] == [(512 // 2 ** j) ** 2 for j in range(1, 6)]
    assert samples["sets"]["leaders"]["values"] is r["quantity"]["leaders"]          # no copy
    assert samples["sets"]["coefficients"]["positive_only"] and "positive_only" not in samples["sets"]["wse"]
    tables = _tables(r)
    for k in L.METHODS:
        t = tables[k]
        assert np.allclose(t["tau_qa"][_qi(t, 0.0)], 2 * np.log2(512) - 2 * np.arange(1, 6))
        assert np.allclose(fit_spectra(t, 2, 5)["tau"][_qi(t, 0.0)], -2.0)
        assert np.all(t["N_a"] == r["valid_cells"][k])
    assert np.all(tables["coefficients"]["N_a"] == 3 * np.asarray(r["valid_cells"]["coefficients"]))
    assert np.all(np.isnan(tables["coefficients"]["tau_qa"][Q < 0]))
    assert np.all(np.isfinite(tables["leaders"]["tau_qa"][Q < 0]))
    assert "tables" not in r and "cumulants" not in r                              # nothing fitted here
    one = L.run_leaders(np.cumsum(np.random.default_rng(0).standard_normal(4096)), scale_min=4,
                        scale_max=64, wse_rule="fixed")
    t1 = _tables(one)["leaders"]
    assert np.allclose(fit_spectra(t1, 2, 6)["tau"][_qi(t1, 0.0)], -1.0)


@pytest.mark.parametrize("H", [0.3, 0.5, 0.8])
def test_fbm_reads_its_hurst_exponent_on_every_method(H):
    """1024², sym4, 4–64 px: c1 within 0.03 of H and |c2| below 0.01 for leaders, p = 2 and the
    fixed-window WSE; the coefficients' ζ(2)/2 and h_min, η(2) where fBm puts them."""
    r = L.run_leaders(fbm2d(1024, H, seed=int(10 * H)), wavelet="sym4", scale_min=4, scale_max=64,
                      wse_rule="fixed")
    for k in L.METHODS:
        c = _c(r, k)
        assert abs(c["c1"] - H) < 0.03, (k, c)
        assert abs(c["c2"]) < 0.01, (k, c)
        assert r["admissibility"][k]["admissible"]
    t = _tables(r)["coefficients"]
    assert abs((fit_spectra(t, 2, 6)["tau"][_qi(t, 2.0)] + 2) / 2 - H) < 0.05
    adm = r["admissibility"]
    assert 0 < adm["h_min"]["original"] < H and abs(adm["eta"]["original"] - 2 * H) < 0.1
    assert r["zero_cells"] == [0] * 6


def test_the_dumeur_window_runs_wider_and_reads_fbm_within_a_tenth():
    r = L.run_leaders(fbm2d(1024, 0.5, seed=5), scale_min=4, scale_max=64, wse_rule="dumeur")
    assert r["wse_schedule"][1] == [(0, 8.0), (1, 9.0)]
    assert r["valid_cells"]["wse"][1] < r["valid_cells"]["leaders"][1]
    assert abs(_c(r, "wse")["c1"] - 0.5) < 0.1


def _cascade(n_levels, w, seed=0):
    """A 2-D binomial cascade: the four quadrant weights of ``w`` permuted at random per cell."""
    rng = np.random.default_rng(seed)
    mu = np.ones((1, 1))
    for _ in range(n_levels):
        ny, nx = mu.shape
        perm = np.stack([rng.permutation(w) for _ in range(ny * nx)]).reshape(ny, nx, 2, 2)
        mu = (mu[:, :, None, None] * perm).transpose(0, 2, 1, 3).reshape(2 * ny, 2 * nx)
    return mu


def test_binomial_cascade_density_needs_gamma_and_then_reads_its_closed_form():
    """The density of a 4-weight cascade has τ(q) = −log2 Σ w^q − 2q (the measure's τ less d·q):
    at γ = 0 leaders and p-leaders are inadmissible (h_min < 0) and far off; at γ = 1 both read
    τ within 0.1 over q ∈ [−2, 4] in the original frame and c2 within 0.02 of the closed form."""
    w = np.array([0.1, 0.2, 0.3, 0.4])
    mu = _cascade(10, w)
    theory = -np.log2((w[None, :] ** Q[:, None]).sum(1)) - 2 * Q
    c2_theory = np.log(2.0) * ((np.log2(w) ** 2).mean() - np.log2(w).mean() ** 2)
    r0 = L.run_leaders(mu, wavelet="haar", scale_min=4, scale_max=32, wse_rule="fixed")
    adm = r0["admissibility"]
    assert not adm["leaders"]["admissible"] and not adm["p-leaders"]["admissible"]
    assert adm["h_min"]["original"] < 0 and adm["leaders"]["suggested_gamma"] >= 1.0
    bad = fit_spectra(_tables(r0)["leaders"], 2, 5)["tau"]
    assert np.abs(bad - theory).max() > 0.5
    r1 = L.run_leaders(mu, wavelet="haar", scale_min=4, scale_max=32, gamma=1.0, wse_rule="fixed")
    assert r1["admissibility"]["leaders"]["admissible"] and r1["admissibility"]["p-leaders"]["admissible"]
    t1 = _tables(r1)
    for k in L.METHODS:
        fit = fit_spectra(t1[k], 2, 5, eta=1.0, frame="original")
        assert np.abs(fit["tau"] - theory).max() < 0.1, (k, fit["tau"] - theory)
        assert abs(_c(r1, k, gamma=1.0)["c2"] - c2_theory) < 0.02, k
        assert fit_spectra(t1[k], 2, 5, eta=1.0, frame="integrated")["tau"][_qi(t1[k], 1.0)] \
            == pytest.approx(fit["tau"][_qi(t1[k], 1.0)] + 1.0)


def test_the_derivative_of_fbm_is_inadmissible_until_integrated_once():
    f = fbm2d(512, 0.5, seed=2)
    dxf = np.roll(f, -1, axis=1) - f
    r0 = L.run_leaders(dxf, scale_min=4, scale_max=32, wse_rule="fixed")
    a0 = r0["admissibility"]
    assert not a0["leaders"]["admissible"] and not a0["p-leaders"]["admissible"] and a0["wse"]["admissible"]
    assert a0["leaders"]["suggested_gamma"] == 1.0 and a0["p-leaders"]["suggested_gamma"] == 1.0
    assert abs(_c(r0, "wse")["c1"] + 0.5) < 0.1
    assert _c(r0, "leaders")["c1"] > 0                                       # the wrong answer
    r1 = L.run_leaders(dxf, scale_min=4, scale_max=32, gamma=1.0, wse_rule="fixed")
    a1 = r1["admissibility"]
    assert a1["leaders"]["admissible"] and a1["p-leaders"]["admissible"]
    assert a1["h_min"]["integrated"] == pytest.approx(a0["h_min"]["original"] + 1.0)
    assert a1["eta"]["integrated"] == pytest.approx(a0["eta"]["original"] + 2.0)
    for k in L.METHODS:
        assert abs(_c(r1, k, gamma=1.0)["c1"] + 0.5) < 0.1, k


# ------------------------------------------------------------------- cusps and chirps, 1-D

def _signal(kind, h=0.5, beta=1.0):
    n = 1 << 18
    u = np.abs((np.arange(n) + 0.5) / n - 0.5 + 0.37 / n)              # x0 off the grid
    return u ** h if kind == "cusp" else u ** h * np.sin(1.0 / u ** beta)


def _cell(j, a):
    """The value at the level-``j`` cell holding x0 or its neighbour (the larger)."""
    k = (1 << 18) // 2 // 2 ** j
    return np.nanmax(a[k - 1:k + 1])


def _at_point(r, per_level):
    return [_cell(j, per_level[j - 1]) for j in range(r["j_min"], r["levels"] + 1)]


def _local(r, method):
    return _cell(r["j_min"], r["local"][method][0])


def _run(f, gamma=0.0):
    return L.run_leaders(f, wavelet="sym4", scale_min=256, scale_max=8192, gamma=gamma,
                         wse_rule="fixed")


@pytest.mark.parametrize("h", [0.3, 0.5])
def test_a_cusp_reads_its_exponent_on_every_method_with_unit_gamma_slope_and_no_oscillation(h):
    r0, r1 = _run(_signal("cusp", h)), _run(_signal("cusp", h), gamma=1.0)
    for k in L.METHODS:
        assert abs(_local(r0, k) - h) < 0.01, k
        assert abs(_local(r1, k) - 1.0 - h) < 0.01, k
    octaves = np.diff(np.log2(_at_point(r0, r0["quantity"]["leaders"])))
    assert np.allclose(octaves, h, atol=0.02)
    assert np.allclose(np.diff(np.log2(_at_point(r1, r1["quantity"]["leaders"]))) - octaves, 1.0, atol=0.02)
    assert np.allclose(_at_point(r0, r0["oscillation"]), 0.0, atol=1e-9)


def test_a_chirp_reads_its_holder_exponent_on_leaders_and_a_larger_weak_exponent():
    """|x|^½ sin(1/|x|): leaders and p-leaders read α = ½ at x₀; the (θ, ω)-leaders read the
    weak exponent, well above; the γ-slope of the leaders is 1 + β = 2 on the undisturbed
    octaves (the oscillation diagnostic); the oscillation ratio is large across the levels."""
    f = _signal("chirp", 0.5, 1.0)
    r0, r1 = _run(f), _run(f, gamma=1.0)
    assert abs(_local(r0, "leaders") - 0.5) < 0.15
    assert abs(_local(r0, "p-leaders") - 0.5) < 0.05
    assert _local(r0, "wse") > 0.9
    gamma_slope = (np.diff(np.log2(_at_point(r1, r1["quantity"]["leaders"])))
                   - np.diff(np.log2(_at_point(r0, r0["quantity"]["leaders"]))))
    assert np.sum(np.abs(gamma_slope - 2.0) < 0.2) >= 2
    assert np.sum(np.asarray(_at_point(r0, r0["oscillation"])) > 3.0) >= 4
    cusp = _run(_signal("cusp", 0.5))
    assert max(_at_point(cusp, cusp["oscillation"])) < 1e-9


def test_the_two_dimensional_cusp_has_no_oscillation_and_reads_near_its_exponent():
    """A radial cusp r^½ sampled with 4 × 4 pixel averaging at 1024²: the oscillation ratio is 0
    at the cusp and its leaders' local exponent lies within 0.25 of ½ over 4–64 px (the pixel
    scale and the mirror edge both lie within the ten octaves of such a field)."""
    n, s = 1024, 4
    yy, xx = np.indices((n * s, n * s))
    rr = np.hypot((yy + 0.5) / s - n / 2, (xx + 0.5) / s - n / 2) / n
    cusp = (rr ** 0.5).reshape(n, s, n, s).mean(axis=(1, 3))
    r = L.run_leaders(cusp, scale_min=4, scale_max=64, wse_rule="fixed")
    for j in range(r["j_min"], r["levels"] + 1):
        k = (n // 2) // 2 ** j
        assert np.nanmax(np.abs(r["oscillation"][j - 1][k - 1:k + 1, k - 1:k + 1])) < 1e-9
    k = (n // 2) // 2 ** r["j_min"]
    assert abs(np.nanmax(r["local"]["leaders"][0][k - 1:k + 1, k - 1:k + 1]) - 0.5) < 0.25


# ---------------------------------------------------------------------------- refusals

def test_run_leaders_refuses_bad_scales_and_small_fields():
    f = fbm2d(64, 0.5, seed=0)
    with pytest.raises(ValueError, match="power of two"):
        L.run_leaders(f, scale_min=3, scale_max=32)
    with pytest.raises(ValueError, match="below scale_max"):
        L.run_leaders(f, scale_min=32, scale_max=32)
    with pytest.raises(ValueError, match="smaller side"):
        L.run_leaders(f, scale_min=4, scale_max=128)
    with pytest.raises(ValueError, match="1-D signal or a 2-D band"):
        L.run_leaders(np.zeros((4, 4, 4)), scale_min=2, scale_max=4)


# ------------------------------------------------------------------------------ edge modes

def test_periodic_continuation_is_translation_invariant_and_keeps_every_cell():
    """A periodic field (the spectral fBm wraps) run periodic: no border, the leaders of the
    field rolled by a whole coarsest cell are the rolled leaders; the mirror loses its border."""
    f = fbm2d(256, 0.5, seed=7)
    a = L.coefficients(f, "sym4", 4, edge="periodic")
    assert a["work_shape"] == (256, 256) and a["edge"] == "periodic"
    assert all(v.all() for v in L.valid_cells(a, 0))
    b = L.coefficients(np.roll(f, (32, 64), axis=(0, 1)), "sym4", 4, edge="periodic")
    for j in range(1, 5):
        assert np.allclose(np.roll(L.leaders(a)[j - 1], (32 // 2 ** j, 64 // 2 ** j), axis=(0, 1)),
                           L.leaders(b)[j - 1], atol=1e-10)
    m = L.coefficients(f, "sym4", 4)
    assert m["edge"] == "mirror" and m["work_shape"] == (512, 512)
    assert not L.valid_cells(m, dwt.border_cells("sym4"))[0][0].any()
    for edge in ("reflect", "zero_pad", "edge_pad"):
        c = L.coefficients(f, "sym4", 4, edge=edge)
        assert c["work_shape"] == (512, 512) and np.isfinite(c["mods"][0]).all()
    with pytest.raises(ValueError, match="edge"):
        L.coefficients(f, "sym4", 4, edge="wrap")


def test_the_real_block_bounds_the_valid_cells_and_a_wrap_of_another_period_is_a_seam():
    """A field whose data occupy a block inside it (a centred Periodic extension's output): only
    the block's cells count and the structure functions count the block's grid. When the field's
    size is its own period and a multiple of 2^J the working image is the field (no seam, no
    border); wrapped to another size, the wrap is a seam the border rule treats as an edge."""
    f = fbm2d(256, 0.5, seed=8)
    big = np.pad(f, ((32, 32), (32, 32)), mode="wrap")                   # 320 = 10 · 32, period 256
    r = L.run_leaders(big, scale_min=4, scale_max=32, wse_rule="fixed", edge="periodic",
                      origin=(32, 32), block=(256, 256))
    assert r["data_shape"] == (256, 256) and r["origin"] == (32, 32) and r["seam"] is False
    for j in range(1, 6):
        v = r["valid"]["leaders"][j - 1]
        lo, hi = 32 // 2 ** j, (32 + 256) // 2 ** j
        assert v[lo:hi, lo:hi].all() and not v[:lo].any() and not v[hi:].any()
    assert r["samples"]["cells"] == [(256 // 2 ** j) ** 2 for j in range(1, 6)]
    t = _tables(r)["leaders"]
    assert np.allclose(t["tau_qa"][_qi(t, 0.0)], 2 * np.log2(256) - 2 * np.arange(1, 6))
    odd = L.run_leaders(np.pad(f, ((0, 8), (0, 8)), mode="wrap"), scale_min=4, scale_max=32,
                        wse_rule="fixed", edge="periodic", block=(256, 256))
    assert odd["seam"] is True                                            # 264 is no period of the working size: a seam
    e = dwt.border_cells("sym4")
    for j in range(1, 6):
        v = odd["valid"]["leaders"][j - 1]
        n = v.shape[0]                                                    # the working size, the grid centred on the 264 px
        assert not v[:e].any() and not v[n - e:].any()                    # the seam's border
    plain = L.run_leaders(f, scale_min=4, scale_max=32, wse_rule="fixed", edge="periodic")
    assert plain["seam"] is False and all(v.all() for v in plain["valid"]["leaders"])


# ------------------------------------------------------- what the fitting tools read off the sets

def test_the_structure_fit_reads_fbm_with_either_weighting_and_shifts_frames():
    """1024², sym4, levels 2–6: ζ(2)/2 and h(1) within 0.03 of H = 0.5 with uniform weights and
    with the valid counts (Wendt 2007's choice); τ = ζ − 2; a lift γ = 1 fitted in the
    integrated frame and taken out in the original one gives the same original exponents."""
    r = L.run_leaders(fbm2d(1024, 0.5, seed=7), wavelet="sym4", scale_min=4, scale_max=64, wse_rule="fixed")
    t = _tables(r)["leaders"]
    for w in (None, t["N_a"]):
        fit = L.structure_fit(t, 2, 6, weights=w)
        assert fit["weights"] == ("uniform" if w is None else "counts") and list(fit["levels"]) == [2, 3, 4, 5, 6]
        assert abs(fit["zeta"][_qi(t, 2.0)] / 2 - 0.5) < 0.03 and abs(fit["h"][_qi(t, 1.0)] - 0.5) < 0.03
        assert np.allclose(fit["tau"], fit["zeta"] - 2) and np.all(fit["tau_err"] >= 0)
        assert abs(fit["D"][_qi(t, 0.0)] - 2.0) < 0.05 and fit["d"] == 2
    lifted = L.run_leaders(fbm2d(1024, 0.5, seed=7), wavelet="sym4", scale_min=4, scale_max=64, gamma=1.0,
                           wse_rule="fixed")
    tl = _tables(lifted)["leaders"]
    integrated = L.structure_fit(tl, 2, 6, gamma=1.0, frame="integrated")
    original = L.structure_fit(tl, 2, 6, gamma=1.0, frame="original")
    assert np.allclose(original["tau"], integrated["tau"] - Q) and np.allclose(original["h"], integrated["h"] - 1)
    assert abs(original["zeta"][_qi(tl, 2.0)] / 2 - 0.5) < 0.03
    narrow = L.structure_fit(t, 2, 3)
    assert np.all(np.isnan(narrow["tau"]))                                 # two levels fit nothing


def test_each_log_cumulant_regresses_on_its_own_range_with_the_cascade_sign():
    from dynamix.core.spectra import cumulant_regression, ln_moments

    r = L.run_leaders(fbm2d(512, 0.5, seed=6), scale_min=4, scale_max=32, wse_rule="fixed")
    C = ln_moments([m[v] for m, v in zip(r["quantity"]["leaders"], r["valid"]["leaders"])])
    assert C.shape == (4, 5)
    c1 = cumulant_regression(C, r["scales"], 2, 5, 1)
    assert abs(c1["c"] - 0.5) < 0.05 and c1["n"] == 4 and 0 < c1["sigma"] < 0.05 and c1["r2"] > 0.99
    c2 = cumulant_regression(C, r["scales"], 2, 5, 2)
    assert abs(c2["c"]) < 0.02 and c2["c"] == -c2["slope"]                  # the cascade sign
    exact = np.array([[0.3 * j for j in range(1, 6)], [0.1 * j for j in range(1, 6)], [0.0] * 5, [0.0] * 5]) * np.log(2.0)
    fit = cumulant_regression(exact, 2.0 ** np.arange(1, 6), 1, 5, 2)
    assert fit["c"] == pytest.approx(-0.1) and fit["sigma"] == pytest.approx(0.0, abs=1e-9) and fit["r2"] == pytest.approx(1.0)
    assert cumulant_regression(exact, 2.0 ** np.arange(1, 6), 1, 1, 1)["n"] == 1
    assert np.isnan(cumulant_regression(exact, 2.0 ** np.arange(1, 6), 1, 1, 1)["c"])
    two = cumulant_regression(exact, 2.0 ** np.arange(1, 6), 1, 2, 1)
    assert two["n"] == 2 and two["c"] == pytest.approx(0.3) and np.isnan(two["sigma"])


def test_the_structure_fit_checks_jaffards_sobolev_condition_on_zeta():
    """Jaffard 1997 Part I (3.6): for q ≥ p > 0 any wavelet scaling function satisfies
    (ζ(q) − ζ(p)) / (q − p) ≥ (ζ(p) − d) / p (the Sobolev embeddings between the L^{p,s}); a fitted
    ζ that violates it is numerically wrong whatever produced it. The fit reports the smallest
    margin over the consecutive positive orders and whether the condition holds."""
    q = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
    H = 0.4
    fbm = L.sobolev_condition(q, q * H, d=2)              # margins 2/p, the smallest at p = 2
    assert fbm["holds"] and fbm["margin"] == pytest.approx(1.0) and fbm["at_q"] == 2.0
    # ζ = qH for q ≤ 1 then flat: the chord slope 0 against (ζ(1) − 2)/1 = H − 2 < 0 still holds …
    assert L.sobolev_condition(q, np.where(q <= 1, q * H, H), d=2)["holds"]
    # … a ζ that drops faster than the Sobolev line allows fails: ζ(2) = ζ(1) − 3 gives a chord of −3
    # against (ζ(1) − 2)/1 = −1.6
    bad = np.where(q <= 1, q * H, q * H - 3.0 * (q - 1))
    res = L.sobolev_condition(q, bad, d=2)
    assert not res["holds"] and res["at_q"] == 1.0 and res["margin"] < 0
    # a structure fit carries it
    r = L.run_leaders(fbm2d(256, 0.5, seed=1), scale_min=4, scale_max=32, wse_rule="fixed")
    fit = L.structure_fit(_tables(r)["leaders"], 2, 5)
    assert fit["sobolev"]["holds"]
