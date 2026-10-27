# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Spec 38: the directional formalism of Ben Slimane et al. (2019) read by Abry et al.'s level
function m = max(j₁/α₁, j₂/α₂) (2015, Prop. 4.7 (3)), on fields with known directional exponents."""
from __future__ import annotations

import types

import numpy as np
import pytest

from dynamix.core import directional as dr
from dynamix.core import fft_policy


def _sheet(n, h1, h2, seed=0):
    """A periodic Gaussian sheet with the product spectrum |k₁|^−(2H₁+1) |k₂|^−(2H₂+1): directional
    exponents H₁ along x, H₂ along y (a fractional Brownian sheet's increments)."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n)
    with np.errstate(divide="ignore"):
        a1 = np.where(k == 0, 0.0, np.abs(k) ** (-(2 * h1 + 1) / 2))
        a2 = np.where(k == 0, 0.0, np.abs(k) ** (-(2 * h2 + 1) / 2))
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    f = np.real(fft_policy.double().ifft2(a2[:, None] * a1[None, :] * z))
    return f / f.std()


def _line(n, h, seed=0):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n)
    with np.errstate(divide="ignore"):
        a = np.where(k == 0, 0.0, np.abs(k) ** (-(2 * h + 1) / 2))
    f = np.real(fft_policy.double().ifft(a * (rng.standard_normal(n) + 1j * rng.standard_normal(n))))
    return f / f.std()


def _run(values, **kw):
    from dynamix.devices.directional_scaling import DirectionalScaling

    v = np.asarray(values, dtype=float)
    field = types.SimpleNamespace(values=v, provenance={}, frame=None,
                                  x_axis=np.arange(v.shape[1], dtype=float), y_axis=np.arange(v.shape[0], dtype=float))
    params = {q.name: q.default for q in DirectionalScaling.params}
    params.update(kw)
    return DirectionalScaling().compute(field, params)["_directional"]


def _at2(d, key):
    return float(np.asarray(d[key])[int(np.flatnonzero(np.isclose(d["p"], 2.0))[0])])


@pytest.fixture(scope="module")
def sheet():
    return _sheet(1024, 0.3, 0.7, seed=1)


def test_the_defaults_read_both_directions_and_fit_every_anisotropy(sheet):
    d = _run(sheet)
    assert d["estimator"] == "level_line"
    assert int(np.isfinite(d["eta_alpha"]).any(axis=0).sum()) == d["alpha"].size
    assert _at2(d, "eta_e1") / 2 == pytest.approx(0.3, abs=0.05)
    assert _at2(d, "eta_e2") / 2 == pytest.approx(0.7, abs=0.05)
    lo, hi = d["alpha_star_e1_interval"][:, int(np.flatnonzero(np.isclose(d["p"], 2.0))[0])]
    assert lo <= 1.4 <= hi + dr.ALPHA_STEP           # the plateau ends at α₁ = 2H₂/(H₁ + H₂), on the grid


def test_the_transposed_field_swaps_the_two_directions(sheet):
    d, t = _run(sheet), _run(sheet.T)
    np.testing.assert_allclose(d["eta_e1"], t["eta_e2"], atol=1e-9)
    np.testing.assert_allclose(d["eta_e2"], t["eta_e1"], atol=1e-9)


def test_an_isotropic_field_reads_the_same_exponent_both_ways():
    d = _run(_sheet(1024, 0.5, 0.5, seed=2))
    assert _at2(d, "eta_e1") / 2 == pytest.approx(0.5, abs=0.05)
    assert _at2(d, "eta_e2") / 2 == pytest.approx(0.5, abs=0.05)
    lo, hi = d["alpha_star_e1_interval"][:, int(np.flatnonzero(np.isclose(d["p"], 2.0))[0])]
    assert lo <= 1.0 <= hi


def test_a_layered_field_reads_its_one_direction_and_no_structure_along_the_other():
    u = np.tile(_line(1024, 0.3, seed=5), (1024, 1))              # u(x): no detail along y
    d = _run(u, edge="periodic")
    assert d["no_structure"] == {"x": False, "y": True}
    assert _at2(d, "eta_e1") / 2 == pytest.approx(0.3, abs=0.05)
    assert not np.isfinite(d["eta_e2"]).any()                     # never 0.000
    bare = _run(u, edge="periodic", coarse_partner="never")
    assert bare["no_structure"]["y"] and not np.isfinite(bare["eta_e2"]).any()


def test_coefficient_integration_shifts_each_direction_by_half_gamma(sheet):
    d, g = _run(sheet), _run(sheet, gamma=1.0)
    for key in ("eta_e1", "eta_e2", "eta_iso"):
        assert (_at2(g, key) - _at2(d, key)) / 2 == pytest.approx(0.5, abs=0.05), key


def test_the_fitted_cells_lie_in_the_knob_range(sheet):
    d = _run(sheet, scale_min=8, scale_max=64)
    for fit in d["diagnostic_fits"]:
        for arm in fit["arms"].values():
            assert all(8 <= px <= 64 for px in arm["fine_px"])


def test_no_directional_exponent_where_the_isotropic_scaling_function_is_not_positive():
    scaling = {"alpha": dr.alpha_grid(), "eta_alpha": np.tile([[0.5], [-0.2]], (1, 15)),
               "eta_alpha_se": np.zeros((2, 15))}
    out = dr.directional_scaling(scaling)
    assert np.isfinite(out["eta_e1"][0]) and not np.isfinite(out["eta_e1"][1])


def test_the_legendre_bound_is_empty_below_zero_and_marks_its_p_max_stretch(sheet):
    """The knee approaches H_e from below as p max grows: H_e − d(1 − α_min)/(p_max α_e)."""
    low, high = _run(sheet, p_max=8.0), _run(sheet, p_max=16.0)
    for key, h in (("L_e1", 0.3), ("L_e2", 0.7)):
        L = np.asarray(high[key]["L"])
        assert np.nanmin(L) >= 0.0
        assert np.asarray(high[key]["limited"]).any()
        assert low[key]["knee"] <= high[key]["knee"] <= h + 0.05


def test_the_legacy_estimator_reads_no_exact_zero(sheet):
    d = _run(sheet, estimator="legacy", scale_max=256)
    eta = np.asarray(d["eta_alpha"])
    assert not np.any(np.isfinite(eta) & (np.abs(eta) < 1e-6))


def test_the_bs19_literal_estimator_keeps_its_numbers(sheet):
    d = _run(sheet, estimator="gamma")
    assert _at2(d, "eta_e1") / 2 == pytest.approx(0.342, abs=0.002)
    assert _at2(d, "eta_e2") / 2 == pytest.approx(0.342, abs=0.002)


# -- Phase B: hyperbolic wavelet leaders (Abry et al. 2015, (3.3)) and the local maps ------------

def test_the_leaders_recursion_is_the_sup_over_sub_rectangles_and_the_neighbourhood():
    rng = np.random.default_rng(0)
    J = 3
    shape = {l: (64 // 2 ** l, 64 // 2 ** l) for l in range(1, J + 1)}
    mods = {(l1, l2): rng.random((shape[l2][0], shape[l1][1])) for l1 in range(1, J + 1) for l2 in range(1, J + 1)}
    coef = {"levels": J, "mods": mods, "valid": {k: np.ones_like(v, bool) for k, v in mods.items()}}
    out = dr.hyperbolic_leaders(coef)
    for (l1, l2), d in out["leaders"].items():
        rows, cols = d.shape
        for (r, c) in ((0, 0), (rows // 2, cols // 3), (rows - 1, cols - 1)):
            best = 0.0
            for dr_ in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = (r + dr_) % rows, (c + dc) % cols
                    for m1 in range(1, l1 + 1):
                        for m2 in range(1, l2 + 1):
                            fx, fy = 2 ** (l1 - m1), 2 ** (l2 - m2)
                            block = mods[(m1, m2)][rr * fy:(rr + 1) * fy, cc * fx:(cc + 1) * fx]
                            best = max(best, float(block.max()))
            assert d[r, c] == pytest.approx(best), (l1, l2, r, c)


def test_the_strip_maps_read_each_half_of_a_two_anisotropy_field():
    n = 1024
    left, right = _sheet(n, 0.3, 0.7, seed=11), _sheet(n, 0.7, 0.3, seed=12)
    f = np.concatenate([left[:, : n // 2], right[:, n // 2:]], axis=1)
    coef = dr.coefficients(f, "sym4", 6, edge="periodic", complete=True)
    maps = dr.strip_maps(dr.hyperbolic_leaders(coef), j_min=2, shape=f.shape)
    for cols, (hx, hy) in ((slice(192, 320), (0.3, 0.7)), (slice(704, 832), (0.7, 0.3))):
        assert np.nanmedian(maps["h_x"][:, cols]) == pytest.approx(hx, abs=0.1)
        assert np.nanmedian(maps["h_y"][:, cols]) == pytest.approx(hy, abs=0.1)
        assert np.nanmedian(maps["anisotropy"][:, cols]) == pytest.approx(2 * hy / (hx + hy), abs=0.25)


def test_the_leader_bound_is_a_level_set_spectrum_bound_over_both_signs_of_p(sheet):
    d = _run(sheet)
    lead = d["leaders"]
    assert (lead["p"] < 0).any() and (lead["p"] > 0).any()
    L = np.asarray(lead["L_alpha"])
    assert L.shape == (d["alpha"].size, lead["h"].size)
    diag = int(np.argmin(np.abs(d["alpha"] - 1.0)))
    top = lead["h"][int(np.nanargmax(L[diag]))]
    assert np.nanmax(L) <= 2.0 + 1e-9 and np.nanmin(L) >= 0.0
    assert 0.15 <= top <= 0.6                       # its peak near the field's smaller exponent


def test_the_device_offers_the_maps_as_outputs(sheet):
    from dynamix.devices.directional_scaling import DirectionalScaling

    dev = DirectionalScaling()
    assert {o.name for o in dev.outputs} == {"h_x", "h_y", "anisotropy"}
    field = types.SimpleNamespace(values=sheet, provenance={}, frame=None,
                                  x_axis=np.arange(1024.0), y_axis=np.arange(1024.0))
    params = {q.name: q.default for q in dev.params}
    res = dev.compute(field, dict(params, edge="periodic"))
    out = dev.compute_output("h_x", sheet, res, dict(params, edge="periodic"), fetch=None)
    assert out["raster"].shape == sheet.shape and np.isfinite(out["raster"]).mean() > 0.9
    assert np.nanmedian(out["raster"]) == pytest.approx(0.3, abs=0.1)
