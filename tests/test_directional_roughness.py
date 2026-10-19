# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Khalil et al. 2006's directional roughness on the WTMMM: the two components' Hurst exponents by
slope and by collapse, the anisotropy against scale, the modulus pdfs and their q-rescaled
collapse, the cut spectra — on fractional Brownian sheets and an isotropic field."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from dynamix.core import anisotropy as an
from dynamix.core.orientation import from_arg

sys.path.insert(0, "tests")
from conftest import FIXED_EDGES, fbm2d, make_field                               # noqa: E402
from test_directional import fbm_sheet                                            # noqa: E402

FIELDS = {"sheet_10_50": (lambda: fbm_sheet(1024, 0.10, 0.50, seed=1), (0.10, 0.50)),
          "sheet_41_44": (lambda: fbm_sheet(1024, 0.41, 0.44, seed=2), (0.41, 0.44)),
          "fbm_50": (lambda: fbm2d(1024, 0.5, seed=3), (0.50, 0.50))}


@pytest.fixture(scope="module")
def wtmm():
    """The WTMM (three octaves, the suite's fixed edges) on the three fields, computed once;
    the registry is restored before the tests see it."""
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES, get_device, validate_params

    saved = dict(DEVICES)
    register_builtin_devices()
    try:
        dev = get_device("wtmm2d")
        p = validate_params(dev, {"n_oct": 3, "n_voice": 4, **FIXED_EDGES})
        out = {}
        for name, (make, truth) in FIELDS.items():
            field = make()
            res = dev.compute(make_field(field), p)
            w = an.wtmmm_by_scale(res["extrema"])
            out[name] = {"field": field, "truth": truth, "scales": np.asarray(res["scales"], float),
                         "wtmmm": w, "components": [an.gradient_plane(s["mod"], s["arg"]) for s in w],
                         "mods": [np.asarray(s["mod"], float) for s in w],
                         "orientations": [from_arg(s["arg"]) for s in w]}
    finally:
        DEVICES.clear()
        DEVICES.update(saved)
    return out


def _window(scales):
    return float(np.log2(scales[1])), float(np.log2(scales[-2]))


def test_the_directional_hurst_exponents_of_the_sheets_and_the_isotropic_field(wtmm):
    """The paper's eqs. 33–36 on one 1024² image: the order and the gap right, the small
    exponent reading low (the WTMM's finite-size underestimate), |H_y − H_x| = 0.03 resolved,
    the collapse within 0.03 of the slopes, the isotropic field reading equal exponents."""
    r = wtmm["sheet_10_50"]
    h = an.directional_hurst(r["components"], r["scales"], *_window(r["scales"]))
    (hx, sx), (hy, sy) = h["median"]
    assert abs(hy - 0.50) < 0.05 and abs(hx - 0.10) < 0.08 and hx < hy
    assert 0 < sx < 0.05 and 0 < sy < 0.05
    assert abs(h["collapse"][0][0] - hx) < 0.03 and abs(h["collapse"][1][0] - hy) < 0.03
    assert abs(h["std"][1][0] - 0.50) < 0.05
    assert h["dominant"] == 0                                 # T_ψ1 carries the sheet's spread (x rough)
    assert h["scales"].size == len(r["scales"]) - 2
    r = wtmm["sheet_41_44"]
    h = an.directional_hurst(r["components"], r["scales"], *_window(r["scales"]))
    (hx, _), (hy, _) = h["median"]
    assert abs(hx - 0.41) < 0.05 and abs(hy - 0.44) < 0.05 and 0.02 <= hy - hx <= 0.09
    r = wtmm["fbm_50"]
    h = an.directional_hurst(r["components"], r["scales"], *_window(r["scales"]))
    (hx, _), (hy, _) = h["median"]
    assert abs(hx - hy) < 0.03 and abs(hx - 0.5) < 0.08 and abs(hy - 0.5) < 0.08
    assert abs(h["collapse"][0][0] - h["collapse"][1][0]) <= 0.03
    none = an.directional_hurst(r["components"][:2], r["scales"][:2], 0.0, 10.0)
    assert all(np.isnan(v) for pair in none["median"] for v in pair)


def test_component_pdfs_share_their_edges_and_collapse_at_the_true_exponents(wtmm):
    r = wtmm["sheet_10_50"]
    lo, hi = _window(r["scales"])
    keep = [k for k in range(r["scales"].size) if lo <= np.log2(r["scales"][k]) <= hi]
    comps = [r["components"][k] for k in keep]
    raw = an.component_pdfs(comps, r["scales"][keep], bins=40)
    assert raw["centres"].shape == (2, 40) and raw["ln_pdf"].shape == (2, len(keep), 40)
    assert np.all(raw["counts"] == np.array([[c[0].size for c in comps]] * 2))
    col = an.component_pdfs(comps, r["scales"][keep], bins=40, rescale=(0.10, 0.50))

    def spread(table):
        """The mean across the bins of the standard deviation of ln P across scales."""
        ln = table["ln_pdf"]                                   # (n_scales, bins)
        ok = np.all(np.isfinite(ln), axis=0)
        return float(np.mean(np.std(ln[:, ok], axis=0))) if ok.any() else np.inf

    assert spread({"ln_pdf": col["ln_pdf"][1]}) < 0.3 * spread({"ln_pdf": raw["ln_pdf"][1]})


def test_the_anisotropy_against_scale_sharpens_on_the_sheet_and_stays_flat_when_isotropic(wtmm):
    r = wtmm["sheet_10_50"]
    a = an.anisotropy_vs_scale(r["orientations"], r["scales"])
    assert a["relative_height"][0] > 3.0 and a["relative_height"][-1] < 2.5
    assert a["relative_height"][0] > a["relative_height"][-1] + 1.0
    assert np.all(np.diff(a["relative_height"]) < 0.5)                  # falling, up to noise
    fold = np.abs(a["angle_deg"])                                       # from the x axis, in (−90, 90]
    assert np.all(fold[:2] <= 10.0)                                     # the rough axis at fine scales
    assert fold[-3:].mean() > fold[:3].mean()                           # eq. 38: tan A ~ a^{H_y − H_x}
    assert np.all(a["angle_deg"] > -90.0) and np.all(a["angle_deg"] <= 90.0)
    assert a["counts"][0] > a["counts"][-1] > 0
    r = wtmm["fbm_50"]
    a = an.anisotropy_vs_scale(r["orientations"], r["scales"])
    assert np.all(np.abs(a["relative_height"][:2] - 1.0) < 0.3)


def test_modulus_pdfs_vanish_at_zero_and_integrate_to_one(wtmm):
    r = wtmm["fbm_50"]
    m = an.modulus_pdfs(r["mods"], r["scales"], bins=50)
    assert m["pdf"].shape == (r["scales"].size, 50)
    dx = m["centres"][1] - m["centres"][0]
    assert np.allclose(m["pdf"].sum(axis=1) * dx, 1.0)
    assert np.all(m["pdf"][:, 0] < m["pdf"].max(axis=1))               # P_a(M) → 0 at M = 0


def test_the_q_rescaled_modulus_collapse_marks_the_usable_q(wtmm):
    """On a monofractal the areas under a^{−qH} M^q P_a(M) are the rescaled moments: equal
    across scales for moderate q, scattered for the large |q| the statistics cannot hold."""
    r = wtmm["fbm_50"]
    lo, hi = _window(r["scales"])
    H, se = an.modulus_hurst(r["mods"], r["scales"], lo, hi)
    assert abs(H - 0.45) < 0.05 and 0 < se < 0.05                          # the WTMM's global H
    q = np.arange(-3.0, 6.01, 0.5)
    c = an.modulus_collapse(r["mods"], r["scales"], H, q, lo, hi, bins=60)
    assert c["curves"].shape == (q.size, c["scales"].size, 60) and c["x"][0] > 0
    assert c["usable"][q == 1.0][0] and c["usable"][q == 2.0][0] and not c["usable"][q == -3.0][0]
    assert set(q[c["usable"]]) >= {-1.0, 0.0, 1.0, 2.0, 3.0}                 # the paper's own range
    assert c["area_spread"][q == 0.0][0] < 1e-9                           # the pdf integrates to one
    assert np.nanmax(c["area_spread"]) > c["area_spread"][q == 1.0][0]  # the extremes scatter more
    tight = an.modulus_collapse(r["mods"], r["scales"], H, q, lo, hi, tolerance=1e-6)
    assert tight["usable"].sum() <= 1


def test_cut_spectra_read_the_directional_and_radial_exponents():
    c = an.cut_spectra(fbm_sheet(512, 0.1, 0.5, seed=1))
    assert abs(c["beta_x"] - 1.2) < 0.1 and abs(c["beta_y"] - 2.0) < 0.1 and abs(c["beta_r"] - 2.2) < 0.15
    assert 0 < c["beta_x_se"] < 0.1 and c["k_x"].size == 256 and c["log2_power"].shape == (512, 512)
    assert c["window_x"][0] > 0 and c["window_x"][1] < 0.5
    iso = an.cut_spectra(fbm2d(512, 0.5, seed=3))
    assert abs(iso["beta_x"] - 2.0) < 0.12 and abs(iso["beta_y"] - 2.0) < 0.12 and abs(iso["beta_r"] - 3.0) < 0.12
    holed = fbm_sheet(256, 0.1, 0.5, seed=4)
    holed[10:20, 10:20] = np.nan
    assert np.isfinite(an.cut_spectra(holed)["beta_x"])


def test_a_transposed_field_swaps_the_two_exponents():
    c = an.cut_spectra(fbm_sheet(512, 0.1, 0.5, seed=1).T)
    assert abs(c["beta_x"] - 2.0) < 0.1 and abs(c["beta_y"] - 1.2) < 0.1
