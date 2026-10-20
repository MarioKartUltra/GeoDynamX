# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Directional scaling functions tool: the directional formalism as a tool of its own on a
field, its curves, its reading, its edge modes and its refusals."""
from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("pywt")

from dynamix.core import dwt                                                    # noqa: E402
from dynamix.model.contracts import CURVE, FIELD                               # noqa: E402
from dynamix.model.device import get_device, keyed_params, validate_params     # noqa: E402

sys.path.insert(0, "tests")
from conftest import fbm2d, make_field                                         # noqa: E402


@pytest.fixture
def dev(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return get_device("directional_scaling")


def test_the_tool_sits_in_regularity_and_offers_curves(dev):
    assert dev.label == "Directional scaling functions" and dev.families == ("Regularity",)
    assert dev.consumes == ((FIELD,),) and dev.produces == {CURVE}
    p = validate_params(dev, {})
    assert (p["wavelet"], p["edge"], p["scale_min"], p["scale_max"], p["gamma"]) == ("sym4", "auto", 4, 64, 0.0)
    assert (p["p_min"], p["p_max"], p["p_step"]) == (0.5, 6.0, 0.5)
    assert list(dev.p_list(p)) == list(np.arange(0.5, 6.01, 0.5))
    assert {"wavelet", "edge", "scale_min", "scale_max", "gamma", "p_min"} <= set(keyed_params(dev, p))
    assert dev.roi_margin(p) == dwt.border_cells("sym4") * 64
    with pytest.raises(ValueError, match="power of two"):
        dev.check(validate_params(dev, {"scale_min": 6}))
    with pytest.raises(ValueError, match="at least three levels"):
        dev.check(validate_params(dev, {"scale_min": 32, "scale_max": 64}))
    with pytest.raises(ValueError, match="p max"):
        dev.check(validate_params(dev, {"p_min": 3.0, "p_max": 1.0}))
    with pytest.raises(ValueError, match="below scale_max"):
        dev.compute(make_field(fbm2d(32, 0.5, seed=1)), p)


def test_fbm_reads_its_hurst_exponent_along_both_axes_and_the_curves_follow(dev):
    from dynamix.model.curves import curves_of

    field = make_field(fbm2d(1024, 0.5, seed=6))
    p = validate_params(dev, {})
    res = dev.compute(field, p)
    d = res["_directional"]
    assert d["eta_alpha"].shape == (12, 15) and list(d["p"]) == list(np.arange(0.5, 6.01, 0.5))
    at_p1 = list(d["p"]).index(1.0)
    assert abs(d["eta_e1"][at_p1] - 0.5) < 0.15 and abs(d["eta_e2"][at_p1] - 0.5) < 0.15
    assert res["fitted_anisotropies"] > 0 and res["warnings"] == []
    rec = res["record"]
    assert rec["edge"] == {"asked": "auto", "used": "mirror"} and rec["levels"] == {"j_min": 2, "J": 6}
    assert rec["alpha_step"] == 1 / 8 and len(rec["levels_per_alpha"]) == 15 and len(rec["eta_e1"]) == 12
    names = [c.name for c in curves_of(dev, res, field, p)]
    assert names == ["η(p, e₁), along x", "η(p, e₂), along y", "η(p), the diagonal α = (1, 1)",
                     "α₁*(p), the anisotropy profile of e₁",
                     "L(H, e₁), the directional Legendre bound", "L(H, e₂), the directional Legendre bound"]
    text = dev.reading(res, p)
    assert text.startswith("η(2, e₁)/2 0.") and "α₁*(2)" in text and "anisotropies keep three unclipped levels" in text
    assert dev.reading({"values": 1}, p) == "no scaling functions — run the transform"
    assert curves_of(dev, {"values": 1}, field, p) == []


def test_the_edge_modes_reach_the_hyperbolic_coefficients(dev):
    from dynamix.core import directional as dr
    from dynamix.devices.periodic_extension import PeriodicExtension
    from dynamix.model.device import defaults_for

    f = fbm2d(256, 0.5, seed=2)
    mirrored = dr.coefficients(f, "sym4", 4)
    periodic = dr.coefficients(f, "sym4", 4, edge="periodic")
    assert mirrored["border_cells"] == dwt.border_cells("sym4") - 1 and periodic["border_cells"] == 0
    assert periodic["edge"] == "periodic" and mirrored["edge"] == "mirror"
    assert all(periodic["valid"][k].sum() > mirrored["valid"][k].sum() for k in mirrored["valid"])
    with pytest.raises(ValueError, match="edge"):
        dr.coefficients(f, "sym4", 4, edge="wrap")
    field = make_field(fbm2d(512, 0.5, seed=3))
    ext = PeriodicExtension().compute(field, defaults_for(PeriodicExtension()))
    assert dev.derived_reading("edge", "auto", ext) == "→ periodic (Periodic extension output)"
    assert dev.derived_reading("edge", "auto", field) == "→ mirror" and dev.derived_reading("edge", "mirror", ext) is None
    p = validate_params(dev, {"scale_max": 32})
    res = dev.compute(ext, p)
    assert res["record"]["edge"] == {"asked": "auto", "used": "periodic"}


def test_the_scale_knob_warns_when_a_level_keeps_too_few_cells(dev):
    field = make_field(fbm2d(128, 0.5, seed=1))
    p = validate_params(dev, {"scale_min": 4, "scale_max": 32})
    assert "scale_max" in dev.knob_warnings(field, p)
    roomy = make_field(fbm2d(256, 0.5, seed=1))
    assert dev.knob_warnings(roomy, validate_params(dev, {"scale_min": 4, "scale_max": 16})) == {}
    assert dev.knob_warnings(None, p) == {}
