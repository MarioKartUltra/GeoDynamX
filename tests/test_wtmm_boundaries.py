# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""WTMM's boundary options and Border cut (spec 25 §3.2, §3.3): the Edge mode, the padding width (the
measured reach by default), the Border cut before the V-chains (measured reach by default), and
``auto``, which on a Periodic extension output means periodic, no padding and no cut."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model.device import defaults_for, get_device
from tests.conftest import make_field
from tests.test_periodic_extension import _smooth


@pytest.fixture
def wtmm(clean_registry):
    register_builtin_devices()
    return get_device("wtmm2d")


def _params(dev, **over):
    return dict(defaults_for(dev), **over)


def _scales(params):
    from dynamix.core.wtmm_backend import compute_scales2d

    return list(compute_scales2d(params["n_oct"], params["n_voice"], params["a_min"]))


def test_the_defaults_and_the_hidden_superseded_choices(wtmm):
    p = {q.name: q for q in wtmm.params}
    assert p["edge"].default == "auto" and set(p["edge"].hidden_choices) == {"cut", "daf_periodic"}
    assert p["pad_width"].default == "measured_reach" and p["border_cut"].default == "none"
    assert p["border_policy"].level == "hidden" and p["coi_k"].level == "hidden"
    assert p["cut_scope"].level == "hidden"


def test_a_default_run_mirrors_to_the_widest_measured_reach_and_cuts_nothing(wtmm):
    """The default is the mirror (JPEG2000's symmetric extension), wide enough that nothing wraps
    round; the padding is cropped off before the maxima are found, and the Border cut is a choice."""
    from dynamix.core.edges import border_widths, pad_widths

    field = make_field(_smooth())
    params = _params(wtmm)
    res = wtmm.compute(field, params)
    widths, _ = pad_widths(_scales(params), "measured_reach", wavelet=params["wavelet"])
    assert res["params"]["pad"] == max(widths) and res["params"]["pad_mode"] == "reflect"
    assert res["_padding"]["widths"] == widths and "_border_cut" not in res
    cut = wtmm.compute(field, _params(wtmm, border_cut="measured_reach"))
    want = border_widths(_scales(params), "measured_reach", wavelet=params["wavelet"], scope="largest")
    assert cut["_border_cut"]["borders"] == want


def test_auto_after_periodic_extension_is_periodic_with_no_padding_and_no_cut(wtmm):
    ext = get_device("periodic_extension")
    field = make_field(_smooth())
    extended = ext.compute(field, dict(defaults_for(ext), layout="origin"))
    res = wtmm.compute(extended, _params(wtmm))
    H, W = field.values.shape
    assert res["params"]["pad_mode"] == "periodic" and res["params"]["pad"] == 0
    assert res["params"]["real_block"] == (H, W) and "_border_cut" not in res
    for e in res["extrema"]:
        assert (np.asarray(e["x"]) < W).all() and (np.asarray(e["y"]) < H).all()
    toggled = wtmm.compute(extended, _params(wtmm, border_cut="measured_reach"))
    assert toggled["_border_cut"]["border_cut"] == "measured_reach"


def test_the_fixed_and_border_percent_widths(wtmm):
    field = make_field(_smooth())
    fixed = wtmm.compute(field, _params(wtmm, pad_width="fixed_px", pad_px=32))
    assert fixed["params"]["pad"] == 32
    pct = wtmm.compute(field, _params(wtmm, pad_width="border_pct", pad_pct=0.72))
    assert pct["params"]["pad"] == int(_scales(_params(wtmm))[-1] * 0.72)


def test_periodic_warns_on_a_raster_that_is_not_periodic(wtmm):
    ramp = make_field(np.add.outer(np.arange(96.0), np.arange(96.0)))
    assert "edge" in wtmm.knob_warnings(ramp, _params(wtmm, edge="periodic"))
    ext = get_device("periodic_extension")
    extended = ext.compute(make_field(_smooth()), defaults_for(ext))
    assert "edge" not in wtmm.knob_warnings(extended, _params(wtmm, edge="periodic"))
    assert "edge" not in wtmm.knob_warnings(extended, _params(wtmm))


def test_extend_then_wtmm_in_one_chain_keeps_no_maxima_in_the_gap(wtmm):
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    field = make_field(_smooth())
    chain = Chain((DeviceRef("periodic_extension", {"layout": "origin"}), DeviceRef("wtmm2d", {}))).materialized()
    out = resolve(Layer(layer_id=1, name="L", source_id="s", chain=chain), field, Cache())
    H, W = field.values.shape
    for e in out.result["extrema"]:
        assert (np.asarray(e["x"]) < W).all() and (np.asarray(e["y"]) < H).all()


def test_a_border_that_covers_the_raster_warns_and_auto_then_cuts_nothing(wtmm):
    """A q-Gaussian at q = 1.5 reaches past 120 px: on a 128 px raster a chosen Border cut by
    measured reach keeps no maxima and says so; ``auto`` cuts nothing there, keeps the maxima near
    the edges and says why; a border that fits keeps quiet."""
    from dynamix.devices.wtmm import resolved_boundary

    field = make_field(_smooth(128, seed=3))
    heavy = _params(wtmm, n_oct=3, n_voice=4, wavelet="q_gaussian", q_tsallis=1.5,
                    border_cut="auto")
    chosen = dict(heavy, border_cut="measured_reach")
    assert "covers this 128 × 128" in wtmm.knob_warnings(field, chosen)["border_cut"]
    assert all(len(level["mod"]) == 0 for level in wtmm.compute(field, chosen)["extrema"])
    assert resolved_boundary(heavy, field) == ("reflect", "none")
    assert wtmm.knob_warnings(field, heavy)["border_cut"].startswith("auto cuts nothing")
    assert any(len(level["mod"]) for level in wtmm.compute(field, heavy)["extrema"])
    assert "would cover" in wtmm.derived_reading("border_cut", "auto", field, heavy)
    fits = _params(wtmm, n_oct=2, n_voice=4, wavelet="gaussian", border_cut="auto")
    assert resolved_boundary(fits, field) == ("reflect", "measured_reach")
    assert "border_cut" not in wtmm.knob_warnings(field, fits)
    assert "border_cut" not in wtmm.knob_warnings(field, dict(heavy, border_cut="none"))


def test_an_unconverged_fine_scale_is_read_beside_the_edge_knob_and_never_warns(wtmm):
    field = make_field(_smooth(128, seed=3))
    params = _params(wtmm, a_min=0.75)       # the Mexican hat's 5.2 and 6.2 px do not converge
    assert "pad_width" not in wtmm.knob_warnings(field, params)
    assert "widest converged reach" in wtmm.derived_reading("edge", "auto", field, params)


def test_extend_fork_then_wtmm_runs_periodic_with_no_maxima_in_the_gap(wtmm, tmp_path):
    """The fork flow (spec 25 §5, test 6): the forked dataset keeps the Periodic extension's real
    block, so WTMM's ``auto`` reads it as periodic with no padding and no cut, and keeps no maxima
    in the gap."""
    from dynamix.core.derivative import carried_from, write_derivative
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices.periodic_extension import PeriodicExtension
    from dynamix.devices.wtmm import resolved_boundary

    field = make_field(_smooth())
    ext = PeriodicExtension().compute(field, dict(defaults_for(PeriodicExtension()), layout="origin"))
    path = tmp_path / "extended.npz"
    write_derivative(str(path), bands=[("as shown", np.asarray(ext.values))], frame=ext.frame,
                     x_axis=ext.x_axis, y_axis=ext.y_axis, name="extended",
                     provenance={"derived": {"from_layer": "L"}, **carried_from(ext)})
    forked = RasterField.from_file(str(path))
    assert resolved_boundary(defaults_for(wtmm), forked) == ("periodic", "none")
    res = wtmm.compute(forked, _params(wtmm, n_oct=2))
    assert res["_padding"]["pad"] == 0 and "_border_cut" not in res
    H, W = field.values.shape
    assert any(len(e["x"]) for e in res["extrema"])
    for e in res["extrema"]:
        assert (np.asarray(e["x"]) < W).all() and (np.asarray(e["y"]) < H).all()
    assert carried_from(field) == {}
