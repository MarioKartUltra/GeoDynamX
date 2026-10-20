# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The sign at each modulus maximum. The gradient maxima have no sign, so each is mapped back
onto the transform of the smoother itself (``operator = none``) at its own position and scale;
the WTMM carries it per maximum as ``sign`` when its Sign knob is on, and the canvas draws the
H-chains in two colours by it, a second swatch for the negative one."""
from __future__ import annotations

import numpy as np

from conftest import fbm2d, make_field

FRONT = {"n_oct": 2, "n_voice": 1, "a_min": 1.0, "wavelet": "mexican", "fracint_alpha": 0.0,
         "smoother_params": {}}


def test_the_scalar_stage_keeps_the_signed_transform():
    from dynamix.core.wtmm_backend import run_cwt_stage

    stage = run_cwt_stage(make_field(fbm2d(64, 0.5, seed=2)), FRONT, operator="none")
    signed = stage["signed"]
    assert signed.shape == stage["mod"].shape
    np.testing.assert_allclose(np.abs(signed), stage["mod"], rtol=1e-6, atol=1e-7)
    assert (signed < 0).any() and (signed > 0).any()


def test_the_wtmm_device_marks_each_maximum_with_the_sign(clean_registry, monkeypatch):
    import dynamix.core.wtmm_backend as backend_module
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    captured = []
    original = backend_module.run_cwt_stage

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.append(out)
        return out

    monkeypatch.setattr(backend_module, "run_cwt_stage", spy)
    field = make_field(fbm2d(64, 0.5, seed=2))
    dev = get_device("wtmm2d")
    params = dict(defaults_for(dev), n_oct=2, n_voice=1, wavelet="mexican", sign=True)
    res = dev.compute(field, params)
    assert len(captured) == 1 and captured[0]["operator"] == "none"
    signed = captured[0]["signed"]
    for j, ext in enumerate(res["extrema"]):
        assert ext["sign"].dtype == np.int8 and ext["sign"].shape == ext["x"].shape
        expected = np.sign(signed[j][ext["y"], ext["x"]]).astype(np.int8)
        np.testing.assert_array_equal(ext["sign"], expected)
    assert any((e["sign"] < 0).any() for e in res["extrema"])
    assert any((e["sign"] > 0).any() for e in res["extrema"])
    off = dev.compute(field, dict(params, sign=False))
    assert "sign" not in off["extrema"][0] and len(captured) == 1


def test_the_canvas_draws_the_h_chains_in_two_colours_by_sign(qtbot):
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    ext = {"x": np.array([2, 3, 4, 5], dtype=np.int64), "y": np.full(4, 3, dtype=np.int64),
           "mod": np.ones(4), "arg": np.zeros(4), "line_id": np.zeros(4, dtype=np.int64),
           "sign": np.array([1, 1, -1, -1], dtype=np.int8)}
    result = {"extrema": [ext], "chains": [], "_shape": (8, 8), "scales": np.array([1.0])}
    canvas.set_result(result, 0)
    xp, _ = canvas.hchain_item.getData()
    xn, _ = canvas.hchain_neg_item.getData()
    assert sorted(xp[np.isfinite(xp)].tolist()) == [2, 3]
    assert sorted(xn[np.isfinite(xn)].tolist()) == [4, 5]
    canvas.set_overlay_colors("#ffffff", "#ff0000", "#00ff00", hchain_neg="#0000ff")
    assert canvas.hchain_neg_item.opts["pen"].color().name() == "#0000ff"
    pixels = canvas.extrema_raster_item.image                  # (x, y, rgba): the raster view
    assert pixels[2, 3].tolist() == [0, 255, 0, 255] and pixels[4, 3].tolist() == [0, 0, 255, 255]
    unsigned = {k: v for k, v in ext.items() if k != "sign"}
    canvas.set_result({**result, "extrema": [unsigned]}, 0)
    xp, _ = canvas.hchain_item.getData()
    xn, _ = canvas.hchain_neg_item.getData()
    assert sorted(xp[np.isfinite(xp)].tolist()) == [2, 3, 4, 5]
    assert xn is None or len(xn) == 0


def test_the_negative_h_chain_colour_is_a_layer_style_with_its_own_swatch():
    from dynamix.shell.canvas import HCHAIN_NEG_COLOR
    from dynamix.shell.main_window import _display_style_of, _hex
    from dynamix.shell.right_panel import _SWATCH_SPECS, _SWATCH_TIPS

    assert _display_style_of(None)["color_hchain_neg"] == _hex(HCHAIN_NEG_COLOR)
    keys = [k for k, _label, _rgb in _SWATCH_SPECS]
    assert keys == ["color_hchain", "color_hchain_neg", "color_vtrail", "color_extrema"]
    assert "color_hchain_neg" in _SWATCH_TIPS
