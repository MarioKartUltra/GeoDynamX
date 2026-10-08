# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Microcanonical MFA tools' band rows in the window: an Outputs group with "band
reconstruction" and "band mask", a radio over Show whose off state is the h-map; the
Reconstruction section holds the band knobs, and a band scrub recomputes only the shown row."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.resolve import output_key, resolve
from dynamix.model.device import declared_outputs, defaults_for, get_device, validate_params

MFA = ("holder_measure", "holder_multiaffine")
RECON_OF = {"holder_measure": "band_recon_measure",
            "holder_multiaffine": "band_recon_multiaffine"}


def _shown(canvas):
    """The raster the canvas shows on top: the product when one is up, else the dataset."""
    return canvas._product if canvas._product is not None else canvas._field


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def win(qtbot, builtins):
    from dynamix.shell.main_window import MainWindow

    v = np.random.default_rng(4).standard_normal((40, 50)).cumsum(0)
    f = RasterField(name="plain", values=v, frame=LocalFrame(), x_axis=np.arange(50.0),
                    y_axis=np.arange(40.0))
    w = MainWindow(steps=())
    qtbot.addWidget(w)
    w.load_field(f, "/nonexistent/plain.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


@pytest.fixture
def h_maps(monkeypatch):
    """The method of every h-map the MFA tools compute, on any thread."""
    from dynamix.devices import holder_methods as hm

    seen: list = []
    real = hm.method_arrays

    def counted(vals, method, params, progress=None):
        seen.append(method)
        return real(vals, method, params, progress)

    monkeypatch.setattr(hm, "method_arrays", counted)
    return seen


def _mfa_layer(win, qtbot, device="holder_measure", **params):
    """Drop ``device`` on the dataset (it spawns a child one tick later) and let it land."""
    desc = [{"device": device, "params": {**defaults_for(get_device(device)), **params}}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def _step(layer):
    return next(s for s in layer.chain.steps if s.device in MFA)


def _key(win, layer, name):
    r = resolve(layer, win._fields[layer.layer_id], win.cache, source_id=layer.source_id)
    out = next(o for o in declared_outputs(get_device(r.analysis_device)) if o.name == name)
    return output_key(r.analysis_device, out, r.analysis_params, r.analysis_key)


def _landed(win, qtbot, layer, name):
    """Wait until ``name``'s output for the layer's current knobs is computed and drawn."""
    qtbot.waitUntil(lambda: not win.is_computing and win._out_request is None, timeout=60000)
    key = _key(win, layer, name)
    qtbot.waitUntil(lambda: key in win.cache and not win.is_computing
                    and win._active_result.get("raster_out") is win.cache.get(key)["raster"],
                    timeout=60000)
    return key


def _row(win, layer, name):
    return win.layer_list._output_items[layer.layer_id][name][0]


def _hidden(win, layer, name):
    return win.layer_list._output_rows[layer.layer_id][name].hide_button.isChecked()


@pytest.mark.parametrize("device", MFA)
def test_an_mfa_layer_shows_its_h_map_with_both_band_rows_hidden(win, qtbot, device):
    layer = _mfa_layer(win, qtbot, device)
    group = win.layer_list._output_groups[layer.layer_id]
    assert [group.child(i).text(0) for i in range(group.childCount())] == [
        "band reconstruction", "band mask"]
    assert _hidden(win, layer, "band") and _hidden(win, layer, "band_mask")
    np.testing.assert_array_equal(_shown(win.canvas).values, win._active_result["h_map"])


@pytest.mark.parametrize("device", MFA)
def test_the_band_row_shows_the_reconstruction_and_hiding_it_brings_the_h_map_back(
        win, qtbot, h_maps, device):
    layer = _mfa_layer(win, qtbot, device, h_lo=-6.0, h_hi=6.0)
    lid = layer.layer_id
    computed = len(h_maps)
    win.layer_list.outputHideToggled.emit(lid, "band", False)
    key = _landed(win, qtbot, layer, "band")
    assert len(h_maps) == computed                                 # the h-map stayed cached
    assert _step(layer).params["show"] == "band"
    recon = get_device(RECON_OF[device])
    ref = recon.compute(win._fields[lid], validate_params(recon, {"h_lo": -6.0, "h_hi": 6.0}))
    np.testing.assert_array_equal(win.cache.get(key)["raster"], ref["raster_out"])
    np.testing.assert_array_equal(_shown(win.canvas).values, ref["raster_out"])
    assert win.layer_list.layer_text(lid).endswith("· band reconstruction")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(lid, "band", True)
    assert _step(layer).params["show"] == "h_map"
    assert _hidden(win, layer, "band") and _hidden(win, layer, "band_mask")
    np.testing.assert_array_equal(_shown(win.canvas).values, win._active_result["h_map"])


def test_the_reconstruction_section_holds_the_band_knobs_and_a_scrub_keeps_the_h_map(
        win, qtbot, h_maps):
    from dynamix.shell.main_window import _RECON_TITLE

    layer = _mfa_layer(win, qtbot, "holder_measure", show="band", h_lo=-6.0, h_hi=6.0)
    first = _landed(win, qtbot, layer, "band")
    panel = win._recon_panel
    assert not win.right_panel._sections[_RECON_TITLE].isHidden()
    assert set(panel.controls) == {"h_lo", "h_hi", "min_island", "max_island", "connectivity"}
    assert not panel.run_button.isEnabled()
    assert panel.run_button.toolTip() == "these rows compute when shown"
    assert " dB · " in panel.readout.text()
    assert _row(win, layer, "band").text(0) == f"band reconstruction · {panel.readout.text()}"
    computed = len(h_maps)
    panel.controls["h_lo"].valueChanged.emit(0.0)
    second = _landed(win, qtbot, layer, "band")
    assert second != first and _step(layer).params["h_lo"] == 0.0
    assert len(h_maps) == computed                                 # only the band recomputed
    np.testing.assert_array_equal(_shown(win.canvas).values, win.cache.get(second)["raster"])


def test_the_band_mask_row_shows_the_band_and_its_share(win, qtbot):
    layer = _mfa_layer(win, qtbot, "holder_multiaffine", h_lo=0.0, h_hi=6.0)
    win.layer_list.outputHideToggled.emit(layer.layer_id, "band_mask", False)
    key = _landed(win, qtbot, layer, "band_mask")
    mask = win.cache.get(key)["raster"]
    h = win._active_result["h_map"]
    np.testing.assert_array_equal(
        mask, (np.isfinite(h) & (h >= 0.0) & (h < 6.0)).astype(np.float32))
    density = win.cache.get(key)["diag"]["band_density"]
    assert density == float((mask > 0).mean())
    share = f"{100.0 * density:.0f}% of pixels"
    assert _row(win, layer, "band_mask").text(0) == f"band mask · {share}"
    assert _hidden(win, layer, "band") and not _hidden(win, layer, "band_mask")


def test_the_vector_view_drapes_the_shown_band(win, qtbot):
    from unittest.mock import MagicMock

    layer = _mfa_layer(win, qtbot, "holder_measure", show="band", h_lo=-6.0, h_hi=6.0)
    key = _landed(win, qtbot, layer, "band")
    seen = []
    win._arrangement = MagicMock()
    win._arrangement.set_layers.side_effect = lambda e: seen.append(
        next(x for x in e if x["layer"] is win.layer))
    win._sync_arrangement(frame_mode=True)
    assert seen[-1]["drape"] is win.cache.get(key)["raster"]
