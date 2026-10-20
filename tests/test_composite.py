# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The multiband composite display law: channel assignment, solo/mute, the mixer panel,
and the window flow that persists the spec per layer."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets  # noqa: F401

from dynamix.shell.canvas import _composite_rgba
from dynamix.shell.right_panel import CompositePanel


def _stack():
    v = np.zeros((4, 5, 3))
    v[..., 0] = 10.0
    v[..., 1] = 20.0
    v[..., 2] = np.linspace(0, 30, 20).reshape(4, 5)
    v[0, 0, :] = np.nan
    return v


def test_channels_follow_the_assignment_and_nan_goes_transparent():
    rgba = _composite_rgba(_stack(), {"r": 2, "g": 0, "b": 1, "solo": [], "mute": []})
    assert rgba.shape == (5, 4, 4) and rgba.dtype == np.uint8
    # constant bands stretch to 0; the ramp band spans 0..255 in R
    assert rgba[..., 0].max() == 255 and rgba[..., 1].max() == 0
    assert rgba[0, 0, 3] == 0 and rgba[1, 1, 3] == 255        # NaN pixel transparent


def test_mute_silences_a_channel_and_one_solo_goes_grayscale():
    muted = _composite_rgba(_stack(), {"r": 2, "g": 0, "b": 1, "solo": [], "mute": [2]})
    assert muted[..., 0].max() == 0                            # R's band muted
    solo = _composite_rgba(_stack(), {"r": 2, "g": 0, "b": 1, "solo": [2], "mute": []})
    assert np.array_equal(solo[..., 0], solo[..., 1])          # grayscale
    assert np.array_equal(solo[..., 1], solo[..., 2])
    assert solo[..., 0].max() == 255


def test_panel_spec_round_trips_and_channels_are_exclusive(qtbot):
    panel = CompositePanel()
    qtbot.addWidget(panel)
    spec = {"r": 0, "g": 1, "b": None, "solo": [1], "mute": [2]}
    panel.set_bands(["b1", "b2", "b3"], spec)
    got = panel.spec()
    assert got["r"] == 0 and got["g"] == 1 and got["b"] is None
    assert got["solo"] == [1] and got["mute"] == [2]
    changes = []
    panel.compositeChanged.connect(changes.append)
    panel._rows[2][0].setCurrentText("R")                      # steal R from band 1
    assert changes and changes[-1]["r"] == 2
    assert panel._rows[0][0].currentText() == "—"


def test_the_composite_honours_its_stretch_type_per_band():
    rng = np.random.default_rng(3)
    v = np.stack([rng.normal(0, 1, (40, 40)), rng.exponential(1, (40, 40)),
                  rng.normal(0, 5, (40, 40))], axis=-1)
    base = {"r": 0, "g": 1, "b": 2, "solo": [], "mute": []}
    eq = _composite_rgba(v, {**base, "stretch": "histogram"})
    counts, _ = np.histogram(eq[..., 1], bins=4, range=(0, 256))
    assert counts.min() > 0.8 * counts.max()                # equalised: every level used
    lin = _composite_rgba(v, {**base, "stretch": "linear"})
    pct = _composite_rgba(v, {**base, "stretch": "percent", "stretch_pct": 2.0})
    assert not np.array_equal(lin, pct)
    sym = np.stack([np.linspace(-3, 3, 16).reshape(4, 4)] * 3, axis=-1)
    bip = _composite_rgba(sym, {**base, "stretch": "bipolar", "stretch_pct": 0.0})
    mid = np.abs(sym[..., 0].T) < 0.25                      # (x, y) orientation
    assert np.all(np.abs(bip[..., 0][mid].astype(int) - 127) <= 32)


def test_the_stretch_control_shows_the_value_its_type_uses_and_round_trips(qtbot):
    panel = CompositePanel()
    qtbot.addWidget(panel)
    panel.set_bands(["b1", "b2", "b3"], {"r": 0, "g": 1, "b": 2, "stretch": "stddev",
                                         "stretch_pct": 5.0, "stretch_k": 1.5})
    assert panel.spec()["stretch"] == "stddev" and panel.stretch_value.value() == 1.5
    assert panel.stretch_value.suffix() == " σ" and not panel.stretch_value.isHidden()
    changes = []
    panel.compositeChanged.connect(changes.append)
    panel.stretch_combo.setCurrentIndex(panel.stretch_combo.findData("histogram"))
    assert changes[-1]["stretch"] == "histogram" and panel.stretch_value.isHidden()
    panel.stretch_combo.setCurrentIndex(panel.stretch_combo.findData("percent"))
    assert panel.stretch_value.value() == 5.0                # its OWN percent, not k
    panel.stretch_value.setValue(1.0)
    assert changes[-1]["stretch_pct"] == 1.0 and changes[-1]["stretch_k"] == 1.5


from tests.test_shell_window import stub_devices, window  # noqa: E402,F401


def test_a_multiband_layer_shows_the_mixer_and_persists_its_spec(window, qtbot):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    win = window
    win._start_worker = lambda: None
    v = np.zeros((6, 7, 4))
    for b in range(4):
        v[..., b] = b
    f = RasterField(name="stack", values=v, frame=LocalFrame(),
                    x_axis=np.arange(7.0), y_axis=np.arange(6.0),
                    provenance={"bands": ["SWIR_Band4/ImageData", "SWIR_Band5/ImageData",
                                          "SWIR_Band6/ImageData", "SWIR_Band7/ImageData"]})
    win.load_field(f, "mem:stack")
    stack_master = win.layer
    assert win._composite_frame.isVisible() or not win.isVisible()  # frame shown (offscreen)
    assert win._composite_frame.isVisibleTo(win)
    assert win.canvas._composite and win.canvas._composite["r"] == 0
    rows = win.composite_panel._rows
    assert len(rows) == 4
    # solo band 4 through the panel: the tag persists and the canvas follows
    rows[3][1].setChecked(True)
    import json

    stored = json.loads(win.layer.tags["ui.composite"])
    assert stored["solo"] == [3]
    assert win.canvas._composite["solo"] == [3]
    # a scalar layer hides the mixer again
    flat = RasterField(name="flat", values=np.zeros((6, 7)), frame=LocalFrame(),
                       x_axis=np.arange(7.0), y_axis=np.arange(6.0))
    win.load_field(flat, "mem:flat")
    assert not win._composite_frame.isVisibleTo(win)
    assert win.canvas._composite is None
    # switching back restores the stored spec
    win.layer_list.select_layer(stack_master.layer_id)
    assert win.canvas._composite and win.canvas._composite["solo"] == [3]



def test_the_composite_stretch_is_saved_with_the_layer_and_is_not_the_display_one(window,
                                                                                   qtbot):
    import json

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    win = window
    win._start_worker = lambda: None
    v = np.random.default_rng(1).normal(size=(6, 7, 3))
    win.load_field(RasterField(name="s", values=v, frame=LocalFrame(), x_axis=np.arange(7.0),
                               y_axis=np.arange(6.0)), "mem:s3")
    master = win.layer
    panel = win.composite_panel
    panel.stretch_combo.setCurrentIndex(panel.stretch_combo.findData("bipolar"))
    stored = json.loads(master.tags["ui.composite"])
    assert stored["stretch"] == "bipolar" and win.canvas._composite["stretch"] == "bipolar"
    assert master.tags.get("ui.stretch") != "bipolar"       # the Display control is untouched


def test_a_byte_picture_opens_at_its_own_range_and_keeps_it_through_the_mixer(window):
    """A stack whose file stores bytes (a colour picture) composites at linear 0..255, so it
    reads as the file shows it: the bytes come back within a count. The mixer's round trip
    keeps that range."""
    import json
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    win = window
    win._start_worker = lambda: None
    rgb = np.zeros((6, 7, 3))
    rgb[..., 0] = np.linspace(10, 200, 7)[None, :]
    rgb[..., 1] = 90.0
    rgb[..., 2] = np.linspace(255, 0, 6)[:, None]
    f = RasterField(name="picture", values=rgb, frame=LocalFrame(), x_axis=np.arange(7.0),
                    y_axis=np.arange(6.0), provenance={"bands": ["R", "G", "B"], "dtype": "uint8"})
    win.load_field(f, "mem:picture")
    spec = win.canvas._composite
    assert spec["stretch"] == "linear" and spec["range"] == [0, 255]
    shown = _composite_rgba(rgb, spec).transpose(1, 0, 2)[..., :3].astype(float)
    assert np.abs(shown - rgb).max() <= 1.0
    win.composite_panel._rows[1][1].setChecked(True)            # solo G through the mixer
    stored = json.loads(win.layer.tags["ui.composite"])
    assert stored["solo"] == [1] and stored["range"] == [0, 255]
    other = RasterField(name="floats", values=rgb / 255.0, frame=LocalFrame(), x_axis=np.arange(7.0),
                        y_axis=np.arange(6.0), provenance={"bands": ["a", "b", "c"]})
    win.load_field(other, "mem:floats")
    assert win.canvas._composite["stretch"] == "percent" and "range" not in win.canvas._composite


def test_a_derivative_with_every_band_inherits_the_origins_mix(window, tmp_path):
    """A refined picture forked from a stack opens drawing with the stack's own composite."""
    import json
    from dynamix.core.derivative import write_derivative
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    win = window
    win._start_worker = lambda: None
    rgb = np.random.default_rng(2).random((6, 7, 3)) * 255.0
    f = RasterField(name="picture", values=rgb, frame=LocalFrame(), x_axis=np.arange(7.0),
                    y_axis=np.arange(6.0), provenance={"bands": ["R", "G", "B"], "dtype": "uint8"})
    win.load_field(f, "mem:picture")
    origin = win.layer
    win.composite_panel._rows[2][2].setChecked(True)            # mute B on the origin
    mix = win.layer.tags["ui.composite"]
    assert json.loads(mix)["mute"] == [2]
    path = str(tmp_path / "refined.npz")
    write_derivative(path, bands=[("R", rgb[..., 0]), ("G", rgb[..., 1]), ("B", rgb[..., 2])],
                     frame=f.frame, x_axis=f.x_axis, y_axis=f.y_axis, name="refined",
                     provenance={"derived": {"from_layer": "picture"}})
    win._open_derivative(path, name="refined", temporary=True, nest_under=None, vectors=False,
                         origin_layer_id=origin.layer_id)
    assert win.layer is not origin
    assert win.layer.tags["ui.composite"] == mix
    assert win.canvas._composite["mute"] == [2]
