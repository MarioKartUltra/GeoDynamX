# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A chain that ends on a field stage shows what it produces (noise shows its noise, a stack
through the composite), and a fork takes that produced field."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets  # noqa: F401

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from tests.test_shell_window import stub_devices, window  # noqa: F401


def _stack(nc=3):
    rng = np.random.default_rng(2)
    v = np.stack([np.round(rng.normal(100, 10, (30, 32))) for _ in range(nc)], axis=-1)
    return RasterField(name="stack", values=v if nc > 1 else v[..., 0], frame=LocalFrame(),
                       x_axis=np.arange(32.0), y_axis=np.arange(30.0),
                       provenance={"bands": [f"B{k + 1}" for k in range(nc)]})


def _add_noise(win, qtbot):
    desc = [{"device": "noise", "params": {"amplitude": 5.0, "seed": 3}}]
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)


@pytest.mark.parametrize("nc", [1, 3])
def test_noise_alone_shows_the_noised_field(window, qtbot, nc):
    win = window
    raw = _stack(nc)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(raw, "mem:stack")
    _add_noise(win, qtbot)
    shown = np.asarray(win.canvas._field.values)
    assert shown.shape == np.asarray(raw.values).shape
    assert not np.array_equal(shown, raw.values)             # the noise is on screen
    assert "noised" in win.statusBar().currentMessage()
    assert win._composite_frame.isVisibleTo(win) == (nc > 1)  # a stack: through the mixer


def test_a_fork_takes_the_noised_stack_band_by_band(window, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw
    from tests.test_roi_shell_flow import _Fork

    win = window
    raw = _stack(3)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(raw, "mem:stack")
    _add_noise(win, qtbot)
    noised = np.asarray(win.canvas._field.values).copy()
    monkeypatch.setattr(mw, "ForkDialog", _Fork({"name": "noised", "bands": [0, 1, 2],
                                                 "vectors": False, "nest": False,
                                                 "temporary": True}))
    before = set(win.project.sources)
    win._on_fork_derivative(win.layer.layer_id)
    assert [l.split(" (")[0] for l in _Fork.offered["raster_labels"]] == ["B1", "B2", "B3"]
    (new,) = set(win.project.sources) - before
    forked = RasterField.from_file(win.project.sources[new].path)
    np.testing.assert_array_equal(forked.values, noised)


def test_the_fork_never_takes_a_previous_layers_result(window, qtbot):
    win = window
    notes = []
    win._notify = lambda msg, *a, **k: notes.append(msg)
    win._active_result = {"h_map": np.zeros((2, 2))}          # stale, from another row
    win._land_field_result(object())                          # a non-field tail landed
    assert win._active_result == {} and win._active_field is None
