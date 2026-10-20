# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The CWT leaders' sample sets in the fitting windows (spec 30 §5.7): the Structure functions
window lists them, seeds its range from the device's fit range, and its canonical estimator on
fractional Brownian motion reads h = H and D = 2, the +d law under all-pixel sampling."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import fbm2d
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


@pytest.fixture
def leaders_result(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for

    register_builtin_devices()
    dev = clean_registry["cwt_leaders"]
    params = defaults_for(dev)
    params.update({"wavelet": "gaussian", "n_oct": 4, "a_min": 1.0, "edge": "reflect", "p": 2.0})
    v = fbm2d(512, 0.5, seed=11)
    field = RasterField(name="fbm", values=v, frame=LocalFrame(), x_axis=np.arange(512.0), y_axis=np.arange(512.0))
    return dev.compute(field, params)


def test_the_structure_functions_window_fits_the_leaders_with_the_plus_d_law(qtbot, leaders_result):
    from dynamix.shell.structure_functions_window import StructureFunctionsWindow

    samples = leaders_result["samples"]
    lo, hi = (float(np.log2(a)) for a in samples["fit_range"])     # the window's axis is log2 a
    window = StructureFunctionsWindow(samples, levels=(lo, hi), source="CWT")
    qtbot.addWidget(window)
    names = [window.set_combo.itemText(i) for i in range(window.set_combo.count())]
    assert names == ["CWT leaders", "CWT p-leaders"]
    assert tuple(window._region.getRegion()) == pytest.approx((lo, hi))
    fit = window.current_fit()
    q = np.asarray(fit["q"])
    at = lambda value: int(np.argmin(np.abs(q - value)))       # noqa: E731
    assert fit["h"][at(0.0)] == pytest.approx(0.5, abs=0.1)
    assert fit["D"][at(0.0)] == pytest.approx(2.0, abs=0.15)
    assert fit["tau"][at(2.0)] == pytest.approx(2 * 0.5 - 2, abs=0.25)


def test_the_raster_tab_draws_the_disc_at_the_scale_shown(qtbot, clean_registry):
    """A landed cwt_leaders result puts a ring of R(a) at the scale shown on the canvas, centred on
    what the result covers (spec 30 §3.5: the "visual of the radii at each scale")."""
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    v = fbm2d(96, 0.5, seed=12)
    field = RasterField(name="disc", values=v, frame=LocalFrame(), x_axis=np.arange(96.0), y_axis=np.arange(96.0))
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(field, "/nonexistent/disc.npz")
    params = defaults_for(get_device("cwt_leaders"))
    params.update({"wavelet": "gaussian", "n_oct": 3, "a_min": 1.0, "edge": "reflect", "p": 0.0})
    desc = [{"device": "cwt_leaders", "params": params}]
    win.strips.set_steps(desc, field=win.field)
    with qtbot.waitSignal(win.resolved, timeout=120000):
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=120000)
    radii = win._active_result["_radii"]
    xs, ys = (np.asarray(a, dtype=float) for a in win.canvas.disc_item.getData())
    assert xs.size == 73
    np.testing.assert_allclose(np.hypot(xs - 48.0, ys - 48.0), radii[0], atol=1e-6)


def test_a_leaders_layer_opens_the_partition_functions_window(qtbot, clean_registry):
    """The Partition functions button reads a leaders result's structure tables: enabled, named
    for what it holds, and the fitter opens on them."""
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.main_window import MainWindow
    register_builtin_devices()
    v = fbm2d(96, 0.5, seed=12)
    field = RasterField(name="tables", values=v, frame=LocalFrame(), x_axis=np.arange(96.0), y_axis=np.arange(96.0))
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(field, "/nonexistent/tables.npz")
    params = defaults_for(get_device("cwt_leaders"))
    params.update({"wavelet": "gaussian", "n_oct": 3, "a_min": 1.0, "edge": "reflect", "p": 0.0})
    desc = [{"device": "cwt_leaders", "params": params}]
    win.strips.set_steps(desc, field=win.field)
    with qtbot.waitSignal(win.resolved, timeout=120000):
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=120000)
    # The leaders have their own Structure functions window, so this button keeps its name.
    assert win._spectrum_button.isEnabled() and win._spectrum_button.text() == "Partition functions…"
    assert win._dh_button.isEnabled()
    win._on_spectrum_button_clicked()
    assert win._multifractal_window is not None


def test_a_leaders_landing_sets_the_wavelet_bar_so_the_fitters_read_physical_lengths(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.main_window import MainWindow
    register_builtin_devices()
    v = fbm2d(96, 0.5, seed=12)
    field = RasterField(name="bar", values=v, frame=LocalFrame(), x_axis=np.arange(96.0), y_axis=np.arange(96.0))
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(field, "/nonexistent/bar.npz")
    params = defaults_for(get_device("cwt_leaders"))
    params.update({"wavelet": "gaussian", "n_oct": 3, "a_min": 1.0, "edge": "reflect", "p": 0.0})
    desc = [{"device": "cwt_leaders", "params": params}]
    win.strips.set_steps(desc, field=win.field)
    with qtbot.waitSignal(win.resolved, timeout=120000):
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=120000)
    assert win._bar_device is not None and win._bar_device[0] == "cwt_leaders"
    assert win._physical_of_log2a() is not None
    win._on_structure_button_clicked()
    assert win._structure_window._xaxis_combo.isEnabled()
    assert win._structure_window.family_plot.getAxis("bottom").labelText == "log₂ a"   # a continuous ladder
