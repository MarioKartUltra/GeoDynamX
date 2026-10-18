# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Curve window and the curves a tool publishes (``model.curves``): the Periodic extension's
power spectrum with its band limit, the measures against β, the lineal path; the main window's
Curves… button follows what is on screen."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dynamix.core.daf import band_limit, radial_spectrum
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.curves import Curve, curves_of
from dynamix.model.device import defaults_for, get_device, validate_params
from test_roi_shell_flow import registered_builtins  # noqa: F401


def _field(values):
    values = np.asarray(values, dtype=np.float64)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(values.shape[1])), y_axis=np.arange(float(values.shape[0])))


def _stripes(n, cycles):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    return np.cos(2.0 * math.pi * cycles / n * x)


def test_the_radial_spectrum_peaks_at_the_stripes_and_its_fraction_reads_the_band_limit():
    n, cycles = 128, 8
    k, power, fraction = radial_spectrum(_stripes(n, cycles))
    assert k[1:][np.argmax(power[1:])] == pytest.approx(cycles / n, abs=1.0 / n)
    assert np.all(np.diff(fraction) >= -1e-12) and fraction[-1] == pytest.approx(1.0)
    f_b = band_limit(_stripes(n, cycles), 0.99)
    assert fraction[np.searchsorted(k, f_b)] >= 0.99 - 1e-9        # the same power, the same limit


def test_each_tool_publishes_its_curves(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    ext = get_device("periodic_extension")
    field = _field(_stripes(96, 6) + 0.05 * np.random.default_rng(1).standard_normal((96, 96)))
    curves = curves_of(ext, None, field, defaults_for(ext))
    assert [c.name for c in curves] == ["power spectrum", "power fraction below k"]
    assert curves[0].logx and curves[0].logy and len(curves[0].marks) == 2
    assert curves[0].marks[0][1].startswith("f_b") and "peak" in curves[0].marks[1][1]
    assert curves[0].marks[1][0] == pytest.approx(6 / 96, abs=1 / 96)
    assert "band limit" in curves[0].note and curves[1].marks == (curves[0].marks[0],)
    lp = get_device("lineal_path")
    res = lp.compute(_field(_stripes(64, 4)), validate_params(lp, {}))
    (curve,) = curves_of(lp, res, None, {})
    assert curve.name.startswith("lineal path") and curve.x.size == curve.y.size
    df = get_device("disorder_function")
    p = validate_params(df, {"periodic": True, "k0": "manual", "lambda0": 16.0, "harmonic_filter": "none"})
    names = [c.name for c in curves_of(df, df.compute(_field(_stripes(128, 8)), p), None, p)]
    assert names == ["mean field disorder δ̄(β)", "total field disorder δ(β)", "rooted mean field disorder"]
    hm = get_device("hessian_measure")
    (mu,) = curves_of(hm, hm.compute(_field(_stripes(64, 4)), validate_params(hm, {"periodic": True})), None, {})
    assert mu.name.startswith("Hessian measure") and mu.logy
    assert curves_of(get_device("wtmm2d"), {}, None, {}) == []       # no hook, no curves
    assert curves_of(lp, {}, None, {}) == []                           # nothing landed yet


def test_the_window_shows_one_curve_at_a_time_with_its_marks_and_log_defaults(qtbot):
    from dynamix.shell.curve_window import CurveWindow

    w = CurveWindow()
    qtbot.addWidget(w)
    k = np.linspace(0.01, 0.5, 50)
    curves = [Curve("spectrum", k, 1 / k ** 2, "k", "P", logx=True, logy=True, marks=((0.1, "f_b"),)),
              Curve("fraction", k, np.linspace(0, 1, 50), "k", "fraction", logx=True)]
    w.set_curves(curves, "smooth")
    assert w.windowTitle() == "Curves — smooth" and w.curve_combo.count() == 2
    assert w.logx_check.isChecked() and w.logy_check.isChecked()
    assert len(w.plot.listDataItems()) == 1 and len(w.plot.plotItem.items) == 2   # the curve and its mark
    w.curve_combo.setCurrentIndex(1)
    assert w.logx_check.isChecked() and not w.logy_check.isChecked()
    assert w.plot.plotItem.getAxis("left").logMode is False
    w.logy_check.setChecked(True)                      # a manual switch redraws in log y
    assert w.plot.plotItem.getAxis("left").logMode is True
    w.set_curves(curves, "smooth")                     # a feed keeps the pick by name
    assert w.curve_combo.currentText() == "fraction"


def test_the_main_windows_curves_button_follows_what_is_on_screen(qtbot, registered_builtins):
    from test_roi_shell_flow import _drop
    from dynamix.shell.main_window import MainWindow
    from tests.test_periodic_extension import _smooth

    win = MainWindow(steps=())
    qtbot.addWidget(win)
    assert not win._curves_button.isEnabled()
    v = _smooth(160, seed=2)
    win.load_field(_field(v), "/nonexistent/smooth.npz")
    assert not win._curves_button.isEnabled()                 # a bare dataset publishes none
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "periodic_extension")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert win._curves_button.isEnabled()
    win._on_curves_button_clicked()
    cw = win._curve_window
    assert cw.isVisible() and cw.curve_combo.currentText() == "power spectrum"
    assert "band limit" in cw.note.text()
    assert len(cw.plot.plotItem.items) == 3                  # the curve, f_b and the peak
    import os

    if os.environ.get("CURVE_PNG"):
        cw.resize(760, 560)
        qtbot.wait(100)
        cw.grab().save(os.environ["CURVE_PNG"])
