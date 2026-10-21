# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A bounded session soak over the fitting tools.

Each round, on a 64 × 64 field with a landed WTMM result and a wavelet-leaders layer beside it
(structure functions need the leaders' sample sets): every tool is opened, replaced while it is
visible, dragged through the shared range, and left closed, hidden or minimized; the generalized
spectrum computes and is cancelled, computes and is closed mid-compute, and computes again; the
active layer switches away and back, which closes and releases the tools, the computing one too. After the round's deferred deletes
and a full collection the session holds what it held after the first round: the same windows and
widgets, no tool object, the same connections on the long-lived objects, no fit registered and no
job running, and a Python object count that grows by less than a small constant per round.
"""
from __future__ import annotations

import gc
import os
import warnings

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from tests.conftest import fbm2d

ROUNDS = 20
#: Python objects a round may leave behind on average, counted from the second round. The count
#: measures flat, within a few objects, from the third round on; a tool window kept alive by one
#: stray connection holds thousands.
GROWTH_PER_ROUND = 10

OPENERS = {
    "_multifractal_window": "_on_spectrum_button_clicked",
    "_spectrum_construction_window": "_on_dh_button_clicked",
    "_cumulants_window": "_on_cumulants_button_clicked",
    "_ld_window": "_on_ld_button_clicked",
    "_generalized_window": "_on_generalized_button_clicked",
}


@pytest.fixture
def session(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    win = MainWindow(steps=(("wtmm2d", {"n_oct": 2, "n_voice": 2}), ("scale_select", {"scale_idx": 0})))
    qtbot.addWidget(win, before_close_func=lambda w: qtbot.waitUntil(lambda: not w.is_computing,
                                                                       timeout=30000))
    field = RasterField(name="fbm", values=fbm2d(64, 0.7, seed=0), frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(field, "mem:soak")
    wtmm = win.layer
    leaders = win.project.add_layer("leaders", wtmm.source_id,
                                    Chain((DeviceRef("wavelet_leaders", {}),)).materialized())
    win.add_layer_row(leaders, win.field)
    _select(qtbot, win, leaders)
    _select(qtbot, win, wtmm)
    return win, wtmm, leaders


def _select(qtbot, win, layer) -> None:
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win.layer_list.select_layer(layer.layer_id)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    assert win.layer is layer


def _drag(qtbot, region, positions) -> None:
    """A scale-range slider moved through ``positions``, one event-loop turn apart."""
    for lo, hi in positions:
        region.setRegion((lo, hi))
        qtbot.wait(1)


def _positions(win, k: int) -> list:
    """Three ranges inside the active result's log2 scales, shifted with the round."""
    a = np.log2(np.asarray(win._active_result["scales"], dtype=np.float64))
    lo, span = float(a.min()), float(a.max() - a.min())
    t = 0.1 * (k % 3)
    return [(lo, lo + span), (lo + t * span, lo + span), (lo, lo + (0.8 - t) * span)]


def _receivers(obj) -> int:
    """Connections on every signal ``obj`` declares."""
    mo = obj.metaObject()
    total = 0
    for i in range(mo.methodCount()):
        method = mo.method(i)
        if method.methodType() == QtCore.QMetaMethod.MethodType.Signal:
            total += obj.receivers(QtCore.SIGNAL(bytes(method.methodSignature()).decode()))
    return total


def _census(win) -> dict:
    from dynamix.shell.generalized_spectrum_window import GeneralizedSpectrumWindow
    from dynamix.shell.large_deviations_window import LargeDeviationsWindow
    from dynamix.shell.log_cumulants_window import LogCumulantsWindow
    from dynamix.shell.multifractal_window import MultifractalWindow
    from dynamix.shell.spectrum_window import SpectrumWindow
    from dynamix.shell.structure_functions_window import StructureFunctionsWindow

    tools = (MultifractalWindow, SpectrumWindow, StructureFunctionsWindow, LogCumulantsWindow,
             LargeDeviationsWindow, GeneralizedSpectrumWindow)
    QtCore.QThreadPool.globalInstance().waitForDone(30000)
    for _ in range(3):
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        QtCore.QCoreApplication.processEvents()
    gc.collect()
    # pytest keeps a record of every warning raised (pyqtgraph's all-NaN bounds among them); the
    # records belong to the harness and stay out of the session's count.
    objects = [o for o in gc.get_objects() if not isinstance(o, (warnings.WarningMessage, Warning))]
    app = QtWidgets.QApplication.instance()
    shared = win._shared_range
    return {
        "windows": len(app.topLevelWidgets()),
        "widgets": len(app.allWidgets()),
        "tool objects": sum(isinstance(o, tools) for o in objects),
        "shared-range subscriptions": shared.receivers(QtCore.SIGNAL("changed(double,double,PyObject)")),
        "connections on the long-lived objects": _receivers(win) + _receivers(shared) + _receivers(app),
        "registered fits": len(win._published_fits),
        "running jobs": QtCore.QThreadPool.globalInstance().activeThreadCount(),
        "python objects": len(objects),
    }


def _round(qtbot, win, wtmm, leaders, k: int) -> None:
    # The leaders layer: structure functions opened, replaced while visible, dragged, closed.
    _select(qtbot, win, leaders)
    for _ in range(2):
        win._on_structure_button_clicked()
    _drag(qtbot, win._structure_window._region, _positions(win, k))
    win._structure_window.close()
    # The WTMM layer: every tool its result offers, each opened and replaced while visible.
    _select(qtbot, win, wtmm)
    for attr, opener in OPENERS.items():
        getattr(win, opener)()
        first = getattr(win, attr)
        getattr(win, opener)()
        assert getattr(win, attr) is not first and not first.isVisible()
    positions = _positions(win, k)
    _drag(qtbot, win._cumulants_window.regions[0], positions[:2])
    _drag(qtbot, win._generalized_window._region, positions[2:])
    assert win._multifractal_window._region.getRegion() == pytest.approx(win._shared_range.value)
    generalized = win._generalized_window
    generalized.apply_button.click()
    generalized.cancel_button.click()
    generalized.apply_button.click()
    generalized.close()                            # mid-compute
    QtCore.QThreadPool.globalInstance().waitForDone(30000)
    qtbot.wait(1)
    assert generalized.current_fit() is None
    assert not any(f.get("kind") == "generalized spectrum" for f in win._published_fits.values())
    win._on_generalized_button_clicked()            # released below by the layer switch, mid-compute
    win._generalized_window.apply_button.click()
    win._cumulants_window.hide()
    win._ld_window.showMinimized()
    win._multifractal_window.close()
    win._spectrum_construction_window.close()
    # Away and back: landing the layer again closes and releases every tool it owns.
    _select(qtbot, win, leaders)
    _select(qtbot, win, wtmm)
    assert all(getattr(win, attr) is None for attr in (*OPENERS, "_structure_window"))


def test_a_session_of_fitting_tools_returns_to_its_baseline_every_round(qtbot, session):
    win, wtmm, leaders = session
    _round(qtbot, win, wtmm, leaders, 0)           # the first round also fills the caches
    baseline = _census(win)
    baseline.pop("python objects")
    assert baseline["tool objects"] == 0 and baseline["registered fits"] == 0
    assert baseline["running jobs"] == 0 and baseline["shared-range subscriptions"] == 0
    counts = []
    for k in range(1, ROUNDS):
        _round(qtbot, win, wtmm, leaders, k)
        now = _census(win)
        counts.append(now.pop("python objects"))
        assert now == baseline, k
    growth = (counts[-1] - counts[0]) / (len(counts) - 1)
    if os.environ.get("DYNAMIX_SOAK_REPORT"):
        print(f"\nbaseline {baseline}\npython objects after rounds 1..{ROUNDS - 1}: {counts}\n"
              f"growth {growth:.1f} per round")
    assert growth < GROWTH_PER_ROUND, counts
