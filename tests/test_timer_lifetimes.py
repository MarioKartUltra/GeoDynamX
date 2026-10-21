# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Deferred callbacks die with the widget that queued them.

A two-argument ``QTimer.singleShot(ms, callable)`` has no receiver: in PySide6 a zero delay, or
any lambda or closure, is called after its widget is destroyed, into a C++ object that is gone. A
positive delay on a bound method of a QObject is connected with that object as receiver, and Qt
drops it with the object. Each test destroys the owner before its callback is due and asserts
nothing reaches the dead widget; a live twin, where one fits, shows the callback still runs.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore
from shiboken6 import isValid

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


def _destroy(widget) -> None:
    """Close ``widget`` and run its deferred delete now, ahead of any queued callback."""
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert not isValid(widget)


def _field(n=32):
    v = np.random.default_rng(0).standard_normal((n, n)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(), x_axis=np.arange(float(n)),
                       y_axis=np.arange(float(n)))


def _picture(ny=10, nx=14, s=5, full=(50, 70)):
    f = RasterField(name="pic@pic5", values=np.arange(ny * nx, dtype=float).reshape(ny, nx),
                    frame=LocalFrame(), x_axis=np.arange(nx, dtype=np.float64),
                    y_axis=np.arange(ny, dtype=np.float64))
    f.provenance.update({"display_stride": s, "full_dims": full, "source": "/nonexistent/pic.tif",
                         "window": {"row_off": 0, "col_off": 0}})
    return f


def test_a_dropped_tree_is_read_back_only_while_its_panel_lives(qtbot):
    from dynamix.model.layer_stack import Group, LayerStack, Row
    from dynamix.shell.layers_panel import LayersPanel

    def panel():
        s = LayerStack()
        s.insert(Group("ds:src0", "dem", auto=True))
        s.insert(Row("src:src0", "raster"), "ds:src0", 0)
        return LayersPanel(s, label_of=lambda rid: rid)

    live, dead = panel(), panel()
    qtbot.addWidget(live)
    with qtbot.capture_exceptions() as raised:
        live._on_dropped()
        dead._on_dropped()
        _destroy(dead)
        qtbot.wait(20)
    assert raised == [] and live._drop_pending is False


def test_a_row_flash_is_cancelled_with_its_panel(qtbot):
    from dynamix.model.chain import Chain
    from dynamix.model.project import Project
    from dynamix.shell.layer_panel import _FLASH_MS, LayerPanel

    def flashing():
        project = Project(title="t")
        layer = project.add_layer("A", project.add_source("/data/a.tif").source_id, Chain())
        p = LayerPanel(project)
        p.add_layer_row(layer, None)
        p.flash_row(layer.layer_id)
        return p, p._row_widgets[layer.layer_id]

    (live, row), (dead, _) = flashing(), flashing()
    qtbot.addWidget(live)
    with qtbot.capture_exceptions() as raised:
        _destroy(dead)
        qtbot.wait(_FLASH_MS + 100)
    assert raised == [] and row.property("flash") == "false"


def test_a_refusal_flash_is_cancelled_with_its_zone(qtbot):
    from dynamix.shell.workflow_zone import _FLASH_MS, WorkflowZone

    live, dead = WorkflowZone(), WorkflowZone()
    qtbot.addWidget(live)
    with qtbot.capture_exceptions() as raised:
        live._flash_refusal()
        dead._flash_refusal()
        _destroy(dead)
        qtbot.wait(_FLASH_MS + 100)
    assert raised == [] and live.property("refused") is False


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def windows(qtbot, builtins):
    """MainWindows a test may destroy itself; any it leaves alive are closed afterwards."""
    made = []
    yield made
    for win in made:
        if isValid(win):
            qtbot.waitUntil(lambda w=win: not w.is_computing, timeout=30000)
            win.close()
            win.deleteLater()


def _window(qtbot, windows, field, path="/nonexistent/f.npz"):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=())
    windows.append(win)
    win.load_field(field, path)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    return win


def test_the_3d_warm_up_is_cancelled_with_its_window(qtbot, windows, monkeypatch):
    from dynamix.shell.main_window import MainWindow

    warmed = []

    def _warm_up_3d(self):
        warmed.append(self)

    # PySide takes a bound method's object as the receiver when the method is found on it by its
    # own name, so the recorder keeps the real method's name.
    monkeypatch.setattr(MainWindow, "_warm_up_3d", _warm_up_3d)
    live, dead = MainWindow(steps=()), MainWindow(steps=())
    windows.extend((live, dead))
    live.show()
    dead.show()
    _destroy(dead)
    qtbot.wait(1700)
    assert warmed == [live]


def _drop(win, *devices):
    from dynamix.model.device import defaults_for, get_device

    desc = [{"device": d, "params": defaults_for(get_device(d)), "bypassed": False, "rack": None}
            for d in devices]
    win.strips.set_steps(desc, field=win.field)
    win._on_chain_edited(desc)


@pytest.mark.parametrize("case", ["armed roi revert", "field stage refusal",
                                  "roi holder replacement", "analyzer fork"])
def test_a_chain_edit_s_deferred_follow_up_dies_with_its_window(qtbot, windows, case):
    """Each branch of a drop that finishes on the next tick: the zone's resync to the reverted
    or replaced chain, and the fork's spawn of its sibling layer."""
    if case == "field stage refusal":
        win = _window(qtbot, windows, _picture(), "/nonexistent/pic.tif")
        _drop(win, "noise")
    elif case == "armed roi revert":
        win = _window(qtbot, windows, _field())
        _drop(win, "wtmm2d_roi")
    elif case == "roi holder replacement":
        win = _window(qtbot, windows, _field())
        win._auto_run_action.setChecked(False)
        win.layer.tags["roi.window"] = "0,0,32,32"
        _drop(win, "mz_edges", "cdf_edges")
    else:
        win = _window(qtbot, windows, _field())
        _drop(win, "mz_edges")
    assert not win.is_computing
    with qtbot.capture_exceptions() as raised:
        _destroy(win)
        qtbot.wait(20)
    assert raised == []


def test_an_output_request_dispatches_nothing_after_its_window_closes(qtbot, windows):
    """The lazy-output dispatch reads the window's closing state first: a closed window starts
    no job, whatever its widgets' state."""
    win = _window(qtbot, windows, _field())
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, "mz_edges")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    win.layer_list.outputHideToggled.emit(win.layer.layer_id, "modulus", False)
    assert win._out_request is not None and not win.is_computing
    with qtbot.capture_exceptions() as raised:
        _destroy(win)
        qtbot.wait(20)
    assert raised == [] and win._thread is None
