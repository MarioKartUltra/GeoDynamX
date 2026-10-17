# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A widget its owner lets go of is freed at once, by reference counting: a reference cycle through a
Qt object leaves it to the cyclic collector instead, which runs only on the GUI thread
(``shell.gc_guard``) and only when its timer fires. One test per cycle the audit found
(``tests/gc_audit.py``)."""
from __future__ import annotations

import gc
import weakref

import shiboken6
from PySide6 import QtWidgets

from dynamix.devices import register_builtin_devices
from dynamix.model.device import defaults_for, get_device
from tests.test_shell_window import stub_devices  # noqa: F401


def _freed_without_the_collector(make) -> bool:
    """Qt deletes the widget, as its owner's ``deleteLater`` does, and the last Python reference
    goes: without a cycle through its wrapper, reference counting frees it there and then. (A
    lambda connected to a child's signal is held by Qt and goes with the deletion.) One is built
    and deleted first: the first of a kind in a process can import a library whose exceptions,
    raised and caught during the import, keep the constructor's frame in a cycle of their own."""
    was = gc.isenabled()
    gc.disable()
    try:
        shiboken6.delete(make())
        obj = make()
        assert isinstance(obj, QtWidgets.QWidget)
        ref = weakref.ref(obj)
        shiboken6.delete(obj)
        del obj
        return ref() is None
    finally:
        if was:
            gc.enable()


def test_a_device_box_is_freed_when_dropped(qtbot, clean_registry):
    from dynamix.shell.workflow_zone import DeviceBox

    register_builtin_devices()
    dev = get_device("wtmm2d")
    assert _freed_without_the_collector(lambda: DeviceBox(0, dev, defaults_for(dev)))


def test_a_rack_box_is_freed_when_dropped(qtbot, clean_registry):
    from dynamix.shell.workflow_zone import RackBox

    assert _freed_without_the_collector(lambda: RackBox("rack", []))


def test_the_right_panel_is_freed_when_dropped(qtbot, clean_registry):
    from dynamix.shell.right_panel import RightPanel

    assert _freed_without_the_collector(lambda: RightPanel(()))


def test_a_main_window_is_freed_when_dropped(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    assert _freed_without_the_collector(MainWindow)


def test_a_main_window_with_its_palette_opened_is_freed_when_dropped(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    def make():
        w = MainWindow()
        w.show_palette()
        return w

    assert _freed_without_the_collector(make)


def test_a_main_window_with_an_inspector_open_is_freed_when_dropped(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow
    from tests.test_shell_window import _FIELD, STUB_CHAIN, _open_inspector

    def make():
        w = MainWindow(steps=STUB_CHAIN)
        with qtbot.waitSignal(w.resolved, timeout=10000):
            w.load_field(_FIELD, "mem:stub")
        qtbot.waitUntil(lambda: w._thread is None, timeout=10000)
        if w.layer.source_id not in w._inspectors:      # the settings can bring it back already open
            _open_inspector(w, w.layer.source_id)
        assert w._inspectors
        return w

    assert _freed_without_the_collector(make)
