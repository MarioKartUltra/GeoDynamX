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
