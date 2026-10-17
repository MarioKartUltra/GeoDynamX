# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Garbage collection on the GUI thread only.

Python's cyclic collector runs in whichever thread trips its allocation threshold. A compute
worker (``shell.worker``) allocates plenty, so a collection can start there and free a Qt object
the GUI thread owns, destroying it off its thread: a crash wherever that happens to be (CI caught
it 'Garbage-collecting' inside a worker's scipy import, and inside a pyqtgraph constructor).
:func:`install` turns automatic collection off and has a timer on the GUI thread collect instead,
so every collection runs on the thread that owns the Qt objects. Each tick collects the young
generation if anything was allocated since the last one: the collector's count is net of
deallocations, so a burst of frees (a rebuilt window's deferred deletes) can hide cyclic garbage
below the threshold a sampled check would wait for. The older generations follow the collector's
own thresholds. Reference counting still frees everything outside a cycle at once.

This is a safety net. The reference cycles that leave Qt objects to the collector at all are what
``tests/gc_audit.py`` finds (``DYNAMIX_GC_AUDIT=1`` over the Qt tests); each one broken is a widget
freed the moment its window lets go of it.
"""
from __future__ import annotations

import gc

from PySide6 import QtCore

__all__ = ["MainThreadCollector", "install", "uninstall"]

_ATTR = "_gdx_main_thread_collector"


class MainThreadCollector(QtCore.QObject):
    """Collects on a GUI-thread timer while automatic collection is off."""

    def __init__(self, parent=None, interval_ms: int = 500):
        super().__init__(parent)
        self._was_enabled = gc.isenabled()
        self._thresholds = gc.get_threshold()
        gc.disable()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.check)
        self._timer.start(int(interval_ms))

    def check(self) -> None:
        """Collect the young generation if anything was allocated, an older one when its count
        passed its threshold."""
        c0, c1, c2 = gc.get_count()
        _t0, t1, t2 = self._thresholds
        if c0 > 0:
            gc.collect(2 if c2 > t2 else 1 if c1 > t1 else 0)

    def stop(self) -> None:
        self._timer.stop()
        if self._was_enabled:
            gc.enable()


def install(app, interval_ms: int = 500) -> MainThreadCollector:
    """The application's collector, made on its first call (``app`` owns it)."""
    collector = getattr(app, _ATTR, None)
    if collector is None:
        collector = MainThreadCollector(app, interval_ms)
        setattr(app, _ATTR, collector)
    return collector


def uninstall(app) -> None:
    """Stop the application's collector, detach it from the application and give the collector
    its automatic mode back."""
    collector = getattr(app, _ATTR, None)
    if collector is not None:
        collector.stop()
        collector.setParent(None)
        setattr(app, _ATTR, None)
