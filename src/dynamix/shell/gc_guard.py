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
own thresholds, and the oldest also waits until no input event has arrived for :data:`IDLE_MS`: a
full collection walks every tracked object, a fraction of a second with one result in memory and
seconds after a long session, so it runs while the hand is off the mouse and the keyboard and
never inside a drag. Reference counting still frees everything outside a cycle at once.

This is a safety net. The reference cycles that leave Qt objects to the collector at all are what
``tests/gc_audit.py`` finds (``DYNAMIX_GC_AUDIT=1`` over the Qt tests); each one broken is a widget
freed the moment its window lets go of it.
"""
from __future__ import annotations

import gc
import types
import weakref

from PySide6 import QtCore

__all__ = ["MainThreadCollector", "install", "uninstall", "weak_method"]

_ATTR = "_gdx_main_thread_collector"

#: How long no input event has arrived before a full collection may run.
IDLE_MS = 1000

_E = QtCore.QEvent.Type
#: The events of a hand on the mouse, the keyboard, the trackpad or a tablet.
_INPUT = frozenset({
    _E.MouseButtonPress, _E.MouseButtonRelease, _E.MouseButtonDblClick, _E.MouseMove, _E.Wheel,
    _E.KeyPress, _E.KeyRelease, _E.TouchBegin, _E.TouchUpdate, _E.TouchEnd, _E.NativeGesture,
    _E.TabletPress, _E.TabletMove, _E.TabletRelease})


class MainThreadCollector(QtCore.QObject):
    """Collects on a GUI-thread timer while automatic collection is off; an application event
    filter restarts the idle clock on every input event."""

    def __init__(self, parent=None, interval_ms: int = 500):
        super().__init__(parent)
        self._was_enabled = gc.isenabled()
        self._thresholds = gc.get_threshold()
        gc.disable()
        self._since_input = QtCore.QElapsedTimer()
        self._since_input.start()
        QtCore.QCoreApplication.instance().installEventFilter(self)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.check)
        self._timer.start(int(interval_ms))

    def eventFilter(self, watched, event) -> bool:
        if event.type() in _INPUT:
            self._since_input.restart()
        return False

    def check(self) -> None:
        """Collect the young generation if anything was allocated, an older one when its count
        passed its threshold, the oldest only once no input has arrived for :data:`IDLE_MS`."""
        c0, c1, c2 = gc.get_count()
        _t0, t1, t2 = self._thresholds
        if c0 > 0:
            full = c2 > t2 and self._since_input.elapsed() >= IDLE_MS
            gc.collect(2 if full else 1 if c1 > t1 else 0)

    def stop(self) -> None:
        self._timer.stop()
        QtCore.QCoreApplication.instance().removeEventFilter(self)
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


def weak_method(fn):
    """``fn`` for a child to keep: a bound method is held weakly, so a widget that stores its
    owner's method makes no reference cycle with the owner; the call does nothing once the owner
    is gone. Any other callable is returned as it is (nothing else would keep a lambda alive)."""
    if not isinstance(fn, types.MethodType):
        return fn
    ref = weakref.WeakMethod(fn)

    def call(*args, **kwargs):
        method = ref()
        return None if method is None else method(*args, **kwargs)

    return call
