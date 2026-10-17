# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Garbage collection on the GUI thread only: Python's cyclic collector runs in whichever thread
trips its threshold, a compute worker included, and a Qt object it frees there is destroyed off the
thread that owns it (CI crashed 'Garbage-collecting' inside a worker's scipy import)."""
from __future__ import annotations

import gc
import threading

import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402


class _Cycle:
    def __init__(self):
        self.me = self


def test_the_collector_turns_automatic_collection_off_and_collects_on_its_timer(qtbot):
    from dynamix.shell import gc_guard

    app = QtWidgets.QApplication.instance()
    was = gc.isenabled()
    collector = gc_guard.install(app, interval_ms=10)
    try:
        assert gc.isenabled() is False
        assert gc_guard.install(app) is collector                  # once per application
        seen = []
        gc.callbacks.append(lambda phase, info: seen.append(threading.current_thread()))
        for _ in range(3 * gc.get_threshold()[0]):
            _Cycle()
        qtbot.waitUntil(lambda: bool(seen), timeout=2000)
        assert all(t is threading.main_thread() for t in seen)
    finally:
        gc.callbacks.pop()
        gc_guard.uninstall(app)
    assert gc.isenabled() is was


def test_a_worker_thread_never_collects_while_the_collector_runs(qtbot):
    from dynamix.shell import gc_guard

    app = QtWidgets.QApplication.instance()
    gc_guard.install(app, interval_ms=1000)
    try:
        seen = []
        gc.callbacks.append(lambda phase, info: seen.append(threading.current_thread()))
        t = threading.Thread(target=lambda: [_Cycle() for _ in range(5 * gc.get_threshold()[0])])
        t.start()
        t.join()
        assert not [x for x in seen if x is not threading.main_thread()]
    finally:
        gc.callbacks.pop()
        gc_guard.uninstall(app)


def test_a_devloop_reload_keeps_the_one_collector(qtbot):
    """The devloop reloads ``gc_guard`` with the other shell modules; the collector lives on the
    application, so the reloaded module's ``install`` finds it and starts no second timer."""
    import importlib

    from PySide6 import QtCore

    from dynamix import devloop
    from dynamix.shell import gc_guard

    app = QtWidgets.QApplication.instance()
    collector = gc_guard.install(app, interval_ms=1000)
    try:
        devloop._reload(["dynamix.shell.gc_guard"])
        reloaded = importlib.import_module("dynamix.shell.gc_guard")
        assert reloaded is gc_guard and reloaded.install(app) is collector
        assert gc.isenabled() is False
        collectors = [o for o in app.findChildren(QtCore.QObject)
                      if type(o).__name__ == "MainThreadCollector"]
        assert collectors == [collector]
        assert [t.isActive() for t in collector.findChildren(QtCore.QTimer)] == [True]
    finally:
        importlib.import_module("dynamix.shell.gc_guard").uninstall(app)


def test_weak_method_holds_a_bound_method_weakly_and_a_function_as_it_is():
    import weakref

    from dynamix.shell import gc_guard

    class Owner:
        def label(self, x):
            return f"#{x}"

    owner = Owner()
    call = gc_guard.weak_method(owner.label)
    assert call(3) == "#3"
    ref = weakref.ref(owner)
    del owner
    assert ref() is None and call(3) is None             # the owner went; the call does nothing
    plain = lambda x: x                                   # noqa: E731
    assert gc_guard.weak_method(plain) is plain           # nothing else would keep it alive
