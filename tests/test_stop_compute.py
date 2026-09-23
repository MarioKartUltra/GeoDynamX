# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The user stop control.

v1's "no cancellation" had one exception already built: the active-layer preempt sets the
worker's cancel flag and lands in _on_cancelled, which redispatches the pending recipe.
This slice adds the USER stop on top of the same machinery: a status-bar button visible
while a worker runs; stopping cancels the in-flight run, restores the transport/strips
instead of redispatching, and SUPPRESSES the auto-redispatch (_dispatch_next's cache probe
would otherwise instantly restart the very compute the user stopped -- the tail is
uncached) until the chain signature changes or a new dispatch happens.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from tests.test_center_views import (STUB_CHAIN, _bare_field,   # noqa: F401
                                     stub_devices)              # noqa: F401  (harness reuse)


class _FakeWorker:
    def __init__(self):
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1


class _FakeThread:
    def quit(self):
        pass

    def wait(self):
        pass


@pytest.fixture
def win(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    w = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.resolved, timeout=10000):
        w.load_field(_bare_field("stop"), "mem:stop")
    return w


def test_stop_compute_cancels_the_worker_and_flags(win):
    fake = _FakeWorker()
    win._worker = fake
    win._stop_compute()
    assert fake.cancelled == 1 and win._user_stopped
    win._worker = None
    win._user_stopped = False
    win._stop_compute()                        # no worker: safe no-op
    assert not win._user_stopped


def test_user_stop_landing_restores_ui_and_suppresses_redispatch(win, monkeypatch):
    calls = []
    monkeypatch.setattr(win, "_start_worker", lambda: calls.append("start"))
    win._worker = _FakeWorker()
    win._thread = _FakeThread()
    win._dispatched = (win.layer.layer_id, win._transform_signature())
    win._active_pending = "compute"            # even a pending preempt: stop outranks it
    win._user_stopped = True
    win._on_cancelled()
    assert calls == []                         # nothing redispatched
    assert win._active_pending is None
    assert win._stopped_sig == win._transform_signature()
    assert win.transport.isEnabled()
    assert not win._user_stopped               # consumed
    assert not win._stop_btn.isVisible()


def test_preempt_landing_still_redispatches(win, monkeypatch):
    calls = []
    monkeypatch.setattr(win, "_start_worker", lambda: calls.append("start"))
    win._worker = _FakeWorker()
    win._thread = _FakeThread()
    win._dispatched = (win.layer.layer_id, win._transform_signature())
    win._active_pending = "compute"
    win._user_stopped = False                  # NOT a user stop: the preempt path
    win._on_cancelled()
    assert calls == ["start"]


def test_dispatch_gate_honors_the_stopped_signature(win, monkeypatch):
    calls = []
    monkeypatch.setattr(win, "_start_worker", lambda: calls.append("start"))
    monkeypatch.setattr(win, "_cache_keys_for", lambda layer: ["missing-key"])
    win._stopped_sig = win._transform_signature()
    win._dispatch_next()
    assert calls == []                         # suppressed
    win._stopped_sig = None
    win._dispatch_next()
    assert calls == ["start"]                  # the gate reopens


def test_stop_button_exists_and_rests_hidden(win):
    assert win._stop_btn is not None
    assert not win._stop_btn.isVisible()
