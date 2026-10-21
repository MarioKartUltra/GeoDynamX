# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Finest-scale progressive preview: the worker emits a finest-scale
``partial`` before the full ``finished`` when ``preview=True``, and the window draws it so the
canvas shows the finest lines while the full stack is still computing.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.project import Project
from dynamix.shell.worker import ResolveWorker

FIXTURE = "tests/fixtures/kam_64.npz"


def _layer(project):
    src = project.add_source("kam")
    return project.add_layer("x", src.source_id,
                             Chain((DeviceRef("wtmm2d", {"n_oct": 2, "n_voice": 2}),)))


def test_worker_emits_finest_preview_before_finished(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    field = RasterField.load_npz(FIXTURE)
    worker = ResolveWorker(_layer(Project()), field, Cache(), "kam", preview=True)
    partials = []
    worker.partial.connect(lambda r: partials.append(r))

    with qtbot.waitSignal(worker.finished, timeout=60000) as sig:
        thread = worker.start()

    assert len(partials) == 1
    prev = partials[0]
    assert prev.get("_preview") is True
    assert len(prev["extrema"]) == 1                     # finest scale only
    assert len(sig.args[0].result["extrema"]) == 4       # full stack = n_oct*n_voice scales
    thread.quit(); thread.wait()


def test_no_preview_when_not_requested(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    field = RasterField.load_npz(FIXTURE)
    worker = ResolveWorker(_layer(Project()), field, Cache(), "kam")   # preview defaults off
    partials = []
    worker.partial.connect(lambda r: partials.append(r))
    with qtbot.waitSignal(worker.finished, timeout=60000):
        thread = worker.start()
    assert partials == []
    thread.quit(); thread.wait()


def test_on_preview_draws_the_finest_scale_on_the_canvas(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.resolve import preview_resolve
    from dynamix.shell.main_window import MainWindow, DEMO_CHAIN

    register_builtin_devices()
    win = MainWindow(steps=DEMO_CHAIN)
    qtbot.addWidget(win)
    field = RasterField.load_npz(FIXTURE)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(field, FIXTURE)

    win.canvas.hchain_item.setData([], [])               # clear, then feed a preview
    prev = preview_resolve(win.layer, win.field)
    assert prev is not None
    win._on_preview(prev)

    hx, _ = win.canvas.hchain_item.getData()
    assert hx is not None and len(hx) > 0                 # the finest scale's lines are drawn
    assert "preview" in win.statusBar().currentMessage().lower()


def test_the_preview_notice_goes_when_the_full_stack_lands(qtbot, clean_registry):
    """The notice says the full stack is still computing; once it has landed, it is gone."""
    from dynamix.devices import register_builtin_devices
    from dynamix.shell.main_window import MainWindow, DEMO_CHAIN

    register_builtin_devices()
    win = MainWindow(steps=DEMO_CHAIN)
    qtbot.addWidget(win)
    shown = []
    win.statusBar().messageChanged.connect(shown.append)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(RasterField.load_npz(FIXTURE), FIXTURE)
    qtbot.waitUntil(lambda: win._thread is None, timeout=60000)
    assert any("preview" in m.lower() for m in shown)            # the preview did show
    assert "preview" not in win.statusBar().currentMessage().lower()


def test_a_layer_that_last_ran_fast_gets_no_preview(qtbot, clean_registry, monkeypatch):
    """The finest-scale preview is its own run before the full resolve (0.7 s of a 3.6 s knob
    change on a 1024² field). A layer with no history gets it; one whose last full compute landed
    under ``PREVIEW_AFTER_MS`` gets none, since the full result arrives before the preview would
    have been worth a look, a three-second knob change included; one that ran long gets it back."""
    from dynamix.devices import register_builtin_devices
    from dynamix.shell import main_window as mw
    from dynamix.shell.main_window import DEMO_CHAIN, PREVIEW_AFTER_MS, MainWindow

    register_builtin_devices()
    previews = []

    class Recording(mw.ResolveWorker):
        def __init__(self, *args, preview=False, **kwargs):
            previews.append(preview)
            super().__init__(*args, preview=preview, **kwargs)

    monkeypatch.setattr(mw, "ResolveWorker", Recording)
    win = MainWindow(steps=DEMO_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(RasterField.load_npz(FIXTURE), FIXTURE)
    qtbot.waitUntil(lambda: win._thread is None, timeout=60000)
    assert previews == [True]                                  # no history yet
    assert win.layer.layer_id in win._last_run_ms              # the landing recorded its time
    win._last_run_ms[win.layer.layer_id] = PREVIEW_AFTER_MS / 2
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._start_worker()
    qtbot.waitUntil(lambda: win._thread is None, timeout=60000)
    assert previews[-1] is False
    win._last_run_ms[win.layer.layer_id] = 3000.0              # the preview would add a fifth to it
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._start_worker()
    qtbot.waitUntil(lambda: win._thread is None, timeout=60000)
    assert previews[-1] is False
    win._last_run_ms[win.layer.layer_id] = PREVIEW_AFTER_MS * 2
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._start_worker()
    qtbot.waitUntil(lambda: win._thread is None, timeout=60000)
    assert previews[-1] is True
