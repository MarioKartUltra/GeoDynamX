# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Finest-scale progressive preview (design 2026-09-14 §3): the worker emits a finest-scale
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
