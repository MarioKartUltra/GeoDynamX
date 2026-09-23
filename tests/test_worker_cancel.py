# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ResolveWorker cancellation (progressive compute, design 2026-09-14 §3).

A cancel requested before/while the resolve runs must land on the ``cancelled`` signal, never
``finished`` or ``error`` -- a superseded compute is not a failure. Exercised on a real QThread
via qtbot, on the in-repo EBSD fixture.
"""
from __future__ import annotations

import pytest

from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.project import Project
from dynamix.shell.worker import ResolveWorker

FIXTURE = "tests/fixtures/kam_64.npz"


def _layer(project):
    src = project.add_source("kam")
    return project.add_layer(
        "x", src.source_id,
        Chain((DeviceRef("wtmm2d", {"n_oct": 2, "n_voice": 2, "a_min": 1.0}),)))


def test_cancel_before_start_emits_cancelled_not_finished(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    field = RasterField.load_npz(FIXTURE)
    p = Project()
    worker = ResolveWorker(_layer(p), field, Cache(), "kam")

    finished = []
    worker.finished.connect(lambda r: finished.append(r))

    worker.cancel()                              # flag set before the run even begins
    with qtbot.waitSignal(worker.cancelled, timeout=30000):
        thread = worker.start()

    assert finished == []                        # never produced a result
    thread.quit(); thread.wait()


def test_uncancelled_worker_finishes_normally(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    field = RasterField.load_npz(FIXTURE)
    p = Project()
    worker = ResolveWorker(_layer(p), field, Cache(), "kam")

    cancelled = []
    worker.cancelled.connect(lambda: cancelled.append(True))

    with qtbot.waitSignal(worker.finished, timeout=60000) as sig:
        thread = worker.start()

    assert cancelled == []
    assert len(sig.args[0].result["extrema"]) == 4       # n_oct*n_voice scales
    thread.quit(); thread.wait()
