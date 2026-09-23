# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Filter edits must stay live while a transform edit is pending.

The design: "While pending, the display keeps showing the LAST RUN's result; filter edits stay
live against it (the hybrid: last-run transform tail + live filter steps)." The bug: ``_reresolve``
returns early when the layer is in ``_pending_layers``, so a filter edit made after an unrun
transform tweak is silently DROPPED until Run -- both reported symptoms at once.

Driven through the real window offscreen (the only honest reproduction of the dispatch path).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.rasterfield import RasterField

FIXTURE = "tests/fixtures/kam_64.npz"


def _chains(win):
    return len((win._active_result or {}).get("chains") or [])


@pytest.fixture
def running_window(qtbot, clean_registry):
    """A window on the fixture with [wtmm2d, chain_holder] computed once. Auto-run is ON via the
    suite-wide conftest default, so ``load_field`` dispatches and emits; we then turn it OFF so a
    transform edit goes PENDING -- the §5e gate under test."""
    from dynamix.devices import register_builtin_devices
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    win = MainWindow(steps=(("wtmm2d", {"n_oct": 3, "n_voice": 4}),
                            ("chain_holder", {"cutoff": -3.0, "estimator": "ols"})))
    qtbot.addWidget(win)
    field = RasterField.load_npz(FIXTURE)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(field, FIXTURE)               # auto-run ON -> computes -> emits
    win._auto_run_action.setChecked(False)           # now edits go through the Run gate
    assert win.layer.layer_id not in win._pending_layers
    return win, qtbot


def test_chain_filter_is_live_after_run(running_window):
    """Baseline: with nothing pending, tightening chain_holder applies immediately, no Run."""
    win, qtbot = running_window
    hi = win._index_of("chain_holder")
    before = _chains(win)
    win._on_param_changed(hi, "cutoff", 0.0)         # raise the floor -> fewer chains
    assert win.is_computing is False
    assert _chains(win) < before, (before, _chains(win))


def test_chain_filter_stays_live_while_a_transform_edit_is_pending(running_window):
    """§5e point 1: a pending aₘᵢₙ must NOT freeze filters. Editing chain_holder while the
    transform block is pending still re-resolves against the LAST RUN's cached tail -- it must
    change the drawn chains without a Run and without a synchronous transform recompute."""
    win, qtbot = running_window
    ai = win._index_of("wtmm2d")
    # nudge a transform param -> block goes pending (auto-run off)
    win._on_param_changed(ai, "a_min", 1.5)
    assert win.layer.layer_id in win._pending_layers   # precondition: pending

    hi = win._index_of("chain_holder")
    before = _chains(win)
    win._on_param_changed(hi, "cutoff", 0.0)           # tighten the chain filter

    assert win.is_computing is False                   # no synchronous 22 s recompute
    assert _chains(win) < before, (
        f"filter edit was dropped while pending: {before} -> {_chains(win)}")
