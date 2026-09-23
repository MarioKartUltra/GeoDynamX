# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The explicit Run gate: transforms never auto-run.

Editing a transform's
params (or a chain edit that changes the transform signature) marks the transform block PENDING
and enables Run; nothing computes until Run (or with Settings.auto_run_wtmm=True, the old
dispatch-on-edit behaviour). While pending, the display keeps the last run; filter edits are
recorded but never trigger a synchronous catch-up."""
from __future__ import annotations

from PySide6 import QtCore

from dynamix.shell.settings import Settings, save_settings
from tests.test_shell_roi_flow import parent_tif, roi_window  # noqa: F401


def _manual(win):
    save_settings(Settings(auto_run_wtmm=False))
    win._auto_run_action.setChecked(False)


def test_transform_edit_marks_pending_and_does_not_dispatch(roi_window):
    win = roi_window
    _manual(win)
    assert win.run_button.isEnabled() is False
    win._on_param_changed(0, "n_oct", 3)                          # wtmm2d: a transform param
    assert win._thread is None                                    # nothing dispatched
    assert win.run_button.isEnabled() is True
    assert win.strips.strip(0).state_dot.property("state") == "pending"


def test_filter_edit_while_pending_never_computes_synchronously(roi_window):
    win = roi_window
    _manual(win)
    win._on_param_changed(0, "n_oct", 3)
    win._on_param_changed(1, "scale_idx", 0)                      # scale_select: a filter
    assert win._thread is None                                    # no catch-up, no GUI-thread run


def test_run_button_dispatches_clears_pending_and_lands(roi_window, qtbot):
    win = roi_window
    _manual(win)
    win._on_param_changed(0, "n_oct", 3)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.run_button.click()
    assert win.run_button.isEnabled() is False
    hx, _ = win.canvas.hchain_item.getData()
    assert hx is not None and len(hx) > 0


def test_chain_edit_that_changes_the_signature_gates_too(roi_window):
    win = roi_window
    _manual(win)
    descriptors = ([{"device": "noise", "params": {"amplitude": 0.05}}]
                   + [{"device": ref.device, "params": dict(ref.params)} for ref in win.layer.chain.steps])
    win.strips.set_steps(descriptors, field=win.field)
    win._on_chain_edited(descriptors)
    assert win._thread is None
    assert win.run_button.isEnabled() is True


def test_auto_run_true_keeps_the_old_dispatch_on_edit(roi_window, qtbot):
    win = roi_window                                              # conftest seeds auto_run True
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_param_changed(0, "n_oct", 3)                      # dispatches immediately


def test_run_with_nothing_pending_always_answers(roi_window):
    win = roi_window                                              # resolved and cached
    _manual(win)
    win._run_transforms()
    assert "already computed" in win.statusBar().currentMessage()
