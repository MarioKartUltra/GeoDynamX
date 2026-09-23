# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Dataset removal + devloop-rebuild adoption (2026-09-19 smoke pass).

Offscreen Qt: the header-row "Remove dataset" cascade (the header itself had NO removal
path -- only layer rows did) and load_field's adoption of an already-populated source (a
rebuilt devloop window used to add a DUPLICATE master and strand every existing row)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.chain import Chain, DeviceRef


def _field(n=48):
    return RasterField(name="f", values=np.random.default_rng(0).normal(size=(n, n)),
                       frame=LocalFrame(), x_axis=np.arange(n, dtype=np.float64),
                       y_axis=np.arange(n, dtype=np.float64))


@pytest.fixture
def window(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _open_with_child(w, f, qtbot=None):
    # Let any worker-dispatched resolve land before the test proceeds (the test_shell_window
    # idiom) -- a live ResolveWorker at teardown takes the NEXT file's teardown down with it.
    if qtbot is not None:
        with qtbot.waitSignal(w.resolved, timeout=5000, raising=False):
            w.load_field(f, "/tmp/fake_boem.tif")
    else:
        w.load_field(f, "/tmp/fake_boem.tif")
    child = w.project.add_layer(
        "f · mz", w.layer.source_id,
        Chain((DeviceRef("mz_edges", {"n_levels": 2}),)).materialized(),
        parent_id=w.layer.layer_id)
    w.add_layer_row(child, f)
    return child


def test_rebuild_adopts_the_restored_family_without_duplicating_the_master(
        qtbot, window):
    from dynamix.shell.main_window import MainWindow

    f = _field()
    _open_with_child(window, f, qtbot)
    assert len(window.project.layers) == 2

    rebuilt = MainWindow()
    qtbot.addWidget(rebuilt)
    rebuilt.project = window.project          # the devloop SESSION restore
    with qtbot.waitSignal(rebuilt.resolved, timeout=5000, raising=False):
        rebuilt.load_field(f, "/tmp/fake_boem.tif")
    assert len(rebuilt.project.layers) == 2   # adopted, never duplicated
    assert rebuilt.layer.parent_id is None
    assert sorted(rebuilt._layer_by_id) == [l.layer_id for l in rebuilt.project.layers]


def test_reopen_in_a_live_window_still_adds_a_master(qtbot, window):
    """The @ov overview flow depends on this: same path, LIVE window with rows -> a second
    root under the same header, exactly as before the adoption fix."""
    f = _field()
    for _ in range(2):
        with qtbot.waitSignal(window.resolved, timeout=5000, raising=False):
            window.load_field(f, "/tmp/fake_boem.tif")
    roots = [l for l in window.project.layers if l.parent_id is None]
    assert len(roots) == 2


def test_remove_dataset_cascades_the_family_and_the_header(qtbot, window, monkeypatch):
    f = _field()
    _open_with_child(window, f, qtbot)
    sid = window.layer.source_id
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    window._on_remove_source_requested(sid)
    assert window.project.layers == []
    assert window.layer_list.topLevelItemCount() == 0
    assert window.layer is None               # _reset_to_empty ran


def test_remove_dataset_confirmation_no_keeps_everything(qtbot, window, monkeypatch):
    f = _field()
    _open_with_child(window, f, qtbot)
    sid = window.layer.source_id
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
    window._on_remove_source_requested(sid)
    assert len(window.project.layers) == 2


def test_locked_layer_refuses_the_dataset(qtbot, window, monkeypatch):
    f = _field()
    child = _open_with_child(window, f, qtbot)
    child.tags["ui.lock"] = "1"
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    window._on_remove_source_requested(window.layer.source_id)
    assert len(window.project.layers) == 2    # refused, nothing removed


def test_removals_resync_the_arrangement(qtbot, window, monkeypatch):
    """The 3-D view must learn about removals (2026-09-20: it kept framing a removed
    dataset's extent): per-layer removal resyncs (camera preserved); dataset removal
    resyncs AND refits the camera -- the subject changed."""
    f = _field()
    child = _open_with_child(window, f, qtbot)
    calls = []

    class _Arr:
        def reset_camera(self):
            calls.append("reset")

    window._arrangement = _Arr()
    monkeypatch.setattr(window, "_sync_arrangement", lambda *a, **k: calls.append("sync"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    window._on_remove_requested(child.layer_id)
    assert calls == ["sync"]                   # layer removal: resync only, camera kept
    calls.clear()
    window._on_remove_source_requested(window.layer.source_id)
    assert calls == ["sync", "reset"]          # dataset removal: resync + refit


def test_surface_apply_all_fans_the_tags_across_the_family(qtbot, window, monkeypatch):
    """The 3-D surface dialog's "apply to every layer of this dataset" (2026-09-20): one
    gesture writes the same surface tags on every non-point sibling, so the family stands
    on the same base instead of per-layer unsynchronized configs."""
    from dynamix.shell import surface_dialog

    f = _field()
    child = _open_with_child(window, f, qtbot)
    master = window.layer if window.layer.parent_id is None else None
    window.layer = window._layer_by_id[child.layer_id]

    class _FakeDialog:
        def __init__(self, **kw):
            pass
        def exec(self):
            from PySide6.QtWidgets import QDialog
            return QDialog.Accepted
        def selection(self):
            return True, "0", True          # z from layer 0 (the master), apply to all

    monkeypatch.setattr(surface_dialog, "SurfaceDialog", _FakeDialog)
    monkeypatch.setattr("dynamix.shell.main_window.SurfaceDialog", _FakeDialog)
    window._on_surface_dialog_requested()
    for l in window.project.layers:
        assert l.tags.get("ui.surface") == "True", l.name
        assert l.tags.get("ui.surface_source") == "0", l.name
