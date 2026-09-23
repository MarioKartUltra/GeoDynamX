# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for ModeRow -- the arrangement view's projection-mode segmented control ("corner segmented
control (pacific / greenwich / mercator / globe), mercator default").

Offscreen VTK: ``pytest.importorskip("pyvista")`` skips this whole module honestly if pyvista isn't
installed. ``ModeRow`` itself needs no pyvista at all (pure Qt, see its own module docstring) --
imported unconditionally regardless -- but the ``ArrangementView`` wiring section below needs the
``viz`` extra the same way ``tests/test_arrangement_mask.py``'s own wiring section does.
"""
from __future__ import annotations

import pytest

from dynamix.core.projection import MODES
from dynamix.shell.arrangement.mode_row import DEFAULT_MODE, ModeRow

# --------------------------------------------------------------------------------------- ModeRow


def test_mode_row_default_is_mercator_and_matches_the_checked_button(qtbot):
    row = ModeRow()
    qtbot.addWidget(row)
    assert row.mode == DEFAULT_MODE == "mercator"
    assert row._buttons["mercator"].isChecked() is True
    assert sum(b.isChecked() for b in row._buttons.values()) == 1     # exclusive: only one


def test_mode_row_has_one_button_per_mode_in_projection_order():
    row = ModeRow()
    assert list(row._buttons) == list(MODES)


def test_mode_row_click_emits_mode_changed(qtbot):
    row = ModeRow()
    qtbot.addWidget(row)
    with qtbot.waitSignal(row.modeChanged, timeout=1000) as sig:
        row._buttons["globe"].click()
    assert sig.args[0] == "globe"
    assert row.mode == "globe"
    assert row._buttons["globe"].isChecked() is True
    assert row._buttons["mercator"].isChecked() is False


def test_mode_row_exclusive_group_switches_the_checked_button(qtbot):
    row = ModeRow()
    qtbot.addWidget(row)
    row._buttons["pacific"].click()
    assert row._buttons["mercator"].isChecked() is False
    assert row._buttons["pacific"].isChecked() is True

    row._buttons["greenwich"].click()
    assert row._buttons["pacific"].isChecked() is False
    assert row._buttons["greenwich"].isChecked() is True


def test_mode_row_reclicking_the_active_mode_does_not_reemit(qtbot):
    row = ModeRow()
    qtbot.addWidget(row)
    seen = []
    row.modeChanged.connect(seen.append)

    row._buttons["mercator"].click()          # already the active/default mode

    assert seen == []


# ------------------------------------------------------------------------ ArrangementView wiring
#
# ModeRow was retired from this view's own face -- the last of the four
# controls (after MaskRow/GroupPalette/Commit in Tasks 4/5) to leave it (see view.py's own module
# docstring, "Correction" note). The two tests below replace
# `test_arrangement_view_builds_a_mode_row_eagerly`/`test_arrangement_view_wires_mode_row_to_scene_
# lazily`, which asserted against `view._mode_row`, an attribute this view no longer has at all --
# mirroring exactly how tests/test_arrangement_mask.py's own "ArrangementView wiring" section was already updated for MaskRow's identical departure.


def test_arrangement_view_no_longer_builds_a_mode_row(qtbot):
    pytest.importorskip("pyvista", reason="pyvista not installed")
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    assert not hasattr(view, "_mode_row")


def test_arrangement_view_set_mode_forwards_to_scene_and_camera_lazily(qtbot):
    pv = pytest.importorskip("pyvista", reason="pyvista not installed")
    pv.OFF_SCREEN = True
    from dynamix.shell.arrangement.camera import MomentumCamera
    from dynamix.shell.arrangement.scene import Scene
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    # No scene yet (a real QtInteractor cannot be built under this harness's mandated offscreen
    # QPA platform -- see tests/test_arrangement_flip.py's documented segfault guard, so
    # activate() is never exercised here): calling the public set_mode() passthrough must not
    # raise -- the same no-op-before-a-scene-exists contract _on_mode_changed always had, and
    # set_mask's own forwarding tests already pin for the mask.
    view.set_mode("globe")

    view._scene = Scene(pv.Plotter(off_screen=True))

    view.set_mode("pacific")
    assert view._scene.mode == "pacific"

    # Attach a camera too -- set_mode's second half (MomentumCamera.note_mode) is a no-op until
    # one exists, exactly like Scene.set_mode just above.
    view._camera = MomentumCamera(view._scene._plotter, view._scene)
    view.set_mode("globe")
    view.set_mode("greenwich")

    assert view._scene.mode == "greenwich"
    assert view._camera._last_flat_mode == "greenwich"     # note_mode kept this current
