# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.view_dialog.ViewDialog.

Pure Qt: no pyvista import anywhere in this file. ``ViewDialog`` holds a real
``ArrangementView`` reference in the live app, but every call it makes on that reference is
already a guarded passthrough (see ``arrangement/view.py``'s own module docstring) -- so these
tests drive it against a lightweight ``_StubView`` that only records calls, the same "stub the
collaborator, unit-test the wiring" idiom ``tests/test_arrangement_mask.py`` already uses for
``MaskRow``. Settings isolation (``DYNAMIX_SETTINGS_PATH``) comes from ``tests/conftest.py``'s
autouse ``_dynamix_settings_isolated`` fixture -- no per-test setup needed here.
"""
from __future__ import annotations

from PySide6 import QtWidgets

from dynamix.shell.settings import load_settings, update_settings
from dynamix.shell.view_dialog import (DEFAULT_BACKGROUND, FRAME_LABELS, ViewDialog,
                                        normalized_view_options)


class _StubView:
    """Records every call a real ``ArrangementView`` passthrough would receive. ``camera``
    (a dict or ``None``) is what :meth:`camera_state` returns -- ``None`` mirrors the real,
    headless (no interactor built) contract this whole suite runs under.

    ``default_scale_space_stretch``/``default_scale_space_max_grid_dim`` return fixed, injectable values (``stretch_default``/``max_grid_dim``)
    rather than recording a call -- the real ``ArrangementView`` methods they stand in for are
    pure reads with no side effect of their own to record; only ``set_scale_space`` (a real
    push) appends to ``self.calls``, mirroring every other passthrough here."""

    def __init__(self, camera=None, *, stretch_default=5.0, max_grid_dim=100.0):
        self.calls: list[tuple] = []
        self._camera = camera
        self._stretch_default = stretch_default
        self._max_grid_dim = max_grid_dim

    def set_mode(self, mode):
        self.calls.append(("set_mode", mode))

    def set_graticule(self, enabled):
        self.calls.append(("set_graticule", enabled))

    def set_vertical_exaggeration(self, factor):
        self.calls.append(("set_vertical_exaggeration", factor))

    def set_background(self, color):
        self.calls.append(("set_background", color))

    def set_scale_space(self, enabled, stretch):
        self.calls.append(("set_scale_space", enabled, stretch))

    def default_scale_space_stretch(self):
        return self._stretch_default

    def default_scale_space_max_grid_dim(self):
        return self._max_grid_dim

    def camera_state(self):
        return self._camera

    def set_camera_state(self, azimuth, elevation, zoom):
        self.calls.append(("set_camera_state", azimuth, elevation, zoom))

    def reset_camera(self):
        self.calls.append(("reset_camera",))


# ------------------------------------------------------------------------------ normalized_view_options


def test_normalized_view_options_defaults_a_missing_or_non_dict_payload():
    assert normalized_view_options(None) == {
        "mode": "mercator", "graticule": False, "vexag": 1.0, "background": DEFAULT_BACKGROUND,
        "scale_space": False, "scale_space_stretch": 0.0,
    }
    assert normalized_view_options(7) == normalized_view_options(None)
    assert normalized_view_options({}) == normalized_view_options(None)


def test_normalized_view_options_falls_back_per_key_on_garbage():
    raw = {"mode": "not-a-mode", "graticule": "nope", "vexag": -3.0, "background": 123,
           "scale_space": "nope", "scale_space_stretch": -5.0}
    out = normalized_view_options(raw)
    assert out == {
        "mode": "mercator", "graticule": False, "vexag": 1.0, "background": DEFAULT_BACKGROUND,
        "scale_space": False, "scale_space_stretch": 0.0,
    }


def test_normalized_view_options_keeps_legal_values():
    raw = {"mode": "globe", "graticule": True, "vexag": 3.5, "background": "#204060",
           "scale_space": True, "scale_space_stretch": 12.5}
    assert normalized_view_options(raw) == raw


# ------------------------------------------------------------------------------------- construction


def test_dialog_builds_headless_with_four_tabs(qtbot):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    assert dialog.isModal() is False
    tabs = dialog.findChild(QtWidgets.QTabWidget)
    assert tabs is not None
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Projection", "Camera", "Frame", "Display",
    ]


def test_mode_row_lives_in_the_projection_tab(qtbot):
    from dynamix.shell.arrangement.mode_row import ModeRow

    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    tabs = dialog.findChild(QtWidgets.QTabWidget)
    assert isinstance(dialog._mode_row, ModeRow)
    assert tabs.widget(0).findChild(ModeRow) is dialog._mode_row


# ------------------------------------------------------------------------------------- Projection


def test_mode_row_click_reaches_view_set_mode_and_persists(qtbot):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    dialog._mode_row._buttons["globe"].click()

    assert ("set_mode", "globe") in view.calls
    assert load_settings().view_options["mode"] == "globe"


def test_mode_row_updates_the_frame_tab_readout(qtbot):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    dialog._mode_row._buttons["globe"].click()

    assert dialog._frame_value.text() == FRAME_LABELS["globe"]


def test_restoring_a_non_default_stored_mode_does_not_reach_view_or_settings(qtbot):
    """The dedup guard (module docstring, "mode path must not double-apply"): a dialog
    constructed with a NON-default persisted mode has to change ModeRow's own checked button on
    construction -- but that restore must reach ONLY ModeRow's internal state, never
    ``view.set_mode``/``update_settings`` (``MainWindow``'s own post-``activate()`` push is the
    one place the stored mode actually reaches the view -- see ``main_window.py``)."""
    update_settings(view_options={"mode": "globe", "graticule": False, "vexag": 1.0,
                                   "background": DEFAULT_BACKGROUND})
    before = load_settings().view_options

    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    assert dialog._mode_row.mode == "globe"       # restored...
    assert view.calls == []                       # ...but silently -- no view call at all
    assert load_settings().view_options == before  # ...and no redundant settings write either


# ------------------------------------------------------------------------------------------ Camera


def test_camera_readout_pulls_on_construction():
    view = _StubView(camera={"azimuth": 12.0, "elevation": -5.0, "zoom": 2.0})
    dialog = ViewDialog(view)

    assert dialog._camera_values["azimuth"] == 12.0
    assert dialog._camera_values["elevation"] == -5.0
    assert dialog._camera_values["zoom"] == 2.0


def test_camera_readout_pulls_again_on_show(qtbot):
    view = _StubView(camera=None)
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    assert dialog._camera_values["azimuth"] == 0.0     # headless at construction -- Param default

    view._camera = {"azimuth": 30.0, "elevation": 10.0, "zoom": 1.5}
    dialog.show()

    assert dialog._camera_values["azimuth"] == 30.0
    assert dialog._camera_values["elevation"] == 10.0
    assert dialog._camera_values["zoom"] == 1.5


def test_camera_control_edit_pushes_all_three_via_set_camera_state(qtbot):
    view = _StubView(camera={"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0})
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    view.calls.clear()

    dialog._camera_controls["azimuth"].valueChanged.emit(45.0)

    assert view.calls == [("set_camera_state", 45.0, 0.0, 1.0)]
    assert dialog._camera_values["azimuth"] == 45.0


def test_reset_button_calls_reset_camera_and_refreshes_readout(qtbot):
    view = _StubView(camera={"azimuth": 99.0, "elevation": 99.0, "zoom": 9.0})
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    view.calls.clear()
    view._camera = {"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0}   # post-reset state

    dialog._on_reset_clicked()

    assert view.calls == [("reset_camera",)]
    assert dialog._camera_values == {"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0}


def test_camera_state_is_never_persisted(qtbot):
    view = _StubView(camera={"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0})
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    dialog._camera_controls["zoom"].valueChanged.emit(2.5)

    stored = load_settings().view_options
    assert "azimuth" not in stored and "elevation" not in stored and "zoom" not in stored


# ------------------------------------------------------------------------------------------- Frame


def test_graticule_toggle_calls_view_and_persists(qtbot):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    dialog._graticule_checkbox.setChecked(True)

    assert ("set_graticule", True) in view.calls
    assert load_settings().view_options["graticule"] is True


# ----------------------------------------------------------------------------------------- Display


def test_vexag_change_calls_view_and_persists(qtbot):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    dialog._vexag_control.valueChanged.emit(2.5)

    assert ("set_vertical_exaggeration", 2.5) in view.calls
    assert load_settings().view_options["vexag"] == 2.5


# --------------------------------------------------- Scale-space


def test_scale_space_checkbox_toggle_computes_a_default_stretch_and_persists(qtbot):
    """The design's own "default stretch when enabling with none set": checking the box with the
    persisted ``scale_space_stretch`` still at its 0.0 sentinel pulls
    ``view.default_scale_space_stretch()`` (here, the stub's own fixed 5.0) as the new value,
    rebinds the control's soft range via ``view.default_scale_space_max_grid_dim()`` (the stub's
    100.0), and pushes ``set_scale_space(True, 5.0)`` -- not ``set_scale_space(True, 0.0)``."""
    view = _StubView(stretch_default=5.0, max_grid_dim=100.0)
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    assert dialog._opts["scale_space_stretch"] == 0.0     # the unset sentinel, pre-toggle

    dialog._scale_space_checkbox.setChecked(True)

    assert ("set_scale_space", True, 5.0) in view.calls
    assert load_settings().view_options["scale_space"] is True
    assert load_settings().view_options["scale_space_stretch"] == 5.0
    assert dialog._scale_space_stretch_control._value == 5.0


def test_scale_space_checkbox_toggle_off_does_not_recompute_stretch(qtbot):
    view = _StubView(stretch_default=5.0)
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    dialog._scale_space_checkbox.setChecked(True)     # -> stretch becomes 5.0 (the default)
    view.calls.clear()

    dialog._scale_space_checkbox.setChecked(False)

    assert ("set_scale_space", False, 5.0) in view.calls
    assert load_settings().view_options["scale_space"] is False
    assert load_settings().view_options["scale_space_stretch"] == 5.0    # untouched, not reset


def test_scale_space_checkbox_toggle_on_with_a_real_stretch_already_chosen_keeps_it(qtbot):
    """A stretch the user already dialed in (or restored from settings) is never silently
    overwritten by the computed default -- only the 0.0 sentinel triggers that."""
    view = _StubView(stretch_default=5.0)
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    dialog._scale_space_stretch_control.valueChanged.emit(42.0)   # a real, user-chosen value
    view.calls.clear()

    dialog._scale_space_checkbox.setChecked(True)

    assert ("set_scale_space", True, 42.0) in view.calls


def test_scale_space_stretch_change_calls_view_and_persists(qtbot):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)

    dialog._scale_space_stretch_control.valueChanged.emit(7.5)

    assert ("set_scale_space", False, 7.5) in view.calls   # checkbox still off -- pushed as-is
    assert load_settings().view_options["scale_space_stretch"] == 7.5


def test_background_button_uses_the_pick_color_seam_never_opens_a_real_dialog(qtbot, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("QColorDialog.getColor must never be called in a test")
    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", _boom)

    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog, "_pick_color", lambda initial: "#204060")

    dialog._background_button.click()

    assert ("set_background", "#204060") in view.calls
    assert load_settings().view_options["background"] == "#204060"


def test_background_cancel_leaves_state_untouched(qtbot, monkeypatch):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    before = dict(dialog._opts)
    monkeypatch.setattr(dialog, "_pick_color", lambda initial: None)

    dialog._background_button.click()

    assert view.calls == []
    assert dialog._opts == before


# ------------------------------------------------------------------------------------ Frame mode
# ViewDialog.set_frame_mode hides ONLY the Projection
# tab -- the Camera (reset/az/el/zoom) and Display (vertical exaggeration/background) tabs stay
# reachable, since both are load-bearing in frame mode too (spec's own words).


def test_set_frame_mode_true_hides_only_the_projection_tab(qtbot):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    tabs = dialog.findChild(QtWidgets.QTabWidget)
    assert all(tabs.isTabVisible(i) for i in range(tabs.count()))   # nothing hidden yet

    dialog.set_frame_mode(True)

    assert tabs.isTabVisible(0) is False       # Projection -- the one meaningless control
    assert tabs.isTabVisible(1) is True        # Camera -- reset/az/el/zoom stay reachable
    assert tabs.isTabVisible(2) is True        # Frame
    assert tabs.isTabVisible(3) is True        # Display -- vertical exaggeration stays reachable


def test_set_frame_mode_false_restores_the_projection_tab(qtbot):
    view = _StubView()
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    tabs = dialog.findChild(QtWidgets.QTabWidget)
    dialog.set_frame_mode(True)

    dialog.set_frame_mode(False)

    assert tabs.isTabVisible(0) is True


def test_frame_mode_camera_and_vexag_controls_stay_fully_functional(qtbot):
    """Not just VISIBLE -- still WIRED: a Camera/Display edit in frame mode reaches ``view``
    exactly as it would in geo mode ("camera reset and vertical exaggeration must
    be reachable there")."""
    view = _StubView(camera={"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0})
    dialog = ViewDialog(view)
    qtbot.addWidget(dialog)
    dialog.set_frame_mode(True)
    view.calls.clear()

    dialog._on_reset_clicked()
    dialog._vexag_control.valueChanged.emit(3.0)
    dialog._scale_space_stretch_control.valueChanged.emit(6.0)

    assert ("reset_camera",) in view.calls
    assert ("set_vertical_exaggeration", 3.0) in view.calls
    assert ("set_scale_space", False, 6.0) in view.calls


# ------------------------------------------------------------------------ persist + re-apply (fresh dialog)


def test_vexag_and_background_persist_and_reapply_on_a_fresh_dialog(qtbot):
    first = ViewDialog(_StubView())
    qtbot.addWidget(first)
    first._vexag_control.valueChanged.emit(4.0)
    monkeypatched_color = "#336699"
    first._opts["background"] = monkeypatched_color
    first._update_background_swatch()
    update_settings(view_options=dict(first._opts))

    second_view = _StubView()
    second = ViewDialog(second_view)
    qtbot.addWidget(second)

    assert second._opts["vexag"] == 4.0
    assert second._opts["background"] == monkeypatched_color
    assert second._vexag_control._value == 4.0


def test_scale_space_persists_and_reapplies_on_a_fresh_dialog(qtbot):
    first = ViewDialog(_StubView(stretch_default=9.0))
    qtbot.addWidget(first)
    first._scale_space_checkbox.setChecked(True)      # -> stretch becomes 9.0 (the default)

    second = ViewDialog(_StubView())
    qtbot.addWidget(second)

    assert second._opts["scale_space"] is True
    assert second._opts["scale_space_stretch"] == 9.0
    assert second._scale_space_checkbox.isChecked() is True
    assert second._scale_space_stretch_control._value == 9.0
