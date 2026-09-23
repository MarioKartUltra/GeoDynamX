# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the three-view cycle (Raster/Vector/Globe), the
visible switcher, and frame-mode layer admission.

Offscreen Qt (``QT_QPA_PLATFORM=offscreen``, mandated repo-wide). A REAL ``ArrangementView`` is
offscreen-safe on its own (``activate()`` detects the offscreen QPA platform and never builds a
live ``pyvistaqt.QtInteractor`` -- ``tests/test_arrangement_resolve.py`` already exercises it this
way), but this file follows the design's instruction instead: inject a small, plain RECORDER
object as ``window._arrangement`` *before* the first view switch, so ``_set_center_view``'s
lazy-build branch (``if self._arrangement is None: ...``) never runs at all -- the assertions
below are about WHICH entries/calls ``MainWindow`` hands the arrangement and WHEN, not about
``ArrangementView``/``Scene`` themselves (covered elsewhere).

Fixtures (``window``, ``loaded``, ``STUB_CHAIN``, ``_FIELD``, ``stub_devices``) come from
``tests/test_shell_window.py``, same convention every other ``tests/test_arrangement_*.py`` file
already follows.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.chain import Chain, DeviceRef
from dynamix.shell.settings import load_settings, update_settings

from tests.test_shell_window import STUB_CHAIN, _FIELD, loaded, stub_devices, window  # noqa: F401


class _FakeArrangement:
    """A simple recorder standing in for ``ArrangementView`` -- exactly the methods
    ``MainWindow`` calls on it across ``_set_center_view``/``_sync_arrangement``. Injected
    directly as ``window._arrangement`` so the real lazy-build branch (which would construct a
    genuine ``ArrangementView`` and touch ``dynamix.shell.arrangement_facade``) never runs.

    ``set_selection_mode`` (the design's last bullet) joined this list
    when ``_set_center_view`` started pushing the current selection mode on every flip-in,
    mirroring the pre-existing ``set_mask`` push just above it in ``main_window.py`` -- see
    ``ArrangementView.set_selection_mode``'s own docstring for the full contract this recorder
    stands in for.

    ``set_scale_space`` joined this list the same way,
    pushed alongside ``set_vertical_exaggeration`` in the SAME gated ("geo" only) block -- see
    ``main_window._set_center_view``'s own docstring/comment for why."""

    def __init__(self):
        self.set_layers_calls: list[list[dict]] = []
        self.colormap_calls: list[str] = []
        self.frame_mode_calls: list[bool] = []
        self.activate_calls = 0
        self.deactivate_calls = 0
        self.mask_calls: list[dict] = []
        self.mode_calls: list[str] = []
        self.graticule_calls: list[bool] = []
        self.vexag_calls: list[float] = []
        self.background_calls: list[str] = []
        self.selection_mode_calls: list[str] = []
        self.scale_space_calls: list[tuple] = []

    def set_layers(self, entries) -> None:
        self.set_layers_calls.append(list(entries))

    def set_frame_mode(self, enabled: bool) -> None:
        self.frame_mode_calls.append(bool(enabled))

    def activate(self) -> None:
        self.activate_calls += 1

    def deactivate(self) -> None:
        self.deactivate_calls += 1

    def set_mask(self, payload: dict) -> None:
        self.mask_calls.append(dict(payload))

    def set_mode(self, mode: str) -> None:
        self.mode_calls.append(mode)

    def set_graticule(self, enabled: bool) -> None:
        self.graticule_calls.append(bool(enabled))

    def set_vertical_exaggeration(self, vexag: float) -> None:
        self.vexag_calls.append(vexag)

    def set_background(self, color: str) -> None:
        self.background_calls.append(color)

    def set_selection_mode(self, mode: str) -> None:
        self.selection_mode_calls.append(mode)

    def set_scale_space(self, enabled: bool, stretch: float) -> None:
        self.scale_space_calls.append((bool(enabled), stretch))

    # 2026-08-29: the window now pushes footprints/previews/reference layers into whatever
    # arrangement it holds; a double that stands in for it accepts them silently.
    def set_footprints(self, footprints):
        self.footprints = list(footprints)

    def set_previews(self, previews):
        self.previews = list(previews)

    def set_reference_layers(self, entries):
        self.reference_layers = list(entries)

    def set_reference_visible(self, ref_id, visible):
        pass

    def set_colormap(self, name):
        self.colormap_calls.append(name)

    def zoom_to_reference(self, ref_id):
        pass

def _bare_field(label: str, *, dx: float = 1.0, dy: float = 1.0) -> RasterField:
    """A field with no CRS in its provenance at all -- LocalFrame, matching
    tests/test_arrangement_resolve.py's own ``_bare_field`` fixture, with dx/dy exposed so a test
    can build a frame that is deliberately INCOMPATIBLE with another bare field's default."""
    return RasterField._from_bare_array(
        _FIELD.copy(), label, frame=LocalFrame(dx=dx, dy=dy, units="px"), name=label)


def _chain_like_stub() -> Chain:
    return Chain(tuple(DeviceRef(n, dict(p)) for n, p in STUB_CHAIN))


class _FakeViewDialog:
    """Stands in for ``dynamix.shell.view_dialog.ViewDialog`` -- only its own
    ``set_frame_mode`` wiring is under test here, not the real
    dialog's tabs (covered in ``tests/test_view_dialog.py``)."""

    def __init__(self):
        self.frame_mode_calls: list[bool] = []

    def set_frame_mode(self, enabled: bool) -> None:
        self.frame_mode_calls.append(bool(enabled))


# --------------------------------------------------------------------------- (a) initial state


def test_opens_in_raster_view_at_stack_index_0(window):
    assert window._center_view == "raster"
    assert window._center_stack.currentIndex() == 0


# --------------------------------------------------------------------------- (b) cycle order


def test_cycle_center_view_walks_raster_vector_geo_raster(loaded):
    loaded._arrangement = _FakeArrangement()

    loaded._cycle_center_view()
    assert loaded._center_view == "vector"

    loaded._cycle_center_view()
    assert loaded._center_view == "geo"

    loaded._cycle_center_view()
    assert loaded._center_view == "raster"
    assert loaded._center_stack.currentIndex() == 0


# --------------------------------------------------------------------------- (c) switcher


def test_switcher_buttons_reflect_and_drive_state(loaded):
    fake = _FakeArrangement()
    loaded._arrangement = fake

    assert loaded._view_switcher_buttons["raster"].isChecked() is True
    assert loaded._view_switcher_buttons["vector"].isChecked() is False
    assert loaded._view_switcher_buttons["geo"].isChecked() is False

    loaded._view_switcher_buttons["vector"].click()

    assert loaded._center_view == "vector"
    assert loaded._view_switcher_buttons["vector"].isChecked() is True
    assert loaded._view_switcher_buttons["raster"].isChecked() is False
    assert fake.frame_mode_calls[-1] is True

    # Driving it the OTHER way -- a direct _set_center_view call -- must update the buttons too.
    loaded._set_center_view("geo")
    assert loaded._view_switcher_buttons["geo"].isChecked() is True
    assert loaded._view_switcher_buttons["vector"].isChecked() is False

    loaded._view_switcher_buttons["raster"].click()
    assert loaded._center_view == "raster"
    assert loaded._view_switcher_buttons["raster"].isChecked() is True


def test_reclicking_the_current_button_is_a_no_op(loaded):
    fake = _FakeArrangement()
    loaded._arrangement = fake
    loaded._set_center_view("vector")
    calls_before = len(fake.set_layers_calls)

    loaded._view_switcher_buttons["vector"].click()

    assert loaded._center_view == "vector"
    assert len(fake.set_layers_calls) == calls_before      # no redundant resync


# --------------------------------------------------------------------------- vector push-set contract
# Fix round 1: pins exactly which of the four view-options calls "vector" pushes --
# see the table in main_window.py's _set_center_view docstring: mode/graticule/vexag are
# mode-dependent chrome (skipped -- no CRS to project a mode against), background is not
# (pushed).


def test_vector_push_set_skips_mode_graticule_vexag_but_pushes_background(loaded):
    fake = _FakeArrangement()
    loaded._arrangement = fake

    loaded._set_center_view("vector")

    assert fake.mode_calls == []
    assert fake.graticule_calls == []
    assert fake.vexag_calls == []
    assert fake.background_calls != []
    # Scale-space is placement-independent (chain points lift the
    # same way in geo or frame mode, Scene's own module docstring), unlike mode/graticule/vexag
    # (which genuinely feed a meaningless-in-frame-mode CRS/projection dialog) -- so, unlike
    # those three, it is pushed UNCONDITIONALLY, alongside background, not gated behind
    # frame_mode. The original assertion here (`scale_space_calls == []`) mirrored vexag's own
    # gated placement per the design, but review traced a real gap that placement
    # created: a session restoring straight into "vector" never runs the geo branch at all, so a
    # persisted scale_space state would silently never reach the Scene -- see
    # test_vector_restore_pushes_persisted_scale_space below for that exact scenario.
    assert fake.scale_space_calls != []


def test_vector_restore_pushes_persisted_scale_space(window):
    """A fresh session restoring straight into a
    persisted "vector" view must still receive the persisted scale-space state -- the exact gap
    the ORIGINAL (vexag-mirroring, frame_mode-gated) push placement created (the live-dialog push
    and the tab-away-then-back path both still worked; only this fresh-into-"vector" restore was
    broken). Uses the plain ``window`` fixture, not ``loaded`` -- ``loaded`` already burns the
    ONE-SHOT restore via its own first ``load_field`` call (``MainWindow.load_field``'s own
    docstring: "A SECOND (or later) load_field call on the same window does not re-apply it"), so
    this scenario needs a window that has never called ``load_field`` yet."""
    fake = _FakeArrangement()
    window._arrangement = fake
    update_settings(center_view="vector",
                     view_options={"mode": "mercator", "graticule": False, "vexag": 1.0,
                                   "background": "#101010", "scale_space": True,
                                   "scale_space_stretch": 8.0})

    window.load_field(_FIELD, "mem:stub")

    assert fake.scale_space_calls == [(True, 8.0)]


def test_geo_push_includes_scale_space_alongside_vexag(loaded):
    fake = _FakeArrangement()
    loaded._arrangement = fake
    update_settings(view_options={"mode": "globe", "graticule": False, "vexag": 1.0,
                                   "background": "#101010", "scale_space": True,
                                   "scale_space_stretch": 12.5})

    loaded._set_center_view("geo")

    assert fake.scale_space_calls == [(True, 12.5)]


def test_set_center_view_pushes_the_current_selection_mode_on_flip_in(loaded):
    """The missing flip-in forwarding
    assertion. ``_set_center_view`` must push ``MainWindow._selection_mode`` into the just-
    activated arrangement on EVERY flip, mirroring the pre-existing ``set_mask`` push
    immediately above it in ``main_window.py`` -- unlike mode/graticule/vexag (mode-dependent
    chrome the vector push deliberately skips, see the section banner above), selection mode is
    placement-independent and pushed in BOTH "vector" and "geo". The mode is set BEFORE the fake
    arrangement is attached, so the only calls recorded here are the flip-in pushes themselves,
    not ``set_selection_mode``'s own separate (and separately real) forward-when-attached path."""
    loaded.set_selection_mode("box")
    fake = _FakeArrangement()
    loaded._arrangement = fake

    loaded._set_center_view("vector")
    assert fake.selection_mode_calls == ["box"]

    loaded._set_center_view("geo")
    assert fake.selection_mode_calls == ["box", "box"]


# --------------------------------------------------------------------------- View dialog push
# _set_center_view pushes the SAME frame_mode bool into
# the View dialog, if one has ever been built -- a no-op when it hasn't (the overwhelmingly common
# case: the dialog is lazy-built, only on its first "View…" click).


def test_set_center_view_pushes_frame_mode_into_an_existing_view_dialog(loaded):
    loaded._arrangement = _FakeArrangement()
    fake_dialog = _FakeViewDialog()
    loaded._view_dialog = fake_dialog

    loaded._set_center_view("vector")
    assert fake_dialog.frame_mode_calls == [True]

    loaded._set_center_view("geo")
    assert fake_dialog.frame_mode_calls == [True, False]


def test_set_center_view_never_touches_a_view_dialog_that_was_never_built(loaded):
    loaded._arrangement = _FakeArrangement()
    assert loaded._view_dialog is None

    loaded._set_center_view("vector")                # must not raise -- guarded no-op

    assert loaded._view_dialog is None


# --------------------------------------------------------------------------- (d) admission: active layer


def test_vector_admits_nongeoreferenced_active_layer_without_no_georeference_status(
        qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    field = _bare_field("bare")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(field, "mem:bare")
    active_id = win.layer.layer_id

    win._set_center_view("vector")
    vector_entry = next(e for e in fake.set_layers_calls[-1] if e["layer"].layer_id == active_id)
    assert vector_entry["status"] != "no-georeference"

    win._set_center_view("geo")
    geo_entry = next(e for e in fake.set_layers_calls[-1] if e["layer"].layer_id == active_id)
    assert geo_entry["status"] == "no-georeference"


# --------------------------------------------------------------------------- (e) admission: incompatible frame


def test_incompatible_frame_excluded_from_vector_but_present_in_geo(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    active_field = _bare_field("active")                  # LocalFrame(dx=1.0, dy=1.0, units=px)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, "mem:active")

    other_field = _bare_field("other", dx=5.0, dy=5.0)     # deliberately incompatible frame
    other_src = win.project.add_source("mem:other")
    other_layer = win.project.add_layer("other", other_src.source_id, _chain_like_stub())
    win.add_layer_row(other_layer, other_field)

    win._set_center_view("vector")
    vector_ids = {e["layer"].layer_id for e in fake.set_layers_calls[-1]}
    assert other_layer.layer_id not in vector_ids
    assert win.layer.layer_id in vector_ids

    win._set_center_view("geo")
    geo_ids = {e["layer"].layer_id for e in fake.set_layers_calls[-1]}
    assert other_layer.layer_id in geo_ids


def test_compatible_frame_is_admitted_into_vector_entries(qtbot, stub_devices):
    """The positive twin of the exclusion test above -- a second layer whose frame DOES match
    the active layer's own must still show up in vector entries (the gate is "compatible", not
    "everyone but the active layer is excluded")."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    active_field = _bare_field("active")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, "mem:active2")

    matching_field = _bare_field("matching")               # same LocalFrame(dx=1, dy=1, units=px)
    matching_src = win.project.add_source("mem:matching")
    matching_layer = win.project.add_layer("matching", matching_src.source_id, _chain_like_stub())
    win.add_layer_row(matching_layer, matching_field)

    win._set_center_view("vector")
    vector_ids = {e["layer"].layer_id for e in fake.set_layers_calls[-1]}
    assert matching_layer.layer_id in vector_ids


# --------------------------------------------------------------------------- AC3: same-frame,
# different-shape exclusion (final branch review, item 1). Two bare rasters both carry the
# IDENTICAL default LocalFrame(dx=1, dy=1, units=px), so frames_compatible alone is not enough --
# without the shape check, two grids of completely different sizes would overlay each other.


def test_same_frame_different_shape_excluded_from_vector_entries(qtbot, stub_devices):
    """A sibling whose ``field.frame`` matches the active layer's own exactly, but whose grid
    ``shape`` does not, is excluded from vector entries -- the shape half of the AC3 gate."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    active_field = _bare_field("active")                        # 16x16, LocalFrame(dx=1,dy=1,px)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, "mem:active-shape")

    different_shape_field = RasterField._from_bare_array(
        np.zeros((8, 8)), "different-shape", frame=LocalFrame(dx=1.0, dy=1.0, units="px"),
        name="different-shape")                                  # SAME frame, DIFFERENT grid shape
    other_src = win.project.add_source("mem:different-shape")
    other_layer = win.project.add_layer("other", other_src.source_id, _chain_like_stub())
    win.add_layer_row(other_layer, different_shape_field)

    win._set_center_view("vector")
    vector_ids = {e["layer"].layer_id for e in fake.set_layers_calls[-1]}
    assert other_layer.layer_id not in vector_ids
    assert win.layer.layer_id in vector_ids

    # The geo tab's own admission rule never looked at frame OR shape -- only georeference --
    # so this sibling (a bare/LocalFrame field, "no-georeference") shows up there regardless,
    # proving the shape gate is Vector-specific, not a blanket exclusion.
    win._set_center_view("geo")
    geo_ids = {e["layer"].layer_id for e in fake.set_layers_calls[-1]}
    assert other_layer.layer_id in geo_ids


def test_same_frame_and_shape_still_admitted(qtbot, stub_devices):
    """The positive twin: matching frame AND matching shape is admitted -- the shape check is an
    ADDITIONAL gate, not a replacement that would exclude everything."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    active_field = _bare_field("active")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, "mem:active-shape2")

    same_shape_field = RasterField._from_bare_array(
        np.zeros((16, 16)), "same-shape", frame=LocalFrame(dx=1.0, dy=1.0, units="px"),
        name="same-shape")
    other_src = win.project.add_source("mem:same-shape")
    other_layer = win.project.add_layer("other", other_src.source_id, _chain_like_stub())
    win.add_layer_row(other_layer, same_shape_field)

    win._set_center_view("vector")
    vector_ids = {e["layer"].layer_id for e in fake.set_layers_calls[-1]}
    assert other_layer.layer_id in vector_ids


# --------------------------------------------------------------------------- fix round 1:
# a visible sibling whose FIELD is a bare ndarray (no ``.frame`` attribute at all) must not crash
# vector admission -- ``load_field``'s own docstring allows a bare array, and an ROI/refined-run
# layer or a devloop round-trip can leave one visible next to a real RasterField active layer.


def test_bare_ndarray_sibling_field_is_excluded_without_crashing_in_vector_mode(qtbot, stub_devices):
    """Reviewer-reproduced Critical: ``field.frame`` was accessed unguarded in the vector
    admission gate, so a visible NON-active layer whose field is a bare ndarray raised
    ``AttributeError`` the instant ``_set_center_view("vector")`` tried to classify it. Must not
    raise; the frameless layer is simply excluded (no entry), while the real, framed active
    layer's own entry is unaffected."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    active_field = _bare_field("active")               # a real RasterField, LocalFrame
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, "mem:active-frame-guard")

    bare_src = win.project.add_source("mem:bare-ndarray")
    bare_layer = win.project.add_layer("bare-ndarray", bare_src.source_id, _chain_like_stub())
    win.add_layer_row(bare_layer, _FIELD.copy())        # plain ndarray -- NO .frame attribute

    win._set_center_view("vector")                      # must not raise AttributeError

    entries = fake.set_layers_calls[-1]
    ids = {e["layer"].layer_id for e in entries}
    assert bare_layer.layer_id not in ids
    assert win.layer.layer_id in ids


def test_bare_ndarray_active_field_emits_no_raster_entry_in_vector_mode(qtbot, stub_devices):
    """The other half of the same guard: when the ACTIVE layer's own field is frameless, its
    entry is skipped too (nothing for Scene's frame-mode placement to call), rather than crashing
    on ``field.frame`` for the active layer specifically."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD.copy(), "mem:bare-active")   # active field is a bare ndarray

    win._set_center_view("vector")                          # must not raise AttributeError

    entries = fake.set_layers_calls[-1]
    ids = {e["layer"].layer_id for e in entries}
    assert win.layer.layer_id not in ids


# --------------------------------------------------------------------------- (d)/(e) points: frame-points status


def test_point_layer_gets_frame_points_status_in_vector_mode(qtbot, stub_devices):
    from dynamix.core.pointset import PointSet
    import numpy as np
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    active_field = _bare_field("active")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, "mem:active3")

    pt_src = win.project.add_source("mem:pts", kind="points")
    pt_layer = win.project.add_layer("pts", pt_src.source_id)
    pset = PointSet(lon=np.array([1.0]), lat=np.array([2.0]))
    win.add_layer_row(pt_layer, pset)

    win._set_center_view("vector")
    entry = next(e for e in fake.set_layers_calls[-1] if e["layer"].layer_id == pt_layer.layer_id)
    assert entry["status"] == "frame-points"

    win._set_center_view("geo")
    entry = next(e for e in fake.set_layers_calls[-1] if e["layer"].layer_id == pt_layer.layer_id)
    assert entry["status"] == "ok"


# --------------------------------------------------------------------------- (f) Settings round trip


def test_settings_center_view_round_trips():
    assert load_settings().center_view == "raster"
    update_settings(center_view="vector")
    assert load_settings().center_view == "vector"
    update_settings(center_view="geo")
    assert load_settings().center_view == "geo"


def test_garbled_center_view_falls_back_to_raster(tmp_path, monkeypatch):
    import json

    from dynamix.shell.settings import settings_path

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "garbled.json"))
    settings_path().write_text(json.dumps({"center_view": "not-a-real-view"}))

    assert load_settings().center_view == "raster"


def test_window_restores_persisted_center_view_after_first_layer_load(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    update_settings(center_view="vector")

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    win._arrangement = _FakeArrangement()
    assert win._center_view == "raster"          # not restored until a layer actually loads

    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:restore")

    assert win._center_view == "vector"
    # NOT asserted here: ``_center_stack.currentIndex()``. The fake recorder above is a plain
    # object (per the design), never added to the real QStackedWidget -- ``setCurrentIndex(1)``
    # is therefore a documented Qt no-op (out-of-range index), which would make a stack-index
    # assertion here meaningless rather than a real check of anything this task changed.


def test_a_second_load_field_does_not_re_apply_the_restored_view(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    update_settings(center_view="vector")

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    win._arrangement = _FakeArrangement()
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:restore-once-a")
    assert win._center_view == "vector"

    win._set_center_view("raster")                # the user switches away on purpose
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD.copy(), "mem:restore-once-b")

    assert win._center_view == "raster"           # a second load must not snap back to vector


# --------------------------------------------------------------------------- (g) _notify


def test_notify_log_tier_writes_to_stderr_only_and_never_touches_the_status_bar(
        window, capsys):
    window._notify("quiet message", "log")

    assert "quiet message" in capsys.readouterr().err
    assert window.statusBar().currentMessage() == ""
    assert window._status_message == ""


def test_notify_status_tier_shows_the_message_with_no_timeout_and_tracks_it(window):
    window._notify("first status", "status")
    assert window.statusBar().currentMessage() == "first status"
    assert window._status_message == "first status"

    window._notify("second status", "status")     # replaces, not appends -- "track and clear"
    assert window.statusBar().currentMessage() == "second status"
    assert window._status_message == "second status"


def test_notify_modal_tier_is_a_status_message_under_offscreen(window, monkeypatch):
    """EQ§7's own gating: ``_on_wtmm_failed`` always writes the status bar and shows the modal
    box ONLY on a real display -- under the offscreen QPA platform this whole suite runs under,
    a modal ``exec()`` would block forever, so this must degrade to status-only."""
    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        lambda *a, **k: calls.append((a, k)))

    window._notify("needs a precondition", "modal")

    assert calls == []                              # never opened -- offscreen
    assert window.statusBar().currentMessage() == "needs a precondition"
    assert window._status_message == "needs a precondition"


def test_dropplaced_reaches_the_status_bar(loaded):
    """``WorkflowZone.dropPlaced`` -> ``MainWindow._notify(msg, "status")``, wired in
    ``_build_strips`` -- a clamped drop's message lands on the status bar, not just the zone's
    own label."""
    from dynamix.shell.browser import DEVICE_MIME
    from tests.test_workflow_zone import _mime, _StubDropEvent

    window = loaded
    assert window.strips is not None
    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "wtmm2d"))
    window.strips.dropEvent(event)

    assert window.statusBar().currentMessage() == window.strips.reading_label.text()
    assert "wtmm2d placed in the transform section" in window.statusBar().currentMessage()


def test_shift_cmd_z_shortcut_routes_to_whichever_strips_is_currently_live(loaded):
    """The binding decision (see the ``QShortcut`` construction site's comment): a
    SEPARATE chord from the plain ⌘Z (no collision), dispatched through a wrapper that
    re-reads ``self.strips`` at activation time so it survives a rebuild."""
    window = loaded
    assert window._key_undo_rack.key() != window._key_undo_transect.key()

    calls = []
    window.strips.undo_removal = lambda: calls.append(window.strips)
    window._key_undo_rack.activated.emit()
    assert calls == [window.strips]

    # After a rebuild, the SAME shortcut must reach the NEW zone, not the torn-down one.
    old_strips = window.strips
    window._build_strips()
    assert window.strips is not old_strips
    calls.clear()
    window.strips.undo_removal = lambda: calls.append(window.strips)
    window._key_undo_rack.activated.emit()
    assert calls == [window.strips]


def test_a_colormap_change_swaps_the_lut_in_place_instead_of_resyncing(loaded):
    win = loaded
    fake = _FakeArrangement(); win._arrangement = fake
    win._set_center_view("vector")
    n_syncs = len(fake.set_layers_calls)
    win._on_display_style_changed("colormap", "magma")
    assert fake.colormap_calls == ["magma"]                       # the in-place LUT swap
    assert len(fake.set_layers_calls) == n_syncs                  # and NO full rebuild


def test_a_colormap_change_with_hillshade_on_still_resyncs(loaded):
    win = loaded
    fake = _FakeArrangement(); win._arrangement = fake
    win._set_center_view("vector")
    win.layer.tags["ui.hillshade"] = "True"                       # baked RGBA: LUT swap can't help
    n_syncs = len(fake.set_layers_calls)
    win._on_display_style_changed("colormap", "magma")
    assert len(fake.set_layers_calls) > n_syncs


# --------------------------------------------------------------------------- ROI child overlay:
# a child ROI crop is the legitimate
# exception to the AC3 shape gate -- same source, ``tags["roi.window"]`` linkage, axes sliced
# from the parent's own -- so parent and child overlay at the correct relative position in the
# Vector tab. Admission only; placement is the scene's existing axes-direct math.


def test_roi_child_is_admitted_into_vector_entries_despite_its_shape(qtbot, stub_devices):
    """Parent active: its child crop (same source, roi.window tag, smaller grid) is admitted."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    active_field = _bare_field("active")                        # 16x16
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, "mem:roi-parent")

    child_field = RasterField._from_bare_array(
        np.zeros((8, 8)), "child", frame=LocalFrame(dx=1.0, dy=1.0, units="px"),
        name="child")
    child_layer = win.project.add_layer("child", win.layer.source_id, _chain_like_stub(),
                                        parent_id=win.layer.layer_id)
    child_layer.tags["roi.window"] = "4,4,8,8"
    win.add_layer_row(child_layer, child_field)

    win._set_center_view("vector")
    vector_ids = {e["layer"].layer_id for e in fake.set_layers_calls[-1]}
    assert child_layer.layer_id in vector_ids
    assert win.layer.layer_id in vector_ids


def test_roi_parent_is_admitted_when_the_child_is_active(qtbot, stub_devices):
    """Child active (creation SELECTS the child, so this is the flow the user actually sees):
    the parent must be admitted alongside it -- the family clause reads the ACTIVE side's tag
    too, not only the sibling's."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    fake = _FakeArrangement()
    win._arrangement = fake

    active_field = _bare_field("active")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, "mem:roi-parent2")
    parent_layer = win.layer

    child_field = RasterField._from_bare_array(
        np.zeros((8, 8)), "child", frame=LocalFrame(dx=1.0, dy=1.0, units="px"),
        name="child")
    child_layer = win.project.add_layer("child", parent_layer.source_id, _chain_like_stub(),
                                        parent_id=parent_layer.layer_id)
    child_layer.tags["roi.window"] = "4,4,8,8"
    win.add_layer_row(child_layer, child_field)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.layer_list.select_layer(child_layer.layer_id)

    win._set_center_view("vector")
    vector_ids = {e["layer"].layer_id for e in fake.set_layers_calls[-1]}
    assert parent_layer.layer_id in vector_ids
    assert child_layer.layer_id in vector_ids
