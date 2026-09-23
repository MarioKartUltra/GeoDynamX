# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""tests/test_right_panel.py -- the right panel scaffold + Display section.

Every test here builds a bare ``MainWindow()``, which unconditionally calls
``register_builtin_devices()`` (``main_window.py``'s own ``__init__``) -- the suite-wide
``_registry_leak_guard`` (tests/conftest.py) requires that be undone afterwards, so each test
also takes ``clean_registry`` (unused directly; its fixture teardown restores ``DEVICES``), same
pattern ``test_shell_layout.py`` already uses for the identical reason.

The plan's own sketch names ``synthetic_layer_window``/``two_layer_window`` fixtures that do not
exist anywhere in the suite (the plan explicitly says to adapt fixture names to what
``tests/test_display_controls.py`` actually builds, reusing its helpers rather than inventing
parallel ones). That file imports ``loaded`` (a stub-chain window whose first resolve has landed)
from ``tests/test_shell_window.py`` and builds a second layer inline
(``test_style_round_trips_across_layer_switches``) rather than through a named fixture -- both
tests below follow that exact precedent.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from dynamix.devices import register_builtin_devices
from dynamix.shell.main_window import MainWindow
from dynamix.shell.right_panel import RightPanel
from tests.test_shell_window import _FIELD, loaded, stub_devices, window  # noqa: F401


def test_display_knobs_live_right_not_left(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    host = w._work_split.widget(2)
    assert w.right_panel.isAncestorOf(w._display_controls["opacity"])
    left = w._work_split.widget(0)
    assert not left.isAncestorOf(w._display_controls["opacity"])
    assert host is w.right_panel or host.isAncestorOf(w.right_panel)


def test_style_edit_reaches_canvas_and_tags(qtbot, loaded):
    w = loaded
    control = w._display_controls["opacity"]
    control.valueChanged.emit(0.5)
    assert w.layer.tags["ui.opacity"] == repr(0.5)


def test_layer_switch_syncs_panel_without_writing_tags(qtbot, loaded):
    """Switching layers must reach the panel through ``set_style_values`` -- non-emitting, so it
    writes no tag on the layer being switched AWAY from (the ``_sync_display_controls`` hook)."""
    w = loaded
    first = w.layer
    second = w.project.add_layer("plain", first.source_id, first.chain)
    w.add_layer_row(second, w._fields[first.layer_id])
    before = dict(first.tags)
    with qtbot.waitSignal(w.resolved, timeout=10000):
        w.layer_list.select_layer(second.layer_id)
    assert dict(first.tags) == before


def test_panel_min_width_is_200_no_max(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.right_panel.minimumWidth() == 200
    assert w.right_panel.maximumWidth() >= 16_000_000     # Qt's own QWIDGETSIZE_MAX: no cap set


def test_fresh_settings_uncollapse_right_slot_to_a_default_width(qtbot, tmp_path, monkeypatch,
                                                                 clean_registry):
    """The right slot hosts the Display section, so a window with no PERSISTED sizes must open it to a sane default instead
    of hiding the knobs it just relocated. A window with stored sizes is untouched -- see the
    sibling test below."""
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._work_split.sizes()[2] > 0


def test_stored_splitter_sizes_still_win_over_the_fresh_default(qtbot, tmp_path, monkeypatch,
                                                                 clean_registry):
    """A stored width AT OR ABOVE the panel's own 200px minimum round-trips exactly -- a size
    BELOW 200 is not a real achievable user gesture (the splitter's ``setChildrenCollapsible``
    lets a drag snap a pane to fully 0, never to an arbitrary point under its minimum), so it is
    not this test's concern.

    ``show()`` + ``waitExposed`` (not a bare ``resize()``) is required here: an un-shown
    top-level widget does not propagate a ``resize()`` to its children's geometry until the
    platform actually exposes it, so the splitter would otherwise still be redistributing sizes
    against its tiny pre-show width -- ``tests/test_arrangement_flip.py`` uses the identical
    ``show()``/``waitExposed`` pairing for the same reason."""
    from dynamix.shell.settings import Settings, save_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    save_settings(Settings(splitter_sizes={"work": [180, 900, 220]}))
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    assert w._work_split.sizes()[0] == 180
    assert w._work_split.sizes()[2] == 220             # non-stretch panes hold their stored size


def test_wheel_over_a_display_knob_scrolls_the_panel_not_the_knob(qtbot, clean_registry):
    """The knob's own value must not move, and the panel's scrollbar must (EQSelect's
    ``_PanelWheelFilter`` precedent, ``app_window.py:1085-1103``). The scrollbar's range is
    forced open here rather than relied on from real layout overflow -- what is under test is
    that the filter forwards the wheel delta to the scrollbar, not that this particular window
    size happens to overflow."""
    from PySide6 import QtCore, QtGui

    w = MainWindow()
    qtbot.addWidget(w)
    control = w._display_controls["opacity"]
    before = control._value
    bar = w.right_panel.verticalScrollBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    event = QtGui.QWheelEvent(
        QtCore.QPointF(5, 5), QtCore.QPointF(5, 5), QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.ScrollUpdate, False)
    QtWidgets.QApplication.sendEvent(control, event)
    assert control._value == before
    assert bar.value() != 0


def test_add_section_appends_below_display(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    extra = QtWidgets.QLabel("extra section body")
    w.right_panel.add_section("Extra", extra)
    assert w.right_panel.isAncestorOf(extra)
    # still ancestor of the Display controls too -- add_section never replaces, only appends
    assert w.right_panel.isAncestorOf(w._display_controls["line_width"])


def test_right_panel_scaffold_is_a_scroll_area():
    assert issubclass(RightPanel, QtWidgets.QScrollArea)


# --------------------------------------------------------------- Display mask (relocated)
#
# The arrangement view's MaskRow (``arrangement/mask_row.py`` -- widget unchanged) moves into a
# view-scoped right-panel section owned by ``MainWindow``; the view keeps only a public
# ``set_mask(payload)`` passthrough (``tests/test_arrangement_mask.py``'s own "ArrangementView
# wiring" section now tests THAT, not a row this view no longer builds).


def test_mask_row_lives_in_right_panel_not_view(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.right_panel.isAncestorOf(w._mask_row)


@pytest.fixture
def arranged_window_with_stub_scene(qtbot, clean_registry):
    """A window flipped to the arrangement, with a REAL ``Scene`` attached directly against a
    plain offscreen ``pv.Plotter`` -- ``tests/test_arrangement_commit.py``'s own ``win`` fixture
    substitution (and ``tests/test_arrangement_mode_row.py``'s per-test ``importorskip``
    convention), reused here rather than inventing a parallel fixture: a real ``QtInteractor``
    cannot be built under this harness's mandated offscreen QPA platform without segfaulting
    (``tests/test_arrangement_flip.py``'s documented guard), so ``activate()`` alone leaves
    ``w._arrangement._scene`` at ``None``."""
    pytest.importorskip("pyvista", reason="pyvista not installed")
    import pyvista as pv
    pv.OFF_SCREEN = True
    from dynamix.shell.arrangement.scene import Scene

    w = MainWindow()
    qtbot.addWidget(w)
    w._toggle_center_view()
    if not w._arrangement.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")
    w._arrangement._scene = Scene(pv.Plotter(off_screen=True))
    return w


def test_mask_edit_reaches_scene_when_arranged(qtbot, arranged_window_with_stub_scene):
    # The design's sketch names this `w.arrangement_view._scene.last_mask`; adapted here
    # to the window's real private `_arrangement` attribute and `Scene`'s own `_mask` TUPLE
    # attribute (`scene.py`'s `set_mask`: `(modulus_pctl, scale_lo, scale_hi)`) -- neither an
    # `arrangement_view` public alias nor a `last_mask` field exists anywhere in this codebase,
    # and the binding constraint is to adapt existing fixtures/attributes, never invent parallel
    # ones (`tests/test_arrangement_mask.py`'s own `view._scene._mask` assertion is the precedent).
    w = arranged_window_with_stub_scene
    w._mask_row.maskChanged.emit({"modulus_pctl": 40.0, "scale_lo": 1, "scale_hi": 0})
    assert w._arrangement._scene._mask == (40.0, 1, 0)


# --------------------------------------------------------------- Groups (relocated)
#
# GroupPalette + the Commit button (``arrangement/group_palette.py`` -- widget unchanged) move
# into a second view-scoped right-panel section owned by ``MainWindow``; the view keeps only a
# public ``set_group_palette(palette)`` passthrough, storing the reference it reads picks INTO
# (``_on_click`` -> ``add_pick``) and snapshots FROM at commit time (``tests/
# test_arrangement_commit.py``'s own "one-snapshot" rule, unchanged -- ``tests/
# test_arrangement_picking.py``'s own "ArrangementView wiring" section now tests the passthrough
# itself, not a palette this view no longer builds).


def test_group_palette_and_commit_button_live_in_right_panel_not_view(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.right_panel.isAncestorOf(w._group_palette)
    assert w.right_panel.isAncestorOf(w._commit_button)


def test_commit_reaches_the_scene_and_emits_groups_committed_when_arranged(
        qtbot, arranged_window_with_stub_scene):
    """Palette hosted in the right panel; a scripted pick -> commit against
    the stub scene still emits one ``groupsCommitted`` per layer with the same payload shape as
    before the relocation (``tests/test_arrangement_commit.py``'s own ``test_groups_committed_
    emitted_once_per_affected_layer``, adapted here only in how the palette/button are reached)."""
    from dynamix.shell.arrangement.group_palette import GROUP_COLORS

    w = arranged_window_with_stub_scene
    w._group_palette.new_group("g")
    w._group_palette.add_pick((1, 0), shift=False)     # layer_id=1 is arbitrary -- no real layer
                                                        # is loaded in this fixture; see its own
                                                        # docstring -- pre-commit membership is
                                                        # independent of whether the layer exists.

    received = []
    w._arrangement.groupsCommitted.connect(lambda lid, groups: received.append((lid, groups)))
    with qtbot.waitSignal(w._arrangement.commitFinished, timeout=1000):
        w._commit_button.click()

    assert received == [(1, {"g": {"chains": [0], "color": list(GROUP_COLORS[0])}})]
    # The pick also reached the scene through the same stored palette reference (Scene.
    # set_group_preview/set_selection, wired by set_group_palette's own membershipChanged connect).
    assert w._arrangement._scene._selection.get(1) == {0}


# --------------------------------------------------------------- Colormap combo, swatches,
# trails checkbox -- the Display-section UI that emits the five color keys.


def test_colormap_combo_has_20_iconed_entries(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    combo = w.right_panel.cmap_combo
    assert combo.count() == 20
    for i in range(combo.count()):
        assert not combo.itemIcon(i).isNull()


def test_selecting_plasma_writes_the_tag_and_recolors_the_canvas(qtbot, loaded):
    w = loaded
    before = w.canvas.image_item.lut

    w.right_panel.cmap_combo.setCurrentText("plasma")

    assert w.layer.tags["ui.colormap"] == "plasma"
    assert not np.array_equal(before, w.canvas.image_item.lut)


def test_swatch_click_with_a_chosen_color_writes_tag_and_updates_the_face(qtbot, loaded,
                                                                          monkeypatch):
    w = loaded
    monkeypatch.setattr(w.right_panel, "_pick_color", lambda initial: "#112233")

    w.right_panel._swatch_buttons["color_hchain"].click()

    assert w.layer.tags["ui.color_hchain"] == "#112233"
    assert "#112233" in w.right_panel._swatch_buttons["color_hchain"].styleSheet()
    assert w.canvas.hchain_item.opts["pen"].color().name() == "#112233"


def test_swatch_click_cancelled_writes_nothing(qtbot, loaded, monkeypatch):
    w = loaded
    monkeypatch.setattr(w.right_panel, "_pick_color", lambda initial: None)

    w.right_panel._swatch_buttons["color_vtrail"].click()

    assert "ui.color_vtrail" not in w.layer.tags


def test_trails_checkbox_flips_vtrail_visibility_end_to_end(qtbot, loaded):
    w = loaded
    assert w.canvas.vtrail_item.isVisible() is False

    w.right_panel.trails_check.setChecked(True)

    assert w.layer.tags["ui.show_trails"] == "True"
    assert w.canvas.vtrail_item.isVisible() is True


def test_set_style_values_syncs_combo_swatches_and_checkbox_without_emitting(qtbot, loaded):
    """Extends the non-emitting contract (``test_layer_switch_syncs_panel_without_writing_
    tags``) onto the three controls this task adds: a programmatic sync must show the new style
    on every one of them without proposing an edit back out through ``styleChanged``."""
    w = loaded
    received = []
    w.right_panel.styleChanged.connect(lambda name, value: received.append((name, value)))
    style = {"opacity": 1.0, "point_size": 3.0, "line_width": 1.0, "colormap": "plasma",
             "color_hchain": "#112233", "color_vtrail": "#445566", "color_extrema": "#778899",
             "show_trails": True}

    w.right_panel.set_style_values(style)

    assert received == []
    assert w.right_panel.cmap_combo.currentText() == "plasma"
    assert "#112233" in w.right_panel._swatch_buttons["color_hchain"].styleSheet()
    assert "#445566" in w.right_panel._swatch_buttons["color_vtrail"].styleSheet()
    assert "#778899" in w.right_panel._swatch_buttons["color_extrema"].styleSheet()
    assert w.right_panel.trails_check.isChecked() is True


def test_wheel_over_the_colormap_combo_scrolls_the_panel_not_the_combo(qtbot, clean_registry):
    """Same precedent as ``test_wheel_over_a_display_knob_scrolls_the_panel_not_the_knob``: the
    existing ``_WheelRedirect``/``_protect_from_wheel`` pair installs on every descendant widget
    of a section at ``add_section`` time, which already includes the combo (it is built and
    parented before ``add_section`` runs) -- this confirms that coverage extends to it too rather
    than assuming it from the knob case alone."""
    from PySide6 import QtCore, QtGui

    w = MainWindow()
    qtbot.addWidget(w)
    combo = w.right_panel.cmap_combo
    before = combo.currentText()
    bar = w.right_panel.verticalScrollBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    event = QtGui.QWheelEvent(
        QtCore.QPointF(5, 5), QtCore.QPointF(5, 5), QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.ScrollUpdate, False)
    QtWidgets.QApplication.sendEvent(combo, event)
    assert combo.currentText() == before
    assert bar.value() != 0


# --------------------------------------------------------------- Topology links
#
# TopologyPanel (``shell/topology_panel.py``) lives in a THIRD right-panel section, not
# view-scoped (unlike Groups/Display mask): a link names two chains regardless of which
# center-stack page is showing. ``MainWindow`` wires ``linkRequested``/``unlinkRequested`` into
# ``self.project.user_links`` (``topology.links.LinkStore``), naming the group palette's last two
# picks -- the identical pick-gesture precedent ``test_commit_reaches_the_scene_and_emits_
# groups_committed_when_arranged`` above already establishes for driving ``GroupPalette.add_pick``
# by hand rather than through a real mouse event.


class _ChainStub:
    """A transform whose result already carries ``"chains"`` -- the one thing STUB_CHAIN's own
    ``_stub_stack`` (test_shell_window.py) never produces (it stops at "extrema"), so this task's
    own point-gathering path (``MainWindow._topology_chain_points``, which reads
    ``result["chains"]`` directly rather than through the canvas) needs a chain-shaped stub of its
    own to exercise end to end."""

    name = "chain_stub"
    params = ()

    def compute(self, field, params, *, progress=None):
        return {"chains": [{"x": [0.0, 1.0], "y": [0.0, 0.0]},
                           {"x": [5.0, 6.0], "y": [5.0, 5.0]}]}

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


@pytest.fixture
def chain_window(qtbot, clean_registry):
    from dynamix.model.device import register_device

    register_builtin_devices()
    register_device(_ChainStub())
    w = MainWindow(steps=(("chain_stub", {}),))
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.resolved, timeout=10000):
        w.load_field(np.zeros((4, 4)), "mem:chainstub")
    return w


class _ChainStubWithScales(_ChainStub):
    """Same chain-shaped result as ``_ChainStub``, plus a real ``"scales"`` array -- the ONE thing
    ``_ChainStub``'s own result never carries, which is why ``self._scales`` stays empty even when
    a ``scale_select`` step sits in the chain (``ScaleSelect.apply`` is a no-op on a result with no
    ``"extrema"`` either way -- ``_layers``'s own ``result.get("extrema") or []`` -- so this stub's
    ``"chains"`` key survives the filter untouched). Lets a chain include a GENUINE scale_select
    step, so ``_current_scale_px()`` has a real px value to report (final branch review, item 2's
    positive case, the twin of ``chain_window``'s own no-scale-select negative case)."""

    name = "chain_stub_with_scales"

    def compute(self, field, params, *, progress=None):
        result = super().compute(field, params, progress=progress)
        result["scales"] = [2.0, 4.0, 8.0]
        return result


@pytest.fixture
def chain_window_with_scale(qtbot, clean_registry):
    from dynamix.model.device import register_device

    register_builtin_devices()
    register_device(_ChainStubWithScales())
    w = MainWindow(steps=(("chain_stub_with_scales", {}), ("scale_select", {"scale_idx": 0})))
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.resolved, timeout=10000):
        w.load_field(np.zeros((4, 4)), "mem:chainstub-scaled")
    return w


def test_topology_panel_lives_in_right_panel_and_is_not_view_scoped(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.right_panel.isAncestorOf(w._topology_panel)
    # walk up from the panel to its section frame and check the view_scoped property the same
    # way RightPanel.add_section itself stamps it (the section must be False, unlike
    # Groups/Display mask, which are True).
    frame = w._topology_panel.parentWidget()
    while frame is not None and not frame.property("view_scoped") is not None:
        frame = frame.parentWidget()
    assert frame is not None
    assert frame.property("view_scoped") is False


def test_link_with_fewer_than_two_picks_is_a_statusbar_noop(qtbot, chain_window):
    w = chain_window
    before = len(w.project.user_links.all())
    w._topology_panel.linkRequested.emit(None)
    assert len(w.project.user_links.all()) == before
    assert "exactly two" in w.statusBar().currentMessage()


def test_link_with_three_picks_is_a_statusbar_noop(qtbot, chain_window):
    """'the last two picks' is gone -- anything other than exactly two
    picks refuses outright rather than guessing which pair was meant. A third pick need not even
    name a real chain (index 2 does not exist in ``_ChainStub``'s own two-chain result): the
    exactly-two check must short-circuit before any geometry is ever touched."""
    w = chain_window
    before = len(w.project.user_links.all())
    w._group_palette.add_pick((w.layer.layer_id, 0), shift=False)
    w._group_palette.add_pick((w.layer.layer_id, 1), shift=True)
    w._group_palette.add_pick((w.layer.layer_id, 2), shift=True)

    w._topology_panel.linkRequested.emit(None)

    assert len(w.project.user_links.all()) == before
    assert "exactly two" in w.statusBar().currentMessage()


def test_link_button_end_to_end_creates_a_link_and_refreshes_the_panel(qtbot, chain_window):
    w = chain_window
    w._group_palette.add_pick((w.layer.layer_id, 0), shift=False)
    w._group_palette.add_pick((w.layer.layer_id, 1), shift=True)

    w._topology_panel.linkRequested.emit(None)     # auto-suggest, no override

    (link,) = w.project.user_links.links_for_layer(w.layer.layer_id)
    assert link.a.transform == "chain_stub" and link.a.kind == "line"
    assert {link.a.obj_id, link.b.obj_id} == {0, 1}
    # Final branch review, item 2: no scale_select step in this chain -> _current_scale_px()
    # reports None, HONESTLY -- no more fabricated 1.0px "measurement". The panel's own row
    # falls back to the "@a=?" reading it already had a branch for.
    assert link.scale_first_contact is None
    assert w._topology_panel._list.count() == 1
    assert "@a=?" in w._topology_panel._list.item(0).text()


def test_unlink_button_end_to_end_removes_the_link_and_refreshes_the_panel(qtbot, chain_window):
    w = chain_window
    w._group_palette.add_pick((w.layer.layer_id, 0), shift=False)
    w._group_palette.add_pick((w.layer.layer_id, 1), shift=True)
    w._topology_panel.linkRequested.emit(None)
    assert len(w.project.user_links.all()) == 1

    w._topology_panel._list.setCurrentRow(0)
    w._topology_panel.unlinkRequested.emit(0)

    assert w.project.user_links.all() == []
    assert w._topology_panel._list.count() == 0


def test_link_honors_the_panels_code_override(qtbot, chain_window):
    w = chain_window
    w._group_palette.add_pick((w.layer.layer_id, 0), shift=False)
    w._group_palette.add_pick((w.layer.layer_id, 1), shift=True)
    from dynamix.topology.codes import LINE, permitted

    override = sorted(permitted(LINE, LINE, 2))[0]

    w._topology_panel.linkRequested.emit(override)

    (link,) = w.project.user_links.all()
    assert link.code == override


def test_link_records_the_real_scale_when_scale_select_is_present(qtbot, chain_window_with_scale):
    """Final branch review, item 2's positive case: a chain that DOES carry a real scale_select
    step stores the genuine px value, not None and not a fabricated fallback."""
    w = chain_window_with_scale
    w._group_palette.add_pick((w.layer.layer_id, 0), shift=False)
    w._group_palette.add_pick((w.layer.layer_id, 1), shift=True)

    w._topology_panel.linkRequested.emit(None)

    (link,) = w.project.user_links.links_for_layer(w.layer.layer_id)
    assert link.scale_first_contact == 2.0
    assert "@a=2" in w._topology_panel._list.item(0).text()


# --------------------------------------------------------------- Display-frame
# geometry for cross-layer links
#
# _topology_chain_points must shift an ROI/refined-run layer's raw chain coordinates by the SAME
# display_offset _apply already applies before a canvas pick -- otherwise two chains that visibly
# touch on screen classify by their raw, un-shifted (and possibly enormously distant) coordinates.


class _PlainChainStub:
    """No ``_roi`` at all -- ``display_offset`` gives it ``(0, 0)``, so its chain's raw and
    display coordinates are identical, exactly like ``wtmm2d`` running on the window it was
    given (``canvas.display_offset``'s own docstring)."""

    name = "plain_chain_stub"
    params = ()

    def compute(self, field, params, *, progress=None):
        return {"chains": [{"x": [100.0, 100.5], "y": [100.0, 100.0]}]}

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


class _ROIChainStubNear:
    """Raw chain far from ``_PlainChainStub``'s own (near the origin, not (100, 100)) -- but
    tagged ``_roi: {"roi": (100, 100)}``, which ``display_offset`` (no ``field.provenance`` on a
    bare stub field, so ``window_offset`` is ``(0, 0)``) turns into a ``(row_off=100,
    col_off=100)`` shift. Shifted, the chain lands exactly 1.0px from the plain stub's own chain
    -- touching, at this test's ``contact_scale=1.0`` fallback -- with bounding boxes that do NOT
    overlap (so the touch branch, not the overlap branch, is what must fire)."""

    name = "roi_chain_stub_near"
    params = ()

    def compute(self, field, params, *, progress=None):
        return {"chains": [{"x": [1.5, 2.0], "y": [0.0, 0.0]}], "_roi": {"roi": (100, 100)}}

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


class _ROIChainStubFar:
    """Raw chain IDENTICAL to ``_PlainChainStub``'s own -- reading raw coordinates alone (the
    pre-fix bug) would call this touching or overlapping. Tagged ``_roi: {"roi": (0, 500)}``, a
    500px shift that moves it far away in the DISPLAY frame -- the converse proof that the
    classification genuinely depends on the shifted geometry, not merely on whichever branch a
    zero offset happens to hit."""

    name = "roi_chain_stub_far"
    params = ()

    def compute(self, field, params, *, progress=None):
        return {"chains": [{"x": [100.0, 100.5], "y": [100.0, 100.0]}],
               "_roi": {"roi": (0, 500)}}

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


def _two_layer_chain_window(qtbot, roi_stub):
    """A window with layer A (plain_chain_stub, the active layer) and layer B (whichever
    ``roi_stub`` instance is passed, a second layer over the same source) -- ``add_layer_row``
    alone is enough for ``_topology_chain_points`` to resolve layer B (it only ever looks the
    layer/field up by id, never requires it to be the ACTIVE one)."""
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.device import register_device

    register_builtin_devices()
    register_device(_PlainChainStub())
    register_device(roi_stub)
    w = MainWindow(steps=(("plain_chain_stub", {}),))
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.resolved, timeout=10000):
        w.load_field(np.zeros((4, 4)), "mem:plain")
    layer_a = w.layer
    layer_b = w.project.add_layer("roi", layer_a.source_id,
                                  Chain((DeviceRef(roi_stub.name, {}),)))
    w.add_layer_row(layer_b, w._fields[layer_a.layer_id])
    return w, layer_a, layer_b


def test_topology_link_classifies_by_display_frame_geometry_touching(qtbot, clean_registry):
    from dynamix.topology.links import _TOUCH_CODE

    w, layer_a, layer_b = _two_layer_chain_window(qtbot, _ROIChainStubNear())
    w._group_palette.add_pick((layer_a.layer_id, 0), shift=False)
    w._group_palette.add_pick((layer_b.layer_id, 0), shift=True)

    w._topology_panel.linkRequested.emit(None)

    (link,) = w.project.user_links.all()
    assert link.code == _TOUCH_CODE


def test_topology_link_classifies_by_display_frame_geometry_disjoint_converse(qtbot,
                                                                              clean_registry):
    """The converse of the test above: raw coordinates that would read as touching/overlapping
    are pushed genuinely far apart by the ROI shift -- disjoint is only correct here because the
    shift was actually applied, not because this scenario would be disjoint either way."""
    from dynamix.topology.links import _DISJOINT_CODE

    w, layer_a, layer_b = _two_layer_chain_window(qtbot, _ROIChainStubFar())
    w._group_palette.add_pick((layer_a.layer_id, 0), shift=False)
    w._group_palette.add_pick((layer_b.layer_id, 0), shift=True)

    w._topology_panel.linkRequested.emit(None)

    (link,) = w.project.user_links.all()
    assert link.code == _DISJOINT_CODE


# --------------------------------------------------------------- Refresh-stable
# unlink selection
#
# _refresh_topology_panel runs on every _apply (a filter knob moving, a scale scrub -- anything),
# not only after a link/unlink. Rebuilding the QListWidget every time drops whatever row a
# half-finished "select a row, then click Unlink" gesture had selected, even though nothing about
# the link set itself changed.


def test_unrelated_refresh_preserves_the_selected_row_for_unlink(qtbot, chain_window):
    w = chain_window
    w._group_palette.add_pick((w.layer.layer_id, 0), shift=False)
    w._group_palette.add_pick((w.layer.layer_id, 1), shift=True)
    w._topology_panel.linkRequested.emit(None)
    assert w._topology_panel._list.count() == 1
    w._topology_panel._list.setCurrentRow(0)
    assert w._topology_panel._list.currentRow() == 0

    # An UNRELATED redraw -- the exact path _apply takes on every resolve, with the link set
    # itself untouched (no link/unlink happened here).
    w._refresh_topology_panel()

    assert w._topology_panel._list.currentRow() == 0     # selection survived
    assert w._topology_panel._list.count() == 1           # and the row itself is still there


def test_a_refresh_with_genuinely_changed_rows_does_rebuild(qtbot, chain_window):
    w = chain_window
    w._group_palette.add_pick((w.layer.layer_id, 0), shift=False)
    w._group_palette.add_pick((w.layer.layer_id, 1), shift=True)
    w._topology_panel.linkRequested.emit(None)
    assert w._topology_panel._list.count() == 1

    w._topology_panel._list.setCurrentRow(0)
    w.project.user_links.unlink(0)          # the link set NOW genuinely differs from what was shown
    w._refresh_topology_panel()

    assert w._topology_panel._list.count() == 0
