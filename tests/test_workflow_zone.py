# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.workflow_zone and the WTMM PRE/CHAINING param exposure it
depends on.

Offscreen Qt (see conftest.py / the runner's QT_QPA_PLATFORM=offscreen), stub-free for the widget
tests: exercises the real registered ``wtmm2d`` device, same convention
tests/test_shell_chain_strip.py already uses for DeviceStrip.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.devices import register_builtin_devices
from dynamix.devices.wtmm import WTMM2D
from dynamix.model.device import defaults_for, get_device


@pytest.fixture
def registered_builtins(clean_registry):
    register_builtin_devices()
    return clean_registry


class _StubDropEvent:
    """A minimal stand-in for ``QDragEnterEvent``/``QDropEvent``: pytest-qt cannot synthesize a
    real native drag reliably offscreen (``workflow_zone``'s module docstring, drag-drop section),
    so every drop test here calls ``dragEnterEvent``/``dropEvent`` directly with one of these
    instead of driving a real ``QDrag`` through Qt's own event queue. Needs exactly what the
    handlers under test read: position, mime data, and the two accept/ignore outcomes."""

    def __init__(self, pos: QtCore.QPoint, mime: QtCore.QMimeData,
                action=QtCore.Qt.MoveAction):
        self._pos = QtCore.QPointF(pos)
        self._mime = mime
        self._action = action
        self.accepted: bool | None = None

    def position(self):
        return self._pos

    def mimeData(self) -> QtCore.QMimeData:
        return self._mime

    def proposedAction(self):
        return self._action

    def acceptProposedAction(self) -> None:
        self.accepted = True

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.accepted = False


class _StubMouseEvent:
    """A minimal stand-in for ``QMouseEvent``, for driving ``_TitleBar``'s own press/move/release
    handlers directly (pure Python, never forwarded into a real Qt C++ call) -- constructing a
    genuine ``QMouseEvent`` is unnecessary ceremony for testing pure threshold arithmetic."""

    def __init__(self, pos: QtCore.QPoint, button=QtCore.Qt.LeftButton, buttons=None):
        self._pos = QtCore.QPoint(pos)
        self._button = button
        self._buttons = button if buttons is None else buttons

    def pos(self) -> QtCore.QPoint:
        return self._pos

    def button(self):
        return self._button

    def buttons(self):
        return self._buttons


def _mime(mime_type: str, payload: str) -> QtCore.QMimeData:
    data = QtCore.QMimeData()
    data.setData(mime_type, payload.encode("utf-8"))
    return data


def _assert_racks_contiguous(descriptors) -> None:
    """The module docstring's own invariant: every non-``None`` rack title appears in exactly ONE
    contiguous run of ``descriptors``. Walks the list once, recording each rack's first
    appearance and failing if that SAME title shows up again after a run of some other value --
    the exact shape an earlier defect produced (a rack silently split into two
    same-titled ``RackBox``es)."""
    seen_and_closed: set[str] = set()
    current = None
    for d in descriptors:
        rack = d.get("rack")
        if rack == current:
            continue
        if current is not None:
            seen_and_closed.add(current)
        assert rack not in seen_and_closed, (
            f"rack {rack!r} reappears in a second, non-contiguous run: {descriptors}")
        current = rack


# --------------------------------------------------------------------------- DeviceBox


def test_device_box_exposes_one_control_per_param_including_the_five_new_ones(
        qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("wtmm2d")
    box = DeviceBox(0, dev, defaults_for(dev))
    qtbot.addWidget(box)

    assert set(box.controls) == {p.name for p in dev.params}
    for new_param in ("smooth", "thresh", "dist2_max", "box_ratio", "similitude"):
        assert new_param in box.controls


def test_device_box_groups_wtmm2d_params_into_pre_and_chaining(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("wtmm2d")
    box = DeviceBox(0, dev, defaults_for(dev))
    qtbot.addWidget(box)

    labels = {lbl.text() for lbl in box.findChildren(QtWidgets.QLabel)}
    assert "PRE" in labels
    assert "CHAINING" in labels


def test_device_box_hides_underscore_prefixed_params(qtbot, registered_builtins):
    """``backproject``'s seven shell-stamped scalars (``_target_crs``,
    ``_target_x0``, ...) must never earn a strip control -- they are system-derived, not something
    a user turns. ``target`` is the one param this box DOES render."""
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("backproject")
    box = DeviceBox(0, dev, defaults_for(dev))
    qtbot.addWidget(box)

    assert set(box.controls) == {"target"}
    for hidden in ("_target_crs", "_target_x0", "_target_dx", "_target_y0", "_target_dy",
                   "_target_nx", "_target_ny"):
        assert hidden not in box.controls


def test_grouped_params_filters_underscore_prefixed_from_every_group(registered_builtins):
    """Direct pin of the mechanism ``_grouped_params`` itself uses: a param whose name starts with
    ``_`` never lands in any group's tuple, including the synthetic single/trailing group --
    same "absent from every tuple" idiom ``param_groups`` filtering already relies on."""
    from dynamix.shell.workflow_zone import _grouped_params

    dev = get_device("backproject")
    groups = _grouped_params(dev)

    rendered = {p.name for _, params in groups for p in params}
    assert rendered == {"target"}


def test_paramchanged_reemits_with_the_param_name(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("wtmm2d")
    box = DeviceBox(0, dev, defaults_for(dev))
    qtbot.addWidget(box)

    with qtbot.waitSignal(box.paramChanged) as sig:
        box.controls["n_oct"].valueChanged.emit(4)
    assert sig.args == ["n_oct", 4]


def test_derived_label_refreshes_when_a_sibling_param_changes(qtbot, registered_builtins):
    """Mirrored from test_shell_units.py's DeviceStrip case:
    ``self._params`` used to be a construction-time snapshot, so flipping ``wavelet`` never
    touched it and the aₘᵢₙ line's λ stayed pinned to whichever wavelet was selected when the box
    was built. ``_on_control_changed`` now writes every edit into ``self._params`` first and
    recomputes ALL derived-reading labels off the refreshed dict, so a wavelet flip -- not just an
    a_min edit -- must update the aₘᵢₙ label's λ immediately."""
    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("wtmm2d")
    box = DeviceBox(0, dev, defaults_for(dev))     # a_min=1.0, wavelet="mexican" (the default)
    qtbot.addWidget(box)

    mexican_lambda = lambda_peak_px(1.0 * 6.0 / 0.86, "mexican", 1)
    gaussian_lambda = lambda_peak_px(1.0 * 6.0 / 0.86, "gaussian", 1)
    assert f"λ {mexican_lambda:.1f} px" in box._derived_labels["a_min"].text()

    box.controls["wavelet"].valueChanged.emit("gaussian")     # as the widget itself would propose

    label_text = box._derived_labels["a_min"].text()
    assert f"λ {gaussian_lambda:.1f} px" in label_text
    assert f"λ {mexican_lambda:.1f} px" not in label_text


def test_bypassed_box_disables_only_its_control_grid_not_the_title_bar(qtbot, registered_builtins):
    """A bypassed box must not cascade-disable its OWN bypass button (or
    the rest of the title bar) -- that would trap it, since re-enabling the button later would inherit a box that can never be clicked back to enabled. The control grid is what
    ``bypassed=True`` actually disables."""
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("scale_select")
    box = DeviceBox(0, dev, defaults_for(dev), bypassed=True)
    qtbot.addWidget(box)
    assert box.isEnabled() is True                 # the frame itself is never disabled
    assert box.name_label.isEnabled() is True       # neither is the title bar
    assert box.controls["scale_idx"].isEnabled() is False   # the control grid IS

    enabled_box = DeviceBox(0, dev, defaults_for(dev), bypassed=False)
    qtbot.addWidget(enabled_box)
    assert enabled_box.controls["scale_idx"].isEnabled() is True


def test_bypass_and_remove_buttons_are_enabled_regardless_of_bypass_state(
        qtbot, registered_builtins):
    """Both buttons are real now -- ``WorkflowZone.chainEdited`` round-trips into
    ``main_window._on_chain_edited``, which rebuilds ``_names``/``_params``/bypass/rack wholesale
    from the payload every time, so there is no stale index-based bookkeeping left for a live
    button to desync (see the module docstring's drag-drop section)."""
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("scale_select")
    for bypassed in (False, True):
        box = DeviceBox(0, dev, defaults_for(dev), bypassed=bypassed)
        qtbot.addWidget(box)
        assert box.bypass_button.isEnabled() is True, bypassed
        assert box.remove_button.isEnabled() is True, bypassed


def test_a_real_click_on_the_remove_button_fires_removerequested(qtbot, registered_builtins):
    """The old regression test inverted: now that the button is wired end to end, a real click
    (``qtbot.mouseClick``, the same real-input simulation the old test used to prove the OPPOSITE)
    must actually fire ``removeRequested``."""
    from PySide6 import QtCore

    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("scale_select")
    box = DeviceBox(0, dev, defaults_for(dev))
    qtbot.addWidget(box)
    box.show()

    received = []
    box.removeRequested.connect(lambda: received.append(True))
    qtbot.mouseClick(box.remove_button, QtCore.Qt.LeftButton)
    assert received == [True]


def test_remove_and_bypass_signals_fire_from_the_buttons_own_api(qtbot, registered_builtins):
    """Driven via the buttons' own API (as the version of this test did) rather than a real
    click -- this one is just confirming the signal wiring itself, independent of the click path
    already covered above."""
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("scale_select")
    box = DeviceBox(0, dev, defaults_for(dev))
    qtbot.addWidget(box)

    with qtbot.waitSignal(box.removeRequested):
        box.removeRequested.emit()

    with qtbot.waitSignal(box.bypassToggled) as sig:
        box.bypass_button.toggle()
    assert sig.args == [True]


# --------------------------------------------------------------------------- SourceBox


def test_set_source_renders_the_fed_by_chip_only_when_given(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import SourceBox

    box = SourceBox()
    qtbot.addWidget(box)
    box.show()                         # isVisible() needs the whole ancestor chain actually shown

    box.set_source("child · refined", None, fed_by="parent", fed_by_layer_id=3)
    assert box.chip_label.text() == "◂ fed by parent"
    assert box.chip_label.isVisible() is True

    box.set_source("plain layer", None)
    assert box.chip_label.text() == ""
    assert box.chip_label.isVisible() is False


def test_set_source_renders_name_and_provenance_text(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import SourceBox

    box = SourceBox()
    qtbot.addWidget(box)

    box.set_source("layer A", {"source": "/data/scan.tif"})

    assert box.name_label.text() == "layer A"
    assert box.provenance_label.text() == "/data/scan.tif"


def test_chip_hover_emits_chipHovered_with_the_parent_layer_id(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import SourceBox

    box = SourceBox()
    qtbot.addWidget(box)
    box.show()                         # enterEvent's own guard reads chip_label.isVisible()
    box.set_source("child · refined", None, fed_by="parent", fed_by_layer_id=9)

    with qtbot.waitSignal(box.chipHovered, timeout=1000) as sig:
        box.enterEvent(QtGui.QEnterEvent(QtCore.QPointF(), QtCore.QPointF(), QtCore.QPointF()))
    assert sig.args == [9]


def test_hover_over_a_box_with_no_chip_emits_nothing(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import SourceBox

    box = SourceBox()
    qtbot.addWidget(box)
    box.set_source("plain layer", None)

    received = []
    box.chipHovered.connect(received.append)
    box.enterEvent(QtGui.QEnterEvent(QtCore.QPointF(), QtCore.QPointF(), QtCore.QPointF()))

    assert received == []


def test_zone_relays_chip_hover_from_its_source_box(qtbot, registered_builtins):
    """``WorkflowZone.chipHovered`` is a straight relay of ``self.source_box.chipHovered`` -- one
    signal ``main_window`` can connect per zone rebuild, without reaching into ``zone.source_box``
    itself."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)

    with qtbot.waitSignal(zone.chipHovered, timeout=1000) as sig:
        zone.source_box.chipHovered.emit(7)
    assert sig.args == [7]


# --------------------------------------------------------------------------- WorkflowZone


def _descriptors(steps):
    return [{"device": name, "params": dict(params), "bypassed": False, "rack": None}
           for name, params in steps]


def test_set_steps_over_default_steps_yields_one_box_per_step(qtbot, registered_builtins):
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors(MainWindow.DEFAULT_STEPS))

    assert zone.strip(0).device.name == "wtmm2d"
    for i, (name, _) in enumerate(MainWindow.DEFAULT_STEPS):
        assert zone.strip(i).device.name == name


def test_select_marks_exactly_one_box_selected(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    zone.select(1)
    assert zone.strip(1).property("selected") == "true"
    assert zone.strip(0).property("selected") != "true"


def test_bypassed_descriptor_renders_a_disabled_control_grid(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    descriptors = _descriptors((("wtmm2d", {}), ("scale_select", {})))
    descriptors[1]["bypassed"] = True
    zone.set_steps(descriptors)

    assert zone.strip(0).controls["n_oct"].isEnabled() is True
    assert zone.strip(1).controls["scale_idx"].isEnabled() is False
    # the box itself, and its title bar, are never disabled by bypass -- see the DeviceBox-level
    # test for why
    assert zone.strip(1).isEnabled() is True
    assert zone.strip(1).name_label.isEnabled() is True


def test_zone_paramchanged_carries_the_step_index(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    with qtbot.waitSignal(zone.paramChanged) as sig:
        zone.strip(1).controls["scale_idx"].valueChanged.emit(2)
    assert sig.args == [1, "scale_idx", 2]


# --------------------------------------------------------------------------- drag-drop


def test_filter_dropped_after_transforms_is_appended_and_emits_chainedited(
        qtbot, registered_builtins):
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}),)))

    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "scale_select"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        zone.dropEvent(event)

    assert event.accepted is True
    assert [d["device"] for d in sig.args[0]] == ["wtmm2d", "scale_select"]
    assert zone.strip(1).device.name == "scale_select"
    # a fresh device drop carries every declared default, not just an override subset
    assert sig.args[0][1]["params"] == defaults_for(get_device("scale_select"))


def test_transform_dropped_after_a_filter_clamps_to_the_transform_section(qtbot, registered_builtins):
    """SUPERSEDES the OLD
    ``test_transform_dropped_after_a_filter_is_refused``, which pinned a SILENT refusal here:
    appending ``wtmm2d`` after ``scale_select`` (a filter) is a transform-after-filter chain, which
    ``Chain.validate`` correctly rejects -- but because every ``DEMO_CHAIN``-seeded zone already
    ends in filters, this exact shape is what most "drop an unused transform" gestures hit on the
    very first try, and the only sign of refusal was a small muted label fading in four seconds, so a refused drop looked like nothing happened. Fixed by smart placement: the drop is no longer refused -- it clamps to the LAST LEGAL
    POSITION (the end of the transform block, :meth:`WorkflowZone._legal_insert_index`) and lands
    there, with a status-tier message naming what happened. ``Chain.validate`` stays the sole
    legality authority; the clamp only ever narrows the candidate index toward a position it was
    always going to accept."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))
    zone.show()                        # isVisible() needs the whole ancestor chain actually shown
    qtbot.wait(10)

    received = []
    zone.chainEdited.connect(lambda d: received.append(d))
    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "wtmm2d"))
    zone.dropEvent(event)

    assert event.accepted is True
    assert len(received) == 1
    assert [d["device"] for d in received[0]] == ["wtmm2d", "wtmm2d", "scale_select"]
    assert [zone.strip(i).device.name for i in range(3)] == \
        ["wtmm2d", "wtmm2d", "scale_select"]           # landed BEFORE the filter, not after it
    assert zone.reading_label.text() == (
        "wtmm2d placed in the transform section -- chains are transforms-then-filters")
    assert zone.reading_label.isVisible() is True


def test_clamped_placement_message_clears_after_the_timer_fires(qtbot, registered_builtins):
    """RENAMED from ``test_refused_drop_warning_clears_after_the
    _timer_fires`` -- this exact scenario (``wtmm2d`` dropped past ``scale_select``) no longer
    refuses at all under smart placement (see the retargeted test just above): it is a SUCCESS,
    clamped into the transform section. A clamped SUCCESS still uses the old four-second fade (only a genuine REFUSAL is persistent now) -- this pins that the fade timer still runs, and
    the label still clears, for this class of message. The body is unchanged from the original
    test; only its name and this note were updated to describe what it now actually exercises."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))
    zone.show()
    qtbot.wait(10)

    zone.dropEvent(_StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "wtmm2d")))
    assert zone.reading_label.text() != ""
    assert zone._warning_timer.interval() == 4000
    assert zone._warning_timer.isActive() is True

    zone._warning_timer.timeout.emit()                 # simulate the 4 s elapsing
    assert zone.reading_label.text() == ""
    assert zone.reading_label.isVisible() is False


# ------------------------------------ Smart placement + persistent refusal


def test_filter_dropped_before_transforms_clamps_to_the_start_of_the_filter_block(
        qtbot, registered_builtins):
    """Item (b): the mirror-image clamp -- a FILTER whose ``before=`` target sits inside the
    transform block is redirected to the START of the filter block (right after the last
    transform), not refused."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}),)))

    target = zone.strip(0)                                  # wtmm2d, the one transform
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(DEVICE_MIME, "modulus_threshold"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        target.dropEvent(event)

    assert [d["device"] for d in sig.args[0]] == ["wtmm2d", "modulus_threshold"]
    assert zone.reading_label.text() == (
        "modulus_threshold placed in the filter section -- chains are transforms-then-filters")


def test_explicit_legal_before_target_is_unchanged_by_the_clamp(qtbot, registered_builtins):
    """Item (c): an explicit, ALREADY-legal ``before=`` target -- a transform inserted directly
    before another existing transform, still fully inside the transform block -- must land exactly
    there. The clamp only ever narrows an ILLEGAL request; it never relocates a legal one, and
    shows no placement message (nothing was clamped)."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("chain_topology", {}), ("scale_select", {}))))

    target = zone.strip(1)                                  # chain_topology, the 2nd transform
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(DEVICE_MIME, "mz_edges"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        target.dropEvent(event)

    assert [d["device"] for d in sig.args[0]] == \
        ["wtmm2d", "mz_edges", "chain_topology", "scale_select"]
    assert zone.reading_label.text() == ""              # no clamp occurred -- nothing to announce


def test_reorder_of_a_transform_into_filter_territory_clamps_the_same_way(
        qtbot, registered_builtins):
    """Item (d): a box REORDER (title-bar drag, ``_REORDER_MIME``) clamps identically to a fresh
    device drop -- dragging ``chain_topology`` onto ``modulus_threshold`` (deep in filter
    territory) is redirected to the end of the transform block, same as :meth:`_handle_device_drop`
    would do for a fresh drop landing in the same illegal spot."""
    from dynamix.shell.workflow_zone import _REORDER_MIME, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors(
        (("wtmm2d", {}), ("chain_topology", {}), ("scale_select", {}), ("modulus_threshold", {}))))

    target = zone.strip(3)                                  # modulus_threshold
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(_REORDER_MIME, "box:1"))  # chain_topology
    with qtbot.waitSignal(zone.chainEdited) as sig:
        target.dropEvent(event)

    assert [d["device"] for d in sig.args[0]] == \
        ["wtmm2d", "chain_topology", "scale_select", "modulus_threshold"]
    assert zone.reading_label.text() == (
        "chain_topology placed in the transform section -- chains are transforms-then-filters")


def test_unknown_mime_payload_is_ignored_not_crashed(qtbot, registered_builtins):
    """Item (e): a genuinely impossible payload -- a mime type none of ``DEVICE_MIME``,
    ``PRESET_MIME`` or ``_REORDER_MIME`` -- is still refused via the existing ``dragEnterEvent``/
    ``dropEvent`` ignore() path (unchanged by this task): nothing is emitted, nothing changes, and
    the event is left un-accepted."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}),)))

    mime = QtCore.QMimeData()
    mime.setData("text/plain", b"not a device")
    event = _StubDropEvent(QtCore.QPoint(0, 0), mime)

    received = []
    zone.chainEdited.connect(lambda d: received.append(d))
    zone.dropEvent(event)

    assert received == []
    assert event.accepted is False
    assert [d["device"] for d in zone._descriptors] == ["wtmm2d"]


def test_refusal_message_persists_past_the_old_4s_window_until_the_next_success(
        qtbot, registered_builtins):
    """Item (f): a genuine refusal -- here, a RACK reorder, which the box-only clamp does not cover
    (a rack can legitimately mix transforms and filters), landing ``wtmm2d`` (a transform, inside
    rack "R") after ``scale_select`` (a filter) -- still hits ``Chain.validate`` and is REVERTED.
    Its message is PERSISTENT: the fade timer never starts at all, so the label is still
    showing well past the OLD 4 s window; only a SUBSEQUENT successful mutation clears it."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import _REORDER_MIME, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps([
        {"device": "wtmm2d", "params": {}, "bypassed": False, "rack": "R"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "R"},
        {"device": "scale_select", "params": {}, "bypassed": False, "rack": None},
    ])
    zone.show()
    qtbot.wait(10)

    received = []
    zone.chainEdited.connect(lambda d: received.append(d))
    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(_REORDER_MIME, "rack:R"))
    zone.dropEvent(event)

    assert received == []
    message = zone.reading_label.text()
    assert message != ""
    assert zone._warning_timer.isActive() is False      # PERSISTENT -- no fade timer scheduled

    qtbot.wait(4100)                                     # well past the OLD 4 s window
    assert zone.reading_label.text() == message         # still showing -- nothing auto-cleared it

    # the next SUCCESSFUL mutation clears it, even though it has no placement message of its own
    event2 = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "hline_length"))
    zone.dropEvent(event2)
    assert zone.reading_label.text() == ""


def test_refusal_flashes_the_zone_red_then_reverts(qtbot, registered_builtins):
    """The drop-refusal flash: ``setProperty("refused", True)`` on the zone itself for
    ``_FLASH_MS``, then reverted -- exercised as a state check (acceptance criterion 1), since the
    visual repaint itself is not something a headless test can observe."""
    from dynamix.shell.workflow_zone import _REORDER_MIME, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps([
        {"device": "wtmm2d", "params": {}, "bypassed": False, "rack": "R"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "R"},
        {"device": "scale_select", "params": {}, "bypassed": False, "rack": None},
    ])

    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(_REORDER_MIME, "rack:R"))
    zone.dropEvent(event)

    assert zone.property("refused") is True
    qtbot.waitUntil(lambda: zone.property("refused") is not True, timeout=1000)


def test_successful_drop_scrolls_the_new_box_into_view(qtbot, registered_builtins, monkeypatch):
    """Item (g): after ANY successful drop, the new box scrolls into view -- a secondary friction point."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}),)))

    calls = []
    monkeypatch.setattr(zone._scroll, "ensureWidgetVisible",
                        lambda w, *a, **k: calls.append(w))

    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "scale_select"))
    with qtbot.waitSignal(zone.chainEdited):
        zone.dropEvent(event)

    assert calls == [zone.strip(1)]


def test_droppplaced_signal_fires_only_on_a_clamped_placement(qtbot, registered_builtins):
    """``dropPlaced`` (the future status-bar wiring): fires ``(device, message)`` when a drop
    had to be clamped, and does NOT fire for an already-legal placement (no clamp, nothing to
    announce beyond the zone's own label)."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    fired = []
    zone.dropPlaced.connect(lambda device, message: fired.append((device, message)))

    # clamped: wtmm2d dropped past the filter
    zone.dropEvent(_StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "wtmm2d")))
    assert fired == [("wtmm2d", "wtmm2d placed in the transform section -- "
                                "chains are transforms-then-filters")]

    fired.clear()
    # not clamped: a filter appended after filters is already legal
    zone.dropEvent(_StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "modulus_threshold")))
    assert fired == []


def test_legal_insert_index_is_a_pure_function_over_the_descriptor_list(qtbot, registered_builtins):
    """Direct pin of the interface the window consumes: ``WorkflowZone._legal_insert_index(device_name,
    requested_index) -> int``, pure descriptor-list math, no Qt event needed to call it."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    assert zone._legal_insert_index("wtmm2d", 2) == 1          # transform clamps back
    assert zone._legal_insert_index("wtmm2d", 0) == 0           # already legal, unchanged
    assert zone._legal_insert_index("modulus_threshold", 0) == 1  # filter clamps forward
    assert zone._legal_insert_index("modulus_threshold", 2) == 2  # already legal, unchanged


def test_preset_dropped_on_the_empty_zone_matches_default_steps(qtbot, registered_builtins):
    """A preset drop creates its OWN named ``RackBox`` -- "Racks" is
    a real browser category, and dragging one of its rows out has to actually produce a rack, or
    nothing in this slice ever reaches one. Device order is preserved; every step carries the
    preset's own name as its rack."""
    from dynamix.shell.browser import PRESET_MIME
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_presets({"WTMM standard": MainWindow.DEFAULT_STEPS})

    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(PRESET_MIME, "WTMM standard"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        zone.dropEvent(event)

    assert [d["device"] for d in sig.args[0]] == [name for name, _ in MainWindow.DEFAULT_STEPS]
    assert all(d["rack"] == "WTMM standard" for d in sig.args[0])
    racks = zone.findChildren(RackBox)
    assert len(racks) == 1
    assert racks[0].title == "WTMM standard"
    assert [zone.strip(i).device.name for i in range(len(MainWindow.DEFAULT_STEPS))] == \
        [name for name, _ in MainWindow.DEFAULT_STEPS]


def test_new_preset_dropped_matches_chain(qtbot, registered_builtins):
    """Coverage for new presets: dropping WTMM + Hölder creates a rack with
    the correct device names and params matching PRESETS."""
    from dynamix.model.presets import PRESETS
    from dynamix.shell.browser import PRESET_MIME
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_presets({"WTMM + Hölder": PRESETS["WTMM + Hölder"]})

    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(PRESET_MIME, "WTMM + Hölder"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        zone.dropEvent(event)

    # Verify the dropped descriptors match the preset exactly
    preset_steps = PRESETS["WTMM + Hölder"]
    descriptors = sig.args[0]

    assert len(descriptors) == len(preset_steps)
    for i, (preset_name, preset_params) in enumerate(preset_steps):
        assert descriptors[i]["device"] == preset_name
        assert descriptors[i]["params"] == preset_params
        assert descriptors[i]["rack"] == "WTMM + Hölder"
        assert descriptors[i]["bypassed"] is False

    # Verify a single RackBox was created with the correct title
    racks = zone.findChildren(RackBox)
    assert len(racks) == 1
    assert racks[0].title == "WTMM + Hölder"


def test_rack_collapse_toggles_member_body_visibility(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    descriptors = _descriptors((("wtmm2d", {}),))
    descriptors += [
        {"device": "chain_length", "params": {}, "bypassed": False, "rack": "post"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "post"},
    ]
    zone.set_steps(descriptors)
    zone.show()
    qtbot.wait(10)

    racks = zone.findChildren(RackBox)
    assert len(racks) == 1
    rack = racks[0]
    assert zone.strip(1).controls["min_len"].isVisible() is True
    assert zone.strip(2).controls["estimator"].isVisible() is True
    # titles stay visible either way -- collapsing hides bodies, not the row of member titles
    assert zone.strip(1).name_label.isVisible() is True
    assert zone.strip(2).name_label.isVisible() is True

    rack.set_collapsed(True)
    assert zone.strip(1).controls["min_len"].isVisible() is False
    assert zone.strip(2).controls["estimator"].isVisible() is False
    assert zone.strip(1).name_label.isVisible() is True
    assert zone.strip(2).name_label.isVisible() is True

    rack.set_collapsed(False)
    assert zone.strip(1).controls["min_len"].isVisible() is True
    assert zone.strip(2).controls["estimator"].isVisible() is True


def test_flatten_order_of_a_racked_arrangement(qtbot, registered_builtins):
    """The exact scenario the design names: [wtmm][RACK(post): chain_length, chain_holder]
    [scale_select] must flatten to that device order -- constructed directly (no drag involved)
    to isolate the rendering/flatten logic from drop-target handling."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    descriptors = _descriptors((("wtmm2d", {}),))
    descriptors += [
        {"device": "chain_length", "params": {}, "bypassed": False, "rack": "post"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "post"},
    ]
    descriptors += _descriptors((("scale_select", {}),))
    zone.set_steps(descriptors)

    assert [zone.strip(i).device.name for i in range(4)] == \
        ["wtmm2d", "chain_length", "chain_holder", "scale_select"]


def test_box_drag_reorders_top_level_boxes(qtbot, registered_builtins):
    """Drop ROUTING is decided by which widget's ``dropEvent`` Qt calls (module docstring's drag-drop section) -- calling ``zone.dropEvent`` directly, as here, is simulating a drop that landed on
    the zone's own top level, appending the dragged box there."""
    from dynamix.shell.workflow_zone import _REORDER_MIME, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors(
        (("wtmm2d", {}), ("scale_select", {}), ("modulus_threshold", {}))))

    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(_REORDER_MIME, "box:1"))   # scale_select
    with qtbot.waitSignal(zone.chainEdited) as sig:
        zone.dropEvent(event)

    assert [d["device"] for d in sig.args[0]] == \
        ["wtmm2d", "modulus_threshold", "scale_select"]


def test_device_drop_into_a_rack_nests(qtbot, registered_builtins):
    """Calling ``rack.dropEvent`` directly simulates a drop that landed on that ``RackBox`` (or
    one of its members, which do not accept drops themselves and so bubble to it) -- see
    ``RackBox``'s own docstring."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    descriptors = _descriptors((("wtmm2d", {}),))
    descriptors.append({"device": "chain_length", "params": {}, "bypassed": False, "rack": "post"})
    zone.set_steps(descriptors)

    rack = zone.findChildren(RackBox)[0]
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(DEVICE_MIME, "chain_holder"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        rack.dropEvent(event)

    got = [(d["device"], d["rack"]) for d in sig.args[0]]
    assert got == [("wtmm2d", None), ("chain_length", "post"), ("chain_holder", "post")]


def test_click_without_movement_selects_via_the_title_bar(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    box1 = zone.strip(1)
    received = []
    box1.selected.connect(lambda: received.append(True))
    box1._title_bar.mousePressEvent(_StubMouseEvent(QtCore.QPoint(5, 5)))
    box1._title_bar.mouseReleaseEvent(_StubMouseEvent(QtCore.QPoint(5, 5)))
    assert received == [True]


def test_click_on_the_box_background_also_selects(qtbot, registered_builtins):
    """Click-to-select is not scoped to the title bar alone -- a click anywhere else on the box's
    own background (not a generated control) still selects, mirroring ``DeviceStrip``'s original
    whole-strip behaviour."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))
    zone.show()
    qtbot.wait(10)

    box1 = zone.strip(1)
    with qtbot.waitSignal(box1.selected):
        qtbot.mouseClick(box1, QtCore.Qt.LeftButton, pos=box1.reading_label.pos())


def test_drag_past_the_threshold_starts_a_box_drag_instead_of_selecting(
        qtbot, registered_builtins, monkeypatch):
    from dynamix.shell.workflow_zone import DeviceBox, WorkflowZone

    # Patched on the CLASS, before construction: ``_TitleBar`` is handed the bound method
    # ``self._start_drag`` resolves to AT CONSTRUCTION time (inside ``DeviceBox.__init__``), so an
    # instance-level ``monkeypatch.setattr(box1, ...)`` afterwards would not be the callable the
    # title bar actually holds -- this also sidesteps ever calling the real ``QDrag.exec()``.
    started = []
    monkeypatch.setattr(DeviceBox, "_start_drag", lambda self: started.append(True))

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    box1 = zone.strip(1)
    received = []
    box1.selected.connect(lambda: received.append(True))

    far = QtWidgets.QApplication.startDragDistance() + 5
    box1._title_bar.mousePressEvent(_StubMouseEvent(QtCore.QPoint(0, 0)))
    box1._title_bar.mouseMoveEvent(_StubMouseEvent(QtCore.QPoint(far, 0)))
    assert started == [True]

    box1._title_bar.mouseReleaseEvent(_StubMouseEvent(QtCore.QPoint(far, 0)))
    assert received == []              # the drag consumed the gesture -- no click-select followed


def test_bypass_toggle_disables_the_control_grid_and_stays_wired_end_to_end(
        qtbot, registered_builtins):
    """Bypass is real now: toggling the button routes through the same ``_commit_or_revert`` gate
    every drop does -- which REBUILDS the zone, so ``box1`` itself
    is torn down and the assertion has to look at the fresh box the rebuild produced, not the
    stale reference."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    box1 = zone.strip(1)
    assert box1.controls["scale_idx"].isEnabled() is True

    with qtbot.waitSignal(box1.bypassToggled):
        box1.bypass_button.toggle()

    fresh = zone.strip(1)
    assert fresh.controls["scale_idx"].isEnabled() is False
    assert zone._descriptors[1]["bypassed"] is True


def test_a_param_edit_survives_a_later_chain_edit(qtbot, registered_builtins):
    """The desync this task closes (module docstring's drag-drop section): a knob turned before a
    drag-drop gesture must not be silently discarded by that gesture's ``chainEdited`` payload."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    zone.strip(1).controls["scale_idx"].valueChanged.emit(3)   # a knob edit, no chain edit yet

    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "modulus_threshold"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        zone.dropEvent(event)

    scale_select = next(d for d in sig.args[0] if d["device"] == "scale_select")
    assert scale_select["params"]["scale_idx"] == 3


# --------------------------------------------------------------------------- bypass ordering


def test_bypassed_filter_still_blocks_a_transform_from_landing_after_it(qtbot, registered_builtins):
    """RETARGETED from the OLD
    ``test_bypassed_filter_still_blocks_a_transform_landing_after_it``, which asserted the drop was
    REFUSED outright. The point still holds -- the clamp is computed over the FULL
    descriptor order, bypassed steps included, so a bypassed step is present-but-INERT for what it
    computes, never for where it sits in the order -- but under smart placement "still
    blocks a transform from landing after it" now means the transform is redirected to land BEFORE
    the bypassed filter, not refused. Un-bypassing the (still legal, still transforms-then-filters)
    arrangement afterward still never raises -- that guarantee is unchanged; only the OBSERVABLE
    outcome of the initial drop is (refusal -> clamp)."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}), ("scale_select", {}))))

    box1 = zone.strip(1)                                   # scale_select
    with qtbot.waitSignal(box1.bypassToggled):
        box1.bypass_button.toggle()
    assert zone._descriptors[1]["bypassed"] is True

    received = []
    zone.chainEdited.connect(lambda d: received.append(d))
    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "wtmm2d"))
    zone.dropEvent(event)

    assert len(received) == 1
    assert [d["device"] for d in received[0]] == ["wtmm2d", "wtmm2d", "scale_select"]
    assert [d["device"] for d in zone._descriptors] == ["wtmm2d", "wtmm2d", "scale_select"]
    assert zone._descriptors[2]["bypassed"] is True         # the bypassed filter itself untouched
    assert zone.strip(1).device.name == "wtmm2d"            # still never lands AFTER the filter

    # and un-bypassing the (still legal) arrangement never raises
    fresh = zone.strip(2)                                    # scale_select, now index 2
    with qtbot.waitSignal(fresh.bypassToggled):
        fresh.bypass_button.toggle()
    assert zone._descriptors[2]["bypassed"] is False


def test_a_device_construction_failure_during_commit_fully_reverts(
        qtbot, registered_builtins, monkeypatch):
    """Rebuilding the boxes (``set_steps``) can raise for reasons ``Chain.validate``
    never sees -- the live trigger is ``chain_classify``'s undeclared FLOAT param bounds crashing
    ``knobs.control_spec`` (routed to its owning task, not fixed here); this proves the ZONE
    survives ANY such failure without depending on that specific bug, by monkeypatching a
    synthetic one instead. Full revert, warning shown, nothing emitted -- same outcome as an
    illegal order."""
    import dynamix.shell.workflow_zone as wz
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    real_make_control = wz.make_control

    def _flaky(spec, value):
        if spec.param.name == "scale_idx":
            raise TypeError("synthetic control-construction failure")
        return real_make_control(spec, value)

    monkeypatch.setattr(wz, "make_control", _flaky)

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("wtmm2d", {}),)))

    received = []
    zone.chainEdited.connect(lambda d: received.append(d))
    event = _StubDropEvent(QtCore.QPoint(10_000, 0), _mime(DEVICE_MIME, "scale_select"))
    zone.dropEvent(event)

    assert received == []
    assert [d["device"] for d in zone._descriptors] == ["wtmm2d"]     # fully reverted
    assert len(zone._boxes) == 1
    assert zone.strip(0).device.name == "wtmm2d"                      # boxes rebuilt to match
    assert zone.reading_label.text() != ""


def test_box_drag_onto_another_box_inserts_before_it(qtbot, registered_builtins):
    """Reorder is insert-at-drop-target now, not append-only -- dropping onto a
    specific ``DeviceBox`` (routed via that box's OWN ``dropEvent``, no pixel hit-testing) inserts
    directly before it."""
    from dynamix.shell.workflow_zone import _REORDER_MIME, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors(
        (("scale_select", {}), ("orientation_wedge", {}), ("modulus_threshold", {}))))

    target = zone.strip(0)                                  # scale_select
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(_REORDER_MIME, "box:2"))  # modulus_threshold
    with qtbot.waitSignal(zone.chainEdited) as sig:
        target.dropEvent(event)

    assert [d["device"] for d in sig.args[0]] == \
        ["modulus_threshold", "scale_select", "orientation_wedge"]


def test_device_drop_onto_a_rack_member_inserts_before_it_nested(qtbot, registered_builtins):
    """Extended into a rack: dropping onto a specific MEMBER box (not the rack's own
    chrome) inserts directly before that member, inheriting its rack membership."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    descriptors = _descriptors((("wtmm2d", {}),))
    descriptors.append({"device": "chain_holder", "params": {}, "bypassed": False, "rack": "post"})
    zone.set_steps(descriptors)

    member = zone.strip(1)                                  # chain_holder, inside "post"
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(DEVICE_MIME, "chain_length"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        member.dropEvent(event)

    got = [(d["device"], d["rack"]) for d in sig.args[0]]
    assert got == [("wtmm2d", None), ("chain_length", "post"), ("chain_holder", "post")]


def test_cross_rack_drag_does_not_split_the_target_rack(qtbot, registered_builtins):
    """The exact repro. Dragging rack B's title bar onto rack A's SECOND (interior) member used to
    splice B into the middle of A's own descriptor span, which ``set_steps``'
    consecutive-grouping then silently rendered as THREE ``RackBox``es -- A split into two
    same-titled boxes around B. A must stay one contiguous ``RackBox``; B lands wholly before (or
    after) it instead."""
    from dynamix.shell.workflow_zone import _REORDER_MIME, RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps([
        {"device": "scale_select", "params": {}, "bypassed": False, "rack": "A"},
        {"device": "orientation_wedge", "params": {}, "bypassed": False, "rack": "A"},
        {"device": "modulus_threshold", "params": {}, "bypassed": False, "rack": "B"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "B"},
    ])

    target = zone.strip(1)                                  # orientation_wedge, A's 2nd member
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(_REORDER_MIME, "rack:B"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        target.dropEvent(event)

    descriptors = sig.args[0]
    _assert_racks_contiguous(descriptors)
    a_positions = [i for i, d in enumerate(descriptors) if d["rack"] == "A"]
    b_positions = [i for i, d in enumerate(descriptors) if d["rack"] == "B"]
    assert set(a_positions).isdisjoint(b_positions)
    assert [descriptors[i]["device"] for i in a_positions] == ["scale_select", "orientation_wedge"]
    assert len(zone.findChildren(RackBox)) == 2             # not three


def test_fresh_device_drop_onto_a_rack_interior_member_snaps_to_the_boundary(
        qtbot, registered_builtins):
    """Test (2) of the same fix: a browser-path device drop (``into_rack=None``,
    routed via the target box's own ``dropEvent``) landing on a rack's INTERIOR member does not
    nest into it -- it snaps to that rack's own start boundary and lands OUTSIDE the rack instead,
    for the same contiguity reason as the rack-drag case above."""
    from dynamix.shell.browser import DEVICE_MIME
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps([
        {"device": "chain_length", "params": {}, "bypassed": False, "rack": "A"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "A"},
    ])

    interior = zone.strip(1)                                # chain_holder, A's 2nd (last) member
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(DEVICE_MIME, "scale_select"))
    with qtbot.waitSignal(zone.chainEdited) as sig:
        interior.dropEvent(event)

    descriptors = sig.args[0]
    _assert_racks_contiguous(descriptors)
    assert [d["device"] for d in descriptors] == ["scale_select", "chain_length", "chain_holder"]
    assert descriptors[0]["rack"] is None                    # landed OUTSIDE the rack
    assert len(zone.findChildren(RackBox)) == 1


def test_into_rack_drop_on_an_interior_member_still_inserts_interior(qtbot, registered_builtins):
    """Test (3) of the same fix, isolating the exception clause directly: when
    the caller explicitly targets the SAME rack an interior ``before`` already belongs to
    (``into_rack`` equal to that rack's own title), the interior position IS honoured -- "the
    dragged item is itself destined for that rack", the fix's own exception. Not reachable through
    today's real routing (``RackBox.dropEvent``/``DeviceBox.dropEvent`` never set both parameters
    at once), so this drives the internal method directly to prove the exception clause itself is
    correct."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps([
        {"device": "chain_length", "params": {}, "bypassed": False, "rack": "A"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "A"},
    ])

    interior = zone.strip(1)                                # chain_holder, A's 2nd member
    index, rack = zone._insertion_target(into_rack="A", before=interior)
    assert (index, rack) == (1, "A")                         # interior position, still joins A


def test_box_reordered_within_its_own_rack_stays_in_it(qtbot, registered_builtins):
    """Not one of the three named tests, but a direct consequence of the same fix
    worth pinning down: a rack member dragged onto a SIBLING within that SAME rack (not the
    first member) must still land inside it, reordered -- the general "interior of a DIFFERENT
    rack" snap only applies across racks, never within one, or the most natural "reorder within a
    rack" gesture would eject the member instead."""
    from dynamix.shell.workflow_zone import _REORDER_MIME, RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps([
        {"device": "chain_length", "params": {}, "bypassed": False, "rack": "A"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "A"},
        {"device": "chain_modulus", "params": {}, "bypassed": False, "rack": "A"},
    ])

    target = zone.strip(2)                                  # chain_modulus, A's LAST member
    event = _StubDropEvent(QtCore.QPoint(0, 0), _mime(_REORDER_MIME, "box:0"))   # chain_length
    with qtbot.waitSignal(zone.chainEdited) as sig:
        target.dropEvent(event)

    descriptors = sig.args[0]
    _assert_racks_contiguous(descriptors)
    assert [d["device"] for d in descriptors] == \
        ["chain_holder", "chain_length", "chain_modulus"]
    assert all(d["rack"] == "A" for d in descriptors)
    assert len(zone.findChildren(RackBox)) == 1


# --------------------------------------------------------------------------- device: new params


class _FakeField:
    def __init__(self, shape=(8, 8)):
        self.values = np.zeros(shape)
        self.frame = None


def test_wtmm2d_threads_smooth_and_thresh_into_the_backend_call(monkeypatch, registered_builtins):
    """The real device test the design asks for: monkeypatch ``run_wtmm2d`` (the actual callee
    ``WTMM2D.compute`` imports and calls -- ``compute`` does ``from dynamix.core.wtmm_backend
    import run_wtmm2d`` freshly on every call, so patching the module attribute before calling
    ``compute`` is what the local import will pick up) and assert the new params reached it."""
    import dynamix.core.wtmm_backend as backend

    captured = {}

    def fake_run_wtmm2d(field, params, *, out_dir=None, backend=None, progress=None, cancel=None):
        captured.update(params)
        return {"chains": [], "extrema": [], "scales": [], "hd_std": None, "hd_cmax": None,
               "npz_path": None, "params": params, "cache_hits": set()}

    monkeypatch.setattr(backend, "run_wtmm2d", fake_run_wtmm2d)

    dev = WTMM2D()
    params = defaults_for(dev)
    params["smooth"] = False
    params["thresh"] = 0.01
    params["similitude"] = 0.6

    dev.compute(_FakeField(), params)

    assert captured["smooth"] is False
    assert captured["thresh"] == 0.01
    assert captured["similitude"] == 0.6
    # the pre-existing params still ride through unchanged
    assert captured["min_chain_len"] == params["min_chain_len"]
    assert captured["dist2_max"] == params["dist2_max"]
    assert captured["box_ratio"] == params["box_ratio"]


def test_wtmm2d_roi_threads_the_same_five_params_into_its_backend_call(
        monkeypatch, registered_builtins):
    from dynamix.devices.wtmm_roi import WTMM2DROI

    captured = {}

    def fake_run_wtmm2d_roi(source, full_dims, roi, params, *, boundary="auto", nodata=None,
                            progress=None):
        captured.update(params)
        return {"chains": [], "extrema": [], "scales": [], "hd_std": None, "hd_cmax": None,
               "npz_path": None, "params": params, "cache_hits": set(), "_shape": (1, 1),
               "_roi": {}, "_roi_margins": [], "_missing_mask": None, "_coi_radii": []}

    import dynamix.roi.halo as halo
    monkeypatch.setattr(halo, "run_wtmm2d_roi", fake_run_wtmm2d_roi)

    dev = WTMM2DROI()
    params = defaults_for(dev)
    params["smooth"] = False
    params["box_ratio"] = 2.0
    params["similitude"] = 0.6

    class _RoiField:
        provenance = {"source": "fake.tif", "full_dims": (32, 32)}
        frame = None

    dev.compute(_RoiField(), params)

    assert captured["smooth"] is False
    assert captured["box_ratio"] == 2.0
    assert captured["similitude"] == 0.6
    assert captured["thresh"] == params["thresh"]
    assert captured["dist2_max"] == params["dist2_max"]


# --------------------------------------------------------------------- DeviceBox.sync_from_result -- data-aware chain-filter readings and hints


@pytest.fixture(scope="module")
def kam64_stack():
    """Real WTMM2D output over the real fixture -- the same one the bug triage measured (73
    chains, OLS Hölder h in [-1.28, 0.69]).

    ``fracint_alpha=0`` pinned (2026-09-15, when the lift started applying on the scalar path):
    the triage numbers above were measured on the unlifted pipeline -- the lift shifts every
    chain's OLS slope by +alpha and re-links chains through the similitude band, which is not
    what these hint/snap regressions pin (tests/test_fracint_scalar.py covers the lift; the
    same pin, same reason, as tests/test_chain_filter_devices.py's ``stack``)."""
    from dynamix.core.rasterfield import RasterField

    field = RasterField.load_npz("tests/fixtures/kam_64.npz")
    return WTMM2D().compute(field, dict(defaults_for(WTMM2D()), fracint_alpha=0.0))


def test_fresh_chain_holder_box_shows_the_kam_64_reading_after_a_result_lands(
        qtbot, registered_builtins, kam64_stack):
    """The design's pinned acceptance number: a freshly dropped chain_holder box (default
    cutoff -3.0, pass-all) reads exactly "kept 73/73 · h ∈ [-1.28, 0.69]" once a real result lands."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("chain_holder", {}),)))
    box = zone.strip(0)
    assert box._params["cutoff"] == -3.0        # the new hard-floor default

    result = box.device.apply(kam64_stack, box._params)
    box.sync_from_result(result)

    assert box.reading_label.text() == "kept 73/73 · h ∈ [-1.28, 0.69]"


def test_hints_rebind_soft_bounds_and_snap_the_value_to_the_data_floor(
        qtbot, registered_builtins, kam64_stack):
    """After hints: knob soft bounds == data range, and the out-of-range default is snapped to
    the data minimum (still pass-all -- the design-B2's "identical behavior... but the very next
    drag-step prunes")."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("chain_holder", {}),)))
    box = zone.strip(0)
    control = box.controls["cutoff"]
    original_step = control._spec.step

    result = box.device.apply(kam64_stack, box._params)
    box.sync_from_result(result)

    assert control._spec.lo == pytest.approx(-1.2845051091543018, abs=1e-6)
    assert control._spec.hi == pytest.approx(0.6942832452452513, abs=1e-6)
    assert control._spec.step != original_step          # re-scaled to the real data spread
    assert box._params["cutoff"] == pytest.approx(control._spec.lo)
    assert control._value == pytest.approx(control._spec.lo)     # display followed the snap
    assert control.isEnabled() is True


def test_rebound_step_reaches_a_real_key_press_on_the_cutoff_control(
        qtbot, registered_builtins, kam64_stack):
    """CRITICAL fix, reproduced against the real device/fixture: the static declared step for chain_holder's cutoff (soft range
    [-1.0, 1.5]) is 0.025; the kam_64-derived rebound step is ~0.0198. A REAL Key_Up on the
    actual control must apply the REBOUND delta, not the static one."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("chain_holder", {}),)))
    box = zone.strip(0)
    control = box.controls["cutoff"]
    static_step = control._spec.step
    assert static_step == pytest.approx(0.025)

    result = box.device.apply(kam64_stack, box._params)
    box.sync_from_result(result)          # rebinds + snaps to the data floor (-1.2845...)
    rebound_step = control._spec.step
    assert rebound_step == pytest.approx(0.019787883543995532, abs=1e-9)

    value_before = control._value
    with qtbot.waitSignal(control.valueChanged) as sig:
        qtbot.keyClick(control, QtCore.Qt.Key_Up)
    delta = sig.args[0] - value_before
    assert delta == pytest.approx(rebound_step)
    assert delta != pytest.approx(static_step)


def test_snap_never_fires_when_the_current_value_is_above_the_data_ceiling(
        qtbot, registered_builtins, kam64_stack):
    """The snap predicate is pass-all-direction ONLY
    (``current < lo``). A deliberately-set HIGH cutoff (above the data's own ceiling -- e.g. a
    non-terminal chain_holder box whose own cutoff is stricter than what the terminal result's
    narrower range would suggest) must never be silently yanked down to ``hi``."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("chain_holder", {"cutoff": 2.0, "estimator": "ols"}),)))
    box = zone.strip(0)
    assert box._params["cutoff"] == 2.0

    # A result whose OLS range tops out well below 2.0 (hard max is 3.0) -- if the snap fired in
    # both directions, this would drag a deliberately strict cutoff back down into the data.
    result = {"chains": kam64_stack["chains"], "_chains_dropped": 0}

    box.sync_from_result(result)

    assert box._params["cutoff"] == 2.0                     # untouched
    assert box.controls["cutoff"]._value == 2.0
    assert box.controls["cutoff"]._spec.hi == pytest.approx(0.6942832452452513, abs=1e-6)


def test_one_soft_step_increase_reduces_kept_count_and_drawn_geometry(
        qtbot, registered_builtins, kam64_stack):
    """The design-B2's own acceptance shape: after hints, ONE soft-step increase of the knob visibly
    prunes chains -- both the kept count and the drawn V-trail geometry point count drop."""
    from dynamix.shell.workflow_zone import WorkflowZone

    from dynamix.shell.canvas import vchain_trails

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("chain_holder", {}),)))
    box = zone.strip(0)

    # Selection contract: on a product-stamped stack, apply() narrows
    # index selections and materialize_selection lands the honest dicts -- in the app, resolve()
    # does the materialization once per redraw.
    from dynamix.core.chain_product import materialize_selection

    result_before = materialize_selection(box.device.apply(kam64_stack, box._params))
    box.sync_from_result(result_before)      # hints land; cutoff snaps to the data floor
    xs_before, _ = vchain_trails(result_before["chains"])

    control = box.controls["cutoff"]
    stepped = control._value + control._spec.step        # ONE soft step, at the re-scaled size
    box._on_control_changed("cutoff", stepped)

    result_after = materialize_selection(box.device.apply(kam64_stack, box._params))
    xs_after, _ = vchain_trails(result_after["chains"])

    assert len(result_before["chains"]) == 73
    assert len(result_after["chains"]) < len(result_before["chains"])
    assert xs_after.size < xs_before.size


def test_metric_unavailable_disables_the_control_and_reads_unavailable(
        qtbot, registered_builtins):
    """Chains present but missing what the Hölder estimate needs (no ``log2_mod``) -- EQSelect's
    own "unavailable, disable, don't apply" degradation tier, not a crash."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("chain_holder", {}),)))
    box = zone.strip(0)

    chains_without_log2_mod = [{"x": np.array([0, 1]), "y": np.array([0, 1]),
                                "log2_scales": np.array([0.0, 1.0])}]
    box.sync_from_result({"chains": chains_without_log2_mod})

    assert box.reading_label.text() == "unavailable — recompute the transform to enable"
    assert box.controls["cutoff"].isEnabled() is False


def test_hints_never_touch_the_declared_param_or_leak_into_the_descriptor(
        qtbot, registered_builtins, kam64_stack):
    """Display-only doctrine (Param's own docstring: soft bounds are "never enforced... must not
    affect cache_key"). ``data_hints`` must never mutate the declared ``Param`` (what a
    Transform's ``cache_key`` and ``validate_params`` actually read), and the one legitimate param
    edit it triggers (the snap) must land through the normal edit path -- never as an extra key."""
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors((("chain_holder", {}),)))
    box = zone.strip(0)
    cutoff_param = [p for p in box.device.params if p.name == "cutoff"][0]
    original_soft_min, original_soft_max = cutoff_param.soft_min, cutoff_param.soft_max

    result = box.device.apply(kam64_stack, box._params)
    box.sync_from_result(result)

    assert cutoff_param.soft_min == original_soft_min
    assert cutoff_param.soft_max == original_soft_max
    assert set(zone._descriptors[0]["params"]) <= {"cutoff", "estimator"}


# --------------------------------------------------------------------------- Rack ×/undo


def _two_rack_descriptors():
    """``[wtmm2d][RACK A: chain_length, chain_holder][RACK B: scale_select,
    modulus_threshold]`` -- one leading transform, two independent racks of filters, so a
    removal/undo test can assert the OTHER rack (and the leading transform) are untouched."""
    return [
        {"device": "wtmm2d", "params": dict(defaults_for(get_device("wtmm2d"))),
         "bypassed": False, "rack": None},
        {"device": "chain_length", "params": {}, "bypassed": False, "rack": "A"},
        {"device": "chain_holder", "params": {}, "bypassed": False, "rack": "A"},
        {"device": "scale_select", "params": {}, "bypassed": False, "rack": "B"},
        {"device": "modulus_threshold", "params": {}, "bypassed": False, "rack": "B"},
    ]


def test_rack_remove_button_removes_only_that_racks_span(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_two_rack_descriptors())

    rack_a = next(r for r in zone.findChildren(RackBox) if r.title == "A")
    with qtbot.waitSignal(zone.chainEdited) as sig:
        rack_a.remove_button.click()

    assert [d["device"] for d in sig.args[0]] == \
        ["wtmm2d", "scale_select", "modulus_threshold"]
    assert [d["rack"] for d in zone._descriptors] == [None, "B", "B"]
    assert zone.findChildren(RackBox)[0].title == "B"    # rack B, and only rack B, remains


def test_two_rack_removals_produce_two_lifo_undos_with_byte_identical_descriptors(
        qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    original = _two_rack_descriptors()
    zone.set_steps(original)
    after_a_removed = [d for d in original if d["rack"] != "A"]

    rack_a = next(r for r in zone.findChildren(RackBox) if r.title == "A")
    rack_a.remove_button.click()
    assert zone._descriptors == after_a_removed

    rack_b = next(r for r in zone.findChildren(RackBox) if r.title == "B")
    rack_b.remove_button.click()
    assert [d["device"] for d in zone._descriptors] == ["wtmm2d"]

    zone.undo_removal()                      # LIFO: undoes B's removal first
    assert zone._descriptors == after_a_removed

    zone.undo_removal()                      # then A's
    assert zone._descriptors == original

    zone.undo_removal()                      # stack now empty -- silent no-op
    assert zone._descriptors == original


def test_rack_removal_status_message_names_title_count_and_undo_shortcut(
        qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_two_rack_descriptors())

    rack_a = next(r for r in zone.findChildren(RackBox) if r.title == "A")
    rack_a.remove_button.click()

    assert zone.reading_label.text() == "Removed rack A (2 devices) — ⇧⌘Z restores"


def test_rack_removed_signal_fires_with_title_and_the_same_message_as_the_zone_label(
        qtbot, registered_builtins):
    """``rackRemoved`` -- the future status-bar wiring, ``dropPlaced``-shaped -- fires once,
    with the exact same message the zone's own ``reading_label`` shows."""
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_two_rack_descriptors())

    rack_a = next(r for r in zone.findChildren(RackBox) if r.title == "A")
    with qtbot.waitSignal(zone.rackRemoved) as sig:
        rack_a.remove_button.click()

    assert sig.args == ["A", zone.reading_label.text()]


def test_undo_removal_is_a_silent_no_op_with_nothing_removed_yet(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_two_rack_descriptors())
    before = [dict(d) for d in zone._descriptors]

    zone.undo_removal()

    assert zone._descriptors == before


def test_q_warning_lights_up_and_clears_in_the_box(qtbot, registered_builtins):
    """The q knob's warning line follows the knob live: empty at the default, a ⚠ line once
    q_mexican's q drops below 0, empty again when it comes back."""
    from dynamix.shell.workflow_zone import DeviceBox

    dev = get_device("holder_multiaffine")
    box = DeviceBox(0, dev, dict(defaults_for(dev), wavelet="q_mexican"))
    qtbot.addWidget(box)
    label = box._derived_labels["q_tsallis"]
    assert label.text() == ""
    box.controls["q_tsallis"].valueChanged.emit(-0.5)
    assert "⚠" in label.text()
    box.controls["q_tsallis"].valueChanged.emit(0.5)
    assert label.text() == ""
