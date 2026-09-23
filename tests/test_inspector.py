# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.inspector -- the per-source floating inspector window.

Offscreen widget tests, the WIDGET half only (test_shell_canvas.py's split convention): a bare
``InspectorWindow`` driven directly against a provenance node, no ``MainWindow``, no worker, no
resolve -- the wiring lives elsewhere.

The flags assertions are the point of the first two tests. The window class is the
``_LassoRegionOverlay`` precedent (``shell/arrangement/view.py``) MINUS its three click-through
settings, and a future copy-paste of that overlay's constructor would silently reintroduce them:
an inspector the user cannot click is an inspector that cannot adjust anything, which is the opposite of an inspector that stays in front while parameters are adjusted. So the divergence is asserted, not commented.

The stub-device registry fixture is copied from tests/test_provenance.py rather than imported --
tests/ is not a package, and duplicating a small fixture is the established pattern here.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore

from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import register_device
from dynamix.model.project import Project
from dynamix.model.provenance import provenance_tree, raster_key, sublayer_key
from dynamix.shell.canvas import Canvas
from dynamix.shell.inspector import (
    NO_RESULT_TEXT, SCALE_NOT_SHOWN_LIVE_TEXT, InspectorWindow)


@pytest.fixture
def _registry(clean_registry, stub_transform, stub_filter):
    register_device(stub_transform)
    register_device(stub_filter)


def _reading(idx: int) -> str:
    """Stands in for ``MainWindow._scale_reading`` -- the window is what knows the units."""
    return f"a = {idx}"


@pytest.fixture
def project(_registry):
    p = Project(title="t")
    source = p.add_source("/data/kam.npz")
    p.add_layer("kam", source.source_id,
                Chain((DeviceRef("t", {"scale": 2}), DeviceRef("t", {"scale": 4}),
                       DeviceRef("f", {"cut": 0.3}))))
    return p


@pytest.fixture
def node(project):
    return provenance_tree(project)[0]


@pytest.fixture
def window(qtbot, node):
    w = InspectorWindow(node.source_id, node.label, node, _reading)
    qtbot.addWidget(w)
    return w


# --------------------------------------------------------------------------- the window itself

def test_window_is_a_top_level_tool_that_stays_on_top(window):
    flags = window.windowFlags()
    assert window.isWindow()
    assert flags & QtCore.Qt.WindowType.WindowType_Mask == QtCore.Qt.WindowType.Tool
    assert flags & QtCore.Qt.WindowType.WindowStaysOnTopHint


def test_window_is_not_input_transparent(window):
    """The deliberate divergence from ``arrangement/view.py``'s overlay: an overlay wants clicks
    to pass through, an inspector must RECEIVE them."""
    flags = window.windowFlags()
    assert not (flags & QtCore.Qt.WindowType.WindowTransparentForInput)
    assert window.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents) is False
    assert window.testAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating) is False


def test_hosts_the_existing_canvas_widget(window):
    """A proven widget was MOVED into the frame, not a second raster view authored beside it."""
    assert isinstance(window.canvas, Canvas)


# --------------------------------------------------------------------------- the sub-layer tree

def test_sublayer_tree_rows_match_the_provenance_tree_order(window, node):
    layer = node.layers[0]
    assert window.row_keys() == [raster_key(node.source_id),
                                 sublayer_key(layer.layer_id, 0),
                                 sublayer_key(layer.layer_id, 1),
                                 sublayer_key(layer.layer_id, 2)]
    # ... and the NESTING is the provenance: raster on top, the layer under it, its products
    # under the layer, in the chain's own step order.
    raster_item = window.tree.topLevelItem(0)
    assert window.tree.topLevelItemCount() == 1
    layer_item = raster_item.child(0)
    assert layer_item.text(0) == "kam"
    assert [layer_item.child(i).text(0) for i in range(layer_item.childCount())] == ["t", "t", "f"]


def test_toggling_a_sublayer_row_emits_its_key(window, node):
    key = sublayer_key(node.layers[0].layer_id, 0)
    seen: list[tuple[str, str, bool]] = []
    window.subLayerToggled.connect(lambda sid, k, vis: seen.append((sid, k, vis)))

    window.rows[key].setCheckState(0, QtCore.Qt.CheckState.Unchecked)

    assert seen == [(node.source_id, key, False)]


def test_opacity_spin_emits_the_new_value(window, node):
    key = sublayer_key(node.layers[0].layer_id, 1)
    seen: list[tuple[str, str, float]] = []
    window.subLayerOpacityChanged.connect(lambda sid, k, v: seen.append((sid, k, v)))

    window.opacity_spins[key].setValue(0.4)

    assert len(seen) == 1
    assert seen[0][:2] == (node.source_id, key)
    assert seen[0][2] == pytest.approx(0.4)


# --------------------------------------------------------------------------- close + the master

def test_close_emits_closed_with_the_source_id(window, node):
    """Closing the window and un-toggling the layer-list button are ONE state;
    this signal is the half the toggle listens to."""
    seen: list[str] = []
    window.closed.connect(seen.append)

    window.close()

    assert seen == [node.source_id]


def test_follow_master_defaults_checked(window):
    assert window.follow_button.isChecked() is True


# ------------------------------------------------- The master track
# "Each dataset's inspector keeps its own scale control ... a layer named 'master' carries a
# transport that drives every inspector that follows it." This file owns the INSPECTOR half:
# what ``set_scale_index`` / ``set_n_scales`` / ``follow_master`` do on their own, with no
# ``MainWindow`` anywhere. ``tests/test_shell_window.py`` owns the wired half (who drives whom).


def _result(n_scales: int) -> dict:
    """A WTMM-SHAPED result carrying its OWN scale count.

    ``scales`` is what makes the inspector's slider range data-derived (sec 8 rule 5) rather than
    a constant; ``extrema``/``_shape`` are what ``show_result``'s draw branch reads, kept real so
    the range assertion runs over the same call the window actually makes."""
    idx = np.arange(4, dtype=np.int64)
    layer = {"x": idx, "y": idx, "mod": np.linspace(1.0, 0.1, 4),
             "arg": np.linspace(0.0, np.pi, 4), "line_id": np.full(4, -1, dtype=np.int64)}
    return {"scales": [2.0 * (k + 1) for k in range(n_scales)], "extrema": [layer],
            "_shape": (8, 8), "params": {}}


class _Renderable:
    """Stands in for ``engine.resolve.Renderable`` -- ``show_result`` reads ``.result`` only."""

    def __init__(self, result: dict):
        self.result = result


def test_following_inspector_adopts_silently(window, node):
    """``set_scale_index`` routes through ``Transport.sync_to`` (the SILENT adopt) on purpose:
    emitting would echo the master's index straight back at whoever just applied it, which is
    the exact bug ``sync_to``'s own docstring records."""
    window.set_n_scales(4)
    seen: list[tuple[str, int]] = []
    window.scaleChanged.connect(lambda sid, i: seen.append((sid, i)))

    window.set_scale_index(2)

    assert window.follow_master is True
    assert window.transport.slider.value() == 2
    assert window.transport.reading_label.text() == _reading(2)   # the READING moved with it
    assert seen == []


def test_inspector_scale_range_comes_from_the_result_not_a_constant(window):
    """Sec 8 rule 5: the range is DATA-derived. A 3-scale result gives slider maximum 2, a
    5-scale one gives 4 -- and the re-range is not this window's own gesture, so it is silent."""
    assert window.transport.slider.maximum() == 0        # nothing known yet
    seen: list[tuple[str, int]] = []
    window.scaleChanged.connect(lambda sid, i: seen.append((sid, i)))

    window.show_result(_Renderable(_result(3)), active=True)

    assert window.transport.slider.maximum() == 2
    assert window.transport.clock.n_scales == 3

    window.show_result(_Renderable(_result(5)), active=True)

    assert window.transport.slider.maximum() == 4
    assert window.transport.clock.n_scales == 5
    assert seen == []


def test_un_following_is_readable_off_the_window(window):
    """``follow_master`` is the property ``MainWindow._drive_follower_inspectors`` asks, so the
    button's state and the answer it gives can never be two different opinions."""
    window.follow_button.setChecked(False)

    assert window.follow_master is False


# --------------------------- The own scale control, routed or said
# The window half of "either route it or say it, but do not ship a reading that contradicts the
# picture". Routing is FREE exactly where the held result still carries the whole stack; where
# the chain already reduced it to one layer (``scale_select``), the picture cannot move and the
# window names the state and the fix instead (sec 8 rule 6).


def _stack_result(n_scales: int) -> dict:
    """A result that still carries the WHOLE stack -- one extrema layer per scale, with a
    different point count each, so a redraw at another index is visible in the drawn data.

    This is what an inspector holds when ``scale_select`` is bypassed or absent from the chain;
    ``_result`` above is the post-``scale_select`` shape (ONE layer), which is what the shipped
    ``DEMO_CHAIN``/``STUB_CHAIN`` produce."""
    layers = []
    for k in range(n_scales):
        m = max(4, 12 - 2 * k)
        idx = np.arange(m, dtype=np.int64)
        layers.append({"x": idx, "y": idx, "mod": np.linspace(1.0, 0.1, m),
                       "arg": np.linspace(0.0, np.pi, m),
                       "line_id": np.full(m, -1, dtype=np.int64)})
    return {"scales": [2.0 * (k + 1) for k in range(n_scales)], "extrema": layers,
            "_shape": (16, 16), "params": {}}


def test_a_multi_scale_result_redraws_at_the_requested_index(window):
    """Route where it is free: no resolve, no compute, no second worker -- the held result is
    re-indexed, which is the whole of what ``scale_select`` does anyway."""
    window.show_result(_Renderable(_stack_result(4)), active=True)
    assert window.picture_scale_idx == 0
    drawn_at_zero = len(window.canvas.extrema_item.getData()[0])

    window.set_scale_index(3)                       # the master driving a follower

    assert window.transport.slider.value() == 3
    assert window.picture_scale_idx == 3
    assert len(window.canvas.extrema_item.getData()[0]) != drawn_at_zero
    # nothing to confess: the reading and the picture name the same scale
    assert window.status_label.text() == NO_RESULT_TEXT


def test_a_single_scale_result_cannot_move_and_the_window_says_so(window):
    """The shipped chain's shape. ``scale_select`` already reduced the stack to ONE layer, so
    the requested scale is simply not in this result -- and a reading that silently named it
    while the canvas showed another is the contradiction this fix exists to remove."""
    window.show_result(_Renderable(_result(4)), active=True)
    drawn = window.canvas.extrema_item.getData()[0].copy()

    window.transport.slider.setValue(2)             # this window's own control, dragged
    window.redraw_at(2)                             # what MainWindow routes that drag into

    np.testing.assert_array_equal(window.canvas.extrema_item.getData()[0], drawn)
    assert window.picture_scale_idx == 0
    text = window.status_label.text()
    assert SCALE_NOT_SHOWN_LIVE_TEXT.format(control=2, picture=0) in text
    assert text.startswith(NO_RESULT_TEXT)          # MainWindow's own sentence is not clobbered


def test_set_scale_index_clamps_to_this_windows_own_range(window):
    """``Transport.sync_to`` is unclamped on the receiving side (``SweepClock.scrub`` assigns
    ``_index`` outright and the label is formatted from whatever it is handed), so a master
    index past THIS window's range would leave the slider at the end while the reading named a
    scale it cannot show -- and the next playback tick would advance from out of range."""
    window.set_n_scales(3)

    window.set_scale_index(7)

    assert window.transport.slider.value() == 2
    assert window.transport.clock.index == 2
    assert window.transport.reading_label.text() == _reading(2)


# ----------------------------------------------- What a RESTORE hands back in
# The window half of persistence. ``MainWindow`` restores at ``load_field``'s tail, which is
# BEFORE the first result has landed -- so a remembered scale index arrives while the transport
# still knows one scale, and a remembered row state must land without the window announcing
# changes nobody made.


def test_a_restored_scale_index_waits_for_the_range_to_arrive(window):
    """The one-shot restore runs before any result: the transport still offers a single scale,
    so a remembered index has nowhere to go yet and is applied when the range is."""
    assert window.transport.slider.maximum() == 0

    window.restore_scale_index(2)

    assert window.transport.slider.value() == 0          # nothing to move to, yet

    window.show_result(_Renderable(_result(4)), active=True)

    assert window.transport.slider.value() == 2
    assert window.transport.reading_label.text() == _reading(2)


def test_a_restored_scale_index_is_applied_once_not_on_every_result(window):
    window.show_result(_Renderable(_result(4)), active=True)
    window.restore_scale_index(3)
    assert window.transport.slider.value() == 3          # the range is already known

    window.transport.slider.setValue(1)                  # the user moves it afterwards
    window.show_result(_Renderable(_result(4)), active=True)

    assert window.transport.slider.value() == 1          # ... and the restore does not fight it


def test_restoring_a_sublayer_row_is_silent(window, node):
    """A restore is not a gesture: announcing it would write the state back as though the user
    had just made it, and every open window would report N sub-layer changes on startup."""
    key = window.row_keys()[1]
    toggled: list[tuple] = []
    dimmed: list[tuple] = []
    window.subLayerToggled.connect(lambda *a: toggled.append(a))
    window.subLayerOpacityChanged.connect(lambda *a: dimmed.append(a))

    window.set_sublayer_state(key, visible=False, opacity=0.4)

    assert window.rows[key].checkState(0) == QtCore.Qt.CheckState.Unchecked
    assert window.opacity_spins[key].value() == pytest.approx(0.4)
    assert window.sublayer_states()[key] == (False, pytest.approx(0.4))
    assert toggled == [] and dimmed == []


def test_a_remembered_row_this_project_no_longer_has_is_ignored(window):
    """Chains change: a key from a chain step that has since been removed names no row here, and
    a restore must drop it rather than raise (a preference cannot be a fault)."""
    window.set_sublayer_state("999:7", visible=False, opacity=0.2)

    assert "999:7" not in window.rows
