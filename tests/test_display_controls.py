# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Per-layer overlay display controls -- opacity, point size, line width as view state.

This extends to the MODEL half of per-layer color prefs: five more
``ui.*`` keys (``colormap``/``color_hchain``/``color_vtrail``/``color_extrema``/``show_trails``),
NOT ``_DISPLAY_PARAMS`` knobs (no float min/max makes sense for a colormap name, a hex string, or
a bool) but read by :func:`_display_style_of` the same tolerant way, and pushed onto the canvas
through the same ``_on_display_style_changed``/``_sync_display_controls`` pair. The right panel builds the combo/swatch/checkbox controls that emit these through ``styleChanged``; these tests
exercise the pipeline directly, as ``_on_display_style_changed`` already lets the three knobs be
exercised above.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.pointset import PointSet
from dynamix.model.layer import Layer
from dynamix.shell.canvas import EXTREMA_COLOR, HCHAIN_COLOR, VTRAIL_COLOR
from dynamix.shell.main_window import _display_style_of

from tests.test_shell_window import (STUB_CHAIN, _FIELD, loaded, stub_devices,  # noqa: F401
                                     window)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def test_display_knob_persists_on_tags_and_restyles_canvas(loaded):
    loaded._on_display_style_changed("opacity", 0.4)
    loaded._on_display_style_changed("point_size", 7.0)
    assert loaded.layer.tags["ui.opacity"] == repr(0.4)
    assert loaded.layer.tags["ui.point_size"] == repr(7.0)
    assert loaded.canvas.extrema_item.opacity() == pytest.approx(0.4)
    # the items that draw in the raster view carry the opacity too
    assert loaded.canvas.extrema_raster_item.opacity() == pytest.approx(0.4)
    assert loaded.canvas.arrow_item.opacity() == pytest.approx(0.4)
    assert loaded.canvas.extrema_item.opts["size"] == pytest.approx(7.0)


def test_line_width_applies_to_every_polyline_overlay(loaded):
    loaded._on_display_style_changed("line_width", 2.5)
    for item in (loaded.canvas.hchain_item, loaded.canvas.vtrail_item,
                 loaded.canvas.seam_item, loaded.canvas.ghost_item):
        assert item.opts["pen"].widthF() == pytest.approx(2.5)
    # the ghost keeps its dashes -- styling never overwrites identity
    from PySide6 import QtCore
    assert loaded.canvas.ghost_item.opts["pen"].style() == QtCore.Qt.DashLine


def test_style_round_trips_across_layer_switches(qtbot, loaded):
    first = loaded.layer
    loaded._on_display_style_changed("opacity", 0.3)
    second = loaded.project.add_layer("plain", first.source_id, first.chain)
    loaded.add_layer_row(second, loaded._fields[first.layer_id])
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(second.layer_id)
    assert loaded.canvas.extrema_item.opacity() == pytest.approx(1.0)   # fresh layer: defaults
    assert loaded.canvas.extrema_raster_item.opacity() == pytest.approx(1.0)
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(first.layer_id)
    assert loaded.canvas.extrema_item.opacity() == pytest.approx(0.3)   # stored style restored
    assert loaded.canvas.extrema_raster_item.opacity() == pytest.approx(0.3)
    assert loaded._display_controls["opacity"]._value == pytest.approx(0.3)


def test_garbled_tag_falls_back_to_default(loaded):
    loaded.layer.tags["ui.opacity"] = "not-a-number"
    loaded._sync_display_controls(loaded.layer)
    assert loaded.canvas.extrema_item.opacity() == pytest.approx(1.0)
    assert loaded.canvas.extrema_raster_item.opacity() == pytest.approx(1.0)


def test_styling_a_locked_layer_is_allowed(loaded):
    loaded.layer.tags["ui.lock"] = "1"
    loaded._on_display_style_changed("opacity", 0.5)
    assert loaded.layer.tags["ui.opacity"] == repr(0.5)
    assert loaded.canvas.extrema_item.opacity() == pytest.approx(0.5)
    assert loaded.canvas.extrema_raster_item.opacity() == pytest.approx(0.5)


# ------------------------------------------------------------------------------------------------
# The five new ui.* keys -- _display_style_of's own tolerant defaults.


def test_new_display_keys_default_when_tags_absent():
    """A layer with no tags at all -- the state every existing project's layer opens in the first
    time it is read under this task -- gets the canvas's OWN constants (converted to hex) and
    ``show_trails`` off, exactly what the canvas already drew before these keys existed."""
    layer = Layer(layer_id=1, name="A", source_id="mem:a")

    style = _display_style_of(layer)

    assert style["colormap"] == "viridis"
    assert style["color_hchain"] == _hex(HCHAIN_COLOR)
    assert style["color_vtrail"] == _hex(VTRAIL_COLOR)
    assert style["color_extrema"] == _hex(EXTREMA_COLOR)
    assert style["show_trails"] is False


def test_layer_with_only_legacy_tags_gets_new_key_defaults():
    """A pre-Task-8 project: its layer already recorded ``ui.opacity`` (the old three knobs) but
    has never heard of the five new keys -- they must still default cleanly, not KeyError, so
    every existing project opens unchanged."""
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    layer.tags["ui.opacity"] = repr(0.5)

    style = _display_style_of(layer)

    assert style["opacity"] == pytest.approx(0.5)
    assert style["colormap"] == "viridis"
    assert style["color_vtrail"] == _hex(VTRAIL_COLOR)
    assert style["show_trails"] is False


def test_new_display_keys_round_trip_from_tags():
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    layer.tags["ui.colormap"] = "plasma"
    layer.tags["ui.color_hchain"] = "#112233"
    layer.tags["ui.color_vtrail"] = "#445566"
    layer.tags["ui.color_extrema"] = "#778899"
    layer.tags["ui.show_trails"] = "True"

    style = _display_style_of(layer)

    assert style["colormap"] == "plasma"
    assert style["color_hchain"] == "#112233"
    assert style["color_vtrail"] == "#445566"
    assert style["color_extrema"] == "#778899"
    assert style["show_trails"] is True


def test_garbled_new_tags_fall_back_to_defaults():
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    layer.tags["ui.color_hchain"] = "not-a-color"
    layer.tags["ui.color_vtrail"] = "#gg0011"          # not hex digits
    layer.tags["ui.color_extrema"] = "#1122"           # wrong length
    layer.tags["ui.show_trails"] = "yes-please"        # only the literal "True" means true

    style = _display_style_of(layer)

    assert style["color_hchain"] == _hex(HCHAIN_COLOR)
    assert style["color_vtrail"] == _hex(VTRAIL_COLOR)
    assert style["color_extrema"] == _hex(EXTREMA_COLOR)
    assert style["show_trails"] is False


def test_empty_colormap_tag_falls_back_to_default():
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    layer.tags["ui.colormap"] = ""

    assert _display_style_of(layer)["colormap"] == "viridis"


# ------------------------------------------------------------------------------------------------
# The new keys pushed onto the canvas through _on_display_style_changed.


def test_colormap_change_persists_and_swaps_canvas_lut(loaded):
    before = loaded.canvas.image_item.lut

    loaded._on_display_style_changed("colormap", "plasma")

    assert loaded.layer.tags["ui.colormap"] == "plasma"
    after = loaded.canvas.image_item.lut
    assert not np.array_equal(before, after)


def test_unknown_colormap_name_does_not_crash_and_keeps_prior_lut(loaded):
    before = loaded.canvas.image_item.lut

    loaded._on_display_style_changed("colormap", "not-a-real-colormap")

    assert loaded.layer.tags["ui.colormap"] == "not-a-real-colormap"    # tag still recorded
    assert np.array_equal(loaded.canvas.image_item.lut, before)         # LUT untouched


def test_color_hchain_change_updates_canvas_pen(loaded):
    loaded._on_display_style_changed("color_hchain", "#112233")

    assert loaded.layer.tags["ui.color_hchain"] == "#112233"
    assert loaded.canvas.hchain_item.opts["pen"].color().name() == "#112233"


def test_color_vtrail_change_updates_canvas_pen(loaded):
    loaded._on_display_style_changed("color_vtrail", "#445566")

    assert loaded.canvas.vtrail_item.opts["pen"].color().name() == "#445566"


def test_color_extrema_change_updates_canvas_brush(loaded):
    loaded._on_display_style_changed("color_extrema", "#778899")

    assert loaded.canvas.extrema_item.opts["brush"].color().name() == "#778899"


def test_overlay_color_change_preserves_the_current_line_width(loaded):
    """``set_overlay_colors`` rebuilds fresh pens -- they must not silently reset to pyqtgraph's
    own default width, dropping whatever ``set_display_style`` last recorded."""
    loaded._on_display_style_changed("line_width", 2.5)

    loaded._on_display_style_changed("color_hchain", "#112233")

    assert loaded.canvas.hchain_item.opts["pen"].widthF() == pytest.approx(2.5)


def test_show_trails_true_persists_and_survives_a_fresh_set_result(loaded):
    loaded._on_display_style_changed("show_trails", True)

    assert loaded.layer.tags["ui.show_trails"] == "True"
    assert loaded.canvas.vtrail_item.isVisible() is True

    loaded._reresolve()      # a fresh set_result -- the toggle must not have been a one-shot poke

    assert loaded.canvas.vtrail_item.isVisible() is True


def test_show_trails_false_is_the_stored_default(loaded):
    assert loaded.layer.tags.get("ui.show_trails") is None
    assert loaded.canvas.vtrail_item.isVisible() is False


def test_new_keys_round_trip_across_layer_switches(qtbot, loaded):
    first = loaded.layer
    loaded._on_display_style_changed("colormap", "plasma")
    loaded._on_display_style_changed("color_vtrail", "#445566")
    second = loaded.project.add_layer("plain", first.source_id, first.chain)
    loaded.add_layer_row(second, loaded._fields[first.layer_id])

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(second.layer_id)
    assert loaded.canvas.vtrail_item.opts["pen"].color().name() == _hex(VTRAIL_COLOR)  # defaults

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(first.layer_id)
    assert loaded.canvas.vtrail_item.opts["pen"].color().name() == "#445566"           # restored


# ------------------------------------------------------------------------------------------------
# Selecting a point layer must not restyle the
# still-displayed raster with the point layer's own (unrelated) prefs.


def test_selecting_a_point_layer_does_not_restyle_the_displayed_raster(loaded):
    """Tune the active raster to magma, then select an unrelated point layer with its OWN
    (default-viridis) colormap tag. Before the fix, ``_select_layer`` -> ``_sync_display_controls``
    -> ``_apply_display_style`` pushed the point layer's colormap/overlay-color prefs onto the
    canvas unconditionally, even though the canvas keeps showing the PREVIOUS raster (a point
    layer's field is not array-like, so ``Canvas.set_field`` is never called for it) -- magma would
    silently snap back to viridis. Canvas and the layer's own stored preference (what the
    arrangement reads via ``_display_style_of``) must still agree afterward."""
    raster_layer = loaded.layer
    loaded._on_display_style_changed("colormap", "magma")
    assert loaded.canvas._colormap == "magma"

    source = loaded.project.add_source("mem:pts", kind="points")
    point_layer = loaded.project.add_layer("pts", source.source_id)
    point_layer.tags["ui.colormap"] = "plasma"           # a DIFFERENT colormap than the raster's
    point_layer.tags["ui.color_points"] = "#112233"
    point_layer.tags["ui.point_size"] = repr(9.0)
    pset = PointSet(lon=np.array([1.0]), lat=np.array([2.0]))
    loaded.add_layer_row(point_layer, pset)

    loaded.layer_list.select_layer(point_layer.layer_id)

    # The raster's own colormap must still be showing -- selecting the point layer must not have
    # applied ITS (different) colormap to the still-displayed raster image.
    assert loaded.canvas._colormap == "magma"
    # Canvas and arrangement (which reads straight off the layer's own tag) must still agree.
    assert _display_style_of(raster_layer)["colormap"] == "magma"
    # The point-relevant setter DID fire -- the panel's own edits still reach the points overlay.
    assert loaded.canvas.points_item.opts["brush"].color().name() == "#112233"
    assert loaded.canvas.points_item.opts["size"] == pytest.approx(9.0)


def test_editing_colormap_while_a_point_layer_is_active_does_not_touch_the_canvas(loaded):
    """Same guard, the OTHER entry point: a display-control edit made WHILE a point layer is
    active (``_on_display_style_changed``) must not reach the canvas's raster-display setters
    either -- it still persists on the point layer's own tags (for whenever THAT layer becomes a
    raster's own drape, or a project reopen), same as any other edit."""
    loaded._on_display_style_changed("colormap", "magma")

    source = loaded.project.add_source("mem:pts", kind="points")
    point_layer = loaded.project.add_layer("pts", source.source_id)
    pset = PointSet(lon=np.array([1.0]), lat=np.array([2.0]))
    loaded.add_layer_row(point_layer, pset)
    loaded.layer_list.select_layer(point_layer.layer_id)

    loaded._on_display_style_changed("colormap", "plasma")

    assert point_layer.tags["ui.colormap"] == "plasma"    # persisted on the ACTIVE (point) layer
    assert loaded.canvas._colormap == "magma"              # canvas untouched


def test_a_dragged_knob_shows_the_value_it_just_proposed(loaded):
    """``DragValue`` never updates its own label on a drag -- the owner must ``set_value`` once
    it has applied the proposal (knob_widgets.py's feedback-loop contract). The Vector-view bug:
the thickness applied but the number on the knob stayed put."""
    loaded._on_display_style_changed("line_width", 2.5)
    control = loaded.right_panel._display_controls["line_width"]
    assert control._value == pytest.approx(2.5)
    assert "2.5" in control.text()


# ----------------------------------------------------------------------------- hillshade

def test_hillshade_defaults_off_with_sun_from_the_north_west(loaded):
    style = _display_style_of(loaded.layer)
    assert style["hillshade"] is False
    assert style["sun_azimuth"] == 315.0 and style["sun_altitude"] == 45.0 and style["z_factor"] == 1.0


def test_hillshade_toggle_persists_and_renders_the_raster_as_shaded_rgba(loaded):
    plain = loaded.canvas.image_item.image
    assert plain.ndim == 2
    loaded._on_display_style_changed("hillshade", True)
    assert loaded.layer.tags["ui.hillshade"] == "True"
    shaded = loaded.canvas.image_item.image
    assert shaded.ndim == 3 and shaded.shape[:2] == plain.shape and shaded.shape[2] == 4
    loaded._on_display_style_changed("sun_azimuth", 90.0)
    assert loaded.layer.tags["ui.sun_azimuth"] == repr(90.0)
    turned = loaded.canvas.image_item.image
    assert turned.shape == shaded.shape and not np.array_equal(turned, shaded)
    loaded._on_display_style_changed("hillshade", False)
    assert loaded.canvas.image_item.image.ndim == 2


def test_hillshade_survives_a_colormap_change_and_a_layer_switch(qtbot, loaded):
    loaded._on_display_style_changed("hillshade", True)
    loaded._on_display_style_changed("colormap", "magma")
    assert loaded.canvas.image_item.image.ndim == 3
    loaded._sync_display_controls(loaded.layer)
    assert loaded.right_panel.hillshade_check.isChecked()
    assert loaded.right_panel._display_controls["sun_azimuth"]._value == 315.0


# -------------------------------------------------------------------------------- stretch

def test_stretch_defaults_to_linear_and_validates_the_tag(loaded):
    style = _display_style_of(loaded.layer)
    assert style["stretch"] == "linear" and style["stretch_pct"] == 2.0
    loaded.layer.tags["ui.stretch"] = "cubist"
    assert _display_style_of(loaded.layer)["stretch"] == "linear"


def test_stretch_changes_the_canvas_image_and_persists(loaded):
    lin = np.array(loaded.canvas.image_item.image, copy=True)
    loaded._on_display_style_changed("stretch", "histogram")
    assert loaded.layer.tags["ui.stretch"] == "histogram"
    hist = loaded.canvas.image_item.image
    assert hist.shape == lin.shape and not np.array_equal(hist, lin)
    assert np.nanmin(hist) >= 0.0 and np.nanmax(hist) <= 1.0
    loaded._on_display_style_changed("stretch", "percent")
    loaded._on_display_style_changed("stretch_pct", 10.0)
    assert loaded.layer.tags["ui.stretch_pct"] == repr(10.0)
    loaded._on_display_style_changed("hillshade", True)
    assert loaded.canvas.image_item.image.ndim == 3          # stretch + hillshade compose
    loaded._sync_display_controls(loaded.layer)
    assert loaded.right_panel.stretch_combo.currentText() == "percent"


# ----------------------------------------------------------------------------- 3-D surface

def test_surface_and_depth_positive_are_per_layer_display_tags(loaded):
    style = _display_style_of(loaded.layer)
    assert style["surface"] is False and style["depth_positive"] is False
    loaded._on_display_style_changed("surface", True)
    loaded._on_display_style_changed("depth_positive", True)
    assert loaded.layer.tags["ui.surface"] == "True" and loaded.layer.tags["ui.depth_positive"] == "True"
    loaded._sync_display_controls(loaded.layer)
    # The surface control is a dialog-opening button whose label mirrors state; depth
    # is a checkbox. The default source is "same".
    assert loaded.right_panel.depth_check.isChecked()
    assert loaded.right_panel.surface_button.text() == "3-D surface: same dataset…"
    assert _display_style_of(loaded.layer)["surface_source"] == "same"
    loaded._on_display_style_changed("surface_source", "some-layer-id")
    assert loaded.layer.tags["ui.surface_source"] == "some-layer-id"
    loaded._sync_display_controls(loaded.layer)
    assert loaded.right_panel.surface_button.text() == "3-D surface: other dataset…"
