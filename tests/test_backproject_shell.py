# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the SHELL half of ``dynamix.devices.backproject``:
``MainWindow``'s stamping of a point layer's ``backproject`` step from a target layer's real
field, the refusal-and-zero path for an unresolvable/non-georeferenced target, and the canvas
scatter overlay's target-match gate, end to end through the real window.

Offscreen Qt, same convention as the rest of the shell suite. ``window``/``stub_devices``/
``_device_drop`` come from ``test_shell_window.py`` (the established cross-file precedent,
``test_point_import.py``'s own docstring names the fixture reuse explicitly). Every step here
goes through a REAL zone gesture -- a device drop, then a param edit -- rather than calling
``MainWindow._on_chain_edited`` with a hand-built descriptor directly: the zone's own
``WorkflowZone._boxes`` list is only rebuilt by ITS OWN ``set_steps`` (inside ``dropEvent`` ->
``_commit_or_revert``), so a bypassed, hand-fed descriptor whose length differs from what the zone
currently holds desyncs the two and crashes the very next redraw (``_set_transform_states``
indexing ``self.strips.strip(i)`` against a stale box count) -- exactly the failure mode
``main_window.py``'s own module docstring warns a REAL gesture, not a hand-fed one, is needed to
avoid.

A REAL georeferenced raster target is needed here, so this whole module needs ``rasterio`` --
``tests/test_geo_mapping.py``'s own BOEM-like GeoTIFF fixture is reused directly
(``test_arrangement_scene.py``'s own established cross-file pattern) rather than adding that
dependency to ``test_point_import.py``, whose other point-layer shell tests all run without it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

from dynamix.core.pointset import PointSet, read_csv_points
from dynamix.geo.mapping import points_lonlat

from tests.test_geo_mapping import _load_field, _write_boem_like_tif
from tests.test_shell_window import _device_drop, stub_devices, window  # noqa: F401


def _target_field(tmp_path, name="target.tif"):
    path = tmp_path / name
    _write_boem_like_tif(path)
    return _load_field(path)


def _valid_lonlat(field, col=5, row=5):
    """A lon/lat pair guaranteed to lie inside ``field``'s own CRS's valid projection domain --
    every ``_target_field``-built fixture shares the identical BOEM-like NAD27 TM domain/extent
    (only pixel VALUES differ across files), so a pixel-center point is always safely in-domain,
    unlike an arbitrary literal lon/lat (which this custom, regionally-scoped TM projection can
    reject outright with rasterio's own "Point outside of projection domain")."""
    lon, lat = points_lonlat(field, [col], [row])
    return lon.tolist(), lat.tolist()


def _add_point_layer(window, lon=(1.0,), lat=(2.0,), name="pts"):
    pset = PointSet(lon=np.asarray(lon, dtype=np.float64), lat=np.asarray(lat, dtype=np.float64))
    source = window.project.add_source(f"mem:{name}", kind="points")
    layer = window.project.add_layer(name, source.source_id)
    window.add_layer_row(layer, pset)
    return layer


def _add_raster_layer(window, field, name="raster"):
    source = window.project.add_source(f"mem:{name}")
    layer = window.project.add_layer(name, source.source_id)
    window.add_layer_row(layer, field)
    return layer


def _drop_backproject(qtbot, window) -> int:
    """A REAL drop of the ``backproject`` device onto the (currently-active point layer's) zone
    -- defaults only (``target=""``), the same "add at defaults" every device drop gets. Returns
    the step index (always the chain's first/only step here, since every point layer in this
    module starts chainless)."""
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.strips.dropEvent(_device_drop("backproject"))
    return window._names.index("backproject")


def _set_target(qtbot, window, idx: int, target: str) -> None:
    """A REAL param edit on the already-dropped step -- the gesture ``DeviceBox``'s generated
    "target" control performs, driven directly (this suite's established "call the handler, not
    the input queue" convention)."""
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window._on_param_changed(idx, "target", target)


# --------------------------------------------------------------------------- stamping


def test_stamp_fires_on_chain_edit_with_a_georeferenced_target(qtbot, window, tmp_path):
    target_field = _target_field(tmp_path)
    _add_raster_layer(window, target_field, "raster")
    lon, lat = _valid_lonlat(target_field)
    point_layer = _add_point_layer(window, lon=lon, lat=lat)
    window.layer_list.select_layer(point_layer.layer_id)

    idx = _drop_backproject(qtbot, window)
    _set_target(qtbot, window, idx, "raster")

    params = window._params[idx]
    assert params["_target_crs"] == target_field.provenance["crs"]
    assert params["_target_nx"] == target_field.nx
    assert params["_target_ny"] == target_field.ny
    assert params["_target_x0"] == pytest.approx(float(target_field.x_axis[0]))
    assert params["_target_dx"] == pytest.approx(
        (float(target_field.x_axis[-1]) - float(target_field.x_axis[0])) / (target_field.nx - 1))
    # Baked into layer.chain too, not just the window's own working copy.
    assert point_layer.chain.steps[0].params["_target_nx"] == target_field.nx


def test_retargeting_a_bound_layer_invalidates_the_cache(qtbot, window, tmp_path):
    """Cache-key honesty end to end: the stamped scalars ride params, so pointing the SAME step
    at a DIFFERENT target is a genuine cache miss, not a silent reuse of the old target's
    compute.

    Checked on ``window.cache``'s own CUMULATIVE ``misses`` counter, not the ``resolved`` signal's
    own ``Renderable.cache_misses`` -- a transform edit always dispatches the worker (which does
    the genuine compute, a real miss on ITS OWN internal resolve) and then re-resolves once more
    synchronously on the GUI thread purely to fetch what the worker just cached (module docstring:
    "a guaranteed cache HIT") -- that second resolve's own ``Renderable`` is what actually reaches
    ``resolved``, so it reports ``cache_misses == 0`` for EVERY transform edit, genuinely new or
    not. ``Cache.misses`` is not reset between those two resolves, so its delta is what actually
    answers "did new work happen".
    """
    field_a = _target_field(tmp_path, "a.tif")
    field_b = _target_field(tmp_path, "b.tif")
    _add_raster_layer(window, field_a, "raster_a")
    _add_raster_layer(window, field_b, "raster_b")
    lon, lat = _valid_lonlat(field_a)      # both fixtures share the identical CRS domain
    point_layer = _add_point_layer(window, lon=lon, lat=lat)
    window.layer_list.select_layer(point_layer.layer_id)
    idx = _drop_backproject(qtbot, window)

    misses_before = window.cache.misses
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window._on_param_changed(idx, "target", "raster_a")
    assert window.cache.misses > misses_before

    misses_before = window.cache.misses
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window._on_param_changed(idx, "target", "raster_b")
    assert window.cache.misses > misses_before     # a genuine recompute, not a stale cache hit
    assert window._params[idx]["_target_crs"] == field_b.provenance["crs"]


# --------------------------------------------------------------------------- refusal


def test_stamp_warns_and_zeros_on_an_unresolvable_target_name(qtbot, window, tmp_path):
    point_layer = _add_point_layer(window)
    window.layer_list.select_layer(point_layer.layer_id)
    idx = _drop_backproject(qtbot, window)

    _set_target(qtbot, window, idx, "no such layer")

    params = window._params[idx]
    assert params["_target_nx"] == 0
    assert params["_target_ny"] == 0
    assert params["_target_crs"] == ""
    warning = window.strips.reading_label.text()
    assert "no such layer" in warning
    assert "no georeference" in warning


def test_stamp_warns_on_a_real_layer_with_no_georeference(qtbot, window, tmp_path):
    bare_field = np.zeros((8, 8))          # a plain array: no `.provenance["crs"]` at all
    _add_raster_layer(window, bare_field, "bare")
    point_layer = _add_point_layer(window)
    window.layer_list.select_layer(point_layer.layer_id)
    idx = _drop_backproject(qtbot, window)

    _set_target(qtbot, window, idx, "bare")

    assert window._params[idx]["_target_nx"] == 0
    warning = window.strips.reading_label.text()
    assert "bare" in warning
    assert "no georeference" in warning


def test_empty_target_stamps_zero_without_a_warning(qtbot, window):
    point_layer = _add_point_layer(window)
    window.layer_list.select_layer(point_layer.layer_id)

    idx = _drop_backproject(qtbot, window)     # defaults only -- target="" already

    assert window._params[idx]["_target_nx"] == 0
    assert window.strips.reading_label.text() == ""


# --------------------------------------------------------------------------- re-stamp on select


def test_select_layer_restamps_once_the_target_becomes_available(qtbot, window, tmp_path):
    """The target didn't exist yet when the step was first pointed at it (a refused, zeroed
    stamp); loading it and then switching BACK onto the point layer re-stamps for free -- the
    ``_select_layer``/``_sync_display_controls`` hook the design names."""
    # Built (but not yet added as a layer) up front, purely to get a lon/lat pair inside its own
    # CRS's valid projection domain -- the fixture's domain/extent is a fixed constant across
    # every file this helper writes, so deriving it early costs nothing extra.
    target_field = _target_field(tmp_path)
    lon, lat = _valid_lonlat(target_field)
    point_layer = _add_point_layer(window, lon=lon, lat=lat)
    window.layer_list.select_layer(point_layer.layer_id)
    idx = _drop_backproject(qtbot, window)
    _set_target(qtbot, window, idx, "raster")
    assert window._params[idx]["_target_nx"] == 0     # not resolvable yet

    raster_layer = _add_raster_layer(window, target_field, "raster")
    window.layer_list.select_layer(raster_layer.layer_id)     # switch away
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.layer_list.select_layer(point_layer.layer_id)  # switch back onto the point layer

    idx = window._names.index("backproject")
    assert window._params[idx]["_target_nx"] == target_field.nx


# --------------------------------------------------------------------------- canvas overlay


def test_canvas_shows_points_when_the_active_raster_is_the_bound_target(qtbot, window, tmp_path):
    """End to end: the target raster is on screen (viewed first, per Canvas.set_field's own
    call sites), then the point layer bound to it is selected and given a `backproject` step --
    the scatter lands on `canvas.points_item`, registered to that raster's own pixel grid."""
    target_field = _target_field(tmp_path)
    raster_layer = _add_raster_layer(window, target_field, "raster")
    window.layer_list.select_layer(raster_layer.layer_id)

    lon, lat = points_lonlat(target_field, [5], [5])
    point_layer = _add_point_layer(window, lon=lon.tolist(), lat=lat.tolist())
    window.layer_list.select_layer(point_layer.layer_id)
    idx = _drop_backproject(qtbot, window)
    _set_target(qtbot, window, idx, "raster")

    x, y = window.canvas.points_item.getData()
    assert x.size == 1
    np.testing.assert_allclose(x[0], 5.0, atol=1e-3)
    np.testing.assert_allclose(y[0], 5.0, atol=1e-3)


def test_canvas_clears_points_when_the_active_raster_is_not_the_bound_target(
        qtbot, window, tmp_path):
    target_field = _target_field(tmp_path, "target.tif")
    other_field = _target_field(tmp_path, "other.tif")
    raster_layer = _add_raster_layer(window, target_field, "raster")
    other_layer = _add_raster_layer(window, other_field, "other")
    window.layer_list.select_layer(raster_layer.layer_id)

    lon, lat = points_lonlat(target_field, [5], [5])
    point_layer = _add_point_layer(window, lon=lon.tolist(), lat=lat.tolist())
    window.layer_list.select_layer(point_layer.layer_id)
    idx = _drop_backproject(qtbot, window)
    _set_target(qtbot, window, idx, "raster")
    assert window.canvas.points_item.data.size == 1

    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.layer_list.select_layer(other_layer.layer_id)

    assert window.canvas.points_item.data.size == 0


def test_canvas_clears_points_when_the_point_layer_is_unbound(qtbot, window, tmp_path):
    target_field = _target_field(tmp_path)
    raster_layer = _add_raster_layer(window, target_field, "raster")
    window.layer_list.select_layer(raster_layer.layer_id)
    point_layer = _add_point_layer(window)
    window.layer_list.select_layer(point_layer.layer_id)

    _drop_backproject(qtbot, window)          # defaults only -- target="" -- unbound

    assert window.canvas.points_item.data.size == 0


# --------------------------------------------------------------------------- Mapping-
# scoped cache identity


def test_two_layers_over_the_same_csv_with_different_mappings_dont_collide(
        qtbot, window, tmp_path):
    """Same CSV (one deduped source), two layers, two DIFFERENT column mappings, both bound to the
    SAME target -- the exact in-session collision the mapping-identity fix names. Before the fix ``resolve()``'s
    cache key was ``(device, source_id, params)`` alone: identical ``source_id`` (path-deduped) and
    identical params (both bound to "raster") meant the SECOND layer's compute was silently served
    the FIRST layer's cached ``points_px`` -- numbers computed from the wrong input. Proven both
    ways: a genuine cache miss for the second layer's own resolve, and its OWN pixel placement
    (not the first layer's) reaching the canvas.
    """
    target_field = _target_field(tmp_path)
    raster_layer = _add_raster_layer(window, target_field, "raster")
    window.layer_list.select_layer(raster_layer.layer_id)

    lon1, lat1 = _valid_lonlat(target_field, col=5, row=5)
    lon2, lat2 = _valid_lonlat(target_field, col=20, row=20)
    csv_path = tmp_path / "quakes.csv"
    csv_path.write_text(f"lonA,latA,lonB,latB\n{lon1[0]},{lat1[0]},{lon2[0]},{lat2[0]}\n")
    mapping1 = {"lon": "lonA", "lat": "latA"}
    mapping2 = {"lon": "lonB", "lat": "latB"}

    source = window.project.add_source(str(csv_path), kind="points")
    layer1 = window.project.add_layer("pts1", source.source_id)
    layer1.tags["points.mapping"] = json.dumps(mapping1)
    window.add_layer_row(layer1, read_csv_points(str(csv_path), mapping=mapping1))

    layer2 = window.project.add_layer("pts2", source.source_id)
    layer2.tags["points.mapping"] = json.dumps(mapping2)
    window.add_layer_row(layer2, read_csv_points(str(csv_path), mapping=mapping2))

    window.layer_list.select_layer(layer1.layer_id)
    idx1 = _drop_backproject(qtbot, window)
    _set_target(qtbot, window, idx1, "raster")
    x1, y1 = window.canvas.points_item.getData()
    np.testing.assert_allclose(x1[0], 5.0, atol=1e-3)
    np.testing.assert_allclose(y1[0], 5.0, atol=1e-3)

    misses_before = window.cache.misses
    window.layer_list.select_layer(layer2.layer_id)
    idx2 = _drop_backproject(qtbot, window)
    _set_target(qtbot, window, idx2, "raster")
    assert window.cache.misses > misses_before   # a genuine miss, not a stale hit off layer1

    x2, y2 = window.canvas.points_item.getData()
    np.testing.assert_allclose(x2[0], 20.0, atol=1e-3)
    np.testing.assert_allclose(y2[0], 20.0, atol=1e-3)
    qtbot.waitUntil(lambda: not window.is_computing, timeout=10000)   # drain before teardown
