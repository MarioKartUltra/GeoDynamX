# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.arrangement.scene -- the arrangement's per-layer raster draping and
4-mode projection switch.

Offscreen VTK: ``pytest.importorskip("pyvista")`` skips this whole module honestly if pyvista
isn't installed (never a fake pass); ``pyvista.OFF_SCREEN = True`` plus ``pv.Plotter(off_screen=
True)`` builds a real plotter with no display required. ``Scene`` itself needs no Qt -- it takes a
plain ``pyvista.Plotter`` directly in every test here (the live app instead hands it a
``pyvistaqt.QtInteractor``, which duck-types as one).

Fixture reuse: synthetic geo fields are built with ``tests/test_geo_mapping.py``'s own
``_write_boem_like_tif``/``_load_field`` helpers (same NAD27 TM survey-foot GeoTIFF fixture), imported directly -- the established cross-file fixture-reuse pattern in this suite (see
``tests/test_arrangement_flip.py``'s own import from ``tests/test_shell_window.py``).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista not installed")
pytest.importorskip("rasterio", reason="rasterio not installed (needed by the geo fixture)")
pv.OFF_SCREEN = True

from dynamix.core.frames import LocalFrame
from dynamix.core.pointset import PointSet
from dynamix.core.projection import project
from dynamix.core.rasterfield import RasterField
from dynamix.geo.mapping import field_lonlat_grid
from dynamix.model.layer import Layer
from dynamix.shell.arrangement.scene import (DEFAULT_MODE, EXTREMA_COLOR, POINTS_COLOR,
                                              ROI_BOUNDS_COLOR, SEAM_COLOR, VTRAIL_COLOR, Scene,
                                              _GRATICULE_NAME)

from tests.test_geo_mapping import (_BOEM_LIKE_WKT, _NORTH_FT, _NX, _NY, _NODATA_CELLS, _PX_FT,
                                     _SENTINEL_CELLS, _WEST_FT, _load_field, _write_boem_like_tif)


def _plotter():
    return pv.Plotter(off_screen=True)


def _geo_field(tmp_path, name):
    path = tmp_path / name
    _write_boem_like_tif(path)
    return _load_field(path)


def _bare_field():
    return RasterField._from_bare_array(np.zeros((8, 8)), "bare", frame=LocalFrame(units="px"),
                                         name="bare")


def _write_asymmetric_tif(path, ny, nx):
    """A GeoTIFF shaped like ``tests/test_geo_mapping.py``'s own BOEM-like fixture (same WKT,
    pixel size, origin) but with ``ny != nx`` -- the shared 64x64 fixture is square, so a
    ``pv.StructuredGrid`` ``dims=[nx, ny, 1]`` vs ``[ny, nx, 1]`` mixup is geometrically
    INVISIBLE there (both read back as the identical tuple ``(64, 64, 1)``). An asymmetric grid
    makes the two orderings distinguishable (no existing test could ever catch a
    dims swap against a square fixture, regardless of what it inspects)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    crs = CRS.from_wkt(_BOEM_LIKE_WKT)
    transform = from_origin(_WEST_FT, _NORTH_FT, _PX_FT, _PX_FT)
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    vals = (-(500.0 + 0.1 * rows + 0.05 * cols)).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx, count=1,
                        dtype="float32", crs=crs, transform=transform) as dst:
        dst.write(vals, 1)


def _asymmetric_field(tmp_path, ny, nx, name="asym.tif"):
    path = tmp_path / name
    _write_asymmetric_tif(path, ny, nx)
    return _load_field(path)


# ------------------------------------------------------------------- Chain/extrema/result fixtures
#
# Shared with tests/test_arrangement_mask.py (imported directly from there -- the same cross-file
# fixture-reuse pattern this file itself uses for test_geo_mapping's helpers, above).


def _make_chain(x0, y0, n=4, slope=-0.5, tags=None, group_color=None):
    """A chain seeded at pixel ``(x0, y0)``, drifting one pixel per scale step (``+idx`` on both
    axes) -- geometrically real enough to project through ``points_lonlat`` into a visible
    polyline, unlike ``tests/test_chain_classify.py``'s own ``_chain`` fixture (constant x0/y0,
    fine for classification-only tests but a degenerate single-point line here). ``log2_mod =
    slope * log2_scales`` with ``log2_scales = arange(n)`` -- an exact line, so ``max(log2_mod)``
    is deterministic and easy to reason about in the masking tests."""
    idx = np.arange(n, dtype=np.int64)
    x = (x0 + idx).astype(np.int64)
    y = (y0 + idx).astype(np.int64)
    log2_scales = idx.astype(np.float64)
    log2_mod = slope * log2_scales
    chain = {"x": x, "y": y, "mod": (2.0 ** log2_mod), "log2_mod": log2_mod,
             "log2_scales": log2_scales}
    if tags is not None:
        chain["tags"] = list(tags)
    if group_color is not None:
        chain["group_color"] = list(group_color)
    return chain


def _make_extrema0(x, y):
    n = len(x)
    return {"x": np.asarray(x, dtype=np.int64), "y": np.asarray(y, dtype=np.int64),
            "mod": np.ones(n), "arg": np.zeros(n), "line_id": np.arange(n, dtype=np.int64)}


def _result(chains, extrema0=None, roi=None):
    out = {"chains": list(chains)}
    if extrema0 is not None:
        out["extrema"] = [extrema0]
    if roi is not None:
        out["_roi"] = {"roi": tuple(roi)}
    return out


def _make_chain_with_scales(x0, y0, log2_scales, slope=-0.5):
    """Like ``_make_chain`` above, but with an EXPLICIT, possibly non-uniform and/or GAPPED
    ``log2_scales`` array instead of always ``arange(n)`` -- the asymmetric-chain and missing-scale-gap fixtures need per-point scale values
    the uniform helper cannot produce."""
    log2_scales = np.asarray(log2_scales, dtype=np.float64)
    n = len(log2_scales)
    idx = np.arange(n, dtype=np.int64)
    x = (x0 + idx).astype(np.int64)
    y = (y0 + idx).astype(np.int64)
    log2_mod = slope * log2_scales
    return {"x": x, "y": y, "mod": (2.0 ** log2_mod), "log2_mod": log2_mod,
            "log2_scales": log2_scales}


# --------------------------------------------------------------------------------------- set_layers


def test_default_mode_is_mercator():
    scene = Scene(_plotter())
    assert scene.mode == "mercator"
    assert DEFAULT_MODE == "mercator"


# ------------------------------------------------------------------------------ themed background
#
# Scene.__init__ gains an optional `background` color string, applied
# via `plotter.set_background` exactly once, at construction. Scene stays Qt-free -- it never
# imports dynamix.shell.theme itself (see the module docstring's own "Themed background" section);
# these tests exercise it with a plain hex literal, the same shape the live view hands in.


def test_background_default_none_leaves_the_plotters_background_untouched():
    plotter = _plotter()
    before = plotter.background_color

    Scene(plotter)

    assert plotter.background_color == before      # no set_background call at all


def test_background_hex_string_sets_the_plotters_background():
    plotter = _plotter()

    scene = Scene(plotter, background="#131313")

    assert scene._plotter.background_color == pv.Color("#131313")


def test_two_ok_entries_get_mesh_actors_and_an_empty_legend(tmp_path):
    scene = Scene(_plotter())
    layer_a = Layer(layer_id=1, name="A", source_id="mem:a")
    layer_b = Layer(layer_id=2, name="B", source_id="mem:b")
    field_a = _geo_field(tmp_path, "a.tif")
    field_b = _geo_field(tmp_path, "b.tif")

    scene.set_layers([
        {"layer": layer_a, "field": field_a, "result": None, "status": "ok"},
        {"layer": layer_b, "field": field_b, "result": None, "status": "ok"},
    ])

    assert scene.actor_count() == 2
    assert scene.legend_lines == []
    assert scene._layer_actors[1] == ["layer-1-raster"]
    assert scene._layer_actors[2] == ["layer-2-raster"]
    assert set(scene._layer_actors[1] + scene._layer_actors[2]) <= set(scene._plotter.actors)


def test_no_georeference_entry_gets_zero_geometry_and_a_legend_line():
    scene = Scene(_plotter())
    layer = Layer(layer_id=3, name="Bare", source_id="mem:bare")

    scene.set_layers([
        {"layer": layer, "field": _bare_field(), "result": None, "status": "no-georeference"},
    ])

    assert scene.actor_count() == 0
    assert scene._layer_actors == {}
    assert len(scene.legend_lines) == 1
    assert "Bare" in scene.legend_lines[0]
    assert "no georeference" in scene.legend_lines[0]


def test_computing_and_error_entries_are_legend_only_never_geometry(tmp_path):
    scene = Scene(_plotter())
    ok_layer = Layer(layer_id=1, name="OK", source_id="mem:ok")
    computing_layer = Layer(layer_id=2, name="Computing", source_id="mem:c")
    error_layer = Layer(layer_id=3, name="Errored", source_id="mem:e")
    ok_field = _geo_field(tmp_path, "ok.tif")

    scene.set_layers([
        {"layer": ok_layer, "field": ok_field, "result": None, "status": "ok"},
        {"layer": computing_layer, "field": ok_field, "result": None, "status": "computing"},
        {"layer": error_layer, "field": ok_field, "result": None, "status": "error:bad params"},
    ])

    assert scene.actor_count() == 1
    assert scene._layer_actors == {1: ["layer-1-raster"]}
    assert len(scene.legend_lines) == 2
    computing_line = next(l for l in scene.legend_lines if "Computing" in l)
    error_line = next(l for l in scene.legend_lines if "Errored" in l)
    assert "computing" in computing_line
    assert "bad params" in error_line


def test_set_layers_replaces_the_previous_set(tmp_path):
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "one.tif")
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])
    assert scene.actor_count() == 1

    scene.set_layers([
        {"layer": layer, "field": None, "result": None, "status": "computing"},
    ])
    assert scene.actor_count() == 0
    assert len(scene.legend_lines) == 1


# ------------------------------------------------------------------------------- NaN sentinel handling


def test_nan_sentinel_region_is_masked_and_point_count_matches_stride(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=7, name="Sentinel", source_id="mem:sentinel")
    field = _geo_field(tmp_path, "sentinel.tif")

    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    name = scene._layer_actors[7][0]
    grid = scene._plotter.actors[name].mapper.dataset
    assert grid.n_points == _NY * _NX          # stride == 1 at this size, per the _stride_for

    values = grid.point_data["value"]
    for r, c in _NODATA_CELLS + _SENTINEL_CELLS:
        assert np.isnan(values[r * _NX + c])   # C-order ravel: row r, col c -> flat index r*nx+c

    finite = values[np.isfinite(values)]
    assert finite.size > 0
    assert np.max(np.abs(finite)) < 1e6        # real data only; the 3.4e38-magnitude sentinel is gone


# ------------------------------------------------------------------------------------------ set_mode


def test_set_mode_rebuilds_with_unchanged_actor_count_but_new_coordinates(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    name = scene._layer_actors[1][0]
    mercator_points = np.array(scene._plotter.actors[name].mapper.dataset.points)
    assert scene.mode == "mercator"

    scene.set_mode("globe")

    assert scene.mode == "globe"
    assert scene.actor_count() == 1
    globe_points = np.array(scene._plotter.actors[name].mapper.dataset.points)
    assert globe_points.shape == mercator_points.shape
    assert not np.allclose(globe_points, mercator_points)


def test_set_mode_matches_projection_project_directly(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])
    scene.set_mode("pacific")

    lon2d, lat2d, _values2d, _stride = field_lonlat_grid(field)
    lon_flat, lat_flat = lon2d.ravel(), lat2d.ravel()
    expected = project(lon_flat, lat_flat, np.zeros_like(lon_flat), mode="pacific")

    name = scene._layer_actors[1][0]
    actual = np.array(scene._plotter.actors[name].mapper.dataset.points)
    np.testing.assert_allclose(actual, expected)


def test_set_mode_rejects_an_unknown_mode_without_disturbing_state(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    with pytest.raises(ValueError):
        scene.set_mode("nonsense")

    assert scene.mode == "mercator"
    assert scene.actor_count() == 1


# --------------------------------------------------------------- lon/lat grid cache
#
# Field_lonlat_grid's own CRS transform is mode-invariant, so set_mode's rebuild used
# to recompute it from scratch on every switch for a field that had not changed at all -- profiling
# found ~86% of a set_mode call's own time there. Scene._lonlat_cache/_lonlat_grid_for fix this;
# these tests use the same counting-monkeypatch pattern tests/test_geometry_cache.py's own
# canvas.py cache tests use, applied to dynamix.shell.arrangement.scene.field_lonlat_grid.


def _counting_field_lonlat_grid(monkeypatch):
    import dynamix.shell.arrangement.scene as scene_mod

    calls = {"n": 0}
    real = scene_mod.field_lonlat_grid

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(scene_mod, "field_lonlat_grid", counting)
    return calls


def test_lonlat_grid_is_cached_across_set_mode_rebuilds(tmp_path, monkeypatch):
    calls = _counting_field_lonlat_grid(monkeypatch)
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])
    assert calls["n"] == 1                     # the first, unavoidable build

    scene.set_mode("globe")
    scene.set_mode("pacific")
    scene.set_mode("mercator")

    assert calls["n"] == 1                     # three more rebuilds, zero more CRS transforms
    assert scene.actor_count() == 1             # the rebuilds still genuinely happened


def test_lonlat_grid_cache_invalidates_when_the_field_object_changes(tmp_path, monkeypatch):
    calls = _counting_field_lonlat_grid(monkeypatch)
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field_a = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field_a, "result": None, "status": "ok"}])
    assert calls["n"] == 1

    field_b = _geo_field(tmp_path, "b.tif")     # same layer_id, a DIFFERENT field object
    scene.set_layers([{"layer": layer, "field": field_b, "result": None, "status": "ok"}])

    assert calls["n"] == 2                      # treated as a genuine miss, not assumed unchanged


def test_lonlat_grid_cache_is_pruned_when_a_layer_disappears(tmp_path):
    scene = Scene(_plotter())
    layer_a = Layer(layer_id=1, name="A", source_id="mem:a")
    layer_b = Layer(layer_id=2, name="B", source_id="mem:b")
    field_a = _geo_field(tmp_path, "a.tif")
    field_b = _geo_field(tmp_path, "b.tif")
    scene.set_layers([
        {"layer": layer_a, "field": field_a, "result": None, "status": "ok"},
        {"layer": layer_b, "field": field_b, "result": None, "status": "ok"},
    ])
    assert set(scene._lonlat_cache) == {1, 2}

    scene.set_layers([{"layer": layer_a, "field": field_a, "result": None, "status": "ok"}])

    assert set(scene._lonlat_cache) == {1}      # layer 2's cached grid is gone, not leaked forever


# --------------------------------------------------------------- raster grid orientation (dims order)


def test_raster_grid_dimensions_pin_x_fastest_not_transposed(tmp_path):
    """Pyvista's ``.points``/``.bounds``/``.n_cells`` readbacks are INDEPENDENT of
    ``.dimensions`` (confirmed empirically), and the shared geo fixture is square (64x64), so
    flipping ``grid.dimensions = [nx, ny, 1]`` to ``[ny, nx, 1]`` in ``scene.py`` -- a plausible
    future "fix" -- would render a transposed surface while every OTHER geometry test in this file
    (which only ever inspects ``.points``/``n_points``) stayed green, because ``(64, 64, 1)`` and
    ``(64, 64, 1)`` are the same tuple either way. This test uses an ASYMMETRIC fixture (``ny !=
    nx``) so the two orderings are distinguishable tuples, and pins both the dimensions metadata
    itself and a genuine (row, col) position probe -- verified (throwaway, reverted) to actually
    FAIL if ``scene.py``'s dims assignment is swapped."""
    scene = Scene(_plotter())
    ny, nx = 48, 64                              # deliberately asymmetric: ny != nx
    layer = Layer(layer_id=1, name="Asym", source_id="mem:asym")
    field = _asymmetric_field(tmp_path, ny=ny, nx=nx)

    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    lon2d, lat2d, _values2d, stride = field_lonlat_grid(field)
    assert stride == 1
    assert lon2d.shape == (ny, nx)                # sanity: the fixture really is asymmetric

    name = scene._layer_actors[1][0]
    grid = scene._plotter.actors[name].mapper.dataset

    dims = tuple(int(d) for d in grid.dimensions)
    assert dims == (nx, ny, 1)                    # pins the orientation directly -- a swap fails here

    r, c = 5, 40                                  # r != c, well inside both (asymmetric) axes
    pts = np.asarray(grid.points).reshape(ny, nx, 3)
    expected = project(np.array([lon2d[r, c]]), np.array([lat2d[r, c]]), np.zeros(1),
                        mode=scene.mode)[0]
    np.testing.assert_allclose(pts[r, c], expected)


# ------------------------------------------------------------------------- per-layer failure isolation


def test_a_raster_build_failure_demotes_only_that_layer_to_an_error_legend_line(tmp_path,
                                                                                 monkeypatch):
    """``_rebuild()`` used to guard only ``NoGeoreference``, so any other
    exception raised while building one layer's actor (a malformed CRS, a corrupt axis, ...) would
    crash the whole flip. Monkeypatches ``field_lonlat_grid`` to raise for one specific field only,
    so the OTHER, healthy layer in the same ``set_layers()`` call proves it still drapes normally."""
    import dynamix.shell.arrangement.scene as scene_mod

    scene = Scene(_plotter())
    good_layer = Layer(layer_id=1, name="Good", source_id="mem:good")
    bad_layer = Layer(layer_id=2, name="Bad", source_id="mem:bad")
    good_field = _geo_field(tmp_path, "good.tif")
    bad_field = _geo_field(tmp_path, "bad.tif")

    real_field_lonlat_grid = scene_mod.field_lonlat_grid

    def _flaky(field, *args, **kwargs):
        if field is bad_field:
            raise ValueError("simulated malformed CRS")
        return real_field_lonlat_grid(field, *args, **kwargs)

    monkeypatch.setattr(scene_mod, "field_lonlat_grid", _flaky)

    scene.set_layers([
        {"layer": good_layer, "field": good_field, "result": None, "status": "ok"},
        {"layer": bad_layer, "field": bad_field, "result": None, "status": "ok"},
    ])

    assert scene.actor_count() == 1
    assert scene._layer_actors == {1: ["layer-1-raster"]}      # the good layer still drapes
    assert len(scene.legend_lines) == 1
    assert "Bad" in scene.legend_lines[0]
    assert "simulated malformed CRS" in scene.legend_lines[0]


# --------------------------------------------------------------------------------------------- clear


def test_clear_removes_every_actor_and_the_legend(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    bare_layer = Layer(layer_id=2, name="Bare", source_id="mem:bare")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([
        {"layer": layer, "field": field, "result": None, "status": "ok"},
        {"layer": bare_layer, "field": _bare_field(), "result": None, "status": "no-georeference"},
    ])
    assert scene.actor_count() == 1
    assert scene.legend_lines != []

    scene.clear()

    assert scene.actor_count() == 0
    assert scene.legend_lines == []
    assert scene._layer_actors == {}

    # set_mode after clear() has nothing to rebuild -- no crash, still empty.
    scene.set_mode("globe")
    assert scene.actor_count() == 0


# --------------------------------------------------------------------------------- vector actors


def test_chains_draw_as_one_polydata_with_a_line_cell_per_chain(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    chains = [_make_chain(5, 5, n=4), _make_chain(10, 10, n=3)]

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result(chains), "status": "ok"},
    ])

    assert scene._layer_actors[1] == ["layer-1-raster", "layer-1-chains"]
    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    assert grid.n_points == 4 + 3
    assert grid.n_cells == 2                     # one polyline cell per chain


def test_chain_with_fewer_than_two_points_is_skipped_but_still_indexed(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    single = _make_chain(5, 5, n=1)
    normal = _make_chain(10, 10, n=3)

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([single, normal]), "status": "ok"},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    assert grid.n_points == 3                    # only the 3-point chain drawn

    starts, chain_indices = scene._chain_lookup[1]
    assert starts.tolist() == [0, 0, 3]           # chain 0: zero-width slice; chain 1: all 3 pts
    assert set(chain_indices.tolist()) == {1}     # every drawn point belongs to chain index 1


def test_chain_color_priority_group_then_seam_then_plain(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    grouped = _make_chain(2, 2, n=2, group_color=[10, 20, 30], tags=["group:fault_a"])
    seamed = _make_chain(6, 6, n=2, tags=["seam_step"])
    plain = _make_chain(20, 20, n=2)

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([grouped, seamed, plain]),
         "status": "ok"},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    colors = np.asarray(grid.point_data["colors"])
    assert list(colors[0][:3]) == [10, 20, 30]           # grouped chain: its own group_color
    assert list(colors[2][:3]) == list(SEAM_COLOR)       # seam-tagged chain: SEAM_COLOR
    assert list(colors[4][:3]) == list(VTRAIL_COLOR)     # plain chain: VTRAIL_COLOR
    assert (colors[:, 3] == 255).all()                   # nothing masked yet: fully opaque


# --------------------------------------------------------------- Per-layer vtrail_color


def test_chain_color_default_vtrail_param_overrides_the_module_constant():
    """Pure unit test of :func:`_chain_color` itself: a plain,
    untagged chain gets whatever ``default_vtrail`` is handed in, falling back to the module's own
    ``VTRAIL_COLOR`` only when ``None`` (every call site that predates this task)."""
    from dynamix.shell.arrangement.scene import _chain_color

    plain = _make_chain(0, 0, n=2)

    assert _chain_color(plain) == VTRAIL_COLOR
    assert _chain_color(plain, default_vtrail=(1, 2, 3)) == (1, 2, 3)
    # group/seam priority is untouched by the parameter
    grouped = _make_chain(0, 0, n=2, group_color=[10, 20, 30], tags=["group:fault_a"])
    assert _chain_color(grouped, default_vtrail=(1, 2, 3)) == (10, 20, 30)
    seamed = _make_chain(0, 0, n=2, tags=["seam_step"])
    assert _chain_color(seamed, default_vtrail=(1, 2, 3)) == SEAM_COLOR


def test_entry_vtrail_color_is_the_plain_chain_fallback(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    plain = _make_chain(20, 20, n=2)

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([plain]), "status": "ok",
         "vtrail_color": (1, 2, 3)},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    colors = np.asarray(grid.point_data["colors"])
    assert list(colors[0][:3]) == [1, 2, 3]


def test_entry_without_vtrail_color_falls_back_to_the_module_constant(tmp_path):
    """Every ``set_layers`` call from before this task omits ``"vtrail_color"`` entirely -- the
    plain chain must still draw exactly ``VTRAIL_COLOR``, unchanged."""
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    plain = _make_chain(20, 20, n=2)

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([plain]), "status": "ok"},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    colors = np.asarray(grid.point_data["colors"])
    assert list(colors[0][:3]) == list(VTRAIL_COLOR)


def test_entry_vtrail_color_does_not_override_group_or_seam_priority(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    grouped = _make_chain(2, 2, n=2, group_color=[10, 20, 30], tags=["group:fault_a"])
    seamed = _make_chain(6, 6, n=2, tags=["seam_step"])
    plain = _make_chain(20, 20, n=2)

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([grouped, seamed, plain]),
         "status": "ok", "vtrail_color": (1, 2, 3)},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    colors = np.asarray(grid.point_data["colors"])
    assert list(colors[0][:3]) == [10, 20, 30]           # grouped: unaffected
    assert list(colors[2][:3]) == list(SEAM_COLOR)       # seam: unaffected
    assert list(colors[4][:3]) == [1, 2, 3]              # plain: the entry's own preference


def test_extrema_actor_draws_the_finest_scale_layer_only(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    ext0 = _make_extrema0([1, 2, 3], [1, 2, 3])
    ext1 = _make_extrema0([40, 41], [40, 41])            # a coarser scale -- must NOT be drawn
    result = {"chains": [], "extrema": [ext0, ext1]}

    scene.set_layers([{"layer": layer, "field": field, "result": result, "status": "ok"}])

    assert scene._layer_actors[1] == ["layer-1-raster", "layer-1-extrema"]
    grid = scene._plotter.actors["layer-1-extrema"].mapper.dataset
    assert grid.n_points == 3


def test_roi_outline_is_a_closed_5point_line_loop_when_roi_present(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    result = _result([], roi=(5, 5, 10, 8))

    scene.set_layers([{"layer": layer, "field": field, "result": result, "status": "ok"}])

    assert scene._layer_actors[1] == ["layer-1-raster", "layer-1-roi"]
    grid = scene._plotter.actors["layer-1-roi"].mapper.dataset
    assert grid.n_points == 5
    pts = np.asarray(grid.points)
    np.testing.assert_allclose(pts[0], pts[4])           # closed loop: first == last


def test_no_roi_key_means_no_roi_actor(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")

    scene.set_layers([{"layer": layer, "field": field, "result": _result([]), "status": "ok"}])

    assert scene._layer_actors[1] == ["layer-1-raster"]


def test_result_none_builds_raster_only_and_no_vector_bookkeeping(tmp_path):
    """Regression: every draping test passes ``result=None`` for an "ok" entry (draping alone is a
    complete entry) -- the vector step must be a pure no-op in that case, not a crash."""
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")

    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    assert scene._layer_actors[1] == ["layer-1-raster"]
    assert scene._chain_lookup == {}
    assert scene._vector_masks == {}


def test_a_vector_build_failure_rolls_back_that_layers_raster_actor_too(tmp_path):
    """The per-layer-isolation contract, extended to vectors: a failure while
    building ONE layer's vector actors must demote the WHOLE layer (raster included) to a single
    "error:<msg>" legend line -- never a raster actor left on screen next to its own error row."""
    scene = Scene(_plotter())
    good_layer = Layer(layer_id=1, name="Good", source_id="mem:good")
    bad_layer = Layer(layer_id=2, name="Bad", source_id="mem:bad")
    good_field = _geo_field(tmp_path, "good.tif")
    bad_field = _geo_field(tmp_path, "bad.tif")

    good_result = _result([_make_chain(5, 5, n=3)])
    # A malformed "extrema" (a string, not a list of per-scale dicts): truthy, so the vector build
    # reaches `extrema_layers[0]` (a single character) and then `.get` on it raises AttributeError.
    bad_result = {"chains": [_make_chain(5, 5, n=3)], "extrema": "not-a-list-of-dicts"}

    scene.set_layers([
        {"layer": good_layer, "field": good_field, "result": good_result, "status": "ok"},
        {"layer": bad_layer, "field": bad_field, "result": bad_result, "status": "ok"},
    ])

    assert scene._layer_actors == {1: ["layer-1-raster", "layer-1-chains"]}
    assert 1 in scene._chain_lookup
    assert 2 not in scene._layer_actors
    assert 2 not in scene._chain_lookup
    assert "layer-2-raster" not in scene._plotter.actors    # rolled back, not orphaned
    # The bad layer's CHAINS actor lands on the plotter (its "chains"
    # are well-formed) before the malformed "extrema" step raises -- an earlier version of
    # _build_vector_actors built its own LOCAL names list and only returned it on success, so this
    # exception discarded that list along with the exception, leaving "layer-2-chains" a
    # PERMANENT orphan: on the plotter, but in no dict any removal path ever walks. Pinned
    # directly, and confirmed it survives a subsequent clear() too (clear() only ever removes what
    # _layer_actors still names -- an untracked actor is invisible to it by construction, so this
    # is the assertion that actually proves the orphan is gone, not merely unlisted).
    assert "layer-2-chains" not in scene._plotter.actors
    assert len(scene.legend_lines) == 1
    assert "Bad" in scene.legend_lines[0]

    scene.clear()
    assert "layer-2-chains" not in scene._plotter.actors
    assert "layer-1-raster" not in scene._plotter.actors
    assert "layer-1-chains" not in scene._plotter.actors


# ------------------------------------------------------------------- points entries


def _points_entry(layer, pointset, color=None):
    return {"layer": layer, "field": pointset, "result": None, "status": "ok",
            "kind": "points", "pointset": pointset, "points_color": color}


def test_points_entry_adds_exactly_one_points_actor():
    scene = Scene(_plotter())
    layer = Layer(layer_id=9, name="Quakes", source_id="mem:q")
    pset = PointSet(lon=np.array([10.0, 20.0, -30.0]), lat=np.array([5.0, -5.0, 40.0]))

    scene.set_layers([_points_entry(layer, pset)])

    assert scene.actor_count() == 1
    assert scene._layer_actors[9] == ["layer-9-points"]
    assert scene.legend_lines == []
    assert "layer-9-points" in scene._plotter.actors


def test_points_entry_drapes_at_surface_z_ignoring_depth():
    """The display law: (lon, lat, ~surface), depth column ignored -- a wildly different depth
    changes nothing about where the point lands. "Surface" is the constant ``_VECTOR_LIFT_KM``
    above the drape: coplanar vector actors lose the depth test to the opaque raster under a
    top-down view. The depth-INDEPENDENCE is what this test exists for."""
    from dynamix.shell.arrangement.scene import _VECTOR_LIFT_KM

    scene = Scene(_plotter())
    layer_shallow = Layer(layer_id=1, name="Shallow", source_id="mem:s")
    layer_deep = Layer(layer_id=2, name="Deep", source_id="mem:d")
    lon, lat = np.array([12.0]), np.array([34.0])
    pset_shallow = PointSet(lon=lon, lat=lat, depth=np.array([1.0]))
    pset_deep = PointSet(lon=lon, lat=lat, depth=np.array([999999.0]))

    scene.set_layers([_points_entry(layer_shallow, pset_shallow),
                      _points_entry(layer_deep, pset_deep)])

    pts_shallow = np.asarray(scene._plotter.actors["layer-1-points"].mapper.dataset.points)
    pts_deep = np.asarray(scene._plotter.actors["layer-2-points"].mapper.dataset.points)
    expected = project(lon, lat, np.full_like(lon, _VECTOR_LIFT_KM), mode=scene.mode)
    np.testing.assert_allclose(pts_shallow, expected)
    np.testing.assert_allclose(pts_deep, expected)


def test_points_entry_uses_its_own_color_falling_back_to_the_module_default():
    scene = Scene(_plotter())
    styled_layer = Layer(layer_id=1, name="Styled", source_id="mem:s")
    plain_layer = Layer(layer_id=2, name="Plain", source_id="mem:p")
    pset = PointSet(lon=np.array([1.0]), lat=np.array([2.0]))

    scene.set_layers([_points_entry(styled_layer, pset, color=(10, 20, 30)),
                      _points_entry(plain_layer, pset, color=None)])

    styled_actor = scene._plotter.actors["layer-1-points"]
    plain_actor = scene._plotter.actors["layer-2-points"]
    assert tuple(int(round(c * 255)) for c in styled_actor.prop.color)[:3] == (10, 20, 30)
    assert tuple(int(round(c * 255)) for c in plain_actor.prop.color)[:3] == POINTS_COLOR


def test_points_entry_rebuilds_with_new_coordinates_on_a_mode_switch():
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="Quakes", source_id="mem:q")
    pset = PointSet(lon=np.array([100.0]), lat=np.array([10.0]))
    scene.set_layers([_points_entry(layer, pset)])
    before = np.asarray(scene._plotter.actors["layer-1-points"].mapper.dataset.points).copy()

    scene.set_mode("globe")

    after = np.asarray(scene._plotter.actors["layer-1-points"].mapper.dataset.points)
    assert scene.actor_count() == 1
    assert not np.allclose(before, after)


def test_a_malformed_pointset_demotes_only_that_layer_to_an_error_legend_line(tmp_path):
    """Same crash-avoidance posture as the raster path's own
    ``test_a_vector_build_failure_rolls_back_that_layers_raster_actor_too``: one layer's bad data
    never takes the whole flip down."""
    scene = Scene(_plotter())
    good_layer = Layer(layer_id=1, name="Good", source_id="mem:good")
    bad_layer = Layer(layer_id=2, name="Bad", source_id="mem:bad")
    good_pset = PointSet(lon=np.array([1.0]), lat=np.array([2.0]))
    # Mismatched lon/lat lengths -- `project`'s own column_stack raises on this shape mismatch.
    bad_pset = PointSet(lon=np.array([1.0, 2.0]), lat=np.array([3.0]))

    scene.set_layers([_points_entry(good_layer, good_pset),
                      _points_entry(bad_layer, bad_pset)])

    assert scene._layer_actors == {1: ["layer-1-points"]}
    assert 2 not in scene._layer_actors
    assert "layer-2-points" not in scene._plotter.actors
    assert len(scene.legend_lines) == 1
    assert "Bad" in scene.legend_lines[0]


# ------------------------------------------------------------------------------------ set_colormap


def test_set_colormap_swaps_the_lut_object_but_preserves_actor_identity(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    actor = scene._plotter.actors["layer-1-raster"]
    old_lut = actor.mapper.lookup_table
    old_range = tuple(actor.mapper.scalar_range)

    scene.set_colormap("magma")

    assert scene._plotter.actors["layer-1-raster"] is actor    # same actor object
    new_lut = actor.mapper.lookup_table
    assert new_lut is not old_lut                              # different LUT object
    assert tuple(new_lut.GetRange()) == pytest.approx(old_range)
    assert new_lut.nan_opacity == 0            # sentinel cells stay transparent post-swap too


def test_set_colormap_is_remembered_across_a_later_set_mode_rebuild(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])
    scene.set_colormap("magma")

    scene.set_mode("globe")

    assert scene._colormap == "magma"          # _add_raster_actor reads this, not a literal
    assert scene.actor_count() == 1


def test_set_colormap_does_not_touch_vector_actors(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    chains = [_make_chain(5, 5, n=3)]
    scene.set_layers([{"layer": layer, "field": field, "result": _result(chains), "status": "ok"}])

    chains_actor = scene._plotter.actors["layer-1-chains"]
    scene.set_colormap("magma")

    assert scene._plotter.actors["layer-1-chains"] is chains_actor


# ------------------------------------------------------------------ Per-layer colormap


def _lut_table(actor):
    """The raster actor's LUT as a plain ``(256, 4)`` uint8 array -- comparable by VALUE, not
    just by object identity."""
    return pv.convert_array(actor.mapper.lookup_table.GetTable())


def test_entry_colormap_is_used_for_that_layers_own_drape(tmp_path):
    """A layer's own ``"colormap"`` entry produces the SAME table a scene-wide
    ``set_colormap`` of the identical name would -- proving the per-layer value actually reaches
    ``add_mesh``'s own ``cmap=`` argument, not merely that it does not crash."""
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")

    scene_wide = Scene(_plotter())
    scene_wide.set_colormap("plasma")
    scene_wide.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])
    expected = _lut_table(scene_wide._plotter.actors["layer-1-raster"])

    per_layer = Scene(_plotter())      # scene-wide default left at "viridis"
    per_layer.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok",
                           "colormap": "plasma"}])
    actual = _lut_table(per_layer._plotter.actors["layer-1-raster"])

    np.testing.assert_array_equal(actual, expected)


def test_entry_without_colormap_falls_back_to_the_scene_wide_default(tmp_path):
    """Every ``set_layers`` call from before this task omits ``"colormap"`` entirely -- the drape
    must still use ``self._colormap`` (:meth:`Scene.set_colormap`'s own target), unchanged."""
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")

    scene = Scene(_plotter())
    scene.set_colormap("magma")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    actual = _lut_table(scene._plotter.actors["layer-1-raster"])
    expected = pv.convert_array(pv.LookupTable(cmap="magma").GetTable())
    np.testing.assert_array_equal(actual, expected)


def test_two_layers_can_each_carry_a_different_colormap_at_once(tmp_path):
    scene = Scene(_plotter())
    layer_a = Layer(layer_id=1, name="A", source_id="mem:a")
    layer_b = Layer(layer_id=2, name="B", source_id="mem:b")
    field_a = _geo_field(tmp_path, "a.tif")
    field_b = _geo_field(tmp_path, "b.tif")

    scene.set_layers([
        {"layer": layer_a, "field": field_a, "result": None, "status": "ok",
         "colormap": "plasma"},
        {"layer": layer_b, "field": field_b, "result": None, "status": "ok",
         "colormap": "magma"},
    ])

    lut_a = _lut_table(scene._plotter.actors["layer-1-raster"])
    lut_b = _lut_table(scene._plotter.actors["layer-2-raster"])
    assert not np.array_equal(lut_a, lut_b)
    np.testing.assert_array_equal(lut_a, pv.convert_array(pv.LookupTable(cmap="plasma").GetTable()))
    np.testing.assert_array_equal(lut_b, pv.convert_array(pv.LookupTable(cmap="magma").GetTable()))


# ------------------------------------------------------------------------------------------- Graticule, vertical exaggeration, runtime background -- the View dialog's
# Camera/Frame/Display capabilities on the Scene side.


def test_set_graticule_true_adds_exactly_one_actor():
    scene = Scene(_plotter())
    before = set(scene._plotter.actors)

    scene.set_graticule(True)

    added = set(scene._plotter.actors) - before
    assert added == {_GRATICULE_NAME}


def test_set_graticule_false_removes_the_actor():
    scene = Scene(_plotter())
    before = set(scene._plotter.actors)
    scene.set_graticule(True)

    scene.set_graticule(False)

    assert set(scene._plotter.actors) == before


def test_set_graticule_true_twice_does_not_add_a_second_actor():
    scene = Scene(_plotter())
    scene.set_graticule(True)
    actor_before = scene._plotter.actors[_GRATICULE_NAME]

    scene.set_graticule(True)          # already on -- must be a no-op, not a second add

    assert scene._plotter.actors[_GRATICULE_NAME] is actor_before


def test_graticule_rebuilds_with_new_coordinates_on_a_mode_switch():
    scene = Scene(_plotter())
    scene.set_graticule(True)
    mercator_points = np.array(scene._plotter.actors[_GRATICULE_NAME].mapper.dataset.points)

    scene.set_mode("globe")

    assert _GRATICULE_NAME in scene._plotter.actors     # still exactly one -- not dropped
    globe_points = np.array(scene._plotter.actors[_GRATICULE_NAME].mapper.dataset.points)
    assert globe_points.shape == mercator_points.shape
    assert not np.allclose(globe_points, mercator_points)


def test_graticule_survives_a_set_layers_rebuild(tmp_path):
    scene = Scene(_plotter())
    scene.set_graticule(True)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")

    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    assert _GRATICULE_NAME in scene._plotter.actors


def test_graticule_off_by_default_and_never_counted_by_actor_count(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])
    assert _GRATICULE_NAME not in scene._plotter.actors

    scene.set_graticule(True)

    assert scene.actor_count() == 1     # the raster only -- the graticule is chrome, not a layer


def test_set_vertical_exaggeration_threads_the_factor_into_every_project_call(tmp_path, monkeypatch):
    """``Scene`` calls the MODULE-LEVEL ``project`` name (``from dynamix.core.projection import
    project``), so patching it on ``dynamix.shell.arrangement.scene`` intercepts every call site
    in this module -- the same monkeypatch target ``test_lonlat_grid_is_cached_across_set_mode_
    rebuilds`` (below) already uses for ``field_lonlat_grid``, applied to project() instead."""
    import dynamix.shell.arrangement.scene as scene_mod

    calls: list[float] = []
    real_project = scene_mod.project

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("vexag"))
        return real_project(*args, **kwargs)

    monkeypatch.setattr(scene_mod, "project", _spy)

    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])
    assert calls and calls[-1] == 1.0          # the untouched default

    calls.clear()
    scene.set_vertical_exaggeration(2.5)

    assert scene._vexag == 2.5
    assert calls                                # set_vertical_exaggeration triggered a real rebuild
    assert all(v == 2.5 for v in calls)          # ...with the NEW factor threaded through every call


def test_set_vertical_exaggeration_rebuilds_raster_actor_geometry(tmp_path):
    """Every drape height is currently always 0 (module docstring), so a plain point-position
    comparison would prove nothing about ``vexag`` reaching ``project`` -- confirmed instead, more
    directly, via the monkeypatch spy above. This test pins the OTHER half: the call is a genuine
    rebuild (matches ``set_mode``'s own "actor count unchanged, same identity name" contract), not
    a silent no-op."""
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    scene.set_vertical_exaggeration(4.0)

    assert scene.actor_count() == 1
    assert scene._plotter.actors["layer-1-raster"] is not None


def test_set_background_at_runtime_sets_the_plotters_background():
    scene = Scene(_plotter())          # no constructor background -- runtime setter only

    scene.set_background("#204060")

    assert scene._plotter.background_color == pv.Color("#204060")


def test_set_background_at_runtime_can_change_a_construction_time_background():
    scene = Scene(_plotter(), background="#131313")

    scene.set_background("#204060")

    assert scene._plotter.background_color == pv.Color("#204060")


# ------------------------------------------------------------------------------------- frame mode
#
# Scene.set_frame_mode -- the Vector tab's placement switch. `_bare_field()` (LocalFrame, no CRS
# in provenance) is the fixture throughout: it is deliberately what today's geo mode CANNOT place
# (NoGeoreference) and what frame mode places without ever touching geo.mapping.


def test_frame_mode_chains_draw_for_a_local_frame_field_with_no_georeference(tmp_path):
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chains = [_make_chain(1, 1, n=3)]

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result(chains), "status": "ok"},
    ])

    assert scene._layer_actors[1] == ["layer-1-raster", "layer-1-chains"]
    assert 1 in scene._chain_lookup
    assert scene.legend_lines == []


def test_frame_mode_false_the_same_entry_shows_the_no_georeference_legend_line(tmp_path):
    scene = Scene(_plotter())          # frame_mode left False -- today's byte-for-byte geo mode
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chains = [_make_chain(1, 1, n=3)]

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result(chains), "status": "ok"},
    ])

    assert scene._layer_actors == {}
    assert 1 not in scene._chain_lookup
    assert len(scene.legend_lines) == 1
    assert "no georeference" in scene.legend_lines[0]


def test_frame_mode_points_entry_demotes_to_a_session_view_only_legend_line():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    layer = Layer(layer_id=9, name="Quakes", source_id="mem:q")
    pset = PointSet(lon=np.array([10.0]), lat=np.array([5.0]))

    scene.set_layers([_points_entry(layer, pset)])

    assert scene.actor_count() == 0
    assert scene._layer_actors == {}
    assert len(scene.legend_lines) == 1
    assert "Quakes" in scene.legend_lines[0]
    assert "session view only (frame mode)" in scene.legend_lines[0]


def test_add_points_geometry_raises_directly_when_called_in_frame_mode():
    """Defense in depth (_add_points_geometry's own docstring): _rebuild itself never reaches
    this call in frame mode (the test above proves that), but the guard must still hold for any
    other caller."""
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    pset = PointSet(lon=np.array([1.0]), lat=np.array([2.0]))

    with pytest.raises(ValueError, match="not drawable in a native frame"):
        scene._add_points_geometry(layer, pset)


def test_set_frame_mode_flip_triggers_a_rebuild_with_a_different_actor_set(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chains = [_make_chain(1, 1, n=3)]
    scene.set_layers([
        {"layer": layer, "field": field, "result": _result(chains), "status": "ok"},
    ])
    assert scene.actor_count() == 0             # geo mode: NoGeoreference -> legend only
    assert len(scene.legend_lines) == 1

    scene.set_frame_mode(True)

    assert scene.actor_count() == 2              # raster + chains
    assert scene.legend_lines == []
    assert 1 in scene._chain_lookup

    scene.set_frame_mode(False)

    assert scene.actor_count() == 0
    assert len(scene.legend_lines) == 1


def test_set_frame_mode_same_value_is_a_no_op(monkeypatch):
    scene = Scene(_plotter())
    calls = {"n": 0}
    real_rebuild = scene._rebuild

    def counting():
        calls["n"] += 1
        real_rebuild()

    monkeypatch.setattr(scene, "_rebuild", counting)

    scene.set_frame_mode(False)         # already False -- must not rebuild

    assert calls["n"] == 0


def test_graticule_never_applied_in_frame_mode():
    scene = Scene(_plotter())
    scene.set_graticule(True)
    assert _GRATICULE_NAME in scene._plotter.actors

    scene.set_frame_mode(True)

    assert _GRATICULE_NAME not in scene._plotter.actors


def test_set_graticule_true_while_already_in_frame_mode_adds_nothing():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)

    scene.set_graticule(True)

    assert _GRATICULE_NAME not in scene._plotter.actors


def test_frame_mode_extrema_and_roi_place_via_the_frame_not_lonlat(tmp_path):
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    ext0 = _make_extrema0([1, 2, 3], [1, 2, 3])
    result = {"chains": [], "extrema": [ext0], "_roi": {"roi": (1, 1, 3, 3)}}

    scene.set_layers([{"layer": layer, "field": field, "result": result, "status": "ok"}])

    assert scene._layer_actors[1] == ["layer-1-raster", "layer-1-extrema", "layer-1-roi"]
    ext_pts = np.asarray(scene._plotter.actors["layer-1-extrema"].mapper.dataset.points)
    expected = np.asarray(field.frame.to_scene(field.x_axis[[1, 2, 3]],
                                               field.y_axis[[1, 2, 3]]), dtype=np.float64)
    # Vector actors ride _frame_vector_lift above the drape (breaks the top-down depth tie);
    # x/y placement is the frame's own.
    expected[:, 2] += Scene._frame_vector_lift(field)
    np.testing.assert_allclose(ext_pts, expected)
    roi_grid = scene._plotter.actors["layer-1-roi"].mapper.dataset
    assert roi_grid.n_points == 5
    roi_pts = np.asarray(roi_grid.points)
    np.testing.assert_allclose(roi_pts[0], roi_pts[4])     # closed loop: first == last


def test_frame_mode_raster_places_points_via_the_frame_not_lonlat(tmp_path):
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()

    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    grid = scene._plotter.actors["layer-1-raster"].mapper.dataset
    assert grid.n_points == field.nx * field.ny
    pts = np.asarray(grid.points)
    x2d, y2d = np.meshgrid(field.x_axis, field.y_axis)
    expected = field.frame.to_scene(x2d.ravel(), y2d.ravel())
    np.testing.assert_allclose(pts, expected)


def test_frame_mode_is_planar_for_geographic_frames_no_world_map_wrap():
    """Frame mode places a GeographicFrame field's own CONTINUOUS axes as planar coordinates --
    never through ``frame.to_scene``, whose ``projection.project`` branch applies the world-map
    ``mod 360`` longitude canonicalisation. Through that branch, under "pacific" a lon range
    crossing 0 (here -30..+12.5, the demo DEM's own span) tears into two edge strips (0..12.5
    far left, 330..360 far right) with the bridging cells smeared across the whole width. The
    assertions pin the planar contract directly: raster x strictly monotonic, spanning exactly
    the axes' own range -- a wrap is impossible, not just absent. Chains placed through
    ``_scene_points`` must land in the SAME planar frame as the drape (the drape/vector
    agreement the raster docstring names)."""
    from dynamix.core.frames import GeographicFrame

    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    ny, nx = 8, 16
    field = RasterField._from_bare_array(np.zeros((ny, nx)), "geo_planar",
                                         frame=GeographicFrame(), name="geo_planar")
    field.x_axis = np.linspace(-30.0, 12.5, nx)          # crosses lon 0 -- the pacific seam
    field.y_axis = np.linspace(-23.0, 19.0, ny)

    chains = [_make_chain(2, 3, n=3)]
    scene.set_layers([{"layer": layer, "field": field, "result": _result(chains),
                       "status": "ok"}])

    pts = np.asarray(scene._plotter.actors["layer-1-raster"].mapper.dataset.points)
    row_x = pts[:nx, 0]                                   # first grid row's x coordinates
    assert np.all(np.diff(row_x) > 0)                     # strictly monotonic: no mod-360 fold
    np.testing.assert_allclose((row_x[0], row_x[-1]), (-30.0, 12.5))
    assert pts[:, 0].max() < 180.0                        # nothing canonicalised to 330..360

    chain_pts = np.asarray(
        scene._plotter.actors[scene._chains_actor_name(1)].mapper.dataset.points)
    assert np.all(chain_pts[:, 0] >= -30.0) and np.all(chain_pts[:, 0] <= 12.5)
    # camera-bounds signature agrees with the drawn planar extent (Edit 3 of the same fix)
    assert scene._frame_bounds_signature() == "-30.000,12.500,-23.000,19.000"


# --------------------------------------------------------------------------------- Per-view
# camera memory (fixes "vector-view raster stretched all weird"). Real vtkCamera state via a plain off-screen
# pv.Plotter (the triage's own bug3_camera_repro.py probe did exactly this) -- no QtInteractor.


def _frame_field(nx=64, ny=32, dx=2.0, dy=-3.0, x0=100.0, y0=-50.0, name="frame"):
    """The triage's own bug3_camera_repro.py fixture, verbatim: an asymmetric, non-uniform,
    negative-dy LocalFrame -- x spans [100, 226], y spans [-143, -50], largest span 126."""
    return RasterField._from_bare_array(
        np.zeros((ny, nx)), name, name=name,
        frame=LocalFrame(dx=dx, dy=dy, x0=x0, y0=y0, units="px"))


def test_geo_to_frame_flip_refits_the_camera_away_from_the_stale_geo_scale(tmp_path):
    """The triage's measured numbers, as a regression fixture: parallel_scale 0.0055 vs. mesh
    span 126, byte-identical camera across the flip. After the fix, a geo->frame flip must land
    the camera on the frame mesh's own (radically different-scale) bounds instead."""
    scene = Scene(_plotter())
    geo_layer = Layer(layer_id=1, name="Geo", source_id="mem:geo")
    geo_field = _geo_field(tmp_path, "geo.tif")
    scene.set_layers([{"layer": geo_layer, "field": geo_field, "result": None, "status": "ok"}])
    stale_position = tuple(scene._plotter.camera.position)
    stale_parallel_scale = scene._plotter.camera.parallel_scale

    frame_layer = Layer(layer_id=2, name="Frame", source_id="mem:frame")
    frame_field = _frame_field()               # x:[100,226] y:[-143,-50], span 126
    mesh_span = 126.0

    scene.set_frame_mode(True)
    scene.set_layers([{"layer": frame_layer, "field": frame_field, "result": None, "status": "ok"}])

    cam = scene._plotter.camera
    assert tuple(cam.position) != stale_position
    assert cam.parallel_scale != pytest.approx(stale_parallel_scale)
    # the triage's own ~11,500x mismatch (0.0055 / 126) must be impossible now -- the fitted
    # half-height is within the new mesh's own order of magnitude, not the old geo one's.
    assert cam.parallel_scale > mesh_span / 100.0


def test_frame_to_geo_restores_the_saved_geo_camera_exactly(tmp_path):
    scene = Scene(_plotter())
    geo_layer = Layer(layer_id=1, name="Geo", source_id="mem:geo")
    geo_field = _geo_field(tmp_path, "geo.tif")
    scene.set_layers([{"layer": geo_layer, "field": geo_field, "result": None, "status": "ok"}])
    geo_position = tuple(scene._plotter.camera.position)
    geo_focal = tuple(scene._plotter.camera.focal_point)
    geo_parallel_scale = scene._plotter.camera.parallel_scale

    frame_layer = Layer(layer_id=2, name="Frame", source_id="mem:frame")
    frame_entry = {"layer": frame_layer, "field": _bare_field(), "result": None, "status": "ok"}
    scene.set_frame_mode(True)
    scene.set_layers([frame_entry])
    scene._plotter.camera.zoom(2.0)             # the user navigates within frame mode

    scene.set_frame_mode(False)
    scene.set_layers([{"layer": geo_layer, "field": geo_field, "result": None, "status": "ok"}])

    cam = scene._plotter.camera
    assert tuple(cam.position) == pytest.approx(geo_position)
    assert tuple(cam.focal_point) == pytest.approx(geo_focal)
    assert cam.parallel_scale == pytest.approx(geo_parallel_scale)


def test_second_frame_entry_restores_the_users_adjusted_camera_not_a_refit(tmp_path):
    scene = Scene(_plotter())
    geo_layer = Layer(layer_id=1, name="Geo", source_id="mem:geo")
    geo_field = _geo_field(tmp_path, "geo.tif")
    scene.set_layers([{"layer": geo_layer, "field": geo_field, "result": None, "status": "ok"}])

    frame_layer = Layer(layer_id=2, name="Frame", source_id="mem:frame")
    frame_entry = {"layer": frame_layer, "field": _bare_field(), "result": None, "status": "ok"}

    scene.set_frame_mode(True)                  # 1st entry -- unknown key, auto re-fit
    scene.set_layers([frame_entry])
    fitted_parallel_scale = scene._plotter.camera.parallel_scale

    scene._plotter.camera.parallel_scale = fitted_parallel_scale * 3.0    # user zooms out
    adjusted_position = tuple(scene._plotter.camera.position)
    adjusted_parallel_scale = scene._plotter.camera.parallel_scale

    scene.set_frame_mode(False)                 # remembers the ADJUSTED frame camera
    scene.set_layers([{"layer": geo_layer, "field": geo_field, "result": None, "status": "ok"}])

    scene.set_frame_mode(True)                  # 2nd entry -- key now known
    scene.set_layers([frame_entry])

    cam = scene._plotter.camera
    assert cam.parallel_scale == pytest.approx(adjusted_parallel_scale)
    assert cam.parallel_scale != pytest.approx(fitted_parallel_scale)
    assert tuple(cam.position) == pytest.approx(adjusted_position)


def test_frame_mode_active_layer_swap_refits_but_same_layer_resync_leaves_camera_untouched():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)

    layer_a = Layer(layer_id=1, name="A", source_id="mem:a")
    field_a = _bare_field()
    entry_a = {"layer": layer_a, "field": field_a, "result": None, "status": "ok"}
    scene.set_layers([entry_a])                 # first-ever entries -> was_empty fit

    navigated_parallel_scale = scene._plotter.camera.parallel_scale * 5.0
    scene._plotter.camera.parallel_scale = navigated_parallel_scale     # user navigates

    # Same-layer RESYNC: identical field/bounds, a FRESH dict -- mirrors a real resolve() call
    # (every Filter always copies, dynamix/engine/resolve.py's own module docstring).
    scene.set_layers([dict(entry_a)])
    assert scene._plotter.camera.parallel_scale == pytest.approx(navigated_parallel_scale)

    # ACTIVE-LAYER SWAP: a different layer, disjoint bounds -> must re-fit.
    layer_b = Layer(layer_id=2, name="B", source_id="mem:b")
    field_b = RasterField._from_bare_array(
        np.zeros((8, 8)), "b", name="b", frame=LocalFrame(x0=1000.0, y0=1000.0, units="px"))
    scene.set_layers([{"layer": layer_b, "field": field_b, "result": None, "status": "ok"}])

    assert scene._plotter.camera.parallel_scale != pytest.approx(navigated_parallel_scale)


def test_camera_key_is_mode_scoped_in_geo_and_bounds_scoped_in_frame():
    scene = Scene(_plotter())
    assert scene.camera_key() == "geo:mercator"
    scene.set_mode("pacific")
    assert scene.camera_key() == "geo:pacific"

    scene.set_frame_mode(True)
    empty_key = scene.camera_key()
    assert empty_key.startswith("frame:")

    layer_a = Layer(layer_id=1, name="A", source_id="mem:a")
    scene.set_layers([{"layer": layer_a, "field": _bare_field(), "result": None, "status": "ok"}])
    key_a = scene.camera_key()
    assert key_a.startswith("frame:") and key_a != empty_key

    layer_b = Layer(layer_id=2, name="B", source_id="mem:b")
    field_b = RasterField._from_bare_array(
        np.zeros((8, 8)), "b", name="b", frame=LocalFrame(x0=1000.0, units="px"))
    scene.set_layers([{"layer": layer_b, "field": field_b, "result": None, "status": "ok"}])
    assert scene.camera_key() != key_a


def test_remember_and_restore_camera_round_trip():
    scene = Scene(_plotter())
    cam = scene._plotter.camera
    cam.position = (1.0, 2.0, 3.0)
    cam.focal_point = (0.0, 0.0, 0.0)
    cam.up = (0.0, 1.0, 0.0)
    cam.parallel_scale = 5.0
    cam.SetParallelProjection(True)
    cam.clipping_range = (0.1, 100.0)

    scene.remember_camera()
    key = scene.camera_key()
    cam.position = (9.0, 9.0, 9.0)               # simulate navigating away

    assert scene.restore_camera(key) is True
    assert tuple(scene._plotter.camera.position) == pytest.approx((1.0, 2.0, 3.0))
    assert scene._plotter.camera.parallel_scale == pytest.approx(5.0)
    assert scene._plotter.camera.GetParallelProjection() == 1


def test_restore_camera_unknown_key_returns_false_and_leaves_camera_untouched():
    scene = Scene(_plotter())
    before = tuple(scene._plotter.camera.position)

    assert scene.restore_camera("nonsense-key") is False

    assert tuple(scene._plotter.camera.position) == before


def test_remember_camera_only_overwrites_the_current_views_own_slot(tmp_path):
    """The 'r'/reset-button flow (ArrangementView.reset_camera -> MomentumCamera.reset ->
    Scene.remember_camera) must touch ONLY the current view's own memory slot -- resetting while
    in frame mode must never clobber a remembered geo camera, and vice versa."""
    scene = Scene(_plotter())
    geo_layer = Layer(layer_id=1, name="Geo", source_id="mem:geo")
    geo_field = _geo_field(tmp_path, "geo.tif")
    scene.set_layers([{"layer": geo_layer, "field": geo_field, "result": None, "status": "ok"}])
    scene.remember_camera()
    geo_key = scene.camera_key()
    geo_state = dict(scene._camera_memory[geo_key])

    frame_layer = Layer(layer_id=2, name="Frame", source_id="mem:frame")
    scene.set_frame_mode(True)
    scene.set_layers([{"layer": frame_layer, "field": _bare_field(), "result": None, "status": "ok"}])
    scene._plotter.camera.parallel_scale *= 2.0   # simulate user navigation in frame mode
    scene.remember_camera()                        # the reset-button's own memory-overwrite step

    assert scene._camera_memory[geo_key] == geo_state
    assert scene.camera_key() != geo_key


# ---------------------------------------- Camera-memory export/import
#
# ``Project`` has an additive ``"cameras"`` key; these two methods
# are the ONLY seam between that key and this ``Scene``'s own live ``_camera_memory`` -- a
# viewpoint that survives a process boundary is the whole goal. No plotter access in either,
# so this module's own off-screen-plotter pattern covers them.


def _camera_payload(**over):
    """One ``Scene._camera_snapshot()`` shape as it looks after a JSON round trip -- every tuple
    already a list, which is exactly what ``Project.cameras`` hands back from a saved file."""
    payload = {
        "position": [1.0, 2.0, 3.0],
        "focal_point": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "parallel_scale": 5.0,
        "parallel_projection": True,
        "clipping_range": [0.1, 100.0],
    }
    payload.update(over)
    return payload


def test_export_camera_memory_is_json_safe():
    scene = Scene(_plotter())
    cam = scene._plotter.camera
    cam.position = (1.0, 2.0, 3.0)
    cam.focal_point = (0.0, 0.0, 0.0)
    cam.up = (0.0, 1.0, 0.0)
    cam.parallel_scale = 5.0
    cam.SetParallelProjection(True)
    cam.clipping_range = (0.1, 100.0)
    scene.remember_camera()
    key = scene.camera_key()

    exported = scene.export_camera_memory()

    # json.dumps would happily flatten a tuple on the way out and never re-tuple it on the way
    # back; equality after a full dumps->loads cycle is what proves the export ALREADY carries
    # the on-disk types, so Project.cameras cannot silently re-type a viewpoint.
    assert json.loads(json.dumps(exported)) == exported
    assert exported[key]["position"] == [1.0, 2.0, 3.0]
    assert exported[key]["parallel_scale"] == pytest.approx(5.0)

    exported[key]["parallel_scale"] = 999.0          # a copy, not a window onto the live dict
    assert scene._camera_memory[key]["parallel_scale"] == pytest.approx(5.0)


def test_import_camera_memory_merges_without_dropping_live_keys():
    """Opening a project must not discard the viewpoints this session already has -- import
    MERGES. A key the payload also names is the file's own, more deliberate, saved viewpoint, so
    that one wins; every key it does not name is left exactly as it was."""
    scene = Scene(_plotter())
    scene.remember_camera()
    live_key = scene.camera_key()
    live_state = dict(scene._camera_memory[live_key])

    scene.import_camera_memory({"frame:0.000,0.000,1.000,1.000": _camera_payload()})

    assert scene._camera_memory[live_key] == live_state        # the live session kept its own
    assert "frame:0.000,0.000,1.000,1.000" in scene._camera_memory

    scene.import_camera_memory({live_key: _camera_payload(parallel_scale=7.0)})

    assert scene._camera_memory[live_key]["parallel_scale"] == pytest.approx(7.0)


def test_import_camera_memory_skips_a_malformed_entry_without_raising():
    """A remembered viewpoint is a preference, not a measurement: a corrupt entry
    is dropped silently rather than blocking the project that carries it."""
    scene = Scene(_plotter())

    scene.import_camera_memory({
        "geo:mercator": _camera_payload(parallel_scale=7.0),
        "frame:truncated": {"position": [1.0, 2.0, 3.0]},      # missing every other field
        "frame:not-a-dict": "nonsense",
        7: _camera_payload(),                                   # not a key a Scene could mint
    })

    assert list(scene._camera_memory) == ["geo:mercator"]
    assert scene._camera_memory["geo:mercator"]["parallel_scale"] == pytest.approx(7.0)


def test_remembered_camera_round_trips_through_export_import_and_restores():
    """The whole of G2, end to end: remember here, export, cross a real JSON boundary, import
    into a Scene that has never seen the key, and restore the SAME viewpoint."""
    scene = Scene(_plotter())
    cam = scene._plotter.camera
    cam.position = (11.0, 12.0, 13.0)
    cam.focal_point = (1.0, 1.0, 1.0)
    cam.up = (0.0, 0.0, 1.0)
    cam.parallel_scale = 42.0
    cam.SetParallelProjection(True)
    cam.clipping_range = (0.5, 500.0)
    scene.remember_camera()
    key = scene.camera_key()
    payload = json.loads(json.dumps(scene.export_camera_memory()))   # exactly what a file holds

    fresh = Scene(_plotter())
    assert fresh.restore_camera(key) is False        # a brand-new Scene remembers nothing

    fresh.import_camera_memory(payload)

    assert fresh.restore_camera(key) is True
    assert tuple(fresh._plotter.camera.position) == pytest.approx((11.0, 12.0, 13.0))
    assert fresh._plotter.camera.parallel_scale == pytest.approx(42.0)
    assert fresh._camera_memory[key] == scene._camera_memory[key]    # re-tupled, snapshot shape


# ------------------------------------------------------------- Scale-space chain stacking
#
# Frame mode (LocalFrame's own pure pass-through ``to_scene``) is the seam used
# for the exact z-math assertions below: geo mode's own flat-view height conversion (``_z_height``,
# ``core/projection.py``) divides by ``KM_PER_DEG``, exactly proportional but not a byte-for-byte
# match to ``stretch * (log2 a - log2 a_min)`` -- frame mode has no such conversion, so it is the
# precise, deterministic seam this task's own z-lift formula is tested against. Geo mode gets its
# own dedicated "byte-identical when disabled" test below, matching the design's instruction.


def test_scale_space_disabled_z_is_zero_geo_mode_byte_identical(tmp_path):
    """Geo mode + checkbox off: the SCALE-SPACE z-lift must never leak into the disabled path
    (default Scene state; set_scale_space is never called here). The constant per-vector drape
    lift -- ``_VECTOR_LIFT_KM`` through ``project`` -- IS present, so "no scale-space lift" means
    a single shared nonzero constant z across every point; scale-VARYING z is what this test
    forbids."""
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    chains = [_make_chain(5, 5, n=4), _make_chain(10, 10, n=3)]

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result(chains), "status": "ok"},
    ])

    pts = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.points)
    assert np.allclose(pts[:, 2], pts[0, 2])            # one constant lift, no scale variation
    assert pts[0, 2] != 0.0                              # and genuinely above the drape


def test_scale_space_enabled_z_matches_stretch_times_log2a_minus_amin_frame_mode():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    stretch = 3.0
    scene.set_scale_space(True, stretch)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    # Asymmetric: different lengths AND non-uniform (but individually gap-free) scale ladders --
    # chain_a starts finer than chain_b, exercising a genuinely cross-chain a_min.
    chain_a = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.5])
    chain_b = _make_chain_with_scales(4, 0, [1.0, 3.0])

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([chain_a, chain_b]), "status": "ok"},
    ])

    pts = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.points)
    all_log2a = np.concatenate([chain_a["log2_scales"], chain_b["log2_scales"]])
    a_min = float(np.min(all_log2a))
    # The scale-space law plus the constant drape lift (breaks the top-down depth tie).
    lift = Scene._frame_vector_lift(field)
    expected_z = stretch * (all_log2a - a_min) + lift
    assert pts[:, 2] == pytest.approx(expected_z, abs=1e-12)
    assert pts[0, 2] == pytest.approx(lift)          # the layer's own finest point sits at the lift


def test_scale_space_disabled_after_being_enabled_returns_to_zero():
    """Toggling OFF must fully retract the lift, not merely stop growing it further."""
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    scene.set_scale_space(True, 4.0)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chain = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.0])
    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([chain]), "status": "ok"},
    ])
    lifted = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.points)[:, 2]
    assert not np.allclose(lifted, lifted[0])            # scale-varying while enabled

    scene.set_scale_space(False, 4.0)

    pts = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.points)
    # OFF retracts to the constant drape lift.
    assert np.allclose(pts[:, 2], Scene._frame_vector_lift(field))


def test_scale_space_stretch_change_rebuilds_with_a_different_z_span():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    scene.set_scale_space(True, 1.0)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chain = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.0])
    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([chain]), "status": "ok"},
    ])
    z_before = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.points)[:, 2].copy()

    scene.set_scale_space(True, 10.0)

    z_after = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.points)[:, 2]
    # Compare SPANS (lift-invariant) -- the constant drape lift shifts both renders identically,
    # so the stretch relation holds on max - min, not on raw max.
    span_before = z_before.max() - z_before.min()
    span_after = z_after.max() - z_after.min()
    assert span_after == pytest.approx(span_before * 10.0)
    assert span_after != pytest.approx(span_before)


def test_scale_space_missing_scale_step_splits_into_two_line_cells():
    """A dropped scale is an honest gap, never a fake straight link.
    ``n_voice=1`` -> expected_step = 1.0; the 2.0 -> 5.0 jump is a 3-step gap (> 1.5x that step),
    splitting the chain into a 3-point run ``[0.0, 1.0, 2.0]`` and a 2-point run ``[5.0, 6.0]``."""
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    scene.set_scale_space(True, 1.0)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chain = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.0, 5.0, 6.0])
    result = _result([chain])
    result["params"] = {"n_voice": 1}

    scene.set_layers([
        {"layer": layer, "field": field, "result": result, "status": "ok"},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    assert grid.n_points == 5              # every point still drawn
    assert grid.n_cells == 2               # ...but as two separate line cells, not a fake link


def test_scale_space_no_gap_stays_one_line_cell():
    """The ``n_voice=1`` ladder's own control case: a genuinely gap-free chain stays ONE cell."""
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    scene.set_scale_space(True, 1.0)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chain = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.0, 3.0])
    result = _result([chain])
    result["params"] = {"n_voice": 1}

    scene.set_layers([
        {"layer": layer, "field": field, "result": result, "status": "ok"},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    assert grid.n_cells == 1


def test_scale_space_colors_distinct_per_scale():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    scene.set_scale_space(True, 1.0)          # color_by_scale defaults True
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chain_a = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.0])
    chain_b = _make_chain_with_scales(4, 0, [1.0, 2.0, 3.0])   # overlaps + one new scale (3.0)

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([chain_a, chain_b]), "status": "ok"},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    colors = np.asarray(grid.point_data["colors"])[:, :3]
    n_scales = len(set(chain_a["log2_scales"].tolist()) | set(chain_b["log2_scales"].tolist()))
    assert n_scales == 4                      # {0.0, 1.0, 2.0, 3.0}
    distinct = {tuple(c) for c in colors}
    assert len(distinct) == n_scales


def test_scale_space_color_by_scale_false_keeps_ordinary_chain_colors():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    scene.set_scale_space(True, 1.0, color_by_scale=False)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chain = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.0])

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([chain]), "status": "ok"},
    ])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    colors = np.asarray(grid.point_data["colors"])
    assert (colors[:, :3] == list(VTRAIL_COLOR)).all()


def test_camera_key_folds_scale_space_state_and_off_stays_byte_identical():
    scene = Scene(_plotter())
    assert scene.camera_key() == "geo:mercator"       # off: pre-Task-6 exact string, unchanged

    scene.set_scale_space(True, 5.0)

    assert scene.camera_key() == "geo:mercator|sspace:on"

    scene.set_scale_space(False, 5.0)

    assert scene.camera_key() == "geo:mercator"        # back to the exact original string


def test_scale_space_toggle_earns_its_own_remembered_framing():
    """A toggled-on cone deserves a fit, not the flat view's stale camera (module docstring's
    "folds in the scale-space flag" section) -- mirrors ``test_remember_camera_only_overwrites_
    the_current_views_own_slot``'s own per-key-isolation idiom above."""
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    scene.set_frame_mode(True)
    chain = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.0])
    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([chain]), "status": "ok"},
    ])
    navigated = scene._plotter.camera.parallel_scale * 7.0
    scene._plotter.camera.parallel_scale = navigated        # user navigates the flat view

    scene.set_scale_space(True, 3.0)                        # first-ever visit -> fresh fit

    assert scene._plotter.camera.parallel_scale != pytest.approx(navigated)

    scene._plotter.camera.parallel_scale *= 2.0              # user navigates the sspace view
    sspace_navigated = scene._plotter.camera.parallel_scale
    scene.set_scale_space(False, 3.0)     # back to flat -- restores the flat view's OWN memory
    scene.set_scale_space(True, 3.0)      # back to sspace -- restores ITS OWN memory, not flat's

    assert scene._plotter.camera.parallel_scale == pytest.approx(sspace_navigated)


def test_set_scale_space_noop_when_unchanged_does_not_rebuild_the_actor():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _bare_field()
    chain = _make_chain_with_scales(0, 0, [0.0, 1.0, 2.0])
    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([chain]), "status": "ok"},
    ])
    scene.set_scale_space(True, 3.0)
    actor_before = scene._plotter.actors["layer-1-chains"]

    scene.set_scale_space(True, 3.0)      # identical state -- must be a no-op, not a rebuild

    assert scene._plotter.actors["layer-1-chains"] is actor_before


def test_default_scale_space_stretch_and_max_grid_dim_from_active_geometry():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = RasterField._from_bare_array(np.zeros((10, 20)), "bare", frame=LocalFrame(units="px"),
                                          name="bare")     # ny=10, nx=20 -> max_grid_dim = 20
    chain_a = _make_chain_with_scales(0, 0, [0.0, 1.0])
    chain_b = _make_chain_with_scales(4, 0, [1.0, 2.0])    # union of scales {0,1,2} -> n_scales=3

    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([chain_a, chain_b]), "status": "ok"},
    ])

    assert scene.default_scale_space_max_grid_dim() == pytest.approx(20.0)
    assert scene.default_scale_space_stretch() == pytest.approx(20.0 / (2.0 * 3))


def test_default_scale_space_stretch_falls_back_to_one_with_no_qualifying_entry():
    scene = Scene(_plotter())
    assert scene.default_scale_space_stretch() == pytest.approx(1.0)
    assert scene.default_scale_space_max_grid_dim() == pytest.approx(1.0)


# --------------------------------------------------------------------------------- palette parity


def test_palette_constants_match_canvas_session_palette():
    """Session palette parity: guarded by its OWN importorskip so this is the only
    place in the arrangement test suite that ever needs pyqtgraph -- scene.py itself never imports
    canvas.py (see its module docstring's "redeclared here, not imported" section)."""
    pytest.importorskip("pyqtgraph", reason="only this parity check needs pyqtgraph")
    from dynamix.shell import canvas

    assert VTRAIL_COLOR == canvas.VTRAIL_COLOR
    assert SEAM_COLOR == canvas.SEAM_COLOR
    assert EXTREMA_COLOR == canvas.EXTREMA_COLOR
    assert ROI_BOUNDS_COLOR == canvas.ROI_BOUNDS_COLOR


# --------------------------------------------------------------------- footprints

from dynamix.geo.footprints import Footprint  # noqa: E402
from dynamix.shell.arrangement.scene import _FOOTPRINT_NAME  # noqa: E402


def _fp(name, west, south, east, north):
    return Footprint(path=f"/data/{name}.tif", name=name, width=10, height=10, count=1,
                     dtype="int16", crs="EPSG:4326", bounds=(west, south, east, north),
                     corners=((west, south), (east, south), (east, north), (west, north)))


def test_set_footprints_draws_one_closed_loop_per_footprint():
    p = _plotter()
    scene = Scene(p)
    scene.set_footprints([_fp("a", 119, -22, 120, -21), _fp("b", 120, -22, 121, -21)])
    actor = p.renderer.actors[_FOOTPRINT_NAME]
    ds = actor.mapper.dataset
    assert ds.n_points == 8 and ds.n_lines == 2
    scene.set_footprints([])
    assert _FOOTPRINT_NAME not in p.renderer.actors


def test_footprints_survive_a_mode_switch_and_reproject():
    p = _plotter()
    scene = Scene(p)
    scene.set_footprints([_fp("a", 119, -22, 120, -21)])
    before = np.array(p.renderer.actors[_FOOTPRINT_NAME].mapper.dataset.points)
    scene.set_mode("globe")
    after = np.array(p.renderer.actors[_FOOTPRINT_NAME].mapper.dataset.points)
    assert after.shape == before.shape and not np.allclose(after, before)


def test_frame_mode_hides_footprints_and_geo_mode_brings_them_back():
    p = _plotter()
    scene = Scene(p)
    scene.set_footprints([_fp("a", 119, -22, 120, -21)])
    scene.set_frame_mode(True)
    assert _FOOTPRINT_NAME not in p.renderer.actors
    scene.set_frame_mode(False)
    assert _FOOTPRINT_NAME in p.renderer.actors


def test_footprints_at_screen_returns_the_footprint_under_the_pixel_smallest_first():
    p = _plotter()
    scene = Scene(p)
    big = _fp("big", 118, -23, 122, -19)
    small = _fp("small", 119, -22, 120, -21)
    scene.set_footprints([big, small])
    p.view_xy(); p.reset_camera()
    viewport = (400, 400)
    from dynamix.core.projection import project
    from dynamix.core.selection import project_points
    centre = project(np.array([119.5]), np.array([-21.5]), np.array([0.0]), mode=scene.mode)
    px, visible = project_points(centre, scene._camera_mvp(viewport), viewport)
    assert visible[0]
    hits = scene.footprints_at_screen(px[0, 0], px[0, 1], viewport)
    assert [h.name for h in hits] == ["small", "big"]
    far = project(np.array([130.0]), np.array([-21.5]), np.array([0.0]), mode=scene.mode)
    px2, _ = project_points(far, scene._camera_mvp(viewport), viewport)
    assert scene.footprints_at_screen(px2[0, 0], px2[0, 1], viewport) == []


def test_footprints_of_one_granule_draw_one_loop_and_hit_as_a_group():
    """An ASTER granule's band files share a footprint (to within metres): one loop on the
    world, not thirteen on top of each other -- and a right-click on it returns every file."""
    p = _plotter()
    scene = Scene(p)
    base = "AST_07XT_00410302004021233_20250402184734_SRF"
    b01 = _fp(f"{base}_VNIR_B01", 119, -22, 120, -21)
    b04 = _fp(f"{base}_SWIR_B04", 119.0001, -22.0001, 120.0001, -21.0001)
    other = _fp("ASTGTMV003_S22E120_dem", 120, -22, 121, -21)
    scene.set_footprints([b01, b04, other])
    ds = p.renderer.actors[_FOOTPRINT_NAME].mapper.dataset
    assert ds.n_lines == 2 and ds.n_points == 8
    p.view_xy(); p.reset_camera()
    viewport = (400, 400)
    from dynamix.core.projection import project
    from dynamix.core.selection import project_points
    centre = project(np.array([119.5]), np.array([-21.5]), np.array([0.0]), mode=scene.mode)
    px, _ = project_points(centre, scene._camera_mvp(viewport), viewport)
    hits = scene.footprints_at_screen(px[0, 0], px[0, 1], viewport)
    assert [h.name for h in hits] == [b01.name, b04.name]


# ------------------------------------------------------------------- previews

from dynamix.core.frames import GeographicFrame  # noqa: E402
from dynamix.shell.arrangement.scene import _PREVIEW_PREFIX  # noqa: E402


def _preview_field(name, west, north, n=16):
    vals = np.random.default_rng(0).random((n, n)); vals[0, :] = np.nan
    ax = west + (np.arange(n) + 0.5) / n; ay = north - (np.arange(n) + 0.5) / n
    return RasterField(name=name, values=vals, frame=GeographicFrame(), x_axis=ax, y_axis=ay,
                       provenance={"crs": "EPSG:4326", "preview": True})


def _preview_actors(p):
    return sorted(n for n in p.renderer.actors if n.startswith(_PREVIEW_PREFIX))


def test_set_previews_drapes_each_field_and_an_empty_list_clears_them():
    p = _plotter()
    scene = Scene(p)
    scene.set_previews([("a", _preview_field("a", 119, -21)), ("b", _preview_field("b", 120, -21))])
    assert len(_preview_actors(p)) == 2
    ds = p.renderer.actors[_preview_actors(p)[0]].mapper.dataset
    assert ds.n_points == 256 and "value" in ds.point_data
    scene.set_previews([])
    assert _preview_actors(p) == []


def test_previews_survive_a_mode_switch_and_hide_in_frame_mode():
    p = _plotter()
    scene = Scene(p)
    scene.set_previews([("a", _preview_field("a", 119, -21))])
    scene.set_mode("globe")
    assert len(_preview_actors(p)) == 1
    scene.set_frame_mode(True)
    assert _preview_actors(p) == []
    scene.set_frame_mode(False)
    assert len(_preview_actors(p)) == 1


# ------------------------------------------------------------ H-lines in the world

def _hline_result():
    ext0 = {"x": np.array([5, 3, 4, 10, 11, 20], np.int64), "y": np.array([2, 2, 2, 7, 7, 9], np.int64),
            "mod": np.ones(6), "arg": np.zeros(6), "line_id": np.array([0, 0, 0, 1, 1, -1], np.int64)}
    return {"chains": [], "extrema": [ext0], "scales": np.array([2.0])}


def test_h_lines_are_drawn_as_polylines_in_the_world():
    p = _plotter()
    scene = Scene(p)
    field = _frame_field()
    layer = Layer(layer_id=0, name="l", source_id="s0")
    scene.set_frame_mode(True)
    scene.set_layers([{"layer": layer, "field": field, "result": _hline_result(), "status": "ok"}])
    name = "layer-0-hlines"
    assert name in p.renderer.actors
    ds = p.renderer.actors[name].mapper.dataset
    assert ds.n_lines == 2 and ds.n_points == 5            # the singleton is not a line
    dots = p.renderer.actors["layer-0-extrema"].mapper.dataset
    assert dots.n_points == 1                               # only the singleton stays a dot


def test_no_h_line_actor_when_nothing_is_labelled():
    p = _plotter()
    scene = Scene(p)
    res = _hline_result(); res["extrema"][0]["line_id"][:] = -1
    scene.set_frame_mode(True)
    scene.set_layers([{"layer": Layer(layer_id=0, name="l", source_id="s0"), "field": _frame_field(),
                       "result": res, "status": "ok"}])
    assert "layer-0-hlines" not in p.renderer.actors


# ------------------------------------------------------------------ hillshade drape

def _sloped_field(n=24):
    y, x = np.mgrid[0:n, 0:n]
    return RasterField(name="slope", values=(x + y).astype(float), frame=LocalFrame(units="m", dx=10.0, dy=10.0),
                       x_axis=np.arange(n) * 10.0, y_axis=np.arange(n) * 10.0)


def test_hillshade_entry_drapes_rgb_colours_instead_of_a_lut():
    p = _plotter()
    scene = Scene(p)
    scene.set_frame_mode(True)
    layer = Layer(layer_id=0, name="slope", source_id="s0")
    scene.set_layers([{"layer": layer, "field": _sloped_field(), "result": None, "status": "ok",
                       "colormap": "viridis", "hillshade": (True, 315.0, 45.0, 1.0)}])
    ds = p.renderer.actors["layer-0-raster"].mapper.dataset
    assert "rgba" in ds.point_data and ds.point_data["rgba"].shape == (24 * 24, 4)
    lit = np.array(ds.point_data["rgba"])
    scene.set_layers([{"layer": layer, "field": _sloped_field(), "result": None, "status": "ok",
                       "colormap": "viridis", "hillshade": (True, 135.0, 45.0, 1.0)}])
    ds2 = p.renderer.actors["layer-0-raster"].mapper.dataset
    assert not np.array_equal(np.array(ds2.point_data["rgba"]), lit)      # the sun moved
    scene.set_colormap("magma")                                            # must not raise on an RGBA actor
    scene.set_layers([{"layer": layer, "field": _sloped_field(), "result": None, "status": "ok",
                       "colormap": "viridis", "hillshade": (False, 315.0, 45.0, 1.0)}])
    assert "value" in p.renderer.actors["layer-0-raster"].mapper.dataset.point_data


# -------------------------------------------------------------- 3-D surface

def _surface_entry(field, on, negate=False, result=None):
    return {"layer": Layer(layer_id=0, name="s", source_id="s0"), "field": field, "result": result,
            "status": "ok", "colormap": "viridis", "surface": (on, negate)}


def test_surface_option_lifts_the_drape_by_the_value_in_frame_mode():
    p = _plotter()
    scene = Scene(p)
    scene.set_frame_mode(True)
    scene.set_vertical_exaggeration(2.0)
    f = _sloped_field()                                   # values 0..46
    scene.set_layers([_surface_entry(f, on=False)])
    flat = np.array(p.renderer.actors["layer-0-raster"].mapper.dataset.points)
    assert np.allclose(flat[:, 2], 0.0)
    scene.set_layers([_surface_entry(f, on=True)])
    z = np.array(p.renderer.actors["layer-0-raster"].mapper.dataset.points)[:, 2]
    assert z.min() == pytest.approx(0.0) and z.max() == pytest.approx(46.0 * 2.0)
    scene.set_layers([_surface_entry(f, on=True, negate=True)])
    z = np.array(p.renderer.actors["layer-0-raster"].mapper.dataset.points)[:, 2]
    assert z.min() == pytest.approx(-46.0 * 2.0) and z.max() == pytest.approx(0.0)


def test_surface_option_moves_extrema_and_h_lines_onto_the_surface():
    p = _plotter()
    scene = Scene(p)
    scene.set_frame_mode(True)
    f = _sloped_field()
    res = _hline_result()                                 # extrema at x 3..5,y 2 / x 10,11,y 7 / x 20,y 9
    scene.set_layers([_surface_entry(f, on=True, result=res)])
    lift = scene._frame_vector_lift(f)
    dots = np.array(p.renderer.actors["layer-0-extrema"].mapper.dataset.points)   # the singleton (20, 9)
    assert dots.shape[0] == 1 and dots[0, 2] == pytest.approx(f.values[9, 20] + lift)
    hl = np.array(p.renderer.actors["layer-0-hlines"].mapper.dataset.points)
    assert hl[:, 2].min() > lift and not np.allclose(hl[:, 2], hl[0, 2])


# ------------------------------------------------------------ reference layers

from dynamix.shell.arrangement.scene import _REFERENCE_PREFIX  # noqa: E402


def _ref_entry(ref_id="ref0", kind="polygon", visible=True, native=None):
    ring = np.array([[119.0, -22.0], [119.5, -22.0], [119.5, -21.5], [119.0, -22.0]])
    return {"ref_id": ref_id, "name": "slumps", "kind": kind, "color": "#ff8800", "visible": visible,
            "lonlat": [[ring]], "native": native}


def test_reference_layers_draw_in_geo_modes_and_reproject_on_mode_switch():
    p = _plotter()
    scene = Scene(p)
    scene.set_reference_layers([_ref_entry(), _ref_entry("ref1", visible=False)])
    a0 = p.renderer.actors[f"{_REFERENCE_PREFIX}ref0"]
    assert a0.mapper.dataset.n_lines == 1 and a0.mapper.dataset.n_points == 4
    assert not p.renderer.actors[f"{_REFERENCE_PREFIX}ref1"].GetVisibility()
    before = np.array(a0.mapper.dataset.points)
    scene.set_mode("globe")
    after = np.array(p.renderer.actors[f"{_REFERENCE_PREFIX}ref0"].mapper.dataset.points)
    assert not np.allclose(before, after)
    scene.set_reference_visible("ref1", True)
    assert p.renderer.actors[f"{_REFERENCE_PREFIX}ref1"].GetVisibility()
    scene.set_reference_layers([])
    assert not any(n.startswith(_REFERENCE_PREFIX) for n in p.renderer.actors)


def test_reference_layers_use_native_coordinates_in_frame_mode_and_hide_without_them():
    p = _plotter()
    scene = Scene(p)
    scene.set_frame_mode(True)
    native_ring = [[np.array([[100.0, 200.0], [300.0, 200.0], [300.0, 400.0], [100.0, 200.0]])]]
    scene.set_reference_layers([_ref_entry("ref0", native=native_ring), _ref_entry("ref1")])
    pts = np.array(p.renderer.actors[f"{_REFERENCE_PREFIX}ref0"].mapper.dataset.points)
    assert pts[:, 0].min() == 100.0 and pts[:, 1].max() == 400.0
    assert f"{_REFERENCE_PREFIX}ref1" not in p.renderer.actors      # no native frame coords: not drawn


def test_zoom_to_reference_frames_the_layer_not_the_raster():
    p = _plotter()
    scene = Scene(p)
    scene.set_reference_layers([_ref_entry()])
    p.camera.focal_point = (0.0, 0.0, 0.0)
    scene.zoom_to_reference("ref0")
    b = p.renderer.actors[f"{_REFERENCE_PREFIX}ref0"].GetBounds()
    centre = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
    assert np.allclose(p.camera.focal_point, centre, atol=1e-3)     # framed on the layer's bounds


# ------------------------------------------------ globe framing follows the DATA
# A geo dataset swap refits the globe camera to the new data. A camera remembered per MODE keeps
# the first dataset's framing (ASTER, Pilbara) when BOEM (Gulf of Mexico) replaces it, and the
# Gulf sits on the far side of the Earth. The first fit frames the data: a fit to EVERY actor
# lets a world-spanning reference layer (BOEM's website/states) shrink a 12 km raster to nothing.

def _lonlat_field(tmp_path, name, lon0, lat0, deg=0.1, n=16):
    import rasterio
    from rasterio.transform import from_origin
    path = tmp_path / name
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(lon0, lat0 + deg, deg / n, deg / n)) as dst:
        dst.write(np.random.default_rng(0).random((n, n)).astype(np.float32), 1)
    return _load_field(path)


def _raster_centre(scene, layer_id):
    b = scene._plotter.renderer.actors[f"layer-{layer_id}-raster"].GetBounds()
    return np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])


def _looks_at_from_outside(scene, centre, within_km):
    cam = scene._plotter.camera
    focal, pos = np.array(cam.focal_point), np.array(cam.position)
    return (np.linalg.norm(focal - centre) < within_km            # framed on the data
            and np.dot(pos - focal, centre) > 0                    # from outside the globe
            and np.linalg.norm(pos - focal) < 3000.0)              # close, not a whole-Earth fit


def test_first_globe_fit_frames_the_data_not_a_world_spanning_reference_layer(tmp_path):
    scene = Scene(_plotter())
    scene.set_mode("globe")
    usa = np.array([[-125.0, 25.0], [-65.0, 25.0], [-65.0, 50.0], [-125.0, 50.0], [-125.0, 25.0]])
    scene.set_reference_layers([{"ref_id": "states", "name": "states", "kind": "polygon", "color": "#ff8800",
                                 "visible": True, "lonlat": [[usa]], "native": None}])
    gulf = _lonlat_field(tmp_path, "gulf.tif", -94.0, 27.0)
    scene.set_layers([{"layer": Layer(layer_id=1, name="BOEM", source_id="mem:b"), "field": gulf,
                       "result": None, "status": "ok"}])
    assert _looks_at_from_outside(scene, _raster_centre(scene, 1), within_km=50.0)


def test_geo_dataset_swap_refits_to_the_new_data_but_adding_a_layer_keeps_the_camera(tmp_path):
    scene = Scene(_plotter())
    scene.set_mode("globe")
    aster = _lonlat_field(tmp_path, "aster.tif", 119.0, -22.0)
    boem = _lonlat_field(tmp_path, "boem.tif", -94.0, 27.0)
    a = {"layer": Layer(layer_id=1, name="ASTER", source_id="mem:a"), "field": aster, "result": None, "status": "ok"}
    b = {"layer": Layer(layer_id=2, name="BOEM", source_id="mem:b"), "field": boem, "result": None, "status": "ok"}
    scene.set_layers([a])
    assert _looks_at_from_outside(scene, _raster_centre(scene, 1), within_km=50.0)
    scene.set_layers([b])                                          # the swap: Pilbara -> Gulf
    assert _looks_at_from_outside(scene, _raster_centre(scene, 2), within_km=50.0)
    before = tuple(scene._plotter.camera.position)
    scene.set_layers([b, a])                                       # adding a layer: navigation is sacred
    assert tuple(scene._plotter.camera.position) == before


# ------------------------------------------ reference layers ride a 3-D surface
# A raster shown as a surface rises above the fixed vector lift, so a reference layer at that lift
# renders UNDER it. Reference layers ride the surface as chains/H-lines do (_surface_ride), from
# their pixel-frame vertices.

def test_reference_layer_rides_a_3d_surface_instead_of_sitting_under_it():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    field = RasterField._from_bare_array(np.full((8, 8), 100.0), "s", frame=LocalFrame(units="px"), name="s")
    scene.set_layers([{"layer": Layer(layer_id=1, name="S", source_id="mem:s"), "field": field,
                       "result": None, "status": "ok", "surface": (True, False)}])
    ring = np.array([[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 2.0]])
    scene.set_reference_layers([{"ref_id": "r", "name": "r", "kind": "polygon", "color": "#ff8800",
                                 "visible": True, "lonlat": [], "native": [[ring]], "pixels": [[ring]]}])
    z = scene._plotter.renderer.actors[f"{_REFERENCE_PREFIX}r"].GetMapper().GetInput().points[:, 2]
    assert z.min() >= 100.0                                        # on the surface, plus the lift


def test_reference_layer_without_pixels_or_on_a_flat_raster_keeps_the_plain_lift():
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    field = RasterField._from_bare_array(np.full((8, 8), 100.0), "s", frame=LocalFrame(units="px"), name="s")
    scene.set_layers([{"layer": Layer(layer_id=1, name="S", source_id="mem:s"), "field": field,
                       "result": None, "status": "ok"}])
    ring = np.array([[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 2.0]])
    scene.set_reference_layers([{"ref_id": "r", "name": "r", "kind": "polygon", "color": "#ff8800",
                                 "visible": True, "lonlat": [], "native": [[ring]], "pixels": [[ring]]}])
    z = scene._plotter.renderer.actors[f"{_REFERENCE_PREFIX}r"].GetMapper().GetInput().points[:, 2]
    assert 0.0 < z.max() < 1.0                                     # flat raster: just the epsilon


def test_entry_show_raster_false_hides_the_drape_and_survives_a_mode_switch(tmp_path):
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "geo.tif")
    entry = {"layer": Layer(layer_id=1, name="G", source_id="mem:g"), "field": field, "result": None,
             "status": "ok", "show_raster": False}
    scene.set_layers([entry])
    assert not scene._plotter.renderer.actors["layer-1-raster"].GetVisibility()
    scene.set_mode("globe")
    assert not scene._plotter.renderer.actors["layer-1-raster"].GetVisibility()
    scene.set_layers([{**entry, "show_raster": True}])
    assert scene._plotter.renderer.actors["layer-1-raster"].GetVisibility()


# ------------------------------ filter-only resync, in place

def _ext_result(n_keep=None, sig="sigA"):
    x = np.arange(10, dtype=np.int64); y = np.zeros(10, dtype=np.int64)
    base = {"x": x, "y": y, "mod": np.linspace(1, 0.1, 10), "arg": np.zeros(10),
            "line_id": np.zeros(10, dtype=np.int64)}
    ext = base if n_keep is None else {k: (v[:n_keep] if hasattr(v, "shape") else v) for k, v in base.items()}
    return {"extrema": [ext], "_shape": (4, 16), "chains": [], "scales": np.array([2.0])}


def test_a_filter_only_resync_keeps_raster_and_camera_and_updates_vectors(tmp_path):
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    field = _frame_field()
    layer = Layer(layer_id=1, name="F", source_id="mem:f")
    e1 = {"layer": layer, "field": field, "result": _ext_result(), "status": "ok", "signature": "s1"}
    scene.set_layers([e1])
    p = scene._plotter
    raster_before = p.renderer.actors["layer-1-raster"]
    cam_before = tuple(p.camera.position)
    hline_name = [n for n in p.renderer.actors if "hline" in n][0]
    n_before = p.renderer.actors[hline_name].GetMapper().GetInput().n_points
    e2 = {"layer": layer, "field": field, "result": _ext_result(n_keep=6), "status": "ok", "signature": "s1"}
    scene.set_layers([e2])
    assert p.renderer.actors["layer-1-raster"] is raster_before      # drape untouched
    assert tuple(p.camera.position) == cam_before                    # camera untouched
    assert p.renderer.actors[hline_name].GetMapper().GetInput().n_points < n_before


def test_a_signature_or_style_change_still_takes_the_full_rebuild(tmp_path):
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    field = _frame_field()
    layer = Layer(layer_id=1, name="F", source_id="mem:f")
    scene.set_layers([{"layer": layer, "field": field, "result": _ext_result(), "status": "ok", "signature": "s1"}])
    raster_before = scene._plotter.renderer.actors["layer-1-raster"]
    scene.set_layers([{"layer": layer, "field": field, "result": _ext_result(n_keep=6), "status": "ok",
                       "signature": "s2"}])                          # the transform re-ran
    assert scene._plotter.renderer.actors["layer-1-raster"] is not raster_before


def test_a_filter_resync_reuses_the_base_hline_ordering_instead_of_rewalking(monkeypatch):
    import dynamix.shell.arrangement.scene as scene_mod
    calls = []
    real = scene_mod.hline_runs
    monkeypatch.setattr(scene_mod, "hline_runs", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    field = _frame_field()
    layer = Layer(layer_id=1, name="F", source_id="mem:f")
    x = np.arange(10, dtype=np.int64); y = np.zeros(10, dtype=np.int64)
    base = {"x": x, "y": y, "mod": np.linspace(1, 0.1, 10), "arg": np.zeros(10),
            "line_id": np.zeros(10, dtype=np.int64)}
    def result(keep=None):
        ext = base if keep is None else {k: (v[keep] if hasattr(v, "shape") else v) for k, v in base.items()}
        return {"extrema": [ext], "_shape": (4, 16), "_ext_base": base, "chains": [],
                "scales": np.array([2.0])}
    scene.set_layers([{"layer": layer, "field": field, "result": result(), "status": "ok", "signature": "s1"}])
    n_walks = len(calls)
    hline_name = [n for n in scene._plotter.renderer.actors if "hline" in n][0]
    cells_before = scene._plotter.renderer.actors[hline_name].GetMapper().GetInput().n_cells
    keep = np.ones(10, bool); keep[4] = False                     # cut the line in the middle
    scene.set_layers([{"layer": layer, "field": field, "result": result(keep), "status": "ok", "signature": "s1"}])
    assert len(calls) == n_walks                                  # NO re-walk on the tweak
    after = scene._plotter.renderer.actors[hline_name].GetMapper().GetInput()
    assert after.n_cells < cells_before or after.n_points < 10    # the cut is visible in the actor


def test_stamped_runs_spare_the_scene_the_landing_walk(monkeypatch):
    import dynamix.shell.arrangement.scene as scene_mod
    from dynamix.core.hlines import hline_runs as real_runs
    calls = []
    monkeypatch.setattr(scene_mod, "hline_runs", lambda *a, **k: (calls.append(1), real_runs(*a, **k))[1])
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    field = _frame_field()
    x = np.arange(10, dtype=np.int64); y = np.zeros(10, dtype=np.int64)
    base = {"x": x, "y": y, "mod": np.linspace(1, 0.1, 10), "arg": np.zeros(10),
            "line_id": np.zeros(10, dtype=np.int64)}
    runs = real_runs(base, (4, 16))
    res = {"extrema": [base], "_shape": (4, 16), "_ext_base": base, "_ext_base_runs": runs,
           "chains": [], "scales": np.array([2.0])}
    scene.set_layers([{"layer": Layer(layer_id=1, name="F", source_id="mem:f"), "field": field,
                       "result": res, "status": "ok", "signature": "s1"}])
    assert calls == []                                            # the walk stayed on the worker
    assert any("hline" in n for n in scene._plotter.renderer.actors)
