# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for Scene.set_mask/set_colormap's identity discipline and MaskRow's wiring.

Offscreen VTK: ``pytest.importorskip("pyvista")`` skips this whole module honestly if pyvista
isn't installed. Fixture reuse: the chain/extrema/result builders and the geo field helper live in
``tests/test_arrangement_scene.py`` (which itself reuses ``tests/test_geo_mapping.py``'s own
GeoTIFF fixture) -- imported directly here, the established cross-file fixture-reuse pattern this
suite already uses throughout.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista not installed")
pytest.importorskip("rasterio", reason="rasterio not installed (needed by the geo fixture)")
pv.OFF_SCREEN = True

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.layer import Layer
from dynamix.shell.arrangement.mask_row import MaskRow
from dynamix.shell.arrangement.scene import Scene

from tests.test_arrangement_scene import _geo_field, _make_chain, _make_extrema0, _plotter, _result
from tests.test_geo_mapping import _BOEM_LIKE_WKT, _NORTH_FT, _PX_FT, _WEST_FT

# --------------------------------------------------------------------------------- Scene.set_mask


def test_set_mask_preserves_actor_and_dataset_identity(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    chains = [_make_chain(2 + 2 * i, 2 + 2 * i, n=4, slope=-0.2 * i) for i in range(6)]
    scene.set_layers([{"layer": layer, "field": field, "result": _result(chains), "status": "ok"}])

    actor_before = scene._plotter.actors["layer-1-chains"]
    dataset_before = actor_before.mapper.dataset

    scene.set_mask(modulus_pctl=50.0, scale_lo=0, scale_hi=0)

    actor_after = scene._plotter.actors["layer-1-chains"]
    assert actor_after is actor_before
    assert actor_after.mapper.dataset is dataset_before


def test_set_mask_modulus_percentile_hides_the_low_scoring_chains(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    # 4 chains, well-separated max(log2_mod) = slope * 4 (log2_scales = [0,1,2,3,4]).
    chains = [_make_chain(2 + 6 * i, 2 + 6 * i, n=5, slope=s)
              for i, s in enumerate((0.1, 0.5, 1.0, 2.0))]
    scene.set_layers([{"layer": layer, "field": field, "result": _result(chains), "status": "ok"}])

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    alpha_before = np.asarray(grid.point_data["colors"])[:, 3]
    assert (alpha_before == 255).all()
    visible_before = (alpha_before == 255).sum()

    scene.set_mask(modulus_pctl=75.0, scale_lo=0, scale_hi=0)

    alpha_after = np.asarray(
        scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])[:, 3]
    visible_after = (alpha_after == 255).sum()
    assert visible_after < visible_before
    assert visible_after > 0                  # the top-scoring chain(s) still show

    # Not path-dependent: recomputed fresh from each layer's own base_rgba every call, never
    # accumulated -- setting pctl back to 0 must restore full visibility, not compound the filter.
    scene.set_mask(modulus_pctl=0.0, scale_lo=0, scale_hi=0)
    alpha_restored = np.asarray(
        scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])[:, 3]
    assert (alpha_restored == 255).all()


def test_set_mask_modulus_pctl_zero_keeps_everything(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    chains = [_make_chain(2 + 6 * i, 2 + 6 * i, n=5, slope=s)
              for i, s in enumerate((0.1, 0.5, 1.0, 2.0))]
    scene.set_layers([{"layer": layer, "field": field, "result": _result(chains), "status": "ok"}])

    scene.set_mask(modulus_pctl=0.0, scale_lo=0, scale_hi=0)

    alpha = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])[:, 3]
    assert (alpha == 255).all()


def test_set_mask_scale_lo_hides_chains_too_shallow_to_reach_it(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    shallow = _make_chain(2, 2, n=2, slope=0.0)        # deepest scale index (hi_scale_idx) = 1
    deep = _make_chain(20, 20, n=6, slope=0.0)         # deepest scale index (hi_scale_idx) = 5
    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([shallow, deep]), "status": "ok"},
    ])

    scene.set_mask(modulus_pctl=0.0, scale_lo=3, scale_hi=0)      # scale_hi=0: no cap

    grid = scene._plotter.actors["layer-1-chains"].mapper.dataset
    colors = np.asarray(grid.point_data["colors"])
    _starts, chain_indices = scene._chain_lookup[1]
    alpha = colors[:, 3]
    assert (alpha[chain_indices == 0] == 0).all()      # shallow chain (1 < 3): hidden
    assert (alpha[chain_indices == 1] == 255).all()    # deep chain (5 >= 3, no cap): still visible


def test_set_mask_scale_range_discriminates_on_both_bounds(tmp_path):
    """Coordinator-adjudicated semantics: a chain is visible iff
    ``scale_lo <= hi_scale_idx <= scale_hi`` (``scale_hi == 0`` -> no cap) -- a chain-DEPTH range,
    not the earlier interval-INTERSECTION test under which every chain's span always starting at
    0 made ``scale_hi`` a dead knob across its whole range. This sweeps BOTH bounds against the
    same shallow/deep pair and asserts each one, independently, changes what's visible."""
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    shallow = _make_chain(2, 2, n=2, slope=0.0)        # hi_scale_idx = 1
    deep = _make_chain(20, 20, n=6, slope=0.0)         # hi_scale_idx = 5
    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([shallow, deep]), "status": "ok"},
    ])
    _starts, chain_indices = scene._chain_lookup[1]

    def _alpha():
        colors = np.asarray(
            scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])
        return colors[:, 3]

    # scale_hi caps the DEEP chain out; scale_lo=0 leaves the shallow one unrestricted from below.
    scene.set_mask(modulus_pctl=0.0, scale_lo=0, scale_hi=3)
    alpha = _alpha()
    assert (alpha[chain_indices == 0] == 255).all()    # shallow (1 <= 3): visible
    assert (alpha[chain_indices == 1] == 0).all()       # deep (5 > 3): hidden -- scale_hi bites

    # scale_lo excludes the SHALLOW chain; scale_hi=0 (no cap) leaves the deep one unrestricted
    # from above.
    scene.set_mask(modulus_pctl=0.0, scale_lo=4, scale_hi=0)
    alpha = _alpha()
    assert (alpha[chain_indices == 0] == 0).all()       # shallow (1 < 4): hidden -- scale_lo bites
    assert (alpha[chain_indices == 1] == 255).all()    # deep (5 >= 4, no cap): visible


def test_set_mask_does_not_touch_extrema_or_roi_actors(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    chains = [_make_chain(2, 2, n=4, slope=0.0)]
    ext0 = _make_extrema0([1, 2], [1, 2])
    result = _result(chains, extrema0=ext0, roi=(0, 0, 5, 5))
    scene.set_layers([{"layer": layer, "field": field, "result": result, "status": "ok"}])

    ex_actor = scene._plotter.actors["layer-1-extrema"]
    roi_actor = scene._plotter.actors["layer-1-roi"]

    scene.set_mask(modulus_pctl=100.0, scale_lo=0, scale_hi=0)

    assert scene._plotter.actors["layer-1-extrema"] is ex_actor
    assert scene._plotter.actors["layer-1-roi"] is roi_actor


def test_set_mode_reapplies_the_active_mask_after_rebuild(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    shallow = _make_chain(2, 2, n=2, slope=0.0)
    deep = _make_chain(20, 20, n=6, slope=0.0)
    scene.set_layers([
        {"layer": layer, "field": field, "result": _result([shallow, deep]), "status": "ok"},
    ])
    scene.set_mask(modulus_pctl=0.0, scale_lo=3, scale_hi=0)

    scene.set_mode("globe")            # the one legitimate full rebuild

    _starts, chain_indices = scene._chain_lookup[1]
    colors = np.asarray(
        scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])
    alpha = colors[:, 3]
    assert (alpha[chain_indices == 0] == 0).all()      # mask still in effect post-rebuild
    assert (alpha[chain_indices == 1] == 255).all()


# ------------------------------------------------------------------------------------- MaskRow


def test_mask_row_default_values(qtbot):
    row = MaskRow()
    qtbot.addWidget(row)
    assert row.values() == {"modulus_pctl": 0.0, "scale_lo": 0, "scale_hi": 0}


def test_mask_row_control_change_emits_maskChanged_with_all_three_keys(qtbot):
    row = MaskRow()
    qtbot.addWidget(row)
    with qtbot.waitSignal(row.maskChanged, timeout=1000) as blocker:
        row._controls["modulus_pctl"].valueChanged.emit(42.0)

    payload = blocker.args[0]
    assert set(payload) == {"modulus_pctl", "scale_lo", "scale_hi"}
    assert payload["modulus_pctl"] == 42.0
    assert payload["scale_lo"] == 0 and payload["scale_hi"] == 0     # untouched controls
    assert row.values()["modulus_pctl"] == 42.0                       # confirmed, not just emitted


def test_mask_row_each_control_updates_only_its_own_value(qtbot):
    row = MaskRow()
    qtbot.addWidget(row)
    row._controls["scale_lo"].valueChanged.emit(7)
    row._controls["scale_hi"].valueChanged.emit(20)
    assert row.values() == {"modulus_pctl": 0.0, "scale_lo": 7, "scale_hi": 20}


# ------------------------------------------------------------------------ ArrangementView wiring
#
# MaskRow itself lives in MainWindow's right panel, not this view (tests/test_right_panel.py covers the row's new home and the
# window-level wiring) -- the view keeps only the forwarding half, a public `set_mask` passthrough
# (`_on_mask_changed`'s former body). These two tests replace
# `test_arrangement_view_wires_mask_row_to_scene_lazily`, which asserted against `view._mask_row`,
# an attribute this view no longer has at all.


def test_arrangement_view_no_longer_builds_a_mask_row(qtbot):
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    assert not hasattr(view, "_mask_row")


def test_arrangement_view_set_mask_forwards_to_scene_lazily(qtbot):
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    # No scene yet (a real QtInteractor cannot be built under this harness's mandated offscreen
    # QPA platform -- see tests/test_arrangement_flip.py's documented segfault guard, so
    # activate() is never exercised here): calling set_mask must not raise -- the same
    # no-op-before-a-scene-exists contract `_on_mask_changed` always had.
    view.set_mask({"modulus_pctl": 33.0, "scale_lo": 0, "scale_hi": 0})

    # Attach a Scene directly against a real offscreen pv.Plotter, the same substitution
    # tests/test_arrangement_scene.py uses throughout, standing in for the QtInteractor Scene
    # would otherwise host.
    view._scene = Scene(_plotter())

    view.set_mask({"modulus_pctl": 33.0, "scale_lo": 7, "scale_hi": 0})

    assert view._scene._mask == (33.0, 7, 0)


# ------------------------------------------------------------------------------------- benchmark


def _big_field(ny, nx, name):
    """A RasterField built directly (no GeoTIFF I/O) so a multi-million-point fixture is cheap to
    construct: field_lonlat_grid/points_lonlat only ever read x_axis/y_axis/provenance/values/
    ny/nx off a field, never its frame or a real file, so this is a legitimate, faster substitute
    for tests/test_geo_mapping.py's own file-writing fixture."""
    x_axis = _WEST_FT + (np.arange(nx) + 0.5) * _PX_FT
    y_axis = _NORTH_FT - (np.arange(ny) + 0.5) * _PX_FT
    values = np.zeros((ny, nx), dtype=np.float64)
    return RasterField(name=name, values=values, frame=LocalFrame(units="px"),
                        x_axis=x_axis, y_axis=y_axis, provenance={"crs": _BOEM_LIKE_WKT})


def _synthetic_chains(n_chains, n_pts, ny, nx, seed):
    rng = np.random.default_rng(seed)
    chains = []
    for _ in range(n_chains):
        x0 = int(rng.integers(0, nx - n_pts - 1))
        y0 = int(rng.integers(0, ny - n_pts - 1))
        idx = np.arange(n_pts, dtype=np.int64)
        x = (x0 + idx).astype(np.int64)
        y = (y0 + idx).astype(np.int64)
        log2_scales = idx.astype(np.float64)
        slope = float(rng.uniform(-1.5, 0.5))
        log2_mod = slope * log2_scales
        chains.append({"x": x, "y": y, "mod": 2.0 ** log2_mod, "log2_mod": log2_mod,
                        "log2_scales": log2_scales})
    return chains


def test_set_mask_benchmark_two_fields_at_the_stride_cap_plus_50k_chain_vertices():
    """The design's benchmark scenario (2 draped fields <= 2M points each + 50k chain vertices),
    reused as the mask benchmark's fixture: 50 `set_mask` calls, mean asserted < 50ms (a generous
    CI bound -- offscreen software rendering on shared CI hardware; the real <16ms bar is
    validated on real hardware and reported in the commit message). The
    number is PRINTED either way, not just asserted, so it lands somewhere a human reads it.

    Includes the render fix: set_mask now ends with an explicit plotter.render() call (it
    did not before -- the mutated "colors" array never reached the screen without one, confirmed
    by an offscreen screenshot probe). This number is therefore the honest one against the spec's
    <16ms real-hardware bar, not the pre-fix figure that never actually painted anything.
    """
    ny = nx = 1414                    # 1_999_396 points: just under field_lonlat_grid's 2M cap
                                       # (default max_points), so stride == 1 -- "at the cap".
    field_a = _big_field(ny, nx, "a")
    field_b = _big_field(ny, nx, "b")
    chains_a = _synthetic_chains(500, 50, ny, nx, seed=1)     # 25,000 vertices
    chains_b = _synthetic_chains(500, 50, ny, nx, seed=2)     # 25,000 vertices -- 50,000 total

    layer_a = Layer(layer_id=1, name="A", source_id="mem:a")
    layer_b = Layer(layer_id=2, name="B", source_id="mem:b")
    scene = Scene(_plotter())
    scene.set_layers([
        {"layer": layer_a, "field": field_a, "result": _result(chains_a), "status": "ok"},
        {"layer": layer_b, "field": field_b, "result": _result(chains_b), "status": "ok"},
    ])
    assert scene.actor_count() == 4    # 2 raster + 2 chains actors

    n_calls = 50
    times = np.empty(n_calls, dtype=np.float64)
    for i in range(n_calls):
        t0 = time.perf_counter()
        scene.set_mask(modulus_pctl=float(i % 100), scale_lo=0, scale_hi=0)
        times[i] = time.perf_counter() - t0

    mean_ms = float(times.mean() * 1000.0)
    max_ms = float(times.max() * 1000.0)
    print(f"\nScene.set_mask: mean {mean_ms:.3f} ms, max {max_ms:.3f} ms over {n_calls} calls "
          f"(2 layers, 50,000 chain vertices, {ny}x{nx} draped fields)")
    assert mean_ms < 50.0, f"set_mask mean {mean_ms:.3f} ms exceeds the generous 50 ms CI bound"
