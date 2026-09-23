# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for Scene.pick/set_selection/set_group_preview and GroupPalette.

Offscreen VTK: ``pytest.importorskip("pyvista")`` skips this whole module honestly if pyvista
isn't installed. Fixture reuse: the chain/extrema/result/field builders live in
``tests/test_arrangement_scene.py``; the big-field/synthetic-chain perf fixtures live in
``tests/test_arrangement_mask.py`` -- both imported directly here, the established cross-file
fixture-reuse pattern this suite already uses throughout.

Additions: the real click -> pick VTK event leg
(``test_on_click_...``, proving the ``viewport=True`` fix -- see ``view.py``'s corrected module
docstring for why the earlier "exercised live only" claim was wrong); a real-camera ``_camera_mvp``
conversion test; a global-nearest-across-two-layers test; ``Scene.set_layers``'s staleness-pruning
contract and ``GroupPalette.prune_layer``.
"""
from __future__ import annotations

import time
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

pv = pytest.importorskip("pyvista", reason="pyvista not installed")
pytest.importorskip("rasterio", reason="rasterio not installed (needed by the geo fixture)")
pv.OFF_SCREEN = True

from dynamix.core.selection import project_points
from dynamix.model.layer import Layer
from dynamix.shell.arrangement.group_palette import GROUP_COLORS, GroupPalette
from dynamix.shell.arrangement.scene import SELECTION_COLOR, Scene

from tests.test_arrangement_mask import _big_field, _synthetic_chains
from tests.test_arrangement_scene import _geo_field, _make_chain, _plotter, _result


def _one_chain_entry(layer_id, name, field, chain, signature=None):
    """``signature`` is the entry's own transform-signature
    field, threaded by ``MainWindow._sync_arrangement`` in the real app -- ``None`` here by
    default (the shape callers not concerned with the fingerprint-based prune predicate keep
    using), settable by name where a test specifically means "the transform genuinely re-ran"."""
    layer = Layer(layer_id=layer_id, name=name, source_id=f"mem:{name}")
    return {"layer": layer, "field": field, "result": _result([chain]), "status": "ok",
            "signature": signature}

# ------------------------------------------------------------------------------- Scene.pick


def _two_chain_layer(tmp_path, slope0=-0.2, slope1=-0.2):
    """A 1-layer scene with two well-separated chains; returns ``(scene, pts3d, chain_indices,
    target)`` where ``target`` is chain 1's first vertex's world xyz -- the point every pick math
    test aims a synthetic camera at."""
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    chain0 = _make_chain(2, 2, n=4, slope=slope0)
    chain1 = _make_chain(30, 30, n=4, slope=slope1)
    scene.set_layers([{"layer": layer, "field": field, "result": _result([chain0, chain1]),
                        "status": "ok"}])
    dataset = scene._plotter.actors["layer-1-chains"].mapper.dataset
    pts3d = np.asarray(dataset.points)
    _starts, chain_indices = scene._chain_lookup[1]
    target_pt_idx = int(np.nonzero(chain_indices == 1)[0][0])
    return scene, pts3d, chain_indices, pts3d[target_pt_idx]


def _centering_mvp(target, scale=1.0e5):
    """A synthetic orthographic-like MVP (w_clip == 1 always) that maps ``target``'s own world
    (x, y) to NDC (0, 0) exactly -- i.e. dead center of any viewport -- and scales every other
    point's offset from it by ``scale``. Chosen large enough that DynamiX's geo pixel spacing
    (a fraction of a degree in Mercator's degree-equivalent y) puts every other chain's vertices
    far outside the [-1, 1] frustum, so only the target itself is ever a visible pick candidate --
    confirmed numerically before writing this fixture (see the task report)."""
    tx, ty = float(target[0]), float(target[1])
    return np.array([
        [scale, 0.0, 0.0, -scale * tx],
        [0.0, scale, 0.0, -scale * ty],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def test_pick_hits_the_right_layer_and_chain_within_8px(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    mvp = _centering_mvp(target)
    viewport = (800, 600)

    hit = scene.pick(400.0, 300.0, viewport, mvp=mvp)

    assert hit == (1, 1)     # layer_id=1, chain_index=1 (chain1, the target's own chain)


def test_pick_and_pick_in_region_resolve_the_correct_chain_from_lifted_scale_space_geometry(tmp_path):
    """Backs up
    ``scene.py``'s own module-docstring claim ("lifted points mean picking works on the lifted
    geometry automatically -- no changes needed") with a real test rather than inspection alone.

    Same construction as :func:`_two_chain_layer` above, but with scale-space enabled and a
    nonzero stretch BEFORE ``set_layers`` -- ``pts3d`` is read straight off the chains actor's
    OWN (already-lifted, non-flat) points, exactly as :meth:`Scene._pick_candidates` does, so a
    correct hit here proves the picking math operates on the lifted geometry itself, not some
    stale flat copy from before the lift."""
    scene = Scene(_plotter())
    scene.set_scale_space(True, 5.0)
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    chain0 = _make_chain(2, 2, n=4, slope=-0.2)
    chain1 = _make_chain(30, 30, n=4, slope=-0.2)
    scene.set_layers([{"layer": layer, "field": field, "result": _result([chain0, chain1]),
                        "status": "ok"}])
    dataset = scene._plotter.actors["layer-1-chains"].mapper.dataset
    pts3d = np.asarray(dataset.points)
    assert not (pts3d[:, 2] == 0.0).all()      # confirms the geometry really is lifted, not flat
    _starts, chain_indices = scene._chain_lookup[1]
    target_pt_idx = int(np.nonzero(chain_indices == 1)[0][0])
    target = pts3d[target_pt_idx]
    mvp = _centering_mvp(target)
    viewport = (800, 600)

    hit = scene.pick(400.0, 300.0, viewport, mvp=mvp)

    assert hit == (1, 1)

    region_hits = scene.pick_in_region(("box", 350.0, 250.0, 450.0, 350.0), viewport, mvp=mvp)

    assert (1, 1) in region_hits


def test_pick_a_few_pixels_off_target_still_hits_within_8px(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    mvp = _centering_mvp(target)
    viewport = (800, 600)

    hit = scene.pick(400.0 + 3.0, 300.0 - 4.0, viewport, mvp=mvp)   # offset (3, 4) -> dist 5.0 < 8

    assert hit == (1, 1)


def test_pick_just_past_8px_of_target_misses(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    mvp = _centering_mvp(target)
    viewport = (800, 600)

    hit = scene.pick(400.0 + 6.0, 300.0 - 6.0, viewport, mvp=mvp)   # offset (6, 6) -> dist ~8.49 > 8

    assert hit is None


def test_pick_returns_none_far_from_every_chain(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    mvp = _centering_mvp(target)
    viewport = (800, 600)

    miss = scene.pick(10.0, 10.0, viewport, mvp=mvp)

    assert miss is None


def test_pick_returns_none_when_no_chains_exist(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "a.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    assert scene.pick(400.0, 300.0, (800, 600)) is None


def test_pick_uses_the_live_camera_when_mvp_is_omitted(tmp_path):
    """``mvp=None`` (the default) must not raise -- it derives the matrix from
    ``self._plotter.camera`` (a real, if arbitrary, offscreen camera). Not aimed at anything in
    particular, so the only contract checked here is "doesn't crash, returns the documented
    type"."""
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    result = scene.pick(400.0, 300.0, (800, 600))
    assert result is None or (isinstance(result, tuple) and len(result) == 2)


def test_camera_mvp_projects_a_known_focal_point_to_the_window_center(tmp_path):
    """Every OTHER pick-math test injects a synthetic ``mvp``, so
    nothing previously exercised ``Scene._camera_mvp``'s conversion of a REAL ``vtkCamera`` (as
    opposed to ``project_points``'s own formula, tested against a hand-built matrix elsewhere).
    A camera looking straight down -z at a KNOWN world point, from a known distance, with a known
    up vector, must project that exact point to the window's CENTER -- hand-computed
    (``(w / 2, h / 2)``) -- both as a direct ``project_points`` check against
    ``Scene._camera_mvp``'s own output, and end-to-end through ``Scene.pick``'s ``mvp=None`` path
    (the target IS the target chain's own vertex here, via ``_two_chain_layer``, so a dead-center
    click must hit chain 1)."""
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)

    cam = scene._plotter.camera
    cam.position = (float(target[0]), float(target[1]), 10.0)
    cam.focal_point = (float(target[0]), float(target[1]), 0.0)
    cam.up = (0.0, 1.0, 0.0)

    viewport = tuple(scene._plotter.window_size)
    w, h = viewport
    mvp = scene._camera_mvp(viewport)
    pts2d, visible = project_points(np.asarray(cam.focal_point).reshape(1, 3), mvp, viewport)

    assert bool(visible[0])
    assert pts2d[0] == pytest.approx((w / 2.0, h / 2.0), abs=1e-6)

    hit = scene.pick(w / 2.0, h / 2.0, viewport)     # mvp=None: derives from the live camera
    assert hit == (1, 1)


def test_pick_returns_the_globally_nearest_candidate_across_layers(tmp_path):
    """``pick`` concatenates every layer's candidates into ONE array
    before calling ``nearest_point`` once -- this proves that end-to-end, not just by code
    inspection. Layer A (added FIRST) carries an in-frustum but ~49px-distant candidate; layer B
    (added SECOND) carries the exact target pixel (distance 0, confirmed numerically -- see the
    task report). If ``pick`` ever regressed to a per-layer, first-match-wins strategy instead of a
    genuine global minimum, layer A (iterated first, since ``self._chain_lookup`` is insertion-
    ordered) would incorrectly win; it must not."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    chain_a = _make_chain(2, 2, n=2, slope=0.0)
    chain_b = _make_chain(10, 10, n=2, slope=0.0)
    scene.set_layers([
        _one_chain_entry(1, "A", field, chain_a),
        _one_chain_entry(2, "B", field, chain_b),
    ])
    target_b = np.asarray(scene._plotter.actors["layer-2-chains"].mapper.dataset.points)[0]

    scale = 100.0
    mvp = np.array([
        [scale, 0.0, 0.0, -scale * float(target_b[0])],
        [0.0, scale, 0.0, -scale * float(target_b[1])],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    viewport = (800, 600)

    hit = scene.pick(400.0, 300.0, viewport, mvp=mvp)

    assert hit == (2, 0)   # layer B (second-added, distance 0) -- not layer A (first-added, ~49px)


# ------------------------------------------------------------------- masked-out chains: unpickable


def test_masked_out_chain_is_not_pickable(tmp_path):
    """A chain hidden by ``set_mask`` (alpha-zeroed, geometry untouched -- see scene.py's module
    docstring) must not be returned by ``pick`` even though it is still real, on-screen geometry a
    naive VTK hit-test would still find."""
    # chain0: slope=+1.0 -> max(log2_mod) = 3 (high). chain1 (the pick target): slope=-1.0 ->
    # max(log2_mod) = 0 (low). modulus_pctl=100 keeps only the top scorer (chain0), hiding chain1.
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path, slope0=1.0, slope1=-1.0)
    mvp = _centering_mvp(target)
    viewport = (800, 600)

    # Sanity: visible (unmasked) first -- the same click DOES hit chain1 before masking.
    assert scene.pick(400.0, 300.0, viewport, mvp=mvp) == (1, 1)

    scene.set_mask(modulus_pctl=100.0, scale_lo=0, scale_hi=0)

    assert scene.pick(400.0, 300.0, viewport, mvp=mvp) is None


def test_masking_then_unmasking_restores_pickability(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path, slope0=1.0, slope1=-1.0)
    mvp = _centering_mvp(target)
    viewport = (800, 600)

    scene.set_mask(modulus_pctl=100.0, scale_lo=0, scale_hi=0)
    assert scene.pick(400.0, 300.0, viewport, mvp=mvp) is None

    scene.set_mask(modulus_pctl=0.0, scale_lo=0, scale_hi=0)
    assert scene.pick(400.0, 300.0, viewport, mvp=mvp) == (1, 1)


# ------------------------------------------------------------------- Scene.pick_in_region
#
# Box/lasso resolution for the 3-D views,
# sharing `pick`'s own candidate assembly (`Scene._pick_candidates`, factored out additively -- see
# `pick`'s own tests above, unmodified, proving the factoring is byte-identical). These tests need
# EXACT, hand-computable screen-pixel placement for more than one point of the SAME chain (to prove
# the "ANY point inside" rule), which real geodetic projection math makes tedious to derive by hand
# -- `_region_test_layer` sidesteps that by overwriting the chains actor's own dataset points
# directly (`grid.points = ...` is an established assignment in this codebase, see scene.py:1267)
# to a hand-picked, evenly-spaced line, then a small custom orthographic-like MVP
# (`_linear_mvp`, the same shape as `_centering_mvp` above) maps that line to known pixels.


def _region_test_layer(tmp_path, n=4, layer_id=1):
    """One layer, one ``n``-point chain, with its chains actor's dataset POINTS overwritten to
    ``(0, 0, 0), (100, 0, 0), (200, 0, 0), ...`` -- full control over screen-pixel placement,
    independent of ``_geo_field``'s own CRS/projection. Returns ``(scene, chain_indices)``."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, f"region-{layer_id}.tif")
    chain = _make_chain(2, 2, n=n, slope=-0.2)
    scene.set_layers([_one_chain_entry(layer_id, "A", field, chain)])
    dataset = scene._plotter.actors[f"layer-{layer_id}-chains"].mapper.dataset
    dataset.points = np.array([[100.0 * i, 0.0, 0.0] for i in range(n)])
    _starts, chain_indices = scene._chain_lookup[layer_id]
    return scene, chain_indices


def _linear_mvp(center_x, scale):
    """A synthetic orthographic-like MVP (``w_clip == 1`` always -- the same shape as
    ``_centering_mvp`` above) mapping world x -> NDC as ``scale * (x - center_x)``; world y/z pass
    through UNSCALED (every ``_region_test_layer`` point has y = z = 0, so NDC y is always 0 --
    every projected point lands on the viewport's own vertical center row)."""
    return np.array([
        [scale, 0.0, 0.0, -scale * center_x],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


# ``_region_test_layer(tmp_path)``'s default (n=4) chain, under ``_linear_mvp(150.0, 1/300)``,
# projects to viewport (800, 600) pixels (200, 300), (333.33, 300), (466.67, 300), (600, 300) --
# hand-verified in the task report. A box/polygon around ONLY the third point (~466.7, 300) is
# used throughout this section to prove the "any point selects the whole chain" rule: the other
# three points of the SAME chain fall outside the region (but stay inside the frustum, so they are
# still candidates -- exactly what would let a wrong "every point" implementation fail this test).
_REGION_MVP = _linear_mvp(150.0, 1.0 / 300.0)
_REGION_VIEWPORT = (800, 600)


def test_pick_in_region_box_selects_chain_with_any_point_inside(tmp_path):
    scene, chain_indices = _region_test_layer(tmp_path)

    picks = scene.pick_in_region(("box", 440.0, 250.0, 500.0, 350.0), _REGION_VIEWPORT,
                                  mvp=_REGION_MVP)

    assert picks == [(1, 0)]


def test_pick_in_region_box_returns_empty_when_nothing_inside(tmp_path):
    scene, chain_indices = _region_test_layer(tmp_path)

    picks = scene.pick_in_region(("box", 0.0, 0.0, 10.0, 10.0), _REGION_VIEWPORT, mvp=_REGION_MVP)

    assert picks == []


def test_pick_in_region_poly_selects_chain_with_any_point_inside(tmp_path):
    scene, chain_indices = _region_test_layer(tmp_path)
    polygon = [(440.0, 250.0), (500.0, 250.0), (500.0, 350.0), (440.0, 350.0)]

    picks = scene.pick_in_region(("poly", polygon), _REGION_VIEWPORT, mvp=_REGION_MVP)

    assert picks == [(1, 0)]


def test_pick_in_region_poly_returns_empty_when_nothing_inside(tmp_path):
    scene, chain_indices = _region_test_layer(tmp_path)
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    picks = scene.pick_in_region(("poly", polygon), _REGION_VIEWPORT, mvp=_REGION_MVP)

    assert picks == []


def test_pick_in_region_excludes_mask_hidden_chains(tmp_path):
    """The box/lasso sibling of ``test_masked_out_chain_is_not_pickable`` above -- same fixture,
    same masking recipe, a small box around the centered target instead of a point click."""
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path, slope0=1.0, slope1=-1.0)
    mvp = _centering_mvp(target)
    viewport = (800, 600)
    box = ("box", 390.0, 290.0, 410.0, 310.0)   # a small box around the centered target (400, 300)

    assert scene.pick_in_region(box, viewport, mvp=mvp) == [(1, 1)]

    scene.set_mask(modulus_pctl=100.0, scale_lo=0, scale_hi=0)     # hides chain1 (see above)

    assert scene.pick_in_region(box, viewport, mvp=mvp) == []


def test_pick_in_region_returns_sorted_unique_pairs_across_layers(tmp_path):
    """Review-pattern companion to ``test_pick_returns_the_globally_nearest_candidate_across_
    layers`` above: layer 2 is inserted FIRST (``self._chain_lookup`` iterates it before layer 1),
    but both chains' points are placed inside the SAME box region -- a correct
    ``pick_in_region`` must still return ``[(1, 0), (2, 0)]``, sorted by the pair itself, not
    ``self._chain_lookup``'s own insertion order."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "region-sorted.tif")
    chain_a = _make_chain(2, 2, n=2, slope=0.0)
    chain_b = _make_chain(10, 10, n=2, slope=0.0)
    scene.set_layers([
        _one_chain_entry(2, "B", field, chain_b),
        _one_chain_entry(1, "A", field, chain_a),
    ])
    scene._plotter.actors["layer-2-chains"].mapper.dataset.points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scene._plotter.actors["layer-1-chains"].mapper.dataset.points = np.array(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    mvp = _linear_mvp(1.5, 0.3)
    viewport = (800, 600)

    picks = scene.pick_in_region(("box", 100.0, 250.0, 700.0, 350.0), viewport, mvp=mvp)

    assert picks == [(1, 0), (2, 0)]


def test_pick_in_region_returns_empty_when_no_chains_exist(tmp_path):
    scene = Scene(_plotter())
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    field = _geo_field(tmp_path, "region-empty.tif")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "ok"}])

    assert scene.pick_in_region(("box", 0.0, 0.0, 800.0, 600.0), (800, 600)) == []


def test_pick_in_region_uses_the_live_camera_when_mvp_is_omitted(tmp_path):
    """``mvp=None`` must not raise -- mirrors ``test_pick_uses_the_live_camera_when_mvp_is_
    omitted`` above; not aimed at anything in particular, so the only contract checked here is
    "doesn't crash, returns the documented type"."""
    scene, chain_indices = _region_test_layer(tmp_path)

    result = scene.pick_in_region(("box", 0.0, 0.0, 800.0, 600.0), (800, 600))

    assert isinstance(result, list)


def test_pick_in_region_unknown_kind_raises(tmp_path):
    scene, chain_indices = _region_test_layer(tmp_path)

    with pytest.raises(ValueError):
        scene.pick_in_region(("circle", 1.0, 2.0, 3.0), _REGION_VIEWPORT, mvp=_REGION_MVP)


# --------------------------------------------------------------------- selection / group re-color


def test_set_selection_recolors_without_rebuild(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    actor_before = scene._plotter.actors["layer-1-chains"]
    dataset_before = actor_before.mapper.dataset

    scene.set_selection({(1, 1)})

    actor_after = scene._plotter.actors["layer-1-chains"]
    assert actor_after is actor_before
    assert actor_after.mapper.dataset is dataset_before

    colors = np.asarray(dataset_before.point_data["colors"])
    chain1_rgb = colors[chain_indices == 1][:, :3]
    chain0_rgb = colors[chain_indices == 0][:, :3]
    assert (chain1_rgb == np.array(SELECTION_COLOR, dtype=np.uint8)).all()
    assert not (chain0_rgb == np.array(SELECTION_COLOR, dtype=np.uint8)).all()


def test_set_selection_empty_restores_unselected_colors(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    scene.set_selection({(1, 1)})
    scene.set_selection(set())

    colors = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])
    chain1_rgb = colors[chain_indices == 1][:, :3]
    assert not (chain1_rgb == np.array(SELECTION_COLOR, dtype=np.uint8)).all()


def test_set_group_preview_recolors_member_chains(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    group_color = (0, 114, 178)

    scene.set_group_preview({"g1": {"chains": [(1, 1)], "color": list(group_color)}})

    colors = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])
    chain1_rgb = colors[chain_indices == 1][:, :3]
    chain0_rgb = colors[chain_indices == 0][:, :3]
    assert (chain1_rgb == np.array(group_color, dtype=np.uint8)).all()
    assert not (chain0_rgb == np.array(group_color, dtype=np.uint8)).all()


def test_selection_color_wins_over_group_preview_color(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    scene.set_group_preview({"g1": {"chains": [(1, 1)], "color": [0, 114, 178]}})
    scene.set_selection({(1, 1)})

    colors = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])
    chain1_rgb = colors[chain_indices == 1][:, :3]
    assert (chain1_rgb == np.array(SELECTION_COLOR, dtype=np.uint8)).all()


def test_selection_and_group_preview_survive_a_mode_rebuild(tmp_path):
    scene, pts3d, chain_indices, target = _two_chain_layer(tmp_path)
    scene.set_group_preview({"g1": {"chains": [(1, 0)], "color": [0, 114, 178]}})
    scene.set_selection({(1, 1)})

    scene.set_mode("globe")

    _starts, chain_indices2 = scene._chain_lookup[1]
    colors = np.asarray(scene._plotter.actors["layer-1-chains"].mapper.dataset.point_data["colors"])
    assert (colors[chain_indices2 == 0][:, :3] == np.array([0, 114, 178], dtype=np.uint8)).all()
    assert (colors[chain_indices2 == 1][:, :3] == np.array(SELECTION_COLOR, dtype=np.uint8)).all()


# ---------------------------------------------------------------------------------- pick benchmark


def test_pick_latency_benchmark(tmp_path):
    """The design's pick bar: <50 ms. Reuses the mask benchmark's 2-field / 50k-chain-vertex
    fixture (tests/test_arrangement_mask.py) -- the same scale the mask benchmark already
    established as representative."""
    ny = nx = 1414
    field_a = _big_field(ny, nx, "a")
    field_b = _big_field(ny, nx, "b")
    chains_a = _synthetic_chains(500, 50, ny, nx, seed=1)
    chains_b = _synthetic_chains(500, 50, ny, nx, seed=2)

    layer_a = Layer(layer_id=1, name="A", source_id="mem:a")
    layer_b = Layer(layer_id=2, name="B", source_id="mem:b")
    scene = Scene(_plotter())
    scene.set_layers([
        {"layer": layer_a, "field": field_a, "result": _result(chains_a), "status": "ok"},
        {"layer": layer_b, "field": field_b, "result": _result(chains_b), "status": "ok"},
    ])

    viewport = (1024, 768)
    n_calls = 50
    times = np.empty(n_calls, dtype=np.float64)
    for i in range(n_calls):
        t0 = time.perf_counter()
        scene.pick(400.0 + i, 300.0, viewport)
        times[i] = time.perf_counter() - t0

    mean_ms = float(times.mean() * 1000.0)
    max_ms = float(times.max() * 1000.0)
    print(f"\nScene.pick: mean {mean_ms:.3f} ms, max {max_ms:.3f} ms over {n_calls} calls "
          f"(2 layers, 50,000 chain vertices)")
    assert mean_ms < 50.0, f"pick mean {mean_ms:.3f} ms exceeds the generous 50 ms CI bound"


# ------------------------------------------------------------- staleness guard (Scene.set_layers)
#
# A background recompute landing on an ALREADY-tracked layer_id hands
# it a brand new result object (a new chains list) -- any (layer_id, chain_index) this scene is
# still holding for it may now name the wrong chain. Scene.set_layers detects this by RESULT-OBJECT
# IDENTITY (not content) and drops that layer's own selection/group-preview entries, reporting
# which layer_ids it pruned.


def test_set_layers_prunes_selection_and_group_preview_on_a_result_identity_change(tmp_path):
    """A result-identity change alone no longer prunes -- the
    entries here carry DIFFERENT ``signature`` values, honestly representing "the transform
    actually re-ran" (a genuine background recompute), which is what must still prune. See
    ``test_set_layers_does_not_prune_a_content_equal_refilter_with_a_new_result_object`` below
    for the companion case this predicate exists to fix: a new object, same content, no prune."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    chain1a = _make_chain(2, 2, n=2, slope=0.0)
    chain2 = _make_chain(10, 10, n=2, slope=0.0)
    entry1a = _one_chain_entry(1, "A", field, chain1a, signature="sig-a")
    entry2 = _one_chain_entry(2, "B", field, chain2, signature="sig-b")
    scene.set_layers([entry1a, entry2])

    scene.set_selection({(1, 0), (2, 0)})
    scene.set_group_preview({"g": {"chains": [(1, 0), (2, 0)], "color": [1, 2, 3]}})
    assert scene._selection[1] == {0} and scene._selection[2] == {0}

    # Simulate a background recompute landing for layer 1: SAME layer_id, a NEW result object,
    # a DIFFERENT signature (the transform's own params genuinely moved).
    chain1b = _make_chain(2, 2, n=2, slope=0.0)
    entry1b = _one_chain_entry(1, "A", field, chain1b, signature="sig-a-v2")
    pruned = scene.set_layers([entry1b, entry2])

    assert pruned == [1]
    assert 1 not in scene._selection
    assert 1 not in scene._group_preview
    assert scene._selection[2] == {0}                     # layer 2 untouched
    assert scene._group_preview[2] == {0: (1, 2, 3)}       # layer 2 untouched


def test_set_layers_does_not_prune_a_content_equal_refilter_with_a_new_result_object(tmp_path):
    """Every real chain has at least one Filter,
    and a Filter always returns a fresh dict (``dynamix/engine/resolve.py``'s own module
    docstring) -- so a REPEATED ``MainWindow._sync_arrangement()`` pass (two landings in a row;
    The multi-layer commit needing one resync per affected layer) mints a NEW
    ``id(result)`` for a layer whose computation has not changed at all. That must not prune: same
    chain COUNT, same transform SIGNATURE -> exploration/commit state survives, even though the
    result object itself is a different one from before."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    chain_a = _make_chain(2, 2, n=2, slope=0.0)
    entry_a = _one_chain_entry(1, "A", field, chain_a, signature="sig-a")
    scene.set_layers([entry_a])
    scene.set_selection({(1, 0)})
    scene.set_group_preview({"g": {"chains": [(1, 0)], "color": [4, 5, 6]}})

    chain_a2 = _make_chain(2, 2, n=2, slope=0.0)      # different object, same shape/content
    entry_a2 = _one_chain_entry(1, "A", field, chain_a2, signature="sig-a")   # SAME signature
    assert entry_a2["result"] is not entry_a["result"]     # the identity that USED to over-fire
    pruned = scene.set_layers([entry_a2])

    assert pruned == []
    assert scene._selection[1] == {0}
    assert scene._group_preview[1] == {0: (4, 5, 6)}


# ------------------------------------------- The positional-digest fingerprint
#
# Round 1's (chain_count, signature) fingerprint is structurally BLIND to a
# count-preserving FILTER change upstream of the tracked chains -- a filter's own params never
# enter the transform signature by design, so a re-selection that swaps in a DIFFERENT, equal-sized
# chain set left the fingerprint unchanged, and a stale (layer_id, chain_index) would silently keep
# naming the WRONG chain. Fixed by replacing the count component with a per-chain POSITIONAL DIGEST
# (Scene._fingerprint / _chain_root) -- order-sensitive, reselection-sensitive, and copy-immune
# (GroupPaint's stamped copies preserve x/y/length exactly, so the commit resync stays intact).


def test_set_layers_prunes_a_count_preserving_reselection(tmp_path):
    """Required test (1): the actual bug this round fixes. Same layer_id, same chain COUNT, same
    transform SIGNATURE -- round 1's fingerprint would have called this content-equal -- but the
    SECOND chain is a genuinely DIFFERENT one (different root position), simulating a count-
    preserving filter re-selection upstream. Must prune."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    chain_kept = _make_chain(2, 2, n=2, slope=0.0)
    chain_old = _make_chain(10, 10, n=2, slope=0.0)
    chain_new = _make_chain(30, 30, n=2, slope=0.0)      # different root, same length -> same count

    entry_before = {"layer": layer, "field": field, "result": _result([chain_kept, chain_old]),
                     "status": "ok", "signature": "sig-a"}
    scene.set_layers([entry_before])
    scene.set_selection({(1, 0), (1, 1)})

    entry_after = {"layer": layer, "field": field, "result": _result([chain_kept, chain_new]),
                    "status": "ok", "signature": "sig-a"}    # SAME count, SAME signature
    pruned = scene.set_layers([entry_after])

    assert pruned == [1]
    assert 1 not in scene._selection


def test_set_layers_prunes_a_pure_reorder_of_the_same_chain_set(tmp_path):
    """Required test (3): order-sensitivity. The identical two chain OBJECTS, merely reordered --
    same count, same signature, same set -- must still prune: a reorder moves which
    (layer_id, chain_index) names which chain just as much as a swap does."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    chain_a = _make_chain(2, 2, n=2, slope=0.0)
    chain_b = _make_chain(10, 10, n=2, slope=0.0)

    entry_before = {"layer": layer, "field": field, "result": _result([chain_a, chain_b]),
                     "status": "ok", "signature": "sig-a"}
    scene.set_layers([entry_before])
    scene.set_selection({(1, 0)})           # picked chain_a, at index 0

    entry_after = {"layer": layer, "field": field, "result": _result([chain_b, chain_a]),
                    "status": "ok", "signature": "sig-a"}     # SAME two chains, REORDERED
    pruned = scene.set_layers([entry_after])

    assert pruned == [1]
    assert 1 not in scene._selection


def test_set_layers_does_not_prune_after_a_real_group_paint_stamped_resync(tmp_path):
    """Required test (2): the commit's-own-resync case from round 1 stays green under the revised
    predicate. Uses the REAL ``GroupPaint.apply()`` (not a hand-built "different object, same
    content" stand-in) -- its stamped copies preserve every chain's x/y/length exactly, so the
    positional digest is identical before and after, and Scene must not prune."""
    from dynamix.devices.groups import GroupPaint, encode_groups

    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    chain0 = _make_chain(2, 2, n=2, slope=0.0)
    chain1 = _make_chain(10, 10, n=2, slope=0.0)
    result_before = _result([chain0, chain1])
    entry_before = {"layer": layer, "field": field, "result": result_before, "status": "ok",
                    "signature": "sig-a"}
    scene.set_layers([entry_before])
    scene.set_selection({(1, 0)})

    groups = {"g": {"signature": "sig-a", "chains": [0], "color": [1, 2, 3]}}
    params = {"spec_json": encode_groups(groups), "signature": "sig-a"}
    result_after = GroupPaint().apply(result_before, params)
    assert result_after is not result_before               # a genuine new object, like a real resync
    assert result_after["chains"] is not result_before["chains"]
    entry_after = {"layer": layer, "field": field, "result": result_after, "status": "ok",
                   "signature": "sig-a"}

    pruned = scene.set_layers([entry_after])

    assert pruned == []
    assert scene._selection[1] == {0}


def test_set_layers_does_not_prune_on_a_repeat_call_with_the_same_result_object(tmp_path):
    """A plain resync (the ``_sync_arrangement`` re-derives entries on every worker landing,
    but a cache-hit layer's result object is the SAME one every time) must not spuriously prune."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    entry = _one_chain_entry(1, "A", field, _make_chain(2, 2, n=2, slope=0.0))
    scene.set_layers([entry])
    scene.set_selection({(1, 0)})

    pruned = scene.set_layers([entry])          # SAME entry, SAME result object

    assert pruned == []
    assert scene._selection[1] == {0}


def test_set_layers_does_not_prune_a_layers_first_ever_result(tmp_path):
    """A layer transitioning from ``result=None`` ("computing") to its first real result is
    routine, not staleness: ``pick()`` could never have returned a hit for it while ``result`` was
    ``None`` (no ``_chain_lookup`` entry existed), so nothing could have been selected or grouped
    for it yet -- see ``set_layers``'s own docstring."""
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    layer = Layer(layer_id=1, name="A", source_id="mem:a")
    scene.set_layers([{"layer": layer, "field": field, "result": None, "status": "computing"}])

    pruned = scene.set_layers([_one_chain_entry(1, "A", field, _make_chain(2, 2, n=2, slope=0.0))])

    assert pruned == []


def test_set_layers_prunes_a_layer_that_disappears_entirely(tmp_path):
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    entry1 = _one_chain_entry(1, "A", field, _make_chain(2, 2, n=2, slope=0.0))
    entry2 = _one_chain_entry(2, "B", field, _make_chain(10, 10, n=2, slope=0.0))
    scene.set_layers([entry1, entry2])
    scene.set_selection({(1, 0)})

    pruned = scene.set_layers([entry2])          # layer 1 removed entirely

    assert pruned == [1]
    assert 1 not in scene._selection


# --------------------------------------------------------------- GroupPalette.prune_layer


def test_prune_layer_removes_only_that_layers_members(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    g = palette.new_group("g")
    palette.add_pick((1, 0), shift=False)
    palette.add_pick((2, 0), shift=True)
    assert palette.groups()[g]["chains"] == [(1, 0), (2, 0)]

    palette.prune_layer(1)

    assert palette.groups()[g]["chains"] == [(2, 0)]
    assert palette.selection() == {(2, 0)}


def test_prune_layer_emits_membership_changed(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    palette.new_group("g")
    palette.add_pick((1, 0), shift=False)

    with qtbot.waitSignal(palette.membershipChanged, timeout=1000) as blocker:
        palette.prune_layer(1)

    assert blocker.args[0]["g"]["chains"] == []


def test_prune_layer_is_a_harmless_no_op_when_nothing_matches(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    g = palette.new_group("g")
    palette.add_pick((2, 0), shift=False)

    palette.prune_layer(1)      # layer 1 has no members here

    assert palette.groups()[g]["chains"] == [(2, 0)]


# ------------------------------------------------------------------------------------ GroupPalette


def test_new_group_auto_names_and_cycles_colors(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)

    name1 = palette.new_group()
    name2 = palette.new_group()

    assert name1 == "Group 1"
    assert name2 == "Group 2"
    groups = palette.groups()
    assert groups[name1]["color"] == list(GROUP_COLORS[0])
    assert groups[name2]["color"] == list(GROUP_COLORS[1])
    assert groups[name1]["chains"] == []


def test_new_group_becomes_active(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)

    name = palette.new_group()

    assert palette.active_group == name


def test_new_group_duplicate_name_raises(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    palette.new_group("mine")

    with pytest.raises(ValueError):
        palette.new_group("mine")


def test_set_active_group_switches(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    a = palette.new_group("a")
    b = palette.new_group("b")
    assert palette.active_group == b

    palette.set_active_group(a)

    assert palette.active_group == a


def test_plain_click_replaces_selection_and_adds_to_active_group(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    g = palette.new_group("g")

    palette.add_pick((1, 5), shift=False)
    assert palette.selection() == {(1, 5)}
    assert palette.groups()[g]["chains"] == [(1, 5)]

    palette.add_pick((1, 9), shift=False)
    assert palette.selection() == {(1, 9)}                       # replaced
    assert palette.groups()[g]["chains"] == [(1, 5), (1, 9)]      # but membership accumulates


def test_shift_click_adds_to_selection_and_group(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    g = palette.new_group("g")

    palette.add_pick((1, 1), shift=False)
    palette.add_pick((1, 2), shift=True)

    assert palette.selection() == {(1, 1), (1, 2)}
    assert palette.groups()[g]["chains"] == [(1, 1), (1, 2)]


def test_pick_with_no_active_group_only_updates_selection(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)

    palette.add_pick((1, 1), shift=False)

    assert palette.selection() == {(1, 1)}
    assert palette.groups() == {}


def test_plain_click_miss_clears_selection(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    palette.new_group("g")
    palette.add_pick((1, 1), shift=False)

    palette.add_pick(None, shift=False)

    assert palette.selection() == set()


def test_shift_click_miss_does_not_clear_selection(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    palette.new_group("g")
    palette.add_pick((1, 1), shift=False)

    palette.add_pick(None, shift=True)

    assert palette.selection() == {(1, 1)}


def test_membership_changed_emits_the_documented_payload_shape(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    g = palette.new_group("g")

    with qtbot.waitSignal(palette.membershipChanged, timeout=1000) as blocker:
        palette.add_pick((1, 3), shift=False)

    payload = blocker.args[0]
    assert payload == {g: {"chains": [(1, 3)], "color": list(GROUP_COLORS[0])}}


def test_picking_a_chain_twice_does_not_duplicate_membership(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    g = palette.new_group("g")

    palette.add_pick((1, 1), shift=True)
    palette.add_pick((1, 1), shift=True)

    assert palette.groups()[g]["chains"] == [(1, 1)]


# --------------------------------------------------------------------- ArrangementView wiring
#
# GroupPalette + the Commit button live in MainWindow's right panel, not this view (tests/test_right_panel.py covers the new home
# and the window-level wiring) -- the view keeps only a public `set_group_palette(palette)`
# passthrough, storing the reference it reads picks into and snapshots from at commit time. These
# tests mirror tests/test_arrangement_mask.py's own Task-4 relocation pins for the identical shape.


def test_arrangement_view_no_longer_builds_a_group_palette_or_commit_button(qtbot):
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    assert view._group_palette is None
    assert not hasattr(view, "_commit_button")


def test_arrangement_view_wires_group_palette_to_scene_lazily(qtbot):
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    # The view no longer builds its own palette -- MainWindow does,
    # and hands the reference over through set_group_palette (here standing in for what
    # _toggle_center_view does for real). No scene yet (a real QtInteractor cannot be built under
    # this harness's mandated offscreen QPA platform -- see tests/test_arrangement_flip.py's
    # documented segfault guard): moving the palette must not raise.
    view.set_group_palette(GroupPalette())
    view._group_palette.new_group("g")
    view._group_palette.add_pick((1, 0), shift=False)

    # Attach a Scene directly against a real offscreen pv.Plotter, the same substitution
    # tests/test_arrangement_mask.py uses for MaskRow wiring.
    view._scene = Scene(_plotter())
    view._group_palette.add_pick((1, 1), shift=True)

    assert view._scene._group_preview.get(1) == {0: tuple(GROUP_COLORS[0]), 1: tuple(GROUP_COLORS[0])}
    assert view._scene._selection.get(1) == {0, 1}


def test_arrangement_view_forwards_scene_pruning_to_the_group_palette(qtbot, tmp_path):
    """The view-layer half: ``ArrangementView.set_layers`` must turn
    ``Scene.set_layers``'s pruned-``layer_id`` report into ``GroupPalette.prune_layer`` calls, so
    the palette's SEPARATELY-held committed-group membership never keeps naming a chain the scene
    itself has already disowned after a simulated recompute (same layer_id, new result object,
    DIFFERENT signature -- a same-signature same-count refilter is
    no longer staleness, so this test's own "recompute" has to be an honest one)."""
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    # Hand the view its palette reference first (see the comment on
    # the sibling test above) -- MainWindow does this once, before set_layers is ever called.
    view.set_group_palette(GroupPalette())
    view._scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    entry1a = _one_chain_entry(1, "A", field, _make_chain(2, 2, n=2, slope=0.0), signature="sig-1")
    view.set_layers([entry1a])

    view._group_palette.new_group("g")
    view._group_palette.add_pick((1, 0), shift=False)
    assert view._group_palette.groups()["g"]["chains"] == [(1, 0)]

    entry1b = _one_chain_entry(1, "A", field, _make_chain(2, 2, n=2, slope=0.0), signature="sig-2")
    view.set_layers([entry1b])          # same layer_id, new result object -- simulated recompute

    assert view._group_palette.groups()["g"]["chains"] == []


# ------------------------------------------------------------ click -> pick (a real VTK event)
#
# The ORIGINAL wiring registered track_click_position with the default
# viewport=False, under which the callback receives a 3-D WORLD-SPACE pick position (pyvista's own
# pick_click_position(), a real VTK geometry pick) -- `x, y = point` on that 3-tuple raised
# ValueError on every real click, confirmed by firing a genuine event exactly as these tests do.
# Fixed by registering with viewport=True (view.py's activate()); these tests prove the fix by
# firing a real LeftButtonPressEvent (SetEventPosition + InvokeEvent) at a PLAIN offscreen
# pv.Plotter substituted in for view._interactor -- the same duck-typed substitution
# tests/test_arrangement_scene.py / tests/test_arrangement_mask.py already use throughout for
# Scene's own pyvistaqt.QtInteractor. No Qt/QtInteractor construction is needed here:
# track_click_position and the VTK event it observes are plain-VTK mechanisms with no Qt dependency
# of their own (see view.py's corrected module docstring for the earlier, wrong "live only" claim).


def test_on_click_dispatches_a_pick_from_a_real_vtk_click_event(qtbot):
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    # No palette attached -- deliberately: this test proves the pick
    # itself still runs even when nothing is listening for the result (see the dedicated no-op
    # test just below for the palette-forwarding half of this same contract).
    view._interactor = pv.Plotter(off_screen=True, window_size=(800, 600))
    view._scene = Scene(view._interactor)
    view._scene.pick = Mock(return_value=(1, 2))
    view._interactor.track_click_position(view._on_click, side="left", viewport=True)

    iren = view._interactor.iren.interactor
    iren.SetEventPosition(123, 456)
    iren.InvokeEvent("LeftButtonPressEvent")

    # VTK's display coords are bottom-left-origin (y=456 up from the bottom, window height 600);
    # _on_click flips to the top-left-origin convention Scene.pick/project_points use: 600 - 456.
    view._scene.pick.assert_called_once_with(123.0, 600.0 - 456.0, (800, 600))


def test_on_click_dispatches_the_pick_result_and_shift_state_to_the_palette(qtbot):
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    # Attach a palette first -- the view no longer builds its own.
    view.set_group_palette(GroupPalette())
    view._interactor = pv.Plotter(off_screen=True, window_size=(800, 600))
    view._scene = Scene(view._interactor)
    view._scene.pick = Mock(return_value=(1, 2))
    view._group_palette.add_pick = Mock()
    view._interactor.track_click_position(view._on_click, side="left", viewport=True)

    iren = view._interactor.iren.interactor
    iren.SetEventPosition(10, 20)
    iren.SetShiftKey(1)
    iren.InvokeEvent("LeftButtonPressEvent")

    view._group_palette.add_pick.assert_called_once_with((1, 2), True)


def test_on_click_is_a_no_op_when_no_group_palette_was_ever_attached(qtbot):
    """Headless guard mirroring set_mask's own no-op-before-anything-
    attached contract -- a real click firing with a scene but no palette (set_group_palette never
    called) must not raise. Scene.pick still runs (see the comment on the sibling test above);
    only the palette hand-off itself is skipped."""
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    view._interactor = pv.Plotter(off_screen=True, window_size=(800, 600))
    view._scene = Scene(view._interactor)
    view._scene.pick = Mock(return_value=(1, 2))
    view._interactor.track_click_position(view._on_click, side="left", viewport=True)

    iren = view._interactor.iren.interactor
    iren.SetEventPosition(1, 1)
    iren.InvokeEvent("LeftButtonPressEvent")     # must not raise -- view._group_palette is None

    view._scene.pick.assert_called_once()


def test_on_click_is_a_no_op_before_a_scene_exists(qtbot):
    """Mirrors _on_mask_changed/_on_membership_changed's own no-scene-yet contract: a click firing
    before activate() has built anything must not raise."""
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    view._interactor = pv.Plotter(off_screen=True, window_size=(800, 600))
    view._interactor.track_click_position(view._on_click, side="left", viewport=True)

    iren = view._interactor.iren.interactor
    iren.SetEventPosition(1, 1)
    iren.InvokeEvent("LeftButtonPressEvent")     # must not raise -- view._scene is still None


# --------------------------------------------------------- ArrangementView._apply_region_pick
#
# The box/lasso release handler -> Scene.pick_in_region ->
# GroupPalette.apply_picks. Same fixture shape as the _on_click tests just above (a plain offscreen
# pv.Plotter substituted for the real QtInteractor, Scene.pick_in_region mocked directly rather than
# exercised end-to-end -- that machinery is already covered exhaustively above).


def test_apply_region_pick_forwards_picks_to_the_palette_with_add_op(qtbot):
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    view.set_group_palette(GroupPalette())
    view._interactor = pv.Plotter(off_screen=True, window_size=(800, 600))
    view._scene = Scene(view._interactor)
    view._scene.pick_in_region = Mock(return_value=[(1, 0), (1, 1)])

    view._apply_region_pick(("box", 0.0, 0.0, 10.0, 10.0), subtract=False)

    view._scene.pick_in_region.assert_called_once_with(("box", 0.0, 0.0, 10.0, 10.0), (800, 600))
    assert view._group_palette.selection() == {(1, 0), (1, 1)}


def test_apply_region_pick_subtract_true_removes_from_the_existing_selection(qtbot):
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    palette = GroupPalette()
    view.set_group_palette(palette)
    palette.apply_picks([(1, 0), (1, 1), (1, 2)], "add")
    view._interactor = pv.Plotter(off_screen=True, window_size=(800, 600))
    view._scene = Scene(view._interactor)
    view._scene.pick_in_region = Mock(return_value=[(1, 0), (1, 1)])

    view._apply_region_pick(("poly", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]), subtract=True)

    assert palette.selection() == {(1, 2)}


def test_apply_region_pick_is_a_no_op_when_no_group_palette_was_ever_attached(qtbot):
    """Mirrors ``test_on_click_is_a_no_op_when_no_group_palette_was_ever_attached`` above -- the
    pick against ``Scene`` still runs regardless of whether anyone is listening for it."""
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    view._interactor = pv.Plotter(off_screen=True, window_size=(800, 600))
    view._scene = Scene(view._interactor)
    view._scene.pick_in_region = Mock(return_value=[(1, 0)])

    view._apply_region_pick(("box", 0.0, 0.0, 10.0, 10.0), subtract=False)   # must not raise

    view._scene.pick_in_region.assert_called_once()


def test_apply_region_pick_is_a_no_op_before_a_scene_exists(qtbot):
    """Mirrors ``test_on_click_is_a_no_op_before_a_scene_exists`` above."""
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    view._apply_region_pick(("box", 0.0, 0.0, 10.0, 10.0), subtract=False)   # must not raise


# ------------------------------------------------------------- _RegionSelectFilter
#
# The gesture-capture event filter installed
# on the live 3-D interactor in ArrangementView.activate(). Its swallow/pass DECISION logic (and
# the region-resolution it triggers on release) are deliberately testable by calling
# `eventFilter(obj, event)` directly with stub events -- no real interactor, no QApplication event
# dispatch, no offscreen segfault risk -- per the module docstring's own "Box/lasso selection"
# section. `_LassoRegionOverlay`'s own PAINTING is the one cocoa-only surface this task ships (see
# that class's own docstring) and is deliberately NOT exercised here.
#
# "Camera untouched" proof: an installed QObject.eventFilter returning True is Qt's own guarantee
# that the event is never delivered to the target object (here, the interactor widget) at all --
# proving `eventFilter` returns True for a swallowed event IS the proof the VTK trackball (which
# only ever sees events Qt actually delivers to the widget) never receives it. No live interactor
# or spied trackball call is needed to establish that; asserting the return value directly is the
# more precise test; a spy on a real interactor's style would only re-derive the same guarantee
# indirectly, at the cost of needing a real interactor this suite cannot build offscreen at all.


def _stub_mouse_event(event_type, pos=(0.0, 0.0), button=None, modifiers=None):
    """A stand-in for a QMouseEvent, built as a ``Mock`` -- ``_RegionSelectFilter.eventFilter``
    only ever calls ``.type()``/``.button()``/``.position()``/``.modifiers()`` on the event it is
    handed, so a full, real ``QMouseEvent`` (fiddly to construct with an exact type/button/
    position/modifier combination) is unnecessary. ``position`` becomes a genuine
    ``QtCore.QPointF`` (not a further mock) since the filter calls REAL methods on it
    (``.x()``, ``.y()``, ``.toPoint()``)."""
    event = Mock()
    event.type.return_value = event_type
    event.position.return_value = QtCore.QPointF(*pos)
    event.button.return_value = (button if button is not None
                                  else QtCore.Qt.MouseButton.LeftButton)
    event.modifiers.return_value = (modifiers if modifiers is not None
                                     else QtCore.Qt.KeyboardModifier.NoModifier)
    return event


def _region_filter(qtbot, mode):
    """One ``ArrangementView`` + one plain child ``QWidget`` (standing in for the live
    interactor -- offscreen-safe, unlike the real ``pyvistaqt.QtInteractor``) + one
    ``_RegionSelectFilter`` wired to it, mode already set. Returns ``(view, widget, filt)``."""
    from dynamix.shell.arrangement.view import ArrangementView, _RegionSelectFilter

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")
    view.set_selection_mode(mode)
    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()          # QRubberBand's own isVisible() needs its PARENT actually shown too
    filt = _RegionSelectFilter(view, widget)
    return view, widget, filt


def test_region_filter_click_mode_passes_every_event_through(qtbot):
    view, widget, filt = _region_filter(qtbot, "click")

    press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 10.0))
    move = _stub_mouse_event(QtCore.QEvent.MouseMove, (20.0, 20.0))
    release = _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (20.0, 20.0))

    assert filt.eventFilter(widget, press) is False
    assert filt.eventFilter(widget, move) is False
    assert filt.eventFilter(widget, release) is False


def test_region_filter_transect_mode_passes_every_event_through(qtbot):
    """The transect gesture does not exist yet -- true pass-through is the
    honest placeholder (see the class's own docstring)."""
    view, widget, filt = _region_filter(qtbot, "transect")

    press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 10.0))
    move = _stub_mouse_event(QtCore.QEvent.MouseMove, (20.0, 20.0))
    release = _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (20.0, 20.0))

    assert filt.eventFilter(widget, press) is False
    assert filt.eventFilter(widget, move) is False
    assert filt.eventFilter(widget, release) is False


def test_region_filter_box_mode_swallows_press_and_move(qtbot):
    """The required gesture-filter contract: in box mode, synthetic move events are
    swallowed -- camera untouched, see the section banner's own "camera untouched" proof note."""
    view, widget, filt = _region_filter(qtbot, "box")

    press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 10.0))
    move = _stub_mouse_event(QtCore.QEvent.MouseMove, (20.0, 20.0))

    assert filt.eventFilter(widget, press) is True
    assert filt.eventFilter(widget, move) is True
    assert filt._band.isVisible()          # the rubber band is up while the drag is in progress


def test_region_filter_lasso_mode_swallows_press_and_move(qtbot):
    view, widget, filt = _region_filter(qtbot, "lasso")

    press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 10.0))
    move = _stub_mouse_event(QtCore.QEvent.MouseMove, (20.0, 20.0))

    assert filt.eventFilter(widget, press) is True
    assert filt.eventFilter(widget, move) is True
    assert filt._pts == [(10.0, 10.0), (20.0, 20.0)]


def test_region_filter_ignores_a_right_button_press_in_box_mode(qtbot):
    """Only a LEFT-button press starts a drag -- a stray right-click in box mode is not a
    gesture this filter recognizes, and must not start one."""
    view, widget, filt = _region_filter(qtbot, "box")

    press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 10.0),
                               button=QtCore.Qt.MouseButton.RightButton)

    assert filt.eventFilter(widget, press) is False
    assert filt._dragging is False


def test_region_filter_release_with_no_prior_press_is_ignored(qtbot):
    """A release event arriving with no drag in progress (``self._dragging`` still False --
    e.g. the widget only just gained the filter) is not this filter's gesture to resolve."""
    view, widget, filt = _region_filter(qtbot, "box")
    view._apply_region_pick = Mock()

    release = _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (20.0, 20.0))

    assert filt.eventFilter(widget, release) is False
    view._apply_region_pick.assert_not_called()


def test_region_filter_box_release_resolves_region_scaled_for_hidpi_with_add_op(qtbot):
    """EQSelect's own ``_scale`` pattern: the widget's logical size (400x300, set
    by ``_region_filter``) is HALF the interactor's own render-window size (800x600) -- every
    captured logical-px coordinate must be DOUBLED before reaching ``_apply_region_pick``."""
    view, widget, filt = _region_filter(qtbot, "box")
    view._apply_region_pick = Mock()
    view._interactor = Mock(window_size=(800, 600))

    press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 20.0))
    move = _stub_mouse_event(QtCore.QEvent.MouseMove, (30.0, 40.0))
    release = _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (50.0, 60.0))

    assert filt.eventFilter(widget, press) is True
    assert filt.eventFilter(widget, move) is True
    assert filt.eventFilter(widget, release) is True
    assert not filt._band.isVisible()      # taken off screen -- the gesture is over

    view._apply_region_pick.assert_called_once_with(("box", 20.0, 40.0, 100.0, 120.0), False)


def test_region_filter_box_release_alt_modifier_reports_subtract(qtbot):
    view, widget, filt = _region_filter(qtbot, "box")
    view._apply_region_pick = Mock()
    view._interactor = Mock(window_size=(800, 600))

    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 20.0)))
    release = _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (50.0, 60.0),
                                 modifiers=QtCore.Qt.KeyboardModifier.AltModifier)

    assert filt.eventFilter(widget, release) is True

    view._apply_region_pick.assert_called_once_with(("box", 20.0, 40.0, 100.0, 120.0), True)


def test_region_filter_box_release_without_any_move_still_resolves(qtbot):
    """Unlike lasso (below), a box needs no minimum point count -- press+release alone already
    define both corners (origin and release), even a zero-area box."""
    view, widget, filt = _region_filter(qtbot, "box")
    view._apply_region_pick = Mock()
    view._interactor = Mock(window_size=(800, 600))

    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 20.0)))
    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (10.0, 20.0)))

    view._apply_region_pick.assert_called_once_with(("box", 20.0, 40.0, 20.0, 40.0), False)


def test_region_filter_lasso_release_with_three_points_resolves_polygon(qtbot):
    view, widget, filt = _region_filter(qtbot, "lasso")
    view._apply_region_pick = Mock()
    view._interactor = Mock(window_size=(800, 600))

    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (0.0, 0.0)))
    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseMove, (10.0, 0.0)))
    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseMove, (10.0, 10.0)))
    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (10.0, 10.0)))

    view._apply_region_pick.assert_called_once_with(
        ("poly", [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]), False)


def test_region_filter_lasso_release_with_fewer_than_three_points_is_a_no_op(qtbot):
    """A press released with no real drag (a single captured point) has no polygon to resolve --
    mirrors ``dynamix.shell.canvas.Canvas.mouseReleaseEvent``'s identical guard for the raster
    canvas's own lasso."""
    view, widget, filt = _region_filter(qtbot, "lasso")
    view._apply_region_pick = Mock()
    view._interactor = Mock(window_size=(800, 600))

    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (0.0, 0.0)))
    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (0.0, 0.0)))

    view._apply_region_pick.assert_not_called()


def test_region_filter_mid_drag_mode_switch_stays_frozen(qtbot):
    """Fix round 1: a hotkey mode-switch mid-drag (`c`/`v` are WindowShortcut-
    context QShortcuts that fire independent of whatever widget holds the mouse grab --
    `main_window.py`) must NOT un-swallow the drag. Once the press below starts a BOX gesture,
    the live selection mode is switched to "lasso" mid-drag; the move must still be swallowed
    (camera untouched) and the release must still resolve as a BOX (the frozen press-time mode),
    not a lasso -- with the band hidden and state reset afterward."""
    view, widget, filt = _region_filter(qtbot, "box")
    view._apply_region_pick = Mock()
    view._interactor = Mock(window_size=(800, 600))

    press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 20.0))
    assert filt.eventFilter(widget, press) is True
    assert filt._mode_at_press == "box"

    view.set_selection_mode("lasso")     # the mid-drag mode switch -- live mode is now "lasso"

    move = _stub_mouse_event(QtCore.QEvent.MouseMove, (30.0, 40.0))
    assert filt.eventFilter(widget, move) is True      # still swallowed -- camera stays untouched
    assert filt._band.isVisible()                       # still the BOX feedback, not a lasso path

    release = _stub_mouse_event(QtCore.QEvent.MouseButtonRelease, (50.0, 60.0))
    assert filt.eventFilter(widget, release) is True

    # Resolved as a BOX (frozen press-time mode) -- not a lasso (the live mode at release).
    view._apply_region_pick.assert_called_once_with(("box", 20.0, 40.0, 100.0, 120.0), False)
    assert not filt._band.isVisible()      # feedback taken down
    assert filt._dragging is False          # state reset
    assert filt._mode_at_press is None


def test_region_filter_press_while_already_dragging_self_heals(qtbot):
    """Defensive stale-state self-heal: `self._dragging` should only ever be True
    between a press and its own matching release -- a SECOND press arriving while it is somehow
    still True (a missed release upstream) must not start a nested gesture on top of the stale
    one. The filter cancels the stale state instead; this press itself is swallowed and dropped,
    not reinterpreted as a fresh gesture start."""
    view, widget, filt = _region_filter(qtbot, "box")
    view._apply_region_pick = Mock()
    view._interactor = Mock(window_size=(800, 600))

    filt.eventFilter(widget, _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (10.0, 20.0)))
    assert filt._dragging is True

    stray_press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (99.0, 99.0))
    assert filt.eventFilter(widget, stray_press) is True   # swallowed, but self-heals, not nests

    assert filt._dragging is False
    assert filt._mode_at_press is None
    assert not filt._band.isVisible()
    view._apply_region_pick.assert_not_called()             # no resolution -- an aborted gesture


def test_deactivate_cancels_an_in_progress_lasso_drag(qtbot):
    """Fix round 1: Tab-away mid-lasso must not strand the top-level, click-
    through `_LassoRegionOverlay` floating over whatever tab is now showing -- undismissable,
    since `WindowTransparentForInput` means no click ever reaches it -- with stale gesture state
    behind it. `ArrangementView.deactivate()` now cancels any in-progress region gesture as part
    of parking the view, with no real interactor/camera ever built (this harness's mandated
    offscreen QPA platform, exactly the ordinary test condition)."""
    from dynamix.shell.arrangement.view import ArrangementView, _RegionSelectFilter

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    view.set_selection_mode("lasso")
    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    view._region_filter = _RegionSelectFilter(view, widget)
    view._region_filter.eventFilter(
        widget, _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (0.0, 0.0)))
    view._region_filter.eventFilter(
        widget, _stub_mouse_event(QtCore.QEvent.MouseMove, (10.0, 0.0)))
    assert view._region_filter._dragging is True

    view.deactivate()      # must not raise -- no real interactor/camera exists in this fixture

    assert view._region_filter._dragging is False
    assert view._region_filter._mode_at_press is None
    assert view._region_filter._pts == []
    assert not view._region_filter._lasso.isVisible()

    # Next press after a simulated "reactivate" starts a clean, fresh gesture.
    view.set_selection_mode("box")
    press = _stub_mouse_event(QtCore.QEvent.MouseButtonPress, (5.0, 5.0))
    assert view._region_filter.eventFilter(widget, press) is True
    assert view._region_filter._dragging is True
    assert view._region_filter._mode_at_press == "box"


def test_lasso_region_overlay_paints_without_crashing(qtbot):
    """The one genuinely NOT-offscreen-testable surface in this task (see the module docstring's
    own "Box/lasso selection" section and ``_LassoRegionOverlay``'s own docstring) is whether the
    overlay actually COMPOSITES correctly on a real display -- there is nothing under this
    harness's mandated ``QT_QPA_PLATFORM=offscreen`` to assert that against. What CAN be proven
    here, and is: the paint recipe itself does not raise. qtbot's own post-test event flush
    already exercises a real ``paintEvent`` on a shown top-level widget even under "offscreen" --
    this test does so explicitly and up front, rather than relying on that implicit teardown
    behaviour, and is exactly the regression class review caught while writing this task (a
    ``QtGui.QPointF`` typo -- ``QPointF`` lives in ``QtCore``, not ``QtGui`` -- crashed this exact
    path the first time these tests ran for real)."""
    from dynamix.shell.arrangement.view import _LassoRegionOverlay

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    widget.resize(100, 100)
    widget.show()
    overlay = _LassoRegionOverlay()
    overlay.begin(widget)
    overlay.set_points([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])

    QtWidgets.QApplication.processEvents()   # must not raise
    overlay.hide()
