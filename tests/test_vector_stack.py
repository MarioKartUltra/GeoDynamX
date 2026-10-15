# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The 3-D scene honours the stack: rank lift, blocks with ticks (pyvista; outside the gates)."""
from __future__ import annotations

import pytest
pytestmark = pytest.mark.pyvista      # a pyvista scene: the allowed-to-fail CI job only

import numpy as np
import pyvista as pv

from dynamix.model.layer import Layer
from dynamix.shell.arrangement.scene import Scene
from test_arrangement_scene import _geo_field, _make_extrema0, _plotter, _result


def _entry(layer_id, field, result, **kw):
    return {"layer": Layer(layer_id=layer_id, name=f"L{layer_id}", source_id=f"mem:{layer_id}"),
            "field": field, "result": result, "status": "ok", **kw}


def _with_shape(result, field):
    result["_shape"] = (field.ny, field.nx)
    return result


def test_blocks_draw_one_point_per_maximum_and_one_tick_each(tmp_path):
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    res = _with_shape(_result([], _make_extrema0([2, 3, 4], [5, 5, 5])), field)
    scene.set_layers([_entry(1, field, res, extrema_style="blocks")])
    names = scene._layer_actors[1]
    (tick_name,) = [n for n in names if n.endswith(":ticks")]
    ticks = pv.wrap(scene._plotter.actors[tick_name].GetMapper().GetInput())
    assert ticks.n_lines == 3                                   # one tick per maximum
    dots = pv.wrap(scene._plotter.actors[scene._extrema_actor_name(1)].GetMapper().GetInput())
    assert dots.n_points == 3                                   # every maximum, none joined


def test_a_higher_rank_lifts_its_vectors_further(tmp_path):
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    field = _geo_field(tmp_path, "a.tif")
    res = _with_shape(_result([], _make_extrema0([2, 3], [5, 5])), field)
    scene.set_layers([_entry(1, field, res, extrema_style="blocks", rank=0),
                      _entry(2, field, dict(res), extrema_style="blocks", rank=3)])
    z = [pv.wrap(scene._plotter.actors[scene._extrema_actor_name(i)].GetMapper().GetInput())
         .points[:, 2].max() for i in (1, 2)]
    assert abs(z[1] / z[0] - 1.75) < 1e-6              # a quarter step per rank


def test_the_stack_hides_extrema_and_chains_and_fades_the_raster(tmp_path):
    from test_arrangement_scene import _make_chain

    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    res = _with_shape(_result([_make_chain(10, 10, 4)], _make_extrema0([2, 3], [5, 5])), field)
    scene.set_layers([_entry(1, field, res, show_extrema=False, show_chains=False,
                             raster_opacity=0.4)])
    names = scene._layer_actors[1]
    assert scene._extrema_actor_name(1) not in names
    assert scene._chains_actor_name(1) not in names
    raster = scene._plotter.actors[scene._mesh_actor_name(1)]
    assert abs(raster.GetProperty().GetOpacity() - 0.4) < 1e-9


def test_a_hidden_outline_row_leaves_the_roi_outline_out(tmp_path):
    scene = Scene(_plotter())
    field = _geo_field(tmp_path, "a.tif")
    res = _with_shape(_result([], _make_extrema0([2, 3], [5, 5]), roi=(1, 1, 10, 10)), field)
    scene.set_layers([_entry(1, field, res)])
    shown = len(scene._layer_actors[1])
    scene.set_layers([_entry(1, field, dict(res), show_outline=False)])
    assert len(scene._layer_actors[1]) == shown - 1


def test_a_filtered_resync_keeps_the_stacks_flags_and_lift(tmp_path):
    """The filter-only fast path rebuilds a layer's vectors in place: with the same stack flags
    and rank lift as a full rebuild."""
    scene = Scene(_plotter())
    scene.set_frame_mode(True)
    field = _geo_field(tmp_path, "a.tif")
    res = _with_shape(_result([], _make_extrema0([2, 3], [5, 5]), roi=(1, 1, 10, 10)), field)
    flags = {"extrema_style": "blocks", "rank": 3, "show_outline": False}
    scene.set_layers([_entry(1, field, res, **flags)])
    def heights():
        mapper = scene._plotter.actors[scene._extrema_actor_name(1)].GetMapper()
        mapper.Update()                       # the actor's pipeline runs on demand
        return np.array(pv.wrap(mapper.GetInput()).points[:, 2])

    z_before = heights()
    scene.set_layers([_entry(1, field, dict(res), **flags)])          # a re-filtered result
    assert not any(n.endswith("-roi") for n in scene._layer_actors[1])
    z_after = heights()
    assert z_after.size == 2 and np.allclose(z_after, z_before)
