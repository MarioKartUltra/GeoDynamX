# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""M-Z chains render through the existing display path -- no new shell code.

Regression proof: the `mz_edges` device landed without touching `dynamix/shell/`. This file proves the shell's EXISTING WTMM display path already
renders an mz_edges bundle correctly, with zero shell edits, because the bundle already carries
every key `Canvas.set_result`/`Scene.set_layers` need (`extrema`, `_shape`, `chains == []`) and
`_update_wavelet_bar`'s own `except (KeyError, ValueError)` guard already handles the unknown
`"mz_spline"` wavelet name by clearing the bar rather than raising.

Construction idioms are reused verbatim from the two authorities named in the design, NOT
from the design's Step-1 sketch, which turned out to have two bugs once checked against real
code (both noted inline, at the test that would have caught them):

- `tests/test_shell_canvas.py`: `Canvas()` + `qtbot.addWidget(canvas)`, module-level `Canvas`
  import (pyqtgraph/PySide6 are this suite's own required "gui" group, never guarded here).
- `tests/test_arrangement_scene.py`: `pytest.importorskip("pyvista")` / `("rasterio")`,
  `pv.OFF_SCREEN = True`, `Scene(plotter)` over a plain `pv.Plotter(off_screen=True)`, and the
  real `set_layers` entry shape (`"layer"`/`"field"`/`"result"`/`"status"`, not the design's `"layer_id"`/`"visible"`) -- plus that file's own cross-file reuse of
  `tests/test_geo_mapping.py`'s BOEM-like GeoTIFF fixture (`_NX = _NY = 64`, the same shape as
  `fbm64`, so an mz_edges bundle computed over `fbm64` has every extrema pixel index in bounds).

``fbm64`` (tests/conftest.py) is a bare ndarray, not a Field -- wrapped in a RasterField here via
a local `_field()` helper, the same fix `tests/test_mz_edges_device.py` made to this exact pattern (a `fbm64.values` sketch is an AttributeError waiting to happen). Duplicated rather than imported: tests/ is not a package, and
test_shell_canvas.py's own docstring already establishes duplicating a small fixture as this
suite's convention for exactly this situation.

One test beyond the design's Step-1 sketch: the design's "Produces" list names a criterion (b)
("the transport sizes to `len(result["scales"])`") the sketch's own four functions never actually
exercise -- `test_transport_sizes_to_the_result_scale_count` closes that directly, against
`dynamix.shell.transport.Transport`, reusing `tests/test_shell_transport.py`'s own construction
idiom (`Transport(n, scale_reading)` + `qtbot.addWidget`).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices
from dynamix.model.device import get_device, validate_params
from dynamix.shell.canvas import Canvas


def _field(values):
    """RasterField wrap of a bare ndarray -- see the module docstring's fixture-reuse note."""
    ny, nx = values.shape
    return RasterField(name="fbm", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(nx, dtype=np.float64),
                       y_axis=np.arange(ny, dtype=np.float64))


@pytest.fixture
def mz_result(clean_registry, fbm64):
    register_builtin_devices()
    dev = get_device("mz_edges")
    return dev.compute(_field(fbm64), validate_params(dev, {"n_levels": 3}))


# --------------------------------------------------------------------------------------- Canvas


def test_canvas_renders_mz_result(qtbot, mz_result):
    """`Canvas.set_result` on a real mz_edges bundle must draw BOTH point patterns the schema
    promises: isolated extrema (`line_id == -1`) on `extrema_item`, and every
    wrap-merged H-line run (`line_id != -1`) through the H-chain polyline path. Expected counts
    are pulled from the bundle itself, not hardcoded -- this stays a check on `set_result`'s
    OWN split (isolated -> points, grouped -> polyline) rather than a frozen pin on fbm64's
    particular statistics, while still being fully falsifiable: a `set_result` that dropped the
    isolated-point split (e.g. drew every extremum as a point, isolated or not) would fail the
    first assertion below; one that dropped the H-chain path entirely would fail the second."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(mz_result, 0)

    ext0 = mz_result["extrema"][0]
    n_iso = int((ext0["line_id"] == -1).sum())
    n_grouped = int((ext0["line_id"] != -1).sum())
    assert n_iso > 0 and n_grouped > 0          # fbm64 at n_levels=3 genuinely exercises both paths

    x, y = canvas.extrema_item.getData()
    assert x is not None and len(x) == n_iso
    assert len(y) == n_iso

    hx, hy = canvas.hchain_item.getData()
    assert hx is not None
    # NaN entries are run separators (hline_polylines), not data -- every grouped point appears
    # exactly once, finite (test_shell_canvas.py's own hline_polylines tests pin this same
    # invariant: "x.size - n_nan == 9, all 9 points present, none dropped").
    assert int(np.isfinite(hx).sum()) == n_grouped
    assert int(np.isfinite(hy).sum()) == n_grouped


def test_scale_scrub_changes_the_drawn_scale(qtbot, mz_result):
    """Falsifiable against a scrub that silently always draws scale 0: scale 0 and scale 2 of
    this bundle have DIFFERENT isolated-extrema counts (asserted below, not assumed), so a
    stuck-at-0 `set_result` would draw scale 2's request with scale 0's count -- checked directly
    (`n2 != expected0`), not merely "some count changed"."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    expected0 = int((mz_result["extrema"][0]["line_id"] == -1).sum())
    expected2 = int((mz_result["extrema"][2]["line_id"] == -1).sum())
    assert expected0 != expected2               # the fixture actually distinguishes the two scales

    canvas.set_result(mz_result, 0)
    n0 = len(canvas.extrema_item.getData()[0])
    assert n0 == expected0

    canvas.set_result(mz_result, 2)
    n2 = len(canvas.extrema_item.getData()[0])
    assert n2 == expected2
    assert n2 != expected0                      # what a stuck-at-scale-0 scrub would wrongly show


def test_transport_sizes_to_the_result_scale_count(qtbot, mz_result):
    """Acceptance criterion (b): "the transport sizes to `len(result["scales"])`".
    `dynamix/shell/main_window.py`'s own wiring (`_land_worker`) is already Transform-agnostic --
    `scales = renderable.result.get("scales"); ...; self._sync_transport(len(self._scales))` --
    and `Transport.set_n_scales`'s generic re-ranging is already exhaustively covered by
    `tests/test_shell_transport.py`, so this is not a re-test of either of those; it is the one
    fact specific to this task: that a REAL mz_edges bundle's `len(result["scales"])` (built in
    the same per-scale loop as `extrema`, `dynamix/core/mz_edges.py`) is a genuine, correct count
    that re-ranges `Transport` when handed to it exactly as `main_window.py` would.

    `before` is read off the actual widget rather than assumed to start at 0, so the final
    assertion is a real before/after comparison -- falsifiable against a `set_n_scales` that
    silently no-ops."""
    from dynamix.shell.transport import Transport

    n_scales = len(mz_result["scales"])
    assert n_scales == len(mz_result["extrema"])     # mz_edges builds both in the same per-scale loop

    transport = Transport(1, lambda idx: str(idx))   # main_window.py's own pre-first-result default
    qtbot.addWidget(transport)
    before = transport.slider.maximum()

    transport.set_n_scales(n_scales)

    assert transport.slider.maximum() == n_scales - 1
    assert transport.slider.maximum() != before      # a genuine re-range, not a silent no-op
    assert transport.clock.n_scales == n_scales


# --------------------------------------------------------------------------------- ScaleSelect


def test_scale_select_filter_composes(mz_result):
    """The design's sketch calls this with `{"scale": 1}` -- `ScaleSelect.params`
    (dynamix/devices/filters.py) names the param `scale_idx`, and `apply` reads
    `params["scale_idx"]` directly with no `validate_params` guard in between, so the design's key
    would raise `KeyError` before any assertion ran. Fixed here, and strengthened past the design's bare `out["_scale_idx"] == 1`: the `is`-identity check proves the filter selected
    the layer genuinely AT index 1 (not e.g. always index 0 while merely stamping the requested
    index), and `_scale_px` ties the composition to mz_edges' own dyadic scale ladder."""
    from dynamix.devices.filters import ScaleSelect

    out = ScaleSelect().apply(dict(mz_result), {"scale_idx": 1})

    assert out["_scale_idx"] == 1
    assert len(out["extrema"]) == 1
    assert out["extrema"][0] is mz_result["extrema"][1]        # genuinely the layer AT index 1
    assert out["_scale_px"] == float(mz_result["scales"][1])   # composes with mz_edges' own scales
    assert out["chains"] == []                                 # mz_edges' own deliberate stamp, untouched


# --------------------------------------------------------------------------------- arrangement Scene


def test_arrangement_scene_accepts_empty_chains(mz_result, tmp_path):
    """The design's sketch entry (`{"layer_id": "L1", "result": ..., "visible": True}`)
    does not match `Scene.set_layers`'s real contract: `tests/test_arrangement_scene.py`'s own
    entries key `"layer"` (a `Layer` instance), `"field"` (a GEOREFERENCED field --
    `_add_layer_geometry` needs `field_lonlat_grid` to succeed or the entry demotes to a
    "no-georeference"/"error:..." legend line instead of drawing), and `"status"`. Built here
    with that file's own BOEM-like GeoTIFF fixture -- same 64x64 grid as `fbm64`/this bundle's
    own `_shape`, so every extrema pixel index lands in bounds.

    Also strengthens the design's `pruned is not None`: `Scene.set_layers` is annotated
    `-> list[int]` and a list is never `None`, so that assertion could not fail regardless of
    behavior -- a dead gate. A genuine crash inside `_build_chain_geometry` on `chains == []`
    (e.g. an unconditional `np.concatenate` over an empty list) would instead be CAUGHT by
    `_rebuild`'s per-layer try/except and demote this layer to an "error:<msg>" legend line, not
    raise through to this test -- so the falsifiable checks are that the layer landed exactly the
    raster + extrema actors this bundle should produce, with NO chains actor and NO legend line,
    not merely that `set_layers` returned.
    """
    pv = pytest.importorskip("pyvista", reason="pyvista not installed")
    pytest.importorskip("rasterio", reason="rasterio not installed (needed by the geo fixture)")
    pv.OFF_SCREEN = True
    from dynamix.model.layer import Layer
    from dynamix.shell.arrangement.scene import Scene
    from tests.test_geo_mapping import _load_field, _write_boem_like_tif

    tif = tmp_path / "mz.tif"
    _write_boem_like_tif(tif)
    field = _load_field(tif)

    plotter = pv.Plotter(off_screen=True)
    scene = Scene(plotter)
    layer = Layer(layer_id=1, name="MZ", source_id="mem:mz")

    pruned = scene.set_layers([
        {"layer": layer, "field": field, "result": mz_result, "status": "ok"},
    ])

    assert pruned == []                                         # nothing stale on a fresh Scene
    assert scene.legend_lines == []                             # never demoted to an error line
    assert scene._layer_actors[1] == ["layer-1-raster", "layer-1-extrema"]
    assert 1 not in scene._chain_lookup                         # chains == [] drew no chains actor
    assert scene.actor_count() == 2
