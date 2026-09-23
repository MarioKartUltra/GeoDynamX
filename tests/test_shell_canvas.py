# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.canvas -- raster, vectorized WTMM chain overlays, LOD, scale bar.

The four module-level helpers (`hline_polylines`, `vchain_trails`, `lod_stride`,
`nice_round_scalebar`) are pure numpy and exercised directly, headless. `Canvas` is a
`pyqtgraph.GraphicsLayoutWidget` exercised offscreen via pytest-qt, smoke-style (construction
plus each public setter runs without raising), mirroring test_shell_transport.py's split between
headless-pure and offscreen-widget coverage.

Fixture note: the `_single_line_ext`/`_branching_ext` point patterns are duplicated from
tests/test_topology_wtmm.py's `_result`/`_cross_result` rather than imported -- tests/ is not a
package, and duplicating a small fixture is the established pattern here.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dynamix.shell.canvas import (
    Canvas,
    HCHAIN_COLOR,
    POINTS_COLOR,
    VTRAIL_COLOR,
    WAVELET_BAR_HEIGHT_FRAC,
    coi_outline,
    dilate_coi_mask,
    hline_polylines,
    lod_stride,
    nice_round_scalebar,
    vchain_trails,
    wavelet_bar_scale_factor,
)


# --- hline_polylines --------------------------------------------------------------------------

def _single_line_ext():
    """One 5-point H-line along y=0 (line_id 0) plus one isolated point (line_id -1) -- same
    shape as tests/test_topology_wtmm.py::_result's ext0."""
    return {
        "x": np.array([0, 1, 2, 3, 4, 9], dtype=np.int64),
        "y": np.zeros(6, dtype=np.int64),
        "mod": np.array([1.0, 9.0, 2.0, 8.0, 1.0, 5.0]),
        "arg": np.zeros(6),
        "line_id": np.array([0, 0, 0, 0, 0, -1], dtype=np.int64),
    }


def test_hline_polylines_single_line_excludes_isolated_point():
    x, y = hline_polylines(_single_line_ext(), (1, 10))
    assert x.size == 5 and y.size == 5             # isolated point (x=9) dropped; one run, no NaN
    assert not np.isnan(x).any() and not np.isnan(y).any()
    assert set(x.tolist()) == {0.0, 1.0, 2.0, 3.0, 4.0}
    assert set(y.tolist()) == {0.0}


def _branching_ext():
    """Plus-shaped single line_id: _order_lines restarts the walk at each of the 4 arms, so the
    line arrives as 3 runs -- same point pattern as tests/test_topology_wtmm.py::_cross_result."""
    pts = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (0, 2), (1, 2), (3, 2), (4, 2)]
    x = np.array([p[0] for p in pts], dtype=np.int64)
    y = np.array([p[1] for p in pts], dtype=np.int64)
    return {
        "x": x, "y": y,
        "mod": np.arange(1.0, len(pts) + 1.0),
        "arg": np.zeros(len(pts)),
        "line_id": np.zeros(len(pts), dtype=np.int64),
    }


def test_hline_polylines_branching_line_has_one_nan_break_per_run_boundary():
    x, y = hline_polylines(_branching_ext(), (5, 5))
    n_nan = int(np.isnan(x).sum())
    assert n_nan == 2                               # 3 runs -> 2 boundaries
    assert int(np.isnan(y).sum()) == 2
    assert x.size - n_nan == 9                      # all 9 points present, none dropped


def test_hline_polylines_two_distinct_lines_get_a_nan_break_between_them():
    x = np.array([0, 1, 2, 0, 1, 2, 9], dtype=np.int64)
    y = np.array([0, 0, 0, 2, 2, 2, 9], dtype=np.int64)
    line_id = np.array([0, 0, 0, 1, 1, 1, -1], dtype=np.int64)
    ext = {"x": x, "y": y, "mod": np.ones(7), "arg": np.zeros(7), "line_id": line_id}
    xo, yo = hline_polylines(ext, (10, 10))
    n_nan = int(np.isnan(xo).sum())
    assert n_nan == 1                               # two unbranched lines -> one boundary
    assert xo.size - n_nan == 6                      # isolated point (9, 9) excluded
    assert 9.0 not in xo[~np.isnan(xo)].tolist()


def test_hline_polylines_empty_ext_returns_empty_arrays():
    ext = {"x": np.zeros(0, dtype=np.int64), "y": np.zeros(0, dtype=np.int64),
           "mod": np.zeros(0), "arg": np.zeros(0), "line_id": np.zeros(0, dtype=np.int64)}
    x, y = hline_polylines(ext, (4, 4))
    assert x.size == 0 and y.size == 0


def test_hline_polylines_all_isolated_returns_empty_arrays():
    ext = {"x": np.array([1, 2], dtype=np.int64), "y": np.array([1, 2], dtype=np.int64),
           "mod": np.ones(2), "arg": np.zeros(2), "line_id": np.array([-1, -1], dtype=np.int64)}
    x, y = hline_polylines(ext, (4, 4))
    assert x.size == 0 and y.size == 0


# --- vchain_trails -----------------------------------------------------------------------------

def test_vchain_trails_two_chains_one_nan_break():
    chains = [
        {"x": np.array([1, 2, 3], dtype=np.int64), "y": np.array([10, 11, 12], dtype=np.int64),
         "mod": np.array([1.0, 2.0, 3.0])},
        {"x": np.array([5, 6, 7], dtype=np.int64), "y": np.array([50, 51, 52], dtype=np.int64),
         "mod": np.array([4.0, 5.0, 6.0])},
    ]
    x, y = vchain_trails(chains)
    assert x.size == 7 and y.size == 7
    assert int(np.isnan(x).sum()) == 1
    assert int(np.isnan(y).sum()) == 1
    finite_x = x[~np.isnan(x)]
    finite_y = y[~np.isnan(y)]
    assert finite_x.tolist() == [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
    assert finite_y.tolist() == [10.0, 11.0, 12.0, 50.0, 51.0, 52.0]


def test_vchain_trails_empty_list_returns_empty_arrays():
    x, y = vchain_trails([])
    assert x.size == 0 and y.size == 0


def test_vchain_trails_skips_empty_chains():
    chains = [
        {"x": np.zeros(0, dtype=np.int64), "y": np.zeros(0, dtype=np.int64), "mod": np.zeros(0)},
        {"x": np.array([1], dtype=np.int64), "y": np.array([2], dtype=np.int64),
         "mod": np.array([9.0])},
    ]
    x, y = vchain_trails(chains)
    assert x.tolist() == [1.0]
    assert y.tolist() == [2.0]


# --- lod_stride --------------------------------------------------------------------------------

def test_lod_stride_large_image_downsamples():
    assert lod_stride((8000, 8000)) == 4


def test_lod_stride_small_image_is_full_res():
    assert lod_stride((512, 512)) == 1


def test_lod_stride_exact_boundary_is_full_res():
    assert lod_stride((2048, 2048)) == 1


def test_lod_stride_one_over_boundary_needs_stride_two():
    assert lod_stride((2049, 2049)) == 2


def test_lod_stride_respects_custom_max_dim():
    assert lod_stride((100, 100), max_dim=32) == 4       # ceil(100 / 32) = 4


# --- nice_round_scalebar -----------------------------------------------------------------------
# Ladder = largest 1/2/5 x 10^n <= width * 0.25, hand-verified:
#   width=10.0  -> target=2.5   -> largest {1,2,5}x10^n <= 2.5   is 2    -> ("2", 2.0)
#   width=43.0  -> target=10.75 -> largest {1,2,5}x10^n <= 10.75 is 10   -> ("10", 10.0)
#   width=0.9   -> target=0.225 -> largest {1,2,5}x10^n <= 0.225 is 0.2  -> ("0.2", 0.2)

def test_nice_round_scalebar_width_10():
    assert nice_round_scalebar(10.0) == ("2", 2.0)


def test_nice_round_scalebar_width_43():
    assert nice_round_scalebar(43.0) == ("10", 10.0)


def test_nice_round_scalebar_width_0_9():
    assert nice_round_scalebar(0.9) == ("0.2", 0.2)


def test_nice_round_scalebar_nonpositive_width_falls_back():
    assert nice_round_scalebar(0.0) == ("1", 1.0)
    assert nice_round_scalebar(-5.0) == ("1", 1.0)


# --- dilate_coi_mask / coi_outline ---------------------------------------------------------------
# Hand-checkable 16x16 mask: a single True pixel at (8, 8). SQUARE (8-connectivity / Chebyshev)
# `iterations=radius` grows it into a filled (2*radius+1)-side square centered on that pixel --
# the dilation choice `dilate_coi_mask` documents and this test pins.

def test_dilate_coi_mask_radius_2_grows_a_single_pixel_into_a_5x5_square():
    mask = np.zeros((16, 16), dtype=bool)
    mask[8, 8] = True
    dilated = dilate_coi_mask(mask, 2)

    expected = np.zeros((16, 16), dtype=bool)
    expected[6:11, 6:11] = True                # Chebyshev disk of radius 2: rows/cols 6..10
    assert np.array_equal(dilated, expected)


def test_dilate_coi_mask_zero_radius_is_a_no_op():
    mask = np.zeros((16, 16), dtype=bool)
    mask[8, 8] = True
    assert np.array_equal(dilate_coi_mask(mask, 0), mask)


def test_dilate_coi_mask_grows_with_radius():
    mask = np.zeros((16, 16), dtype=bool)
    mask[8, 8] = True
    small = dilate_coi_mask(mask, 1)
    large = dilate_coi_mask(mask, 3)
    assert small.sum() < large.sum()
    assert np.all(small <= large)               # radius-1 disk is a strict subset of radius-3's


def test_coi_outline_matches_the_extrema_overlays_coordinate_convention():
    """Cross-convention check, replacing a self-referential test that pinned a transposition bug.
The old test compared `coi_outline`'s output to a FRESH call to
    `pg.functions.isocurve` using THAT function's own (row, col) axis order -- the same order the
    bug used -- so it passed identically whether or not `coi_outline`'s x/y were swapped: it was
    checking the helper against a re-derivation of itself, never against the convention the rest
    of the app actually draws in.

    `coi_outline` must instead match every OTHER overlay this module draws: x=column, y=row (see
    `hline_polylines`, `vchain_trails`, and `Canvas.set_result`'s
    `extrema_item.setData(x=ext["x"], y=ext["y"])`). A non-square mask with an off-center, non-
    square (wider-than-tall) blob makes the two conventions numerically distinguishable -- a
    square or dead-centered blob's bracket would look identical either way, which is exactly how
    the original bug (mask[8, 8] in a 16x16 array) went unnoticed.
    """
    ny, nx = 40, 90
    row0, row1 = 10, 14                     # 4 rows tall
    col0, col1 = 60, 75                     # 15 cols wide -- unmistakably wider than tall
    mask = np.zeros((ny, nx), dtype=bool)
    mask[row0:row1, col0:col1] = True

    x, y = coi_outline(mask)
    fx, fy = x[~np.isnan(x)], y[~np.isnan(y)]

    assert fx.size > 0 and fy.size > 0
    # x brackets the blob's COLUMN span, y its ROW span -- swapped, x would instead bracket the
    # ROW span (10..14) and y would bracket the COLUMN span (60..75).
    assert fx.min() <= col0 and fx.max() >= col1 - 1
    assert fy.min() <= row0 and fy.max() >= row1 - 1
    assert (fx.max() - fx.min()) > (fy.max() - fy.min())      # wider than tall, same as the mask

    # An extremum at the SAME (row, col), placed with the ordinary extrema convention (x=col,
    # y=row -- what `Canvas.set_result` hands `extrema_item.setData`), must land INSIDE that
    # bracket: the two overlays have to agree on where "this pixel" is on screen.
    ext_row, ext_col = (row0 + row1 - 1) / 2, (col0 + col1 - 1) / 2
    ext_x, ext_y = float(ext_col), float(ext_row)
    assert fx.min() <= ext_x <= fx.max()
    assert fy.min() <= ext_y <= fy.max()


def test_coi_outline_empty_mask_returns_empty_arrays():
    x, y = coi_outline(np.zeros((16, 16), dtype=bool))
    assert x.size == 0 and y.size == 0


# --- wavelet_bar_scale_factor --------------------------------------------------------------------

def test_wavelet_bar_scale_factor_scales_the_taller_curve_to_the_target_height():
    theta = np.array([0.0, 4.0, 0.0])            # peak-to-peak 4
    psi = np.array([-1.0, 1.0])                  # peak-to-peak 2
    factor = wavelet_bar_scale_factor(theta, psi, target_height=1.0)
    assert factor == pytest.approx(0.25)          # 4 * 0.25 == 1.0, the taller curve's new span
    assert (np.ptp(psi) * factor) < 1.0            # the shorter curve stays shorter, unchanged rel.


def test_wavelet_bar_scale_factor_is_a_no_op_for_a_flat_pair():
    assert wavelet_bar_scale_factor(np.zeros(4), np.zeros(4), target_height=1.0) == 1.0


def test_wavelet_bar_scale_factor_is_a_no_op_for_empty_curves():
    assert wavelet_bar_scale_factor(np.empty(0), np.empty(0), target_height=1.0) == 1.0


def test_wavelet_bar_scale_factor_is_a_no_op_for_a_nonpositive_target():
    theta = np.array([0.0, 4.0])
    psi = np.array([0.0, 2.0])
    assert wavelet_bar_scale_factor(theta, psi, target_height=0.0) == 1.0
    assert wavelet_bar_scale_factor(theta, psi, target_height=-1.0) == 1.0


# --- Canvas widget smoke tests -----------------------------------------------------------------

def _synthetic_result(shape=(64, 64)):
    """WTMM2D-result-shaped dict at the fbm64 fixture's own (64, 64) shape -- same fixture shape
    as tests/test_topology_wtmm.py::_result, shifted to fit inside a 64x64 grid."""
    ext0 = {
        "x": np.array([10, 11, 12, 13, 14, 40], dtype=np.int64),
        "y": np.full(6, 5, dtype=np.int64),
        "mod": np.array([1.0, 9.0, 2.0, 8.0, 1.0, 5.0]),
        "arg": np.zeros(6),
        "line_id": np.array([0, 0, 0, 0, 0, -1], dtype=np.int64),
    }
    chains = [
        {"x": np.array([11], dtype=np.int64), "y": np.array([5], dtype=np.int64),
         "mod": np.array([9.0])},
        {"x": np.array([13], dtype=np.int64), "y": np.array([5], dtype=np.int64),
         "mod": np.array([8.0])},
    ]
    return {"extrema": [ext0], "chains": chains, "scales": np.array([1.0]),
            "_shape": shape, "params": {"wavelet": "mexican"}}


def _empty_ext():
    """An extrema layer with no points -- COI tests only care about `_missing_mask`/`_coi_radii`,
    so the H-chain/extrema machinery `set_result` also runs gets nothing to draw."""
    return {"x": np.zeros(0, dtype=np.int64), "y": np.zeros(0, dtype=np.int64),
            "mod": np.zeros(0), "arg": np.zeros(0), "line_id": np.zeros(0, dtype=np.int64)}


def _roi_result_with_missing(shape=(16, 16), missing_at=(8, 8), radii=(2,)):
    """A `run_wtmm2d_roi`-shaped result (dynamix/roi/halo.py) carrying just enough for
    `_update_coi_outline`: one missing pixel and one radius per scale layer."""
    mask = np.zeros(shape, dtype=bool)
    mask[missing_at] = True
    n = len(radii)
    return {"extrema": [_empty_ext()] * n, "chains": [], "scales": np.arange(1.0, n + 1.0),
            "_shape": shape, "params": {}, "_missing_mask": mask, "_coi_radii": list(radii)}


# --- COI outline (Canvas) ------------------------------------------------------------------------

def test_coi_outline_lands_inside_the_roi_bounds_on_a_non_square_roi(qtbot):
    """End-to-end regression for the same transposition bug, driven through the
    REAL `Canvas.set_result` pipeline rather than `coi_outline` alone -- ``_update_coi_outline``,
    ``_update_roi_bounds`` and the extrema draw all run for real here.

    The ROI is far wider than it is tall (10 rows, 200 cols) with an asymmetric offset
    (``row_off != col_off``), and the missing-data blob sits far from BOTH axes' zero -- a
    transposed outline lands its ROW-sized extent where the wide COLUMN axis is checked (passes by
    accident, the box is generous) but its COLUMN-sized extent where the narrow ROW axis is
    checked (fails hard), which is exactly the asymmetry the fixture is built to expose. An
    isolated extremum planted at the SAME (row, col) the missing blob occupies must also land
    inside the outline it produces -- the two overlays agreeing on where "this pixel" is.
    """
    canvas = Canvas()
    qtbot.addWidget(canvas)
    h, w = 10, 200                           # non-square: far wider than tall
    row_off, col_off = 100, 300              # asymmetric offset
    blob_row, blob_col = 5, 150              # off-center on both axes
    mask = np.zeros((h, w), dtype=bool)
    mask[blob_row, blob_col] = True
    ext = {"x": np.array([blob_col], dtype=np.int64), "y": np.array([blob_row], dtype=np.int64),
          "mod": np.array([1.0]), "arg": np.array([0.0]), "line_id": np.array([-1], dtype=np.int64)}
    result = {
        "extrema": [ext], "chains": [], "scales": np.array([1.0]), "_shape": (h, w), "params": {},
        "_missing_mask": mask, "_coi_radii": [2],
        "_roi": {"source": "mem:parent", "roi": (row_off, col_off, h, w), "boundary": "auto"},
    }

    canvas.set_result(result, 0)

    x, y = canvas.coi_item.getData()
    assert x is not None and x.size > 0
    fx, fy = x[~np.isnan(x)], y[~np.isnan(y)]
    bx, by = canvas.roi_bounds_item.getData()
    assert fx.min() >= bx.min() and fx.max() <= bx.max()
    assert fy.min() >= by.min() and fy.max() <= by.max()

    ex, ey = canvas.extrema_item.getData()
    assert ex[0] >= fx.min() and ex[0] <= fx.max()
    assert ey[0] >= fy.min() and ey[0] <= fy.max()


def test_coi_outline_appears_when_both_keys_present(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(_roi_result_with_missing(), 0)

    x, y = canvas.coi_item.getData()
    assert x is not None and x.size > 0


def test_coi_outline_absent_without_missing_mask(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    result = _roi_result_with_missing()
    del result["_missing_mask"]

    canvas.set_result(result, 0)

    x, _ = canvas.coi_item.getData()
    assert x is None or x.size == 0


def test_coi_outline_absent_without_coi_radii(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    result = _roi_result_with_missing()
    del result["_coi_radii"]

    canvas.set_result(result, 0)

    x, _ = canvas.coi_item.getData()
    assert x is None or x.size == 0


def test_coi_outline_absent_for_an_ordinary_non_roi_result(qtbot, fbm64):
    """The common case: a plain (non-ROI) result never carries either key, so no outline is ever
    drawn -- exercised through the same `_synthetic_result` fixture the rest of the file uses."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(fbm64)

    canvas.set_result(_synthetic_result(fbm64.shape), 0)

    x, _ = canvas.coi_item.getData()
    assert x is None or x.size == 0


def test_coi_outline_grows_with_scale_index(qtbot):
    """Acceptance criterion 3: outline area grows
    monotonically with scale -- driven through two scale layers of the SAME result, radii 1 and
    4, so a bigger `_coi_radii` entry must produce a strictly wider drawn outline."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    result = _roi_result_with_missing(radii=(1, 4))

    canvas.set_result(result, 0)
    x0, _ = canvas.coi_item.getData()
    span0 = float(np.nanmax(x0) - np.nanmin(x0))

    canvas.set_result(result, 1)
    x1, _ = canvas.coi_item.getData()
    span1 = float(np.nanmax(x1) - np.nanmin(x1))

    assert span1 > span0


def test_coi_outline_cache_invalidates_on_a_genuinely_new_mask_array(qtbot):
    """Cache invalidation, driven twice: a SECOND, DIFFERENT `_missing_mask` array (a new layer, a
    re-run transform) at the SAME scale index must not be drawn from the first mask's cached
    dilation. If the cache were keyed on scale index alone (never checking mask identity), this
    would silently redraw result_a's outline under result_b."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    result_a = _roi_result_with_missing(missing_at=(8, 8))
    canvas.set_result(result_a, 0)
    xa, ya = canvas.coi_item.getData()

    result_b = _roi_result_with_missing(missing_at=(2, 2))     # a genuinely NEW mask array
    canvas.set_result(result_b, 0)
    xb, yb = canvas.coi_item.getData()

    assert (float(np.nanmin(xa)), float(np.nanmin(ya))) != (float(np.nanmin(xb)), float(np.nanmin(yb)))
    assert float(np.nanmin(xb)) < float(np.nanmin(xa))          # centered near (2, 2), not (8, 8)


# --- COI cache: the real pipeline's dict churn ---------------------------------
#
# `ScaleSelect.apply()` (dynamix/devices/filters.py, the filter every scale-scrub tick runs
# through) does `out = dict(result)` -- a FRESH dict, a new `id()`, on EVERY tick -- while
# `out["_missing_mask"]` is a plain shallow-copied reference to the SAME array each time. A cache
# keyed on `id(result)` invalidates on every tick and never hits in real usage; these tests drive
# that exact shallow-copy pattern and spy on both expensive stages (`dilate_coi_mask`'s scipy call
# and `coi_outline`'s pyqtgraph isocurve call) to prove the cache actually holds.

def _spy(monkeypatch, module, name):
    """Wrap `module.name` to count calls while still delegating to the real function -- a list
    (not a bare counter) so a failing assertion's `len(calls)` is easy to read at a glance."""
    calls = []
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


def test_coi_outline_cache_survives_scale_select_style_dict_churn(qtbot, monkeypatch):
    """Three ticks, three FRESH `dict(result)` wrappers sharing ONE `_missing_mask` array, same
    scale index throughout -- exactly `ScaleSelect.apply()`'s own pattern. Both the dilation and
    the isocurve computation must run exactly ONCE."""
    import dynamix.shell.canvas as canvas_mod

    dilate_calls = _spy(monkeypatch, canvas_mod, "dilate_coi_mask")
    outline_calls = _spy(monkeypatch, canvas_mod, "coi_outline")

    canvas = Canvas()
    qtbot.addWidget(canvas)

    base = _roi_result_with_missing()
    for _ in range(3):
        canvas.set_result(dict(base), 0)          # ScaleSelect.apply()'s own `out = dict(result)`

    assert len(dilate_calls) == 1
    assert len(outline_calls) == 1

    x, y = canvas.coi_item.getData()
    assert x is not None and x.size > 0             # the outline still actually got drawn


def test_coi_outline_cache_invalidates_when_the_mask_array_itself_changes(qtbot, monkeypatch):
    """Dict churn alone must NOT be mistaken for "nothing changed": once a truly NEW mask array
    arrives (a new layer, a re-run transform), the dilation must re-run."""
    import dynamix.shell.canvas as canvas_mod

    dilate_calls = _spy(monkeypatch, canvas_mod, "dilate_coi_mask")

    canvas = Canvas()
    qtbot.addWidget(canvas)

    result_a = _roi_result_with_missing(missing_at=(8, 8))
    canvas.set_result(dict(result_a), 0)
    canvas.set_result(dict(result_a), 0)              # churned dict, SAME mask array -> cache hit
    assert len(dilate_calls) == 1

    result_b = _roi_result_with_missing(missing_at=(2, 2))    # a genuinely NEW mask array
    canvas.set_result(result_b, 0)
    assert len(dilate_calls) == 2


def test_coi_outline_scale_revisit_after_a_sweep_hits_the_cache(qtbot, monkeypatch):
    """Scrubbing scale 0 -> 1 -> 0 (dict-churned at every tick, like the real chain) must not
    recompute scale 0's dilation the second time it's shown -- one dilation per DISTINCT scale
    index, not one per `set_result` call."""
    import dynamix.shell.canvas as canvas_mod

    dilate_calls = _spy(monkeypatch, canvas_mod, "dilate_coi_mask")

    canvas = Canvas()
    qtbot.addWidget(canvas)

    result = _roi_result_with_missing(radii=(1, 4))
    canvas.set_result(dict(result), 0)
    canvas.set_result(dict(result), 1)
    canvas.set_result(dict(result), 0)                # revisit

    assert len(dilate_calls) == 2                       # scale 0 and scale 1, each exactly once


def test_set_field_and_set_result_construct_without_error(qtbot, fbm64):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_field(fbm64)
    canvas.set_result(_synthetic_result(fbm64.shape), 0)

    assert canvas.image_item.image is not None
    hx, _ = canvas.hchain_item.getData()
    assert hx.size == 5                              # 5-point line, no NaN break
    assert canvas.extrema_item.data.size == 1         # one isolated extremum


def test_clear_overlays_empties_every_wtmm_item_but_leaves_the_raster(qtbot, fbm64):
    """The layer-panel hide gesture (``MainWindow._on_hide_toggled``) calls this to take the
    ACTIVE layer's overlays off screen. Every item ``set_result`` can populate goes back to empty;
    the raster itself (``image_item``, set by ``set_field``, untouched by ``clear_overlays``) is
    unaffected -- hiding a layer's WTMM overlay is not the same as hiding the raster it was drawn
    over."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_field(fbm64)
    canvas.set_result(_synthetic_result(fbm64.shape), 0)
    assert canvas.extrema_item.data.size > 0
    hx, _ = canvas.hchain_item.getData()
    assert hx.size > 0
    vx, _ = canvas.vtrail_item.getData()
    assert vx.size > 0

    canvas.clear_overlays()

    assert canvas.extrema_item.data.size == 0
    for item in (canvas.hchain_item, canvas.vtrail_item, canvas.seam_item, canvas.ghost_item,
                canvas.coi_item, canvas.roi_bounds_item):
        x, _ = item.getData()
        assert x is None or len(x) == 0
    assert canvas.image_item.image is not None        # the raster survives


def test_set_field_keeps_full_resolution_extent_under_lod_decimation(qtbot):
    """C-1: a decimated raster must still OCCUPY its full-resolution extent in data space.

    ``setImage`` alone sizes an ``ImageItem`` at one data unit per stored sample, so a stride-4
    display of a (4096, 3000) raster claimed a 750x1024 rectangle while the extrema, the chain
    polylines and the scale bar were all drawn at full-resolution pixel coordinates -- every
    overlay sat four times too far out. The rect is the registration: the image spans the
    ORIGINAL pixel grid regardless of how many samples were kept to draw it.

    ``max_dim`` is threaded through ``set_field`` so this can be forced on a cheap array instead
    of allocating a real 12-megapixel one.

    Measured through ``mapRectToView``, not ``boundingRect`` alone: pyqtgraph's ``setRect``
    applies a TRANSFORM, so ``boundingRect`` keeps reporting the stored sample count in item
    coordinates. Data space is where the overlays live, so data space is where registration has
    to be asserted.
    """
    canvas = Canvas()
    qtbot.addWidget(canvas)

    values = np.zeros((40, 30), dtype=np.float32)
    assert lod_stride(values.shape, max_dim=10) == 4          # decimation really is in play
    canvas.set_field(values, max_dim=10)

    rect = canvas.image_item.mapRectToView(canvas.image_item.boundingRect())
    # Exact blocks: each of the ceil(n/4) drawn samples covers
    # EXACTLY 4 pixels, so the rect is ceil(30/4)*4 = 32 wide (overhanging the last, partial
    # block by 2) -- the old n-wide rect squeezed samples to 3.75 px and drifted by up to one
    # displayed sample at the far edge. Still the full-resolution grid, never the sample count.
    assert (rect.width(), rect.height()) == (32.0, 40.0)
    # Center registration (edges drawn "between two pixels"): integer
    # data coordinate j is the CENTER of pixel j, so the image spans [-0.5, n-0.5].
    assert (rect.x(), rect.y()) == (-0.5, -0.5)


def test_set_field_full_resolution_extent_matches_an_undecimated_raster(qtbot):
    """The same data-space extent, whether or not a stride was applied -- one registration."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    values = np.zeros((40, 32), dtype=np.float32)          # dims a multiple of the stride:
    canvas.set_field(values)                                   # stride 1
    plain = canvas.image_item.mapRectToView(canvas.image_item.boundingRect())
    canvas.set_field(values, max_dim=10)                       # stride 4
    # exact blocks tile a multiple-of-stride grid exactly: the same data-space extent
    assert canvas.image_item.mapRectToView(canvas.image_item.boundingRect()) == plain


def _flagged_result(shape=(64, 64), show_ghosts=True):
    """`_synthetic_result` extended with the (`dynamix.devices.chain_classify`) optional
    result keys: the second of its two chains gets a truthy `tags`, and a third, separate chain
    is added under `chains_excluded` -- the three routing cases `Canvas.set_result` must split:
    plain -> `vtrail_item`, flagged -> `seam_item`, excluded -> `ghost_item`."""
    result = _synthetic_result(shape)
    result["chains"][1]["tags"] = ["seam_step"]
    result["chains_excluded"] = [
        {"x": np.array([20], dtype=np.int64), "y": np.array([30], dtype=np.int64),
         "mod": np.array([3.0])},
    ]
    result["_show_ghosts"] = show_ghosts
    return result


def test_seam_flagged_chain_renders_on_seam_item_not_vtrail(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(_flagged_result(), 0)

    sx, sy = canvas.seam_item.getData()
    assert sx.tolist() == [13.0]                  # the flagged chain's (x=13, y=5) point, only
    assert sy.tolist() == [5.0]

    vx, _ = canvas.vtrail_item.getData()
    assert vx.tolist() == [11.0]                   # the plain chain stayed on vtrail_item, alone


def test_ghost_chain_renders_on_ghost_item(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(_flagged_result(), 0)

    gx, gy = canvas.ghost_item.getData()
    assert gx.tolist() == [20.0]
    assert gy.tolist() == [30.0]


def test_show_ghosts_false_hides_ghost_item(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(_flagged_result(show_ghosts=False), 0)

    assert canvas.ghost_item.isVisible() is False


def test_show_ghosts_true_shows_ghost_item(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(_flagged_result(show_ghosts=True), 0)

    assert canvas.ghost_item.isVisible() is True


def test_seam_item_renders_even_when_trails_toggle_is_off(qtbot):
    """The point of flagging: a seam-tagged chain must stay visible independent of
    `_show_trails`, which governs only the ROUTINE (unflagged) drift trails on `vtrail_item`."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    assert canvas._show_trails is False             # default -- see test_set_show_trails_* below

    canvas.set_result(_flagged_result(), 0)

    assert canvas.seam_item.isVisible() is True
    sx, _ = canvas.seam_item.getData()
    assert sx.size > 0


def test_absent_tags_and_chains_excluded_render_bit_identically_to_plain_result(qtbot, fbm64):
    """Backward-compat guard: a result carrying none of the optional keys (`tags`,
    `chains_excluded`, `_show_ghosts` -- every result before that device existed, and every result
    from a chain a `chain_classify` device never touched) must draw `seam_item`/`ghost_item` empty
    and `vtrail_item` exactly as it did before this task."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(fbm64)

    canvas.set_result(_synthetic_result(fbm64.shape), 0)

    sx, _ = canvas.seam_item.getData()
    gx, _ = canvas.ghost_item.getData()
    assert sx is None or sx.size == 0
    assert gx is None or gx.size == 0
    vx, _ = canvas.vtrail_item.getData()
    finite_vx = vx[~np.isnan(vx)]
    assert finite_vx.tolist() == [11.0, 13.0]        # both chains, same as pre-Task-4 behavior


def test_set_show_trails_toggles_vtrail_visibility(qtbot, fbm64):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(fbm64)
    canvas.set_result(_synthetic_result(fbm64.shape), 0)
    assert canvas.vtrail_item.isVisible() is False    # off by default

    canvas.set_show_trails(True)
    assert canvas.vtrail_item.isVisible() is True

    canvas.set_show_trails(False)
    assert canvas.vtrail_item.isVisible() is False


def test_show_trails_set_before_a_result_stays_visible_once_it_lands(qtbot, fbm64):
    """The design's scenario: ``ui.show_trails=True`` (applied before a result exists yet, the
    shape a fresh layer switch takes -- ``_sync_display_controls`` runs before ``_start_worker``)
    must still show the trail once ``set_result`` actually draws one."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(fbm64)
    canvas.set_show_trails(True)

    canvas.set_result(_synthetic_result(fbm64.shape), 0)

    assert canvas.vtrail_item.isVisible() is True
    vx, _ = canvas.vtrail_item.getData()
    assert vx.size > 0


# ------------------------------------------------------------------------------------ Colors


def test_set_colormap_swaps_the_image_items_lookup_table(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    before = canvas.image_item.lut

    canvas.set_colormap("plasma")

    after = canvas.image_item.lut
    assert not np.array_equal(before, after)


def test_set_colormap_default_at_construction_is_viridis(qtbot):
    import pyqtgraph as pg

    canvas = Canvas()
    qtbot.addWidget(canvas)

    expected = pg.colormap.get("viridis").getLookupTable(nPts=256)
    assert np.array_equal(canvas.image_item.lut, expected)


def test_set_colormap_unknown_name_keeps_the_current_lut_and_does_not_raise(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_colormap("plasma")
    before = canvas.image_item.lut

    canvas.set_colormap("not-a-real-colormap-xyz")     # must not raise

    assert np.array_equal(canvas.image_item.lut, before)


def test_set_overlay_colors_rebuilds_hchain_vtrail_and_extrema_only(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    seam_before = canvas.seam_item.opts["pen"].color().name()
    ghost_before = canvas.ghost_item.opts["pen"].color().name()

    canvas.set_overlay_colors("#112233", "#445566", "#778899")

    assert canvas.hchain_item.opts["pen"].color().name() == "#112233"
    assert canvas.vtrail_item.opts["pen"].color().name() == "#445566"
    assert canvas.extrema_item.opts["brush"].color().name() == "#778899"
    # seam/ghost are flag/exclusion identity colors, never a layer preference -- untouched
    assert canvas.seam_item.opts["pen"].color().name() == seam_before
    assert canvas.ghost_item.opts["pen"].color().name() == ghost_before


def test_set_overlay_colors_preserves_the_current_line_width(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_display_style(line_width=3.0)

    canvas.set_overlay_colors("#112233", "#445566", "#778899")

    assert canvas.hchain_item.opts["pen"].widthF() == pytest.approx(3.0)
    assert canvas.vtrail_item.opts["pen"].widthF() == pytest.approx(3.0)


def test_set_overlay_colors_does_not_override_a_committed_groups_own_color(qtbot, fbm64):
    """``classify_chains``' own priority (group color wins, then seam, then plain): a per-layer
    swatch changes the PLAIN trail only -- a grouped chain keeps its own ``group_color``."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(fbm64)
    result = _synthetic_result(fbm64.shape)
    result["chains"][0]["tags"] = ["group:fault_a"]
    result["chains"][0]["group_color"] = [10, 20, 30]
    canvas.set_result(result, 0)

    canvas.set_overlay_colors("#112233", "#445566", "#778899")

    group_item = next(iter(canvas._group_items.values()))
    assert group_item.opts["pen"].color().getRgb()[:3] == (10, 20, 30)


def test_set_scale_bar_text_sets_label_and_clears(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    # `isHidden()` reflects the widget's OWN setVisible() flag; `isVisible()` also requires the
    # whole ancestor chain (including this never-`.show()`n Canvas) to be on-screen, so it would
    # read False here regardless of what set_scale_bar_text did.
    canvas.set_scale_bar_text("2 km")
    assert canvas.scale_bar_label.text() == "2 km"
    assert canvas.scale_bar_label.property("reading") == "true"
    assert canvas.scale_bar_label.isHidden() is False

    canvas.set_scale_bar_text("")
    assert canvas.scale_bar_label.isHidden() is True


def test_set_wavelet_bar_draws_and_clears(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    x_theta = np.linspace(-4, 4, 32)
    theta = np.exp(-x_theta ** 2 / 2)
    x_psi = np.linspace(-4, 4, 40)
    psi = x_psi * np.exp(-x_psi ** 2 / 2)
    canvas.set_wavelet_bar(x_theta, theta, x_psi, psi)
    tx, ty = canvas.smoother_item.getData()
    wx, wy = canvas.wavelet_item.getData()
    assert tx.size == 32 and ty.size == 32
    assert wx.size == 40 and wy.size == 40

    canvas.set_wavelet_bar(None, None, None, None)
    tx2, ty2 = canvas.smoother_item.getData()
    wx2, wy2 = canvas.wavelet_item.getData()
    assert tx2 is None or tx2.size == 0
    assert wx2 is None or wx2.size == 0


def test_set_result_skips_wavelet_bar_when_result_has_no_wavelet_params(qtbot, fbm64):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(fbm64)
    result = _synthetic_result(fbm64.shape)
    del result["params"]                              # no wavelet-transform provenance at all

    canvas.set_result(result, 0)
    tx, _ = canvas.smoother_item.getData()
    wx, _ = canvas.wavelet_item.getData()
    assert tx is None or tx.size == 0
    assert wx is None or wx.size == 0


def test_mexican_psi_span_exceeds_gaussian_psi_span_at_equal_scale(qtbot):
    """At the SAME nominal scale, the mexican psi section spans more px than the
    gaussian psi section -- `kernel_section`'s own support formula
    (+/-(outer_extremum + 3) sigma) puts mexican psi's outer extremum farther out (2.5243 sigma)
    than gaussian psi's (1.0 sigma), so ``(2.5243+3)/(1+3)`` times whichever support-floor the two
    happen to share. Pinned as the simple `mexican_span > gaussian_span` per the design,
    rather than the exact ratio."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    a = 5.0
    canvas._update_wavelet_bar({"params": {"wavelet": "gaussian"}, "scales": np.array([a])}, 0)
    gx, _ = canvas.wavelet_item.getData()
    gaussian_span = float(gx.max() - gx.min())

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([a])}, 0)
    mx, _ = canvas.wavelet_item.getData()
    mexican_span = float(mx.max() - mx.min())

    assert mexican_span > gaussian_span


def test_wavelet_bar_populates_without_wtmm_installed(qtbot, monkeypatch):
    """The bar's kernels come from `dynamix.core.scale_units`, which is pure numpy --
    `wtmm` plays no part in this path at all any more. Simulated here by making `import wtmm`
    (and `import wtmm.wavelet_scalebar`) raise, the way it would on a checkout that never
    installed the optional, private `wtmm` package: the bar must still populate,
    proving the old `except ImportError: return` fallback path is gone, not merely untested."""
    import sys
    monkeypatch.setitem(sys.modules, "wtmm", None)
    monkeypatch.delitem(sys.modules, "wtmm.wavelet_scalebar", raising=False)

    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([7.0])}, 0)

    tx, ty = canvas.smoother_item.getData()
    wx, wy = canvas.wavelet_item.getData()
    assert tx is not None and tx.size > 0 and ty is not None and ty.size > 0
    assert wx is not None and wx.size > 0 and wy is not None and wy.size > 0
    text = canvas.wavelet_label.text()
    assert "g2" in text and "g3" in text and "7.0" in text


def test_wavelet_bar_label_pins_exact_string_for_a_gaussian_result(qtbot):
    """Exact label pin: gaussian, a=10.0, no field on record -> px-only half."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas._update_wavelet_bar({"params": {"wavelet": "gaussian"}, "scales": np.array([10.0])}, 0)

    assert canvas.wavelet_label.text() == "θ g0 · ψ g1 @ a = 10.0 · σ 2.3 px · λ 14.1 px"


def test_wavelet_bar_label_pins_exact_string_for_a_mexican_result(qtbot):
    """Exact label pin: mexican, a=105.9, a 1 m/px field (physical half
    included). dx=1.0 m/px keeps sigma/lambda's physical readings numerically equal to their px
    readings, which is why the same digits appear twice in the expected string for two different
    reasons (a px count vs. a rounded metre reading)."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    field = SimpleNamespace(
        values=np.zeros((16, 16)),
        x_axis=np.arange(16.0),                    # 1 px per axis step -> 1 m/px
        frame=SimpleNamespace(units="px"),
        provenance={"crs": "metre"},
    )
    canvas.set_field(field)

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([105.9])}, 0)

    assert (canvas.wavelet_label.text()
            == "θ g2 · ψ g3 @ a = 105.9 · σ 23.8 px ≈ 23.8 m · λ 86.5 px ≈ 86.5 m")


def test_wavelet_bar_maps_wtmm2d_wavelet_choice_through_smoothing(qtbot, monkeypatch):
    """`_update_wavelet_bar` must resolve WTMM2D's OWN `wavelet` choice ("mexican", its default)
    to the matching kernels, not always draw `kernel_section`'s bare defaults. Checked here for
    psi (deriv_order=1); theta's own render is pinned by
    `test_wavelet_duo_theta_curve_matches_deriv_order_0_render` below.

    Captured via a spy on `set_wavelet_bar` rather than reading back `wavelet_item`'s stored
    data: `set_wavelet_bar` corner-anchors and amplitude-normalizes its input with display-only
    math, so comparing what `_update_wavelet_bar` PASSES it is the exact, unmodified check;
    comparing the post-anchor, post-normalization stored curve would need to reproduce that math
    in the test too.

    `kernel_section` already samples over its own support -- there is no separate clip
    step any more (see `test_wavelet_bar_draws_kernel_sections_own_support_no_separate_clip`), so
    the comparison is against its bare output.
    """
    from dynamix.core.scale_units import kernel_section

    canvas = Canvas()
    qtbot.addWidget(canvas)

    captured = {}
    original = canvas.set_wavelet_bar

    def spy(x_theta, theta, x_psi, psi, label=""):
        captured["x_psi"], captured["psi"], captured["label"] = x_psi, psi, label
        original(x_theta, theta, x_psi, psi, label)

    monkeypatch.setattr(canvas, "set_wavelet_bar", spy)

    scale = 4.0
    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([scale])}, 0)

    assert captured, "set_wavelet_bar was never called"
    expected_x, expected_psi = kernel_section(scale, "mexican", 1)
    assert np.allclose(captured["x_psi"], expected_x)
    assert np.allclose(captured["psi"], expected_psi)


def test_wavelet_bar_follows_the_selected_scale_not_the_layer_index(qtbot, monkeypatch):
    """The footprint must show the scale being LOOKED AT, not scale 0 forever.

    ``ScaleSelect`` hands on a one-layer ``extrema`` list, so every caller passes ``scale_idx=0``
    while stamping the absolute index it picked as ``_scale_idx``. Indexing ``scales`` with the
    layer index therefore pins the kernel to the finest scale even as the extrema underneath it
    change with every scrub -- an overlay quietly contradicting the picture it sits on.
    """
    from dynamix.core.scale_units import kernel_section

    canvas = Canvas()
    qtbot.addWidget(canvas)

    captured = {}
    monkeypatch.setattr(
        canvas, "set_wavelet_bar",
        lambda x_theta, theta, x_psi, psi, label="": captured.update(
            x_psi=x_psi, psi=psi, label=label))

    canvas._update_wavelet_bar(
        {"params": {"wavelet": "mexican"}, "scales": np.array([2.0, 4.0, 8.0]),
         "_scale_idx": 2}, 0)          # layer index 0, scale index 2

    expected_x, expected_psi = kernel_section(8.0, "mexican", 1)
    assert np.allclose(captured["x_psi"], expected_x)
    assert np.allclose(captured["psi"], expected_psi)
    assert "8.0" in captured["label"]              # the label follows the same scale


def test_wavelet_bar_draws_kernel_sections_own_support_no_separate_clip(qtbot):
    """I-3, SUPERSEDED: the bar used to clip a `wtmm`-rendered kernel -- numeric support
    ~20a, a squiggle spanning most of a zoomed view -- down to a readable +/-3a window.
    `kernel_section` is already sampled over its OWN support
    (+/-(outer_extremum + 3) sigma, at least +/-4 sigma), comfortably inside that old ~20a, so
    there is no separate clip step any more: the drawn extent must match `kernel_section`'s own
    output exactly.

    Only the X span is checked: WIDTH stays true data px (unaffected by the amplitude
    normalization `_draw_wavelet` now applies), which is exactly what this asserts -- a pure
    x-shift (the corner anchor) never changes `max() - min()`.
    """
    from dynamix.core.scale_units import kernel_section

    canvas = Canvas()
    qtbot.addWidget(canvas)

    a = 7.0
    expected_x, _ = kernel_section(a, "mexican", 1)
    assert 0 < (expected_x.max() - expected_x.min()) < 20.0 * a   # well inside the old ~20a span

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([a])}, 0)

    wx, _ = canvas.wavelet_item.getData()
    assert wx is not None and wx.size > 8                   # still a curve, not a stub
    assert np.allclose(wx.max() - wx.min(), expected_x.max() - expected_x.min())


def test_wavelet_duo_theta_curve_matches_deriv_order_0_render(qtbot):
    """The theta curve drawn on `smoother_item` must be `kernel_section(..., 0)`'s shape -- the
    SAME curve up to the shared amplitude-normalization factor and the corner-anchor translation
    `_draw_wavelet` applies (both display-only). Comparing NORMALIZED (zero-mean, unit
    peak-to-peak) curves sidesteps having to know that factor or the anchor point:
    `(arr - mean) / ptp` is invariant under any `arr -> arr*c + d` with `c > 0`, which is exactly
    what the draw-time transform is."""
    from dynamix.core.scale_units import kernel_section

    canvas = Canvas()
    qtbot.addWidget(canvas)

    a = 5.0
    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([a])}, 0)
    _, drawn_theta = canvas.smoother_item.getData()

    expected_x, expected_theta = kernel_section(a, "mexican", 0)

    def normalize(arr):
        arr = np.asarray(arr, dtype=np.float64)
        span = np.ptp(arr)
        return (arr - arr.mean()) / span if span > 0 else arr - arr.mean()

    assert drawn_theta.size == expected_theta.size
    assert np.allclose(normalize(drawn_theta), normalize(expected_theta), atol=1e-9)


def test_wavelet_duo_normalizes_taller_curve_to_the_bar_height_fraction(qtbot):
    """The taller curve's ON-SCREEN height is
    normalized to `WAVELET_BAR_HEIGHT_FRAC` of the current view's y-range -- checked here against
    an explicit, padding-free `setRange` so the view's y-extent is exactly known."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.view.setRange(xRange=(0, 100), yRange=(0, 300), padding=0)

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([4.0])}, 0)

    _, theta = canvas.smoother_item.getData()
    _, psi = canvas.wavelet_item.getData()
    theta_span = float(np.ptp(theta)) if theta is not None and len(theta) else 0.0
    psi_span = float(np.ptp(psi)) if psi is not None and len(psi) else 0.0
    taller = max(theta_span, psi_span)

    (x0, x1), (y0, y1) = canvas.view.viewRange()
    assert taller == pytest.approx(WAVELET_BAR_HEIGHT_FRAC * (y1 - y0), rel=1e-6)


def test_wavelet_bar_label_names_the_resolved_wavelet_and_scale(qtbot):
    """A curve with no label is not a reading. It must say WHICH wavelets (the resolved registry
    names -- "g2" for theta, "g3" for psi, WTMM2D's "mexican" default) at WHICH scale, in the
    theme's mono reading font like every other reading in the shell."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([7.0])}, 0)

    text = canvas.wavelet_label.text()
    assert "g2" in text and "g3" in text and "7.0" in text
    assert canvas.wavelet_label.property("reading") == "true"
    assert canvas.wavelet_label.isHidden() is False


def test_wavelet_duo_label_has_metres_for_a_metre_crs_field(qtbot):
    """The physical half (`≈ {phys}`) goes through `dynamix.shell.units.px_to_metres` on whatever
    field `set_field` last retained. A field with a bare "metre" CRS unit converts 1:1, so the
    label must end in " m"."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    field = SimpleNamespace(
        values=np.zeros((16, 16)),
        x_axis=np.arange(16.0),                    # 1 px per axis step
        frame=SimpleNamespace(units="px"),
        provenance={"crs": "metre"},                # linear_unit_to_m's bare-name shortcut
    )
    canvas.set_field(field)

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([7.0])}, 0)

    text = canvas.wavelet_label.text()
    assert "g2" in text and "g3" in text
    assert "≈" in text and text.rstrip().endswith("m")


def test_wavelet_duo_label_precision_matches_the_task_6_pinned_format(qtbot):
    """Supersedes the earlier `.4g` pin: the design's label format spec fixes sigma's
    and lambda's physical readings at `.3g` (`{sigma * m_per_px:.3g}`, `{lam * m_per_px:.3g}`),
    matching the illustrative `σ 23.8 px ≈ 953 m` / `λ 86.5 px ≈ 3.46 km` examples in the design,
    not the shell's usual `.4g` (`_scale_reading`/`_update_scale_bar`'s scale bar,
    `RoiPanel._physical_text`). A non-round conversion factor (1.23456 m/px, unlike the plain 1:1
    fixtures elsewhere in this file) makes the two precisions numerically distinguishable, for
    both sigma and lambda independently."""
    from dynamix.core.scale_units import lambda_peak_px, sigma_px

    canvas = Canvas()
    qtbot.addWidget(canvas)
    field = SimpleNamespace(
        values=np.zeros((16, 16)),
        x_axis=1.23456 * np.arange(16.0),          # dx = 1.23456 m/px, deliberately non-round
        frame=SimpleNamespace(units="px"),
        provenance={"crs": "metre"},
    )
    canvas.set_field(field)

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([7.0])}, 0)

    text = canvas.wavelet_label.text()
    sigma_phys = sigma_px(7.0) * 1.23456
    lam_phys = lambda_peak_px(7.0, "mexican", 1) * 1.23456
    assert f"{sigma_phys:.3g} m" in text
    assert f"{lam_phys:.3g} m" in text
    assert f"{sigma_phys:.4g} m" not in text
    assert f"{lam_phys:.4g} m" not in text


def test_wavelet_duo_label_omits_physical_half_for_a_unitless_field(qtbot):
    """No field on record (`px_to_metres(None)` returns a `None` factor) -- omit `≈ {phys}`
    rather than fabricate a physical number, exactly as `dynamix.shell.units.px_to_metres`'s own
    contract requires of every caller."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas._update_wavelet_bar({"params": {"wavelet": "mexican"}, "scales": np.array([7.0])}, 0)

    text = canvas.wavelet_label.text()
    assert "g2" in text and "g3" in text and "7.0" in text
    assert "≈" not in text


def test_wavelet_label_clears_with_the_curve(qtbot, fbm64):
    """A non-wavelet layer draws no curve, so it must show no label either -- a stale "psi g3"
    over a layer that never went through a wavelet transform is exactly the misleading overlay
    the semantic guard exists to prevent."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(fbm64)
    result = _synthetic_result(fbm64.shape)
    del result["params"]

    canvas.set_result(result, 0)
    assert canvas.wavelet_label.text() == ""
    assert canvas.wavelet_label.isHidden() is True


def test_wavelet_and_scale_bar_pens_differ(qtbot):
    """Same pen as the distance bar was half of why the footprint read as an artifact. The
    footprint is an annotation about the analysis; the distance bar is a full-strength reading."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    assert canvas.wavelet_item.opts["pen"].color() != canvas.scale_bar_line.opts["pen"].color()
    assert canvas.smoother_item.opts["pen"].color() != canvas.scale_bar_line.opts["pen"].color()
    assert canvas.smoother_item.opts["pen"].color() != canvas.wavelet_item.opts["pen"].color()


def test_wavelet_bar_skips_when_wavelet_choice_is_unmappable(qtbot):
    """"morlet" is not a WTMM2D `wavelet` choice (those are "mexican"/"gaussian", both of which
    resolve) -- it stands in here for any name `dynamix.core.scale_units`' kernel registry doesn't
    know, which `kernel_section`/`analyzing_names`/`lambda_peak_px` reject with `KeyError`/
    `ValueError`. The bar must be cleared, not raise through to the caller or fall back to a
    misleading default kernel."""
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas._update_wavelet_bar({"params": {"wavelet": "morlet"}, "scales": np.array([4.0])}, 0)

    tx, _ = canvas.smoother_item.getData()
    wx, _ = canvas.wavelet_item.getData()
    assert tx is None or tx.size == 0
    assert wx is None or wx.size == 0


# --- points overlay --------------------------------------------


def test_points_color_default_is_a_distinct_cool_hue():
    """Not amber (VTRAIL_COLOR/the ROI band's One-Accent exception), not the extrema/seam grays --
    see POINTS_COLOR's own module comment."""
    assert POINTS_COLOR == (0x4F, 0xC3, 0xF7)
    assert POINTS_COLOR != VTRAIL_COLOR
    assert POINTS_COLOR != HCHAIN_COLOR


def test_points_item_constructed_empty_in_the_default_color(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    assert canvas.points_item.data.size == 0
    assert canvas.points_item.opts["brush"].color().getRgb()[:3] == POINTS_COLOR


def test_set_points_result_feeds_the_scatter_item_with_the_stamped_arrays(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    result = {"points_px": {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])},
              "inside": np.array([True, True, False])}

    canvas.set_points_result(result)

    x, y = canvas.points_item.getData()
    assert x.tolist() == [1.0, 2.0, 3.0]
    assert y.tolist() == [4.0, 5.0, 6.0]


def test_set_points_style_sets_brush_and_size(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_points_style("#112233", 7.0)

    assert canvas.points_item.opts["brush"].color().name() == "#112233"
    assert canvas.points_item.opts["size"] == pytest.approx(7.0)


def test_clear_points_empties_the_scatter_item_and_leaves_other_overlays_alone(qtbot, fbm64):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(fbm64)
    canvas.set_result(_synthetic_result(fbm64.shape), 0)
    canvas.set_points_result({"points_px": {"x": np.array([1.0]), "y": np.array([2.0])},
                              "inside": np.array([True])})
    assert canvas.points_item.data.size == 1
    hx_before, _ = canvas.hchain_item.getData()

    canvas.clear_points()

    assert canvas.points_item.data.size == 0
    hx_after, _ = canvas.hchain_item.getData()
    np.testing.assert_array_equal(hx_before, hx_after)   # untouched by clear_points


def test_clear_field_also_clears_points(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((8, 8)))
    canvas.set_points_result({"points_px": {"x": np.array([1.0]), "y": np.array([2.0])},
                              "inside": np.array([True])})

    canvas.clear_field()

    assert canvas.points_item.data.size == 0


def test_canvas_module_does_not_import_wtmm_at_top_level():
    """The wavelet-bar math is a LAZY import inside a method -- importing dynamix.shell.canvas
    must not pull in wtmm/wtmm_ebsd, so the shell stays importable on a checkout without them."""
    import dynamix.shell.canvas as canvas_mod

    assert "wtmm" not in canvas_mod.__dict__


def test_geographic_south_first_field_displays_north_up(qtbot):
    """2026-08-19 fix: a GEOGRAPHIC field stored south-first (ascending latitude axis -- the
    demo DEM) un-inverts the ViewBox so north draws at the top, matching the Vector tab; every
    other field keeps image convention (row 0 top). The ViewBox flip carries image AND overlays
    together, so registration needs no separate assertion -- the single inversion flag IS the
    behavior."""
    from dynamix.core.frames import GeographicFrame, LocalFrame
    from dynamix.core.rasterfield import RasterField

    canvas = Canvas()
    qtbot.addWidget(canvas)
    assert canvas.view.state["yInverted"] is True          # construction default: image convention

    south_first = RasterField._from_bare_array(np.zeros((8, 8)), "geo", name="geo",
                                               frame=GeographicFrame())
    south_first.y_axis = np.linspace(-23.0, 19.0, 8)       # ascending latitude = south-first rows
    canvas.set_field(south_first)
    assert canvas.view.state["yInverted"] is False          # north-up

    north_first = RasterField._from_bare_array(np.zeros((8, 8)), "geo2", name="geo2",
                                               frame=GeographicFrame())
    north_first.y_axis = np.linspace(19.0, -23.0, 8)        # descending = already north-up
    canvas.set_field(north_first)
    assert canvas.view.state["yInverted"] is True

    pixel = RasterField._from_bare_array(np.zeros((8, 8)), "px", name="px",
                                         frame=LocalFrame(units="px"))
    canvas.set_field(pixel)
    assert canvas.view.state["yInverted"] is True           # EBSD/pixel maps: image convention


# --------------------------- filter tweaks mask cached H-line geometry (2026-08-30, "laggy")

def _big_lines_result(keep=None):
    import numpy as np
    x = np.arange(12, dtype=np.int64); y = np.repeat(np.arange(3, dtype=np.int64) * 2, 4)
    x[4:8] = np.arange(4); x[8:] = np.arange(4)
    lid = np.repeat(np.arange(3, dtype=np.int64), 4)
    base = {"x": x, "y": y, "mod": np.linspace(1, 0.1, 12), "arg": np.zeros(12), "line_id": lid}
    ext = base if keep is None else {k: (v[keep] if hasattr(v, "shape") else v) for k, v in base.items()}
    return {"extrema": [ext], "_shape": (8, 16), "_ext_base": base, "chains": []}, base


def test_a_filtered_result_masks_the_cached_base_geometry_instead_of_rewalking(qtbot):
    import numpy as np
    from dynamix.shell import canvas as canvas_mod
    from dynamix.shell.canvas import Canvas
    c = Canvas(); qtbot.addWidget(c)
    c.set_field(np.zeros((8, 16)))
    full, base = _big_lines_result()
    calls = []
    real = canvas_mod.hline_polylines
    canvas_mod.hline_polylines = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        c.set_result(full, 0)
        n_walks = len(calls)
        keep = np.ones(12, bool); keep[1] = False              # drop one mid-line point
        filtered, _ = _big_lines_result(keep)
        filtered["_ext_base"] = base                            # same identity: the cached stack
        c.set_result(filtered, 0)
        assert len(calls) == n_walks                            # NO re-walk on a filter tweak
    finally:
        canvas_mod.hline_polylines = real
    hx, hy = c.hchain_item.getData()
    fin = np.isfinite(hx) & np.isfinite(hy)
    drawn = {(int(xx), int(yy)) for xx, yy in zip(hx[fin], hy[fin])}
    assert (1, 0) not in drawn                                  # the dropped vertex is masked out
    assert (0, 0) in drawn and (2, 0) in drawn                  # its neighbours still draw


def test_stamped_runs_spare_the_canvas_the_landing_walk(qtbot):
    """With _ext_base_runs stamped by the transform, the canvas builds its base geometry from
    them -- hline_polylines (the main-thread _order_lines walk) is never called (2026-08-30)."""
    import numpy as np
    from dynamix.core.hlines import hline_runs
    from dynamix.shell import canvas as canvas_mod
    from dynamix.shell.canvas import Canvas
    c = Canvas(); qtbot.addWidget(c)
    c.set_field(np.zeros((8, 16)))
    full, base = _big_lines_result()
    full["_ext_base_runs"] = hline_runs(base, (8, 16))
    calls = []
    real = canvas_mod.hline_polylines
    canvas_mod.hline_polylines = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        c.set_result(full, 0)
        assert calls == []                                        # no walk at landing
    finally:
        canvas_mod.hline_polylines = real
    hx, hy = c.hchain_item.getData()
    fin = np.isfinite(hx) & np.isfinite(hy)
    assert {(int(a), int(b)) for a, b in zip(hx[fin], hy[fin])} == \
        {(int(a), int(b)) for a, b in zip(base["x"], base["y"])}  # same vertices drawn


# --- selection fast path ----------------------------------------

def _prod_result():
    """A product-stamped result: scale 0 has a 3-pt line, a 4-pt line and a singleton;
    scale 1 one 3-pt line. Three chains, one only 1 scale deep."""
    from dynamix.core.chain_product import attach_chain_product

    l0 = {"x": np.arange(8, dtype=np.int64), "y": np.zeros(8, dtype=np.int64),
          "mod": np.linspace(1.0, 0.3, 8), "arg": np.zeros(8),
          "line_id": np.array([0, 0, 0, 1, 1, 1, 1, -1], dtype=np.int64)}
    l1 = {"x": np.arange(3, dtype=np.int64), "y": np.full(3, 2, dtype=np.int64),
          "mod": np.array([0.7, 0.6, 0.5]), "arg": np.zeros(3),
          "line_id": np.zeros(3, dtype=np.int64)}

    def chain(xs, ys, mods, scales):
        mod = np.asarray(mods, float)
        with np.errstate(divide="ignore"):
            lm = np.log2(np.abs(mod))
        return {"x": np.asarray(xs, np.int64), "y": np.asarray(ys, np.int64), "mod": mod,
                "log2_mod": lm, "log2_scales": np.log2(np.asarray(scales, float))[:mod.size]}

    chains = [chain([0, 0], [0, 2], [1.0, 0.7], [1.0, 2.0]),
              chain([3, 1], [0, 2], [0.9, 0.6], [1.0, 2.0]),
              chain([5], [0], [0.8], [1.0, 2.0])]
    return attach_chain_product({"extrema": [l0, l1], "chains": chains,
                                 "scales": np.asarray([1.0, 2.0]), "_shape": (8, 64)})


@pytest.mark.skip(reason="selection mechanism disabled 2026-09-14 pending progressive-compute redesign")
def test_selection_fast_path_draws_the_selected_h_chains(qtbot):
    from dynamix.core.chain_product import materialize_selection
    from dynamix.devices.filters import HLineLength, ScaleSelect

    res = _prod_result()
    out = ScaleSelect().apply(res, {"scale_idx": 0})
    out = HLineLength().apply(out, {"min_len": 4, "max_len": 0})   # keeps only the 4-pt line
    out = materialize_selection(out)

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((8, 64)))
    canvas.set_result(out, 0)

    hx, hy = canvas.hchain_item.getData()
    finite = np.isfinite(hx)
    assert set(hx[finite].tolist()) == {3.0, 4.0, 5.0, 6.0}
    assert set(np.asarray(hy)[finite].tolist()) == {0.0}
    assert canvas.extrema_item.data.size == 1          # the singleton survives as a dot


@pytest.mark.skip(reason="selection mechanism disabled 2026-09-14 pending progressive-compute redesign")
def test_selection_fast_path_caps_v_trails_and_says_so(qtbot):
    from dynamix.core.chain_product import materialize_selection
    from dynamix.devices.chain_filters import ChainLengthFilter

    res = _prod_result()
    out = materialize_selection(ChainLengthFilter().apply(res, {"min_len": 2}))

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((8, 64)))
    canvas.draw_cap = 1
    canvas.set_result(out, 0)

    vx, _vy = canvas.vtrail_item.getData()
    assert np.isfinite(vx).sum() == 2                  # ONE 2-point chain drawn, not two
    assert canvas.cap_note is not None and "1 of 2" in canvas.cap_note

    canvas.draw_cap = 6000
    canvas.set_result(out, 0)
    assert np.isfinite(canvas.vtrail_item.getData()[0]).sum() == 4
    assert canvas.cap_note is None


# --- perf-critical geometry correctness (2026-09-14) -------------------------------------------

def test_hline_base_geometry_vectorized_matches_the_loop():
    """The vectorized run scatter must be byte-identical to the old per-run list.extend loop --
    it is the scale-scrub hot path (~70 ms -> vectorized), correctness cannot drift."""
    from dynamix.shell.canvas import _hline_base_geometry
    rng = np.random.default_rng(1)
    n = 60
    base = {"x": rng.integers(0, 40, n).astype(np.int64),
            "y": rng.integers(0, 40, n).astype(np.int64)}
    runs = [np.array([0, 1, 2, 3]), np.array([7, 8]), np.array([12, 13, 14]), np.array([20, 21])]

    def loop_ref(base, shape, runs):
        bx = np.asarray(base["x"], float); by = np.asarray(base["y"], float)
        xs, ys = [], []; nan = np.array([np.nan])
        for r in runs:
            xs.extend((bx[r], nan)); ys.extend((by[r], nan))
        return np.concatenate(xs[:-1]), np.concatenate(ys[:-1])

    hx, hy, vert = _hline_base_geometry(base, (40, 40), runs)
    rx, ry = loop_ref(base, (40, 40), runs)
    np.testing.assert_array_equal(np.nan_to_num(hx, nan=-1), np.nan_to_num(rx, nan=-1))
    np.testing.assert_array_equal(np.nan_to_num(hy, nan=-1), np.nan_to_num(ry, nan=-1))
    # vert_pos is row*nx+col on finite vertices, -1 on separators
    fin = np.isfinite(hx)
    assert (vert[~fin] == -1).all()
    np.testing.assert_array_equal(vert[fin], (hy[fin].astype(np.int64) * 40 + hx[fin].astype(np.int64)))


def test_cap_polylines_keeps_the_longest_and_reports_totals():
    from dynamix.shell.canvas import cap_polylines
    nan = np.nan
    # three polylines of length 2, 4, 1 separated by NaN
    x = np.array([0., 1, nan, 5, 6, 7, 8, nan, 9])
    y = x.copy()
    cx, cy, kept, total = cap_polylines(x, y, 2)     # keep the 2 longest (len 4 and 2)
    assert (kept, total) == (2, 3)
    finite = np.isfinite(cx)
    assert set(cx[finite].tolist()) == {0., 1., 5., 6., 7., 8.}   # the length-1 line dropped

    same_x, same_y, k, t = cap_polylines(x, y, 10)   # under the cap -> untouched, no copy
    assert (k, t) == (3, 3) and same_x is x


# ------------------------------------------------------- subpixel display (2026-09-20 knob)


def test_hline_polylines_prefers_the_subpixel_channels():
    """Extrema carrying x_sub/y_sub (the interpolate knob) draw at the FLOAT positions; the
    ordering walk itself still runs on the integer support (grid adjacency orders a line)."""
    ext = _single_line_ext()
    ext = dict(ext, x_sub=ext["x"] + 0.3, y_sub=ext["y"] - 0.2)
    x, y = hline_polylines(ext, (1, 10))
    assert x.size == 5                                     # same walk, same run
    np.testing.assert_allclose(sorted(x.tolist()), [0.3, 1.3, 2.3, 3.3, 4.3])
    np.testing.assert_allclose(y, -0.2)


def test_hline_base_geometry_runs_path_draws_float_but_keys_integer():
    """The runs path: drawn vertices move to the subpixel positions, while ``vert_pos`` -- the
    pixel identity the filter-membership gather keys on -- stays derived from the integer
    support (a rint over a half-pixel offset would be ambiguous at ties)."""
    from dynamix.shell.canvas import _hline_base_geometry
    rng = np.random.default_rng(5)
    n = 20
    base = {"x": rng.integers(1, 39, n).astype(np.int64),
            "y": rng.integers(1, 39, n).astype(np.int64)}
    base["x_sub"] = base["x"] + rng.uniform(-0.5, 0.5, n)
    base["y_sub"] = base["y"] + rng.uniform(-0.5, 0.5, n)
    runs = [np.array([0, 1, 2]), np.array([5, 6])]

    hx, hy, vert = _hline_base_geometry(base, (40, 40), runs)
    fin = np.isfinite(hx)
    order = np.concatenate(runs)
    np.testing.assert_array_equal(hx[fin], base["x_sub"][order])
    np.testing.assert_array_equal(hy[fin], base["y_sub"][order])
    np.testing.assert_array_equal(vert[fin], base["y"][order] * 40 + base["x"][order])


def test_hline_base_geometry_fallback_path_draws_float_but_keys_integer():
    """Same guarantees on the no-runs fallback (the walk happens here): float vertices via the
    pixel-position lookup, integer vert_pos identity."""
    from dynamix.shell.canvas import _hline_base_geometry
    ext = _single_line_ext()
    ext = dict(ext, x_sub=ext["x"] + 0.25, y_sub=ext["y"] + 0.1)
    hx, hy, vert = _hline_base_geometry(ext, (1, 10), None)
    fin = np.isfinite(hx)
    assert fin.sum() == 5                                   # isolated point excluded, as ever
    np.testing.assert_allclose(sorted(hx[fin].tolist()), [0.25, 1.25, 2.25, 3.25, 4.25])
    np.testing.assert_allclose(hy[fin], 0.1)
    np.testing.assert_array_equal(np.sort(vert[fin]), np.array([0, 1, 2, 3, 4]))
