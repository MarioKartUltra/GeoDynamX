# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The headless ROI runner.

read ROI + margin once (real data where the file has it, reflect-padded past its edge), run
the field stage + analyzer on that window, crop to the ROI, re-label lines inside it.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402
from dynamix.roi.runner import read_processing_window  # noqa: E402


def _tif(path, h=40, w=50, nodata=None, hole=None):
    rows, cols = np.mgrid[0:h, 0:w]
    vals = (rows * 1000 + cols).astype(np.float32)
    if hole is not None:
        vals[hole] = -9999.0
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:32615", nodata=nodata,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(vals, 1)
    return vals.astype(np.float64)


def _picture_of(path, full):
    """A display picture of ``path`` -- the runner must read NATIVE pixels off the file."""
    f = RasterField(name="pic", values=np.zeros((4, 5)), frame=LocalFrame(),
                    x_axis=np.arange(5.0), y_axis=np.arange(4.0))
    f.provenance.update({"display_stride": 10, "full_dims": full, "source": str(path),
                         "window": {"row_off": 0, "col_off": 0}})
    return f


def test_an_interior_window_is_the_file_slice(tmp_path):
    vals = _tif(tmp_path / "a.tif")
    win, info = read_processing_window(_picture_of(tmp_path / "a.tif", (40, 50)),
                                       (10, 12, 8, 9), margin=3)
    assert win.values.shape == (14, 15)
    assert np.array_equal(win.values, vals[7:21, 9:24])
    assert info["offset"] == 3 and info["reflected_edges"] == ()
    assert info["real"].all() and info["real_frac"] == 1.0
    assert win.provenance["window"] == {"row_off": 7, "col_off": 9}


def test_a_corner_roi_reflects_only_the_missing_part(tmp_path):
    """The design:
    real data wherever the file has it; only the deficit is mirrored, about the edge pixel."""
    vals = _tif(tmp_path / "a.tif")
    win, info = read_processing_window(_picture_of(tmp_path / "a.tif", (40, 50)),
                                       (0, 0, 8, 9), margin=3)
    v = win.values
    assert np.array_equal(v[3:, 3:], vals[0:11, 0:12])            # the real part
    assert np.array_equal(v[2, 3:], vals[1, 0:12])                  # mirror about row 0
    assert np.array_equal(v[0, 3:], vals[3, 0:12])
    assert np.array_equal(v[3:, 1], vals[0:11, 2])                  # mirror about col 0
    assert set(info["reflected_edges"]) == {"N", "W"}
    assert not info["real"][:3].any() and not info["real"][:, :3].any()
    assert info["real"][3:, 3:].all()


def test_the_window_carries_its_real_georeference(tmp_path):
    _tif(tmp_path / "a.tif")
    win, _ = read_processing_window(_picture_of(tmp_path / "a.tif", (40, 50)),
                                    (10, 12, 8, 9), margin=3)
    assert win.x_axis[3] == pytest.approx(500000.0 + 2.0 * (12 + 0.5))
    assert win.y_axis[3] == pytest.approx(3200000.0 - 2.0 * (10 + 0.5))
    assert win.frame.dx == pytest.approx(2.0)


def test_nodata_reaches_the_tool_as_nan(tmp_path):
    _tif(tmp_path / "n.tif", nodata=-9999.0, hole=(slice(12, 14), slice(5, 35)))
    win, _ = read_processing_window(_picture_of(tmp_path / "n.tif", (40, 50)),
                                    (10, 12, 8, 9), margin=3)
    assert np.isnan(win.values[12 - 7:14 - 7, :]).all()
    assert np.isfinite(win.values[:12 - 7]).all()


def test_a_dataset_with_no_file_is_sliced_from_memory_and_reflected_at_its_edges():
    vals = np.arange(20 * 30, dtype=float).reshape(20, 30)
    f = RasterField(name="npz", values=vals, frame=LocalFrame(dx=5.0, dy=5.0),
                    x_axis=5.0 * np.arange(30), y_axis=5.0 * np.arange(20))
    win, info = read_processing_window(f, (1, 25, 6, 5), margin=2)
    assert np.array_equal(win.values[2:8, 1:7], vals[1:7, 24:30])  # window row 0 = file row -1
    assert np.array_equal(win.values[2:8, 7], vals[1:7, 28])       # mirror about the last col
    assert np.array_equal(win.values[0, 1:7], vals[1, 24:30])      # mirror about row 0
    assert set(info["reflected_edges"]) == {"N", "E"}
    assert win.x_axis[2] == pytest.approx(5.0 * 25)


def test_a_windowed_field_maps_file_rects_through_its_window_offset():
    """A legacy crop / centred native window: its values start at (8, 12) of the file."""
    vals = np.arange(16 * 20, dtype=float).reshape(16, 20)
    f = RasterField(name="crop", values=vals, frame=LocalFrame(),
                    x_axis=np.arange(12, 32, dtype=float), y_axis=np.arange(8, 24, dtype=float))
    f.provenance["window"] = {"row_off": 8, "col_off": 12}
    win, info = read_processing_window(f, (10, 14, 4, 4), margin=1)
    assert np.array_equal(win.values, vals[1:7, 1:7])
    assert info["reflected_edges"] == ()


def test_a_whole_loaded_file_is_sliced_from_memory_even_with_a_source(tmp_path):
    """Everything needed is already in memory at native resolution: no file read (format-
    agnostic -- a netCDF grid's container path must never be re-read as band 1)."""
    vals = np.arange(20 * 30, dtype=float).reshape(20, 30)
    f = RasterField(name="whole", values=vals, frame=LocalFrame(),
                    x_axis=np.arange(30.0), y_axis=np.arange(20.0))
    f.provenance.update({"source": str(tmp_path / "does-not-exist.nc"), "full_dims": (20, 30)})
    win, _ = read_processing_window(f, (5, 5, 6, 6), margin=2)
    assert np.array_equal(win.values, vals[3:13, 3:13])


# ------------------------------------------------------------ C2: generic crop + relabel

from dynamix.roi.runner import crop_result_to_roi, split_lines  # noqa: E402


def _level(xs, ys, ids):
    xs = np.asarray(xs, dtype=np.int64)
    return {"x": xs, "y": np.asarray(ys, dtype=np.int64),
            "mod": np.arange(xs.size, dtype=float), "arg": np.zeros(xs.size),
            "line_id": np.asarray(ids, dtype=np.int64),
            "x_sub": xs + 0.25, "y_sub": np.asarray(ys, dtype=float) - 0.25}


def test_crop_keeps_points_inside_and_shifts_every_positional_channel():
    lvl = _level([1, 3, 12, 5], [3, 4, 5, 20], [0, 0, 1, -1])
    out = crop_result_to_roi({"extrema": [lvl]}, offset=2, h=10, w=9, win_shape=(14, 13))
    got = out["extrema"][0]
    assert got["x"].tolist() == [1] and got["y"].tolist() == [2]      # only (3, 4) is inside
    assert got["x_sub"].tolist() == [1.25] and got["y_sub"].tolist() == [1.75]
    assert got["mod"].tolist() == [1.0]                                # subset in step


def test_crop_slices_window_shaped_rasters_and_stacks_and_drops_draw_orderings():
    h_map = np.arange(14 * 13, dtype=float).reshape(14, 13)
    stack = np.stack([h_map, h_map + 1])
    res = {"h_map": h_map, "stack": stack, "scales": np.arange(14.0), "chains": [],
           "_xs_runs": [1], "_xs_closed": [1], "_ext_base_runs": [1], "_hline_runs": [1],
           "chain_product": object(), "_selection": {}}
    out = crop_result_to_roi(res, offset=2, h=10, w=9, win_shape=(14, 13))
    assert np.array_equal(out["h_map"], h_map[2:12, 2:11])
    assert np.array_equal(out["stack"], stack[:, 2:12, 2:11])
    assert np.array_equal(out["scales"], np.arange(14.0))              # 1-D: untouched
    for k in ("_xs_runs", "_xs_closed", "_ext_base_runs", "_hline_runs", "chain_product",
              "_selection"):
        assert k not in out


def test_a_line_that_leaves_and_reenters_the_roi_becomes_two_lines():
    # line 0 runs along row 1, climbs out of the ROI (row -1) and comes back in at x = 9
    xs = [2, 3, 4, 5, 6, 7, 7, 8, 9, 9]
    ys = [1, 1, 1, 1, 1, 1, 0, -1, 0, 1]
    lvl = {"x": np.asarray(xs) + 2, "y": np.asarray(ys) + 2, "mod": np.zeros(10),
           "arg": np.zeros(10), "line_id": np.zeros(10, dtype=np.int64)}
    out = crop_result_to_roi({"extrema": [lvl]}, offset=2, h=10, w=10, win_shape=(14, 14))
    ids = split_lines(out["extrema"][0])
    first, second = set(ids[:7].tolist()), set(ids[7:].tolist())
    assert len(first) == 1 and len(second) == 1 and first != second
    assert -1 not in first | second


def test_relabel_never_merges_and_isolates_singletons():
    lvl = {"x": np.array([0, 1, 5, 6, 9]), "y": np.array([0, 0, 0, 0, 9]),
           "line_id": np.array([0, 1, 2, 2, -1])}
    ids = split_lines(lvl)
    assert ids[0] == -1 and ids[1] == -1          # adjacent but different lines: never merged;
    assert ids[2] == ids[3] and ids[2] >= 0       # each is now a lone point
    assert ids[4] == -1                           # an isolated extremum stays isolated


# ------------------------------------------------------------ C3: run_on_region

from dynamix.roi.runner import roi_margin_of, run_on_region  # noqa: E402


def _blur(v):
    """3x3 mean, edge-replicated -- a kernel of reach 1."""
    p = np.pad(v, 1, mode="edge")
    return sum(p[1 + dy:1 + dy + v.shape[0], 1 + dx:1 + dx + v.shape[1]]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0


class _Blur:
    name = "blur"
    params = ()

    def roi_margin(self, params):
        return 1

    def compute(self, field, params, *, progress=None):
        out = _blur(np.asarray(field.values))
        ys, xs = np.nonzero(out > np.nanmean(out))
        return {"raster_out": out, "chains": [], "scales": np.array([1.0]),
                "extrema": [{"x": xs, "y": ys, "mod": out[ys, xs], "arg": 0 * out[ys, xs],
                             "line_id": np.zeros(xs.size, dtype=np.int64)}],
                "_frame": field.frame, "_shape": out.shape}


class _AddThousand:
    """A field-stage transform (field -> field), like ``noise``."""
    name = "add"
    params = ()
    field_stage = True

    def roi_margin(self, params):
        return 0

    def compute(self, field, params, *, progress=None):
        import dataclasses
        return dataclasses.replace(field, values=np.asarray(field.values) + 1000.0)


class _Undeclared:
    name = "plain"
    params = ()

    def compute(self, field, params, *, progress=None):
        return {"raster_out": np.asarray(field.values).copy(), "extrema": [], "chains": []}


class _OwnStep:
    name = "own"
    params = ()
    seen = {}

    def roi_margin(self, params):
        return 4

    def compute(self, field, params, *, progress=None):          # pragma: no cover
        raise AssertionError("compute_roi must be preferred")

    def compute_roi(self, window_field, core, params, *, info=None, progress=None):
        _OwnStep.seen.update(core=core, shape=window_field.values.shape, info=info)
        r0, c0, h, w = core
        return {"raster_out": np.asarray(window_field.values)[r0:r0 + h, c0:c0 + w],
                "extrema": [], "chains": []}


def _field(h=40, w=50):
    v = np.random.default_rng(1).standard_normal((h, w)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(dx=3.0, dy=3.0),
                       x_axis=3.0 * np.arange(w), y_axis=3.0 * np.arange(h))


def test_an_interior_roi_equals_the_same_region_of_a_whole_field_run():
    f = _field()
    res = run_on_region([(_Blur(), {})], f, (10, 12, 8, 9))
    whole = _blur(f.values)
    assert np.allclose(res["raster_out"], whole[10:18, 12:21])
    assert res["_shape"] == (8, 9)
    assert res["_roi"]["roi"] == (10, 12, 8, 9) and res["_roi"]["margin"] == 1
    assert res["_roi"]["margin_declared"] is True
    x_axis, y_axis = res["_roi_axes"]
    assert np.allclose(x_axis, 3.0 * np.arange(12, 21)) and np.allclose(y_axis, 3.0 * np.arange(10, 18))
    lvl = res["extrema"][0]
    assert lvl["x"].min() >= 0 and lvl["x"].max() < 9 and lvl["y"].max() < 8


def test_a_field_stage_runs_over_the_whole_window_before_the_analyzer():
    """Noise on an ROI layer covers the ROI and its surrounding pixels before processing."""
    f = _field()
    res = run_on_region([(_AddThousand(), {}), (_Blur(), {})], f, (10, 12, 8, 9))
    assert np.allclose(res["raster_out"], _blur(f.values + 1000.0)[10:18, 12:21])


def test_an_undeclared_plugin_runs_with_no_margin_and_says_so():
    f = _field()
    res = run_on_region([(_Undeclared(), {})], f, (10, 12, 8, 9))
    assert res["_roi"]["margin"] == 0 and res["_roi"]["margin_declared"] is False
    assert np.array_equal(res["raster_out"], f.values[10:18, 12:21])
    assert roi_margin_of(_Undeclared(), {}) == (0, False)


def test_a_tools_own_roi_step_gets_the_window_and_the_core_inside_it():
    f = _field()
    res = run_on_region([(_OwnStep(), {})], f, (10, 12, 8, 9))
    assert _OwnStep.seen["core"] == (4, 4, 8, 9)
    assert _OwnStep.seen["shape"] == (16, 17)
    assert _OwnStep.seen["info"]["offset"] == 4
    assert np.array_equal(res["raster_out"], f.values[10:18, 12:21])
    assert res["_roi"]["roi"] == (10, 12, 8, 9)
