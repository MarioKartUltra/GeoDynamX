# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Band buses: sends by reference, the same-grid law, and windows aligned geometrically --
identical to the ROI runner's own read, reflected margin included."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dynamix.core.bus import grid_mismatch, materialize, native_grid, sends_for_field
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


def _field(values, x0=500_000.0, y0=4_000_000.0, d=30.0, crs="EPSG:32611", name="f"):
    ny, nx = values.shape[:2]
    return RasterField(name=name, values=np.asarray(values, dtype=np.float64),
                       frame=LocalFrame(x0=x0, y0=y0, dx=d, dy=d, units="metre"),
                       x_axis=x0 + d * np.arange(nx) + d / 2,
                       y_axis=y0 - d * np.arange(ny) - d / 2,
                       units="metre", provenance={"crs": crs})


def _npz(tmp_path, name, values, **kw):
    f = _field(values, name=name, **kw)
    p = tmp_path / f"{name}.npz"
    f.save_npz(p)
    return f, str(p)


def test_sends_stack_in_order_on_the_whole_grid(tmp_path):
    rng = np.random.default_rng(0)
    a, pa = _npz(tmp_path, "a", rng.normal(size=(20, 24)))
    b, pb = _npz(tmp_path, "b", rng.normal(size=(20, 24)))
    sends = sends_for_field(b, pb, "B") + sends_for_field(a, pa, "A")
    out = materialize(a, sends)
    assert out.values.shape == (20, 24, 2)
    np.testing.assert_array_equal(out.values[..., 0], b.values)
    np.testing.assert_array_equal(out.values[..., 1], a.values)
    assert out.provenance["bands"] == ["B · band 1", "A · band 1"]
    assert out.provenance["crs"] == "EPSG:32611"
    np.testing.assert_array_equal(out.x_axis, a.x_axis)


def test_an_roi_window_matches_the_runners_own_read_reflected_margin_included(tmp_path):
    """The bus on an ROI must see exactly the pixels the ROI runner reads off the dataset --
    here the window overhangs the NW corner, so part of it is reflect-padded."""
    from dynamix.roi.runner import read_processing_window

    rng = np.random.default_rng(1)
    host, path = _npz(tmp_path, "host", rng.normal(size=(30, 40)).cumsum(0))
    window, info = read_processing_window(host, (2, 3, 10, 12), 5)
    assert set(info["reflected_edges"]) == {"N", "W"}
    bused = materialize(window, sends_for_field(host, path, "H"))
    np.testing.assert_array_equal(bused.values, window.values)


def test_a_multiband_file_sends_by_band_index(tmp_path):
    stack = np.stack([np.full((6, 7), float(k)) for k in range(3)], axis=-1)
    f, p = _npz(tmp_path, "stack", stack)
    sends = sends_for_field(f, p, "S")
    assert [s["band"] for s in sends] == [0, 1, 2]
    out = materialize(f, [sends[2], sends[0]])
    assert out.values[0, 0, 0] == 2.0 and out.values[0, 0, 1] == 0.0


def test_one_send_is_a_2d_plane_and_no_sends_is_refused(tmp_path):
    f, p = _npz(tmp_path, "one", np.ones((5, 6)))
    assert materialize(f, sends_for_field(f, p, "O")).values.shape == (5, 6)
    with pytest.raises(ValueError, match="no sends"):
        materialize(f, [])


def test_a_geotiff_send_aligns_by_coordinates_into_a_sub_window(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    v = np.arange(40 * 50, dtype="float32").reshape(40, 50)
    tif = tmp_path / "g.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=40, width=50, count=1,
                       dtype="float32", crs="EPSG:32611",
                       transform=from_origin(500_000.0, 4_000_000.0, 30.0, 30.0)) as dst:
        dst.write(v, 1)
    # a target covering file rows 10..19, cols 20..29, carrying absolute axes
    target = _field(np.zeros((10, 10)), x0=500_000.0 + 20 * 30.0,
                    y0=4_000_000.0 - 10 * 30.0)
    send = {"path": str(tif), "subdataset": None, "band": None, "label": "g"}
    out = materialize(target, [send])
    np.testing.assert_array_equal(out.values, v[10:20, 20:30])


def test_grid_mismatch_names_the_reason():
    base = native_grid(_field(np.zeros((10, 12))))
    assert grid_mismatch(base, base) is None
    assert "grid" in grid_mismatch(base, native_grid(_field(np.zeros((10, 13)))))
    assert "CRS" in grid_mismatch(base, native_grid(_field(np.zeros((10, 12)),
                                                           crs="EPSG:4326")))
    assert "step" in grid_mismatch(base, native_grid(_field(np.zeros((10, 12)), d=15.0)))
    assert "anchor" in grid_mismatch(base, native_grid(_field(np.zeros((10, 12)),
                                                              x0=600_000.0)))


def test_hdf4_container_sends_read_each_band_by_its_grid_id(tmp_path):
    pytest.importorskip("pyhdf")
    from dynamix.core.ingest import load_grid_stack
    from tests.test_ingest import _aster_like

    try:
        path = _aster_like(tmp_path)
    except Exception as exc:                          # pragma: no cover - env guard
        pytest.skip(f"pyhdf cannot create HDF4 groups here: {exc}")
    stack = load_grid_stack(path, ["VNIR_Band1/ImageData", "VNIR_Band2/ImageData"])
    sends = sends_for_field(stack, str(path), "VNIR")
    assert [s["subdataset"] for s in sends] == ["VNIR_Band1/ImageData",
                                                "VNIR_Band2/ImageData"]
    assert sends[0]["label"] == "VNIR · VNIR_Band1"
    out = materialize(stack, list(reversed(sends)))
    np.testing.assert_array_equal(out.values[..., 0], stack.values[..., 1])


def test_the_bus_device_runs_its_json_sends_and_refuses_a_result(tmp_path, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import get_device, validate_params

    register_builtin_devices()
    dev = get_device("bus")
    assert dev.field_stage and dev.roi_margin({}) == 0
    f, p = _npz(tmp_path, "d", np.full((4, 5), 7.0))
    params = validate_params(dev, {"_sends": json.dumps(sends_for_field(f, p, "D"))})
    assert np.all(dev.compute(f, params).values == 7.0)
    with pytest.raises(ValueError, match="HEAD"):
        dev.compute({"extrema": []}, params)


def test_a_tool_after_the_bus_on_an_roi_analyses_the_sends_not_the_host(tmp_path,
                                                                       clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device, validate_params
    from dynamix.roi.runner import run_on_region

    register_builtin_devices()
    rng = np.random.default_rng(3)
    host, _ph = _npz(tmp_path, "host", np.zeros((40, 40)))       # a flat host: no variance
    a, pa = _npz(tmp_path, "a", rng.normal(size=(40, 40)).cumsum(0))
    b, pb = _npz(tmp_path, "b", rng.normal(size=(40, 40)).cumsum(1))
    bus, pca = get_device("bus"), get_device("pca")
    sends = sends_for_field(a, pa, "A") + sends_for_field(b, pb, "B")
    steps = [(bus, validate_params(bus, {"_sends": json.dumps(sends)})),
             (pca, validate_params(pca, defaults_for(pca)))]
    res = run_on_region(steps, host, (5, 5, 20, 20))
    assert res["_shape"] == (20, 20)
    evr = np.asarray(res["explained_var_ratio"])
    assert evr.size == 2 and np.isfinite(evr).all()          # two real bands, not the flat host


# ------------------------------------------- live layer sends

def test_shown_plane_is_what_a_layer_shows():
    from dynamix.core.bus import shown_plane

    f = _field(np.arange(20.0).reshape(4, 5))
    assert shown_plane({}, f) is f                                    # no transforms: itself
    produced = _field(np.ones((4, 5)))
    assert shown_plane(produced, f) is produced                       # a field stage's field
    res = {"_shape": (4, 5), "raster_out": np.full((4, 5), 7.0), "h_map": np.zeros((4, 5))}
    p = shown_plane(res, f)
    assert np.all(p.values == 7.0) and np.array_equal(p.x_axis, f.x_axis)
    with pytest.raises(ValueError, match="no raster"):
        shown_plane({"extrema": [], "_shape": (4, 5)}, f)
    with pytest.raises(ValueError, match="band rows"):
        shown_plane(_field(np.zeros((4, 5, 3))), f)


def test_a_layer_send_aligns_like_a_file_send_reflected_margin_included():
    from dynamix.core.bus import register_plane
    from dynamix.roi.runner import read_processing_window

    rng = np.random.default_rng(9)
    host = _field(rng.normal(size=(30, 40)).cumsum(0))
    register_plane("stamp-A", host)
    send = {"layer": 3, "stamp": "stamp-A", "label": "recon A"}
    np.testing.assert_array_equal(materialize(host, [send]).values, host.values)
    window, _info = read_processing_window(host, (2, 3, 10, 12), 5)
    np.testing.assert_array_equal(materialize(window, [send]).values, window.values)


def test_a_layer_send_not_computed_yet_says_so():
    with pytest.raises(ValueError, match="not computed yet"):
        materialize(_field(np.zeros((4, 5))),
                    [{"layer": 1, "stamp": "never-filed", "label": "recon B"}])
