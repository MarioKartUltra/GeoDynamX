# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Sends for inpaint and fill: the bus's send form with a role, each read on its own grid
over the window that covers the target."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("pywt")
rasterio = pytest.importorskip("rasterio")

from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.inpaint import sends as sn  # noqa: E402
from dynamix.core.inpaint.grid import georef, relate  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402

UTM = "EPSG:32615"


def _tif(path, values, x0, y0, px, crs=UTM):
    from rasterio.transform import Affine

    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs=crs, transform=Affine(px, 0, x0, 0, -px, y0),
                       nodata=-9999.0) as dst:
        dst.write(values.astype(np.float32), 1)
    return str(path)


def _target(values, x0=500000.0, y0=4000000.0, px=30.0, name="t", **prov):
    ny, nx = values.shape
    return RasterField(name=name, values=values, frame=LocalFrame(x0=x0 + px / 2, y0=y0 - px / 2,
                                                                   dx=px, dy=px, units="m"),
                       x_axis=x0 + px * (np.arange(nx) + 0.5), y_axis=y0 - px * (np.arange(ny) + 0.5),
                       provenance={"crs": UTM, "source": f"/{name}.tif", **prov})


def _send(path, role="value", label=None, **extra):
    return {"path": path, "subdataset": None, "band": None, "label": label or role, "role": role,
            **extra}


def test_parse_roles_validates_roles_and_the_weights_binding():
    text = json.dumps([_send("/a.tif"), _send("/b.tif", "weights", "w", **{"for": "value"}),
                       {"path": "/c.tif", "subdataset": None, "band": None, "label": "c"}])
    sends = sn.parse_roles(text)
    assert [s["role"] for s in sends] == ["value", "weights", "value"]
    assert sends[1]["for"] == "value"
    with pytest.raises(ValueError, match="role"):
        sn.parse_roles(json.dumps([_send("/a.tif", "colour")]))
    with pytest.raises(ValueError, match="weights.*for"):
        sn.parse_roles(json.dumps([_send("/a.tif", "weights", "w")]))
    assert sn.parse_roles("[]") == []


def test_send_georef_reads_a_geotiff_an_npz_and_a_layer_plane(tmp_path):
    from dynamix.core.bus import register_plane

    p = _tif(tmp_path / "g.tif", np.ones((40, 50)), 500000.0, 4000000.0, 30.0)
    g = sn.send_georef(_send(p))
    assert g["crs"] == UTM and g["shape"] == (40, 50) and g["georeferenced"]
    assert g["x0"] == 500015.0 and g["dx"] == 30.0 and g["y0"] == 3999985.0 and g["dy"] == -30.0
    f = _target(np.ones((8, 8)), name="n")
    f.save_npz(tmp_path / "n.npz")
    g = sn.send_georef(_send(str(tmp_path / "n.npz")))
    assert g["crs"] == UTM and g["shape"] == (8, 8)
    register_plane("stampA", _target(np.ones((6, 7)), name="plane"))
    g = sn.send_georef({"layer": 3, "stamp": "stampA", "label": "L", "role": "value"})
    assert g["shape"] == (6, 7) and g["crs"] == UTM and g["source"] == "layer:3"


def test_a_send_on_the_same_grid_reads_exactly_the_targets_window(tmp_path):
    vals = np.arange(100 * 120, dtype=np.float64).reshape(100, 120)
    p = _tif(tmp_path / "g.tif", vals, 500000.0, 4000000.0, 30.0)
    t = _target(np.zeros((40, 50)), x0=500000.0 + 30.0 * 10, y0=4000000.0 - 30.0 * 20)
    guide = sn.materialize_guide(_send(p, label="same"), t)
    assert guide.relation.kind == "same" and guide.relation.offset == (0, 0)
    assert guide.field.values.shape == (40, 50)
    np.testing.assert_array_equal(guide.field.values, vals[20:60, 10:60])
    assert guide.field.x_axis[0] == pytest.approx(t.x_axis[0])
    assert guide.label == "same" and guide.role == "value" and guide.stamp


def test_a_coarser_nested_send_reads_the_blocks_that_cover_the_target_plus_one(tmp_path):
    vals = np.arange(50 * 60, dtype=np.float64).reshape(50, 60)
    p = _tif(tmp_path / "c.tif", vals, 500000.0, 4000000.0, 60.0)
    t = _target(np.zeros((40, 50)), x0=500000.0 + 30.0 * 11, y0=4000000.0 - 30.0 * 21)
    guide = sn.materialize_guide(_send(p, label="coarse"), t)
    assert guide.relation.kind == "nested" and guide.relation.ratio == 2
    full = relate(georef(t), sn.send_georef(_send(p)))
    r0, c0, r1, c1 = sn.window_on(sn.send_georef(_send(p)), georef(t), full)
    assert (r0, c0) == (9, 4) and (r1, c1) == (32, 32)      # rows 10..30, cols 5..30, + 1 each side
    assert guide.field.values.shape == (r1 - r0, c1 - c0)
    assert guide.field.values[1, 1] == vals[10, 5]
    # the window's own origin: guide row 9 starts at target row 2 * 9 - 21 = -3 (the block
    # holding target row 0 is guide row 10, which starts at -1; the margin adds one block)
    assert guide.relation.offset == (-3, -3)


def test_a_foreign_send_reads_the_window_its_mapping_covers(tmp_path):
    vals = np.random.default_rng(0).random((200, 200))
    # UTM 15N (500000, 4000000) is lon -93.0, lat 36.13: a lon/lat raster over that spot
    p = _tif(tmp_path / "ll.tif", vals, -93.05, 36.2, 0.0005, crs="EPSG:4326")
    t = _target(np.zeros((40, 50)), x0=500000.0, y0=4000000.0)
    guide = sn.materialize_guide(_send(p, label="dem"), t)
    assert guide.relation.kind == "foreign" and guide.relation.mapping_source == "crs"
    assert 0 < guide.field.values.shape[0] < 200 and guide.field.provenance["crs"] == "EPSG:4326"
    g = georef(guide.field)
    assert g["lattice"] is None and g["georeferenced"]


def test_a_send_entirely_off_the_target_is_refused_by_name(tmp_path):
    p = _tif(tmp_path / "far.tif", np.ones((20, 20)), 900000.0, 4900000.0, 30.0)
    t = _target(np.zeros((40, 50)))
    with pytest.raises(ValueError, match="far.*outside"):
        sn.materialize_guide(_send(p, label="far"), t)


def test_a_send_without_a_georeference_is_refused_by_name(tmp_path):
    bare = RasterField(name="bare", values=np.ones((8, 8)), frame=LocalFrame(),
                       x_axis=np.arange(8.0) + 0.5, y_axis=np.arange(8.0) + 0.5)
    bare.save_npz(tmp_path / "bare.npz")
    t = _target(np.zeros((40, 50)))
    with pytest.raises(ValueError, match="notebook.*no mapping"):
        sn.materialize_guide(_send(str(tmp_path / "bare.npz"), label="notebook"), t)


def test_two_sds_of_one_hdf4_file_on_different_grids_are_not_one_grid(tmp_path):
    pytest.importorskip("pyhdf")
    from pyhdf.SD import SD, SDC

    from dynamix.core.ingest import load_grid

    p = tmp_path / "two.hdf"
    sd = SD(str(p), SDC.WRITE | SDC.CREATE)
    for name, shape in (("fine", (40, 50)), ("coarse", (20, 25))):
        s = sd.create(name, SDC.FLOAT64, shape)
        s[:] = np.ones(shape)
        s.endaccess()
    sd.end()
    fine = load_grid(p, subdataset="fine")
    send = {"path": str(p), "subdataset": "coarse", "band": None, "label": "coarse", "role": "value"}
    with pytest.raises(ValueError, match="coarse.*no mapping"):   # no lattice: nothing relates them
        sn.materialize_guide(send, fine)
    same = {"path": str(p), "subdataset": "fine", "band": None, "label": "fine again", "role": "value"}
    assert sn.materialize_guide(same, fine).relation.kind == "same"


def test_an_roi_window_target_reads_the_matching_sub_window(tmp_path):
    vals = np.arange(100 * 120, dtype=np.float64).reshape(100, 120)
    p = _tif(tmp_path / "g.tif", vals, 500000.0, 4000000.0, 30.0)
    # a processing window of a 100 x 120 dataset: rows 20..60, cols 10..60, as the runner cuts it
    t = _target(np.zeros((40, 50)), x0=500000.0 + 30.0 * 10, y0=4000000.0 - 30.0 * 20,
                full_dims=(100, 120), window={"row_off": 20, "col_off": 10})
    guide = sn.materialize_guide(_send(p, label="same"), t)
    np.testing.assert_array_equal(guide.field.values, vals[20:60, 10:60])


def test_a_layer_send_not_computed_yet_is_named():
    t = _target(np.zeros((8, 8)))
    with pytest.raises(ValueError, match="pending layer.*not computed") as ei:
        sn.materialize_guide({"layer": 9, "stamp": "nope", "label": "pending layer",
                              "role": "value"}, t)
    assert str(ei.value).count("pending layer") == 1


def test_resolve_guides_binds_weights_to_their_send_and_refuses_another_grid(tmp_path):
    vals = np.ones((40, 50))
    p = _tif(tmp_path / "v.tif", vals, 500000.0, 4000000.0, 30.0)
    w = _tif(tmp_path / "w.tif", 0.5 * vals, 500000.0, 4000000.0, 30.0)
    w2 = _tif(tmp_path / "w2.tif", np.ones((20, 25)), 500000.0, 4000000.0, 60.0)
    t = _target(np.zeros((40, 50)))
    text = json.dumps([_send(p, "value", "v"), _send(w, "weights", "w", **{"for": "v"}),
                       _send(w2, "weights", "wt", **{"for": "target"})])
    guides = sn.resolve_guides(text, t)
    assert [g.label for g in guides] == ["v", "wt"]          # the bound weights ride on v
    assert guides[0].weights is not None and guides[0].weights.values[0, 0] == 0.5
    assert guides[1].role == "weights" and guides[1].weights_for == "target"
    bad = json.dumps([_send(p, "value", "v"), _send(w2, "weights", "w2", **{"for": "v"})])
    with pytest.raises(ValueError, match="w2.*grid"):
        sn.resolve_guides(bad, t)
    orphan = json.dumps([_send(w, "weights", "w", **{"for": "nobody"})])
    with pytest.raises(ValueError, match="nobody"):
        sn.resolve_guides(orphan, t)


def test_resolved_guides_drive_the_engine(tmp_path):
    from dynamix.core.inpaint.engine import run_inpaint
    from dynamix.devices.inpaint import PARAMS

    rng = np.random.default_rng(1)
    truth = rng.standard_normal((64, 64)).cumsum(0).cumsum(1)
    y = truth.copy()
    y[20:40, 20:40] = np.nan
    coarse = truth.reshape(32, 2, 32, 2).mean(axis=(1, 3))
    p = _tif(tmp_path / "prior.tif", coarse, 500000.0, 4000000.0, 60.0)
    t = _target(y)
    params = {q.name: q.default for q in PARAMS}
    params.update(route="sparsity", n_iter=20, heldout_blocks=0,
                  _sends=json.dumps([_send(p, "value", "prior")]))
    res = run_inpaint(t, y, params, sn.resolve_guides(params["_sends"], t))
    assert res["_grid_relations"]["prior"]["kind"] == "nested"
    inside = ~np.isfinite(y)
    err = np.abs(res["raster_out"][inside] - truth[inside]).mean()
    plain = run_inpaint(t, y, {**params, "_sends": "[]"}, [])
    assert err < np.abs(plain["raster_out"][inside] - truth[inside]).mean()


def test_an_unreferenced_same_band_send_over_an_roi_window_is_same(tmp_path):
    vals = np.arange(100 * 120, dtype=np.float64).reshape(100, 120)
    band = RasterField(name="band", values=vals, frame=LocalFrame(),
                       x_axis=np.arange(120.0) + 0.5, y_axis=np.arange(100.0) + 0.5)
    band.save_npz(tmp_path / "band.npz")
    src = str(tmp_path / "band.npz")
    t = RasterField(name="win", values=np.zeros((40, 50)), frame=LocalFrame(),
                    x_axis=0.5 + 10 + np.arange(50.0), y_axis=0.5 + 20 + np.arange(40.0),
                    provenance={"source": src, "full_dims": (100, 120),
                                "window": {"row_off": 20, "col_off": 10}})
    guide = sn.materialize_guide(_send(src, label="band"), t)
    assert guide.relation.kind == "same" and guide.relation.mapping_source == "file"
    assert guide.relation.offset == (0, 0)
    assert guide.field.values.shape == (40, 50)
    np.testing.assert_array_equal(guide.field.values, vals[20:60, 10:60])


def test_a_bound_weights_send_of_another_extent_is_refused(tmp_path):
    p = _tif(tmp_path / "v.tif", np.ones((40, 50)), 500000.0, 4000000.0, 30.0)
    w = _tif(tmp_path / "w.tif", np.ones((30, 50)), 500000.0, 4000000.0, 30.0)
    t = _target(np.zeros((40, 50)))
    text = json.dumps([_send(p, "value", "v"), _send(w, "weights", "w", **{"for": "v"})])
    with pytest.raises(ValueError, match="same extent"):
        sn.resolve_guides(text, t)


def test_duplicate_send_labels_are_refused():
    with pytest.raises(ValueError, match="share this label"):
        sn.parse_roles(json.dumps([_send("/a.tif", label="x"), _send("/b.tif", label="x")]))


def test_a_rewritten_file_changes_the_guides_stamp(tmp_path):
    t = _target(np.zeros((8, 8)))
    f = _target(np.ones((8, 8)), name="n")
    f.save_npz(tmp_path / "n.npz")
    a = sn.materialize_guide(_send(str(tmp_path / "n.npz")), t).stamp
    f2 = _target(np.full((8, 8), 2.0), name="n")
    f2.save_npz(tmp_path / "n.npz")
    assert sn.materialize_guide(_send(str(tmp_path / "n.npz")), t).stamp != a
