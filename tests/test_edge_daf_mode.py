# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The DAF periodic boundary as an edge mode (spec 22 §3.4): where a scale's halo runs past the
raster, its real block is made periodic by the self-tuned DAF extension and transformed without
padding; the tuning goes into _edge and the .gprv; where the data are not band-limited the run
reflects, and the edge knob says why."""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dynamix.core.rasterfield import RasterField  # noqa: E402
from dynamix.devices.wtmm import WTMM2D  # noqa: E402
from dynamix.devices.wtmm_roi import WTMM2DROI  # noqa: E402
from dynamix.model.device import defaults_for, validate_params  # noqa: E402
from test_edge_taper import _field, _labyrinth  # noqa: E402
from test_roi_halo import _write_tif  # noqa: E402

TUNING = {"f_b", "gap", "M", "sigma", "lam", "ppw"}


def _roi_params(**kw):
    dev = WTMM2DROI()
    p = dict(defaults_for(dev), n_oct=2, n_voice=4, a_min=1.0, fracint_alpha=0.0, roi_row=0,
             roi_col=0, roi_h=64, roi_w=64, edge="daf_periodic")
    p.update(kw)
    return validate_params(dev, p)


def _roi_field(tif):
    return RasterField.from_geotiff_window(tif, row_off=0, col_off=0, height=96, width=96)


def test_a_corner_roi_runs_on_the_daf_extension_and_records_its_tuning(tmp_path):
    tif = _write_tif(tmp_path / "p.tif", _labyrinth(256))
    res = WTMM2DROI().compute(_roi_field(tif), _roi_params())
    edge = res["_edge"]
    assert edge["mode"] == "daf_periodic" and edge["requested"] == "daf_periodic"
    assert TUNING <= edge.keys() and edge.get("fallback") is None and edge["M"] == 12
    assert edge["ppw"] >= 6.0 and edge["gap"] >= 8
    assert not any("mirror" in w for w in res.get("_warnings", []))
    assert sum(len(l["x"]) for l in res["extrema"]) > 0
    for lvl in res["extrema"]:
        assert ((lvl["x"] >= 0) & (lvl["x"] < 64) & (lvl["y"] >= 0) & (lvl["y"] < 64)).all()


def test_the_daf_fill_differs_from_reflection_at_the_edge_only(tmp_path):
    tif = _write_tif(tmp_path / "p.tif", _labyrinth(256))
    daf = WTMM2DROI().compute(_roi_field(tif), _roi_params())
    refl = WTMM2DROI().compute(_roi_field(tif), _roi_params(edge="reflect"))
    far = lambda l: (l["y"] >= 40) & (l["x"] >= 40)          # noqa: E731 - beyond the finest reach
    a = {(int(y), int(x)) for l in daf["extrema"][:1] for y, x in zip(l["y"][far(l)], l["x"][far(l)])}
    b = {(int(y), int(x)) for l in refl["extrema"][:1] for y, x in zip(l["y"][far(l)], l["x"][far(l)])}
    assert a == b and a


def test_not_band_limited_data_fall_back_to_reflection_and_say_why(tmp_path):
    x = np.arange(256.0)
    stripes = np.sin(2 * np.pi * x / 5.0)[None, :].repeat(256, axis=0).astype(np.float32)
    tif = _write_tif(tmp_path / "p.tif", stripes)
    p = _roi_params()
    res = WTMM2DROI().compute(_roi_field(tif), p)
    assert res["_edge"]["mode"] == "reflect"
    assert "points per wavelength" in res["_edge"]["fallback"]
    assert WTMM2DROI().knob_warnings(_roi_field(tif), p)["edge"] == res["_edge"]["fallback"]


def test_nodata_in_the_halo_is_zero_filled_before_the_solve(tmp_path):
    v = _labyrinth(256)
    v[2:6, 70:76] = -9999.0
    tif = _write_tif(tmp_path / "p.tif", v, nodata=-9999.0)
    res = WTMM2DROI().compute(_roi_field(tif), _roi_params(roi_col=64))
    assert res["_edge"]["mode"] == "daf_periodic"
    assert all(np.isfinite(l["mod"]).all() for l in res["extrema"])
    supported = np.asarray(res["hd_std"]["N_a"]) > 0         # a scale no chain reaches is NaN
    assert np.isfinite(np.asarray(res["hd_std"]["tau_qa"], float)[:, supported]).all()


def test_the_region_runner_matches_the_file_path(tmp_path):
    from test_roi_wtmm_window import _KEYS, _picture, _points

    from dynamix.roi.halo import run_wtmm2d_roi
    from dynamix.roi.runner import run_on_region

    tif = tmp_path / "s.tif"
    _write_tif(tif, _labyrinth(96, k0=1 / 10))
    p = dict(defaults_for(WTMM2D()), n_oct=2, n_voice=2, a_min=1.0, edge="daf_periodic")
    params = validate_params(WTMM2D(), p)
    roi = (0, 0, 32, 32)
    got = run_on_region([(WTMM2D(), params)], _picture(tif), roi)
    want = run_wtmm2d_roi(str(tif), (96, 96), roi, dict({k: params[k] for k in _KEYS},
                                                        edge="daf_periodic"))
    assert got["_edge"]["mode"] == want["_edge"]["mode"] == "daf_periodic"
    assert got["_edge"]["gap"] == want["_edge"]["gap"]
    for g, w in zip(got["extrema"], want["extrema"]):
        pg, pw = _points(g), _points(w)
        assert pg.keys() == pw.keys()
        # float32: the two paths transform the same pixels on differently padded windows, so the
        # moduli agree to the engine's float32 round-off, not bit for bit.
        assert np.allclose([pg[k] for k in pw], [pw[k] for k in pw], rtol=1e-5, atol=1e-7)


def test_a_whole_raster_runs_on_its_periodic_extension():
    v = _labyrinth(128)
    p = validate_params(WTMM2D(), dict(defaults_for(WTMM2D()), n_oct=2, n_voice=4, a_min=1.0,
                                       fracint_alpha=0.0, edge="daf_periodic"))
    res = WTMM2D().compute(_field(v), p)
    assert res["_edge"]["mode"] == "daf_periodic" and TUNING <= res["_edge"].keys()
    for lvl in res["extrema"]:
        assert ((lvl["x"] >= 0) & (lvl["x"] < 128) & (lvl["y"] >= 0) & (lvl["y"] < 128)).all()
    assert res["chains"] and all(((c["x"] >= 0) & (c["x"] < 128)).all() for c in res["chains"])
    assert np.isfinite(np.asarray(res["hd_std"]["tau_qa"], float)).any()
