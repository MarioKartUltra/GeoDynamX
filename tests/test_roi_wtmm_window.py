# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""wtmm2d's own ROI step: per-scale halos sliced from the runner's
in-memory window must reproduce the proven file-reading path (``run_wtmm2d_roi``, whose own
oracle is ``tests/test_roi_halo.py``) -- extrema, moduli and chains -- then chain inside the ROI.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402

_KEYS = ("n_oct", "n_voice", "a_min", "wavelet", "fracint_alpha", "interpolate", "detector",
         "min_chain_len", "smooth", "thresh", "dist2_max", "box_ratio", "similitude")


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(7)
    v = rng.standard_normal((96, 96)).cumsum(0).cumsum(1).astype(np.float32)
    path = tmp_path / "s.tif"
    with rasterio.open(path, "w", driver="GTiff", height=96, width=96, count=1, dtype="float32",
                       crs="EPSG:32615", transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as d:
        d.write(v, 1)
    return path


def _picture(path):
    f = RasterField(name="pic", values=np.zeros((10, 10)), frame=LocalFrame(),
                    x_axis=np.arange(10.0), y_axis=np.arange(10.0))
    f.provenance.update({"display_stride": 10, "full_dims": (96, 96), "source": str(path),
                         "window": {"row_off": 0, "col_off": 0}})
    return f


def _params():
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.model.device import defaults_for, validate_params

    p = defaults_for(WTMM2D())
    p.update(n_oct=2, n_voice=2, a_min=1.0)
    return validate_params(WTMM2D(), p)


def _points(level):
    return {(int(y), int(x)): float(m) for y, x, m in zip(level["y"], level["x"], level["mod"])}


@pytest.mark.parametrize("roi", [(30, 34, 24, 28), (0, 0, 24, 28)], ids=["interior", "corner"])
def test_wtmm2d_on_a_region_reproduces_the_file_halo_path(source, roi):
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.roi.halo import run_wtmm2d_roi
    from dynamix.roi.runner import run_on_region

    params = _params()
    got = run_on_region([(WTMM2D(), params)], _picture(source), roi)
    want = run_wtmm2d_roi(str(source), (96, 96), roi, {k: params[k] for k in _KEYS},
                          boundary="auto")
    assert got["_shape"] == (roi[2], roi[3])
    assert len(got["extrema"]) == len(want["extrema"])
    for g, w in zip(got["extrema"], want["extrema"]):
        pg, pw = _points(g), _points(w)
        assert pg.keys() == pw.keys()
        assert np.allclose([pg[k] for k in pw], [pw[k] for k in pw], rtol=1e-6, atol=1e-9)
    assert len(got["chains"]) == len(want["chains"])
    assert [m["real_frac"] for m in got["_roi_margins"]] == \
           [m["real_frac"] for m in want["_roi_margins"]]
    assert "chain_product" in got                       # draw-ready, like every wtmm result


def test_wtmm2d_declares_its_coarsest_scales_halo_as_its_margin():
    from dynamix.core.wtmm_backend import compute_scales2d
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.roi.halo import halo_margin

    params = _params()
    want = halo_margin(max(compute_scales2d(params["n_oct"], params["n_voice"], params["a_min"])))
    assert WTMM2D().roi_margin(params) == want


def test_dither_is_honoured_on_an_roi_run(source):
    """Final review: wtmm2d's dither knob was silently ignored on ROI runs -- a
    retune that recomputes and changes nothing. The window is dithered like a whole field."""
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.model.device import validate_params
    from dynamix.roi.runner import run_on_region

    import rasterio

    # quantise the source so it HAS a lattice for dither to break
    with rasterio.open(source, "r+") as d:
        v = d.read(1)
        d.write((np.round(v * 2.0) / 2.0).astype(np.float32), 1)
    plain = dict(_params())
    dith = validate_params(WTMM2D(), {**plain, "dither": True})
    roi = (30, 34, 24, 28)
    a = run_on_region([(WTMM2D(), plain)], _picture(source), roi)
    b = run_on_region([(WTMM2D(), dith)], _picture(source), roi)
    assert any(_points(x) != _points(y) for x, y in zip(a["extrema"], b["extrema"]))
