# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A real ASTER L1T scene: a transplanted natural mask, the three routes, the readouts.
Runs only with GEODYNAMIX_ASTER_L1T set to the scene folder."""
from __future__ import annotations

import glob
import json
import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("pywt")
pytest.importorskip("rasterio")
pytest.importorskip("scipy")

SCENE = os.environ.get("GEODYNAMIX_ASTER_L1T")
pytestmark = pytest.mark.skipif(not SCENE, reason="GEODYNAMIX_ASTER_L1T not set")


def _band(path, k):
    import rasterio

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    with rasterio.open(path) as src:
        values = np.asarray(src.read(k), dtype=np.float64)
        t, crs = src.transform, src.crs
    ny, nx = values.shape
    x = t.c + t.a * (np.arange(nx) + 0.5)
    y = t.f + t.e * (np.arange(ny) + 0.5)
    return RasterField(name=f"{pathlib.Path(path).stem}:b{k}", values=values,
                       frame=LocalFrame(x0=x[0], y0=y[0], dx=abs(t.a), dy=abs(t.e), units="m"),
                       x_axis=x, y_axis=y, provenance={"crs": str(crs), "source": str(path)})


def _crop(field, r0, c0, h, w):
    import dataclasses

    return dataclasses.replace(field, values=field.values[r0:r0 + h, c0:c0 + w],
                               x_axis=field.x_axis[c0:c0 + w], y_axis=field.y_axis[r0:r0 + h])


def test_three_routes_on_a_transplanted_natural_mask(capsys, tmp_path):
    from scipy.ndimage import label

    from dynamix.core.inpaint.engine import run_inpaint
    from dynamix.core.inpaint.sends import resolve_guides
    from dynamix.devices.inpaint import PARAMS

    (vtif,) = glob.glob(os.path.join(SCENE, "*_V.tif"))
    (ttif,) = glob.glob(os.path.join(SCENE, "*_T.tif"))
    v3 = _crop(_band(vtif, 3), 1200, 1800, 1024, 1024)        # 15 m browse band 3; 1024^2 on clear ground
    v1 = _crop(_band(vtif, 1), 1200, 1800, 1024, 1024)
    bright = v3.values > np.nanquantile(v3.values, 0.99)
    ids, n = label(bright, structure=np.ones((3, 3)))
    sizes = np.bincount(ids.ravel())[1:]
    big = int(np.argmax(sizes)) + 1
    assert sizes[big - 1] > 400, "no natural region over 400 px to transplant"
    mask = np.roll(ids == big, 600, axis=0)
    y = v3.values.copy()
    y[mask] = np.nan
    target = _crop(v3, 0, 0, 1024, 1024)
    target.values = y
    v1.save_npz(tmp_path / "v1.npz")
    value = {"path": str(tmp_path / "v1.npz"), "subdataset": None, "band": None,
             "label": "V band 1", "role": "value"}
    structure = {"path": ttif, "subdataset": None, "band": 0, "label": "T band 1", "role": "structure"}
    guides = resolve_guides(json.dumps([value, structure]), target)
    kinds = {g.label: g.relation.kind for g in guides}
    assert kinds["V band 1"] == "same" and kinds["T band 1"] == "foreign"   # the T grid is 37.5 m off
    params = {p.name: p.default for p in PARAMS}
    params.update(heldout_blocks=8, heldout_side=16, seed=0)
    rows = []
    for route in ("sparsity", "cascade", "both"):
        use = guides if route == "sparsity" else [g for g in guides if g.role == "value"]
        res = run_inpaint(target, y, {**params, "route": route}, use)
        fine = np.mean([v for k, v in res["_amp_distance"].items() if k.startswith("1/")])
        rows.append((route, res["_heldout"]["rms_rel"], res["_variance_error"], fine,
                     res["_iterations"], res["_stop"]))
        assert np.isfinite(res["raster_out"]).all()
    with capsys.disabled():
        print("\nASTER L1T 3N, transplanted mask of", int(mask.sum()), "px:")
        for route, rms, var_err, fine, it, stop in rows:
            print(f"  {route:9s} held-out RMS {rms:.4f}  var err {var_err:.4f}  "
                  f"fine ln|c| distance {fine:.3f}  {it} it ({stop})")
    assert all(np.isfinite(r[1]) for r in rows)
