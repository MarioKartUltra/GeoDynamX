# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Follow-detection parity against the REAL xsmurf C (built for the exact-port slice).

Both stacks receive the IDENTICAL kapa/kapap/mod/arg images (float32, the C's own type), so
this pins DETECTION and the VALUE CHANNEL alone: ``xsmurf_wrapper.find_extrema_follow``
(w2_folow_contour + _get_interpolated_modulus_, followVersion=1) versus
``dynamix.core.xsmurf_follow``'s ports of the same functions. The kapa/kapap fields
themselves are cross-checked between our ``follow2d.kapa_fields`` port and the wrapper's
``gkapa_numpy``/``gkapap_numpy`` first, so a mismatch localizes to the right layer.

Machine-specific by nature (needs the compiled XSmurfWrapper); skips cleanly when absent.
"""
from __future__ import annotations

import numpy as np
import pytest

xw = pytest.importorskip("xsmurf_wrapper")

from conftest import fbm2d                                            # noqa: E402
from dynamix.core import xsmurf_follow as xf                          # noqa: E402
from dynamix.core.follow2d import kapa_fields                         # noqa: E402
from dynamix.core.wtmm_backend import _cwt2d_numpy, compute_scales2d  # noqa: E402

N = 96


@pytest.fixture(scope="module")
def images():
    field = fbm2d(N, 0.6, seed=7)
    scales = compute_scales2d(2, 2, 1.0)
    raw = _cwt2d_numpy(field, scales, derivs="all", verbose=False)
    per_scale = []
    for si in range(len(scales)):
        d = {k: np.asarray(raw[k][si], dtype=np.float64) for k in
             ("dx", "dy", "dxx", "dxy", "dyy", "dxxx", "dxxy", "dxyy", "dyyy")}
        kapa, kapap = kapa_fields(raw, si)
        mod = np.hypot(d["dx"], d["dy"])
        arg = np.arctan2(d["dy"], d["dx"])
        # float32 for BOTH stacks -- the C's real type; identical inputs by construction
        per_scale.append({k: v.astype(np.float32) for k, v in
                          {"kapa": kapa, "kapap": kapap, "mod": mod, "arg": arg}.items()}
                         | {"scale": float(scales[si]), "dstacks": d})
    return per_scale


def test_our_kapa_fields_match_the_wrapper_kernels(images):
    for im in images:
        d = im["dstacks"]
        wk = xw.gkapa_numpy(d["dx"], d["dy"], d["dxx"], d["dxy"], d["dyy"])
        wkp = xw.gkapap_numpy(d["dx"], d["dy"], d["dxx"], d["dxy"], d["dyy"],
                              d["dxxx"], d["dxxy"], d["dxyy"], d["dyyy"])
        np.testing.assert_allclose(im["kapa"], wk.astype(np.float32), rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(im["kapap"], wkp.astype(np.float32), rtol=2e-6, atol=1e-6)


def test_follow_detection_point_sets_are_identical(images):
    for im in images:
        ximg = xw.find_extrema_follow(xw.XImage.from_numpy(im["kapa"]),
                                      xw.XImage.from_numpy(im["kapap"]),
                                      xw.XImage.from_numpy(im["mod"]),
                                      xw.XImage.from_numpy(im["arg"]),
                                      im["scale"], thresh=0.0, version=1)
        theirs = ximg.get_extrema_arrays()
        b = set(zip(np.asarray(theirs["x"], int).tolist(),
                    np.asarray(theirs["y"], int).tolist()))
        ys, xs = xf.follow_contour(im["kapa"].astype(np.float64),
                                   im["kapap"].astype(np.float64))
        a = set(zip(xs.tolist(), ys.tolist()))
        assert a == b, (f"scale {im['scale']}: {len(a - b)} only ours, "
                        f"{len(b - a)} only theirs of {len(b)}")


def test_follow_moduli_match_the_v1_interpolation(images):
    for im in images:
        ximg = xw.find_extrema_follow(xw.XImage.from_numpy(im["kapa"]),
                                      xw.XImage.from_numpy(im["kapap"]),
                                      xw.XImage.from_numpy(im["mod"]),
                                      xw.XImage.from_numpy(im["arg"]),
                                      im["scale"], thresh=0.0, version=1)
        theirs = ximg.get_extrema_arrays()
        tx = np.asarray(theirs["x"], int); ty = np.asarray(theirs["y"], int)
        tmod = np.asarray(theirs["mod"], np.float64)
        mod64 = im["mod"].astype(np.float64); kapa64 = im["kapa"].astype(np.float64)
        ours = np.array([xf.interpolated_modulus(mod64, kapa64, int(x), int(y))
                         for x, y in zip(tx, ty)])
        np.testing.assert_allclose(ours, tmod, rtol=5e-6, atol=1e-6)
