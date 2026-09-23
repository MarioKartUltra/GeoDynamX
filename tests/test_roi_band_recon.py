# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Band reconstruction on a region."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


def _field(n=96):
    v = np.random.default_rng(5).standard_normal((n, n)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(), x_axis=np.arange(float(n)),
                       y_axis=np.arange(float(n)))


def _params(device, **kw):
    from dynamix.model.device import defaults_for, validate_params

    p = defaults_for(device)
    p.update(kappa=4.0, n_scales=4, **kw)
    if "h_lo" in p:                                   # the band devices: a wide band
        p.update(h_lo=-5.0, h_hi=5.0)
    return validate_params(device, p)


@pytest.mark.parametrize("cls,method", [("BandReconMeasure", "measure"),
                                        ("BandReconMultiaffine", "multiaffine")])
def test_the_hmap_uses_the_margin_and_the_reconstruction_only_the_roi(cls, method):
    from dynamix.core import microcanonical as mc
    import dynamix.devices.band_recon as br
    from dynamix.devices.holder_methods import holder_roi_margin, method_arrays
    from dynamix.roi.runner import read_processing_window, run_on_region

    dev = getattr(br, cls)()
    params = _params(dev)
    f, roi = _field(), (30, 34, 24, 28)
    m = dev.roi_margin(params)
    assert m == holder_roi_margin(method, params) and m > 0

    res = run_on_region([(dev, params)], f, roi)
    r, c, h, w = roi
    for key in ("raster_out", "h_map", "r2_map", "band_mask"):
        assert res[key].shape == (h, w), key
    # h(x) was estimated over the WINDOW (ROI + margin), then cut to the ROI ...
    win, _info = read_processing_window(f, roi, m)
    h_win, _r2, _s = method_arrays(np.asarray(win.values), method, params)
    np.testing.assert_allclose(res["h_map"], h_win[m:m + h, m:m + w], equal_nan=True)
    # ... and the reconstruction sees the ROI's own values only
    roi_vals = np.asarray(f.values)[r:r + h, c:c + w]
    want, _psnr, _err = mc.reconstruct_from_msc(roi_vals, res["band_mask"])
    np.testing.assert_allclose(res["raster_out"], want)


def test_the_conflated_band_recon_follows_its_estimator_param():
    from dynamix.devices.band_recon import BandRecon
    from dynamix.devices.holder_methods import holder_roi_margin

    dev = BandRecon()
    p = _params(dev)
    method = "multiaffine" if p["estimator"] == "multiaffine" else "measure"
    assert dev.roi_margin(p) == holder_roi_margin(method, p)


def test_a_heavier_tail_needs_a_wider_margin():
    from dynamix.devices.holder_methods import HolderMeasure, holder_roi_margin

    g = _params(HolderMeasure(), wavelet="gaussian")
    lz = _params(HolderMeasure(), wavelet="lorentzian", beta=1.0)
    assert holder_roi_margin("measure", lz) > holder_roi_margin("measure", g)
