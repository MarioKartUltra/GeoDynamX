# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""mz_edges device: registration -> real compute -> zero-miss law through resolve.

Ordering convention per tests/test_wtmm_roi_device.py: structural first, then behavior.

``fbm64`` (tests/conftest.py) is a bare ndarray, not a Field -- every real compute call below
wraps it in a RasterField first, mirroring tests/test_wtmm_backend.py's own construction idiom
(``RasterField(name=..., values=fbm64, frame=LocalFrame(units="px"), x_axis=..., y_axis=...)``).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices
from dynamix.model.device import DEVICES, get_device, is_transform, validate_params


def _field(values):
    ny, nx = values.shape
    return RasterField(name="fbm", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(nx, dtype=np.float64),
                       y_axis=np.arange(ny, dtype=np.float64))


@pytest.fixture
def mz(clean_registry):
    register_builtin_devices()
    return get_device("mz_edges")


# --------------------------------------------------------------------------- structural


def test_registered_and_transform(mz):
    assert mz.name == "mz_edges"
    assert is_transform(mz)


def test_param_surface(mz):
    names = [p.name for p in mz.params]
    assert names == ["n_levels", "algorithm", "border", "colocate_l1", "dither",
                     "interpolate", "recon_live", "kappa", "clip", "run_mode", "iterations",
                     "tolerance", "coarse", "mode", "show"]
    defaults = {p.name: p.default for p in mz.params}
    assert defaults == {"n_levels": 4, "algorithm": "lastwave", "border": "mirror",
                        "colocate_l1": False, "dither": False, "interpolate": False,
                        "recon_live": True, "kappa": 1.0, "clip": False, "run_mode": "converge",
                        "iterations": 20, "tolerance": 1e-3, "coarse": "full",
                        "mode": "separable", "show": "edges"}


def test_validate_rejects_unknown_keys(mz):
    with pytest.raises(Exception):
        validate_params(mz, {"n_levels": 4, "bogus": 1})


# --------------------------------------------------------------------------- cache_key


def test_different_params_produce_different_cache_keys(mz):
    """Idiom authority: test_wtmm_roi_device.py's
    test_two_different_rois_produce_two_different_cache_keys. Pins the params into the key --
    a cache_key that dropped them (e.g. ``_k(self.name, source_id, {})``) would still pass every
    other test in this file, since resolve() never calls the device's own cache_key method."""
    base = validate_params(mz, {"n_levels": 3})
    other = validate_params(mz, {"n_levels": 4})
    assert mz.cache_key("src0", base) != mz.cache_key("src0", other)


def test_same_params_produce_the_same_cache_key(mz):
    """Idiom authority: test_wtmm_roi_device.py's
    test_same_roi_and_params_produce_the_same_cache_key."""
    params = validate_params(mz, {"n_levels": 3, "coarse": "thumbnail", "dither": True})
    assert mz.cache_key("src0", params) == mz.cache_key("src0", dict(params))


# --------------------------------------------------------------------------- real compute


def test_compute_bundle_over_fbm(mz, fbm64):
    field = _field(fbm64)
    params = validate_params(mz, {"algorithm": "printed"})
    res = mz.compute(field, params)
    assert res["chains"] == []
    assert len(res["extrema"]) == params["n_levels"]
    assert len(res["mz_maxima"]) == params["n_levels"]
    assert res["_shape"] == field.values.shape
    assert res["_frame"] is field.frame
    assert "wavelet" not in res["params"] and "alpha" not in res["params"]
    assert res["wavelet"] == "mz_spline"                  # the printed algorithm's spline
    assert res["algorithm"] == "printed"
    assert res["scales"].dtype == np.float64
    # dither defaults to False: analyze() must not have measured/applied an LSB dither.
    assert res["lsb"] is None
    assert res["coarse_policy"] == "full"


def test_compute_lastwave_bundle_over_fbm(mz, fbm64):
    field = _field(fbm64)
    params = validate_params(mz, {})
    res = mz.compute(field, params)
    assert res["algorithm"] == "lastwave" and res["chains"] == []
    assert len(res["extrema"]) == params["n_levels"]
    assert res["_shape"] == field.values.shape
    assert res["_frame"] is field.frame
    assert res["scales"].dtype == np.float64
    assert res["_display_offset"] == (-0.5, -0.5)
    for level in res["extrema"]:
        assert {"x", "y", "mod", "arg", "line_id"} <= set(level)


def test_compute_honors_dither_and_leaves_coarse_to_the_outputs(mz, fbm64):
    """The default-params compute above can't tell a device that plumbs dither through from one
    that silently drops it, since analyze()'s own default (dither=False) coincides with this
    device's param default. Force it (and Coarse) off their defaults: fbm64 is continuous
    float64 data, so measure_lsb (invoked only when dither=True) finds a positive lsb. Coarse is
    view-only: the analysis always keeps the full coarse channel, and the 2^J thumbnail
    (64 % 2**3 == 0) is derived from it by the lazy "thumbnail" output."""
    field = _field(fbm64)
    params = validate_params(mz, {"algorithm": "printed", "n_levels": 3,
                                  "coarse": "thumbnail", "dither": True})
    res = mz.compute(field, params)
    assert res["coarse_policy"] == "full"
    assert res["lsb"] is not None
    thumb = mz.compute_output("thumbnail", field.values, res, params, fetch=None)
    assert thumb["raster"].shape == (8, 8) and thumb["display_stride"] == 8
    lastwave = validate_params(mz, {"n_levels": 3, "coarse": "thumbnail"})
    res = mz.compute(field, lastwave)
    thumb = mz.compute_output("thumbnail", field.values, res, lastwave, fetch=None)
    assert thumb["raster"].shape == (8, 8) and thumb["display_stride"] == 8


@pytest.mark.parametrize("algorithm", ["lastwave", "printed"])
def test_compute_refuses_oversized_ladder(mz, fbm64, algorithm):
    # n_levels=8 passes param validation (max=10) but 2**8 > 64: compute must refuse
    field = _field(fbm64)
    params = validate_params(mz, {"n_levels": 8, "algorithm": algorithm})
    with pytest.raises(ValueError):
        mz.compute(field, params)


# --------------------------------------------------------------------------- zero-miss law


def test_zero_miss_law_through_resolve(mz, fbm64):
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.project import Project

    field = _field(fbm64)
    project = Project()
    src = project.add_source("synthetic://fbm64")
    layer = project.add_layer(
        "mz", src.source_id, Chain((DeviceRef("mz_edges", {"n_levels": 3}),)))

    cache = Cache()
    first = resolve(layer, field, cache)
    again = resolve(layer, field, cache)
    assert first.cache_misses >= 1
    assert again.cache_misses == 0
    assert again.result["extrema"] is first.result["extrema"]
