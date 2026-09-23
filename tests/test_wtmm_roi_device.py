# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the wtmm2d_roi device: the halo engine, wearing the Transform contract.

Structural/registration tests come first (cheap, no rasterio needed for most of them), then a
real compute over a small synthetic parent GeoTIFF, then the zero-miss law at the engine level --
the property the whole device split exists for: a downstream filter change must not touch the
cached ROI transform.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
from rasterio.transform import from_origin

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.filters import ScaleSelect
from dynamix.devices.wtmm_roi import WTMM2DROI
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import (defaults_for, is_transform, register_device,
                                  validate_params)
from dynamix.model.project import Project
from dynamix.roi.halo import MIN_A_MIN

_CRS = "EPSG:32615"

# n_oct=1, n_voice=2, a_min=1.0 -> 2 scales, margins 18 and 25 px (checked by hand against
# dynamix.roi.halo.halo_margin) -- small and fast, and the exact halo/oracle claim is
# tests/test_roi_halo.py's job, not this file's.
_SMALL_WTMM = {"n_oct": 1, "n_voice": 2, "a_min": 1.0}


def _synthetic_parent(n=80, seed=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    f = 3.0 * np.sin(2 * np.pi * xx / 23.0) * np.cos(2 * np.pi * yy / 19.0)
    for _ in range(8):
        cy, cx = rng.uniform(4, n - 4, 2)
        f += rng.uniform(-6.0, 6.0) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(2.0, 6.0) ** 2))
    return (f - f.mean()).astype(np.float32)


def _write_tif(path, values):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs=_CRS,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(values, 1)
    return str(path)


def _bare_field(name="npz-loaded"):
    """A field with no recorded source -- the npz-loaded case the device must refuse."""
    values = np.zeros((16, 16))
    return RasterField(name=name, values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(16, dtype=np.float64),
                       y_axis=np.arange(16, dtype=np.float64))


@pytest.fixture
def parent_tif(tmp_path):
    values = _synthetic_parent()
    return _write_tif(tmp_path / "parent.tif", values)


@pytest.fixture
def parent_field(parent_tif):
    """A field whose provenance records the parent's own path -- what a real geotiff layer has."""
    return RasterField.from_geotiff_window(parent_tif, row_off=0, col_off=0, height=80, width=80)


# --------------------------------------------------------------------------- structural


def test_wtmm2d_roi_is_registered_by_the_builtin_list():
    from dynamix.devices import BUILTIN_DEVICES

    assert WTMM2DROI in BUILTIN_DEVICES


def test_wtmm2d_roi_registers_under_its_own_name(clean_registry):
    device = WTMM2DROI()
    register_device(device)
    from dynamix.model.device import DEVICES

    assert DEVICES["wtmm2d_roi"] is device


def test_wtmm2d_roi_is_structurally_a_transform():
    device = WTMM2DROI()
    assert is_transform(device)
    assert hasattr(device, "compute") and hasattr(device, "cache_key")
    assert not hasattr(device, "apply")


def test_params_include_the_roi_six_and_the_wtmm_five():
    names = {p.name for p in WTMM2DROI().params}
    assert {"n_oct", "n_voice", "a_min", "wavelet", "min_chain_len"} <= names
    assert {"roi_row", "roi_col", "roi_h", "roi_w", "boundary"} <= names


def test_a_min_hard_floor_is_raised_to_one():
    """The carry-forward: the engine refuses a_min < 1.0 (contaminated-numbers regime), so the
    Param must not let the UI ask for a value it would reject."""
    a_min = next(p for p in WTMM2DROI().params if p.name == "a_min")
    assert a_min.min == 1.0 == MIN_A_MIN
    with pytest.raises(ValueError, match="below min"):
        a_min.validate(0.5)


def test_every_param_declares_a_label():
    for p in WTMM2DROI().params:
        assert p.label, p.name


def test_defaults_are_internally_valid():
    device = WTMM2DROI()
    assert validate_params(device, {}) == defaults_for(device)


# --------------------------------------------------------------------------- cache_key


def test_two_different_rois_produce_two_different_cache_keys():
    device = WTMM2DROI()
    base = dict(_SMALL_WTMM, wavelet="mexican", min_chain_len=2, boundary="auto",
               roi_row=0, roi_col=0, roi_h=32, roi_w=32)
    other = dict(base, roi_row=16, roi_col=16)
    assert device.cache_key("src0", base) != device.cache_key("src0", other)


def test_same_roi_and_params_produce_the_same_cache_key():
    device = WTMM2DROI()
    params = dict(_SMALL_WTMM, wavelet="mexican", min_chain_len=2, boundary="auto",
                  roi_row=8, roi_col=8, roi_h=32, roi_w=32)
    assert device.cache_key("src0", params) == device.cache_key("src0", dict(params))


# --------------------------------------------------------------------------- provenance guard


def test_missing_provenance_source_raises_a_clear_valueerror():
    device = WTMM2DROI()
    params = validate_params(device, {})
    with pytest.raises(ValueError, match="provenance"):
        device.compute(_bare_field(), params)


# --------------------------------------------------------------------------- real compute


def test_compute_on_a_synthetic_parent_returns_canonical_keys_and_roi_extras(parent_field):
    device = WTMM2DROI()
    params = validate_params(device, dict(
        _SMALL_WTMM, roi_row=16, roi_col=16, roi_h=32, roi_w=32, boundary="auto"))

    result = device.compute(parent_field, params)

    for key in ("chains", "extrema", "scales", "hd_std", "hd_cmax", "npz_path", "params",
               "cache_hits"):
        assert key in result, key
    assert result["_shape"] == (32, 32)
    assert result["_roi"]["roi"] == (16, 16, 32, 32)
    assert result["_roi"]["boundary"] == "auto"
    assert len(result["_roi_margins"]) == len(result["scales"]) == 2
    assert result["_frame"] is parent_field.frame


def test_compute_consumes_a_stamped_full_dims_without_reading_geotiff_info(parent_field,
                                                                           monkeypatch):
    """dynamix.shell.opening stamps ``provenance["full_dims"]`` on every field it opens;
    the device must use that directly rather than re-deriving the same numbers with a second
    ``geotiff_info`` read -- one fewer info read per compute. Proven by spy: with the stamp
    present, ``geotiff_info`` must never be called at all."""
    import dynamix.core.rasterfield as rasterfield_module

    parent_field.provenance["full_dims"] = (80, 80)

    def _refuse(*a, **k):
        raise AssertionError("geotiff_info must not be called when full_dims is already stamped")
    monkeypatch.setattr(rasterfield_module, "geotiff_info", _refuse)

    device = WTMM2DROI()
    params = validate_params(device, dict(
        _SMALL_WTMM, roi_row=16, roi_col=16, roi_h=32, roi_w=32, boundary="auto"))

    result = device.compute(parent_field, params)

    assert result["_shape"] == (32, 32)
    assert result["_roi"]["roi"] == (16, 16, 32, 32)


def test_compute_does_not_overwrite_the_rois_own_shape_with_the_parents(parent_field):
    """run_wtmm2d_roi already set _shape to the ROI's own (h, w) -- the device must not stomp it
    with the parent field's shape the way wtmm2d.compute does for the whole-raster case."""
    device = WTMM2DROI()
    params = validate_params(device, dict(
        _SMALL_WTMM, roi_row=16, roi_col=16, roi_h=32, roi_w=32, boundary="auto"))

    result = device.compute(parent_field, params)

    assert result["_shape"] == (32, 32)
    assert result["_shape"] != tuple(parent_field.values.shape)


# --------------------------------------------------------------------------- zero-miss law


def test_zero_miss_law_holds_for_a_downstream_filter_change(clean_registry, parent_tif):
    """THE property this device split exists for: turning scale_select's knob after wtmm2d_roi
    has run must reuse the cached ROI transform, not recompute it. Mirrors
    tests/test_shell_window.py's test_filter_knob_drag_is_zero_miss, at the engine level."""
    register_device(WTMM2DROI())
    register_device(ScaleSelect())

    project = Project()
    src = project.add_source(parent_tif)
    field = RasterField.from_geotiff_window(parent_tif, row_off=0, col_off=0, height=80, width=80)

    wtmm_step = DeviceRef("wtmm2d_roi", dict(
        _SMALL_WTMM, roi_row=16, roi_col=16, roi_h=32, roi_w=32))
    layer = project.add_layer(
        "roi", src.source_id, Chain((wtmm_step, DeviceRef("scale_select", {"scale_idx": 0}))))

    cache = Cache()
    first = resolve(layer, field, cache)
    assert first.cache_misses == 1
    assert first.result["_scale_idx"] == 0

    layer.chain = Chain(
        (wtmm_step, DeviceRef("scale_select", {"scale_idx": 1}))
    ).materialized()
    second = resolve(layer, field, cache)

    assert second.cache_misses == 0
    assert second.cache_hits >= 1
    assert second.result["_scale_idx"] == 1


def test_roi_compute_stamps_the_chain_product(parent_field):
    """The ROI transform stamps the same draw-ready CSR bundle the
    whole-raster transform does -- worker-side, over the ROI's own shape."""
    from dynamix.core.chain_product import build_chain_product

    device = WTMM2DROI()
    params = validate_params(device, dict(
        _SMALL_WTMM, roi_row=16, roi_col=16, roi_h=32, roi_w=32, boundary="auto"))

    result = device.compute(parent_field, params)

    p = result.get("chain_product")
    assert p is not None, "wtmm2d_roi must stamp result['chain_product']"
    assert result.get("_hline_runs") is not None
    ref = build_chain_product(result["extrema"], result["chains"], result["scales"],
                              result["_shape"], runs=result["_hline_runs"])
    assert set(p) == set(ref)
    for key in ref:
        np.testing.assert_array_equal(np.asarray(p[key]), np.asarray(ref[key]),
                                      err_msg=f"key {key!r} differs")
