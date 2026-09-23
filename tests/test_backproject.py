# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.devices.backproject -- the Transform that registers a point layer's lon/lat
onto ANOTHER layer's pixel grid, from seven shell-stamped scalars alone.

Fixture reuse: the target raster is ``tests/test_geo_mapping.py``'s own NAD27 Transverse-Mercator,
US-survey-foot GeoTIFF fixture (``_write_boem_like_tif``/``_load_field``) -- the established
cross-file fixture-reuse pattern this suite already uses (``tests/test_arrangement_scene.py`` does
the same). Skipped honestly, whole-module, when ``rasterio`` is absent -- the device's own
compute() lazily imports it, same as ``dynamix.geo.mapping``.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

from dynamix.core.pointset import PointSet
from dynamix.devices.backproject import Backproject
from dynamix.geo.mapping import lonlat_to_pixels, points_lonlat

from tests.test_geo_mapping import _load_field, _write_boem_like_tif


def _target_field(tmp_path, name="target.tif"):
    path = tmp_path / name
    _write_boem_like_tif(path)
    return _load_field(path)


def _scalars_from(field, target_name="raster") -> dict:
    """The seven stamped scalars a target's own axes/CRS produce -- the SAME derivation
    ``main_window._backproject_scalars_for`` performs (duplicated here, deliberately: this test
    must stay independent of that method's own bug, or a shared helper could make both sides wrong
    together)."""
    nx, ny = int(field.nx), int(field.ny)
    dx = (float(field.x_axis[-1]) - float(field.x_axis[0])) / (nx - 1)
    dy = (float(field.y_axis[-1]) - float(field.y_axis[0])) / (ny - 1)
    return {"target": target_name, "_target_crs": field.provenance["crs"],
            "_target_x0": float(field.x_axis[0]), "_target_dx": dx,
            "_target_y0": float(field.y_axis[0]), "_target_dy": dy,
            "_target_nx": nx, "_target_ny": ny}


_ZERO_PARAMS = {"target": "", "_target_crs": "", "_target_x0": 0.0, "_target_dx": 0.0,
                "_target_y0": 0.0, "_target_dy": 0.0, "_target_nx": 0, "_target_ny": 0}


# ------------------------------------------------------------------------------ registration


def test_backproject_is_a_builtin_device(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES, is_transform

    register_builtin_devices()
    assert "backproject" in DEVICES
    assert is_transform(DEVICES["backproject"])


# ------------------------------------------------------------------------------ numerics


def test_points_px_matches_lonlat_to_pixels_against_the_real_field(tmp_path):
    """The whole point of the scalar-reconstruction design: a device that never sees the target
    field object must still land pixel-for-pixel where `lonlat_to_pixels` (run directly against
    the real field) does."""
    field = _target_field(tmp_path)
    cols = np.array([0, 5, 20, 63], dtype=np.intp)
    rows = np.array([0, 10, 40, 63], dtype=np.intp)
    lon, lat = points_lonlat(field, cols, rows)
    pset = PointSet(lon=lon, lat=lat)
    params = _scalars_from(field)

    result = Backproject().compute(pset, params)

    expected_cols, expected_rows = lonlat_to_pixels(field, lon, lat)
    np.testing.assert_allclose(result["points_px"]["x"], expected_cols, atol=1e-6)
    np.testing.assert_allclose(result["points_px"]["y"], expected_rows, atol=1e-6)
    # And also recovers the original pixel indices themselves, to the same tolerance.
    np.testing.assert_allclose(result["points_px"]["x"], cols.astype(np.float64), atol=1e-4)
    np.testing.assert_allclose(result["points_px"]["y"], rows.astype(np.float64), atol=1e-4)


def test_result_carries_target_name_and_shape(tmp_path):
    field = _target_field(tmp_path)
    lon, lat = points_lonlat(field, [0], [0])
    pset = PointSet(lon=lon, lat=lat)
    params = _scalars_from(field, target_name="my raster")

    result = Backproject().compute(pset, params)

    assert result["_target"] == "my raster"
    assert result["_shape"] == (field.ny, field.nx)
    assert "_unbound" not in result


def test_attrs_pass_through_unchanged(tmp_path):
    field = _target_field(tmp_path)
    lon, lat = points_lonlat(field, [0, 5], [0, 5])
    pset = PointSet(lon=lon, lat=lat, attrs={"mag": np.array([3.1, 4.2])})
    params = _scalars_from(field)

    result = Backproject().compute(pset, params)

    np.testing.assert_array_equal(result["attrs"]["mag"], [3.1, 4.2])


# ------------------------------------------------------------------------------ inside mask


def test_inside_mask_true_within_grid_false_off_raster(tmp_path):
    """Interior pixels (never the exact 0/(n-1) edge -- the CRS round trip's own float noise can
    tip an EXACT boundary pixel a hair negative, e.g. -3.6e-6, the same tolerance
    ``test_geo_mapping.py``'s own round-trip test accepts via ``atol=1e-4``) map back inside;
    a point shifted a whole degree west -- far beyond that noise floor -- does not."""
    field = _target_field(tmp_path)
    cols = np.array([5, 58], dtype=np.intp)
    rows = np.array([5, 58], dtype=np.intp)
    lon_in, lat_in = points_lonlat(field, cols, rows)
    # One degree west of an in-bounds point -- definitely off the raster (mirrors
    # test_geo_mapping.py's own off-raster construction).
    lon_off = lon_in[:1] - 1.0
    lat_off = lat_in[:1]
    lon = np.concatenate([lon_in, lon_off])
    lat = np.concatenate([lat_in, lat_off])
    pset = PointSet(lon=lon, lat=lat)
    params = _scalars_from(field)

    result = Backproject().compute(pset, params)

    assert result["inside"].tolist() == [True, True, False]


# ------------------------------------------------------------------------------ unbound no-op


def test_empty_target_is_an_honest_unbound_no_op():
    pset = PointSet(lon=np.array([1.0, 2.0]), lat=np.array([3.0, 4.0]))

    result = Backproject().compute(pset, dict(_ZERO_PARAMS))

    assert result["points_px"] is None
    assert result["_unbound"] is True
    assert result["_target"] == ""
    assert result["_shape"] == (0, 0)
    assert result["inside"].shape == (2,)
    assert not result["inside"].any()


def test_named_target_with_zeroed_grid_spec_is_still_unbound():
    """A target NAME can be present (the shell always stamps zeros alongside its refusal warning,
    never blanking the text the user typed) while the grid spec itself is zeroed -- still an
    honest no-op, not a crash from dividing by a zero step."""
    pset = PointSet(lon=np.array([1.0]), lat=np.array([2.0]))
    params = dict(_ZERO_PARAMS, target="no such layer")

    result = Backproject().compute(pset, params)

    assert result["points_px"] is None
    assert result["_unbound"] is True
    assert result["_target"] == "no such layer"


def test_unbound_result_has_the_same_keys_as_the_bound_one(tmp_path):
    field = _target_field(tmp_path)
    lon, lat = points_lonlat(field, [0], [0])
    pset = PointSet(lon=lon, lat=lat)
    bound = Backproject().compute(pset, _scalars_from(field))
    unbound = Backproject().compute(pset, dict(_ZERO_PARAMS))

    assert set(unbound) - {"_unbound"} == set(bound)


# ------------------------------------------------------------------------------ cache_key


def test_cache_key_changes_when_a_stamped_scalar_changes(tmp_path):
    dev = Backproject()
    field = _target_field(tmp_path)
    base = _scalars_from(field)
    retargeted = dict(base, _target_x0=base["_target_x0"] + 1.0)

    k1 = dev.cache_key("mem:points", base)
    k2 = dev.cache_key("mem:points", retargeted)

    assert k1 != k2


def test_cache_key_stable_for_identical_params(tmp_path):
    dev = Backproject()
    field = _target_field(tmp_path)
    params = _scalars_from(field)

    assert dev.cache_key("mem:points", dict(params)) == dev.cache_key("mem:points", dict(params))
