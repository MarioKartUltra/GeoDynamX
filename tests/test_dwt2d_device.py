# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dwt2d: the periodized orthogonal pyramid as a Decomposition tool with per-level rows."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pywt")

from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402
from dynamix.devices.dwt2d import DWT2D  # noqa: E402
from dynamix.model.device import Output, defaults_for, validate_params  # noqa: E402


def _field(ny=64, nx=96, seed=0):
    v = np.random.default_rng(seed).standard_normal((ny, nx)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(), x_axis=np.arange(float(nx)),
                       y_axis=np.arange(float(ny)))


def _params(**over):
    dev = DWT2D()
    return validate_params(dev, {**defaults_for(dev), **over})


def test_compute_gives_the_coarsest_approximation_and_the_dyadic_scales():
    dev, f = DWT2D(), _field()
    res = dev.compute(f, _params(level=3))
    assert res["raster_out"].shape == (64, 96) and res["chains"] == [] and res["extrema"] == []
    np.testing.assert_array_equal(res["scales"], [2.0, 4.0, 8.0])
    assert res["_dwt_levels"] == 3 and res["_shape"] == (64, 96)
    assert abs(res["raster_out"].mean() - f.values.mean()) < 1e-3 * abs(f.values).max()
    res0 = dev.compute(f, _params(level=0))
    assert res0["_dwt_levels"] == 6                         # full depth of the 128 x 192 working image


def test_a_stack_is_refused_and_the_declaration_says_field_only():
    """The contract-honesty test (Task 6.3) pins that the refusal and the declaration agree."""
    dev = DWT2D()
    stack = RasterField(name="s", values=np.zeros((32, 32, 3)), frame=LocalFrame(),
                        x_axis=np.arange(32.0), y_axis=np.arange(32.0))
    with pytest.raises(ValueError, match="single-band"):
        dev.compute(stack, _params())
    assert dev.consumes == (("Field",),)


def test_the_per_level_bands_sum_back_to_the_field():
    dev, f = DWT2D(), _field()
    params = _params(level=3)
    res = dev.compute(f, params)
    parts = [dev.compute_output("approximation", f.values, res, params, fetch=None, level=2)["raster"]]
    for level in range(3):
        for band in ("horizontal", "vertical", "diagonal"):
            out = dev.compute_output(band, f.values, res, params, fetch=None, level=level)
            assert out["diag"] == {"level": level, "scale_px": 2.0 ** (level + 1)}
            parts.append(out["raster"])
    np.testing.assert_allclose(sum(p.astype(np.float64) for p in parts), f.values,
                               atol=1e-4 * abs(f.values).max())
    approx0 = dev.compute_output("approximation", f.values, res, params, fetch=None, level=0)
    assert approx0["raster"].shape == (64, 96) and approx0["diag"]["scale_px"] == 2.0


def test_cycle_spinning_averages_shifts_and_keeps_the_sum_identity():
    dev, f = DWT2D(), _field()
    params = _params(level=2, cycle_spin=True, n_spins=4)
    res = dev.compute(f, params)
    a = dev.compute_output("horizontal", f.values, res, params, fetch=None, level=0)["raster"]
    b = dev.compute_output("horizontal", f.values, res, _params(level=2), fetch=None, level=0)["raster"]
    assert not np.allclose(a, b)
    parts = [dev.compute_output("approximation", f.values, res, params, fetch=None, level=1)["raster"]]
    parts += [dev.compute_output(band, f.values, res, params, fetch=None, level=level)["raster"]
              for level in range(2) for band in ("horizontal", "vertical", "diagonal")]
    np.testing.assert_allclose(sum(p.astype(np.float64) for p in parts), f.values,
                               atol=1e-4 * abs(f.values).max())


def test_a_level_past_the_stack_reads_the_coarsest_and_an_unknown_output_is_refused():
    dev, f = DWT2D(), _field()
    params = _params(level=2)
    res = dev.compute(f, params)
    out = dev.compute_output("diagonal", f.values, res, params, fetch=None, level=9)
    assert out["diag"]["level"] == 1
    with pytest.raises(ValueError, match="no lazy output"):
        dev.compute_output("modulus", f.values, res, params, fetch=None, level=0)


def test_declarations_outputs_margin_and_keys():
    dev = DWT2D()
    assert [o.name for o in dev.outputs] == ["approximation", "horizontal", "vertical", "diagonal"]
    assert all(isinstance(o, Output) and o.lazy and o.per_level and o.kind == "raster"
               for o in dev.outputs)
    assert dev.roi_margin(_params(level=0)) == 14 * 8 and dev.roi_margin(_params(level=2)) == 14 * 4
    assert dev.roi_margin(_params(wavelet="bl3", level=0)) == 256
    assert dev.view({"raster_out": 1}, _params()) == {"raster_out": 1}
    assert dev.cache_key("s", _params()) == dev.cache_key("s", _params(show="diagonal"))
    assert dev.cache_key("s", _params()) != dev.cache_key("s", _params(level=2))


def test_nan_pixels_are_filled_for_the_transform_and_restored():
    dev, f = DWT2D(), _field()
    v = f.values.copy()
    v[10:20, 10:20] = np.nan
    g = RasterField(name="g", values=v, frame=f.frame, x_axis=f.x_axis, y_axis=f.y_axis)
    res = dev.compute(g, _params(level=2))
    assert np.isnan(res["raster_out"][15, 15]) and np.isfinite(res["raster_out"][0, 0])
    out = dev.compute_output("horizontal", g.values, res, _params(level=2), fetch=None, level=0)
    assert np.isnan(out["raster"][15, 15]) and np.isfinite(out["raster"][40, 40])
