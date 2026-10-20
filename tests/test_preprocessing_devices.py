# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pytest

from conftest import make_field
from dynamix.devices.interpolation import (DAFSmoothing, DyadicInterpolation,
                                           InterpolatingWaveletFilter, InterpolatingWavelets)
from dynamix.model.device import defaults_for
from dynamix.model.contracts import FIELD, STACK
from dynamix.engine.cache import StageStore


def test_refined_axes_and_provenance_preserve_original_sample_coordinates():
    field = make_field(shape=(12, 15))
    dev = DyadicInterpolation()
    p = {**defaults_for(dev), "levels": 2}
    out = dev.compute(field, p)
    np.testing.assert_array_equal(out.values[::4, ::4], field.values)
    np.testing.assert_allclose(out.x_axis[::4], field.x_axis)
    np.testing.assert_allclose(out.y_axis[::4], field.y_axis)
    assert out.provenance["preprocessing"][-1]["derived_sampling"]


def test_refined_grid_forks_before_analysis_and_keeps_history(clean_registry):
    from conftest import make_layer
    from dynamix.devices import register_builtin_devices
    from dynamix.core.derivative import carried_from
    from dynamix.engine.resolve import resolve
    from dynamix.engine.cache import Cache

    register_builtin_devices()
    field = make_field(shape=(32, 32))
    layer = make_layer(("dyadic_interpolation", {}), ("interpolating_wavelets", {}))
    with pytest.raises(ValueError, match="Fork derivative dataset"):
        resolve(layer, field, Cache())
    produced = DyadicInterpolation().compute(field, defaults_for(DyadicInterpolation()))
    carried = carried_from(produced)
    assert "preprocessing" in carried and "grid.requires_fork" not in carried


def test_wavelet_threshold_reuses_coefficients_and_publishes_additive_bands(monkeypatch):
    import dynamix.core.interpolating as iw

    field = make_field(shape=(31, 37))
    calls = []
    original = iw.pyramid
    def counted(*args, **kw):
        calls.append(1)
        return original(*args, **kw)
    monkeypatch.setattr(iw, "pyramid", counted)
    cache = StageStore()
    dev = InterpolatingWaveletFilter()
    p = defaults_for(dev)
    zero = dev.compute(field, p, stage_cache=cache)
    np.testing.assert_allclose(zero.values, field.values, atol=1e-13)
    filtered = dev.compute(field, {**p, "threshold": .1}, stage_cache=cache)
    assert len(calls) == 1 and not np.allclose(filtered.values, zero.values)
    analysis = InterpolatingWavelets()
    ap = defaults_for(analysis)
    res = analysis.compute(field, ap, stage_cache=cache)
    assert len(calls) == 1
    total = res["raster_out"].astype(float)
    for j in range(3):
        for band in ("horizontal", "vertical", "diagonal"):
            total += analysis.compute_output(band, field.values, res, ap, fetch=None, level=j)["raster"]
    np.testing.assert_allclose(total, field.values, atol=2e-5)


def test_daf_smoothing_preserves_dc_and_matches_analytic_periodic_gain():
    from dynamix.core import fft_policy
    from dynamix.core.daf import daf_gain

    fft_policy.configure("pyfftw", 64)
    x = np.broadcast_to(np.cos(2*np.pi*np.arange(64)/8), (32, 64))
    field = make_field(values=3+x)
    dev = DAFSmoothing()
    p = {**defaults_for(dev), "edge": "periodic", "sigma": 4.}
    out = dev.compute(field, p)
    np.testing.assert_allclose(out.values, 3+daf_gain(2*np.pi/8, 4., 12)*x, atol=2e-13)


@pytest.mark.parametrize("edge", ["reflect", "periodic"])
def test_one_dd_refinement_of_512_is_1023_with_half_spacing(edge):
    field = make_field(shape=(512, 512))
    dev = DyadicInterpolation()
    p = {**defaults_for(dev), "edge": edge}
    out = dev.compute(field, p)
    assert out.values.shape == (1023, 1023)
    np.testing.assert_array_equal(out.values[::2, ::2], field.values)
    np.testing.assert_allclose(out.x_axis[::2], field.x_axis)
    assert out.x_axis[1] - out.x_axis[0] == pytest.approx((field.x_axis[1] - field.x_axis[0])/2)
    assert dev.derived_reading("levels", 1, field, p).startswith("1023 × 1023")


@pytest.mark.parametrize("edge", ["reflect", "periodic"])
def test_dd_on_saved_roi_uses_region_pixels_and_axes(clean_registry, edge):
    from conftest import make_layer
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.resolve import resolve
    from dynamix.engine.cache import Cache

    register_builtin_devices()
    field = make_field(shape=(40, 50))
    layer = make_layer(("dyadic_interpolation", {"edge": edge}), window="3,4,12,15")
    out = resolve(layer, field, Cache()).result
    assert out.values.shape == (23, 29)
    np.testing.assert_array_equal(out.values[::2, ::2], field.values[3:15, 4:19])
    np.testing.assert_allclose(out.x_axis[::2], field.x_axis[4:19])


def test_dyadic_interpolation_refines_each_band_of_a_stack():
    """A colour picture (a band stack) refines band by band: the result is the stack of the
    single-band refinements, on the refined grid, with the source shape recorded per plane."""
    rng = np.random.default_rng(4)
    stack = rng.random((12, 15, 3))
    field = make_field(values=stack)
    dev = DyadicInterpolation()
    p = {**defaults_for(dev), "levels": 2}
    out = dev.compute(field, p)
    assert out.values.shape == (45, 57, 3)
    for b in range(3):
        single = dev.compute(make_field(values=stack[..., b]), p)
        np.testing.assert_array_equal(out.values[..., b], single.values)
    assert out.provenance["preprocessing"][-1]["source_shape"] == [12, 15]
    assert (FIELD,) in DyadicInterpolation.consumes and (STACK,) in DyadicInterpolation.consumes


def test_daf_smoothing_runs_band_by_band():
    from dynamix.core import fft_policy

    fft_policy.configure("pyfftw", 64)
    x = np.broadcast_to(np.cos(2 * np.pi * np.arange(64) / 8), (32, 64))
    stack = np.stack([3 + x, 1 + 2 * x], axis=-1)
    dev = DAFSmoothing()
    p = {**defaults_for(dev), "edge": "periodic", "sigma": 4.}
    out = dev.compute(make_field(values=stack), p)
    assert out.values.shape == (32, 64, 2)
    for b in range(2):
        np.testing.assert_allclose(out.values[..., b], dev.compute(make_field(values=stack[..., b]), p).values)
