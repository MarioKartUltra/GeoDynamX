# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Periodic extension (DAF), spec 25 §3.1: the field plus a DAF-filled gap after its last row and
column, so that the whole is periodic; the real block is stamped so a scale-space tool after it
keeps no maxima in the gap; below the DAF's band limit the field passes through with the reason."""
from __future__ import annotations

import numpy as np

from dynamix.devices import register_builtin_devices
from dynamix.model.device import defaults_for, get_device
from tests.conftest import make_field


def _device():
    register_builtin_devices()
    return get_device("periodic_extension")


def _smooth(n=192, seed=1):
    """A band-limited field large enough for the DAF stencil and its gap (spec 22 §3.4): a random
    spectrum low-passed and transformed through the FFT policy, then cut out of the larger result so
    that it is not periodic (an FFT-built field alone would be)."""
    from dynamix.core import fft_policy

    big = 2 * n
    rng = np.random.default_rng(seed)
    k = np.hypot(np.fft.fftfreq(big)[:, None], np.fft.rfftfreq(big)[None, :])
    spectrum = (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) * np.exp(-(k / 0.1) ** 2)
    smooth = fft_policy.double().irfft2(spectrum, s=(big, big))
    smooth = smooth[n // 3:n // 3 + n, n // 5:n // 5 + n]
    return (smooth - smooth.mean()) / smooth.std()


def test_the_output_is_the_field_plus_a_filled_gap_with_its_real_block_stamped(clean_registry):
    dev = _device()
    f = make_field(_smooth())
    out = dev.compute(f, defaults_for(dev))
    H, W = f.values.shape
    gy, gx = out.values.shape[0] - H, out.values.shape[1] - W
    assert gy > 0 and gx > 0
    np.testing.assert_array_equal(out.values[:H, :W], f.values)
    assert out.provenance["edges.filled"] == {"rows": H, "cols": W}
    assert len(out.x_axis) == W + gx and len(out.y_axis) == H + gy
    assert np.allclose(np.diff(out.x_axis), np.diff(f.x_axis)[0])


def test_the_extended_field_is_periodic_across_its_wrap(clean_registry):
    dev = _device()
    f = make_field(_smooth())
    v = dev.compute(f, defaults_for(dev)).values
    neighbour = np.sqrt(np.mean(np.diff(v, axis=0) ** 2) + np.mean(np.diff(v, axis=1) ** 2))
    wrap = np.sqrt(np.mean((v[0] - v[-1]) ** 2) + np.mean((v[:, 0] - v[:, -1]) ** 2))
    plain = f.values
    seam = np.sqrt(np.mean((plain[0] - plain[-1]) ** 2) + np.mean((plain[:, 0] - plain[:, -1]) ** 2))
    assert wrap <= 2.0 * neighbour and seam > 3.0 * wrap


def test_below_the_band_limit_the_field_passes_through_with_the_reason(clean_registry):
    dev = _device()
    noise = make_field(np.random.default_rng(0).standard_normal((64, 64)))
    out = dev.compute(noise, defaults_for(dev))
    np.testing.assert_array_equal(out.values, noise.values)
    assert "edges.filled" not in out.provenance
    warnings = dev.knob_warnings(noise, defaults_for(dev))
    assert warnings and "points per wavelength" in next(iter(warnings.values()))
    assert dev.knob_warnings(make_field(_smooth()), defaults_for(dev)) == {}


def test_it_sits_in_data_preparation(clean_registry):
    from dynamix.model.families import families_of

    dev = _device()
    assert dev.label == "Periodic extension (DAF)" and families_of(dev) == ("Data preparation",)
