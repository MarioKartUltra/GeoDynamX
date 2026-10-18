# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The padded Edge modes against a direct convolution (spec 25 §5, test 3). The FFT's convolution
is circular, so a padded mode is right only where the pad is wide enough that nothing wraps round.
The reference convolves directly with the analysing wavelet's taps and xsmurf's index rules beyond
the raster (mirror, clamp, zero), separably: the Gaussian's x-analyser is g'(x) g(y), the Mexican
hat's a sum of two such products. Its taps are the 1-D inverse transforms of the backend's own
filters on a long grid, so the two differ only at the boundary.

At ε = 10⁻⁶ the measured reach brings the moduli to float32 tolerance of the scale's largest; at the
default ε = 10⁻³ to that order, since ε of the kernel's mass is left outside by definition. The old
fixed 32 px leaves the 64 px scale 8 to 21 % off near the edges."""
from __future__ import annotations

import numba
import numpy as np
import pytest

from dynamix.core.edges import pad_widths
from dynamix.core.wtmm_backend import get_backend
from tests.test_periodic_extension import _smooth

#: The taps' grid, and their half-length: past every reach measured here.
_GRID, _HALF = 2048, 300
_RULES = {"reflect": 0, "edge_pad": 1, "zero_pad": 2}


@numba.njit(cache=True)
def _index(i, n, rule):
    """xsmurf's index rule beyond ``[0, n)``: mirror without repeating the edge sample (period
    2(n - 1)), clamp to the edge, or -1 for zero."""
    if 0 <= i < n:
        return i
    if rule == 1:
        return 0 if i < 0 else n - 1
    if rule == 2:
        return -1
    p = 2 * (n - 1)
    i = abs(i) % p
    return i if i < n else p - i


@numba.njit(cache=True)
def _separable(f, hx, hy, half, rule):
    """``Σ f[rule(i - m), rule(j - n)] hy[m] hx[n]`` over ``|m|, |n| ≤ half``: rows first, over the
    rows the column pass reads."""
    H, W = f.shape
    rows = np.zeros((H + 2 * half, W))
    for a in range(H + 2 * half):
        ii = _index(a - half, H, rule)
        if ii < 0:
            continue
        for j in range(W):
            s = 0.0
            for k in range(2 * half + 1):
                jj = _index(j - (k - half), W, rule)
                if jj >= 0:
                    s += f[ii, jj] * hx[k]
            rows[a, j] = s
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            s = 0.0
            for k in range(2 * half + 1):
                s += rows[i + 2 * half - k, j] * hy[k]
            out[i, j] = s
    return out


def _taps(spectrum, scale):
    """The taps ``h[-half..half]`` of a 1-D filter ``spectrum(k · scale)`` (k in cycles/sample)."""
    from dynamix.core import fft_policy

    k = np.fft.fftfreq(_GRID) * scale
    h = np.asarray(fft_policy.double().ifft(np.asarray(spectrum(k), dtype=np.complex128)))
    taps = np.concatenate([h[-_HALF:], h[:_HALF + 1]])
    part = taps.imag if np.abs(taps.imag).max() > np.abs(taps.real).max() else taps.real
    assert np.abs(part[[0, -1]]).max() < 1e-12 * np.abs(part).max()              # past its reach
    return np.ascontiguousarray(part)


def _direct_modulus(f, scale, wavelet, mode):
    """The modulus from the direct path. The x-analyser's filter is ``i sx θ̂``: the Gaussian
    ``i sx e^{-sx²} · e^{-sy²}``, the Mexican hat ``i sx (sx² + sy²) e^{-sx²-sy²}``; each term's x
    factor is odd and imaginary in frequency, so its taps are real after the ``i``."""
    g = lambda s: np.exp(-s * s)
    if wavelet == "gaussian":
        terms = [(lambda s: s * g(s), g)]
    else:
        terms = [(lambda s: s ** 3 * g(s), g), (lambda s: s * g(s), lambda s: s * s * g(s))]
    rule = _RULES[mode]

    def component(values):
        # the filter is i·(odd real); its inverse transform is -(the odd factor's imaginary taps)
        return sum(-_separable(values, _taps(a, scale), _taps(b, scale), _HALF, rule)
                   for a, b in terms)

    tx = component(f)
    ty = component(np.ascontiguousarray(f.T)).T
    return np.hypot(tx, ty)


@pytest.mark.parametrize("wavelet", ["gaussian", "mexican"])
@pytest.mark.parametrize("mode", ["reflect", "edge_pad", "zero_pad"])
def test_a_padded_mode_is_the_direct_convolution_once_the_pad_is_the_measured_reach(
        double_fft, wavelet, mode):
    f = _smooth(128, seed=3)
    scales = [16.0, 64.0]
    backend = get_backend("python")
    direct = [_direct_modulus(f, s, wavelet, mode) for s in scales]

    def worst(pad):
        mod = backend.cwt2d(f, scales, wavelet=wavelet, pad=pad, pad_mode=mode)["mod"]
        return [float(np.abs(np.asarray(m) - d).max() / d.max()) for m, d in zip(mod, direct)]

    tight, _ = pad_widths(scales, "measured_reach", wavelet=wavelet, eps=1e-6)
    default, _ = pad_widths(scales, "measured_reach", wavelet=wavelet, eps=1e-3)
    assert max(worst(max(tight))) < 1e-6
    assert max(worst(max(default))) < 1e-3
    if mode != "zero_pad":           # zeros wrap round as zeros: any pad holds for them
        assert worst(32)[1] > 1e-2
