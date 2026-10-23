# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Independent-axis analysis-filter previews for the hyperbolic DWT basis."""
from __future__ import annotations

import functools

import numpy as np

from dynamix.core import dwt


@functools.lru_cache(maxsize=48)
def axis_preview(bank: str, index: int, depth: int) -> dict:
    """Exact equivalent analysis filter and response; index 0 is the coarsest scaling branch.

    ``taps`` are the cascade in convolution order. A coefficient is the inner product of the
    signal with those taps reversed, so ``profile`` is the reversed, peak-normalised atom.
    The frequency response is evaluated from the filter polynomial, without an FFT or source
    resampling. Profiles are centred on their energy centroid for display only.
    """
    index, depth = int(index), int(depth)
    if not 1 <= depth <= 10 or not 0 <= index <= depth:
        raise ValueError("hyperbolic preview needs depth 1..10 and index 0..depth")
    wavelet = dwt.wavelet(bank)
    level = index or depth
    h, g = np.asarray(wavelet.dec_lo), np.asarray(wavelet.dec_hi)
    taps = np.ones(1)
    for stage in range(level):
        filt = g if index and stage == level-1 else h
        step = 2**stage
        expanded = np.zeros(taps.size + (filt.size-1)*step)
        for k, tap in enumerate(filt):
            expanded[k*step:k*step+taps.size] += tap*taps
        taps = expanded
    atom = taps[::-1]
    energy = atom*atom
    centre = float(np.dot(np.arange(atom.size), energy)/energy.sum())
    # Resolve the narrowest coarse-scale passband without a large dense Fourier matrix.
    frequency = np.linspace(0., .5, max(2049, 32 * 2**level + 1))
    response = np.ones(frequency.size, dtype=complex)
    for stage in range(level):
        filt = (g if index and stage == level-1 else h)
        angle = -2j*np.pi*frequency*2**stage
        response *= sum(tap*np.exp(angle*k) for k, tap in enumerate(filt))
    magnitude = np.abs(response)
    magnitude /= max(float(magnitude.max()), np.finfo(float).tiny)
    peak = int(np.argmax(magnitude))
    wavelength = float(1/frequency[peak]) if index and frequency[peak] > 0 else float('nan')
    return {"bank": bank, "index": index, "level": level, "branch": "ψ" if index else "φ",
            "x": np.arange(atom.size)-centre, "taps": taps,
            "profile": atom / max(float(np.abs(atom).max()), np.finfo(float).tiny),
            "frequency": frequency, "magnitude": magnitude, "wavelength": wavelength,
            "cell_px": 2**level, "support_px": taps.size}


def tensor_preview(x: dict, y: dict, limit: int = 192) -> dict:
    """A bounded display of the tensor-product filter, with its original pixel coordinates."""
    sx = max(1, int(np.ceil(x['taps'].size/limit)))
    sy = max(1, int(np.ceil(y['taps'].size/limit)))
    return {"values": np.outer(y['profile'][::sy], x['profile'][::sx]),
            "origin": (float(x['x'][0]), float(y['x'][0])), "pitch": (sx, sy)}
