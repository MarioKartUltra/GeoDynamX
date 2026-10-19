# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Fourier fractional integration, distinct from a wavelet coefficient lift.

Bessel potential: (1 + (length_px |k|)^2)^(-s/2), Dumeur et al.
(2025), Eq. 12. Riesz uses (length_px |k|)^(-s) at nonzero
frequencies. Both preserve the mean. k is angular frequency per pixel.
"""
from __future__ import annotations

import numpy as np

from dynamix.core import fft_policy


def integrate(values, order, *, operator="bessel", length_px=0., edge="mirror"):
    x = np.asarray(values, dtype=float)
    s = float(order)
    if not np.isfinite(s) or abs(s) > 12:
        raise ValueError("fractional order must be finite and within -12..12")
    if s == 0:
        return x.copy()
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("fractional integration needs a finite 2D band; fill nodata explicitly first")
    if operator not in ("bessel", "riesz") or edge not in ("mirror", "periodic"):
        raise ValueError("integration requires bessel/riesz and mirror/periodic boundaries")
    length = float(length_px) or min(x.shape)/(2*np.pi)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("integration reference length must be finite and positive")
    # One even reflection per axis creates a periodic working image. This is
    # a whole-domain operation: no finite halo gives the same fractional field.
    work = np.pad(x, ((0, x.shape[0]), (0, x.shape[1])), mode="symmetric") if edge == "mirror" else x
    k = np.hypot(2*np.pi*np.fft.fftfreq(work.shape[0])[:, None],
                 2*np.pi*np.fft.rfftfreq(work.shape[1])[None, :])
    if operator == "bessel":
        gain = (1+(length*k)**2)**(-s/2)
    else:
        with np.errstate(divide="ignore"):
            gain = np.where(k > 0, np.where(k > 0, length*k, 1.)**(-s), 1.)
    fft = fft_policy.double()
    spectrum = np.asarray(fft.rfft2(work))
    y = np.asarray(fft.irfft2(spectrum*gain, s=work.shape)).real
    return y[:x.shape[0], :x.shape[1]]
