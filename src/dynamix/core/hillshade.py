# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Shaded relief of a 2-D field — the classic hillshade, headless (2026-08-29).

ESRI/Horn convention over a NORTH-UP array (row index increases southward, column index
eastward): ``dz/dx`` from east minus west, ``dz/dy`` from south minus north, slope from the
gradient magnitude scaled by ``z_factor`` (units of z per unit of x/y — 1.0 when elevation and
pixel spacing share a unit), aspect as the downslope direction, and

    shade = cos(zenith)·cos(slope) + sin(zenith)·sin(slope)·cos(azimuth − aspect)

clipped to [0, 1]. The field's own pixel spacing (``dx``, ``dy`` in the frame's units) is what
makes the slope physical -- the caller passes ``field.frame.dx``-style values, never 1 px.

A DISPLAY product: the shell multiplies a colormap by it under the chains. It never enters the
analysis path (scales stay in pixels; nothing here resamples).
"""
from __future__ import annotations

import numpy as np


def hillshade(values, dx: float, dy: float, azimuth: float = 315.0, altitude: float = 45.0,
              z_factor: float = 1.0) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"hillshade needs a 2-D field, got shape {z.shape} — pick one component first")
    if z.shape[0] < 2 or z.shape[1] < 2:
        return np.full(z.shape, np.nan)
    # central differences; NaN spreads only to immediate neighbours, which then stay NaN
    dzdy, dzdx = np.gradient(z, float(dy), float(dx))          # rows = y (southward), cols = x
    dzdx *= float(z_factor)
    dzdy *= float(z_factor)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)                            # ESRI: downslope direction, math angle
    zenith = np.radians(90.0 - float(altitude))
    az = np.radians((360.0 - float(azimuth) + 90.0) % 360.0)    # compass -> math convention
    shade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(az - aspect)
    out = np.clip(shade, 0.0, 1.0)
    out[~np.isfinite(shade) | ~np.isfinite(z)] = np.nan       # a missing cell stays missing
    return out
