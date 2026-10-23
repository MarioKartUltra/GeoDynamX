# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Hemisphere projections of lines and planes, independent of plotting libraries."""
from __future__ import annotations

import numpy as np

from dynamix.core.orientation import Orientation, Frame


def project(trend: Orientation, plunge, *, projection="schmidt", hemisphere="lower"):
    """Bearing/plunge to a unit net (east right, north up); plunge is positive down.

    Antipodal lines are folded into the selected hemisphere. Pixel angles must first be
    registered or explicitly converted to a sample-axis bearing convention.
    """
    if trend.frame is Frame.PIXEL:
        raise ValueError("register pixel axes before projecting a 3-D orientation")
    if projection not in ("schmidt", "wulff") or hemisphere not in ("lower", "upper"):
        raise ValueError("unknown projection or hemisphere")
    t, p = np.broadcast_arrays(np.asarray(trend.degrees, float), np.asarray(plunge, float))
    if np.any(np.abs(p) > 90):
        raise ValueError("plunge must lie between -90 and 90 degrees")
    flip = p < 0 if hemisphere == "lower" else p > 0
    t = np.radians(t + np.where(flip, 180, 0))
    p = np.radians(np.abs(p))
    r = np.sqrt(np.maximum(0, 1 - np.sin(p))) if projection == "schmidt" else np.tan(np.pi / 4 - p / 2)
    return r * np.sin(t), r * np.cos(t)


def plane_pole(strike: Orientation, dip):
    """Right-hand-rule strike/dip to the lower-hemisphere pole."""
    d = np.asarray(dip, float)
    if np.any((d < 0) | (d > 90)):
        raise ValueError("dip must lie between 0 and 90 degrees")
    return Orientation((np.asarray(strike.degrees) - 90) % 360, strike.frame), 90 - d


def plane_trace(strike: Orientation, dip, *, projection="schmidt", hemisphere="lower"):
    """Projected great-circle trace, sampled in its plane and split at the horizon."""
    s, d = np.radians(float(strike.degrees)), np.radians(float(dip))
    if not 0 <= d <= np.pi / 2:
        raise ValueError("dip must lie between 0 and 90 degrees")
    along = np.array([np.sin(s), np.cos(s), 0.])
    down = np.array([np.cos(s)*np.cos(d), -np.sin(s)*np.cos(d), np.sin(d)])
    u = np.linspace(0, 2*np.pi if d == 0 else np.pi, 361 if d == 0 else 181)
    v = along[:, None]*np.cos(u) + down[:, None]*np.sin(u)
    t = Orientation(np.degrees(np.arctan2(v[0], v[1])), strike.frame)
    return project(t, np.degrees(np.arcsin(np.clip(v[2], -1, 1))), projection=projection, hemisphere=hemisphere)


def graticule(*, projection="schmidt", hemisphere="lower", spacing=15):
    """Great-circle meridians and small-circle parallels of the hemisphere net."""
    u = np.linspace(0, np.pi, 181)
    curves = []
    def line(east, north, down):
        t = Orientation(np.degrees(np.arctan2(east, north)), Frame.TRUE)
        return project(t, np.degrees(np.arcsin(np.clip(down, -1, 1))),
                       projection=projection, hemisphere=hemisphere)
    for angle in np.arange(spacing, 180, spacing):
        beta = np.radians(angle)
        curves.append(line(np.sin(u)*np.cos(beta), np.cos(u), np.sin(u)*np.sin(beta)))
    for angle in np.arange(-90+spacing, 90, spacing):
        lat = np.radians(angle)
        curves.append(line(np.cos(lat)*np.cos(u), np.full(u.shape,np.sin(lat)), np.cos(lat)*np.sin(u)))
    return curves


def rose_histogram(orientation: Orientation, *, bins=36, axial=False):
    """Zero-centred angular sectors; axial antipodes share one original observation count."""
    if int(bins) != bins or bins < 1:
        raise ValueError('rose bin count must be a positive integer')
    values = np.asarray(orientation.degrees, dtype=float).ravel()
    values = values[np.isfinite(values)]
    period = 180. if axial else 360.
    width = period/int(bins)
    edges = np.arange(int(bins)+1)*width-width/2
    folded = (values+width/2)%period-width/2
    counts, _ = np.histogram(folded, edges)
    return {'edges': edges, 'counts': counts, 'fraction': counts/max(values.size, 1),
            'total': int(values.size), 'period': period, 'frame': orientation.frame.value,
            'axial': bool(axial)}


def rose_coordinates(degrees, radius, *, frame: Frame):
    """Plot coordinates: bearings clockwise from north; pixel +90 points toward +row/down."""
    angle = np.radians(np.asarray(degrees, float))
    if frame is Frame.PIXEL:
        return radius*np.cos(angle), -radius*np.sin(angle)
    return radius*np.sin(angle), radius*np.cos(angle)
