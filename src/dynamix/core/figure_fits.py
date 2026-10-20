# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Straight segments and range fits in explicitly transformed plot coordinates."""
from __future__ import annotations

import numpy as np


def manual_slope(points, *, factor=1.) -> dict:
    p = np.asarray(points, float)
    if p.shape != (2, 2) or not np.isfinite(p).all():
        raise ValueError("a manual slope requires two finite endpoints")
    dx = p[1, 0] - p[0, 0]
    slope = float((p[1, 1]-p[0, 1])/dx) if dx != 0 else float("nan")
    return {"method": "manual", "endpoints": p.tolist(), "slope": slope,
            "intercept": float(p[0, 1]-slope*p[0, 0]), "value": factor*slope}


def range_fit(x, y, lo, hi, *, factor=1.) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo, hi = sorted((float(lo), float(hi)))
    keep = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    x, y = x[keep], y[keep]
    out = {"method": "range", "bounds": [lo, hi], "count": int(x.size),
           "slope": np.nan, "intercept": np.nan, "se": np.nan, "value": np.nan}
    if x.size < 3 or np.ptp(x) == 0:
        return out
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope*x+intercept)
    se = np.sqrt(np.sum(residual**2)/(x.size-2)/np.sum((x-x.mean())**2))
    out.update(slope=float(slope), intercept=float(intercept), se=float(se), value=float(factor*slope),
               included=[float(x.min()), float(x.max())],
               r2=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)) if np.ptp(y)>0 else np.nan)
    return out
