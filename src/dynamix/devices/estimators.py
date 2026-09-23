# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Named, registered Hölder estimators over the chain-dict schema.

The extensibility contract: quantities like "slope of log2|W| vs
log2 a" are pluggable components. A learned estimator registers under a new
name and appears in every device dropdown built from this registry. Pure
numpy over plain dicts -- no optional-dependency imports, ever.
"""
from __future__ import annotations
from typing import Callable
import numpy as np

def _ols(chain: dict) -> float:
    s = np.asarray(chain.get("log2_scales", ()), dtype=np.float64)
    m = np.asarray(chain.get("log2_mod", ()), dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(m)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(s[ok], m[ok], 1)[0])

def _max_local_slope(chain: dict) -> float:
    s = np.asarray(chain.get("log2_scales", ()), dtype=np.float64)
    m = np.asarray(chain.get("log2_mod", ()), dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(m)
    if ok.sum() < 2:
        return float("nan")
    ds, dm = np.diff(s[ok]), np.diff(m[ok])
    good = ds != 0
    return float((dm[good] / ds[good]).max()) if good.any() else float("nan")

HOLDER_ESTIMATORS: dict[str, Callable[[dict], float]] = {}

def register_holder_estimator(name: str, fn: Callable[[dict], float]) -> None:
    if name in HOLDER_ESTIMATORS:
        raise ValueError(f"holder estimator {name!r} already registered")
    HOLDER_ESTIMATORS[name] = fn

register_holder_estimator("ols", _ols)
register_holder_estimator("max", _max_local_slope)
