# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Spatial samples for direction figures, retaining their source-point identities."""
from __future__ import annotations

import numpy as np


def direction_samples(result: dict, population: str = "wtmmm") -> list:
    """One point per native position per scale, with original indices and chain membership.

    ``wtmm`` includes all horizontal-chain points, including endpoints; ``wtmmm`` uses
    the detector's existing local-peak reduction. Neither requires vertical links.
    """
    if population not in ("wtmm", "wtmmm"):
        raise ValueError("population must be wtmm or wtmmm")
    extrema = result.get("extrema") or []
    if population == "wtmmm":
        from dynamix.core.anisotropy import wtmmm_by_scale
        reduced = wtmmm_by_scale(extrema)
    else:
        reduced = extrema
    out = []
    for original, points in zip(extrema, reduced):
        n = len(original.get("x", ()))
        if population == "wtmm" and n and "line_id" not in original:
            raise ValueError("All WTMM requires horizontal-chain membership")
        xy = np.column_stack((original.get("x", ()), original.get("y", ())))
        lookup = {tuple(p): i for i, p in reversed(list(enumerate(xy)))}
        chosen = np.array([lookup[tuple(p)] for p in
                           np.column_stack((points.get("x", ()), points.get("y", ())))], dtype=int)
        ids = np.asarray(original.get("line_id", np.full(n, -1)))
        if population == "wtmm":
            chosen = chosen[ids[chosen] >= 0]
        if chosen.size:
            _, unique = np.unique(xy[chosen], axis=0, return_index=True)
            chosen = chosen[np.sort(unique)]
        row = {k: np.asarray(original.get(k, ()))[chosen] for k in ("x", "y", "mod", "arg")}
        row.update(line_id=ids[chosen], point_id=chosen)
        out.append(row)
    return out


def tangent_arguments(ext: dict, shape) -> np.ndarray:
    """Local geometric tangents of ordered chain runs, NaN where no tangent exists."""
    from dynamix.core.hlines import hline_runs

    out = np.full(len(ext.get("x", ())), np.nan)
    for run in hline_runs(ext, shape):
        x, y = np.asarray(ext["x"])[run], np.asarray(ext["y"])[run]
        out[run] = np.arctan2(np.gradient(y.astype(float)), np.gradient(x.astype(float)))
    return out
