# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""What a run reports: held-out blocks and their error, the paper's variance error, and the
``ln|eta|`` and ``ln|c|`` distributions inside the regions against the clear ones. Consistent
ratios alone do not show that the amplitudes are right, so both are read.
"""
from __future__ import annotations

import numpy as np

__all__ = ["band_hists", "error_map", "errors", "heldout", "hist_distance"]


def heldout(valid, n_blocks: int, side: int, margin: int, rng) -> np.ndarray:
    """Up to ``n_blocks`` ``side``-pixel squares of ``valid`` pixels, ``margin`` from any
    invalid pixel and from each other, drawn in ``rng``'s order."""
    from dynamix.core.inpaint._kernels import kernels

    v = np.ascontiguousarray(np.asarray(valid, dtype=bool).view(np.uint8))
    out = np.zeros(v.shape, dtype=np.uint8)
    if int(n_blocks) <= 0:
        return out.astype(bool)
    cand = rng.permutation(v.size).astype(np.int64)
    kernels()["heldout_blocks"](v, cand, int(side), int(margin), int(n_blocks), out)
    return out.astype(bool)


def errors(x, y, where) -> dict:
    """``rms_rel``: the RMS of ``x - y`` over ``where`` relative to ``y``'s 0.5-99.5 % range;
    ``var_err``: the paper's variance error, the mean square over the variance of ``y`` there."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    sel = np.asarray(where, dtype=bool) & np.isfinite(y) & np.isfinite(x)
    n = int(sel.sum())
    if n == 0:
        return {"rms_rel": float("nan"), "var_err": float("nan"), "n": 0}
    d = x[sel] - y[sel]
    fin = y[np.isfinite(y)]
    span = float(np.quantile(fin, 0.995) - np.quantile(fin, 0.005)) if fin.size else 1.0
    var = float(y[sel].var())
    mse = float((d ** 2).mean())
    return {"rms_rel": float(np.sqrt(mse)) / (span if span > 0 else 1.0),
            "var_err": mse / var if var > 0 else float("nan"), "n": n}


def error_map(x, y, where) -> np.ndarray:
    sel = np.asarray(where, dtype=bool)
    return np.where(sel, np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)),
                    np.nan)


def hist_distance(a, b) -> float:
    """The Kolmogorov distance of two samples (the maximum gap of their empirical CDFs)."""
    a, b = np.sort(np.ravel(a)), np.sort(np.ravel(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.abs(fa - fb).max())


def _pairs(parent, child):
    fp, fc = 1e-8 * np.abs(parent).max(), 1e-8 * np.abs(child).max()
    P = np.repeat(np.repeat(parent, 2, 0), 2, 1)
    ok_c = np.abs(child) >= fc
    ok = ok_c & (np.abs(P) >= fp)
    return ok, ok_c, np.log(np.abs(child[ok])) - np.log(np.abs(P[ok])), np.log(np.abs(child[ok_c]))


def band_hists(coeffs, rho, clear, abacus=None, nbins: int = 60, *, shift: int) -> dict:
    """``{(level, orientation): {"eta": {...}, "amp": {...}}}`` for levels ``1..J-1``: the
    ``ln|eta|`` and ``ln|c|`` samples of the children inside the regions (``rho > 0.9``) and
    of the clear children, histogrammed on common edges with their Kolmogorov distance; the
    child bands and their masks are rolled by the bank's child ``shift`` (the pairing the
    abacus and the inference use); a clear ETA pair needs the parent clear too; the amplitude sample needs only the child.
    A level with no inside or no clear pair has distance
    ``nan``."""
    from dynamix.core.inpaint.priors import aligned

    J = len(coeffs) - 1
    out = {}
    for j in range(1, J):
        for o in range(3):
            inside = aligned(np.asarray(rho[j - 1]) > 0.9, o, shift)
            clr_c = aligned(np.asarray(clear[j - 1]) == 1, o, shift)
            clr = clr_c & np.repeat(np.repeat(np.asarray(clear[j]) == 1, 2, 0), 2, 1)
            parent = np.asarray(coeffs[J + 1 - (j + 1)][o], dtype=np.float64)
            child = aligned(np.asarray(coeffs[J + 1 - j][o], dtype=np.float64), o, shift)
            ok, ok_c, eta, amp = _pairs(parent, child)
            entry = {}
            for key, values, edges, sel_in, sel_clr in (
                    ("eta", eta, None if abacus is None else abacus.edges, inside[ok], clr[ok]),
                    ("amp", amp, None, inside[ok_c], clr_c[ok_c])):
                a, b = values[sel_in], values[sel_clr]
                if edges is None:
                    lo, hi = (np.quantile(np.concatenate([a, b]), [0.005, 0.995])
                              if a.size + b.size else (0.0, 1.0))
                    edges = np.linspace(lo, hi if hi > lo else lo + 1.0, nbins + 1)
                entry[key] = {"edges": edges, "inside": np.histogram(a, bins=edges)[0],
                              "clear": np.histogram(b, bins=edges)[0],
                              "distance": hist_distance(a, b), "n_inside": int(a.size),
                              "n_clear": int(b.size)}
            out[(j, o)] = entry
    return out
