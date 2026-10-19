# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Wavelet leaders, p-leaders and (θ, ω)-leaders on the critically sampled dyadic pyramid of
:mod:`dynamix.core.dwt`, with the readouts the regularity tool publishes: structure-function
tables in the partition schema, log-cumulants, admissibility and a local exponent per cell.

Level j = 1 is the finest; a level-j cell λ has side 2^j px and, once every band is rolled onto
the field's dyadic grid by :func:`dwt.registration`, owns the 2^d cells of level j − 1 under it.
Coefficients are L¹ (2^{−dj/2} times PyWavelets' orthonormal ones), so a quantity built from
them scales like 2^{jh} at a singularity of exponent h. Over the three bands of the 2-D pyramid
(the one band in 1-D), with 3λ the 3^d block of cells around λ:

- the leader ℓ(λ) = sup |d| over the cells of 3λ and all their descendants (Jaffard; Dumeur,
  Saës, Abry, Ciuciu, Wendt and Jaffard 2025, Definition 1.2);
- the p-leader ℓ^{(p)}(λ) = (Σ over the same set of |d|^p 2^{−d (j − j')})^{1/p} (Jaffard, Melot,
  Leonarduzzi, Wendt, Abry, Roux and Torres 2016), times the finite-resolution factor
  ((1 − 2^{−jη}) / (1 − 2^{−η}))^{−1/p} when the coefficients' η(p) is positive (Leonarduzzi,
  Wendt, Abry, Jaffard, Roux and Torres 2017);
- the (θ, ω)-leader ℓ^w(λ) = sup |d| over the levels j, j − 1, …, j − D and, at each, the cells
  whose centre lies within ω coarse cells (sup norm) of λ's centre (Dumeur et al. 2025,
  Definition 2.2), the window of Meyer's weak scaling exponent; its schedule (D and ω per level)
  is fixed, or the paper's θ(j) = j + j^β, ω(j) = j in the resolution index.

The oscillation ratio o = log2(ℓ / ℓ^w) is 0 at a cusp and large at a chirp when the window
lies inside 3λ. γ is the pseudo-fractional integration of the coefficients (× 2^{γj}); every
readout carries the integrated frame as computed and the original one shifted back.
"""
from __future__ import annotations

import math

import numpy as np

from dynamix.core import dwt

__all__ = ["BANDS", "EDGE_MODES", "METHODS", "admissibility", "coefficients", "cumulants", "leaders",
           "local_exponent", "oscillation", "paint", "partition_table", "pleaders", "run_leaders",
           "valid_cells", "wse", "wse_schedule"]

BANDS = ("horizontal", "vertical", "diagonal")
METHODS = ("leaders", "p-leaders", "wse")
#: How the field is continued to the working image the periodized transform sees, by numpy's
#: pad mode: ``mirror`` repeats the edge sample and folds (one period of an even image,
#: continuous at every edge, the WTMM's ``reflect`` by another name), ``reflect`` folds without
#: repeating it, ``edge_pad`` and ``zero_pad`` hold a constant, ``periodic`` wraps the field
#: itself — for a field that is periodic already (a Periodic extension's output), where no edge
#: exists and no border is lost.
EDGE_MODES = {"mirror": "symmetric", "reflect": "reflect", "edge_pad": "edge", "zero_pad": "constant",
              "periodic": "wrap"}


# ----------------------------------------------------------------------------------- coefficients

def _extend(x, W, edge: str):
    """``x`` continued to the working shape ``W`` by ``edge`` (:data:`EDGE_MODES`): the mirror
    through :func:`dwt.mirror` (the DWT tools' own construction), the others by numpy's pad
    after the data."""
    if edge == "mirror":
        return dwt.mirror(x, W)
    if edge not in EDGE_MODES:
        raise ValueError(f"leaders: edge {edge!r}, expected one of {tuple(EDGE_MODES)}")
    pad = tuple((0, int(w) - int(n)) for w, n in zip(W, x.shape))
    return np.pad(x, pad, mode=EDGE_MODES[edge])


def coefficients(values, name: str, level: int, *, gamma: float = 0.0, edge: str = "mirror",
                 origin=(0, 0), block=None) -> dict:
    """The detail moduli of the field continued by ``edge`` to the working image, L¹, registered
    and integrated by 2^{γj}: ``mods[j-1]`` is the stack ``(n_bands, *cells)`` of |d| at level
    j = 1..``level``; ``bad[j-1]`` marks the cells holding a non-finite pixel (filled with the
    finite mean for the transform); ``block[j-1]`` is the cell range per axis, ``(lo, hi)``, that
    lies inside the data — the whole field, or the real block of a Periodic extension's output
    starting at ``origin`` with ``block`` pixels per axis. Under ``periodic`` the working image is
    the field wrapped to the next multiple of 2^J per axis, not doubled: a periodic field has no
    edge to hide."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"leaders: a 1-D signal or a 2-D band, not shape {x.shape}")
    J = int(level)
    finite = np.isfinite(x)
    filled = np.where(finite, x, float(x[finite].mean()) if finite.any() else 0.0)
    if edge == "periodic":
        W = tuple(int(math.ceil(int(n) / 2 ** J)) * 2 ** J for n in x.shape)
    else:
        W = dwt.working_shape(x.shape, J)
    work = _extend(filled, W, edge)
    bad = _extend((~finite).astype(np.float64), W, "zero_pad" if edge in ("zero_pad", "edge_pad") else edge) > 0.5
    size = tuple(int(n) for n in x.shape) if block is None else tuple(int(n) for n in block)
    start = tuple(int(o) for o in origin)
    reg = dwt.registration(name, J)
    d = x.ndim
    mods = []
    if d == 2:
        pyr = dwt.pyramid(work, name, J)
        for j in range(1, J + 1):
            rolled = [np.roll(b, reg[(j, band)][:2], axis=(0, 1))
                      for b, band in zip(pyr[J + 1 - j], BANDS)]
            mods.append(np.abs(np.stack(rolled)) * 2.0 ** ((gamma - 1.0) * j))
    else:
        pywt = dwt.require_pywt()
        coeffs = pywt.wavedec(work, dwt.wavelet(name), mode="periodization", level=J)
        for j in range(1, J + 1):
            rolled = np.roll(coeffs[J + 1 - j], reg[(j, "detail")][0])
            mods.append(np.abs(rolled)[None] * 2.0 ** ((gamma - 0.5) * j))
    cells = [tuple((int(math.ceil(o / 2 ** j)), int((o + n) // 2 ** j)) for o, n in zip(start, size))
             for j in range(1, J + 1)]
    # Wrapping a field to a working size that is not its own period puts a seam at the working
    # image's wrap: a border there, like any padded edge.
    seam = edge == "periodic" and tuple(W) != tuple(x.shape)
    return {"mods": mods, "bad": [_pool_to(bad, j, np.any) for j in range(1, J + 1)],
            "block": cells, "levels": J, "ndim": d, "shape": tuple(x.shape), "work_shape": tuple(W),
            "data_shape": size, "origin": start, "edge": edge, "seam": seam,
            "registration": reg, "gamma": float(gamma), "wavelet": name}


def _pool(a, op):
    """One pooling step: the 2^d children of every cell reduced by ``op`` (max or sum)."""
    shape = []
    for n in a.shape:
        shape += [n // 2, 2]
    return op(a.reshape(shape), axis=tuple(range(1, 2 * a.ndim, 2)))


def _pool_to(a, level: int, op):
    for _ in range(int(level)):
        a = _pool(a, op)
    return a


def _box(a, radius: int, op):
    """The separable box of half-width ``radius`` cells reduced by ``op``, periodic on the
    working grid (the mirror extension makes it one period of an even image)."""
    out = a
    for ax in range(a.ndim):
        acc = out
        for t in range(1, int(radius) + 1):
            acc = op(op(acc, np.roll(out, t, axis=ax)), np.roll(out, -t, axis=ax))
        out = acc
    return out


def _window_max(fine, depth: int, width: float):
    """The sup over the fine cells (``depth`` levels below) whose centre lies within ``width``
    coarse cells of each coarse cell's centre, on the coarse grid."""
    f = 2 ** int(depth)
    o = (f - 1) / 2.0                      # the coarse centre, in fine cells from fine cell f k
    lo, hi = math.ceil(o - width * f - 1e-9), math.floor(o + width * f + 1e-9)
    out = fine
    for ax in range(fine.ndim):
        acc = None
        for t in range(lo, hi + 1):
            r = np.roll(out, -t, axis=ax)
            acc = r if acc is None else np.maximum(acc, r)
        out = acc
    return out[tuple(slice(0, None, f) for _ in range(fine.ndim))]


def valid_cells(coef: dict, margin: int, seam_margin: int = 0) -> list:
    """Per level, the cells that enter maps and statistics: inside the data's block at least
    ``margin`` cells from each of its edges, at least ``seam_margin`` cells from the working
    image's wrap (the seam a wrapped field of another period has there), and no cell with a
    non-finite pixel within ``margin`` cells."""
    out = []
    for bad, block in zip(coef["bad"], coef["block"]):
        v = np.zeros(bad.shape, dtype=bool)
        window = [(max(lo + margin, seam_margin), min(hi - margin, n - seam_margin))
                  for (lo, hi), n in zip(block, bad.shape)]
        if all(hi > lo for lo, hi in window):
            v[tuple(slice(lo, hi) for lo, hi in window)] = True
        if bad.any():
            v &= ~_box(bad, margin, np.logical_or) if margin else ~bad
        out.append(v)
    return out


# ------------------------------------------------------------------------------------ the quantities

def leaders(coef: dict) -> list:
    """ℓ_j per level: the sup over the bands and over the subtree of 3λ."""
    out, S = [], None
    for mods in coef["mods"]:
        top = mods.max(axis=0)
        S = top if S is None else np.maximum(top, _pool(S, np.max))
        out.append(_box(S, 1, np.maximum))
    return out


def pleaders(coef: dict, p: float, *, eta: "float | None" = None) -> list:
    """ℓ^{(p)}_j per level, the finer levels weighted 2^{−d} per level down; the
    finite-resolution factor applied when ``eta`` (the coefficients' η(p)) is positive."""
    d, p = coef["ndim"], float(p)
    out, T = [], None
    for j, mods in enumerate(coef["mods"], start=1):
        top = (mods ** p).sum(axis=0)
        T = top if T is None else top + 2.0 ** (-d) * _pool(T, np.sum)
        lp = _box(T, 1, np.add) ** (1.0 / p)
        if eta is not None and eta > 0:
            lp = lp * ((1.0 - 2.0 ** (-j * eta)) / (1.0 - 2.0 ** (-eta))) ** (-1.0 / p)
        out.append(lp)
    return out


def wse_schedule(levels: int, *, rule: str = "dumeur", depth: int = 1, width: float = 1.0,
                 beta: float = 0.25, log2_n: float = 10.0) -> list:
    """Per level j = 1..``levels``, the ``(depth, half-width)`` pairs of the (θ, ω) window, the
    half-width in coarse cells. ``fixed``: depths 0..``depth`` at ``width``. ``dumeur``: the
    paper's θ(j) = j + j^β and ω(j') = j' in the resolution index j_r = ``log2_n`` − j (the
    field's log2 side less the level), i.e. the depth round(j_r^β) and the half-width j_r + D at
    depth D (Dumeur et al. 2025, Figures 4 and 5: β = 1/4)."""
    out = []
    for j in range(1, int(levels) + 1):
        if rule == "fixed":
            out.append([(D, float(width)) for D in range(int(depth) + 1)])
        elif rule == "dumeur":
            jr = max(1.0, float(log2_n) - j)
            out.append([(D, jr + D) for D in range(int(round(jr ** beta)) + 1)])
        else:
            raise ValueError(f"wse rule {rule!r}: 'dumeur' or 'fixed'")
    return out


def wse(coef: dict, schedule: list) -> list:
    """ℓ^w_j per level: the sup of |d| over the window ``schedule[j-1]`` lists as (depth,
    half-width) pairs, each depth taken at the level that many steps finer."""
    mods, out = coef["mods"], []
    for j in range(1, coef["levels"] + 1):
        acc = None
        for D, w in schedule[j - 1]:
            if j - int(D) < 1:
                continue
            win = _window_max(mods[j - int(D) - 1].max(axis=0), int(D), float(w))
            acc = win if acc is None else np.maximum(acc, win)
        out.append(acc)
    return out


def oscillation(lead: list, weak: list) -> list:
    """o_j = log2(ℓ_j / ℓ^w_j) per level, NaN where either is 0."""
    out = []
    for a, b in zip(lead, weak):
        with np.errstate(divide="ignore", invalid="ignore"):
            o = np.log2(a / b)
        out.append(np.where(np.isfinite(o), o, np.nan))
    return out


# --------------------------------------------------------------------------------------- readouts

def _slope(x, y) -> tuple:
    """OLS slope and R² of ``y`` against ``x`` over the finite pairs; NaN below two points."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    s, b = np.polyfit(x[m], y[m], 1)
    res = y[m] - (s * x[m] + b)
    tot = y[m] - y[m].mean()
    return float(s), float(1.0 - (res @ res) / max(float(tot @ tot), 1e-300))


def admissibility(coef: dict, valid: list, j_min: int, p: float) -> dict:
    """h_min, the slope of log2 max |d| against the level over the fitted levels and the valid
    cells of every band, and η(p), the slope of log2 mean |d|^p; both as computed on the
    integrated coefficients and shifted back to the original frame (h_min − γ, η − pγ).
    Verdicts: leaders need h_min > 0, p-leaders η(p) > 0, (θ, ω)-leaders nothing; for a method
    that fails, the total γ in steps of ½ that would clear 0 by ¼ (pymultifracs' rule)."""
    J, gamma, p = coef["levels"], coef["gamma"], float(p)
    js = np.arange(int(j_min), J + 1, dtype=np.float64)
    top, mean = [], []
    for j in range(int(j_min), J + 1):
        m = coef["mods"][j - 1][:, valid[j - 1]]
        top.append(np.log2(m.max()) if m.size and m.max() > 0 else np.nan)
        mean.append(np.log2(np.mean(m ** p)) if m.size and np.mean(m ** p) > 0 else np.nan)
    h_min, _ = _slope(js, top)
    eta, _ = _slope(js, mean)

    def suggest(margin_needed: float, per_gamma: float) -> float:
        if not np.isfinite(margin_needed) or margin_needed <= 0.0:
            return gamma
        return gamma + math.ceil(margin_needed / per_gamma / 0.5) * 0.5

    return {"gamma": gamma, "p": p,
            "h_min": {"integrated": h_min, "original": h_min - gamma},
            "eta": {"integrated": eta, "original": eta - p * gamma},
            "leaders": {"admissible": bool(h_min > 0), "suggested_gamma": suggest(0.25 - h_min, 1.0)},
            "p-leaders": {"admissible": bool(eta > 0), "suggested_gamma": suggest(0.25 - eta, p)},
            "wse": {"admissible": True, "suggested_gamma": gamma}}


def partition_table(per_level: list, valid: list, q_list, shape, *,
                    positive_only: bool = False) -> dict:
    """The structure functions of one quantity in the partition schema the fitter reads:
    ``tau_qa`` = log2 of the mean of M^q over the valid cells of each level times the field's
    cell count there, Π_i n_i / 2^j, so that the count scales exactly as 2^{−dj} whatever the
    border takes and τ(q) = ζ(q) − d (the critically sampled cells standing where the WTMM's
    maxima count stands); ``h_qa``, ``D_qa``, their σ's, ``N_a`` (the valid cells) and
    ``lnT_moments``; rows with q ≤ 0 NaN when ``positive_only`` (the bare coefficients, whose
    zeros make negative moments meaningless)."""
    from dynamix.core.microcanonical import partition_tables

    samples = [np.asarray(m)[..., v].ravel() if np.asarray(m).ndim > v.ndim else np.asarray(m)[v]
               for m, v in zip(per_level, valid)]
    scales = 2.0 ** np.arange(1, len(per_level) + 1)
    table = partition_tables(samples, scales, np.asarray(q_list, dtype=np.float64), d=0.0)
    with np.errstate(divide="ignore"):
        shift = np.array([np.log2(np.prod([int(n) // 2 ** j for n in shape]) / max(x.size, 1))
                          for j, x in enumerate(samples, start=1)])
    table["tau_qa"] += shift[None, :]
    table["D_qa"] -= shift[None, :]
    if positive_only:
        rows = table["q_list"] <= 0
        for k in ("tau_qa", "h_qa", "D_qa", "sigma_tau_qa", "sigma_h_qa", "sigma_D_qa"):
            table[k][rows] = np.nan
    return table


def cumulants(table: dict, j_min: int, j_max: int, gamma: float) -> dict:
    """The log-cumulants c1..c3 of ln M over the fitted levels, from the table's per-level
    moments, in the app's cascade convention (c2 > 0 for a cascade: τ(q) = c1 q − c2 q²/2 + …),
    as computed and in the original frame (c1 − γ)."""
    from dynamix.core.spectra import cumulants_from_moments

    integrated = cumulants_from_moments(table["lnT_moments"], table["scales"], float(j_min),
                                        float(j_max), max_order=3)
    original = dict(integrated)
    original["c1"] = integrated["c1"] - float(gamma)
    return {"integrated": integrated, "original": original}


def local_exponent(per_level: list, valid: list, j_min: int) -> tuple:
    """Per cell of level ``j_min``, the slope of log2 M against the level over that cell and its
    ancestors up to the coarsest level, with its R²; NaN where any level is invalid or zero."""
    J = len(per_level)
    base = np.asarray(per_level[j_min - 1]).shape
    rows = []
    for j in range(j_min, J + 1):
        a, v = np.asarray(per_level[j - 1], dtype=np.float64), valid[j - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(v & (a > 0), np.log2(np.where(a > 0, a, 1.0)), np.nan)
        rows.append(paint(y, j - j_min, base))
    Y = np.stack(rows)
    x = np.arange(j_min, J + 1, dtype=np.float64)
    xm = x - x.mean()
    ym = Y.mean(axis=0)
    slope = np.tensordot(xm, Y - ym, axes=(0, 0)) / float(xm @ xm)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = Y - ym - slope[None] * xm.reshape((-1,) + (1,) * ym.ndim)
        r2 = 1.0 - (res ** 2).sum(axis=0) / ((Y - ym) ** 2).sum(axis=0)
    return slope, r2


def paint(cells, level: int, shape) -> np.ndarray:
    """A level-``level`` cell grid painted over its 2^level-px blocks, cropped to ``shape``."""
    out = np.asarray(cells)
    f = 2 ** int(level)
    for ax in range(out.ndim):
        out = np.repeat(out, f, axis=ax)
    return out[tuple(slice(0, int(n)) for n in shape)]


# ------------------------------------------------------------------------------------------- run

def run_leaders(values, *, wavelet: str = "sym4", scale_min: int = 4, scale_max: int = 64,
                gamma: float = 0.0, p: float = 2.0, wse_rule: str = "dumeur", wse_depth: int = 1,
                wse_width: float = 1.0, wse_beta: float = 0.25, q_list=None, edge: str = "mirror",
                origin=(0, 0), block=None, progress=None) -> dict:
    """The three quantities and every readout on one band (or 1-D signal): the pyramid to
    J = log2 ``scale_max``, the fit over the levels from log2 ``scale_min``; the tables keyed
    by method plus ``coefficients``; the cumulants and the local exponent per method. ``edge``
    continues the field (:data:`EDGE_MODES`); under ``periodic`` no border is lost. ``origin``
    and ``block`` name the data's real block inside the field (a Periodic extension's output)."""
    from dynamix.core.partition_tables import DEFAULT_Q_LIST

    J, j_min = _level_of(scale_max, "scale_max"), _level_of(scale_min, "scale_min")
    if j_min >= J:
        raise ValueError(f"leaders: scale_min {scale_min} px must be below scale_max {scale_max} px")
    x = np.asarray(values, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"leaders: a 1-D signal or a 2-D band, not shape {x.shape}")
    if min(x.shape) < 2 ** J:
        raise ValueError(f"leaders: the field's smaller side {min(x.shape)} px is below scale_max "
                         f"{2 ** J} px")
    q = np.asarray(DEFAULT_Q_LIST if q_list is None else q_list, dtype=np.float64)
    tick = progress or (lambda frac: None)

    coef = coefficients(x, wavelet, J, gamma=gamma, edge=edge, origin=origin, block=block)
    tick(0.3)
    # The border the continuation costs: the bank's cells for a padded edge, none on a torus;
    # a wrapped field whose period is not the working size has a seam at the wrap, a border there.
    border = dwt.border_cells(wavelet)
    e = 0 if edge == "periodic" else border
    seam = border if coef["seam"] else 0
    e_coef, seam_coef = max(e - 1, 0), max(seam - 1, 0)
    schedule = wse_schedule(J, rule=wse_rule, depth=wse_depth, width=wse_width, beta=wse_beta,
                            log2_n=math.log2(min(coef["data_shape"])))
    reach = 0 if edge == "periodic" and not coef["seam"] else max(math.ceil(w) for level in schedule for _, w in level)
    valid = {"coefficients": valid_cells(coef, e_coef, seam_coef), "leaders": valid_cells(coef, e, seam),
             "p-leaders": valid_cells(coef, e, seam),
             "wse": valid_cells(coef, e_coef + (reach if edge != "periodic" else 0), seam_coef + (reach if coef["seam"] else 0))}
    adm = admissibility(coef, valid["coefficients"], j_min, p)
    eta = adm["eta"]["integrated"]
    quantity = {"leaders": leaders(coef),
                "p-leaders": pleaders(coef, p, eta=eta if np.isfinite(eta) else None),
                "wse": wse(coef, schedule)}
    osc = oscillation(quantity["leaders"], quantity["wse"])
    tick(0.6)
    for m, v in zip(quantity["leaders"], valid["leaders"]):
        v &= np.isfinite(m)
    tables = {k: partition_table(quantity[k], valid[k], q, coef["data_shape"]) for k in METHODS}
    tables["coefficients"] = partition_table(coef["mods"], valid["coefficients"], q, coef["data_shape"],
                                             positive_only=True)
    cums = {k: cumulants(tables[k], j_min, J, gamma) for k in METHODS}
    local = {k: local_exponent(quantity[k], valid[k], j_min) for k in METHODS}
    tick(1.0)
    return {"levels": J, "j_min": j_min, "scales": 2.0 ** np.arange(1, J + 1), "ndim": x.ndim,
            "shape": coef["shape"], "edge": edge, "seam": coef["seam"], "origin": coef["origin"],
            "data_shape": coef["data_shape"],
            "wavelet": wavelet, "gamma": float(gamma), "p": float(p),
            "coefficients": coef["mods"], "quantity": quantity, "oscillation": osc, "valid": valid,
            "tables": tables, "cumulants": cums, "admissibility": adm, "local": local,
            "registration": coef["registration"], "border_cells": e, "wse_schedule": schedule,
            "valid_cells": {k: [int(v.sum()) for v in vs] for k, vs in valid.items()},
            "zero_cells": [int(((m == 0) & v).sum())
                           for m, v in zip(quantity["leaders"], valid["leaders"])]}


def _level_of(scale_px, name: str) -> int:
    j = math.log2(float(scale_px))
    if scale_px <= 1 or abs(j - round(j)) > 1e-9:
        raise ValueError(f"leaders: {name} {scale_px} px is not a power of two above 1")
    return int(round(j))
