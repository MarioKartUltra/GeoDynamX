# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The two priors of the inpainting engine and the masks they read on the pyramid.

* ``Sparsity``: the cycle-spun soft threshold of every detail band (the author's notebooks),
  with the threshold ``lambda = lam_rel sigma sqrt(2 ln N)``.
* ``Cascade``: Pottier, Turiel and Garcon 2008, every parent inferred from its children by
  maximum likelihood over the abacus of ``ln|eta|`` and blended by ``rho`` (eqs. 13-16).
* ``Abacus``: the histogram of ``ln|eta| = ln|c| - ln|p|`` over the clear parent-child pairs,
  one curve per orientation pooled over levels (the paper's universality, Fig. 8).

A coefficient is *clear* when no masked pixel lies within its atom's support, a dilation test
on the mask; ``rho`` is the approximation of the mask through the same pyramid divided by
``2**j`` and clipped to [0, 1] (for sym7 and bl3 the unclipped value spans -0.19..1.36).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamix.core import dwt

__all__ = ["Abacus", "Cascade", "Sparsity", "aligned", "auto_lambda", "clear_masks",
           "clear_pixels", "make_abacus", "rho_pyramid", "unaligned"]


def _hits(mask, half: int, step: int, shape) -> np.ndarray:
    from dynamix.core.inpaint._kernels import kernels

    m = np.ascontiguousarray(np.asarray(mask, dtype=bool).view(np.uint8))
    integ = np.zeros((m.shape[0] + 1, m.shape[1] + 1), dtype=np.int64)
    kernels()["integral"](m, integ)
    out = np.zeros(tuple(int(n) for n in shape), dtype=np.uint8)
    kernels()["window_any"](integ, int(half), int(step), 0, out)
    return out


def clear_masks(mask, name: str, levels: int) -> list:
    """Per level 1..``levels``: 1 on the coefficient grid where the atom's support (its
    ``2**j`` block plus ``(L - 1) 2**(j-1)`` each side) holds no masked pixel."""
    L = dwt.filter_length(name)
    ny, nx = np.shape(mask)
    out = []
    for j in range(1, int(levels) + 1):
        step = 2 ** j
        hit = _hits(mask, (L - 1) * 2 ** (j - 1), step, (ny // step, nx // step))
        out.append(np.ascontiguousarray(1 - hit))
    return out


def clear_pixels(mask, name: str, level: int) -> np.ndarray:
    """1 where no masked pixel lies within ``(L - 1) 2**(level-1)`` pixels: the clear set of
    the stationary transform's bands at ``level``."""
    L = dwt.filter_length(name)
    hit = _hits(mask, (L - 1) * 2 ** (int(level) - 1), 1, np.shape(mask))
    return np.ascontiguousarray(1 - hit)


def rho_pyramid(mask, name: str, levels: int) -> list:
    """Per level 1..``levels``: the mask's approximation over ``2**j``, clipped to [0, 1]."""
    approx = dwt.approximations(np.asarray(mask, dtype=np.float64), name, levels)
    return [np.clip(a / 2.0 ** (j + 1), 0.0, 1.0) for j, a in enumerate(approx)]


def _low_pass_axis(o: int):
    """The axis along which band ``o``'s pairing carries the bank's child shift: columns for
    the horizontal band, rows for the vertical, none for the diagonal."""
    return {0: 1, 1: 0}.get(int(o))


def aligned(a, o: int, s: int):
    """A child-level array of orientation ``o`` rolled so that index ``2k`` holds what was at
    ``2k + s``: the bank's pairing along the low-pass axis (exact under periodization)."""
    axis = _low_pass_axis(o)
    return a if axis is None or s == 0 else np.roll(a, -int(s), axis=axis)


def unaligned(a, o: int, s: int):
    axis = _low_pass_axis(o)
    return a if axis is None or s == 0 else np.roll(a, int(s), axis=axis)


def _collect_pairs(coeffs, clear, name_or_shift) -> tuple:
    """Per orientation, the ``ln|eta|`` of every clear parent-child pair over levels 1..J-1
    and whether the child carries the parent's sign; ``name_or_shift`` is the bank (its
    child shift is measured) or an integer shift."""
    from dynamix.core.inpaint._kernels import kernels

    s = name_or_shift if isinstance(name_or_shift, int) else dwt.child_shift(name_or_shift)
    J = len(coeffs) - 1
    ratios = [[], [], []]
    signs = [[], [], []]
    for j in range(1, J):
        for o in range(3):
            parent = np.ascontiguousarray(coeffs[J + 1 - (j + 1)][o], dtype=np.float64)
            child = np.ascontiguousarray(aligned(coeffs[J + 1 - j][o], o, s), dtype=np.float64)
            clear_c = np.ascontiguousarray(aligned(clear[j - 1], o, s))
            fp, fc = 1e-8 * np.abs(parent).max(), 1e-8 * np.abs(child).max()
            out = np.empty(child.size)
            n = kernels()["clear_pairs"](parent, child, clear[j], clear_c, fp, fc, out)
            ratios[o].append(out[:n].copy())
            P = np.repeat(np.repeat(parent, 2, 0), 2, 1)
            ok = ((np.abs(P) >= fp) & (np.abs(child) >= fc) & (clear_c == 1)
                  & (np.repeat(np.repeat(clear[j], 2, 0), 2, 1) == 1))
            signs[o].append((np.sign(P[ok]) == np.sign(child[ok])).astype(np.float64))
    return ([np.concatenate(r) if r else np.zeros(0) for r in ratios],
            [np.concatenate(g) if g else np.zeros(0) for g in signs])


def make_abacus(fields, masks, name: str, levels: int, nbins: int = 200) -> "Abacus":
    """A corpus abacus over several fields (the "Make abacus" action): each field, with its
    non-finite pixels and its region mask (``None`` for none) excluded, mirror-extended and
    decomposed to ``levels``; the clear pairs pooled per orientation; ``scenes`` names the
    fields. Saved with :meth:`Abacus.save`, read by ``abacus = file``."""
    ratios, signs = [[], [], []], [[], [], []]
    scenes = []
    for i, (field, mask) in enumerate(zip(fields, masks)):
        vals = np.asarray(getattr(field, "values", field), dtype=np.float64)
        finite = np.isfinite(vals)
        region = ~finite if mask is None else (~finite | np.asarray(mask, dtype=bool))
        shape = dwt.working_shape(vals.shape, levels)
        x = dwt.mirror(np.where(finite, vals, vals[finite].mean() if finite.any() else 0.0), shape)
        clear = clear_masks(dwt.mirror(region.astype(np.float64), shape) > 0.5, name, levels)
        r, g = _collect_pairs(dwt.pyramid(x, name, levels), clear, name)
        for o in range(3):
            ratios[o].append(r[o])
            signs[o].append(g[o])
        scenes.append(str(getattr(field, "name", i)))
    ab = Abacus.from_ratios([np.concatenate(r) for r in ratios], [np.concatenate(g) for g in signs],
                            name, nbins)
    ab.scenes = scenes
    return ab


@dataclasses.dataclass
class Abacus:
    """The density of ``ln|eta|`` per orientation on ``edges`` (``nbins + 1``), smoothed by a
    Gaussian of one bin and floored at 1e-6; ``lo`` and ``hi`` are the 0.5 % and 99.5 %
    quantiles of the clear pairs, ``sign_freq`` the share of children with the parent's sign."""

    edges: np.ndarray
    hist: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    wavelet: str
    n_pairs: np.ndarray
    sign_freq: np.ndarray
    scenes: list = dataclasses.field(default_factory=list)

    @property
    def log_hist(self) -> np.ndarray:
        return np.log(self.hist)

    @property
    def cdf(self) -> np.ndarray:
        c = np.cumsum(self.hist, axis=1)
        return c / c[:, -1:]

    @classmethod
    def from_ratios(cls, ratios, signs, name: str, nbins: int = 200) -> "Abacus":
        from scipy.ndimage import gaussian_filter1d

        pooled = np.concatenate([np.asarray(r, dtype=np.float64) for r in ratios])
        if pooled.size == 0:
            raise ValueError("no clear parent-child pair at all: every coefficient touches a "
                             "region (or a held-out block)")
        lo_all, hi_all = np.quantile(pooled, [0.005, 0.995])
        edges = np.linspace(lo_all, hi_all, int(nbins) + 1)
        hist, lo, hi, n, sf = [], [], [], [], []
        for r, s in zip(ratios, signs):
            r = np.asarray(r, dtype=np.float64)
            h = np.histogram(r, bins=edges)[0].astype(np.float64) if r.size else np.zeros(nbins)
            h = gaussian_filter1d(h, 1.0, mode="nearest")
            h = np.maximum(h / max(h.sum(), 1.0), 1e-6)
            hist.append(h / h.sum())
            q = np.quantile(r, [0.005, 0.995]) if r.size else (lo_all, hi_all)
            lo.append(float(q[0]))
            hi.append(float(q[1]))
            n.append(int(r.size))
            sf.append(float(np.mean(s)) if np.size(s) else 0.5)
        return cls(edges, np.stack(hist), np.asarray(lo), np.asarray(hi), str(name),
                   np.asarray(n), np.asarray(sf))

    @classmethod
    def from_pyramid(cls, coeffs, clear, name: str, nbins: int = 200, *, shift=None) -> "Abacus":
        """Over levels ``j = 1..J-1``: the pairs of a clear parent at ``j + 1`` and its clear
        children at ``j`` (the (2k, 2k+1)^2 rule, the child band rolled by the bank's child
        shift along its low-pass axis), both above ``1e-8`` of their level's maximum, pooled
        per orientation. ``clear[i]`` is level ``i + 1``'s mask. ``shift`` overrides
        :func:`dwt.child_shift` (tests)."""
        ratios, signs = _collect_pairs(coeffs, clear, name if shift is None else shift)
        return cls.from_ratios(ratios, signs, name, nbins)

    def save(self, path) -> None:
        np.savez(path, edges=self.edges, hist=self.hist, lo=self.lo, hi=self.hi,
                 wavelet=np.array(self.wavelet), n_pairs=self.n_pairs,
                 sign_freq=self.sign_freq, scenes=np.array(list(self.scenes), dtype=str))

    @classmethod
    def load(cls, path, wavelet: "str | None" = None) -> "Abacus":
        with np.load(path, allow_pickle=False) as z:
            ab = cls(z["edges"], z["hist"], z["lo"], z["hi"], str(z["wavelet"]), z["n_pairs"],
                     z["sign_freq"], [str(s) for s in z["scenes"]])
        if wavelet is not None and ab.wavelet != wavelet:
            raise ValueError(f"{path}: an abacus of the {ab.wavelet} bank cannot drive a "
                             f"{wavelet} cascade -- make one for {wavelet}")
        return ab


class Sparsity:
    """The cycle-spun soft threshold: a fresh shift per call from ``rng``."""

    def __init__(self, name: str, level: int, lam: float, rng):
        self.name, self.level, self.lam, self.rng = str(name), int(level), float(lam), rng

    def prox(self, g, t: float) -> np.ndarray:
        z, shift = dwt.cycle_shift(np.asarray(g, dtype=np.float64), self.rng, self.level)
        c = dwt.soft_threshold(dwt.pyramid(z, self.name, self.level), self.lam * float(t))
        return dwt.cycle_unshift(dwt.inverse(c, self.name), shift)

    def penalty(self, x) -> float:
        return self.lam * dwt.detail_l1(dwt.pyramid(x, self.name, self.level))


def auto_lambda(y, clear1, name: str, lam_rel: float) -> float:
    """``lam_rel sigma sqrt(2 ln N)``, ``sigma`` the median absolute clear level-1 detail of
    ``y`` over 0.6745 (the notebooks' rule)."""
    details = dwt.pyramid(y, name, 1)[1]
    sel = np.asarray(clear1, dtype=bool)
    d = np.concatenate([np.abs(b[sel]).ravel() for b in details])
    if d.size == 0:
        d = np.concatenate([np.abs(b).ravel() for b in details])
    sigma = float(np.median(d)) / 0.6745
    return float(lam_rel) * sigma * float(np.sqrt(2.0 * np.log(np.size(y))))
