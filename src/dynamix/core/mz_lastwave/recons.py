# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Reconstruction from the multiscale edges: LastWave's ``e2recons``, ported exactly.

The authors' alternating projections (``ext2_proj.c``): an initial pass projects a zero
transform onto the edge constraints and synthesises it; every iteration then analyses the current
image, projects its transform and synthesises it again. The projection of level l runs along each
row of ``Wx`` and each column of ``Wy`` between consecutive anchors (the first sample with target
0, the extrema strictly inside, the last sample with target 0), adding the ``W2_interp`` sinh
interpolation of the two anchor errors with decay ``a``, and sets the last sample to 0. With
clipping on, the modulus between anchors is then held to a running minimum towards the segment
minimum wherever the argument stays along the axis. The coarse channel is replaced by the pinned
one before every synthesis.

The decay is ``a = exp(−κ / 2**l)``. At κ = ``KAPPA_LASTWAVE`` = 2 ln 5.8 the C's own line
``a = 1 / pow(5.8, 2 / 2**l)`` computes it, the same number to rounding and bit for bit with the
C; κ = 1 is the published constant.

Two coarse modes extend the authors' algorithm: ``"thumbnail"`` pins the coarse decoded from the
working field's 2**J subsample by trigonometric interpolation (the paper's coding mode), and
``"none"`` pins a zero coarse, reconstructing from the edges alone.

In ``"mirror"`` mode the whole algorithm runs on the 2N mirrored working field, from the
working-field extrema and coarse (``pipeline.analyze``, ``Transform.S_full[J]``), and only the
result is cropped to the field's grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dynamix.core.mz_edges import _fourier_upsample_torus, _snr_db
from dynamix.core.mz_lastwave._kernels import kernels
from dynamix.core.mz_lastwave.transform import BORDERS, Transform, _decompose, _recompose

#: The decay constant of LastWave's ``ext2_proj.c``: ``1 / 5.8**(2 / s) = exp(−κ / s)``.
KAPPA_LASTWAVE = 2 * math.log(5.8)

COARSE = ("full", "thumbnail", "none")
MODES = ("fixed", "converge")

#: Consecutive rises of the constraint residual that stop a ``"converge"`` run.
_RISING_RUN = 3


@dataclass(frozen=True, eq=False)
class ReconState:
    """Where a reconstruction stands, enough to continue it exactly (LastWave's ``-i``).

    ``image`` is the current iterate on the working field; ``resid`` the constraint residual of
    every iterate 0..``iterations``; ``change`` the relative change of the last iteration (None
    after the initial pass alone); ``best_image`` the minimum-residual iterate on the field's grid
    and ``best_iteration`` its index; ``detail`` the edges-only score target (``coarse="none"``);
    ``settings`` what the run was computed with. A state is never modified: continuing from it
    returns a new one. The transform workspace is rebuilt by each call (it holds 3J + 3 arrays of
    the working field, and a state may be kept in a cache).
    """

    image: np.ndarray
    iterations: int
    resid: tuple
    change: float | None
    best_image: np.ndarray
    best_iteration: int
    detail: np.ndarray | None
    settings: tuple


def _decay(kappa, l):
    s = 2 ** l
    if kappa == KAPPA_LASTWAVE:
        return 1.0 / math.pow(5.8, 2.0 / s)
    return math.exp(-kappa / s)


class _Engine:
    """One call's workspace: the transform arrays on the working field, the anchors of every
    level (``W2_point_repr_cartesian``), the decays and the pinned coarse."""

    def __init__(self, extrep, pin, shape, J, border, kappa, clip):
        self.t = Transform(shape, J, border)
        self.pin = pin
        self.clip = clip
        self.decay = [None] + [_decay(kappa, l) for l in range(1, J + 1)]
        self.ext, self.hor, self.ver, self.mag, self.idx = [None], [None], [None], [None], [None]
        for l in range(1, J + 1):
            ext = np.ascontiguousarray(extrep.mask[l], dtype=np.bool_)
            h, v = extrep.cartesian(l)
            self.ext.append(ext)
            self.hor.append(np.ascontiguousarray(h))
            self.ver.append(np.ascontiguousarray(v))
            self.mag.append(np.ascontiguousarray(extrep.denormalised(l)[0]))
            self.idx.append(np.flatnonzero(ext))
        if clip:
            self.mod = np.empty(self.t.full_shape)
            self.arg = np.empty(self.t.full_shape)

    def zero(self):
        for l in range(1, self.t.J + 1):
            self.t.Wx_full[l].fill(0.0)
            self.t.Wy_full[l].fill(0.0)

    def analyse(self, img):
        """The transform of ``img`` into the workspace; returns its constraint residual,
        ``sqrt(Σ (Wx − hor)² + (Wy − ver)²)`` over the extrema of every level."""
        _decompose(self.t, img)
        r2 = 0.0
        for l in range(1, self.t.J + 1):
            i = self.idx[l]
            dx = self.t.Wx_full[l].ravel()[i] - self.hor[l].ravel()[i]
            dy = self.t.Wy_full[l].ravel()[i] - self.ver[l].ravel()[i]
            r2 += float(np.dot(dx, dx) + np.dot(dy, dy))
        return math.sqrt(r2)

    def project(self):
        """``W2_point_repr_projection`` on the workspace transform, in place."""
        k = kernels()
        for l in range(1, self.t.J + 1):
            h, v = self.t.Wx_full[l], self.t.Wy_full[l]
            k.proj1_rows(h, self.ext[l], self.hor[l], self.decay[l])
            k.proj1_cols(v, self.ext[l], self.ver[l], self.decay[l])
            if self.clip:
                k.polar(h, v, self.mod, self.arg)
                k.clip_rows(self.mod, self.arg, self.ext[l], self.mag[l])
                k.clip_cols(self.mod, self.arg, self.ext[l], self.mag[l])
                k.cartesian(self.mod, self.arg, h, v)

    def synthesise(self, out):
        _recompose(self.t, self.pin, out)


def _pinned_coarse(coarse_image, coarse, J, border, shape):
    """The coarse the synthesis pins: the working field's own (``"full"``), the decode of the
    working field's 2**J thumbnail (``"thumbnail"``), or zeros (``"none"``).

    The thumbnail is the working-field subsample ``coarse_image[::2**J, ::2**J]``. In mirror mode
    LastWave's coarse sits half a sample off the grid, so on the 2N field it is whole-sample
    symmetric (``S[i] == S[-i mod 2N]``): its thumbnail holds ``ny / 2**J + 1`` distinct rows,
    the fold row at ``ny`` included, and a half-sample mirror of the field's own quadrant would
    misplace the mirrored half by one sample. The trigonometric decode splits the Nyquist term,
    so it takes an even thumbnail per axis, which a periodic field gives when its grid divides
    by 2**(J+1)."""
    if coarse == "full":
        return coarse_image
    if coarse == "none":
        return np.zeros(coarse_image.shape)
    ny, nx = shape
    step = 2 ** J
    if ny % step or nx % step:
        raise ValueError(f"the thumbnail coarse needs the grid divisible by 2**J = {step}; "
                         f"this grid is {ny} x {nx}")
    thumb = np.ascontiguousarray(coarse_image[::step, ::step])
    if thumb.shape[0] % 2 or thumb.shape[1] % 2:
        raise ValueError(
            f"the thumbnail coarse needs an even number of 2**J samples per axis (its "
            f"trigonometric decode splits the Nyquist term); this {ny} x {nx} grid with border "
            f"{border!r} gives {thumb.shape[0]} x {thumb.shape[1]}: use a grid divisible by "
            f"2**(J+1) = {2 * step}")
    return np.ascontiguousarray(_fourier_upsample_torus(thumb, step), dtype=np.float64)


def _relative_change(new, old):
    """``‖new − old‖ / ‖new‖``; 0 for two identical images."""
    diff = float(np.linalg.norm(new - old))
    norm = float(np.linalg.norm(new))
    if norm > 0:
        return diff / norm
    return 0.0 if diff == 0 else math.inf


def _rising(resid):
    """The residual rose over each of the last ``_RISING_RUN`` iterations."""
    return len(resid) > _RISING_RUN and all(
        resid[-i] > resid[-i - 1] for i in range(1, _RISING_RUN + 1))


def _converge_stop(iterations, change, resid, tol, cap):
    if change is not None and change < tol:
        return "converged"
    if _rising(resid):
        return "residual rising"
    if iterations >= cap:
        return "cap"
    return None


def e2recons(values, extrep, coarse_image, J, *, k=20, kappa=1.0, clip=False, coarse="full",
             border="mirror", mode="fixed", tol=1e-3, state=None, progress=None, cancel=None):
    """Reconstruct a field from the extrema of its working field.

    ``extrep`` and ``coarse_image`` are the working field's (``pipeline.analyze`` and
    ``Transform.S_full[J]``, the fact-scaled coarse); ``values`` is the field itself, which scores
    the result. ``coarse`` is ``"full"``, ``"thumbnail"`` or ``"none"``. Without ``state`` the
    initial pass runs first (iteration 0); with one, the run continues from it.

    ``mode="fixed"`` adds ``k`` iterations. ``mode="converge"`` iterates up to ``k`` iterations in
    total and stops early when ``‖f_k − f_{k−1}‖ / ‖f_k‖ < tol`` ("converged") or when the
    constraint residual has risen over ``_RISING_RUN`` iterations running ("residual rising"),
    returning the minimum-residual iterate; at the cap it stops with "cap". A state that already
    meets its stop returns at once with the same image. ``cancel()`` is consulted once per
    iteration and a true value raises ``ComputeCancelled``; ``progress(msg, frac)`` follows each
    iteration.

    Returns ``(image, diag, state)``: the (ny, nx) float64 image; ``diag`` with ``iterations``
    (the total, the initial pass counting as 0), ``stop`` ("fixed", "converged", "cap" or
    "residual rising"), ``resid`` (per iterate), ``best_iteration``, ``snr_db`` (mean-removed,
    against the field, or for ``coarse="none"`` against the field minus the coarse as the
    synthesis delivers it), ``kappa``, ``clip``, ``coarse``, ``border`` and ``mode``; and the
    :class:`ReconState` to continue from.
    """
    if coarse not in COARSE:
        raise ValueError(f"unknown coarse {coarse!r}; choices: {COARSE}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choices: {MODES}")
    if border not in BORDERS:
        raise ValueError(f"border must be one of {BORDERS}; got {border!r}")
    values = np.asarray(values, dtype=np.float64)
    ny, nx = values.shape
    J, k = int(J), int(k)
    full_shape = (ny, nx) if border == "periodic" else (2 * ny, 2 * nx)
    coarse_image = np.ascontiguousarray(coarse_image, dtype=np.float64)
    if coarse_image.shape != full_shape or extrep.mask[1].shape != full_shape:
        raise ValueError(
            f"e2recons takes the extrema and coarse of the working field, {full_shape[0]} x "
            f"{full_shape[1]} for a {ny} x {nx} field with border {border!r} "
            f"(pipeline.analyze and Transform.S_full[J]); got a {coarse_image.shape} coarse and "
            f"{extrep.mask[1].shape} extrema")
    settings = ((ny, nx), J, border, float(kappa), bool(clip), coarse)
    if state is not None and state.settings != settings:
        raise ValueError(f"the state was computed with {state.settings}, this call asks for "
                         f"{settings} (shape, J, border, kappa, clip, coarse)")

    def engine():
        pin = _pinned_coarse(coarse_image, coarse, J, border, (ny, nx))
        return _Engine(extrep, pin, (ny, nx), J, border, kappa, bool(clip))

    if state is None:
        eng = engine()
        img = np.empty(full_shape)
        detail = None
        eng.zero()
        if coarse == "none":
            _recompose(eng.t, coarse_image, img)
            detail = values - img[:ny, :nx]
        eng.project()
        eng.synthesise(img)
        resid = [eng.analyse(img)]
        iterations, change, owned = 0, None, True
        best, best_it = img[:ny, :nx].copy(), 0
    else:
        eng = None
        img, iterations, change, owned = state.image, state.iterations, state.change, False
        resid = list(state.resid)
        best, best_it, detail = state.best_image, state.best_iteration, state.detail

    added, spare = 0, None
    while True:
        if mode == "fixed":
            if added >= k:
                stop = "fixed"
                break
        else:
            stop = _converge_stop(iterations, change, resid, tol, k)
            if stop is not None:
                break
        if cancel is not None and cancel():
            from dynamix.core.wtmm_backend import ComputeCancelled
            raise ComputeCancelled("M–Z reconstruction cancelled")
        if eng is None:
            eng = engine()
            eng.analyse(img)                       # the transform of the state's image
        if spare is None:
            spare = np.empty(full_shape)
        eng.project()
        eng.synthesise(spare)
        change = _relative_change(spare, img)
        img, spare = spare, (img if owned else None)
        owned = True
        resid.append(eng.analyse(img))
        iterations += 1
        added += 1
        if resid[-1] < resid[best_it]:
            best, best_it = img[:ny, :nx].copy(), iterations
        if progress is not None:
            done = added if mode == "fixed" else iterations
            progress(f"M–Z reconstruction {done}/{k}", done / k)

    out = best.copy() if stop == "residual rising" else img[:ny, :nx].copy()
    snr = _snr_db(values if detail is None else detail, out)
    diag = {"iterations": iterations, "stop": stop, "resid": list(resid),
            "best_iteration": best_it, "snr_db": snr, "kappa": float(kappa), "clip": bool(clip),
            "coarse": coarse, "border": border, "mode": mode}
    new = ReconState(image=img, iterations=iterations, resid=tuple(resid), change=change,
                     best_image=best, best_iteration=best_it, detail=detail, settings=settings)
    return out, diag, new
