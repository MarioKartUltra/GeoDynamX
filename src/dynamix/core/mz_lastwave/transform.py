# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""LastWave's dyadic wavelet transform ``dwt2d`` and its inverse ``dwt2r``, by direct FIR
convolution with the p3.1 (analysis) and p3.2 (synthesis) filters.

Level l (1..J) filters at dilation ``2**(l-1)``: the coarse ``S_l`` is H1 along the rows then the
columns of ``S_{l-1}``; ``Wx_l`` is G1 along the rows (K1, a unit Dirac, along the columns) and
``Wy_l`` K1 along the rows then G1 along the columns. The S chain runs unscaled; every level is then
multiplied by ``fact(l)`` (``ddecomp2.c``). Synthesis divides the coarse by ``fact(J)`` and each
level's W by ``fact(l)`` before the H2 / G2 / K2 filters.

Two corrections to the C, neither of which changes a square J >= 2 result: every convolution uses
the length of its own axis, so non-square grids transform and reconstruct correctly; and synthesis
divides ``S_J`` by the same ``fact(J)`` analysis multiplied it by, which makes J = 1 an identity.

Registration in index space (x = column j, y = row i): the input and the reconstruction sit at
(j, i); ``S_l`` and levels l >= 2 at (j − ½, i − ½); level 1 ``Wx`` at (j − ½, i) and ``Wy`` at
(j, i − ½).
"""
from __future__ import annotations

import numpy as np

from dynamix.core.mz_edges import _mirror2d
from dynamix.core.mz_lastwave._kernels import kernels

#: The p3.1 (H1, G1, K1) and p3.2 (H2, G2, K2) filters as ``(size, shift, symmetry, values)``,
#: exactly as LastWave's filter files hold them: ``values[0]`` is the centre tap and
#: ``values[1:]`` the one-sided taps, mirrored with the symmetry sign.
FILTERS = {
    "H1": (3, 1, 1.0, (0.0, 0.375, 0.125)),
    "G1": (2, 1, -1.0, (0.0, 0.5)),
    "K1": (1, 0, 1.0, (1.0,)),
    "H2": (3, -1, 1.0, (0.0, 0.375, 0.125)),
    "G2": (4, -1, -1.0, (0.0, -0.6875, -0.21875, -0.03125)),
    "K2": (4, 0, 1.0, (0.65625, 0.1171875, 0.046875, 0.0078125)),
}

#: The p3.1 factors block (about 1/λ_l per level; the filter reader keeps up to twelve).
FACT1 = (0.667969, 0.890625, 0.971591, 0.994186) + (1.0,) * 8

BORDERS = ("mirror", "periodic")


def fact(l):
    """The level-l scaling ``dwt2d`` applies to S_l, Wx_l and Wy_l (``ddecomp2.c``)."""
    return 2.0 ** 0.5 / 1.8 if l == 1 else 2.0 ** (l / 2.0)


def _l1r1(shift, scale):
    """The offsets of the first left and right taps (``convol2.c``): a half-sample shift at scale
    1 for a shifted filter, centred dilated taps above."""
    if scale == 1:
        return {1: (1, 0), -1: (0, 1), 0: (1, 1)}[shift]
    return (scale // 2, scale // 2) if shift else (scale, scale)


def _taps(name, scale):
    size, shift, sym, f = FILTERS[name]
    l1, r1 = _l1r1(shift, scale)
    return np.asarray(f, np.float64), float(sym), l1, r1, size


class Transform:
    """The transform of one field, and the workspace that computed it.

    ``S[l]``, ``Wx[l]``, ``Wy[l]`` (l = 1..J; index 0 unused) are the fact-scaled channels on the
    field's own ``shape``. The algorithm runs on ``full_shape``: the field itself for
    ``border="periodic"``, its 2N mirror for ``"mirror"``, whose arrays ``S_full``, ``Wx_full``,
    ``Wy_full`` hold every level and of which ``S``, ``Wx``, ``Wy`` are the primary-quadrant
    views. Synthesis reads the full arrays and leaves them unchanged.
    """

    def __init__(self, shape, J, border):
        ny, nx = shape
        self.J = J
        self.shape = (ny, nx)
        self.border = border
        self.full_shape = (ny, nx) if border == "periodic" else (2 * ny, 2 * nx)
        blank = lambda: np.empty(self.full_shape)   # noqa: E731
        self.S_full = [None] + [blank() for _ in range(J)]
        self.Wx_full = [None] + [blank() for _ in range(J)]
        self.Wy_full = [None] + [blank() for _ in range(J)]
        self._t1, self._t2, self._t3 = blank(), blank(), blank()
        crop = self._primary
        self.S = [None] + [crop(a) for a in self.S_full[1:]]
        self.Wx = [None] + [crop(a) for a in self.Wx_full[1:]]
        self.Wy = [None] + [crop(a) for a in self.Wy_full[1:]]

    def _primary(self, a):
        ny, nx = self.shape
        return a if a.shape == self.shape else a[:ny, :nx]


def _decompose(t, x):
    """``dwt2d``'s arithmetic on the working field ``x`` (``t.full_shape``), into ``t``'s full
    arrays. K1 is the unit Dirac, so its passes are the identity and are skipped."""
    k = kernels()
    for l in range(t.J):
        s = 2 ** l
        f, sym, l1, r1, size = _taps("H1", s)
        k.conv_rows(x, t._t1, f, sym, l1, r1, s, size)
        k.conv_cols(t._t1, t.S_full[l + 1], f, sym, l1, r1, s, size, 1.0)
        g, gsym, gl1, gr1, gsize = _taps("G1", s)
        k.conv_rows(x, t._t1, g, gsym, gl1, gr1, s, gsize)
        fct = fact(l + 1)
        k.scale_into(t._t1, t.Wx_full[l + 1], fct)
        k.conv_cols(x, t.Wy_full[l + 1], g, gsym, gl1, gr1, s, gsize, fct)
        x = t.S_full[l + 1]
    for l in range(1, t.J + 1):
        k.scale_into(t.S_full[l], t.S_full[l], fact(l))
    return t


def _recompose(t, SJ, out):
    """``dwt2r``'s arithmetic from ``t``'s full W arrays and a fact-scaled coarse ``SJ`` (both
    ``t.full_shape``) into ``out``. ``t``'s arrays are read, never written."""
    k = kernels()
    S = t._t3
    k.scale_into(SJ, S, 1.0 / fact(t.J))
    for l in range(t.J - 1, -1, -1):
        s = 2 ** l
        inv = 1.0 / fact(l + 1)
        dst = out if l == 0 else t._t3
        f, sym, l1, r1, size = _taps("H2", s)
        k.conv_rows(S, t._t1, f, sym, l1, r1, s, size)
        k.conv_cols(t._t1, dst, f, sym, l1, r1, s, size, 1.0)
        g, gsym, gl1, gr1, gsize = _taps("G2", s)
        c, csym, cl1, cr1, csize = _taps("K2", s)
        k.scale_into(t.Wx_full[l + 1], t._t2, inv)
        k.conv_rows(t._t2, t._t1, g, gsym, gl1, gr1, s, gsize)
        k.conv_cols(t._t1, t._t2, c, csym, cl1, cr1, s, csize, 1.0)
        k.add_into(dst, t._t2)
        k.scale_into(t.Wy_full[l + 1], t._t2, inv)
        k.conv_rows(t._t2, t._t1, c, csym, cl1, cr1, s, csize)
        k.conv_cols(t._t1, t._t2, g, gsym, gl1, gr1, s, gsize, 1.0)
        k.add_into(dst, t._t2)
        S = dst
    return out


def dwt2d(values, J, *, border="mirror"):
    """The J-level transform of a 2-D field.

    ``border="periodic"`` runs LastWave's periodic transform on the field as given;
    ``"mirror"`` runs the identical algorithm on the field's 2N mirror and crops every channel to
    the field's own grid. Raises ``ValueError`` for J < 1 or a grid shorter than 2**J on a side.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"M–Z (LastWave) transforms a 2-D field; got {x.ndim} dimensions")
    if border not in BORDERS:
        raise ValueError(f"border must be one of {BORDERS}; got {border!r}")
    J = int(J)
    if J < 1:
        raise ValueError(f"M–Z (LastWave) needs at least one level; got J = {J}")
    ny, nx = x.shape
    if 2 ** J > min(ny, nx):
        raise ValueError(f"M–Z (LastWave) needs 2**J <= the shorter side of the grid: J = {J} "
                         f"asks for {2 ** J} px and the grid is {ny} x {nx}")
    work = np.ascontiguousarray(_mirror2d(x) if border == "mirror" else x)
    return _decompose(Transform(x.shape, J, border), work)


def idwt2d(t):
    """The field reconstructed from ``t`` (``dwt2r``), on the field's own grid."""
    out = np.empty(t.full_shape)
    _recompose(t, t.S_full[t.J], out)
    return out if out.shape == t.shape else out[:t.shape[0], :t.shape[1]].copy()


def polar(t, l):
    """``(M, A)`` of level l: ``M = sqrt(Wx² + Wy²)``, ``A = atan2(Wy, Wx)`` (0 where both
    are 0)."""
    h = np.ascontiguousarray(t.Wx[l])
    v = np.ascontiguousarray(t.Wy[l])
    M = np.empty(t.shape)
    A = np.empty(t.shape)
    kernels().polar(h, v, M, A)
    return M, A


def _symmetry_centre(profile):
    """``(centre, "sym" | "anti")`` of a finitely supported 1-D profile: the midpoint of its
    support (first to last non-zero sample, in index units) and whether the profile is even or
    odd about it. Measures where a channel registers an impulse; the support must not wrap
    around the ends."""
    p = np.asarray(profile, dtype=np.float64)
    nz = np.flatnonzero(p)
    if nz.size == 0:
        raise ValueError("an all-zero profile has no centre")
    lo, hi = int(nz[0]), int(nz[-1])
    seg = p[lo:hi + 1]
    tol = 1e-12 * float(np.max(np.abs(seg)))
    if np.all(np.abs(seg - seg[::-1]) <= tol):
        kind = "sym"
    elif np.all(np.abs(seg + seg[::-1]) <= tol):
        kind = "anti"
    else:
        raise ValueError("the profile is neither even nor odd about its support midpoint")
    return (lo + hi) / 2, kind
