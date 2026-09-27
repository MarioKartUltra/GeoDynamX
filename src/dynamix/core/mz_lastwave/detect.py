# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Multiscale edge detection: LastWave's ``extrema2`` (``W2_compute_point``), ported exactly.

At every level the gradient ``(Wx, Wy)`` is taken to polar form, ``M = sqrt(Wx² + Wy²)`` and
``A = atan2(Wy, Wx)`` (0 where both vanish). A point is an extremum when ``M`` is a local maximum
along its gradient direction quantised to the x or y axis, with a neighbour whose gradient points
the opposite way counted as lower (the authors' default orientation rule); a point whose comparison
axis leaves the grid is skipped. At level 1 a second pass in raster order fills the gaps of
diagonal edges; it reads the extrema it has already added, so it runs sequentially. The magnitudes
are then normalised as the C stores them, ``(M · factors[l−1]) · (1 / fact_l)``, where ``factors``
are the p3.1 G filter's per-level factors and ``fact_l`` the level scaling of the transform.

``colocate_l1`` detects level 1 on the 2-tap co-located gradient ``Wx1c[i, j] = ½(Wx1[i−1, j] +
Wx1[i, j])``, ``Wy1c[i, j] = ½(Wy1[i, j−1] + Wy1[i, j])``, which sits at (j − ½, i − ½) like the
coarser levels; the stored magnitude and argument stay those of the raw ``Wx1``, ``Wy1`` at the
detected index. The neighbour before the first row (column) follows the transform's border: the
last row (column) for ``"periodic"``, and the first row (column) itself for ``"mirror"``, which is
its neighbour on the 2N mirrored field the arrays were cropped from.

The detection kernels are written in the numba subset of Python and compiled on first use
(``cache=True``, ``fastmath=False``); the polar and cartesian conversions are the transform's
kernels. numba is required and imported lazily.
"""
from __future__ import annotations

import math

import numpy as np

from dynamix.core.mz_lastwave._kernels import _NEEDS_NUMBA, kernels
from dynamix.core.mz_lastwave.transform import BORDERS, FACT1, fact

_KERNELS: dict = {}


def _g1_factor(l: int) -> float:
    """``filterg1->factors[l−1]``; 1 beyond the factors the filter holds."""
    return FACT1[l - 1] if l <= len(FACT1) else 1.0


# The kernels below are written in the numba-friendly subset of Python and compiled on first use
# by _detect_kernels(), which also rebinds the helper names to their compiled versions: numba
# resolves a kernel's calls through its module globals when it compiles, and a kernel that reached
# its helpers through closure cells instead would miss the on-disk cache in every new process.

def _double_dir_py(argument):
    """``W2_double_dir``: the argument quantised to the axes, 0, 2, 4 or 6 (in 45° steps)."""
    fdir = argument * 2.0 / math.pi + 4.0
    rdir = int(fdir - 0.5 if fdir < 0 else fdir + 0.5)
    return 2 * (rdir % 4)


def _direction_py(argument):
    """``W2_direction``'s first direction: the argument quantised to 0..7 (in 45° steps)."""
    fdir = argument * 4.0 / math.pi + 8.0
    rdir = int(fdir - 0.5 if fdir < 0 else fdir + 0.5)
    return rdir % 8


def _opposit_dir_py(arg1, arg2):
    """``W2_opposit_dir``: the two gradients point within 3 × 45° of opposite ways."""
    diff = arg1 - arg2
    if diff > math.pi:
        diff = diff - 2 * math.pi
    elif diff < -math.pi:
        diff = diff + 2 * math.pi
    d = _direction(diff)
    return d == 3 or d == 4 or d == 5


def _axis_max_py(m, k, k0):
    """The orientation-free maximum test along the axis with flat offset ``k0``."""
    bol2 = m[k] > m[k + k0] and m[k] >= m[k - k0]
    bol4 = m[k] > m[k - k0] and m[k] >= m[k + k0]
    return bol2 or bol4


_double_dir = _double_dir_py                    # rebound to the njit versions on compile
_direction = _direction_py
_opposit_dir = _opposit_dir_py
_axis_max = _axis_max_py


def _compute_point_py(m, a, nrow, ncol, level, ext):
    """``W2_compute_point`` with orientation on, marking the extrema in the flat bool ``ext``."""
    for i in range(nrow):
        for j in range(ncol):
            k = i * ncol + j
            d = _double_dir(a[k])
            if d == 0:
                k0 = -1
            elif d == 4:
                k0 = 1
            elif d == 2:
                k0 = -ncol
            else:
                k0 = ncol
            if (((i == 0 or i == nrow - 1) and (d == 2 or d == 6))
                    or ((j == 0 or j == ncol - 1) and (d == 0 or d == 4))):
                continue
            bol1 = m[k] > m[k + k0] or _opposit_dir(a[k], a[k + k0])
            bol2 = bol1 and (m[k] >= m[k - k0] or _opposit_dir(a[k], a[k - k0]))
            bol3 = m[k] > m[k - k0] or _opposit_dir(a[k], a[k - k0])
            bol4 = bol3 and (m[k] >= m[k + k0] or _opposit_dir(a[k], a[k + k0]))
            if bol2 or bol4:
                ext[k] = True

    if level == 1:
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                k = i * ncol + j
                d0 = _direction(a[k])
                if d0 == 1 or d0 == 5:
                    bol1 = ext[k - 1 + ncol] and (ext[k - ncol] or ext[k - ncol + 1] or ext[k + 1])
                    bol2 = ext[k + 1 - ncol] and (ext[k + ncol] or ext[k + ncol - 1] or ext[k - 1])
                elif d0 == 3 or d0 == 7:
                    bol1 = ext[k + 1 + ncol] and (ext[k - ncol] or ext[k - ncol - 1] or ext[k - 1])
                    bol2 = ext[k - 1 - ncol] and (ext[k + ncol] or ext[k + ncol + 1] or ext[k + 1])
                else:
                    bol1 = False
                    bol2 = False
                if (bol1 or bol2) and not ext[k]:
                    if _axis_max(m, k, 1) or _axis_max(m, k, ncol):
                        ext[k] = True


def _detect_kernels() -> dict:
    """Lazily njit(cache=True, fastmath=False)-compile the detection kernels.

    Raises ``RuntimeError`` naming numba when it is not installed.
    """
    global _double_dir, _direction, _opposit_dir, _axis_max
    if not _KERNELS:
        try:
            from numba import njit
        except ImportError:
            raise RuntimeError(_NEEDS_NUMBA) from None
        jit = njit(cache=True, fastmath=False)
        _double_dir = jit(_double_dir_py)
        _direction = jit(_direction_py)
        _opposit_dir = jit(_opposit_dir_py)
        _axis_max = jit(_axis_max_py)
        _KERNELS["compute_point"] = jit(_compute_point_py)
    return _KERNELS


def _polar(hor: np.ndarray, ver: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.empty(hor.shape)
    a = np.empty(hor.shape)
    kernels().polar(hor, ver, m, a)
    return m, a


def _colocated(hor: np.ndarray, ver: np.ndarray, border: str) -> tuple[np.ndarray, np.ndarray]:
    """The level-1 ``(Wx1c, Wy1c)``: ``Wx1`` averaged with the row before, ``Wy1`` with the column
    before, the neighbour of the first row (column) taken as ``border`` sets it."""
    if border == "periodic":
        up = np.roll(hor, 1, axis=0)
        left = np.roll(ver, 1, axis=1)
    else:
        up = np.concatenate([hor[:1], hor[:-1]], axis=0)
        left = np.concatenate([ver[:, :1], ver[:, :-1]], axis=1)
    return 0.5 * (up + hor), 0.5 * (left + ver)


class Extrep:
    """The extrema representation of ``extrema2``: per level l = 1..J (index 0 unused),
    ``mask[l]`` (bool, ny × nx), ``mag[l]`` (the normalised magnitude as the C stores it) and
    ``arg[l]`` (the gradient argument), both 0 away from the extrema."""

    def __init__(self, J: int, mask: list, mag: list, arg: list):
        self.J = J
        self.mask = mask
        self.mag = mag
        self.arg = arg

    def denormalised(self, l: int) -> tuple[np.ndarray, np.ndarray]:
        """``(mag, arg)`` of level ``l`` with the magnitude back in transform units, as
        ``W2_point_pic_denormalize`` leaves it for the reconstruction."""
        return (self.mag[l] / _g1_factor(l)) * fact(l), self.arg[l]

    def cartesian(self, l: int) -> tuple[np.ndarray, np.ndarray]:
        """``(hor, ver)`` = ``mag·cos(arg)``, ``mag·sin(arg)`` of the denormalised extrema of level
        ``l`` (``W2_point_repr_cartesian``); 0 away from the extrema."""
        mag, arg = self.denormalised(l)
        h = np.empty(mag.shape)
        v = np.empty(mag.shape)
        kernels().cartesian(mag, arg, h, v)
        return h, v


def extrema2(Wx: list, Wy: list, J: int, *, colocate_l1: bool = False,
             border: str = "periodic") -> Extrep:
    """The extrema of levels 1..J of a dyadic transform (``Wx``, ``Wy`` indexed 1..J, as ``dwt2d``
    leaves them, level scaling included). ``border`` is the border the transform ran with; it
    matters only to ``colocate_l1``."""
    if border not in BORDERS:
        raise ValueError(f"border must be one of {BORDERS}; got {border!r}")
    compute_point = _detect_kernels()["compute_point"]
    mask, mag, arg = [None], [None], [None]
    for l in range(1, J + 1):
        hor = np.ascontiguousarray(Wx[l], dtype=np.float64)
        ver = np.ascontiguousarray(Wy[l], dtype=np.float64)
        m, a = _polar(hor, ver)
        if l == 1 and colocate_l1:
            dm, da = _polar(*_colocated(hor, ver, border))
        else:
            dm, da = m, a
        nrow, ncol = hor.shape
        ext = np.zeros((nrow, ncol), dtype=np.bool_)
        compute_point(dm.ravel(), da.ravel(), nrow, ncol, l, ext.ravel())
        inv = 1.0 / fact(l)
        mask.append(ext)
        mag.append(np.where(ext, (m * _g1_factor(l)) * inv, 0.0))
        arg.append(np.where(ext, a, 0.0))
    return Extrep(J, mask, mag, arg)
