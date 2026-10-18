# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The orthogonal dyadic pyramids behind ``dwt2d``, ``inpaint`` and ``fill``: PyWavelets
(imported lazily) in ``periodization`` mode only, on a working image that is the field
mirror-extended once in each axis and rounded up to a multiple of ``2**levels``.

Periodization is the one mode whose synthesis is the analysis's adjoint, so the solver's
gradient steps are exact and the cascade's parent-child indexing is a fixed integer rule:
parent ``k`` over children ``2k`` and ``2k + 1``, shifted along the low-pass axis of the
horizontal and vertical bands by the bank's :func:`child_shift` (the scaling filter's delay,
measured once per bank). The mirror extension hides the field's edges from the periodic
transform. Reflect and symmetric modes are redundant and not adjoint; the Mallat-Zhong tool
keeps its own extension in ``mz_edges``.

Banks: PyWavelets' symlets, Daubechies and coiflets, plus ``bl3``, the Battle-Lemarie order-3
(cubic B-spline) bank of Pottier, Turiel and Garcon 2008, built here in the Fourier domain
(Mallat, A Wavelet Tour, section 7.2.3) and truncated at a coefficient magnitude of 1e-8.
Biorthogonal banks are not offered: their synthesis is not the analysis's adjoint.

The stationary transform (``swt2``, ``norm=True``) is a Parseval tight frame on the full grid
with ``iswt2`` as its exact adjoint; it is the measurement operator of a structure guide.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

__all__ = ["BANKS", "approximations", "atom_centroid", "battle_lemarie_filters", "border_cells",
           "child_shift", "cycle_shift", "cycle_unshift", "dc_gain", "detail_l1", "filter_length",
           "full_depth", "hyperbolic", "inverse", "max_level", "mirror", "only_band", "pyramid", "quarter",
           "registration", "require_pywt", "soft_threshold", "stationary", "stationary_inverse", "wavelet",
           "working_shape"]

BANKS = ("sym4", "sym5", "sym6", "sym7", "sym8", "db4", "db5", "db6", "db7", "db8",
         "coif4", "coif5", "coif6", "bl3")
BANDS = ("approximation", "horizontal", "vertical", "diagonal")
_NEEDS_PYWT = ("the DWT tools (dwt2d, inpaint, fill) need PyWavelets; install it with "
               "pip install PyWavelets")
_BL3 = None
_CHILD_SHIFT: dict = {}
_CENTROID: dict = {}
_BORDER: dict = {}


def require_pywt():
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(_NEEDS_PYWT) from exc
    return pywt


def battle_lemarie_filters(order: int = 3, tol: float = 1e-8, n: int = 4096,
                           kmax: int = 32) -> tuple:
    """``(dec_lo, dec_hi, rec_lo, rec_hi)`` of the orthonormalised B-spline bank of degree
    ``order``: ``H(w) = sqrt(2) cos(w/2)**(order+1) sqrt(S(w) / S(2w))`` with
    ``S(w) = sum_k |B(w + 2 pi k)|**2``, sampled on ``n`` points, inverted, truncated at
    ``tol``, padded to an even length and renormalised to a DC gain of ``sqrt(2)``."""
    if int(order) % 2 == 0:
        raise ValueError(f"battle_lemarie_filters: order {order} is even; the construction "
                         "holds for odd orders (3 is the paper's)")
    pywt = require_pywt()
    m = int(order) + 1
    omega = 2.0 * np.pi * np.arange(n) / n
    k = np.arange(-kmax, kmax + 1)

    def spectrum_sum(om):
        arg = (om[:, None] + 2.0 * np.pi * k[None, :]) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(arg) < 1e-12, 1.0, np.sin(arg) / arg)
        return (s ** (2 * m)).sum(1)

    H = np.sqrt(2.0) * np.cos(omega / 2.0) ** m * np.sqrt(spectrum_sum(omega)
                                                           / spectrum_sum(2.0 * omega))
    from dynamix.core.fft_policy import double

    h = np.roll(double().ifft(H).real, n // 2)
    keep = np.nonzero(np.abs(h) > tol)[0]
    h = h[keep[0]:keep[-1] + 1]
    if len(h) % 2:
        h = np.append(h, 0.0)
    h *= np.sqrt(2.0) / h.sum()
    return tuple(np.asarray(f, dtype=np.float64) for f in pywt.orthogonal_filter_bank(h))


def wavelet(name: str):
    """The PyWavelets ``Wavelet`` of a bank in :data:`BANKS` (or ``haar``, the test bank the
    exact-cascade oracles use; the devices never offer it); ``bl3`` is built once."""
    global _BL3
    pywt = require_pywt()
    if name == "haar":
        return pywt.Wavelet("haar")
    if name == "bl3":
        if _BL3 is None:
            _BL3 = pywt.Wavelet("bl3", filter_bank=battle_lemarie_filters())
            # Orthogonal up to the 1e-8 truncation (Parseval and adjointness hold to about
            # 1e-7); PyWavelets cannot tell for a custom bank and would warn on swt2 norm=True.
            _BL3.orthogonal = True
        return _BL3
    if name not in BANKS:
        raise ValueError(f"{name!r} is not one of the orthogonal banks {BANKS} (biorthogonal "
                         "banks such as bior2.2 have no adjoint synthesis)")
    return pywt.Wavelet(name)


def filter_length(name: str) -> int:
    return int(wavelet(name).dec_len)


def dc_gain(name: str) -> float:
    return float(np.sum(wavelet(name).dec_lo))


def working_shape(shape, levels: int) -> tuple:
    """``(2 ny, 2 nx)`` rounded up to a multiple of ``2**levels``."""
    m = 2 ** int(levels)
    return tuple(int(math.ceil(2 * int(n) / m)) * m for n in shape[:2])


def mirror(x, shape=None) -> np.ndarray:
    """``x`` padded symmetrically to half of ``shape`` (default ``(2 ny, 2 nx)``), then
    followed by its flips: one period of an even periodic image, continuous at every edge.
    A 1-D ``x`` is extended the same way along its one axis."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        m = 2 * x.size if shape is None else int(shape[0])
        x = np.pad(x, (0, m // 2 - x.size), mode="symmetric")
        return np.concatenate([x, x[::-1]])
    ny, nx = x.shape[:2]
    my, mx = (2 * ny, 2 * nx) if shape is None else (int(shape[0]), int(shape[1]))
    x = np.pad(x, ((0, my // 2 - ny), (0, mx // 2 - nx)), mode="symmetric")
    m = np.concatenate([x, x[::-1]], axis=0)
    return np.concatenate([m, m[:, ::-1]], axis=1)


def quarter(x, ny: int, nx: int) -> np.ndarray:
    return np.asarray(x)[:int(ny), :int(nx)]


def full_depth(shape) -> int:
    """The paper's depth: ``floor(log2(min side)) - 1``, at least 1."""
    return max(1, int(math.floor(math.log2(min(int(shape[0]), int(shape[1]))))) - 1)


def max_level(shape, name: str) -> int:
    return int(require_pywt().dwt_max_level(min(int(shape[0]), int(shape[1])),
                                           filter_length(name)))


def pyramid(x, name: str, level: int) -> list:
    """``wavedec2`` in periodization; PyWavelets' past-max-level warning is silenced because
    the cascade uses the paper's full depth on purpose."""
    pywt = require_pywt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pywt.wavedec2(np.asarray(x, dtype=np.float64), wavelet(name),
                             mode="periodization", level=int(level))


def inverse(coeffs, name: str) -> np.ndarray:
    return require_pywt().waverec2(coeffs, wavelet(name), mode="periodization")


def hyperbolic(x, name: str, level: int) -> dict:
    """The fully separable (hyperbolic) dyadic transform: the one-dimensional periodized DWT
    along x to depth ``level``, then along y to the same depth on every row band, one array per
    pair ``(j1, j2)`` of detail levels (1 the finest, the cell 2^j1 × 2^j2 px), the terms with
    the scaling function along either axis left out. In L¹ normalisation, 2^{−(j1+j2)/2} times
    PyWavelets' orthonormal coefficients, so the equal-level diagonal is the ordinary pyramid's
    diagonal band at its own L¹ scale (2^{−j} times ``wavedec2``'s), and on a product
    u(x) v(y) each pair is the outer product of the two one-dimensional details. The hyperbolic
    basis of DeVore, Konyagin and Temlyakov holds every anisotropy at once: the anisotropic
    structure functions of the directional formalism read it along the index sets Γ_j(α)."""
    pywt = require_pywt()
    w = wavelet(name)
    J = int(level)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = pywt.wavedec(np.asarray(x, dtype=np.float64), w, mode="periodization", level=J,
                            axis=1)
        for i1, band in enumerate(rows[1:]):              # rows[1] is the coarsest x detail
            j1 = J - i1
            cols = pywt.wavedec(band, w, mode="periodization", level=J, axis=0)
            for i2, c in enumerate(cols[1:]):
                j2 = J - i2
                out[(j1, j2)] = c * 2.0 ** (-(j1 + j2) / 2.0)
    return out


def approximations(x, name: str, levels: int) -> list:
    """The approximation at every level 1..``levels`` of the periodized pyramid."""
    pywt = require_pywt()
    w = wavelet(name)
    out, a = [], np.asarray(x, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(levels)):
            a, _details = pywt.dwt2(a, w, mode="periodization")
            out.append(a)
    return out


def stationary(x, name: str, level: int) -> list:
    return require_pywt().swt2(np.asarray(x, dtype=np.float64), wavelet(name),
                               level=int(level), norm=True, trim_approx=True)


def stationary_inverse(coeffs, name: str) -> np.ndarray:
    return require_pywt().iswt2(coeffs, wavelet(name), norm=True)


def child_shift(name: str) -> int:
    """The integer number of child cells by which a parent sits off the midpoint of its
    children ``(2k, 2k+1)`` along the low-pass axis of the horizontal and vertical bands
    (the scaling filter's delay under periodization), measured once per bank on the
    horizontal band at level 2 by the amplitude-weighted centroid of one coefficient's atom
    (on a 1024 px grid with the parent at the centre, so no atom wraps) and rounded half up;
    an exact half rounds up, so the paper's ``(2k, 2k+1)`` pairing wins a tie (``bl3 0``).
    The kernels pair children ``(2k + s, 2k + s + 1)`` by rolling the child band by ``-s``."""
    if name not in _CHILD_SHIFT:
        n = 1024
        zero = pyramid(np.zeros((n, n)), name, 3)
        cols = np.arange(n)[None, :]

        def centre_col(level, k):
            c = [np.zeros_like(zero[0])] + [tuple(np.zeros_like(d) for d in lev) for lev in zero[1:]]
            c[-level][0][k, k] = 1.0
            atom = np.abs(inverse(c, name))
            rel = cols - n // 2
            return float((atom * rel).sum() / atom.sum())

        off = (centre_col(3, n // 16) - 0.5 * (centre_col(2, n // 8) + centre_col(2, n // 8 + 1))) / 4.0
        _CHILD_SHIFT[name] = int(math.floor(off + 0.5 + 1e-3))
    return _CHILD_SHIFT[name]


def atom_centroid(name: str, level: int, atom: str) -> float:
    """The offset, in px, of the level-``level`` wavelet (``psi``) or scaling (``phi``) atom's
    amplitude-weighted centroid from its nominal cell centre ``2**level * (k + 1/2) - 1/2``,
    measured once per bank on a 1-D periodized grid long enough that no atom wraps."""
    key = (name, int(level), atom)
    if key not in _CENTROID:
        pywt = require_pywt()
        j = int(level)
        n = 1 << (j + 8)
        k0 = n // 2 ** j // 2
        coeffs = [np.zeros(n // 2 ** j)] + [np.zeros(n // 2 ** i) for i in range(j, 0, -1)]
        coeffs[0 if atom == "phi" else 1][k0] = 1.0
        a = np.abs(pywt.waverec(coeffs, wavelet(name), mode="periodization"))
        _CENTROID[key] = float((a * np.arange(n)).sum() / a.sum()) - (2 ** j * (k0 + 0.5) - 0.5)
    return _CENTROID[key]


def _cell_shift(offset_px: float, level: int) -> int:
    """The nearest whole cell; an exact half rounds up, the :func:`child_shift` tie rule."""
    return int(math.floor(offset_px / 2 ** int(level) + 0.5 + 1e-3))


def registration(name: str, levels: int) -> dict:
    """``{(j, band): (shift_rows, shift_cols, residual_rows_px, residual_cols_px)}`` for the
    three detail bands at j = 1..``levels``, and ``(j, "detail"): (shift, residual_px)`` for the
    one band of a 1-D transform: the whole cells by which a coefficient's atom sits off its own
    cell, so that rolling each band by its shift puts every level of every band on the field's
    dyadic grid (parent k over children 2k, 2k + 1 per axis), and what remains in px. A 2-D atom
    is the wavelet atom along the band's band-pass axis times the scaling atom along the other,
    so its centroid is the pair of 1-D ones; along the low-pass axis the shift settles to
    :func:`child_shift` by level 4 (level 3 for most banks) and can differ below."""
    out = {}
    for j in range(1, int(levels) + 1):
        psi, phi = atom_centroid(name, j, "psi"), atom_centroid(name, j, "phi")
        for band, (ry, rx) in (("horizontal", (psi, phi)), ("vertical", (phi, psi)),
                               ("diagonal", (psi, psi))):
            sy, sx = _cell_shift(ry, j), _cell_shift(rx, j)
            out[(j, band)] = (sy, sx, ry - sy * 2 ** j, rx - sx * 2 ** j)
        s = _cell_shift(psi, j)
        out[(j, "detail")] = (s, psi - s * 2 ** j)
    return out


def border_cells(name: str, tol: float = 1e-4) -> int:
    """Cells per edge whose coefficients the mirror extension can change: the cells, counted
    from an edge of the field, whose registered atom (wavelet or scaling, the 2-D atom being
    separable) carries at least ``tol`` of its L¹ mass beyond that edge, the largest count over
    both edges and levels 1..6, plus one for the leaders' 3 × 3 block. At 1e-4 the leaders of
    every bank agree with the periodic parent's to 1e-3 of the level mean on the cells kept.
    Measured once per bank: sym4 5, sym7 6, db4 5, bl3 13."""
    key = (name, float(tol))
    if key not in _BORDER:
        pywt = require_pywt()
        w = wavelet(name)
        e = 0
        for j in range(1, 7):
            n = 1 << (j + 8)
            k0 = n // 2 ** j // 2
            for atom in ("psi", "phi"):
                coeffs = [np.zeros(n // 2 ** j)] + [np.zeros(n // 2 ** i) for i in range(j, 0, -1)]
                coeffs[0 if atom == "phi" else 1][k0] = 1.0
                a = np.abs(pywt.waverec(coeffs, w, mode="periodization"))
                cum = np.cumsum(a / a.sum())
                c = k0 + _cell_shift(atom_centroid(name, j, atom), j)
                m = np.arange(64)
                left = int((cum[2 ** j * (c - m) - 1] >= tol).sum())
                right = int((1.0 - cum[2 ** j * (c + 1 + m) - 1] >= tol).sum())
                e = max(e, left, right)
        _BORDER[key] = e + 1
    return _BORDER[key]


def cycle_shift(x, rng, level: int) -> tuple:
    """``(rolled, (dy, dx))`` with ``|dy|, |dx| <= min(2**level, 32)`` drawn from ``rng``."""
    r = min(2 ** int(level), 32)
    dy, dx = (int(v) for v in rng.integers(-r, r + 1, size=2))
    return np.roll(x, (dy, dx), axis=(0, 1)), (dy, dx)


def cycle_unshift(x, shift) -> np.ndarray:
    return np.roll(x, (-int(shift[0]), -int(shift[1])), axis=(0, 1))


def soft_threshold(coeffs, t: float) -> list:
    """Every detail band shrunk by ``t``; the approximation untouched."""
    return [coeffs[0]] + [tuple(np.sign(d) * np.maximum(np.abs(d) - t, 0.0) for d in lev)
                          for lev in coeffs[1:]]


def detail_l1(coeffs) -> float:
    return float(sum(np.abs(d).sum() for lev in coeffs[1:] for d in lev))


def only_band(coeffs, level: int, band: str) -> list:
    """``coeffs`` with every band zeroed except ``band`` at ``level`` (1 = finest; the
    ``approximation`` at level ``j`` keeps the stored approximation and every detail coarser
    than ``j``, so at level ``J`` it is the stored approximation alone)."""
    J = len(coeffs) - 1
    out = [np.zeros_like(coeffs[0])] + [tuple(np.zeros_like(d) for d in lev)
                                        for lev in coeffs[1:]]
    if band == "approximation":
        out[0] = coeffs[0]
        for j in range(int(level) + 1, J + 1):
            out[J + 1 - j] = coeffs[J + 1 - j]
        return out
    i = BANDS.index(band) - 1
    lev = list(out[J + 1 - int(level)])
    lev[i] = coeffs[J + 1 - int(level)][i]
    out[J + 1 - int(level)] = tuple(lev)
    return out


def _flat(coeffs) -> np.ndarray:
    return np.concatenate([np.ravel(coeffs[0])] + [np.ravel(d) for lev in coeffs[1:]
                                                   for d in lev])
