# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
# Portions translated from xsmurf, Copyright (C) 1999 Centre de Recherche Paul Pascal,
# Bordeaux, France (N. Decoster, P. Kestener, S. Roux, A. Arneodo), GPL-2.0 -- see NOTICE.
"""Pure-numpy WTMM backend seam for the spatial multifractal workbench.

This module is the "universal binary" analysis layer for 2D (and, in later
slices, 1D and tensor) wavelet-transform-modulus-maxima pipelines. It is
imported from the reload path and must therefore stay importable on a
machine with ONLY pure-Python wheels: at module level it pulls in nothing
but :mod:`numpy` and the standard library. Everything that requires the
sibling analysis packages (``wtmm``, ``wtmm_ebsd``), the Apple-GPU ``mlx``
array library, ``scipy``, or ``numba`` is imported lazily inside the
function/method that actually needs it, so a plain ``import
dynamix.core.wtmm_backend`` never drags any of them into ``sys.modules``.

This slice implements the scale ladder, the 2D continuous wavelet
transform (CWT) stage, pure-numpy non-maximum-suppression (NMS) extrema
detection with within-scale line labeling, cross-scale chaining, 2D
partition-function delegation, the EBSD alpha-Jacobian tensor pipeline, and
the thin 1D pipeline delegation:
:func:`compute_scales2d`, :class:`PythonWTMMBackend` (with its ``cwt2d``,
``extrema2d``, ``single_maxima2d``, ``chains2d``, ``partition2d``,
``tensor2d``, ``cwt1d``, ``extrema1d``, ``chains1d`` and ``partition1d``
methods), the private maxima-line ordering kernels (:func:`_walk_lines_py`,
:func:`_smooth_dips_py`, njit-compiled on demand by :func:`_line_kernels`,
driven by :func:`_order_lines`), the shared single-scale NMS core
(:func:`_nms_extrema_scale`), the private numpy transcription
of ``wtmm_ebsd.cwt2d.cwt_2d_f32`` (:func:`_cwt2d_numpy`, its FFTs from the
FFT policy) used as the engine when ``mlx`` is unavailable, the numpy
transcription of ``wtmm_ebsd.cwt2d.tensor_svd_3x2``
(:func:`_tensor_svd_3x2_numpy`), the pure-Python ``xsmurf_wrapper``
stand-ins that ``tensor2d`` dependency-injects into
``wtmm_ebsd.twtmm.alpha_jacobian_twtmm`` (:class:`_XsmShim`,
:class:`_XImageShim`, :class:`_ExtImageShim` and the
:func:`_make_chain_adapter` chainer), the chain ``.npz`` export/import pair
(:func:`export_chains_npz`, :func:`load_chains_npz` -- schema v3, byte
-compatible with ``topo_wtmm.export_extrema``'s v2 via delegation to
:func:`dynamix.core.reflayers.load_topo_extrema`), and the staged pipeline
orchestrator :func:`run_wtmm2d` -- a pure, headless ``cwt2d`` ->
``extrema2d`` -> ``chains2d`` -> ``partition2d`` driver with per-stage
``.npz`` caching on the scalar path and a single run-level cache on the
tensor path (see its docstring for the cache-key chain).

NaN policy
----------
``PythonWTMMBackend.cwt2d`` fills any non-finite (NaN/inf) pixel with the
field's finite mean BEFORE the FFT (xsmurf/EBSD convention: NaN marks
masked-out pixels, and the wavelet transform itself has no concept of a
mask). The original NaN mask is intentionally NOT threaded through this
return value -- ``PythonWTMMBackend.extrema2d`` instead recomputes the
invalid mask directly from the caller-supplied original field (its
``field`` kwarg) and dilates it by the per-scale support (``ceil(scale)``
px) before dropping extrema, so ``cwt2d`` itself stays a pure, stateless
field -> {"mod", "arg"} map.

``PythonWTMMBackend.tensor2d`` applies the same fill to its (ny, nx, 3)
input before handing it to ``wtmm_ebsd.twtmm``, but PER COMPONENT (the
three components carry independent offsets, so one global mean would punch
an artificial step into all but one of them). Unlike ``cwt2d``, which falls
back to 0.0 for an all-NaN field, ``tensor2d`` raises ``ValueError`` when a
component has no finite pixel at all -- there is no signal to transform and
a silent zero-chain result would be undiagnosable.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

from dynamix.core import chaining

__all__ = [
    "DEFAULT_Q",
    "compute_scales2d",
    "PythonWTMMBackend",
    "get_backend",
    "resolve_tensor_svd_3x2",
    "export_chains_npz",
    "load_chains_npz",
    "run_wtmm2d",
    "run_wtmm2d_preview",
    "ComputeCancelled",
]


class ComputeCancelled(Exception):
    """Raised by :func:`run_wtmm2d` at a stage boundary when its ``cancel`` predicate is true.

    Progressive-compute design: a new ``a_min`` edit kills the in-flight
    full-stack compute instead of letting it finish. Python threads can't be hard-killed, so the
    staged loop checks ``cancel()`` cooperatively BEFORE each stage's compute+save -- raising this
    (never returning a half-built result) so nothing partial is ever cached. Callers (the worker)
    catch it and simply drop the abandoned run.
    """

# Curated GUI default q-grid (coarser than wtmm_ebsd's DEFAULT_Q_LIST 0.1 grid,
# which is tuned for offline batch runs rather than an interactive workbench).
DEFAULT_Q = np.arange(-3.0, 6.0 + 1e-9, 0.5)


def compute_scales2d(n_oct: int, n_voice: int, a_min: float = 1.0) -> np.ndarray:
    """Logarithmically spaced 2D CWT scales (xsmurf convention).

    Transcribes ``wtmm_ebsd.cwt2d.compute_scales`` exactly (norm = 6.0/0.86;
    ``scales[o, v] = a_min * 2 ** (o + v / n_voice) * norm`` for octave ``o``
    in ``range(n_oct)`` and voice ``v`` in ``range(n_voice)``, flattened with
    ``o`` the slow index and ``v`` the fast index) WITHOUT depending on
    ``wtmm_ebsd`` itself -- ``wtmm_ebsd.cwt2d`` hard-imports ``mlx`` at module
    level, and this function must keep working on the pure-Python-wheel path.

    Returns
    -------
    np.ndarray
        float64, shape ``(n_oct * n_voice,)``.
    """
    norm = 6.0 / 0.86
    octave, voice = np.meshgrid(np.arange(n_oct), np.arange(n_voice), indexing="ij")
    exponents = octave + voice / n_voice
    return (a_min * (2.0 ** exponents) * norm).reshape(-1).astype(np.float64)


#: The derivative stacks the transform computes, in the vendored order.
_DERIVATIVE_NAMES = ("dx", "dy", "dxx", "dxy", "dyy", "dxxx", "dxxy", "dxyy", "dyyy")


def _build_wavelet_filters_numpy(kx: np.ndarray, ky: np.ndarray, scale: float,
                                  wavelet: str = "gaussian", params: "dict | None" = None,
                                  dtype=np.float32) -> dict:
    """Fourier-domain derivative filters at one scale (numpy transcription).

    Exact transcription of ``wtmm_ebsd.cwt2d._build_wavelet_filters_f32``,
    swapping ``mlx.core`` array ops for numpy/complex64 ones. Every
    derivative filter is ``(i*sx)**p * (i*sy)**q * smoother`` for derivative
    order ``(p, q)`` in x/y, with ``smoother = gauss`` for the 'gaussian'
    wavelet or ``smoother = |k|^2 * gauss`` (the "mexican" smoother) for
    'mexican' -- transcribed as explicit real/imaginary parts (not via
    complex exponentiation) to match the reference bit-for-bit in structure.

    Any other name is one of :data:`dynamix.core.smoothers.NAMES`: its θ̂ is evaluated on
    the grid and the nine filters follow the same rule (:func:`smoothers.derivative_filters`);
    ``params`` are that smoother's parameters (``alpha``, ``q``). ``dtype`` is the real
    precision: float32 (complex64 filters, the reference's) or float64 at 64-bit FFTs.
    """
    sx = (kx * scale).astype(dtype)
    sy = (ky * scale).astype(dtype)
    gauss = np.exp(-(sx * sx + sy * sy)).astype(dtype)
    zero = np.zeros_like(gauss)
    complex_dtype = np.result_type(dtype, np.complex64)

    def to_complex(r, i):
        return (r + 1j * i).astype(complex_dtype)

    if wavelet == "mexican":
        k2 = sx * sx + sy * sy
        return {
            "dx":   to_complex(zero,                    sx * k2 * gauss),
            "dy":   to_complex(zero,                    sy * k2 * gauss),
            "dxx":  to_complex(-sx * sx * k2 * gauss,    zero),
            "dxy":  to_complex(-sy * sx * k2 * gauss,    zero),
            "dyy":  to_complex(-sy * sy * k2 * gauss,    zero),
            "dxxx": to_complex(zero, -sx * sx * sx * k2 * gauss),
            "dxxy": to_complex(zero, -sx * sx * sy * k2 * gauss),
            "dxyy": to_complex(zero, -sx * sy * sy * k2 * gauss),
            "dyyy": to_complex(zero, -sy * sy * sy * k2 * gauss),
        }
    if wavelet != "gaussian":
        from dynamix.core.smoothers import NAMES, derivative_filters, smoother
        if wavelet not in NAMES:
            raise ValueError(f"unknown wavelet {wavelet!r}; expected one of {NAMES}")
        sm = smoother(wavelet, params)
        return derivative_filters(sx, sy, sm.theta_hat(sx, sy), _DERIVATIVE_NAMES, dtype=dtype)
    return {
        "dx":   to_complex(zero,                    sx * gauss),
        "dy":   to_complex(zero,                    sy * gauss),
        "dxx":  to_complex(-sx * sx * gauss,         zero),
        "dxy":  to_complex(-sy * sx * gauss,         zero),
        "dyy":  to_complex(-sy * sy * gauss,         zero),
        "dxxx": to_complex(zero, -sx * sx * sx * gauss),
        "dxxy": to_complex(zero, -sx * sx * sy * gauss),
        "dxyy": to_complex(zero, -sx * sy * sy * gauss),
        "dyyy": to_complex(zero, -sy * sy * sy * gauss),
    }


#: What the FFT torus holds beyond the real data, by edge mode (spec 25): the numpy pad mode, or None
#: for ``periodic``, where the raster itself is the torus and nothing is padded. Every convolution here
#: is circular; the padding decides what the wrap-around runs across.
PAD_MODES = {"reflect": "reflect", "edge_pad": "edge", "zero_pad": "constant", "periodic": None}


def _padded(image, pad: int, pad_mode: str):
    """``(padded image, the pad actually applied)`` for ``pad_mode`` (:data:`PAD_MODES`)."""
    if pad_mode not in PAD_MODES:
        raise ValueError(f"unknown pad_mode {pad_mode!r}; expected one of {tuple(PAD_MODES)}")
    np_mode = PAD_MODES[pad_mode]
    image = np.asarray(image)
    if np_mode is None or int(pad) <= 0:
        return image, 0
    return np.pad(image, int(pad), mode=np_mode), int(pad)


def _cwt2d_numpy(image: np.ndarray, scales, *, pad: int = 32, derivs: str = "first",
                  wavelet: str = "gaussian", verbose: bool = True, fft=None,
                  params: "dict | None" = None, pad_mode: str = "reflect") -> dict:
    """Numpy/FFT transcription of ``wtmm_ebsd.cwt2d.cwt_2d_f32``.

    Signature-compatible with the ``mlx`` original so either engine can be
    injected into ``wtmm_ebsd.twtmm.alpha_jacobian_twtmm`` unchanged (the
    later tensor slice). Same reflect padding, same ``np.fft.fftfreq``
    wavenumber grids, same per-scale filter construction (see
    :func:`_build_wavelet_filters_numpy`), same modulus/argument
    computation, same crop back to the input shape, float32 outputs -- float64 throughout when
    the FFT backend runs at 64 bits.

    Parameters mirror ``cwt_2d_f32`` exactly; see that docstring. ``params`` are the
    smoother's parameters for the registry's names (:func:`_build_wavelet_filters_numpy`).
    """
    # ``fft``: an fft_policy backend (FFTW3 at 32/64-bit, or mlx); None takes the policy's
    # active one. Its precision sets the real dtype of the whole transform.
    if fft is None:
        from dynamix.core import fft_policy
        fft = fft_policy.active()
    _F = fft
    real = np.float64 if getattr(fft, "precision", 32) == 64 else np.float32
    image = np.asarray(image)
    ny, nx = image.shape
    padded, pad = _padded(image, pad, pad_mode)
    padded = padded.astype(real)
    ny_p, nx_p = padded.shape
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    kx_np = np.fft.fftfreq(nx_p).astype(real)
    ky_np = np.fft.fftfreq(ny_p).astype(real)
    kx, ky = np.meshgrid(kx_np, ky_np)
    image_fft = _F.fft2(padded.astype(np.result_type(real, np.complex64)))

    if derivs == "first":
        names = ["dx", "dy"]
    else:
        names = ["dx", "dy", "dxx", "dxy", "dyy", "dxxx", "dxxy", "dxyy", "dyyy"]

    n_scales = len(scales)
    result = {n: np.zeros((n_scales, ny, nx), dtype=real) for n in names}
    result["mod"] = np.zeros((n_scales, ny, nx), dtype=real)
    result["arg"] = np.zeros((n_scales, ny, nx), dtype=real)
    for i, scale in enumerate(scales):
        filters = _build_wavelet_filters_numpy(kx, ky, scale, wavelet=wavelet, params=params,
                                               dtype=real)
        for name in names:
            conv = _F.ifft2(image_fft * filters[name])
            result[name][i] = conv.real.astype(real)[crop]
        dx_s, dy_s = result["dx"][i], result["dy"][i]
        result["mod"][i] = np.sqrt(dx_s * dx_s + dy_s * dy_s)
        result["arg"][i] = np.arctan2(dy_s, dx_s).astype(real)
        if verbose and (i % max(1, n_scales // 4) == 0 or i == n_scales - 1):
            print(f"  Scale {i}/{n_scales - 1}: a={scale:.1f}")
    return result


def _cwt2d_mlx(image: np.ndarray, scales, *, pad: int = 32, derivs: str = "first",
               wavelet: str = "gaussian", verbose: bool = True,
               params: "dict | None" = None, pad_mode: str = "reflect") -> dict:
    """The vendored ``cwt_2d_f32`` loop on the device with the registry's filters.

    Same reflect padding, wavenumber grids, one ``fft2`` of the padded image, per scale
    one ``ifft2`` per derivative, same modulus/argument and crop, float32 outputs. The
    filters come from :func:`_build_wavelet_filters_numpy` on the host (θ̂ of the
    fractional B-spline and the q-Gaussian are host formulas) and are moved to the device
    per scale; the products and inverse transforms run there. Every ``mx.eval`` precedes a
    host read so an mlx error raises instead of aborting inside the buffer read.
    """
    import mlx.core as mx

    image = np.asarray(image)
    ny, nx = image.shape
    padded, pad = _padded(image, pad, pad_mode)
    padded = padded.astype(np.float32)
    ny_p, nx_p = padded.shape
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    kx_np = np.fft.fftfreq(nx_p).astype(np.float32)
    ky_np = np.fft.fftfreq(ny_p).astype(np.float32)
    kx, ky = np.meshgrid(kx_np, ky_np)
    image_fft = mx.fft.fft2(mx.array(padded))

    names = ["dx", "dy"] if derivs == "first" else list(_DERIVATIVE_NAMES)
    n_scales = len(scales)
    result = {n: np.zeros((n_scales, ny, nx), dtype=np.float32) for n in names}
    result["mod"] = np.zeros((n_scales, ny, nx), dtype=np.float32)
    result["arg"] = np.zeros((n_scales, ny, nx), dtype=np.float32)
    for i, scale in enumerate(scales):
        filters = _build_wavelet_filters_numpy(kx, ky, scale, wavelet=wavelet, params=params)
        for name in names:
            conv = mx.fft.ifft2(image_fft * mx.array(filters[name]))
            mx.eval(conv)
            result[name][i] = np.array(conv.real)[crop]
        dx_s, dy_s = result["dx"][i], result["dy"][i]
        result["mod"][i] = np.sqrt(dx_s * dx_s + dy_s * dy_s)
        result["arg"][i] = np.arctan2(dy_s, dx_s).astype(np.float32)
        if verbose and (i % max(1, n_scales // 4) == 0 or i == n_scales - 1):
            print(f"  Scale {i}/{n_scales - 1}: a={scale:.1f}")
    mx.eval()
    return result


def _tensor_svd_3x2_numpy(gx1, gy1, gx2, gy2, gx3, gy3):
    """Closed-form JᵀJ eigendecomposition SVD of a 3x2 Jacobian, per pixel.

    Exact transcription of ``wtmm_ebsd.cwt2d.tensor_svd_3x2`` (that function
    is already pure numpy in the reference, so this is a direct copy).

    All inputs are (ny, nx) float arrays. Returns (sigma_max, sigma_min,
    arg) where arg is the orientation of the leading right singular vector
    (rad).
    """
    a = gx1 * gx1 + gx2 * gx2 + gx3 * gx3
    b = gx1 * gy1 + gx2 * gy2 + gx3 * gy3
    d = gy1 * gy1 + gy2 * gy2 + gy3 * gy3
    trace = a + d
    det = a * d - b * b
    disc = np.sqrt(np.maximum((trace / np.float32(2)) ** 2 - det,
                               np.float32(0)))
    lam1 = trace / np.float32(2) + disc
    lam2 = np.maximum(trace / np.float32(2) - disc, np.float32(0))
    sigma_max = np.sqrt(np.maximum(lam1, np.float32(0))).astype(np.float32)
    sigma_min = np.sqrt(lam2).astype(np.float32)
    vx = lam1 - d
    vy = b
    vnorm = np.sqrt(vx ** 2 + vy ** 2) + np.float32(1e-30)
    arg = np.arctan2(vy / vnorm, vx / vnorm).astype(np.float32)
    return sigma_max, sigma_min, arg


def resolve_tensor_svd_3x2():
    """Lazily resolve the 3x2 tensor-SVD callable: the app's own transcription
    :func:`_tensor_svd_3x2_numpy`; the vendored ``wtmm_ebsd.cwt2d.tensor_svd_3x2`` behind
    :data:`USE_VENDORED` until the last port."""
    if USE_VENDORED:
        try:
            from dynamix._vendor.wtmm_ebsd.cwt2d import tensor_svd_3x2
            return tensor_svd_3x2
        except ImportError:
            pass
    return _tensor_svd_3x2_numpy


# ---------------------------------------------------------------------------
# Maxima-line ordering kernels (xsmurf `ssm` / `smooth_chains` support)
#
# These two functions are written in the numba-friendly subset of Python
# (scalar loops, preallocated output arrays, no fancy indexing) and are
# njit(cache=True)-compiled ON FIRST USE by `_line_kernels()` -- numba stays a
# lazy import, and if it is missing at all the plain-Python versions below are
# used unchanged (correct, just slower).
# ---------------------------------------------------------------------------
_LINE_KERNELS: dict = {}

# 8-neighbourhood offsets, 4-connected first so the walk's "nearest unvisited
# neighbour" tie-break naturally prefers a step of length 1 over sqrt(2).
_OFF_X = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int64)
_OFF_Y = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int64)

# `single_max.c::init_pos_incr` -- the CLOCKWISE neighbour order the
# `_is_single_max_` run count sweeps (entry 8 repeats entry 0, which the
# circular `np.roll` below reproduces):
#       0   1   2
#       7   .   3
#       6   5   4
_SM_OFF_X = np.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=np.int64)
_SM_OFF_Y = np.array([-1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int64)


def _walk_lines_py(x, y, gidx, members, starts, grid, nx, ny,
                   off_x, off_y, stamp, order_out, seg_out):
    """Order every labelled maxima line along itself; return the segment count.

    ``members``/``starts`` are a CSR grouping of the labelled points by line
    (``members[starts[g]:starts[g + 1]]`` are the point indices of line ``g``),
    ``gidx[i]`` is point ``i``'s line index (``-1`` for singletons) and
    ``grid[y * nx + x]`` maps a pixel back to its point index (``-1`` if empty).

    Each line is traversed by a nearest-neighbour walk started from an
    *endpoint* (a point with exactly one not-yet-visited within-line
    8-neighbour); a closed loop has no such point and starts at its first
    member instead. A line that branches leaves points unvisited when the walk
    dead-ends, so the walk simply restarts on the remainder -- every restart
    opens a NEW segment, which keeps the downstream "along the line"
    comparisons from ever bridging two topologically separate branches.
    """
    n_lines = starts.shape[0] - 1
    seg = 0
    for g in range(n_lines):
        lo = starts[g]
        hi = starts[g + 1]
        pos = lo
        while pos < hi:
            # --- pick this segment's starting point ---------------------
            start = -1
            fallback = -1
            for k in range(lo, hi):
                i = members[k]
                if stamp[i] == g:
                    continue
                if fallback < 0:
                    fallback = i
                cnt = 0
                for t in range(8):
                    xx = x[i] + off_x[t]
                    yy = y[i] + off_y[t]
                    if xx < 0 or xx >= nx or yy < 0 or yy >= ny:
                        continue
                    j = grid[yy * nx + xx]
                    if j >= 0 and gidx[j] == g and stamp[j] != g:
                        cnt += 1
                if cnt == 1:
                    start = i
                    break
            if start < 0:
                start = fallback
            # --- walk ----------------------------------------------------
            cur = start
            while cur >= 0:
                order_out[pos] = cur
                seg_out[pos] = seg
                stamp[cur] = g
                pos += 1
                best = -1
                best_d = 9
                for t in range(8):
                    xx = x[cur] + off_x[t]
                    yy = y[cur] + off_y[t]
                    if xx < 0 or xx >= nx or yy < 0 or yy >= ny:
                        continue
                    j = grid[yy * nx + xx]
                    if j >= 0 and gidx[j] == g and stamp[j] != g:
                        d = off_x[t] * off_x[t] + off_y[t] * off_y[t]
                        if d < best_d:
                            best_d = d
                            best = j
                cur = best
            seg += 1
    return seg


def _smooth_dips_py(mod, seg):
    """xsmurf ``ssm -smooth``: single forward pass of 3-point dip filling.

    Transcribes ``xsmurf_wrapper.core.smooth_chains`` exactly: sliding a
    3-point window along the ordered line, a middle point that is a strict dip
    between two higher neighbours (``m1 > m2 and m3 > m2``) is replaced by
    ``(m1 + m3) / 2``. The pass is sequential and in place, so a replaced value
    is the ``m1`` seen by the next window -- exactly as in the reference (which
    advances ``e1 = e2`` AFTER writing ``e2.mod``).
    """
    n = mod.shape[0]
    for i in range(1, n - 1):
        if seg[i - 1] != seg[i] or seg[i] != seg[i + 1]:
            continue
        m1 = mod[i - 1]
        m2 = mod[i]
        m3 = mod[i + 1]
        if m1 > m2 and m3 > m2:
            mod[i] = 0.5 * (m1 + m3)


def _line_kernels() -> dict:
    """Lazily njit(cache=True)-compile the maxima-line kernels."""
    if not _LINE_KERNELS:
        try:
            from numba import njit
        except Exception:                                   # pragma: no cover
            _LINE_KERNELS["walk"] = _walk_lines_py
            _LINE_KERNELS["smooth"] = _smooth_dips_py
        else:
            _LINE_KERNELS["walk"] = njit(cache=True)(_walk_lines_py)
            _LINE_KERNELS["smooth"] = njit(cache=True)(_smooth_dips_py)
    return _LINE_KERNELS


def _order_lines(x, y, line_id, grid, nx, ny):
    """Order every labelled maxima line along itself (``_walk_lines_py`` driver).

    Builds the CSR grouping of the labelled points (``line_id >= 0``; singletons
    are excluded, they are one-point "lines" with nothing to order) and runs the
    njit-compiled nearest-neighbour walk.

    Returns
    -------
    (order, seg, starts)
        ``order`` : (m,) int64 point indices, grouped by line and ordered along
        each line. ``seg`` : (m,) int64 segment id, incremented at every walk
        restart, so a branching line yields several contiguous segments.
        ``starts`` : (n_lines + 1,) int64 CSR offsets into ``order``, one slice
        per ``line_id`` component. All three are empty when nothing is labelled.
    """
    kern = _line_kernels()
    walk = kern["walk"]
    n = x.size

    lab_idx = np.nonzero(line_id >= 0)[0]
    empty = np.empty(0, dtype=np.int64)
    if not lab_idx.size:
        return empty, empty, np.zeros(1, dtype=np.int64)

    gidx = np.full(n, -1, dtype=np.int64)
    _, inv = np.unique(line_id[lab_idx], return_inverse=True)
    inv = inv.astype(np.int64).ravel()
    gidx[lab_idx] = inv
    srt = np.argsort(inv, kind="stable")
    members = lab_idx[srt]
    counts = np.bincount(inv)
    starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts)]).astype(np.int64)

    order = np.empty(members.size, dtype=np.int64)
    seg = np.empty(members.size, dtype=np.int64)
    stamp = np.full(n, -1, dtype=np.int64)
    walk(x, y, gidx, members, starts, grid, nx, ny,
         _OFF_X, _OFF_Y, stamp, order, seg)
    return order, seg, starts


#: The nms detector's rules for a pixel tied with a neighbour along the gradient. ``asymmetric`` is xsmurf's NMS path (``wtmm2d`` under ``useNMaxSup 1``:
#: ``Remove_Gradient_NonMaxima_Slice_2D``, edge/extrema_core.c): strict toward the gradient, >=
#: against it, so one pixel of a tied pair survives, and the nearest pixel within ~1.8 deg of an
#: axis. ``strict`` keeps neither, ``keep_both`` (>= both sides, interpolated) both.
TIE_RULES = ("asymmetric", "strict", "keep_both")
#: xsmurf's ``_EPSILON_DERIVATIVE_``: past it a normalised gradient component reads the nearest pixel.
_NEAREST_AXIS = 0.9995


def _nms_extrema_scale(mod, arg, *, thresh: float = 1e-3, invalid=None,
                       radius: int = 0, grid=None, real_block=None, real_origin=(0, 0),
                       tie_rule: str = "asymmetric") -> dict:
    """Single-scale NMS modulus maxima -- the core shared by both entry points.

    :meth:`PythonWTMMBackend.extrema2d` calls this once per scale, and
    :meth:`_XsmShim.wtmm2d` calls it once for the single scale it is handed, so
    the xsmurf-shim path and the public 2D pipeline detect extrema with the
    exact same code (see the ``extrema2d`` docstring for the algorithm).

    Parameters
    ----------
    mod, arg : (ny, nx) array
        Wavelet-gradient modulus and angle (rad) at ONE scale.
    thresh : float
        Fraction of ``mod.max()`` below which extrema are dropped.
    invalid : (ny, nx) bool array or None
        Originally-NaN mask; when given (and non-empty) it is dilated by
        ``radius`` pixels (8-connectivity) and the covered extrema are dropped.
    radius : int
        Dilation radius in pixels, normally ``ceil(scale)``.
    grid : (2, ny, nx) float array or None
        Precomputed ``np.mgrid`` row/column coordinates; built on demand when
        ``None`` (``extrema2d`` builds it once and reuses it across scales).
    tie_rule : str
        One of :data:`TIE_RULES`.

    Returns
    -------
    dict
        ``{"x", "y" (int64), "mod", "arg" (float64), "line_id" (int64)}``.
    """
    from scipy.ndimage import binary_dilation, label, map_coordinates

    if tie_rule not in TIE_RULES:
        raise ValueError(f"unknown tie_rule {tie_rule!r}; expected one of {TIE_RULES}")
    m = np.asarray(mod, dtype=np.float64)
    a = np.asarray(arg, dtype=np.float64)
    ny, nx = m.shape
    if grid is None:
        grid = np.mgrid[0:ny, 0:nx].astype(np.float64)
    yy, xx = grid

    cx = np.cos(a)
    sy = np.sin(a)
    fore = map_coordinates(m, [yy + sy, xx + cx], order=1, mode="nearest")
    back = map_coordinates(m, [yy - sy, xx - cx], order=1, mode="nearest")
    if tie_rule != "keep_both":
        near = (np.abs(cx) > _NEAREST_AXIS) | (np.abs(sy) > _NEAREST_AXIS)
        if near.any():
            # the C's (int)(x + g + 0.5): truncation toward zero; edges read the nearest pixel
            def nearest(dy, dx):
                r = np.clip(np.trunc(yy + dy + 0.5).astype(np.int64), 0, ny - 1)
                c = np.clip(np.trunc(xx + dx + 0.5).astype(np.int64), 0, nx - 1)
                return m[r, c]

            fore = np.where(near, nearest(sy, cx), fore)
            back = np.where(near, nearest(-sy, -cx), back)

    if real_block is None:
        mmax = m.max()
    else:
        # A Periodic extension output (spec 25 §3.1): the real data are one block, at the origin
        # or at the centred layout's offset, the rest a filled gap -- its maxima are not the
        # data's, and its moduli set no threshold.
        rows, cols = (int(v) for v in real_block)
        r0, c0 = (int(v) for v in (real_origin or (0, 0)))
        mmax = m[r0:r0 + rows, c0:c0 + cols].max()
    if tie_rule == "keep_both":
        peak = (m >= fore) & (m >= back)
    elif tie_rule == "asymmetric":
        peak = (m > fore) & (m >= back)
    else:
        peak = (m > fore) & (m > back)
    mask = peak & (m >= thresh * mmax)
    if real_block is not None:
        mask[:r0, :] = False
        mask[r0 + rows:, :] = False
        mask[:, :c0] = False
        mask[:, c0 + cols:] = False

    if invalid is not None and invalid.any():
        # 8-connectivity (full 3x3) structuring element so each iteration grows
        # the invalid region by one pixel in Chebyshev distance (a square halo),
        # matching the "support touches an originally-NaN pixel" contract
        # language for a square dilation footprint rather than the smaller
        # diamond (L1) footprint scipy's default cross gives.
        dilated = (binary_dilation(invalid, structure=np.ones((3, 3)), iterations=radius)
                   if radius > 0 else invalid)
        mask &= ~dilated

    labels, n_labels = label(mask, structure=np.ones((3, 3)))
    y_idx, x_idx = np.nonzero(mask)
    line_id = labels[y_idx, x_idx].astype(np.int64)
    if n_labels > 0:
        counts = np.bincount(labels.reshape(-1), minlength=n_labels + 1)
        isolated = counts[line_id] <= 1
        line_id = line_id - 1              # labels are 1..n_labels -> 0..n_labels-1
        line_id[isolated] = -1

    return {
        "x": x_idx.astype(np.int64),
        "y": y_idx.astype(np.int64),
        "mod": m[y_idx, x_idx].astype(np.float64),
        "arg": a[y_idx, x_idx].astype(np.float64),
        "line_id": line_id,
    }


# ---------------------------------------------------------------------------
# The pure-Python `xsmurf_wrapper` stand-in injected into wtmm_ebsd.twtmm.
#
# `wtmm_ebsd.twtmm.alpha_jacobian_twtmm` owns the CUSTOM tensor math (per-scale
# kappa_ij from the 3-component CWT, Pantleon Nye-alpha assembly, the 3x2 and
# 6x2 Jacobian SVDs, the mode modulus/arg fields) and takes its
# xsmurf-dependent stages as INJECTED callables. These shims supply exactly the
# surface it touches, so the custom math is reused rather than transcribed and
# no `xsmurf` module is ever imported.
# ---------------------------------------------------------------------------
class _XImageShim:
    """Stand-in for ``xsmurf_wrapper.XImage`` (the ``from_numpy`` path only).

    ``alpha_jacobian_twtmm`` builds one of these per rotated derivative field
    and immediately hands it to :meth:`_XsmShim.wtmm2d`, so the only state it
    needs is the array itself plus the image dimensions.
    """

    __slots__ = ("data", "lx", "ly")

    def __init__(self, arr):
        self.data = np.ascontiguousarray(arr, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError(f"XImage expects a 2-D array; got {self.data.shape}")
        self.ly, self.lx = (int(n) for n in self.data.shape)

    @classmethod
    def from_numpy(cls, arr) -> "_XImageShim":
        """``xsmurf_wrapper.XImage.from_numpy`` equivalent."""
        return cls(arr)


class _ExtImageShim:
    """Stand-in for ``xsmurf_wrapper.XExtImage`` (one scale's NMS extrema).

    Exposes the consumed ``XExtImage`` interface: :attr:`extr_nb`, :attr:`lx`,
    :meth:`get_extrema_arrays` and :meth:`get_lines`, plus the private
    :meth:`to_extrema_dict` the chain adapter uses to hand the points back to
    :meth:`PythonWTMMBackend.chains2d` in the native ``extrema2d`` schema (the
    within-scale ``line_id`` labels ride along, so nothing is re-labelled).
    """

    __slots__ = ("_extrema", "lx", "ly", "scale", "extr_nb")

    def __init__(self, extrema: dict, lx: int, ly: int, scale: float):
        self._extrema = extrema
        self.lx = int(lx)
        self.ly = int(ly)
        self.scale = float(scale)
        self.extr_nb = int(extrema["x"].size)

    def to_extrema_dict(self) -> dict:
        """The points in :meth:`PythonWTMMBackend.extrema2d`'s dict schema."""
        return dict(self._extrema)

    def get_extrema_arrays(self) -> dict:
        """``{"x", "y", "pos", "mod", "arg"}`` 1-D arrays; ``pos = y * lx + x``."""
        e = self._extrema
        return {
            "x": e["x"].copy(),
            "y": e["y"].copy(),
            "pos": (e["y"] * self.lx + e["x"]).astype(np.int64),
            "mod": e["mod"].copy(),
            "arg": e["arg"].copy(),
        }

    def get_lines(self) -> list:
        """Within-scale maxima lines as ``[{"extrema_pos": (m,) int64}, ...]``.

        Each entry is ONE ordered maxima-line segment: the points of a
        ``line_id`` component walked along themselves from an endpoint
        (:func:`_order_lines`), so consecutive positions are always
        8-neighbours and the list can be drawn as a polyline directly. A
        BRANCHING component yields one entry per branch, mirroring xsmurf's
        ordered ``line->ext_lst`` (which is likewise a simple walk); singleton
        components (``line_id == -1``) have no line and are excluded.
        """
        e = self._extrema
        x, y = e["x"], e["y"]
        if x.size == 0:
            return []
        grid = np.full(self.ly * self.lx, -1, dtype=np.int64)
        grid[y * self.lx + x] = np.arange(x.size, dtype=np.int64)
        order, seg, _starts = _order_lines(x, y, e["line_id"], grid, self.lx, self.ly)
        if not order.size:
            return []
        pos = (y[order] * self.lx + x[order]).astype(np.int64)
        bounds = np.nonzero(np.diff(seg))[0] + 1
        return [{"extrema_pos": p} for p in np.split(pos, bounds)]


class _XsmShim:
    """The pure-Python ``xsm`` module stand-in injected into ``wtmm_ebsd.twtmm``.

    ``alpha_jacobian_twtmm`` touches EXACTLY three names on its injected ``xsm``
    (verified against ``wtmm_ebsd/twtmm.py``) and this class provides those and
    nothing more::

        xsm.XImage.from_numpy(arr)                       -> _XImageShim
        ext, _, _ = xsm.wtmm2d(xim_dx, xim_dy, scale, thresh=...)
        xsm.compute_holder_exponents(chains)             -> [{"h": ...}, ...]

    Every member is a ``staticmethod``/class attribute, so the CLASS ITSELF is
    what gets injected (no instantiation) -- matching how ``twtmm`` uses the
    real ``xsmurf_wrapper`` module object.
    """

    XImage = _XImageShim

    @staticmethod
    def wtmm2d(xim_dx, xim_dy, scale, thresh: float = 1e-3):
        """Single-scale gradient NMS -- ``(ext_image, None, None)``.

        ``twtmm`` unpacks the result as ``ext, _, _`` and only ever uses the
        first element, so the modulus/argument images the real ``wtmm2d``
        returns as elements 2 and 3 are not materialised here.

        The modulus/argument are recovered from the two rotated derivative
        images (``mod = hypot(dx, dy)``, ``arg = arctan2(dy, dx)``, which
        round-trips ``twtmm``'s ``dx_r = mod*cos(arg)`` / ``dy_r =
        mod*sin(arg)`` exactly) and fed to :func:`_nms_extrema_scale` -- the
        SAME code path :meth:`PythonWTMMBackend.extrema2d` runs per scale.
        """
        dx = np.asarray(xim_dx.data, dtype=np.float64)
        dy = np.asarray(xim_dy.data, dtype=np.float64)
        mod = np.hypot(dx, dy)
        arg = np.arctan2(dy, dx)
        extrema = _nms_extrema_scale(mod, arg, thresh=thresh)
        return _ExtImageShim(extrema, lx=xim_dx.lx, ly=xim_dx.ly, scale=scale), None, None

    @staticmethod
    def compute_holder_exponents(chains) -> list:
        """``[{"h": h} for ch in chains]`` with ``h`` the OLS slope of ``log2_mod`` against
        ``log2_scales`` (``dynamix.core.chain_stats._chain_ols_holder``, the app's own
        reimplementation of the vendored ``chain_ols_holder``; the vendored original stays behind
        :data:`USE_VENDORED` until the last port).

        One entry per chain, positionally aligned with ``chains`` (the real
        wrapper additionally drops chains shorter than 2 scales, which would
        break that alignment; the slope is NaN for those instead).
        """
        if USE_VENDORED:
            chain_filters = _load_wtmm_ebsd_module("chain_filters")
            return [{"h": chain_filters.chain_ols_holder(ch)} for ch in chains]
        from dynamix.core.chain_stats import _chain_ols_holder

        return [{"h": _chain_ols_holder(ch)} for ch in chains]


def _load_wtmm_ebsd_module(name: str):
    """Lazily import ``wtmm_ebsd.<name>``, tolerating a missing ``mlx``.

    ``wtmm_ebsd/__init__.py`` eagerly re-exports every submodule, including
    ``cwt2d``, which hard-imports ``mlx.core``. On the universal-binary path
    (pure-Python wheels, no ``mlx``) the ordinary package import therefore
    fails even though the module we want -- ``twtmm`` or ``chain_filters`` --
    needs nothing but numpy. So the ``ImportError`` fallback loads that single
    source file directly from the located package directory, bypassing the
    package ``__init__``. The normal import is always tried first, so whenever
    ``mlx`` IS present the genuine ``wtmm_ebsd.<name>`` module object (and its
    identity in ``sys.modules``) is what callers get.
    """
    import importlib

    try:
        return importlib.import_module(f"dynamix._vendor.wtmm_ebsd.{name}")
    except ImportError:
        import importlib.util
        import sys
        from pathlib import Path

        alias = f"_eqselect_wtmm_ebsd_{name}"
        cached = sys.modules.get(alias)
        if cached is not None:
            return cached
        pkg_spec = importlib.util.find_spec("dynamix._vendor.wtmm_ebsd")   # does NOT exec __init__
        locations = list(getattr(pkg_spec, "submodule_search_locations", None) or [])
        if not locations:
            raise
        path = Path(locations[0]) / f"{name}.py"
        if not path.is_file():
            raise
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(alias, None)
            raise
        return module


#: Test-only (spec 09, Phase B): True routes the 2-D CWT on mlx and the tensor SVD through the
#: vendored ``wtmm_ebsd.cwt2d`` originals in place of the app's own transcriptions, so the two can
#: be compared until the last port; the app never sets it.
USE_VENDORED: bool = False

#: The app-wide default cwt2d engine. ``None`` = follow the FFT policy (mlx on Apple Silicon,
#: else FFTW3); "mlx"/"fftw" force one engine for every call that
#: does not pass its own ``_engine``. Set by the shell from ``Settings.compute_engine`` at
#: startup and on menu toggle. NEVER consulted by cache keys -- the backends-agree-to-float32
#: law makes engine choice infrastructure, not physics, so switching engines must keep every
#: stage cache valid.
_DEFAULT_ENGINE: "str | None" = None


def set_default_engine(engine: "str | None") -> None:
    """Set the app-wide default cwt2d engine: None/"auto" (follow the FFT policy,
    :mod:`dynamix.core.fft_policy`), "mlx" or "fftw"."""
    global _DEFAULT_ENGINE
    if engine in (None, "auto"):
        _DEFAULT_ENGINE = None
    elif engine in ("mlx", "fftw"):
        _DEFAULT_ENGINE = engine
    else:
        raise ValueError(f"unknown engine {engine!r}; expected 'auto', 'mlx' or 'fftw'")


def get_default_engine() -> "str | None":
    return _DEFAULT_ENGINE


def _observed_mlx_cwt(function, *args, **kwargs):
    """Record the explicit MLX call without changing the vendored implementation."""
    import mlx.core as mx
    from dynamix.core import fft_policy
    fft_policy.include_observed([{"precision": "float32", "backend": {
        "name": "mlx", "version": str(mx.__version__), "planner": None, "binding": None}}])
    return function(*args, **kwargs)


def _resolve_cwt2d_engine():
    """Lazily resolve the CWT operator injected as ``cwt_2d_f32``.

    On mlx the app's own :func:`_cwt2d_mlx` (the vendored ``cwt_2d_f32`` loop with the
    registry's filters; the original stays behind :data:`USE_VENDORED`), else the numpy
    transcription :func:`_cwt2d_numpy` on the policy's FFT. All three are
    signature-compatible -- ``(image, scales, *, pad, derivs, wavelet, verbose)`` returning a
    dict keyed by derivative name -- and implement ``derivs="all"``, which the tensor path
    needs for the wavelet Hessian.
    """
    from dynamix.core import fft_policy

    be = fft_policy.active()
    if be.name == "mlx":
        try:
            import mlx.core  # noqa: F401
            if USE_VENDORED:
                from dynamix._vendor.wtmm_ebsd.cwt2d import cwt_2d_f32
                from functools import partial
                return partial(_observed_mlx_cwt, cwt_2d_f32)
            from functools import partial
            return partial(_observed_mlx_cwt, _cwt2d_mlx)
        except ImportError:
            pass
    # The transcription on the policy backend: FFTW3 at the policy precision, or mlx's FFT
    # when the vendored mlx loop cannot be imported
    import functools
    return functools.partial(_cwt2d_numpy, fft=be)


def _cut_edge_extrema(extrema: dict, lx: int, ly: int, border_percent: float) -> dict:
    """``xsmurf_wrapper.cut_edge`` equivalent: inner-ROI crop of extrema.

    Mirrors the wrapper's convention exactly (``core.pyx::cut_edge``, itself
    xsmurf's ``rm_ext``): ``border_percent`` is the fraction of the image to
    KEEP, centred -- ``0.72`` keeps the central 72% by trimming
    ``int((1 - 0.72) / 2 * lx)`` = 14% from each side. Values ``>= 1.0`` trim
    nothing (the C computes a non-positive border and its bounds then cover the
    whole image), which is why ``twtmm``'s ``border_pct=1.0`` default is a
    no-op. The retained box is ``border <= coord <= extent - border``.
    """
    if border_percent is None or border_percent >= 1.0:
        return extrema
    bx = int((1.0 - border_percent) / 2.0 * lx)
    by = int((1.0 - border_percent) / 2.0 * ly)
    x, y = extrema["x"], extrema["y"]
    keep = ((x >= bx) & (x <= lx - bx) & (y >= by) & (y <= ly - by))
    if keep.all():
        return extrema
    return {k: v[keep] for k, v in extrema.items()}


def _make_chain_adapter(backend, *, box_ratio: float = 1.0,
                         dist2_max: float = 50.0, min_len: int = 2, band_per: str = "voice"):
    """Build the ``chain_and_extract`` callable injected into ``wtmm_ebsd.twtmm``.

    Accepts the full signature ``twtmm`` calls it with -- ``(ext_images,
    scales, method=, a_min=, n_oct=, n_vox=, similitude=, smooth=,
    border_percent=, dist_frac=)`` -- and implements it as
    ``border_percent`` crop -> :meth:`PythonWTMMBackend.chains2d`.

    ``similitude`` and ``smooth`` are FORWARDED (``chains2d`` is an exact
    xsmurf ``vchain`` port, so they map 1:1 onto the modulus-ratio band and
    ``ssm -smooth``). ``box_ratio``, ``dist2_max`` and ``min_len`` are closed
    over from :meth:`PythonWTMMBackend.tensor2d`'s arguments, since ``twtmm``
    has no parameters for them.

    ACCEPTED-BUT-IGNORED (documented deviation): ``method`` -- ``'greedy'`` is
    the only chaining method implemented, the ``'anchor'`` variant being an
    xsmurf-wrapper experiment; ``dist_frac`` -- an ``'anchor'``-only parameter;
    ``a_min``, ``n_oct``, ``n_vox`` -- the wrapper needs these to recompute the
    scale ladder for its box sizes, whereas ``chains2d`` derives the box
    directly from the ``scales`` array it is handed. Residual xsmurf parity is
    Phase C's benchmark harness.
    """
    def chain_and_extract(ext_images, scales, method="greedy", a_min=1.0,
                          n_oct=4, n_vox=8, similitude=0.8, smooth=False,
                          border_percent=1.0, dist_frac=0.1, **_ignored):
        extrema = [_cut_edge_extrema(ext.to_extrema_dict(), ext.lx, ext.ly,
                                      border_percent)
                   for ext in ext_images]
        return backend.chains2d(extrema, scales, similitude=similitude,
                                box_ratio=box_ratio, dist2_max=dist2_max,
                                min_len=min_len, smooth=smooth, band_per=band_per)

    return chain_and_extract


#: How the similitude band applies: per "voice", between neighbouring scales of the ladder (xsmurf's
#: vchain), or "octave" over a whole octave, each step taking its share.
BAND_PER = ("voice", "octave")


def chainable_h(similitude: float, n_voice: int, eta: float = 0.0,
                band_per: str = "voice") -> "tuple[float, float] | None":
    """The Hölder exponents a chain can carry through the similitude band, ``(h_min, h_max)``,
    or None with no band (``similitude`` 0). Along a chain the modulus goes as a^(h + eta), so a
    step of the ladder (a factor 2^(1/n_voice)) multiplies it by 2^((h + eta)/n_voice); the
    band admits ratios in (s, 1/s) per voice, or per octave with ``band_per="octave"``."""
    if similitude <= 0:
        return None
    width = -float(np.log2(similitude))
    if band_per == "voice":
        width *= int(n_voice)
    return (-width - float(eta), width - float(eta))


class PythonWTMMBackend:
    """Pure-Python WTMM backend; always works (the universal-binary path).

    This slice implements the 2D CWT stage (``cwt2d``), NMS extrema
    detection with within-scale line labeling (``extrema2d``), the xsmurf
    ``ssm`` single-maxima reduction (``single_maxima2d``), the xsmurf
    ``vchain`` cross-scale chaining (``chains2d``), the partition-function
    delegation (``partition2d``), the EBSD alpha-Jacobian tensor pipeline
    (``tensor2d``, which dependency-injects the pure-Python xsmurf shims into
    ``wtmm_ebsd.twtmm``) and the 1D pipeline.
    """

    name = "python"

    def cwt2d(self, field, scales, *, wavelet: str = "gaussian", pad: int = 32,
              derivs: str = "first", smoother_params: "dict | None" = None,
              _engine: "str | None" = None, pad_mode: str = "reflect") -> dict:
        """2D continuous wavelet transform.

        Parameters
        ----------
        field : (ny, nx) array, float, NaN allowed
            NaN/inf pixels are filled with the field's finite mean before
            the FFT (see the module NaN-policy docstring).
        scales : sequence of float
            Wavelet scales in pixels (e.g. from :func:`compute_scales2d`).
        wavelet : str
            One of :data:`dynamix.core.smoothers.NAMES`. "gaussian" and "mexican" run on
            the vendored loops; the other names run on the app's own loops
            (:func:`_cwt2d_mlx` on the device, :func:`_cwt2d_numpy` otherwise). Any other
            value raises ``ValueError`` -- checked here, before engine dispatch, because the
            ``mlx`` engine calls ``wtmm_ebsd.cwt2d.cwt_2d_f32`` directly and never touches
            :func:`_build_wavelet_filters_numpy`.
        pad : int
            Pad width in pixels.
        pad_mode : str
            What the FFT torus holds beyond the field (:data:`PAD_MODES`): ``"reflect"``
            (mirrored), ``"edge_pad"`` (the edge values repeated), ``"zero_pad"``, or
            ``"periodic"`` (nothing padded; the field itself is the torus).
        smoother_params : dict or None
            The registry smoother's parameters (``{"alpha": ...}``, ``{"q": ...}``);
            validated here before dispatch; ignored by the Gaussian pair.
        _engine : {None, "mlx", "fftw"}
            Private test hook. ``None`` (default) follows the FFT policy (mlx
            on Apple Silicon, else FFTW3). ``"mlx"`` forces the mlx engine and
            raises ``ImportError`` if mlx is unavailable. ``"fftw"`` forces the
            numpy transcription on FFTW3 (used to parity-test the two engines).

        Returns
        -------
        dict
            ``{"mod": (n_sc, ny, nx) float32, "arg": (n_sc, ny, nx) float32}``
            -- ``|grad(G_a * f)|`` and its angle (rad). Other derivative
            keys the engines compute internally are dropped from this
            public return.
        """
        field = np.asarray(field, dtype=np.float64)
        finite = np.isfinite(field)
        if not np.all(finite):
            fill_value = field[finite].mean() if finite.any() else 0.0
            field = np.where(finite, field, fill_value)

        from dynamix.core.smoothers import NAMES, smoother as _smoother
        if wavelet not in NAMES:
            raise ValueError(f"unknown wavelet {wavelet!r}; expected one of {NAMES}")
        registry = wavelet not in ("gaussian", "mexican")
        if registry:
            _smoother(wavelet, smoother_params)       # the parameter check, before dispatch

        engine = _engine
        if engine is None:
            engine = _DEFAULT_ENGINE          # the app-wide master setting, when one is set
        if engine is None:
            # Follow the FFT policy -- mlx (32-bit) on Apple Silicon, else FFTW3 at the
            # configured precision.
            from dynamix.core import fft_policy
            engine = {"mlx": "mlx", "pyfftw": "fftw"}[fft_policy.active().name]

        if engine == "mlx":
            import mlx.core  # noqa: F401  (raises ImportError if unavailable)
            if USE_VENDORED and not registry and pad_mode == "reflect":
                from dynamix._vendor.wtmm_ebsd.cwt2d import cwt_2d_f32
                raw = _observed_mlx_cwt(cwt_2d_f32, field, scales, pad=pad, derivs=derivs,
                                  wavelet=wavelet, verbose=False)
            else:
                raw = _observed_mlx_cwt(_cwt2d_mlx, field, scales, pad=pad, derivs=derivs,
                                 wavelet=wavelet, verbose=False, params=smoother_params,
                                 pad_mode=pad_mode)
        elif engine == "fftw":
            from dynamix.core import fft_policy
            be = fft_policy.active()
            if be.name != "pyfftw":
                be = fft_policy.make("pyfftw", 32)
            raw = _cwt2d_numpy(field, scales, pad=pad, derivs=derivs,
                                wavelet=wavelet, verbose=False, fft=be, params=smoother_params,
                                pad_mode=pad_mode)
        else:
            raise ValueError(f"unknown cwt2d engine {engine!r}")

        # float32 (xsmurf's contract), float64 when the FFTs ran at 64 bits
        real = np.float64 if np.asarray(raw["mod"]).dtype == np.float64 else np.float32
        out = {
            "mod": np.asarray(raw["mod"], dtype=real),
            "arg": np.asarray(raw["arg"], dtype=real),
        }
        if derivs == "all":
            # The follow detector's derivative stacks -- both engines produce the
            # same nine keys (the numpy filter bank is an exact transcription of the mlx one).
            for name in ("dx", "dy", "dxx", "dxy", "dyy", "dxxx", "dxxy", "dxyy", "dyyy"):
                out[name] = np.asarray(raw[name], dtype=real)
        return out

    def extrema2d(self, cwt: dict, scales, *, thresh: float = 1e-3,
                   field: "np.ndarray | None" = None, real_block=None, real_origin=(0, 0),
                   tie_rule: str = "asymmetric") -> list:
        """Per-scale non-maximum-suppression (NMS) modulus maxima.

        For each scale, a pixel is kept when its modulus beats its
        bilinearly-interpolated neighbours one step along the (unit)
        gradient direction and one step against it, ties broken by ``tie_rule``
        (:data:`TIE_RULES`; xsmurf's: > along, >= against) -- the standard
        Canny-style directional NMS, but using the *wavelet* gradient angle
        (``cwt["arg"]``) rather than a re-derived image gradient, and fully
        vectorized over the pixel grid (``scipy.ndimage.map_coordinates`` on
        the whole array at once, no pixel loop). A global threshold
        (``mod >= thresh * mod.max()`` at that scale) is ANDed in.

        Within-scale connected components of the NMS mask (8-connectivity)
        are labelled as "lines" via ``scipy.ndimage.label``; isolated
        (size-1) components get ``line_id = -1`` so downstream chaining/
        display code can distinguish a true ridge segment from a lone
        speckle.

        Parameters
        ----------
        cwt : dict
            ``{"mod", "arg"}`` as returned by :meth:`cwt2d`, shape
            ``(n_sc, ny, nx)``.
        scales : sequence of float
            Same scales used to produce ``cwt`` (only used here to size the
            NaN-dilation radius, ``ceil(scale)`` pixels).
        thresh : float
            Fraction of the per-scale modulus max below which extrema are
            dropped.
        field : (ny, nx) array or None
            Original (possibly NaN-bearing) field. When given, extrema whose
            support -- the original invalid mask dilated by ``ceil(scale)``
            pixels -- touches a pixel that was NaN/inf in ``field`` are
            dropped (the NaN-fill in ``cwt2d`` would otherwise fabricate
            spurious ridge structure right at the mask boundary).
        real_block : (rows, cols) or None
            A Periodic extension output's real data, one block (spec 25 §3.1): no
            extrema outside it, and the threshold's maximum is the block's.
        real_origin : (row0, col0)
            Where that block starts: the origin, or the centred layout's offset.

        Returns
        -------
        list[dict]
            One dict per scale: ``{"x", "y" (int64), "mod", "arg" (float64),
            "line_id" (int64)}``, all shape ``(m,)`` for that scale's ``m``
            surviving extrema.
        """
        mod_stack = np.asarray(cwt["mod"], dtype=np.float64)
        arg_stack = np.asarray(cwt["arg"], dtype=np.float64)
        n_sc, ny, nx = mod_stack.shape

        invalid = None
        if field is not None:
            invalid = ~np.isfinite(np.asarray(field))

        grid = np.mgrid[0:ny, 0:nx].astype(np.float64)

        # The per-scale detection itself lives in `_nms_extrema_scale`, which the
        # `_XsmShim.wtmm2d` single-scale entry point shares verbatim.
        return [
            _nms_extrema_scale(mod_stack[s], arg_stack[s], thresh=thresh,
                                invalid=invalid, radius=int(np.ceil(scales[s])),
                                grid=grid, real_block=real_block, real_origin=real_origin,
                                tie_rule=tie_rule)
            for s in range(n_sc)
        ]

    def single_maxima2d(self, extrema: list, *, smooth: bool = False) -> list:
        """xsmurf ``ssm``: an EXACT port of ``single_max.c::_is_single_max_``.

        The WTMM skeleton proper (the "WTMMM" points) is not the full NMS
        extrema set: xsmurf keeps only the points that pass ``_is_single_max_``
        (``search_single_max``), after ``ssm -smooth``'s optional dip filling
        along each maxima line. Only those points take part in vertical
        (cross-scale) chaining -- see :meth:`chains2d`.

        The C's test is purely LOCAL -- it never consults the line ordering,
        only the per-scale extrema mask (``array``, built from *all* extrema at
        that scale, not just same-line ones) -- and has two parts:

        (a) **Exactly two neighbour runs.** Sweeping the 8 neighbours in the
            clockwise ``init_pos_incr`` order with a 9-entry wraparound
            (``pos_incr[8] == pos_incr[0]``), count the positions where
            neighbour ``i`` holds an extremum AND neighbour ``i+1`` does not
            (or is outside the image). That count is the number of contiguous
            runs of extrema around the point, and the C requires it to be
            EXACTLY 2. So an interior "through" point of a line qualifies,
            while a line END (1 run), a singleton (0 runs) and a T/X junction
            (3+ runs) are all REJECTED -- "The ends of a line can't be in this
            list".
        (b) **Locally maximal modulus.** ``mod`` must be ``>=`` the modulus of
            every 8-neighbour that holds an extremum (the C returns 0 as soon
            as ``ext->mod < array[pos1]->mod``, so equal moduli survive).

        Both parts are pure mask/shift operations, computed here with 8 shifted
        boolean/float lookups through the pixel->point grid; there is no
        per-point Python loop.

        The ordered-line walk (``_walk_lines_py``, njit ``cache=True``) is
        retained ONLY for ``smooth=True``: ``ssm -smooth`` overwrites ``e->mod``
        in place before the test, so the smoothed modulus is what feeds BOTH
        tests above and what the returned dicts carry.

        Deviation (unavoidable, negligible): the C's ``_is_in_image_`` derives
        ``x = pos % lx`` from a flat index, so a neighbour off the left/right
        edge silently wraps onto the adjacent row and is tested there. This
        port uses the true geometric 8-neighbourhood (out-of-grid neighbours
        simply hold no extremum), which is what the C intends.

        Parameters
        ----------
        extrema : list[dict]
            Per-scale extrema as returned by :meth:`extrema2d` (needs
            ``{"x", "y", "mod", "arg", "line_id"}``).
        smooth : bool
            Apply ``ssm -smooth``'s dip filling along each ordered line before
            the ``_is_single_max_`` test. Off by default: xsmurf's WTMM scripts
            call ``ssm`` without it. When on, as in xsmurf (which overwrites
            ``e->mod`` in place) the SMOOTHED modulus is what the returned
            dicts carry, so cross-scale comparisons see the same values.

        Returns
        -------
        list[dict]
            One dict per scale with the same schema as :meth:`extrema2d`
            (``{"x", "y", "mod", "arg", "line_id"}``), a subset of the input
            points kept in the input's (raster) order.
        """
        out = []
        for e in extrema:
            x = np.asarray(e["x"], dtype=np.int64)
            y = np.asarray(e["y"], dtype=np.int64)
            mod = np.asarray(e["mod"], dtype=np.float64)
            arg = np.asarray(e["arg"], dtype=np.float64)
            line_id = np.asarray(e["line_id"], dtype=np.int64)
            n = x.size
            if n == 0:
                out.append({"x": x, "y": y, "mod": mod, "arg": arg,
                            "line_id": line_id})
                continue

            # pixel -> point lookup (extrema pixels are unique by construction).
            # `nx`/`ny` bound the extrema, not the source image: a neighbour
            # outside this box cannot hold an extremum, and both C branches
            # treat "outside the image" and "no extremum here" identically.
            nx = int(x.max()) + 1
            ny = int(y.max()) + 1
            grid = np.full(ny * nx, -1, dtype=np.int64)
            grid[y * nx + x] = np.arange(n, dtype=np.int64)

            mod_s = mod.copy()
            if smooth:
                mod_s = self._smooth_chains(x, y, mod, line_id, grid, nx, ny)

            # --- `_is_single_max_`, vectorized -------------------------------
            # nbr_ext[k] / nbr_mod[k]: does clockwise neighbour k hold an
            # extremum, and (if so) what is its (smoothed) modulus.
            nbr_ext = np.empty((8, n), dtype=bool)
            nbr_mod = np.empty((8, n), dtype=np.float64)
            for k in range(8):
                xx = x + _SM_OFF_X[k]
                yy = y + _SM_OFF_Y[k]
                inside = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
                flat = np.where(inside, yy * nx + xx, 0)
                j = np.where(inside, grid[flat], -1)
                is_ext = j >= 0
                nbr_ext[k] = is_ext
                nbr_mod[k] = np.where(is_ext, mod_s[np.maximum(j, 0)], -np.inf)

            # (a) count extremum -> non-extremum transitions around the ring;
            #     np.roll(-1) IS the C's pos_incr[i+1] with pos_incr[8] == [0].
            runs = (nbr_ext & ~np.roll(nbr_ext, -1, axis=0)).sum(axis=0)
            # (b) no 8-neighbour extremum may have a strictly larger modulus
            keep = (runs == 2) & (mod_s >= nbr_mod.max(axis=0))

            sel = np.nonzero(keep)[0]                 # already in raster order
            out.append({
                "x": x[sel],
                "y": y[sel],
                "mod": mod_s[sel],
                "arg": arg[sel],
                "line_id": line_id[sel],
            })
        return out

    @staticmethod
    def _smooth_chains(x, y, mod, line_id, grid, nx, ny):
        """xsmurf ``ssm -smooth``: dip filling ALONG each ordered maxima line.

        Each line (``line_id`` component of :meth:`extrema2d`) is ORDERED along
        itself by a nearest-neighbour walk from an endpoint (``_walk_lines_py``,
        njit-compiled with ``cache=True``); branches become separate segments so
        a smoothing window never bridges two topologically separate branches.
        Singletons (``line_id == -1``) are one-point lines and are left alone.

        Returns a copy of ``mod`` with the smoothed values in the INPUT point
        order (xsmurf overwrites ``e->mod`` in place, so this is the modulus
        every later stage sees).
        """
        smooth_dips = _line_kernels()["smooth"]
        order, seg, _starts = _order_lines(x, y, line_id, grid, nx, ny)
        if not order.size:
            return mod.copy()

        mod_ord = mod[order].copy()
        smooth_dips(mod_ord, seg)
        mod_s = mod.copy()
        mod_s[order] = mod_ord
        return mod_s

    def chains2d(self, extrema: list, scales, *, similitude: float = 0.8,
                 box_ratio: float = 1.0, dist2_max: float = 50.0,
                 min_len: int = 2, smooth: bool = False, band_per: str = "voice",
                 angle_test: bool = False, angle_threshold: float = 0.8,
                 radius_rule: str = "xsmurf", radius_c: float = 1.0,
                 tie_break: str = "distance") -> list:
        """Cross-scale WTMM chains -- an exact port of xsmurf's ``vchain``.

        Transcribes ``vert_chain`` (``xsmurf/wt2d/chain2.c``) together with the
        ``eiChain.tcl`` driver (``xsmurf_wrapper.chaining.chain``), which is the
        code path xsmurf uses for BOTH the scalar and the tensor 2D WTMM:

        1. **Single maxima only.** :meth:`single_maxima2d` (xsmurf ``ssm``,
           optionally preceded by ``smooth_chains``) reduces each scale to the
           maxima of the modulus along its maxima lines; ``vert_chain`` fills
           its ``up_array`` from ``line->gr_lst``, i.e. those points and no
           others, so they are the ONLY points that chain vertically.
        2. **Eligibility.** Working from the finest scale up, a point at scale
           ``s`` may look for a partner at ``s+1`` only if it already carries a
           link from below (``do_ext->down``) or ``s == 0``
           (``is_first``). Chains are therefore finest-anchored: no chain is
           ever born at mid-scale.
        3. **Candidacy.** A coarser single maximum ``q`` is a candidate for the
           finer point ``p`` when it lies in the Chebyshev box of half-size
           ``B = max(1, int(box_ratio * log2(a_{s+1}) * 2))`` around ``p``
           (``_ComputeBox_``, with ``a_{s+1}`` the xsmurf-normalized coarser
           scale exactly as ``chaining.chain`` computes ``box_size``), its
           modulus ratio is inside the similitude band ``similitude <
           mod_q / mod_p < 1 / similitude``, and the SQUARED euclidean
           distance is below ``dist2_max`` (``ExtChnDistance_`` returns a
           squared distance; the C's hard-coded ``dist < 50`` is ~7.07 px).
        4. **Selection.** ``p`` takes the nearest admissible candidate. The C
           scans the box with ``dist < dist_min``, so the FIRST minimum in
           raster (y-major) order wins a tie; the single-maxima arrays are in
           that same raster order here, so the tie-break is "lowest candidate
           index".
        5. **Competition.** A coarser point keeps only its CLOSEST claimant
           (``dist1 < dist2`` -> steal, incumbent wins ties). The loser is NOT
           re-queued for a second-best partner -- the C's re-queue line is
           commented out -- so its chain simply TERMINATES at scale ``s``,
           keeping the segment it has accumulated. This is what makes maxima
           lines merge as the scale grows, and hence what makes ``N(a)`` decay
           and ``tau(0) ~ -D_F`` come out negative.

        Because the competition's outcome is "each coarser point is won by its
        nearest claimant", the fixed point of the C's sequential ``while``
        loop is order-independent and is computed here in two vectorized
        passes per scale pair -- per-finer nearest admissible candidate
        (``cKDTree`` k-nearest with a bounded escalation of ``k``), then
        per-coarser argmin claimant (``np.lexsort`` + first-occurrence mask).
        There is no Python loop over extrema anywhere.

        Parameters
        ----------
        extrema : list[dict]
            Per-scale extrema as returned by :meth:`extrema2d`, one dict per
            entry of ``scales``, index 0 = finest scale.
        scales : sequence of float
            Same scales used to produce ``extrema`` (xsmurf-normalized, i.e.
            straight from :func:`compute_scales2d`).
        similitude : float
            Modulus-ratio band half-width; candidates must satisfy
            ``similitude < mod_coarse / mod_fine < 1 / similitude``.
        box_ratio : float
            Multiplier on the search-box half-size (xsmurf ``box_ratio``).
        dist2_max : float
            Hard cap on the SQUARED linking distance in pixels (xsmurf's 50).
        min_len : int
            Chains spanning fewer than this many scales are dropped.
        smooth : bool
            Forwarded to :meth:`single_maxima2d` (xsmurf ``ssm -smooth``).
        angle_test, angle_threshold : bool, float
            Off by default (xsmurf: the test is commented out of ``vchain``). On, a candidate
            also needs ``1 - |Δarg|/π > angle_threshold`` for Δarg or Δarg − π (a gradient
            flipped by π is accepted).
        radius_rule, radius_c : str, float
            ``"xsmurf"`` (default): the box and the squared-distance cap above.
            ``"scale_proportional"``: radius ``r = radius_c · a_{s+1} · (a_{s+1}/a_s − 1)`` (the
            normalized scales are already in pixels, ``a_px = a · 6 / 0.86``), as the box
            half-size and the distance cap.
        tie_break : str
            ``"distance"`` (default): a coarser maximum keeps its closest claimant (xsmurf).
            ``"modulus"``: the claimant whose modulus is closest (smallest ``|ln ratio|``).

        Returns
        -------
        list[dict]
            One dict per surviving chain, matching the ``wtmm_ebsd`` chain
            schema exactly so ``wtmm_ebsd.chain_filters`` and
            ``wtmm_ebsd.partition.build_hd_from_chains`` consume it
            unchanged: ``{"x": (k,) int64, "y": (k,) int64, "mod": (k,)
            float64, "log2_mod": (k,) float64, "log2_scales": (k,)
            float64}``, index 0 = finest scale, ``k`` = scales spanned.
        """
        from scipy.spatial import cKDTree

        scales = np.asarray(scales, dtype=np.float64)
        n_sc = len(scales)
        sm = self.single_maxima2d(extrema, smooth=smooth)
        xs = [np.asarray(e["x"], dtype=np.float64) for e in sm]
        ys = [np.asarray(e["y"], dtype=np.float64) for e in sm]
        mods = [np.asarray(e["mod"], dtype=np.float64) for e in sm]
        pts = [np.column_stack([xs[s], ys[s]]) for s in range(n_sc)]

        if band_per not in BAND_PER:
            raise ValueError(f"band_per {band_per!r}; known: {BAND_PER}")
        if radius_rule not in ("xsmurf", "scale_proportional"):
            raise ValueError(f"radius_rule {radius_rule!r}; known: ('xsmurf', 'scale_proportional')")
        if tie_break not in ("distance", "modulus"):
            raise ValueError(f"tie_break {tie_break!r}; known: ('distance', 'modulus')")
        args = [np.asarray(e["arg"], dtype=np.float64) for e in sm] if angle_test else None

        # --- vchain, scale pair by scale pair (the ONLY loop over scales) ---
        parent = [np.full(len(pts[s]), -1, dtype=np.int64) for s in range(n_sc)]
        linked = np.zeros(len(pts[0]), dtype=bool)      # unused at s == 0
        for s in range(n_sc - 1):
            n_f, n_c = len(pts[s]), len(pts[s + 1])
            nxt_linked = np.zeros(n_c, dtype=bool)
            elig = (np.arange(n_f, dtype=np.int64) if s == 0
                    else np.nonzero(linked)[0])
            if n_f == 0 or n_c == 0 or elig.size == 0:
                linked = nxt_linked
                continue

            if radius_rule == "scale_proportional":
                radius = radius_c * scales[s + 1] * (scales[s + 1] / scales[s] - 1.0)
                box, cap2 = max(1.0, radius), radius * radius
            else:
                box, cap2 = max(1, int(box_ratio * np.log2(scales[s + 1]) * 2)), dist2_max
            r_max = float(np.sqrt(cap2))
            # The modulus-ratio band: xsmurf's, per voice (one step of the ladder); per octave, the
            # similitude bounds the ratio over a whole octave and this step takes its share.
            lo_band = (similitude ** float(np.log2(scales[s + 1] / scales[s]))
                       if band_per == "octave" and similitude > 0 else similitude)
            hi_band = 1.0 / lo_band if lo_band > 0 else np.inf
            fx, fy = xs[s][elig], ys[s][elig]
            fmod = mods[s][elig]
            cx, cy, cmod = xs[s + 1], ys[s + 1], mods[s + 1]
            tree = cKDTree(pts[s + 1])

            best_j = np.full(elig.size, -1, dtype=np.int64)
            best_d2 = np.full(elig.size, np.inf)
            todo = np.arange(elig.size, dtype=np.int64)

            # k escalation: the k-nearest query can in principle truncate away
            # an admissible candidate when the k closest coarser points all
            # fail the band/box test. Rows that are still unresolved AND whose
            # k-th neighbour was itself inside r_max (i.e. the query really was
            # truncated) are retried with a larger k, ending at k = n_c, which
            # is exhaustive -- so the result is exact, and in practice only the
            # first level ever runs.
            k_levels = sorted({min(16, n_c), min(64, n_c), n_c})
            for k in k_levels:
                if todo.size == 0:
                    break
                d, j = tree.query(np.column_stack([fx[todo], fy[todo]]), k=k,
                                  distance_upper_bound=r_max)
                d = np.atleast_2d(d.reshape(todo.size, -1))
                j = np.atleast_2d(j.reshape(todo.size, -1))
                ok = np.isfinite(d) & (j < n_c)
                jj = np.where(ok, j, 0)
                d2 = d * d
                ratio = np.where(ok, cmod[jj] / fmod[todo][:, None], 0.0)
                adm = (ok
                       & (np.abs(cx[jj] - fx[todo][:, None]) <= box)
                       & (np.abs(cy[jj] - fy[todo][:, None]) <= box)
                       & (ratio > lo_band) & (ratio < hi_band)
                       & (d2 < cap2))
                if angle_test:
                    turn = np.abs(np.angle(np.exp(1j * (args[s + 1][jj]
                                                        - args[s][elig][todo][:, None]))))
                    adm &= np.maximum(1.0 - turn / np.pi, turn / np.pi) > angle_threshold
                # nearest admissible, ties broken by lowest candidate index
                key_d = np.where(adm, d2, np.inf)
                key_j = np.where(adm, jj, n_c)
                pick = np.lexsort((key_j, key_d), axis=-1)[:, 0]
                got = np.take_along_axis(adm, pick[:, None], 1)[:, 0]
                if got.any():
                    rows = todo[got]
                    p = pick[got]
                    best_j[rows] = jj[got, p]
                    best_d2[rows] = d2[got, p]
                truncated = np.isfinite(d[:, -1]) if k < n_c else np.zeros(todo.size, bool)
                todo = todo[~got & truncated]

            # --- pass (b): each coarser single max keeps its closest claimant (xsmurf), or the
            # claimant whose modulus is closest (tie_break="modulus")
            has = best_j >= 0
            if has.any():
                f_idx = elig[has]
                c_idx = best_j[has]
                d2_win = (best_d2[has] if tie_break == "distance"
                          else np.abs(np.log(cmod[c_idx] / fmod[has])))
                order = np.lexsort((d2_win, c_idx))     # by target, then dist
                first = np.ones(order.size, dtype=bool)
                first[1:] = c_idx[order][1:] != c_idx[order][:-1]
                win = order[first]
                parent[s][f_idx[win]] = c_idx[win]
                nxt_linked[c_idx[win]] = True
            linked = nxt_linked

        # --- walk ALL finest-scale seeds simultaneously (vectorized over
        # seeds; the loop here is over scales only, to advance the pointer
        # chase and gather per-scale values) ---
        n0 = len(pts[0])
        idx_mat = np.full((n0, n_sc), -1, dtype=np.int64)
        x_mat = np.zeros((n0, n_sc), dtype=np.float64)
        y_mat = np.zeros((n0, n_sc), dtype=np.float64)
        mod_mat = np.zeros((n0, n_sc), dtype=np.float64)

        cur = np.arange(n0, dtype=np.int64)
        for s in range(n_sc):
            valid = cur >= 0
            idx_mat[valid, s] = cur[valid]
            if valid.any():
                x_mat[valid, s] = xs[s][cur[valid]]
                y_mat[valid, s] = ys[s][cur[valid]]
                mod_mat[valid, s] = mods[s][cur[valid]]
            if s < n_sc - 1:
                nxt = np.full(n0, -1, dtype=np.int64)
                if valid.any():
                    nxt[valid] = parent[s][cur[valid]]
                cur = nxt

        # Contiguous finest-anchored chains -> valid entries are a row-wise
        # prefix, so a simple count of non-(-1) entries IS the chain length.
        lengths = (idx_mat != -1).sum(axis=1)
        keep = np.nonzero(lengths >= min_len)[0]

        x_int = x_mat.astype(np.int64)
        y_int = y_mat.astype(np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log2_mod_mat = np.log2(np.abs(mod_mat))
        log2_scales_full = np.log2(scales)

        # Materialization: the wtmm_ebsd list-of-dicts interface boundary.
        # Everything numeric is already precomputed above -- this loop only
        # slices each kept row at its own (precomputed) length.
        chains = []
        for i in keep:
            k = int(lengths[i])
            chains.append({
                "x": x_int[i, :k].copy(),
                "y": y_int[i, :k].copy(),
                "mod": mod_mat[i, :k].copy(),
                "log2_mod": log2_mod_mat[i, :k].copy(),
                "log2_scales": log2_scales_full[:k].copy(),
            })
        return chains

    def partition2d(self, chains: list, scales, q_list=None, *,
                     min_chain_len: int = 2):
        """2D partition function (thin delegation to ``wtmm_ebsd``).

        Delegates to ``wtmm_ebsd.partition.build_hd_from_chains`` UNCHANGED
        -- the custom Boltzmann-average / Chhabra-Jensen math (and its
        sample-variance error propagation) lives there and is never
        transcribed here.

        Parameters
        ----------
        chains : list[dict]
            Chain dicts as returned by :meth:`chains2d` (only ``"mod"`` is
            read by ``build_hd_from_chains``).
        scales : sequence of float
            Same scales used to produce ``chains``.
        q_list : sequence of float or None
            q grid; ``None`` -> ``wtmm_ebsd.partition.DEFAULT_Q_LIST`` (a
            finer 0.1-spaced grid tuned for offline batch runs -- distinct
            from this module's coarser interactive-GUI default,
            :data:`DEFAULT_Q`).
        min_chain_len : int
            Chains shorter than this are dropped before accumulation
            (forwarded unchanged).

        Returns
        -------
        (hd_std, hd_cmax) : tuple[dict, dict]
            Exactly ``build_hd_from_chains``'s return value.
        """
        if USE_VENDORED:
            from dynamix._vendor.wtmm_ebsd.partition import DEFAULT_Q_LIST, build_hd_from_chains
        else:
            from dynamix.core.partition_tables import DEFAULT_Q_LIST, build_hd_from_chains

        if q_list is None:
            q_list = DEFAULT_Q_LIST
        return build_hd_from_chains(chains, scales, q_list, min_chain_len=min_chain_len)

    def tensor2d(self, field3, scales, *,
                 svd_modes=("sigma_max", "sigma_min", "alpha_sigma_max",
                            "alpha_sigma_min", "modL", "modT"),
                 wavelet: str = "gaussian", wavelet_hessian: str = "gaussian",
                 use_wavelet_hessian: bool = True, fracint_alpha: float = 0.0,
                 thresh: float = 1e-3, pad: int = 32, similitude: float = 0.8,
                 box_ratio: float = 1.0, dist2_max: float = 50.0,
                 min_len: int = 2, smooth: bool = True, band_per: str = "voice") -> dict:
        """EBSD alpha-Jacobian tensor WTMM (delegates to ``wtmm_ebsd.twtmm``).

        DRY rule (non-negotiable): the CUSTOM tensor math is REUSED, never
        transcribed. This method is a thin adapter around
        ``wtmm_ebsd.twtmm.alpha_jacobian_twtmm``, which owns the per-scale
        kappa_ij from the 3-component CWT, the Pantleon Nye-alpha assembly, the
        3x2 kappa-Jacobian and 6x2 alpha-Jacobian SVDs and the per-mode
        modulus/argument fields. Only its three xsmurf-injected stages are
        replaced, by pure-Python equivalents:

        ==================  ====================================================
        injected            supplied here
        ==================  ====================================================
        ``cwt_2d_f32``      :func:`_resolve_cwt2d_engine` (mlx original, else
                            :func:`_cwt2d_numpy`)
        ``xsm``             :class:`_XsmShim` (NMS via
                            :func:`_nms_extrema_scale`, Hölder via
                            ``wtmm_ebsd.chain_filters.chain_ols_holder``)
        ``chain_and_extract``  :func:`_make_chain_adapter` ->
                            :meth:`chains2d` (the exact xsmurf ``vchain`` port)
        ==================  ====================================================

        No ``xsmurf`` module is imported at any point.

        Parameters
        ----------
        field3 : (ny, nx, 3) array, float, NaN allowed
            Demeaned log-orientation field (a ``RasterField.values`` with
            ``n_components == 3``). Any other shape raises ``ValueError``.
            Non-finite (NaN/inf) pixels are filled with their OWN component's
            finite mean before the transform, per the module NaN policy; a
            component with no finite pixel at all raises ``ValueError``.
        scales : sequence of float
            Wavelet scales in pixels (e.g. from :func:`compute_scales2d`).
        svd_modes : tuple of str
            Which SVD-modulus modes to run scalar WTMM on; a subset of
            ``{"sigma_max", "sigma_min", "alpha_sigma_max", "alpha_sigma_min",
            "modL", "modT"}``.
        wavelet, wavelet_hessian, use_wavelet_hessian, fracint_alpha, pad
            Forwarded to ``alpha_jacobian_twtmm`` unchanged.
            ``use_wavelet_hessian=True`` makes it request ``derivs="all"`` so
            the alpha-Jacobian is exact rather than a pixel-level gradient.
        thresh : float
            Per-scale NMS modulus threshold (fraction of that scale's max).
        similitude, box_ratio, dist2_max, min_len, smooth
            Chaining parameters; reach :meth:`chains2d` through the injected
            adapter (``similitude`` and ``smooth`` via ``twtmm``'s own
            ``similitude``/``smooth_chain`` arguments, the rest closed over).

        Returns
        -------
        dict
            ``{mode: {"chains": list[chain dict], "holders": list[{"h"}],
            "extrema": per-scale list[dict] in the :meth:`extrema2d` schema,
            "mod": (n_sc, ny, nx) float32, "scales": (n_sc,) float64}}`` for
            each requested mode.
        """
        for _name, _value in (("wavelet", wavelet), ("wavelet_hessian", wavelet_hessian)):
            if _value not in ("gaussian", "mexican"):
                raise ValueError(
                    f"unknown {_name} {_value!r}; expected 'gaussian' or 'mexican'"
                )

        field3 = np.asarray(field3)
        if field3.ndim != 3 or field3.shape[-1] != 3:
            raise ValueError(
                f"tensor2d expects a 3-component field of shape (ny, nx, 3); "
                f"got {field3.shape}"
            )
        field3 = field3.astype(np.float64, copy=False)

        # NaN policy (module docstring): masked-out pixels are filled with the
        # finite mean BEFORE the FFT, PER COMPONENT -- the three components of a
        # log-orientation field carry independent offsets, so a single global
        # mean would punch an artificial step into every component but one.
        # Without this fill the FFT smears one NaN across the whole padded image,
        # every derivative comes back all-NaN, the NMS mask goes all-False and
        # tensor2d silently returns zero chains.
        finite = np.isfinite(field3)
        if not finite.all():
            n_finite = finite.sum(axis=(0, 1))                       # (3,)
            dead = np.nonzero(n_finite == 0)[0]
            if dead.size:
                raise ValueError(
                    f"tensor2d: component(s) {dead.tolist()} of field3 are "
                    f"entirely non-finite (no finite pixel to take a mean over); "
                    f"there is no signal to transform"
                )
            fill = np.where(finite, field3, 0.0).sum(axis=(0, 1)) / n_finite
            field3 = np.where(finite, field3, fill)

        scales = np.asarray(scales, dtype=np.float64)
        if USE_VENDORED:
            twtmm = _load_wtmm_ebsd_module("twtmm")
        else:
            from dynamix.core import tensor_wtmm as twtmm

        svd_wtmm = twtmm.alpha_jacobian_twtmm(
            field3, scales,
            cwt_2d_f32=_resolve_cwt2d_engine(),
            xsm=_XsmShim,
            chain_and_extract=_make_chain_adapter(
                self, box_ratio=box_ratio, dist2_max=dist2_max, min_len=min_len,
                band_per=band_per),
            fracint_alpha=fracint_alpha,
            pad=pad,
            wavelet=wavelet,
            wavelet_hessian=wavelet_hessian,
            use_wavelet_hessian=use_wavelet_hessian,
            similitude=similitude,
            smooth_chain=smooth,
            thresh=thresh,
            svd_modes=tuple(svd_modes),
            verbose=False,
        )

        return {
            mode: {
                "chains": res["chains"],
                "holders": res["holders"],
                "extrema": [ext.to_extrema_dict() for ext in res["ext_images"]],
                "mod": res["mod"],
                "scales": scales.copy(),
            }
            for mode, res in svd_wtmm.items()
        }

    def cwt1d(self, signal, a_min, n_oct, n_voice, *, wavelet: str = "g2",
              expo: float = -1.0) -> tuple:
        """1D continuous wavelet transform (thin delegation to ``wtmm.cwt``).

        Engine: the FFT policy's -- ``wtmm.cwt.cwtd_mlx`` when the policy runs
        on mlx, else ``wtmm.cwt.cwtd_fftw`` (FFTW3).

        Returns
        -------
        (coeffs, scales, valid_ranges)
            ``coeffs`` : (n_oct*n_voice, n) float64
            ``scales`` : (n_oct*n_voice,) float64
            ``valid_ranges`` : list[(first, last)] per scale
        """
        signal = np.asarray(signal, dtype=np.float64)
        from dynamix.core import fft_policy
        policy = fft_policy.active().name          # mlx or FFTW3, per policy
        try:
            if policy != "mlx":
                raise ImportError
            import mlx.core  # noqa: F401
        except ImportError:
            pass
        else:
            from dynamix._vendor.wtmm.cwt import cwtd_mlx
            return cwtd_mlx(signal, a_min, n_oct, n_voice, wavelet_name=wavelet, expo=expo)
        from dynamix._vendor.wtmm.cwt import cwtd_fftw
        return cwtd_fftw(signal, a_min, n_oct, n_voice, wavelet_name=wavelet, expo=expo)

    def extrema1d(self, coeffs, scales, *, valid_ranges=None,
                  epsilon: float = 1e-6) -> tuple:
        """1D WTMM extrema (thin delegation to ``wtmm.extrema.compute_extrep``).

        ``compute_extrep`` unconditionally ``print()``s a "Total extrema:
        N" summary line; that is redirected to an in-memory buffer here
        (and discarded) so using this backend as a library stays quiet.
        """
        import contextlib
        import io

        from dynamix._vendor.wtmm.extrema import compute_extrep

        with contextlib.redirect_stdout(io.StringIO()):
            return compute_extrep(coeffs, scales, valid_ranges=valid_ranges, epsilon=epsilon)

    def chains1d(self, extrep_abs, extrep_ord) -> tuple:
        """1D WTMM chaining (thin delegation to ``wtmm.chains.chain_all``)."""
        from dynamix._vendor.wtmm.chains import chain_all
        return chain_all(extrep_abs, extrep_ord)

    def partition1d(self, extrep_ord, n_oct, n_voice, a_min, q_list, *,
                    coarser_links=None, min_chain_voices=None) -> dict:
        """1D partition function (thin delegation to
        ``wtmm.partition.compute_partition_function``)."""
        from dynamix._vendor.wtmm.partition import compute_partition_function
        return compute_partition_function(
            extrep_ord, n_oct, n_voice, a_min, q_list,
            coarser_links=coarser_links, min_chain_voices=min_chain_voices,
        )


def get_backend(name: str = "python"):
    """Resolve a :class:`WTMMBackend` by name.

    ``"python"`` -> :class:`PythonWTMMBackend`. ``"xsmurf"`` -> Phase C's
    xsmurf-accelerated backend is not built yet, so this warns
    (``UserWarning``) and falls back to :class:`PythonWTMMBackend`. Any
    other name raises ``ValueError``.
    """
    if name == "python":
        return PythonWTMMBackend()
    if name == "xsmurf":
        warnings.warn(
            "backend 'xsmurf' is not available yet (Phase C); "
            "falling back to PythonWTMMBackend",
            UserWarning,
            stacklevel=2,
        )
        return PythonWTMMBackend()
    raise ValueError(f"unknown WTMM backend {name!r}")


# ---------------------------------------------------------------------------
# Chain npz export/load (schema v3, byte-compatible with topo_wtmm.export_extrema's v2)
#
# v2 files (``topo_wtmm.export_extrema``) carry lon/lat/elev_m in ``h_xyz``/``v_xyz`` and
# are loaded by ``dynamix.core.reflayers.load_topo_extrema`` (untouched -- that loader applies
# the legacy ``depth_km = -elev_m / 1000`` relief convention). v3 is the SAME CSR layout
# (flat point arrays + per-chain offsets) plus a handful of additive keys, but its z
# column is always 0.0 (Phase A never writes a relief/elevation z into a v3 file, even
# for a GeographicFrame field) -- so the v3 loader never applies that flip. Only files
# stamped ``schema_version < 3`` (or missing the key entirely, like hand-built legacy
# fixtures) carry the elev_m convention, and those are delegated to
# ``load_topo_extrema`` VERBATIM rather than re-implemented here.
# ---------------------------------------------------------------------------

def export_chains_npz(path, *, chains: list, extrema: list, scales, field,
                      params: dict) -> dict:
    """Write WTMM chains (H = within-scale maxima lines, V = cross-scale chains) to a
    schema-v3 ``.npz`` that :func:`load_chains_npz` (and, transparently, the legacy
    :func:`dynamix.core.reflayers.load_topo_extrema`) can read back.

    Two chain sets, each as flat frame-unit coordinates + CSR offsets so they
    reconstruct as polylines, mirroring ``topo_wtmm.export_extrema``'s v2 layout:

      HORIZONTAL (each scale's within-scale maxima-line components, ``extrema[s]
      ["line_id"]``, walked along themselves via :func:`_order_lines` -- the SAME
      numba-cache ordering kernel :meth:`PythonWTMMBackend.single_maxima2d` and
      :class:`_ExtImageShim` use, not duplicated here):
        ``h_xyz`` (Nh, 3) float32, ``h_off`` (n_h+1,) int64 CSR boundaries,
        ``h_scale`` (n_h,) int32 scale index, ``h_len`` (n_h,) int32 chain length,
        ``h_mod`` (Nh,) float32 per-point ``|W|``.
      VERTICAL (:meth:`PythonWTMMBackend.chains2d` output, concatenated):
        ``v_xyz`` (Nv, 3) float32, ``v_off`` (n_v+1,) int64, ``v_persist`` (n_v,)
        int32 (= ``len(chain["mod"])``), ``v_scale`` (Nv,) int32 per-point scale
        index (``0..k-1`` within each chain, concatenated), ``v_mod`` (Nv,) float32
        per-point ``|W|``.

    Coordinates are frame units, NOT projected scene coordinates: ``xyz[:, 0] =
    field.x_axis[x]``, ``xyz[:, 1] = field.y_axis[y]`` (lon/lat for a
    :class:`~dynamix.core.frames.GeographicFrame` field, x/y for a
    :class:`~dynamix.core.frames.LocalFrame` one) and ``xyz[:, 2] = 0.0`` always in Phase
    A -- the legacy ``elev_m`` relief convention only ever applied to
    ``topo_wtmm``'s relief exports and has no Phase-A equivalent (a raster field's
    VALUES are a separate scalar layer, not a z-coordinate).

    Parameters
    ----------
    chains : list[dict]
        :meth:`PythonWTMMBackend.chains2d` output.
    extrema : list[dict]
        Per-scale :meth:`PythonWTMMBackend.extrema2d` output (index-aligned with
        ``scales``); H chains are built from each entry's ``"line_id"`` components.
    scales : sequence of float
        Same scales used to produce ``extrema``/``chains``.
    field : dynamix.core.rasterfield.RasterField
        Supplies ``x_axis``/``y_axis`` (pixel -> frame-unit coordinate lookup),
        ``frame`` (for ``frame_kind``/``frame_meta``) and ``name`` (-> ``region``).
    params : dict
        JSON-safe WTMM run parameters (provenance); round-trips through
        :func:`load_chains_npz` as ``te["params"]``.

    Returns
    -------
    dict
        ``{"path": Path(path), "n_horizontal": int, "n_vertical": int, "n_scales": int}``.
    """
    from dynamix.core.frames import frame_to_meta

    scales = np.asarray(scales, dtype=np.float64)
    x_axis = np.asarray(field.x_axis, dtype=np.float64)
    y_axis = np.asarray(field.y_axis, dtype=np.float64)
    ny, nx = field.ny, field.nx

    # --- H chains: within-scale maxima lines, ordered along themselves --------
    h_pts, h_off, h_scale, h_len, h_mod = [], [0], [], [], []
    for si, e in enumerate(extrema):
        x = np.asarray(e["x"], dtype=np.int64)
        y = np.asarray(e["y"], dtype=np.int64)
        if x.size == 0:
            continue
        mod = np.asarray(e["mod"], dtype=np.float64)
        line_id = np.asarray(e["line_id"], dtype=np.int64)
        grid = np.full(ny * nx, -1, dtype=np.int64)
        grid[y * nx + x] = np.arange(x.size, dtype=np.int64)
        order, seg, _starts = _order_lines(x, y, line_id, grid, nx, ny)
        if not order.size:
            continue
        bounds = np.nonzero(np.diff(seg))[0] + 1
        # Materialization: one iteration per maxima-LINE SEGMENT (few), not per
        # point -- everything per-point (`x[idx]`, `x_axis[xs]`, ...) is a vector op.
        for idx in np.split(order, bounds):
            if idx.size < 2:            # a lone point is not a polyline
                continue
            xs, ys = x[idx], y[idx]
            h_pts.append(np.column_stack(
                [x_axis[xs], y_axis[ys], np.zeros(xs.size)]))
            h_off.append(h_off[-1] + int(idx.size))
            h_scale.append(si)
            h_len.append(int(idx.size))
            h_mod.append(mod[idx])

    # --- V chains: chains2d output, concatenated -------------------------------
    v_pts, v_off, v_persist, v_scale, v_mod = [], [0], [], [], []
    for ch in chains:                   # materialization: one iteration per CHAIN (few)
        cx = np.asarray(ch["x"], dtype=np.int64)
        cy = np.asarray(ch["y"], dtype=np.int64)
        cmod = np.asarray(ch["mod"], dtype=np.float64)
        k = cx.size
        v_pts.append(np.column_stack([x_axis[cx], y_axis[cy], np.zeros(k)]))
        v_off.append(v_off[-1] + k)
        v_persist.append(int(cmod.size))
        v_scale.append(np.arange(k, dtype=np.int32))
        v_mod.append(cmod)

    h_xyz = (np.concatenate(h_pts) if h_pts else np.zeros((0, 3))).astype(np.float32)
    h_mod_arr = (np.concatenate(h_mod) if h_mod else np.zeros(0)).astype(np.float32)
    v_xyz = (np.concatenate(v_pts) if v_pts else np.zeros((0, 3))).astype(np.float32)
    v_scale_arr = (np.concatenate(v_scale) if v_scale
                   else np.zeros(0, dtype=np.int32)).astype(np.int32)
    v_mod_arr = (np.concatenate(v_mod) if v_mod else np.zeros(0)).astype(np.float32)

    np.savez_compressed(
        path,
        h_xyz=h_xyz, h_off=np.asarray(h_off, np.int64),
        h_scale=np.asarray(h_scale, np.int32), h_len=np.asarray(h_len, np.int32),
        h_mod=h_mod_arr,
        v_xyz=v_xyz, v_off=np.asarray(v_off, np.int64),
        v_persist=np.asarray(v_persist, np.int32), v_scale=v_scale_arr,
        v_mod=v_mod_arr,
        schema_version=3,
        frame_kind=field.frame.kind,
        frame_meta=json.dumps(frame_to_meta(field.frame)),
        params=json.dumps(params),
        scales=scales, n_scales=len(scales),
        region=str(field.name),
    )
    return dict(path=Path(path), n_horizontal=len(h_scale), n_vertical=len(v_persist),
                n_scales=len(scales))


def load_chains_npz(path) -> dict:
    """Load a WTMM chain ``.npz``, transparently handling both schema v2 (legacy
    ``topo_wtmm.export_extrema``) and v3 (:func:`export_chains_npz`) files.

    Files with no ``schema_version`` key, or ``schema_version < 3``, are legacy v2
    (or older) and are delegated to :func:`dynamix.core.reflayers.load_topo_extrema`
    VERBATIM -- that loader (and its ``depth_km = -elev_m / 1000`` relief
    convention) is untouched and must keep loading existing topo ``.npz`` files
    byte-for-byte identically; this function returns EXACTLY its dict, no extra
    keys added.

    v3 files get the same CSR unpack (producing ``h_segments``/``h_scale``/
    ``h_len``/``v_segments``/``v_persist`` + the per-point ``v_scale_segments``,
    ``h_mod_segments``, ``v_mod_segments``) but WITHOUT any ``-z/1000`` depth
    flip -- a v3 z column is always 0.0 (see :func:`export_chains_npz`), for both
    local and geographic frames, so unlike the v2 path there is no elevation to
    convert. Segments with fewer than 2 points are dropped, consistently with
    ``load_topo_extrema`` (and the per-chain attributes stay aligned to the kept
    segments). Also attaches ``"frame"`` (:func:`dynamix.core.frames.frame_from_meta`
    of the stored ``frame_meta``) and ``"params"`` (``json.loads`` of the stored
    run-parameter provenance).

    Returns
    -------
    dict
        v2: exactly ``load_topo_extrema``'s return value. v3: ``{"h_segments",
        "h_scale", "h_len", "h_mod_segments", "v_segments", "v_persist",
        "v_scale_segments", "v_mod_segments", "n_scales", "scales", "region",
        "frame", "params"}``.
    """
    d = np.load(path, allow_pickle=True)
    if "schema_version" not in d.files or int(d["schema_version"]) < 3:
        from dynamix.core.extrema_io import load_topo_extrema
        return load_topo_extrema(path)

    from dynamix.core.frames import frame_from_meta

    def _unpack(xyz, off, chain_attrs, point_attrs):
        """CSR unpack: chain_attrs are per-chain arrays sliced by index i; point_attrs
        are per-point arrays sliced [a:b]. Drops chains with < 2 points."""
        segs = []
        kept_chain = [[] for _ in chain_attrs]
        kept_point = [[] for _ in point_attrs]
        for i in range(len(off) - 1):
            a, b = int(off[i]), int(off[i + 1])
            s = xyz[a:b]
            if len(s) < 2:
                continue
            segs.append(np.asarray(s, dtype=np.float64))
            for j, at in enumerate(chain_attrs):
                kept_chain[j].append(at[i])
            for j, at in enumerate(point_attrs):
                kept_point[j].append(np.asarray(at[a:b]))
        return segs, [np.asarray(k) for k in kept_chain], kept_point

    h_segs, (h_scale, h_len), (h_mod_segs,) = _unpack(
        d["h_xyz"], d["h_off"], [d["h_scale"], d["h_len"]], [d["h_mod"]])
    v_segs, (v_persist,), (v_scale_segs, v_mod_segs) = _unpack(
        d["v_xyz"], d["v_off"], [d["v_persist"]], [d["v_scale"], d["v_mod"]])

    return dict(
        h_segments=h_segs, h_scale=h_scale, h_len=h_len, h_mod_segments=h_mod_segs,
        v_segments=v_segs, v_persist=v_persist, v_scale_segments=v_scale_segs,
        v_mod_segments=v_mod_segs,
        n_scales=int(d["n_scales"]), scales=np.asarray(d["scales"]),
        region=str(d["region"]),
        frame=frame_from_meta(json.loads(str(d["frame_meta"]))),
        params=json.loads(str(d["params"])),
    )


# ---------------------------------------------------------------------------
# run_wtmm2d: staged pipeline orchestrator
#
# Pure, headless driver: cwt2d -> extrema2d -> chains2d -> partition2d (scalar
# path) or tensor2d -> partition2d (tensor path). Every heavy dependency stays
# behind the methods it calls (PythonWTMMBackend, export_chains_npz) -- this
# section itself only ever touches numpy/hashlib/json/pathlib, so importing
# dynamix.core.wtmm_backend remains safe on the pure-Python-wheel path.
# ---------------------------------------------------------------------------

_WTMM2D_PARAM_DEFAULTS = {
    "mode": "scalar",
    "svd_mode": "sigma_max",
    "wavelet": "gaussian",
    "a_min": 1.0,
    "n_oct": 4,
    "n_voice": 6,
    "thresh": 1e-3,
    "similitude": 0.8,
    # The similitude band per voice, one step of the ladder (xsmurf), or per octave (``chainable_h``).
    "band_per": "voice",
    "box_ratio": 1.0,
    "dist2_max": 50.0,
    # ``ssm -smooth``'s dip filling along each maxima line before the single-max test: an xsmurf
    # option its WTMM scripts never pass, so off here too.
    "smooth": False,
    "min_chain_len": 2,
    # The chainer (dynamix.core.chaining): the benchmark by default, so the chains stage's key is
    # what it was for every recipe that names none.
    "chainer": "xsmurf_bench",
    # Spec 14's chaining options, off by default (xsmurf's behaviour); see chaining.OPTION_DEFAULTS.
    **chaining.OPTION_DEFAULTS,
    "q_list": DEFAULT_Q,
    "pad": 32,
    # What the FFT torus holds beyond the field (PAD_MODES): mirrored by default, as it always was.
    "pad_mode": "reflect",
    # A Periodic extension output's real block, (rows, cols) (spec 25 §3.1): no maxima in the filled
    # gap, and the threshold from the real data. None: the whole field is data.
    "real_block": None,
    # Where that block starts: the origin, or the centred layout's offset; (0, 0) keeps the keys.
    "real_origin": (0, 0),
    # The nms detector's tie rule (TIE_RULES): xsmurf's asymmetric test; ``keep_both`` is the rule
    # before it, whose stage keys it keeps.
    "tie_rule": "asymmetric",
    # The Border cut (spec 25 §3.3; dynamix.core.edges.border_widths): off here, so a direct caller's
    # run is unchanged; the WTMM tool sets its own default.
    "border_cut": "none",
    "border_pct": 0.72,
    "border_k": 2.0,
    "border_scope": "largest",
    "edge_eps": 1e-3,
    # No pseudo-fractional integration unless asked: with eta = 0 a chain's log2-modulus slope is
    # its Hölder exponent itself; eta > 0 reads h + eta, which the spectrum fit undoes.
    "fracint_alpha": 0.0,
    # Subpixel/parity knob: parabolic refinement of each NMS maximum along its gradient
    # direction (dynamix.core.subpixel). Off by default -- byte-identical pipeline.
    # Inert under detector="follow", whose crossing offset and crossing-interpolated modulus
    # ARE the interpolation (the k_edge inert-knob precedent; still keyed).
    "interpolate": False,
    # The detection method -- "nms" (bilinear non-maxima suppression, the xsmurf
    # wtmm2d/Malandain path) or "follow" (kappa zero-crossing with the kappa' < 0 gate,
    # xsmurf's scalar-2D historical default; see dynamix.core.follow2d for the ported
    # gkapa/gkapap formulas and the discretization).
    "detector": "nms",
    "use_wavelet_hessian": True,
    # The registry smoother's parameters (dynamix.core.smoothers: {"alpha": ...} for
    # frac_bspline, {"q": ...} for q_gaussian); None for the Gaussian pair, which leaves the
    # pair's cache keys as they were.
    "smoother_params": None,
}

_WTMM2D_CACHE_DIRNAME = "wtmm_cache"


def _resolve_wtmm2d_params(params: dict) -> dict:
    """Merge ``params`` onto :data:`_WTMM2D_PARAM_DEFAULTS`, rejecting typos.

    Any key not in the default set raises ``ValueError`` naming it -- this is
    the ONLY guard against a silently-ignored misspelled parameter (e.g.
    ``n_octaves`` instead of ``n_oct``).
    """
    unknown = sorted(set(params) - set(_WTMM2D_PARAM_DEFAULTS))
    if unknown:
        raise ValueError(
            f"run_wtmm2d: unknown params key(s) {unknown}; "
            f"expected a subset of {sorted(_WTMM2D_PARAM_DEFAULTS)}"
        )
    resolved = dict(_WTMM2D_PARAM_DEFAULTS)
    resolved.update(params)
    return resolved


def _field_data_hash(field) -> str:
    """sha1 over the field's values + x_axis + y_axis bytes (cache-key input)."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(field.values).tobytes())
    h.update(np.ascontiguousarray(field.x_axis).tobytes())
    h.update(np.ascontiguousarray(field.y_axis).tobytes())
    return h.hexdigest()


def _hash_key_dict(key_dict: dict) -> str:
    """Deterministic sha1 hex digest of a stage's relevant resolved params.

    ``sorted(key_dict.items())`` sorts by (unique, string) key, so array-typed
    values (``q_list``) are never compared against each other during the
    sort. Each value is then fed into the digest as its EXACT bytes rather
    than via ``repr()``: ``repr()`` of an ndarray is a lossy DISPLAY encoding
    -- numpy silently summarizes arrays over 1000 elements (``...`` in the
    middle) and caps float precision at 8 digits, so two genuinely different
    large ``q_list`` arrays can ``repr()`` identically and therefore hash
    identically, corrupting the cache (a stage silently reports a hit and
    returns the WRONG array's cached result). ``tobytes() + dtype + shape``
    has none of that summarization/truncation.
    """
    h = hashlib.sha1()
    for k, v in sorted(key_dict.items()):
        h.update(repr(k).encode())
        h.update(b"=")
        if isinstance(v, np.ndarray):
            h.update(v.tobytes())
            h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode())
        else:
            h.update(repr(v).encode())
        h.update(b";")
    return h.hexdigest()


def _stage_cache_path(out_dir, field_name, stage: str, key_dict: dict) -> Path:
    key_hash = _hash_key_dict(key_dict)
    return (Path(out_dir) / _WTMM2D_CACHE_DIRNAME / str(field_name)
            / f"{stage}-{key_hash[:12]}.npz")


def _atomic_save(path: Path, save_fn, result) -> None:
    """Write a stage cache file atomically: ``save_fn`` writes to a sibling
    temp path (name still ``*.npz`` so ``np.savez_compressed`` doesn't quietly
    append its own ``.npz`` suffix onto something else), then ``os.replace``
    swaps it onto ``path`` in one filesystem op. Without this, a process
    killed mid-``savez_compressed`` leaves a truncated file AT the final
    cache path, which ``path.exists()`` accepts forever and every later
    ``run_wtmm2d`` call dies loading it. ``os``/``uuid`` are imported here,
    lazily, to keep this module's top-level imports at numpy/json/hashlib/
    pathlib/warnings only.
    """
    import os
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp-{uuid.uuid4().hex[:8]}{path.suffix}")
    try:
        save_fn(tmp, result)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _run_stage(stage: str, key_dict: dict, out_dir, field_name: str, compute_fn,
               save_fn, load_fn, cache_hits: set, progress) -> object:
    """Run one cacheable pipeline stage: load-on-hit, else compute + save.

    ``progress(stage, 0.0)`` fires immediately before ``compute_fn()`` (never
    on a cache hit); ``progress(stage, 1.0)`` fires after either a hit or a
    completed compute. ``cache_hits`` gains ``stage`` only on a hit.

    A cache file that exists but fails to load (truncated/corrupt -- e.g. left
    behind by a killed process before :func:`_atomic_save` existed, or any
    other on-disk damage) is treated exactly like a miss: silently recomputed,
    and the cache file is atomically rewritten below, so a corrupt cache
    self-heals on the very next call instead of wedging every future run
    behind a permanent load error.
    """
    path = None
    if out_dir is not None:
        path = _stage_cache_path(out_dir, field_name, stage, key_dict)
        if path.exists():
            try:
                result = load_fn(path)
            except Exception:
                pass
            else:
                cache_hits.add(stage)
                if progress is not None:
                    progress(stage, 1.0)
                return result

    if progress is not None:
        progress(stage, 0.0)
    result = compute_fn()
    if path is not None:
        _atomic_save(path, save_fn, result)
    if progress is not None:
        progress(stage, 1.0)
    return result


# --- per-stage npz (de)serialization ----------------------------------------

def _save_cwt(path, cwt: dict) -> None:
    """``kapa``/``kapap`` (the follow detector's fields) ride along when present --
    the follow variant of the cwt stage caches four stacks instead of two."""
    arrays = {"mod": np.asarray(cwt["mod"]), "arg": np.asarray(cwt["arg"])}
    if "kapa" in cwt:
        arrays["kapa"] = np.asarray(cwt["kapa"])
        arrays["kapap"] = np.asarray(cwt["kapap"])
    np.savez_compressed(path, **arrays)


def _load_cwt(path) -> dict:
    d = np.load(path)
    out = {"mod": d["mod"], "arg": d["arg"]}
    if "kapa" in d:
        out["kapa"] = d["kapa"]
        out["kapap"] = d["kapap"]
    return out


def _extrema_to_arrays(extrema: list) -> dict:
    """Per-scale extrema list -> CSR-flattened arrays (x/y/mod/arg/line_id + offsets).

    ``x_sub``/``y_sub`` (the subpixel channels) are flattened too when present --
    present on every layer or none (the refinement is a whole-stack pass), so presence on the
    first layer decides."""
    lens = [len(e["x"]) for e in extrema]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    def cat(key, dtype):
        if extrema:
            return np.concatenate([np.asarray(e[key], dtype=dtype) for e in extrema])
        return np.zeros(0, dtype=dtype)

    out = {
        "x": cat("x", np.int64), "y": cat("y", np.int64),
        "mod": cat("mod", np.float64), "arg": cat("arg", np.float64),
        "line_id": cat("line_id", np.int64), "off": off,
    }
    if extrema and "x_sub" in extrema[0]:
        out["x_sub"] = cat("x_sub", np.float64)
        out["y_sub"] = cat("y_sub", np.float64)
    return out


def _arrays_to_extrema(d) -> list:
    """Inverse of :func:`_extrema_to_arrays`; ``d`` is any ``{"x", "y", ...}`` mapping.

    ``x_sub``/``y_sub`` come back when the arrays carry them; a stage-cache npz without the
    subpixel channels loads without them -- npz mappings support ``in`` and this never
    KeyErrors on their absence."""
    off = np.asarray(d["off"])
    x, y, mod, arg, line_id = d["x"], d["y"], d["mod"], d["arg"], d["line_id"]
    has_sub = "x_sub" in d
    out = []
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1])
        layer = {"x": np.asarray(x[a:b], dtype=np.int64),
                 "y": np.asarray(y[a:b], dtype=np.int64),
                 "mod": np.asarray(mod[a:b], dtype=np.float64),
                 "arg": np.asarray(arg[a:b], dtype=np.float64),
                 "line_id": np.asarray(line_id[a:b], dtype=np.int64)}
        if has_sub:
            layer["x_sub"] = np.asarray(d["x_sub"][a:b], dtype=np.float64)
            layer["y_sub"] = np.asarray(d["y_sub"][a:b], dtype=np.float64)
        out.append(layer)
    return out


def _save_extrema(path, extrema: list) -> None:
    np.savez_compressed(path, **_extrema_to_arrays(extrema))


def _load_extrema(path) -> list:
    d = np.load(path)
    return _arrays_to_extrema(d)


def _chains_to_arrays(chains: list) -> dict:
    """Chain list -> CSR (x/y/mod flat + offsets); ``log2_mod``/``log2_scales`` are
    NOT stored -- they are pure functions of ``mod``/``scales`` and are rebuilt by
    :func:`_arrays_to_chains` on load."""
    lens = [len(c["x"]) for c in chains]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    def cat(key, dtype):
        if chains:
            return np.concatenate([np.asarray(c[key], dtype=dtype) for c in chains])
        return np.zeros(0, dtype=dtype)

    return {"x": cat("x", np.int64), "y": cat("y", np.int64),
            "mod": cat("mod", np.float64), "off": off}


def _arrays_to_chains(d, scales) -> list:
    """Inverse of :func:`_chains_to_arrays`; ``d`` is any ``{"x", "y", "mod", "off"}``
    mapping. Rebuilds ``log2_mod = log2(|mod|)`` and ``log2_scales = log2(scales)[:k]``
    per chain, exactly as :meth:`PythonWTMMBackend.chains2d` computes them."""
    off = np.asarray(d["off"])
    x, y, mod = d["x"], d["y"], np.asarray(d["mod"], dtype=np.float64)
    log2_scales_full = np.log2(np.asarray(scales, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_mod_full = np.log2(np.abs(mod))
    out = []
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1])
        k = b - a
        out.append({
            "x": np.asarray(x[a:b], dtype=np.int64), "y": np.asarray(y[a:b], dtype=np.int64),
            "mod": mod[a:b].copy(),
            "log2_mod": log2_mod_full[a:b].copy(),
            "log2_scales": log2_scales_full[:k].copy(),
        })
    return out


def _save_chains(path, chains: list) -> None:
    np.savez_compressed(path, **_chains_to_arrays(chains))


def _load_chains(path, scales) -> list:
    d = np.load(path)
    return _arrays_to_chains(d, scales)


def _partition_to_arrays(hd_std: dict, hd_cmax: dict) -> dict:
    """The two ``build_hd_from_chains`` dicts' arrays, "std_"/"cmax_"-prefixed."""
    data = {}
    for k, v in hd_std.items():
        data[f"std_{k}"] = np.asarray(v)
    for k, v in hd_cmax.items():
        data[f"cmax_{k}"] = np.asarray(v)
    return data


def _arrays_to_partition(d) -> tuple:
    hd_std, hd_cmax = {}, {}
    keys = d.files if hasattr(d, "files") else list(d.keys())
    for key in keys:
        if key.startswith("std_"):
            hd_std[key[4:]] = d[key]
        elif key.startswith("cmax_"):
            hd_cmax[key[5:]] = d[key]
    return hd_std, hd_cmax


def _save_partition(path, result: tuple) -> None:
    hd_std, hd_cmax = result
    np.savez_compressed(path, **_partition_to_arrays(hd_std, hd_cmax))


def _load_partition(path) -> tuple:
    d = np.load(path)
    return _arrays_to_partition(d)


def _save_tensor_stage(path, result: dict) -> None:
    """The tensor run-level cache product: chains CSR + extrema CSR + hd arrays."""
    data = {}
    for k, v in _chains_to_arrays(result["chains"]).items():
        data[f"chain_{k}"] = v
    for k, v in _extrema_to_arrays(result["extrema"]).items():
        data[f"extrema_{k}"] = v
    data.update(_partition_to_arrays(result["hd_std"], result["hd_cmax"]))
    np.savez_compressed(path, **data)


def _load_tensor_stage(path, scales) -> dict:
    d = np.load(path)
    chain_arrays = {k[len("chain_"):]: d[k] for k in d.files if k.startswith("chain_")}
    extrema_arrays = {k[len("extrema_"):]: d[k] for k in d.files if k.startswith("extrema_")}
    hd_std, hd_cmax = _arrays_to_partition(d)
    return {
        "chains": _arrays_to_chains(chain_arrays, scales),
        "extrema": _arrays_to_extrema(extrema_arrays),
        "hd_std": hd_std, "hd_cmax": hd_cmax,
    }


# --- pipeline drivers --------------------------------------------------------

def _apply_fracint2d(cwt: dict, scales, fracint_alpha: float) -> dict:
    """Pseudo-fractional integration of order eta on the scalar 2D transform (Wendt 2009).

    The per-scale ``a**fracint_alpha`` lift the reference tensor path applies to its WT
    derivatives (``wtmm_ebsd/twtmm.py:154-165``: ``dx, dy *= a**alpha``), applied here to the
    scalar transform's modulus instead -- the identical operation, since a positive per-scale
    scalar gives ``|grad| *= a**alpha`` with the gradient angle untouched (the lift is available on BOTH paths, not only tensor). Distinct from the 1D
    path's in-CWT ``a**expo`` normalization (``expo=-1.0``, LastWave) -- same ``a^power`` family,
    different stage and sign convention; never conflate or double-apply them.

    ``alpha == 0`` returns the input dict itself (the reference's own ``!= 0`` guard). Otherwise
    a NEW ``mod`` array is built -- the input is never mutated, because the cached cwt stage may
    own it -- and ``arg`` passes through as the same object.
    """
    if fracint_alpha == 0:
        return cwt
    mod = np.asarray(cwt["mod"])
    lifted = np.empty_like(mod)
    for si, a in enumerate(scales):
        np.multiply(mod[si], np.float32(float(a) ** fracint_alpha), out=lifted[si])
    return dict(cwt, mod=lifted)


def _run_wtmm2d_scalar(field, resolved: dict, *, out_dir, backend, progress,
                       cancel=None, stage_cache=None) -> dict:
    backend = backend if backend is not None else get_backend("python")
    field_hash = _field_data_hash(field)
    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    cache_hits: set = set()

    def _ck():
        # Cooperative cancellation (progressive-compute design §2b): checked BEFORE each stage's
        # compute+save, so a set flag raises before any work or any cache write for that stage.
        if cancel is not None and cancel():
            raise ComputeCancelled()

    def run_stage(stage, key_dict, out_dir, field_name, compute_fn,
                  save_fn, load_fn, cache_hits, progress):
        # The engine owns the store. The backend/precision are part of this in-memory
        # identity even though the historical on-disk format predates those settings.
        import copy

        memory_key = None
        if stage_cache is not None:
            from dynamix.core import fft_policy

            policy = fft_policy.active()
            memory_key = "wtmm2d:" + stage + ":" + _hash_key_dict({
                **key_dict, "engine": policy.name, "precision": policy.precision,
                "backend": type(backend).__module__ + "." + type(backend).__qualname__,
            })
            if memory_key in stage_cache:
                result = stage_cache[memory_key]
                cache_hits.add(stage)
                if progress is not None:
                    progress(stage, 1.0)
                # Line ordering and output annotations can mutate maxima/chains. Keep
                # their saved stages isolated; the large CWT arrays are only read.
                return result if stage == "cwt" else copy.deepcopy(result)

        def checked_compute():
            result = compute_fn()
            _ck()
            return result

        from dynamix.core import fft_policy
        with fft_policy.observing() as stage_context:
            result = _run_stage(stage, key_dict, out_dir, field_name, checked_compute,
                                save_fn, load_fn, cache_hits, progress)
        _ck()
        if memory_key is not None:
            stage_cache[memory_key] = result if stage == "cwt" else copy.deepcopy(result)
            if hasattr(stage_cache, "set_context"):
                stage_cache.set_context(memory_key, stage_context)
        return result

    follow = resolved["detector"] == "follow"
    cwt_key = {
        "field_hash": field_hash, "wavelet": resolved["wavelet"],
        "a_min": resolved["a_min"], "n_oct": resolved["n_oct"],
        "n_voice": resolved["n_voice"], "pad": resolved["pad"],
    }
    if resolved["pad_mode"] != "reflect":         # reflect keeps the stage keys it always had
        cwt_key["pad_mode"] = resolved["pad_mode"]
    if resolved["smoother_params"] is not None:
        cwt_key["smoother_params"] = tuple(sorted(resolved["smoother_params"].items()))
    if follow:
        # The follow variant carries kappa/kappa' alongside mod/arg -- a DIFFERENT stage
        # content, so it must never collide with a plain-nms cwt cache entry.
        cwt_key = dict(cwt_key, variant="follow")

    def compute_cwt():
        if not follow:
            return backend.cwt2d(field.values, scales, wavelet=resolved["wavelet"],
                                 pad=resolved["pad"], pad_mode=resolved["pad_mode"],
                                 smoother_params=resolved["smoother_params"])
        # Follow needs the nine derivative stacks per scale. One engine call per scale keeps
        # the transient footprint at eleven single-scale rasters (all-scales derivs='all' on a
        # 4096^2 field would be ~GBs); kappa/kappa' come out of one fused numba pass and the
        # raw derivatives are dropped immediately (xsmurf's own gkapa-then-delete order).
        from dynamix.core.follow2d import kapa_fields
        n_sc = len(scales)
        mod = arg = kapa = kapap = None
        for si, a in enumerate(scales):
            one = backend.cwt2d(field.values, [a], wavelet=resolved["wavelet"],
                                pad=resolved["pad"], pad_mode=resolved["pad_mode"], derivs="all",
                                smoother_params=resolved["smoother_params"])
            if mod is None:
                ny, nx = one["mod"].shape[1:]
                mod = np.empty((n_sc, ny, nx), dtype=one["mod"].dtype)
                arg = np.empty_like(mod)
                kapa = np.empty_like(mod)
                kapap = np.empty_like(mod)
            mod[si] = one["mod"][0]
            arg[si] = one["arg"][0]
            kapa[si], kapap[si] = kapa_fields(one, 0)
            if progress is not None:
                progress(f"cwt+kappa {si + 1}/{n_sc}", (si + 1) / n_sc)
        return {"mod": mod, "arg": arg, "kapa": kapa, "kapap": kapap}

    _ck()
    cwt = run_stage("cwt", cwt_key, out_dir, field.name, compute_cwt,
                     _save_cwt, _load_cwt, cache_hits, progress)

    # Applied AFTER the cached cwt stage, so the expensive transform stays reusable across
    # fracint_alpha values; every downstream stage key carries fracint_alpha from here on
    # (lifted moduli change extrema VALUES and the similitude chaining band, not positions).
    cwt = _apply_fracint2d(cwt, scales, resolved["fracint_alpha"])

    extrema_key = dict(cwt_key, thresh=resolved["thresh"],
                       fracint_alpha=resolved["fracint_alpha"],
                       interpolate=resolved["interpolate"],
                       detector=resolved["detector"])
    if resolved["detector"] == "nms" and resolved["tie_rule"] != "keep_both":
        extrema_key["tie_rule"] = resolved["tie_rule"]
    real_block = resolved["real_block"]
    real_origin = tuple(int(v) for v in (resolved.get("real_origin") or (0, 0)))
    if real_block is not None:                    # the whole field as data keeps its stage keys
        real_block = (int(real_block[0]), int(real_block[1]))
        extrema_key["real_block"] = real_block
        if real_origin != (0, 0):                 # the block at the origin keeps its stage keys
            extrema_key["real_origin"] = real_origin

    def compute_extrema():
        if follow:
            # kappa zero-crossings with the kappa' < 0 gate (dynamix.core.follow2d). The
            # LIFTED mod supplies values (fracint is a per-scale positive scalar: kappa's
            # crossings are invariant under it, so the raw kappa stacks decide WHERE and the
            # lifted modulus supplies the amplitudes -- the exact division of labor the NMS
            # path has). Follow's crossing offset and crossing-interpolated modulus are its
            # native channels, so the interpolate knob is inert here (keyed regardless).
            # Exact port: dynamix.core.xsmurf_follow -- w2_folow_contour detection +
            # _get_interpolated_modulus_ v1 values + search_lines chaining, parity-proven
            # against the wrapper oracle (test_xsmurf_follow_parity). The follow2d
            # discretization stays in core, unused by this path.
            from dynamix.core.xsmurf_follow import follow_extrema_scale_exact
            invalid = None
            fvals = np.asarray(field.values)
            if not np.all(np.isfinite(fvals)):
                invalid = ~np.isfinite(fvals)
            found = [
                follow_extrema_scale_exact(cwt["mod"][si], cwt["arg"][si],
                                           cwt["kapa"][si], cwt["kapap"][si],
                                           thresh=resolved["thresh"], invalid=invalid,
                                           radius=int(np.ceil(scales[si])))
                for si in range(len(scales))
            ]
            if real_block is not None:
                # Follow thresholds inside its exact port, against the whole array; the gap's
                # maxima are dropped here, and their line runs index the uncut points.
                from dynamix.core import edges as _edges
                found = _edges.drop_outside(found, real_block, real_origin)
            return found
        ext = backend.extrema2d(cwt, scales, thresh=resolved["thresh"], field=field.values,
                                real_block=real_block, real_origin=real_origin,
                                tie_rule=resolved["tie_rule"])
        if resolved["interpolate"]:
            # Parabolic refinement along the gradient (dynamix.core.subpixel): the refined
            # MODULUS replaces ``mod`` -- it feeds chaining and Z(q,a), the xsmurf follow /
            # LastWave-1D value channel -- and float ``x_sub``/``y_sub`` ride alongside the
            # integer support, which stays untouched (M-Z reconstruction is pixel-exact).
            # Runs on the LIFTED mod stack -- the same raster the NMS itself ran on.
            from dynamix.core.subpixel import refine_extrema_stack
            ext = refine_extrema_stack(ext, cwt["mod"])
        return ext

    _ck()
    extrema = run_stage("extrema", extrema_key, out_dir, field.name, compute_extrema,
                        _save_extrema, _load_extrema, cache_hits, progress)

    # The Border cut (spec 25 §3.3), in xsmurf's order: each scale's maxima inside its border go
    # before the V-chains link them (study.tcl: follow, rm_ext, then vchain), so a border maximum
    # never competes for a link. The stored extrema stage stays the uncut one.
    border_info = None
    if resolved["border_cut"] != "none":
        from dynamix.core import edges as _edges
        shape2 = tuple(field.values.shape)[:2]
        borders = _edges.border_widths(
            list(scales), resolved["border_cut"], wavelet=resolved["wavelet"],
            smoother_params=resolved["smoother_params"], eps=resolved["edge_eps"],
            pct=resolved["border_pct"], k=resolved["border_k"], scope=resolved["border_scope"],
            run_grid=max(shape2) + 2 * int(resolved["pad"]))
        extrema = _edges.drop_in_cone(extrema, borders, (0, 0), shape2)
        for level in extrema:            # the follow detector's line runs index the uncut points
            level.pop("_xs_runs", None), level.pop("_xs_closed", None)
        border_info = {"border_cut": resolved["border_cut"], "border_scope": resolved["border_scope"],
                       "borders": borders}

    chains_key = dict(extrema_key, similitude=resolved["similitude"],
                      box_ratio=resolved["box_ratio"], dist2_max=resolved["dist2_max"],
                      smooth=resolved["smooth"], min_chain_len=resolved["min_chain_len"])
    if border_info is not None:                   # no cut: the stage keys stay as they were
        chains_key.update(border_cut=resolved["border_cut"], border_scope=resolved["border_scope"],
                          border_pct=resolved["border_pct"], border_k=resolved["border_k"],
                          edge_eps=resolved["edge_eps"])
    if resolved["band_per"] != "voice":          # xsmurf's band keeps the stage keys it had
        chains_key["band_per"] = resolved["band_per"]
    if resolved["chainer"] != chaining.DEFAULT:
        chains_key["chainer"] = resolved["chainer"]
    for name, default in chaining.OPTION_DEFAULTS.items():   # off: the stage keys stay as they were
        if resolved[name] != default:
            chains_key[name] = resolved[name]

    def compute_chains():
        return chaining.link(extrema, scales, resolved, backend,
                             shape=tuple(field.values.shape)[:2])

    _ck()
    chains = run_stage("chains", chains_key, out_dir, field.name, compute_chains,
                        _save_chains, lambda p: _load_chains(p, scales),
                        cache_hits, progress)

    partition_key = dict(chains_key, q_list=np.asarray(resolved["q_list"]))

    def compute_partition():
        return backend.partition2d(chains, scales, q_list=resolved["q_list"],
                                   min_chain_len=resolved["min_chain_len"])

    _ck()
    hd_std, hd_cmax = run_stage(
        "partition", partition_key, out_dir, field.name, compute_partition,
        _save_partition, _load_partition, cache_hits, progress)

    npz_path = None
    if out_dir is not None:
        wtmm_dir = Path(out_dir) / "wtmm"
        wtmm_dir.mkdir(parents=True, exist_ok=True)
        npz_path = wtmm_dir / f"{field.name}.npz"
        export_chains_npz(npz_path, chains=chains, extrema=extrema, scales=scales,
                          field=field, params=_json_safe_params(resolved))

    out = {
        "chains": chains, "extrema": extrema, "scales": scales,
        "hd_std": hd_std, "hd_cmax": hd_cmax, "npz_path": npz_path,
        "params": resolved, "cache_hits": cache_hits,
    }
    if border_info is not None:
        out["_border_cut"] = border_info
    if resolved["detector"] == "follow":
        # The C's own hsearch ordering (search_lines walk) becomes the display runs --
        # closed rings carry their LINE_CLOSED flag so both views can close the polyline.
        # attach_chain_product keeps a pre-stamped _hline_runs (fills only when absent).
        from dynamix.core.xsmurf_follow import runs_for_layer
        shape = tuple(field.values.shape)[:2]
        pairs = [runs_for_layer(layer, shape) for layer in extrema]
        for layer in extrema:
            layer.pop("_xs_runs", None), layer.pop("_xs_closed", None)
        out["_hline_runs"] = [p[0] for p in pairs]
        out["_hline_closed"] = [p[1] for p in pairs]
    return out


def _run_wtmm2d_tensor(field, resolved: dict, *, out_dir, backend, progress) -> dict:
    backend = backend if backend is not None else get_backend("python")
    field_hash = _field_data_hash(field)
    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    cache_hits: set = set()

    # Run-level cache key: ALL resolved params (q_list normalized to an ndarray
    # for a deterministic repr) plus the field-data hash.
    tensor_key = dict(resolved)
    tensor_key["q_list"] = np.asarray(resolved["q_list"])
    tensor_key["field_hash"] = field_hash

    def compute_tensor():
        svd_mode = resolved["svd_mode"]
        tensor_out = backend.tensor2d(
            field.values, scales, svd_modes=(svd_mode,),
            wavelet=resolved["wavelet"],
            use_wavelet_hessian=resolved["use_wavelet_hessian"],
            fracint_alpha=resolved["fracint_alpha"], thresh=resolved["thresh"],
            pad=resolved["pad"], similitude=resolved["similitude"],
            box_ratio=resolved["box_ratio"], dist2_max=resolved["dist2_max"],
            min_len=resolved["min_chain_len"], smooth=resolved["smooth"],
            band_per=resolved["band_per"],
        )[svd_mode]
        chains = tensor_out["chains"]
        extrema = tensor_out["extrema"]
        hd_std, hd_cmax = backend.partition2d(
            chains, scales, q_list=resolved["q_list"],
            min_chain_len=resolved["min_chain_len"])
        return {"chains": chains, "extrema": extrema, "hd_std": hd_std, "hd_cmax": hd_cmax}

    result = _run_stage("tensor", tensor_key, out_dir, field.name, compute_tensor,
                        _save_tensor_stage, lambda p: _load_tensor_stage(p, scales),
                        cache_hits, progress)

    chains, extrema = result["chains"], result["extrema"]
    hd_std, hd_cmax = result["hd_std"], result["hd_cmax"]

    npz_path = None
    if out_dir is not None:
        wtmm_dir = Path(out_dir) / "wtmm"
        wtmm_dir.mkdir(parents=True, exist_ok=True)
        npz_path = wtmm_dir / f"{field.name}.npz"
        export_chains_npz(npz_path, chains=chains, extrema=extrema, scales=scales,
                          field=field, params=_json_safe_params(resolved))

    return {
        "chains": chains, "extrema": extrema, "scales": scales,
        "hd_std": hd_std, "hd_cmax": hd_cmax, "npz_path": npz_path,
        "params": resolved, "cache_hits": cache_hits,
    }


def _json_safe_params(resolved: dict) -> dict:
    """A copy of ``resolved`` fit for ``export_chains_npz``'s ``json.dumps`` --
    only ``q_list`` (an ndarray internally, for :meth:`PythonWTMMBackend.partition2d`)
    needs converting, to a plain ``list``."""
    out = dict(resolved)
    out["q_list"] = np.asarray(resolved["q_list"]).tolist()
    return out


def run_wtmm2d(field, params: dict, *, out_dir=None, backend=None, progress=None,
               cancel=None, stage_cache=None) -> dict:
    """Staged pipeline orchestrator (pure, headless): cwt -> extrema -> chains -> partition.

    Resolves ``params`` against the contract defaults (see
    :data:`_WTMM2D_PARAM_DEFAULTS`) -- ``mode="scalar"``, ``svd_mode="sigma_max"``,
    ``wavelet="gaussian"``, ``a_min=1.0``, ``n_oct=4``, ``n_voice=6``,
    ``thresh=1e-3``, ``similitude=0.8``, ``box_ratio=1.0``, ``dist2_max=50.0``,
    ``smooth=True``, ``min_chain_len=2``, ``q_list=DEFAULT_Q``, ``pad=32``,
    ``fracint_alpha=0.0``, ``use_wavelet_hessian=True`` -- and raises ``ValueError``
    naming any key NOT in that set (a typo guard).

    ``mode="tensor"`` requires ``field.n_components == 3`` (``ValueError``
    otherwise) and delegates to :meth:`PythonWTMMBackend.tensor2d` for the one
    requested ``svd_mode``, then partitions its chains; because
    ``alpha_jacobian_twtmm`` is monolithic, this path caches at RUN level (one
    key over every resolved param + the field-data hash, stage name
    ``"tensor"``). ``mode="scalar"`` requires ``field.n_components == 1`` and
    runs the four-stage pipeline (``cwt``, ``extrema``, ``chains``,
    ``partition``), each independently cached.

    Staged caching (with ``out_dir`` set)
    --------------------------------------
    Each stage saves ``out_dir/wtmm_cache/<field.name>/<stage>-<sha1(upstream
    params)[:12]>.npz`` and a later call with unchanged upstream params reloads
    it (recorded in the returned ``"cache_hits"``) instead of recomputing; a
    param edit recomputes ONLY that stage and everything downstream of it.
    Cache-key chain: ``cwt`` <- (field data hash, wavelet, a_min, n_oct,
    n_voice, pad); ``extrema`` <- cwt key + thresh; ``chains`` <- extrema key +
    similitude, box_ratio, dist2_max, smooth, min_chain_len; ``partition`` <-
    chains key + q_list. An engine-owned ``stage_cache`` also retains the scalar
    stages in memory, independently of disk export. Its keys include the active
    FFT engine and precision. With ``out_dir=None`` nothing is exported and
    ``"npz_path"`` comes back ``None``.

    Whenever ``out_dir`` is given, :func:`export_chains_npz` also always writes
    (or overwrites) ``out_dir/wtmm/<field.name>.npz`` with the resolved,
    JSON-safe params (``q_list`` as a plain list) -- this is unconditional, not
    itself cached.

    Parameters
    ----------
    field : dynamix.core.rasterfield.RasterField
    params : dict
        See the resolvable keys above; unknown keys raise ``ValueError``.
    out_dir : path-like or None
        Enables both per-stage caching (scalar) / run-level caching (tensor)
        and the chain ``.npz`` export. ``None`` disables both.
    backend : PythonWTMMBackend or None
        ``None`` -> :func:`get_backend` ``"python"``.
    progress : callable(stage: str, frac: float) or None
        Called ``(stage, 0.0)`` immediately before computing that stage (NEVER
        on a cache hit) and ``(stage, 1.0)`` after it completes or hits. The
        GUI marshals this to Qt signals; this module knows nothing about Qt.

    Returns
    -------
    dict
        ``{"chains", "extrema", "scales", "hd_std", "hd_cmax",
        "npz_path" (Path or None), "params" (resolved dict), "cache_hits"
        (set[str])}``.
    """
    resolved = _resolve_wtmm2d_params(params)
    mode = resolved["mode"]
    if mode not in ("scalar", "tensor"):
        raise ValueError(f"run_wtmm2d: unknown mode {mode!r}; expected 'scalar' or 'tensor'")

    if mode == "tensor":
        if field.n_components != 3:
            raise ValueError(
                f"run_wtmm2d: mode='tensor' requires field.n_components == 3; "
                f"got {field.n_components} (field {field.name!r})"
            )
        # Cancellation is scalar-only in v1 (the tensor path is one monolithic call with no
        # stage seam to check between); a set flag before it still raises here.
        if cancel is not None and cancel():
            raise ComputeCancelled()
        return _run_wtmm2d_tensor(field, resolved, out_dir=out_dir, backend=backend,
                                  progress=progress)

    if field.n_components != 1:
        raise ValueError(
            f"run_wtmm2d: mode='scalar' requires field.n_components == 1; "
            f"got {field.n_components} (field {field.name!r})"
        )
    return _run_wtmm2d_scalar(field, resolved, out_dir=out_dir, backend=backend,
                              progress=progress, cancel=cancel, stage_cache=stage_cache)


def run_wtmm2d_preview(field, params: dict, *, backend=None) -> dict:
    """Finest-scale-only WTMM: the instant preview for progressive compute (design §2a).

    Computes the wavelet transform and within-scale (H) maxima lines for the FINEST scale alone
    (``compute_scales2d(...)[:1]``) -- cheap, and value-identical to :func:`run_wtmm2d`'s scale-0
    layer for the same ``a_min`` (``tests/test_progressive_compute.py`` pins this). No cross-scale
    chaining (``chains``), no partition -- those need the whole stack and belong to the background
    full run. Uncached by design: it is cheap, and it must never write into the full run's stage
    cache (different stage set, same ``a_min``).

    Returns the ordinary result shape a canvas already draws (``extrema`` = 1-layer list,
    ``_hline_runs``, ``_shape``, ``_frame``, ``scales`` = finest only, ``chains`` = []), plus
    ``_preview = True`` so a consumer can label it "preview — computing full stack…".
    """
    resolved = _resolve_wtmm2d_params(params)
    if resolved["mode"] != "scalar":
        raise ValueError("run_wtmm2d_preview: scalar mode only (finest-scale H-lines)")
    if field.n_components != 1:
        raise ValueError(
            f"run_wtmm2d_preview: requires field.n_components == 1; got {field.n_components}")
    backend = backend if backend is not None else get_backend("python")
    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])[:1]
    follow = resolved["detector"] == "follow"
    cwt = backend.cwt2d(field.values, scales, wavelet=resolved["wavelet"],
                        pad=resolved["pad"], pad_mode=resolved["pad_mode"],
                        derivs="all" if follow else "first",
                        smoother_params=resolved["smoother_params"])
    if follow:
        from dynamix.core.follow2d import kapa_fields
        kapa, kapap = kapa_fields(cwt, 0)
        cwt = {"mod": cwt["mod"], "arg": cwt["arg"],
               "kapa": kapa[None], "kapap": kapap[None]}
    # Same lift as the full scalar run (_run_wtmm2d_scalar) -- the preview's value-identity
    # with the full run's scale-0 layer holds THROUGH the lift.
    cwt = _apply_fracint2d(cwt, scales, resolved["fracint_alpha"])
    if follow:
        # Same detection as the full run's extrema stage (value-identity holds THROUGH it).
        from dynamix.core.xsmurf_follow import follow_extrema_scale_exact as follow_extrema_scale
        invalid = None
        fvals = np.asarray(field.values)
        if not np.all(np.isfinite(fvals)):
            invalid = ~np.isfinite(fvals)
        extrema = [follow_extrema_scale(cwt["mod"][0], cwt["arg"][0],
                                        cwt["kapa"][0], cwt["kapap"][0],
                                        thresh=resolved["thresh"], invalid=invalid,
                                        radius=int(np.ceil(scales[0])))]
    else:
        extrema = backend.extrema2d(cwt, scales, thresh=resolved["thresh"], field=field.values,
                                    tie_rule=resolved["tie_rule"])
        if resolved["interpolate"]:
            # Same refinement as the full run's extrema stage -- the preview's value-identity
            # with the full run's scale-0 layer (test_progressive_compute) holds THROUGH it.
            from dynamix.core.subpixel import refine_extrema_stack
            extrema = refine_extrema_stack(extrema, cwt["mod"])

    from dynamix.core.hlines import hline_runs      # lazy: hlines imports _order_lines from here
    shape = tuple(field.values.shape)
    if follow:
        from dynamix.core.xsmurf_follow import runs_for_layer
        pairs = [runs_for_layer(layer, shape[:2]) for layer in extrema]
        for layer in extrema:
            layer.pop("_xs_runs", None), layer.pop("_xs_closed", None)
        runs = [p[0] for p in pairs]
        closed = [p[1] for p in pairs]
    else:
        runs = [hline_runs(layer, shape[:2]) for layer in extrema]
        closed = None
    out = {
        "chains": [], "extrema": extrema, "scales": scales,
        "hd_std": None, "hd_cmax": None, "npz_path": None,
        "params": resolved, "cache_hits": set(),
        "_frame": getattr(field, "frame", None), "_shape": shape,
        "_hline_runs": runs,
        "_preview": True,
    }
    if closed is not None:
        out["_hline_closed"] = closed
    return out
