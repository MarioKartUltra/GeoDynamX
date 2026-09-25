# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""fft_policy.py -- the app-wide FFT engine + precision POLICY (DynamiX-native, 2026-09-22).

Every FFT runs on mlx or FFTW3, never numpy by default. So:

- ``auto``: mlx on Apple Silicon (when installed), FFTW3 (pyfftw) everywhere else;
- 64-bit always runs on FFTW3 (mlx is single precision only);
- a requested engine that is not installed falls to the other;
- numpy's FFT only when NEITHER is installed, with a RuntimeWarning.

Set once at startup from ``Settings.compute_engine`` / ``Settings.compute_precision`` (changes apply after a restart) via :func:`configure`. Every FFT consumer calls :func:`active`
at call time. Engine and precision are INFRASTRUCTURE, never part of a cache key: the project's
backends-agree-to-float32 law (tests/test_fft_policy.py pins the agreement for every engine).

``fftbackend.py`` and ``mzlib.py`` are VERBATIM copies of the research repo
(tests/test_mzlib_port.py) and are never edited: the policy reaches mzlib by reassigning its
``FFT`` module global in :func:`configure` (and :func:`reset` restores the import-time one).
pyfftw and mlx are imported lazily, inside the backends.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["ENGINES", "PRECISIONS", "active", "configure", "make", "precision_note", "reset",
           "resolve"]

ENGINES = ("auto", "mlx", "fftw")
PRECISIONS = (32, 64)


def _cast(a, precision, real=False):
    if real:
        return np.asarray(a, dtype=np.float32 if precision == 32 else np.float64)
    return np.asarray(a, dtype=np.complex64 if precision == 32 else np.complex128)


class _PyFFTW:
    """FFTW3 through pyfftw.interfaces.numpy_fft -- single-precision plans at 32 (xsmurf's
    own choice), double at 64; plan-cached and threaded."""
    name = "pyfftw"

    def __init__(self, precision=32, threads=None):
        import pyfftw

        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30.0)
        self._m = pyfftw.interfaces.numpy_fft
        self._t = threads or os.cpu_count() or 1
        self.precision = int(precision)

    def fft(self, a, axis=-1):
        return self._m.fft(_cast(a, self.precision), axis=axis, threads=self._t)

    def ifft(self, a, axis=-1):
        return self._m.ifft(_cast(a, self.precision), axis=axis, threads=self._t)

    def fft2(self, a, axes=(-2, -1)):
        return self._m.fft2(_cast(a, self.precision), axes=axes, threads=self._t)

    def ifft2(self, a, axes=(-2, -1)):
        return self._m.ifft2(_cast(a, self.precision), axes=axes, threads=self._t)

    def rfft(self, a, axis=-1):
        return self._m.rfft(_cast(a, self.precision, real=True), axis=axis, threads=self._t)

    def irfft(self, a, n=None, axis=-1):
        return self._m.irfft(_cast(a, self.precision), n=n, axis=axis, threads=self._t)

    def rfft2(self, a, axes=(-2, -1)):
        return self._m.rfft2(_cast(a, self.precision, real=True), axes=axes, threads=self._t)

    def irfft2(self, a, s=None, axes=(-2, -1)):
        return self._m.irfft2(_cast(a, self.precision), s=s, axes=axes, threads=self._t)


class _MLX:
    """mlx.core.fft, single precision always (complex64/float32), converting at the
    numpy<->mlx boundary on every call."""
    name = "mlx"

    def __init__(self, precision=32):
        import mlx.core as mx

        self._mx = mx
        self.precision = 32

    def _c(self, a):
        return self._mx.array(np.asarray(a).astype(np.complex64))

    def _r(self, a):
        return self._mx.array(np.asarray(a).astype(np.float32))

    def fft(self, a, axis=-1):
        return np.array(self._mx.fft.fft(self._c(a), axis=axis))

    def ifft(self, a, axis=-1):
        return np.array(self._mx.fft.ifft(self._c(a), axis=axis))

    def fft2(self, a, axes=(-2, -1)):
        return np.array(self._mx.fft.fft2(self._c(a), axes=list(axes)))

    def ifft2(self, a, axes=(-2, -1)):
        return np.array(self._mx.fft.ifft2(self._c(a), axes=list(axes)))

    def rfft(self, a, axis=-1):
        return np.array(self._mx.fft.rfft(self._r(a), axis=axis))

    def irfft(self, a, n=None, axis=-1):
        return np.array(self._mx.fft.irfft(self._c(a), n=n, axis=axis))

    def rfft2(self, a, axes=(-2, -1)):
        return np.array(self._mx.fft.rfft2(self._r(a), axes=list(axes)))

    def irfft2(self, a, s=None, axes=(-2, -1)):
        return np.array(self._mx.fft.irfft2(self._c(a), s=None if s is None else list(s),
                                            axes=list(axes)))


class _Numpy:
    """numpy's FFT -- the warned LAST RESORT (neither mlx nor pyfftw installed)."""
    name = "numpy"

    def __init__(self, precision=32):
        self.precision = int(precision)

    def fft(self, a, axis=-1):
        return np.fft.fft(_cast(a, self.precision), axis=axis)

    def ifft(self, a, axis=-1):
        return np.fft.ifft(_cast(a, self.precision), axis=axis)

    def fft2(self, a, axes=(-2, -1)):
        return np.fft.fft2(_cast(a, self.precision), axes=axes)

    def ifft2(self, a, axes=(-2, -1)):
        return np.fft.ifft2(_cast(a, self.precision), axes=axes)

    def rfft(self, a, axis=-1):
        return np.fft.rfft(_cast(a, self.precision, real=True), axis=axis)

    def irfft(self, a, n=None, axis=-1):
        return np.fft.irfft(_cast(a, self.precision), n=n, axis=axis)

    def rfft2(self, a, axes=(-2, -1)):
        return np.fft.rfft2(_cast(a, self.precision, real=True), axes=axes)

    def irfft2(self, a, s=None, axes=(-2, -1)):
        return np.fft.irfft2(_cast(a, self.precision), s=s, axes=axes)


_BACKENDS = {"pyfftw": _PyFFTW, "mlx": _MLX, "numpy": _Numpy}
_ACTIVE = None
_CONFIG = ("auto", 32)
_MZLIB_LEGACY = None


def _importable(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def resolve(engine="auto", precision=32, *, platform=None, machine=None, have=None):
    """``(backend_name, precision)`` for a requested engine/precision on this machine."""
    import platform as _pf
    import sys
    import warnings

    precision = 64 if int(precision) == 64 else 32
    platform = sys.platform if platform is None else platform
    machine = _pf.machine() if machine is None else machine
    if have is None:
        have = {"mlx": _importable("mlx"), "pyfftw": _importable("pyfftw")}
    apple_silicon = platform == "darwin" and machine == "arm64"
    want = engine if engine in ("mlx", "fftw") else ("mlx" if apple_silicon else "fftw")
    if precision == 64 or not apple_silicon:
        want = "fftw"
    if want == "mlx" and have.get("mlx"):
        return "mlx", 32
    if have.get("pyfftw"):
        return "pyfftw", precision
    if have.get("mlx") and precision == 32:
        return "mlx", 32
    warnings.warn("neither mlx nor pyfftw is installed -- falling back to numpy's FFT, which "
                  "is slow (pip install pyfftw)", RuntimeWarning, stacklevel=2)
    return "numpy", precision


def precision_note(engine: str, precision: int, **machine) -> str:
    """The Settings warning for 64-bit FFTs where the engine would otherwise be mlx: mlx has a
    64-bit float only on the CPU and its FFT still returns complex64 (it has no complex128), so
    64-bit FFTs run on FFTW3 on the CPU instead of mlx on the GPU. Empty otherwise. ``machine``
    passes :func:`resolve`'s ``platform`` / ``machine`` / ``have``."""
    if int(precision) != 64 or resolve(engine, 32, **machine)[0] != "mlx":
        return ""
    return "⚠ 64-bit FFTs run on FFTW3 on the CPU, not on mlx's GPU: mlx has no 64-bit FFT"


def make(name: str, precision: int = 32):
    """A backend instance (``"pyfftw"``, ``"mlx"`` or ``"numpy"``) at ``precision``."""
    return _BACKENDS[name](precision=precision)


def configure(engine="auto", precision=32):
    """Set the app-wide policy backend (and route mzlib through it); returns the resolved
    ``(name, precision)``."""
    global _ACTIVE, _CONFIG, _MZLIB_LEGACY
    name, prec = resolve(engine, precision)
    _ACTIVE = make(name, prec)
    _CONFIG = (engine, int(precision))
    from dynamix.core import mzlib

    if _MZLIB_LEGACY is None:
        _MZLIB_LEGACY = mzlib.FFT
    mzlib.FFT = _ACTIVE
    return name, prec


def active():
    """The app-wide policy backend (``auto``/32 until :func:`configure` runs)."""
    global _ACTIVE
    if _ACTIVE is None:
        name, prec = resolve(*_CONFIG)
        _ACTIVE = make(name, prec)
    return _ACTIVE


def reset():
    """Back to the unconfigured default (tests): mzlib gets its import-time FFT back."""
    global _ACTIVE, _CONFIG, _MZLIB_LEGACY
    if _MZLIB_LEGACY is not None:
        from dynamix.core import mzlib

        mzlib.FFT = _MZLIB_LEGACY
        _MZLIB_LEGACY = None
    _ACTIVE = None
    _CONFIG = ("auto", 32)
