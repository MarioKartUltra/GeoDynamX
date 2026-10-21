# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""fft_policy.py -- the app-wide FFT engine + precision POLICY (DynamiX-native).

Every FFT runs on mlx or FFTW3, never numpy. So:

- ``auto``: mlx on Apple Silicon (when installed), FFTW3 (pyfftw) everywhere else;
- 64-bit always runs on FFTW3 (mlx is single precision only);
- a requested engine that is not installed falls to the other;
- neither installed is an error: pyfftw is a dependency on every platform.

Set once at startup from ``Settings.compute_engine`` / ``Settings.compute_precision`` (changes apply after a restart) via :func:`configure`. Every FFT consumer calls :func:`active`
at call time. The engine is INFRASTRUCTURE, never part of a cache key: the project's
backends-agree-to-float32 law (tests/test_fft_policy.py pins the agreement for every engine).
Precision reaches a key only as an ordinary device param (``precision``, f32 or f64): the engine
runs a device's compute inside :func:`using` its precision, so :func:`active` hands every FFT in it
a backend at that precision; mlx has no 64-bit FFT, so a 64-bit request there runs on FFTW3 and
:func:`resolve` says so in its note, never a silent downcast.

``fftbackend.py`` and ``mzlib.py`` are VERBATIM copies of the research repo
(tests/test_mzlib_port.py) and are never edited: the policy reaches mzlib by reassigning its
``FFT`` module global, at this module's import, to a proxy that follows :func:`active`.
pyfftw and mlx are imported lazily, inside the backends.
"""
from __future__ import annotations

import contextlib
import functools
import os
import sys
import threading

import numpy as np

__all__ = ["ENGINES", "PRECISIONS", "active", "bits", "configure", "configured_precision",
           "context", "double", "make", "note", "precision_note", "reset", "resolve", "scope_for",
           "using", "observing", "observed", "include_observed", "mlx_memory", "trim_mlx_cache"]

ENGINES = ("auto", "mlx", "fftw")
PRECISIONS = (32, 64)


_OBSERVERS = threading.local()


@contextlib.contextmanager
def observing():
    """Collect contexts of FFT calls, including the work represented by reused stages."""
    calls = []
    stack = getattr(_OBSERVERS, "stack", ())
    _OBSERVERS.stack = (*stack, calls)
    try:
        yield calls
    finally:
        _OBSERVERS.stack = stack


def observed():
    stack = getattr(_OBSERVERS, "stack", ())
    return list(stack[-1]) if stack else []


def include_observed(contexts):
    for calls in getattr(_OBSERVERS, "stack", ()):
        for ctx in contexts:
            if ctx not in calls:
                calls.append(ctx)


def _observed_fft(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        if getattr(_OBSERVERS, "stack", ()):
            if self.name == "pyfftw":
                import pyfftw
                import pyfftw.config
                backend = {"name": "fftw", "version": str(pyfftw.fftw_version),
                           "planner": str(pyfftw.config.PLANNER_EFFORT),
                           "binding": f"pyfftw {pyfftw.__version__}"}
            else:
                backend = {"name": "mlx", "version": str(self._mx.__version__),
                           "planner": None, "binding": None}
            include_observed([{"precision": f"float{self.precision}", "backend": backend}])
        return method(self, *args, **kwargs)
    return call


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

    @_observed_fft
    def fft(self, a, axis=-1):
        return self._m.fft(_cast(a, self.precision), axis=axis, threads=self._t)

    @_observed_fft
    def ifft(self, a, axis=-1):
        return self._m.ifft(_cast(a, self.precision), axis=axis, threads=self._t)

    @_observed_fft
    def fft2(self, a, axes=(-2, -1)):
        return self._m.fft2(_cast(a, self.precision), axes=axes, threads=self._t)

    @_observed_fft
    def ifft2(self, a, axes=(-2, -1)):
        return self._m.ifft2(_cast(a, self.precision), axes=axes, threads=self._t)

    @_observed_fft
    def rfft(self, a, axis=-1):
        return self._m.rfft(_cast(a, self.precision, real=True), axis=axis, threads=self._t)

    @_observed_fft
    def irfft(self, a, n=None, axis=-1):
        return self._m.irfft(_cast(a, self.precision), n=n, axis=axis, threads=self._t)

    @_observed_fft
    def rfft2(self, a, axes=(-2, -1)):
        return self._m.rfft2(_cast(a, self.precision, real=True), axes=axes, threads=self._t)

    @_observed_fft
    def irfft2(self, a, s=None, axes=(-2, -1)):
        return self._m.irfft2(_cast(a, self.precision), s=s, axes=axes, threads=self._t)


class _MLX:
    """mlx.core.fft, single precision always (complex64/float32), converting at the
    numpy<->mlx boundary on every call. A 64-bit request is refused: :func:`resolve` sends it
    to FFTW3 with a note, never a silent downcast."""
    name = "mlx"

    def __init__(self, precision=32):
        import mlx.core as mx

        if bits(precision) == 64:
            raise ValueError(MLX_NO_DOUBLE)
        self._mx = mx
        self.precision = 32

    def _c(self, a):
        return self._mx.array(np.asarray(a).astype(np.complex64))

    def _r(self, a):
        return self._mx.array(np.asarray(a).astype(np.float32))

    def _np(self, x):
        # Evaluate before converting: an mlx error raised inside np.array's buffer-protocol
        # read cannot cross that C boundary and aborts the process; mx.eval raises it.
        self._mx.eval(x)
        return np.array(x)

    @_observed_fft
    def fft(self, a, axis=-1):
        return self._np(self._mx.fft.fft(self._c(a), axis=axis))

    @_observed_fft
    def ifft(self, a, axis=-1):
        return self._np(self._mx.fft.ifft(self._c(a), axis=axis))

    @_observed_fft
    def fft2(self, a, axes=(-2, -1)):
        return self._np(self._mx.fft.fft2(self._c(a), axes=list(axes)))

    @_observed_fft
    def ifft2(self, a, axes=(-2, -1)):
        return self._np(self._mx.fft.ifft2(self._c(a), axes=list(axes)))

    @_observed_fft
    def rfft(self, a, axis=-1):
        return self._np(self._mx.fft.rfft(self._r(a), axis=axis))

    @_observed_fft
    def irfft(self, a, n=None, axis=-1):
        return self._np(self._mx.fft.irfft(self._c(a), n=n, axis=axis))

    @_observed_fft
    def rfft2(self, a, axes=(-2, -1)):
        return self._np(self._mx.fft.rfft2(self._r(a), axes=list(axes)))

    @_observed_fft
    def irfft2(self, a, s=None, axes=(-2, -1)):
        return self._np(self._mx.fft.irfft2(self._c(a), s=None if s is None else list(s),
                                            axes=list(axes)))


_BACKENDS = {"pyfftw": _PyFFTW, "mlx": _MLX}
_ACTIVE = None
_CONFIG = ("auto", 32)
_DOUBLE = None


def _importable(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


#: The note when a 64-bit FFT is asked where the engine would be mlx.
MLX_NO_DOUBLE = "mlx has no double FFT; using FFTW3 double"
#: The note for the reserved CUDA engine.
CUDA_NOT_INSTALLED = "CUDA is not installed; using the default engine"


def bits(precision) -> int:
    """32 or 64 from ``"f32"``/``"f64"``, ``"float32"``/``"float64"`` or the number itself."""
    text = str(precision)
    return 64 if text in ("64", "f64", "float64") else 32


def resolve(engine="auto", precision=32, *, platform=None, machine=None, have=None):
    """``(backend_name, precision, note)`` for a requested engine/precision on this machine. The
    note is empty unless the request could not be met as asked (64-bit where the engine would be
    mlx, or the reserved ``cuda``); a RuntimeError when neither mlx nor pyfftw is installed."""
    import platform as _pf
    import sys

    precision = bits(precision)
    platform = sys.platform if platform is None else platform
    machine = _pf.machine() if machine is None else machine
    if have is None:
        have = {"mlx": _importable("mlx"), "pyfftw": _importable("pyfftw")}
    note = ""
    if engine == "cuda":
        note, engine = CUDA_NOT_INSTALLED, "auto"
    apple_silicon = platform == "darwin" and machine == "arm64"
    want = engine if engine in ("mlx", "fftw") else ("mlx" if apple_silicon else "fftw")
    if precision == 64 and want == "mlx" and (apple_silicon or have.get("mlx")) and have.get("mlx"):
        note = note or MLX_NO_DOUBLE
    if precision == 64 or not apple_silicon:
        want = "fftw"
    if want == "mlx" and have.get("mlx"):
        return "mlx", 32, note
    if have.get("pyfftw"):
        return "pyfftw", precision, note
    if have.get("mlx") and precision == 32:
        return "mlx", 32, note
    raise RuntimeError("neither mlx nor pyfftw is installed; every FFT runs on one of them "
                       "(pip install pyfftw)")


def precision_note(engine: str, precision: int, **machine) -> str:
    """The Settings warning for 64-bit FFTs where the engine would otherwise be mlx: mlx has a
    64-bit float only on the CPU and its FFT still returns complex64 (it has no complex128), so
    64-bit FFTs run on FFTW3 on the CPU instead of mlx on the GPU. Empty otherwise. ``machine``
    passes :func:`resolve`'s ``platform`` / ``machine`` / ``have``."""
    if bits(precision) != 64 or resolve(engine, 32, **machine)[0] != "mlx":
        return ""
    return "⚠ 64-bit FFTs run on FFTW3 on the CPU, not on mlx's GPU: mlx has no 64-bit FFT"


def make(name: str, precision: int = 32):
    """A backend instance (``"pyfftw"`` or ``"mlx"``) at ``precision``."""
    return _BACKENDS[name](precision=precision)


def double():
    """FFTW3 at 64-bit, whatever the configured engine: filter design and kernel sections that
    are computed once in float64 (mlx has no double FFT)."""
    global _DOUBLE
    if _DOUBLE is None:
        _DOUBLE = make(resolve("fftw", 64)[0], 64)
    return _DOUBLE


def configure(engine="auto", precision=32):
    """Set the app-wide policy backend (mzlib follows it); returns the resolved
    ``(name, precision)``."""
    global _ACTIVE, _CONFIG
    name, prec, _note = resolve(engine, precision)
    _ACTIVE = make(name, prec)
    _CONFIG = (engine, bits(precision))
    _BY_PRECISION.clear()
    return name, prec


#: The precision a device's compute asked for, per thread (the engine computes in workers).
_SCOPE = threading.local()
#: Backends made for a scoped precision other than the configured one, by precision.
_BY_PRECISION: dict = {}


@contextlib.contextmanager
def using(precision):
    """Every :func:`active` call in this thread, inside the block, answers with a backend at
    ``precision`` (``"f32"``/``"f64"``/32/64) on the configured engine: the engine runs a device's
    compute, outputs and preview inside its ``precision`` param. Nests; restores on exit."""
    previous = getattr(_SCOPE, "precision", None)
    _SCOPE.precision = bits(precision)
    try:
        yield
    finally:
        _SCOPE.precision = previous


def scope_for(device, params: dict):
    """:func:`using` the device's ``precision`` param when it declares one, else a no-op."""
    if any(getattr(p, "name", None) == "precision" for p in getattr(device, "params", ())):
        return using(params.get("precision", "f32"))
    return contextlib.nullcontext()


def configured_precision() -> int:
    """The app-wide precision :func:`configure` set (32 until it runs): the FFTs outside a tool's
    compute run at it, and a newly placed tool's knob starts at it."""
    return _CONFIG[1]


def _wanted(precision) -> int:
    return bits(precision) if precision is not None else (getattr(_SCOPE, "precision", None)
                                                          or _CONFIG[1])


def note(precision=None) -> str:
    """The fallback note for ``precision`` (the scoped or configured one by default) on the
    configured engine; empty when it runs as asked."""
    return resolve(_CONFIG[0], _wanted(precision))[2]


def context(precision=None) -> dict:
    """The execution context of an FFT run at ``precision`` (the scoped or configured one by
    default) on the configured engine, in gprv 0.2's terms (§6.2): the precision actually used,
    the backend that runs it (name, library version, planner, binding) and the fallback note
    when the request could not run as asked."""
    name, prec, why = resolve(_CONFIG[0], _wanted(precision))
    if name == "pyfftw":
        import pyfftw
        import pyfftw.config

        backend = {"name": "fftw", "version": str(pyfftw.fftw_version),
                   "planner": str(pyfftw.config.PLANNER_EFFORT),
                   "binding": f"pyfftw {pyfftw.__version__}"}
    else:
        import mlx.core as mx

        backend = {"name": "mlx", "version": str(mx.__version__), "planner": None,
                   "binding": None}
    out = {"precision": "float64" if prec == 64 else "float32", "backend": backend}
    if why:
        out["fallback"] = why
    return out


def active():
    """The policy backend: at the scoped precision inside :func:`using`, else the app-wide one
    (``auto``/32 until :func:`configure` runs)."""
    global _ACTIVE
    scoped = getattr(_SCOPE, "precision", None)
    if scoped is not None and scoped != _CONFIG[1]:
        be = _BY_PRECISION.get(scoped)
        if be is None:
            name, prec, _note = resolve(_CONFIG[0], scoped)
            be = _BY_PRECISION[scoped] = make(name, prec)
        return be
    if _ACTIVE is None:
        name, prec, _note = resolve(*_CONFIG)
        _ACTIVE = make(name, prec)
    return _ACTIVE


def reset():
    """Back to the unconfigured default (tests); mzlib keeps following the policy."""
    global _ACTIVE, _CONFIG
    _ACTIVE = None
    _CONFIG = ("auto", 32)
    _BY_PRECISION.clear()
    _SCOPE.precision = None


#: mlx's buffer cache past which a finished resolve hands it back to the system (1 GiB).
MLX_CACHE_LIMIT = 1 << 30


def mlx_memory() -> dict | None:
    """mlx's allocator: ``{"active": bytes in live arrays, "cache": bytes of freed buffers it keeps
    for reuse}``; ``None`` while mlx is not loaded, which off Apple Silicon is always."""
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return None
    return {"active": int(mx.get_active_memory()), "cache": int(mx.get_cache_memory())}


def trim_mlx_cache() -> bool:
    """Hand mlx's cached buffers back to the system once they pass :data:`MLX_CACHE_LIMIT`; True
    when it did. mlx keeps every freed buffer for reuse and a compute on the same field reuses
    them, so below the limit the cache stays warm; a session of knob changes at 2048² otherwise
    holds several GB of it."""
    mx = sys.modules.get("mlx.core")
    if mx is None or mx.get_cache_memory() <= MLX_CACHE_LIMIT:
        return False
    mx.clear_cache()
    return True


class _Following:
    """mzlib's ``FFT`` global: every attribute is the :func:`active` backend's at the time of
    the call, so mzlib runs on the configured engine, and inside a device's :func:`using` scope
    at its precision, without being edited."""

    def __getattr__(self, name):
        return getattr(active(), name)


def _route_mzlib() -> None:
    from dynamix.core import mzlib

    mzlib.FFT = _Following()


_route_mzlib()
