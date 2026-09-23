"""fftbackend.py -- pluggable FFT backend seam for cwtlib.py / mzlib.py.

Selection: the RECON_FFT environment variable ("numpy" | "pyfftw" | "mlx"), default
"numpy". Each backend exposes fft/ifft/fft2/ifft2 with numpy.fft's own call signature
(array, axis=-1 for the 1-D pair, axes=(-2,-1) for the 2-D pair) so cwtlib.py/mzlib.py
can call FFT.fft2(x) exactly where they used to call np.fft.fft2(x).

The numpy backend is a direct passthrough -- no wrapping, no copy, no dtype coercion --
so selecting it (the default) is bit-identical to every gate script's behavior before
this seam existed (verified in 12_backends.py's numpy_exact_passthrough gate). fftfreq
is deliberately NOT part of the seam: frequency-grid indexing is the same integer
arithmetic for every backend, so cwtlib.py/mzlib.py keep calling np.fft.fftfreq
directly -- there is nothing to swap.

pyfftw path: pyfftw.interfaces.numpy_fft, a drop-in for np.fft, with the interfaces
cache enabled (repeated same-shape calls reuse FFTW's plan instead of re-planning every
time) and threaded.

mlx path: converts float64/complex128 numpy arrays to mx.array complex64 at the
boundary, runs the transform on-device, converts the result back to a numpy complex64
array. This is a boundary conversion, not a native-mlx operator chain -- a full-mlx
Frame (keeping intermediates as mx.array across analyze/synthesize so there is no
per-call host<->device round trip) is future work. Precision is fp32 throughout the
mlx path: the backends must agree to float32 tolerance.

pyfftw and mlx are imported lazily, inside each backend's __init__ -- selecting
"numpy" (the default) never requires either package to be installed.
"""
import os
import numpy as np


class NumpyBackend:
    """np.fft, called directly. The reference backend; every other backend is judged
    against this one's output."""
    name = "numpy"

    def fft(self, a, axis=-1):
        return np.fft.fft(a, axis=axis)

    def ifft(self, a, axis=-1):
        return np.fft.ifft(a, axis=axis)

    def fft2(self, a, axes=(-2, -1)):
        return np.fft.fft2(a, axes=axes)

    def ifft2(self, a, axes=(-2, -1)):
        return np.fft.ifft2(a, axes=axes)


class PyFFTWBackend:
    """pyfftw.interfaces.numpy_fft -- a drop-in for np.fft, plan-cached and threaded."""
    name = "pyfftw"

    def __init__(self, threads=None):
        import pyfftw
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30.0)
        self._m = pyfftw.interfaces.numpy_fft
        self._threads = threads or os.cpu_count() or 1

    def fft(self, a, axis=-1):
        return self._m.fft(a, axis=axis, threads=self._threads)

    def ifft(self, a, axis=-1):
        return self._m.ifft(a, axis=axis, threads=self._threads)

    def fft2(self, a, axes=(-2, -1)):
        return self._m.fft2(a, axes=axes, threads=self._threads)

    def ifft2(self, a, axes=(-2, -1)):
        return self._m.ifft2(a, axes=axes, threads=self._threads)


class MLXBackend:
    """mlx.core.fft, complex64, converting at the numpy<->mlx boundary on every call
    (see module docstring)."""
    name = "mlx"

    def __init__(self):
        import mlx.core as mx
        self._mx = mx

    def _to_mx(self, a):
        return self._mx.array(np.asarray(a).astype(np.complex64))

    def fft(self, a, axis=-1):
        return np.array(self._mx.fft.fft(self._to_mx(a), axis=axis))

    def ifft(self, a, axis=-1):
        return np.array(self._mx.fft.ifft(self._to_mx(a), axis=axis))

    def fft2(self, a, axes=(-2, -1)):
        return np.array(self._mx.fft.fft2(self._to_mx(a), axes=list(axes)))

    def ifft2(self, a, axes=(-2, -1)):
        return np.array(self._mx.fft.ifft2(self._to_mx(a), axes=list(axes)))


_BACKENDS = {"numpy": NumpyBackend, "pyfftw": PyFFTWBackend, "mlx": MLXBackend}


def get_backend(name=None):
    """name=None reads the RECON_FFT env var (default "numpy"). Each call constructs a
    fresh instance -- cheap for all three (a module import plus, for pyfftw, an
    idempotent cache-enable call)."""
    name = name or os.environ.get("RECON_FFT", "numpy")
    if name not in _BACKENDS:
        raise ValueError(f"unknown RECON_FFT backend {name!r}; choices: {sorted(_BACKENDS)}")
    return _BACKENDS[name]()
