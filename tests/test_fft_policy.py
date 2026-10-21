# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""FFT engine + precision policy: mlx by default on Apple Silicon, FFTW3 elsewhere; engine and 32/64-bit precision are settable, and 64-bit falls to FFTW3. numpy's FFT is never an engine: with neither mlx nor pyfftw installed the policy refuses.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import fft_policy as fb


def _have(mlx=True, pyfftw=True):
    return {"mlx": mlx, "pyfftw": pyfftw}


@pytest.mark.parametrize("platform,machine,have,want", [
    ("darwin", "arm64", _have(), ("mlx", 32)),                 # Apple Silicon: mlx
    ("win32", "AMD64", _have(mlx=False), ("pyfftw", 32)),      # Windows: FFTW3
    ("linux", "x86_64", _have(mlx=False), ("pyfftw", 32)),
    ("darwin", "arm64", _have(mlx=False), ("pyfftw", 32)),     # mlx missing -> FFTW3
])
def test_auto_picks_mlx_on_apple_silicon_and_fftw3_elsewhere(platform, machine, have, want):
    assert fb.resolve("auto", 32, platform=platform, machine=machine, have=have)[:2] == want


def test_64_bit_always_falls_to_fftw3_because_mlx_is_32_bit_only():
    kw = dict(platform="darwin", machine="arm64", have=_have())
    assert fb.resolve("auto", 64, **kw) == ("pyfftw", 64, fb.MLX_NO_DOUBLE)
    assert fb.resolve("mlx", 64, **kw) == ("pyfftw", 64, fb.MLX_NO_DOUBLE)
    assert fb.resolve("fftw", 32, **kw) == ("pyfftw", 32, "")
    assert fb.resolve("fftw", 64, **kw) == ("pyfftw", 64, "")           # asked for FFTW: no note
    assert fb.resolve("mlx", 32, platform="win32", machine="AMD64",
                      have=_have(mlx=False)) == ("pyfftw", 32, "")


def test_without_mlx_or_pyfftw_the_policy_refuses():
    for precision in (32, 64):
        with pytest.raises(RuntimeError, match="pyfftw"):
            fb.resolve("auto", precision, platform="win32", machine="AMD64",
                       have=_have(mlx=False, pyfftw=False))
    with pytest.raises(KeyError):
        fb.make("numpy")


def test_configure_sets_the_active_backend_and_its_precision():
    try:
        fb.configure("fftw", 64)
        be = fb.active()
        assert (be.name, be.precision) == ("pyfftw", 64)
        assert be.fft2(np.ones((4, 4))).dtype == np.complex128
        fb.configure("fftw", 32)
        assert fb.active().fft2(np.ones((4, 4))).dtype == np.complex64
    finally:
        fb.reset()


def test_mzlib_follows_the_policy_without_editing_it():
    """mzlib and fftbackend are VERBATIM research copies (test_mzlib_port): the policy
    reaches mzlib by reassigning its FFT module global, never by editing either file. The
    global follows the active backend from import on, configured or not."""
    from dynamix.core import mzlib

    one = np.ones((4, 4))
    try:
        fb.configure("fftw", 64)
        assert mzlib.FFT.name == "pyfftw" and mzlib.FFT.fft2(one).dtype == np.complex128
        fb.configure("fftw", 32)
        assert mzlib.FFT.fft2(one).dtype == np.complex64
    finally:
        fb.reset()
    assert mzlib.FFT.name == fb.active().name


def test_the_double_backend_is_fftw3_at_64_bit():
    """Filter design and kernel sections run once in float64, whatever the configured engine."""
    try:
        fb.configure("auto", 32)
        be = fb.double()
        assert (be.name, be.precision) == ("pyfftw", 64)
        assert be.ifft(np.ones(8)).dtype == np.complex128
    finally:
        fb.reset()


@pytest.mark.parametrize("engine,precision,tol", [("fftw", 64, 1e-12), ("fftw", 32, 2e-5),
                                                   ("mlx", 32, 2e-5)])
def test_every_engine_agrees_with_the_reference_transform(engine, precision, tol):
    """Backends must agree to float32 tolerance (64-bit FFTW3 to float64)."""
    if engine == "mlx":
        pytest.importorskip("mlx.core")
    be = fb.make(engine if engine != "fftw" else "pyfftw", precision)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 48))
    scale = np.abs(np.fft.fft2(x)).max()
    for got, want in ((be.fft2(x), np.fft.fft2(x)),
                      (be.ifft2(be.fft2(x)), x.astype(complex)),
                      (be.rfft2(x), np.fft.rfft2(x)),
                      (be.fft(x[0]), np.fft.fft(x[0])),
                      (be.rfft(x[0]), np.fft.rfft(x[0]))):
        assert np.abs(np.asarray(got) - want).max() <= tol * scale
    np.testing.assert_allclose(be.irfft2(be.rfft2(x), s=x.shape), x, atol=tol * scale)
    np.testing.assert_allclose(be.irfft(be.rfft(x[0]), n=48), x[0], atol=tol * scale)


# ------------------------------------------------ every FFT consumer follows the policy

_TRANSFORMS = ("fft", "ifft", "fft2", "ifft2", "rfft", "irfft", "rfft2", "irfft2")
_ENGINES = [("fftw", 32), ("fftw", 64), ("mlx", 32)]


def _field(n=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)).cumsum(0).cumsum(1)


def _run_consumers():
    """One small call into every FFT-using analysis path."""
    from dynamix.core import microcanonical as mc
    from dynamix.core import wavelet_skeleton as wsk
    from dynamix.core.mz_edges import analyze
    from dynamix.core.wtmm_backend import get_backend

    v = _field()
    out = {
        "measure": mc.measure_projections(mc.gradient_measure(v), [1.0, 2.0, 4.0]),
        "ricker": mc.ricker_projections(v, [1.0, 2.0, 4.0]),
        "skeleton_W1": wsk.gradient_wt(v, 3.0)[0],
        "cwt2d": get_backend("python").cwt2d(v, [2.0, 4.0])["mod"],
        "mz": analyze(v, 2)["extrema"][0]["mod"],
    }
    return out


def _run_kernel_builders():
    """The small float64 FFTs: filter design, the scale bar's profiles, a smoother's section and
    the 1-D wavelet transform."""
    pytest.importorskip("pywt")
    from dynamix.core import wavelet_profiles as wp
    from dynamix.core.dwt import battle_lemarie_filters
    from dynamix.core.smoothers import smoother
    from dynamix.core.wtmm_backend import get_backend

    battle_lemarie_filters(3)
    wp.mz_profile(2)
    wp.dwt_profile("db4", 3)
    wp.tang_you_profile("tang_you", 4.0)
    smoother("frac_bspline", {"alpha": 2.5}).section(2.0, 1)
    get_backend("python").cwt1d(np.cumsum(np.random.default_rng(0).standard_normal(256)),
                                1.0, 2, 4)


@pytest.mark.parametrize("engine,precision", _ENGINES)
def test_no_numpy_fft_transform_runs_under_the_policy(engine, precision, monkeypatch):
    """-- a trap on every numpy FFT
    TRANSFORM (fftfreq/fftshift are index arithmetic, not transforms)."""
    if engine == "mlx":
        pytest.importorskip("mlx.core")
    fb.configure(engine, precision)

    def _trap(*a, **k):
        raise AssertionError("numpy FFT transform called under the policy")

    for name in _TRANSFORMS:
        monkeypatch.setattr(np.fft, name, _trap)
    _run_consumers()
    _run_kernel_builders()


def test_no_numpy_fft_transform_runs_before_the_policy_is_configured(monkeypatch):
    """Headless use (tests, scripts, batch runs) never calls configure: the defaults still
    run on mlx or FFTW3, mzlib's paths included."""
    fb.reset()

    def _trap(*a, **k):
        raise AssertionError("numpy FFT transform called before configure")

    for name in _TRANSFORMS:
        monkeypatch.setattr(np.fft, name, _trap)
    _run_consumers()
    _run_kernel_builders()


@pytest.mark.parametrize("engine,precision", _ENGINES)
def test_every_consumer_agrees_with_the_double_reference(engine, precision):
    """Backends agree to float32 tolerance -- 64-bit FFTW3 to float64."""
    if engine == "mlx":
        pytest.importorskip("mlx.core")
    from dynamix.core.wtmm_backend import set_default_engine

    fb.configure("fftw", 64)
    set_default_engine("fftw")
    try:
        ref = _run_consumers()             # the reference: every consumer on FFTW3 at 64-bit
    finally:
        set_default_engine(None)
    fb.configure(engine, precision)
    got = _run_consumers()
    tol = 1e-9 if precision == 64 else 5e-4
    for key in ("measure", "ricker", "skeleton_W1"):
        a, b = np.asarray(got[key], float), np.asarray(ref[key], float)
        assert np.abs(a - b).max() <= tol * np.abs(b).max(), key
    # The WTMM wavelet transform's DATA contract is float32 end to end (its input is
    # quantised to float32 and mod/arg are stored float32 -- wtmm_ebsd.cwt_2d_f32's and
    # xsmurf's own convention), so every engine agrees with it to float32, whatever the FFT
    # precision (measured 2.4e-7 relative at 64-bit).
    a, b = np.asarray(got["cwt2d"], float), np.asarray(ref["cwt2d"], float)
    assert np.abs(a - b).max() <= 1e-5 * np.abs(b).max(), "cwt2d"


# ------------------------------------------------ Hölder support at every engine/precision

@pytest.mark.parametrize("engine,precision", _ENGINES)
def test_flat_patches_have_no_exponent_at_every_engine_and_precision(engine, precision):
    """The 09-18 +-200 fix must hold on FFTW3 32/64 and mlx: no support is decided from the
    DATA (Turiel's mu(B_r) > 0 over the kernel's reach), never from FFT round-off."""
    if engine == "mlx":
        pytest.importorskip("mlx.core")
    from dynamix.core import microcanonical as mc

    fb.configure(engine, precision)
    rng = np.random.default_rng(0)
    f = rng.normal(0.0, 1.0, (128, 128)).cumsum(axis=1)
    f[:, :64] = 3.7
    scales = np.geomspace(1, 8, 6)
    for T in (mc.measure_projections(mc.gradient_measure(f), scales),
              mc.ricker_projections(f, scales)):
        h, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
        assert np.isnan(h[:, :48]).all()
        finite = h[np.isfinite(h)]
        assert finite.size > 0 and np.abs(finite).max() < 50.0


@pytest.mark.parametrize("engine,precision", _ENGINES)
@pytest.mark.parametrize("method,wavelet", [("measure", "gaussian"), ("measure", "lorentzian"),
                                            ("measure", "q_gaussian"),
                                            ("multiaffine", "g2"),
                                            ("multiaffine", "lorentzian_marr")])
def test_nodata_never_gets_an_exponent_for_any_kernel(engine, precision, method, wavelet):
    """Heavy tails used to report their own tail exponent inside nodata (a Lorentzian's
    2beta - d): NaN in, NaN out, for every kernel, engine and precision."""
    if engine == "mlx":
        pytest.importorskip("mlx.core")
    from dynamix.devices.holder_methods import method_arrays

    fb.configure(engine, precision)
    rng = np.random.default_rng(1)
    v = rng.standard_normal((96, 96)).cumsum(0).cumsum(1)
    v[30:60, 30:60] = np.nan
    params = {"estimator": "regression", "r_min": 1.0, "kappa": 4.0, "n_scales": 4,
              "wavelet": wavelet, "beta": 1.0, "q_tsallis": 1.5, "frac_n": 2.0}
    h, _r2, _s = method_arrays(v, method, params)
    assert np.isnan(h[30:60, 30:60]).all()
    assert np.isfinite(h).mean() > 0.3


def test_the_settings_note_warns_when_64_bit_takes_the_ffts_off_mlx():
    """mlx has float64 only on the CPU and its FFT returns complex64 (no complex128), so 64-bit
    FFTs run on FFTW3 on the CPU -- worth a warning exactly where mlx would otherwise run."""
    from dynamix.core.fft_policy import precision_note

    mac = dict(platform="darwin", machine="arm64", have={"mlx": True, "pyfftw": True})
    assert precision_note("auto", 64, **mac).startswith("⚠")
    assert precision_note("mlx", 64, **mac).startswith("⚠")
    assert precision_note("auto", 32, **mac) == ""
    assert precision_note("fftw", 64, **mac) == ""
    win = dict(platform="win32", machine="AMD64", have={"mlx": False, "pyfftw": True})
    assert precision_note("auto", 64, **win) == ""


def test_an_mlx_error_is_a_python_exception_never_an_abort():
    """mlx evaluates lazily. An error raised while ``np.array`` pulls an unevaluated array
    through the buffer protocol cannot cross that C boundary, so the process aborts (SIGABRT)
    instead of raising. The backend must evaluate first, so a worker thread can report the
    error. Run in a subprocess: the unfixed failure kills the interpreter."""
    pytest.importorskip("mlx.core")
    import subprocess
    import sys

    script = """
import mlx.core as mx, numpy as np
from dynamix.core.fft_policy import make
be = make("mlx")
cap = mx.device_info()["max_buffer_length"]
rows = cap // (8 * 8 * 12000) + 1                  # complex64 stack just over the Metal cap
mx.fft.fft2 = lambda a, axes=None: mx.zeros((8, rows, 12000), dtype=mx.complex64)
try:
    be.fft2(np.zeros((2, 2)))
except RuntimeError as exc:
    print("raised:", exc)
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr[-500:]
    assert "raised:" in run.stdout and "metal::malloc" in run.stdout


def test_mlx_returns_its_buffer_cache_only_past_the_limit(monkeypatch):
    import sys
    from types import SimpleNamespace

    cached, cleared = [fb.MLX_CACHE_LIMIT], []
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(
        get_active_memory=lambda: 5, get_cache_memory=lambda: cached[0],
        clear_cache=lambda: cleared.append(cached[0])))
    assert fb.mlx_memory() == {"active": 5, "cache": fb.MLX_CACHE_LIMIT}
    assert not fb.trim_mlx_cache() and cleared == []
    cached[0] += 1
    assert fb.trim_mlx_cache() and cleared == [fb.MLX_CACHE_LIMIT + 1]
    assert fb.MLX_CACHE_LIMIT == 1 << 30


def test_without_mlx_loaded_there_is_no_mlx_memory_to_report_or_trim(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    assert fb.mlx_memory() is None and fb.trim_mlx_cache() is False
