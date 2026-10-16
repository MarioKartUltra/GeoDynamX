# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""FFT precision as an ordinary device param, the engine as a setting.

``precision`` (f32/f64) sits on every tool whose compute runs FFTs. It reaches the key like any
other param, except at its default, so every saved chain keeps its key. The engine never reaches
a key. The engine runs a tool's compute inside its precision, so every FFT in it, mzlib's
included, runs at that precision; mlx has no double FFT, so a 64-bit request there runs on FFTW3
with a note.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from dynamix.core import fft_policy as fb
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import DEVICES, defaults_for, keyed_params
from dynamix.model.project import Project

#: The tools whose compute, outputs or preview run FFTs (measured: every other field tool runs
#: none at any setting).
FFT_TOOLS = ("band_recon", "band_recon_measure", "band_recon_multiaffine", "holder_map",
             "holder_measure", "holder_multiaffine", "mz_edges", "ssa2d", "wavelet_skeleton",
             "wtmm2d", "wtmm2d_roi")
NO_FFT_TOOLS = ("cdf_edges", "inpaint", "tucker_HOOI_HOSVD", "pm_edges", "dwt2d")


@pytest.fixture
def builtin(clean_registry):
    register_builtin_devices()
    return clean_registry


def _field(n=64, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, n)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))


def test_every_fft_tool_declares_precision_and_no_other_does(builtin):
    for name in FFT_TOOLS:
        p = {q.name: q for q in DEVICES[name].params}["precision"]
        assert p.default == "f32" and p.choices == ("f32", "f64"), name
        assert not p.keyed_at_default and not p.view, name
    for name in NO_FFT_TOOLS:
        assert "precision" not in {q.name for q in DEVICES[name].params}, name


def test_the_default_leaves_the_key_and_f64_enters_it(builtin):
    for name in FFT_TOOLS:
        d = DEVICES[name]
        params = defaults_for(d)
        assert "precision" not in keyed_params(d, params), name
        assert keyed_params(d, dict(params, precision="f64"))["precision"] == "f64", name


def test_using_scopes_the_precision_to_its_thread():
    fb.configure("fftw", 32)
    seen = {}
    with fb.using("f64"):
        assert (fb.active().name, fb.active().precision) == ("pyfftw", 64)
        t = threading.Thread(target=lambda: seen.update(other=fb.active().precision))
        t.start()
        t.join()
        with fb.using("f32"):
            assert fb.active().precision == 32
        assert fb.active().precision == 64
    assert fb.active().precision == 32 and seen["other"] == 32


def test_mlx_refuses_64_bits_and_the_note_says_where_it_ran():
    pytest.importorskip("mlx.core")
    with pytest.raises(ValueError, match="double"):
        fb.make("mlx", 64)
    mac = dict(platform="darwin", machine="arm64", have={"mlx": True, "pyfftw": True})
    assert fb.resolve("auto", 64, **mac) == ("pyfftw", 64, fb.MLX_NO_DOUBLE)
    fb.configure("mlx", 32)
    assert fb.note("f32") == ""
    with fb.using("f64"):
        assert fb.active().name == "pyfftw" and fb.note() == fb.MLX_NO_DOUBLE


def _precisions_seen(monkeypatch):
    """Every precision a policy backend ran a transform at."""
    seen = set()
    for cls in (fb._PyFFTW, fb._MLX):
        for meth in ("fft", "ifft", "fft2", "ifft2", "rfft", "irfft", "rfft2", "irfft2"):
            orig = getattr(cls, meth)

            def spy(self, *a, _orig=orig, **k):
                seen.add(self.precision)
                return _orig(self, *a, **k)

            monkeypatch.setattr(cls, meth, spy)
    return seen


@pytest.mark.parametrize("tool", ["holder_map", "ssa2d", "wavelet_skeleton", "wtmm2d"])
def test_the_engine_computes_a_tool_at_its_precision(builtin, monkeypatch, tool):
    fb.configure("fftw", 32)
    seen = _precisions_seen(monkeypatch)
    project = Project()
    src = project.add_source("/data/x.tif")
    for precision in ("f32", "f64"):
        layer = project.add_layer(precision, src.source_id,
                                  Chain([DeviceRef(tool, {"precision": precision})]))
        seen.clear()
        resolve(layer, _field(), Cache())
        assert seen == {32 if precision == "f32" else 64}, (tool, precision, seen)


def test_the_engine_never_reaches_a_key(builtin):
    project = Project()
    src = project.add_source("/data/x.tif")
    layer = project.add_layer("x", src.source_id, Chain([DeviceRef("holder_map", {})]))
    keys = []
    for engine in ("fftw", "auto"):
        fb.configure(engine, 32)
        keys.append(resolve(layer, _field(32), Cache()).analysis_key)
    assert keys[0] == keys[1]


def test_the_wavelet_transform_follows_the_precision_end_to_end(builtin):
    """At 64 bits the transform's filters, FFTs, modulus and argument are float64; at 32 bits
    the float32 contract (xsmurf's) holds."""
    from dynamix.core.wtmm_backend import compute_scales2d, get_backend

    v = _field(48).values
    scales = compute_scales2d(2, 2)
    fb.configure("fftw", 32)
    lo = get_backend("python").cwt2d(v, scales, derivs="all")
    with fb.using("f64"):
        hi = get_backend("python").cwt2d(v, scales, derivs="all")
    assert lo["mod"].dtype == np.float32 and lo["dxx"].dtype == np.float32
    assert hi["mod"].dtype == np.float64 and hi["dxx"].dtype == np.float64
    np.testing.assert_allclose(lo["mod"], hi["mod"], rtol=1e-4, atol=1e-5 * hi["mod"].max())


def test_the_reading_names_the_fallback_only_where_it_happens():
    from dynamix.devices.precision import precision_reading

    fb.configure("fftw", 32)
    assert precision_reading({"precision": "f64"}) is None
    pytest.importorskip("mlx.core")
    fb.configure("mlx", 32)
    assert precision_reading({"precision": "f32"}) is None
    assert precision_reading({"precision": "f64"}) == fb.MLX_NO_DOUBLE


def test_the_context_is_what_the_gprv_writer_records():
    """gprv 0.2 §6.2's execution context: the precision actually used, the backend that ran it
    (name, library version, planner, binding) and the fallback note when there was one."""
    fb.configure("fftw", 32)
    c = fb.context("f64")
    assert set(c) == {"precision", "backend"} and c["precision"] == "float64"
    assert set(c["backend"]) == {"name", "version", "planner", "binding"}
    assert c["backend"]["name"] == "fftw" and c["backend"]["planner"] == "FFTW_ESTIMATE"
    assert c["backend"]["binding"].startswith("pyfftw ")
    assert fb.context()["precision"] == "float32"
    pytest.importorskip("mlx.core")
    fb.configure("mlx", 32)
    assert fb.context()["backend"]["name"] == "mlx" and fb.context()["backend"]["planner"] is None
    with fb.using("f64"):
        c = fb.context()
    assert c["backend"]["name"] == "fftw" and c["fallback"] == fb.MLX_NO_DOUBLE


def test_a_new_tool_starts_at_the_settings_precision(builtin):
    """Settings › FFT precision is the default of a newly placed tool's knob: at 64 the tool
    carries f64 in its recipe (and its key); a tool without the knob gets its defaults."""
    from dynamix.devices.precision import placement_params

    fb.configure("fftw", 64)
    assert placement_params(DEVICES["holder_map"])["precision"] == "f64"
    assert placement_params(DEVICES["cdf_edges"]) == defaults_for(DEVICES["cdf_edges"])
    fb.configure("fftw", 32)
    assert placement_params(DEVICES["holder_map"]) == defaults_for(DEVICES["holder_map"])
