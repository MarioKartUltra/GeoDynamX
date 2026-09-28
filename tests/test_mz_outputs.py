# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""mz_edges' lazily computed outputs through the engine: the coarse channel, its 2^J thumbnail,
the reconstruction from multiscale edges (with and without the coarse channel) and its residual.
The knobs that shape them (Show and the reconstruction knobs) are view-only, so flipping one never
re-runs the edge analysis; each output is cached under its own key with the analysis upstream.
The output tests here run the printed algorithm (``algorithm="printed"``); the LastWave engine's
outputs are pinned in ``tests/test_mz_edges_lastwave_device.py``."""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from dynamix.core import mz_edges as mz
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import output_key, resolve, resolve_output
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import declared_outputs, get_device, keyed_params, validate_params
from dynamix.model.layer import Layer

DEM = pathlib.Path(__file__).resolve().parents[1] / "docs" / "demo" / "dem_crop.npz"

_VIEW = ("recon_live", "kappa", "clip", "run_mode", "iterations", "tolerance", "coarse", "mode",
         "show")


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture(scope="module")
def dem() -> RasterField:
    if not DEM.is_file():
        pytest.skip("the demo raster is not in this checkout")
    return RasterField.load_npz(str(DEM))


def _synthetic(ny: int = 64, nx: int = 64, seed: int = 0) -> RasterField:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((ny, nx)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(nx)), y_axis=np.arange(float(ny)))


def _layer(**params) -> Layer:
    return Layer(layer_id=1, name="mz", source_id="s",
                 chain=Chain((DeviceRef("mz_edges", dict(params)),)).materialized())


def _printed(**params) -> Layer:
    """A layer on the printed algorithm at its former default of 10 iterations."""
    return _layer(**{"algorithm": "printed", "iterations": 10, **params})


def _output(name: str):
    return next(o for o in declared_outputs(get_device("mz_edges")) if o.name == name)


def _counting(monkeypatch) -> list:
    """Replace ``core.mz_edges.reconstruct`` with a wrapper that records each call's coarse."""
    calls: list = []
    real = mz.reconstruct

    def counted(values, bundle, **kw):
        calls.append(kw.get("coarse", "full"))
        return real(values, bundle, **kw)

    monkeypatch.setattr(mz, "reconstruct", counted)
    return calls


def _close32(got, want) -> None:
    """Equal to float32 precision, relative to the field's own magnitude."""
    want = np.asarray(want, dtype=np.float64)
    tol = 1e-6 * float(np.abs(want).max())
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want, rtol=1e-6, atol=tol)


# --------------------------------------------------------------------------- the surface


def test_param_order_choices_and_view_only_knobs(builtins):
    dev = get_device("mz_edges")
    by = {p.name: p for p in dev.params}
    assert [p.name for p in dev.params] == ["n_levels", "algorithm", "border", "colocate_l1",
                                            "dither", "interpolate", "recon_live", "kappa",
                                            "clip", "run_mode", "iterations", "tolerance",
                                            "coarse", "mode", "show"]
    assert by["show"].choices == ("edges", "coarse", "thumbnail", "recon", "recon_edges_only",
                                  "residual")
    assert by["show"].default == "edges"
    it = by["iterations"]
    assert (it.default, it.min, it.max, it.soft_min, it.soft_max, it.label) == (
        20, 1, 500, 1, 50, "Iterations")
    assert by["mode"].choices == mz.RECON_MODES
    assert (by["mode"].default, by["mode"].label) == ("separable", "Mode")
    assert by["coarse"].choices == ("full", "thumbnail")
    assert {p.name for p in dev.params if p.view} == set(_VIEW)
    params = {p.name: p.default for p in dev.params}
    assert set(keyed_params(dev, params)) == {"n_levels", "algorithm", "border", "colocate_l1",
                                              "dither", "interpolate"}


def test_declared_outputs(builtins):
    outs = declared_outputs(get_device("mz_edges"))
    assert [o.name for o in outs] == ["edges", "coarse", "thumbnail", "recon",
                                      "recon_edges_only", "residual", "recon_preview"]
    assert [o.label for o in outs] == ["edges", "coarse", "thumbnail", "recon (edges + coarse)",
                                       "recon (edges only)", "residual", "recon preview"]
    assert [o.kind for o in outs] == ["vector"] + ["raster"] * 6
    assert [o.lazy for o in outs] == [False] + [True] * 6
    assert _output("thumbnail").grid == "stride"
    assert all(o.grid == "native" for o in outs if o.name != "thumbnail")
    recon = ("kappa", "clip", "run_mode", "iterations", "tolerance", "coarse", "mode")
    assert _output("recon").params == recon
    assert _output("recon_edges_only").params == tuple(p for p in recon if p != "coarse")
    assert _output("residual").params == recon
    assert _output("recon_preview").params == ("kappa", "clip", "coarse")
    assert _output("coarse").params == () and _output("thumbnail").params == ()
    # every show value past "edges" names a lazy output, and every lazy output but the
    # one-iteration preview (drawn on the recon row) is a show value
    show = next(p for p in get_device("mz_edges").params if p.name == "show")
    assert set(show.choices[1:]) == {o.name for o in outs if o.lazy} - {"recon_preview"}


@pytest.mark.parametrize("name,value", [("coarse", "thumbnail"), ("show", "recon"),
                                        ("iterations", 5), ("mode", "set_points"),
                                        ("kappa", 3.0), ("clip", True), ("run_mode", "fixed"),
                                        ("tolerance", 1e-4), ("recon_live", False)])
def test_flipping_a_view_knob_is_a_cache_hit_for_the_analysis(builtins, name, value):
    field, cache = _synthetic(), Cache()
    first = resolve(_layer(n_levels=3), field, cache)
    again = resolve(_layer(n_levels=3, **{name: value}), field, cache)
    assert first.cache_misses == 1
    assert again.cache_misses == 0 and again.cache_hits == 1
    assert again.analysis_key == first.analysis_key
    assert again.result["extrema"] is first.result["extrema"]  # the analysis is shown unchanged
    assert again.result["params"][name] == value         # ...under the current view knobs
    assert first.result["params"][name] != value
    assert cache.get(first.analysis_key)["params"][name] != value  # the cache is not mutated


def test_the_analysis_always_uses_the_full_coarse(builtins):
    """Coarse only chooses what the reconstruction pins; the edge analysis is the same bundle."""
    res = resolve(_printed(n_levels=3, coarse="thumbnail"), _synthetic(), Cache()).result
    assert res["coarse_policy"] == "full"
    assert res["coarse_thumb"] is None


def test_the_device_cache_key_ignores_view_knobs(builtins):
    dev = get_device("mz_edges")
    a = {p.name: p.default for p in dev.params}
    b = {**a, "show": "recon", "iterations": 3, "mode": "set_points", "coarse": "thumbnail",
         "kappa": 2.0, "clip": True, "run_mode": "fixed", "tolerance": 1e-2,
         "recon_live": False}
    assert dev.cache_key("src", a) == dev.cache_key("src", b)
    assert dev.cache_key("src", a) != dev.cache_key("src", {**a, "n_levels": 3})
    for name, value in (("algorithm", "printed"), ("border", "periodic"),
                        ("colocate_l1", True)):
        assert dev.cache_key("src", a) != dev.cache_key("src", {**a, name: value})


# --------------------------------------------------------------------------- the outputs


def test_recon_equals_the_core_reconstruction(builtins, dem):
    out = resolve_output(_printed(n_levels=4), dem, Cache(), "recon")
    ref, ref_diag = mz.reconstruct(dem.values, mz.analyze(dem.values, 4), n_iter=10)
    assert out["raster"].dtype == np.float32
    assert out["raster"].shape == dem.values.shape
    _close32(out["raster"], ref)
    assert out["diag"]["n_iter"] == 10
    assert out["diag"]["coarse"] == "full" and out["diag"]["mode"] == "separable"
    assert out["diag"]["snr_db"] == pytest.approx(ref_diag["snr_db"], rel=1e-6)


def test_an_iterations_change_rekeys_the_recon_alone(builtins, dem, monkeypatch):
    calls = _counting(monkeypatch)
    cache = Cache()
    ten = resolve_output(_printed(n_levels=4), dem, cache, "recon")
    r10 = resolve(_printed(n_levels=4), dem, cache)
    r5 = resolve(_printed(n_levels=4, iterations=5), dem, cache)
    assert r5.cache_misses == 0 and r5.analysis_key == r10.analysis_key    # analysis untouched
    k10 = output_key("mz_edges", _output("recon"), r10.analysis_params, r10.analysis_key)
    k5 = output_key("mz_edges", _output("recon"), r5.analysis_params, r5.analysis_key)
    assert k5 != k10 and k10 in cache and k5 not in cache
    five = resolve_output(_printed(n_levels=4, iterations=5), dem, cache, "recon")
    assert len(calls) == 2 and k5 in cache
    assert five["diag"]["n_iter"] == 5 and ten["diag"]["n_iter"] == 10
    assert not np.array_equal(five["raster"], ten["raster"])
    resolve_output(_printed(n_levels=4, iterations=5), dem, cache, "recon")
    assert len(calls) == 2                                                  # cached
    # the edges-only recon does not read Coarse: flipping it keeps that key
    r_thumb = resolve(_printed(n_levels=4, coarse="thumbnail"), dem, cache)
    assert (output_key("mz_edges", _output("recon_edges_only"), r_thumb.analysis_params,
                       r_thumb.analysis_key)
            == output_key("mz_edges", _output("recon_edges_only"), r10.analysis_params,
                          r10.analysis_key))
    assert (output_key("mz_edges", _output("recon"), r_thumb.analysis_params, r_thumb.analysis_key)
            != k10)


def test_residual_plus_recon_is_the_field_and_reuses_the_recon(builtins, dem, monkeypatch):
    calls = _counting(monkeypatch)
    cache = Cache()
    recon = resolve_output(_printed(n_levels=4), dem, cache, "recon")
    resid = resolve_output(_printed(n_levels=4), dem, cache, "residual")
    assert calls == ["full"]                               # no second reconstruction
    assert resid["raster"].dtype == np.float32
    _close32(resid["raster"].astype(np.float64) + recon["raster"], dem.values)
    assert resid["diag"] == recon["diag"]


def test_residual_first_computes_the_recon_under_its_own_key(builtins, monkeypatch):
    calls = _counting(monkeypatch)
    field, cache = _synthetic(), Cache()
    resid = resolve_output(_printed(n_levels=3), field, cache, "residual")
    r = resolve(_printed(n_levels=3), field, cache)
    assert output_key("mz_edges", _output("recon"), r.analysis_params, r.analysis_key) in cache
    recon = resolve_output(_printed(n_levels=3), field, cache, "recon")
    assert calls == ["full"]
    _close32(resid["raster"].astype(np.float64) + recon["raster"], field.values)


def test_recon_edges_only_pins_no_coarse(builtins, monkeypatch):
    calls = _counting(monkeypatch)
    field = _synthetic()
    out = resolve_output(_printed(n_levels=3, coarse="thumbnail"), field, Cache(),
                         "recon_edges_only")
    ref, _d = mz.reconstruct(field.values, mz.analyze(field.values, 3), n_iter=10,
                             coarse="none")
    assert calls[0] == "none"
    assert out["diag"]["coarse"] == "none"
    _close32(out["raster"], ref)


def test_the_coarse_output_is_the_full_resolution_coarse_channel(builtins, dem):
    out = resolve_output(_printed(n_levels=4), dem, Cache(), "coarse")
    assert out["raster"].dtype == np.float32 and out["raster"].shape == dem.values.shape
    _close32(out["raster"], mz.coarse_image(dem.values, mz.analyze(dem.values, 4)))
    assert out["diag"] == {}


def test_the_thumbnail_sits_on_its_own_stride_grid(builtins, dem):
    out = resolve_output(_printed(n_levels=4), dem, Cache(), "thumbnail")
    ny, nx = dem.values.shape
    assert out["display_stride"] == 2 ** 4
    assert tuple(out["full_dims"]) == (ny, nx)
    assert out["raster"].shape == (math.ceil(ny / 16), math.ceil(nx / 16))
    assert out["raster"].dtype == np.float32
    torus = mz._coarse_for(dem.values, mz.analyze(dem.values, 4), 4)    # unregistered S_J
    _close32(out["raster"], torus[:ny:16, :nx:16])


@pytest.mark.parametrize("algorithm", ["printed", "lastwave"])
def test_a_thumbnail_coarse_on_a_non_divisible_grid_refuses_the_recon(builtins, monkeypatch,
                                                                      algorithm):
    """60 x 60 at J = 4: the thumbnail cannot be pinned (the recon and residual refuse with the
    reason, nothing cached) while the thumbnail itself is drawn with a partial last block."""
    field, cache = _synthetic(60, 60), Cache()
    layer = _layer(algorithm=algorithm, n_levels=4, coarse="thumbnail")
    for name in ("recon", "residual"):
        with pytest.raises(ValueError, match="divisible"):
            resolve_output(layer, field, cache, name)
    r = resolve(layer, field, cache)
    for name in ("recon", "residual"):
        assert output_key("mz_edges", _output(name), r.analysis_params, r.analysis_key) \
            not in cache
    thumb = resolve_output(layer, field, cache, "thumbnail")
    assert thumb["raster"].shape == (4, 4)                 # ceil(60 / 16)
    assert thumb["display_stride"] == 16 and tuple(thumb["full_dims"]) == (60, 60)
    # the same grid reconstructs with the full coarse, and edges-only needs no coarse at all
    full = _layer(algorithm=algorithm, n_levels=4)
    assert resolve_output(full, field, cache, "recon")["raster"].shape == (60, 60)
    assert resolve_output(layer, field, cache, "recon_edges_only")["raster"].shape == (60, 60)


def test_the_fractional_order_left_the_device_and_stays_in_the_core(builtins):
    """The device offers the spline alone; the fractional-order core still reconstructs."""
    dev = get_device("mz_edges")
    for knob in ({"wavelet": "frac_bspline"}, {"alpha": 2.5}):
        with pytest.raises(ValueError, match="unknown parameter"):
            validate_params(dev, {"n_levels": 3, **knob})
    field = _synthetic()
    img, diag = mz.reconstruct(field.values,
                               mz.analyze(field.values, 3, wavelet="frac_bspline", alpha=2.5))
    assert diag["wavelet"] == "frac_bspline"
    assert diag["alpha"] == pytest.approx(2.5)
    assert diag["status"] != "diverging"
    assert np.isfinite(img).all() and img.shape == field.values.shape


def test_progress_reaches_the_reconstruction(builtins):
    seen: list = []
    resolve_output(_printed(n_levels=3, iterations=3), _synthetic(), Cache(), "recon",
                   progress=lambda msg, frac: seen.append((msg, frac)))
    assert ("mz reconstruction 3/3", 1.0) in seen


@pytest.mark.parametrize("algorithm", ["printed", "lastwave"])
def test_a_cancelled_recon_caches_nothing(builtins, algorithm):
    from dynamix.core.wtmm_backend import ComputeCancelled

    field, cache = _synthetic(), Cache()
    layer = _layer(algorithm=algorithm, n_levels=3)
    resolve(layer, field, cache)                          # the analysis lands first
    with pytest.raises(ComputeCancelled):
        resolve_output(layer, field, cache, "recon", cancel=lambda: True)
    r = resolve(layer, field, cache)
    assert output_key("mz_edges", _output("recon"), r.analysis_params, r.analysis_key) \
        not in cache
    assert r.analysis_key in cache


def test_an_unknown_output_name_is_refused_by_the_device(builtins):
    dev = get_device("mz_edges")
    field = _synthetic()
    params = {p.name: p.default for p in dev.params}
    result = dev.compute(field, {**params, "n_levels": 3})
    with pytest.raises(ValueError, match="nope"):
        dev.compute_output("nope", field.values, result, params, fetch=None)


# --------------------------------------------------------------------------- persistence


def test_save_and_reopen_keeps_the_shown_output_and_its_knobs(builtins):
    from dynamix.model.project import Project

    p = Project()
    src = p.add_source("synthetic://mz")
    knobs = {"n_levels": 3, "algorithm": "printed", "show": "recon", "iterations": 7,
             "mode": "set_points", "coarse": "thumbnail", "border": "periodic",
             "colocate_l1": True, "kappa": 2.0, "clip": True, "run_mode": "fixed",
             "tolerance": 1e-4}
    layer = p.add_layer("mz", src.source_id, Chain((DeviceRef("mz_edges", dict(knobs)),)),
                        tags={"ui.edges_hidden": "1"})
    back = Project.from_payload(p.to_payload())
    again = next(l for l in back.layers if l.layer_id == layer.layer_id)
    step = again.chain.steps[0]
    assert step.device == "mz_edges"
    assert {k: step.params[k] for k in knobs} == knobs
    assert again.tags.get("ui.edges_hidden") == "1"
    # lazy outputs are not saved: the reopened layer recomputes the recon when asked
    out = resolve_output(again, _synthetic(), Cache(), "recon")
    assert out["diag"]["n_iter"] == 7 and out["diag"]["mode"] == "set_points"
    assert out["diag"]["coarse"] == "thumbnail"
