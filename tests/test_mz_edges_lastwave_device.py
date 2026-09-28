# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""mz_edges on the LastWave engine, through the engine: the knob surface, the analysis bundle,
the lazy outputs (coarse, thumbnail, the one-iteration preview, the reconstruction continued from
it, the residual), the refusals, the ROI margin, and the printed-algorithm path kept selectable."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from dynamix.core import mz_edges as mz
from dynamix.core import mz_lastwave as lw
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import output_key, resolve, resolve_output
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import (declared_outputs, defaults_for, get_device, keyed_params,
                                  validate_params)
from dynamix.model.layer import Layer
from dynamix.model.param import ParamKind

NY, NX, J = 96, 80, 3

_NAMES = ["n_levels", "algorithm", "border", "colocate_l1", "dither", "interpolate",
          "recon_live", "kappa", "clip", "run_mode", "iterations", "tolerance", "coarse", "mode",
          "recon_levels", "per_level", "near_radius", "alpha_check", "alpha_tol",
          "alpha_fallback", "show"]
_SELECT_KNOBS = ("recon_levels", "per_level", "near_radius", "alpha_check", "alpha_tol",
                 "alpha_fallback")
_RECON_KNOBS = ("kappa", "clip", "run_mode", "iterations", "tolerance", "coarse", "mode")
#: The Reconstruction section's knobs: Live (when work is dispatched, keyed nowhere) and the
#: settings a reconstruction reads.
_SECTION_KNOBS = ("recon_live",) + _RECON_KNOBS + _SELECT_KNOBS
_LASTWAVE = ("algorithm", ("lastwave",))
_PRINTED = ("algorithm", ("printed",))


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(ny: int = NY, nx: int = NX, seed: int = 0, values=None) -> RasterField:
    if values is None:
        values = np.random.default_rng(seed).standard_normal((ny, nx)).cumsum(0).cumsum(1)
    ny, nx = values.shape
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(nx)), y_axis=np.arange(float(ny)))


def _layer(**params) -> Layer:
    return Layer(layer_id=1, name="mz", source_id="s",
                 chain=Chain((DeviceRef("mz_edges", {"n_levels": J, **params}),)).materialized())


def _output(name: str):
    return next(o for o in declared_outputs(get_device("mz_edges")) if o.name == name)


def _spy(monkeypatch) -> list:
    """Record every ``mz_lastwave.e2recons`` call: ``(k, mode, state passed, state returned)``."""
    calls: list = []
    real = lw.e2recons

    def spy(*args, **kw):
        out = real(*args, **kw)
        calls.append((kw.get("k"), kw.get("mode"), kw.get("state"), out[2]))
        return out

    monkeypatch.setattr(lw, "e2recons", spy)
    return calls


def _analysis(values, border="mirror", colocate_l1=False):
    return lw.analyze(values, J, border=border, colocate_l1=colocate_l1)


# --------------------------------------------------------------------------- the knob surface


def test_param_names_order_defaults_and_sections(builtins):
    dev = get_device("mz_edges")
    by = {p.name: p for p in dev.params}
    assert [p.name for p in dev.params] == _NAMES
    assert "wavelet" not in by and "alpha" not in by
    assert defaults_for(dev) == {
        "n_levels": 4, "algorithm": "lastwave", "border": "mirror", "colocate_l1": False,
        "dither": False, "interpolate": False, "recon_live": False, "kappa": 1.0, "clip": False,
        "run_mode": "converge", "iterations": 20, "tolerance": 1e-3, "coarse": "full",
        "mode": "separable", "recon_levels": "", "per_level": False, "near_radius": 1,
        "alpha_check": False, "alpha_tol": 0.5, "alpha_fallback": 0.0, "show": "edges"}
    assert by["algorithm"].choices == ("lastwave", "printed")
    assert by["border"].choices == ("mirror", "periodic")
    assert by["colocate_l1"].label == "Co-locate level 1"
    assert (by["recon_live"].kind, by["recon_live"].label) == (ParamKind.BOOL, "Live")
    k = by["kappa"]
    assert (k.min, k.max, k.soft_min, k.soft_max, k.label) == (0.1, 10.0, 0.5, 4.0, "Decay κ")
    assert by["clip"].label == "Clip"
    assert (by["run_mode"].choices, by["run_mode"].label) == (("converge", "fixed"), "Run mode")
    it = by["iterations"]
    assert (it.min, it.max, it.label) == (1, 500, "Iterations")
    tol = by["tolerance"]
    assert (tol.min, tol.max, tol.label) == (1e-6, 1e-1, "Tolerance")
    assert by["coarse"].choices == ("full", "thumbnail")
    assert by["mode"].choices == mz.RECON_MODES
    assert by["show"].choices == ("edges", "coarse", "thumbnail", "recon", "recon_edges_only",
                                  "residual")
    assert {p.name for p in dev.params if p.section == "reconstruction"} == set(_SECTION_KNOBS)
    assert all(p.section == "" for p in dev.params if p.name not in _SECTION_KNOBS)
    assert {p.name for p in dev.params if p.view} == set(_SECTION_KNOBS) | {"show"}
    assert set(keyed_params(dev, defaults_for(dev))) == {
        "n_levels", "algorithm", "border", "colocate_l1", "dither", "interpolate"}


def test_a_knob_that_does_not_apply_to_the_algorithm_says_so(builtins):
    """The context-sensitive rule: each algorithm's own knobs are active only under it."""
    by = {p.name: p for p in get_device("mz_edges").params}
    for name in ("border", "colocate_l1", "recon_live", "kappa", "clip", "run_mode",
                 "tolerance"):
        assert by[name].active_when == _LASTWAVE, name
    for name in ("dither", "interpolate", "mode"):
        assert by[name].active_when == _PRINTED, name
    for name in ("n_levels", "algorithm", "iterations", "coarse", "show"):
        assert by[name].active_when is None, name


def test_the_recon_outputs_key_every_reconstruction_setting(builtins):
    names = [o.name for o in declared_outputs(get_device("mz_edges"))]
    assert names == ["edges", "coarse", "thumbnail", "recon", "recon_edges_only", "residual",
                     "recon_preview"]
    for name in ("recon", "residual"):
        assert set(_output(name).params) == set(_RECON_KNOBS + _SELECT_KNOBS)
    assert (set(_output("recon_edges_only").params)
            == set(_RECON_KNOBS + _SELECT_KNOBS) - {"coarse"})
    assert _output("recon_preview").params == ("kappa", "clip", "coarse") + _SELECT_KNOBS
    assert all(_output(n).selects for n in ("recon", "recon_edges_only", "residual",
                                            "recon_preview"))
    assert not any(_output(n).selects for n in ("edges", "coarse", "thumbnail"))
    assert _output("recon_preview").lazy and _output("recon_preview").kind == "raster"
    # Live chooses when work is dispatched and changes no result: no output reads it
    assert all("recon_live" not in o.params for o in declared_outputs(get_device("mz_edges")))


@pytest.mark.parametrize("name,value", [("kappa", 2.0), ("clip", True), ("run_mode", "fixed"),
                                        ("iterations", 5), ("tolerance", 1e-2),
                                        ("coarse", "thumbnail"), ("mode", "set_points"),
                                        ("recon_live", False)])
def test_flipping_a_reconstruction_knob_is_an_analysis_cache_hit(builtins, name, value):
    field, cache = _field(), Cache()
    first = resolve(_layer(), field, cache)
    again = resolve(_layer(**{name: value}), field, cache)
    assert first.cache_misses == 1
    assert again.cache_misses == 0 and again.cache_hits == 1
    assert again.analysis_key == first.analysis_key
    assert again.result["extrema"] is first.result["extrema"]
    assert again.result["params"][name] == value


@pytest.mark.parametrize("name,value", [("algorithm", "printed"), ("border", "periodic"),
                                        ("colocate_l1", True), ("n_levels", 2)])
def test_an_analysis_knob_rekeys_the_analysis(builtins, name, value):
    field, cache = _field(), Cache()
    first = resolve(_layer(), field, cache)
    again = resolve(_layer(**{name: value}), field, cache)
    assert again.cache_misses == 1 and again.analysis_key != first.analysis_key


# --------------------------------------------------------------------------- the analysis


@pytest.mark.parametrize("border,colocate", [("mirror", False), ("periodic", False),
                                             ("mirror", True)])
def test_the_bundle_is_the_lastwave_extrema_on_the_fields_grid(builtins, border, colocate):
    field = _field()
    res = resolve(_layer(border=border, colocate_l1=colocate), field, Cache()).result
    t, ex = _analysis(field.values, border, colocate)
    assert res["algorithm"] == "lastwave"
    assert res["chains"] == []
    assert res["_shape"] == (NY, NX) and res["_frame"] is field.frame
    assert res["_display_offset"] == (-0.5, -0.5)
    np.testing.assert_array_equal(res["scales"], 2.0 ** np.arange(1, J + 1))
    assert res["scales"].dtype == np.float64
    assert len(res["extrema"]) == J
    for l, level in enumerate(res["extrema"], start=1):
        mask, mag, arg = lw.primary_extrema(t, ex, l)
        y, x = np.nonzero(mask)
        assert level["x"].dtype == np.int64 and level["y"].dtype == np.int64
        np.testing.assert_array_equal(level["x"], x)
        np.testing.assert_array_equal(level["y"], y)
        np.testing.assert_array_equal(level["mod"], mag[mask])
        np.testing.assert_array_equal(level["arg"], arg[mask])
        assert level["line_id"].dtype == np.int64 and level["line_id"].shape == x.shape
        assert x.size > 0
    # the lazy outputs' inputs alone: the working field's scaled S_J and its extrema
    kept_s, kept_ex = res["_lastwave"]
    assert kept_s.shape == t.full_shape
    np.testing.assert_array_equal(kept_s, t.S_full[J])
    assert kept_ex.J == J
    for l in range(1, J + 1):
        np.testing.assert_array_equal(kept_ex.mask[l], ex.mask[l])


def test_line_ids_are_the_8_connected_components_of_each_level(builtins):
    """Points of one 8-connected component of the level's mask on the field's grid share an id,
    distinct components never do, and a component of one point is -1 (not a line)."""
    from scipy import ndimage

    res = resolve(_layer(), _field(), Cache()).result
    _s, ex = res["_lastwave"]
    for l, level in enumerate(res["extrema"], start=1):
        mask = ex.mask[l][:NY, :NX]
        lab, _n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        comp = lab[level["y"], level["x"]]
        sizes = np.bincount(lab.ravel())
        single = sizes[comp] == 1
        assert np.all(level["line_id"][single] == -1)
        lines = level["line_id"][~single]
        assert np.all(lines >= 0)
        pairs = set(zip(comp[~single].tolist(), lines.tolist()))
        assert len(pairs) == len({c for c, _ in pairs}) == len({i for _, i in pairs})


def test_a_nodata_pixel_is_refused_with_the_count(builtins):
    values = _field().values.copy()
    values[10, 12] = np.nan
    values[40, 3] = np.inf
    for algorithm in ("lastwave", "printed"):
        with pytest.raises(ValueError, match="2 nodata"):
            resolve(_layer(algorithm=algorithm), _field(values=values), Cache())


def test_a_grid_too_small_for_j_is_refused(builtins):
    with pytest.raises(ValueError, match=r"2\*\*J"):
        resolve(_layer(n_levels=5), _field(ny=24, nx=80), Cache())


def test_a_missing_numba_names_it(builtins, monkeypatch):
    monkeypatch.setattr(lw._kernels, "_KERNELS", None)
    monkeypatch.setitem(sys.modules, "numba", None)
    with pytest.raises(RuntimeError, match="numba"):
        resolve(_layer(), _field(), Cache())


def test_printed_reproduces_the_printed_algorithm_bundle(builtins):
    field = _field()
    res = resolve(_layer(algorithm="printed", dither=True, interpolate=True), field,
                  Cache()).result
    ref = mz.analyze(field.values, J, coarse="full", dither=True, interpolate=True)
    assert res["algorithm"] == "printed"
    assert "_display_offset" not in res and "_lastwave" not in res
    assert res["wavelet"] == "mz_spline" and res["lsb"] == ref["lsb"]
    np.testing.assert_array_equal(res["scales"], ref["scales"])
    for got, want in zip(res["extrema"], ref["extrema"]):
        assert set(got) == set(want)
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])
    for got, want in zip(res["mz_maxima"], ref["mz_maxima"]):
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- the outputs


def test_the_coarse_output_of_a_constant_field_is_the_constant(builtins):
    field = _field(values=np.full((NY, NX), 7.25))
    out = resolve_output(_layer(), field, Cache(), "coarse")
    assert out["raster"].dtype == np.float32 and out["raster"].shape == (NY, NX)
    np.testing.assert_allclose(out["raster"], 7.25, rtol=1e-6)
    assert out["display_offset"] == (-0.5, -0.5)


@pytest.mark.parametrize("border", ["mirror", "periodic"])
def test_coarse_and_thumbnail_are_s_j_in_the_fields_units(builtins, border):
    field, cache = _field(), Cache()
    t, _ex = _analysis(field.values, border)
    want = t.S[J] / lw.fact(J)
    coarse = resolve_output(_layer(border=border), field, cache, "coarse")
    np.testing.assert_array_equal(coarse["raster"], want.astype(np.float32))
    assert coarse["diag"] == {}
    thumb = resolve_output(_layer(border=border), field, cache, "thumbnail")
    step = 2 ** J
    np.testing.assert_array_equal(thumb["raster"], want[::step, ::step].astype(np.float32))
    assert thumb["display_stride"] == step and tuple(thumb["full_dims"]) == (NY, NX)
    assert thumb["display_offset"] == (-0.5, -0.5)


@pytest.mark.parametrize("knobs", [
    {},
    {"run_mode": "fixed", "iterations": 7, "kappa": lw.KAPPA_LASTWAVE, "clip": True},
    {"run_mode": "converge", "iterations": 40, "tolerance": 1e-2, "coarse": "thumbnail",
     "border": "periodic"},
    {"run_mode": "fixed", "iterations": 1, "kappa": 2.5, "colocate_l1": True},
])
def test_recon_equals_e2recons_for_the_same_knobs(builtins, knobs):
    field = _field()
    out = resolve_output(_layer(**knobs), field, Cache(), "recon")
    p = validate_params(get_device("mz_edges"), {"n_levels": J, **knobs})
    t, ex = _analysis(field.values, p["border"], p["colocate_l1"])
    ref, diag, _state = lw.e2recons(field.values, ex, t.S_full[J], J, k=p["iterations"],
                                    kappa=p["kappa"], clip=p["clip"], coarse=p["coarse"],
                                    border=p["border"], mode=p["run_mode"], tol=p["tolerance"])
    assert out["raster"].dtype == np.float32 and out["raster"].shape == (NY, NX)
    np.testing.assert_array_equal(out["raster"], ref.astype(np.float32))
    assert out["diag"]["iterations"] == diag["iterations"]
    assert out["diag"]["stop"] == diag["stop"]
    assert out["diag"]["snr_db"] == pytest.approx(diag["snr_db"], rel=1e-12)
    assert out["state"].iterations == diag["iterations"]
    assert "display_offset" not in out                     # the recon is on the input grid


def test_the_preview_is_the_initial_pass_and_one_iteration(builtins, monkeypatch):
    calls = _spy(monkeypatch)
    field, cache = _field(), Cache()
    prev = resolve_output(_layer(kappa=2.0, clip=True), field, cache, "recon_preview")
    assert [(k, mode, state) for k, mode, state, _new in calls] == [(1, "fixed", None)]
    t, ex = _analysis(field.values)
    ref, diag, _s = lw.e2recons(field.values, ex, t.S_full[J], J, k=1, kappa=2.0, clip=True)
    np.testing.assert_array_equal(prev["raster"], ref.astype(np.float32))
    assert prev["diag"]["iterations"] == 1 and prev["state"].iterations == 1
    # keyed on kappa / clip / coarse alone: a run-setting change keeps the preview's key
    r1 = resolve(_layer(kappa=2.0, clip=True), field, cache)
    r2 = resolve(_layer(kappa=2.0, clip=True, iterations=9, run_mode="fixed", tolerance=1e-2),
                 field, cache)
    k1 = output_key("mz_edges", _output("recon_preview"), r1.analysis_params, r1.analysis_key)
    k2 = output_key("mz_edges", _output("recon_preview"), r2.analysis_params, r2.analysis_key)
    assert k1 == k2 and k1 in cache
    r3 = resolve(_layer(kappa=2.5, clip=True), field, cache)
    assert output_key("mz_edges", _output("recon_preview"), r3.analysis_params,
                      r3.analysis_key) != k1


def test_run_continues_from_the_cached_preview(builtins, monkeypatch):
    """Iterations is the total in both run modes: a fixed run adds what the preview has not
    done, a converge run caps the total; either way the image is the one-call result."""
    calls = _spy(monkeypatch)
    field, cache = _field(), Cache()
    prev = resolve_output(_layer(run_mode="fixed", iterations=6), field, cache, "recon_preview")
    fixed = resolve_output(_layer(run_mode="fixed", iterations=6), field, cache, "recon")
    assert len(calls) == 2
    k, mode, state, _new = calls[1]
    assert (k, mode) == (5, "fixed") and state is prev["state"]
    assert fixed["diag"]["iterations"] == 6
    conv = resolve_output(_layer(run_mode="converge", iterations=30), field, cache, "recon")
    assert len(calls) == 3                                  # the preview is a cache hit
    k, mode, state, _new = calls[2]
    assert (k, mode) == (30, "converge") and state is prev["state"]
    t, ex = _analysis(field.values)
    for out, run in ((fixed, dict(k=6, mode="fixed")), (conv, dict(k=30, mode="converge"))):
        ref, diag, _s = lw.e2recons(field.values, ex, t.S_full[J], J, **run)
        np.testing.assert_array_equal(out["raster"], ref.astype(np.float32))
        assert out["diag"]["iterations"] == diag["iterations"]


def test_flipping_live_is_a_cache_hit_for_every_output(builtins, monkeypatch):
    """Live only chooses when the window dispatches work: every output keeps its key, so
    nothing already cached is computed again."""
    calls = _spy(monkeypatch)
    field, cache = _field(), Cache()
    names = ("coarse", "thumbnail", "recon_preview", "recon", "recon_edges_only", "residual")
    live = {n: resolve_output(_layer(), field, cache, n) for n in names}
    ran = len(calls)
    r1 = resolve(_layer(), field, cache)
    r2 = resolve(_layer(recon_live=False), field, cache)
    assert r2.cache_misses == 0 and r2.analysis_key == r1.analysis_key
    for n in names:
        assert (output_key("mz_edges", _output(n), r2.analysis_params, r2.analysis_key)
                == output_key("mz_edges", _output(n), r1.analysis_params, r1.analysis_key))
        manual = resolve_output(_layer(recon_live=False), field, cache, n)
        np.testing.assert_array_equal(manual["raster"], live[n]["raster"])
    assert len(calls) == ran


def test_a_recon_without_a_preview_lands_the_preview_first(builtins):
    field, cache = _field(), Cache()
    resolve_output(_layer(), field, cache, "recon")
    r = resolve(_layer(), field, cache)
    assert output_key("mz_edges", _output("recon_preview"), r.analysis_params,
                      r.analysis_key) in cache


def test_running_again_on_a_converged_result_returns_it_unchanged(builtins):
    field, cache = _field(), Cache()
    layer = _layer(run_mode="converge", iterations=200, tolerance=1e-2)
    first = resolve_output(layer, field, cache, "recon")
    assert first["diag"]["stop"] in ("converged", "residual rising")
    again = resolve_output(layer, field, cache, "recon")
    assert again is first


def test_recon_edges_only_pins_a_zero_coarse(builtins):
    field = _field()
    out = resolve_output(_layer(coarse="thumbnail", run_mode="fixed", iterations=4), field,
                         Cache(), "recon_edges_only")
    t, ex = _analysis(field.values)
    ref, diag, _s = lw.e2recons(field.values, ex, t.S_full[J], J, k=4, coarse="none")
    np.testing.assert_array_equal(out["raster"], ref.astype(np.float32))
    assert out["diag"]["coarse"] == "none" and out["diag"]["iterations"] == 4


def test_residual_plus_recon_is_the_field(builtins, monkeypatch):
    calls = _spy(monkeypatch)
    field, cache = _field(), Cache()
    layer = _layer(run_mode="fixed", iterations=3)
    recon = resolve_output(layer, field, cache, "recon")
    n = len(calls)
    resid = resolve_output(layer, field, cache, "residual")
    assert len(calls) == n                                  # the recon is reused
    assert resid["raster"].dtype == np.float32
    want = np.asarray(field.values, dtype=np.float64)
    got = resid["raster"].astype(np.float64) + recon["raster"]
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6 * np.abs(want).max())
    assert resid["diag"] == recon["diag"]


def test_the_thumbnail_coarse_needs_a_divisible_grid(builtins):
    field, cache = _field(ny=60, nx=60), Cache()
    layer = _layer(n_levels=4, coarse="thumbnail")
    for name in ("recon_preview", "recon", "residual"):
        with pytest.raises(ValueError, match="divisible"):
            resolve_output(layer, field, cache, name)
    r = resolve(layer, field, cache)
    for name in ("recon_preview", "recon", "residual"):
        assert output_key("mz_edges", _output(name), r.analysis_params, r.analysis_key) \
            not in cache
    assert resolve_output(layer, field, cache, "recon_edges_only")["raster"].shape == (60, 60)


def test_progress_and_cancel_reach_the_reconstruction(builtins):
    from dynamix.core.wtmm_backend import ComputeCancelled

    seen: list = []
    resolve_output(_layer(run_mode="fixed", iterations=3), _field(), Cache(), "recon",
                   progress=lambda msg, frac: seen.append((msg, frac)))
    assert any(msg.startswith("M–Z reconstruction") and frac == pytest.approx(1.0)
               for msg, frac in seen)
    field, cache = _field(), Cache()
    resolve(_layer(), field, cache)
    with pytest.raises(ComputeCancelled):
        resolve_output(_layer(), field, cache, "recon", cancel=lambda: True)
    r = resolve(_layer(), field, cache)
    for name in ("recon", "recon_preview"):
        assert output_key("mz_edges", _output(name), r.analysis_params, r.analysis_key) \
            not in cache


def test_the_printed_algorithm_keeps_its_own_reconstruction(builtins):
    field = _field()
    layer = _layer(algorithm="printed", iterations=4, mode="set_points")
    out = resolve_output(layer, field, Cache(), "recon")
    ref, diag = mz.reconstruct(field.values, mz.analyze(field.values, J), n_iter=4,
                               mode="set_points")
    np.testing.assert_array_equal(out["raster"], ref.astype(np.float32))
    assert out["diag"]["n_iter"] == 4 and out["diag"]["mode"] == "set_points"
    assert out["diag"]["iterations"] == 4 and out["diag"]["stop"] == "fixed"
    with pytest.raises(ValueError, match="lastwave"):
        resolve_output(layer, field, Cache(), "recon_preview")


# --------------------------------------------------------------------------- the ROI margin


def _fir_reach(name: str, scale: int) -> int:
    """How far one filter of ``transform.FILTERS`` reads at dilation ``scale`` (px, one side)."""
    from dynamix.core.mz_lastwave.transform import FILTERS, _l1r1

    size, shift, _sym, _f = FILTERS[name]
    l1, r1 = _l1r1(shift, scale)
    return max(l1, r1) + (size - 2) * scale if size > 1 else 0


@pytest.mark.parametrize("n_levels", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("colocate", [False, True])
def test_the_lastwave_margin_is_the_fir_reach_plus_the_detection_probe(builtins, n_levels,
                                                                       colocate):
    dev = get_device("mz_edges")
    params = validate_params(dev, {"n_levels": n_levels, "colocate_l1": colocate})
    reach = sum(_fir_reach("H1", 2 ** (l - 1)) for l in range(1, n_levels)) \
        + _fir_reach("G1", 2 ** (n_levels - 1))
    level1 = _fir_reach("G1", 1) + 2 + int(colocate)
    assert dev.roi_margin(params) == max(reach + 1, level1)


def test_the_printed_margin_is_the_measured_impulse_reach(builtins):
    dev = get_device("mz_edges")
    params = validate_params(dev, {"n_levels": 3, "algorithm": "printed"})
    assert dev.roi_margin(params) == mz.mz_impulse_reach(3) + 2


@pytest.mark.parametrize("border,colocate", [("mirror", False), ("periodic", False),
                                             ("mirror", True)])
@pytest.mark.parametrize("n_levels", [1, 2, 3])
def test_an_interior_roi_reproduces_the_whole_field(builtins, border, colocate, n_levels):
    """The margin keeps the window's border (mirror seam or periodic wrap) out of the ROI."""
    from dynamix.roi.runner import crop_result_to_roi, run_on_region, split_lines

    dev = get_device("mz_edges")
    params = validate_params(dev, {"n_levels": n_levels, "border": border,
                                   "colocate_l1": colocate})
    m = dev.roi_margin(params)
    n = 2 * (m + 8) + 48
    rng = np.random.default_rng(11)
    v = rng.standard_normal((n, n)).cumsum(0).cumsum(1)
    v[:, n // 2:] += 40.0 * v.std() / n
    f = _field(values=v)
    roi = (m + 8, m + 8, 48, 48)
    ref = crop_result_to_roi(dev.compute(f, params), roi[0], 48, 48, (n, n))
    for lvl in ref["extrema"]:
        lvl["line_id"] = split_lines(lvl)
    got = run_on_region([(dev, params)], f, roi)
    assert len(ref["extrema"]) == len(got["extrema"]) == n_levels
    for a, b in zip(ref["extrema"], got["extrema"]):
        pa = set(zip(a["y"].tolist(), a["x"].tolist()))
        pb = set(zip(b["y"].tolist(), b["x"].tolist()))
        assert pa == pb, f"{len(pa ^ pb)} positions differ"
        order_a = np.lexsort((a["x"], a["y"]))
        order_b = np.lexsort((b["x"], b["y"]))
        np.testing.assert_array_equal(a["mod"][order_a], b["mod"][order_b])
