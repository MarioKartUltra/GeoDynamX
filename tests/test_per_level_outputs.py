# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Per-scale raster outputs. An output declared ``per_level`` has one cache entry per level of the
analysis's scale stack: ``output_key`` folds the level in and ``resolve_output`` hands it to
``compute_output``; every other output's key and call are unchanged. Each tool computes a level
with its own per-scale routine and the analysis's params, so the level-k raster shows what the
analysis detected at level k: with the NMS detector and Interpolate off, the NMS maxima of
wtmm2d's level-k modulus are its stored level-k extrema, and mz_edges' level-l modulus reads each
stored maximum's own ``mod``. The CDF / PM evolution runs once for every level. A band row of a
multiband stack works as any field does."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache, cache_key
from dynamix.engine.resolve import output_key, resolve, resolve_output
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import (Output, declared_outputs, defaults_for, get_device,
                                  keyed_params, register_device, validate_params)
from dynamix.model.layer import Layer
from dynamix.model.param import Param, ParamKind

FAST_WTMM = {"n_oct": 2, "n_voice": 2}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(values, name="f") -> RasterField:
    ny, nx = values.shape[:2]
    return RasterField(name=name, values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(nx)), y_axis=np.arange(float(ny)))


def _smooth(ny=48, nx=56, seed=0):
    return np.random.default_rng(seed).standard_normal((ny, nx)).cumsum(0).cumsum(1)


def _layer(*steps, tags=None) -> Layer:
    chain = Chain(tuple(DeviceRef(n, dict(p)) for n, p in steps)).materialized()
    return Layer(layer_id=1, name="L", source_id="s", chain=chain, tags=dict(tags or {}))


def _out(device: str, name: str) -> Output:
    return next(o for o in declared_outputs(get_device(device)) if o.name == name)


# --------------------------------------------------------------------- the engine contract


class _Levels:
    """A fake analysis with a per-level output and a plain one."""

    name = "fake_levels"
    params = (Param("k", ParamKind.INT, default=1, min=1, max=9),)
    outputs = (Output("plain", "raster", lazy=True),
               Output("each", "raster", lazy=True, per_level=True))

    def __init__(self):
        self.calls: list = []

    def compute(self, field, params, *, progress=None):
        return {"extrema": [], "chains": [], "scales": np.array([1.0, 2.0, 4.0])}

    def cache_key(self, source_id, params):
        return ""

    def compute_output(self, name, values, result, params, *, fetch, progress=None,
                       cancel=None, **extra):
        self.calls.append((name, extra))
        if name == "plain":
            return {"raster": np.zeros((2, 2), np.float32), "diag": {}}
        return {"raster": np.full((2, 2), extra["level"], np.float32), "diag": {}}


@pytest.fixture
def fake(clean_registry):
    device = _Levels()
    register_device(device)
    return device


def test_output_is_not_per_level_by_default():
    assert Output("x", "raster").per_level is False


def test_a_plain_output_key_ignores_the_level():
    out = Output("plain", "raster", lazy=True, params=("n",))
    want = cache_key("dev/plain", "", {"n": 3}, upstream="A")
    assert output_key("dev", out, {"n": 3}, "A") == want
    assert output_key("dev", out, {"n": 3}, "A", level=5) == want


def test_a_per_level_output_key_folds_the_level_in():
    out = Output("each", "raster", lazy=True, per_level=True)
    keys = {output_key("dev", out, {}, "A", level=k) for k in range(3)}
    assert len(keys) == 3
    assert output_key("dev", out, {}, "A") == output_key("dev", out, {}, "A", level=0)
    assert output_key("dev", out, {}, "A", level=2) == cache_key(
        "dev/each", "", {"_level": 2}, upstream="A")


def test_resolve_output_hands_the_level_to_per_level_outputs_only(fake):
    field, cache = _field(_smooth()), Cache()
    layer = _layer(("fake_levels", {}))
    resolve_output(layer, field, cache, "plain", level=2)
    a = resolve_output(layer, field, cache, "each", level=2)
    b = resolve_output(layer, field, cache, "each", level=1)
    assert fake.calls == [("plain", {}), ("each", {"level": 2}), ("each", {"level": 1})]
    assert float(a["raster"][0, 0]) == 2.0 and float(b["raster"][0, 0]) == 1.0
    misses = cache.misses
    resolve_output(layer, field, cache, "each", level=2)            # its own cached entry
    resolve_output(layer, field, cache, "plain", level=0)           # one entry for every level
    assert cache.misses == misses and len(fake.calls) == 3


def test_no_level_is_level_zero(fake):
    field, cache = _field(_smooth()), Cache()
    layer = _layer(("fake_levels", {}))
    resolve_output(layer, field, cache, "each")
    assert fake.calls == [("each", {"level": 0})]
    resolve_output(layer, field, cache, "each", level=0)
    assert len(fake.calls) == 1


# --------------------------------------------------------------------- wtmm2d


def test_wtmm2d_declares_modulus_and_argument_rows_off_by_default(builtins):
    dev = get_device("wtmm2d")
    outs = declared_outputs(dev)
    assert [o.name for o in outs] == ["modulus", "argument"]
    assert all(o.lazy and o.per_level and o.kind == "raster" and o.row for o in outs)
    show = next(p for p in dev.params if p.name == "show")
    assert show.view and show.default == "edges"
    assert show.choices == ("edges", "modulus", "argument")
    assert "show" not in keyed_params(dev, defaults_for(dev))      # the analysis key is unchanged


def test_wtmm2d_analysis_key_does_not_move_with_show(builtins):
    field, cache = _field(_smooth()), Cache()
    a = resolve(_layer(("wtmm2d", FAST_WTMM)), field, cache)
    b = resolve(_layer(("wtmm2d", {**FAST_WTMM, "show": "modulus"})), field, cache)
    assert a.analysis_key == b.analysis_key and b.cache_misses == 0


@pytest.mark.parametrize("detector", ["nms", "follow"])
def test_run_wtmm2d_level_is_that_level_of_the_full_runs_cwt_stage(builtins, monkeypatch,
                                                                   detector):
    """The lifted stack the full run's extrema stage reads, captured where ``run_wtmm2d``
    lifts it, against one level at a time. ``wtmm_levels`` binds the lift at import, so the
    capture sees the full run's one call only."""
    from dynamix.core import wtmm_backend
    from dynamix.core.wtmm_levels import run_wtmm2d_level

    lifted = []
    real = wtmm_backend._apply_fracint2d

    def capture(cwt, scales, alpha):
        out = real(cwt, scales, alpha)
        lifted.append(out)
        return out

    monkeypatch.setattr(wtmm_backend, "_apply_fracint2d", capture)
    values = _smooth()
    params = {"n_oct": 2, "n_voice": 2, "a_min": 1.0, "wavelet": "mexican",
              "fracint_alpha": 0.7, "detector": detector}
    scales = wtmm_backend.run_wtmm2d(_field(values), params)["scales"]
    (stage,) = lifted
    for k in range(len(scales)):
        one = run_wtmm2d_level(values, params, k)
        assert one["scale"] == float(scales[k])
        np.testing.assert_array_equal(one["mod"], stage["mod"][k])
        np.testing.assert_array_equal(one["arg"], stage["arg"][k])
    assert len(lifted) == 1
    with pytest.raises(ValueError, match="outside"):
        run_wtmm2d_level(values, params, len(scales))


@pytest.mark.parametrize("extra", [{}, {"dither": True}, {"fracint_alpha": 0.0},
                                   {"wavelet": "gaussian"}])
def test_the_nms_maxima_of_the_level_k_modulus_are_the_stored_level_k_extrema(builtins,
                                                                              extra):
    from dynamix.core.wtmm_backend import _nms_extrema_scale

    values = np.round(_smooth() * 4.0) / 4.0           # a measurable value lattice (dither)
    field, cache = _field(values), Cache()
    layer = _layer(("wtmm2d", {**FAST_WTMM, **extra}))
    r = resolve(layer, field, cache)
    raw = cache.get(r.analysis_key)
    params = r.analysis_params
    for k, stored in enumerate(raw["extrema"]):
        mod = resolve_output(layer, field, cache, "modulus", level=k)
        arg = resolve_output(layer, field, cache, "argument", level=k)
        assert mod["raster"].dtype == np.float32 and mod["raster"].shape == values.shape
        assert mod["diag"] == {"level": k, "scale_px": float(raw["scales"][k])}
        got = _nms_extrema_scale(mod["raster"].astype(np.float64),
                                 arg["raster"].astype(np.float64), thresh=params["thresh"],
                                 radius=int(np.ceil(raw["scales"][k])))
        np.testing.assert_array_equal(got["x"], stored["x"])
        np.testing.assert_array_equal(got["y"], stored["y"])
        np.testing.assert_array_equal(got["mod"], stored["mod"])


def test_a_level_past_the_stack_reads_the_coarsest(builtins):
    field, cache = _field(_smooth()), Cache()
    layer = _layer(("wtmm2d", FAST_WTMM))
    out = resolve_output(layer, field, cache, "modulus", level=99)
    assert out["diag"]["level"] == 3


def test_a_band_row_of_a_stack_gets_its_bands_levels(builtins):
    """A multiband stack's band row (band_select ahead of wtmm2d) is a Field like any other: the
    level raster is that band's, and a band removed from the dataset re-keys it."""
    from dynamix.core.wtmm_levels import run_wtmm2d_level

    stack = np.stack([_smooth(seed=s) for s in (1, 2, 3)], axis=-1)
    field, cache = _field(stack), Cache()
    steps = (("band_select", {"band": 2}), ("wtmm2d", FAST_WTMM))
    layer = _layer(*steps)
    out = resolve_output(layer, field, cache, "modulus", level=1)
    want = run_wtmm2d_level(stack[..., 1], {"n_oct": 2, "n_voice": 2, "a_min": 1.0,
                                            "wavelet": "mexican", "fracint_alpha": 1.0}, 1)
    np.testing.assert_array_equal(out["raster"], want["mod"])
    reduced = _layer(*steps, tags={"data.bands": "b1,b2"})
    r1 = resolve(layer, field, cache)
    r2 = resolve(reduced, field, cache)
    o = _out("wtmm2d", "modulus")
    assert (output_key("wtmm2d", o, r1.analysis_params, r1.analysis_key, level=1)
            != output_key("wtmm2d", o, r2.analysis_params, r2.analysis_key, level=1))


# --------------------------------------------------------------------- wtmm2d_roi


def test_wtmm2d_roi_level_reads_its_stored_maxima_on_the_roi_grid(builtins, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    n = 80
    values = _smooth(n, n, seed=5).astype(np.float32)
    path = str(tmp_path / "parent.tif")
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="float32", crs="EPSG:32615",
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(values, 1)
    field = RasterField.from_geotiff_window(path, row_off=0, col_off=0, height=n, width=n)
    roi = {"n_oct": 1, "n_voice": 2, "roi_row": 20, "roi_col": 24, "roi_h": 30, "roi_w": 28}
    grid = RasterField.from_geotiff_window(path, row_off=20, col_off=24, height=30, width=28)
    for boundary in ("auto", "reflective"):
        cache = Cache()
        layer = _layer(("wtmm2d_roi", {**roi, "boundary": boundary}))
        r = resolve(layer, field, cache)
        raw = cache.get(r.analysis_key)
        for k, stored in enumerate(raw["extrema"]):
            out = resolve_output(layer, field, cache, "modulus", level=k)
            assert out["raster"].shape == (30, 28)
            np.testing.assert_array_equal(out["raster"][stored["y"], stored["x"]],
                                          stored["mod"].astype(np.float32))
            np.testing.assert_array_equal(out["roi_axes"][0], grid.x_axis)
            np.testing.assert_array_equal(out["roi_axes"][1], grid.y_axis)


# --------------------------------------------------------------------- mz_edges


@pytest.mark.parametrize("extra", [{}, {"border": "periodic"}, {"colocate_l1": True},
                                   {"algorithm": "printed"},
                                   {"algorithm": "printed", "dither": True}])
def test_mz_level_modulus_reads_each_stored_maximum(builtins, extra):
    values = np.round(_smooth(64, 56) * 8.0) / 8.0
    field, cache = _field(values), Cache()
    layer = _layer(("mz_edges", {"n_levels": 3, **extra}))
    r = resolve(layer, field, cache)
    raw = cache.get(r.analysis_key)
    for idx, stored in enumerate(raw["extrema"]):
        mod = resolve_output(layer, field, cache, "modulus", level=idx)
        arg = resolve_output(layer, field, cache, "argument", level=idx)
        assert mod["diag"] == {"level": idx, "scale_px": float(2 ** (idx + 1))}
        np.testing.assert_array_equal(mod["raster"][stored["y"], stored["x"]],
                                      stored["mod"].astype(np.float32))
        np.testing.assert_array_equal(arg["raster"][stored["y"], stored["x"]],
                                      stored["arg"].astype(np.float32))
    lastwave = r.analysis_params["algorithm"] == "lastwave"
    assert ("display_offset" in mod) is lastwave


def test_mz_level_rows_join_the_declared_outputs(builtins):
    dev = get_device("mz_edges")
    by = {o.name: o for o in declared_outputs(dev)}
    assert by["modulus"].per_level and by["argument"].per_level
    assert not any(o.per_level for n, o in by.items() if n not in ("modulus", "argument"))
    show = next(p for p in dev.params if p.name == "show")
    assert show.choices[-2:] == ("modulus", "argument")


# --------------------------------------------------------------------- cdf_edges / pm_edges


@pytest.mark.parametrize("variant", ["linear", "nonlinear"])
def test_cdf_snapshots_are_the_analysis_evolution(builtins, variant):
    from dynamix.core import cdf as lcdf
    from dynamix.devices.cdf_edges import CDFEdges

    values = _smooth(40, 48, seed=3)
    values = (values - values.mean()) / values.std()
    field, cache = _field(values), Cache()
    layer = _layer(("cdf_edges", {"n_levels": 3, "variant": variant}))
    r = resolve(layer, field, cache)
    raw = cache.get(r.analysis_key)
    snaps = CDFEdges()._snapshots(values, r.analysis_params)
    np.testing.assert_array_equal(snaps[-1].real, raw["filtered"])
    np.testing.assert_array_equal(snaps[-1].imag, raw["edge_channel"])
    for k, snap in enumerate(snaps):                       # edge_mode "mz": grad-Re NMS
        mask, _coeff = lcdf.grad_nms(snap.real, float(r.analysis_params["floor"]))
        ys, xs = np.where(mask)
        np.testing.assert_array_equal(xs, raw["extrema"][k]["x"])
        np.testing.assert_array_equal(ys, raw["extrema"][k]["y"])
    re = resolve_output(layer, field, cache, "re_scale", level=2)
    im = resolve_output(layer, field, cache, "im_scale", level=2)
    np.testing.assert_array_equal(re["raster"], raw["filtered"].astype(np.float32))
    np.testing.assert_array_equal(im["raster"], raw["edge_channel"].astype(np.float32))
    assert re["diag"] == {"level": 2, "scale_px": 4.0}


def test_cdf_evolves_once_for_every_level_and_channel(builtins, monkeypatch):
    from dynamix.devices.cdf_edges import CDFEdges

    runs = []
    real = CDFEdges._snapshots

    def counted(self, *a, **k):
        runs.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(CDFEdges, "_snapshots", counted)
    field, cache = _field(_smooth(40, 48, seed=3)), Cache()
    layer = _layer(("cdf_edges", {"n_levels": 3}))
    for level in range(3):
        resolve_output(layer, field, cache, "re_scale", level=level)
        resolve_output(layer, field, cache, "im_scale", level=level)
    assert runs == [1]
    snaps = resolve_output(layer, field, cache, "snapshots")
    assert snaps["raster"].shape == (3, 2, 40, 48) and snaps["raster"].dtype == np.float32


def test_cdf_show_of_a_lazy_row_leaves_the_display_to_the_window(builtins):
    from dynamix.devices.cdf_edges import CDFEdges

    dev = CDFEdges()
    params = validate_params(dev, {"n_levels": 2})
    res = dev.compute(_field(_smooth(32, 32)), params)
    for show in ("re_scale", "im_scale"):
        out = dev.view(res, dict(params, show=show))
        assert "raster_out" not in out and out["params"]["show"] == show


def test_pm_snapshots_are_the_analysis_evolution_and_run_once(builtins, monkeypatch):
    from dynamix.devices.pm_edges import PMEdges

    runs = []
    real = PMEdges._snapshots

    def counted(self, *a, **k):
        runs.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(PMEdges, "_snapshots", counted)
    values = _smooth(40, 48, seed=3)
    values = (values - values.mean()) / values.std()
    field, cache = _field(values), Cache()
    layer = _layer(("pm_edges", {"n_levels": 3}))
    r = resolve(layer, field, cache)
    raw = cache.get(r.analysis_key)
    outs = [resolve_output(layer, field, cache, "filtered_scale", level=k) for k in range(3)]
    assert runs == [1]
    np.testing.assert_array_equal(outs[-1]["raster"], raw["filtered"].astype(np.float32))
    assert [o["diag"]["scale_px"] for o in outs] == [1.0, 2.0, 4.0]
    dev = get_device("pm_edges")
    assert [o.name for o in declared_outputs(dev) if o.row] == ["edges", "filtered",
                                                                 "filtered_scale"]
