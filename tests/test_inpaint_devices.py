# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""inpaint and fill as devices: through the engine's resolve, on a stack, on an ROI, with
their rows, readings and stamps."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("pywt")
pytest.importorskip("scipy")

from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402
from dynamix.devices.inpaint import PARAMS, Fill, Inpaint  # noqa: E402
from dynamix.engine.cache import Cache  # noqa: E402
from dynamix.engine.resolve import resolve, resolve_output  # noqa: E402
from dynamix.model.chain import Chain, DeviceRef  # noqa: E402
from dynamix.model.device import defaults_for, register_device, validate_params  # noqa: E402
from dynamix.model.layer import Layer  # noqa: E402


def _field(values, name="f", x0=500015.0, **prov):
    ny, nx = values.shape[:2]
    return RasterField(name=name, values=values, frame=LocalFrame(x0=x0, y0=3999985.0,
                                                                   dx=30.0, dy=30.0, units="m"),
                       x_axis=x0 + 30.0 * np.arange(nx), y_axis=3999985.0 - 30.0 * np.arange(ny),
                       provenance={"crs": "EPSG:32615", "source": f"/{name}.tif", **prov})


def _gapped(seed=0, shape=(64, 64)):
    v = np.random.default_rng(seed).standard_normal(shape).cumsum(0).cumsum(1)
    y = v.copy()
    y[20:36, 24:40] = np.nan
    return v, y


FAST = {"n_iter": 3, "heldout_blocks": 2, "heldout_side": 4}


@pytest.fixture
def devices(clean_registry):
    register_device(Inpaint())
    register_device(Fill())

    class _Reader:
        """A stand-in analyzer: it reports what it received."""

        name = "fake_reader"
        params = ()

        def compute(self, field, params, *, progress=None):
            return {"extrema": [], "chains": [], "got": np.asarray(field.values),
                    "name": field.name}

        def cache_key(self, source_id, params):
            return ""

    register_device(_Reader())
    return clean_registry


def _layer(*steps, tags=None):
    return Layer(layer_id=1, name="L", source_id="s",
                 chain=Chain(tuple(DeviceRef(n, dict(p)) for n, p in steps)).materialized(),
                 tags=dict(tags or {}))


def test_inpaint_resolves_fills_and_flips_its_view_without_recomputing(devices):
    _truth, y = _gapped()
    field, cache = _field(y), Cache()
    r = resolve(_layer(("inpaint", FAST)), field, cache)
    assert r.cache_misses == 1 and r.analysis_device == "inpaint"
    res = r.result
    known = np.isfinite(y)
    np.testing.assert_array_equal(res["raster_out"][known], y[known].astype(res["raster_out"].dtype))
    assert np.isfinite(res["raster_out"]).all() and res["raster_out"].dtype == np.float32   # _store at 32 bit
    assert res["_view_note"] == "filled" and res["_shape"] == (64, 64)
    again = resolve(_layer(("inpaint", {**FAST, "show": "residual"})), field, cache)
    assert again.cache_misses == 0
    residual = resolve_output(_layer(("inpaint", FAST)), field, cache, "residual")
    assert np.isnan(residual["raster"][25, 30]) and np.isfinite(residual["raster"][0, 0])
    assert "rms_rel" in residual["diag"]
    held = resolve_output(_layer(("inpaint", FAST)), field, cache, "heldout")
    assert np.isfinite(held["raster"]).sum() == 2 * 16 and held["diag"]["n_blocks"] == 2


def test_the_reading_names_the_route_the_relations_and_the_numbers(devices):
    _truth, y = _gapped()
    dev = Inpaint()
    res = dev.compute(_field(y), validate_params(dev, {**defaults_for(dev), **FAST}))
    text = dev.reading(res, {})
    assert text.startswith("sparsity · 1 gap ·") and "it" in text and "var err" in text and "held-out" in text
    plain = dev.compute(_field(np.ones((32, 32))), validate_params(dev, defaults_for(dev)))
    assert dev.reading(plain, {}) == "no region declared"


def test_a_stack_needs_its_target_band_named_and_fills_that_band_only(devices):
    truth, y = _gapped()
    stack = np.stack([y, truth * 2.0, truth + 1.0], axis=-1)
    field = _field(stack, bands=["b1", "b2", "b3"])
    dev = Inpaint()
    with pytest.raises(ValueError, match="target_band.*b1.*b2.*b3"):
        dev.compute(field, validate_params(dev, {**defaults_for(dev), **FAST}))
    res = dev.compute(field, validate_params(dev, {**defaults_for(dev), **FAST, "target_band": "b1"}))
    assert res["raster_out"].shape == (64, 64) and res["_band_index"] == 0
    with pytest.raises(ValueError, match="target_band"):
        dev.compute(field, validate_params(dev, {**defaults_for(dev), **FAST, "target_band": "b9"}))
    filled = Fill().compute(field, validate_params(Fill(), {**defaults_for(Fill()), "n_iter": 3,
                                                           "target_band": "b1"}))
    assert filled.values.shape == (64, 64, 3)
    np.testing.assert_array_equal(filled.values[..., 1], truth * 2.0)
    assert np.isfinite(filled.values[..., 0]).all()


def test_an_roi_layer_runs_on_the_window_and_crops_to_the_region(devices):
    truth, y = _gapped(shape=(96, 96))
    field, cache = _field(y), Cache()
    layer = _layer(("inpaint", FAST), tags={"roi.window": "10,12,48,50"})
    r = resolve(layer, field, cache)
    res = r.result
    assert res["_shape"] == (48, 50) and res["raster_out"].shape == (48, 50)
    assert res["fill_mask"].shape == (48, 50) and (res["fill_mask"] > 0).any()
    known = np.isfinite(y[10:58, 12:62])
    np.testing.assert_array_equal(res["raster_out"][known],
                                  y[10:58, 12:62][known].astype(res["raster_out"].dtype))


def test_an_roi_inpaint_run_can_be_cancelled(devices):
    from dynamix.core.wtmm_backend import ComputeCancelled
    from dynamix.roi.runner import run_on_region

    _truth, y = _gapped(shape=(96, 96))
    dev = Inpaint()
    with pytest.raises(ComputeCancelled):
        run_on_region([(dev, validate_params(dev, {**defaults_for(dev), "n_iter": 50}))], _field(y),
                      (10, 12, 48, 50), cancel=lambda: True)


def test_inpaint_through_resolve_reads_a_nested_and_a_foreign_send(devices, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    truth, y = _gapped()
    coarse = truth.reshape(32, 2, 32, 2).mean(axis=(1, 3))
    p = tmp_path / "prior.tif"
    with rasterio.open(p, "w", driver="GTiff", height=32, width=32, count=1, dtype="float32",
                       crs="EPSG:32615", transform=Affine(60.0, 0.0, 500000.0, 0.0, -60.0, 4000000.0)) as dst:
        dst.write(coarse.astype(np.float32), 1)
    f = tmp_path / "foreign.tif"
    with rasterio.open(f, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32",
                       crs="EPSG:32615", transform=Affine(30.0, 0.0, 500012.0, 0.0, -30.0, 4000000.0)) as dst:
        dst.write(truth.astype(np.float32), 1)                # 0.4 px off the grid: foreign
    sends = [{"path": str(p), "subdataset": None, "band": None, "label": "prior", "role": "value"},
             {"path": str(f), "subdataset": None, "band": None, "label": "other day", "role": "value"}]
    field, cache = _field(y), Cache()
    r = resolve(_layer(("inpaint", {**FAST, "_sends": json.dumps(sends)})), field, cache)
    rel = r.result["_grid_relations"]
    assert rel["prior"]["kind"] == "nested" and rel["prior"]["ratio"] == 2
    assert rel["other day"]["kind"] == "foreign"
    with pytest.raises(ValueError, match="other day.*foreign"):
        resolve(_layer(("inpaint", {**FAST, "route": "cascade", "_sends": json.dumps(sends)})),
                field, Cache())


def test_fill_passes_all_with_region_source_and_no_polygons_and_clones_otherwise(devices):
    truth, y = _gapped()
    dev = Fill()
    base = defaults_for(dev)
    assert "show" not in base and "heldout_blocks" not in base
    assert dev.pass_all(validate_params(dev, {**base, "mask_source": "region"}))
    assert not dev.pass_all(validate_params(dev, {**base, "mask_source": "nodata"}))
    field = _field(y)
    same = dev.compute(field, validate_params(dev, {**base, "mask_source": "region"}))
    assert same is field
    clone = dev.compute(field, validate_params(dev, {**base, "n_iter": 3}))
    assert clone is not field and clone.name.startswith("f+fill") and len(clone.name) > len("f+fill")
    assert np.isfinite(clone.values).all()
    np.testing.assert_array_equal(clone.values[np.isfinite(y)], y[np.isfinite(y)])
    fill = clone.provenance["fill"]
    assert fill["fill_kind"] == "estimate" and fill["fill_route"] == "sparsity"
    assert fill["filled_fraction"] == pytest.approx(256 / 4096)
    assert set(fill) >= {"grid_relations", "seed", "iterations", "returned", "stop", "variance_error"}
    json.dumps(clone.provenance)
    other = dev.compute(field, validate_params(dev, {**base, "n_iter": 3, "seed": 4}))
    assert other.name != clone.name                         # two fills never share a stage identity
    empty = dev.compute(_field(truth), validate_params(dev, {**base, "n_iter": 3}))
    assert empty.provenance["fill"]["filled_fraction"] == 0.0 and empty.name != "f"
    assert dev.reading(clone, {}).startswith("filled 6 % · sparsity · analysis inside the region reads a reconstruction")


def test_the_engine_stamps_fill_on_the_analyzer_result_downstream(devices):
    from dynamix.model.contracts import fill_of, stamp_fill

    _truth, y = _gapped()
    field, cache = _field(y), Cache()
    r = resolve(_layer(("fill", {"n_iter": 3}), ("fake_reader", {})), field, cache)
    assert r.result["_fill"]["filled_fraction"] == pytest.approx(256 / 4096)
    assert r.result["name"].startswith("f+fill") and np.isfinite(r.result["got"]).all()
    assert Fill().reading(r.result, {}).startswith("filled 6 %")
    plain = resolve(_layer(("fake_reader", {})), field, Cache())
    assert "_fill" not in plain.result
    assert fill_of({"_fill": {"a": 1}}) == {"a": 1} and fill_of(field) is None
    assert stamp_fill({"x": 1}, field) == {"x": 1}
    assert stamp_fill(field, field) is field
    tail = resolve(_layer(("fill", {"n_iter": 3})), field, Cache())
    assert tail.result.provenance["fill"]["fill_route"] == "sparsity"
    roi = resolve(_layer(("fill", {"n_iter": 3}), ("fake_reader", {})), _field(_gapped(shape=(96, 96))[1]),
                  Cache())                                        # whole field; the ROI path:
    layer = _layer(("fill", {"n_iter": 3}), ("fake_reader", {}), tags={"roi.window": "10,12,48,50"})
    roi = resolve(layer, _field(_gapped(shape=(96, 96))[1]), Cache())
    assert roi.result["_fill"]["fill_route"] == "sparsity"


def test_inactive_params_follow_the_sends_and_the_regions(devices):
    dev = Inpaint()
    base = validate_params(dev, defaults_for(dev))
    off = dev.inactive_params(base, [])
    assert {"prior_weight", "structure_weight", "structure_levels", "veil_model", "veil_levels",
            "veil_passes", "sun_azimuth", "sun_elevation"} <= off
    with_value = dev.inactive_params(base, [{"role": "value"}])
    assert "prior_weight" not in with_value and "structure_weight" in with_value
    with_structure = dev.inactive_params(base, [{"role": "structure"}])
    assert "structure_weight" not in with_structure and "structure_levels" not in with_structure
    veiled = dev.inactive_params({**base, "mask_source": "threshold", "threshold_kind": "veil"}, [])
    assert "veil_model" not in veiled and "sun_azimuth" not in veiled
    seeded = dev.inactive_params(base, [{"role": "veil_seed"}])
    assert "veil_model" not in seeded
    assert set(p.name for p in Fill.params) == set(p.name for p in PARAMS) - {"show", "heldout_blocks", "heldout_side"}


def test_cancel_reaches_the_loop(devices):
    from dynamix.core.wtmm_backend import ComputeCancelled

    _truth, y = _gapped()
    dev = Inpaint()
    with pytest.raises(ComputeCancelled):
        dev.compute(_field(y), validate_params(dev, {**defaults_for(dev), "n_iter": 50}),
                    cancel=lambda: True)


def _dem_send(tmp_path, truth, x0=300000.0):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    p = tmp_path / "dem.tif"
    with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32",
                       crs="EPSG:32615", transform=Affine(30.0, 0.0, x0, 0.0, -30.0, 4000000.0)) as dst:
        dst.write((truth * 5.0).astype(np.float32), 1)
    return json.dumps([{"path": str(p), "subdataset": None, "band": None, "label": "dem",
                        "role": "structure", "shade": True}])


def test_the_scene_sun_stamp_feeds_the_shading_unless_the_knobs_were_set(devices, tmp_path):
    from dynamix.core.inpaint.engine import _bearing_of_up

    truth, y = _gapped()
    sun = {"azimuth": 100.0, "elevation": 30.0, "north": "true"}
    field = _field(y, x0=300015.0, sun=sun)
    sends = _dem_send(tmp_path, truth)
    dev = Inpaint()
    b, _frame = _bearing_of_up(field)
    assert abs(((b + 180) % 360) - 180) > 1.0                # the fixture has a convergence
    base = {**defaults_for(dev), **FAST, "heldout_blocks": 0, "_sends": sends}
    res = dev.compute(field, validate_params(dev, base))
    used = res["_sun"]["dem"]
    assert used["frame"] == "grid" and used["elevation"] == pytest.approx(30.0)
    assert abs(((used["azimuth"] - (100.0 - b) + 180) % 360) - 180) < 0.05
    assert abs(((used["azimuth"] - (100.0 + b) + 180) % 360) - 180) > 1.0
    given = dev.compute(field, validate_params(dev, {**base, "sun_azimuth": 200.0}))
    assert given["_sun"]["dem"]["azimuth"] == 200.0 and given["_sun"]["dem"]["frame"] == "grid"
    assert base["sun_azimuth"] == 0.0                       # the caller's params are untouched


def test_the_sun_knobs_stay_active_while_a_shaded_send_uses_them(devices):
    dev = Inpaint()
    base = validate_params(dev, defaults_for(dev))
    veil = {"veil_model", "veil_levels", "veil_passes"}
    sun = {"sun_azimuth", "sun_elevation"}
    shaded = dev.inactive_params(base, [{"role": "structure", "shade": True}])
    assert not (sun & shaded) and veil <= shaded
    assert (veil | sun) <= dev.inactive_params(base, [{"role": "structure"}])


def test_the_fill_product_gives_up_its_file_source_and_reflects_from_memory(devices, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    from dynamix.roi.runner import read_processing_window

    _truth, y = _gapped()
    p = tmp_path / "raw.tif"
    with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32",
                       crs="EPSG:32615", transform=Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)) as dst:
        dst.write(y.astype(np.float32), 1)
    field = _field(y)
    field.provenance["source"] = str(p)
    dev = Fill()
    clone = dev.compute(field, validate_params(dev, {**defaults_for(dev), "n_iter": 3}))
    assert "source" not in clone.provenance and clone.provenance["fill_source"] == str(p)
    window, _info = read_processing_window(clone, (16, 20, 24, 24), 4)
    inside = np.asarray(window.values)[4 + 4:4 + 20, 4 + 4:4 + 20]      # the filled region
    assert np.isfinite(inside).all()
    np.testing.assert_allclose(inside, np.asarray(clone.values)[20:36, 24:40])
