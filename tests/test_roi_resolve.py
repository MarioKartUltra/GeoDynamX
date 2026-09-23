# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""engine.resolve and the ROI runner."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import register_device
from dynamix.model.layer import Layer
from dynamix.model.param import Param, ParamKind


def _picture():
    f = RasterField(name="pic", values=np.zeros((10, 14)), frame=LocalFrame(),
                    x_axis=np.arange(14.0), y_axis=np.arange(10.0))
    f.provenance.update({"display_stride": 5, "full_dims": (50, 70),
                         "source": "/nonexistent.tif", "window": {"row_off": 0, "col_off": 0}})
    return f


class _SourceReader:
    """Stands in for wtmm2d_roi: reads its own pixels off provenance['source']."""

    name = "src_reader"
    reads_source = True
    params = (Param("k", ParamKind.INT, default=1, min=0, max=9),)

    def compute(self, field, params, *, progress=None):
        return {"k": params["k"]}

    def cache_key(self, source_id, params):
        return f"{source_id}:{params['k']}"


def _layer(*names):
    return Layer(layer_id=1, name="L", source_id="s",
                 chain=Chain(tuple(DeviceRef(n, {}) for n in names)).materialized())


def test_no_tool_runs_on_the_display_picture(clean_registry, stub_transform):
    register_device(stub_transform)
    with pytest.raises(ValueError, match="ROI"):
        resolve(_layer("t"), _picture(), Cache())


def test_the_picture_itself_still_resolves_with_no_transform(clean_registry):
    out = resolve(_layer(), _picture(), Cache())
    assert out.transforms_run == ()


def test_a_device_that_reads_its_own_source_is_not_refused(clean_registry):
    register_device(_SourceReader())
    out = resolve(_layer("src_reader"), _picture(), Cache())
    assert out.result["k"] == 1


def test_an_ordinary_field_is_untouched(clean_registry, stub_transform):
    register_device(stub_transform)
    f = RasterField(name="f", values=np.zeros((4, 4)), frame=LocalFrame(),
                    x_axis=np.arange(4.0), y_axis=np.arange(4.0))
    assert resolve(_layer("t"), f, Cache()).result["scale"] == 4


# ------------------------------------------------------------ C4: resolve runs ROI layers

class _Blur:
    name = "blur_t"
    params = ()

    def roi_margin(self, params):
        return 1

    def compute(self, field, params, *, progress=None):
        v = np.asarray(field.values)
        p = np.pad(v, 1, mode="edge")
        out = sum(p[1 + a:1 + a + v.shape[0], 1 + b:1 + b + v.shape[1]]
                  for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9.0
        return {"raster_out": out, "extrema": [], "chains": [], "_shape": out.shape}

    def cache_key(self, source_id, params):
        return f"blur:{source_id}"


class _Add:
    name = "add_t"
    params = ()
    field_stage = True

    def roi_margin(self, params):
        return 0

    def compute(self, field, params, *, progress=None):
        import dataclasses
        return dataclasses.replace(field, values=np.asarray(field.values) + 1000.0)

    def cache_key(self, source_id, params):
        return f"add:{source_id}"


def _roi_layer(window, *names):
    lay = _layer(*names)
    lay.tags["roi.window"] = window
    return lay


def _data():
    v = np.random.default_rng(3).standard_normal((30, 40)).cumsum(0)
    return RasterField(name="d", values=v, frame=LocalFrame(), x_axis=np.arange(40.0),
                       y_axis=np.arange(30.0))


def test_an_roi_layer_resolves_to_an_roi_shaped_result_and_filters_see_it(clean_registry,
                                                                         stub_filter):
    register_device(_Blur())
    register_device(stub_filter)
    out = resolve(_roi_layer("10,12,8,9", "blur_t", "f"), _data(), Cache())
    assert out.result["raster_out"].shape == (8, 9)
    assert out.result["_roi"]["roi"] == (10, 12, 8, 9)
    assert out.result["cut"] == 0.5                       # the filter ran on the ROI result
    assert out.transforms_run == ("blur_t",)


def test_two_windows_on_one_source_never_share_a_cache_line(clean_registry):
    register_device(_Blur())
    cache, f = Cache(), _data()
    a = resolve(_roi_layer("10,12,8,9", "blur_t"), f, cache).result
    b = resolve(_roi_layer("0,0,8,9", "blur_t"), f, cache).result
    assert len(cache) == 2
    assert not np.array_equal(a["raster_out"], b["raster_out"])


def test_the_field_stage_and_the_analyzer_run_as_one_region_step(clean_registry):
    register_device(_Add())
    register_device(_Blur())
    f = _data()
    out = resolve(_roi_layer("10,12,8,9", "add_t", "blur_t"), f, Cache())
    assert out.transforms_run == ("add_t", "blur_t")
    whole = _Blur().compute(RasterField(name="w", values=f.values + 1000.0, frame=f.frame,
                                        x_axis=f.x_axis, y_axis=f.y_axis), {})["raster_out"]
    assert np.allclose(out.result["raster_out"], whole[10:18, 12:21])


# ------------------------------------------------------------ Preview gate

class _Previewable:
    name = "prev_t"
    params = ()
    calls = []

    def compute(self, field, params, *, progress=None):
        return {"k": 1}

    def preview(self, field, params):
        _Previewable.calls.append(np.asarray(field.values).shape)
        return {"extrema": [], "chains": []}

    def cache_key(self, source_id, params):
        return f"prev:{source_id}"


def test_no_progressive_preview_runs_on_a_picture_or_for_an_roi_layer(clean_registry):
    """Final review: the finest-scale preview ran the analyzer on the WHOLE
    picture for an ROI child -- resampled pixels, slow on BOEM, drawn misregistered."""
    from dynamix.engine.resolve import preview_resolve

    register_device(_Previewable())
    _Previewable.calls.clear()
    assert preview_resolve(_roi_layer("10,12,8,9", "prev_t"), _data()) is None
    assert preview_resolve(_layer("prev_t"), _picture()) is None
    assert _Previewable.calls == []
    assert preview_resolve(_layer("prev_t"), _data()) is not None     # ordinary: unchanged


def test_a_field_stage_alone_on_an_roi_never_runs_on_the_picture(clean_registry):
    """Final review: an ROI layer holding only a field stage (noise) had no region step, and
    the picture refusal waved it through because roi.window was set."""
    register_device(_Add())
    with pytest.raises(ValueError, match="analyzer"):
        resolve(_roi_layer("10,12,8,9", "add_t"), _picture(), Cache())
