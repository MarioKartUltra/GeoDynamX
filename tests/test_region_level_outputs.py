# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Region cell outputs reuse their landed analysis and preserve window placement."""
import numpy as np
import pytest

from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve, resolve_output
from dynamix.model.device import get_device
from tests.test_roi_outputs import _field, _layer, builtins


@pytest.mark.parametrize("roi", [(40, 44, 32, 40), (0, 0, 32, 40)])
def test_region_levels_reuse_analysis_and_match_window_route(builtins, monkeypatch, roi):
    field = _field()
    layer = _layer("wavelet_leaders", roi=roi, wavelet="db2", scale_min=2,
                   scale_max=8, with_wse=True)
    dev = get_device("wavelet_leaders")
    outputs = [o.name for o in dev.outputs if o.lazy]
    cache = Cache()
    landed = resolve(layer, field, cache, source_id="s")
    cache.mark_shown(layer.layer_id, landed.analysis_key)
    raw = cache.get(landed.analysis_key)
    from dynamix.roi.runner import crop_result_to_roi
    shape = (roi[2] + 2 * raw["_roi"]["margin"], roi[3] + 2 * raw["_roi"]["margin"])
    # Nested cell pyramids must survive even when a cell grid matches the window shape.
    quantity = {"leaders": [np.zeros(shape)]}
    assert crop_result_to_roi({"_quantity": quantity}, raw["_roi"]["margin"],
                              roi[2], roi[3], shape)["_quantity"] is quantity
    references = {}
    with monkeypatch.context() as patch:
        patch.setattr(type(dev), "region_outputs_from_result", False, raising=False)
        legacy = Cache()
        for name in outputs:
            for level in range(len(raw["scales"])):
                references[name, level] = resolve_output(layer, field, legacy, name,
                                                         source_id="s", level=level)

    def unexpected(*args, **kwargs):
        pytest.fail("resolving a region level reran the leaders analysis")

    monkeypatch.setattr(type(dev), "compute", unexpected)
    for name, level in references:
        value = resolve_output(layer, field, cache, name, source_id="s", level=level)
        expected = references[name, level]
        np.testing.assert_allclose(value["cells"], expected["cells"], rtol=1e-6,
                                   atol=1e-7, equal_nan=True,
                                   err_msg="same leaders cells at float32 output precision")
        assert value["paint"] == expected["paint"]
        assert value["_output_identity"] == expected["_output_identity"]
        for actual, wanted in zip(value["roi_axes"], expected["roi_axes"]):
            np.testing.assert_array_equal(actual, wanted)
        assert resolve_output(layer, field, cache, name, source_id="s", level=level) is value
    assert not any(k.endswith("|window") for k in cache._store)
    cache.max_bytes = 1
    for name, level in list(references)[::-1] + list(references):
        value = resolve_output(layer, field, cache, name, source_id="s", level=level)
        assert value["paint"] == references[name, level]["paint"]
