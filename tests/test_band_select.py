# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The band_select field stage: one band of a multiband stack as the working field."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.band_select import BandSelect


def _stack():
    v = np.zeros((4, 5, 3))
    for b in range(3):
        v[..., b] = b + 1
    return RasterField(name="stack", values=v, frame=LocalFrame(),
                       x_axis=np.arange(5.0), y_axis=np.arange(4.0),
                       provenance={"bands": ["VNIR_Band1/ImageData",
                                             "VNIR_Band2/ImageData",
                                             "VNIR_Band3N/ImageData"], "crs": "EPSG:32650"})


def test_band_select_slices_the_named_band_and_keeps_the_grid():
    out = BandSelect().compute(_stack(), {"band": 2})
    assert out.values.shape == (4, 5) and np.all(out.values == 2.0)
    assert out.provenance["band"] == "VNIR_Band2/ImageData"
    assert out.provenance["crs"] == "EPSG:32650"          # georeference rides through
    assert np.array_equal(out.x_axis, np.arange(5.0))
    assert out.name.endswith("VNIR_Band2/ImageData")


def test_band_select_refuses_a_single_band_field_and_a_missing_band():
    f2d = RasterField(name="flat", values=np.zeros((4, 5)), frame=LocalFrame(),
                      x_axis=np.arange(5.0), y_axis=np.arange(4.0))
    with pytest.raises(ValueError, match="multiband"):
        BandSelect().compute(f2d, {"band": 1})
    with pytest.raises(ValueError, match="band 7"):
        BandSelect().compute(_stack(), {"band": 7})


def test_band_select_registers_as_a_field_stage(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import get_device

    register_builtin_devices()
    dev = get_device("band_select")
    assert dev.field_stage and dev.roi_margin({"band": 1}) == 0
