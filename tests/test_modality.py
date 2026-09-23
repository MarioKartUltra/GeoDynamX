# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""tests/test_modality.py"""
import numpy as np
from dynamix.core.modality import MODALITIES, register_modality


class _Dummy:
    name = "dummy"
    def compute(self, field, params, *, out_dir=None, progress=None):
        return {"n": int(np.isfinite(field.values).sum()), "params": params}
    def overlay(self, result):
        return None
    def detail_widget_factory(self):
        return None

def test_register_and_dispatch():
    register_modality(_Dummy())
    assert "dummy" in MODALITIES
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    rf = RasterField(name="t", values=np.ones((2, 2)), frame=LocalFrame(),
                     x_axis=np.arange(2.0), y_axis=np.arange(2.0))
    out = MODALITIES["dummy"].compute(rf, {"a": 1})
    assert out == {"n": 4, "params": {"a": 1}}
    MODALITIES.pop("dummy")

def test_duplicate_name_raises():
    import pytest
    register_modality(_Dummy())
    with pytest.raises(ValueError):
        register_modality(_Dummy())
    MODALITIES.pop("dummy")
