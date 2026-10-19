# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The WTMM's Analyse knob on a Periodic extension output: ``real_data`` keeps the maxima to the
real block and takes the threshold from it; ``whole_field`` analyses the extended field, the
filled gap included, so the two spectra can be compared."""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dynamix.devices.periodic_extension import PeriodicExtension  # noqa: E402
from dynamix.devices.wtmm import WTMM2D  # noqa: E402
from dynamix.model.device import defaults_for, keyed_params, validate_params  # noqa: E402
from test_edge_taper import _field, _labyrinth  # noqa: E402


def _extended():
    """The extension with the data at the origin: these tests place the gap by position."""
    ext = PeriodicExtension().compute(_field(_labyrinth(128)),
                                      dict(defaults_for(PeriodicExtension()), layout="origin"))
    assert "edges.filled" in ext.provenance
    return ext


def _params(**kw):
    dev = WTMM2D()
    p = dict(defaults_for(dev), n_oct=2, n_voice=4, a_min=1.0, fracint_alpha=0.0)
    p.update(kw)
    return validate_params(dev, p)


def test_real_data_is_the_default_and_enters_the_key_only_when_set():
    dev = WTMM2D()
    assert defaults_for(dev)["extent"] == "real_data"
    assert "extent" not in keyed_params(dev, defaults_for(dev))
    assert keyed_params(dev, dict(defaults_for(dev), extent="whole_field"))["extent"] == "whole_field"


def test_whole_field_finds_maxima_in_the_gap_and_real_data_does_not():
    ext = _extended()
    rows, cols = ext.provenance["edges.filled"]["rows"], ext.provenance["edges.filled"]["cols"]
    assert ext.values.shape[0] > rows and ext.values.shape[1] > cols
    real = WTMM2D().compute(ext, _params())
    whole = WTMM2D().compute(ext, _params(extent="whole_field"))
    for lvl in real["extrema"]:
        assert ((lvl["x"] < cols) & (lvl["y"] < rows)).all()
    in_gap = sum(int(((lvl["x"] >= cols) | (lvl["y"] >= rows)).sum()) for lvl in whole["extrema"])
    assert in_gap > 0
    assert real["_padding"]["extent"] == "real_data" and whole["_padding"]["extent"] == "whole_field"


def test_the_centred_layout_rolls_the_gap_around_the_data_and_the_real_block_follows():
    dev = PeriodicExtension()
    field = _field(_labyrinth(128))
    origin = dev.compute(field, dict(defaults_for(dev), layout="origin"))
    centred = dev.compute(field, defaults_for(dev))                   # centred is the default
    H, W = field.values.shape
    gy, gx = centred.values.shape[0] - H, centred.values.shape[1] - W
    r0, c0 = gy // 2, gx // 2
    assert gy > 0 and gx > 0 and centred.values.shape == origin.values.shape
    stamp = centred.provenance["edges.filled"]
    assert (stamp["row0"], stamp["col0"]) == (r0, c0)
    assert "row0" not in origin.provenance["edges.filled"]        # the origin stamp as it was
    assert "layout" not in keyed_params(dev, defaults_for(dev))
    # the same periodic field, rolled; the data where they were, in pixels and in world units
    assert np.array_equal(centred.values, np.roll(np.roll(origin.values, r0, axis=0), c0, axis=1))
    assert np.array_equal(centred.values[r0:r0 + H, c0:c0 + W], np.asarray(field.values, float))
    assert centred.x_axis[c0] == field.x_axis[0] and centred.y_axis[r0] == field.y_axis[0]
    res = WTMM2D().compute(centred, _params())
    assert res["params"]["real_block"] == (H, W) and res["params"]["real_origin"] == (r0, c0)
    for lvl in res["extrema"]:
        assert ((lvl["x"] >= c0) & (lvl["x"] < c0 + W) & (lvl["y"] >= r0) & (lvl["y"] < r0 + H)).all()
    assert sum(len(lvl["x"]) for lvl in res["extrema"]) > 0
