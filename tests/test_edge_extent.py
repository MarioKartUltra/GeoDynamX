# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The WTMM's Analyse knob on a Periodic extension output: ``real_data`` keeps the maxima to the
real block and takes the threshold from it; ``whole_field`` analyses the extended field, the
filled gap included, so the two spectra can be compared."""
from __future__ import annotations

import pytest

rasterio = pytest.importorskip("rasterio")

from dynamix.devices.periodic_extension import PeriodicExtension  # noqa: E402
from dynamix.devices.wtmm import WTMM2D  # noqa: E402
from dynamix.model.device import defaults_for, keyed_params, validate_params  # noqa: E402
from test_edge_taper import _field, _labyrinth  # noqa: E402


def _extended():
    ext = PeriodicExtension().compute(_field(_labyrinth(128)), defaults_for(PeriodicExtension()))
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
