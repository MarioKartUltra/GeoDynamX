# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The edge knobs (spec 22 §3.2.2–3): reflect by default with a ⚠ that names the scales whose halo
runs past the raster and the mirror lines' false maxima; cut at the cone of influence."""
from __future__ import annotations

import types

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dynamix.devices.wtmm import WTMM2D  # noqa: E402
from dynamix.devices.wtmm_roi import WTMM2DROI  # noqa: E402
from dynamix.model.device import defaults_for, keyed_params, validate_params  # noqa: E402
from test_roi_halo import _synthetic_parent, _write_tif  # noqa: E402

EDGE_KEYS = {"edge", "cut_scope", "edge_eps"}


def _stub(full=(384, 384)):
    return types.SimpleNamespace(provenance={"full_dims": full})


def _roi_params(**kw):
    dev = WTMM2DROI()
    p = dict(defaults_for(dev), n_oct=2, n_voice=4, a_min=1.0, fracint_alpha=0.0,
             roi_h=64, roi_w=64)
    p.update(kw)
    return validate_params(dev, p)


@pytest.mark.parametrize("dev", [WTMM2D(), WTMM2DROI()], ids=["wtmm2d", "wtmm2d_roi"])
def test_edge_params_are_out_of_the_key_at_their_defaults(dev):
    names = {p.name for p in dev.params}
    assert EDGE_KEYS <= names
    assert not EDGE_KEYS & set(keyed_params(dev, defaults_for(dev)))
    assert defaults_for(dev)["edge"] == "reflect"


def test_reflect_at_a_corner_warns_naming_scales_widths_and_mirror_maxima():
    msg = WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=0, roi_col=0))["a_min"]
    assert "mirror" in msg and "reflection" in msg and "N/W" in msg and "8 scale(s)" in msg
    assert "edge = cut" in msg
    assert WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=160, roi_col=160)) == {}


def test_a_near_edge_roi_warns_only_for_the_scales_that_reach_past_it():
    msg = WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=30, roi_col=160))["a_min"]
    assert "N edge" in msg and "8 scale(s)" not in msg


def test_reflective_boundary_always_warns():
    msg = WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=160, roi_col=160,
                                                         boundary="reflective"))["a_min"]
    assert "mirror" in msg and "E/N/S/W" in msg


def test_cut_carries_no_mirror_warning():
    assert WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=0, roi_col=0, edge="cut")) == {}


def test_the_corner_warning_rides_the_run_and_the_record(tmp_path):
    from dynamix.core.rasterfield import RasterField

    tif = _write_tif(tmp_path / "p.tif", _synthetic_parent(384, seed=7))
    field = RasterField.from_geotiff_window(tif, row_off=0, col_off=0, height=96, width=96)
    res = WTMM2DROI().compute(field, _roi_params(roi_row=0, roi_col=0))
    assert any("mirror" in w for w in res["_warnings"])
    assert res["_edge"]["mode"] == "reflect" and res["_edge"]["device"] == "wtmm2d_roi"
