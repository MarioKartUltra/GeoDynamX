# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the real devices over the copied WTMM backend.

These run the actual pipeline on the in-repo EBSD fixture rather than a synthetic stand-in, so
they pin the behaviour a user will actually see. The fixture is 64x64 at 70 um/px.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.chain_product import materialize_selection
from dynamix.core.rasterfield import RasterField
from dynamix.devices.filters import (ModulusThreshold, OrientationWedge, ScaleSelect,
                                     _wrapped_delta)
from dynamix.devices.wtmm import WTMM2D
from dynamix.model.device import defaults_for, validate_params

FIXTURE = "tests/fixtures/kam_64.npz"


@pytest.fixture(scope="module")
def stack():
    """The computed WTMM stack. Module-scoped: it costs ~1.5 s and nothing here mutates it.

    ``defaults_for``, not a hand-spelled dict: the five values this used to list by hand
    (n_oct=3, n_voice=4, a_min=1.0, wavelet="mexican", min_chain_len=2) were already exactly
    WTMM2D's own declared defaults, so pulling the full set from the device schema is the same
    recipe -- plus whatever any future change adds, without another one-by-one edit
    here every time the schema grows.
    """
    field = RasterField.load_npz(FIXTURE)
    return WTMM2D().compute(field, defaults_for(WTMM2D()))


# --------------------------------------------------------------------------- the transform

def test_wtmm_produces_one_extrema_layer_per_scale(stack):
    assert len(stack["extrema"]) == len(stack["scales"]) == 12


def test_every_layer_carries_the_arg_the_wedge_filters_on(stack):
    for layer in stack["extrema"]:
        assert "arg" in layer and len(layer["arg"]) == len(layer["x"])


def test_extrema_count_falls_off_with_scale(stack):
    """Coarser scales must resolve fewer features. If this inverts, the scale axis is reversed."""
    counts = [len(l["x"]) for l in stack["extrema"]]
    assert counts[0] > counts[-1]
    assert counts[0] > 500 and counts[-1] < 100


def test_cache_key_is_param_sensitive_and_stable():
    d = WTMM2D()
    a = d.cache_key("src", {"n_oct": 3, "n_voice": 4})
    assert a == d.cache_key("src", {"n_oct": 3, "n_voice": 4})
    assert a != d.cache_key("src", {"n_oct": 4, "n_voice": 4})


def test_wavelet_choice_is_mexican_or_gaussian_not_morlet():
    """"morlet" was never implemented -- the copied core's ``cwt2d`` silently fell through to
    the gaussian branch for any non-"mexican" value, so a "morlet" layer was always actually
    gaussian. The device schema must not offer a choice that lies about what it computes."""
    d = WTMM2D()
    assert validate_params(d, {"wavelet": "gaussian"})["wavelet"] == "gaussian"
    with pytest.raises(ValueError, match="not one of"):
        validate_params(d, {"wavelet": "morlet"})


# --------------------------------------------------------------------------- scale select

def test_scale_select_picks_one_layer_and_reports_its_size(stack):
    out = ScaleSelect().apply(stack, {"scale_idx": 5})
    assert len(out["extrema"]) == 1
    assert out["_scale_idx"] == 5
    assert out["_scale_px"] == pytest.approx(float(stack["scales"][5]))


def test_scale_select_clamps_rather_than_raising(stack):
    """A stale project naming a scale index beyond a smaller stack must degrade, not crash."""
    assert ScaleSelect().apply(stack, {"scale_idx": 999})["_scale_idx"] == 11


def test_scale_select_does_not_mutate_the_cached_stack(stack):
    before = len(stack["extrema"])
    ScaleSelect().apply(stack, {"scale_idx": 3})
    assert len(stack["extrema"]) == before, "filter mutated the cached transform result"


# --------------------------------------------------------------------------- orientation

@pytest.mark.parametrize("a,b,expected", [
    (175.0, 10.0, 15.0),      # across the wrap: axial, so 15 apart and not 165
    (10.0, 175.0, 15.0),
    (0.0, 90.0, 90.0),        # the maximum axial separation
    (0.0, 180.0, 0.0),        # 180 is the same orientation
    (190.0, 10.0, 0.0),       # unwrapped input folds correctly
])
def test_axial_separation(a, b, expected):
    assert float(_wrapped_delta(np.array([a]), b)[0]) == pytest.approx(expected)


def test_wedge_keeps_a_subset_and_a_full_wedge_keeps_everything(stack):
    # Selection contract: on a product-stamped result, apply()
    # narrows index selections; materialize_selection lands the honest filtered dicts --
    # in the app, resolve() does this once per redraw.
    one = ScaleSelect().apply(stack, {"scale_idx": 2})
    n_all = len(one["extrema"][0]["x"])

    full = OrientationWedge().apply(one, {"centre": 40.0, "half_width": 90.0, "north": "grid"})
    assert len(full["extrema"][0]["x"]) == n_all

    narrow = materialize_selection(
        OrientationWedge().apply(one, {"centre": 40.0, "half_width": 15.0, "north": "grid"}))
    assert 0 < len(narrow["extrema"][0]["x"]) < n_all


def test_wedge_selects_different_subsets_at_different_strikes(stack):
    """Anisotropy: two orthogonal wedges must not select the same points."""
    one = ScaleSelect().apply(stack, {"scale_idx": 2})
    w = OrientationWedge()
    a = materialize_selection(
        w.apply(one, {"centre": 40.0, "half_width": 15.0, "north": "grid"}))["extrema"][0]["x"]
    b = materialize_selection(
        w.apply(one, {"centre": 130.0, "half_width": 15.0, "north": "grid"}))["extrema"][0]["x"]
    assert len(a) != len(b)


def test_wedge_masks_every_per_point_array_consistently(stack):
    """x, y, mod and arg must stay index-aligned, or a point's modulus belongs to another point."""
    one = ScaleSelect().apply(stack, {"scale_idx": 2})
    out = materialize_selection(
        OrientationWedge().apply(one, {"centre": 40.0, "half_width": 15.0, "north": "grid"}))
    layer = out["extrema"][0]
    n = len(layer["x"])
    for key in ("y", "mod", "arg"):
        assert len(layer[key]) == n, key


# --------------------------------------------------------------------------- modulus

def test_modulus_threshold_removes_weak_extrema(stack):
    one = ScaleSelect().apply(stack, {"scale_idx": 2})
    n_all = len(one["extrema"][0]["x"])
    out = materialize_selection(ModulusThreshold().apply(one, {"frac": 0.35}))
    kept = out["extrema"][0]
    assert 0 < len(kept["x"]) < n_all
    assert float(np.min(kept["mod"])) >= 0.35 * float(np.nanmax(one["extrema"][0]["mod"]))


def test_zero_threshold_is_a_no_op(stack):
    one = ScaleSelect().apply(stack, {"scale_idx": 2})
    assert ModulusThreshold().apply(one, {"frac": 0.0}) is one


# --------------------------------------------------------------------------- chain product

def test_transform_stamps_the_chain_product(stack):
    """The transform stamps the draw-ready CSR bundle worker-side, so
    no landing or filter tweak ever re-derives geometry. It must be exactly what the builder
    produces from the stack's own extrema/chains and the already-stamped ordering runs."""
    from dynamix.core.chain_product import build_chain_product

    p = stack.get("chain_product")
    assert p is not None, "wtmm2d must stamp result['chain_product']"
    ref = build_chain_product(stack["extrema"], stack["chains"], stack["scales"],
                              stack["_shape"], runs=stack["_hline_runs"])
    assert set(p) == set(ref)
    for key in ref:
        np.testing.assert_array_equal(np.asarray(p[key]), np.asarray(ref[key]),
                                      err_msg=f"key {key!r} differs")
