# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Filters as index selections over the chain product.

The model is EQSelect's: metrics are STATIC columns computed once over the immutable product;
a filter's whole effect is an intersection of index selections -- ``keep_h``/``keep_v`` per
chain, ``keep_h_pts``/``keep_iso_pts`` per point -- O(n_chains)/O(n_points) boolean work with
no dict ever copied. Legacy consumers keep seeing honest filtered dicts through ONE
materialization pass (``materialize_selection``), and the selection state is identity-bound to
the result's ``extrema``/``chains`` objects so any device that rewrites them outside the model
self-invalidates it.

Static-metric semantics note (deliberate): a metric always refers to
the TRANSFORM's own unfiltered product -- e.g. ``hline_length`` judges a line's original point
count and ``modulus_threshold``'s peak is the original layer's -- so filter order cannot change
what a knob value means. For the canonical chains (scale_select + metric filters) this is
value-identical to the sequential dict path, which the golden test pins.
"""
from __future__ import annotations

import numpy as np
import pytest

# DISABLED: the index-selection filter mechanism these pin was unsound (froze the app on
# min |W| / scale scrub) and is gated off at ``chain_product._SELECTION_ENABLED``. Retained (not
# deleted) as the spec for the progressive (finest-first) compute redesign that will re-enable
# it. Unskip when that flag flips.
pytest.skip("selection mechanism disabled pending the progressive-compute redesign",
            allow_module_level=True)

from dynamix.core.chain_product import (attach_chain_product, materialize_selection,
                                        selection_of)
from dynamix.devices.chain_filters import (ChainHolderFilter, ChainLengthFilter,
                                           ChainModulusFilter)
from dynamix.devices.filters import (HLineLength, HLineModulus, ModulusThreshold,
                                     OrientationWedge, ScaleSelect)
from dynamix.model.device import defaults_for

SHAPE = (8, 64)


def _layer(mods, line_ids, args, y=0):
    mod = np.asarray(mods, dtype=np.float64)
    n = mod.size
    return {"x": np.arange(n, dtype=np.int64), "y": np.full(n, y, dtype=np.int64),
            "mod": mod, "arg": np.asarray(args, dtype=np.float64),
            "line_id": np.asarray(line_ids, dtype=np.int64)}


def _chain(xs, ys, mods, scales):
    mod = np.asarray(mods, dtype=np.float64)
    k = mod.size
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_mod = np.log2(np.abs(mod))
    return {"x": np.asarray(xs, dtype=np.int64), "y": np.asarray(ys, dtype=np.int64),
            "mod": mod, "log2_mod": log2_mod,
            "log2_scales": np.log2(np.asarray(scales, dtype=np.float64))[:k]}


def _result():
    """Two scales; l0: lines 0 (3 pts) and 1 (2 pts) + two singletons; l1: one line + one
    singleton. Three chains, the last only one scale deep."""
    scales = [1.0, 2.0]
    l0 = _layer([1.0, 0.8, 0.6, 0.5, 0.4, 0.9, 0.2],
                [0, 0, 0, 1, 1, -1, -1],
                [0.0, 0.1, 0.2, 1.5, 1.6, 0.05, 1.55], y=0)
    l1 = _layer([0.7, 0.3, 0.2, 0.65], [0, 0, 0, -1],
                [0.3, 0.4, 0.5, 1.2], y=2)
    chains = [
        _chain([0, 0], [0, 2], [1.0, 0.7], scales),
        _chain([3, 1], [0, 2], [0.5, 0.3], scales),
        _chain([5], [0], [0.9], scales),
    ]
    res = {"extrema": [l0, l1], "chains": chains,
           "scales": np.asarray(scales), "_shape": SHAPE}
    return attach_chain_product(res)


def _strip(res):
    """The same result without the product/selection -- forces the legacy dict path."""
    return {k: v for k, v in res.items()
            if k not in ("chain_product", "_selection", "_hline_runs")}


def _params(device, **over):
    d = defaults_for(device)
    d.update(over)
    return d


# --------------------------------------------------------------------------- the state itself

def test_attach_stamps_an_identity_bound_all_pass_selection():
    res = _result()
    sel = selection_of(res)
    assert sel is not None
    assert sel["keep_h"] is None and sel["keep_v"] is None
    assert sel["keep_h_pts"] is None and sel["keep_iso_pts"] is None


def test_rewriting_chains_outside_the_model_invalidates_the_selection():
    res = _result()
    rogue = dict(res)
    rogue["chains"] = list(res["chains"])      # same content, new object -- a rogue rewrite
    assert selection_of(rogue) is None


def test_rewriting_extrema_outside_the_model_invalidates_the_selection():
    res = _result()
    rogue = dict(res)
    rogue["extrema"] = [dict(l) for l in res["extrema"]]
    assert selection_of(rogue) is None


# --------------------------------------------------------------------------- aware filters

def test_scale_select_narrows_chains_and_iso_to_the_scale():
    res = _result()
    p = res["chain_product"]
    out = ScaleSelect().apply(res, _params(ScaleSelect(), scale_idx=1))
    sel = selection_of(out)
    assert sel is not None
    np.testing.assert_array_equal(sel["keep_h"], np.flatnonzero(p["h_scale"] == 1))
    np.testing.assert_array_equal(sel["keep_iso_pts"], np.asarray(p["iso_scale"]) == 1)
    assert out["extrema"][0] is res["extrema"][1]      # dict behavior retained (a ref, no copy)
    assert out["_scale_idx"] == 1


def test_hline_length_is_a_range_over_the_original_length_column():
    res = _result()
    p = res["chain_product"]
    out = HLineLength().apply(res, _params(HLineLength(), min_len=3))
    sel = selection_of(out)
    np.testing.assert_array_equal(sel["keep_h"], np.flatnonzero(p["h_len"] >= 3))
    assert sel["keep_iso_pts"] is None                 # singletons pass through untouched
    assert out["extrema"] is res["extrema"]            # no dict ever copied


def test_modulus_threshold_narrows_points_against_the_static_scale_peak():
    res = _result()
    p = res["chain_product"]
    out = ModulusThreshold().apply(res, _params(ModulusThreshold(), frac=0.5))
    sel = selection_of(out)
    pt_scale = np.repeat(p["h_scale"], np.diff(p["h_off"]))
    np.testing.assert_array_equal(
        sel["keep_h_pts"], p["h_mod"] >= 0.5 * p["scale_peak"][pt_scale])
    np.testing.assert_array_equal(
        sel["keep_iso_pts"], p["iso_mod"] >= 0.5 * p["scale_peak"][p["iso_scale"]])
    assert out["extrema"] is res["extrema"]


def test_wedge_narrows_points_and_lets_missing_args_pass():
    res = _result()
    bare = [{k: v for k, v in layer.items() if k != "arg"} for layer in res["extrema"]]
    res_bare = attach_chain_product(
        {"extrema": bare, "chains": res["chains"], "scales": res["scales"], "_shape": SHAPE})
    out = OrientationWedge().apply(
        res_bare, _params(OrientationWedge(), centre=0.0, half_width=10.0))
    sel = selection_of(out)
    assert sel["keep_h_pts"] is None or bool(np.all(sel["keep_h_pts"]))   # NaN arg = pass

    out2 = OrientationWedge().apply(res, _params(OrientationWedge(), centre=0.0, half_width=10.0))
    sel2 = selection_of(out2)
    deg = np.degrees(res["chain_product"]["h_arg"]) % 180.0
    d = np.abs(deg - 0.0) % 180.0
    inside = np.minimum(d, 180.0 - d) <= 10.0
    np.testing.assert_array_equal(sel2["keep_h_pts"], inside)


def test_hline_modulus_sup_and_mean_are_ranges_over_the_metric_columns():
    res = _result()
    p = res["chain_product"]
    for mode, col in (("sup", "h_mod_sup"), ("mean", "h_mod_mean")):
        out = HLineModulus().apply(res, _params(HLineModulus(), frac=0.5, mode=mode))
        sel = selection_of(out)
        floor = 0.5 * p["scale_peak"][p["h_scale"]]
        np.testing.assert_array_equal(sel["keep_h"], np.flatnonzero(p[col] >= floor),
                                      err_msg=mode)
        # singletons get the point-wise rule (a one-point line IS its own sup)
        np.testing.assert_array_equal(
            sel["keep_iso_pts"], p["iso_mod"] >= 0.5 * p["scale_peak"][p["iso_scale"]],
            err_msg=mode)
        assert out["extrema"] is res["extrema"]


def test_hline_modulus_segments_mode_falls_back_and_invalidates():
    """The segments cut is genuinely path-shaped, not a metric range -- it takes the honest
    dict path (via one materialization) and the selection dies with it, so the views fall
    back too rather than drawing a stale subset."""
    res = _result()
    out = HLineModulus().apply(res, _params(HLineModulus(), frac=0.6, mode="segments"))
    assert selection_of(out) is None
    ref = HLineModulus().apply(_strip(res), _params(HLineModulus(), frac=0.6, mode="segments"))
    for got, want in zip(out["extrema"], ref["extrema"]):
        np.testing.assert_array_equal(got["mod"], want["mod"])
        np.testing.assert_array_equal(got["line_id"], want["line_id"])


def test_chain_filters_narrow_keep_v_without_rewriting_chains():
    res = _result()
    p = res["chain_product"]

    out = ChainLengthFilter().apply(res, _params(ChainLengthFilter(), min_len=2))
    sel = selection_of(out)
    np.testing.assert_array_equal(sel["keep_v"], np.flatnonzero(p["v_persist"] >= 2))
    assert out["chains"] is res["chains"]
    assert out["_chains_dropped"] == 1

    out = ChainModulusFilter().apply(res, _params(ChainModulusFilter(), threshold=-0.6))
    sel = selection_of(out)
    vals = p["v_max_log2_mod"]
    np.testing.assert_array_equal(sel["keep_v"],
                                  np.flatnonzero(np.isfinite(vals) & (vals >= -0.6)))

    out = ChainHolderFilter().apply(res, _params(ChainHolderFilter(), cutoff=0.4))
    sel = selection_of(out)
    vals = p["v_holder_ols"]
    np.testing.assert_array_equal(sel["keep_v"],
                                  np.flatnonzero(np.isfinite(vals) & (vals >= 0.4)))


def test_selections_compose_by_intersection():
    res = _result()
    p = res["chain_product"]
    out = ScaleSelect().apply(res, _params(ScaleSelect(), scale_idx=0))
    out = HLineLength().apply(out, _params(HLineLength(), min_len=3))
    sel = selection_of(out)
    np.testing.assert_array_equal(
        sel["keep_h"], np.flatnonzero((p["h_scale"] == 0) & (p["h_len"] >= 3)))


# --------------------------------------------------------------------------- materialization

def test_materialize_on_an_all_pass_selection_is_an_identity():
    res = _result()
    out = materialize_selection(res)
    assert out["extrema"] is res["extrema"]
    assert out["chains"] is res["chains"]


def test_materialized_chains_share_the_original_dict_objects():
    res = _result()
    out = ChainLengthFilter().apply(res, _params(ChainLengthFilter(), min_len=2))
    m = materialize_selection(out)
    assert m["chains"] == [res["chains"][0], res["chains"][1]]
    assert m["chains"][0] is res["chains"][0]          # list-index, never a copy
    assert selection_of(m) is not None                 # still live for the views


def test_materialize_equals_the_dict_path_for_the_canonical_chain():
    """The golden equivalence: [scale_select, hline_length, hline_modulus sup,
    modulus_threshold, orientation_wedge, chain_length] via selections + one materialization
    produces value-identical extrema/chains to the sequential legacy dict path."""
    res = _result()
    steps = (
        (ScaleSelect(), dict(scale_idx=0)),
        (HLineLength(), dict(min_len=2)),
        (HLineModulus(), dict(frac=0.4, mode="sup")),
        (ModulusThreshold(), dict(frac=0.45)),
        (OrientationWedge(), dict(centre=0.0, half_width=30.0)),
        (ChainLengthFilter(), dict(min_len=2)),
    )
    fast = res
    for dev, over in steps:
        fast = dev.apply(fast, _params(dev, **over))
    fast = materialize_selection(fast)

    ref = _strip(res)
    for dev, over in steps:
        ref = dev.apply(ref, _params(dev, **over))

    assert len(fast["extrema"]) == len(ref["extrema"]) == 1
    for key in ("x", "y", "mod", "arg", "line_id"):
        np.testing.assert_array_equal(fast["extrema"][0][key], ref["extrema"][0][key],
                                      err_msg=f"extrema key {key!r}")
    assert fast["chains"] == ref["chains"]


def test_layer_point_keep_matches_the_materialized_mask():
    """The one per-layer keep mask, shared by the materializer and the scene's fast path --
    both must scatter the identical selection back onto the layer's own point indices."""
    from dynamix.core.chain_product import layer_point_keep

    res = _result()
    out = ScaleSelect().apply(res, _params(ScaleSelect(), scale_idx=0))
    out = ModulusThreshold().apply(out, _params(ModulusThreshold(), frac=0.5))
    sel = selection_of(out)
    m = materialize_selection(out)
    keep0 = layer_point_keep(res["chain_product"], sel, 0, len(res["extrema"][0]["x"]))
    np.testing.assert_array_equal(np.asarray(res["extrema"][0]["x"])[keep0],
                                  m["extrema"][0]["x"])
