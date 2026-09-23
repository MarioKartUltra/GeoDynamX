# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
from dynamix.devices.chain_classify import ChainClassify, chain_straightness
from dynamix.devices.estimators import HOLDER_ESTIMATORS

def _chain(slope, n=6, x0=10, y0=10):
    # log2_scales = [0..n-1] -> a = 2**log2_scales = [1, 2, 4, 8, 16, 32] at the default n=6.
    # ChainClassify's default fit_a_min=2.0 (the fit-floor doctrine's lower edge) drops only the
    # a=1 point; 5 of 6 survive the window, still an EXACT line, so the OLS slope every step/delta
    # test below asserts is unaffected by the mask.
    log2_scales = np.arange(n, dtype=np.float64)
    return {"x": np.full(n, x0, dtype=np.int64), "y": np.full(n, y0, dtype=np.int64),
            "mod": 2.0 ** (slope * log2_scales), "log2_mod": slope * log2_scales,
            "log2_scales": log2_scales}

def _extrema_line(x0=10, y0=10, npts=20, straight=True, line_id=7):
    xs = np.arange(x0, x0 + npts, dtype=np.int64)
    ys = np.full(npts, y0, dtype=np.int64)
    arg = np.zeros(npts) if straight else np.random.default_rng(0).uniform(-np.pi, np.pi, npts)
    return {"x": xs, "y": ys, "mod": np.ones(npts), "arg": arg,
            "line_id": np.full(npts, line_id, dtype=np.int64)}

def _result(chains, ext):
    return {"chains": chains, "extrema": [ext], "scales": np.array([1.0])}

def _params(**over):
    d = {p.name: p.default for p in ChainClassify.params}; d.update(over); return d

def test_ols_estimator_recovers_slope():
    assert abs(HOLDER_ESTIMATORS["ols"](_chain(-1.0)) - (-1.0)) < 1e-9

def test_step_seam_tagged():
    out = ChainClassify().apply(_result([_chain(0.0)], _extrema_line(straight=True)), _params())
    assert out["chains"][0]["tags"] == ["seam_step"]
    assert out["chains"][0]["tag_origin"] == "device:chain_classify"

def test_delta_seam_tagged():
    out = ChainClassify().apply(_result([_chain(-1.0)], _extrema_line(straight=True)), _params())
    assert out["chains"][0]["tags"] == ["seam_delta"]

def test_wiggly_line_not_tagged():
    out = ChainClassify().apply(_result([_chain(0.0)], _extrema_line(straight=False)), _params())
    assert out["chains"][0].get("tags", []) == []

def test_exclude_moves_tagged_and_keeps_evidence():
    out = ChainClassify().apply(
        _result([_chain(0.0), _chain(0.7)], _extrema_line(straight=True)),
        _params(action="exclude"))
    assert len(out["chains"]) == 1 and len(out["chains_excluded"]) == 1
    assert out["_chains_dropped"] == 1 and out["_show_ghosts"] is True

def test_missing_arg_key_noops():
    ext = _extrema_line(); del ext["arg"]
    res = _result([_chain(0.0)], ext)
    out = ChainClassify().apply(res, _params())
    assert out["chains"][0].get("tags", []) == []   # cannot claim straightness -> no seam tag

def test_points_below_the_fit_floor_are_excluded_so_no_h_and_no_tag():
    """The design's fit-floor doctrine (σ ≥ 3 px, i.e. a ≳ 1.9 at the default fit_a_min=2.0): a chain
    whose every point sits BELOW the floor must not be classified at all, even though its raw
    log2_mod/log2_scales are an exact step-seam line (slope 0.0) over a perfectly straight
    H-line -- without the fit-window mask this would tag seam_step; with it, zero points survive,
    the estimator's own <2-points guard returns NaN, and NaN fails every band comparison."""
    n = 3
    log2_scales = np.array([-2.0, -1.0, 0.0])   # a = 0.25, 0.5, 1.0 -- all under fit_a_min=2.0
    chain = {"x": np.full(n, 10, dtype=np.int64), "y": np.full(n, 10, dtype=np.int64),
             "mod": np.ones(n), "log2_mod": np.zeros(n), "log2_scales": log2_scales}

    out = ChainClassify().apply(_result([chain], _extrema_line(straight=True)), _params())

    assert out["chains"][0].get("tags", []) == []


def test_scales_above_l_over_8_are_excluded_when_shape_is_known():
    """The doctrine's OTHER edge, a ≤ L/8: a chain that is an exact step-seam line (slope 0.0)
    only WITHIN the fit window, and diverges outside it, must be read off the window, not off
    every point regardless of scale -- proving ``_shape`` actually narrows the fit rather than
    being read and ignored. shape=(64, 64) -> L/8 = 8, so with fit_a_min=2.0 the legal window is
    a in [2, 8] (log2_scales in [1, 3]); log2_scales=4 (a=16) sits past it."""
    log2_scales = np.array([1.0, 2.0, 3.0, 4.0])           # a = 2, 4, 8, 16
    log2_mod = np.array([0.0, 0.0, 0.0, 9.0])               # flat (h=0) inside the window, a kink
    #                                                          past it -- OLS over all 4 points
    #                                                          would NOT recover slope 0.0.
    chain = {"x": np.full(4, 10, dtype=np.int64), "y": np.full(4, 10, dtype=np.int64),
             "mod": 2.0 ** log2_mod, "log2_mod": log2_mod, "log2_scales": log2_scales}
    result = _result([chain], _extrema_line(straight=True))
    result["_shape"] = (64, 64)

    out = ChainClassify().apply(result, _params())

    assert out["chains"][0]["tags"] == ["seam_step"]


def test_duplicate_estimator_name_raises():
    import pytest
    from dynamix.devices.estimators import register_holder_estimator
    with pytest.raises(ValueError):
        register_holder_estimator("ols", lambda c: 0.0)
