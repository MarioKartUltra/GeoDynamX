# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""fracint_alpha on the SCALAR 2D path: the pseudo-fractional
integration lift is available on BOTH pipelines, not only tensor.

The reference family lifts only its tensor path's WT derivatives (``twtmm.py:154-165``:
``dx,dy *= a**alpha``), but the lift is a per-scale POSITIVE SCALAR -- on the scalar transform
``|grad| *= a**alpha`` with the gradient angle untouched is the identical operation, so there was
never anything tensor-specific about it. ``_apply_fracint2d`` applies it right after the (cached)
cwt stage; every downstream stage cache key carries ``fracint_alpha`` from there on, while the
expensive transform itself stays reusable across eta values.

Consequences pinned here: extrema POSITIONS are invariant (within-scale NMS and the per-scale
``thresh * max`` floor both scale through), extrema MODULI lift by exactly ``a**alpha`` per
scale, the cwt stage cache survives an alpha change while the extrema stage recomputes, and the
preview applies the same lift (its value-identity with the full run's scale-0 layer, pinned in
``test_progressive_compute.py``, now runs THROUGH the lift at the default alpha=1).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.rasterfield import RasterField
from dynamix.core.wtmm_backend import _apply_fracint2d, run_wtmm2d, run_wtmm2d_preview

FIXTURE = "tests/fixtures/kam_64.npz"
PARAMS = {"n_oct": 2, "n_voice": 2, "a_min": 1.0}


@pytest.fixture(scope="module")
def field():
    return RasterField.load_npz(FIXTURE)


# --------------------------------------------------------------------------------- the helper

def test_apply_fracint2d_zero_is_the_identity_object():
    """alpha == 0 returns the INPUT dict itself (the reference's own ``!= 0`` guard) -- no copy,
    no new arrays."""
    cwt = {"mod": np.ones((2, 4, 4), np.float32), "arg": np.zeros((2, 4, 4), np.float32)}
    assert _apply_fracint2d(cwt, [1.0, 2.0], 0.0) is cwt


def test_apply_fracint2d_lifts_mod_per_scale_and_never_mutates_the_input():
    """``mod[si] * a_si**alpha`` exactly; ``arg`` passes through as the SAME object (a positive
    scalar cannot turn a gradient); the input arrays stay untouched -- the cwt stage cache may
    own them."""
    rng = np.random.default_rng(0)
    mod = rng.random((3, 8, 8)).astype(np.float32)
    arg = rng.random((3, 8, 8)).astype(np.float32)
    cwt = {"mod": mod, "arg": arg}
    keep = mod.copy()
    scales = [1.0, 2.0, 4.0]
    out = _apply_fracint2d(cwt, scales, 1.5)
    np.testing.assert_array_equal(cwt["mod"], keep)
    assert out["arg"] is arg
    assert out["mod"].dtype == np.float32
    for si, a in enumerate(scales):
        np.testing.assert_allclose(out["mod"][si], keep[si] * np.float32(a ** 1.5), rtol=1e-6)


# --------------------------------------------------------------------------------- the pipeline

def test_scalar_run_lifts_extrema_moduli_but_not_positions(field):
    """Same maxima, lifted values: per scale the lifted run's extrema sit at the SAME (x, y)
    (NMS compares within a scale, so a per-scale scalar cancels) with ``mod`` exactly
    ``a**alpha`` times the unlifted run's."""
    base = run_wtmm2d(field, dict(PARAMS, fracint_alpha=0.0))
    lifted = run_wtmm2d(field, dict(PARAMS, fracint_alpha=1.0))
    assert len(base["extrema"]) == len(lifted["extrema"])
    for a, lo, hi in zip(base["scales"], base["extrema"], lifted["extrema"]):
        np.testing.assert_array_equal(lo["x"], hi["x"])
        np.testing.assert_array_equal(lo["y"], hi["y"])
        np.testing.assert_allclose(hi["mod"], lo["mod"] * np.float32(a), rtol=1e-5)


def test_fracint_change_reuses_the_cwt_stage_cache_but_not_extrema(field, tmp_path):
    """The lift sits AFTER the cached cwt stage, so scrubbing alpha re-runs extrema onward but
    never the FFTs: the second run hits the cwt cache and misses the extrema cache."""
    run_wtmm2d(field, dict(PARAMS, fracint_alpha=0.0), out_dir=tmp_path)
    second = run_wtmm2d(field, dict(PARAMS, fracint_alpha=1.0), out_dir=tmp_path)
    assert "cwt" in second["cache_hits"]
    assert "extrema" not in second["cache_hits"]


def test_preview_applies_the_same_lift(field):
    """The finest-first preview lifts its single scale exactly like the full run does, by the
    preview's OWN reported finest scale (``compute_scales2d``'s units, not ``a_min`` itself) --
    at ``a_min != 1`` the lift is numerically visible even on scale 0."""
    p = dict(PARAMS, a_min=2.0)
    base = run_wtmm2d_preview(field, dict(p, fracint_alpha=0.0))
    lifted = run_wtmm2d_preview(field, dict(p, fracint_alpha=1.0))
    a0 = float(base["scales"][0])
    np.testing.assert_array_equal(base["extrema"][0]["x"], lifted["extrema"][0]["x"])
    np.testing.assert_allclose(lifted["extrema"][0]["mod"],
                               base["extrema"][0]["mod"] * np.float32(a0), rtol=1e-5)
