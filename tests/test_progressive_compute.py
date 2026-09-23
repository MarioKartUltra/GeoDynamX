# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Headless primitives for progressive finest-first compute (spec 2026-09-14 §2).

Two things, both pure/headless (the GUI streaming + cancel wiring is a separate devloop slice):

* ``run_wtmm2d_preview`` — finest-scale-only WT + H-lines, value-identical to the full run's
  scale-0 layer for the same ``a_min``, but without the full stack / V-chains / partition.
* cooperative cancellation — ``run_wtmm2d(..., cancel=pred)`` raises ``ComputeCancelled`` at a
  stage boundary when ``pred()`` is true, and never caches a partial result.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.rasterfield import RasterField
from dynamix.core.wtmm_backend import ComputeCancelled, run_wtmm2d, run_wtmm2d_preview

FIXTURE = "tests/fixtures/kam_64.npz"
PARAMS = {"n_oct": 2, "n_voice": 2, "a_min": 1.0}


@pytest.fixture(scope="module")
def field():
    return RasterField.load_npz(FIXTURE)


@pytest.fixture(scope="module")
def full(field):
    return run_wtmm2d(field, PARAMS)


def test_preview_returns_only_the_finest_scale(field, full):
    prev = run_wtmm2d_preview(field, PARAMS)
    assert prev["_preview"] is True
    assert len(prev["extrema"]) == 1
    assert len(prev["scales"]) == 1
    assert prev["scales"][0] == pytest.approx(full["scales"][0])
    assert prev["chains"] == []                     # no cross-scale chaining in a preview


def test_preview_finest_layer_matches_the_full_run(field, full):
    """The whole point: the preview's finest scale IS the full run's scale-0 layer, so the
    finest-first frame the user sees is the real answer, not an approximation."""
    prev = run_wtmm2d_preview(field, PARAMS)
    a, b = prev["extrema"][0], full["extrema"][0]
    for key in ("x", "y", "mod", "arg", "line_id"):
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]), err_msg=key)


def test_preview_carries_hline_runs_for_the_finest_scale(field):
    prev = run_wtmm2d_preview(field, PARAMS)
    assert "_hline_runs" in prev and len(prev["_hline_runs"]) == 1
    assert prev["_shape"] == tuple(field.values.shape)


def test_preview_is_much_cheaper_than_the_full_run(field):
    import time
    t0 = time.perf_counter(); run_wtmm2d_preview(field, PARAMS); tp = time.perf_counter() - t0
    t0 = time.perf_counter(); run_wtmm2d(field, PARAMS); tf = time.perf_counter() - t0
    assert tp < tf                                  # finest-only beats the whole ladder


def test_cancel_before_any_stage_raises_and_caches_nothing(field, tmp_path):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return True                                 # cancelled from the very first check

    with pytest.raises(ComputeCancelled):
        run_wtmm2d(field, PARAMS, out_dir=tmp_path, cancel=cancel)
    assert calls["n"] >= 1
    # nothing partial written to the stage cache
    assert not list((tmp_path / "wtmm_cache").rglob("*.npz")) if (tmp_path / "wtmm_cache").exists() else True


def test_cancel_none_is_the_ordinary_full_run(field, full):
    r = run_wtmm2d(field, PARAMS, cancel=None)
    assert len(r["extrema"]) == len(full["extrema"])
    assert len(r["chains"]) == len(full["chains"])


def test_cancel_partway_stops_before_finishing(field):
    """A cancel that trips after the first stage boundary must raise, not return a full result."""
    seen = {"n": 0}

    def cancel():
        seen["n"] += 1
        return seen["n"] > 1                         # allow the first check, trip the next

    with pytest.raises(ComputeCancelled):
        run_wtmm2d(field, PARAMS, cancel=cancel)


# --- cancel threaded through resolve -> compute -> run_wtmm2d (design §3 worker link) ---------

def test_resolve_threads_cancel_into_the_wtmm_compute(field, clean_registry):
    """The 22 s lives inside wtmm2d.compute, so resolve must carry a cancel predicate down to
    run_wtmm2d's stage loop -- a set flag raises ComputeCancelled out of resolve, uncached."""
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.project import Project

    register_builtin_devices()
    p = Project(); src = p.add_source("kam")
    layer = p.add_layer("x", src.source_id,
                        Chain((DeviceRef("wtmm2d", dict(PARAMS)),
                               DeviceRef("scale_select", {"scale_idx": 0}))))
    cache = Cache()
    with pytest.raises(ComputeCancelled):
        resolve(layer, field, cache, cancel=lambda: True)
    # nothing cached: a subsequent uncancelled resolve still recomputes and succeeds
    r = resolve(layer, field, cache, cancel=lambda: False)
    assert len(r.result["extrema"]) == 1


def test_resolve_without_cancel_is_unchanged(field, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.project import Project

    register_builtin_devices()
    p = Project(); src = p.add_source("kam")
    layer = p.add_layer("x", src.source_id, Chain((DeviceRef("wtmm2d", dict(PARAMS)),)))
    r = resolve(layer, field, Cache())               # no cancel kwarg at all
    assert len(r.result["extrema"]) == len(r.result["scales"])
