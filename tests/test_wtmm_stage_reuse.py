# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np

from dynamix.core import fft_policy
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.core.wtmm_backend import PythonWTMMBackend, run_wtmm2d
from dynamix.engine.cache import StageStore


def test_moment_increment_never_generates_an_order_above_the_requested_maximum():
    from dynamix.devices.wtmm import q_list
    from dynamix.core.partition_tables import moment_grid

    np.testing.assert_allclose(q_list({"q_min": 0, "q_max": 1, "q_step": .6}), [0, .6])
    np.testing.assert_allclose(moment_grid(-.3, .3, .1), [-.3, -.2, -.1, 0, .1, .2, .3])


def test_downstream_edits_reuse_upstream_stages_and_match_a_fresh_run(monkeypatch):
    fft_policy.configure("pyfftw", 64)
    rng = np.random.default_rng(41)
    field = RasterField(name="reuse", values=rng.normal(size=(48, 48)), frame=LocalFrame(),
                        x_axis=np.arange(48.), y_axis=np.arange(48.))
    backend = PythonWTMMBackend()
    calls = dict.fromkeys(("cwt2d", "extrema2d", "chains2d", "partition2d"), 0)
    for name in calls:
        original = getattr(backend, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(backend, name, counted)
    store = StageStore(maxsize=16)
    params = {"n_oct": 2, "n_voice": 2, "a_min": 2.0}
    first = run_wtmm2d(field, params, backend=backend, stage_cache=store)
    assert calls == dict.fromkeys(calls, 1)
    q_params = {**params, "q_list": [-2., 0., 2.]}
    repartitioned = run_wtmm2d(field, q_params, backend=backend, stage_cache=store)
    assert calls == {"cwt2d": 1, "extrema2d": 1, "chains2d": 1, "partition2d": 2}
    assert repartitioned["cache_hits"] == {"cwt", "extrema", "chains"}
    chain_params = {**q_params, "box_ratio": 1.5}
    rechained = run_wtmm2d(field, chain_params, backend=backend, stage_cache=store)
    assert calls == {"cwt2d": 1, "extrema2d": 1, "chains2d": 2, "partition2d": 3}
    fresh = run_wtmm2d(field, chain_params)
    np.testing.assert_allclose(rechained["hd_std"]["tau_qa"], fresh["hd_std"]["tau_qa"], equal_nan=True)
    # A consumer annotates a result by rebinding a key; the stored stage keeps its own dicts
    # (the arrays themselves are shared read-only, see the test below).
    first["extrema"][0]["mod"] = np.full_like(first["extrema"][0]["mod"], -1)
    again = run_wtmm2d(field, params, backend=backend, stage_cache=store)
    assert np.all(again["extrema"][0]["mod"] >= 0)
    run_wtmm2d(field, {**params, "thresh": .1}, backend=backend, stage_cache=store)
    assert calls["cwt2d"] == 1 and calls["extrema2d"] == 2
    field.values[0, 0] += 1
    run_wtmm2d(field, params, backend=backend, stage_cache=store)
    assert calls["cwt2d"] == 2
    store.clear()
    run_wtmm2d(field, params, backend=backend, stage_cache=store)
    assert calls["cwt2d"] == 3


def test_a_stage_hit_shares_its_arrays_read_only_and_isolates_its_structure():
    """The in-memory stage store hands back the maxima and chains without copying their arrays
    (a deep copy of 366 000 maxima on twelve scales cost half a second per knob change): the
    arrays are shared and read-only, so nothing downstream can change the cached numbers, and
    the dicts and lists are fresh, so a downstream ``pop`` of an annotation leaves the store's
    entry whole."""
    import numpy as np
    import pytest
    from conftest import fbm2d, make_field
    from dynamix.core.wtmm_backend import run_wtmm2d

    field = make_field(fbm2d(128, 0.5, seed=4))
    params = {"n_oct": 2, "n_voice": 1, "a_min": 1.0, "wavelet": "mexican", "fracint_alpha": 0.0,
              "smoother_params": {}}
    cache: dict = {}
    first = run_wtmm2d(field, params, stage_cache=cache)
    second = run_wtmm2d(field, params, stage_cache=cache)
    assert {"cwt", "extrema", "chains"} <= set(second["cache_hits"])
    a, b = first["extrema"][0], second["extrema"][0]
    assert a is not b and np.shares_memory(a["x"], b["x"]) and not b["x"].flags.writeable
    stored = next(v for k, v in cache.items() if ":extrema:" in k)
    b.pop("mod")                                               # a downstream annotation edit
    assert "mod" in stored[0] and "mod" in a
    with pytest.raises(ValueError):
        b["x"][0] = -1                                         # the numbers are the store's
