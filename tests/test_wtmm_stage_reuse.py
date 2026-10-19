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
    # Consumers can annotate a result without modifying its stored stage.
    first["extrema"][0]["mod"][:] = -1
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
