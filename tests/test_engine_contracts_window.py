# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The window predicts the keys the engine builds for a chain holding a step that does not
apply: ``_cache_keys_for`` threads only the applying transforms, ``_output_keys`` and
``_shown_lazy`` key the engine's analysis and its selection, the transform signature leaves the
step out, and a band stack's chain is read as a stack."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.resolve import output_key, resolve, resolve_output, selection_recipe
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.contracts import FIELD, STACK
from dynamix.model.device import declared_outputs, get_device

WTMM = {"n_oct": 2, "n_voice": 2}
MZ = {"n_levels": 3, "iterations": 2, "run_mode": "fixed"}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _values():
    return np.random.default_rng(7).standard_normal((64, 64)).cumsum(0).cumsum(1)


def _window(qtbot, values):
    from dynamix.shell.main_window import MainWindow

    w = MainWindow(steps=())
    qtbot.addWidget(w)
    ny, nx = values.shape[:2]
    field = RasterField(name="f", values=values, frame=LocalFrame(),
                        x_axis=np.arange(float(nx)), y_axis=np.arange(float(ny)))
    w.load_field(field, "/nonexistent/f.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


@pytest.fixture
def win(qtbot, builtins):
    return _window(qtbot, _values())


@pytest.fixture
def stack_win(qtbot, builtins):
    v = _values()
    return _window(qtbot, np.stack([v, v[::-1], v[:, ::-1]], axis=-1))


def _with_chain(layer, *steps):
    """``layer`` (a copy) with ``steps`` as its chain; the window's own layer is untouched."""
    chain = Chain(tuple(DeviceRef(n, dict(p)) for n, p in steps)).materialized()
    return dataclasses.replace(layer, chain=chain)


def _resolved(win, layer):
    return resolve(layer, win._fields[layer.layer_id], win.cache, source_id=layer.source_id)


@pytest.mark.parametrize("steps", [
    (("mz_edges", MZ), ("chain_topology", {})),
    (("pca", {}), ("wtmm2d", WTMM)),
])
def test_the_predicted_keys_skip_a_transform_that_does_not_apply(win, steps):
    layer = _with_chain(win.layer, *steps)
    r = _resolved(win, layer)
    assert [s.applies for s in r.step_states].count(False) == 1
    assert win._cache_keys_for(layer) == [r.analysis_key]
    assert win._arr_tail_key(layer) in win.cache


def test_the_output_keys_are_the_engines_past_an_inactive_transform(win):
    layer = _with_chain(win.layer, ("mz_edges", MZ), ("chain_topology", {}))
    field = win._fields[layer.layer_id]
    r = _resolved(win, layer)
    keys = win._output_keys(layer)
    for out in declared_outputs(get_device("mz_edges")):
        if out.lazy:
            assert keys[out.name] == output_key(
                "mz_edges", out, r.analysis_params, r.analysis_key,
                selection_recipe(layer, FIELD) if out.selects else None)
    resolve_output(layer, field, win.cache, "coarse", source_id=layer.source_id)
    assert keys["coarse"] in win.cache


def test_an_inactive_filter_reaches_neither_predicted_nor_shown_output_keys(win):
    view = {**MZ, "show": "recon_edges_only", "recon_live": True}
    plain = _with_chain(win.layer, ("mz_edges", view), ("hline_length", {"min_len": 4}))
    layer = _with_chain(win.layer, ("mz_edges", view), ("chain_length", {"min_len": 5}),
                        ("hline_length", {"min_len": 4}))
    r = _resolved(win, layer)
    out = next(o for o in declared_outputs(get_device("mz_edges"))
               if o.name == "recon_edges_only")
    want = output_key("mz_edges", out, r.analysis_params, r.analysis_key,
                      selection_recipe(plain, FIELD))
    assert win._output_keys(layer)["recon_edges_only"] == want
    assert win._shown_lazy(layer, r) == (out, want)


def test_the_transform_signature_leaves_out_a_transform_that_does_not_apply(qtbot, builtins):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=(("pca", {"n_components": 3}), ("wtmm2d", WTMM)))
    qtbot.addWidget(win)
    win._start_worker_for = lambda layer: None      # the signature is read; nothing computes
    values = _values()
    win.load_field(RasterField(name="f", values=values, frame=LocalFrame(),
                               x_axis=np.arange(64.0), y_axis=np.arange(64.0)),
                   "/nonexistent/f.npz")
    assert [ref.device for ref in win.layer.chain.steps] == ["pca", "wtmm2d"]
    assert [name for name, _p in win._transform_signature()[1]] == ["wtmm2d"]
    assert [name for name, _p in win._transform_signature_for(win.layer)[1]] == ["wtmm2d"]


def test_a_band_stacks_chain_is_read_as_a_stack(stack_win):
    win = stack_win
    assert win._layer_input(win.layer) == win._layer_input() == STACK
    dataset = _with_chain(win.layer, ("pca", {}))
    r = _resolved(win, dataset)
    assert r.transforms_run == ("pca",)
    assert win._cache_keys_for(dataset) == [r.analysis_key]
    assert [name for name, _p in win._transform_signature_for(dataset)[1]] == ["pca"]
    band = next(l for l in win._layer_by_id.values() if l.parent_id == win.layer.layer_id)
    row = _with_chain(band, *[(s.device, s.params) for s in band.chain.steps], ("wtmm2d", WTMM))
    r = _resolved(win, row)
    assert r.transforms_run == ("band_select", "wtmm2d")
    assert win._cache_keys_for(row) == [*win._cache_keys_for(row)[:1], r.analysis_key]
