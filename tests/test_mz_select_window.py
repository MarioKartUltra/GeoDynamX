# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Reconstruction section's level table and Per-level filters in the window: the table writes
``recon_levels`` through the step's box and re-keys the reconstruction; with Per-level filters on,
a filter knob edits the level on show and the Scale slider loads each level's own settings."""
from __future__ import annotations

import json

import numpy as np

from dynamix.engine.resolve import output_key, resolve, selection_recipe
from dynamix.model.device import declared_outputs, defaults_for, get_device
from tests.test_mz_reconstruction_panel import (J, _mz_layer, _step,  # noqa: F401
                                                builtins, values, win)


def _key(win, layer, name):
    """The engine's key for ``name`` of ``layer``, its selection folded in; the window predicts
    the same one."""
    r = resolve(layer, win._fields[layer.layer_id], win.cache, source_id=layer.source_id)
    out = next(o for o in declared_outputs(get_device("mz_edges")) if o.name == name)
    key = output_key(r.analysis_device, out, r.analysis_params, r.analysis_key,
                     selection_recipe(layer))
    assert win._output_keys(layer)[name] == key
    return key


def _rack(win, qtbot, *filters):
    """mz_edges with the Scale slider and ``filters`` in its rack, landed."""
    layer = _mz_layer(win, qtbot)
    params = {**defaults_for(get_device("mz_edges")), **_step(layer).params}
    desc = [{"device": "mz_edges", "params": params},
            {"device": "scale_select", "params": {"scale_idx": 0}}]
    desc += [{"device": n, "params": dict(p)} for n, p in filters]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def test_the_level_table_writes_the_states_and_rekeys_the_recon(win, qtbot):
    layer = _rack(win, qtbot)
    before = _key(win, layer, "recon")
    table = win._recon_panel.levels
    label, state, source = table.rows[0]
    with qtbot.waitSignal(win.resolved, timeout=10000):
        state.setCurrentIndex(table.STATES.index("coder"))
    data = json.loads(_step(layer).params["recon_levels"])
    assert data["levels"]["1"] == {"state": "coder", "source": 2}
    assert _key(win, layer, "recon") != before
    shown = win._active_result["extrema"][0]
    raw = win.cache.get(win._cache_keys_for(layer)[-1])
    assert np.array_equal(shown["x"], raw["extrema"][1]["x"])
    assert "coder from level 2" in table.reading.text()


def test_the_preset_is_the_papers_coding_mode(win, qtbot):
    layer = _rack(win, qtbot)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win._recon_panel.levels.preset_button.click()
    levels = json.loads(_step(layer).params["recon_levels"])["levels"]
    assert [levels[str(l)]["state"] for l in range(1, J + 1)] == ["coder", "own", "coder", "all"]
    assert levels["1"]["source"] == levels["3"]["source"] == 2


def test_a_filter_edit_changes_the_recon_key(win, qtbot):
    layer = _rack(win, qtbot, ("hline_length", {"min_len": 1}))
    before = _key(win, layer, "recon")
    i = win._names.index("hline_length")
    win._on_param_changed(i, "min_len", 8)
    assert _key(win, layer, "recon") != before


def test_per_level_filters_follow_the_scale_slider(win, qtbot):
    layer = _rack(win, qtbot, ("hline_length", {"min_len": 1}))
    i = win._names.index("hline_length")
    s = win._names.index("scale_select")
    win._on_recon_knob_changed("per_level", True)
    data = json.loads(_step(layer).params["recon_levels"])
    assert all(data["filters"][str(l)]["hline_length"]["min_len"] == 1 for l in range(1, J + 1))
    win._on_param_changed(s, "scale_idx", 1)                 # level 2
    win._on_param_changed(i, "min_len", 9)
    data = json.loads(_step(layer).params["recon_levels"])
    assert data["filters"]["2"]["hline_length"]["min_len"] == 9
    assert data["filters"]["1"]["hline_length"]["min_len"] == 1
    win._on_scale_changed(0)                                  # back to level 1: its own setting
    assert win._params[i]["min_len"] == 1
    assert win.strips.strip(i).controls["min_len"]._value == 1
    win._on_scale_changed(1)
    assert win._params[i]["min_len"] == 9
