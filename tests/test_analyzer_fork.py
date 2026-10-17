# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The fork decision (_analyzer_fork, main_window): a DIFFERENT primary analyzer dropped on
an analyzed layer forks a sibling representation; ordinary edits commit as before. Returns
(sibling_steps, shipped_parent_indices) -- the shipping law: consumers follow their producer,
so a consumer stranded on the parent whose producers include the NEW analyzer ships over."""
from __future__ import annotations

import pytest

from dynamix.model.chain import DeviceRef
from dynamix.model.contracts import FIELD, STACK
from dynamix.shell.main_window import _analyzer_fork, _contract_fork


def _d(*names, bypassed=()):
    return [{"device": n, "params": {}, "bypassed": n in bypassed, "rack": None}
            for n in names]


def _devices(steps):
    return [s["device"] for s in steps]


def test_append_of_a_different_analyzer_forks_it_alone(builtins):
    sibling, shipped = _analyzer_fork(["holder_map"], [False], _d("holder_map", "wtmm2d"))
    assert _devices(sibling) == ["wtmm2d"] and shipped == []


def test_preset_replacing_the_analyzer_forks_the_whole_preset(builtins):
    sibling, shipped = _analyzer_fork(["holder_map"], [False],
                                      _d("wtmm2d", "scale_select", "chain_holder"))
    assert _devices(sibling) == ["wtmm2d", "scale_select", "chain_holder"]
    assert shipped == []


def test_multiset_diff_ships_a_reincluded_duplicate(builtins):
    """A preset re-including a device the parent also holds must still fork a copy -- the
    name-subtraction hole."""
    sibling, _ = _analyzer_fork(["holder_map", "scale_select"], [False, False],
                                _d("holder_map", "scale_select", "wtmm2d", "scale_select"))
    assert _devices(sibling) == ["wtmm2d", "scale_select"]


def test_stranded_consumers_ship_to_the_new_analyzer(builtins):
    """The shipping law: chain consumers sitting uselessly on a holder layer move to the
    forked wtmm sibling (as indices, so params travel)."""
    sibling, shipped = _analyzer_fork(
        ["holder_map", "chain_holder", "scale_select"], [False, False, False],
        _d("holder_map", "chain_holder", "scale_select", "wtmm2d"))
    assert _devices(sibling) == ["wtmm2d"]
    assert shipped == [1, 2]


def test_consumers_of_the_parents_own_analyzer_never_move(builtins):
    sibling, shipped = _analyzer_fork(
        ["wtmm2d", "scale_select", "chain_holder"], [False, False, False],
        _d("wtmm2d", "scale_select", "chain_holder", "holder_map"))
    assert _devices(sibling) == ["holder_map"]
    assert shipped == []


def test_same_analyzer_edits_never_fork():
    assert _analyzer_fork(["wtmm2d"], [False], _d("wtmm2d", "scale_select")) is None
    assert _analyzer_fork(["wtmm2d", "scale_select"], [False, False],
                          _d("scale_select", "wtmm2d")) is None      # reorder
    assert _analyzer_fork(["wtmm2d"], [False], _d()) is None          # removal


def test_unanalyzed_layer_edits_never_fork():
    assert _analyzer_fork([], [], _d("wtmm2d")) is None
    assert _analyzer_fork(["noise"], [False], _d("noise", "wtmm2d")) is None


def test_bypassed_analyzer_does_not_count_as_the_head():
    assert _analyzer_fork(["holder_map"], [True], _d("holder_map", "wtmm2d")) is None


def test_mz_edges_onto_wtmm_forks_the_extrema_rep(builtins):
    sibling, shipped = _analyzer_fork(["wtmm2d", "scale_select"], [False, False],
                                      _d("wtmm2d", "scale_select", "mz_edges"))
    assert _devices(sibling) == ["mz_edges"]
    assert shipped == []          # scale_select reads wtmm2d too -- it stays home


def test_master_root_never_takes_an_analyzer_directly(builtins):
    """Root protection: a primary dropped on a RAW master forks a child; the
    master's own recipe stays empty. Non-root raw layers keep the ordinary edit."""
    sibling, shipped = _analyzer_fork([], [], _d("holder_map"), root=True)
    assert _devices(sibling) == ["holder_map"] and shipped == []
    sibling, shipped = _analyzer_fork([], [], _d("wtmm2d", "scale_select"), root=True)
    assert _devices(sibling) == ["wtmm2d", "scale_select"]
    assert _analyzer_fork([], [], _d("holder_map"), root=False) is None


def test_master_keeps_filters_and_noise_without_forking(builtins):
    """Field-prep and filter edits on the master are ordinary edits; a later analyzer drop
    forks WITHOUT taking noise along (documented: prep ships via a preset, not the fork)."""
    assert _analyzer_fork([], [], _d("noise"), root=True) is None
    sibling, shipped = _analyzer_fork(["noise"], [False], _d("noise", "wtmm2d"), root=True)
    assert _devices(sibling) == ["wtmm2d"] and shipped == []


def test_split_holder_tools_are_primary_analyzers(builtins):
    """Split holder tools: each method is its own representation -- dropping the OTHER method
    (or the conflated predecessor's sibling) forks a sibling layer, exactly as wtmm2d does."""
    sibling, shipped = _analyzer_fork(["holder_multiaffine"], [False],
                                      _d("holder_multiaffine", "holder_measure"))
    assert _devices(sibling) == ["holder_measure"] and shipped == []
    sibling, shipped = _analyzer_fork(["holder_map"], [False],
                                      _d("holder_map", "holder_multiaffine"))
    assert _devices(sibling) == ["holder_multiaffine"] and shipped == []


def test_band_dialog_commit_picks_the_variant_of_the_recipes_producer(builtins):
    """A Band-dialog commit creates the band_recon VARIANT whose engine params are the
    recipe's h-map producer's verbatim; the conflated holder_map keeps the conflated
    band_recon (byte-compatible for saved projects)."""
    from dynamix.shell.main_window import _band_device_for, _band_step_for

    assert _band_device_for(["holder_measure"]) == "band_recon_measure"
    assert _band_device_for(["holder_multiaffine", "scale_select"]) == \
        "band_recon_multiaffine"
    assert _band_device_for(["holder_map"]) == "band_recon"
    assert _band_device_for([]) == "band_recon"

    assert _band_step_for(["holder_measure", "band_recon_measure"]) == "band_recon_measure"
    assert _band_step_for(["band_recon"]) == "band_recon"
    assert _band_step_for(["holder_measure"]) is None


def test_wavelet_skeleton_is_a_primary_analyzer(builtins):
    """The medial-axis extractor is its own representation: dropping it on a wtmm2d layer
    forks a sibling, exactly as mz_edges does."""
    sibling, shipped = _analyzer_fork(["wtmm2d"], [False],
                                      _d("wtmm2d", "wavelet_skeleton"))
    assert _devices(sibling) == ["wavelet_skeleton"] and shipped == []


def test_wavelet_skeleton_is_a_chain_producer_for_the_shipping_law(builtins):
    """The skeleton emits the extrema schema the chain consumers read, so it belongs in
    _CHAIN_PRODUCERS -- otherwise a fork STRIPS live, tuned filters off a skeleton layer
    (shipped=[1,2] where the identical wtmm2d-parent fork ships nothing), and stranded
    consumers never ship TO a skeleton sibling."""
    sibling, shipped = _analyzer_fork(
        ["wavelet_skeleton", "scale_select", "orientation_wedge"], [False] * 3,
        _d("wavelet_skeleton", "scale_select", "orientation_wedge", "mz_edges"))
    assert shipped == []                       # live consumers stay on the skeleton layer
    sibling, shipped = _analyzer_fork(
        ["holder_map", "scale_select"], [False, False],
        _d("holder_map", "scale_select", "wavelet_skeleton"))
    assert _devices(sibling) == ["wavelet_skeleton"]
    assert shipped == [1]                      # stranded consumer ships to the sibling


def test_cdf_edges_is_a_primary_analyzer_and_chain_producer(builtins):
    sibling, shipped = _analyzer_fork(["wtmm2d"], [False], _d("wtmm2d", "cdf_edges"))
    assert _devices(sibling) == ["cdf_edges"] and shipped == []
    sibling, shipped = _analyzer_fork(["cdf_edges", "scale_select"], [False, False],
                                      _d("cdf_edges", "scale_select", "mz_edges"))
    assert shipped == []                       # its consumers stay on their producer


def test_pm_edges_is_a_primary_analyzer_and_chain_producer(builtins):
    """Perona-Malik proper is the fifth peer analyzer: dropping it on an
    analyzed layer forks a sibling, and its extrema pyramid feeds the same chain
    consumers, so its consumers must stay put on a fork exactly as cdf_edges' do."""
    sibling, shipped = _analyzer_fork(["wtmm2d"], [False], _d("wtmm2d", "pm_edges"))
    assert _devices(sibling) == ["pm_edges"] and shipped == []
    sibling, shipped = _analyzer_fork(["pm_edges", "scale_select"], [False, False],
                                      _d("pm_edges", "scale_select", "mz_edges"))
    assert shipped == []                       # its consumers stay on their producer


# -- the fork and the shipping law read from the declared contracts (_contract_fork) ---------


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _refs(*names):
    return [DeviceRef(n, {}) for n in names]


def test_derived_primaries_are_the_table_plus_the_decompositions(builtins):
    from dynamix.model.device import DEVICES, get_device
    from dynamix.model.forking import is_primary
    from dynamix.shell.main_window import _PRIMARY_ANALYZERS

    derived = {n for n in DEVICES if is_primary(get_device(n))}
    assert derived == set(_PRIMARY_ANALYZERS) | {"pca", "tucker_HOOI_HOSVD", "tucker_havok",
                                                  "ssa2d", "dwt2d", "inpaint"}


def test_derived_shipping_is_the_table_apart_from_the_readers_of_chains_and_alpha(builtins):
    from dynamix.model.contracts import next_step_state
    from dynamix.shell.main_window import _CHAIN_PRODUCERS, _CONSUMER_PRODUCERS

    def producers(consumer):
        out = []
        for p in _CHAIN_PRODUCERS:
            alone = [DeviceRef(p, {})]
            topology = alone + [DeviceRef("chain_topology", {})]
            if (next_step_state(alone, FIELD, consumer).applies
                    or next_step_state(topology, FIELD, consumer).applies):
                out.append(p)
        return tuple(out)

    expected = dict(_CONSUMER_PRODUCERS)
    for c in ("chain_holder", "chain_modulus", "chain_length", "chain_classify", "group_paint",
              "group_filter", "chain_topology", "min_vchains"):
        expected[c] = ("wtmm2d", "wtmm2d_roi")
    expected["hline_holder"] = ("mz_edges",)
    assert {c: producers(c) for c in _CONSUMER_PRODUCERS} == expected


def test_contract_fork_forks_a_decomposition_dropped_on_an_analyzed_layer(builtins):
    sibling, shipped = _contract_fork(_refs("wtmm2d"), [False], _d("wtmm2d", "ssa2d"))
    assert _devices(sibling) == ["ssa2d"] and shipped == []
    assert _analyzer_fork(["wtmm2d"], [False], _d("wtmm2d", "ssa2d")) is None   # the table


def test_contract_fork_ships_chain_filters_only_to_a_chain_producer(builtins):
    old = _refs("holder_measure", "chain_holder", "orientation_wedge")
    sibling, shipped = _contract_fork(old, [False] * 3,
                                      _d("holder_measure", "chain_holder", "orientation_wedge",
                                         "mz_edges"))
    assert _devices(sibling) == ["mz_edges"] and shipped == [2]
    _, table = _analyzer_fork([r.device for r in old], [False] * 3,
                              _d("holder_measure", "chain_holder", "orientation_wedge",
                                 "mz_edges"))
    assert table == [1, 2]                               # the table shipped chain_holder too
    _, shipped = _contract_fork(old, [False] * 3,
                                _d("holder_measure", "chain_holder", "orientation_wedge",
                                   "wtmm2d"))
    assert shipped == [1, 2]


def test_contract_fork_ships_topology_with_its_reader(builtins):
    sibling, shipped = _contract_fork(
        _refs("holder_map", "chain_topology", "min_vchains"), [False] * 3,
        _d("holder_map", "chain_topology", "min_vchains", "wtmm2d"))
    assert _devices(sibling) == ["wtmm2d"] and shipped == [1, 2]


def test_contract_fork_ships_hline_holder_to_the_analysis_that_stamps_alpha(builtins):
    sibling, shipped = _contract_fork(_refs("wtmm2d", "hline_holder"), [False, False],
                                      _d("wtmm2d", "hline_holder", "mz_edges"))
    assert _devices(sibling) == ["mz_edges"] and shipped == [1]


def test_contract_fork_keeps_consumers_of_the_parents_own_analysis(builtins):
    sibling, shipped = _contract_fork(
        _refs("wtmm2d", "orientation_wedge", "chain_holder"), [False] * 3,
        _d("wtmm2d", "orientation_wedge", "chain_holder", "holder_map"))
    assert _devices(sibling) == ["holder_map"] and shipped == []


def test_contract_fork_leaves_ordinary_edits_alone(builtins):
    assert _contract_fork(_refs("wtmm2d"), [False], _d("wtmm2d", "orientation_wedge")) is None
    assert _contract_fork(_refs("wtmm2d", "orientation_wedge"), [False, False],
                          _d("orientation_wedge", "wtmm2d")) is None          # reorder
    assert _contract_fork(_refs("wtmm2d"), [False], _d()) is None             # removal
    assert _contract_fork([], [], _d("wtmm2d")) is None
    assert _contract_fork(_refs("noise"), [False], _d("noise", "wtmm2d")) is None
    assert _contract_fork(_refs("holder_map"), [True], _d("holder_map", "wtmm2d")) is None


def test_contract_fork_on_a_stack_master_forks_a_decomposition_and_keeps_noise(builtins):
    sibling, shipped = _contract_fork([], [], _d("pca"), root=True, layer_input=STACK)
    assert _devices(sibling) == ["pca"] and shipped == []
    sibling, shipped = _contract_fork(_refs("noise"), [False], _d("noise", "tucker_HOOI_HOSVD"),
                                      root=True, layer_input=STACK)
    assert _devices(sibling) == ["tucker_HOOI_HOSVD"] and shipped == []
    assert _contract_fork(_refs("noise"), [False], _d("noise"), root=True,
                          layer_input=STACK) is None


def test_contract_fork_carries_a_bus_rows_bus_into_the_sibling(builtins):
    """tucker dropped on a [bus, pca] row analyses the bus's stack: the sibling starts with
    the bus, its sends included."""
    sends = '[{"path": "/a.npz", "subdataset": null, "band": null, "label": "a"}]'
    old = [DeviceRef("bus", {"_sends": sends}), DeviceRef("pca", {})]
    sibling, shipped = _contract_fork(old, [False, False],
                                      [{"device": "bus", "params": {"_sends": sends}},
                                       {"device": "pca", "params": {}},
                                       {"device": "tucker_HOOI_HOSVD", "params": {}}])
    assert _devices(sibling) == ["bus", "tucker_HOOI_HOSVD"] and shipped == []
    assert sibling[0]["params"] == {"_sends": sends}


def test_contract_fork_carries_a_band_rows_band_select_into_the_sibling(builtins):
    """mz_edges dropped on an analysed band row [band_select, wtmm2d] analyses that band, and
    the shipping law is judged after the band_select both chains start with."""
    band = {"band": 2, "_band_id": "B2"}
    old = [DeviceRef("band_select", dict(band)), DeviceRef("wtmm2d", {}),
           DeviceRef("chain_holder", {})]
    proposal = [{"device": "band_select", "params": dict(band)},
                {"device": "wtmm2d", "params": {}}, {"device": "chain_holder", "params": {}},
                {"device": "mz_edges", "params": {}}]
    sibling, shipped = _contract_fork(old, [False] * 3, proposal, layer_input=STACK)
    assert _devices(sibling) == ["band_select", "mz_edges"] and shipped == []
    assert sibling[0]["params"] == band
    old = [DeviceRef("band_select", dict(band)), DeviceRef("holder_measure", {}),
           DeviceRef("orientation_wedge", {})]
    proposal = [{"device": "band_select", "params": dict(band)},
                {"device": "holder_measure", "params": {}},
                {"device": "orientation_wedge", "params": {}},
                {"device": "mz_edges", "params": {}}]
    sibling, shipped = _contract_fork(old, [False] * 3, proposal, layer_input=STACK)
    assert _devices(sibling) == ["band_select", "mz_edges"] and shipped == [2]
