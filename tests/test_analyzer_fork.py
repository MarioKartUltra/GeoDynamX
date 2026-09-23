# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The fork decision (_analyzer_fork, main_window): a DIFFERENT primary analyzer dropped on
an analyzed layer forks a sibling representation; ordinary edits commit as before. Returns
(sibling_steps, shipped_parent_indices) -- the shipping law: consumers follow their producer,
so a consumer stranded on the parent whose producers include the NEW analyzer ships over."""
from __future__ import annotations

from dynamix.shell.main_window import _analyzer_fork


def _d(*names, bypassed=()):
    return [{"device": n, "params": {}, "bypassed": n in bypassed, "rack": None}
            for n in names]


def _devices(steps):
    return [s["device"] for s in steps]


def test_append_of_a_different_analyzer_forks_it_alone():
    sibling, shipped = _analyzer_fork(["holder_map"], [False], _d("holder_map", "wtmm2d"))
    assert _devices(sibling) == ["wtmm2d"] and shipped == []


def test_preset_replacing_the_analyzer_forks_the_whole_preset():
    sibling, shipped = _analyzer_fork(["holder_map"], [False],
                                      _d("wtmm2d", "scale_select", "chain_holder"))
    assert _devices(sibling) == ["wtmm2d", "scale_select", "chain_holder"]
    assert shipped == []


def test_multiset_diff_ships_a_reincluded_duplicate():
    """A preset re-including a device the parent also holds must still fork a copy -- the
    name-subtraction hole (2026-09-17)."""
    sibling, _ = _analyzer_fork(["holder_map", "scale_select"], [False, False],
                                _d("holder_map", "scale_select", "wtmm2d", "scale_select"))
    assert _devices(sibling) == ["wtmm2d", "scale_select"]


def test_stranded_consumers_ship_to_the_new_analyzer():
    """The shipping law: chain consumers sitting uselessly on a holder layer move to the
    forked wtmm sibling (as indices, so params travel)."""
    sibling, shipped = _analyzer_fork(
        ["holder_map", "chain_holder", "scale_select"], [False, False, False],
        _d("holder_map", "chain_holder", "scale_select", "wtmm2d"))
    assert _devices(sibling) == ["wtmm2d"]
    assert shipped == [1, 2]


def test_consumers_of_the_parents_own_analyzer_never_move():
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


def test_mz_edges_onto_wtmm_forks_the_extrema_rep():
    sibling, shipped = _analyzer_fork(["wtmm2d", "scale_select"], [False, False],
                                      _d("wtmm2d", "scale_select", "mz_edges"))
    assert _devices(sibling) == ["mz_edges"]
    assert shipped == []          # scale_select reads wtmm2d too -- it stays home


def test_master_root_never_takes_an_analyzer_directly():
    """Root protection (2026-09-18): a primary dropped on a RAW master forks a child; the
    master's own recipe stays empty. Non-root raw layers keep the ordinary edit."""
    sibling, shipped = _analyzer_fork([], [], _d("holder_map"), root=True)
    assert _devices(sibling) == ["holder_map"] and shipped == []
    sibling, shipped = _analyzer_fork([], [], _d("wtmm2d", "scale_select"), root=True)
    assert _devices(sibling) == ["wtmm2d", "scale_select"]
    assert _analyzer_fork([], [], _d("holder_map"), root=False) is None


def test_master_keeps_filters_and_noise_without_forking():
    """Field-prep and filter edits on the master are ordinary edits; a later analyzer drop
    forks WITHOUT taking noise along (documented: prep ships via a preset, not the fork)."""
    assert _analyzer_fork([], [], _d("noise"), root=True) is None
    sibling, shipped = _analyzer_fork(["noise"], [False], _d("noise", "wtmm2d"), root=True)
    assert _devices(sibling) == ["wtmm2d"] and shipped == []


def test_split_holder_tools_are_primary_analyzers():
    """2026-09-19 split: each method is its own representation -- dropping the OTHER method
    (or the conflated predecessor's sibling) forks a sibling layer, exactly as wtmm2d does."""
    sibling, shipped = _analyzer_fork(["holder_multiaffine"], [False],
                                      _d("holder_multiaffine", "holder_measure"))
    assert _devices(sibling) == ["holder_measure"] and shipped == []
    sibling, shipped = _analyzer_fork(["holder_map"], [False],
                                      _d("holder_map", "holder_multiaffine"))
    assert _devices(sibling) == ["holder_multiaffine"] and shipped == []


def test_band_dialog_commit_picks_the_variant_of_the_recipes_producer():
    """The 2026-09-19 source split: a Band-dialog commit creates the band_recon VARIANT whose
    engine params are the recipe's h-map producer's verbatim; the conflated holder_map keeps
    the conflated band_recon (the old path, byte-compatible for saved projects)."""
    from dynamix.shell.main_window import _band_device_for, _band_step_for

    assert _band_device_for(["holder_measure"]) == "band_recon_measure"
    assert _band_device_for(["holder_multiaffine", "scale_select"]) == \
        "band_recon_multiaffine"
    assert _band_device_for(["holder_map"]) == "band_recon"
    assert _band_device_for([]) == "band_recon"

    assert _band_step_for(["holder_measure", "band_recon_measure"]) == "band_recon_measure"
    assert _band_step_for(["band_recon"]) == "band_recon"
    assert _band_step_for(["holder_measure"]) is None


def test_wavelet_skeleton_is_a_primary_analyzer():
    """The medial-axis extractor is its own representation: dropping it on a wtmm2d layer
    forks a sibling, exactly as mz_edges does."""
    sibling, shipped = _analyzer_fork(["wtmm2d"], [False],
                                      _d("wtmm2d", "wavelet_skeleton"))
    assert _devices(sibling) == ["wavelet_skeleton"] and shipped == []


def test_wavelet_skeleton_is_a_chain_producer_for_the_shipping_law():
    """Adversarial review (2026-09-19): the skeleton emits the extrema schema the chain
    consumers read, so it belongs in _CHAIN_PRODUCERS -- otherwise a fork STRIPS live,
    tuned filters off a skeleton layer (measured before the fix: shipped=[1,2] where the
    identical wtmm2d-parent fork ships nothing), and stranded consumers never ship TO a
    skeleton sibling."""
    sibling, shipped = _analyzer_fork(
        ["wavelet_skeleton", "scale_select", "orientation_wedge"], [False] * 3,
        _d("wavelet_skeleton", "scale_select", "orientation_wedge", "mz_edges"))
    assert shipped == []                       # live consumers stay on the skeleton layer
    sibling, shipped = _analyzer_fork(
        ["holder_map", "scale_select"], [False, False],
        _d("holder_map", "scale_select", "wavelet_skeleton"))
    assert _devices(sibling) == ["wavelet_skeleton"]
    assert shipped == [1]                      # stranded consumer ships to the sibling


def test_cdf_edges_is_a_primary_analyzer_and_chain_producer():
    sibling, shipped = _analyzer_fork(["wtmm2d"], [False], _d("wtmm2d", "cdf_edges"))
    assert _devices(sibling) == ["cdf_edges"] and shipped == []
    sibling, shipped = _analyzer_fork(["cdf_edges", "scale_select"], [False, False],
                                      _d("cdf_edges", "scale_select", "mz_edges"))
    assert shipped == []                       # its consumers stay on their producer


def test_pm_edges_is_a_primary_analyzer_and_chain_producer():
    """Perona-Malik proper (2026-09-21) is the fifth peer analyzer: dropping it on an
    analyzed layer forks a sibling, and its extrema pyramid feeds the same chain
    consumers, so its consumers must stay put on a fork exactly as cdf_edges' do."""
    sibling, shipped = _analyzer_fork(["wtmm2d"], [False], _d("wtmm2d", "pm_edges"))
    assert _devices(sibling) == ["pm_edges"] and shipped == []
    sibling, shipped = _analyzer_fork(["pm_edges", "scale_select"], [False, False],
                                      _d("pm_edges", "scale_select", "mz_edges"))
    assert shipped == []                       # its consumers stay on their producer
