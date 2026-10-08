# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The fork rule and the shipping law read from the devices' declared contracts
(dynamix.model.forking): which drops fork a sibling, which steps move to it, and what the browser
offers as the next drop."""
from __future__ import annotations

import pytest

from dynamix.model.chain import DeviceRef
from dynamix.model.contracts import FIELD, STACK, next_step_state


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _primary(name):
    from dynamix.model.device import get_device
    from dynamix.model.forking import is_primary

    return is_primary(get_device(name))


def test_primary_analyzers_read_a_raster_and_make_something_else(builtins):
    for name in ("wtmm2d", "mz_edges", "wavelet_skeleton", "cdf_edges", "pm_edges",
                 "holder_measure", "holder_multiaffine", "holder_map", "band_recon",
                 "band_recon_measure", "band_recon_multiaffine", "pca", "tucker_HOOI_HOSVD",
                 "tucker_havok", "ssa2d", "dwt2d", "inpaint"):
        assert _primary(name), name


def test_field_stages_result_tools_filters_and_self_reading_devices_never_fork(builtins):
    for name in ("noise", "band_select", "bus", "fill", "chain_topology", "backproject",
                 "wtmm2d_roi", "derived_vectors", "stub_wavelet", "orientation_wedge",
                 "chain_holder", "min_vchains"):
        assert not _primary(name), name


def test_a_device_declaring_no_contracts_never_forks(stub_transform):
    from dynamix.model.forking import is_primary

    assert not is_primary(stub_transform)


def test_ships_to_moves_a_step_that_applies_only_to_the_siblings_analysis(builtins):
    from dynamix.model.forking import ships_to

    hmap = (DeviceRef("holder_measure", {}),)
    wtmm = (DeviceRef("wtmm2d", {}),)
    mz = (DeviceRef("mz_edges", {}),)
    assert ships_to("chain_holder", hmap, wtmm)
    assert not ships_to("chain_holder", hmap, mz)          # mz_edges makes no V-chains
    assert ships_to("orientation_wedge", hmap, mz)
    assert not ships_to("orientation_wedge", wtmm, mz)     # it applies to both: it stays
    assert ships_to("hline_modulus", mz, wtmm, FIELD, {"frac": 0.0, "mode": "sup_all_scales"})
    assert not ships_to("hline_modulus", mz, wtmm, FIELD, {"frac": 0.0, "mode": "sup"})


def test_drop_states_on_a_field_master(builtins):
    from dynamix.model.device import DEVICES
    from dynamix.model.forking import drop_states

    states = drop_states((), FIELD, master=True)
    assert set(states) == set(DEVICES)
    assert states["wtmm2d"].applies and states["noise"].applies
    assert not states["pca"].applies and not states["chain_holder"].applies
    assert states["pca"] == next_step_state((), FIELD, "pca")


def test_drop_states_offer_stack_tools_on_a_stack_and_single_band_tools_on_its_band_rows(
        builtins):
    from dynamix.model.forking import drop_states

    stack = drop_states((), STACK, master=True)
    for name in ("pca", "tucker_HOOI_HOSVD", "noise", "band_select"):
        assert stack[name].applies, name
    for name in ("wtmm2d", "mz_edges", "holder_measure", "ssa2d"):
        assert not stack[name].applies, name
    band_row = drop_states((DeviceRef("band_select", {"band": 1}),), STACK)
    for name in ("wtmm2d", "mz_edges", "holder_measure", "ssa2d", "tucker_HOOI_HOSVD"):
        assert band_row[name].applies, name
    assert not band_row["pca"].applies
    bus_row = drop_states((DeviceRef("bus", {}),), FIELD)
    assert bus_row["pca"].applies and not bus_row["wtmm2d"].applies


def test_drop_states_judge_a_forking_analyzer_on_the_layers_input(builtins):
    from dynamix.model.forking import drop_states

    on_wtmm = drop_states((DeviceRef("wtmm2d", {}),), FIELD)
    for name in ("mz_edges", "holder_measure", "ssa2d", "chain_holder", "chain_topology",
                 "orientation_wedge", "wtmm2d_roi"):
        assert on_wtmm[name].applies, name
    assert not on_wtmm["noise"].applies               # appended after the analysis
    cdf = (DeviceRef("cdf_edges", {}),)
    on_cdf = drop_states(cdf, FIELD)
    assert on_cdf["orientation_wedge"].applies
    assert on_cdf["chain_holder"] == next_step_state(cdf, FIELD, "chain_holder")
    assert not on_cdf["chain_holder"].applies


def test_routing_head_is_the_leading_run_of_band_select_and_bus(builtins):
    from dynamix.model.forking import routing_head

    band, bus = DeviceRef("band_select", {"band": 2}), DeviceRef("bus", {})
    assert routing_head((band, DeviceRef("wtmm2d", {}))) == (band,)
    assert routing_head((bus, DeviceRef("pca", {}))) == (bus,)
    assert routing_head((DeviceRef("noise", {}), band)) == ()
    assert routing_head(()) == ()


def test_drop_states_judge_a_forking_analyzer_after_the_layers_routing_steps(builtins):
    """A sibling forked from a band row or a bus row starts with its band_select or bus, so a
    second analyzer is judged on the band or the bus's stack, never on the dataset's input."""
    from dynamix.model.forking import drop_states

    band = DeviceRef("band_select", {"band": 1})
    band_row = drop_states((band, DeviceRef("wtmm2d", {})), STACK)
    assert band_row["mz_edges"] == next_step_state((band,), STACK, "mz_edges")
    assert band_row["mz_edges"].applies and band_row["ssa2d"].applies
    assert not band_row["pca"].applies
    bus = DeviceRef("bus", {})
    bus_row = drop_states((bus, DeviceRef("pca", {})), FIELD)
    assert bus_row["tucker_HOOI_HOSVD"].applies
    assert bus_row["wtmm2d"] == next_step_state((bus,), FIELD, "wtmm2d")
    assert not bus_row["wtmm2d"].applies


def test_drop_states_on_an_roi_holder_judge_an_analyzer_in_the_heading_ones_place(builtins):
    from dynamix.model.forking import drop_states

    noise = DeviceRef("noise", {})
    states = drop_states((noise, DeviceRef("wtmm2d", {})), FIELD, roi_holder=True)
    assert states["mz_edges"] == next_step_state((noise,), FIELD, "mz_edges")
    assert states["mz_edges"].applies


def test_drop_states_under_a_shown_level_grey_scale_select_alone(builtins):
    from dynamix.model.contracts import SCALE_BY_TRANSPORT, StepState
    from dynamix.model.forking import drop_states

    wtmm = (DeviceRef("wtmm2d", {}),)
    plain = drop_states(wtmm, FIELD)
    viewed = drop_states(wtmm, FIELD, level_view=True)
    assert plain["scale_select"].applies
    assert viewed["scale_select"] == StepState(False, SCALE_BY_TRANSPORT)
    assert {n: s for n, s in viewed.items() if n != "scale_select"} == \
        {n: s for n, s in plain.items() if n != "scale_select"}
