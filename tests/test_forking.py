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
                 "tucker_havok", "ssa2d"):
        assert _primary(name), name


def test_field_stages_result_tools_filters_and_self_reading_devices_never_fork(builtins):
    for name in ("noise", "band_select", "bus", "chain_topology", "backproject",
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
