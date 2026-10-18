# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tool families and display names: a device's ``label`` and ``families`` are presentation only,
an undeclared device is shown by its id under "Other", and an empty ``families`` lists a device in
no family."""
from __future__ import annotations

import pytest

from dynamix.model.device import DEVICES, get_device
from dynamix.model.families import FAMILY_ORDER, display_name, families_of, label_of


class _Bare:
    name = "third_party"


class _Labelled:
    name = "wtmm2d"
    label = "WTMM 2D"
    families = ("Edge detection", "Regularity")


class _Superseded:
    name = "holder_map"
    families = ()


def test_the_family_order_is_the_browser_tree():
    assert FAMILY_ORDER == ("Data preparation", "Edge detection", "Regularity", "Skeletons",
                            "Decomposition", "Reconstruction", "Topology", "Selection: extrema",
                            "Selection: V-chains", "Points", "Microstructure", "Labs", "Other")


def test_an_undeclared_device_is_its_id_under_other():
    device = _Bare()
    assert label_of(device) == "third_party"
    assert families_of(device) == ("Other",)
    assert display_name(device) == "third_party"


def test_a_labelled_device_shows_its_label_then_its_id():
    device = _Labelled()
    assert label_of(device) == "WTMM 2D"
    assert families_of(device) == ("Edge detection", "Regularity")
    assert display_name(device) == "WTMM 2D  (wtmm2d)"


def test_an_empty_families_tuple_lists_the_device_in_no_family():
    device = _Superseded()
    assert families_of(device) == ()
    assert label_of(device) == "holder_map"
    assert display_name(device) == "holder_map"


# --- the builtin devices ----------------------------------------------------------------------

#: id -> display name, for every tool that declares one.
LABELS = {
    "band_select": "Band select",
    "noise": "Noise / dither",
    "periodic_extension": "Periodic extension (DAF)",
    "wtmm2d": "WTMM 2D",
    "wtmm2d_roi": "WTMM 2D, ROI",
    "mz_edges": "Mallat–Zhong edges",
    "cdf_edges": "CDF edges",
    "pm_edges": "Perona–Malik edges",
    "holder_measure": "Microcanonical MFA, measure",
    "holder_multiaffine": "Microcanonical MFA, multiaffine",
    "wavelet_skeleton": "Wavelet skeleton",
    "lineal_path": "Lineal path",
    "invariant_measures": "Invariant measures (Gunaratne)",
    "pca": "PCA",
    "tucker_HOOI_HOSVD": "Tucker (HOOI / HOSVD)",
    "ssa2d": "2-D SSA",
    "dwt2d": "DWT",
    "inpaint": "Inpaint",
    "fill": "Fill",
    "band_recon_measure": "Band reconstruction, measure",
    "band_recon_multiaffine": "Band reconstruction, multiaffine",
    "chain_topology": "Chain topology",
    "scale_select": "Scale",
    "orientation_wedge": "Orientation wedge",
    "modulus_threshold": "Modulus threshold",
    "hline_length": "H-chain length",
    "hline_modulus": "H-chain modulus",
    "hline_holder": "H-chain Hölder",
    "chain_holder": "Chain Hölder",
    "chain_modulus": "Chain modulus",
    "chain_length": "Chain length",
    "chain_classify": "Seam classifier",
    "min_vchains": "Min V-chains per H-chain",
    "group_filter": "Group filter",
    "backproject": "Back-project points",
}

#: family -> the ids listed under it.
FAMILY_TREE = {
    "Data preparation": {"band_select", "noise", "fill", "periodic_extension"},
    "Edge detection": {"wtmm2d", "wtmm2d_roi", "mz_edges", "cdf_edges", "pm_edges"},
    "Regularity": {"wtmm2d", "wtmm2d_roi", "mz_edges", "holder_measure", "holder_multiaffine"},
    "Skeletons": {"wavelet_skeleton"},
    "Decomposition": {"pca", "tucker_HOOI_HOSVD", "ssa2d", "dwt2d"},
    "Reconstruction": {"mz_edges", "holder_measure", "holder_multiaffine", "ssa2d",
                       "tucker_HOOI_HOSVD", "inpaint"},
    "Topology": {"chain_topology"},
    "Selection: extrema": {"orientation_wedge", "modulus_threshold", "hline_length",
                           "hline_modulus", "hline_holder"},
    "Selection: V-chains": {"chain_holder", "chain_modulus", "chain_length", "chain_classify",
                            "min_vchains", "group_filter"},
    "Points": {"backproject"},
    "Microstructure": {"lineal_path", "invariant_measures"},
    "Other": set(),
    "Labs": set(),                        # experimental plugins declare it; no built-in does
}

#: Superseded and shell-placed devices: registered, listed in no family.
NO_FAMILY = ("holder_map", "tucker_havok", "band_recon", "scale_select", "bus",
             "derived_vectors", "group_paint")


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def test_every_registered_device_is_listed_in_a_family_or_in_none(builtins):
    stubs = {n for n in DEVICES if n.startswith("stub_")}
    assert stubs == {"stub_wavelet", "stub_holder", "stub_wedge"}
    assert set(DEVICES) == set(LABELS) | set(NO_FAMILY) | stubs
    for name, device in DEVICES.items():
        assert hasattr(device, "families") == (name not in stubs), name


def test_every_listed_tool_declares_its_display_name(builtins):
    for name, label in LABELS.items():
        assert get_device(name).label == label, name
    assert display_name(get_device("chain_classify")) == "Seam classifier  (chain_classify)"


def test_the_family_tree_is_the_family_table(builtins):
    tree = {family: set() for family in FAMILY_ORDER}
    for name, device in DEVICES.items():
        if name.startswith("stub_"):
            continue
        for family in families_of(device):
            tree[family].add(name)
    assert tree == FAMILY_TREE


def test_superseded_and_shell_placed_devices_are_in_no_family(builtins):
    for name in NO_FAMILY:
        assert families_of(get_device(name)) == (), name


def test_the_band_recon_variants_keep_their_label_in_no_family(builtins):
    """Their band reconstruction is the band rows of holder_measure and holder_multiaffine: the
    variants stay registered with their labels and are listed in no family."""
    for name, label in (("band_recon_measure", "Band reconstruction, measure"),
                        ("band_recon_multiaffine", "Band reconstruction, multiaffine")):
        device = get_device(name)
        assert families_of(device) == (), name
        assert display_name(device) == f"{label}  ({name})"


def test_scale_select_keeps_its_label_in_no_family(builtins):
    """The Transport sets the shown level: scale_select stays registered with its label "Scale"
    and is listed in no family."""
    device = get_device("scale_select")
    assert families_of(device) == ()
    assert display_name(device) == "Scale  (scale_select)"
