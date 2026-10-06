# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tool families and display names: a device's ``label`` and ``families`` are presentation only,
an undeclared device is shown by its id under "Other", and an empty ``families`` lists a device in
no family."""
from __future__ import annotations

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
                            "Selection: V-chains", "Points", "Other")


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
