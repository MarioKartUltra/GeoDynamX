# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tool families and display names: how the browser and the chain strip present a device.

A device may declare ``label`` (its display name) and ``families`` (the scientific families it is
listed under; a tool may sit in several). Both are presentation only: no saved chain, cache key,
rack file or provenance step identifier reads them, so the device id stays the identity. A device
that declares no families is listed under "Other" by its id; one that declares an empty tuple is
superseded or shell-placed and is listed in no family.
"""
from __future__ import annotations

FAMILY_ORDER = ("Data preparation", "Edge detection", "Regularity", "Skeletons", "Decomposition",
                "Reconstruction", "Topology", "Selection: extrema", "Selection: V-chains",
                "Points", "Microstructure", "Labs", "Other")


def label_of(device) -> str:
    """The device's display name, or its id when it declares none."""
    return getattr(device, "label", "") or device.name


def families_of(device) -> tuple[str, ...]:
    """The families the device declares, or ``("Other",)`` when it declares none."""
    declared = getattr(device, "families", None)
    return ("Other",) if declared is None else tuple(declared)


def display_name(device) -> str:
    """``"<label>  (<id>)"`` when the label differs from the id, else the id."""
    label = label_of(device)
    return label if label == device.name else f"{label}  ({device.name})"


def tooltip_of(device) -> str:
    """The browser's tooltip: the display name, then the device's origin (authors, licence,
    citation, URL; ``dynamix.model.origin``), one per line."""
    from dynamix.model.origin import origin_of_device

    return "\n".join([display_name(device), *origin_of_device(device).lines()])
