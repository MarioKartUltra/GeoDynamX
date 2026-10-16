# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The optional capabilities a device may have, by name.

The engine and the shell used to learn them by probing attributes (``getattr(device,
"field_stage", False)`` and ten more). A device may now declare ``capabilities = frozenset({...})``
with the names below, and the engine, the ROI runner, the fork logic and the shell ask :func:`has`,
which reads the declaration when there is one and probes only a device that declares nothing;
:func:`differences` lists every registered device whose declaration disagrees with its probes."""
from __future__ import annotations

from dynamix.model.device import DEVICES

#: Capability name -> the attributes the engine or the shell probes for it. A flag attribute counts
#: when truthy; a method or a declaration counts when present and not None/empty.
PROBES: dict[str, tuple[str, ...]] = {
    "FieldStage": ("field_stage",),
    "SourceReader": ("reads_source",),
    "VectorSource": ("vector_source",),
    "RegionAware": ("roi_margin", "compute_roi"),
    "Cancellable": ("wants_cancel",),
    "Previewable": ("preview",),
    "Viewable": ("view",),
    "OutputProducer": ("outputs",),
    "SelectionAware": ("selection_aware",),
    "SelfReporting": ("reading", "data_hints"),
    "StripExtras": ("derived_reading", "param_groups", "check", "knob_warnings"),
}
CAPABILITIES: tuple[str, ...] = tuple(PROBES)


def _present(device, attr: str) -> bool:
    value = getattr(device, attr, None)
    if value is None or value is False:
        return False
    if isinstance(value, (tuple, list, dict, frozenset, set)):
        return len(value) > 0
    return True


def probed(device) -> frozenset[str]:
    """The capabilities the probes find on ``device``."""
    return frozenset(name for name, attrs in PROBES.items()
                     if any(_present(device, a) for a in attrs))


def declared(device) -> frozenset[str] | None:
    """The capabilities ``device`` declares, or None when it declares nothing."""
    value = getattr(device, "capabilities", None)
    if value is None:
        return None
    unknown = sorted(set(value) - set(CAPABILITIES))
    if unknown:
        raise ValueError(f"{getattr(device, 'name', device)!r}: unknown capabilities {unknown}")
    return frozenset(value)


def has(device, name: str) -> bool:
    """Whether ``device`` has capability ``name``: by its declaration when it has one, else by the probes."""
    if name not in PROBES:
        raise KeyError(f"unknown capability {name!r}; known: {CAPABILITIES}")
    d = declared(device)
    return name in (d if d is not None else probed(device))


def undeclared() -> list[str]:
    """Registered devices that declare no capabilities (the probes still serve them)."""
    return sorted(name for name, device in DEVICES.items() if declared(device) is None)


def differences() -> dict[str, tuple[set[str], set[str]]]:
    """Registered devices whose declaration disagrees with their probes:
    ``name -> (declared only, probed only)``; a device that declares nothing is not listed."""
    out = {}
    for name, device in DEVICES.items():
        d = declared(device)
        if d is None:
            continue
        p = probed(device)
        if d != p:
            out[name] = (set(d - p), set(p - d))
    return out
