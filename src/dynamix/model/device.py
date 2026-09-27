# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Devices: the units a chain is built from.

Two kinds, and the split is load-bearing because it is where interactivity comes from. A
``Transform`` is expensive and cached, keyed by a content hash of (source, params); a ``Filter`` is
a cheap pure function over an already-computed result and must never show a progress bar.

Generalises ``dynamix.core.modality.AnalysisModality``, which described whole analyses
(compute/overlay/detail_widget_factory) rather than chainable units, and coupled the seam to Qt.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable

from dynamix.model.param import Param


@runtime_checkable
class Device(Protocol):
    name: str
    params: tuple[Param, ...]

    # Optional: def check(self, params: dict) -> None
    #   Cross-parameter validation, for constraints no single Param can express (a min/max pair,
    #   a scale range that must be ascending). Raise ValueError. Called by validate_params.


@runtime_checkable
class Transform(Device, Protocol):
    """Expensive. Its result is cached and reused across filter changes."""

    def compute(self, field, params: dict, *, progress=None) -> dict: ...

    def cache_key(self, source_id: str, params: dict) -> str: ...

    # Optional: outputs: tuple[Output, ...]
    #   What the tool produces that a person can show or hide on its own (``declared_outputs``).
    # Optional: def compute_output(self, name, values, result, params, *, fetch, progress=None,
    #                              cancel=None) -> dict
    #   Computes a lazy output from the cached analysis ``result`` and the analysed ``values``;
    #   ``fetch(other)`` resolves another lazy output of the same layer through the cache. Called
    #   by ``dynamix.engine.resolve.resolve_output``.


@runtime_checkable
class Filter(Device, Protocol):
    """Cheap and pure. Runs on every redraw, inside the 16 ms budget."""

    def apply(self, result: dict, params: dict) -> dict: ...


@dataclasses.dataclass(frozen=True)
class Output:
    """One thing a tool produces that a person can show or hide on its own. ``kind`` is
    "raster" or "vector"; an eager output is already in the analysis result under ``name``;
    a lazy one is computed on request by ``device.compute_output`` and cached under its own key,
    which folds in the view-only ``params`` it depends on. ``grid`` is "native", or "stride"
    for a raster drawn on its own coarser grid."""

    name: str
    kind: str
    lazy: bool = False
    params: tuple[str, ...] = ()
    label: str = ""
    grid: str = "native"


def declared_outputs(device) -> tuple[Output, ...]:
    """The outputs ``device`` declares (its ``outputs`` attribute), or ``()`` for one that
    declares none."""
    return tuple(getattr(device, "outputs", ()))


DEVICES: dict[str, Device] = {}


def register_device(device: Device) -> None:
    """Register by name. Raises ValueError on a duplicate, as modality.register_modality does, and
    on a device that is neither shape.

    The shape check is worth more than it looks. Kind is decided structurally, so a device whose
    ``cache_key`` is misspelled is not rejected -- it silently becomes a Filter, and the chain then
    fails much later with "chain must be transforms-then-filters", which misdiagnoses a typo as an
    ordering mistake and sends the author looking in the wrong file.
    """
    name = getattr(device, "name", "")
    if not name:
        raise ValueError(f"device {device!r} declares no name")
    if not isinstance(getattr(device, "params", None), tuple):
        raise ValueError(f"device {name!r} declares no params tuple")
    if not is_transform(device) and not hasattr(device, "apply"):
        missing = [a for a in ("compute", "cache_key") if not hasattr(device, a)]
        raise ValueError(
            f"device {name!r} is neither a Transform nor a Filter: a Transform needs compute + "
            f"cache_key (missing {missing}), a Filter needs apply"
        )
    if name in DEVICES:
        raise ValueError(f"device {name!r} is already registered")
    DEVICES[name] = device


def get_device(name: str) -> Device:
    if name not in DEVICES:
        raise KeyError(f"no device named {name!r}; known: {sorted(DEVICES)}")
    return DEVICES[name]


def is_transform(device: Device) -> bool:
    """True for a Transform. Checked structurally, so a device needs no base class."""
    return hasattr(device, "compute") and hasattr(device, "cache_key")


def defaults_for(device: Device) -> dict[str, Any]:
    return {p.name: p.default for p in device.params}


def validate_params(device: Device, params: dict) -> dict[str, Any]:
    """Fill defaults, validate every value, and reject unknown keys."""
    schema = {p.name: p for p in device.params}
    unknown = set(params) - set(schema)
    if unknown:
        raise ValueError(
            f"{device.name}: unknown parameter(s) {sorted(unknown)}; known: {sorted(schema)}"
        )
    out = {}
    for name, p in schema.items():
        out[name] = p.validate(params[name]) if name in params else p.default
    check = getattr(device, "check", None)
    if check is not None:
        check(out)            # cross-parameter validation; raises ValueError
    return out


def keyed_params(device: Device, params: dict) -> dict[str, Any]:
    """``params`` without the device's view-only params (``Param.view``) -- what a cache key and
    a transform signature are built from. A display selector must never be keyed: switching
    which computed output is shown would otherwise recompute the whole tool."""
    view = {p.name for p in getattr(device, "params", ()) if p.view}
    return {k: v for k, v in params.items() if k not in view}


def apply_view(device: Device, result, params: dict):
    """The device's ``view(result, params)`` applied to a (cached) result, or ``result`` itself
    for a device with no view step. Runs after the cache, on every resolve."""
    view = getattr(device, "view", None)
    return result if view is None else view(result, params)
