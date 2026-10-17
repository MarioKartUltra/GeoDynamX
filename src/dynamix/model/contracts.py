# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Data contracts: the named kinds of data a chain passes between devices, and the one rule that
says whether a step applies.

A device declares ``consumes`` -- alternatives, any one of which is enough, each a tuple of
contracts that must all be available -- and ``produces``, the contracts its result carries, where
``RECEIVED`` stands for everything the step received. Either may instead be a function of the
validated params. A device that declares no ``consumes`` always applies, and a Transform that
declares no ``produces`` counts as producing every contract (``ALL_CONTRACTS``): what its result
holds is unknown, so every step after it applies and a third-party device keeps working unchanged.

The rule: the layer's input is available before the first step. A Transform applies when one of
its alternatives is available and then replaces the available set with what it produces; a Filter
applies when one of its alternatives is available and leaves the set as it is. A step that does
not apply leaves the set as it is. Under a shown level (``level_view``) the engine has already
reduced the extrema stack to the level the Transport shows, so ``scale_select`` never applies.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from dynamix.model.device import DEVICES, defaults_for, get_device, is_transform, validate_params

FIELD, STACK, POINTS = "Field", "Stack", "Points"
EXTREMA, ALPHA, VCHAINS, TOPOLOGY, HMAP, COMPONENTS = (
    "ExtremaStack", "PointAlpha", "VChains", "Topology", "HMap", "Components")
CURVE = "Curve"
#: In a ``produces`` declaration: every contract the step received passes through.
RECEIVED = "<received>"

#: Contract -> its meaning, phrased to read after "needs" and after "the layer's input is".
CONTRACTS: dict[str, str] = {
    FIELD: "a single-band field",
    STACK: "a band stack",
    POINTS: "a point set",
    EXTREMA: "an extrema stack",
    ALPHA: "a per-point α",
    VCHAINS: "V-chains",
    TOPOLOGY: "the H/V topology",
    HMAP: "an h-map",
    COMPONENTS: "decomposition components",
    CURVE: "a curve (r, L(r))",
}

#: Every contract: what a Transform that declares no ``produces`` counts as producing.
ALL_CONTRACTS = frozenset(CONTRACTS)

#: Why ``scale_select`` does not apply under a shown level.
SCALE_BY_TRANSPORT = "the scale is set by the Transport"

#: Contract -> the result keys it covers. ``PointAlpha`` lives inside each extrema level
#: (``extrema[i]["alpha"]``), so it is checked per level; the raster and point contracts are the
#: field or point set itself.
CONTRACT_KEYS: dict[str, tuple[str, ...]] = {
    FIELD: (),
    STACK: (),
    POINTS: (),
    EXTREMA: ("extrema", "scales", "_shape"),
    ALPHA: (),
    VCHAINS: ("chains",),
    TOPOLOGY: ("topology", "_topology_vids"),
    HMAP: ("h_map", "r2_map"),
    COMPONENTS: ("raster_out",),
    CURVE: ("r", "L"),
}


@dataclasses.dataclass(frozen=True)
class StepState:
    """Whether a chain step applies; ``reason`` says why not when it does not."""

    applies: bool
    reason: str = ""


def input_contract(field) -> str:
    """The contract a layer's input satisfies: a point set, a band stack ``(ny, nx, nc)`` with
    ``nc > 1``, or a single-band field. ``field`` may be a field object or its bare values array;
    anything else (``None``, an opaque object) reads as a single-band field."""
    from dynamix.core.pointset import PointSet

    if isinstance(field, PointSet):
        return POINTS
    values = np.asarray(getattr(field, "values", field))
    if values.ndim == 3 and values.shape[-1] > 1:
        return STACK
    return FIELD


def consumes_of(device, params: dict) -> tuple[tuple[str, ...], ...]:
    """The device's ``consumes`` for these params: any-of alternatives, each all-of. ``()`` for a
    device that declares none."""
    decl = getattr(device, "consumes", None)
    if decl is None:
        return ()
    if callable(decl):
        decl = decl(params)
    return tuple(tuple(alt) for alt in decl)


def produces_of(device, params: dict, received: frozenset) -> frozenset:
    """What the device's result carries, given what it received: ``RECEIVED`` expands to
    ``received``; a device that declares no ``produces`` counts as producing ``ALL_CONTRACTS``."""
    decl = getattr(device, "produces", None)
    if decl is None:
        return ALL_CONTRACTS
    if callable(decl):
        decl = decl(params)
    out = set()
    for contract in decl:
        if contract == RECEIVED:
            out |= set(received)
        else:
            out.add(contract)
    return frozenset(out)


def _label(device) -> str:
    from dynamix.model.families import label_of

    return label_of(device)


def _join(labels: Sequence[str]) -> str:
    labels = list(labels)
    if len(labels) < 2:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def producers_of(contract: str) -> tuple[str, ...]:
    """Registered device ids whose produces, at default params, include ``contract``; the stubs
    and the devices in no family (superseded or shell-placed) are left out. Registration order."""
    from dynamix.model.families import families_of

    out = []
    for name, device in DEVICES.items():
        if name.startswith("stub_") or not families_of(device):
            continue
        decl = getattr(device, "produces", None)
        if decl is None:
            continue
        if callable(decl):
            decl = decl(defaults_for(device))
        if contract in decl:
            out.append(name)
    return tuple(out)


def _state(device, params: dict, available: frozenset, analysis, layer_input: str,
           level_view: bool) -> StepState:
    if level_view and device.name == "scale_select":
        return StepState(False, SCALE_BY_TRANSPORT)
    alternatives = consumes_of(device, params)
    if not alternatives or any(set(alt) <= available for alt in alternatives):
        return StepState(True)
    missing = []
    for alt in alternatives:
        first = next(c for c in alt if c not in available)
        if first not in missing:
            missing.append(first)
    needs = " or ".join(CONTRACTS.get(c, c) for c in missing)
    producers = []
    for c in missing:
        for name in producers_of(c):
            if name not in producers:
                producers.append(name)
    labels = [_label(get_device(n)) for n in producers]
    reason = f"needs {needs}"
    if labels:
        reason += f": {_join(labels)} {'produces' if len(labels) == 1 else 'produce'} it"
    if analysis is None:
        reason += f"; the layer's input is {CONTRACTS.get(layer_input, layer_input)}"
    else:
        reason += f"; the current analysis ({_label(analysis)}) does not"
    return StepState(False, reason)


def _walk(steps, layer_input: str, level_view: bool):
    """``(states, available, analysis)`` after ``steps``: one StepState per step, the contracts
    available after the last step, and the last applying Transform (``None`` when none ran)."""
    available = frozenset({layer_input})
    analysis = None
    states = []
    for ref in steps:
        device = get_device(ref.device)
        params = validate_params(device, ref.params)
        state = _state(device, params, available, analysis, layer_input, level_view)
        states.append(state)
        if state.applies and is_transform(device):
            available = produces_of(device, params, available)
            analysis = device
    return states, available, analysis


def step_states(steps, layer_input: str, *, level_view: bool = False) -> list[StepState]:
    """One StepState per step of ``steps`` (``DeviceRef``s, chain order) on a layer whose input
    is ``layer_input``. ``level_view``: the chain is resolved at a shown level, so
    ``scale_select`` does not apply."""
    return _walk(steps, layer_input, level_view)[0]


def chain_contracts(steps, layer_input: str) -> frozenset:
    """The contracts a layer's chain leaves available after its last applying step."""
    return _walk(steps, layer_input, False)[1]


def next_step_state(steps, layer_input: str, device_name: str, params: dict | None = None, *,
                    level_view: bool = False) -> StepState:
    """Whether ``device_name`` would apply if added to ``steps``: a Transform after the
    transforms, a Filter at the end. Filters never change the available set, so both positions
    see the set the whole chain leaves. ``level_view`` as in :func:`step_states`."""
    _states, available, analysis = _walk(steps, layer_input, level_view)
    device = get_device(device_name)
    return _state(device, validate_params(device, params or {}), available, analysis,
                  layer_input, level_view)


# --- the fill stamp ----------------------------------------------------------------------

def fill_of(source):
    """The fill stamp a step's input carries: a result's ``_fill``, a field's
    ``provenance["fill"]`` (written by the ``fill`` step), else ``None``."""
    if isinstance(source, dict):
        return source.get("_fill")
    prov = getattr(source, "provenance", None) or {}
    return prov.get("fill")


def stamp_fill(result, source):
    """An analyzer's result stamped with the fill of the field it analysed (``_fill``), so a
    reading downstream can say that analysis inside the region reads a reconstruction.
    Nothing is stamped on a field tail or when the input carries no fill."""
    if not isinstance(result, dict):
        return result
    stamp = fill_of(source)
    if stamp is None:
        return result
    out = dict(result)
    out["_fill"] = stamp
    return out
