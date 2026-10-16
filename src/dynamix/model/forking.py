# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The fork rule and the shipping law, read from the devices' declared contracts.

A primary analyzer reads the layer's raster (a single-band field or a band stack) and makes
something that is no longer a raster. Dropping a different one on a layer a primary already heads,
or on a dataset's master row, forks a sibling layer over the same input. A step on the parent moves
to the sibling when it applies to the sibling's analysis and not to the parent's.
"""
from __future__ import annotations

from collections.abc import Iterable

from dynamix.model.chain import DeviceRef
from dynamix.model import capabilities
from dynamix.model.contracts import FIELD, STACK, consumes_of, next_step_state, produces_of
from dynamix.model.device import DEVICES, defaults_for, get_device, is_transform

_RASTERS = frozenset((FIELD, STACK))


def is_primary(device) -> bool:
    """True for a Transform that consumes a single-band field or a band stack in one of its
    declared alternatives and, given that input, produces at least one contract and no raster.
    A device that reads its own data (``reads_source``: the ROI WTMM reading the source file;
    ``vector_source``: a derivative dataset's vector loader) is part of how its layer is read and
    never counts."""
    if (not is_transform(device) or capabilities.has(device, "SourceReader")
            or capabilities.has(device, "VectorSource")):
        return False
    params = defaults_for(device)
    for alternative in consumes_of(device, params):
        received = frozenset(alternative)
        produced = produces_of(device, params, received)
        if received & _RASTERS and produced and produced.isdisjoint(_RASTERS):
            return True
    return False


def ships_to(consumer_name: str, parent_analysis, sibling_analysis, layer_input: str = FIELD,
             params: dict | None = None) -> bool:
    """Whether a step on a parent layer moves to a sibling forked from it: it applies after the
    sibling's transforms and does not apply after the parent's. ``parent_analysis`` and
    ``sibling_analysis`` are the enabled Transform steps (``DeviceRef``) of the two chains, both
    over ``layer_input`` (a forked sibling reads its parent's field); ``params`` are the step's
    own (``None``: its defaults)."""
    return (next_step_state(sibling_analysis, layer_input, consumer_name, params).applies
            and not next_step_state(parent_analysis, layer_input, consumer_name, params).applies)


#: The routing steps: they decide which raster a layer's analysis reads (one band of a stack, a
#: bus's stack of sends), so a sibling forked from the layer carries them over.
_ROUTING = ("band_select", "bus")


def routing_head(steps: Iterable[DeviceRef]) -> tuple[DeviceRef, ...]:
    """The leading run of routing steps (``band_select``, ``bus``) of ``steps``, as the same
    ``DeviceRef`` objects in order: the steps a sibling forked from the layer starts with, ahead
    of its own analysis."""
    head = []
    for ref in steps:
        if ref.device not in _ROUTING:
            break
        head.append(ref)
    return tuple(head)


def drop_states(steps, layer_input: str, *, master: bool = False,
                roi_holder: bool = False, level_view: bool = False) -> dict:
    """``{device id: StepState}`` for every registered device as the next drop on a layer whose
    enabled Transform steps are ``steps`` (``DeviceRef``; filters never change which contracts are
    available, so they are left out), over an input of kind ``layer_input``.

    A primary analyzer, or a device that reads the source file, dropped where a primary already
    heads the chain or on a dataset's master row forks a sibling over the same input that starts
    with the layer's routing steps (:func:`routing_head`), so it is judged after those steps
    alone. On an ROI-holder layer it takes the heading primary's place, so it is judged on the
    steps ahead of that primary. Every other device is judged as appended to ``steps``.
    ``level_view``: the layer is shown at a level, so ``scale_select`` does not apply
    (:func:`~dynamix.model.contracts.next_step_state`)."""
    steps = tuple(steps)
    head = next((i for i, ref in enumerate(steps) if is_primary(get_device(ref.device))), None)
    if roi_holder:
        base = steps if head is None else steps[:head]
    elif master or head is not None:
        base = routing_head(steps)
    else:
        base = steps
    states = {}
    for name, device in DEVICES.items():
        heads = is_primary(device) or capabilities.has(device, "SourceReader")
        states[name] = next_step_state(base if heads else steps, layer_input, name,
                                       level_view=level_view)
    return states
