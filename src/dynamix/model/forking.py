# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The fork rule and the shipping law, read from the devices' declared contracts.

A primary analyzer reads the layer's raster (a single-band field or a band stack) and makes
something that is no longer a raster. Dropping a different one on a layer a primary already heads,
or on a dataset's master row, forks a sibling layer over the same input. A step on the parent moves
to the sibling when it applies to the sibling's analysis and not to the parent's.
"""
from __future__ import annotations

from dynamix.model.contracts import FIELD, STACK, consumes_of, next_step_state, produces_of
from dynamix.model.device import defaults_for, is_transform

_RASTERS = frozenset((FIELD, STACK))


def is_primary(device) -> bool:
    """True for a Transform that consumes a single-band field or a band stack in one of its
    declared alternatives and, given that input, produces at least one contract and no raster.
    A device that reads its own data (``reads_source``: the ROI WTMM reading the source file;
    ``vector_source``: a derivative dataset's vector loader) is part of how its layer is read and
    never counts."""
    if (not is_transform(device) or getattr(device, "reads_source", False)
            or getattr(device, "vector_source", False)):
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
