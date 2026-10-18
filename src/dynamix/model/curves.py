# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A curve a tool publishes for the Curve window: one y against one x, with its axis labels,
the axes' default scaling and the vertical marks worth reading off it.

A device offers its curves through ``curves(result, field, params) -> list[Curve]``, where
``result`` is the active result (a dict) or the produced field of a chain that ends on a field
stage, and ``field`` the layer's own field. The window shows one at a time, named.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Curve", "curves_of"]


@dataclasses.dataclass(frozen=True)
class Curve:
    name: str
    x: np.ndarray
    y: np.ndarray
    xlabel: str = ""
    ylabel: str = ""
    logx: bool = False
    logy: bool = False
    #: ``(x, label)`` pairs drawn as vertical lines.
    marks: tuple = ()
    note: str = ""


def curves_of(device, result, field, params: dict) -> list:
    """The device's curves for ``result`` over ``field``, or none when it offers no curves or
    has nothing to show yet."""
    hook = getattr(device, "curves", None)
    if hook is None:
        return []
    try:
        return list(hook(result, field, params) or [])
    except (KeyError, ValueError, TypeError):
        return []
