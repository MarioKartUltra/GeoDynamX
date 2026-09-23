# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Param -> ControlSpec: what knob a declaration earns. Pure logic; widgets live in
knob_widgets.py so this module imports without Qt.

INT params have a minimum nudge step of 1; the fine modifier collapses to normal for integers.
"""
from __future__ import annotations

import dataclasses

from dynamix.model.param import Param, ParamKind

_NUDGE_DIVISIONS = 100
_MOD_SCALE = {"normal": 1.0, "coarse": 10.0, "fine": 0.1}


@dataclasses.dataclass(frozen=True)
class ControlSpec:
    param: Param
    widget: str
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    decimals: int = 0
    wrap: float | None = None


def control_spec(param: Param) -> ControlSpec:
    k = param.kind
    if k is ParamKind.BOOL:
        return ControlSpec(param, "toggle")
    if k is ParamKind.TEXT:
        # Read-only by default (a reading, per ParamKind.TEXT's own docstring) -- ``editable``
        # (Param's own opt-in) earns a real line-edit instead, for the one TEXT
        # param the user must actually type into (group_filter.group).
        return ControlSpec(param, "text_edit" if param.editable else "label")
    if k is ParamKind.CHOICE:
        return ControlSpec(param, "cycle")
    if k is ParamKind.ANGLE:
        step = param.wrap / _NUDGE_DIVISIONS
        return ControlSpec(param, "drag_value", lo=0.0, hi=param.wrap, step=step,
                           decimals=1, wrap=param.wrap)
    lo = param.soft_min if param.soft_min is not None else param.min
    hi = param.soft_max if param.soft_max is not None else param.max
    if k is ParamKind.INT:
        return ControlSpec(param, "drag_value", lo=lo, hi=hi, step=1, decimals=0)
    return ControlSpec(param, "drag_value", lo=lo, hi=hi,
                       step=(hi - lo) / _NUDGE_DIVISIONS, decimals=2)


def rebind_range(spec: ControlSpec, lo: float, hi: float) -> ControlSpec:
    """A COPY of ``spec`` with its SOFT bounds (and the step size they imply) replaced by
    ``lo``/``hi`` -- the same ``(hi - lo) / _NUDGE_DIVISIONS`` formula ``control_spec`` itself
    uses, so a rebound control steps at 1/100th of whatever range it was just given, exactly as a
    freshly-built one would (a data-derived hint re-scales a
    control's drag/nudge sensitivity to the range the data actually spans, e.g. ``[-1.28, 0.69]``
    for the kam_64 fixture's OLS Hölder chains, rather than the device's static declared soft
    range).

    Deliberately does NOT touch ``spec.param`` -- hard bounds (``Param.min``/``max``, what
    ``nudge`` actually clamps to) and ``cache_key`` both read the ORIGINAL ``Param``, never this
    ``ControlSpec``, so nothing here can affect either (``Param``'s own docstring: "a user-retuned
    slider range is view state ... must not affect cache_key"). INT params keep ``step=1``
    (widening/narrowing the soft range does not change the minimum meaningful nudge for an
    integer); a degenerate ``hi <= lo`` keeps the spec's own existing step rather than dividing by
    zero or going negative.
    """
    if spec.param.kind is ParamKind.INT:
        step = spec.step
    elif hi > lo:
        step = (hi - lo) / _NUDGE_DIVISIONS
    else:
        step = spec.step
    return dataclasses.replace(spec, lo=lo, hi=hi, step=step)


def nudge(value, spec_or_param: ControlSpec | Param, direction: int, modifier: str = "normal"):
    """Increment value by a step adjusted by modifier.

    For INT params, the minimum step is 1; the fine modifier collapses to normal.
    ANGLE params wrap rather than clamp. All others respect hard bounds.

    ``spec_or_param`` accepts EITHER a bare ``Param`` (reconstructs a fresh ``ControlSpec`` via
    ``control_spec`` -- the param's STATIC declared soft range, exactly the old behavior every
    existing caller/test relies on) OR a live ``ControlSpec`` (uses its step AS GIVEN, picking up
    a data-rebound range from :func:`rebind_range`).

    **Data-scaled steps (were dead state for real gestures).** Before this fix,
    ``DragValue.keyPressEvent``/``mouseMoveEvent`` passed ``self._spec.param`` here, which threw
    away any rebind and reconstructed the STATIC spec from scratch every call -- a control whose
    soft bounds had just been rebound by ``data_hints`` still
    nudged/dragged at its old, static step size. Those two call sites now pass ``self._spec``
    (the live one) instead -- see ``knob_widgets.py``.
    """
    spec = spec_or_param if isinstance(spec_or_param, ControlSpec) else control_spec(spec_or_param)
    param = spec.param
    step = spec.step * _MOD_SCALE[modifier] * direction
    if param.kind is ParamKind.INT:
        step = direction * max(1, round(abs(step)))
    new = value + step
    if param.kind is ParamKind.ANGLE:
        return new % param.wrap
    if param.kind is ParamKind.INT:
        new = round(new)
    if param.min is not None:
        new = max(param.min, new)
    if param.max is not None:
        new = min(param.max, new)
    return new


def format_reading(value, param: Param) -> str:
    spec = control_spec(param)
    if param.kind is ParamKind.INT:
        text = f"{int(value)}"
    elif param.kind in (ParamKind.FLOAT, ParamKind.ANGLE):
        text = f"{value:.{spec.decimals}f}"
        # A reading must never round a nonzero value to a string that reads as zero or
        # misstates its magnitude -- wtmm2d's thresh default of 1e-3 displayed as "0.00",
        # which is a false reading on an instrument whose readings are the product. Values
        # below the fixed format's resolution fall back to significant figures instead.
        if value != 0 and abs(value) < 10 ** -spec.decimals:
            text = f"{value:.4g}"
    else:
        text = str(value)
    return f"{text} {param.units}".rstrip()
