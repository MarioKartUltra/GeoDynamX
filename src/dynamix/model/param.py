# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Parameter declarations. A device declares these; the GUI generates its knobs from them.

This is the highest-leverage abstraction in the design: a new device is a pure-Python file and
zero GUI code. If a device ever needs a bespoke widget, that is a signal the schema is wrong.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any


class ParamKind(str, enum.Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    CHOICE = "choice"
    ANGLE = "angle"        # periodic; wraps rather than clamps
    TEXT = "text"          # opaque string; no range, no editor -- a reading, not a knob


@dataclasses.dataclass(frozen=True)
class Param:
    """One knob. ``wrap`` is the period for ANGLE params (180 for orientation mod pi, 360 for a
    full azimuth); it is required for ANGLE and meaningless otherwise.

    ``min``/``max`` are hard bounds: validation limits enforced by ``validate()``. They belong to
    the ordered kinds only -- an ANGLE wraps rather than clamps, and CHOICE and BOOL have no range,
    so declaring bounds on those is refused rather than ignored. ``soft_min``/
    ``soft_max`` are only the default span of the generated slider -- the sub-range of the legal
    one a device suggests as visually useful. They are never enforced; ``validate()`` ignores
    them entirely. A user-retuned slider range is view state, not a param value, and must not
    affect ``cache_key``."""

    name: str
    kind: ParamKind
    default: Any
    min: float | None = None
    max: float | None = None
    units: str = ""
    choices: tuple[str, ...] = ()
    label: str = ""
    wrap: float | None = None
    soft_min: float | None = None
    soft_max: float | None = None
    #: TEXT-only opt-in (default False, no other behavior change): a TEXT param normally renders
    #: as a read-only reading (``knobs.control_spec``'s "label" widget) -- correct for something
    #: like ``group_paint.spec_json``, which the commit transaction writes and nothing else should
    #: ever type into. A param the USER must set by hand (``group_filter.group``)
    #: sets this True to earn an editable line-edit control ("text_edit") instead. Ignored for
    #: every other kind -- there is nothing here for a non-TEXT param to opt into.
    editable: bool = False
    #: A DISPLAY selector (which computed output is shown -- ``show``, ``component``): never part
    #: of a cache key and never read by ``compute``; the device's ``view(result, params)`` applies
    #: it after the cache, so switching it is a cache hit, not a recompute
    #: (``dynamix.model.device.keyed_params``).
    view: bool = False
    #: ``(other_param, allowed_values)``: this knob only APPLIES while ``other_param`` holds one of
    #: ``allowed_values`` -- the knob panel greys it out otherwise (it is still validated, stored
    #: and keyed as before). ``None`` = always applies. Several conditions that must ALL hold
    #: are a tuple of such pairs: ``(("wavelet", ("q_gaussian",)), ("q_pairing", ("fixed",)))``.
    active_when: tuple | None = None
    #: Where the shell draws the knob: ``""`` on the device strip, or the name of a right-panel
    #: section (``"reconstruction"``). Presentation only: validation, defaults and cache keys
    #: never read it.
    section: str = ""

    def __post_init__(self):
        if self.kind is ParamKind.CHOICE:
            if not self.choices:
                raise ValueError(f"{self.name}: CHOICE param needs choices")
            if self.default not in self.choices:
                raise ValueError(f"{self.name}: default {self.default!r} not one of {self.choices}")
        if self.kind is ParamKind.ANGLE and not self.wrap:
            raise ValueError(f"{self.name}: ANGLE param needs a wrap period (e.g. 180.0)")
        if self.kind in (ParamKind.ANGLE, ParamKind.CHOICE, ParamKind.BOOL, ParamKind.TEXT):
            declared = [n for n in ("min", "max", "soft_min", "soft_max")
                        if getattr(self, n) is not None]
            if declared:
                raise ValueError(
                    f"{self.name}: a {self.kind.value} param has no range, so {declared} would be "
                    f"silently ignored -- an ANGLE wraps rather than clamps, and CHOICE and BOOL "
                    f"are not ordered. Narrow the wrap period or the choices instead."
                )
        if (self.soft_min is not None and self.soft_max is not None
                and self.soft_min > self.soft_max):
            raise ValueError(
                f"{self.name}: soft_min {self.soft_min} must not exceed soft_max {self.soft_max}"
            )
        if self.soft_min is not None and self.min is not None and self.soft_min < self.min:
            raise ValueError(f"{self.name}: soft_min {self.soft_min} is below min {self.min}")
        if self.soft_max is not None and self.max is not None and self.soft_max > self.max:
            raise ValueError(f"{self.name}: soft_max {self.soft_max} is above max {self.max}")
        if self.soft_min is not None and self.max is not None and self.soft_min > self.max:
            raise ValueError(f"{self.name}: soft_min {self.soft_min} is above max {self.max}")
        if self.soft_max is not None and self.min is not None and self.soft_max < self.min:
            raise ValueError(f"{self.name}: soft_max {self.soft_max} is below min {self.min}")
        self._check_default()

    def _check_default(self) -> None:
        """A declaration must describe a device completely, and that includes its own default.

        Without this check ``Param("n", INT, default=99, min=1, max=8)`` would construct happily,
        and 99 would flow straight into ``cache_key`` and ``compute`` because ``validate_params``
        fills an absent param from ``default`` without validating it.

        The default is put through ``validate`` rather than range-checked raw, so it is coerced the
        same way a user-supplied value would be -- an ANGLE wraps, an INT-valued float narrows.
        Otherwise an ANGLE default of 190 and a user typing 190 would produce different cache keys
        for the same dial position.
        """
        try:
            checked = self.validate(self.default)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{self.name}: default {self.default!r} is not a legal value "
                             f"for this param ({exc})") from exc
        object.__setattr__(self, "default", checked)     # frozen dataclass: this is the idiom

    def validate(self, value: Any) -> Any:
        """Coerce and range-check, or raise ValueError naming the parameter."""
        if self.kind is ParamKind.BOOL:
            return bool(value)
        if self.kind is ParamKind.TEXT:
            return str(value)
        if self.kind is ParamKind.CHOICE:
            if value not in self.choices:
                raise ValueError(f"{self.name}: {value!r} is not one of {self.choices}")
            return value
        if self.kind is ParamKind.INT:
            if int(value) != value:
                raise ValueError(f"{self.name}: {value!r} is not an integer")
            value = int(value)
        elif self.kind is ParamKind.ANGLE:
            # Periodic: wrap, never clamp. Returning here skips the min/max check below, which is
            # only safe because __post_init__ refuses to let an ANGLE declare bounds at all --
            # otherwise they would be accepted and then silently ignored.
            return float(value) % float(self.wrap)
        else:
            value = float(value)
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.name}: {value} is below min {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.name}: {value} is above max {self.max}")
        return value

    def to_payload(self) -> dict:
        d = dataclasses.asdict(self)
        d["kind"] = self.kind.value
        d["choices"] = list(self.choices)
        return d

    @classmethod
    def from_payload(cls, d: dict) -> "Param":
        d = dict(d)
        d["kind"] = ParamKind(d["kind"])
        d["choices"] = tuple(d.get("choices", ()))
        return cls(**d)
