# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A chain: an ordered list of device references applied to one layer's source.

Stores REFERENCES (registry name + params), never device instances, so a session serialises to
JSON without pickling behaviour and a chain loaded on another machine binds to that machine's
registry.
"""
from __future__ import annotations

import dataclasses

from dynamix.model.device import get_device, is_transform, validate_params


@dataclasses.dataclass(frozen=True)
class DeviceRef:
    device: str
    params: dict = dataclasses.field(default_factory=dict)


def _normalized(ref: DeviceRef) -> DeviceRef:
    """The single definition of a normalised step: the device resolves, and its params are filled
    with defaults, coerced and validated.

    ``validate``, ``materialized`` and ``from_payload`` all route through this. They used to hold
    three separate opinions about what "normalised" meant -- ``validate`` discarding the filled
    dict that ``materialized`` kept -- which is how an in-memory chain and its round-tripped twin
    came to differ.
    """
    return DeviceRef(ref.device, validate_params(get_device(ref.device), ref.params))


@dataclasses.dataclass(frozen=True)
class Chain:
    steps: tuple[DeviceRef, ...] = ()

    @property
    def transforms(self) -> tuple[DeviceRef, ...]:
        """The leading transforms. Raises on an interleaved chain -- see :meth:`_split`."""
        return self._split()[0]

    @property
    def filters(self) -> tuple[DeviceRef, ...]:
        """The trailing filters. Raises on an interleaved chain -- see :meth:`_split`."""
        return self._split()[1]

    def _split(self) -> tuple[tuple[DeviceRef, ...], tuple[DeviceRef, ...]]:
        """(transforms, filters), rejecting an interleaved chain rather than sorting it.

        Filtering the steps by kind quietly turned an invalid ``['f', 't']`` into transforms
        ``['t']`` and filters ``['f']``. Phase 2 is the caller: it would have executed a different
        recipe than the one written, with nothing raised anywhere.
        """
        transforms: list[DeviceRef] = []
        filters: list[DeviceRef] = []
        for ref in self.steps:
            if is_transform(get_device(ref.device)):
                if filters:
                    raise ValueError(
                        f"chain must be transforms-then-filters: transform {ref.device!r} "
                        f"appears after a filter"
                    )
                transforms.append(ref)
            else:
                filters.append(ref)
        return tuple(transforms), tuple(filters)

    def validate(self) -> "Chain":
        """Raise unless every device resolves, every param is legal, and no transform follows a
        filter. Returns self so it can be chained."""
        for ref in self.steps:
            _normalized(ref)          # KeyError if unknown, ValueError if a param is illegal
        self._split()                 # ValueError if interleaved
        return self

    def materialized(self) -> "Chain":
        """Validate, and fill every step's params with its device defaults, in memory.

        A chain built by hand is then as fully specified as one loaded from a payload. Without it,
        an in-memory chain and its round-tripped twin differ.
        """
        return Chain(tuple(_normalized(r) for r in self.steps)).validate()

    def to_payload(self) -> dict:
        return {"steps": [{"device": r.device, "params": dict(r.params)} for r in self.steps]}

    @classmethod
    def from_payload(cls, payload: dict) -> "Chain":
        """Rebuild, materialising defaults so a loaded chain is fully specified."""
        steps = tuple(
            _normalized(DeviceRef(s["device"], s.get("params", {})))
            for s in payload.get("steps", [])
        )
        return cls(steps).validate()
