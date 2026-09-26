# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""bus -- the rack-head stage that routes raw bands, by reference, into one stack.

Sits FIRST in a layer's chain: whatever field arrives (the host dataset, or an ROI's
processing window of it) only fixes the GRID; the output is the send list's planes read on
that grid (:func:`dynamix.core.bus.materialize`). A multi-input tool dropped after it (pca,
tucker's band mode) consumes the bus's stack exactly as it would an imported one. The send
list is written by the shell's routing dialog into the hidden ``_sends`` parameter, so the
cache key is the routing itself: re-route and the bus recomputes; route back and it is a hit.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind


class Bus:
    """Raw bands routed into one ``(ny, nx, n)`` stack on the host's grid."""

    name = "bus"
    #: A FIELD STAGE (field -> field) at the head of the rack; per-pixel reads, no margin.
    field_stage = True

    def roi_margin(self, params: dict) -> int:
        return 0

    params = (
        Param("_sends", ParamKind.TEXT, default="[]", label="Sends"),
    )

    def compute(self, field, params: dict, *, progress=None):
        from dynamix.core.bus import materialize, parse_sends

        if isinstance(field, dict):
            raise ValueError("bus sits at the HEAD of the rack — drag it before every tool")
        return materialize(field, parse_sends(params.get("_sends")))

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
