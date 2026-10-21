# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The content-hash key of one transform application.

It lives at the bottom layer so the ROI code, below the engine, keys its stage groups the same way
the engine keys its results; ``dynamix.engine.cache`` re-exports it at its historical path.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def cache_key(device_name: str, source_id: str, params: dict, *,
              upstream: str | None = None, origin: str | None = None) -> str:
    """Stable key for one transform application.

    Params are serialised sorted, so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` -- the same
    recipe -- produce the same key.

    ``upstream`` is the key of the transform immediately before this one in the chain, or ``None``
    for the first. **It is not optional in practice.** A key computed only from
    ``(device, source, params)`` is correct for a chain with a single transform and silently WRONG
    for any chain with two, because the second transform's result depends on the first's output and
    nothing in its own identity records that. Two chains differing only in a preprocessing step
    would then share a cached result — no crash, just numbers computed from the wrong input.

    Threading the upstream key makes each entry depend on its whole lineage: source, every
    preceding step, and its own params.

    ``origin`` is a plugin device's origin (``dynamix.model.plugins.origin_of``); it enters the key
    only for a non-builtin, so every built-in key is what it always was.
    """
    fields = {"device": device_name, "source": source_id, "params": params, "upstream": upstream}
    if origin and origin != "builtin":     # a plugin's file or package: a changed file is a new recipe
        fields["origin"] = origin
    payload = json.dumps(
        fields,
        sort_keys=True, separators=(",", ":"), default=_jsonable,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _jsonable(o: Any):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
