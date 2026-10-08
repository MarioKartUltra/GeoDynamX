# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Content-hash keyed store for expensive transform results.

The key is derived from the source identity plus the transform's own params, and from nothing else.
That is what lets a filter change -- a scale slider, an orientation wedge -- reuse a computed WTMM
stack instead of triggering a recompute that takes seconds to minutes.

Anything that is *view* state (a slider's span, a colour ramp) must never reach a key, or adjusting
the view would silently invalidate the science.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import numpy as np


def cache_key(device_name: str, source_id: str, params: dict, *,
              upstream: str | None = None) -> str:
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
    """
    payload = json.dumps(
        {"device": device_name, "source": source_id, "params": params, "upstream": upstream},
        sort_keys=True, separators=(",", ":"), default=_jsonable,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _jsonable(o: Any):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def stage_nbytes(value) -> int:
    """Bytes of every ndarray inside ``value``: a dict's values, a list's or tuple's items, a
    field's ``values`` and axes, an array itself. Anything else weighs nothing."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(stage_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(stage_nbytes(v) for v in value)
    values = getattr(value, "values", None)
    if isinstance(values, np.ndarray):
        return int(values.nbytes) + stage_nbytes(getattr(value, "x_axis", None)) \
            + stage_nbytes(getattr(value, "y_axis", None))
    return 0


class StageStore:
    """A bounded, byte-counted mapping for a transform's intermediate stages, so a change of a
    downstream knob recomputes the cheap half only. Insertion order is recency; a get refreshes
    it. Nothing here is pinned, and ``Cache.clear`` empties it with the results."""

    def __init__(self, maxsize: int = 8, max_bytes: int | None = None):
        self.maxsize = int(maxsize)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self._store: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def __contains__(self, key) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, key):
        if key not in self._store:
            self.misses += 1
            raise KeyError(key)
        self.hits += 1
        self._store[key] = self._store.pop(key)
        return self._store[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value) -> None:
        self._store.pop(key, None)
        self._store[key] = value
        while len(self._store) > self.maxsize:
            del self._store[next(iter(self._store))]
        if self.max_bytes is not None:
            sizes = [(k, stage_nbytes(v)) for k, v in self._store.items()]
            total = sum(n for _k, n in sizes)
            for k, n in sizes:
                if total <= self.max_bytes or k == key:
                    continue
                del self._store[k]
                total -= n

    def put(self, key, value) -> None:
        self[key] = value

    def clear(self) -> None:
        self._store.clear()


class Cache:
    """In-memory store, bounded LRU. A scrub such as the ``noise`` transform's seed/amplitude
    mints a full-raster field clone plus a complete WTMM result per step; an unbounded store
    retains every one of them, which on a 4096-px window is enough memory pressure to stall the
    app. ``maxsize`` bounds the UNPINNED entry count; a hit refreshes recency, so the entries a user
    is actively scrubbing between are the last to go, and the capacity is generous enough that a
    multi-layer arrangement's working set never thrashes into recomputes.

    Pins are a freeze contract, honored exactly as promised: a pinned key is never evicted,
    however old or large, and rides ABOVE the bound rather than shrinking it for the rest.

    ``max_bytes`` (``None``: no byte bound) bounds the bytes of the unpinned raster OUTPUTS --
    entries whose value is a dict holding a ``"raster"`` ndarray, what ``resolve_output`` caches
    -- because one 4096² float32 level is about 67 MB and a scrub through the levels mints one
    per level. Past it the least-recent of them go first. An analysis result, a field (a
    multiband stack included) and a pinned output are never counted or evicted by it, and the
    newest entry always stays, so an output larger than the whole budget is still served."""

    def __init__(self, maxsize: int = 64, max_bytes: int | None = None):
        self.maxsize = int(maxsize)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self._store: dict[str, Any] = {}          # insertion-ordered; order IS recency
        self._pinned: set[str] = set()
        self.hits = 0
        self.misses = 0
        #: Intermediate stages of transforms that declare ``uses_stage_cache``.
        self.stages = StageStore(8, self.max_bytes)

    def _touch(self, key: str) -> None:
        self._store[key] = self._store.pop(key)   # move to the recent end

    def _evict_over_capacity(self) -> None:
        while len(self._store) - len(self._pinned & self._store.keys()) > self.maxsize:
            victim = next((k for k in self._store if k not in self._pinned), None)
            if victim is None:
                return
            del self._store[victim]

    def _evict_over_bytes(self, newest: str) -> None:
        """Drop the least-recent unpinned raster outputs while their bytes exceed ``max_bytes``,
        never ``newest`` (the entry just stored)."""
        if self.max_bytes is None:
            return
        sizes = [(k, _raster_nbytes(v)) for k, v in self._store.items()
                 if k not in self._pinned]
        total = sum(n for _k, n in sizes)
        for k, n in sizes:                           # insertion order: least recent first
            if total <= self.max_bytes:
                return
            if n and k != newest:
                del self._store[k]
                total -= n

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        if key in self._store:
            self.hits += 1
            self._touch(key)
            return self._store[key]
        self.misses += 1
        value = compute()
        self._store[key] = value
        self._evict_over_capacity()
        self._evict_over_bytes(key)
        return value

    def get(self, key: str, default=None):
        if key in self._store:
            self._touch(key)
        return self._store.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self._store.pop(key, None)
        self._store[key] = value
        self._evict_over_capacity()
        self._evict_over_bytes(key)

    def invalidate(self, key: str) -> bool:
        """Drop one entry. Returns whether it was present.

        Presence is tested with ``in`` rather than by inspecting the popped value, so an entry
        whose value is legitimately ``None`` still reports as present and still gets dropped.
        """
        present = key in self._store
        self._store.pop(key, None)
        return present

    def clear(self) -> None:
        self._store.clear()
        self.stages.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def pin(self, key: str) -> None:
        """Pin a key to prevent eviction. The key need not exist in the store yet."""
        self._pinned.add(key)

    def unpin(self, key: str) -> None:
        """Unpin a key, allowing it to be evicted. No-op if the key is not pinned."""
        self._pinned.discard(key)

    def is_pinned(self, key: str) -> bool:
        """Return whether a key is pinned."""
        return key in self._pinned

    @property
    def pinned(self) -> frozenset[str]:
        """Return a frozenset of all pinned keys."""
        return frozenset(self._pinned)


def _raster_nbytes(value) -> int:
    """The bytes of ``value``'s ``"raster"`` when it is a raster output (a dict holding a
    ``"raster"`` ndarray), else 0."""
    if isinstance(value, dict):
        raster = value.get("raster")
        if isinstance(raster, np.ndarray):
            return int(raster.nbytes)
    return 0
