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


class Cache:
    """In-memory store, bounded LRU (2026-08-30). The original was deliberately unbounded --
    "a WTMM stack is large but there are few of them" -- an assumption the ``noise`` transform's
    seed/amplitude scrubs broke: each step minted a full-raster field clone plus a complete WTMM
    result, retained forever ("the app was not responding": memory pressure on a 4096-px window).
    ``maxsize`` bounds the UNPINNED entry count; a hit refreshes recency, so the entries a user
    is actively scrubbing between are the last to go, and the capacity is generous enough that a
    multi-layer arrangement's working set never thrashes into recomputes.

    Pins are a freeze contract, honored exactly as promised: a pinned key is never evicted,
    however old or large, and rides ABOVE the bound rather than shrinking it for the rest."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = int(maxsize)
        self._store: dict[str, Any] = {}          # insertion-ordered; order IS recency
        self._pinned: set[str] = set()
        self.hits = 0
        self.misses = 0

    def _touch(self, key: str) -> None:
        self._store[key] = self._store.pop(key)   # move to the recent end

    def _evict_over_capacity(self) -> None:
        while len(self._store) - len(self._pinned & self._store.keys()) > self.maxsize:
            victim = next((k for k in self._store if k not in self._pinned), None)
            if victim is None:
                return
            del self._store[victim]

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        if key in self._store:
            self.hits += 1
            self._touch(key)
            return self._store[key]
        self.misses += 1
        value = compute()
        self._store[key] = value
        self._evict_over_capacity()
        return value

    def get(self, key: str, default=None):
        if key in self._store:
            self._touch(key)
        return self._store.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self._store.pop(key, None)
        self._store[key] = value
        self._evict_over_capacity()

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
