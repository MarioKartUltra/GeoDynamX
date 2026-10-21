# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Content-hash keyed store for expensive transform results.

The key is derived from the source identity plus the transform's own params, and from nothing else.
That is what lets a filter change -- a scale slider, an orientation wedge -- reuse a computed WTMM
stack instead of triggering a recompute that takes seconds to minutes.

Anything that is *view* state (a slider's span, a colour ramp) must never reach a key, or adjusting
the view would silently invalidate the science. The key itself is ``dynamix.core.keys.cache_key``,
re-exported here at its historical path.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from dynamix.core.deferred import Deferred
from dynamix.core.keys import cache_key  # noqa: F401


def stage_nbytes(value) -> int:
    """Bytes of every ndarray inside ``value``: a dict's values, a list's or tuple's items, a
    field's ``values`` and axes, an array itself, a deferred product once it is built (counting
    never builds it). Anything else weighs nothing."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Deferred):
        return sum(stage_nbytes(v) for v in value.values()) if value.built else 0
    if isinstance(value, dict):
        return sum(stage_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(stage_nbytes(v) for v in value)
    values = getattr(value, "values", None)
    if isinstance(values, np.ndarray):
        return int(values.nbytes) + stage_nbytes(getattr(value, "x_axis", None)) \
            + stage_nbytes(getattr(value, "y_axis", None))
    return 0


#: Bytes one Python object in a cached value holds beyond its array data: an ndarray's header with
#: its shape and strides, a small dict or list, a scalar. tracemalloc over the wtmm2d results of
#: 256² and 512² fBm fields measured 123.9 B per object at both sizes, counting the objects as
#: :func:`retained_nbytes` reaches them.
OBJECT_BYTES = 124


def retained_nbytes(value) -> int:
    """An estimate of the bytes ``value`` keeps alive: :data:`OBJECT_BYTES` for each Python object
    reached (``value``, a dict's values, a list's or tuple's items, recursively), each ndarray's
    data, a field's values and axes (:func:`stage_nbytes`), and the ``nbytes`` any other object
    states for itself (a topology graph's nodes and edges). A set's members and an object's
    attributes are not reached."""
    if isinstance(value, np.ndarray):
        return OBJECT_BYTES + int(value.nbytes)
    if isinstance(value, dict):
        return OBJECT_BYTES + sum(retained_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return OBJECT_BYTES + sum(retained_nbytes(v) for v in value)
    if isinstance(getattr(value, "values", None), np.ndarray):
        return OBJECT_BYTES + stage_nbytes(value)
    own = getattr(value, "nbytes", None)
    return OBJECT_BYTES + (own if isinstance(own, int) else 0)


class StageStore:
    """A bounded, byte-counted mapping for a transform's intermediate stages, so a change of a
    downstream knob recomputes the cheap half only. Insertion order is recency; a get refreshes
    it. Nothing here is pinned, and ``Cache.clear`` empties it with the results."""

    def __init__(self, maxsize: int = 8, max_bytes: int | None = None):
        self.maxsize = int(maxsize)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self._store: dict[str, Any] = {}
        self._fft_contexts: dict[str, list] = {}
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
        from dynamix.core.fft_policy import include_observed
        include_observed(self._fft_contexts.get(key, ()))
        return self._store[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value) -> None:
        from dynamix.core.fft_policy import observed
        self._fft_contexts[key] = observed()
        self._store.pop(key, None)
        self._store[key] = value
        while len(self._store) > self.maxsize:
            victim = next(iter(self._store))
            del self._store[victim]
            self._fft_contexts.pop(victim, None)
        if self.max_bytes is not None:
            sizes = [(k, stage_nbytes(v)) for k, v in self._store.items()]
            total = sum(n for _k, n in sizes)
            for k, n in sizes:
                if total <= self.max_bytes or k == key:
                    continue
                del self._store[k]
                self._fft_contexts.pop(k, None)
                total -= n

    def put(self, key, value) -> None:
        self[key] = value

    def set_context(self, key, contexts) -> None:
        """Narrow a stage's context to its own compute scope."""
        if key in self._store:
            self._fft_contexts[key] = list(contexts)

    def clear(self) -> None:
        self._store.clear()
        self._fft_contexts.clear()


class Cache:
    """In-memory store, bounded LRU. A scrub such as the ``noise`` transform's seed/amplitude
    mints a full-raster field clone plus a complete WTMM result per step; an unbounded store
    retains every one of them, which on a 4096-px window is enough memory pressure to stall the
    app. ``maxsize`` bounds the UNPINNED entry count; a hit refreshes recency, so the entries a user
    is actively scrubbing between are the last to go, and the capacity is generous enough that a
    multi-layer arrangement's working set never thrashes into recomputes.

    Pins are a freeze contract, honored exactly as promised: a pinned key is never evicted,
    however old or large, and rides ABOVE the bound rather than shrinking it for the rest.

    ``max_results`` (``None``: no such bound) bounds the unpinned analysis results (every dict
    entry but a raster output), because each transform knob value mints a whole result: at 2048²
    a WTMM result is 600 MB of arrays and its chain topology millions of Python objects.

    ``max_bytes`` (``None``: no byte bound) bounds the bytes of every unpinned entry, each
    weighed once at insertion by :func:`retained_nbytes` (its arrays and its Python objects); a
    value two entries share, as a ``chain_topology`` result shares its ``wtmm2d`` result's, counts
    once. Past either bound the least-recent entries go first, but never the newest (so an entry
    larger than the whole budget is still served), never the analysis a layer shows
    (:meth:`mark_shown`), and never an entry another stored entry names as its upstream: a stored
    transform tail always finds its lineage whole, so a resolve whose tail is cached, which the
    shell runs on the GUI thread, computes nothing."""

    def __init__(self, maxsize: int = 64, max_bytes: int | None = None,
                 max_results: int | None = None):
        self.maxsize = int(maxsize)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.max_results = None if max_results is None else int(max_results)
        self._store: dict[str, Any] = {}          # insertion-ordered; order IS recency
        self._pinned: set[str] = set()
        self.hits = 0
        self.misses = 0
        #: Intermediate stages of transforms that declare ``uses_stage_cache``.
        self.stages = StageStore(8, self.max_bytes)
        self._landed_views: dict = {}
        #: Per entry, the weight of each value it holds, by the value's id.
        self._weights: dict[str, dict[int, int]] = {}
        #: Per entry, the key of the entry its compute consumed, where the caller named it.
        self._upstream: dict[str, str] = {}
        #: Per layer id, the key of the analysis that layer shows.
        self._shown: dict = {}
        #: The key stored last, kept until another is stored.
        self._newest: str | None = None

    def _touch(self, key: str) -> None:
        self._store[key] = self._store.pop(key)   # move to the recent end

    def _nbytes(self, keys) -> int:
        """The weight of the entries under ``keys``, a value they share counted once."""
        parts: dict[int, int] = {}
        for k in keys:
            parts.update(self._weights.get(k, {}))
        return sum(parts.values())

    def _evictable(self, keys) -> list[str]:
        """``keys`` in recency order without what eviction keeps: a pinned or shown entry, the
        newest and every entry a stored entry names as its upstream."""
        keep = {self._newest, *self._shown.values(),
                *(self._upstream.get(k) for k in list(self._store))}
        return [k for k in keys if k not in keep and k not in self._pinned]

    def _drop(self, key: str) -> None:
        self._store.pop(key, None)
        self._weights.pop(key, None)
        self._upstream.pop(key, None)
        self._landed_views.pop(key, None)

    def _evict_over_capacity(self) -> None:
        """Drop the least-recent evictable entries while more than ``maxsize`` entries, or more
        than ``max_results`` analysis results, are unpinned."""
        self._landed_views = {k: v for k, v in self._landed_views.items() if k in self._store}
        while True:
            unpinned = [k for k in list(self._store) if k not in self._pinned]
            results = [k for k in unpinned if _is_result(self._store.get(k))]
            if len(unpinned) > self.maxsize:
                pool = unpinned
            elif self.max_results is not None and len(results) > self.max_results:
                pool = results
            else:
                return
            victims = self._evictable(pool)
            if not victims:
                return
            self._drop(victims[0])

    def _evict_over_bytes(self) -> None:
        """Drop the least-recent evictable entries while the unpinned ones weigh more than
        ``max_bytes``."""
        if self.max_bytes is None:
            return
        while True:
            unpinned = [k for k in list(self._store) if k not in self._pinned]
            if self._nbytes(unpinned) <= self.max_bytes:
                return
            victims = self._evictable(unpinned)
            if not victims:
                return
            self._drop(victims[0])

    def _insert(self, key: str, value: Any, upstream: str | None) -> None:
        """Store ``value``, weighed by its values; one another stored entry holds is weighed
        already, which spares a ``chain_topology`` result a second walk of its ``wtmm2d``
        result's chains."""
        self._store.pop(key, None)
        self._weights.pop(key, None)
        self._store[key] = value
        self._newest = key
        known = {i: n for weights in list(self._weights.values()) for i, n in weights.items()}
        parts = value.values() if isinstance(value, dict) else (value,)
        self._weights[key] = {id(v): known[id(v)] if id(v) in known else retained_nbytes(v)
                              for v in parts}
        if upstream is None:
            self._upstream.pop(key, None)
        else:
            self._upstream[key] = upstream
        self._evict_over_capacity()
        self._evict_over_bytes()

    def get_or_compute(self, key: str, compute: Callable[[], Any], *,
                       upstream: str | None = None) -> Any:
        """The entry under ``key``, computed and stored on a miss. ``upstream`` is the key of the
        entry ``compute`` consumes, kept while this one is stored."""
        if key in self._store:
            self.hits += 1
            self._touch(key)
            return self._store[key]
        self.misses += 1
        value = compute()
        self._insert(key, value, upstream)
        return value

    def get(self, key: str, default=None):
        if key in self._store:
            self._touch(key)
        return self._store.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self._landed_views.pop(key, None)
        self._insert(key, value, None)

    def landed_result(self, key, value, identity):
        """Keep a stable view identity while its underlying cached result is unchanged."""
        from dynamix.engine.execution import attach
        prior = self._landed_views.get(key)
        if prior is not None and prior[0] is value and prior[1] == identity:
            return prior[2]
        result = attach(value, _landed=identity)
        if result is not value and key in self._store:
            self._landed_views[key] = (value, identity, result)
        self._landed_views = {k: v for k, v in self._landed_views.items() if k in self._store}
        return result

    def mark_shown(self, layer_id, key: str | None) -> None:
        """Record ``key`` as the analysis ``layer_id`` shows (``None``: none, the layer is gone).
        A shown entry stays through every bound, and so does each entry upstream of it, so the
        layer's next filter change or scale tick is a cache hit; the analysis it showed before
        becomes evictable."""
        if key is None:
            self._shown.pop(layer_id, None)
        else:
            self._shown[layer_id] = key
        self._evict_over_capacity()
        self._evict_over_bytes()

    def report(self) -> dict:
        """What the three memory budgets hold, apart. ``results``: this store's entries, with the
        bytes and the analysis results of the unpinned ones, which ``max_bytes`` and
        ``max_results`` bound, and the pinned ones above them; ``stages``: the stage store's
        entries and array bytes; ``mlx``: mlx's allocator
        (:func:`~dynamix.core.fft_policy.mlx_memory`), ``None`` while mlx is not loaded."""
        from dynamix.core.fft_policy import mlx_memory

        keys = list(self._store)
        pinned = [k for k in keys if k in self._pinned]
        unpinned = [k for k in keys if k not in self._pinned]
        stages = list(self.stages._store.values())
        return {
            "results": {"entries": len(keys), "bytes": self._nbytes(unpinned),
                        "max_bytes": self.max_bytes,
                        "analysis": sum(_is_result(self._store.get(k)) for k in unpinned),
                        "max_results": self.max_results,
                        "pinned": len(pinned), "pinned_bytes": self._nbytes(pinned)},
            "stages": {"entries": len(stages), "bytes": sum(stage_nbytes(v) for v in stages),
                       "max_bytes": self.stages.max_bytes},
            "mlx": mlx_memory(),
        }

    def invalidate(self, key: str) -> bool:
        """Drop one entry. Returns whether it was present.

        Presence is tested with ``in`` rather than by inspecting the popped value, so an entry
        whose value is legitimately ``None`` still reports as present and still gets dropped.
        """
        present = key in self._store
        self._drop(key)
        return present

    def clear(self) -> None:
        self._store.clear()
        self._landed_views.clear()
        self._weights.clear()
        self._upstream.clear()
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


def _is_result(value) -> bool:
    """Whether ``value`` is an analysis result: a dict other than a raster output."""
    return isinstance(value, dict) and not _raster_nbytes(value)


def _raster_nbytes(value) -> int:
    """The bytes of ``value``'s ``"raster"`` when it is a raster output (a dict holding a
    ``"raster"`` ndarray), else 0."""
    if isinstance(value, dict):
        raster = value.get("raster")
        if isinstance(raster, np.ndarray):
            return int(raster.nbytes)
    return 0
