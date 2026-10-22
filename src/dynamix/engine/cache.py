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
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from functools import wraps
import threading
import time


import numpy as np

from dynamix.core.deferred import Deferred
from dynamix.core.keys import cache_key  # noqa: F401


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


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
    if getattr(value, "built", True) is False:          # a deferred product: weighing it must not build it
        return OBJECT_BYTES
    if isinstance(value, Deferred):
        return OBJECT_BYTES + sum(retained_nbytes(v) for v in value.values())
    own = getattr(value, "nbytes", None)
    return OBJECT_BYTES + (own if isinstance(own, int) else 0)


class StageStore:
    """A bounded, byte-counted mapping for a transform's intermediate stages, so a change of a
    downstream knob recomputes the cheap half only. Insertion order is recency; a get refreshes
    it. Nothing here is pinned, and ``Cache.clear`` empties it with the results."""

    def __init__(self, maxsize: int = 8, max_bytes: int | None = None):
        self._lock = threading.RLock()
        self.maxsize = int(maxsize)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self._store: dict[str, Any] = {}
        self._fft_contexts: dict[str, list] = {}
        self.hits = 0
        self.misses = 0

    @_locked
    def __contains__(self, key) -> bool:
        return key in self._store

    @_locked
    def __len__(self) -> int:
        return len(self._store)

    @_locked
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

    @_locked
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

    @_locked
    def set_context(self, key, contexts) -> None:
        """Narrow a stage's context to its own compute scope."""
        if key in self._store:
            self._fft_contexts[key] = list(contexts)

    @_locked
    def clear(self) -> None:
        self._store.clear()
        self._fft_contexts.clear()



class Cache:
    """Bounded RAM store with optional asynchronous disk spill and memory-pressure release. A scrub such as the ``noise`` transform's seed/amplitude
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
                 max_results: int | None = None, *, disk_dir=None,
                 disk_bytes: int | None = None, pressure_floor: int = 0,
                 spill_threshold: int = 1_000_000):
        self._lock = threading.RLock()
        self.maxsize = int(maxsize)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.max_results = None if max_results is None else int(max_results)
        self._store: dict[str, Any] = {}          # insertion-ordered; order IS recency
        self._pinned: set[str] = set()
        self.hits = 0
        self.misses = 0
        #: Intermediate stages of transforms that declare ``uses_stage_cache``.
        self.stages = StageStore(8, 2_000_000_000)
        self._landed_views: dict = {}
        #: Per entry, the weight of each value it holds, by the value's id.
        self._weights: dict[str, dict[int, int]] = {}
        #: Per entry, the key of the entry its compute consumed, where the caller named it.
        self._upstream: dict[str, str] = {}
        #: Per layer id, the key of the analysis that layer shows.
        self._shown: dict = {}
        #: The key stored last, kept until another is stored.
        self._newest: str | None = None
        self._meta = {}
        self._versions = {}
        self._sequence = 0
        self._pending = {}
        self._computing = {}
        self._events = deque(maxlen=1000)
        self._executor = None
        self._closed = False
        self._disk_epoch = 0
        self.pressure_floor = int(pressure_floor)
        self.spill_threshold = int(spill_threshold)
        if disk_dir is None:
            self.disk = None
        else:
            from dynamix.engine.disk_cache import DiskStore
            self.disk = DiskStore(disk_dir, disk_bytes)
            import atexit
            import weakref
            reference = weakref.ref(self)
            def cleanup():
                cache = reference()
                if cache is not None:
                    cache.close()
            atexit.register(cleanup)

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
                *(upstream for _future, upstream in self._computing.values()),
                *(self._upstream.get(k) for k in list(self._store))}
        return [k for k in keys if k not in keep and k not in self._pinned and k not in self._pending]

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
            unpinned = [k for k in list(self._store) if k not in self._pinned and k not in self._pending]
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
            self._retire(self._victim(victims))

    def _evict_over_bytes(self) -> None:
        """Drop the least-recent evictable entries while the unpinned ones weigh more than
        ``max_bytes``."""
        if self.max_bytes is None:
            return
        while True:
            unpinned = [k for k in list(self._store) if k not in self._pinned and k not in self._pending]
            if self._nbytes(unpinned) <= self.max_bytes:
                return
            victims = self._evictable(unpinned)
            if not victims:
                return
            self._retire(self._victim(victims))

    def _insert(self, key: str, value: Any, upstream: str | None, label: str | None = None) -> None:
        """Store ``value``, weighed by its values; one another stored entry holds is weighed
        already, which spares a ``chain_topology`` result a second walk of its ``wtmm2d``
        result's chains."""
        self._store.pop(key, None)
        self._weights.pop(key, None)
        self._store[key] = value
        if key not in self._versions:
            self._sequence += 1
            self._versions[key] = self._sequence
        self._meta[key] = {**self._meta.get(key, {}), "label": label or self._meta.get(key, {}).get("label", key),
                           "touched": time.time(), "hits": self._meta.get(key, {}).get("hits", 0)}
        self._newest = key
        known = {i: n for weights in list(self._weights.values()) for i, n in weights.items()}
        parts = value.values() if isinstance(value, dict) else (value,)
        self._weights[key] = {id(v): known[id(v)] if id(v) in known else retained_nbytes(v)
                              for v in parts}
        if upstream is None:
            self._upstream.pop(key, None)
        else:
            self._upstream[key] = upstream
        self._event("insert", key)
        self._evict_over_capacity()
        self._evict_over_bytes()

    def get_or_compute(self, key: str, compute: Callable[[], Any], *,
                       upstream: str | None = None, label: str | None = None) -> Any:
        """Read either tier, or compute outside the cache lock on a miss."""
        absent = object()
        value = self.get(key, absent)
        if value is not absent:
            with self._lock:
                self.hits += 1
            return value
        with self._lock:
            if key in self._computing:
                future, _upstream = self._computing[key]
                owner = False
            else:
                future = Future()
                self._computing[key] = (future, upstream)
                owner = True
                version = self._versions.get(key, 0)
        if not owner:
            return future.result()
        try:
            value = self.load(key, absent)
            if value is absent:
                with self._lock:
                    self.misses += 1
                    self._event("miss", key, label=label)
                value = compute()
                with self._lock:
                    if key in self._store:
                        value = self._store[key]
                    elif self._versions.get(key, 0) == version:
                        self._insert(key, value, upstream, label)
            else:
                with self._lock:
                    self.hits += 1
            future.set_result(value)
            self.check_pressure()
            return value
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._computing.pop(key, None)


    def get(self, key: str, default=None):
        """Read RAM without I/O; worker resolves use get_or_compute to consult disk."""
        with self._lock:
            if key in self._store:
                self._touch(key)
                self._meta[key]["touched"] = time.time()
                self._meta[key]["hits"] += 1
                self._event("hit in RAM", key)
                return self._store[key]
        return default

    @_locked
    def ready(self, key):
        """Whether a result and its cached lineage can be read synchronously from RAM."""
        seen = set()
        while key is not None:
            if key in seen or key not in self._store:
                return False
            seen.add(key)
            key = self._upstream.get(key)
        return True

    def put(self, key: str, value: Any, *, label: str | None = None,
            upstream: str | None = None) -> None:
        self.invalidate(key)
        with self._lock:
            self._insert(key, value, upstream, label)
        self.check_pressure()

    @_locked
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

    @_locked
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

    @_locked
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
        with self.stages._lock:
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
        """Remove an entry from both tiers and revoke any pending spill."""
        with self._lock:
            present = key in self._store or key in self._computing or bool(self.disk and self.disk.record(key))
            if key in self._pending or key in self._computing:
                self._sequence += 1
                self._versions[key] = self._sequence
            else:
                self._versions.pop(key, None)
            if present:
                self._event("drop", key)
            self._drop(key)
            self._meta.pop(key, None)
        if self.disk:
            self.disk.remove(key)
        return present

    def clear(self) -> None:
        with self._lock:
            keys = set(self._store) | set(self._pending) | set(self._computing)
            if self.disk:
                keys.update(self.disk.records())
        for key in keys:
            self.invalidate(key)
        self.stages.clear()

    @_locked
    def __contains__(self, key: str) -> bool:
        # The shell uses membership to select its immediate GUI-thread cache-hit path.
        return key in self._store

    @_locked
    def __len__(self) -> int:
        return len(self._store)

    @_locked
    def pin(self, key: str) -> None:
        """Pin a key to prevent eviction. The key need not exist in the store yet."""
        self._pinned.add(key)

    @_locked
    def unpin(self, key: str) -> None:
        """Unpin a key, allowing it to be evicted. No-op if the key is not pinned."""
        self._pinned.discard(key)

    @_locked
    def is_pinned(self, key: str) -> bool:
        """Return whether a key is pinned."""
        return key in self._pinned

    @property
    @_locked
    def pinned(self) -> frozenset[str]:
        """Return a frozenset of all pinned keys."""
        return frozenset(self._pinned)


    def _event(self, action, key='', *, label=None, elapsed_ms=0, detail=''):
        self._events.append({"time": time.time(), "thread": threading.current_thread().name,
                             "event": action, "key": key,
                             "label": label or self._meta.get(key, {}).get("label", key),
                             "bytes": self._nbytes([key]), "ms": elapsed_ms, "detail": detail})

    def _victim(self, candidates):
        return next((k for k in candidates if self._nbytes([k]) >= self.spill_threshold), candidates[0])

    def _retire(self, key):
        if (self.disk is None or self._closed or
                self._nbytes([key]) < self.spill_threshold):
            self._event('drop', key)
            self._drop(key)
            self._meta.pop(key, None)
            if key not in self._computing:
                self._versions.pop(key, None)
            return
        if self.disk.record(key):
            self._event('spill', key, detail='existing disk copy')
            self._drop(key)
            return
        if self._pending:
            self._event('drop', key, detail='disk writer busy; bounded pending memory')
            self._drop(key)
            self._meta.pop(key, None)
            if key not in self._computing:
                self._versions.pop(key, None)
            return
        self._schedule_spill(key)

    def _schedule_spill(self, key):
        if key in self._pending or self._closed:
            return False
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cache-writer')
        version = self._versions.get(key, 0)
        metadata = {**self._meta[key], 'upstream': self._upstream.get(key),
                    'kind': 'lazy output' if _is_output(self._store[key]) else
                            'region window' if key.endswith('|window') else 'analysis result'}
        self._pending[key] = version
        self._executor.submit(self._write_spill, key, self._store[key], version,
                              self._disk_epoch, metadata)
        return True

    def _write_spill(self, key, value, version, epoch, metadata):
        started = time.perf_counter()
        reason = ''
        try:
            kept = self.disk.write(key, value, version, metadata)
        except Exception as exc:
            kept, reason = False, f'{type(exc).__name__}: {exc}'
        with self._lock:
            self._pending.pop(key, None)
            current = (self._versions.get(key, 0) == version and epoch == self._disk_epoch
                       and not self._closed)
            if current:
                self._event('refusal to pickle' if reason else 'spill', key,
                            elapsed_ms=(time.perf_counter()-started)*1000,
                            detail=reason or ('' if kept else 'disk budget; entry dropped'))
                if key in self._evictable([key]) and self._store.get(key) is value:
                    self._drop(key)
                self._evict_over_capacity()
                self._evict_over_bytes()
            retained = set(self._store) | set(self._pending) | set(self._computing) | set(self.disk.records())
            self._meta = {k: v for k, v in self._meta.items() if k in retained}
            self._versions = {k: v for k, v in self._versions.items() if k in retained}
        if not current:
            self.disk.remove(key, version)
        if current:
            self.check_pressure()

    def spill(self, key) -> bool:
        """Queue an evictable large entry; retained RAM remains counted until writing ends."""
        with self._lock:
            if (self.disk is None or key not in self._store or
                    key not in self._evictable([key]) or self._nbytes([key]) < self.spill_threshold):
                return False
            if self._pending and not self.disk.record(key):
                self._event('spill deferred', key, detail='disk writer busy')
                return False
            self._retire(key)
            return True

    def load(self, key, default=None):
        """Read a disk entry outside the cache lock and promote it to RAM."""
        with self._lock:
            if key in self._store:
                self._touch(key)
                return self._store[key]
            if self.disk is None or not self.disk.record(key):
                return default
            version = self._versions.get(key, 0)
            epoch = self._disk_epoch
        started = time.perf_counter()
        try:
            value, metadata = self.disk.read(key)
        except Exception as exc:
            self.disk.remove(key)
            with self._lock:
                self._event('drop', key, detail=f'load failed: {type(exc).__name__}')
            return default
        with self._lock:
            if self._versions.get(key, 0) != version or epoch != self._disk_epoch or self._closed:
                return default
            if key in self._store:
                return self._store[key]
            self._insert(key, value, metadata.get('upstream'), metadata.get('label'))
            self._event('hit on disk', key, elapsed_ms=(time.perf_counter()-started)*1000)
            self._event('load', key)
        self.check_pressure()
        return value

    def wait_for_spills(self):
        """Wait for pending writes, including any evictions their completion schedules."""
        while True:
            with self._lock:
                executor = self._executor
                pending = bool(self._pending)
            if executor is None:
                return
            executor.submit(lambda: None).result()
            with self._lock:
                if not self._pending:
                    return
            if not pending:
                return

    def check_pressure(self):
        """Release evictable memory when the system reports less than its reserve floor."""
        if not self.pressure_floor or self._closed:
            return
        import psutil
        if psutil.virtual_memory().available >= self.pressure_floor:
            return
        self.stages.clear()
        from dynamix.core.fft_policy import mlx_memory
        if mlx_memory() is not None:
            import mlx.core as mx
            mx.clear_cache()
        while psutil.virtual_memory().available < self.pressure_floor:
            with self._lock:
                candidates = self._evictable(list(self._store))
                if not candidates:
                    return
                self._retire(self._victim(candidates))


    def clear_unpinned(self):
        with self._lock:
            keys = [k for k in self._store if k not in self._pinned]
        with self._lock:
            for key in keys:
                self._event('drop', key, detail='cleared RAM tier')
                self._drop(key)

    def clear_disk(self):
        with self._lock:
            self._disk_epoch += 1
        if self.disk:
            self.disk.clear()
        with self._lock:
            self._event('drop', detail='cleared disk tier')

    def snapshot(self):
        """Copy consistent metadata for the inspector without walking retained products."""
        with self._lock:
            records = self.disk.records() if self.disk else {}
            rows = []
            upstreams = set(self._upstream.values())
            for key in dict.fromkeys([*self._store, *records]):
                in_ram = key in self._store
                metadata = self._meta.get(key, records.get(key, {}))
                holders = []
                if key in self._pinned:
                    holders.append('pinned')
                if key in self._shown.values():
                    holders.append('shown')
                if key in upstreams:
                    holders.append('upstream')
                if key == self._newest and in_ram:
                    holders.append('newest')
                kind = ('lazy output' if _is_output(self._store[key]) else
                        'region window' if key.endswith('|window') else 'analysis result') if in_ram else records[key]['kind']
                rows.append({**metadata, 'key': key, 'kind': kind, 'tier': 'RAM' if in_ram else 'disk',
                             'bytes': self._nbytes([key]) if in_ram else records[key]['bytes'],
                             'holders': holders, 'pending': key in self._pending,
                             'can_spill': bool(self.disk and in_ram and not holders and
                                               self._nbytes([key]) >= self.spill_threshold)})
            return {'report': self.report(), 'entries': sorted(rows, key=lambda r:r.get('touched',0), reverse=True),
                    'events': list(self._events), 'shown': dict(self._shown),
                    'disk': {'bytes': sum(r['bytes'] for r in records.values()),
                             'max_bytes': self.disk.max_bytes if self.disk else 0,
                             'folder': str(self.disk.folder) if self.disk else '',
                             'pending': len(self._pending)},
                    'pressure_floor': self.pressure_floor, 'hits': self.hits, 'misses': self.misses}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        if executor:
            executor.shutdown(wait=True)
        if self.disk:
            self.disk.close()



def _is_output(value) -> bool:
    """Whether ``value`` is a lazy output: a dict holding a ``"raster"`` ndarray, or a pyramid
    level travelling as its decimated ``"cells"`` with their ``"paint"`` placement."""
    return isinstance(value, dict) and (bool(_raster_nbytes(value))
                                        or (isinstance(value.get("cells"), np.ndarray) and "paint" in value))


def _is_result(value) -> bool:
    """Whether ``value`` is an analysis result: a dict other than a lazy output."""
    return isinstance(value, dict) and not _is_output(value)


def _raster_nbytes(value) -> int:
    """The bytes of ``value``'s ``"raster"`` when it is a raster output (a dict holding a
    ``"raster"`` ndarray), else 0."""
    if isinstance(value, dict):
        raster = value.get("raster")
        if isinstance(raster, np.ndarray):
            return int(raster.nbytes)
    return 0
