# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Values a result carries under their usual keys but builds on their first read.

A transform's landing pays only for what the landing reads; a product read later (the partition
tables a fitting window opens on, the chain product an export writes) is built by its first
reader and kept on the result, so every later reader gets the same object. The first read may come
from the worker or from the GUI thread: a lock makes it the only build, and a second reader waits
for it.

A :class:`Deferred` is a ``Mapping``, not a ``dict``: a presence test (``is not None``, ``in``)
never builds it, and the stage store's ``_freeze`` and ``_own_structure`` and the engine's
``stage_nbytes`` pass it through unread.
"""
from __future__ import annotations

import copy
import threading
from collections.abc import Mapping


def _same(value):
    return value


class Once:
    """``build()``'s value, computed on the first call and kept. A deep copy of an unbuilt one
    shares the builder and builds for itself; of a built one, copies the value. Pickling carries
    the value, built if need be, since the builder's closure cannot travel."""

    __slots__ = ("_build", "_value", "_lock")

    def __init__(self, build):
        self._build, self._value, self._lock = build, None, threading.Lock()

    def __deepcopy__(self, memo):
        if self._build is not None:
            return Once(self._build)
        twin = Once(None)
        twin._value = copy.deepcopy(self._value, memo)
        return twin

    def __reduce__(self):
        return _same, (self(),)

    @property
    def built(self) -> bool:
        return self._build is None

    def __call__(self):
        if self._build is not None:
            with self._lock:
                if self._build is not None:
                    self._value = self._build()
                    self._build = None
        return self._value


class Deferred(Mapping):
    """The mapping ``build()`` returns, built when a key, its length or its keys are first read."""

    __slots__ = ("_once",)

    def __init__(self, build):
        self._once = Once(build)

    def __deepcopy__(self, memo):
        twin = Deferred(None)
        twin._once = copy.deepcopy(self._once, memo)
        return twin

    def __reduce__(self):
        return _same, (dict(self._once()),)

    @property
    def built(self) -> bool:
        return self._once.built

    def __getitem__(self, key):
        return self._once()[key]

    def __iter__(self):
        return iter(self._once())

    def __len__(self):
        return len(self._once())


def deferred_pair(build) -> tuple[Deferred, Deferred]:
    """Two mappings filled by one call of ``build()``, which returns them as a pair."""
    both = Once(build)
    return Deferred(lambda: both()[0]), Deferred(lambda: both()[1])
