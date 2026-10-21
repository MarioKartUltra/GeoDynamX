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

import threading
from collections.abc import Mapping


class Once:
    """``build()``'s value, computed on the first call and kept."""

    __slots__ = ("_build", "_value", "_lock")

    def __init__(self, build):
        self._build, self._value, self._lock = build, None, threading.Lock()

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
