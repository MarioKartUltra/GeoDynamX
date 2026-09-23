# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.engine.cache."""
from __future__ import annotations

import pytest

from dynamix.engine.cache import Cache


def test_cache_pin_unpin():
    c = Cache()
    c.put("k", 1)
    c.pin("k")
    assert c.is_pinned("k") and "k" in c.pinned
    c.unpin("k")
    assert not c.is_pinned("k")


def test_cache_pinned_property_is_frozenset():
    c = Cache()
    c.put("k1", 1)
    c.put("k2", 2)
    c.pin("k1")
    c.pin("k2")
    pinned = c.pinned
    assert isinstance(pinned, frozenset)
    assert pinned == frozenset(["k1", "k2"])


def test_cache_pin_nonexistent_key():
    c = Cache()
    c.pin("nonexistent")
    assert c.is_pinned("nonexistent")


def test_cache_unpin_nonexistent_key():
    c = Cache()
    c.unpin("nonexistent")
    assert not c.is_pinned("nonexistent")


# ------------------------------------------------- bounded LRU (2026-08-30, "not responding")
# The unbounded store assumed "few WTMM stacks"; the noise device's seed/amplitude scrubs mint
# a ~134 MB field clone plus a full WTMM result PER STEP, retained forever -> memory pressure,
# beachball. Bounded LRU, pins exempt (the frozen-layer contract the docstring promised).

def test_capacity_evicts_the_least_recently_used_unpinned_entry():
    c = Cache(maxsize=3)
    for k in ("a", "b", "c"):
        c.put(k, k.upper())
    assert c.get("a") == "A"                   # refresh a: b is now the LRU
    c.put("d", "D")
    assert "b" not in c and all(k in c for k in ("a", "c", "d"))


def test_get_or_compute_hit_refreshes_recency():
    c = Cache(maxsize=2)
    c.put("a", 1); c.put("b", 2)
    c.get_or_compute("a", lambda: 99)          # hit: refresh a
    c.put("c", 3)
    assert "b" not in c and "a" in c


def test_pinned_entries_survive_eviction_even_when_oldest():
    c = Cache(maxsize=2)
    c.put("old", 1); c.pin("old")
    c.put("b", 2); c.put("c", 3); c.put("d", 4)
    assert "old" in c and c.get("old") == 1
    assert len(c) <= 3                          # the pin rides above the bound; the rest is bounded


def test_default_capacity_is_finite_but_generous():
    c = Cache()
    assert 32 <= c.maxsize < 10_000
