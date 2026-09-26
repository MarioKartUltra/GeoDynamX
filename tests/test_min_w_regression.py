# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Regression: interleaved selection-aware and non-aware filters must not freeze the app.

The failure guarded: the min |W| frac filter stops working and the scale bar lags badly. Root
cause: the selection/materialize layer back-references each chain to the ORIGINAL transform
layer; when a non-aware filter (min_vchains) rewrites extrema between a selection-aware
scale_select and modulus_threshold, ``materialize_selection`` applies stale ``h_src`` indices
to an already-shrunk layer and raises IndexError inside a Qt signal handler -- Qt swallows it
and the canvas silently stops redrawing.

These assert the OBSERVABLE contract, through the real engine on the real fixture:
* the DEMO-shaped interleaved chain resolves without raising, repeatedly;
* raising ``modulus_threshold``'s fraction monotonically REDUCES kept extrema (min |W| works);
* the kept count for a given fraction is STABLE across repeated resolves (no cached-layer
  corruption -- the monotonic 144724->78045->26666 shrink the bug produced).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.project import Project

FIXTURE = "tests/fixtures/kam_64.npz"


def _kept(result) -> int:
    return sum(len(l["x"]) for l in (result.get("extrema") or []))


@pytest.fixture
def demo_chain(clean_registry):
    register_builtin_devices()
    field = RasterField.load_npz(FIXTURE)
    p = Project()
    src = p.add_source("kam")
    cache = Cache()

    def resolve_with(frac):
        steps = (
            DeviceRef("wtmm2d", {"n_oct": 2, "n_voice": 2}),
            DeviceRef("chain_topology", {}),
            DeviceRef("scale_select", {"scale_idx": 0}),
            DeviceRef("min_vchains", {"min_vchains": 1}),
            DeviceRef("modulus_threshold", {"frac": frac}),
        )
        layer = p.add_layer("x", src.source_id, Chain(steps))
        return resolve(layer, field, cache).result

    return resolve_with


def test_interleaved_chain_resolves_without_raising(demo_chain):
    demo_chain(0.0)          # pre-fix: IndexError on the SECOND resolve of this chain shape
    demo_chain(0.3)


def test_modulus_fraction_monotonically_reduces_kept_extrema(demo_chain):
    counts = [_kept(demo_chain(f)) for f in (0.0, 0.1, 0.3, 0.5)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1]                    # it actually does something


def test_kept_count_is_stable_across_repeated_resolves(demo_chain):
    """The corruption signature: the same fraction gave a DIFFERENT (shrinking) count each time
    because a cached layer was being mutated. Same fraction -> same count, always."""
    a = _kept(demo_chain(0.2))
    b = _kept(demo_chain(0.2))
    c = _kept(demo_chain(0.2))
    assert a == b == c, (a, b, c)
