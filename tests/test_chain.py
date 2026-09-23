# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.model.chain -- ordering is an invariant, not a convention."""
from __future__ import annotations

import pytest

from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import register_device


@pytest.fixture
def _registry(clean_registry, stub_transform, stub_filter):
    """Register stub devices for the test."""
    register_device(stub_transform)
    register_device(stub_filter)


def test_transforms_then_filters_is_accepted(_registry):
    c = Chain((DeviceRef("t", {"scale": 2}), DeviceRef("f", {"cut": 0.2})))
    c.validate()
    assert [r.device for r in c.transforms] == ["t"]
    assert [r.device for r in c.filters] == ["f"]


def test_a_transform_after_a_filter_is_rejected(_registry):
    """Interleaving is what makes cache correctness hard, so it is refused at
    construction."""
    c = Chain((DeviceRef("f", {}), DeviceRef("t", {})))
    with pytest.raises(ValueError, match="transforms-then-filters"):
        c.validate()


def test_filters_only_is_valid(_registry):
    Chain((DeviceRef("f", {}),)).validate()


def test_empty_chain_is_valid(_registry):
    Chain(()).validate()


def test_validate_rejects_an_unknown_device(_registry):
    with pytest.raises(KeyError, match="no device"):
        Chain((DeviceRef("nope", {}),)).validate()


def test_validate_rejects_a_bad_param(_registry):
    with pytest.raises(ValueError, match="above max"):
        Chain((DeviceRef("t", {"scale": 99}),)).validate()


def test_payload_round_trip_fills_defaults(_registry):
    c = Chain((DeviceRef("t", {}), DeviceRef("f", {"cut": 0.25})))
    back = Chain.from_payload(c.to_payload())
    assert back.steps[0].params == {"scale": 4}          # default materialised
    assert back.steps[1].params == {"cut": 0.25}


def test_splitting_an_interleaved_chain_raises_rather_than_reordering(_registry):
    """``['f', 't']`` used to yield transforms ``['t']`` and filters ``['f']`` -- a chain nobody
    wrote, executed by Phase 2 with nothing raised anywhere."""
    c = Chain((DeviceRef("f", {}), DeviceRef("t", {})))
    with pytest.raises(ValueError, match="transforms-then-filters"):
        c.transforms
    with pytest.raises(ValueError, match="transforms-then-filters"):
        c.filters


def test_materialized_fills_defaults_without_a_payload_round_trip(_registry):
    """The in-memory half of the same guarantee: a chain built by hand must be as fully specified
    as one loaded from a file, or the two are not the same recipe."""
    c = Chain((DeviceRef("t", {}), DeviceRef("f", {"cut": 0.25}))).materialized()
    assert c.steps[0].params == {"scale": 4}
    assert c.steps[1].params == {"cut": 0.25}
    assert Chain.from_payload(c.to_payload()) == c       # identical to its round-tripped twin


def test_materialized_leaves_the_original_untouched(_registry):
    """Chain is frozen; materialized returns a new one rather than filling in place."""
    original = Chain((DeviceRef("t", {}),))
    filled = original.materialized()
    assert original.steps[0].params == {}
    assert filled is not original and filled.steps[0].params == {"scale": 4}


def test_materialized_rejects_what_validate_rejects(_registry):
    with pytest.raises(ValueError, match="above max"):
        Chain((DeviceRef("t", {"scale": 99}),)).materialized()
    with pytest.raises(KeyError, match="no device"):
        Chain((DeviceRef("nope", {}),)).materialized()
    with pytest.raises(ValueError, match="transforms-then-filters"):
        Chain((DeviceRef("f", {}), DeviceRef("t", {}))).materialized()
