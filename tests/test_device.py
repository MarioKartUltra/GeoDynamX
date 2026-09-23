# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.model.device -- the two-kind protocol and its registry."""
from __future__ import annotations

import pytest

from dynamix.model.device import (defaults_for, get_device, is_transform, register_device,
                                  validate_params)


def test_register_and_retrieve(stub_transform, clean_registry):
    register_device(stub_transform)
    assert get_device("t").name == "t"


def test_duplicate_name_raises(stub_transform, clean_registry):
    register_device(stub_transform)
    with pytest.raises(ValueError, match="already registered"):
        register_device(stub_transform)


def test_unknown_name_raises(clean_registry):
    with pytest.raises(KeyError, match="no device"):
        get_device("nope")


def test_transform_and_filter_are_distinguished(stub_transform, stub_filter, clean_registry):
    register_device(stub_transform)
    register_device(stub_filter)
    assert is_transform(stub_transform) is True
    assert is_transform(stub_filter) is False


def test_defaults_come_from_the_schema(stub_transform):
    assert defaults_for(stub_transform) == {"scale": 4}


def test_validate_params_fills_defaults_and_checks_ranges(stub_transform):
    assert validate_params(stub_transform, {}) == {"scale": 4}
    assert validate_params(stub_transform, {"scale": 7}) == {"scale": 7}
    with pytest.raises(ValueError, match="above max"):
        validate_params(stub_transform, {"scale": 99})


def test_registering_a_device_of_neither_shape_is_rejected(clean_registry):
    """A mistyped cache_key is worse than silent: kind is decided structurally, so the device
    becomes a Filter and the chain fails later with 'chain must be transforms-then-filters',
    blaming the ordering for what is a typo."""
    from dynamix.model.param import Param, ParamKind

    class Typo:
        name = "typo"
        params = (Param("scale", ParamKind.INT, default=1),)

        def compute(self, field, params, *, progress=None):
            return {}

        def cache_ky(self, source_id, params):     # noqa: the typo IS the test
            return ""

    with pytest.raises(ValueError, match="neither a Transform nor a Filter"):
        register_device(Typo())
    assert "typo" not in clean_registry


def test_registering_a_device_without_a_name_is_rejected(clean_registry):
    class Nameless:
        name = ""
        params = ()

        def apply(self, result, params):
            return result

    with pytest.raises(ValueError, match="no name"):
        register_device(Nameless())


def test_validate_params_rejects_unknown_keys(stub_transform):
    """A typo'd param must fail loudly, not be silently ignored -- it would look like the knob
    simply had no effect."""
    with pytest.raises(ValueError, match="unknown parameter"):
        validate_params(stub_transform, {"scal": 4})
