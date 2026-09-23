# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""THE PHASE 1 GATE. Three devices chosen to stress the param schema, not flatter it.

If any of these cannot be expressed as a Param tuple plus an optional check() hook, the
generated-knob design is wrong and phases 3 and 4 must be replanned before they depend on it.
"""
from __future__ import annotations

import pytest

from dynamix.devices import BUILTIN_DEVICES, register_builtin_devices
from dynamix.devices.stubs import StubHolder, StubWavelet, StubWedge
from dynamix.model.device import DEVICES, defaults_for, get_device, is_transform, validate_params
from dynamix.model.param import Param, ParamKind


def test_every_stub_declares_a_complete_schema():
    """The whole bet: a device's knobs are fully described by its declaration, with no GUI code.

    This asserted ``set(defaults_for(d)) == {p.name for p in d.params}``, which is the definition
    of ``defaults_for`` restated -- it could not fail for any input. What Phase 4 actually needs is
    checked instead: distinct names to key the knobs by, a label to put on each one, a declaration
    that survives the trip to a payload and back, and defaults that are themselves legal values.
    """
    for d in (StubWavelet(), StubHolder(), StubWedge()):
        assert d.params, d.name
        names = [p.name for p in d.params]
        assert len(set(names)) == len(names), f"{d.name}: duplicate param name in {names}"
        for p in d.params:
            assert p.label, f"{d.name}.{p.name}: no label for the generated knob"
            assert Param.from_payload(p.to_payload()) == p, f"{d.name}.{p.name}: no round trip"
        assert validate_params(d, {}) == defaults_for(d), d.name


def test_importing_the_devices_package_registers_nothing(clean_registry):
    """Registration is explicit, never an import side effect: importing a module must not mutate
    global state, or test isolation becomes a question of import order."""
    import importlib

    importlib.reload(importlib.import_module("dynamix.devices"))
    assert DEVICES == {}


def test_register_builtin_devices_populates_the_registry(clean_registry):
    """Nothing under src/ called register_device, so open_project raised 'no device named ...' for
    any real project unless the caller had registered by hand."""
    added = register_builtin_devices()
    # The three gate stubs plus the real devices the application actually drives.
    assert {"stub_wavelet", "stub_holder", "stub_wedge"} <= set(added)
    assert {"wtmm2d", "scale_select", "orientation_wedge", "modulus_threshold"} <= set(added)
    assert set(DEVICES) == set(added)
    assert get_device("stub_wavelet").name == "stub_wavelet"
    assert get_device("wtmm2d").name == "wtmm2d"
    assert len(added) == len(BUILTIN_DEVICES)


def test_register_builtin_devices_is_idempotent(clean_registry):
    """A second window, or a plugin that also registers, must not crash the application."""
    register_builtin_devices()
    before = dict(DEVICES)
    assert register_builtin_devices() == []                  # nothing new, and no raise
    assert set(DEVICES) == set(before)


def test_transform_and_filters_are_correctly_classified():
    assert is_transform(StubWavelet()) is True
    assert is_transform(StubHolder()) is False
    assert is_transform(StubWedge()) is False


def test_mixed_kind_transform_validates():
    d = StubWavelet()
    p = validate_params(d, {"n_octaves": 6, "wavelet": "morlet", "normalise": False})
    assert p == {"n_octaves": 6, "wavelet": "morlet", "normalise": False}
    with pytest.raises(ValueError, match="not one of"):
        validate_params(d, {"wavelet": "haar"})


def test_cache_key_is_stable_and_param_sensitive():
    d = StubWavelet()
    a = d.cache_key("src", validate_params(d, {"n_octaves": 4}))
    b = d.cache_key("src", validate_params(d, {"n_octaves": 4}))
    c = d.cache_key("src", validate_params(d, {"n_octaves": 5}))
    assert a == b and a != c


def test_paired_param_cross_validation_is_rejected():
    """min > max is illegal, and no single Param can know that -- this is what check() is for."""
    d = StubHolder()
    validate_params(d, {"h_min": 0.1, "h_max": 0.8})
    with pytest.raises(ValueError, match="h_min .* h_max"):
        validate_params(d, {"h_min": 0.9, "h_max": 0.2})


def test_angle_param_wraps_and_carries_its_frame():
    """An orientation is (angle, frame). The wedge centre wraps mod 180; the frame is a choice."""
    d = StubWedge()
    p = validate_params(d, {"centre": 190.0, "half_width": 15.0, "north": "true"})
    assert p["centre"] == pytest.approx(10.0)
    assert p["north"] == "true"
    assert next(q for q in d.params if q.name == "centre").kind is ParamKind.ANGLE
    with pytest.raises(ValueError, match="not one of"):
        validate_params(d, {"north": "magnetic_declination"})


def test_wedge_accepts_a_lineament_across_the_wrap():
    """A wedge centred at 10 deg must capture a 175 deg lineament. This is the query structural
    geologists actually run, and getting it wrong is silent."""
    d = StubWedge()
    p = validate_params(d, {"centre": 10.0, "half_width": 20.0, "north": "true"})
    assert d.apply({"orientations": [175.0, 100.0]}, p)["orientations"] == [175.0]
