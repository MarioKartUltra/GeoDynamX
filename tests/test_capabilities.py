# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The eleven duck-typed capabilities have names; a device may declare them, the probes remain."""
from __future__ import annotations

from dynamix.devices import register_builtin_devices
from dynamix.model.capabilities import (CAPABILITIES, PROBES, declared, differences, has, probed,
                                         undeclared)


def test_the_eleven_capabilities_are_named_and_each_has_a_probe():
    assert len(CAPABILITIES) == 11 and set(CAPABILITIES) == set(PROBES)
    assert "FieldStage" in CAPABILITIES and "StripExtras" in CAPABILITIES


def test_declared_wins_over_probed_and_probed_is_the_fallback():
    class A:
        name = "a"
        params = ()
        field_stage = True

    class B(A):
        capabilities = frozenset({"Cancellable"})

    assert has(A(), "FieldStage") and not has(B(), "FieldStage") and has(B(), "Cancellable")
    assert declared(A()) is None and "FieldStage" in probed(A())


def test_every_builtin_declares_what_it_probes(clean_registry):
    register_builtin_devices()
    assert undeclared() == []
    assert differences() == {}


def test_a_device_that_declares_nothing_is_reported_and_a_wrong_declaration_is_listed(clean_registry):
    class Bare:
        name = "bare"
        params = ()
        field_stage = True

    class Wrong(Bare):
        name = "wrong"
        capabilities = frozenset({"Cancellable"})

    clean_registry["bare"], clean_registry["wrong"] = Bare(), Wrong()
    assert undeclared() == ["bare"]
    assert differences() == {"wrong": ({"Cancellable"}, {"FieldStage"})}


def test_the_engine_reads_the_declaration_and_probes_only_a_device_that_declares_nothing(clean_registry):
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from tests.conftest import make_field, make_layer

    seen = {}

    class Quiet:
        """Sets ``wants_cancel`` but declares no capability: the declaration wins, no cancel."""
        name = "quiet"
        params = ()
        families = ("Labs",)
        consumes = (("Field",),)
        produces = frozenset({"Field"})
        capabilities = frozenset()
        wants_cancel = True

        def compute(self, field, params, progress=None, **kw):
            seen[self.name] = sorted(kw)
            return {"values": field.values}

        def cache_key(self, params):
            return (self.name,)

    class Loud(Quiet):
        """Declares nothing at all: the probe finds ``wants_cancel`` and the cancel is passed."""
        name = "loud"
        capabilities = None

    clean_registry["quiet"], clean_registry["loud"] = Quiet(), Loud()
    for name in ("quiet", "loud"):
        resolve(make_layer((name, {})), make_field(), Cache(), cancel=lambda: False)
    assert seen == {"quiet": [], "loud": ["cancel"]}
