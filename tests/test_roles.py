# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The shell's device sets come from declarations on the devices; the old hand tables are the baseline."""
from __future__ import annotations

from dynamix.devices import register_builtin_devices
from dynamix.model.roles import (band_devices, band_for_analyzer, chain_producers, consumer_producers,
                                 is_dev, primary_analyzers, shell_placed, superseded)


def test_the_derived_sets_equal_the_hand_tables(clean_registry):
    from dynamix.shell import browser, main_window

    register_builtin_devices()
    assert primary_analyzers() == main_window._PRIMARY_ANALYZERS
    assert chain_producers() == main_window._CHAIN_PRODUCERS
    assert consumer_producers() == main_window._CONSUMER_PRODUCERS
    assert band_devices() == main_window._BAND_DEVICES
    assert band_for_analyzer() == main_window._BAND_FOR_ANALYZER
    assert set(superseded()) == set(browser._SUPERSEDED)
    assert set(shell_placed()) == set(browser._SHELL_PLACED)      # registration order differs
    for name, device in clean_registry.items():
        expected = (name.startswith(browser._DEV_PREFIX) or name in browser._SUPERSEDED
                    or name in browser._SHELL_PLACED or not device.families)
        assert is_dev(name, device) == expected, name


def test_a_new_device_joins_the_sets_by_declaration_alone(clean_registry):
    class NewAnalyzer:
        name = "new_analyzer"
        params = ()
        families = ("Labs",)
        role = "primary"
        chain_producer = True

        def compute(self, field, params, progress=None):
            return {}

        def cache_key(self, params):
            return ("new_analyzer",)

    class OldOne:
        name = "old_one"
        params = ()
        families = ()
        superseded_by = "new_analyzer"

        def apply(self, extrema, params):
            return extrema

    clean_registry["new_analyzer"], clean_registry["old_one"] = NewAnalyzer(), OldOne()
    assert primary_analyzers() == ("new_analyzer",) and chain_producers() == ("new_analyzer",)
    assert superseded() == {"old_one": "new_analyzer"} and is_dev("old_one", OldOne())
    assert not is_dev("new_analyzer", NewAnalyzer())
