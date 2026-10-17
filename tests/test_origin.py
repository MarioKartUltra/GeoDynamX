# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every device carries an origin: authors, an accepted licence, a citation and a URL; the derived
devices say what they were ported from; the browser's tooltip shows it."""
from __future__ import annotations

import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model.device import get_device
from dynamix.model.families import display_name, tooltip_of
from dynamix.model.origin import (ACCEPTED_LICENSES, AUTHOR, FIRST_PARTY, KINDS, Origin,
                                  origin_of_device)


def test_an_origin_refuses_an_unknown_kind_an_unaccepted_licence_and_no_author():
    with pytest.raises(ValueError, match="kind"):
        Origin("borrowed", (AUTHOR,), "MIT")
    with pytest.raises(ValueError, match="licence"):
        Origin("third_party", ("Someone",), "WTFPL")
    with pytest.raises(ValueError, match="author"):
        Origin("first_party", (), "MIT")
    assert set(KINDS) == {"first_party", "derived", "third_party"}


def test_a_device_that_declares_nothing_is_first_party_and_a_wrong_declaration_is_refused():
    class Quiet:
        name = "quiet"
        params = ()

    class Wrong(Quiet):
        name = "wrong"
        origin = "MIT"

    assert origin_of_device(Quiet()) is FIRST_PARTY
    with pytest.raises(TypeError, match="wrong"):
        origin_of_device(Wrong())


def test_every_builtin_has_an_accepted_origin_and_the_ports_say_so(clean_registry):
    register_builtin_devices()
    for name, device in clean_registry.items():
        o = origin_of_device(device)
        assert o.license in ACCEPTED_LICENSES and o.authors, name
    wtmm = origin_of_device(get_device("wtmm2d"))
    assert wtmm.kind == "derived" and wtmm.license == "GPL-2.0-only" and "xsmurf" in wtmm.citation
    assert "Alain Arneodo" in wtmm.authors and AUTHOR in wtmm.authors
    assert origin_of_device(get_device("wtmm2d_roi")) is wtmm
    mz = origin_of_device(get_device("mz_edges"))
    assert mz.kind == "derived" and "LastWave" in mz.citation and "Stéphane Mallat" in mz.authors
    assert origin_of_device(get_device("noise")) is FIRST_PARTY


def test_the_tooltip_carries_the_origin_lines(clean_registry):
    register_builtin_devices()
    wtmm = get_device("wtmm2d")
    tip = tooltip_of(wtmm)
    assert tip.startswith(display_name(wtmm) + "\n")
    assert "GPL-2.0-only (derived)" in tip and "Arneodo" in tip
    assert tooltip_of(get_device("noise")).splitlines()[1:] == [AUTHOR, "GPL-2.0-or-later",
                                                                FIRST_PARTY.url]
