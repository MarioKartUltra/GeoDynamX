# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The D(h) comparison board (``dynamix.model.spectra_board``): the sent spectra in legend order,
replaced in place when one is sent again under its identity, renamed for display only."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dynamix.model.spectra_board import KINDS, SentSpectrum, SpectraBoard


def _entry(kind="canonical", window=(1.0, 5.0), name="canonical (Arneodo WTMM) · A", **kw):
    h = np.linspace(0.2, 0.8, 7)
    identity = {"tool": {"recipe": "k1", "layer_id": "L1", "layer": "A"}, "scale_window": window,
                "fit_mode": "naive", "table": "std", "settings": {"eta": 0.0}}
    return SentSpectrum(name=name, full_name=name, kind=kind, h=h, D=2.0 - 4.0 * (h - 0.5) ** 2,
                        identity=identity, **kw)


def test_add_rename_remove_move_and_visibility():
    board = SpectraBoard()
    a = board.add(_entry())
    b = board.add(_entry(kind="histogram", name="microcanonical histogram · A"))
    c = board.add(_entry(kind="large_deviations", name="A · leaders, large deviations"))
    assert [e.key for e in board.entries()] == [a, b, c] and len(board) == 3
    board.move(c, -1)
    assert [e.key for e in board.entries()] == [a, c, b]
    board.move(a, -5)                                   # stops at the front
    board.move(b, +3)                                   # and at the back
    assert [e.key for e in board.entries()] == [a, c, b]
    board.set_visible(b, False)
    board.set_show_errors(a, False)
    board.set_show_tangent(a, False)
    board.set_colour(a, "#123456")
    got = board.get(a)
    assert not board.get(b).visible and not got.show_errors and not got.show_tangent
    assert got.colour == "#123456"
    board.remove(c)
    assert [e.key for e in board.entries()] == [a, b]
    with pytest.raises(KeyError):
        board.remove(c)


def test_a_name_is_display_only():
    board = SpectraBoard()
    key = board.add(_entry())
    board.rename(key, "  WTMM  ")
    entry = board.get(key)
    assert entry.name == "WTMM" and entry.full_name == "canonical (Arneodo WTMM) · A"
    assert entry.record()["name"] == "WTMM"
    assert entry.record()["full_name"] == "canonical (Arneodo WTMM) · A"
    board.rename(key, "")                               # empty restores the producer's name
    assert entry.name == entry.full_name


def test_sending_again_under_one_identity_replaces_in_place():
    board = SpectraBoard()
    first = board.add(_entry())
    other = board.add(_entry(kind="histogram"))
    board.rename(first, "mine")
    board.set_colour(first, "#00ff00")
    board.set_visible(first, False)
    again = _entry()
    again.D = again.D + 0.1
    assert board.add(again) == first
    assert len(board) == 2 and [e.key for e in board.entries()] == [first, other]
    entry = board.get(first)
    assert entry is again and np.allclose(entry.D, _entry().D + 0.1)
    assert (entry.name, entry.colour, entry.visible) == ("mine", "#00ff00", False)


def test_sending_again_keeps_the_bars_and_tangent_toggles():
    """The display state survives a send again: bars and tangent switched off stay off."""
    tangent = {"q_c": -1.0, "h_c": 0.9, "D_c": 1.2, "slope": -1.0, "intercept": 2.1}
    board = SpectraBoard()
    key = board.add(_entry(dD=np.full(7, 0.05), tangent=tangent))
    board.set_show_errors(key, False)
    board.set_show_tangent(key, False)
    board.add(_entry(dD=np.full(7, 0.04), tangent=tangent))
    entry = board.get(key)
    assert len(board) == 1 and not entry.show_errors and not entry.show_tangent
    np.testing.assert_allclose(entry.dD, 0.04)
    board.set_show_errors(key, True)
    board.add(_entry(dD=np.full(7, 0.03), tangent=tangent))
    assert board.get(key).show_errors and not board.get(key).show_tangent


def test_a_new_scale_window_fit_mode_or_kind_makes_a_new_entry():
    board = SpectraBoard()
    board.add(_entry(window=(1.0, 5.0)))
    board.add(_entry(window=(2.0, 5.0)))
    changed_mode = _entry()
    changed_mode.identity["fit_mode"] = "focus"
    board.add(changed_mode)
    board.add(_entry(kind="boxcount"))
    assert len(board) == 4


def test_the_entry_checks_its_arrays_and_tangent():
    with pytest.raises(ValueError, match="kind"):
        _entry(kind="microcanonical")
    with pytest.raises(ValueError, match="length"):
        SentSpectrum(name="x", full_name="x", kind="canonical", h=[0.1, 0.2], D=[1.0])
    with pytest.raises(ValueError, match="dD"):
        _entry(dD=np.ones(3))
    with pytest.raises(ValueError, match="q and tau"):
        _entry(q=np.arange(3.0))
    with pytest.raises(ValueError, match="tangent"):
        _entry(tangent={"q_c": -1.0})
    entry = _entry(dD=np.full(7, 0.05), q=np.arange(3.0), tau=np.arange(3.0) - 2.0,
                   tangent={"q_c": -1.0, "h_c": 0.9, "D_c": 1.2, "slope": -1.0, "intercept": 2.1})
    assert entry.has_errors and entry.h.dtype == np.float64
    assert set(KINDS) >= {"canonical", "histogram", "generalized", "cwt_leaders"}


def test_the_record_is_plain_json_and_says_what_was_drawn():
    entry = _entry(dD=np.full(7, 0.05),
                   tangent={"q_c": -1.0, "h_c": 0.9, "D_c": 1.2, "slope": -1.0, "intercept": 2.1})
    entry.identity["settings"]["grid"] = np.array([np.nan, 1.0])
    record = entry.record()
    assert json.loads(json.dumps(record)) == record
    assert record["scale_window"] == [1.0, 5.0] and record["fit_mode"] == "naive"
    assert record["table"] == "std" and record["errors"] is True
    assert record["identity"]["tool"]["layer"] == "A"
    assert record["identity"]["settings"]["grid"] == [None, 1.0]
    assert record["tangent"]["q_c"] == -1.0
    entry.show_errors = entry.show_tangent = False
    assert entry.record()["errors"] is False and entry.record()["tangent"] is None
    assert _entry().record()["errors"] is False                 # a producer without bars
