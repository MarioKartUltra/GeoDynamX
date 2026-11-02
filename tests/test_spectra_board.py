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
    assert entry.renamed and entry.default_name == "canonical (Arneodo WTMM) · A"
    board.rename(key, "")                               # empty restores the producer's name
    assert entry.name == entry.full_name and not entry.renamed


def test_an_empty_rename_restores_the_producer_s_display_name_and_a_send_again_keeps_a_rename():
    """The producer's display name (a short name) is what an empty rename restores, never the
    full name; a send again keeps a name the user gave and takes the producer's new one
    otherwise."""
    board = SpectraBoard()
    first = _entry()
    first.full_name += " · log2 a 1–5"
    first = SentSpectrum(**{**{k: getattr(first, k) for k in ("full_name", "kind", "h", "D", "identity")},
                            "name": "WTMM (no sup) · a 1–5"})
    key = board.add(first)
    assert first.default_name == "WTMM (no sup) · a 1–5" and not first.renamed
    board.rename(key, "WTMM (no sup) · a 1–5")          # its own name again: not a rename
    assert not board.get(key).renamed
    again = SentSpectrum(name="WTMM (no sup) · a 1–5 new", full_name=first.full_name, kind="canonical",
                         h=first.h, D=first.D, identity=first.identity)
    board.add(again)
    assert board.get(key).name == "WTMM (no sup) · a 1–5 new"   # not renamed: the producer's name
    board.rename(key, "mine")
    assert board.get(key).renamed
    third = SentSpectrum(name="WTMM (no sup) · a 1–5", full_name=first.full_name, kind="canonical",
                         h=first.h, D=first.D, identity=first.identity)
    board.add(third)
    assert (third.name, third.renamed, third.default_name) == ("mine", True, "WTMM (no sup) · a 1–5")
    board.rename(key, "   ")
    assert (third.name, third.renamed) == ("WTMM (no sup) · a 1–5", False)


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


PRODUCER = {"window": "SpectrumWindow", "tool": {"recipe": "k1", "layer_id": "L1", "layer": "A"}}


def test_replace_swaps_an_entry_keeping_its_display_state_and_a_given_name():
    """A replace puts a new fit in place of an entry: its key, its legend position, colour,
    visibility, bars and tangent switches and a name the user gave stay; a producer's name is
    the new fit's."""
    board = SpectraBoard()
    first = board.add(_entry(window=(1.0, 5.0), name="WTMM (sup) · a 1–5", producer=PRODUCER))
    other = board.add(_entry(kind="histogram"))
    board.set_colour(first, "#00ff00")
    board.set_visible(first, False)
    board.set_show_errors(first, False)
    new = _entry(window=(2.0, 6.0), name="WTMM (sup) · a 2–6", producer=PRODUCER)
    assert board.replace(first, new) == first
    assert [e.key for e in board.entries()] == [first, other] and board.get(first) is new
    assert (new.colour, new.visible, new.show_errors) == ("#00ff00", False, False)
    assert new.name == "WTMM (sup) · a 2–6" and not new.renamed       # the default name recomputed
    board.rename(first, "mine")
    board.replace(first, _entry(window=(3.0, 6.0), name="WTMM (sup) · a 3–6", producer=PRODUCER))
    entry = board.get(first)
    assert (entry.name, entry.renamed, entry.default_name) == ("mine", True, "WTMM (sup) · a 3–6")
    assert entry.identity["scale_window"] == (3.0, 6.0)
    with pytest.raises(KeyError):
        board.replace("nowhere", _entry())


def test_a_replace_identical_to_another_entry_drops_the_later_of_the_two():
    """Replacing an entry by a spectrum exactly identical to another entry would leave two of a
    kind: the earlier of the two in board order takes the spectrum, with its own key, colour and
    name, and the later goes, whichever of them was replaced."""
    board = SpectraBoard()
    a = board.add(_entry(window=(1.0, 5.0), producer=PRODUCER))
    h = board.add(_entry(kind="histogram"))
    b = board.add(_entry(window=(2.0, 5.0), producer=PRODUCER))
    board.set_colour(a, "#00ff00")
    twin = _entry(window=(1.0, 5.0), name="again", producer=PRODUCER)
    assert board.replace(b, twin) == a                       # b replaced by a's twin: a holds it, b goes
    assert [e.key for e in board.entries()] == [a, h] and board.get(a) is twin
    assert twin.colour == "#00ff00" and board.latest_of("canonical", PRODUCER) == a
    c = board.add(_entry(window=(3.0, 5.0), producer=PRODUCER))
    board.move(c, -2)                                        # c before a
    assert board.replace(a, _entry(window=(3.0, 5.0), producer=PRODUCER)) == c
    assert [e.key for e in board.entries()] == [c, h]


def test_latest_of_names_the_most_recently_sent_entry_of_a_kind_and_producer():
    """The most recent send or replace of one kind from one producer, whatever the legend order;
    another kind or producer is never it; None when there is none."""
    board = SpectraBoard()
    assert board.latest_of("canonical", PRODUCER) is None
    a = board.add(_entry(window=(1.0, 5.0), producer=PRODUCER))
    b = board.add(_entry(window=(2.0, 5.0), producer=PRODUCER))
    board.add(_entry(kind="histogram", producer=PRODUCER))
    board.add(_entry(window=(3.0, 5.0), producer={**PRODUCER, "window": "MultifractalWindow"}))
    assert board.latest_of("canonical", PRODUCER) == b
    board.move(b, -1)                                   # the legend order is not the sending order
    assert board.latest_of("canonical", PRODUCER) == b
    board.add(_entry(window=(1.0, 5.0), producer=PRODUCER))   # identical to a: a updated, now latest
    assert board.latest_of("canonical", PRODUCER) == a and len(board) == 4
    board.replace(b, _entry(window=(2.5, 5.0), producer=PRODUCER))
    assert board.latest_of("canonical", PRODUCER) == b
    board.remove(b)
    assert board.latest_of("canonical", PRODUCER) == a
    assert board.latest_of("boxcount", PRODUCER) is None


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


def test_an_entry_with_bars_names_their_kind_in_its_record_and_tooltip():
    """Every bar spans ±1σ: the record says so and carries the entry's ``err_kind`` and
    ``err_method``; the tooltip adds them under the full name; an entry without bars says
    nothing of them."""
    entry = _entry(dD=np.full(7, 0.05), err_kind="fit SE", err_method="The slope's standard error.")
    record = entry.record()
    assert (record["bars"], record["err_kind"], record["err_method"]) == ("±1σ", "fit SE", "The slope's standard error.")
    assert entry.tooltip() == "canonical (Arneodo WTMM) · A\nbars ±1σ: fit SE\nThe slope's standard error."
    entry.show_errors = False                                   # hidden bars keep their kind on record
    assert entry.record()["errors"] is False and entry.record()["err_kind"] == "fit SE"
    bare = _entry()
    assert bare.tooltip() == bare.full_name
    assert (bare.record()["bars"], bare.record()["err_kind"], bare.record()["err_method"]) == (None, None, None)
