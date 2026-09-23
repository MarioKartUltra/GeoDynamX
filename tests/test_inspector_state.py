# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.model.inspector_state -- the floating inspector's remembered window state.

This state is a PREFERENCE, never a measurement: it lives in
``Settings.view_options["inspectors"]``, not in the project document, and every honesty rule
asserted below follows from that one fact. A missing entry means the window is closed. A corrupt
entry is DROPPED, never raised -- a preference that cannot be read must not stop the app starting.
A payload written by a later build, carrying keys this one has never heard of, still opens.

The keys of ``InspectorState.sublayers`` are ``model.provenance``'s ``raster_key`` /
``sublayer_key`` and nothing else, so the tests spell them that way
rather than as literals: the remembered row and the row itself cannot come to mean different
things if only one function ever names them.
"""
from __future__ import annotations

import json

from dynamix.model.inspector_state import (InspectorState, SubLayerState,
                                           inspectors_from_payload, inspectors_to_payload)
from dynamix.model.provenance import raster_key, sublayer_key


def _two_states() -> dict[str, InspectorState]:
    """One fully-populated inspector and one bare default -- the two ends of the shape."""
    return {
        "src0": InspectorState(
            source_id="src0", open=True, geometry=(120, 80, 640, 520), scale_idx=3,
            follow_master=False,
            sublayers={raster_key("src0"): SubLayerState(visible=False, opacity=0.4),
                       sublayer_key(2, 0): SubLayerState(),
                       sublayer_key(2, 1): SubLayerState(opacity=0.75)}),
        "src1": InspectorState(source_id="src1"),
    }


def test_missing_entry_means_closed():
    states = inspectors_from_payload({"src0": {"open": True}})
    assert set(states) == {"src0"}
    assert "src1" not in states
    # ... and what the caller falls back to for the source with no entry is a CLOSED window.
    assert InspectorState(source_id="src1").open is False
    # A settings file that has never seen an inspector, and one whose key is the wrong type.
    assert inspectors_from_payload(None) == {}
    assert inspectors_from_payload({}) == {}
    assert inspectors_from_payload(["src0"]) == {}


def test_round_trip_through_payload_is_exact():
    states = _two_states()
    assert inspectors_from_payload(inspectors_to_payload(states)) == states


def test_payload_is_json_serializable():
    from dynamix.shell.settings import load_settings, update_settings

    states = _two_states()
    payload = inspectors_to_payload(states)
    blob = json.dumps(payload)  # this is what save_settings does to view_options
    assert inspectors_from_payload(json.loads(blob)) == states

    # And Settings.view_options accepts it unchanged -- no schema edit to settings.py.
    update_settings(view_options={"graticule": True, "inspectors": payload})
    reloaded = load_settings().view_options
    assert reloaded["graticule"] is True
    assert inspectors_from_payload(reloaded["inspectors"]) == states


def test_a_malformed_entry_is_dropped_not_raised():
    blob = {"src0": {"open": True, "scale_idx": 2},
            "src1": "not an object at all",
            "src2": {"open": True, "geometry": "the whole screen"},
            "src3": {"open": True, "geometry": [10, 20]},
            "src4": {"open": True, "sublayers": ["1:0"]},
            "src5": {"open": True, "scale_idx": "halfway"},
            "src6": {"open": True, "sublayers": {sublayer_key(1, 0): {"opacity": "dim"}}}}
    states = inspectors_from_payload(blob)
    assert set(states) == {"src0"}
    assert states["src0"].scale_idx == 2


def test_follow_master_defaults_to_true():
    assert InspectorState(source_id="src0").follow_master is True
    assert inspectors_from_payload({"src0": {"open": True}})["src0"].follow_master is True
    # ... and an inspector the user took OFF the master stays off.
    assert inspectors_from_payload({"src0": {"follow_master": False}})["src0"].follow_master is False


def test_sublayer_defaults_are_visible_and_opaque():
    assert SubLayerState() == SubLayerState(visible=True, opacity=1.0)
    key = sublayer_key(7, 2)
    state = inspectors_from_payload({"src0": {"sublayers": {key: {}}}})["src0"]
    assert state.sublayers[key] == SubLayerState(visible=True, opacity=1.0)


def test_unknown_extra_keys_in_a_payload_are_ignored():
    blob = {"src0": {"open": True, "scale_idx": 1, "docked": True, "monitor": 2,
                     "sublayers": {raster_key("src0"): {"visible": False, "blend": "screen"}}}}
    states = inspectors_from_payload(blob)
    assert states["src0"].open is True and states["src0"].scale_idx == 1
    assert states["src0"].sublayers[raster_key("src0")] == SubLayerState(visible=False)
