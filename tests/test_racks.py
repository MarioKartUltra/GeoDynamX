# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The rack file store and the rack drop rule (dynamix.model.racks): one JSON file per rack,
a bad file listed with its reason, split-and-merge onto a chain, and the applicability of a rack
on a layer."""
from __future__ import annotations

import json
from importlib import metadata

import pytest

from dynamix.devices import register_builtin_devices


@pytest.fixture
def builtins(clean_registry):
    register_builtin_devices()
    return clean_registry


@pytest.fixture
def store(tmp_path):
    """A rack directory of its own: the suite's settings file sits in ``tmp_path`` itself, as
    the real one sits beside ``racks/``."""
    return tmp_path / "racks"


def _rack(name="Hölder chain", steps=(("wtmm2d", {"n_oct": 4}), ("chain_topology", {}),
                                      ("chain_holder", {}))):
    from dynamix.model.racks import Rack, RackStep

    return Rack(name, tuple(RackStep(d, dict(p)) for d, p in steps))


# --------------------------------------------------------------------------- the file format


def test_payload_round_trips_through_json(builtins):
    from dynamix.model.racks import (RACK_FORMAT, RACK_SCHEMA, Rack, RackStep, rack_from_payload,
                                     rack_to_payload)

    rack = Rack("WTMM + Hölder mine",
                (RackStep("wtmm2d", {"n_oct": 3}), RackStep("chain_holder", {}, bypassed=True)),
                description="mine", created="2026-10-05T12:00:00+00:00", app_version="0.1")
    payload = json.loads(json.dumps(rack_to_payload(rack)))
    assert payload["format"] == RACK_FORMAT and payload["schema"] == RACK_SCHEMA
    assert payload["steps"][1] == {"device": "chain_holder", "params": {}, "bypassed": True}
    assert rack_from_payload(payload) == rack


def test_save_then_load_gives_the_rack_back_stamped(builtins, store):
    from dynamix.model.racks import load_racks, save_rack

    rack = _rack()
    path = save_rack(store, rack)
    assert path == store / "holder-chain.json"
    (entry,) = load_racks(store)
    assert entry.problem == "" and entry.path == path
    assert entry.rack.name == rack.name and entry.rack.steps == rack.steps
    assert entry.rack.created.endswith("+00:00")
    assert entry.rack.app_version == metadata.version("geodynamix-beta")
    assert entry.rack.builtin is False


def test_a_given_created_stamp_is_kept(builtins, store):
    import dataclasses

    from dynamix.model.racks import load_racks, save_rack

    save_rack(store, dataclasses.replace(_rack(), created="2026-01-01T00:00:00+00:00"))
    assert load_racks(store)[0].rack.created == "2026-01-01T00:00:00+00:00"


def test_an_unknown_device_or_param_is_listed_with_the_reason(builtins, store):
    from dynamix.model.racks import RACK_FORMAT, load_racks

    store.mkdir()

    def write(stem, steps):
        (store / f"{stem}.json").write_text(json.dumps(
            {"format": RACK_FORMAT, "schema": 1, "name": stem.upper(), "steps": steps}))

    write("a", [{"device": "no_such_device", "params": {}}])
    write("b", [{"device": "wtmm2d", "params": {"bogus": 1}}])
    write("c", [{"device": "chain_holder", "params": {}}, {"device": "wtmm2d", "params": {}}])
    (store / "d.json").write_text("{not json")
    write("e", [{"device": "wtmm2d", "params": {}}])
    (store / "f.json").write_text(json.dumps(
        {"format": RACK_FORMAT, "schema": 2, "name": "F", "steps": []}))

    by_file = {e.path.stem: e for e in load_racks(store)}
    assert by_file["a"].rack.name == "A" and "no_such_device" in by_file["a"].problem
    assert "unknown parameter" in by_file["b"].problem
    assert "transforms-then-filters" in by_file["c"].problem
    assert by_file["d"].rack is None and by_file["d"].problem.startswith("d.json")
    assert by_file["e"].problem == ""
    assert by_file["f"].rack is None and "newer than this build" in by_file["f"].problem


def test_a_missing_directory_holds_no_racks(tmp_path):
    from dynamix.model.racks import load_racks

    assert load_racks(tmp_path / "nowhere") == []


def test_save_refuses_a_file_holding_a_different_rack(builtins, store):
    from dynamix.model.racks import load_racks, save_rack

    save_rack(store, _rack("A b"))
    with pytest.raises(FileExistsError, match="'A b'"):
        save_rack(store, _rack("a-b"))                    # same file name, other rack
    save_rack(store, _rack("A b", steps=(("mz_edges", {}),)))   # the same rack: replaced
    (entry,) = load_racks(store)
    assert entry.rack.name == "A b" and [s.device for s in entry.rack.steps] == ["mz_edges"]


def test_delete_removes_the_racks_file(builtins, store):
    from dynamix.model.racks import delete_rack, load_racks, save_rack

    save_rack(store, _rack("One"))
    save_rack(store, _rack("Two"))
    delete_rack(store, "One")
    assert [e.rack.name for e in load_racks(store)] == ["Two"]
    with pytest.raises(KeyError):
        delete_rack(store, "One")


def test_builtin_racks_are_the_presets_read_only(builtins):
    from dynamix.model.presets import PRESETS
    from dynamix.model.racks import builtin_racks

    racks = builtin_racks()
    assert [r.name for r in racks] == list(PRESETS)
    assert all(r.builtin for r in racks)
    for rack in racks:
        assert [(s.device, s.params) for s in rack.steps] == \
            [(d, dict(p)) for d, p in PRESETS[rack.name]]
