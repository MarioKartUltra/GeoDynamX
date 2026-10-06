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
from dynamix.model.chain import Chain, DeviceRef


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


def _d(*names, rack=None, bypassed=()):
    return [{"device": n, "params": {}, "bypassed": n in bypassed, "rack": rack} for n in names]


# --------------------------------------------------------------------------- save as rack


def test_rack_from_descriptors_keeps_every_param_and_bypass(builtins):
    from dynamix.model.device import get_device, validate_params
    from dynamix.model.racks import RackStep, rack_from_descriptors

    descriptors = [
        {"device": "cdf_edges", "params": {"n_levels": 3, "show": "filtered"}, "bypassed": False,
         "rack": "CDF edges"},
        {"device": "hline_length", "params": {"min_len": 7, "max_len": 0}, "bypassed": True,
         "rack": None},
    ]
    rack = rack_from_descriptors("mine", descriptors)
    cdf = validate_params(get_device("cdf_edges"), {"n_levels": 3, "show": "filtered"})
    assert rack.name == "mine"
    assert rack.steps == (RackStep("cdf_edges", cdf, False),
                          RackStep("hline_length", {"min_len": 7, "max_len": 0}, True))
    assert rack.steps[0].params["show"] == "filtered"          # a view-only param rides along


def test_a_band_rows_own_band_and_a_routed_bus_stay_with_their_layer(builtins):
    """A band row's band_select (its _band_id) and a bus's sends name the layer the rack was
    saved from; the rack keeps the analysis, so it drops onto any other band row."""
    from dynamix.model.racks import rack_from_descriptors

    row = [{"device": "band_select", "params": {"band": 2, "_band_id": "#1"}, "bypassed": False},
           {"device": "wtmm2d", "params": {}, "bypassed": False}]
    assert [s.device for s in rack_from_descriptors("r", row).steps] == ["wtmm2d"]
    chosen = [{"device": "band_select", "params": {"band": 2, "_band_id": ""}},
              {"device": "wtmm2d", "params": {}}]
    assert [s.device for s in rack_from_descriptors("r", chosen).steps] == \
        ["band_select", "wtmm2d"]
    bus = [{"device": "bus", "params": {"_sends": '[{"path": "a.tif", "band": 1}]'}},
           {"device": "pca", "params": {}}]
    assert [s.device for s in rack_from_descriptors("r", bus).steps] == ["pca"]
    unrouted = [{"device": "bus", "params": {"_sends": "[]"}}, {"device": "pca", "params": {}}]
    assert [s.device for s in rack_from_descriptors("r", unrouted).steps] == ["bus", "pca"]


# --------------------------------------------------------------------------- split and merge


def test_merge_puts_rack_transforms_after_the_chains_and_filters_last(builtins):
    from dynamix.model.racks import merge_rack

    merged = merge_rack(_d("wtmm2d", "modulus_threshold", "hline_length"),
                        _rack("Topology cut", (("chain_topology", {}), ("chain_holder", {}))))
    assert [d["device"] for d in merged] == ["wtmm2d", "chain_topology", "modulus_threshold",
                                             "hline_length", "chain_holder"]
    assert [d["rack"] for d in merged] == [None, "Topology cut", None, None, "Topology cut"]
    Chain(tuple(DeviceRef(d["device"], d["params"]) for d in merged)).validate()


def test_merge_leaves_out_a_rack_transform_the_chain_already_runs(builtins):
    """The rack's wtmm2d would run a second WTMM; the chain's own wtmm2d, with its params, stays
    the analysis and the rack's other steps join it."""
    from dynamix.model.racks import merge_rack

    merged = merge_rack(_d("wtmm2d", "modulus_threshold"), _rack())
    assert [(d["device"], d["params"], d["rack"]) for d in merged] == [
        ("wtmm2d", {}, None), ("chain_topology", {}, "Hölder chain"),
        ("modulus_threshold", {}, None), ("chain_holder", {}, "Hölder chain")]
    bypassed = merge_rack(_d("wtmm2d", bypassed=("wtmm2d",)), _rack())   # not running: kept
    assert [(d["device"], d["bypassed"]) for d in bypassed] == [
        ("wtmm2d", True), ("wtmm2d", False), ("chain_topology", False), ("chain_holder", False)]
    assert bypassed[1]["params"] == {"n_oct": 4}
    filters = merge_rack(_d("wtmm2d", "chain_holder"), _rack())          # a filter is kept
    assert [d["device"] for d in filters] == ["wtmm2d", "chain_topology", "chain_holder",
                                              "chain_holder"]


def test_merge_onto_an_empty_chain_is_the_rack(builtins):
    from dynamix.model.racks import merge_rack

    merged = merge_rack([], _rack())
    assert [(d["device"], d["params"], d["bypassed"], d["rack"]) for d in merged] == [
        ("wtmm2d", {"n_oct": 4}, False, "Hölder chain"),
        ("chain_topology", {}, False, "Hölder chain"),
        ("chain_holder", {}, False, "Hölder chain")]


def test_a_filters_only_rack_lands_after_the_chains_filters(builtins):
    from dynamix.model.racks import merge_rack

    merged = merge_rack(_d("wtmm2d", "scale_select"),
                        _rack("Cuts", (("modulus_threshold", {"frac": 0.2}),
                                       ("chain_length", {}))))
    assert [d["device"] for d in merged] == ["wtmm2d", "scale_select", "modulus_threshold",
                                             "chain_length"]


def test_merge_keeps_bypass_flags_and_leaves_its_input_alone(builtins):
    from dynamix.model.racks import Rack, RackStep, merge_rack

    chain = _d("wtmm2d", "modulus_threshold", bypassed=("modulus_threshold",))
    before = json.dumps(chain)
    rack = Rack("R", (RackStep("mz_edges", {}), RackStep("hline_length", {}, bypassed=True)))
    merged = merge_rack(chain, rack)
    assert [(d["device"], d["bypassed"]) for d in merged] == [
        ("wtmm2d", False), ("mz_edges", False), ("modulus_threshold", True),
        ("hline_length", True)]
    assert json.dumps(chain) == before


def test_every_rack_title_names_one_contiguous_run(builtins):
    from dynamix.model.racks import contiguous_rack_titles, merge_rack

    merged = merge_rack(_d("wtmm2d", "chain_holder", rack="X"),
                        _rack("Y", (("mz_edges", {}), ("hline_length", {}))))
    assert [d["rack"] for d in contiguous_rack_titles(merged)] == ["X", "Y", "X (2)", "Y (2)"]
    taken = [{"rack": "X"}, {"rack": "X (2)"}, {"rack": "X"}]
    assert [d["rack"] for d in contiguous_rack_titles(taken)] == ["X", "X (2)", "X (3)"]
    whole = _d("wtmm2d", "chain_holder", rack="X")
    assert contiguous_rack_titles(whole) == whole
