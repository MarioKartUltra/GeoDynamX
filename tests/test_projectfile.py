# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.model.projectfile -- saving, opening and relocating a project.

The portability behaviour is the point: a project saved beside its data must open after the whole
folder is moved or sent to someone else.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import register_device
from dynamix.model.project import EXT, Project, TransectRecord
from dynamix.model.projectfile import (ProjectFormatError, ProjectSchemaError, changed_sources,
                                       missing_sources, open_project, relocate_source,
                                       resolve_source, save_project, sha256_of)
from dynamix.topology.links import ObjRef


@pytest.fixture
def _registry(clean_registry, stub_transform):
    """Register stub devices for the test."""
    register_device(stub_transform)


def _project_with_data(tmp_path):
    """A project saved beside its raster, as a user would organise it.

    Deliberately calls ``add_source`` the ordinary way, with no hash: a drift test that pre-hashes
    by hand hides a ``save_project`` that never hashes anything. The hash must come from the save.
    """
    data = tmp_path / "data"
    data.mkdir()
    raster = data / "dem.tif"
    raster.write_bytes(b"not really a geotiff, but it hashes")
    p = Project(title="T")
    s = p.add_source(str(raster))
    p.add_layer("dem", s.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    return p, tmp_path / f"proj{EXT}", raster


def test_save_and_open_round_trips(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    back = open_project(path)
    assert back.title == "T"
    assert back.layers[0].chain.steps[0].params == {"scale": 2}


def test_save_stamps_timestamps_and_version(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    back = open_project(path)
    assert back.created and back.modified and back.app_version


def test_paths_under_the_project_directory_are_stored_relative(tmp_path, _registry):
    """This is what makes a project portable -- an absolute path breaks the moment the folder
    moves or is sent to someone else."""
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    payload = json.loads(path.read_text())
    assert payload["sources"][0]["path"] == "data/dem.tif"
    assert not payload["sources"][0]["path"].startswith("/")


def test_a_moved_project_folder_still_resolves_its_sources(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    moved = tmp_path.parent / "moved_project"
    tmp_path.rename(moved)
    new_path = moved / path.name
    back = open_project(new_path)
    sid = back.layers[0].source_id
    assert resolve_source(back, sid, new_path).is_file()
    assert missing_sources(back, new_path) == []


def test_source_kind_round_trips_through_save_and_open(tmp_path, _registry):
    """A project with one raster + one point source: the point source's ``kind`` must survive a
    real save/open cycle (through ``to_payload``/``from_payload``'s JSON round trip), not just
    the in-memory dataclass round trip ``test_project.py`` already pins."""
    data = tmp_path / "data"
    data.mkdir()
    raster = data / "dem.tif"
    raster.write_bytes(b"not really a geotiff, but it hashes")
    quakes = data / "quakes.csv"
    quakes.write_text("lon,lat\n1.0,2.0\n")

    p = Project(title="T")
    raster_src = p.add_source(str(raster))
    points_src = p.add_source(str(quakes), kind="points")
    p.add_layer("dem", raster_src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    p.add_layer("quakes", points_src.source_id)

    path = tmp_path / f"proj{EXT}"
    save_project(p, path)
    back = open_project(path)

    kinds = {s.path: s.kind for s in back.sources.values()}
    assert kinds[str(raster)] == "raster"
    assert kinds[str(quakes)] == "points"


def test_a_project_file_saved_before_kind_existed_opens_every_source_as_raster(tmp_path):
    """A hand-written payload with no ``"kind"`` key anywhere (what every ``.dynamix`` file
    written before this task looks like) must still open -- every source on it as ``"raster"``,
    never a ``KeyError``."""
    payload = {
        "format": "dynamix-project", "schema": 1, "app_version": "", "created": "", "modified": "",
        "title": "old", "description": "",
        "sources": [{"source_id": "src0", "path": "/data/dem.tif"}],
        "layers": [], "topologies": [],
    }
    path = tmp_path / f"proj{EXT}"
    path.write_text(json.dumps(payload))
    back = open_project(path)
    assert back.sources["src0"].kind == "raster"


def test_symlinked_project_directory_still_stores_relative_paths(tmp_path, _registry):
    """A pytest ``tmp_path`` is already ``.resolve()``d, so ``tmp_path == tmp_path.resolve()`` and
    a bug that compares an unresolved absolute source path against a resolved project directory
    can never surface through it. A real symlink does surface it: on macOS ``/tmp`` (and any
    symlinked mount, e.g. iCloud Desktop/Documents) resolves through a ``/private/...`` prefix, so
    a project directory reached through a symlink must still recognise its own sources as living
    under it -- both when saving and after the whole tree is relocated."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to("real")  # relative target: keeps resolving correctly if the parent moves
    data = link / "data"
    data.mkdir()
    raster = data / "dem.tif"
    raster.write_bytes(b"not really a geotiff, but it hashes")

    p = Project(title="T")
    s = p.add_source(str(raster))
    p.add_layer("dem", s.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    path = link / f"proj{EXT}"
    save_project(p, path)

    payload = json.loads(path.read_text())
    assert payload["sources"][0]["path"] == "data/dem.tif"
    assert not payload["sources"][0]["path"].startswith("/")

    moved = tmp_path.parent / "moved_symlinked_project"
    tmp_path.rename(moved)
    new_path = moved / "link" / f"proj{EXT}"
    back = open_project(new_path)
    assert missing_sources(back, new_path) == []


def test_paths_outside_the_project_directory_stay_absolute(tmp_path, _registry):
    """A raster on a shared volume must not become a broken ../../.. climb."""
    outside = tmp_path.parent / "elsewhere.tif"
    outside.write_bytes(b"x")
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    p = Project()
    s = p.add_source(str(outside))
    p.add_layer("x", s.source_id, Chain(()))
    path = proj_dir / f"p{EXT}"
    save_project(p, path)
    assert json.loads(path.read_text())["sources"][0]["path"].startswith("/")
    outside.unlink()


def _save_as_fixture(tmp_path, decoy: bool):
    """A project in ``a/`` holding ``a/data/dem.tif``, and a destination ``b/`` that may or may not
    have an unrelated file sitting at the very same offset."""
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "data").mkdir(parents=True)
    (b / "data").mkdir(parents=True)
    (a / "data" / "dem.tif").write_bytes(b"THE REAL RASTER")
    if decoy:
        (b / "data" / "dem.tif").write_bytes(b"A DIFFERENT RASTER ENTIRELY")

    p = Project(title="T")
    s = p.add_source(str(a / "data" / "dem.tif"))
    p.add_layer("dem", s.source_id, Chain(()))
    return open_project(save_project(p, a / f"p{EXT}")), b / f"p{EXT}"


def test_save_as_does_not_rebind_a_source_to_a_decoy_at_the_destination(tmp_path, _registry):
    """``save_project`` with an arbitrary path IS "Save As". A stored relative path belongs to the
    directory it was READ from; re-interpreting it against the destination silently swapped the
    user's data for whatever unrelated file happened to sit at the same offset there -- with no
    missing-source warning, because something was indeed found."""
    opened, dest = _save_as_fixture(tmp_path, decoy=True)
    reread = open_project(save_project(opened, dest))
    sid = next(iter(reread.sources))
    assert resolve_source(reread, sid, dest).read_bytes() == b"THE REAL RASTER"
    assert missing_sources(reread, dest) == []
    assert changed_sources(reread, dest) == []


def test_save_as_into_an_unrelated_directory_does_not_lose_the_source(tmp_path, _registry):
    """The same bug's other face: with nothing at the destination offset the source was simply
    lost instead of being rebound to a stranger."""
    opened, dest = _save_as_fixture(tmp_path, decoy=False)
    reread = open_project(save_project(opened, dest))
    sid = next(iter(reread.sources))
    assert missing_sources(reread, dest) == []
    assert resolve_source(reread, sid, dest).read_bytes() == b"THE REAL RASTER"
    assert json.loads(dest.read_text())["sources"][0]["path"].startswith("/")


def test_reopening_and_saving_in_place_keeps_the_stored_path_relative(tmp_path, _registry):
    """Absolute in memory must not leak onto disk: an open/save cycle in the same folder has to
    round-trip back to the portable relative form, or the first re-save breaks portability."""
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    save_project(open_project(path), path)
    assert json.loads(path.read_text())["sources"][0]["path"] == "data/dem.tif"


def test_open_project_absolutises_source_paths_in_memory(tmp_path, _registry):
    """The invariant itself, stated once: relative on disk, absolute in memory."""
    p, path, raster = _project_with_data(tmp_path)
    save_project(p, path)
    assert json.loads(path.read_text())["sources"][0]["path"] == "data/dem.tif"
    back = open_project(path)
    stored = back.sources[next(iter(back.sources))].path
    assert Path(stored).is_absolute()
    assert Path(stored).read_bytes() == raster.read_bytes()


def test_missing_source_is_reported_not_raised(tmp_path, _registry):
    """Opening must succeed so the user can relocate the file -- refusing to open a project whose
    data moved is worse than opening it with a warning."""
    p, path, raster = _project_with_data(tmp_path)
    save_project(p, path)
    raster.unlink()
    back = open_project(path)
    assert missing_sources(back, path) == [back.layers[0].source_id]


def test_save_hashes_a_source_the_caller_never_hashed(tmp_path, _registry):
    """The hash must be computed BY the save. Phase 4's file dialog calls ``add_source(path)`` and
    nothing else, so a hash that only ever arrives from a caller means drift detection is inert for
    every real project."""
    p, path, raster = _project_with_data(tmp_path)
    sid = next(iter(p.sources))
    assert p.sources[sid].sha256 is None                     # the ordinary caller passed none
    save_project(p, path)
    assert p.sources[sid].sha256 == sha256_of(raster)        # stamped on the live SourceRef
    assert json.loads(path.read_text())["sources"][0]["sha256"] == sha256_of(raster)


def test_changed_source_is_detected_by_hash(tmp_path, _registry):
    """Reproducibility: if the raster changed under a saved analysis, the chain no longer
    reproduces the same figure and the user must be told. Nothing here hashes by hand -- that is
    the point, the whole path from an ordinary ``add_source`` to a drift report must work."""
    p, path, raster = _project_with_data(tmp_path)
    save_project(p, path)
    raster.write_bytes(b"different bytes entirely")
    back = open_project(path)
    assert changed_sources(back, path) == [back.layers[0].source_id]


def test_a_source_that_does_not_resolve_keeps_its_hash_and_still_saves(tmp_path, _registry):
    """An unmounted drive must not cost the user their work, nor erase the only record of what the
    data used to be."""
    p, path, raster = _project_with_data(tmp_path)
    save_project(p, path)
    sid = next(iter(p.sources))
    stamped = p.sources[sid].sha256
    assert stamped
    raster.unlink()
    save_project(p, path)                                    # must not raise
    assert p.sources[sid].sha256 == stamped
    assert json.loads(path.read_text())["sources"][0]["sha256"] == stamped


def test_re_saving_re_stamps_and_clears_a_drift_warning(tmp_path, _registry):
    """Accepting changed data is the other half of reporting it: a warning with no way to clear it
    is noise the user learns to ignore."""
    p, path, raster = _project_with_data(tmp_path)
    save_project(p, path)
    raster.write_bytes(b"deliberately regenerated raster")
    back = open_project(path)
    assert changed_sources(back, path) == [back.layers[0].source_id]
    save_project(back, path)
    assert changed_sources(open_project(path), path) == []


def test_unchanged_source_is_not_flagged(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    back = open_project(path)
    assert changed_sources(back, path) == []


def test_reopening_then_re_adding_the_same_raster_reuses_the_source(tmp_path, _registry):
    """I1: a reopened project used to hold ``data/dem.tif`` while a UI re-adding the same raster
    passes an absolute path, so the dedupe missed and the project grew a second SourceRef for one
    file -- divergent cache keys for identical data, and a relocation that fixes only half the
    layers."""
    p, path, raster = _project_with_data(tmp_path)
    save_project(p, path)
    back = open_project(path)
    assert len(back.sources) == 1
    ref = back.add_source(str(raster))                       # as a file dialog would
    assert len(back.sources) == 1
    assert ref.source_id == back.layers[0].source_id


def test_re_adding_through_a_symlinked_directory_reuses_the_source(tmp_path, _registry):
    """The same failure by another route: on macOS a file dialog can hand back ``/tmp/...`` while
    the project holds the resolved ``/private/tmp/...``. Same file, so same source."""
    real = tmp_path / "real"
    (real / "data").mkdir(parents=True)
    raster = real / "data" / "dem.tif"
    raster.write_bytes(b"raster bytes")
    link = tmp_path / "link"
    link.symlink_to("real")

    p = Project(title="T")
    s = p.add_source(str(raster))
    p.add_layer("dem", s.source_id, Chain(()))
    again = p.add_source(str(link / "data" / "dem.tif"))
    assert again.source_id == s.source_id
    assert len(p.sources) == 1


def test_add_source_fills_in_metadata_when_it_dedupes(_registry):
    """Dropping the incoming hash and label on the floor made the second call a silent no-op."""
    p = Project()
    first = p.add_source("/data/a.tif")
    assert first.sha256 is None and first.label == ""
    again = p.add_source("/data/a.tif", sha256="abc123", label="Bathymetry")
    assert again is first
    assert first.sha256 == "abc123" and first.label == "Bathymetry"


def test_created_survives_a_second_save_while_modified_advances(tmp_path, _registry, monkeypatch):
    """I2: metadata went into the payload but never back onto the Project, so ``project.created``
    stayed empty and every save stamped a fresh creation date -- a document forever one save old."""
    p, path, _ = _project_with_data(tmp_path)
    stamps = iter(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-06-01T12:00:00Z"])
    monkeypatch.setattr("dynamix.model.projectfile._now", lambda: next(stamps))

    save_project(p, path)
    created, modified = p.created, p.modified
    assert created and modified and p.app_version

    save_project(p, path)
    assert p.created == created                              # not restamped
    assert p.modified > modified                             # but the save was recorded
    assert open_project(path).created == created


def test_a_project_that_cannot_be_reopened_is_not_written(tmp_path, _registry):
    """I4: Layer.from_payload validates through Chain.from_payload, Layer.to_payload does not, so
    assigning ``layer.chain`` directly (which Phase 4 must -- there is no set_chain) then saving
    succeeded and opening raised. Save-succeeds / open-fails is a data-loss shape."""
    p, path, _ = _project_with_data(tmp_path)
    p.layers[0].chain = Chain((DeviceRef("nope", {}),))
    with pytest.raises(KeyError, match="no device"):
        save_project(p, path)
    assert not path.exists()                                 # and nothing was half-written


def test_save_rejects_an_out_of_range_param_naming_the_layer(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    p.layers[0].chain = Chain((DeviceRef("t", {"scale": 99}),))
    with pytest.raises(ValueError, match="above max"):
        save_project(p, path)
    assert not path.exists()


def test_relocate_source_repoints_and_rehashes(tmp_path, _registry):
    """I5: missing_sources and changed_sources reported problems the model gave no way to fix,
    while SourceRef's docstring promised relocating a moved file was one edit."""
    p, path, raster = _project_with_data(tmp_path)
    save_project(p, path)
    sid = next(iter(p.sources))

    moved = tmp_path / "elsewhere"
    moved.mkdir()
    raster.rename(moved / "dem.tif")
    (moved / "dem.tif").write_bytes(b"relocated AND edited on the way")

    back = open_project(path)
    assert missing_sources(back, path) == [sid]

    ref = relocate_source(back, sid, moved / "dem.tif")
    assert ref.source_id == sid
    assert missing_sources(back, path) == []
    assert changed_sources(back, path) == []                 # re-hashed, not merely re-pointed


def test_relocate_source_rejects_an_unknown_id(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    with pytest.raises(KeyError, match="no source named"):
        relocate_source(p, "src99", tmp_path)


def test_relocate_source_rejects_a_path_that_is_not_a_file(tmp_path, _registry):
    """Replacing a broken reference with another broken one is a typo, not an intention."""
    p, path, _ = _project_with_data(tmp_path)
    sid = next(iter(p.sources))
    with pytest.raises(ValueError, match="no file at"):
        relocate_source(p, sid, tmp_path / "does_not_exist.tif")


def test_opening_a_foreign_json_fails_clearly(tmp_path):
    path = tmp_path / f"x{EXT}"
    path.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ProjectFormatError, match="not a DynamiX project"):
        open_project(path)


def test_opening_a_non_json_file_fails_as_a_format_error(tmp_path):
    """One error type for 'this is not my file', whatever shape the rubbish takes. A raw
    JSONDecodeError leaks the parser at the UI, which cannot sensibly catch it."""
    path = tmp_path / f"x{EXT}"
    path.write_text("this is a GeoTIFF, not JSON at all")
    with pytest.raises(ProjectFormatError, match="not valid JSON"):
        open_project(path)


def test_opening_a_json_array_fails_as_a_format_error(tmp_path):
    """A JSON array used to raise AttributeError: 'list' object has no attribute 'get'."""
    path = tmp_path / f"x{EXT}"
    path.write_text(json.dumps([{"format": "dynamix-project"}]))
    with pytest.raises(ProjectFormatError, match="expected an object"):
        open_project(path)


def test_opening_a_json_scalar_fails_as_a_format_error(tmp_path):
    path = tmp_path / f"x{EXT}"
    path.write_text("42")
    with pytest.raises(ProjectFormatError, match="expected an object"):
        open_project(path)


def test_resolve_source_names_the_unknown_id(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    with pytest.raises(KeyError, match="no source named 'src99'; known:"):
        resolve_source(p, "src99", path)


def test_opening_a_future_schema_fails_clearly(tmp_path, _registry):
    """Refuse rather than silently misread a newer file -- a partially understood project
    reproduces the wrong figure under the right filename."""
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    payload = json.loads(path.read_text())
    payload["schema"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ProjectSchemaError, match="newer"):
        open_project(path)


#: Keys that belong to a param DECLARATION and must never reach a saved chain. A device declares
#: these; a project stores only the values a user chose.
DECLARATION_KEYS = ("soft_min", "soft_max", "units", "wrap", "choices", "label", "min", "max")


def test_a_saved_chain_carries_values_only_never_declaration_keys(tmp_path, clean_registry):
    """A project is a recipe: it stores the VALUE a user chose for each knob, never the declaration
    that generated the knob. A slider's soft range, units and label are view state. If any of them
    reached the params dict it would feed cache_key -- and retuning a slider, a purely visual
    change, would invalidate an expensive cached WTMM stack.

    Run against all three stub devices at once, because they were chosen to cover every kind that
    carries declaration-only metadata: soft bounds (StubHolder), choices (StubWavelet), and a wrap
    period with units (StubWedge).
    """
    register_builtin_devices()
    raster = tmp_path / "dem.tif"
    raster.write_bytes(b"raster")
    p = Project(title="all three")
    s = p.add_source(str(raster))
    p.add_layer("l", s.source_id, Chain((
        DeviceRef("stub_wavelet", {"n_octaves": 6, "wavelet": "morlet"}),
        DeviceRef("stub_holder", {"h_min": 0.0, "h_max": 1.0}),
        DeviceRef("stub_wedge", {"centre": 40.0, "half_width": 20.0}),
    )))
    payload = json.loads(save_project(p, tmp_path / f"decl{EXT}").read_text())

    steps = [st for layer in payload["layers"] for st in layer["chain"]["steps"]]
    assert {st["device"] for st in steps} == {"stub_wavelet", "stub_holder", "stub_wedge"}
    for st in steps:
        assert not set(DECLARATION_KEYS) & set(st["params"]), f"{st['device']}: {st['params']}"

    # Nowhere in the recipe at all, however deeply nested. Scoped to layers because a SourceRef
    # legitimately carries its own "label".
    recipe = json.dumps(payload["layers"])
    for key in DECLARATION_KEYS:
        assert f'"{key}"' not in recipe, f"declaration key {key!r} leaked into the saved recipe"


def test_save_adds_the_extension_when_omitted(tmp_path, _registry):
    p, _, _ = _project_with_data(tmp_path)
    written = save_project(p, tmp_path / "noext")
    assert written.suffix == EXT and written.is_file()


# --------------------------------------------------------------- User_links (additive)
#
# Project.user_links is a live LinkStore, serialised under an additive "user_links" key -- same
# additive-key discipline topologies already established (test_topology_project.py's own
# "predating topologies" test is the precedent this mirrors for links).


def test_user_links_round_trip(tmp_path, _registry):
    import numpy as np

    from dynamix.topology.links import ObjRef, suggest_code

    p, path, _ = _project_with_data(tmp_path)
    a, b = ObjRef("L1", "wtmm2d", "line", 0), ObjRef("L2", "mz_edges", "line", 1)
    code = suggest_code(np.array([[0, 0]]), np.array([[9, 9]]), 1.0)
    p.user_links.link(a, b, code=code, scale_first_contact=3.0)

    save_project(p, path)
    back = open_project(path)

    (link,) = back.user_links.links_for_layer("L1")
    assert link.a == a and link.b == b
    assert link.code == code
    assert link.scale_first_contact == 3.0


def test_project_without_user_links_still_opens(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    assert open_project(path).user_links.all() == []


def test_a_project_file_saved_before_user_links_existed_opens_with_an_empty_store(clean_registry):
    """A payload written before this field existed has no "user_links" key at all."""
    register_builtin_devices()
    p = Project.from_payload({
        "format": "dynamix-project", "schema": 1, "title": "old",
        "sources": [{"source_id": "src0", "path": "/tmp/dem.npz", "sha256": None, "label": ""}],
        "layers": [{"layer_id": 0, "name": "wtmm of dem", "source_id": "src0",
                    "chain": {"steps": [{"device": "wtmm2d", "params": {"n_oct": 3}}]},
                    "visible": True, "tags": {}}],
    })
    assert p.user_links.all() == []


# --------------------------------------------------------------------------- Transects
#
# Project.transects is a plain list of TransectRecord, serialised under an additive "transects"
# key -- same discipline user_links established just above (and topologies before that).


def test_transects_round_trip(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    p.transects.append(TransectRecord(transect_id=3, a=(1.5, 2.5), b=(9.0, 2.5),
                                      visible=False, buffer_px=33.0))

    save_project(p, path)
    back = open_project(path)

    (record,) = back.transects
    assert record.transect_id == 3
    assert record.a == (1.5, 2.5) and record.b == (9.0, 2.5)
    assert record.visible is False
    assert record.buffer_px == 33.0


def test_project_without_transects_still_opens(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    save_project(p, path)
    assert open_project(path).transects == []


def test_a_project_file_saved_before_transects_existed_opens_with_an_empty_list(clean_registry):
    """A payload written before this field existed has no "transects" key at all."""
    register_builtin_devices()
    p = Project.from_payload({
        "format": "dynamix-project", "schema": 1, "title": "old",
        "sources": [{"source_id": "src0", "path": "/tmp/dem.npz", "sha256": None, "label": ""}],
        "layers": [{"layer_id": 0, "name": "wtmm of dem", "source_id": "src0",
                    "chain": {"steps": [{"device": "wtmm2d", "params": {"n_oct": 3}}]},
                    "visible": True, "tags": {}}],
    })
    assert p.transects == []
    assert p.layers[0].name == "wtmm of dem"       # the rest of the legacy payload still loads


def test_rois_round_trip(tmp_path, _registry):
    """A general ROI and a drawn-for-a-layer ROI both survive a save/open cycle intact."""
    p, path, _ = _project_with_data(tmp_path)
    layer_id = p.layers[0].layer_id
    general = p.add_roi(100, 200, 64, 128, label="the 040 wedge")
    drawn = p.add_roi(4, 5, 16, 32, layer_ids=(layer_id,))
    p.layers[0].roi_id = drawn.roi_id

    save_project(p, path)
    back = open_project(path)

    first, second = back.rois
    assert (first.roi_id, second.roi_id) == (general.roi_id, drawn.roi_id)
    assert (first.row, first.col, first.h, first.w) == (100, 200, 64, 128)
    assert first.layer_ids == () and first.label == "the 040 wedge"
    assert second.layer_ids == (layer_id,)
    assert back.layers[0].roi_id == drawn.roi_id


def test_a_project_file_saved_before_rois_existed_opens_with_an_empty_list(clean_registry):
    """A payload written before ROIs existed has no "rois" key and no layer "roi_id"."""
    register_builtin_devices()
    p = Project.from_payload({
        "format": "dynamix-project", "schema": 1, "title": "old",
        "sources": [{"source_id": "src0", "path": "/tmp/dem.npz", "sha256": None, "label": ""}],
        "layers": [{"layer_id": 0, "name": "wtmm of dem", "source_id": "src0",
                    "chain": {"steps": [{"device": "wtmm2d", "params": {"n_oct": 3}}]},
                    "visible": True, "tags": {}}],
    })
    assert p.rois == []
    assert p.layers[0].roi_id is None
    assert p.layers[0].name == "wtmm of dem"       # the rest of the legacy payload still loads


# ----------------------------------------------- Camera memory, sec 10 A5
#
# A remembered viewpoint is project truth, not a per-machine preference: it survives the process
# under an additive "cameras" object keyed by ``Scene.camera_key()``.

_CAMERA_KEY = "frame:0.00,0.00,64.00,64.00"
_CAMERA = {
    "position": (1.0, 2.0, 3.0),
    "focal_point": (0.0, 0.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "parallel_scale": 42.5,
    "parallel_projection": True,
    "clipping_range": (0.5, 500.0),
}


def test_cameras_survive_a_save_open_cycle(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    p.cameras[_CAMERA_KEY] = dict(_CAMERA)

    save_project(p, path)

    on_disk = json.loads(path.read_text())
    assert _CAMERA_KEY in on_disk["cameras"]                 # the JSON names the key
    assert on_disk["cameras"][_CAMERA_KEY]["position"] == [1.0, 2.0, 3.0]

    back = open_project(path)
    assert back.cameras == {_CAMERA_KEY: _CAMERA}            # tuples, not lists, in memory


def test_two_save_cycles_of_a_camera_are_byte_identical(tmp_path, _registry):
    """The tuple/list re-tupling guard: open->save must not re-type what save->open produced.

    Only the "cameras" object is compared -- ``save_project`` re-stamps ``modified`` on every
    write by design, so the whole file cannot be byte-identical and never could be.
    """
    p, path, _ = _project_with_data(tmp_path)
    p.cameras[_CAMERA_KEY] = dict(_CAMERA)

    save_project(p, path)
    first = json.dumps(json.loads(path.read_text())["cameras"], indent=2)

    save_project(open_project(path), path)
    second = json.dumps(json.loads(path.read_text())["cameras"], indent=2)

    assert first == second


# ---------------------------------------------- Annotations, sec 10 A4
#
# A note pinned to a place in the world travels with the file, under an additive "annotations"
# list: one pinned to a POINT and one pinned to chain #7 of a layer's mz_edges transform, whose
# ``target.node_id()`` string must come back identical to the one that was saved.


def test_annotations_round_trip(tmp_path, _registry):
    p, path, _ = _project_with_data(tmp_path)
    layer_id = p.layers[0].layer_id
    pinned = p.add_annotation("the 040 wedge starts here", point=(12.5, 33.0))
    ref = ObjRef(layer_id=layer_id, transform="mz_edges", kind="line", obj_id=7)
    on_chain = p.add_annotation("chain 7 is the fault trace", target=ref)

    save_project(p, path)

    on_disk = json.loads(path.read_text())
    assert on_disk["annotations"][0]["point"] == [12.5, 33.0]
    assert on_disk["annotations"][1]["target"]["kind"] == "line"

    back = open_project(path)
    first, second = back.annotations
    assert (first.annotation_id, second.annotation_id) == (pinned.annotation_id,
                                                           on_chain.annotation_id)
    assert first.point == (12.5, 33.0) and first.target is None
    assert first.text == "the 040 wedge starts here"
    assert second.point is None
    assert second.target == ref
    assert second.target.node_id() == ref.node_id()


def test_a_project_file_saved_before_annotations_existed_opens_with_an_empty_list(clean_registry):
    """A payload written before annotations existed has no "annotations" key at all."""
    register_builtin_devices()
    p = Project.from_payload({
        "format": "dynamix-project", "schema": 1, "title": "old",
        "sources": [{"source_id": "src0", "path": "/tmp/dem.npz", "sha256": None, "label": ""}],
        "layers": [{"layer_id": 0, "name": "wtmm of dem", "source_id": "src0",
                    "chain": {"steps": [{"device": "wtmm2d", "params": {"n_oct": 3}}]},
                    "visible": True, "tags": {}}],
    })
    assert p.annotations == []
    assert p.layers[0].name == "wtmm of dem"       # the rest of the legacy payload still loads


# ------------------------------------------- The pre-v2 file gate, sec 8
#
# The four keys slice 1 added -- "rois", "cameras", "annotations" and the layer's "roi_id" -- are
# tested one at a time above, each against its own stripped payload. This is the gate rule 7
# states in its own words: ONE file written before ANY of them existed, carrying real layer,
# chain, transect and link content, opens with that content identical and every new field at its
# empty default, with no warning and no migration step.


def test_a_pre_v2_project_file_carrying_none_of_the_new_keys_opens_unchanged(tmp_path,
                                                                             _registry):
    """A project file from before slice 1: no "rois"/"cameras"/"annotations", no layer "roi_id".

    Written as a real file and read through ``open_project`` (not ``Project.from_payload``), so
    the whole read path is on trial, and read under ``catch_warnings`` so "no warning" is asserted
    rather than assumed.
    """
    data = tmp_path / "data"
    data.mkdir()
    raster = data / "dem.tif"
    raster.write_bytes(b"not really a geotiff, but it hashes")

    payload = {
        "format": "dynamix-project", "schema": 1,
        "app_version": "0.1.0", "created": "2026-08-01T09:00:00", "modified": "2026-08-01T09:30:00",
        "title": "pre-v2", "description": "written before slice 1 existed",
        "sources": [{"source_id": "src0", "path": "data/dem.tif", "sha256": None,
                     "label": "the DEM", "collapsed": False, "kind": "raster"}],
        "layers": [{"layer_id": 0, "name": "dem", "source_id": "src0",
                    "chain": {"steps": [{"device": "t", "params": {"scale": 2}}]},
                    "visible": True, "tags": {"note": "keep"}, "parent_id": None}],
        "topologies": [],
        "user_links": {
            "nodes": [
                {"node_id": "0:wtmm2d:line:0", "kind": "line", "space_dim": None,
                 "anchor": [0, "wtmm2d", "line", 0]},
                {"node_id": "0:mz_edges:line:1", "kind": "line", "space_dim": None,
                 "anchor": [0, "mz_edges", "line", 1]},
            ],
            "edges": [
                {"family": "spatial", "a": "0:wtmm2d:line:0", "b": "0:mz_edges:line:1",
                 "code": 31, "space_dim": 2, "via": None, "scale_first_contact": 3.0},
            ],
        },
        "transects": [{"transect_id": 3, "a": [1.5, 2.5], "b": [9.0, 2.5],
                       "visible": False, "buffer_px": 33.0}],
    }
    # The premise of the test, asserted rather than trusted: none of the four new keys is here.
    assert "rois" not in payload and "cameras" not in payload and "annotations" not in payload
    assert "roi_id" not in payload["layers"][0]

    path = tmp_path / f"legacy{EXT}"
    path.write_text(json.dumps(payload, indent=2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = open_project(path)
    assert caught == []

    # Every new field at its empty default -- absence is not a fault.
    assert back.rois == []
    assert back.cameras == {}
    assert back.annotations == []
    assert back.layers[0].roi_id is None

    # ...and the content that WAS in the file is identical.
    assert back.title == "pre-v2" and back.description == "written before slice 1 existed"
    assert [s.source_id for s in back.sources.values()] == ["src0"]
    assert back.sources["src0"].label == "the DEM"
    assert back.sources["src0"].path == str(raster)          # absolutised against the file's dir
    layer = back.layers[0]
    assert (layer.layer_id, layer.name, layer.source_id) == (0, "dem", "src0")
    assert [(s.device, s.params) for s in layer.chain.steps] == [("t", {"scale": 2})]
    assert layer.visible is True and layer.tags == {"note": "keep"} and layer.parent_id is None
    (transect,) = back.transects
    assert (transect.transect_id, transect.a, transect.b) == (3, (1.5, 2.5), (9.0, 2.5))
    assert transect.visible is False and transect.buffer_px == 33.0
    (link,) = back.user_links.all()
    assert link.a == ObjRef(0, "wtmm2d", "line", 0)
    assert link.b == ObjRef(0, "mz_edges", "line", 1)
    assert link.code == 31 and link.scale_first_contact == 3.0

    # No migration step: re-saving writes the new keys empty and changes nothing that was there.
    save_project(back, path)
    on_disk = json.loads(path.read_text())
    assert on_disk["rois"] == [] and on_disk["cameras"] == {} and on_disk["annotations"] == []
    assert on_disk["layers"][0]["roi_id"] is None
    assert on_disk["transects"] == payload["transects"]
