# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The .gprv writer: a record for an export, assembled from the project, the layer's chain and the
engine's keys, checked by the pinned gprv library and written beside the export.

A record holds only what produced the print (gprv 0.2 §5.1): the dataset, the applying steps with
their parameters and how they ran, and the export. Replaying the recorded recipe reproduces the
exported bytes. A record is never overwritten: a second export to the same name is a new record,
a sibling branch from the same dataset, and the earlier one moves aside under its own hash.
"""
from __future__ import annotations

import hashlib
import json
import subprocess

import numpy as np
import pytest

gprv = pytest.importorskip("gprv")
rasterio = pytest.importorskip("rasterio")

from dynamix.core import fft_policy  # noqa: E402
from dynamix.devices import register_builtin_devices  # noqa: E402
from dynamix.engine.cache import Cache  # noqa: E402
from dynamix.engine.resolve import resolve  # noqa: E402
from dynamix.model.chain import Chain, DeviceRef  # noqa: E402
from dynamix.model.project import Project  # noqa: E402


@pytest.fixture
def builtin(clean_registry):
    register_builtin_devices()
    return clean_registry


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _dem(path, n=64, seed=3):
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, n)).cumsum(0).cumsum(1).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:32611", transform=from_origin(500000, 4000000, 30, 30)) as ds:
        ds.write(v, 1)
    return path


def _project(tmp_path, steps):
    from dynamix.core.ingest import load_grid

    dem = _dem(tmp_path / "dem.tif")
    project = Project()
    src = project.add_source(str(dem), sha256=_sha(dem))
    layer = project.add_layer("dem", src.source_id, Chain(tuple(DeviceRef(n, dict(p))
                                                                 for n, p in steps)))
    return project, layer, load_grid(dem)


def _export_chains(path, result, field):
    """The shell's chain export (``MainWindow._export_chain_product``), headless."""
    from dynamix.core.chain_product import export_chain_product_npz

    params = json.loads(json.dumps(result.get("params") or {}, default=list))
    export_chain_product_npz(path, result["chain_product"], field=field, params=params)


_WTMM = ("wtmm2d", {"n_oct": 2, "n_voice": 4, "precision": "f64"})


def test_figure_record_keeps_sample_population_and_manual_endpoints(builtin, tmp_path):
    from dynamix.engine.gprv_export import write_record
    fft_policy.configure("fftw",64)
    project,layer,field = _project(tmp_path,[_WTMM])
    result = resolve(layer,field,Cache()).result
    out = tmp_path / "figure.png"
    out.write_bytes(b"figure render")
    recipe = {"population":"wtmm", "projection":"schmidt", "scales":[1,2,4],
              "fit":{"method":"manual", "endpoints":[[1,5],[3,1]], "slope":-2}}
    path = write_record(project,layer,out,export="figure_png",result=result,
                        print_params={"figure":recipe})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert gprv.check(path.read_bytes()) == []
    assert record["processStep"][-1]["processingInformation"]["runTimeParameters"]["figure"] == recipe


def _strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def _no_time(record: dict) -> dict:
    out = json.loads(json.dumps(record))
    out["processStep"][-1].pop("stepDateTime", None)
    return out


def test_a_chain_export_writes_a_checked_record_beside_it(builtin, tmp_path):
    from dynamix.engine.gprv_export import write_record

    fft_policy.configure("fftw", 64)
    project, layer, field = _project(tmp_path, [_WTMM, ("scale_select", {"scale_idx": 0})])
    result = resolve(layer, field, Cache()).result
    out = tmp_path / "chains.npz"
    _export_chains(out, result, field)
    path = write_record(project, layer, out, export="chains_npz", result=result)

    assert path == tmp_path / "chains.npz.gprv"
    data = path.read_bytes()
    assert gprv.check(data) == []
    record = json.loads(data)
    jsonschema = pytest.importorskip("jsonschema")

    jsonschema.validate(record, gprv.schema(record["gprv"]))
    assert record["gprv"] == "0.2"
    source = next(s for s in record["source"] if s["id"] == "s1")
    assert source["sourceCitation"] == {"title": "dem.tif", "sha256": _sha(tmp_path / "dem.tif")}
    steps = record["processStep"]
    assert [s["processingInformation"]["identifier"]["code"] for s in steps] == [
        "wtmm2d/chains", "scale_select/chains", "export/chains_npz"]
    wtmm = steps[0]
    params = wtmm["processingInformation"]["runTimeParameters"]
    assert params["n_oct"] == 2 and params["precision"] == "f64" and "show" not in params
    ctx = wtmm["executionContext"]
    assert ctx["precision"] == "float64" and ctx["backend"]["name"] == "fftw"
    (third,) = ctx["thirdParty"]
    origin = next(a for a in wtmm["processingInformation"]["algorithm"]
                  if isinstance(a, dict) and a["id"] == third["component"])["codeOrigin"]
    assert origin["license"] == "GPL-2.0-only" and "xsmurf" in origin["citation"]
    software = wtmm["processingInformation"]["softwareReference"]
    assert software["title"] == "GeoDynamix" and software["onlineResource"][0]["linkage"].endswith(
        "GeoDynamX")
    printed = steps[-1]
    assert printed["output"][0]["sourceCitation"] == {"title": "chains.npz", "sha256": _sha(out)}
    assert printed["source"] == [steps[1]["output"][0]["id"]]
    assert not [s for s in _strings(record) if s.startswith(str(tmp_path))], "absolute paths"


def test_replaying_the_recorded_recipe_reproduces_the_export(builtin, tmp_path):
    from dynamix.engine.gprv_export import recipe_from_record, write_record

    fft_policy.configure("fftw", 64)
    project, layer, field = _project(tmp_path, [_WTMM])
    result = resolve(layer, field, Cache()).result
    out = tmp_path / "chains.npz"
    _export_chains(out, result, field)
    record = json.loads(write_record(project, layer, out, export="chains_npz",
                                     result=result).read_bytes())

    again = project.add_layer("replay", layer.source_id, Chain(tuple(recipe_from_record(record))))
    replay = tmp_path / "replay.npz"
    rendered = resolve(again, field, Cache())
    _export_chains(replay, rendered.result, field)
    assert _sha(replay) == record["processStep"][-1]["output"][0]["sourceCitation"]["sha256"]
    outcome = next(r for r in record["processStep"][0]["report"] if r["name"] == "outcome")
    assert rendered.analysis_key == outcome["description"]["cache_key"]


def test_a_view_param_leaves_the_record_and_a_keyed_param_changes_it(builtin, tmp_path):
    from dynamix.engine.gprv_export import record_for

    project, layer, field = _project(tmp_path, [_WTMM, ("scale_select", {"scale_idx": 0})])
    out = tmp_path / "chains.npz"
    out.write_bytes(b"stand-in export")
    base = _no_time(record_for(project, layer, out, export="chains_npz"))

    layer.chain = Chain((DeviceRef("wtmm2d", {**_WTMM[1], "show": "modulus"}),
                         DeviceRef("scale_select", {"scale_idx": 0})))
    assert _no_time(record_for(project, layer, out, export="chains_npz")) == base

    layer.chain = Chain((DeviceRef("wtmm2d", {**_WTMM[1], "n_voice": 6}),
                         DeviceRef("scale_select", {"scale_idx": 0})))
    assert _no_time(record_for(project, layer, out, export="chains_npz")) != base


def test_a_step_that_does_not_apply_is_left_out(builtin, tmp_path):
    """gprv 0.2 §5.1: only what produced the print. A chain filter after a Hölder map does not
    apply (no V-chains reach it), so adding it leaves the record as it was."""
    from dynamix.engine.gprv_export import record_for
    from dynamix.model.contracts import step_states

    project, layer, field = _project(tmp_path, [("holder_map", {})])
    out = tmp_path / "hmap.npz"
    out.write_bytes(b"stand-in export")
    base = _no_time(record_for(project, layer, out, export="derived_npz"))
    layer.chain = Chain((DeviceRef("holder_map", {}), DeviceRef("chain_length", {})))
    assert not step_states(layer.chain.steps, "Field")[1].applies
    assert _no_time(record_for(project, layer, out, export="derived_npz")) == base


def test_a_region_layer_records_its_window_and_halo(builtin, tmp_path):
    from dynamix.engine.gprv_export import record_for

    project, layer, field = _project(tmp_path, [("holder_map", {})])
    layer.tags["roi.window"] = "8,8,32,32"
    result = resolve(layer, field, Cache()).result
    out = tmp_path / "hmap.npz"
    out.write_bytes(b"stand-in export")
    record = record_for(project, layer, out, export="derived_npz", result=result)
    step = record["processStep"][0]
    roi = step["processingInformation"]["runTimeParameters"]["roi"]
    assert roi["window"] == [8, 8, 32, 32] and roi["margin"] >= 0
    outcome = next(r["description"] for r in step["report"] if r["name"] == "outcome")
    halo = outcome["roi"]
    assert set(halo) >= {"reflected_edges", "real_frac"}


def test_a_rack_is_recorded_by_name_with_its_steps(builtin, tmp_path):
    from dynamix.engine.gprv_export import record_for

    project, layer, field = _project(tmp_path, [_WTMM, ("scale_select", {"scale_idx": 0})])
    out = tmp_path / "chains.npz"
    out.write_bytes(b"stand-in export")
    record = record_for(project, layer, out, export="chains_npz", racks=[None, "Fine scales"])
    steps = record["processStep"]
    assert "Fine scales" not in steps[0]["description"]
    assert 'rack "Fine scales"' in steps[1]["description"]


def test_a_second_export_is_a_new_record_and_the_first_moves_aside(builtin, tmp_path):
    from dynamix.engine.gprv_export import write_record

    project, layer, field = _project(tmp_path, [_WTMM])
    out = tmp_path / "chains.npz"
    out.write_bytes(b"first export")
    result = resolve(layer, field, Cache()).result
    first = write_record(project, layer, out, export="chains_npz", result=result)
    first_bytes = first.read_bytes()
    out.write_bytes(b"second export")
    second = write_record(project, layer, out, export="chains_npz", result=result)
    assert second == first and gprv.check(second.read_bytes()) == []
    aside = tmp_path / f"chains.npz.{gprv.sha256(first_bytes)[:12]}.gprv"
    assert aside.read_bytes() == first_bytes                    # kept, never rewritten
    a, b = json.loads(first_bytes), json.loads(second.read_bytes())
    assert a["source"] == b["source"]                           # siblings from one dataset


def test_a_dataset_with_a_record_is_cited_as_the_parent(builtin, tmp_path):
    from dynamix.engine.gprv_export import record_for, write_record

    project, layer, field = _project(tmp_path, [_WTMM])
    out = tmp_path / "chains.npz"
    out.write_bytes(b"an export")
    write_record(project, layer, out, export="chains_npz",
                 result=resolve(layer, field, Cache()).result)      # chains.npz.gprv
    child = project.add_source(str(out), sha256=_sha(out))
    child_layer = project.add_layer("child", child.source_id, Chain((DeviceRef(*_WTMM),)))
    out2 = tmp_path / "child.npz"
    out2.write_bytes(b"a child export")
    record = record_for(project, child_layer, out2, export="chains_npz")
    source = next(s for s in record["source"] if s["id"] == "s1")
    parent = (tmp_path / "chains.npz.gprv").read_bytes()
    assert source["sourceMetadata"] == [{"title": "chains.npz.gprv", "sha256": gprv.sha256(parent)}]


def test_uncommitted_source_changes_ride_as_a_patch(tmp_path):
    from dynamix.engine.gprv_export import code_identity

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "c"], check=True)
    clean = code_identity(repo)
    assert len(clean["commit"]) == 40 and clean["patch"] is None and clean["checked"]
    (repo / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "src" / "b.py").write_text("y = 3\n", encoding="utf-8")
    dirty = code_identity(repo)
    assert dirty["commit"] == clean["commit"]
    assert b"+x = 2" in dirty["patch"] and b"+y = 3" in dirty["patch"]


def test_an_export_gets_its_own_folder_and_its_record_is_found_beside_it(tmp_path):
    """Like a shapefile's sidecars: the export, its record and its patch share a folder named
    after the export, and opening the exported file finds the record by name."""
    from dynamix.engine.gprv_export import bundle_path, record_beside

    assert bundle_path(tmp_path / "chains.npz") == tmp_path / "chains" / "chains.npz"
    assert bundle_path(tmp_path / "chains" / "chains.npz") == tmp_path / "chains" / "chains.npz"
    exported = tmp_path / "x.npz"
    exported.write_bytes(b"x")
    assert record_beside(exported) is None
    (tmp_path / "x.npz.gprv").write_bytes(b"{}")
    assert record_beside(exported) == tmp_path / "x.npz.gprv"
