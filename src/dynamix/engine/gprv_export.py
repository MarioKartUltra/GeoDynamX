# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The ``.gprv`` writer: one provenance record (gprv 0.2) for one export of one layer.

The record holds only what produced the export (gprv 0.2 §5.1): the dataset, every chain step that
applies, in run order, with its parameters, its key and how it ran, and the export itself. A step
that does not apply on what reaches it, and every view-only or inactive parameter, stays out, so
the same export reached through two editing histories gives the same record apart from the export
time. The format and its checks are the ``gprv`` library's (GeoDynamX-Provenance, pinned in
``pyproject.toml``); it is imported only when a record is written.

An export lives in its own folder (:func:`bundle_path`): the exported file, its record
``<file>.gprv`` and, when the code that ran differs from its commit, the patch that turns the
commit into that code. Opening the exported file later finds the record beside it
(:func:`record_beside`), and a record written from that dataset cites it as the parent. A record
is never overwritten: a second export to the same name is a new record, a sibling branch from the
same dataset, and the earlier record moves aside under its own hash.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

from dynamix.code_identity import checked_identity, code_identity
from dynamix.core import fft_policy
from dynamix.engine.cache import cache_key
from dynamix.engine.resolve import _roi_prefix, source_identity, step_key
from dynamix.engine.execution import metadata, recipe_identity
from dynamix.model.chain import DeviceRef
from dynamix.model.contracts import FIELD, step_states
from dynamix.model.device import declared_outputs, get_device, is_transform, keyed_params, validate_params
from dynamix.model.origin import REPOSITORY, origin_of_device

__all__ = ["CODE_SPACE", "EXPORTS", "bundle_path", "code_identity", "record_beside",
           "record_for", "recipe_from_record", "write_record"]

CODE_SPACE = "geodynamix"
DISTRIBUTION = "geodynamix-beta"

#: The exports that write a record: the export id (``export/<id>`` is the print step's
#: identifier), the print step's description and the output name the steps before it carry.
EXPORTS = {
    "figure_png": ("Scientific figure with explicit sample, scale, fit and projection parameters.", "figure"),
    "chains_npz": ("Export of the chain product (horizontal and vertical chains) as the "
                   "EQSelect-compatible schema-v4 npz.", "chains"),
    "derived_npz": ("Export of the derived raster as an npz (values, axes, frame).", "raster"),
}

#: The libraries a step's results can depend on: import name -> distribution name.
_LIBRARIES = {"numpy": "numpy", "numba": "numba", "scipy": "scipy", "pyfftw": "pyFFTW",
              "mlx": "mlx", "pywt": "PyWavelets", "rasterio": "rasterio"}

_HALO = ("roi", "boundary", "margin", "reflected_edges", "real_frac", "margin_declared")


def bundle_path(chosen) -> Path:
    """Where an export picked as ``chosen`` is written: inside a folder named after it
    (``<dir>/<stem>/<name>``), or at ``chosen`` itself when its folder already carries that name
    (a second export into the same bundle)."""
    chosen = Path(chosen)
    if chosen.parent.name == chosen.stem:
        return chosen
    return chosen.parent / chosen.stem / chosen.name


def record_beside(path) -> Path | None:
    """The record of the file at ``path`` (``<file>.gprv`` beside it), or ``None``."""
    record = Path(str(path) + ".gprv")
    return record if record.is_file() else None


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()



def _libraries() -> dict:
    out = {"python": platform.python_version()}
    for module, dist in _LIBRARIES.items():
        if module not in sys.modules:
            out[module] = None
            continue
        try:
            out[module] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[module] = getattr(sys.modules[module], "__version__", None)
    return out


def _platform() -> str:
    return f"{platform.platform(terse=True)}-{platform.machine()}"


def _active(param, params: dict) -> bool:
    cond = param.active_when
    if not cond:
        return True
    pairs = (cond,) if isinstance(cond[0], str) else cond
    return all(params.get(name) in allowed for name, allowed in pairs)


def _value(value, *, private: bool = False):
    """A parameter value as the record keeps it: a private parameter's JSON text parsed (a
    structured value is an object, gprv 0.2 §5.4), a path by its base name (a record holds no
    absolute paths, §11)."""
    if isinstance(value, str):
        text = value.strip()
        if private and text[:1] in "[{":
            try:
                return json.loads(text)
            except ValueError:
                pass
        if os.path.isabs(value):
            return os.path.basename(value)
    return value


def _step_warnings(device, params: dict, edge: "dict | None" = None) -> dict:
    """What the step's knobs warn of (the device's ``knob_warnings``, the strip's ⚠), keyed by the
    knob in parameter order, each a list of its messages: a report's description is text or an
    object (gprv 0.2 §4.2), so the warnings group by knob as the outcome groups by component. The
    messages follow from the parameters, the configured warning levels and, where the step's
    result records one (``edge``, its ``_edge``), the parent the run read; each names the level it
    was judged against."""
    hook = getattr(device, "knob_warnings", None)
    if hook is None:
        return {}
    field = None
    if edge and edge.get("parent_shape"):
        field = types.SimpleNamespace(provenance={"full_dims": tuple(edge["parent_shape"])})
    found = hook(field, params)
    return {p.name: found[p.name].split("\n\n") for p in device.params if p.name in found}


def _padding_report(result: dict, params: dict) -> dict:
    """Spec 25's edge report for a padded or periodic run: the mode, the padding rule, the pad the
    transform used (the widest scale's), the Border cut, and each scale's own width and border."""
    padding, cut = result["_padding"], result.get("_border_cut") or {}
    scales = [float(a) for a in (result.get("scales") if result.get("scales") is not None else ())]
    borders = list(cut.get("borders") or [0] * len(scales))
    report = {"device": "wtmm2d", "mode": padding["edge"], "pad_width": padding["pad_width"],
              "pad": int(padding["pad"]), "eps": params["edge_eps"],
              "border_cut": cut.get("border_cut", "none"),
              "scales": [{"a": a, "padding": int(w), "border": int(b)}
                         for a, w, b in zip(scales, padding["widths"], borders)]}
    if cut:
        report["border_scope"] = cut["border_scope"]
    if padding.get("notes"):
        report["notes"] = list(padding["notes"])
    return report


def _recorded_params(device, params: dict) -> dict:
    """The step's parameters: every declared one except the view-only and the inactive ones,
    plus the private ones a recipe carries (``_sends``, ``_path``)."""
    declared = {p.name: p for p in device.params}
    out = {}
    for name, value in params.items():
        p = declared.get(name)
        if p is not None and (p.view or not _active(p, params)):
            continue
        out[name] = _value(value, private=name.startswith("_"))
    return out


def _dataset(project, layer) -> dict:
    source = project.sources[layer.source_id]
    path = Path(source.path) if source.path else None
    name = path.name if path else source.source_id
    sha = source.sha256 or (_sha256_file(path) if path and path.is_file() else None)
    what = f"The dataset {source.label or name}"
    if source.bands:
        what += f" (bands {', '.join(str(b) for b in source.bands)} of {name})"
    out = {"id": "s1", "description": what + "."}
    citation = {"title": name}
    if sha:
        citation["sha256"] = sha
    out["sourceCitation"] = citation
    parent = record_beside(path) if path else None
    if parent is not None and sha:
        try:
            data = parent.read_bytes()
            printed = json.loads(data)["processStep"][-1]["output"][0]["sourceCitation"]["sha256"]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            out["description"] += f" Its record {parent.name} is unreadable; it was not cited."
        else:
            if printed == sha:
                out["sourceMetadata"] = [{"title": parent.name,
                                          "sha256": hashlib.sha256(data).hexdigest()}]
            else:
                out["description"] += f" Its record {parent.name} describes other bytes."
    return out


def _label(device) -> str:
    return getattr(device, "label", "") or device.name


def _origin_component(device, code: dict, cid: str) -> dict:
    origin = origin_of_device(device)
    edition = code["commit"] or code["version"]
    return {"id": cid,
            "description": f"{_label(device)}: {origin.kind.replace('_', ' ')} code by "
                           f"{', '.join(origin.authors)}",
            "citation": {"title": _label(device),
                         "identifier": [{"code": device.name, "codeSpace": CODE_SPACE}]},
            "codeOrigin": {"kind": "repository", "source": f"git+{REPOSITORY}@{edition}",
                           "name": device.name, "version": edition,
                           "contentHash": {"algorithm": "sha256", "covers": "none"},
                           "license": origin.license, "citation": origin.citation,
                           "editable": code["patch"] is not None}}


def _fft_component(ctx: dict, cid: str) -> dict:
    backend = ctx["backend"]
    citation = {"title": {"fftw": "FFTW", "mlx": "MLX"}.get(backend["name"], backend["name"]),
                "edition": backend["version"]}
    if backend.get("binding"):
        citation["otherCitationDetails"] = [backend["binding"]]
    return {"id": cid, "description": "the FFT backend", "citation": citation}


def record_for(project, layer, export_path, *, export: str, layer_input: str = FIELD,
               racks=None, result=None, print_params: dict | None = None,
               code: dict | None = None, patch_name: str | None = None) -> dict:
    """The record (a dict) for ``layer``'s export written at ``export_path``.

    ``export`` names the export (:data:`EXPORTS`); ``layer_input`` is the contract the chain starts
    from (which steps apply); ``racks`` gives, per chain step, the name of the rack it came from or
    ``None``; ``result`` is the resolved result (its region stamps go into the outcome);
    ``print_params`` are the export's own parameters (the level shown, the marked levels);
    ``code`` is :func:`code_identity` (asked when omitted); ``patch_name`` is the base name of the
    patch file when the code differs from its commit."""
    description, output = EXPORTS[export]
    export_path = Path(export_path)
    code = code if code is not None else code_identity()
    captured = metadata(result).get("_execution")
    if captured is None:
        # A recipe-only draft has no computation stamps. Load its declared
        # backend metadata before listing libraries so the draft is stable.
        for ref in layer.chain.steps:
            p = validate_params(get_device(ref.device), ref.params)
            if "precision" in p:
                fft_policy.context(p["precision"])
    libraries, plat = _libraries(), _platform()
    if captured:
        libraries = {key: value for entry in captured
                     for key, value in entry["context"]["libraries"].items()}
        plat = captured[0]["context"]["platform"]
    steps = list(layer.chain.steps)
    racks = list(racks) if racks is not None else [None] * len(steps)
    if len(racks) != len(steps):
        raise ValueError("one rack entry per recipe step is required")
    landed = metadata(result).get("_landed", {})
    states = step_states(steps, layer_input, level_view=landed.get("level") is not None)
    applying = [(ref, racks[i]) for i, (ref, st) in enumerate(zip(steps, states)) if st.applies]
    region = _roi_prefix(layer, [ref for ref, _rack in applying])
    sid = source_identity(layer)

    sources = [_dataset(project, layer)]
    reads = ["s1"]
    patch_source = []
    if code["patch"] is not None and patch_name:
        sources.append({"id": "patch", "scope": {"level": "software"},
                        "description": "Uncommitted changes to the GeoDynamix source: git apply "
                                       "turns the cited commit into the code that ran.",
                        "sourceCitation": {"title": patch_name,
                                           "sha256": hashlib.sha256(code["patch"]).hexdigest()}})
        patch_source = ["patch"]
    details = [f"python {libraries['python']}"] + [
        f"{name} {v}" for name, v in libraries.items() if name != "python" and v]
    if not code["checked"]:
        details.append("working tree not checked against the commit (git unavailable)")
    software = {"id": "gdx", "title": "GeoDynamix", "edition": code["commit"] or code["version"]
                or "unknown", "onlineResource": [{"linkage": REPOSITORY}],
                "otherCitationDetails": details}

    evidence = iter(metadata(result).get("_execution", []))
    record_steps, defined, upstream = [], set(), None
    for k, (ref, rack) in enumerate(applying, start=1):
        device = get_device(ref.device)
        params = validate_params(device, ref.params)
        run = _recorded_params(device, params)
        own = next(evidence, None) if is_transform(device) else None
        if own is not None and own["device"] != device.name:
            raise ValueError("landed execution steps differ from the export recipe")
        algorithm = []
        context = dict(own["context"]) if own else {"libraries": libraries, "platform": plat}
        fft_runs = own["fft"] if own else ([fft_policy.context(params["precision"])]
                                           if "precision" in params else [])
        if own is None and fft_runs:
            context.update(fft_runs[0])
        for ctx in fft_runs:
            suffix = f"_{ctx['precision']}" if len(fft_runs) > 1 else ""
            cid = f"fft_{ctx['backend']['name']}{suffix}"
            algorithm.append(cid if cid in defined else _fft_component(ctx, cid))
            defined.add(cid)
            run[cid] = {"precision": ctx["precision"], "planner": ctx["backend"]["planner"]}
        origin = origin_of_device(device)
        if origin.kind != "first_party" or origin.citation:
            cid = f"code_{device.name}"
            algorithm.append(cid if cid in defined else _origin_component(device, code, cid))
            defined.add(cid)
            if origin.kind != "first_party":
                context["thirdParty"] = [{"component": cid}]
        outcome = {}
        if is_transform(device):
            upstream = step_key(device, sid, params, upstream)
            outcome["cache_key"] = upstream
        if region and k == len(region):
            r, c, h, w = (int(v) for v in layer.tags["roi.window"].split(","))
            from dynamix.roi.runner import roi_margin_of

            stamps = (result or {}).get("_roi") if isinstance(result, dict) else None
            margin = (stamps or {}).get("margin")         # the halo the run read, when it says
            run["roi"] = {"window": [r, c, h, w],
                          "margin": int(margin) if margin is not None
                          else sum(roi_margin_of(d, p)[0] for d, p in region)}
            if stamps:
                outcome["roi"] = {key: stamps[key] for key in _HALO if key in stamps}
        processing = {"identifier": {"code": f"{device.name}/{output}", "codeSpace": CODE_SPACE},
                      "softwareReference": software if k == 1 else "gdx",
                      "runTimeParameters": run}
        if algorithm:
            processing["algorithm"] = algorithm
        text = f"{_label(device)} ({device.name})"
        if rack:
            text += f', from the rack "{rack}"'
        step = {"id": f"p{k}", "description": text + ".", "source": reads + patch_source,
                "processingInformation": processing, "executionContext": context,
                "output": [{"id": f"d{k}"}]}
        reports = [{"name": "outcome", "description": outcome}] if outcome else []
        if own and own.get("outcome"):
            reports.append({"name": "method", "description": own["outcome"]})
        edge = (result or {}).get("_edge") if isinstance(result, dict) else None
        if not (isinstance(edge, dict) and edge.get("device") == device.name):
            edge = None
        warned = _step_warnings(device, params, edge)
        if warned:
            reports.append({"name": "warning", "description": warned})
        if edge is not None:
            # Spec 22: what filled each scale's reach -- the halo, the whole parent, the
            # padding, the cut border -- and the reach itself with the grids it was measured on.
            reports.append({"name": "edge", "description": json.loads(json.dumps(edge))})
        elif (device.name == "wtmm2d" and isinstance(result, dict)
              and result.get("_padding") is not None):
            reports.append({"name": "edge", "description": _padding_report(result, params)})
        if reports:
            step["report"] = reports
        record_steps.append(step)
        reads = [f"d{k}"]

    printing = {"identifier": {"code": f"export/{export}", "codeSpace": CODE_SPACE},
                "softwareReference": software if not record_steps else "gdx"}
    print_params = dict(print_params or {})
    product = metadata(result).get("_output_identity")
    if product:
        owner = metadata(result)["_execution"][-1]["device"]
        print_params.setdefault("output", f"{owner}/{product['name']}")
        if product["level"] is not None:
            print_params.setdefault("level", product["level"])
        print_params.update(product.get("params", {}))
    if landed.get("level") is not None:
        print_params["shown_level"] = landed["level"]
    if print_params:
        printing["runTimeParameters"] = {key: _value(v) for key, v in print_params.items()}
    print_context = {"libraries": libraries, "platform": plat}
    output_runs = metadata(result).get("_output_execution") or []
    if output_runs:
        own = output_runs[-1]
        print_context = dict(own["context"])
        algorithm = []
        values = printing.setdefault("runTimeParameters", {})
        for ctx in own["fft"]:
            cid = f"output_fft_{ctx['backend']['name']}_{ctx['precision']}"
            algorithm.append(_fft_component(ctx, cid))
            values[cid] = {"precision": ctx["precision"], "planner": ctx["backend"]["planner"]}
        if algorithm:
            printing["algorithm"] = algorithm
    record_steps.append({
        "id": "print", "description": description,
        "stepDateTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": reads, "processingInformation": printing,
        "executionContext": print_context,
        "output": [{"id": "o1", "sourceCitation": {"title": export_path.name,
                                                   "sha256": _sha256_file(export_path)}}]})
    dataset = sources[0]["sourceCitation"]["title"]
    return {"gprv": "0.2", "source": sources, "processStep": record_steps,
            "statement": f'{export_path.name}: the {output} of the layer "{layer.name}" over '
                         f"{dataset}, computed and exported by GeoDynamix."}


def write_record(project, layer, export_path, **kw) -> Path:
    """Write :func:`record_for`'s record as ``<export>.gprv`` beside the export, checked by the
    gprv library first (it refuses a record that breaks the format), and the patch beside it
    when the code differs from its commit. An earlier record at that name moves aside to
    ``<export>.<hash12>.gprv``; it is never rewritten. Returns the record's path."""
    import gprv

    export_path = Path(export_path)
    result = kw.get("result")
    proof = metadata(result).get("_landed")
    if proof is None:
        raise ValueError("no landed computation identity; run this recipe before writing provenance")
    level = kw.pop("level", proof.get("level"))
    expected = recipe_identity(layer, kw.get("layer_input", FIELD), level=level)
    if proof != expected:
        raise ValueError("current recipe differs from the landed computation; run it before writing provenance")
    if expected["analysis_key"] is not None and not metadata(result).get("_execution"):
        raise ValueError("no compute-time execution context; run this recipe before writing provenance")
    if isinstance(result, dict) and result.get("_preview"):
        raise ValueError("the landed result is a preview; wait for the full computation")
    product = metadata(result).get("_output_identity")
    if product:
        from dynamix.engine.resolve import output_key, selection_recipe
        states = step_states(layer.chain.steps, kw.get("layer_input", FIELD), level_view=level is not None)
        refs = [ref for ref, state in zip(layer.chain.steps, states)
                if state.applies and is_transform(get_device(ref.device))]
        ref = refs[-1]
        device = get_device(ref.device)
        params = validate_params(device, ref.params)
        output = next((o for o in declared_outputs(device) if o.name == product["name"] and o.lazy), None)
        if output is None:
            raise ValueError("landed lazy output is not declared by the current analysis")
        selection = selection_recipe(layer, kw.get("layer_input", FIELD)) if output.selects else None
        actual_key = output_key(device.name, output, params, expected["analysis_key"], selection,
                                level=product["level"])
        if (product["key"] != actual_key or (output.per_level and level is not None
                                             and product["level"] != level)):
            raise ValueError("selected output differs from the landed raster; compute it before export")
        declared_level = (kw.get("print_params") or {}).get("level", product["level"])
        if declared_level != product["level"]:
            raise ValueError("export level differs from the landed output")
        declared_product = (kw.get("print_params") or {}).get("output")
        if declared_product and declared_product != f"{device.name}/{output.name}":
            raise ValueError("export label differs from the landed output")
    actual_code = checked_identity()
    code = kw.pop("code", None) or actual_code
    if code != actual_code:
        raise ValueError("supplied code identity differs from process startup")
    patch_name = None
    if code["patch"] is not None:
        patch_name = f"geodynamix-{hashlib.sha256(code['patch']).hexdigest()[:12]}.patch"
        patch = export_path.parent / patch_name
        if not patch.exists():
            patch.write_bytes(code["patch"])
    record = record_for(project, layer, export_path, code=code, patch_name=patch_name, **kw)
    problems = gprv.check_obj(gprv.normalize(record))
    if problems:
        raise gprv.RecordError(problems)
    target = Path(str(export_path) + ".gprv")
    if target.exists():
        old = target.read_bytes()
        aside = target.with_name(f"{export_path.name}.{gprv.sha256(old)[:12]}.gprv")
        if aside.exists() and aside.read_bytes() == old:
            target.unlink()                          # the same bytes are already kept aside
        else:
            target.rename(aside)
    gprv.write(target, record)
    return target


def recipe_from_record(record: dict) -> list[DeviceRef]:
    """The chain a record's steps ran, to replay it: one :class:`DeviceRef` per GeoDynamix step
    before the export, with its recorded parameters (the components' own groups and the region
    window left out)."""
    components = {a["id"] for st in record["processStep"]
                  for a in (st.get("processingInformation") or {}).get("algorithm", [])
                  if isinstance(a, dict)}
    out = []
    for st in record["processStep"][:-1]:
        info = st.get("processingInformation") or {}
        ident = info.get("identifier") or {}
        if ident.get("codeSpace") != CODE_SPACE:
            continue
        device = ident["code"].split("/", 1)[0]
        params = {k: (json.dumps(v) if k.startswith("_") and isinstance(v, (dict, list)) else v)
                  for k, v in (info.get("runTimeParameters") or {}).items()
                  if k not in components and k != "roi"}
        out.append(DeviceRef(device, params))
    return out
