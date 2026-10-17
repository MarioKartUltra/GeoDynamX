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
from datetime import datetime, timezone
from pathlib import Path

from dynamix.core import fft_policy
from dynamix.engine.cache import cache_key
from dynamix.engine.resolve import _roi_prefix, source_identity
from dynamix.model.chain import DeviceRef
from dynamix.model.contracts import FIELD, step_states
from dynamix.model.device import get_device, is_transform, keyed_params, validate_params
from dynamix.model.origin import REPOSITORY, origin_of_device

__all__ = ["CODE_SPACE", "EXPORTS", "bundle_path", "code_identity", "record_beside",
           "record_for", "recipe_from_record", "write_record"]

CODE_SPACE = "geodynamix"
DISTRIBUTION = "geodynamix-beta"

#: The exports that write a record: the export id (``export/<id>`` is the print step's
#: identifier), the print step's description and the output name the steps before it carry.
EXPORTS = {
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


def _git_dirs(root: Path) -> tuple[Path, Path] | None:
    """``(gitdir, commondir)`` of the checkout at ``root`` (a worktree's ``.git`` is a file)."""
    dot = root / ".git"
    if dot.is_dir():
        return dot, dot
    if dot.is_file():
        text = dot.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = (root / text.split(":", 1)[1].strip()).resolve()
            common = gitdir
            if (gitdir / "commondir").is_file():
                common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
            return gitdir, common
    return None


def _head_commit(gitdir: Path, common: Path) -> str | None:
    """The commit at HEAD, read from the files (no git process)."""
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head or None
    ref = head.split(":", 1)[1].strip()
    for base in (gitdir, common):
        if (base / ref).is_file():
            return (base / ref).read_text(encoding="utf-8").strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    return None


def _git(root: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30)


def code_identity(root=None) -> dict:
    """The code that runs: ``{"commit", "version", "patch", "checked"}``.

    ``root`` defaults to the checkout this module runs from. ``commit`` is read from the git files;
    ``None`` for an install without a checkout, where ``version`` (the distribution's) names the
    code. ``patch``: the bytes that ``git apply`` turns the commit into the source that ran
    (``src/``, untracked files included), or ``None`` when it matches the commit. ``checked`` is
    False when git could not be asked (not installed, timed out), so the record says the working
    tree was not checked."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    try:
        version = importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = None
    dirs = _git_dirs(root)
    if dirs is None:
        return {"commit": None, "version": version, "patch": None, "checked": False}
    commit = _head_commit(*dirs)
    try:
        status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", "src")
        if status.returncode != 0:
            raise OSError(status.stderr.decode(errors="replace"))
        lines = status.stdout.decode().splitlines()
        if not lines:
            return {"commit": commit, "version": version, "patch": None, "checked": True}
        patch = _git(root, "diff", "--binary", "HEAD", "--", "src").stdout
        for line in lines:
            if line.startswith("?? "):
                path = line[3:].strip().strip('"')
                patch += _git(root, "diff", "--binary", "--no-index", "--", "/dev/null",
                              path).stdout
        return {"commit": commit, "version": version, "patch": patch or None, "checked": True}
    except (OSError, subprocess.SubprocessError):
        return {"commit": commit, "version": version, "patch": None, "checked": False}


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
        data = parent.read_bytes()
        try:
            printed = json.loads(data)["processStep"][-1]["output"][0]["sourceCitation"]["sha256"]
        except (ValueError, KeyError, IndexError, TypeError):
            printed = None
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
    libraries, plat = _libraries(), _platform()
    steps = list(layer.chain.steps)
    racks = list(racks) if racks is not None else [None] * len(steps)
    states = step_states(steps, layer_input)
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

    record_steps, defined, upstream = [], set(), None
    for k, (ref, rack) in enumerate(applying, start=1):
        device = get_device(ref.device)
        params = validate_params(device, ref.params)
        run = _recorded_params(device, params)
        algorithm, context = [], {"libraries": libraries, "platform": plat}
        if "precision" in params:
            ctx = fft_policy.context(params["precision"])
            context.update({key: ctx[key] for key in ("precision", "backend", "fallback")
                            if key in ctx})
            cid = f"fft_{ctx['backend']['name']}"
            algorithm.append(cid if cid in defined else _fft_component(ctx, cid))
            defined.add(cid)
            run[cid] = {"precision": ctx["precision"], "planner": ctx["backend"]["planner"]}
        if origin_of_device(device).kind != "first_party":
            cid = f"code_{device.name}"
            algorithm.append(cid if cid in defined else _origin_component(device, code, cid))
            defined.add(cid)
            context["thirdParty"] = [{"component": cid}]
        outcome = {}
        if is_transform(device):
            upstream = cache_key(device.name, sid, keyed_params(device, params), upstream=upstream)
            outcome["cache_key"] = upstream
        if region and k == len(region):
            r, c, h, w = (int(v) for v in layer.tags["roi.window"].split(","))
            from dynamix.roi.runner import roi_margin_of

            run["roi"] = {"window": [r, c, h, w],
                          "margin": sum(roi_margin_of(d, p)[0] for d, p in region)}
            stamps = (result or {}).get("_roi") if isinstance(result, dict) else None
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
        if outcome:
            step["report"] = [{"name": "outcome", "description": outcome}]
        record_steps.append(step)
        reads = [f"d{k}"]

    printing = {"identifier": {"code": f"export/{export}", "codeSpace": CODE_SPACE},
                "softwareReference": software if not record_steps else "gdx"}
    if print_params:
        printing["runTimeParameters"] = {key: _value(v) for key, v in print_params.items()}
    record_steps.append({
        "id": "print", "description": description,
        "stepDateTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": reads, "processingInformation": printing,
        "executionContext": {"libraries": libraries, "platform": plat},
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
    code = kw.pop("code", None) or code_identity()
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
