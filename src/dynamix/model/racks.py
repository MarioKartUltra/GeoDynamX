# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Racks: named, reusable chains, one JSON file each.

A rack file is ``{"format": "geodynamix-rack", "schema": 1, "name", "description", "steps":
[{"device", "params", "bypassed"}], "created", "app_version"}``. Reading never raises for a bad
file: :func:`load_racks` lists it with the reason, so a rack naming a device or a parameter this
build does not have stays visible and never stops the other racks from loading. Headless: the
shell passes the directory (``dynamix.shell.settings.racks_dir``).

A rack lands on a chain by :func:`merge_rack` (split and merge): its transforms after the chain's
transforms, its filters after the chain's filters, so the order stays transforms-then-filters.
"""
from __future__ import annotations

import dataclasses
import json
import re
import unicodedata
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from dynamix.model.chain import Chain, DeviceRef
from dynamix.model import capabilities
from dynamix.model.contracts import StepState, next_step_state
from dynamix.model.forking import is_primary, routing_head
from dynamix.model.device import get_device, is_transform, validate_params
from dynamix.model.presets import PRESETS

RACK_FORMAT = "geodynamix-rack"
RACK_SCHEMA = 1


@dataclasses.dataclass(frozen=True)
class RackStep:
    device: str
    params: dict
    bypassed: bool = False


@dataclasses.dataclass(frozen=True)
class Rack:
    name: str
    steps: tuple[RackStep, ...]
    description: str = ""
    created: str = ""
    app_version: str = ""
    builtin: bool = False


@dataclasses.dataclass(frozen=True)
class RackEntry:
    """One rack the browser lists. ``problem`` non-empty: listed greyed with that reason (an
    unreadable file has ``rack`` None and is named by its file)."""

    rack: Rack | None
    path: Path | None
    problem: str = ""


def rack_to_payload(rack: Rack) -> dict:
    return {"format": RACK_FORMAT, "schema": RACK_SCHEMA, "name": rack.name,
            "description": rack.description,
            "steps": [{"device": s.device, "params": dict(s.params), "bypassed": bool(s.bypassed)}
                      for s in rack.steps],
            "created": rack.created, "app_version": rack.app_version}


def rack_from_payload(d) -> Rack:
    """The rack a payload describes. Raises ValueError for anything that is not a rack of a
    schema this build reads; device and parameter names are checked by :func:`load_racks`."""
    if not isinstance(d, dict) or d.get("format") != RACK_FORMAT:
        raise ValueError(f"not a {RACK_FORMAT} file")
    schema = d.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < 1:
        raise ValueError(f"rack schema {schema!r} is not a positive integer")
    if schema > RACK_SCHEMA:
        raise ValueError(f"rack schema {schema} is newer than this build reads ({RACK_SCHEMA})")
    name = d.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("the rack has no name")
    raw = d.get("steps")
    if not isinstance(raw, list):
        raise ValueError("the rack's steps are not a list")
    steps = []
    for s in raw:
        if not isinstance(s, dict) or not isinstance(s.get("device"), str):
            raise ValueError(f"a step names no device: {s!r}")
        params = s.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{s['device']}: params are not a mapping")
        steps.append(RackStep(s["device"], dict(params), bool(s.get("bypassed", False))))
    return Rack(name=name, steps=tuple(steps), description=str(d.get("description", "")),
                created=str(d.get("created", "")), app_version=str(d.get("app_version", "")))


def _problem(rack: Rack) -> str:
    """Why this build cannot use ``rack`` ("" when it can): an unknown device, a param its
    device rejects, or an order that is not transforms-then-filters."""
    for step in rack.steps:
        try:
            device = get_device(step.device)
        except KeyError:
            return f"unknown device {step.device!r}"
        try:
            validate_params(device, step.params)
        except (TypeError, ValueError) as exc:
            return str(exc)
    try:
        Chain(tuple(DeviceRef(s.device, s.params) for s in rack.steps)).validate()
    except ValueError as exc:
        return str(exc)
    return ""


def _slug(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-") or "rack"


def _app_version() -> str:
    try:
        return metadata.version("geodynamix-beta")
    except metadata.PackageNotFoundError:
        return ""


def load_racks(directory: Path) -> list[RackEntry]:
    """Every ``*.json`` in ``directory``, by file name. A missing directory holds no racks."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.json")):
        try:
            rack = rack_from_payload(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            entries.append(RackEntry(None, path, f"{path.name}: {exc}"))
            continue
        entries.append(RackEntry(rack, path, _problem(rack)))
    return entries


def save_rack(directory: Path, rack: Rack) -> Path:
    """Write ``rack`` to ``<directory>/<slug of its name>.json`` and return the path. The file
    of the same rack is replaced; a file holding a different rack (two names, one slug) is
    refused with FileExistsError. ``created`` is kept when set, else stamped now (UTC);
    ``app_version`` is this build's."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(rack.name)}.json"
    if path.exists():
        try:
            held = rack_from_payload(json.loads(path.read_text(encoding="utf-8"))).name
        except (OSError, ValueError):
            held = None
        if held != rack.name:
            what = f"the rack {held!r}" if held is not None else "a file that is not a rack"
            raise FileExistsError(f"{path} holds {what}; choose another name")
    stamped = dataclasses.replace(
        rack, builtin=False, app_version=_app_version(),
        created=rack.created or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.write_text(json.dumps(rack_to_payload(stamped), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def delete_rack(directory: Path, name: str) -> None:
    """Remove the file holding the rack ``name``; KeyError when no file in ``directory`` does."""
    for entry in load_racks(directory):
        if entry.rack is not None and entry.rack.name == name:
            entry.path.unlink()
            return
    raise KeyError(f"no rack named {name!r} in {directory}")


def builtin_racks() -> list[Rack]:
    """The read-only racks of ``model.presets.PRESETS``, in its order."""
    return [Rack(name, tuple(RackStep(device, dict(params)) for device, params in steps),
                 builtin=True)
            for name, steps in PRESETS.items()]


def _binds_layer(d: dict) -> bool:
    """A step whose hidden params name the layer it was saved from: a band row's own band
    (``band_select`` with ``_band_id``) or a bus's routed sends. It is that layer's identity, so
    a rack leaves it out and stays usable on any other band row or dataset."""
    params = d.get("params", {})
    if d["device"] == "band_select":
        return bool(params.get("_band_id"))
    if d["device"] == "bus":
        return params.get("_sends", "[]") not in ("", "[]")
    return False


_RESET_IN_RACK = {"inpaint": ("_sends", "_mask_polygons"), "fill": ("_sends", "_mask_polygons")}


def _portable(d: dict) -> dict:
    """A step with the params that name THIS layer's sends and drawn regions reset, so the
    rack keeps the tool and its knobs and the sends are declared again where it lands."""
    reset = _RESET_IN_RACK.get(d["device"])
    if not reset:
        return d
    params = {**d.get("params", {}), **{k: "[]" for k in reset}}
    return {**d, "params": params}


def rack_from_descriptors(name: str, descriptors) -> Rack:
    """A rack of the chain the workflow zone holds: every step with all its params filled
    (view-only included) and its bypass state, in order. ``created`` is stamped by
    :func:`save_rack`."""
    return Rack(name, tuple(
        RackStep(d["device"], validate_params(get_device(d["device"]), d.get("params", {})),
                 bool(d.get("bypassed", False)))
        for d in map(_portable, descriptors) if not _binds_layer(d)))


def merge_rack(descriptors: list[dict], rack: Rack) -> list[dict]:
    """``descriptors`` with ``rack`` dropped in, split and merged: the rack's transforms after the
    chain's transforms, its filters after the chain's filters. A rack Transform whose device id
    already runs (not bypassed) in the chain is left out: the chain's own step stays the analysis
    and the rack's other steps join it. Each new descriptor carries ``"rack": rack.name``; the
    input list is not modified."""
    chain = [{**d, "params": dict(d.get("params", {}))} for d in descriptors]
    boundary = next((i for i, d in enumerate(chain)
                     if not is_transform(get_device(d["device"]))), len(chain))
    running = {d["device"] for d in chain
               if not d.get("bypassed") and is_transform(get_device(d["device"]))}
    new = [{"device": s.device, "params": dict(s.params), "bypassed": bool(s.bypassed),
            "rack": rack.name} for s in rack.steps
           if not (s.device in running and is_transform(get_device(s.device)))]
    transforms = [d for d in new if is_transform(get_device(d["device"]))]
    filters = [d for d in new if not is_transform(get_device(d["device"]))]
    return chain[:boundary] + transforms + chain[boundary:] + filters


def contiguous_rack_titles(descriptors: list[dict]) -> list[dict]:
    """``descriptors`` with every rack title naming one contiguous run, as the workflow zone draws
    a rack. A merge can split a rack in two; each later run of a title takes the first free
    ``"<title> (k)"``, k = 2, 3, ..."""
    out = [dict(d) for d in descriptors]
    taken = {d.get("rack") for d in out} - {None}
    seen: set[str] = set()
    i = 0
    while i < len(out):
        title = out[i].get("rack")
        j = i
        while j < len(out) and out[j].get("rack") == title:
            j += 1
        if title is not None:
            if title in seen:
                k = 2
                while f"{title} ({k})" in taken:
                    k += 1
                title = f"{title} ({k})"
                taken.add(title)
                for d in out[i:j]:
                    d["rack"] = title
            seen.add(title)
        i = j
    return out


def rack_state(rack: Rack, steps, layer_input: str, *, master: bool = False,
               roi_holder: bool = False) -> StepState:
    """Whether ``rack`` can head a drop on a layer whose enabled Transform steps are ``steps``
    (``DeviceRef``) over ``layer_input``; ``master`` and ``roi_holder`` as in
    ``dynamix.model.forking.drop_states``.

    The step judged is the rack's first enabled step that :func:`merge_rack` keeps (a Transform
    the chain already runs is left out), with its own params, the way ``drop_states`` judges its
    device. A primary analyzer, or a device that reads the source file, dropped where a primary
    already heads the chain or on a dataset's master row forks a sibling that starts with the
    layer's routing steps (``dynamix.model.forking.routing_head``: a band row's ``band_select``,
    a bus row's ``bus``), so it is judged after those alone. On an ROI-holder layer it takes the
    heading primary's place, so it is judged on the steps ahead of that primary. Every other step
    is judged as appended to ``steps``."""
    steps = tuple(steps)
    running = {ref.device for ref in steps}
    first = next((s for s in rack.steps if not s.bypassed
                  and not (s.device in running and is_transform(get_device(s.device)))), None)
    if first is None:
        return StepState(True)
    device = get_device(first.device)
    head = next((i for i, ref in enumerate(steps) if is_primary(get_device(ref.device))), None)
    if not (is_primary(device) or capabilities.has(device, "SourceReader")):
        base = steps
    elif roi_holder:
        base = steps if head is None else steps[:head]
    elif master or head is not None:
        base = routing_head(steps)
    else:
        base = steps
    return next_step_state(base, layer_input, first.device, first.params)
