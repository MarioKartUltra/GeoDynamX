# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Reading and writing ``.dynamix`` project files.

Three behaviours carry the weight here:

*Portability.* A source under the project directory is stored RELATIVE to the project file, so a
folder containing a project and its rasters can be moved, copied or sent to a colleague and still
open. A source outside that directory stays absolute rather than becoming a fragile ``../..``
climb across volumes.

*One path representation.* Relative on disk, ABSOLUTE in memory. :func:`open_project` absolutises
every source against the directory it read the file from; :func:`save_project` relativises on the
way out. A ``Project`` therefore never carries a relative path whose meaning depends on a directory
it does not remember -- which is what made "Save As" into an unrelated folder rebind a source to
whatever happened to sit at the same offset there, and what made deduplicating a source by path
compare a stored ``data/dem.tif`` against an incoming absolute path and conclude they differ.

*Honesty about drift.* :func:`save_project` records a sha256 of every source it can read, so drift
detection works for the ordinary caller who never computed a hash of their own. Opening reports
which sources are missing and which have changed, rather than silently reproducing a different
figure under the same filename. Neither condition refuses to open -- the user needs the project
loaded in order to fix it, which is what :func:`relocate_source` is for.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

from dynamix.model.project import EXT, FORMAT, SCHEMA, Project, SourceRef

_CHUNK = 1 << 20


class ProjectFormatError(ValueError):
    """The file is not a DynamiX project."""


class ProjectSchemaError(ValueError):
    """The file is a DynamiX project of a schema this build cannot read."""


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _app_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("geodynamix-beta")        # GeoDynamix_Beta's own distribution
        except PackageNotFoundError:
            return version("dynamix")
    except Exception:
        return "0.0.0+unknown"


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _absolute(source_path) -> Path:
    """An in-memory source path as an absolute path.

    In-memory paths are absolute by invariant, so this is usually the identity. A relative one can
    only have been handed straight to ``add_source`` by a caller, and is interpreted the way every
    other Python API would interpret it: against the process's working directory. It is never
    reinterpreted against a project directory -- doing that is how a saved-elsewhere project bound
    itself to a stranger's raster.
    """
    p = Path(source_path)
    return p if p.is_absolute() else Path.cwd() / p


def _store_path(source_path: str, project_dir: Path) -> str:
    """Relative to the project directory when the source lives under it, else absolute.

    Both sides must be resolved before comparing: on macOS ``/tmp`` (and any symlinked mount,
    e.g. iCloud Desktop/Documents) is itself a symlink to ``/private/tmp``, so an unresolved
    absolute source compared against a resolved project directory fails ``relative_to`` even
    when the source genuinely sits under the project -- silently defeating portability.
    """
    abs_p = _absolute(source_path).resolve()
    project_dir = project_dir.resolve()
    try:
        return abs_p.relative_to(project_dir).as_posix()
    except ValueError:
        return str(abs_p)


def save_project(project: Project, path) -> Path:
    """Write ``project`` to ``path``, adding the extension if omitted. Returns the path written.

    Called with any path this is also "Save As" -- there is no other function -- so it must bind
    sources by their absolute in-memory identity rather than by re-reading a stored relative path
    against the destination.

    Every source that resolves to a readable file is hashed here, on both the payload and the live
    ``SourceRef``. A source that does not resolve keeps whatever hash it had: clearing it would
    throw away the only evidence of what the data used to be, and raising would refuse to save work
    merely because a drive is unmounted. Re-saving therefore also re-stamps, which is how a user
    accepts a change reported by :func:`changed_sources`.

    Metadata is written back onto ``project`` rather than only into the payload, so ``created``
    marks when the document was first saved instead of being restamped by every save.
    """
    path = Path(path)
    if path.suffix != EXT:
        path = path.with_suffix(EXT)

    project.validate()          # never write a project that cannot be reopened; before any I/O

    path.parent.mkdir(parents=True, exist_ok=True)
    project_dir = path.parent

    project.app_version = _app_version()
    project.created = project.created or _now()
    project.modified = _now()

    for ref in project.sources.values():
        resolved = _absolute(ref.path)
        if resolved.is_file():
            ref.sha256 = sha256_of(resolved)

    payload = project.to_payload()
    for entry in payload["sources"]:
        entry["path"] = _store_path(entry["path"], project_dir)
    path.write_text(json.dumps(payload, indent=2))
    return path


def open_project(path) -> Project:
    """Read a project. Raises on a foreign or too-new file; missing or changed sources are
    reported by :func:`missing_sources` / :func:`changed_sources`, not raised.

    Every source path is absolutised against the project file's own directory, so the returned
    ``Project`` no longer depends on where it was read from.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProjectFormatError(
            f"{path.name} is not a DynamiX project (not valid JSON: {exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise ProjectFormatError(
            f"{path.name} is not a DynamiX project (top level is a JSON "
            f"{type(payload).__name__}, expected an object)"
        )
    if payload.get("format") != FORMAT:
        raise ProjectFormatError(
            f"{path.name} is not a DynamiX project (format={payload.get('format')!r})"
        )
    schema = int(payload.get("schema", 0))
    if schema > SCHEMA:
        raise ProjectSchemaError(
            f"{path.name} uses schema {schema}, newer than this build's {SCHEMA}; upgrade DynamiX"
        )
    project = Project.from_payload(payload)
    project_dir = path.parent.resolve()
    for ref in project.sources.values():
        stored = Path(ref.path)
        ref.path = str(stored if stored.is_absolute() else project_dir / stored)
    return project


def resolve_source(project: Project, source_id: str, project_path) -> Path:
    """Absolute path of a source. In-memory paths are absolute, so ``project_path`` only matters
    for a ``Project`` assembled by hand rather than opened."""
    if source_id not in project.sources:
        raise KeyError(f"no source named {source_id!r}; known: {sorted(project.sources)}")
    p = Path(project.sources[source_id].path)
    return p if p.is_absolute() else (Path(project_path).parent / p)


def relocate_source(project: Project, source_id: str, new_path) -> SourceRef:
    """Point a source at a different file and re-hash it. Returns the updated ``SourceRef``.

    The remedy :func:`missing_sources` and :func:`changed_sources` lacked -- they reported problems
    the model gave no way to fix. Because layers reference a source by id, this is the single edit
    that repairs every layer over that raster. Refuses a path that is not a readable file: the
    whole point is to repair a broken reference, so replacing it with another broken one is a
    typo, not an intention.
    """
    if source_id not in project.sources:
        raise KeyError(f"no source named {source_id!r}; known: {sorted(project.sources)}")
    p = _absolute(new_path)
    if not p.is_file():
        raise ValueError(f"{source_id}: no file at {p}")
    ref = project.sources[source_id]
    ref.path = str(p)
    ref.sha256 = sha256_of(p)
    return ref


def missing_sources(project: Project, project_path) -> list[str]:
    return [sid for sid in project.sources
            if not resolve_source(project, sid, project_path).is_file()]


def changed_sources(project: Project, project_path) -> list[str]:
    """Source ids whose bytes no longer match the hash recorded at save time. Sources with no
    recorded hash, or that are missing, are not reported here."""
    out = []
    for sid, ref in project.sources.items():
        if not ref.sha256:
            continue
        resolved = resolve_source(project, sid, project_path)
        if resolved.is_file() and sha256_of(resolved) != ref.sha256:
            out.append(sid)
    return out
