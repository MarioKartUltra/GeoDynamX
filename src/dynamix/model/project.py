# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The project document: what a user saves and opens.

A project is a RECIPE, not an archive. It records sources, chains and metadata -- never computed
results. Re-opening recomputes, reusing the content-hash-keyed transform cache that lives outside
the file, so a project stays small, diffs in git, and reproduces a figure rather than storing a
picture of one.

Layers name a ``source_id`` into the project's source registry rather than carrying a path, so
several layers over one raster share a source and relocating a moved file is a single edit.
"""
from __future__ import annotations

import dataclasses
import pathlib
from pathlib import Path

from dynamix.model.chain import Chain
from dynamix.model.layer import Layer
from dynamix.topology.links import LinkStore, ObjRef
from dynamix.topology.model import TopologyModel

FORMAT = "dynamix-project"
SCHEMA = 1
EXT = ".dynamix"


def _path_key(path: str) -> str:
    """Comparison key for deciding whether two paths name the same raster.

    Symlinks and ``..`` segments are resolved where the OS allows, so a source opened from a
    project (absolute, resolved against the project directory) still matches the same file picked
    afresh in a file dialog -- on macOS that is the difference between ``/tmp/...`` and
    ``/private/tmp/...``. Falls back to the raw string rather than raising: failing to deduplicate
    is a nuisance, refusing to add a source is a broken file dialog.
    """
    try:
        return str(Path(path).resolve())
    except OSError:
        return path


#: the design point 3, the MODEL half of typed sources: the
#: layer panel's typed rows wait for the user's mockups, but the vocabulary is committed now so
#: persistence and callers can start using it. "raster" (grids), "points" (catalogues),
#: "vector" (reference geometry), "object" (chain products / annotations -- e.g. a chain-product
#: npz loaded as its own source). ``SourceRef.kind`` stays a free string on load (older projects
#: and forward compatibility), but new call sites pick from here.
SOURCE_KINDS = ("raster", "points", "vector", "object")


@dataclasses.dataclass
class SourceRef:
    """One input raster. ``path`` is ABSOLUTE in memory and stored relative to the project file
    when it sits under the project directory -- see :mod:`dynamix.model.projectfile`, which owns
    both halves of that invariant. ``sha256`` lets opening detect that the data changed under a
    saved analysis; :func:`~dynamix.model.projectfile.relocate_source` is how the user fixes it."""

    source_id: str
    path: str
    sha256: str | None = None
    label: str = ""
    collapsed: bool = False
    #: ``"raster"`` (default) or ``"points"`` -- a CSV point catalogue. Read via ``.get`` on load (below) so a project saved before this field existed
    #: opens unchanged, as ``"raster"``: every source that predates point layers WAS a raster.
    kind: str = "raster"
    #: The DATASET itself hidden (its raster drape / canvas image), its layers' products still
    #: shown -- the source header row's own H (2026-08-29, the design: "the toggle sits on the
    #: source"). Additive: ``.get`` on load, so older projects open with it False.
    hidden: bool = False
    #: A TEMPORARY derivative dataset (2026-09-23): its file lives in this session's scratch
    #: folder, and a saved project leaves it out (``Project.to_payload``) until it is saved as a
    #: file of its own. Additive (``.get`` on load).
    temporary: bool = False
    #: Show this dataset's row nested INSIDE the row of source ``nest_under`` (a derivative
    #: placed under the dataset it came from). Display only; additive (``.get`` on load).
    nest_under: str | None = None
    #: The bands this dataset is made of when it is NOT the file as a whole -- a container's
    #: grid ids (an imported sensor group) or ``"#k"`` band indices of a multiband file (a
    #: stack a band was removed from). ``None`` = the whole file. Part of the source's
    #: IDENTITY (``Project.add_source`` dedups on path AND bands): two groups imported from
    #: one file are two datasets. Additive (``.get`` on load).
    bands: list | None = None

    def to_payload(self) -> dict:
        return {"source_id": self.source_id, "path": self.path,
                "sha256": self.sha256, "label": self.label, "collapsed": self.collapsed,
                "kind": self.kind, "hidden": self.hidden, "temporary": self.temporary,
                "nest_under": self.nest_under, "bands": self.bands}

    @classmethod
    def from_payload(cls, d: dict) -> "SourceRef":
        return cls(source_id=d["source_id"], path=d["path"],
                   sha256=d.get("sha256"), label=d.get("label", ""),
                   collapsed=d.get("collapsed", False), kind=d.get("kind", "raster"),
                   hidden=bool(d.get("hidden", False)),
                   temporary=bool(d.get("temporary", False)), nest_under=d.get("nest_under"),
                   bands=list(d["bands"]) if d.get("bands") else None)


@dataclasses.dataclass
class TransectRecord:
    """One A->A' transect drawn on the raster canvas.

    ``a``/``b`` are ``(x, y)`` PIXEL coordinates in the canvas's own display frame, already
    AUTO-ORIENTED (``dynamix.core.transect.orient_endpoints``) at draw time -- this record only
    ever carries what the canvas gesture already produced; it never re-derives orientation.

    ``transect_id`` is ``TransectPanel``'s own MONOTONIC counter value, never reused even after a
    delete (the collision lesson:
    a name derived from ``len(transects) + 1`` can repeat after a middle delete, and a repeated
    name silently overwrites a still-live drawn item) -- an identity, not a list position, so it
    is what persistence keys on and what an undo restores unchanged.
    """

    transect_id: int
    a: tuple[float, float]
    b: tuple[float, float]
    visible: bool = True
    buffer_px: float = 20.0

    def to_payload(self) -> dict:
        return {"transect_id": self.transect_id, "a": list(self.a), "b": list(self.b),
                "visible": self.visible, "buffer_px": self.buffer_px}

    @classmethod
    def from_payload(cls, d: dict) -> "TransectRecord":
        return cls(transect_id=int(d["transect_id"]),
                   a=(float(d["a"][0]), float(d["a"][1])),
                   b=(float(d["b"][0]), float(d["b"][1])),
                   visible=bool(d.get("visible", True)),
                   buffer_px=float(d.get("buffer_px", 20.0)))


@dataclasses.dataclass
class RoiRecord:
    """One region of interest as a world object.

    ``row``/``col``/``h``/``w`` are in SOURCE-FILE coordinates -- the frame ``MainWindow.
    _on_roi_create`` already writes (``shell/main_window.py:2093-2095``), which adds the loaded
    window's offsets to what the user drew on screen. Storing the display frame instead would
    address a region nobody selected on any raster too large to load whole.

    ``layer_ids`` carries the two forms the user required, both of which must exist: EMPTY is the
    GENERAL ROI -- a region of the source applicable to any layer over it -- and non-empty is the
    ROI drawn FOR those layers, the child-layer path that already exists. The record is addressable
    either way; ``roi_id`` is the id a ``Layer.roi_id`` names and an ``ObjRef``-style reference can
    embed, minted monotonically by :meth:`Project.add_roi` and never reused.
    """

    roi_id: str
    row: int
    col: int
    h: int
    w: int
    layer_ids: tuple[int, ...] = ()
    label: str = ""
    visible: bool = True
    #: The source this region belongs to (ROIs are saved ON a
    #: dataset, and a project can hold several). Additive: ``None`` on projects saved before.
    source_id: str | None = None

    def to_payload(self) -> dict:
        return {"roi_id": self.roi_id, "row": int(self.row), "col": int(self.col),
                "h": int(self.h), "w": int(self.w),
                "layer_ids": [int(i) for i in self.layer_ids],
                "label": self.label, "visible": bool(self.visible),
                "source_id": self.source_id}

    @classmethod
    def from_payload(cls, d: dict) -> "RoiRecord":
        return cls(roi_id=d["roi_id"], row=int(d["row"]), col=int(d["col"]),
                   h=int(d["h"]), w=int(d["w"]),
                   layer_ids=tuple(int(i) for i in d.get("layer_ids") or ()),
                   label=d.get("label", ""), visible=bool(d.get("visible", True)),
                   source_id=d.get("source_id"))


class ReferenceLayerRecord:
    """A GIS vector file drawn OVER the data as interpretation (2026-08-29, BOEM's seafloor
    anomaly shapefiles): the FILE is the source of truth, this record is where it is, whether it
    is shown and in what colour. Geometry is re-read on open, never serialised (DDIA: derived,
    recomputable). ``ref_id`` is ``"ref{N}"``, monotonic like the other ids here."""

    def __init__(self, ref_id: str, path: str, name: str = "", visible: bool = True,
                 color: str = "#ffaa00"):
        self.ref_id = ref_id
        self.path = path
        self.name = name or pathlib.Path(path).stem
        self.visible = bool(visible)
        self.color = color

    def to_payload(self) -> dict:
        return {"ref_id": self.ref_id, "path": self.path, "name": self.name,
                "visible": bool(self.visible), "color": self.color}

    @classmethod
    def from_payload(cls, d: dict) -> "ReferenceLayerRecord":
        return cls(str(d["ref_id"]), str(d["path"]), str(d.get("name", "")),
                   bool(d.get("visible", True)), str(d.get("color", "#ffaa00")))

    def __eq__(self, other):
        return isinstance(other, ReferenceLayerRecord) and self.to_payload() == other.to_payload()

    def __repr__(self):
        return f"ReferenceLayerRecord({self.ref_id!r}, {self.path!r})"


#: Colours handed to successive reference layers (distinct from the WTMM overlays' white/amber
#: family so an interpretation never reads as a measurement).
REFERENCE_COLORS = ("#ff8800", "#00e5ff", "#ff4fd8", "#7cff4f", "#ffe14f", "#4f8bff", "#ff5f5f")


@dataclasses.dataclass
class AnnotationRecord:
    """One note pinned to a place in the world.

    ``point`` is ``(x, y)`` in the canvas's own DISPLAY frame -- the same convention
    ``TransectRecord.a``/``b`` already carry (above), stated here so the two never diverge; an
    annotation and a transect drawn in one gesture must land in one frame.

    ``target`` is an :class:`~dynamix.topology.links.ObjRef`: which layer, which transform
    produced the geometry, what KIND of object it is and its index within that transform's output.
    The kind is validated against ``codes.KINDS`` by ``ObjRef`` itself (``topology/links.py``), and
    that is precisely what makes a label attachable to a feature of ANY dimension: a
    ``"point"``, a chain (``"line"``), or a whole hierarchy of chains across scales made into a
    tree (``"surface"``/``"body"``). Point-pinned came first, but never a point ALONE.

    A record needs at least one of the two: a note that names neither a place nor an object is
    not pinned to anything, and it is refused at construction rather than saved as a mystery.

    ``annotation_id`` is monotonic and never reused, even after a middle delete -- the collision
    lesson ``TransectRecord`` records above, for the same reason: a repeated id silently
    overwrites a still-live note.

    KNOWN LIMITATION of v2: ``ObjRef.obj_id`` renumbers when a chain is
    re-run with different parameters, so a target can come to name a different object. The answer
    is persisted chain products, which these slices do not build; the instability is
    named here, not fixed here.
    """

    annotation_id: int
    text: str
    point: tuple[float, float] | None = None
    target: ObjRef | None = None
    visible: bool = True

    def __post_init__(self) -> None:
        if self.point is None and self.target is None:
            raise ValueError(
                "an annotation must be pinned to something: pass point=(x, y) in the canvas "
                "display frame, or target=ObjRef(layer_id, transform, kind, obj_id), or both")

    def to_payload(self) -> dict:
        return {"annotation_id": int(self.annotation_id), "text": self.text,
                "point": (None if self.point is None
                          else [float(self.point[0]), float(self.point[1])]),
                "target": (None if self.target is None else {
                    "layer_id": self.target.layer_id,
                    "transform": self.target.transform,
                    "kind": self.target.kind,
                    "obj_id": self.target.obj_id}),
                "visible": bool(self.visible)}

    @classmethod
    def from_payload(cls, d: dict) -> "AnnotationRecord":
        point = d.get("point")
        target = d.get("target")
        return cls(annotation_id=int(d["annotation_id"]), text=d.get("text", ""),
                   point=(None if point is None
                          else (float(point[0]), float(point[1]))),
                   target=(None if target is None else ObjRef(**target)),
                   visible=bool(d.get("visible", True)))


def _camera_to_payload(state: dict) -> dict:
    """One ``Scene._camera_snapshot()`` dict as JSON -- its tuples flattened to lists.

    ``position``/``focal_point``/``up``/``clipping_range`` are tuples in memory
    (``shell/arrangement/scene.py``'s ``_camera_snapshot``); JSON has no tuple, so they go out as
    lists and :func:`_camera_from_payload` re-tuples them on the way in. Without that half, an
    open->save cycle would silently re-type a viewpoint from tuples to lists.
    """
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in state.items()}


def _camera_from_payload(d: dict) -> dict:
    """The inverse of :func:`_camera_to_payload`: every JSON list becomes a tuple again, so a
    viewpoint read from disk compares equal to the one that was remembered live."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


@dataclasses.dataclass
class Project:
    title: str = ""
    description: str = ""
    sources: dict[str, SourceRef] = dataclasses.field(default_factory=dict)
    layers: list[Layer] = dataclasses.field(default_factory=list)
    topologies: list[TopologyModel] = dataclasses.field(default_factory=list)
    #: User-asserted links between chains -- a LIVE LinkStore,
    #: not a payload dict, mirroring how ``topologies`` holds live ``TopologyModel``s rather than
    #: their own ``to_payload()`` shape. ``to_payload``/``from_payload`` below are the ONE seam
    #: where this becomes JSON: additive "user_links" key, absent on every pre-existing project
    #: file, which ``from_payload``'s ``.get(..., {})`` -> ``LinkStore.from_payload({})`` turns
    #: into an honestly empty store rather than raising on the missing key.
    user_links: LinkStore = dataclasses.field(default_factory=LinkStore)
    #: Drawn transects -- additive "transects" payload
    #: key, same discipline as ``user_links`` just above: a plain list here (not a Store class --
    #: transects have no cross-references to resolve, unlike a link's two ``ObjRef``s, so a list
    #: of small dataclasses is the whole model). ``TransectPanel`` is the one place that reads
    #: this list's ORDER as meaningful (row order); ``Project`` itself has no opinion on it.
    transects: list[TransectRecord] = dataclasses.field(default_factory=list)
    #: Regions of interest as world objects -- additive
    #: "rois" payload key, same discipline as ``transects`` just above. A plain list: an ROI has
    #: no cross-references to resolve, and the general form (empty ``layer_ids``) is deliberately
    #: not indexed by layer, because it belongs to no single one.
    rois: list[RoiRecord] = dataclasses.field(default_factory=list)
    #: Remembered viewpoints -- additive "cameras" payload
    #: key. Keys are ``Scene.camera_key()`` strings ("geo:<mode>", "frame:<bounds-sig>", either
    #: with an "|sspace:on" suffix); values are the ``Scene._camera_snapshot()`` shape.
    #:
    #: Per PROJECT, deliberately, unlike ``Settings.center_view``, which is per MACHINE: which
    #: pane a user last had open is a workspace preference, but where the camera stood when a
    #: fabric read at 040 is closer to a measurement -- it belongs to the analysis, travels with
    #: the file to a collaborator, and is worth keeping across the process. ``Scene``
    #: owns the live dict; this is only where it rests between sessions.
    cameras: dict[str, dict] = dataclasses.field(default_factory=dict)
    #: Notes pinned to the world -- additive "annotations"
    #: payload key, same discipline as ``transects`` and ``rois`` above. A plain list: an
    #: annotation's one cross-reference is its own ``ObjRef``, which carries everything needed to
    #: re-find its object, so there is nothing for a Store class to resolve.
    annotations: list[AnnotationRecord] = dataclasses.field(default_factory=list)
    #: reference layers (2026-08-29): additive key ``"reference_layers"``; see ReferenceLayerRecord.
    reference_layers: list = dataclasses.field(default_factory=list)
    created: str = ""
    modified: str = ""
    app_version: str = ""
    _next_layer_id: int = 0
    _next_source_id: int = 0
    _next_roi_id: int = 0
    _next_annotation_id: int = 0
    _next_ref_id: int = 0

    def add_source(self, path: str, *, label: str = "",
                   sha256: str | None = None, kind: str = "raster",
                   bands: "list | None" = None) -> SourceRef:
        """Register a source, reusing an existing entry when the path already names the same file.

        Two layers over one raster must share a source, or relocating it means N edits and the
        cache keys diverge for identical data. Comparing raw strings was not enough once a project
        had been through a save/open cycle.

        ``kind`` defaults to ``"raster"`` (every existing call site is unchanged); ``point_import.
        load_points`` is the one caller that passes ``kind="points"``. Not applied to an EXISTING
        (deduped) source -- a path's kind cannot legitimately change between two opens of it, so
        there is nothing to reconcile there, unlike ``sha256``/``label`` which a later open may
        legitimately update.

        ``bands`` (a container group's grid ids, or ``"#k"`` indices) joins the identity: the
        same file with a different band list is a different dataset (its own row, its own
        cache lines).
        """
        key = _path_key(path)
        want = list(bands) if bands else None
        for existing in self.sources.values():
            if _path_key(existing.path) == key and (existing.bands or None) == want:
                if sha256:
                    existing.sha256 = sha256
                if label:
                    existing.label = label
                return existing
        ref = SourceRef(source_id=f"src{self._next_source_id}", path=path,
                        sha256=sha256, label=label, kind=kind, bands=want)
        self._next_source_id += 1
        self.sources[ref.source_id] = ref
        return ref

    def add_layer(self, name: str, source_id: str, chain: Chain | None = None, *,
                  visible: bool = True, tags: dict | None = None,
                  parent_id: int | None = None) -> Layer:
        if source_id not in self.sources:
            raise KeyError(f"no source named {source_id!r}; known: {sorted(self.sources)}")
        layer = Layer(layer_id=self._next_layer_id, name=name, source_id=source_id,
                      chain=(chain or Chain()).materialized(), visible=visible,
                      tags=dict(tags or {}), parent_id=parent_id)
        self._next_layer_id += 1
        self.layers.append(layer)
        return layer

    def add_roi(self, row: int, col: int, h: int, w: int, *,
                layer_ids: tuple[int, ...] = (), label: str = "",
                source_id: str | None = None) -> RoiRecord:
        """Register a region of interest and return it. Coordinates are SOURCE-FILE ones.

        Leaving ``layer_ids`` empty mints the GENERAL form -- an ROI applicable to any layer over
        the source; naming layers mints the drawn-for-those-layers form. Ids are monotonic and
        never reused, so a ``Layer.roi_id`` keeps naming the same record after a middle delete
        (the collision lesson ``TransectRecord`` records).
        """
        roi = RoiRecord(roi_id=f"roi{self._next_roi_id}", row=int(row), col=int(col),
                        h=int(h), w=int(w), layer_ids=tuple(int(i) for i in layer_ids),
                        label=label, source_id=source_id)
        self._next_roi_id += 1
        self.rois.append(roi)
        return roi

    def add_reference_layer(self, path: str, *, name: str = "", color: str | None = None,
                            visible: bool = True) -> ReferenceLayerRecord:
        """Register a vector file as a reference layer; the colour cycles through
        REFERENCE_COLORS unless given. Registering the same path twice returns the existing record."""
        for r in self.reference_layers:
            if r.path == path:
                return r
        rec = ReferenceLayerRecord(f"ref{self._next_ref_id}", path, name=name, visible=visible,
                                   color=color or REFERENCE_COLORS[self._next_ref_id % len(REFERENCE_COLORS)])
        self._next_ref_id += 1
        self.reference_layers.append(rec)
        return rec

    def remove_reference_layer(self, ref_id: str) -> bool:
        """Unregister a reference layer. The FILE is never touched: the record is only where
        the layer is and how it shows, so removing it just forgets the layer -- re-opening the
        file brings it back."""
        for i, r in enumerate(self.reference_layers):
            if r.ref_id == ref_id:
                del self.reference_layers[i]
                return True
        return False

    def add_annotation(self, text: str, *, point: tuple[float, float] | None = None,
                       target: ObjRef | None = None) -> AnnotationRecord:
        """Pin a note to a point, to an object, or to both, and return it.

        ``point`` is in the canvas DISPLAY frame; ``target`` is an ``ObjRef`` of any topology kind.
Passing neither raises -- see :class:`AnnotationRecord`. Ids are monotonic and
        never reused, so a reference to a note keeps naming the same record after a middle delete.
        """
        note = AnnotationRecord(
            annotation_id=self._next_annotation_id, text=text,
            point=(None if point is None else (float(point[0]), float(point[1]))),
            target=target)
        self._next_annotation_id += 1
        self.annotations.append(note)
        return note

    def add_topology(self, model: TopologyModel) -> TopologyModel:
        self.topologies.append(model)
        return model

    def remove_layer(self, layer_id: int, *, cascade: bool = True) -> list[int]:
        """Remove a layer and optionally its children.

        Args:
            layer_id: The ID of the layer to remove.
            cascade: If True (default), recursively remove child layers. If False, raise
                ValueError if the layer has children.

        Returns:
            List of removed layer IDs, children first, then the parent.

        Raises:
            KeyError: If the layer_id is not found.
            ValueError: If cascade=False and the layer has children.
        """
        # Find the layer
        layer = next((l for l in self.layers if l.layer_id == layer_id), None)
        if layer is None:
            raise KeyError(f"no layer with id {layer_id}")

        # Find children
        children = [l for l in self.layers if l.parent_id == layer_id]

        if children and not cascade:
            raise ValueError(f"layer {layer_id} has {len(children)} child layer(s); "
                           "set cascade=True to remove them")

        # Recursively remove children first
        removed_ids = []
        for child in children:
            removed_ids.extend(self.remove_layer(child.layer_id, cascade=True))

        # Remove the parent layer
        self.layers.remove(layer)
        removed_ids.append(layer_id)

        return removed_ids

    def validate(self) -> "Project":
        """Raise unless every layer's chain resolves and validates. Returns self.

        ``add_layer`` materialises the chain it is given, but nothing stops Phase 4 assigning
        ``layer.chain`` directly -- there is no ``set_chain``. Without this, such a project saves
        happily and then fails to open, which is a data-loss shape: the user is told everything
        went fine and discovers otherwise only once the work is gone. ``save_project`` calls this
        before touching disk; callers wanting to check earlier can call it themselves.
        """
        for layer in self.layers:
            try:
                layer.chain.validate()
            except (KeyError, ValueError) as exc:
                detail = exc.args[0] if exc.args else exc
                raise type(exc)(f"layer {layer.layer_id} {layer.name!r}: {detail}") from exc
        return self

    def temporary_sources(self) -> list:
        """The temporary derivative datasets -- what a save leaves out."""
        return [s for s in self.sources.values() if s.temporary]

    def to_payload(self) -> dict:
        # Temporary derivative datasets (their files are this session's scratch) stay out of
        # a saved project, with their layers and ROIs; the live project keeps them.
        temporary = {s.source_id for s in self.temporary_sources()}
        return {
            "format": FORMAT,
            "schema": SCHEMA,
            "app_version": self.app_version,
            "created": self.created,
            "modified": self.modified,
            "title": self.title,
            "description": self.description,
            "sources": [s.to_payload() for s in self.sources.values() if not s.temporary],
            "layers": [l.to_payload() for l in self.layers if l.source_id not in temporary],
            "topologies": [t.to_payload() for t in self.topologies],
            "user_links": self.user_links.to_payload(),
            "transects": [t.to_payload() for t in self.transects],
            "rois": [r.to_payload() for r in self.rois if r.source_id not in temporary],
            "cameras": {k: _camera_to_payload(v) for k, v in self.cameras.items()},
            "annotations": [a.to_payload() for a in self.annotations],
            "reference_layers": [r.to_payload() for r in self.reference_layers],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "Project":
        sources = {s["source_id"]: SourceRef.from_payload(s)
                   for s in payload.get("sources", [])}
        layers = [Layer.from_payload(d) for d in payload.get("layers", [])]
        topologies = [TopologyModel.from_payload(t) for t in payload.get("topologies", [])]
        user_links = LinkStore.from_payload(payload.get("user_links", {}))
        transects = [TransectRecord.from_payload(d) for d in payload.get("transects", [])]
        rois = [RoiRecord.from_payload(d) for d in payload.get("rois", [])]
        cameras = {k: _camera_from_payload(v)
                   for k, v in payload.get("cameras", {}).items()}
        annotations = [AnnotationRecord.from_payload(d)
                       for d in payload.get("annotations", [])]
        reference_layers = [ReferenceLayerRecord.from_payload(d)
                            for d in payload.get("reference_layers", [])]
        p = cls(title=payload.get("title", ""), description=payload.get("description", ""),
                sources=sources, layers=layers, topologies=topologies, user_links=user_links,
                transects=transects, rois=rois, cameras=cameras, annotations=annotations,
                reference_layers=reference_layers,
                created=payload.get("created", ""),
                modified=payload.get("modified", ""),
                app_version=payload.get("app_version", ""))
        p._next_layer_id = max((l.layer_id for l in layers), default=-1) + 1
        p._next_source_id = max(
            (int(sid[3:]) for sid in sources if sid.startswith("src") and sid[3:].isdigit()),
            default=-1) + 1
        p._next_roi_id = max(
            (int(r.roi_id[3:]) for r in rois
             if r.roi_id.startswith("roi") and r.roi_id[3:].isdigit()),
            default=-1) + 1
        p._next_annotation_id = max(
            (a.annotation_id for a in annotations), default=-1) + 1
        p._next_ref_id = max(
            (int(r.ref_id[3:]) for r in reference_layers
             if r.ref_id.startswith("ref") and r.ref_id[3:].isdigit()),
            default=-1) + 1
        return p
