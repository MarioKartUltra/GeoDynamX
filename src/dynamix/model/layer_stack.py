# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The layer stack: what the Map, Vector and Globe views draw, in which order, how opaque.

Rows point at existing content by id -- ``src:<source_id>`` (a dataset's raster),
``prod:<layer_id>`` (a layer's shown raster), ``ext:<layer_id>`` (its extrema),
``chn:<layer_id>`` (its vertical chains), ``ch:<bus_layer_id>:<n>`` (a bus channel). Groups nest;
a ``stack`` group is a bus. Children lists are top first: the first child draws on top. Display
state only -- analysis never reads it.
"""
from __future__ import annotations

import dataclasses

ROW_KINDS = ("raster", "product", "extrema", "chains", "channel")


@dataclasses.dataclass
class Row:
    id: str
    kind: str
    visible: bool = True
    opacity: float = 1.0
    moved: bool = False


@dataclasses.dataclass
class Group:
    id: str
    name: str
    kind: str = "group"                    # "group" or "stack" (a bus)
    visible: bool = True
    opacity: float = 1.0
    auto: bool = False                     # made by placement: removed once a removal empties it
    moved: bool = False
    children: list = dataclasses.field(default_factory=list)


class LayerStack:
    def __init__(self) -> None:
        self.root = Group("root", "")
        self._next_group = 0

    # -- lookup ----------------------------------------------------------------------------
    def _locate(self, node_id, group=None):
        group = self.root if group is None else group
        for i, child in enumerate(group.children):
            if child.id == node_id:
                return group, i
            if isinstance(child, Group):
                hit = self._locate(node_id, child)
                if hit is not None:
                    return hit
        return None

    def node(self, node_id):
        if node_id == "root":
            return self.root
        hit = self._locate(node_id)
        return None if hit is None else hit[0].children[hit[1]]

    def parent_of(self, node_id):
        hit = self._locate(node_id)
        return None if hit is None else hit[0]

    def row_ids(self) -> list[str]:
        return [r.id for r, _v, _o in self.drawn(include_hidden=True)]

    def drawn(self, *, include_hidden: bool = False) -> list:
        """``(row, visible, opacity)`` bottom first, with visibility and opacity inherited
        through every enclosing group."""
        out = []

        def walk(group, vis, opa):
            for child in reversed(group.children):
                v, o = vis and child.visible, opa * child.opacity
                if isinstance(child, Group):
                    walk(child, v, o)
                elif v or include_hidden:
                    out.append((child, v, o))

        walk(self.root, True, 1.0)
        return out

    # -- edits -----------------------------------------------------------------------------
    def insert(self, node, parent_id: str = "root", index: int = 0) -> None:
        self.node(parent_id).children.insert(index, node)

    def _contains(self, group, node_id) -> bool:
        return group.id == node_id or self._locate(node_id, group) is not None

    def move(self, node_id, parent_id, index) -> None:
        node = self.node(node_id)
        if isinstance(node, Group) and self._contains(node, parent_id):
            raise ValueError("a group cannot move into itself")
        old_parent, i = self._locate(node_id)
        target = self.node(parent_id)
        del old_parent.children[i]
        if old_parent is target and i < index:
            index -= 1
        target.children.insert(index, node)
        node.moved = True

    def remove(self, node_id) -> None:
        hit = self._locate(node_id)
        if hit is None:
            return
        parent, i = hit
        del parent.children[i]
        while parent is not self.root and parent.auto and not parent.children:
            grand, j = self._locate(parent.id)
            del grand.children[j]
            parent = grand

    def new_group(self, name: str, parent_id: str = "root", index: int = 0) -> Group:
        group = Group(f"grp:{self._next_group}", name)
        self._next_group += 1
        self.insert(group, parent_id, index)
        return group

    def ungroup(self, group_id) -> None:
        parent, i = self._locate(group_id)
        group = parent.children.pop(i)
        parent.children[i:i] = group.children

    # -- payload ---------------------------------------------------------------------------
    def to_payload(self, skip=frozenset()) -> list:
        def out(node):
            if isinstance(node, Group):
                return {"id": node.id, "name": node.name, "kind": node.kind,
                        "visible": node.visible, "opacity": node.opacity, "auto": node.auto,
                        "moved": node.moved,
                        "children": [out(c) for c in node.children if c.id not in skip]}
            return {"id": node.id, "kind": node.kind, "visible": node.visible,
                    "opacity": node.opacity, "moved": node.moved}

        return [out(c) for c in self.root.children if c.id not in skip]

    @classmethod
    def from_payload(cls, payload) -> "LayerStack":
        stack = cls()

        def build(d):
            if not isinstance(d, dict) or not isinstance(d.get("id"), str):
                return None
            common = {"visible": bool(d.get("visible", True)),
                      "opacity": float(d.get("opacity", 1.0)), "moved": bool(d.get("moved", False))}
            if "children" in d:
                kids = [k for k in (build(c) for c in d.get("children") or []) if k is not None]
                return Group(d["id"], str(d.get("name", "")), kind=d.get("kind", "group"),
                             auto=bool(d.get("auto", False)), children=kids, **common)
            return Row(d["id"], str(d.get("kind", "")), **common)

        stack.root.children = [n for n in (build(d) for d in payload or []) if n is not None]
        ids = [int(g.id[4:]) for g in _groups(stack.root) if g.id.startswith("grp:")
               and g.id[4:].isdigit()]
        stack._next_group = max(ids, default=-1) + 1
        return stack


def _groups(group):
    for child in group.children:
        if isinstance(child, Group):
            yield child
            yield from _groups(child)


@dataclasses.dataclass(frozen=True)
class LayerInfo:
    """What placement needs to know about one layer (the shell builds it)."""
    layer_id: int
    source_id: str
    produces: frozenset
    roi_id: str | None = None
    roi_label: str | None = None
    channels: tuple | None = None          # a bus: its send labels in send order
    shows_raster: bool = False             # its tool's Show can put a raster output up


_PRODUCT = frozenset({"HMap", "Components"})


def rows_for(info: LayerInfo) -> list:
    """``(row_id, kind)`` a layer contributes, top first."""
    out = []
    if "VChains" in info.produces:
        out.append((f"chn:{info.layer_id}", "chains"))
    if "ExtremaStack" in info.produces:
        out.append((f"ext:{info.layer_id}", "extrema"))
    if info.produces & _PRODUCT or info.shows_raster:
        out.append((f"prod:{info.layer_id}", "product"))
    return out


def _auto_group(stack, gid, name, parent_id="root", kind="group"):
    group = stack.node(gid)
    if group is None:
        group = Group(gid, name, kind=kind, auto=True)
        stack.insert(group, parent_id, 0)
    return group


def reconcile(stack: LayerStack, sources, layers, near=None) -> int:
    """Give content without a row a row by the default placement, and remove rows whose
    content is gone. Rows already present never move. ``near`` maps a new row id to an
    existing row id: the new row goes to the top of that row's group (a fork). Returns the
    number of rows removed."""
    near = near or {}
    labels = dict(sources)
    wanted = {f"src:{sid}" for sid, _label in sources}
    for info in layers:
        wanted.update(rid for rid, _k in rows_for(info))
        wanted.update(f"ch:{info.layer_id}:{n}" for n in range(len(info.channels or ())))
    gone = [rid for rid in stack.row_ids() if rid not in wanted]
    for rid in gone:
        stack.remove(rid)

    for sid, label in sources:
        rid = f"src:{sid}"
        if stack.node(rid) is not None:
            continue
        anchor = near.get(rid)
        parent = stack.parent_of(anchor) if anchor else None
        if parent is None:
            parent = _auto_group(stack, f"ds:{sid}", label)
            stack.insert(Row(rid, "raster"), parent.id, len(parent.children))
        else:
            stack.insert(Row(rid, "raster"), parent.id, 0)

    for info in layers:
        if info.channels is not None:
            target = _auto_group(stack, f"stack:{info.layer_id}", "Stack", kind="stack")
            for n in range(len(info.channels)):
                rid = f"ch:{info.layer_id}:{n}"
                if stack.node(rid) is None:
                    stack.insert(Row(rid, "channel"), target.id, len(target.children))
        else:
            target = _auto_group(stack, f"ds:{info.source_id}",
                                 labels.get(info.source_id, info.source_id))
            if info.roi_id is not None:
                target = _auto_group(stack, f"roig:{info.roi_id}", f"ROI {info.roi_label}",
                                     parent_id=target.id)
        for rid, kind in reversed(rows_for(info)):
            if stack.node(rid) is None:
                stack.insert(Row(rid, kind), target.id, 0)
    return len(gone)
