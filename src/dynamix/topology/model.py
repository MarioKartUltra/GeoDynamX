# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The topology document: typed nodes and typed relation edges.

Two node sorts (geometric elements; dataset references) and two edge families in this slice
(spatial -- validated against the permitted Zlatanova sets at assertion time; derivation -- the
dataset paper-trail). "temporal" is a reserved family tag, not implemented. Anchors are small
JSON-able tuples of indices naming what an element is; the model never copies data arrays.
"""
from __future__ import annotations

import dataclasses

from dynamix.topology.codes import KINDS, permitted

DATASET = "dataset"
FAMILIES = ("spatial", "derivation")   # "temporal" reserved


@dataclasses.dataclass(frozen=True)
class Node:
    node_id: str
    kind: str
    space_dim: int | None = None
    anchor: tuple = ()


@dataclasses.dataclass(frozen=True)
class Edge:
    family: str
    a: str
    b: str
    code: int | None = None
    space_dim: int | None = None
    via: str | None = None
    #: The scale (in px -- the project's "scales are stored in pixels" invariant) at which two
    #: objects were first observed in contact, for a USER-asserted link (``topology/links.py``'s
    #: ``LinkStore``). Appended, optional, default None: every pre-existing edge (derivation edges,
    #: hand-asserted spatial edges from ``provenance``/tests) never sets it and is unaffected.
    scale_first_contact: float | None = None


class TopologyModel:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node_id: str, kind: str, space_dim: int | None = None,
                 anchor: tuple = ()) -> Node:
        if kind not in KINDS and kind != DATASET:
            raise ValueError(f"unknown kind {kind!r}; known: {KINDS + (DATASET,)}")
        if node_id in self.nodes:
            raise ValueError(f"node {node_id!r} already exists")
        node = Node(node_id, kind, space_dim, tuple(anchor))
        self.nodes[node_id] = node
        return node

    def _node(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            raise ValueError(f"unknown node {node_id!r}")
        return self.nodes[node_id]

    def assert_spatial(self, code: int, a: str, b: str, *, space_dim: int,
                       scale_first_contact: float | None = None) -> Edge:
        na, nb = self._node(a), self._node(b)
        try:
            allowed = permitted(na.kind, nb.kind, space_dim)
        except KeyError as e:
            raise ValueError(str(e)) from None
        if code not in allowed:
            raise ValueError(
                f"relation {code} is impossible between {na.kind} and {nb.kind} in dim "
                f"{space_dim}; permitted: {sorted(allowed)}"
            )
        edge = Edge("spatial", a, b, code=code, space_dim=space_dim,
                    scale_first_contact=scale_first_contact)
        self.edges.append(edge)
        return edge

    def add_derivation(self, child: str, parent: str, *, via: str) -> Edge:
        for nid in (child, parent):
            if self._node(nid).kind != DATASET:
                raise ValueError(f"derivation edges join dataset nodes; {nid!r} is "
                                 f"{self._node(nid).kind!r}")
        edge = Edge("derivation", child, parent, via=via)
        self.edges.append(edge)
        return edge

    def edges_of(self, node_id: str, *, family: str | None = None) -> list[Edge]:
        return [e for e in self.edges
                if node_id in (e.a, e.b) and (family is None or e.family == family)]

    def neighbors(self, node_id: str, *, family: str | None = None) -> list[str]:
        out = []
        for e in self.edges_of(node_id, family=family):
            other = e.b if e.a == node_id else e.a
            if other not in out:
                out.append(other)
        return out

    def nodes_where(self, *, kind: str | None = None, anchor0=None) -> list[Node]:
        return [n for n in self.nodes.values()
                if (kind is None or n.kind == kind)
                and (anchor0 is None or (n.anchor and n.anchor[0] == anchor0))]

    def to_payload(self) -> dict:
        return {
            "nodes": [{"node_id": n.node_id, "kind": n.kind, "space_dim": n.space_dim,
                       "anchor": list(n.anchor)} for n in self.nodes.values()],
            "edges": [{"family": e.family, "a": e.a, "b": e.b, "code": e.code,
                       "space_dim": e.space_dim, "via": e.via,
                       "scale_first_contact": e.scale_first_contact} for e in self.edges],
        }

    @classmethod
    def from_payload(cls, d: dict) -> "TopologyModel":
        m = cls()
        for n in d.get("nodes", []):
            m.add_node(n["node_id"], n["kind"], n.get("space_dim"),
                       tuple(n.get("anchor", ())))
        for e in d.get("edges", []):
            if e["family"] == "spatial":
                m.assert_spatial(e["code"], e["a"], e["b"], space_dim=e["space_dim"],
                                 scale_first_contact=e.get("scale_first_contact"))
            elif e["family"] == "derivation":
                m.add_derivation(e["a"], e["b"], via=e["via"])
            else:
                raise ValueError(f"unknown edge family {e['family']!r}")
        return m


def provenance(project) -> "TopologyModel":
    """Derivation edges read off a project's layers: each layer's output is a dataset derived
    from its source through its chain. Duck-typed (``project.sources`` / ``project.layers`` /
    ``layer.chain.steps``) so this stdlib-only module never imports dynamix.model. ``via`` is a
    readable, deterministic JSON description of the operations -- the paper trail; the engine's
    content-hash cache keys remain the audit mechanism for whether a trail still reproduces."""
    import json

    m = TopologyModel()
    for source_id in project.sources:
        m.add_node(f"src:{source_id}", DATASET, anchor=("source", source_id))
    for layer in project.layers:
        node = f"layer:{layer.layer_id}"
        m.add_node(node, DATASET, anchor=("layer", layer.layer_id))
        # default=str: a param arrives as whatever the caller put in it -- a numpy scalar out of
        # a device's own params is not JSON-serialisable, and a paper trail that raises is worse
        # than one that spells an int64 the way repr does (engine/cache._jsonable does the same).
        via = json.dumps([{"device": ref.device, "params": ref.params}
                          for ref in layer.chain.steps],
                         sort_keys=True, separators=(",", ":"), default=str)
        m.add_derivation(node, f"src:{layer.source_id}", via=via)
    return m
