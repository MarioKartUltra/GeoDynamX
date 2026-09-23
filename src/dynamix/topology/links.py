# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""User-asserted topology links: a person clicking two chains and saying "these touch" (or
"overlap", or "these are disjoint but worth recording anyway").

Stdlib + numpy only -- mirrors ``topology/model.py``'s own Qt-free posture, so this module (and
the shell panel that drives it) never drags ``dynamix.shell`` or its Qt dependency into the
analysis core.

**Nodes name geometric objects, not results.** ``ObjRef(layer_id, transform, kind, obj_id)``
identifies "chain #7 out of layer L2's mz_edges transform" -- ``transform`` is the layer chain's
last TRANSFORM device name (the thing that actually produced the geometry), ``obj_id`` is the
object's index within that transform's output (a chain index, for every kind this slice ever
produces -- ``kind`` is always ``"line"`` here, chains being 1-D curves regardless of which
transform drew them). ``node_id()`` is the string identity :class:`~dynamix.topology.model.
TopologyModel` keys nodes by; two ``ObjRef``s that describe the same object always produce the
same id, so re-linking the same pair of chains after a redraw reuses the same nodes rather than
minting duplicates.

**suggest_code's three matrices.** Derived from :mod:`dynamix.topology.codes`'s own documented
9-intersection convention (module docstring + ``point_code``'s worked example, ``codes.py:9-11,
216-225``) rather than copied from its ``_NAMED`` table (that table only names R^1 line-line
codes; the codes are dimension-INDEPENDENT bit patterns, only which ones are *geometrically
possible* changes with dimension, which is exactly what ``permitted()`` gates). Every pair not
listed is empty by construction -- only asserting what genuinely differs between the three
regimes, the same discipline ``point_code`` already models:

- **disjoint** -- ``{ee, eb, ei, be, ie}``: both exteriors overlap (true of any two bounded curves
  in the plane), each one's boundary/interior sits entirely in the other's exterior, and NEITHER
  boundary touches the other at all (``bb=ib=bi=ii=0``). Encodes to 31 -- the same integer the
  R^1 table calls "disjoint", because it is the identical bit pattern; only the surrounding
  ``permitted("line","line",2)`` gate differs by dimension, and 31 is a member of it.
- **touch ("meet")** -- disjoint's own five pairs plus ``bb`` (the two boundaries share a point --
  an endpoint touching, interiors and interior/boundary crossings all still empty). Encodes to
  287, the R^1 table's own "meet".
- **overlap** -- ``{ii, bi, ib, ee, eb, ei, be, ie}``: interiors intersect, and each one's
  boundary meets the other's interior (``bi``, ``ib``) -- everything a genuine crossing implies --
  while the two boundaries themselves happen not to coincide (``bb=0``, the generic case; a
  boundary-on-boundary crossing is a different, rarer code this coarse heuristic does not
  distinguish). Encodes to 255, the R^1 table's own "overlap".

All three are asserted, at import time, to be members of ``permitted("line", "line", 2)`` -- the
hard gate the design mandates: a suggestion this module could ever emit that is not a real 2-D
line-line relation is a bug in the matrices above, and it fails loudly before a single test runs,
let alone before it reaches ``LinkStore.link``'s own ``assert_spatial`` gate a second time.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamix.topology.codes import LINE, encode, permitted
from dynamix.topology.model import TopologyModel

__all__ = ["ObjRef", "LinkEntry", "LinkStore", "suggest_code"]


@dataclasses.dataclass(frozen=True)
class ObjRef:
    """One geometric object, named well enough to re-find: which layer, which transform produced
    it, what kind of object it is (must be a member of ``codes.KINDS`` -- validated here, at
    construction, so a bad kind never reaches ``TopologyModel.add_node`` to fail there instead
    with a less specific error), and its index within that transform's own output."""

    layer_id: object
    transform: str
    kind: str
    obj_id: object

    def __post_init__(self) -> None:
        from dynamix.topology.codes import KINDS

        if self.kind not in KINDS:
            raise ValueError(f"ObjRef kind {self.kind!r} is not a topology kind; known: {KINDS}")

    def node_id(self) -> str:
        return f"{self.layer_id}:{self.transform}:{self.kind}:{self.obj_id}"


@dataclasses.dataclass(frozen=True)
class LinkEntry:
    """One user-asserted link, as :meth:`LinkStore.links_for_layer`/iteration hand it back."""

    a: ObjRef
    b: ObjRef
    code: int
    space_dim: int
    scale_first_contact: float | None = None


# --- suggest_code's three 9-intersection matrices (see module docstring) -----------------------

_DISJOINT_PAIRS = frozenset({"ee", "eb", "ei", "be", "ie"})
_TOUCH_PAIRS = _DISJOINT_PAIRS | {"bb"}
_OVERLAP_PAIRS = frozenset({"ii", "bi", "ib", "ee", "eb", "ei", "be", "ie"})

_DISJOINT_CODE = encode(_DISJOINT_PAIRS)
_TOUCH_CODE = encode(_TOUCH_PAIRS)
_OVERLAP_CODE = encode(_OVERLAP_PAIRS)

# The hard gate (design): a matrix that produces a code outside the permitted line-line-in-R^2 set
# is a bug in THIS module, not something a caller should ever discover through a raised ValueError
# three calls later. Fails at import time.
_LL2 = permitted(LINE, LINE, 2)
for _name, _code in (("disjoint", _DISJOINT_CODE), ("touch", _TOUCH_CODE),
                     ("overlap", _OVERLAP_CODE)):
    if _code not in _LL2:
        raise AssertionError(
            f"suggest_code's {_name!r} matrix encodes to {_code}, which is not in "
            f"permitted('line','line',2) -- the matrix in links.py is wrong"
        )
del _name, _code


def _bbox(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pts.min(axis=0), pts.max(axis=0)


def _bbox_overlap_with_mutual_interior_samples(a: np.ndarray, b: np.ndarray) -> bool:
    """Bounding-box overlap, PLUS at least one point of each set actually falling inside the
    other's box -- the design's "bounding-box overlap with both interiors sampled inside each
    other". Box overlap alone would also fire for two chains whose boxes touch at a corner with
    neither curve genuinely entering the other's span; requiring a real sample from each side
    inside the other's box is the cheap way to tell "crossing" from "nearby"."""
    amin, amax = _bbox(a)
    bmin, bmax = _bbox(b)
    if not (np.all(amin <= bmax) and np.all(bmin <= amax)):
        return False
    a_in_b = np.any(np.all((a >= bmin) & (a <= bmax), axis=1))
    b_in_a = np.any(np.all((b >= amin) & (b <= amax), axis=1))
    return bool(a_in_b and b_in_a)


def suggest_code(pts_a, pts_b, contact_scale: float) -> int:
    """A 9-intersection code for the relation between two point sets ``pts_a``/``pts_b`` (each an
    ``(N, 2)`` array of a chain's own pixel coordinates), restricted to the three regimes this
    heuristic can actually tell apart: disjoint, touching, overlapping (see module docstring for
    the exact matrices). Never returns a code outside ``permitted("line", "line", 2)`` -- the
    module-level self-check above already proves the three constants satisfy that; this function
    only ever returns one of them.

    ``contact_scale`` is the distance (in px -- the project's "scales are stored in pixels") below
    which the two sets count as touching rather than disjoint -- ordinarily the analysis scale
    the two chains were drawn at (see ``main_window``'s own derivation), so "in contact" means
    "closer than the feature size currently being examined", not an arbitrary pixel count.
    """
    a = np.atleast_2d(np.asarray(pts_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(pts_b, dtype=np.float64))
    if _bbox_overlap_with_mutual_interior_samples(a, b):
        return _OVERLAP_CODE
    min_dist = np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1))
    return _DISJOINT_CODE if min_dist > contact_scale else _TOUCH_CODE


class LinkStore:
    """A live collection of user-asserted links over a private :class:`TopologyModel`.

    The wrapped model is honestly append-only (``TopologyModel`` never deletes an edge, by
    design -- see its own module docstring); this store's ``_entries`` list is the actual source
    of truth for "what links exist right now", and :meth:`unlink` mutates THAT. ``to_payload``
    rebuilds a fresh model from ``_entries`` before serialising, so a payload never carries an
    edge the caller already unlinked -- ``unlink`` is honest about what gets saved without ever
    needing to delete anything out of a ``TopologyModel``'s own edge list.

    Each node is stamped with an ``anchor`` -- ``(layer_id, transform, kind, obj_id)``, the exact
    four fields of the ``ObjRef`` that named it -- so a payload round-trip can rebuild the
    ``ObjRef``s themselves from the model's own nodes, with no second registry to keep in sync.
    """

    def __init__(self) -> None:
        self._model = TopologyModel()
        self._entries: list[LinkEntry] = []

    def _ensure_node(self, ref: ObjRef) -> None:
        node_id = ref.node_id()
        if node_id not in self._model.nodes:
            self._model.add_node(node_id, ref.kind,
                                 anchor=(ref.layer_id, ref.transform, ref.kind, ref.obj_id))

    def link(self, a: ObjRef, b: ObjRef, code: int, *, space_dim: int = 2,
             scale_first_contact: float | None = None) -> LinkEntry:
        """Assert a relation between ``a`` and ``b``. Raises ``ValueError`` for a ``code``
        impossible between their kinds at ``space_dim`` (delegated to ``TopologyModel.
        assert_spatial`` -- the one validation authority, not re-implemented here)."""
        self._ensure_node(a)
        self._ensure_node(b)
        self._model.assert_spatial(code, a.node_id(), b.node_id(), space_dim=space_dim,
                                   scale_first_contact=scale_first_contact)
        entry = LinkEntry(a, b, code, space_dim, scale_first_contact)
        self._entries.append(entry)
        return entry

    def unlink(self, index: int) -> None:
        """Drop the link at ``index`` (the position :meth:`all` / a panel's own row order gives
        it) from the store's live list. The wrapped model keeps the edge -- see the class
        docstring -- so nothing here needs it to support deletion."""
        del self._entries[index]

    def links_for_layer(self, layer_id) -> list[LinkEntry]:
        """Every current link naming ``layer_id`` on either end."""
        return [e for e in self._entries if e.a.layer_id == layer_id or e.b.layer_id == layer_id]

    def all(self) -> list[LinkEntry]:
        """Every current link, in creation order -- a copy, so a caller cannot mutate the store's
        own bookkeeping through it."""
        return list(self._entries)

    def to_payload(self) -> dict:
        fresh = TopologyModel()
        for e in self._entries:
            for ref in (e.a, e.b):
                node_id = ref.node_id()
                if node_id not in fresh.nodes:
                    fresh.add_node(node_id, ref.kind,
                                   anchor=(ref.layer_id, ref.transform, ref.kind, ref.obj_id))
            fresh.assert_spatial(e.code, e.a.node_id(), e.b.node_id(), space_dim=e.space_dim,
                                 scale_first_contact=e.scale_first_contact)
        return fresh.to_payload()

    @classmethod
    def from_payload(cls, d: dict) -> "LinkStore":
        model = TopologyModel.from_payload(d)
        store = cls()
        store._model = model
        for edge in model.edges:
            if edge.family != "spatial":
                continue
            a_ref = ObjRef(*model.nodes[edge.a].anchor)
            b_ref = ObjRef(*model.nodes[edge.b].anchor)
            store._entries.append(
                LinkEntry(a_ref, b_ref, edge.code, edge.space_dim, edge.scale_first_contact))
        return store
