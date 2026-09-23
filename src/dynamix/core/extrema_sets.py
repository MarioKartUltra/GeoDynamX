# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Named sets of WTMM topo-extrema chains (pure numpy/json, NO GUI/pyvista).

The analogue of :class:`eqselect.groups.GroupManager` for the topo-extrema overlay: where a *group*
is a subset of the catalog's ``event_id`` universe, an *extrema set* is a named / coloured /
toggleable subset of the extrema **chains** produced by :func:`topo_wtmm.export_extrema` and loaded
via :func:`dynamix.core.extrema_io.load_topo_extrema`.

Each set carries two id lists -- ``h_chains`` (within-scale contours, "L" in the sketch) and
``v_chains`` (cross-scale maxima lines, "ℓ") -- because a selection can mix both (a by-V lineage,
a by-H scale-slice, or -- once Stage 3 lands the H↔V incidence -- their cross-query). Ids index the
flattened chain lists of the *source* ``.npz``; the manager records that provenance (``region`` +
chain counts) so a reloaded set is interpretable.

Design mirrors ``GroupManager``: monotonic ``set_id`` counter, a fixed qualitative palette keyed by
id, vectorized membership math (``np.unique`` / ``np.intersect1d``), and an exact JSON round-trip.
The only Python loops iterate over the (few) sets, never over chains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EXTREMA_SETS_JSON = "extrema_sets.json"
SCHEMA = "extrema_sets/1"

# Fixed 10-colour qualitative palette (matplotlib "tab10"); assigned by ``set_id % len`` so the same
# id always maps to the same colour, deterministically.
_PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def _default_color(set_id: int) -> str:
    return _PALETTE[int(set_id) % len(_PALETTE)]


def _default_name(set_id: int) -> str:
    return f"Extrema {int(set_id)}"


#: EQSelect-era flat keys, migrated into ``tags`` on load. See :func:`_migrate_tags`.
_LEGACY_TAG_KEYS = ("slab", "group")


def _clean_tags(tags) -> dict[str, str]:
    """Normalise a tag mapping to ``str -> str``, dropping empty values.

    ``""`` means *no tag*, so it must never persist as a dangling key — under EQSelect that would
    have emitted a meaningless ``[[slab_unnamed]]`` graph edge.
    """
    return {str(k): str(v) for k, v in (tags or {}).items() if v}


def _migrate_tags(rec: dict) -> dict[str, str]:
    """Tags for one serialised set, migrating EQSelect's flat ``slab`` / ``group`` keys.

    Those artifacts exist on disk in the slab repo today, so dropping the keys silently would lose
    real data. An explicit ``tags`` entry always wins over a legacy key of the same name.
    """
    tags = _clean_tags(rec.get("tags"))
    for key in _LEGACY_TAG_KEYS:
        if rec.get(key):
            tags.setdefault(key, str(rec[key]))
    return tags


@dataclass(eq=False)
class ExtremaSet:
    """A named/coloured/toggleable subset of extrema chains.

    ``h_chains`` / ``v_chains`` are always sorted-unique 1-D ``int64`` arrays of valid chain indices.
    (``eq=False`` because the array fields make dataclass equality ambiguous.)
    """

    set_id: int
    name: str
    color: str
    visible: bool
    h_chains: np.ndarray
    v_chains: np.ndarray
    tags: dict[str, str] = field(default_factory=dict)
    """Free-form metadata, e.g. ``{"slab": "Kermadec", "group": "13a"}``.

    Generalised from EQSelect's fixed ``slab`` / ``group`` fields, which existed to emit
    ``[[slab_<x>]]`` / ``[[group_<x>]]`` Obsidian graph edges. The concept survives; the
    earthquake vocabulary does not. An empty value is never stored — ``""`` means no tag.
    """


class ExtremaSetManager:
    """Owns the set of extrema sets defined over a source ``.npz``'s H/V chain universe."""

    def __init__(self, n_h_chains: int, n_v_chains: int, region: str = "region", n_scales: int = 1):
        self.n_h_chains = int(n_h_chains)
        self.n_v_chains = int(n_v_chains)
        self.region = str(region)
        self.n_scales = int(n_scales)
        self._sets: dict[int, ExtremaSet] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ internals
    def _clean(self, ids, n: int) -> np.ndarray:
        """Sorted-unique ids restricted to the valid range ``[0, n)`` (vectorized)."""
        arr = np.asarray(ids if ids is not None else [], dtype=np.int64).ravel()
        if arr.size == 0:
            return np.zeros(0, np.int64)
        arr = np.unique(arr)
        return arr[(arr >= 0) & (arr < n)]

    def _get(self, sid: int) -> ExtremaSet:
        if sid not in self._sets:
            raise KeyError(f"no extrema set with id {sid!r}")
        return self._sets[sid]

    # ------------------------------------------------------------------ inspection
    @property
    def sets(self) -> list[ExtremaSet]:
        """All sets, ordered by ``set_id``."""
        return [self._sets[k] for k in sorted(self._sets)]

    def __len__(self) -> int:
        return len(self._sets)

    def __contains__(self, sid) -> bool:
        return sid in self._sets

    # ------------------------------------------------------------------ mutations
    def add_set(self, h_chains=None, v_chains=None, name=None, color=None,
                tags=None) -> int:
        """Create a set from ``h_chains`` / ``v_chains``; returns its fresh ``set_id``.

        Ids are de-duplicated and clamped to the valid chain range. ``name`` / ``color`` default to a
        generated label and the palette colour for the new id. ``tags`` is optional free-form
        metadata; empty values are dropped, mirroring :meth:`set_tags`.
        """
        sid = self._next_id
        self._next_id += 1
        self._sets[sid] = ExtremaSet(
            set_id=sid,
            name=_default_name(sid) if name is None else name,
            color=_default_color(sid) if color is None else color,
            visible=True,
            h_chains=self._clean(h_chains, self.n_h_chains),
            v_chains=self._clean(v_chains, self.n_v_chains),
            tags=_clean_tags(tags),
        )
        return sid

    def relabel_tag(self, key, old, new) -> int:
        """Update every set whose ``key`` tag equals ``old`` so it reads ``new``, keeping a rename
        from dangling references to the old value. Returns the number of sets updated.

        Only the named ``key`` is considered: a different key holding the same value is untouched.
        """
        n = 0
        for s in self._sets.values():
            if s.tags.get(key) == old:
                s.tags[key] = new
                n += 1
        return n

    def set_tags(self, sid, **tags) -> None:
        """Set or remove tags on a set. An empty value removes that key; keys not named here are
        left unchanged, so updating one tag never disturbs the others."""
        s = self._get(sid)
        for k, v in tags.items():
            if v:
                s.tags[str(k)] = str(v)
            else:
                s.tags.pop(str(k), None)

    def assign(self, sid, h_chains=None, v_chains=None) -> None:
        """Union more chains into an existing set (de-duplicated, stays sorted)."""
        s = self._get(sid)
        s.h_chains = np.union1d(s.h_chains, self._clean(h_chains, self.n_h_chains))
        s.v_chains = np.union1d(s.v_chains, self._clean(v_chains, self.n_v_chains))

    def rename(self, sid, name) -> None:
        self._get(sid).name = name

    def recolor(self, sid, color) -> None:
        self._get(sid).color = color

    def set_visible(self, sid, vis) -> None:
        self._get(sid).visible = bool(vis)

    def delete(self, sid) -> None:
        self._get(sid)                       # validate -> KeyError if absent
        del self._sets[sid]

    def merge(self, sids, name=None) -> int:
        """Union the members of ``sids`` into a new set, delete the originals, return the new id."""
        sids = list(sids)
        members = [self._get(s) for s in sids]                       # validates every id first
        h = np.unique(np.concatenate([m.h_chains for m in members])) if members else np.zeros(0, np.int64)
        v = np.unique(np.concatenate([m.v_chains for m in members])) if members else np.zeros(0, np.int64)
        new_id = self.add_set(h, v, name=name)
        for s in sids:
            del self._sets[s]
        return new_id

    # ------------------------------------------------------------------ persistence
    def to_payload(self) -> dict:
        """JSON-serializable snapshot (sets + source provenance)."""
        return {
            "schema": SCHEMA,
            "region": self.region,
            "n_scales": self.n_scales,
            "n_h_chains": self.n_h_chains,
            "n_v_chains": self.n_v_chains,
            "sets": [
                {"set_id": int(s.set_id), "name": s.name, "color": s.color,
                 "visible": bool(s.visible), "tags": dict(s.tags),
                 "h_chains": s.h_chains.tolist(), "v_chains": s.v_chains.tolist()}
                for s in self.sets
            ],
        }

    def save(self, out_dir) -> None:
        """Write ``extrema_sets.json`` (the full snapshot, including provenance) under ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / EXTREMA_SETS_JSON).write_text(json.dumps(self.to_payload(), indent=2))

    @classmethod
    def load(cls, out_dir) -> "ExtremaSetManager":
        """Reconstruct a manager from ``extrema_sets.json`` (exact round-trip)."""
        payload = json.loads((Path(out_dir) / EXTREMA_SETS_JSON).read_text())
        mgr = cls(payload.get("n_h_chains", 0), payload.get("n_v_chains", 0),
                  region=payload.get("region", "region"), n_scales=payload.get("n_scales", 1))
        for rec in payload.get("sets", []):
            sid = int(rec["set_id"])
            mgr._sets[sid] = ExtremaSet(
                set_id=sid, name=rec["name"], color=rec["color"], visible=bool(rec["visible"]),
                h_chains=np.asarray(rec.get("h_chains", []), np.int64),
                v_chains=np.asarray(rec.get("v_chains", []), np.int64),
                tags=_migrate_tags(rec),
            )
            mgr._next_id = max(mgr._next_id, sid + 1)
        return mgr
