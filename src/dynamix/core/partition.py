"""Reversible keep/prune partitions over a dataset's id-universe (pure numpy/json, NO GUI/pyvista).

"Deleting" points is **non-destructive**: instead of removing ids from the immutable universe, they
move into a PRUNE set; **keep = universe − prune** is the derived complement (never stored, so the two
can't drift). One class serves every EQSelect dataset that is an id-universe -- earthquake ``event_id``
s and WTMM extrema chain/node ids alike -- so the same keep/prune machinery (and the same keep/prune
graph nodes) works across all of them.

Design mirrors :class:`eqselect.groups.GroupManager`: an immutable sorted-unique universe, fully
vectorized membership math (``np.isin`` / ``np.setdiff1d`` / ``np.union1d``), optional per-id prune
**reasons** for provenance (an id has exactly one -- its latest), and an exact JSON round-trip. The
only Python loops iterate over the (few) distinct *reasons*, never over ids.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRUNE_JSON = "prune.json"
SCHEMA = "prune/1"
DEFAULT_REASON = "unspecified"


class Partition:
    """A reversible keep/prune partition of an id-universe. ``prune``/``unprune`` move ids in/out of
    the prune set; ``kept``/``pruned`` are the two views; each prune carries a reason."""

    def __init__(self, universe, name: str = "dataset"):
        arr = np.asarray(universe)
        if arr.size == 0:
            arr = arr.astype("int64")            # avoid an empty float64 universe
        self.universe = np.unique(arr)           # sorted, unique, immutable membership set
        self.name = str(name)
        self._by_reason: dict[str, np.ndarray] = {}   # reason -> disjoint sorted-unique id array

    # ------------------------------------------------------------------ internals
    def _empty(self) -> np.ndarray:
        return np.empty(0, self.universe.dtype)

    def _clean(self, ids) -> np.ndarray:
        """Sorted-unique intersection of ``ids`` with the universe (vectorized, valid-only)."""
        arr = np.asarray(ids)
        if arr.size == 0:
            return self._empty()
        return np.intersect1d(arr, self.universe)

    def _drop_empty(self) -> None:
        self._by_reason = {r: a for r, a in self._by_reason.items() if a.size}

    # ------------------------------------------------------------------ inspection
    @property
    def pruned(self) -> np.ndarray:
        """All pruned ids (sorted-unique)."""
        if not self._by_reason:
            return self._empty()
        return np.unique(np.concatenate(list(self._by_reason.values())))

    @property
    def kept(self) -> np.ndarray:
        """The complement -- universe minus the prune set."""
        return np.setdiff1d(self.universe, self.pruned)

    @property
    def n_pruned(self) -> int:
        return int(self.pruned.size)

    @property
    def n_kept(self) -> int:
        return int(self.universe.size - self.pruned.size)

    def is_pruned(self, ids) -> np.ndarray:
        """Boolean mask (aligned to ``ids``) -- True where the id is pruned."""
        return np.isin(np.asarray(ids), self.pruned)

    def keep_mask(self, ids) -> np.ndarray:
        """Boolean mask (aligned to ``ids``) -- True where the id is KEPT (not pruned). Use this to
        AND prune into a display/selection mask."""
        return ~self.is_pruned(ids)

    def reasons(self) -> dict:
        """``reason -> sorted id list`` for the currently-pruned ids (JSON-friendly)."""
        return {r: a.tolist() for r, a in sorted(self._by_reason.items()) if a.size}

    def reason_counts(self) -> dict:
        """``reason -> count`` -- the breakdown rendered on the prune graph node."""
        return {r: int(a.size) for r, a in sorted(self._by_reason.items()) if a.size}

    # ------------------------------------------------------------------ mutations
    def prune(self, ids, reason: str | None = None) -> int:
        """Move ``ids`` into the prune set under ``reason`` (default 'unspecified'). Reassigning an
        already-pruned id updates its reason. Returns the number of *newly* pruned ids."""
        reason = reason or DEFAULT_REASON
        clean = self._clean(ids)
        if clean.size == 0:
            return 0
        newly = int(np.setdiff1d(clean, self.pruned).size)
        for r in list(self._by_reason):          # loop over the FEW distinct reasons, never over ids
            if r != reason:
                self._by_reason[r] = np.setdiff1d(self._by_reason[r], clean)
        self._by_reason[reason] = np.union1d(self._by_reason.get(reason, self._empty()), clean)
        self._drop_empty()
        return newly

    def unprune(self, ids) -> int:
        """Restore ``ids`` (remove from the prune set). Returns the number restored."""
        clean = self._clean(ids)
        if clean.size == 0:
            return 0
        restored = int(np.intersect1d(clean, self.pruned).size)
        for r in list(self._by_reason):
            self._by_reason[r] = np.setdiff1d(self._by_reason[r], clean)
        self._drop_empty()
        return restored

    def clear(self) -> int:
        """Restore everything (empty the prune set). Returns how many were pruned."""
        n = self.n_pruned
        self._by_reason = {}
        return n

    # ------------------------------------------------------------------ persistence
    def to_payload(self) -> dict:
        """JSON-serializable snapshot: the prune set (by reason) + provenance. keep is NOT stored."""
        return {
            "schema": SCHEMA,
            "name": self.name,
            "universe_size": int(self.universe.size),
            "n_pruned": self.n_pruned,
            "reasons": self.reasons(),
        }

    def save(self, out_dir, filename: str = PRUNE_JSON) -> None:
        """Write the prune partition JSON under ``out_dir``. Idempotent: when nothing is pruned it
        REMOVES any existing file (so a full restore + re-save doesn't leave a stale prune.json that
        would later resurrect the un-pruned points)."""
        target = Path(out_dir) / filename
        if not self._by_reason:
            target.unlink(missing_ok=True)               # no dir created just to check
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_payload(), indent=2))

    @classmethod
    def load(cls, universe, out_dir, filename: str = PRUNE_JSON) -> "Partition":
        """Reconstruct a partition over ``universe`` from a saved JSON (pruned ids re-clamped to the
        universe, so a universe change drops now-invalid ids gracefully)."""
        payload = json.loads((Path(out_dir) / filename).read_text())
        p = cls(universe, name=payload.get("name", "dataset"))
        for reason, ids in payload.get("reasons", {}).items():
            p.prune(ids, reason)
        return p
