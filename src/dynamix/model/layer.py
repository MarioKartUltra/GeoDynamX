# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A layer: one source raster and the chain applied to it. The Ableton track."""
from __future__ import annotations

import dataclasses

from dynamix.model.chain import Chain


@dataclasses.dataclass
class Layer:
    layer_id: int
    name: str
    source_id: str
    chain: Chain = dataclasses.field(default_factory=Chain)
    visible: bool = True
    tags: dict[str, str] = dataclasses.field(default_factory=dict)
    parent_id: int | None = None
    #: The :class:`~dynamix.model.project.RoiRecord` this layer was cut from, or ``None`` for a
    #: layer that is not an ROI. A string id -- ``"roi0"``
    #: -- following ``SourceRef.source_id``'s convention rather than ``parent_id``'s int, because
    #: an ROI id must be embeddable in an ``ObjRef``-style reference. Read via ``.get`` on load
    #: (below) so a project saved before ROIs were objects opens unchanged, as ``None``.
    roi_id: str | None = None

    def to_payload(self) -> dict:
        return {"layer_id": int(self.layer_id), "name": self.name,
                "source_id": self.source_id, "chain": self.chain.to_payload(),
                "visible": bool(self.visible), "tags": dict(self.tags),
                "parent_id": None if self.parent_id is None else int(self.parent_id),
                "roi_id": self.roi_id}

    @classmethod
    def from_payload(cls, d: dict) -> "Layer":
        return cls(layer_id=int(d["layer_id"]), name=d["name"], source_id=d["source_id"],
                   chain=Chain.from_payload(d.get("chain", {})),
                   visible=bool(d.get("visible", True)), tags=dict(d.get("tags") or {}),
                   parent_id=d.get("parent_id"), roi_id=d.get("roi_id"))
