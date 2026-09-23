# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The provenance tree: what came from what, in the order it was made.

This is the shape the per-source floating inspector renders:
one node per SOURCE dataset, the layers over it, and inside each layer one sub-layer row per
product of its chain. **The ordering IS the processing provenance** -- so
nothing here sorts, dedupes or prettifies: sources come in ``project.sources`` insertion order,
layers in ``project.layers`` order, a layer's products in its chain's own step order, and a child
layer (an ROI cut from another layer, ``Layer.parent_id``) nests under ITS OWN parent recursively.
That is the same three-level grouping ``shell/layer_panel.py``'s ``add_layer_row`` already draws;
this module is the headless half of it, so the inspector and the layer list cannot disagree.

**Registry-free by construction.** Products are read off ``Chain.steps``' plain ``device`` STRINGS
(``model/chain.py``), never through ``Chain.transforms``/``filters``, which call ``get_device`` to
split the chain and raise ``KeyError`` wherever the devices of a saved project are not registered
in this process. A provenance view is a view of a DOCUMENT; it must render a project whose devices
this build has never heard of, and a test asserts exactly that.

Pure stdlib, no Qt: the model layer stays importable headless.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path


def sublayer_key(layer_id: int, index: int) -> str:
    """The stable id of ONE product row: layer plus its position in that layer's chain.

    This function and :func:`raster_key` are the ONLY keys the inspector's persisted state
    (``model/inspector_state.py``) and its tree rows (``shell/inspector.py``) may use, so the
    remembered visibility of a row and the row itself cannot come to mean different things.
    """
    return f"{layer_id}:{index}"


def raster_key(source_id: str) -> str:
    """The key of the source's OWN raster row -- the thing every product below it derives from."""
    return f"{source_id}:raster"


@dataclasses.dataclass(frozen=True)
class ProductNode:
    """One product of one chain step: ``index`` is its position in ``Chain.steps``, ``device`` the
    registry NAME as written in the document (a string, never a resolved device)."""

    key: str
    device: str
    index: int


@dataclasses.dataclass(frozen=True)
class LayerNode:
    """One layer, its products in chain order, and the layers cut FROM it (ROI children)."""

    layer_id: int
    name: str
    roi_id: str | None = None
    products: tuple[ProductNode, ...] = ()
    children: tuple["LayerNode", ...] = ()


@dataclasses.dataclass(frozen=True)
class SourceNode:
    """One source dataset -- what an inspector window opens on (SOURCE, not per
    layer). ``label`` is what to title it with; the raster's own row key is
    ``raster_key(source_id)``, not stored here, because it is derivable and a second copy of a key
    is a second opinion about it."""

    source_id: str
    label: str
    layers: tuple[LayerNode, ...] = ()


def _label_for(source) -> str:
    """The source's own label, else its file stem, else the bare id -- a node always names
    something. The stem is what ``layer_panel._ensure_source_header`` shows; the user-set
    ``SourceRef.label`` wins over it here, since a user who named a dataset meant that name."""
    stem = Path(source.path).stem if source.path else ""
    return source.label or stem or source.source_id


def _products_of(layer) -> tuple[ProductNode, ...]:
    return tuple(
        ProductNode(key=sublayer_key(layer.layer_id, i), device=step.device, index=i)
        for i, step in enumerate(layer.chain.steps)
    )


def _layer_node(layer, children_by_parent: dict[int, list]) -> LayerNode:
    return LayerNode(
        layer_id=layer.layer_id,
        name=layer.name,
        roi_id=layer.roi_id,
        products=_products_of(layer),
        children=tuple(_layer_node(child, children_by_parent)
                       for child in children_by_parent.get(layer.layer_id, ())),
    )


def provenance_tree(project) -> list[SourceNode]:
    """The whole document as source -> layer -> product rows, in processing order.

    A source with no layers still yields a node: the raster exists and is inspectable on its own.
    A layer whose ``parent_id`` names no layer in the project is treated as top-level rather than
    dropped -- a row that silently vanishes from a provenance view is worse than one shown a level
    too high.
    """
    known = {layer.layer_id for layer in project.layers}
    children_by_parent: dict[int, list] = {}
    roots_by_source: dict[str, list] = {}
    for layer in project.layers:
        if layer.parent_id is not None and layer.parent_id in known:
            children_by_parent.setdefault(layer.parent_id, []).append(layer)
        else:
            roots_by_source.setdefault(layer.source_id, []).append(layer)

    return [
        SourceNode(
            source_id=source_id,
            label=_label_for(source),
            layers=tuple(_layer_node(layer, children_by_parent)
                         for layer in roots_by_source.get(source_id, ())),
        )
        for source_id, source in project.sources.items()
    ]
