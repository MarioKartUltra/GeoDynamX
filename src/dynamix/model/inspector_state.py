# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""What a floating inspector remembers between sessions: open, where, at what scale.

Keyed by SOURCE id, because the inspector is per source dataset and not per layer ("the toggle sits on the source"), with one :class:`SubLayerState` per row of the source's
provenance tree -- the source raster and every product derived from it, each with its own on/off
and opacity.

**This is a PREFERENCE, not a measurement**. It persists in
``Settings.view_options["inspectors"]`` (``shell/settings.py``), NOT in the project document: a
window position is a fact about one machine, while a remembered camera viewpoint (``Project.
cameras``) is closer to a measurement and lives in the document. ``view_options`` is already a
free-form JSON dict round-tripped by ``dataclasses.asdict``/``json.dumps``, so storing this there
needs no schema change to ``settings.py`` at all -- what it needs is a SHAPE, and this module is
it: headless, stdlib-only, testable without Qt.

Three honesty rules, all of them consequences of "preference, not measurement":

1. A MISSING entry means the window is closed. There is no "unknown" state to surface.
2. A MALFORMED entry is DROPPED silently, never raised -- a preference cannot be a fault.
One unreadable window position must not stop the app opening, and the visible
   consequence of the drop is the honest one: that inspector comes back closed.
3. UNKNOWN keys are ignored rather than rejected, so a settings file written by a later build
   still opens here (sec 8 rule 7, additive-only).

``follow_master`` defaults to ``True``: an inspector joins the master transport's scale sweep
unless the user takes it off, which is what comparing two extrema sets across scales wants. The master drives the scale INDEX only; coefficient calibration is a later slice.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SubLayerState:
    """One row inside an inspector window: the source's raster, or one product of one chain.

    Rows are addressed by ``model.provenance``'s :func:`~dynamix.model.provenance.raster_key` and
    :func:`~dynamix.model.provenance.sublayer_key` -- the ONLY key functions this state and the
    inspector's tree rows may use, so a remembered visibility and the row it belongs to cannot
    come to mean different things.

    Defaults are visible and fully opaque: a product the user has never touched shows as the
    chain produced it.
    """

    visible: bool = True
    opacity: float = 1.0

    def to_payload(self) -> dict:
        return {"visible": bool(self.visible), "opacity": float(self.opacity)}

    @classmethod
    def from_payload(cls, d: dict) -> "SubLayerState":
        """Raises on anything that is not a readable row; the CALLER drops the whole entry
        (see :func:`inspectors_from_payload`), which is rule 2 above."""
        if not isinstance(d, dict):
            raise TypeError("a sub-layer state is a JSON object")
        return cls(visible=bool(d.get("visible", True)), opacity=float(d.get("opacity", 1.0)))


@dataclasses.dataclass
class InspectorState:
    """One source's inspector window, as it should come back next time.

    ``geometry`` is ``(x, y, width, height)`` in screen coordinates -- what ``QWidget.geometry()``
    reports and ``setGeometry`` accepts -- or ``None`` for "wherever the platform puts it", which
    is what a window that has never been moved deserves.

    ``scale_idx`` is an INDEX into the result's scales, never a scale value in pixels: scales are
    stored in pixels and converted for display, and an index is the one
    thing that survives a re-run whose scale vector changed length. The range it is read against
    is data-derived from the result, never a constant.
    """

    source_id: str
    open: bool = False
    geometry: tuple[int, int, int, int] | None = None
    scale_idx: int = 0
    follow_master: bool = True
    sublayers: dict[str, SubLayerState] = dataclasses.field(default_factory=dict)

    def to_payload(self) -> dict:
        """JSON-safe: the geometry tuple flattens to a list, since JSON has no tuple.

        ``source_id`` is deliberately NOT written into the entry -- the mapping key already is the
        source id, and a second copy of a key is a second opinion about it.
        """
        return {"open": bool(self.open),
                "geometry": (None if self.geometry is None
                             else [int(v) for v in self.geometry]),
                "scale_idx": int(self.scale_idx),
                "follow_master": bool(self.follow_master),
                "sublayers": {str(k): v.to_payload() for k, v in self.sublayers.items()}}

    @classmethod
    def from_payload(cls, source_id: str, d: dict) -> "InspectorState":
        """The inverse of :meth:`to_payload`; ``source_id`` comes from the mapping key.

        Every field is ``.get``-defaulted, so an entry written before a field existed reads as
        that field's default. Anything genuinely unreadable RAISES here and is dropped by
        :func:`inspectors_from_payload` -- the two halves are split that way so the drop happens
        in exactly one place and can be tested there.
        """
        if not isinstance(d, dict):
            raise TypeError("an inspector entry is a JSON object")
        geometry = d.get("geometry")
        if geometry is not None:
            x, y, w, h = (int(v) for v in geometry)
            geometry = (x, y, w, h)
        sublayers = d.get("sublayers") or {}
        if not isinstance(sublayers, dict):
            raise TypeError("sublayers is a JSON object keyed by provenance row key")
        return cls(source_id=str(source_id),
                   open=bool(d.get("open", False)),
                   geometry=geometry,
                   scale_idx=int(d.get("scale_idx", 0)),
                   follow_master=bool(d.get("follow_master", True)),
                   sublayers={str(k): SubLayerState.from_payload(v)
                              for k, v in sublayers.items()})


def inspectors_from_payload(blob) -> dict[str, InspectorState]:
    """``Settings.view_options["inspectors"]`` -> live states, keyed by ``source_id``.

    ``blob`` is whatever ``.get("inspectors")`` returned: ``None`` on a settings file that has
    never seen an inspector, or -- on a hand-edited or half-written one -- anything at all. A
    non-mapping blob yields ``{}`` and a malformed ENTRY is skipped, both silently: rule 2. The
    good entries beside it survive, which is the point of dropping per entry rather than per file.
    """
    if not isinstance(blob, dict):
        return {}
    states: dict[str, InspectorState] = {}
    for source_id, entry in blob.items():
        try:
            state = InspectorState.from_payload(source_id, entry)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            continue
        states[state.source_id] = state
    return states


def inspectors_to_payload(states: dict[str, InspectorState]) -> dict[str, dict]:
    """Live states -> the JSON-safe blob to hand ``update_settings(view_options=...)``.

    Keyed by each record's OWN ``source_id`` rather than by the mapping key it arrived under, so
    the id that comes back is the id the record carries.
    """
    return {state.source_id: state.to_payload() for state in states.values()}
