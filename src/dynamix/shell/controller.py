# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Transport and key handling: the first feature split out of ``main_window.py``. The controller owns
what the transport and the key commands do (the play toggle, the selection-mode cycle, the rack-drop
undo, the zoom to the dataset) and hands them in as the handlers of those commands; the window keeps
the widgets they act on and its old method names as one-line aliases for the callers that use them."""
from __future__ import annotations

import weakref


#: V's own cycle order (the non-click modes only): C owns click directly, and EQSelect's own V
#: never visits it either. Lives here because the cycle is the controller's to walk.
SELECTION_MODE_CYCLE = ("box", "lasso", "transect")


class CommandController:
    """The handlers of the transport and key commands for one window."""

    def __init__(self, window) -> None:
        self._window = weakref.ref(window)

    @property
    def _w(self):
        """The window, held weakly: it owns this controller, and a strong reference back would be
        a reference cycle through every widget it has."""
        return self._window()

    def toggle_transport(self) -> None:
        """Space (``transport.play``): the play button's own toggle, so button and key agree."""
        self._w.transport.play_button.toggle()

    def cancel_transect(self) -> None:
        self._w.canvas.cancel_transect()

    def undo_transect(self) -> None:
        self._w._transect_panel.undo()

    def undo_rack_drop(self) -> None:
        """Shift+Ctrl+Z: re-reads the window's zone at activation time, so it reaches whichever
        WorkflowZone is live now, even across a rebuild; a no-op before one is built."""
        zone = self._w.strips
        if zone is not None:
            zone.undo_removal()

    def selection_click(self) -> None:
        self._w.set_selection_mode("click")

    def selection_cycle(self) -> None:
        """V: walks box -> lasso -> transect -> box, skipping click; pressed in click mode it starts
        the cycle at box (there is no principled next mode for a cycle click was never part of)."""
        w = self._w
        if w._selection_mode not in SELECTION_MODE_CYCLE:
            w.set_selection_mode(SELECTION_MODE_CYCLE[0])
            return
        i = SELECTION_MODE_CYCLE.index(w._selection_mode)
        w.set_selection_mode(SELECTION_MODE_CYCLE[(i + 1) % len(SELECTION_MODE_CYCLE)])

    def zoom_to_dataset(self) -> None:
        """Frame the view in front on the selected layer. Raster: its ROI window when it runs on
        one, else the raster on screen (the same framing an open does). Map: the Layers panel's
        current row or group, else the layer's dataset group. Vector and Globe: the camera reset
        the scene's own r key makes."""
        w = self._w
        if w._center_view == "map":
            stack = w.project.layer_stack
            nid = w.layers_panel.current_id() if w.layers_panel is not None else None
            if nid is None and w.layer is not None:
                nid = next((n for n in (f"ds:{w.layer.source_id}", f"prod:{w.layer.layer_id}",
                                        f"src:{w.layer.source_id}") if stack.node(n) is not None),
                           None)
            if nid is not None:
                w._on_layers_zoom(nid)
            return
        if w._center_view in ("vector", "geo"):
            if w._arrangement is not None:
                w._arrangement.reset_camera()
            return
        window = w.layer.tags.get("roi.window") if w.layer is not None else None
        if window:
            r, c, h, wd = (int(v) for v in window.split(","))
            w.canvas.view.setRange(xRange=(c - 0.5, c + wd - 0.5),
                                   yRange=(r - 0.5, r + h - 0.5), padding=0.02)
            return
        w.canvas.view.autoRange(items=[w.canvas.image_item])

    def handlers(self) -> dict:
        return {
            "transport.play": self.toggle_transport,
            "edit.cancel_transect": self.cancel_transect,
            "edit.undo_transect": self.undo_transect,
            "edit.undo_rack_drop": self.undo_rack_drop,
            "view.selection.click": self.selection_click,
            "view.selection.cycle": self.selection_cycle,
            "view.zoom_to_dataset": self.zoom_to_dataset,
        }
