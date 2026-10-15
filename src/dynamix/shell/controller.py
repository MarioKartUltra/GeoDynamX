# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Transport and key handling: the first feature split out of ``main_window.py``. The controller owns
what the transport and the key commands do (the play toggle, the selection-mode cycle, the rack-drop
undo, the zoom to the dataset) and hands them in as the handlers of those commands; the window keeps
the widgets they act on and its old method names as one-line aliases for the callers that use them."""
from __future__ import annotations


class CommandController:
    """The handlers of the transport and key commands for one window."""

    def __init__(self, window) -> None:
        self._w = window

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
        from dynamix.shell.main_window import _SELECTION_MODE_CYCLE

        w = self._w
        if w._selection_mode not in _SELECTION_MODE_CYCLE:
            w.set_selection_mode(_SELECTION_MODE_CYCLE[0])
            return
        i = _SELECTION_MODE_CYCLE.index(w._selection_mode)
        w.set_selection_mode(_SELECTION_MODE_CYCLE[(i + 1) % len(_SELECTION_MODE_CYCLE)])

    def zoom_to_dataset(self) -> None:
        """Fit the raster view to the selected row: its ROI window when it runs on one, else the
        raster on screen (the same framing an open does)."""
        w = self._w
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
