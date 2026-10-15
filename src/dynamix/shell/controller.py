# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Transport and key handling: the first feature split out of ``main_window.py``. The controller owns
what the transport and the key commands do; the window hands it in as the handlers of those commands and
keeps only the widgets they act on."""
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
        self._w._undo_rack_removal()

    def selection_click(self) -> None:
        self._w.set_selection_mode("click")

    def selection_cycle(self) -> None:
        self._w._cycle_selection_mode()

    def zoom_to_dataset(self) -> None:
        self._w._zoom_to_dataset()

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
