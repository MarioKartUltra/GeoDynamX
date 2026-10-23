# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The colour-ramp editor: pyqtgraph's gradient (click to add a stop, drag to move it, double-click
to recolour, right-click to remove), a name, and Save or Delete into the user's ramp library
(:mod:`dynamix.shell.ramps`)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from dynamix.shell import ramps

#: The stops a built-in colormap starts the editor with.
_SEED_STOPS = 5


def _ticks(name: str) -> list:
    """``name``'s stops as gradient ticks: a ramp's own, a built-in colormap sampled evenly."""
    from dynamix.shell.canvas import DEFAULT_COLORMAP, _resolve_cmap

    cmap = _resolve_cmap(name) or _resolve_cmap(DEFAULT_COLORMAP)
    stops = ramps.ramp_stops(name)
    positions = [p for p, _c in stops] if stops else list(np.linspace(0.0, 1.0, _SEED_STOPS))
    colours = cmap.map(np.asarray(positions, dtype=float), mode="byte")
    return [(float(p), tuple(int(v) for v in c[:3]) + (255,)) for p, c in zip(positions, colours)]


class RampEditorDialog(QtWidgets.QDialog):
    def __init__(self, start: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colour ramp")
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel("Click the bar to add a stop, drag to move it, double-click to "
                                "recolour, right-click to remove.")
        hint.setProperty("muted", "true")
        layout.addWidget(hint)
        self.gradient = pg.GradientWidget(orientation="bottom")
        self.gradient.setMinimumWidth(360)
        self.gradient.restoreState({"mode": "rgb", "ticks": _ticks(start)})
        layout.addWidget(self.gradient)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Name"))
        own = start in ramps.ramp_names()
        self.name_edit = QtWidgets.QLineEdit(start if own else "")
        row.addWidget(self.name_edit, 1)
        layout.addLayout(row)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        save = buttons.addButton("Save", QtWidgets.QDialogButtonBox.AcceptRole)
        delete = buttons.addButton("Delete", QtWidgets.QDialogButtonBox.DestructiveRole)
        delete.setEnabled(own)
        save.clicked.connect(lambda: self.save() and self.accept())
        delete.clicked.connect(self._delete)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        #: The ramp the dialog saved, or None (cancelled, or the ramp was deleted).
        self.saved_name = None

    def stops(self) -> list:
        """The gradient's stops as ``(position, "#rrggbb")``, in order."""
        ticks = self.gradient.saveState()["ticks"]
        return sorted((round(float(p), 6), "#{:02x}{:02x}{:02x}".format(*(int(v) for v in c[:3])))
                      for p, c in ticks)

    def save(self) -> bool:
        try:
            ramps.save_ramp(self.name_edit.text(), self.stops())
        except ValueError as exc:
            self.message.setText(str(exc))
            return False
        self.saved_name = self.name_edit.text().strip()
        return True

    def _delete(self) -> None:
        ramps.delete_ramp(self.name_edit.text().strip())
        self.saved_name = None
        self.accept()
