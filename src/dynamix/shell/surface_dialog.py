# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""SurfaceDialog: the 3-D surface configuration for one layer.

Replaces the bare "3-D surface" checkbox semantics with the full choice the Vector view now
supports: OFF (flat drape), z from the SAME dataset (the pre-existing behavior -- a DEM stands
up on its own values), or z from ANOTHER loaded raster layer -- the drape case, e.g. a
holder_map h(x) layer standing on a DEM's elevation. "Other" offers only the ELIGIBLE loaded
rasters the caller passes in (same grid shape, not a point layer -- ``main_window`` owns that
admission rule, exactly as it owns backproject's target resolution; this dialog never inspects
layers itself).

Self-contained pure Qt (no other ``dynamix.shell.*`` import), same shape as
``topology_panel``/``transect_panel`` in devloop's dependency table. Modal ``exec()`` by
design -- unlike the floating analysis windows this is a two-click configuration pick, not a
workspace to keep open beside the canvas.
"""
from __future__ import annotations

from PySide6 import QtWidgets

__all__ = ["SurfaceDialog"]


class SurfaceDialog(QtWidgets.QDialog):
    """Pick the 3-D surface mode for one layer.

    ``layers`` is ``[(layer_id, display_name), ...]`` -- the eligible z-source rasters.
    ``selection()`` returns ``(enabled, source)`` where ``source`` is ``"same"`` or the chosen
    layer_id; it is only meaningful after ``exec()`` returned ``Accepted``.
    """

    def __init__(self, *, enabled: bool, source: str, layers: list, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("3-D surface")

        layout = QtWidgets.QVBoxLayout(self)
        self._off_radio = QtWidgets.QRadioButton("Off — flat drape")
        self._same_radio = QtWidgets.QRadioButton("Same dataset — this layer's values as height")
        self._other_radio = QtWidgets.QRadioButton("Other dataset — heights from another layer:")
        layout.addWidget(self._off_radio)
        layout.addWidget(self._same_radio)
        layout.addWidget(self._other_radio)

        self._layer_combo = QtWidgets.QComboBox()
        for layer_id, name in layers:
            self._layer_combo.addItem(name, layer_id)
        self._layer_combo.setEnabled(False)
        row = QtWidgets.QHBoxLayout()
        row.addSpacing(24)
        row.addWidget(self._layer_combo, 1)
        layout.addLayout(row)
        self._other_radio.toggled.connect(self._layer_combo.setEnabled)
        # 2026-09-20 (user: layers "are not synchronized"): one gesture stands the whole
        # dataset's family on the same choice -- the caller fans the tags out.
        self._all_box = QtWidgets.QCheckBox("Apply to every layer of this dataset")
        layout.addWidget(self._all_box)
        # No eligible siblings -> "Other" is honestly unavailable rather than an empty combo.
        if not layers:
            self._other_radio.setEnabled(False)

        if not enabled:
            self._off_radio.setChecked(True)
        elif source != "same" and self._select_layer(source):
            self._other_radio.setChecked(True)
        else:
            self._same_radio.setChecked(True)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_layer(self, layer_id: str) -> bool:
        for i in range(self._layer_combo.count()):
            if self._layer_combo.itemData(i) == layer_id:
                self._layer_combo.setCurrentIndex(i)
                return True
        return False

    def selection(self) -> tuple:
        """``(enabled, source, apply_all)`` -- meaningful after Accepted."""
        apply_all = self._all_box.isChecked()
        if self._off_radio.isChecked():
            return False, "same", apply_all
        if self._other_radio.isChecked() and self._layer_combo.count():
            return True, str(self._layer_combo.currentData()), apply_all
        return True, "same", apply_all
