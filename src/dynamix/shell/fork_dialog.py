# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ForkDialog: "Fork derivative dataset…" -- what the derivative takes, where it sits, whether
it is kept.

- **Raster**: every raster the result holds (:func:`dynamix.core.derivative.raster_choices`,
  the one on screen first); each checked one becomes a band of the derivative, in list order.
- **Vector**: the extrema and maxima lines the result shows (after its filters).
- Raster, vector, or both -- one file either way.
- **Placement**: a dataset row of its own, or nested inside the dataset it came from.
- **Keep**: saved to a file (a Save-As dialog follows), or temporary -- this session only, left
  out of a saved project until "Save derivative as…".

The shell does the writing; this only asks. Fast Fork (Y) skips the dialog: :func:`fast_fork_choice`
turns the Fast Fork settings into the same choices, and :class:`FastForkSettingsDialog` edits them.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["FastForkSettingsDialog", "ForkDialog", "fast_fork_choice"]


class ForkDialog(QtWidgets.QDialog):
    """Pick the bands, the vectors, the placement and the keep for a derivative dataset."""

    def __init__(self, *, name: str, raster_labels: list, vector_counts: tuple,
                 parent_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fork derivative dataset")
        layout = QtWidgets.QVBoxLayout(self)

        note = QtWidgets.QLabel(
            "A derivative is written once and opens as a dataset of its own: tools run on it "
            "like on any raw dataset, and it never re-processes when its source changes.")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        self._name = QtWidgets.QLineEdit(name)
        form.addRow("Name", self._name)
        layout.addLayout(form)

        self._raster_box = QtWidgets.QGroupBox("Raster — checked rasters become its bands")
        self._raster_box.setCheckable(True)
        box_lay = QtWidgets.QVBoxLayout(self._raster_box)
        self._bands = QtWidgets.QListWidget()
        for i, label in enumerate(raster_labels):
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if i == 0 else QtCore.Qt.Unchecked)
            self._bands.addItem(item)
        box_lay.addWidget(self._bands)
        has_raster = bool(raster_labels)
        self._raster_box.setChecked(has_raster)
        self._raster_box.setEnabled(has_raster)
        layout.addWidget(self._raster_box, 1)

        points, lines = vector_counts
        has_vectors = points > 0 or lines > 0
        self._vector_box = QtWidgets.QGroupBox(
            f"Vector — {points} extrema points, {lines} maxima lines (as shown)")
        self._vector_box.setCheckable(True)
        self._vector_box.setChecked(has_vectors and not has_raster)
        self._vector_box.setEnabled(has_vectors)
        layout.addWidget(self._vector_box)

        place = QtWidgets.QGroupBox("Placement")
        place_lay = QtWidgets.QVBoxLayout(place)
        self._own = QtWidgets.QRadioButton("A dataset row of its own")
        self._nest = QtWidgets.QRadioButton(f"Nested inside {parent_name}")
        self._own.setChecked(True)
        place_lay.addWidget(self._own)
        place_lay.addWidget(self._nest)
        layout.addWidget(place)

        keep = QtWidgets.QGroupBox("Keep")
        keep_lay = QtWidgets.QVBoxLayout(keep)
        self._file = QtWidgets.QRadioButton("Save to a file… (permanent)")
        self._temporary = QtWidgets.QRadioButton(
            "Temporary — this session only; left out of a saved project until "
            "“Save derivative as…”")
        self._file.setChecked(True)
        keep_lay.addWidget(self._file)
        keep_lay.addWidget(self._temporary)
        layout.addWidget(keep)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self._buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Fork")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._name.textChanged.connect(lambda _t: self._refresh_ok())
        self._bands.itemChanged.connect(lambda _i: self._refresh_ok())
        self._raster_box.toggled.connect(lambda _c: self._refresh_ok())
        self._vector_box.toggled.connect(lambda _c: self._refresh_ok())
        self._refresh_ok()
        self.resize(460, 520)

    def _checked_bands(self) -> list:
        if not (self._raster_box.isEnabled() and self._raster_box.isChecked()):
            return []
        return [i for i in range(self._bands.count())
                if self._bands.item(i).checkState() == QtCore.Qt.Checked]

    def _vectors(self) -> bool:
        return self._vector_box.isEnabled() and self._vector_box.isChecked()

    def _refresh_ok(self) -> None:
        ok = bool(self._name.text().strip()) and (bool(self._checked_bands()) or self._vectors())
        self._buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(ok)

    def choices(self) -> dict:
        """``{"name", "bands" (indices into the raster list), "vectors", "nest", "temporary"}``."""
        return {"name": self._name.text().strip(), "bands": self._checked_bands(),
                "vectors": self._vectors(), "nest": self._nest.isChecked(),
                "temporary": self._temporary.isChecked()}


def fast_fork_choice(settings: dict, *, name: str, rasters: list, produced: bool, vectors: tuple) -> dict:
    """The dialog's choices from the Fast Fork ``settings``: every raster when they say ``all``
    or when the layer shows a produced field (its bands are what is on screen), else the raster
    shown, the first; the vectors when the settings ask for them or no raster is there to take."""
    bands = list(range(len(rasters))) if settings["rasters"] == "all" or produced else [0][:len(rasters)]
    take_vectors = (settings["vectors"] or not bands) and sum(vectors) > 0
    return {"name": name, "bands": bands, "vectors": bool(take_vectors), "nest": settings["nest"],
            "temporary": settings["temporary"]}


class FastForkSettingsDialog(QtWidgets.QDialog):
    """Settings > Fast Fork…: what Y forks without asking."""

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fast Fork")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Y forks the active layer's result as set here, with no dialog."))

        def pair(title, first, second, second_on):
            box = QtWidgets.QGroupBox(title)
            lay = QtWidgets.QVBoxLayout(box)
            a, b = QtWidgets.QRadioButton(first), QtWidgets.QRadioButton(second)
            (b if second_on else a).setChecked(True)
            lay.addWidget(a)
            lay.addWidget(b)
            layout.addWidget(box)
            return b

        self._nest = pair("Placement", "A dataset row of its own", "Nested inside the dataset it came from",
                          settings["nest"])
        self._file = pair("Keep", "Temporary, this session only", "Save to a file… (permanent)",
                          not settings["temporary"])
        self._all_rasters = pair("Rasters", "The raster shown", "Every raster the result holds",
                                 settings["rasters"] == "all")
        self._vectors = QtWidgets.QCheckBox("Also the extrema and maxima lines shown")
        self._vectors.setChecked(settings["vectors"])
        layout.addWidget(self._vectors)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choices(self) -> dict:
        """The settings entry: ``{"nest", "temporary", "rasters" ("shown" | "all"), "vectors"}``."""
        return {"nest": self._nest.isChecked(), "temporary": not self._file.isChecked(),
                "rasters": "all" if self._all_rasters.isChecked() else "shown",
                "vectors": self._vectors.isChecked()}
