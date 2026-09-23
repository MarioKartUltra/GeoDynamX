# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""RoiPanel: the precision half of the ROI gesture.

A drag is fast and approximate; the numbers it produced are the thing that ends up in a cache key
and in the paper trail. So the box the user drew arrives here as four EDITABLE pixel fields, each
with its physical size beside it, and nothing is analysed until Create is pressed.

**A panel, never a dialog.** The spec says so in as many words, and DESIGN.md's no-dialog rule is
why: a modal box stops the app to ask a question, and this question is one the user answers by
looking at the raster behind it. The panel lives in the left sidebar under the layer list, hidden
until there is an ROI to talk about. The same rule governs the too-small case -- an ROI under
:data:`~dynamix.roi.halo.MIN_ROI_SIDE` px per side disables Create and says why in a muted line,
rather than accepting the gesture and then popping an error.

**The device is the authority, not this file.** The boundary choices, the minimum side and every
field's label are read off ``WTMM2DROI.params`` rather than restated here: the panel edits that
device's parameters, and a second copy of its declarations would be free to drift out of agreement
with the engine that enforces them. Reading them from the class needs no registry -- ``params`` is
a plain class attribute.

**One conversion.** The physical readouts come from the ``(factor, unit)`` pair the caller already
got out of :func:`dynamix.shell.units.px_to_metres` -- handed in whole rather than recomputed, so
this panel can never disagree with the transport, the scale bar or a knob's derived reading. A
``None`` factor (or a "px" unit) means there is no physical number to state, and the readout is
EMPTY rather than a fabricated 1.0 -- exactly what units.py exists to prevent.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.devices.wtmm_roi import WTMM2DROI
from dynamix.shell.knob_widgets import CycleButton
from dynamix.shell.knobs import control_spec

__all__ = ["RoiPanel"]

#: The four ROI geometry params, in the order they are shown. Names match the device's.
_ROI_FIELDS = ("roi_row", "roi_col", "roi_h", "roi_w")

_PARAMS = {p.name: p for p in WTMM2DROI.params}

#: Both sides must reach this, and the device's own Param hard-min is where that number lives
#: (it mirrors ``dynamix.roi.halo.MIN_ROI_SIDE``, which the engine enforces itself).
_MIN_SIDE = int(_PARAMS["roi_h"].min)

#: Width of the pixel entry fields. A five-digit row offset is the widest thing they ever hold,
#: and the left panel is only 220 px wide -- letting them expand would push the physical readout
#: off the edge of the zone.
_EDIT_WIDTH = 56


class RoiPanel(QtWidgets.QFrame):
    """The drawn box, as four editable pixel fields plus a boundary choice and Create.

    ``show_roi`` fills it and reveals it; ``createRequested`` carries the final integers out. The
    panel knows nothing about layers, chains or the project -- the window does the spawning.
    """

    #: Create was pressed: ``{"roi_row", "roi_col", "roi_h", "roi_w", "boundary"}``, ints and the
    #: choice string, ready to go straight into a ``wtmm2d_roi`` step's params.
    createRequested = QtCore.Signal(dict)

    #: The drawn box as a plain windowed CHILD DATASET (same dict shape as ``createRequested``;
    #: ``boundary`` rides along but the child ignores it -- a crop has no halo).
    childRequested = QtCore.Signal(dict)

    #: Arm the hover-ghost placement for the panel's current (h, w) -- the click repositions
    #: row/col in these fields (never creates anything by itself).
    placeRequested = QtCore.Signal(int, int)

    #: The fields currently describe a legal ROI (same dict shape as ``createRequested``).
    #: Emitted from every edit that leaves :meth:`values` non-None, so the canvas can keep the
    #: drawn band tracking the NUMBERS -- typing a row/col with no visual echo left the user
    #: setting coordinates blind (2026-08-09 first-use report).
    valuesEdited = QtCore.Signal(dict)

    #: Save ROI was pressed -- the same dict shape as
    #: ``createRequested``; the window saves it as a ``RoiRecord`` on the displayed source.
    saveRequested = QtCore.Signal(dict)

    #: A saved ROI was picked in the list: its ``roi_id`` (the ACTIVE ROI a tool drop runs on).
    roiActivated = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("strip", "true")
        self._edits: dict[str, QtWidgets.QLineEdit] = {}
        self._physical: dict[str, QtWidgets.QLabel] = {}
        self._px_to_phys: tuple[float | None, str] = (None, "px")
        self._boundary = _PARAMS["boundary"].default
        self._blocked = ""              # why Create is refused for the CURRENT target, if it is
        self._dims: tuple[int, int] | None = None    # the drawn-on field's (h, w), when known

        column = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        for row, name in enumerate(_ROI_FIELDS):
            label = QtWidgets.QLabel(_PARAMS[name].label or name)
            label.setProperty("muted", "true")
            edit = QtWidgets.QLineEdit()
            # Mono comes from the theme's own QLineEdit rule (theme.py) -- every entry field in
            # this app is already mono, so there is nothing to set (and nothing to set it WITH,
            # under the Theme Rule).
            edit.setFixedWidth(_EDIT_WIDTH)
            edit.textChanged.connect(self._on_edited)
            physical = QtWidgets.QLabel("")
            physical.setProperty("reading", "true")
            physical.setProperty("muted", "true")
            grid.addWidget(label, row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(physical, row, 2)
            self._edits[name] = edit
            self._physical[name] = physical
        column.addLayout(grid)

        boundary_row = QtWidgets.QHBoxLayout()
        boundary_label = QtWidgets.QLabel(_PARAMS["boundary"].label or "boundary")
        boundary_label.setProperty("muted", "true")
        boundary_row.addWidget(boundary_label)
        self.boundary_button = CycleButton(control_spec(_PARAMS["boundary"]), self._boundary)
        self.boundary_button.valueChanged.connect(self._on_boundary_changed)
        boundary_row.addWidget(self.boundary_button)
        column.addLayout(boundary_row)

        self.message_label = QtWidgets.QLabel("")
        self.message_label.setProperty("muted", "true")
        self.message_label.setWordWrap(True)
        column.addWidget(self.message_label)

        # Save is the primary
        # action -- the region is saved on the dataset, and any tool dropped while it is active
        # runs on it (native pixels + the tool's own margin). The legacy one-click stays as
        # "Save + run WTMM" (createRequested; the window also saves the region it runs on).
        self.save_button = QtWidgets.QPushButton("Save ROI")
        self.save_button.clicked.connect(self._on_save_clicked)
        column.addWidget(self.save_button)

        self.create_button = QtWidgets.QPushButton("Save + run WTMM")
        self.create_button.clicked.connect(self._on_create_clicked)
        column.addWidget(self.create_button)

        # 2026-09-21: the SECOND thing a drawn box can become -- a plain windowed CHILD DATASET (native-pixel crop,
        # empty chain, any tool). Deliberately NOT gated on ``_blocked``: that block is about
        # nesting the wtmm2d_roi ANALYSIS, and a child crop is legal wherever the box itself is.
        # NO margin knob here: the ROI defines the CORE only.
        # However much surrounding context an analysis needs is a property of the TOOL
        # (max wavelet scale's COI, kernel support, diffusion domain), decided at COMPUTE
        # time -- the child carries its parent linkage (source, absolute window, full
        # dims) precisely so tools can sample their own apron later, never a number typed
        # here.
        #
        # "Place box": arm a hover ghost of the current h x w on the canvas (the
        # armed-placement gesture the wtmm2d_roi drop already has); the click sets
        # row/col in these fields -- it never creates anything by itself.
        self.place_button = QtWidgets.QPushButton("Place box")
        self.place_button.clicked.connect(self._on_place_clicked)
        column.addWidget(self.place_button)
        self.child_button = QtWidgets.QPushButton("Child dataset")
        self.child_button.clicked.connect(self._on_child_clicked)
        column.addWidget(self.child_button)
        # The crop-copy workflow is superseded by saved ROIs (tools read
        # the region + their margin off the file); unhooked from the panel, handler kept.
        self.child_button.setVisible(False)

        # The saved ROIs of the displayed dataset; the selected row is the ACTIVE ROI.
        saved_label = QtWidgets.QLabel("Saved ROIs")
        saved_label.setProperty("muted", "true")
        column.addWidget(saved_label)
        self.saved_list = QtWidgets.QListWidget()
        self.saved_list.setMaximumHeight(96)
        self.saved_list.currentRowChanged.connect(self._on_saved_row_changed)
        column.addWidget(self.saved_list)
        # "Deselect": no ROI active -- a drop then runs on the whole dataset (or, on a display
        # picture, is refused: nothing runs on the picture itself).
        self.deselect_button = QtWidgets.QPushButton("Deselect")
        self.deselect_button.clicked.connect(self._on_deselect_clicked)
        column.addWidget(self.deselect_button)
        self._saved_ids: list[str] = []
        self._setting_saved = False

        self.setVisible(False)          # nothing to say until something has been drawn

    # -- the outside world -----------------------------------------------------------------
    def show_roi(self, row: int, col: int, h: int, w: int, px_to_phys, blocked: str = "",
                dims: tuple[int, int] | None = None) -> None:
        """Fill the fields from a drawn box and reveal the panel.

        ``px_to_phys`` is :func:`dynamix.shell.units.px_to_metres`'s return value, ``(factor,
        unit)``, passed straight through -- see the module docstring on why it is handed in rather
        than recomputed here.

        ``blocked`` is why Create must be refused for the layer this box was drawn on, or ``""``.
        It is a property of the TARGET, not of the numbers -- an ROI of an ROI is not supported in
        this slice however legal the rectangle is -- so it arrives per-show rather than being
        something the panel could work out for itself, and a subsequent unblocked ``show_roi``
        clears it. Drawing the box on such a layer stays allowed; only creating from it does not.

        ``dims`` is the drawn-on field's own ``(height, width)``, or ``None`` when the caller has
        no honest answer -- :meth:`values` upper-bounds the edited box against it, when given, so
        an edited row/col/h/w that walks the ROI off the displayed image is refused the same way a
        too-small side is, rather than being handed to Create and failing on the strip instead.
        """
        self._px_to_phys = px_to_phys
        self._blocked = blocked
        self._dims = dims
        for name, value in zip(_ROI_FIELDS, (row, col, h, w)):
            self._edits[name].setText(str(int(value)))
        self.setVisible(True)
        self._refresh()

    def set_saved(self, rois, active: str | None = None) -> None:
        """Show the saved ROIs ``[(roi_id, label), ...]`` with ``active`` selected; silent (no
        ``roiActivated`` echo). Never changes the panel's visibility -- that stays the window's
        call (a layer switch hides the panel so a stale box cannot act on a new target)."""
        self._setting_saved = True
        try:
            self.saved_list.clear()
            self._saved_ids = [str(rid) for rid, _label in rois]
            for rid, label in rois:
                self.saved_list.addItem(f"{label}  ({rid})")
            row = self._saved_ids.index(active) if active in self._saved_ids else -1
            self.saved_list.setCurrentRow(row)
        finally:
            self._setting_saved = False

    def edit(self, name: str) -> QtWidgets.QLineEdit:
        """The pixel entry field for one ROI param -- ``panel.edit("roi_h")``."""
        return self._edits[name]

    def physical(self, name: str) -> str:
        """The physical readout currently shown beside ``name`` (``""`` when there is none)."""
        return self._physical[name].text()

    def values(self) -> dict | None:
        """The four fields as ints plus the boundary choice, or ``None`` if they do not currently
        describe a legal ROI (unparsable, negative, a side under the engine's minimum, or -- when
        :meth:`show_roi` was given ``dims`` -- reaching past the drawn-on field's own extent)."""
        out: dict = {}
        for name in _ROI_FIELDS:
            try:
                out[name] = int(self._edits[name].text())
            except ValueError:
                return None
        if out["roi_row"] < 0 or out["roi_col"] < 0:
            return None
        if out["roi_h"] < _MIN_SIDE or out["roi_w"] < _MIN_SIDE:
            return None
        if self._dims is not None:
            dh, dw = self._dims
            if out["roi_row"] + out["roi_h"] > dh or out["roi_col"] + out["roi_w"] > dw:
                return None
        out["boundary"] = self._boundary
        return out

    # -- internals -------------------------------------------------------------------------
    def _on_boundary_changed(self, value) -> None:
        self._boundary = value

    def _on_edited(self, _text: str) -> None:
        self._refresh()

    def _refresh(self) -> None:
        """Re-state every physical readout and the Create button's availability from the fields as
        they stand. One method, called from every edit, so the two can never disagree."""
        factor, unit = self._px_to_phys
        for name in _ROI_FIELDS:
            self._physical[name].setText(self._physical_text(name, factor, unit))
        current = self.values()
        self.create_button.setEnabled(current is not None and not self._blocked)
        self.save_button.setEnabled(current is not None)     # saving is legal on any layer
        self.child_button.setEnabled(current is not None)
        self.place_button.setEnabled(current is not None)
        self.message_label.setText(self._message())
        if current is not None:
            self.valuesEdited.emit(current)

    def _physical_text(self, name: str, factor, unit: str) -> str:
        if factor is None or unit == "px":
            return ""
        try:
            px = int(self._edits[name].text())
        except ValueError:
            return ""
        return f"{px * factor:.4g} {unit}"

    def _message(self) -> str:
        """Why Create is unavailable, or ``""``. Muted text in the panel -- never a dialog.

        The TARGET's objection comes first: when the layer cannot take an ROI at all, saying so is
        more use than commenting on a rectangle that was never the problem.
        """
        if self._blocked:
            return self._blocked
        if self.values() is not None:
            return ""
        try:
            h = int(self._edits["roi_h"].text())
            w = int(self._edits["roi_w"].text())
        except ValueError:
            return "row, col, height and width must be whole pixel counts"
        if h < _MIN_SIDE or w < _MIN_SIDE:
            return (f"ROI is {h}×{w} px; both sides must be at least {_MIN_SIDE} px "
                    "(a sliver carries no multi-scale structure)")
        if self._dims is not None:
            try:
                r = int(self._edits["roi_row"].text())
                c = int(self._edits["roi_col"].text())
            except ValueError:
                return "row, col, height and width must be whole pixel counts"
            dh, dw = self._dims
            if r >= 0 and c >= 0 and (r + h > dh or c + w > dw):
                return f"ROI reaches past the {dh}×{dw} px image"
        return "row and col must be at or inside the raster's origin"

    def _on_create_clicked(self) -> None:
        values = self.values()
        # The button is disabled in both refusal cases; re-checked here because emitting this
        # signal spawns a layer, and a guard that only lives in a widget's enabled state is one
        # programmatic click away from not existing.
        if values is not None and not self._blocked:
            self.createRequested.emit(values)

    def _on_save_clicked(self) -> None:
        values = self.values()
        if values is not None:
            self.saveRequested.emit(values)

    def _on_deselect_clicked(self) -> None:
        self._setting_saved = True
        try:
            self.saved_list.setCurrentRow(-1)
        finally:
            self._setting_saved = False
        self.roiActivated.emit("")

    def _on_saved_row_changed(self, row: int) -> None:
        if self._setting_saved or not (0 <= row < len(self._saved_ids)):
            return
        self.roiActivated.emit(self._saved_ids[row])

    def _on_child_clicked(self) -> None:
        values = self.values()
        if values is not None:          # a child crop is legal wherever the box itself is
            self.childRequested.emit(values)

    def _on_place_clicked(self) -> None:
        values = self.values()
        if values is not None:
            self.placeRequested.emit(int(values["roi_h"]), int(values["roi_w"]))
