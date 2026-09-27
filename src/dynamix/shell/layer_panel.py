# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""LayerPanel: the layer list, grouped by source, with per-layer hide/lock/freeze/remove/rename.

Replaces the flat ``QListWidget`` + ``_CHILD_PREFIX`` indentation hack in ``main_window.py`` with a
real ``QTreeWidget`` of THREE levels: one source-header row per open raster (its file stem, expand
state round-tripped both ways with ``SourceRef.collapsed`` -- see :meth:`LayerPanel.set_project`),
one row per layer under its source, and (for a grouped ROI layer) one row per child nested under
its own parent LAYER row -- an ROI is a child of the layer it was drawn on, not of the source
directly, exactly as ``main_window.roi_chain``'s grouping already implies.

Each layer row carries a small widget (:class:`_LayerRow`) of three checkable ``QToolButton``s --
text glyphs ``"H"``/``"L"``/``"F"``, icon art being explicitly out of scope for this slice -- and
the panel re-emits their toggles as ``hideToggled``/``lockToggled``/``freezeToggled(layer_id,
bool)``. It stores no flag state of its own: ``layer.visible`` and ``layer.tags["ui.lock"]``/
``["ui.freeze"]`` are the single source of truth, read once at row construction and otherwise owned
by ``main_window`` (the enforcement lives there, not here).

**Hide is v1-scoped.** This panel only ever toggles ONE layer's flag and tells the window; it has
no idea whether that layer is the one currently on screen, and it draws nothing itself. Composing
several simultaneously-VISIBLE layers into one picture is the geographic-views slice's business --
today ``main_window`` reads ``layer.visible`` for the ACTIVE layer only, clearing its canvas
overlays the moment its own row is hidden (see ``MainWindow._on_hide_toggled``).

A SOURCE HEADER row carries a widget of its own (:class:`_SourceRow`): the per-source inspector
toggle, re-emitted as ``inspectorToggled(source_id, bool)``. It is the ONE signal here that carries
a ``str`` rather than an ``int`` layer id, because spec 5d makes the inspector per SOURCE dataset
("the layer list becomes a tree rooted at sources; ``inspectorToggled`` carries a ``source_id``,
not a ``layer_id``") -- and this tree was already rooted at sources, so what 5d asks for here is a
button on the header, not a second tree. The panel stores no open/closed state of its own either:
``main_window`` owns the window registry and pushes the truth back through
:meth:`LayerPanel.set_inspector_open` when a window closes itself.

Context menu (right-click a layer row): Rename (opens the same inline ``QLineEdit`` editor a
double-click on the name column does -- ``QTreeWidget``'s own edit machinery, not a custom one),
Remove (``removeRequested``), New refined run (``refinedRunRequested``). A source-header row gets no context menu: it names a raster, not a layer, and
``removeRequested``'s signature (``int`` layer id) has nothing to carry for one.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

#: Item-data roles on column 0. A LAYER row carries its ``layer_id`` under ``_LAYER_ID_ROLE``; a
#: SOURCE HEADER row carries its ``source_id`` under ``_SOURCE_ID_ROLE`` instead -- never both on
#: the same item, which is how every handler below tells a header apart from a layer row.
_LAYER_ID_ROLE = QtCore.Qt.UserRole
_SOURCE_ID_ROLE = QtCore.Qt.UserRole + 1
#: A saved-ROI row carries its ``roi_id`` here -- and no layer id: an ROI is a pixel
#: window on its dataset, drawn as an outline, not a layer.
_ROI_ID_ROLE = QtCore.Qt.UserRole + 2
#: A band row under a stack dataset: the band's 0-based index (-1 on the "Bands" row
#: itself), with its dataset's source id under _BAND_SOURCE_ROLE.
_BAND_ROLE = QtCore.Qt.UserRole + 3
_BAND_SOURCE_ROLE = QtCore.Qt.UserRole + 4
#: An output row under a result's "Outputs" group: the output's name under _OUTPUT_KEY_ROLE
#: (unset on the group row itself) and its layer's id under _OUTPUT_LAYER_ROLE. Never
#: _LAYER_ID_ROLE: an output is a product of its layer, and every rename/remove/menu handler
#: below keys off that role.
_OUTPUT_KEY_ROLE = QtCore.Qt.UserRole + 5
_OUTPUT_LAYER_ROLE = QtCore.Qt.UserRole + 6


def layer_outputs(layer) -> tuple:
    """``(step, outputs)`` for ``layer``: its chain's LAST transform step and the outputs that
    step's device declares, or ``(None, ())`` when the last transform declares none (a filter
    never does; an unregistered device counts as declaring none)."""
    from dynamix.model.device import declared_outputs, get_device, is_transform

    last = None
    for step in layer.chain.steps:
        try:
            device = get_device(step.device)
        except KeyError:
            continue
        if is_transform(device):
            last = (step, device)
    if last is None:
        return None, ()
    outputs = declared_outputs(last[1])
    return (last[0], outputs) if outputs else (None, ())


def shown_output(step, outputs) -> str | None:
    """The raster output ``step``'s view-only ``show`` param puts on the canvas, or ``None``
    when it names none (the raw field is up)."""
    show = step.params.get("show") if step is not None else None
    return show if any(o.name == show and o.kind == "raster" for o in outputs) else None

#: The chip-hover feedback:
#: "chip hover emits chipHovered(parent_layer_id) -> LayerPanel.flash_row(layer_id) (temporary
#: selection-color property, 1 s timer)". The hover gesture elsewhere calls :meth:`LayerPanel.flash_row`; this module only has to make the flash
#: itself (set the property, clear it after one second) correct and independently testable.
_FLASH_MS = 1000


class _LayerRow(QtWidgets.QWidget):
    """One layer row's trailing widget: the three H/L/F tool buttons, nothing else.

    Initial checked state is read from ``layer`` ONCE, at construction, before any signal is
    connected -- the same order ``chain_strip.DeviceStrip``'s and ``workflow_zone.DeviceBox``'s own
    bypass button already use for exactly this reason: seeding ``setChecked`` before ``toggled`` is
    wired can never itself fire a spurious toggle.
    """

    hideToggled = QtCore.Signal(bool)
    lockToggled = QtCore.Signal(bool)
    freezeToggled = QtCore.Signal(bool)

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)

        self.hide_button = self._make_button("H", "Hide this layer's products (extrema, chains)", not layer.visible)
        self.lock_button = self._make_button(
            "L", "Lock (params read-only)", layer.tags.get("ui.lock") == "1")
        self.freeze_button = self._make_button(
            "F", "Freeze (lock + pin cached results)", layer.tags.get("ui.freeze") == "1")
        for button in (self.hide_button, self.lock_button, self.freeze_button):
            row.addWidget(button)

        self.hide_button.toggled.connect(self.hideToggled.emit)
        self.lock_button.toggled.connect(self.lockToggled.emit)
        self.freeze_button.toggled.connect(self.freezeToggled.emit)

    @staticmethod
    def _make_button(text: str, tooltip: str, checked: bool) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setCheckable(True)
        button.setChecked(checked)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setProperty("rowToggle", "true")      # theme.py's black/white toggle rule
        return button


class _RoiRow(QtWidgets.QWidget):
    """A saved ROI row's trailing widget: its H only -- it hides the ROI's OUTLINE, never the
    results computed on it."""

    hideToggled = QtCore.Signal(bool)

    def __init__(self, roi, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)
        row.addStretch(1)
        self.hide_button = _LayerRow._make_button(
            "H", "Hide this ROI's outline (results on it stay)", not roi.visible)
        row.addWidget(self.hide_button)
        self.hide_button.toggled.connect(self.hideToggled.emit)


class _OutputRow(QtWidgets.QWidget):
    """An output row's trailing widget: its H only. On a raster output H checked means "not the
    one on the canvas" (the rows are a radio); on the vector "edges" output it hides the maxima
    drawing."""

    hideToggled = QtCore.Signal(bool)

    def __init__(self, output, hidden: bool, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)
        row.addStretch(1)
        tip = ("Hide the edges drawing (filters and tables still read them)"
               if output.kind == "vector" else
               "Hide this output; clear H to show it in place of the data "
               "(one raster output at a time)")
        self.hide_button = _LayerRow._make_button("H", tip, hidden)
        self.tooltip = tip
        row.addWidget(self.hide_button)
        self.hide_button.toggled.connect(self.hideToggled.emit)


class _SourceRow(QtWidgets.QWidget):
    """One SOURCE header row's trailing widget: the inspector toggle, nothing else.

    Same construction order as :class:`_LayerRow`, for the same reason its docstring gives:
    ``setChecked`` is seeded BEFORE ``toggled`` is connected, so seeding an already-open
    inspector's button can never itself fire a spurious toggle back at the window that opened it.

    The glyph is the text ``"I"``, matching the H/L/F rows -- icon art is explicitly out of scope
    until the mockups land (spec 5c is PENDING).

    The leading stretch is the one pixel decision here, and it is load-bearing for the render
    criterion: ``QHBoxLayout`` spreads its surplus space around a lone non-expanding widget, which
    would leave the toggle floating in the middle of the column rather than at the row's RIGHT
    EDGE where the acceptance criterion (and a rack-style layer list) puts it.
    """

    inspectorToggled = QtCore.Signal(bool)
    hideToggled = QtCore.Signal(bool)          # the DATASET's own hide
    lockToggled = QtCore.Signal(bool)          # the master layer's, once one attaches
    freezeToggled = QtCore.Signal(bool)

    def __init__(self, open_: bool = False, hidden: bool = False, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)
        row.addStretch(1)

        # The dataset's hide: the raster itself, everywhere it is drawn -- the layer rows' own H
        # under this header hide their PRODUCTS (extrema, chains) and never the raster, which is
        # what "hide the dataset and show only extrema" needs.
        self.hide_button = QtWidgets.QToolButton()
        self.hide_button.setCheckable(True)
        self.hide_button.setChecked(hidden)
        self.hide_button.setText("H")
        self.hide_button.setToolTip("Hide dataset (the raster; its layers' products stay)")
        self.hide_button.setProperty("rowToggle", "true")
        row.addWidget(self.hide_button)
        self.hide_button.toggled.connect(self.hideToggled.emit)

        # The dataset row IS its master layer's row: the master's lock/freeze sit here,
        # hidden until a master attaches (a bare header has none).
        self.lock_button = _LayerRow._make_button("L", "Lock (params read-only)", False)
        self.freeze_button = _LayerRow._make_button(
            "F", "Freeze (lock + pin cached results)", False)
        for button in (self.lock_button, self.freeze_button):
            button.setVisible(False)
            row.addWidget(button)
        self.lock_button.toggled.connect(self.lockToggled.emit)
        self.freeze_button.toggled.connect(self.freezeToggled.emit)

        self.inspector_button = QtWidgets.QToolButton()
        self.inspector_button.setCheckable(True)
        self.inspector_button.setChecked(open_)
        self.inspector_button.setText("I")
        self.inspector_button.setToolTip("Inspector (floating window for this source)")
        self.inspector_button.setProperty("rowToggle", "true")
        row.addWidget(self.inspector_button)

        self.inspector_button.toggled.connect(self.inspectorToggled.emit)

    def attach_master(self, layer) -> None:
        """Show the master layer's L/F, seeded from its flags without emitting."""
        for button, tag in ((self.lock_button, "ui.lock"), (self.freeze_button, "ui.freeze")):
            button.blockSignals(True)
            button.setChecked(layer.tags.get(tag) == "1")
            button.blockSignals(False)
            button.setVisible(True)

    def detach_master(self) -> None:
        for button in (self.lock_button, self.freeze_button):
            button.setVisible(False)


class LayerPanel(QtWidgets.QTreeWidget):
    """The layer list. See the module docstring for the three-level grouping and the hide v1 scope.

    ``project`` is optional (default ``None``): a bare panel still renders headers and rows, just
    with no model-backed collapse memory (the header falls back to always-expanded) -- a test can
    build one without a ``Project`` at all. ``main_window`` always has a ``Project`` ready before it
    builds this panel and passes it straight in; :meth:`set_project` exists for a caller that does
    not.
    """

    layerSelected = QtCore.Signal(int)
    hideToggled = QtCore.Signal(int, bool)
    lockToggled = QtCore.Signal(int, bool)
    freezeToggled = QtCore.Signal(int, bool)
    removeRequested = QtCore.Signal(int)
    #: Header-row "Remove dataset": carries the SOURCE id -- the shell
    #: owns the confirmation and the family cascade.
    removeSourceRequested = QtCore.Signal(str)
    renameRequested = QtCore.Signal(int, str)
    refinedRunRequested = QtCore.Signal(int)
    #: ``(source_id, open)`` -- the per-SOURCE inspector toggle (spec 5d). A ``str`` id, unlike
    #: every signal above it: an inspector opens on a SOURCE, never on a layer.
    inspectorToggled = QtCore.Signal(str, bool)
    sourceHideToggled = QtCore.Signal(str, bool)   # (source_id, hidden): the dataset's own hide
    #: A saved-ROI row became current: its ``roi_id`` (the next tool dropped runs on it).
    roiSelected = QtCore.Signal(str)
    #: ``(roi_id, hidden)`` -- an ROI row's H: its outline only.
    roiHideToggled = QtCore.Signal(str, bool)
    #: "Delete layer": remove ONE layer, its children move up to its parent.
    removeLayerOnlyRequested = QtCore.Signal(int)
    #: "Delete ROI" / Delete on an ROI row -- the shell refuses while results use it.
    removeRoiRequested = QtCore.Signal(str)
    #: Delete with several rows selected: ``(layer_ids, roi_ids, source_ids)``, one confirmation.
    removeManyRequested = QtCore.Signal(list, list, list)
    #: "Fork derivative dataset…" on a result row: the shell asks what to fork, then writes it.
    forkDerivativeRequested = QtCore.Signal(int)
    #: "Save derivative as…" on a TEMPORARY derivative's dataset row.
    saveDerivativeRequested = QtCore.Signal(str)
    #: "Route bands to a new bus…" -- a layer of the dataset that will host the bus.
    busRequested = QtCore.Signal(int)
    #: "Edit bus sends…" on a layer whose rack starts with a bus.
    busEditRequested = QtCore.Signal(int)
    #: "Remove band from project" / Delete on a band row: the band LAYER's id.
    removeBandRequested = QtCore.Signal(int)
    #: "Build band stack from N selected layers": the selected layer ids, in tree order.
    stackRequested = QtCore.Signal(list)
    #: ``(layer_id, output name, hidden)`` -- an output row's H under a result's "Outputs".
    outputHideToggled = QtCore.Signal(int, str, bool)

    def __init__(self, project=None, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        # Column 0 (the NAME column) takes the surplus; column 1 (the button widgets) takes only
        # what its buttons need. Why: with no resize mode set at all,
        # column 0 sits at ``QHeaderView``'s fixed 100 px default while the LAST section stretches
        # -- harmless while every header was ``setFirstColumnSpanned(True)`` (a spanned row ignores
        # the columns entirely), but the moment un-spanning made the header respect column 0, a
        # real file stem painted as ``gebco_20...``. ``header.text(0)`` is item DATA and cannot see
        # that; ``test_a_long_source_stem_is_not_elided_in_the_header`` can.
        # ``setStretchLastSection`` defaults to True and OVERRIDES the resize mode of the last
        # section, so without turning it off first the two modes below fight and settle on a 50/50
        # split -- better than 100 px, but it still elides the stem at panel widths the render
        # criterion uses, and it hands half the panel to a button column that needs ~150 px.
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.setHeaderHidden(True)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Shift = a range, cmd = individual rows; Delete acts on all of them.
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self._project = None
        self._source_headers: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._layer_items: dict[int, QtWidgets.QTreeWidgetItem] = {}
        self._row_widgets: dict[int, _LayerRow] = {}
        self._source_rows: dict[str, _SourceRow] = {}
        #: source_id -> its MASTER layer id: the first root layer of a source, whose row IS the
        #: dataset row.
        self._masters: dict[str, int] = {}
        self._roi_items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._band_groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._roi_rows: dict[str, _RoiRow] = {}
        #: layer_id -> its "Outputs" group row, and -> {output name: row widget} for the rows
        #: under it.
        self._output_groups: dict[int, QtWidgets.QTreeWidgetItem] = {}
        self._output_rows: dict[int, dict[str, _OutputRow]] = {}
        #: layer_id -> {output name: (row item, label, kind)}, parallel to _output_rows.
        self._output_items: dict[int, dict[str, tuple]] = {}

        self.currentItemChanged.connect(self._on_current_item_changed)
        self.itemChanged.connect(self._on_item_changed)
        self.itemExpanded.connect(self._on_item_expanded)
        self.itemCollapsed.connect(self._on_item_collapsed)
        self.customContextMenuRequested.connect(self._on_context_menu)

        self.set_project(project)

    # -- project wiring ----------------------------------------------------------------------
    def set_project(self, project) -> None:
        """The ``Project`` a header's expand state reads/writes ``SourceRef.collapsed`` on. Kept
        separate from ``add_layer_row`` (which must keep its exact today's-method signature so
        ``load_field``/ROI-create call sites are untouched) rather than folded into it."""
        self._project = project

    def reset_project(self, project) -> None:
        """Empty the tree and rebind to ``project`` — the open-project path's clean slate.

        Signals are blocked for the duration: tearing down rows must not fire selection or
        item-changed handlers against a window whose own registries are mid-rebuild (the same
        hazard :meth:`remove_rows` guards against, for the whole tree at once)."""
        self.blockSignals(True)
        try:
            self.clear()
            self._source_headers.clear()
            self._layer_items.clear()
            self._row_widgets.clear()
            self._source_rows.clear()
            self._masters.clear()
            self._roi_items.clear()
            self._roi_rows.clear()
            self._output_groups.clear()
            self._output_rows.clear()
            self._output_items.clear()
            self.set_project(project)
        finally:
            self.blockSignals(False)

    # -- building the tree ---------------------------------------------------------------------
    def add_layer_row(self, layer, field) -> None:
        """Add one row for ``layer``, grouped under its source's header (creating the header the
        first time that source is seen) and, for a grouped ROI layer (``layer.parent_id`` set),
        nested under its OWN parent's row instead -- two levels under the header, matching
        ``main_window.roi_chain``'s "an ROI is a child of the layer it was drawn on".

        SAME name and signature as the method ``load_field`` and ROI-create already call on
        ``main_window`` -- see ``main_window.MainWindow.add_layer_row``, which now delegates its
        own tree-row bookkeeping here after doing its OWN ``_row_layers``/``_fields`` bookkeeping,
        unchanged.
        """
        header = self._ensure_source_header(layer.source_id, field)
        if layer.parent_id is None and layer.source_id not in self._masters:
            self._attach_master(header, layer)
            self.sync_output_rows(layer)
            return
        parent_item = (self._layer_items.get(layer.parent_id)
                       if layer.parent_id is not None else None)
        roi_id = getattr(layer, "roi_id", None)
        if roi_id is not None and layer.parent_id is not None:
            # A result computed ON a saved ROI sits under that ROI's row.
            roi_item = self._ensure_roi_row(roi_id)
            if roi_item is not None:
                parent_item = roi_item
        if parent_item is None:
            parent_item = header
        band_row = self._is_band_layer(layer)
        if band_row:
            parent_item = self._band_group(layer.source_id) or parent_item

        # Built DETACHED (no tree/parent argument) and given its flags and _LAYER_ID_ROLE data
        # BEFORE ``addChild`` puts it in the tree -- ``setData`` on an item that
        # is ALREADY in the tree fires ``itemChanged`` immediately, and ``_on_item_changed`` reads
        # that exact role to decide "this is a rename": constructing the item already-parented (as
        # this used to) meant every single ``add_layer_row`` call emitted a spurious
        # ``renameRequested(layer_id, layer.name)`` for a name nobody edited.
        item = QtWidgets.QTreeWidgetItem([layer.name])
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        item.setData(0, _LAYER_ID_ROLE, layer.layer_id)
        parent_item.addChild(item)
        item.setExpanded(True)
        self._layer_items[layer.layer_id] = item

        row = _LayerRow(layer)
        lid = layer.layer_id
        row.hideToggled.connect(lambda checked, lid=lid: self.hideToggled.emit(lid, checked))
        row.lockToggled.connect(lambda checked, lid=lid: self.lockToggled.emit(lid, checked))
        row.freezeToggled.connect(lambda checked, lid=lid: self.freezeToggled.emit(lid, checked))
        self.setItemWidget(item, 1, row)
        self._row_widgets[lid] = row
        if band_row:
            item.setExpanded(False)
            self.sync_band_rows(layer.source_id)
        self.sync_output_rows(layer)

        self.refresh_master_rows()

    def _ensure_source_header(self, source_id: str, field) -> QtWidgets.QTreeWidgetItem:
        """The header row for ``source_id``, creating it (once) from the SOURCE's own file stem --
        not the layer's name, which for anything but the first layer opened over a source (a
        second layer, an ROI) is not the file's name at all. Falls back to ``field.name`` (what
        ``RasterField.from_file``/``open_field`` stamps from the SAME path) when no ``Project`` is
        set, and to the bare ``source_id`` if even that is unavailable -- a header always renders
        something, never raises."""
        item = self._source_headers.get(source_id)
        if item is not None:
            return item
        source = self._project.sources.get(source_id) if self._project is not None else None
        stem = Path(source.path).stem if (source is not None and source.path) else ""
        if not stem:
            stem = getattr(field, "name", None) or source_id
        # A derivative dataset is named at the fork (its file name is only storage), and may
        # nest inside the row of the dataset it came from.
        text = (getattr(source, "label", "") or stem) if source is not None else stem
        nest = self._source_headers.get(getattr(source, "nest_under", None) or "")
        item = QtWidgets.QTreeWidgetItem(nest if nest is not None else self, [text])
        # NOT spanned (this argument used to be ``True``): a header whose column 0
        # spans the whole row covers column 1, and an item widget put there would never be shown.
        # Un-spanning is what makes the inspector toggle below visible at all; the label in column
        # 0 is unchanged, which ``test_header_text_still_reads_the_file_stem`` guards.
        item.setFirstColumnSpanned(False)
        item.setData(0, _SOURCE_ID_ROLE, source_id)
        collapsed = bool(source.collapsed) if source is not None else False
        item.setExpanded(not collapsed)
        self._source_headers[source_id] = item

        row = _SourceRow(hidden=bool(getattr(source, "hidden", False)))
        row.inspectorToggled.connect(
            lambda checked, sid=source_id: self.inspectorToggled.emit(sid, checked))
        row.hideToggled.connect(
            lambda checked, sid=source_id: self.sourceHideToggled.emit(sid, checked))
        row.lockToggled.connect(
            lambda checked, sid=source_id: self._emit_for_master(self.lockToggled, sid, checked))
        row.freezeToggled.connect(
            lambda checked, sid=source_id: self._emit_for_master(self.freezeToggled, sid,
                                                                 checked))
        self.setItemWidget(item, 1, row)
        self._source_rows[source_id] = row
        return item

    # -- band rows (a stack dataset's bands) ---------------------------------------------------
    def _is_band_layer(self, layer) -> bool:
        """A band row: a layer tagged ``band.id`` directly under its dataset's master."""
        return bool(layer.tags.get("band.id")) and layer.parent_id is not None \
            and self._masters.get(layer.source_id) == layer.parent_id

    def _band_group(self, source_id: str):
        """The collapsed "Bands (n)" row first under the dataset row, created on demand."""
        group = self._band_groups.get(source_id)
        header = self._source_headers.get(source_id)
        if group is not None or header is None:
            return group
        self.blockSignals(True)          # detached build: the add_layer_row itemChanged trap
        try:
            group = QtWidgets.QTreeWidgetItem(["Bands"])
            group.setData(0, _BAND_ROLE, -1)
            group.setData(0, _BAND_SOURCE_ROLE, source_id)
            group.setToolTip(0, "this dataset's bands — drop a tool on one to run it on that "
                                "band; right-click / Delete removes a band from the project")
            header.insertChild(0, group)
            group.setExpanded(False)
        finally:
            self.blockSignals(False)
        self._band_groups[source_id] = group
        return group

    def sync_band_rows(self, source_id: str, _names=None) -> None:
        """Keep the "Bands (n)" row honest: its count, and gone when no band rows are left.
        (The band rows themselves are layers, added and removed like any row.)"""
        group = self._band_groups.get(source_id)
        if group is None:
            return
        if group.childCount() == 0 or _names == []:
            self._band_groups.pop(source_id, None)
            parent = group.parent()
            if parent is not None:
                parent.removeChild(group)
            return
        group.setText(0, f"Bands ({group.childCount()})")

    # -- output rows (what a result produces, each shown or hidden on its own) -------------------
    def sync_output_rows(self, layer) -> None:
        """Keep ``layer``'s "Outputs" group matching what its chain's last transform declares
        (:func:`layer_outputs`): built collapsed the first time, rebuilt when the declared set
        changes, removed when there is none. A new row reads its state from the layer once, at
        construction (the :class:`_LayerRow` rule): a raster row is hidden unless the step's
        ``show`` names it, the vector row is hidden while the ``ui.edges_hidden`` tag is set.
        Idempotent; the window calls it after every chain edit."""
        lid = layer.layer_id
        item = self._layer_items.get(lid)
        step, outputs = layer_outputs(layer) if item is not None else (None, ())
        if [o.name for o in outputs] == list(self._output_items.get(lid, {})):
            return
        self._forget_output_rows(lid)
        if not outputs:
            return
        shown = shown_output(step, outputs)
        edges_hidden = layer.tags.get("ui.edges_hidden") == "1"
        rows: dict[str, _OutputRow] = {}
        items: dict[str, tuple] = {}
        self.blockSignals(True)          # detached build: the add_layer_row itemChanged trap
        try:
            group = QtWidgets.QTreeWidgetItem(["Outputs"])
            group.setData(0, _OUTPUT_LAYER_ROLE, lid)
            group.setToolTip(0, "what this result produces — clear H on a raster row to show it "
                                "in place of the data (one at a time); H on edges hides the "
                                "edges drawing")
            item.insertChild(0, group)
            group.setExpanded(False)
            for out in outputs:
                label = out.label or out.name
                child = QtWidgets.QTreeWidgetItem([label])
                child.setData(0, _OUTPUT_KEY_ROLE, out.name)
                child.setData(0, _OUTPUT_LAYER_ROLE, lid)
                group.addChild(child)
                hidden = edges_hidden if out.kind == "vector" else out.name != shown
                row = _OutputRow(out, hidden)
                row.hideToggled.connect(
                    lambda checked, lid=lid, name=out.name:
                    self.outputHideToggled.emit(lid, name, checked))
                self.setItemWidget(child, 1, row)
                rows[out.name] = row
                items[out.name] = (child, label, out.kind)
        finally:
            self.blockSignals(False)
        self._output_groups[lid] = group
        self._output_rows[lid] = rows
        self._output_items[lid] = items

    def set_output_state(self, layer_id: int, shown: str | None, edges_hidden: bool,
                         notes: dict[str, str] | None = None,
                         disabled: dict[str, str] | None = None) -> None:
        """Push the model's output state into ``layer_id``'s rows without emitting: ``shown``
        is the raster output on the canvas (``None``: the raw field), ``edges_hidden`` the
        vector row's H. ``notes[name]`` is text shown after that row's name; ``disabled[name]``
        greys that row out with the reason as its tooltip. A layer without rows is a no-op.
        Each button's own signals are blocked for its ``setChecked`` (never the tree's)."""
        rows = self._output_rows.get(layer_id)
        if not rows:
            return
        notes, disabled = notes or {}, disabled or {}
        for name, row in rows.items():
            child, label, kind = self._output_items[layer_id][name]
            hidden = bool(edges_hidden) if kind == "vector" else name != shown
            row.hide_button.blockSignals(True)
            try:
                row.hide_button.setChecked(hidden)
            finally:
                row.hide_button.blockSignals(False)
            text = f"{label} · {notes[name]}" if notes.get(name) else label
            if child.text(0) != text:
                child.setText(0, text)
            reason = disabled.get(name)
            child.setDisabled(reason is not None)
            child.setToolTip(0, reason or "")
            row.hide_button.setEnabled(reason is None)
            row.hide_button.setToolTip(reason or row.tooltip)

    def _forget_output_rows(self, layer_id: int) -> None:
        """Drop ``layer_id``'s output rows from the registries and take its group out of the
        tree (the dataset row outlives its master, so its group must go explicitly)."""
        group = self._output_groups.pop(layer_id, None)
        self._output_rows.pop(layer_id, None)
        self._output_items.pop(layer_id, None)
        if group is not None and group.parent() is not None:
            group.parent().removeChild(group)

    def _attach_master(self, header, layer) -> None:
        """Make ``header`` the row of ``layer``, its source's master: one row per dataset. Its H
        hides the raster, its L/F are the master's, selecting it selects the master, and the
        master's children nest under it. Signals are blocked for the ``setData`` (the
        ``add_layer_row`` spurious-rename trap)."""
        self.blockSignals(True)
        try:
            header.setData(0, _LAYER_ID_ROLE, layer.layer_id)
        finally:
            self.blockSignals(False)
        row = self._source_rows[layer.source_id]
        row.attach_master(layer)
        self._masters[layer.source_id] = layer.layer_id
        self._layer_items[layer.layer_id] = header
        self._row_widgets[layer.layer_id] = row

    def _emit_for_master(self, signal, source_id: str, checked: bool) -> None:
        layer_id = self._masters.get(source_id)
        if layer_id is not None:
            signal.emit(layer_id, checked)

    # -- saved-ROI rows --------------------------------------------------------------------------
    def _dataset_name(self, source_id: str) -> str:
        """What an ROI row is named after: the dataset's master layer, else its row's text."""
        master_id = self._masters.get(source_id)
        if self._project is not None and master_id is not None:
            master = next((l for l in self._project.layers if l.layer_id == master_id), None)
            if master is not None:
                return master.name
        header = self._source_headers.get(source_id)
        return header.text(0) if header is not None else source_id

    def _ensure_roi_row(self, roi_id: str):
        """The row for saved ROI ``roi_id`` under its dataset row, created once from the
        project's ``RoiRecord``; ``None`` when the record, or its dataset's row, does not exist."""
        item = self._roi_items.get(roi_id)
        if item is not None:
            return item
        if self._project is None:
            return None
        roi = next((r for r in self._project.rois if r.roi_id == roi_id), None)
        if roi is None:
            return None
        header = self._source_headers.get(roi.source_id)
        if header is None:
            return None
        # Detached + data set BEFORE addChild: the add_layer_row spurious-itemChanged trap.
        item = QtWidgets.QTreeWidgetItem([f"{self._dataset_name(roi.source_id)} @{roi.label}"])
        item.setData(0, _ROI_ID_ROLE, roi_id)
        header.addChild(item)
        item.setExpanded(True)
        self._roi_items[roi_id] = item
        row = _RoiRow(roi)
        row.hideToggled.connect(
            lambda checked, rid=roi_id: self.roiHideToggled.emit(rid, checked))
        self.setItemWidget(item, 1, row)
        self._roi_rows[roi_id] = row
        return item

    def sync_roi_rows(self) -> None:
        """One row per saved ROI of every dataset on the list; rows whose record is gone go
        too. Idempotent -- the window calls it wherever the saved ROIs may have changed."""
        if self._project is None:
            return
        live = {r.roi_id for r in self._project.rois}
        for roi in self._project.rois:
            self._ensure_roi_row(roi.roi_id)
        for roi_id in [rid for rid in self._roi_items if rid not in live]:
            self._forget_roi_row(roi_id, take=True)

    def _forget_roi_row(self, roi_id: str, *, take: bool) -> None:
        item = self._roi_items.pop(roi_id, None)
        self._roi_rows.pop(roi_id, None)
        if take and item is not None and item.parent() is not None:
            item.parent().removeChild(item)

    def select_roi(self, roi_id: str) -> None:
        """Make saved ROI ``roi_id``'s row current (emits ``roiSelected``)."""
        item = self._roi_items.get(roi_id)
        if item is None:
            return
        ancestor = item.parent()
        while ancestor is not None:
            ancestor.setExpanded(True)
            ancestor = ancestor.parent()
        self.setCurrentItem(item)

    def remove_rows(self, layer_ids) -> None:
        """Remove every row named in ``layer_ids`` (the ``Project.remove_layer`` return:
        children-first, then the parent -- order does not matter here, since each removal is
        independent). A source header left with zero children afterward is pruned too: an empty
        header naming a file with nothing under it is worse than no header at all.

        Signals are blocked for the whole operation. Removing the CURRENT item makes Qt promote
        some other row to current on its own -- a sibling, a parent, whatever the view's own
        removal heuristic lands on -- and without this guard that promotion fires a real
        ``currentItemChanged`` -> ``layerSelected`` for a row nobody asked to select (a caller mid-cascade-removal saw ``layerSelected`` for a row about to be removed
        two lines later, or for an unrelated survivor Qt merely happened to land on). The caller
        picks the actual survivor explicitly, through :meth:`select_layer`, once this call
        returns -- that is the ONE selection this operation is allowed to produce.

        The current item is explicitly cleared (to ``None``) ONLY when it was itself one of the
        removed rows -- never unconditionally (an earlier version cleared it every
        time, which silently dropped the panel's selection highlight when removing a row that
        WASN'T current at all; Qt's own default -- leave an unrelated current item alone -- was
        already correct there and needed no help). Clearing when it WAS removed exists for a
        subtler reason than the blocked signal alone covers: Qt's own silent promotion (still
        blocked, but still REAL as far as its internal current-index is concerned) can land on the
        exact survivor the caller is about to pass to ``select_layer`` -- e.g. the only layer left
        under a source, after its one sibling was removed. ``setCurrentItem`` on an item that is
        ALREADY current is not a transition, so it would not re-fire the signal either, and the
        caller's explicit reselect would silently do nothing. Clearing first guarantees the
        caller's own ``select_layer`` is always a genuine None -> survivor transition.

        ``blockSignals`` is a flag, not a counter -- this pair does not nest. No current caller
        calls ``remove_rows`` from inside another blocked region, but a future one that did would
        have the INNER call's ``finally`` unblock signals early, for the OUTER call's remaining
        work. Fine today; worth remembering if that ever changes.
        """
        self.blockSignals(True)
        try:
            current = self.currentItem()
            current_removed = False
            touched_headers: set[QtWidgets.QTreeWidgetItem] = set()
            for layer_id in layer_ids:
                self._forget_output_rows(layer_id)
                item = self._layer_items.pop(layer_id, None)
                self._row_widgets.pop(layer_id, None)
                if item is None:
                    continue
                if item is current:
                    current_removed = True
                source_id = item.data(0, _SOURCE_ID_ROLE)
                if source_id is not None:
                    # The master's row is the dataset row: drop the master from it; the row
                    # goes below only if nothing else is left under it.
                    item.setData(0, _LAYER_ID_ROLE, None)
                    self._masters.pop(source_id, None)
                    self.sync_band_rows(str(source_id), [])      # the bands go with it
                    row = self._source_rows.get(source_id)
                    if row is not None:
                        row.detach_master()
                    touched_headers.add(item)
                    continue
                parent = item.parent()
                if parent is None:
                    self.takeTopLevelItem(self.indexOfTopLevelItem(item))
                    continue
                parent.removeChild(item)
                touched_headers.add(parent)
                band_source = parent.data(0, _BAND_SOURCE_ROLE)
                if band_source is not None:
                    self.sync_band_rows(str(band_source))
            for source_id, header in list(self._source_headers.items()):
                # A dataset row still carrying its master is not empty -- it IS a layer row.
                if (header in touched_headers and header.childCount() == 0
                        and header.data(0, _LAYER_ID_ROLE) is None):
                    self._take_header(header)
                    del self._source_headers[source_id]
                    self._source_rows.pop(source_id, None)   # never leak a pruned header's row
                    self._masters.pop(source_id, None)
                    self._forget_roi_rows_of(header)
            if current_removed:
                self.setCurrentItem(None)
        finally:
            self.blockSignals(False)

    # -- selection -------------------------------------------------------------------------------
        self.refresh_master_rows()

    def select_layer(self, layer_id: int) -> None:
        """Make ``layer_id``'s row current, expanding every ancestor first -- a layer nested under
        a collapsed header (or a collapsed parent layer) must still become selectable, and
        selecting it is as good a reason as any to reveal it."""
        item = self._layer_items.get(layer_id)
        if item is None:
            return
        ancestor = item.parent()
        while ancestor is not None:
            ancestor.setExpanded(True)
            ancestor = ancestor.parent()
        self.setCurrentItem(item)

    def current_layer_id(self) -> int | None:
        item = self.currentItem()
        if item is None:
            return None
        layer_id = item.data(0, _LAYER_ID_ROLE)
        return None if layer_id is None else int(layer_id)

    def layer_text(self, layer_id: int) -> str | None:
        item = self._layer_items.get(layer_id)
        return None if item is None else item.text(0)

    def count(self) -> int:
        """Number of LAYER rows tracked (source headers excluded) -- the tree equivalent of the
        old flat ``QListWidget.count()`` call sites already use."""
        return len(self._layer_items)

    def remove_source_row(self, source_id: str) -> None:
        """Drop a source HEADER row (the dataset-removal tail): the shell removes the layer
        rows through :meth:`remove_rows` first, so this only takes the emptied group item.
        A derivative dataset nested inside it is lifted to the top level first -- it is a
        dataset of its own and outlives the one it came from."""
        item = self._source_headers.get(source_id)
        if item is None:
            return
        nested = [str(c.data(0, _SOURCE_ID_ROLE))
                  for c in (item.child(i) for i in range(item.childCount()))
                  if c.data(0, _SOURCE_ID_ROLE) is not None]
        for nested_id in nested:
            self._lift_dataset(nested_id)
        self._take_header(item)
        self._source_headers.pop(source_id, None)
        self._source_rows.pop(source_id, None)
        self._masters.pop(source_id, None)
        self._forget_roi_rows_of(item)

    def _take_header(self, header) -> None:
        """Take a dataset row out of the tree, top-level or nested."""
        parent = header.parent()
        if parent is not None:
            parent.removeChild(header)
        else:
            idx = self.indexOfTopLevelItem(header)
            if idx >= 0:
                self.takeTopLevelItem(idx)

    def _lift_dataset(self, source_id: str) -> None:
        """A nested dataset row moves to the top level: its dataset outlives the one it sat in.
        ``nest_under`` is display state (like ``collapsed``), cleared here. The rows are
        REBUILT, never moved -- Qt deletes an item's row widgets when it leaves the tree -- the
        way "Delete layer" re-nests rows."""
        source = self._project.sources.get(source_id) if self._project is not None else None
        if source is not None:
            source.nest_under = None
        layers = ([l for l in self._project.layers if l.source_id == source_id]
                  if self._project is not None else [])
        current = self.current_layer_id()
        self.blockSignals(True)
        try:
            self.remove_rows([l.layer_id for l in layers])
            self.remove_source_row(source_id)
            for layer in layers:
                self.add_layer_row(layer, None)
            self.sync_roi_rows()
            if current is not None and current in self._layer_items:
                self.select_layer(current)
        finally:
            self.blockSignals(False)

    def _forget_roi_rows_of(self, header) -> None:
        for roi_id in [rid for rid, it in self._roi_items.items() if it.parent() is header]:
            self._forget_roi_row(roi_id, take=False)

    def refresh_master_rows(self) -> None:
        """A no-op, kept as the call ``main_window`` makes after every add/remove and after chain
        edits. The master has no row of its own (the dataset row IS its row,
        :meth:`_attach_master`), so there is nothing to collapse -- and hiding a lone SECOND root
        layer (what the scan below would find) is not wanted.

        The unreachable scan below is the record of the rule it implemented: hide each source's
        MASTER row while it is the source's only layer AND its chain is empty. One row until
        there is something to distinguish: the master row appears the moment its chain gains a
        step or a child forks. A hidden master still works through the header (header clicks
        promote onto it -- selection, rack, Delete all reach it), so nothing is lost while it
        is collapsed. It is a cheap full scan over the handful of headers."""
        return
        if self._project is None:
            return
        layers_by_id = {l.layer_id: l for l in self._project.layers}
        for i in range(self.topLevelItemCount()):
            header = self.topLevelItem(i)
            if header.data(0, _LAYER_ID_ROLE) is not None:
                continue
            rows = [header.child(j) for j in range(header.childCount())]
            rows = [r for r in rows if r.data(0, _LAYER_ID_ROLE) is not None]
            for row in rows:
                layer = layers_by_id.get(int(row.data(0, _LAYER_ID_ROLE)))
                # Forked/ROI children nest UNDER the master row, so "only layer" means: sole
                # direct row AND no nested children of its own.
                collapse = (len(rows) == 1 and row.childCount() == 0 and layer is not None
                            and layer.parent_id is None and not layer.chain.steps)
                row.setHidden(bool(collapse))

    def set_layer_name(self, layer_id: int, name: str) -> None:
        """Push a rename INTO the row's displayed text. Signals are blocked for the call: this is
        ``main_window`` writing its OWN (possibly sanitised) ``layer.name`` back after a
        ``renameRequested`` it just handled, and without the guard that ``setText`` would re-fire
        ``itemChanged`` -> a second, redundant ``renameRequested`` for the very edit that is
        already being applied."""
        item = self._layer_items.get(layer_id)
        if item is None:
            return
        self.blockSignals(True)
        try:
            item.setText(0, name)
        finally:
            self.blockSignals(False)

    def set_inspector_open(self, source_id: str, open_: bool) -> None:
        """Push an inspector's OWN open/closed state back into its header toggle -- the other half
        of ``InspectorWindow.closed``: a window the user closed on the window itself and the
        button that opened it are ONE state (spec 5b).

        Signals are blocked for the call, the same guard :meth:`set_layer_name` uses and for the
        same reason: without it ``setChecked`` would re-fire ``inspectorToggled`` for the very
        transition already being applied, telling ``main_window`` to close a window that just
        closed itself (or to reopen it).

        This is the SECOND blocked region in the file, so :meth:`remove_rows`'s closing note now
        has a second way to bite: ``blockSignals`` is a flag, not a counter. If anything ever routes
        ``InspectorWindow.closed`` -> here from inside a ``remove_rows`` cascade (a pruned header
        closing the window it owned), this call's ``finally`` unblocks signals for the OUTER call's
        remaining work. Not reachable today -- no caller does -- but it is the reason to close a
        pruned header's window AFTER ``remove_rows`` returns, not from inside it."""
        row = self._source_rows.get(source_id)
        if row is None:
            return
        self.blockSignals(True)
        try:
            row.inspector_button.setChecked(bool(open_))
        finally:
            self.blockSignals(False)

    def flash_row(self, layer_id: int) -> None:
        """The chip-hover feedback, stubbed: mark the row's button widget ``flash="true"``
        and clear it one second later. No QSS rule keys off ``flash`` yet (icon/chrome art is not
        this slice); the property and its timed clear are the part the wiring needs to exist
        and be correct today.

        The timer is a REAL ``QTimer`` parented to ``self`` (the panel), not the static
        ``QTimer.singleShot(ms, callable)`` form -- that form owns nothing, so a panel (or row)
        torn down before the second elapses leaves it firing anyway, into a widget whose
        underlying C++ object is already gone (``RuntimeError: ... already deleted``, caught only
        by a later, unrelated test running right as the stray timer landed). Parenting to ``self``
        makes Qt's own ownership tree cancel the pending shot the moment the panel is destroyed.
        """
        widget = self._row_widgets.get(layer_id)
        if widget is None:
            return
        widget.setProperty("flash", "true")
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._clear_flash(layer_id))
        timer.start(_FLASH_MS)

    def _clear_flash(self, layer_id: int) -> None:
        widget = self._row_widgets.get(layer_id)
        if widget is None:
            return
        widget.setProperty("flash", "false")
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # -- tree signals ------------------------------------------------------------------------
    def _on_current_item_changed(self, current, previous) -> None:
        if current is None:
            return
        roi_id = current.data(0, _ROI_ID_ROLE)
        if roi_id is not None:
            self.roiSelected.emit(str(roi_id))
            return
        band_source = current.data(0, _BAND_SOURCE_ROLE)
        if band_source is not None:
            master = self._masters.get(str(band_source))   # the Bands row shows its dataset
            if master is not None:
                self.layerSelected.emit(int(master))
            return
        output_layer = current.data(0, _OUTPUT_LAYER_ROLE)
        if output_layer is not None:
            self.layerSelected.emit(int(output_layer))     # an output row shows its layer
            return
        layer_id = current.data(0, _LAYER_ID_ROLE)
        if layer_id is not None:
            self.layerSelected.emit(int(layer_id))
            return
        # A source HEADER became current: promote
        # the click onto the source's first layer row -- the master, root-protected to stay
        # the raw dataset -- so the header is a real selection target instead of a no-op that
        # strands whatever rack was up. setCurrentItem re-enters this handler with a real
        # layer row, which emits normally.
        for i in range(current.childCount()):
            child = current.child(i)
            if child.data(0, _LAYER_ID_ROLE) is not None:
                self.setCurrentItem(child)
                return

    def _on_item_changed(self, item, column: int) -> None:
        """A layer row's name column was edited (the built-in inline ``QLineEdit`` editor, from a
        double-click or the context menu's Rename). Header rows never carry ``_LAYER_ID_ROLE`` and
        are never made editable in the first place, so this only ever fires for a real rename."""
        if column != 0:
            return
        layer_id = item.data(0, _LAYER_ID_ROLE)
        if layer_id is not None:
            self.renameRequested.emit(int(layer_id), item.text(0))

    def _on_item_expanded(self, item) -> None:
        self._sync_collapsed(item, False)

    def _on_item_collapsed(self, item) -> None:
        self._sync_collapsed(item, True)

    def _sync_collapsed(self, item, collapsed: bool) -> None:
        """The OTHER direction of the header <-> ``SourceRef.collapsed`` sync: a user's own
        expand/collapse click (or a ``select_layer`` auto-reveal) writes straight back into the
        model, same as ``_ensure_source_header`` reads it out. A no-op for anything that is not a
        source header (``_SOURCE_ID_ROLE`` unset) -- a layer row can also be expanded/collapsed
        (it may have ROI children), but that has no ``SourceRef`` counterpart to write into."""
        if self._project is None:
            return
        source_id = item.data(0, _SOURCE_ID_ROLE)
        if source_id is None:
            return
        source = self._project.sources.get(source_id)
        if source is not None:
            source.collapsed = collapsed

    # -- context menu --------------------------------------------------------------------------
    def _build_context_menu(self, item, layer_id: int) -> QtWidgets.QMenu:
        """Split out from :meth:`_on_context_menu` so a test can drive the three actions directly
        (``.trigger()``) without going through the real, blocking ``QMenu.exec`` popup loop --
        the same reason none of ``knob_widgets.py``'s own ``contextMenuEvent`` gestures are driven
        end-to-end in that module's tests either."""
        menu = QtWidgets.QMenu(self)
        layer = next((l for l in (self._project.layers if self._project is not None else [])
                      if l.layer_id == layer_id), None)
        if layer is not None and self._is_band_layer(layer):
            # A band row: removing it removes the BAND from the dataset (never the file).
            band_action = menu.addAction("Remove band from project")
            band_action.triggered.connect(lambda: self.removeBandRequested.emit(layer_id))
            fork_action = menu.addAction("Fork derivative dataset…")
            fork_action.triggered.connect(lambda: self.forkDerivativeRequested.emit(layer_id))
            bus_action = menu.addAction("Route bands to a new bus…")
            bus_action.triggered.connect(lambda: self.busRequested.emit(layer_id))
            return menu
        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(lambda: self.editItem(item, 0))
        # Two deletes: this layer alone (its children move up to its parent), or this layer
        # and everything under it.
        only_action = menu.addAction("Delete layer")
        only_action.triggered.connect(lambda: self.removeLayerOnlyRequested.emit(layer_id))
        remove_action = menu.addAction("Delete layer and children")
        remove_action.triggered.connect(lambda: self.removeRequested.emit(layer_id))
        refined_action = menu.addAction("New refined run")
        refined_action.triggered.connect(lambda: self.refinedRunRequested.emit(layer_id))
        # A DERIVATIVE dataset: what this result shows, written once as a dataset of its own --
        # unlike a child, it never re-processes.
        fork_action = menu.addAction("Fork derivative dataset…")
        fork_action.triggered.connect(lambda: self.forkDerivativeRequested.emit(layer_id))
        layer = next((l for l in (self._project.layers if self._project is not None else [])
                      if l.layer_id == layer_id), None)
        steps = layer.chain.steps if layer is not None else []
        if steps and steps[0].device == "bus":
            edit_bus = menu.addAction("Edit bus sends…")
            edit_bus.triggered.connect(lambda: self.busEditRequested.emit(layer_id))
        bus_action = menu.addAction("Route bands to a new bus…")
        bus_action.triggered.connect(lambda: self.busRequested.emit(layer_id))
        return menu

    def _build_source_context_menu(self, source_id: str) -> QtWidgets.QMenu:
        """A dataset row's menu: "Remove dataset" (the shell owns the confirmation and cascade),
        preceded by "Save derivative as…" on a TEMPORARY derivative."""
        menu = QtWidgets.QMenu(self)
        source = self._project.sources.get(source_id) if self._project is not None else None
        if source is not None and source.temporary:
            save_action = menu.addAction("Save derivative as…")
            save_action.triggered.connect(lambda: self.saveDerivativeRequested.emit(source_id))
        master = next((l for l in (self._project.layers if self._project is not None else [])
                       if l.source_id == source_id and l.parent_id is None), None)
        if master is not None:
            bus_action = menu.addAction("Route bands to a new bus…")
            bus_action.triggered.connect(lambda: self.busRequested.emit(master.layer_id))
        remove_src = menu.addAction("Remove dataset")
        remove_src.triggered.connect(lambda: self.removeSourceRequested.emit(source_id))
        return menu

    def _sort_selection(self, items) -> tuple:
        """``(layer_ids, roi_ids, source_ids, band_layer_ids)`` of a multi-selection (a
        dataset row counts as its source; band rows apart -- removing one removes a band)."""
        layer_ids, roi_ids, source_ids = [], [], []
        for it in items:
            if it.data(0, _SOURCE_ID_ROLE) is not None:
                source_ids.append(str(it.data(0, _SOURCE_ID_ROLE)))
            elif it.data(0, _ROI_ID_ROLE) is not None:
                roi_ids.append(str(it.data(0, _ROI_ID_ROLE)))
            elif it.data(0, _LAYER_ID_ROLE) is not None:
                layer_ids.append(int(it.data(0, _LAYER_ID_ROLE)))
        by_id = {l.layer_id: l for l in (self._project.layers
                                         if self._project is not None else [])}
        bands = [i for i in layer_ids if i in by_id and self._is_band_layer(by_id[i])]
        return [i for i in layer_ids if i not in bands], roi_ids, source_ids, bands

    def _remove_selection(self, items) -> None:
        layer_ids, roi_ids, source_ids, bands = self._sort_selection(items)
        if layer_ids or roi_ids or source_ids:
            self.removeManyRequested.emit(layer_ids, roi_ids, source_ids)
        for i in bands:
            self.removeBandRequested.emit(i)

    def _build_multi_context_menu(self, items) -> QtWidgets.QMenu:
        """Right-click on a row inside a multi-selection: act on ALL selected rows."""
        menu = QtWidgets.QMenu(self)
        stackable = [int(it.data(0, _LAYER_ID_ROLE)) for it in items
                     if it.data(0, _LAYER_ID_ROLE) is not None]
        if len(stackable) >= 2:
            stack = menu.addAction(f"Build band stack from {len(stackable)} selected layers")
            stack.triggered.connect(lambda: self.stackRequested.emit(stackable))
        delete = menu.addAction(f"Delete {len(items)} selected rows")
        delete.triggered.connect(lambda: self._remove_selection(items))
        return menu

    def _build_roi_context_menu(self, roi_id: str) -> QtWidgets.QMenu:
        """An ROI row's menu: Delete ROI (the shell refuses while results use it)."""
        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction("Delete ROI")
        delete_action.triggered.connect(lambda: self.removeRoiRequested.emit(roi_id))
        return menu

    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        selected = self.selectedItems()
        if item in selected and len(selected) > 1:
            self._build_multi_context_menu(selected).exec(self.viewport().mapToGlobal(pos))
            return
        roi_id = item.data(0, _ROI_ID_ROLE)
        if roi_id is not None:
            self._build_roi_context_menu(str(roi_id)).exec(self.viewport().mapToGlobal(pos))
            return
        if item.data(0, _BAND_SOURCE_ROLE) is not None:
            return                                        # the Bands row: nothing to do
        layer_id = item.data(0, _LAYER_ID_ROLE)
        if layer_id is None or item.data(0, _SOURCE_ID_ROLE) is not None:
            # A source header names a raster, not a layer -- so it gets the DATASET action:
            # one entry,
            # the whole family; the shell handler owns the confirmation and cascade.
            source_id = item.data(0, _SOURCE_ID_ROLE)
            if source_id is None:
                return
            self._build_source_context_menu(str(source_id)).exec(
                self.viewport().mapToGlobal(pos))
            return
        menu = self._build_context_menu(item, int(layer_id))
        menu.exec(self.viewport().mapToGlobal(pos))

    # -- keyboard ------------------------------------------------------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Spec 4.4's Delete-key remove: Delete OR Backspace -- this app is macOS-only,
        and Backspace is the physical key labeled "delete" on a Mac keyboard; there is no separate
        forward-delete key to also bind. Emits the exact same ``removeRequested`` the context
        menu's Remove action does, so ``main_window`` owns one confirm/refuse flow (locked layer,
        cascade confirm) for both gestures rather than a second implementation of it here.

        A source-header row current (``current_layer_id()`` is ``None``: it names no layer) or any
        other key defers to Qt's own handling -- consistent with every other current-item read in
        this class (``_on_context_menu``, ``_on_current_item_changed``).
        """
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            selected = self.selectedItems()
            if len(selected) > 1:
                self._remove_selection(selected)     # one batch, one confirmation
                return
            item = self.currentItem()
            roi_id = item.data(0, _ROI_ID_ROLE) if item is not None else None
            if roi_id is not None:
                self.removeRoiRequested.emit(str(roi_id))
                return
            if item is not None and item.data(0, _BAND_SOURCE_ROLE) is not None:
                return                                    # the Bands row itself
            layer_id = self.current_layer_id()
            layer = next((l for l in (self._project.layers if self._project is not None
                                      else []) if l.layer_id == layer_id), None)
            if layer is not None and self._is_band_layer(layer):
                self.removeBandRequested.emit(int(layer_id))
                return
            source_id = item.data(0, _SOURCE_ID_ROLE) if item is not None else None
            if source_id is not None:
                # The dataset row: Delete removes the dataset (the shell confirms once).
                self.removeSourceRequested.emit(str(source_id))
                return
            layer_id = self.current_layer_id()
            if layer_id is not None:
                self.removeRequested.emit(layer_id)
                return
        super().keyPressEvent(event)


class ReferencePanel(QtWidgets.QListWidget):
    """The reference layers (GIS vector files drawn OVER the data): one checkable
    row per layer with its colour square. Sits under the layer list; a checkbox toggles
    visibility everywhere (canvas + world). Lives in this module so devloop's reload registry
    needs no new entry. Outward wires: ``visibilityToggled(ref_id, visible)``,
    ``zoomRequested(ref_id)`` and ``removeRequested(ref_id)``."""

    visibilityToggled = QtCore.Signal(str, bool)
    zoomRequested = QtCore.Signal(str)        # double-click: fit the view to this layer's extent
    removeRequested = QtCore.Signal(str)      # context menu: unload the layer (file untouched)

    def __init__(self, parent=None):
        super().__init__(parent)
        # ExtendedSelection (a shp package opens many layers and each had to
        # be unchecked one at a time): shift/ctrl-click selects a range or individual multiples,
        # and toggling ONE selected row's checkbox propagates to every selected row.
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._propagating = False
        self.setToolTip("Reference layers — interpretation drawn over the data; tick to show, "
                        "double-click to zoom, right-click to remove. Shift/⌘-click to select "
                        "several, then tick one to toggle them all")
        self.itemChanged.connect(self._on_item_changed)
        self.itemDoubleClicked.connect(
            lambda item: self.zoomRequested.emit(str(item.data(QtCore.Qt.UserRole))))
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def _on_context_menu(self, pos) -> None:
        """Remove the row under the cursor -- or, when it is part of a multi-selection, every
        selected row (the same propagation rule as the visibility checkbox). Removing only
        unloads: the file stays on disk, and the menu label says so."""
        item = self.itemAt(pos)
        if item is None:
            return
        selected = self.selectedItems()
        rows = selected if item in selected and len(selected) > 1 else [item]
        menu = QtWidgets.QMenu(self)
        label = ("Remove layer" if len(rows) == 1 else f"Remove {len(rows)} layers")
        action = menu.addAction(f"{label} (file kept on disk)")
        if self._exec_menu(menu, self.mapToGlobal(pos)) is action:
            for row in rows:
                self.removeRequested.emit(str(row.data(QtCore.Qt.UserRole)))

    def _exec_menu(self, menu, global_pos):
        """The one popup call, separated so an offscreen test can choose an action without a
        real menu event loop (which never returns when nobody can click)."""
        return menu.exec(global_pos)

    def set_records(self, records) -> None:
        """Mirror the project's ReferenceLayerRecords (non-emitting)."""
        self.blockSignals(True)
        self.clear()
        for r in records:
            item = QtWidgets.QListWidgetItem(r.name)
            item.setData(QtCore.Qt.UserRole, r.ref_id)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if r.visible else QtCore.Qt.Unchecked)
            pix = QtGui.QPixmap(12, 12)
            pix.fill(QtGui.QColor(r.color))
            item.setIcon(QtGui.QIcon(pix))
            self.addItem(item)
        self.blockSignals(False)
        self.setVisible(self.count() > 0)

    def _on_item_changed(self, item) -> None:
        state = item.checkState()
        checked = state == QtCore.Qt.Checked
        # When the toggled row is part of a multi-selection, apply the new state to every
        # selected row -- the classic "select several, tick one, toggle all".
        selected = self.selectedItems()
        if not self._propagating and item in selected and len(selected) > 1:
            self._propagating = True
            try:
                for other in selected:
                    if other is not item and other.checkState() != state:
                        other.setCheckState(state)      # re-enters here, propagating guard on
                        self.visibilityToggled.emit(str(other.data(QtCore.Qt.UserRole)), checked)
            finally:
                self._propagating = False
        self.visibilityToggled.emit(str(item.data(QtCore.Qt.UserRole)), checked)
