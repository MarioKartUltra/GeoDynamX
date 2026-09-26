# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""TransectPanel: the right panel's "Transect" section.

The list of drawn transects -- one row per :class:`~dynamix.model.project.TransectRecord`, named
``T{transect_id}`` via a MONOTONIC counter that NEVER reuses a value, even across a delete + undo
(the collision lesson worth
restating here since it is the whole reason this widget looks the way it does:
``_next_transect_line_name``'s own docstring records that ``len(transects) + 1`` collides after a
middle delete -- two live rows can derive the SAME name, and ``add_mesh(name=...)`` then silently
overwrites the first one's drawn actor, so one line vanishes from the screen while its list row
still shows. A counter that only ever increases cannot repeat a value no matter how many
add/delete/undo cycles happen in between -- this is a strict improvement on EQSelect's own scheme,
which has to re-mint fresh actor names on every undo specifically to dodge this; here the ORIGINAL
``transect_id`` is always safe to restore unchanged, because nothing else could ever have taken it
while it was gone.

Per-row visibility checkbox; ONE "Buffer (px)" spin that edits the CURRENTLY SELECTED row's own
buffer (EQSelect's own "live buffer of the selected transect... re-selects the swath on every
tick", ported as a single shared control rather than one spin per row -- simpler,
and the swath/highlight only ever apply to the selected transect anyway); Plot button (+
double-click); Delete with an UNDO STACK (a list of removed ``TransectRecord``s, popped by
:meth:`undo` -- ``MainWindow`` wires the actual ⌘Z ``QShortcut`` at the window level, the SAME
``WindowShortcut``-context idiom ``c``/``v``/Space already use there, since a shortcut scoped to
this widget alone would only fire while it happens to hold keyboard focus).

Pure Qt -- no import from ``main_window``/``canvas``, mirroring ``topology_panel.py``'s own
decoupling: this widget knows only about the plain
:class:`~dynamix.model.project.TransectRecord` dataclass and its own signals.
:meth:`MainWindow.__init__` wires the canvas gesture in (``Canvas.transectDrawn`` ->
:meth:`add_record`) and reads :meth:`records`/calls :meth:`set_records` for persistence -- this
widget never touches ``Project`` or ``Canvas`` directly.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.model.project import TransectRecord

__all__ = ["TransectPanel"]

#: EQSelect's own default ("buffer_km, 1-5000 km" -- the pixel-space analog here is
#: 1-500 px, the design's own range) -- a fresh transect and the buffer spin both start here.
_DEFAULT_BUFFER_PX = 20.0

_ID_ROLE = QtCore.Qt.UserRole


class TransectPanel(QtWidgets.QWidget):
    #: Every add / delete / undo / visibility-toggle / buffer-edit -- the FULL current record
    #: list, in on-screen row order. ``MainWindow`` forwards this straight into
    #: ``Canvas.set_transects`` and ``Project.transects`` (the live persistence copy) on every
    #: emission -- the same "always emits, whatever it changed" convention
    #: ``GroupPalette.apply_picks`` already keeps.
    recordsChanged = QtCore.Signal(list)

    #: The row selection changed -- the now-current ``TransectRecord``, or ``None`` (nothing
    #: selected, or the list emptied). ``MainWindow``'s listener highlights the line on the
    #: canvas (thicker) and swath-selects chains within its buffer
    #: (``GroupPalette.apply_picks(op="replace")``). Also re-fired by a buffer-spin edit on the
    #: selected row (EQSelect's own "re-selects the swath on every tick") -- the swath is always a
    #: function of "the selected record, as it currently stands", not a one-time snapshot taken at
    #: selection time.
    selectionChanged = QtCore.Signal(object)

    #: The Plot button, or a double-click on a row -- the ``TransectRecord`` to sample and show.
    plotRequested = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._list = QtWidgets.QListWidget()
        self._list.currentRowChanged.connect(self._on_current_row_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._list)

        buffer_row = QtWidgets.QHBoxLayout()
        buffer_row.addWidget(QtWidgets.QLabel("Buffer (px)"))
        self._buffer_spin = QtWidgets.QSpinBox()
        self._buffer_spin.setRange(1, 500)
        self._buffer_spin.setValue(int(_DEFAULT_BUFFER_PX))
        self._buffer_spin.valueChanged.connect(self._on_buffer_changed)
        buffer_row.addWidget(self._buffer_spin)
        layout.addLayout(buffer_row)

        button_row = QtWidgets.QHBoxLayout()
        self._plot_button = QtWidgets.QPushButton("Plot")
        self._plot_button.clicked.connect(self._on_plot_clicked)
        button_row.addWidget(self._plot_button)
        self._delete_button = QtWidgets.QPushButton("Delete")
        self._delete_button.clicked.connect(self.delete_selected)
        button_row.addWidget(self._delete_button)
        layout.addLayout(button_row)

        # transect_id -> record / list item. A plain dict already preserves insertion order,
        # and nothing here ever reorders except append-only add/restore, so iterating it in
        # insertion order and iterating the QListWidget's own rows agree -- but :meth:`records`
        # walks the LIST WIDGET (the on-screen order), not this dict, since that is the order a
        # future re-ordering gesture (out of this task's scope) would actually change.
        self._records: dict[int, TransectRecord] = {}
        self._rows: dict[int, QtWidgets.QListWidgetItem] = {}
        self._next_id = 1
        self._deleted_stack: list[TransectRecord] = []

    # -- gestures -> records -------------------------------------------------------------------

    def add_record(self, a, b) -> TransectRecord:
        """A finished canvas gesture (``Canvas.transectDrawn``) -- append a new row, select it
        (which pushes its buffer into the spin and fires :attr:`selectionChanged`, swath-selecting
        it immediately), and emit :attr:`recordsChanged`."""
        record = TransectRecord(transect_id=self._next_id, a=(float(a[0]), float(a[1])),
                                b=(float(b[0]), float(b[1])), visible=True,
                                buffer_px=_DEFAULT_BUFFER_PX)
        self._next_id += 1
        self._insert_row(record)
        self._list.setCurrentRow(self._list.count() - 1)
        self.recordsChanged.emit(self.records())
        return record

    def delete_selected(self) -> None:
        """Remove the current row, pushing it onto the undo stack. A no-op with nothing selected
        (matches ``TopologyPanel``'s own ``_on_unlink_clicked`` guard).

        ``takeItem`` FIRST, ``_records``/``_rows`` cleanup AFTER -- not the other order. Removing
        the current row makes Qt reselect an adjacent one (or none) SYNCHRONOUSLY, inside the
        ``takeItem`` call itself, which re-enters this widget's own ``currentRowChanged`` ->
        :attr:`selectionChanged` -> a listener that may call :meth:`records` right back in (e.g.
        ``MainWindow._on_transect_selection_changed``'s own ``Canvas.set_transects`` push).
        :meth:`records` walks the LIST WIDGET's current row count; deleting the dict entries first
        would leave that mid-``takeItem`` callback looking up an id `records()` still iterates
        over but `_records` no longer has -- a ``KeyError`` from inside someone else's signal
        handler, caught and reproduced by ``tests/test_shell_window.py::
        test_deleting_a_transect_removes_its_line_from_the_canvas``.
        """
        row = self._list.currentRow()
        record = self._record_at_row(row)
        if record is None:
            return
        self._deleted_stack.append(record)
        self._list.takeItem(row)          # Qt reselects an adjacent row on its own, if any exist
        del self._records[record.transect_id]
        del self._rows[record.transect_id]
        self.recordsChanged.emit(self.records())

    def undo(self) -> None:
        """Pop the most recently deleted record and restore it -- appended at the END of the
        list (simplest honest placement; re-inserting at its original position is unneeded
        complexity this task does not need to solve) with its ORIGINAL ``transect_id`` intact (see
        the module docstring: the monotonic counter guarantees nothing else could have taken it
        meanwhile). A no-op with an empty stack ("Nothing to undo", EQSelect's own silent-by-design
        refusal shape -- there is nothing here for a caller to react to either way,
        so no signal is the honest response)."""
        if not self._deleted_stack:
            return
        record = self._deleted_stack.pop()
        self._insert_row(record)
        self._list.setCurrentRow(self._list.count() - 1)
        self.recordsChanged.emit(self.records())

    # -- reads ---------------------------------------------------------------------------------

    def records(self) -> list[TransectRecord]:
        """Every record, in ON-SCREEN row order -- the exact shape ``Project.transects`` and
        ``Canvas.set_transects`` both want."""
        out = []
        for row in range(self._list.count()):
            transect_id = self._list.item(row).data(_ID_ROLE)
            out.append(self._records[transect_id])
        return out

    def selected_id(self) -> int | None:
        record = self._record_at_row(self._list.currentRow())
        return None if record is None else record.transect_id

    # -- persistence (silent -- mirrors right_panel.set_style_values / topology_panel.set_rows) -

    def set_records(self, records) -> None:
        """Replace the whole list from a project reopen. Silent -- emits nothing, matching every
        other "sync FROM the model" setter in this shell (``RightPanel.set_style_values``,
        ``TopologyPanel.set_rows``); the caller (``MainWindow._open_project_path``) pushes the
        result into ``Canvas.set_transects`` itself right after."""
        self._list.blockSignals(True)
        self._list.clear()
        self._records = {}
        self._rows = {}
        self._deleted_stack = []
        max_id = 0
        for record in records:
            self._insert_row(record)
            max_id = max(max_id, record.transect_id)
        self._next_id = max_id + 1
        self._list.blockSignals(False)

    # -- internals -------------------------------------------------------------------------------

    def _insert_row(self, record: TransectRecord) -> None:
        item = QtWidgets.QListWidgetItem(f"T{record.transect_id}")
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked if record.visible else QtCore.Qt.Unchecked)
        item.setData(_ID_ROLE, record.transect_id)
        self._records[record.transect_id] = record
        self._rows[record.transect_id] = item
        self._list.addItem(item)          # after every other setup -- avoids a spurious itemChanged

    def _record_at_row(self, row: int) -> TransectRecord | None:
        if row < 0 or row >= self._list.count():
            return None
        return self._records.get(self._list.item(row).data(_ID_ROLE))

    def _on_current_row_changed(self, row: int) -> None:
        record = self._record_at_row(row)
        if record is not None:
            self._buffer_spin.blockSignals(True)
            self._buffer_spin.setValue(int(record.buffer_px))
            self._buffer_spin.blockSignals(False)
        self.selectionChanged.emit(record)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        """The per-row visibility checkbox (the only thing that can change an already-inserted
        item -- rows are never text-editable)."""
        record = self._records.get(item.data(_ID_ROLE))
        if record is None:
            return
        record.visible = item.checkState() == QtCore.Qt.Checked
        self.recordsChanged.emit(self.records())

    def _on_buffer_changed(self, value: int) -> None:
        record = self._record_at_row(self._list.currentRow())
        if record is None:
            return
        record.buffer_px = float(value)
        self.recordsChanged.emit(self.records())
        self.selectionChanged.emit(record)          # re-select the swath at the new buffer

    def _on_plot_clicked(self) -> None:
        record = self._record_at_row(self._list.currentRow())
        if record is not None:
            self.plotRequested.emit(record)

    def _on_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        record = self._records.get(item.data(_ID_ROLE))
        if record is not None:
            self.plotRequested.emit(record)
