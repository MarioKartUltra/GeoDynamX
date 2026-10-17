# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""RackEditor: build, reorder, tune and validate a rack without a layer.

The family tree on the left is a :class:`~dynamix.shell.browser.DeviceBrowser`; a device row
dragged from it lands in the step list on the right, which reorders by drag. The selected step's
knobs are a workflow-zone :class:`~dynamix.shell.workflow_zone.DeviceBox`. The validation line
runs ``Chain.validate`` over every step (bypassed ones included, as the zone does) and the
contracts' ``step_states`` over the enabled steps from the chosen input. Built-in racks open
read-only; "Duplicate to edit" turns the open rack into an unsaved user copy.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.contracts import FIELD, POINTS, STACK, step_states
from dynamix.devices.precision import placement_params
from dynamix.model.device import defaults_for, get_device, validate_params
from dynamix.model.families import display_name, label_of
from dynamix.model.racks import Rack, RackStep, delete_rack, load_racks, save_rack
from dynamix.shell.browser import DEVICE_MIME, DeviceBrowser
from dynamix.shell.workflow_zone import DeviceBox

#: The inputs a rack is validated from, worded as a layer's input is.
_INPUTS = (("single-band field", FIELD), ("band stack", STACK), ("point set", POINTS))

#: A step row's key into ``RackEditor._steps``.
_STEP_ROLE = QtCore.Qt.UserRole

_BYPASS_TIP = "checked: bypassed"


class _StepList(QtWidgets.QListWidget):
    """The rack's steps. Reorders by internal drag (the model's ``rowsMoved`` tells the editor);
    a device row dragged from the tree is added where it lands."""

    deviceDropped = QtCore.Signal(str, int)          # (device id, row; -1 appends)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(DEVICE_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(DEVICE_MIME):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        md = event.mimeData()
        if md.hasFormat(DEVICE_MIME):
            row = self.indexAt(event.position().toPoint()).row()
            self.deviceDropped.emit(bytes(md.data(DEVICE_MIME)).decode("utf-8"), row)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class RackEditor(QtWidgets.QDialog):
    """One rack at a time, picked in the Rack box ("New rack" or any listed rack).

    ``entries`` are the browser's ``RackEntry`` rows (built-in and user); entries with a problem
    are not offered. Saving and deleting write ``directory`` and emit :attr:`racksChanged`.
    """

    racksChanged = QtCore.Signal()

    def __init__(self, directory: Path, entries, name: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rack Editor")
        self._directory = Path(directory)
        self._builtins = {e.rack.name: e.rack for e in entries
                          if e.rack is not None and e.rack.builtin and not e.problem}
        self._steps: dict[int, dict] = {}
        self._next_id = 0
        self._original: str | None = None
        self._read_only = False
        self._created = ""
        self.knob_box: DeviceBox | None = None
        self._knob_sid: int | None = None

        self.rack_combo = QtWidgets.QComboBox()
        self.rack_combo.currentIndexChanged.connect(self._on_rack_chosen)
        self.name_edit = QtWidgets.QLineEdit()
        self.description_edit = QtWidgets.QLineEdit()
        self.browser = DeviceBrowser()
        self.steps_list = _StepList()
        self.steps_list.deviceDropped.connect(self.add_step)
        self.steps_list.model().rowsMoved.connect(self._revalidate)
        self.steps_list.currentRowChanged.connect(self._show_knobs)
        self.steps_list.itemChanged.connect(self._on_item_changed)
        self._knob_area = QtWidgets.QScrollArea()
        self._knob_area.setWidgetResizable(True)
        self.input_combo = QtWidgets.QComboBox()
        for text, contract in _INPUTS:
            self.input_combo.addItem(text, contract)
        self.input_combo.currentIndexChanged.connect(self._revalidate)
        self.validation_label = QtWidgets.QLabel("")
        self.validation_label.setWordWrap(True)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setProperty("muted", "true")
        self.duplicate_button = QtWidgets.QPushButton("Duplicate to edit")
        self.duplicate_button.clicked.connect(self.duplicate)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete)
        self.save_as_button = QtWidgets.QPushButton("Save as…")
        self.save_as_button.clicked.connect(self.save_as)
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)

        form = QtWidgets.QFormLayout()
        form.addRow("Rack", self.rack_combo)
        form.addRow("Name", self.name_edit)
        form.addRow("Description", self.description_edit)
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(self.steps_list)
        right.addWidget(self._knob_area)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.browser)
        split.addWidget(right)
        check = QtWidgets.QHBoxLayout()
        check.addWidget(QtWidgets.QLabel("Validate from a"))
        check.addWidget(self.input_combo)
        check.addStretch(1)
        buttons = QtWidgets.QHBoxLayout()
        for button in (self.duplicate_button, self.delete_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        for button in (self.save_as_button, self.save_button, close_button):
            buttons.addWidget(button)
        outer = QtWidgets.QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(split, 1)
        outer.addLayout(check)
        outer.addWidget(self.validation_label)
        outer.addWidget(self.status_label)
        outer.addLayout(buttons)

        self._fill_combo()
        self.load(name)

    # -- which rack ---------------------------------------------------------------------------
    def _user_racks(self) -> dict:
        return {e.rack.name: e for e in load_racks(self._directory)
                if e.rack is not None and not e.problem}

    def _fill_combo(self) -> None:
        self.rack_combo.blockSignals(True)
        self.rack_combo.clear()
        self.rack_combo.addItem("New rack", None)
        for name in list(self._builtins) + list(self._user_racks()):
            self.rack_combo.addItem(name, name)
        self.rack_combo.blockSignals(False)

    def _on_rack_chosen(self, index: int) -> None:
        self.load(self.rack_combo.itemData(index))

    def load(self, name: str | None) -> None:
        """Show the rack ``name`` (a built-in read-only), or an empty new rack for ``None``."""
        users = self._user_racks()
        rack = self._builtins.get(name) or (users[name].rack if name in users else None)
        self._read_only = rack is not None and rack.builtin
        self._original = rack.name if rack is not None and not rack.builtin else None
        self._created = rack.created if self._original is not None else ""
        self.name_edit.setText(rack.name if rack is not None else "")
        self.description_edit.setText(rack.description if rack is not None else "")
        self._steps = {}
        self.steps_list.blockSignals(True)
        self.steps_list.clear()
        self.steps_list.blockSignals(False)
        for step in (rack.steps if rack is not None else ()):
            self._append({"device": step.device, "params": dict(step.params),
                          "bypassed": step.bypassed}, -1)
        self.rack_combo.blockSignals(True)
        self.rack_combo.setCurrentIndex(max(self.rack_combo.findData(name), 0)
                                        if name is not None else 0)
        self.rack_combo.blockSignals(False)
        self.status_label.setText("built-in rack: Duplicate to edit" if self._read_only else "")
        self._apply_read_only()
        if self.steps_list.count():
            self.steps_list.setCurrentRow(0)
        else:
            self._show_knobs(-1)
        self._revalidate()

    # -- steps ----------------------------------------------------------------------------------
    def descriptors(self) -> list[dict]:
        """The steps in list order: ``{"device", "params", "bypassed"}``."""
        return [self._steps[self.steps_list.item(r).data(_STEP_ROLE)]
                for r in range(self.steps_list.count())]

    def _append(self, descriptor: dict, row: int) -> int:
        sid = self._next_id
        self._next_id += 1
        self._steps[sid] = descriptor
        item = QtWidgets.QListWidgetItem(display_name(get_device(descriptor["device"])))
        item.setData(_STEP_ROLE, sid)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked if descriptor["bypassed"] else QtCore.Qt.Unchecked)
        item.setToolTip(_BYPASS_TIP)
        self.steps_list.blockSignals(True)
        if 0 <= row < self.steps_list.count():
            self.steps_list.insertItem(row, item)
        else:
            self.steps_list.addItem(item)
        self.steps_list.blockSignals(False)
        return self.steps_list.row(item)

    def add_step(self, device_name: str, row: int = -1) -> None:
        """A device dropped from the tree (or added by a caller): its defaults, enabled."""
        if self._read_only:
            return
        device = get_device(device_name)
        at = self._append({"device": device_name, "params": placement_params(device),
                           "bypassed": False}, row)
        self.steps_list.setCurrentRow(at)
        self._revalidate()

    def _row_of(self, sid: int) -> int:
        return next(r for r in range(self.steps_list.count())
                    if self.steps_list.item(r).data(_STEP_ROLE) == sid)

    def _remove_step(self, sid: int) -> None:
        if self._read_only:
            return
        self.steps_list.takeItem(self._row_of(sid))
        del self._steps[sid]
        self._revalidate()

    def _on_item_changed(self, item) -> None:
        sid = item.data(_STEP_ROLE)
        self._set_bypassed(sid, item.checkState() == QtCore.Qt.Checked)

    def _set_bypassed(self, sid: int, bypassed: bool) -> None:
        self._steps[sid]["bypassed"] = bypassed
        item = self.steps_list.item(self._row_of(sid))
        self.steps_list.blockSignals(True)
        item.setCheckState(QtCore.Qt.Checked if bypassed else QtCore.Qt.Unchecked)
        self.steps_list.blockSignals(False)
        if self.knob_box is not None and self._knob_sid == sid:
            self.knob_box.bypass_button.blockSignals(True)
            self.knob_box.bypass_button.setChecked(bypassed)
            self.knob_box.bypass_button.blockSignals(False)
            self.knob_box.set_bypassed(bypassed)
        self._revalidate()

    def _on_param_changed(self, sid: int, name: str, value) -> None:
        self._steps[sid]["params"][name] = value
        self._revalidate()

    def _show_knobs(self, row: int) -> None:
        old = self._knob_area.takeWidget()
        if old is not None:
            old.deleteLater()
        self.knob_box = None
        self._knob_sid = None
        if row < 0:
            return
        sid = self.steps_list.item(row).data(_STEP_ROLE)
        d = self._steps[sid]
        device = get_device(d["device"])
        box = DeviceBox(row, device, {**defaults_for(device), **d["params"]},
                        bypassed=d["bypassed"])
        box.paramChanged.connect(lambda name, value, sid=sid: self._on_param_changed(sid, name,
                                                                                      value))
        box.bypassToggled.connect(lambda checked, sid=sid: self._set_bypassed(sid, checked))
        box.removeRequested.connect(lambda sid=sid: self._remove_step(sid))
        box.setEnabled(not self._read_only)
        self._knob_area.setWidget(box)
        self.knob_box = box
        self._knob_sid = sid

    # -- validation -----------------------------------------------------------------------------
    def _revalidate(self, *_args) -> None:
        descriptors = self.descriptors()
        states: dict[int, object] = {}
        try:
            Chain(tuple(DeviceRef(d["device"], d["params"]) for d in descriptors)).validate()
        except (KeyError, ValueError) as exc:
            ok, text = False, str(exc)
        else:
            enabled = [i for i, d in enumerate(descriptors) if not d["bypassed"]]
            refs = [DeviceRef(descriptors[i]["device"], descriptors[i]["params"])
                    for i in enabled]
            states = dict(zip(enabled, step_states(refs, self.input_combo.currentData())))
            lines = [f"{label_of(get_device(descriptors[i]['device']))}: {s.reason}"
                     for i, s in states.items() if not s.applies]
            ok = bool(descriptors)
            text = ("\n".join(lines) if lines
                    else f"every step applies on a {self.input_combo.currentText()}"
                    if descriptors else "drag tools from the tree to build the rack")
        self.validation_label.setText(text)
        muted = self.palette().color(QtGui.QPalette.Disabled, QtGui.QPalette.Text)
        self.steps_list.blockSignals(True)
        for row in range(self.steps_list.count()):
            item = self.steps_list.item(row)
            state = states.get(row)
            if state is not None and not state.applies:
                item.setForeground(muted)
                item.setToolTip(state.reason)
            else:
                item.setData(QtCore.Qt.ForegroundRole, None)
                item.setToolTip(_BYPASS_TIP)
        self.steps_list.blockSignals(False)
        user = not self._read_only
        self.save_button.setEnabled(user and ok)
        self.save_as_button.setEnabled(user and ok)
        self.delete_button.setEnabled(user and self._original is not None)
        self.duplicate_button.setEnabled(self._read_only)

    def _apply_read_only(self) -> None:
        ro = self._read_only
        self.name_edit.setReadOnly(ro)
        self.description_edit.setReadOnly(ro)
        self.steps_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop if ro
                                        else QtWidgets.QAbstractItemView.InternalMove)
        self.steps_list.blockSignals(True)
        for row in range(self.steps_list.count()):
            item = self.steps_list.item(row)
            flags = item.flags() & ~QtCore.Qt.ItemIsUserCheckable
            item.setFlags(flags if ro else flags | QtCore.Qt.ItemIsUserCheckable)
        self.steps_list.blockSignals(False)
        if self.knob_box is not None:
            self.knob_box.setEnabled(not ro)

    # -- the rack as a whole --------------------------------------------------------------------
    def current_rack(self) -> Rack:
        """What Save writes: every step with all its params filled, and its bypass state."""
        return Rack(name=self.name_edit.text().strip(),
                    steps=tuple(RackStep(d["device"],
                                         validate_params(get_device(d["device"]), d["params"]),
                                         bool(d["bypassed"]))
                                for d in self.descriptors()),
                    description=self.description_edit.text().strip(), created=self._created)

    def duplicate(self) -> None:
        """A built-in rack becomes an unsaved user copy named "<name> copy"."""
        if not self._read_only:
            return
        self._read_only = False
        self._original = None
        self._created = ""
        self.name_edit.setText(f"{self.name_edit.text()} copy")
        self.status_label.setText("")
        self._apply_read_only()
        self._revalidate()

    def _can_take(self, name: str) -> bool:
        if not name:
            self.status_label.setText("name the rack to save it")
            return False
        if name in self._builtins:
            self.status_label.setText(f"{name} is a built-in rack; choose another name")
            return False
        if name != self._original and name in self._user_racks():
            reply = QtWidgets.QMessageBox.question(
                self, "Replace rack", f"A rack named {name} exists. Replace it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            return reply == QtWidgets.QMessageBox.Yes
        return True

    def save(self) -> bool:
        """Write the rack under the Name field. A saved rack whose name changed is renamed."""
        name = self.name_edit.text().strip()
        if self._read_only or not self._can_take(name):
            return False
        try:
            save_rack(self._directory, self.current_rack())
            if self._original is not None and self._original != name:
                delete_rack(self._directory, self._original)
        except (OSError, KeyError) as exc:
            self.status_label.setText(f"not saved: {exc}")
            return False
        self._original = name
        self._created = self._user_racks()[name].rack.created
        self._fill_combo()
        self.rack_combo.blockSignals(True)
        self.rack_combo.setCurrentIndex(self.rack_combo.findData(name))
        self.rack_combo.blockSignals(False)
        self.status_label.setText(f"saved {name}")
        self._revalidate()
        self.racksChanged.emit()
        return True

    def save_as(self) -> bool:
        """Save a copy under a new name; the rack it was opened from stays."""
        name, ok = QtWidgets.QInputDialog.getText(self, "Save rack as", "Rack name:",
                                                  QtWidgets.QLineEdit.Normal,
                                                  self.name_edit.text())
        if not ok or not name.strip():
            return False
        previous, previous_created = self._original, self._created
        self._original, self._created = None, ""
        self.name_edit.setText(name.strip())
        if not self.save():
            self._original, self._created = previous, previous_created
            return False
        return True

    def delete(self) -> None:
        """Delete the saved user rack this editor opened, after a confirmation."""
        if self._original is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Delete rack", f"Delete the rack {self._original}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            delete_rack(self._directory, self._original)
        except (OSError, KeyError) as exc:
            self.status_label.setText(f"not deleted: {exc}")
            return
        self._fill_combo()
        self.load(None)
        self.racksChanged.emit()
