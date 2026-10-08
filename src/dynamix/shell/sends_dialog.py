# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""SendsDialog: the bus dialog with a role per send (value, structure, weights, mask, veil
seed), the send a weights raster weights, a shade mark for a DEM, and each candidate's grid
relation to the host (same, nested, foreign). Any grid can send: ``inpaint`` and ``fill``
read every send on its own grid."""
from __future__ import annotations

import dataclasses

from PySide6 import QtCore, QtWidgets

from dynamix.core.inpaint.engine import ROLES
from dynamix.shell.bus_dialog import BusDialog, _key

__all__ = ["SendsDialog"]

_ROLE_LABELS = {"value": "value", "structure": "structure", "weights": "weights",
                "mask": "mask", "veil_seed": "veil seed"}


@dataclasses.dataclass
class _Row:
    label: str
    key: tuple
    role: QtWidgets.QComboBox
    target: QtWidgets.QComboBox
    shade: QtWidgets.QCheckBox
    #: The binding a weights send arrived with, kept until its measurement's row exists
    #: (a send listed before the send it weights) or the user changes the combo.
    want_for: object = None


class SendsDialog(BusDialog):
    """``candidates`` as the bus dialog's; ``sends`` the step's current sends with their roles;
    ``relation_of(send) -> str`` names each candidate's relation to the host."""

    def __init__(self, candidates, sends=(), *, relation_of=None, title="Sends", parent=None):
        self._relation_of = relation_of
        self._rows: list = []
        self._roles = {_key(s): s for s in sends}

        self._filling = True        # rows built now take their binding from the send
        super().__init__(candidates, sends, title=title, parent=parent)
        self.findChild(QtWidgets.QLabel).setText(
            "Tick rasters to send. Any grid can send: the relation to the host is shown; a "
            "foreign value or structure send closes the cascade routes.")
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            key = item.data(0, QtCore.Qt.UserRole)
            if key is not None and relation_of is not None:
                try:
                    rel = relation_of(self._by_key[tuple(key)])
                except Exception as exc:                     # noqa: BLE001 -- shown, not raised
                    rel = f"unreadable: {exc}"
                item.setText(0, f"{item.text(0)} — {rel}")
            it += 1
        # The combo is the only source of a binding from here on; ``want_for`` served the
        # initial fill, when a weights send could be listed before the send it weights.
        for row in self._rows:
            row.want_for = None
        self._filling = False

    def _append(self, send: dict) -> None:
        super()._append(send)
        item = self.order.item(self.order.count() - 1)
        key = tuple(item.data(QtCore.Qt.UserRole))
        widget = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(widget)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.addWidget(QtWidgets.QLabel(send.get("label") or send["path"]), 1)
        role = QtWidgets.QComboBox()
        for r in ROLES:
            role.addItem(_ROLE_LABELS[r], r)
        current = self._roles.get(key, send)
        role.setCurrentIndex(ROLES.index(str(current.get("role", "value") or "value")))
        target = QtWidgets.QComboBox()
        shade = QtWidgets.QCheckBox("shade (DEM)")
        shade.setChecked(bool(current.get("shade", False)))
        lay.addWidget(role)
        lay.addWidget(target)
        lay.addWidget(shade)
        row = _Row(send.get("label") or send["path"], key, role, target, shade,
                   want_for=(current.get("for") or None) if getattr(self, "_filling", False) else None)
        self._rows.append(row)
        role.currentIndexChanged.connect(lambda _i, r=row: self._refresh_targets())
        target.currentIndexChanged.connect(lambda _i, r=row: setattr(r, "want_for", None))
        self._refresh_targets()
        item.setSizeHint(widget.sizeHint())
        self.order.setItemWidget(item, widget)

    def _refresh_targets(self) -> None:
        for row in self._rows:
            keep = str(row.want_for) if row.want_for else row.target.currentText()
            row.target.blockSignals(True)
            row.target.clear()
            row.target.addItem("target")
            for other in self._rows:
                if other is not row and other.role.currentData() != "weights":
                    row.target.addItem(other.label)
            if keep and row.target.findText(keep) >= 0:
                row.target.setCurrentText(keep)
            row.target.blockSignals(False)
            row.target.setEnabled(row.role.currentData() == "weights")

    def rows(self) -> list:
        order = [tuple(self.order.item(i).data(QtCore.Qt.UserRole)) for i in range(self.order.count())]
        by_key = {r.key: r for r in self._rows}
        return [by_key[k] for k in order if k in by_key]

    def _on_item_changed(self, item, _column) -> None:
        super()._on_item_changed(item, _column)
        keys = {tuple(self.order.item(i).data(QtCore.Qt.UserRole)) for i in range(self.order.count())}
        self._rows = [r for r in self._rows if r.key in keys]
        self._refresh_targets()

    def sends(self) -> list:
        out = []
        for row in self.rows():
            send = dict(self._by_key[row.key])
            send["role"] = row.role.currentData()
            send["shade"] = bool(row.shade.isChecked())
            if send["role"] == "weights":
                send["for"] = str(row.want_for) if row.want_for else row.target.currentText()
            else:
                send.pop("for", None)
            out.append(send)
        return out
