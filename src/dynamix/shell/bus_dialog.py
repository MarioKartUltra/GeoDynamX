# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Routing bands into a bus: pick raw bands from any dataset on the bus's grid, order them.

Left, every loaded dataset with a checkbox per band -- a dataset on another grid is listed
greyed out with the reason (the same-grid law). Right, the bus's send ORDER, which is the
band axis a tool receives: ticking a band appends it, unticking removes it, and dragging
reorders.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["BusDialog"]


def _key(send: dict) -> tuple:
    if "layer" in send:                   # a live layer send: its layer is its identity
        return ("layer", int(send["layer"]), None)
    return (str(send.get("path")), send.get("subdataset"), send.get("band"))


class BusDialog(QtWidgets.QDialog):
    """``candidates``: ``[{"label", "sends": [send...], "reason": None | str}]``;
    ``sends``: the bus's current send list (pre-ticked, in order). OK -> :meth:`sends`."""

    def __init__(self, candidates, sends=(), *, title="Route bands to a bus", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._by_key: dict = {}
        self._loading = True
        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(QtWidgets.QLabel(
            "Tick raw bands to send. Only datasets on this bus's grid can send; a processed "
            "plane enters by forking a derivative first."))
        row = QtWidgets.QHBoxLayout()
        outer.addLayout(row)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Datasets and bands"])
        row.addWidget(self.tree, 3)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Bus order — drag to reorder"))
        self.order = QtWidgets.QListWidget()
        self.order.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.order.setDefaultDropAction(QtCore.Qt.MoveAction)
        right.addWidget(self.order)
        row.addLayout(right, 2)
        current = {_key(s) for s in sends}
        for cand in candidates:
            top = QtWidgets.QTreeWidgetItem(self.tree, [cand["label"]])
            reason = cand.get("reason")
            if reason:
                top.setText(0, f"{cand['label']} — {reason}")
                top.setToolTip(0, reason)
                top.setFlags(top.flags() & ~QtCore.Qt.ItemIsEnabled)
            for send in cand["sends"]:
                self._by_key[_key(send)] = send
                child = QtWidgets.QTreeWidgetItem(top, [send.get("label") or send["path"]])
                child.setData(0, QtCore.Qt.UserRole, _key(send))
                if reason:
                    child.setFlags(child.flags() & ~QtCore.Qt.ItemIsUserCheckable)
                else:
                    child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                    child.setCheckState(0, QtCore.Qt.Checked if _key(send) in current
                                        else QtCore.Qt.Unchecked)
            top.setExpanded(not reason)
        for send in sends:                      # keep the bus's own order, not tree order
            self._by_key.setdefault(_key(send), send)
            self._append(send)
        self.tree.itemChanged.connect(self._on_item_changed)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._loading = False

    def _append(self, send: dict) -> None:
        item = QtWidgets.QListWidgetItem(send.get("label") or send["path"])
        item.setData(QtCore.Qt.UserRole, _key(send))
        self.order.addItem(item)

    def _on_item_changed(self, item, _column) -> None:
        if self._loading:
            return
        key = item.data(0, QtCore.Qt.UserRole)
        if key is None:
            return
        key = tuple(key)
        rows = [i for i in range(self.order.count())
                if tuple(self.order.item(i).data(QtCore.Qt.UserRole)) == key]
        if item.checkState(0) == QtCore.Qt.Checked:
            if not rows:
                self._append(self._by_key[key])
        else:
            for i in reversed(rows):
                self.order.takeItem(i)

    def sends(self) -> list:
        """The routed sends, in the bus's (possibly reordered) order."""
        return [dict(self._by_key[tuple(self.order.item(i).data(QtCore.Qt.UserRole))])
                for i in range(self.order.count())]
