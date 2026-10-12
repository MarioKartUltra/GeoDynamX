# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Layers panel: the layer stack as a tree -- an eye per row and group, an opacity, drag
to reorder and regroup, a note when the current view cannot draw a row. It edits the stack
and says so (``stackChanged``); the views redraw from the stack."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.layer_stack import Group

_ID = QtCore.Qt.UserRole


class LayersPanel(QtWidgets.QWidget):
    stackChanged = QtCore.Signal()
    rowSelected = QtCore.Signal(str)
    zoomRequested = QtCore.Signal(str)

    def __init__(self, stack, label_of, parent=None):
        super().__init__(parent)
        self._stack, self._label_of = stack, label_of
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._reasons: dict[str, str] = {}
        self._building = False
        self._drop_pending = False
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["layer", "opacity"])
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._on_current)
        self.tree.customContextMenuRequested.connect(self._menu)
        # A QTreeWidget drag takes the item out and inserts it again: rowsInserted, never
        # rowsMoved.
        self.tree.model().rowsInserted.connect(self._on_dropped)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)
        self.refresh()

    # -- build ------------------------------------------------------------------------------
    def set_stack(self, stack) -> None:
        """Show another stack (a project opened in its place)."""
        self._stack = stack
        self.refresh()

    def refresh(self) -> None:
        self._building = True
        self.tree.clear()
        self._items.clear()
        self._spins.clear()
        for node in self._stack.root.children:
            self._add(None, node)
        self.tree.expandAll()
        self._building = False
        self._apply_reasons()

    def _add(self, parent_item, node) -> None:
        text = node.name if isinstance(node, Group) else self._label_of(node.id)
        item = QtWidgets.QTreeWidgetItem([text])
        item.setData(0, _ID, node.id)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsDragEnabled
                      | (QtCore.Qt.ItemIsDropEnabled if isinstance(node, Group)
                         else QtCore.Qt.NoItemFlags))
        if not isinstance(node, Group):
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDropEnabled)
        item.setCheckState(0, QtCore.Qt.Checked if node.visible else QtCore.Qt.Unchecked)
        (self.tree.addTopLevelItem(item) if parent_item is None else parent_item.addChild(item))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.1)
        spin.setValue(node.opacity)
        spin.valueChanged.connect(lambda v, nid=node.id: self._set_opacity(nid, v))
        self.tree.setItemWidget(item, 1, spin)
        self._items[node.id], self._spins[node.id] = item, spin
        if isinstance(node, Group):
            for child in node.children:
                self._add(item, child)

    # -- accessors used by tests and MainWindow -------------------------------------------
    def item_for(self, node_id):
        return self._items[node_id]

    def opacity_spin(self, node_id):
        return self._spins[node_id]

    def select_row(self, node_id) -> None:
        item = self._items.get(node_id)
        if item is not None:
            self._building = True
            self.tree.setCurrentItem(item)
            self._building = False

    def highlight_layer(self, layer_id: int) -> None:
        for nid, item in self._items.items():
            font = item.font(0)
            font.setBold(nid.endswith(f":{layer_id}"))
            item.setFont(0, font)

    def relabel(self) -> None:
        """Rows name what they show (a product row its layer's Show): refresh the names in
        place, the tree and its widgets untouched."""
        self._building = True
        for nid, item in self._items.items():
            if not isinstance(self._stack.node(nid), Group):
                item.setText(0, self._label_of(nid))
        self._building = False

    def set_reasons(self, reasons: dict) -> None:
        self._reasons = dict(reasons)
        self._apply_reasons()

    def _apply_reasons(self) -> None:
        for nid, item in self._items.items():
            reason = self._reasons.get(nid)
            color = self.palette().color(QtGui.QPalette.Text)
            if reason:
                color.setAlpha(110)
            item.setForeground(0, color)
            item.setToolTip(0, reason or "")

    # -- edits ------------------------------------------------------------------------------
    def _on_item_changed(self, item, column) -> None:
        if self._building or column != 0:
            return
        node = self._stack.node(item.data(0, _ID))
        if node is None:
            return
        visible = item.checkState(0) == QtCore.Qt.Checked
        if node.visible != visible:
            node.visible = visible
            self.stackChanged.emit()

    def _set_opacity(self, node_id, value) -> None:
        self._stack.node(node_id).opacity = float(value)
        self.stackChanged.emit()

    def new_group_above(self, node_id, name):
        parent = self._stack.parent_of(node_id)
        index = parent.children.index(self._stack.node(node_id))
        group = self._stack.new_group(name, parent.id, index)
        self._stack.move(node_id, group.id, 0)
        self.refresh()
        self.stackChanged.emit()
        return group

    def ungroup(self, group_id) -> None:
        self._stack.ungroup(group_id)
        self.refresh()
        self.stackChanged.emit()

    def move(self, node_id, parent_id, index) -> None:
        self._stack.move(node_id, parent_id, index)
        self.refresh()
        self.stackChanged.emit()

    def _on_dropped(self, *_args) -> None:
        """A drag in the tree: once the drop has finished, read the new tree back into the
        stack, node by node."""
        if self._building or self._drop_pending:
            return
        self._drop_pending = True
        QtCore.QTimer.singleShot(0, self._read_back)

    def _read_back(self) -> None:
        self._drop_pending = False

        def walk(parent_item, parent_id):
            count = (self.tree.topLevelItemCount() if parent_item is None
                     else parent_item.childCount())
            for i in range(count):
                item = (self.tree.topLevelItem(i) if parent_item is None
                        else parent_item.child(i))
                nid = item.data(0, _ID)
                if self._stack.node(nid) is None:
                    continue
                parent = self._stack.parent_of(nid)
                if parent.id != parent_id or parent.children.index(self._stack.node(nid)) != i:
                    self._stack.move(nid, parent_id, i)
                walk(item, nid)

        walk(None, "root")
        self.refresh()
        self.stackChanged.emit()

    def _on_current(self, current, _previous) -> None:
        if current is not None and not self._building:
            self.rowSelected.emit(current.data(0, _ID))

    def _menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        nid = item.data(0, _ID)
        menu = QtWidgets.QMenu(self)
        menu.addAction("New group").triggered.connect(lambda: self.new_group_above(nid, "Group"))
        if isinstance(self._stack.node(nid), Group):
            menu.addAction("Rename…").triggered.connect(lambda: self._rename(nid))
            menu.addAction("Ungroup").triggered.connect(lambda: self.ungroup(nid))
        menu.addAction("Show only this").triggered.connect(lambda: self._solo(nid))
        menu.addAction("Zoom to").triggered.connect(lambda: self.zoomRequested.emit(nid))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _rename(self, group_id) -> None:
        group = self._stack.node(group_id)
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename group", "Name", text=group.name)
        if ok and name:
            group.name = name
            self.refresh()
            self.stackChanged.emit()

    def _solo(self, node_id) -> None:
        keep = {node_id}
        parent = self._stack.parent_of(node_id)
        while parent is not None and parent.id != "root":
            keep.add(parent.id)
            parent = self._stack.parent_of(parent.id)
        for row, _v, _o in self._stack.drawn(include_hidden=True):
            row.visible = row.id in keep
        self.refresh()
        self.stackChanged.emit()
