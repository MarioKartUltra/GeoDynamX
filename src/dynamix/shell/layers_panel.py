# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Layers panel: the layer stack as a tree -- an eye and a name per row and group, one
opacity slider and up/down buttons for the selected row, drag to reorder and regroup, a note
when the current view cannot draw a row, and a handle under the tree to make the panel taller.
It edits the stack and says so (``stackChanged``); the views redraw from the stack."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.layer_stack import Group

_ID = QtCore.Qt.UserRole
DEFAULT_HEIGHT = 240
MIN_HEIGHT = 110


class _Grip(QtWidgets.QFrame):
    """A handle under the tree: drag it to make the tree taller or shorter."""

    resized = QtCore.Signal(int)          # the height, on release

    def __init__(self, target, parent=None):
        super().__init__(parent)
        self._target = target
        self._start = None
        self.setFixedHeight(7)
        self.setCursor(QtCore.Qt.SizeVerCursor)
        self.setToolTip("Drag to resize the layer list")
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)

    def mousePressEvent(self, event) -> None:
        self._start = (event.globalPosition().y(), self._target.height())

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            y0, h0 = self._start
            self._target.setFixedHeight(max(MIN_HEIGHT, int(h0 + event.globalPosition().y() - y0)))

    def mouseReleaseEvent(self, event) -> None:
        if self._start is not None:
            self._start = None
            self.resized.emit(self._target.height())


class LayersPanel(QtWidgets.QWidget):
    stackChanged = QtCore.Signal()
    opacityChanged = QtCore.Signal()      # a slider tick: the views re-fade, nothing else moved
    rowSelected = QtCore.Signal(str)
    zoomRequested = QtCore.Signal(str)
    heightChanged = QtCore.Signal(int)

    def __init__(self, stack, label_of, parent=None, height: int = DEFAULT_HEIGHT):
        super().__init__(parent)
        self._stack, self._label_of = stack, label_of
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._reasons: dict[str, str] = {}
        self._building = False
        self._drop_pending = False
        self._eye_clicked = False

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setToolTip("Opacity of the selected layer or group")
        self.opacity_label = QtWidgets.QLabel("100%")
        self.opacity_label.setMinimumWidth(36)
        self.up_button = QtWidgets.QToolButton()
        self.up_button.setArrowType(QtCore.Qt.UpArrow)
        self.up_button.setToolTip("Move up (draws over the rows below it)")
        self.down_button = QtWidgets.QToolButton()
        self.down_button.setArrowType(QtCore.Qt.DownArrow)
        self.down_button.setToolTip("Move down")
        self.group_button = QtWidgets.QToolButton()
        self.group_button.setText("Group")
        self.group_button.setToolTip("Put the selected row in a new group")
        bar = QtWidgets.QHBoxLayout()             # the slider has its row: a narrow panel
        bar.setContentsMargins(0, 0, 0, 0)        # keeps it long enough to drag
        bar.addWidget(QtWidgets.QLabel("Opacity"))
        bar.addWidget(self.opacity_slider, 1)
        bar.addWidget(self.opacity_label)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.up_button)
        buttons.addWidget(self.down_button)
        buttons.addWidget(self.group_button)
        buttons.addStretch(1)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setUniformRowHeights(True)
        self.tree.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.setFixedHeight(max(MIN_HEIGHT, int(height)))
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.currentItemChanged.connect(lambda *_a: self._sync_controls())
        self.tree.customContextMenuRequested.connect(self._menu)
        # A QTreeWidget drag takes the item out and inserts it again: rowsInserted, never
        # rowsMoved.
        self.tree.model().rowsInserted.connect(self._on_dropped)
        self.grip = _Grip(self.tree)
        self.grip.resized.connect(self.heightChanged.emit)

        self.opacity_slider.valueChanged.connect(self._on_slider)
        self.opacity_slider.sliderReleased.connect(self.stackChanged.emit)
        self.up_button.clicked.connect(lambda: self._step(-1))
        self.down_button.clicked.connect(lambda: self._step(+1))
        self.group_button.clicked.connect(self._group_current)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addLayout(bar)
        layout.addLayout(buttons)
        layout.addWidget(self.tree)
        layout.addWidget(self.grip)
        self.refresh()

    # -- build ------------------------------------------------------------------------------
    def set_stack(self, stack) -> None:
        """Show another stack (a project opened in its place)."""
        self._stack = stack
        self.refresh()

    def refresh(self) -> None:
        current = self.current_id()
        self._building = True
        self.tree.clear()
        self._items.clear()
        for node in self._stack.root.children:
            self._add(None, node)
        self.tree.expandAll()
        item = self._items.get(current)
        if item is not None:
            self.tree.setCurrentItem(item)
        self._building = False
        self._apply_reasons()
        self._sync_controls()

    def _text(self, node) -> str:
        name = node.name if isinstance(node, Group) else self._label_of(node.id)
        return name if node.opacity >= 1.0 else f"{name}  {round(node.opacity * 100)}%"

    def _add(self, parent_item, node) -> None:
        item = QtWidgets.QTreeWidgetItem([self._text(node)])
        item.setData(0, _ID, node.id)
        flags = item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsDragEnabled
        if isinstance(node, Group):
            flags |= QtCore.Qt.ItemIsDropEnabled
        else:
            flags &= ~QtCore.Qt.ItemIsDropEnabled
        item.setFlags(flags)
        item.setCheckState(0, QtCore.Qt.Checked if node.visible else QtCore.Qt.Unchecked)
        if isinstance(node, Group):
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
        (self.tree.addTopLevelItem(item) if parent_item is None else parent_item.addChild(item))
        self._items[node.id] = item
        if isinstance(node, Group):
            for child in node.children:
                self._add(item, child)

    # -- accessors used by tests and MainWindow -------------------------------------------
    def item_for(self, node_id):
        return self._items[node_id]

    def current_id(self):
        item = self.tree.currentItem()
        return item.data(0, _ID) if item is not None else None

    def select_row(self, node_id) -> None:
        item = self._items.get(node_id)
        if item is not None:
            self._building = True
            self.tree.setCurrentItem(item)
            self._building = False
            self._sync_controls()

    def set_tree_height(self, height: int) -> None:
        self.tree.setFixedHeight(max(MIN_HEIGHT, int(height)))

    def highlight_layer(self, layer_id: int) -> None:
        """The selected layer's rows read in italics (groups keep their bold)."""
        for nid, item in self._items.items():
            if isinstance(self._stack.node(nid), Group):
                continue
            font = item.font(0)
            font.setItalic(nid.split(":")[1:2] == [str(layer_id)])
            item.setFont(0, font)

    def relabel(self) -> None:
        """Rows name what they show (a product row its layer's Show): refresh the names in
        place, the tree untouched."""
        self._building = True
        for nid, item in self._items.items():
            node = self._stack.node(nid)
            if node is not None:
                text = self._text(node)
                if item.text(0) != text:
                    item.setText(0, text)
                    item.setToolTip(0, self._reasons.get(nid) or text)
        self._building = False

    def set_reasons(self, reasons: dict) -> None:
        self._reasons = dict(reasons)
        self._apply_reasons()

    def _apply_reasons(self) -> None:
        self._building = True
        for nid, item in self._items.items():
            reason = self._reasons.get(nid)
            color = self.palette().color(QtGui.QPalette.Text)
            if reason:
                color.setAlpha(110)
            item.setForeground(0, color)
            item.setToolTip(0, reason or item.text(0))
        self._building = False

    def _sync_controls(self) -> None:
        """The slider and the buttons follow the selected row."""
        nid = self.current_id()
        node = self._stack.node(nid) if nid else None
        for widget in (self.opacity_slider, self.up_button, self.down_button, self.group_button):
            widget.setEnabled(node is not None)
        value = round((node.opacity if node is not None else 1.0) * 100)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(value)
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"{value}%")

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
            self._eye_clicked = True                     # the click that follows selects nothing
            self.stackChanged.emit()

    def _on_slider(self, value: int) -> None:
        nid = self.current_id()
        node = self._stack.node(nid) if nid else None
        if node is None:
            return
        node.opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self._building = True
        self._items[nid].setText(0, self._text(node))
        self._building = False
        self.opacityChanged.emit()
        if not self.opacity_slider.isSliderDown():       # a key or a click: no release follows
            self.stackChanged.emit()

    def _step(self, delta: int) -> None:
        """Move the selected row one place up (``-1``) or down (``+1``) in its group."""
        nid = self.current_id()
        node = self._stack.node(nid) if nid else None
        if node is None:
            return
        parent = self._stack.parent_of(nid)
        i = parent.children.index(node)
        if not 0 <= i + delta < len(parent.children):
            return
        self.move(nid, parent.id, i - 1 if delta < 0 else i + 2)

    def _group_current(self) -> None:
        nid = self.current_id()
        if nid is not None:
            self.new_group_above(nid, "Group")

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
        self.select_row(node_id)
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

    def _on_clicked(self, item, _column) -> None:
        """A click on a row's name selects it; a click on its eye only toggles the eye."""
        if self._eye_clicked:
            self._eye_clicked = False
            return
        if item is not None and not self._building:
            self.rowSelected.emit(item.data(0, _ID))

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
