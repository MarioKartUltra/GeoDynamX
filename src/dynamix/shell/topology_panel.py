# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""TopologyPanel: the right panel's "Topology" section.

Pure Qt -- no imports from ``main_window`` (mirrors ``arrangement/mask_row.py``'s and
``arrangement/group_palette.py``'s own decoupling: this widget knows nothing about ``Project``,
``LinkStore`` or ``Layer``, only about the plain dicts/ints ``MainWindow`` hands it). NOT
view-scoped -- links name chains by ``(layer_id, transform, kind, obj_id)``, not by which
center-stack page happens to be showing, so this section stays visible in every one of the three
views (unlike ``Groups``/``Display mask``, which are arrangement-only).

Two outward signals carry every gesture: ``linkRequested(object)`` -- an int code override chosen
from the combo, or ``None`` for "let MainWindow's own suggest_code call decide" -- and
``unlinkRequested(int)``, the currently-selected row's position. Everything else (which two chains
a "Link" click actually joins, resolving their points, calling ``suggest_code``, mutating the
store) is ``MainWindow``'s job; this widget only ever displays what it is told via :meth:`set_rows`
/ :meth:`set_chain_graph_count` / :meth:`set_code_choices`.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.topology.codes import name_for, permitted


class TopologyPanel(QtWidgets.QWidget):
    #: int code chosen in the override combo, or None for "auto" (MainWindow calls suggest_code).
    linkRequested = QtCore.Signal(object)
    #: the row index (into the list this panel is currently showing) to drop.
    unlinkRequested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._chain_graph_label = QtWidgets.QLabel("chain graph: —")
        self._chain_graph_label.setProperty("muted", "true")
        layout.addWidget(self._chain_graph_label)

        self._list = QtWidgets.QListWidget()
        layout.addWidget(self._list)

        combo_row = QtWidgets.QHBoxLayout()
        combo_row.addWidget(QtWidgets.QLabel("Code"))
        self._code_combo = QtWidgets.QComboBox()
        self._code_combo.addItem("(auto)", None)
        combo_row.addWidget(self._code_combo, 1)
        layout.addLayout(combo_row)

        button_row = QtWidgets.QHBoxLayout()
        self._link_button = QtWidgets.QPushButton("Link")
        self._link_button.clicked.connect(self._on_link_clicked)
        button_row.addWidget(self._link_button)
        self._unlink_button = QtWidgets.QPushButton("Unlink")
        self._unlink_button.clicked.connect(self._on_unlink_clicked)
        button_row.addWidget(self._unlink_button)
        layout.addLayout(button_row)

    # -- MainWindow -> panel (display only) ---------------------------------------------------

    def set_chain_graph_count(self, n: int | None) -> None:
        """``n`` is the active layer's cached ``chain_topology`` edge count, or ``None`` when no
        such product is cached for it right now (no chain_topology step in the chain, or nothing
        resolved yet) -- shown as an honest em-dash rather than a misleading "0 edges"."""
        self._chain_graph_label.setText(
            "chain graph: —" if n is None else f"chain graph: {n} edges")

    def set_code_choices(self, kind_a: str, kind_b: str, space_dim: int) -> None:
        """Repopulate the override combo from ``permitted(kind_a, kind_b, space_dim)``.

        Final branch review, item 4: this docstring used to describe a call site that does not
        exist -- there is no per-pick re-population hook anywhere. ``MainWindow`` calls this
        exactly ONCE, at panel construction, with the fixed ``("line", "line", 2)``: every
        ``ObjRef`` this slice ever builds is kind ``"line"`` (a chain, whichever transform drew
        it) in a 2-D pixel space, so the override combo's choice set is invariant for the
        window's whole life. The ``kind_a``/``kind_b``/``space_dim`` parameters stay general on
        purpose -- this widget itself has no opinion on what gets linked -- for a future slice
        that links other kinds/dimensions and would need to re-populate per pick.
        """
        self._code_combo.blockSignals(True)
        self._code_combo.clear()
        self._code_combo.addItem("(auto)", None)
        for code in sorted(permitted(kind_a, kind_b, space_dim)):
            label = name_for(code, kind_a, kind_b, space_dim) or str(code)
            self._code_combo.addItem(f"{label} ({code})", code)
        self._code_combo.blockSignals(False)

    def set_rows(self, rows: list[dict]) -> None:
        """``rows``: one dict per current link -- ``{"code", "kind_a", "kind_b", "space_dim",
        "a_id", "b_id", "scale_first_contact"}`` (``a_id``/``b_id`` are the two nodes' own
        ``ObjRef.node_id()`` strings). Replaces the list wholesale -- this panel holds no link
        state of its own, only what it was last told to show."""
        self._list.clear()
        for row in rows:
            name = name_for(row["code"], row["kind_a"], row["kind_b"], row["space_dim"])
            label = name if name is not None else str(row["code"])
            scale = row.get("scale_first_contact")
            scale_text = f"@a={scale:g}px" if scale is not None else "@a=?"
            self._list.addItem(f"{label}  {row['a_id']} <-> {row['b_id']}  {scale_text}")

    # -- panel -> MainWindow (gestures) --------------------------------------------------------

    def _on_link_clicked(self) -> None:
        self.linkRequested.emit(self._code_combo.currentData())

    def _on_unlink_clicked(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self.unlinkRequested.emit(row)
