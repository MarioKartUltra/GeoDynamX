# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The D(h) comparison board on the shell side: the board as a QObject that says when it changes,
the path every fitting window sends a spectrum by, and the board's inspector.

The main window owns one :class:`SpectraBoardModel` (``MainWindow.spectra_board``); a window
finds it through its parents (:func:`board_of`). A sender builds a
:class:`~dynamix.model.spectra_board.SentSpectrum` from its fit (:func:`from_fit`) and hands it to
:func:`send`; the board's ``sent`` signal carries the status line naming what was sent and
``boardChanged`` redraws the comparison."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.spectra_board import SentSpectrum, SpectraBoard

__all__ = ["BOARD_HUES", "SpectraBoardInspector", "SpectraBoardModel", "ancestor_with", "board_of",
           "from_fit", "generalized_entry", "say", "send", "tool_identity"]

#: One hue per sent spectrum, in sending order: DATA identity (which spectrum a curve is), the
#: Theme-Rule carve-out the partition fitter's source hues document, as numeric RGB.
BOARD_HUES = ((242, 169, 59), (76, 201, 240), (241, 91, 181), (158, 240, 26), (255, 107, 107),
              (179, 136, 255), (64, 224, 168), (255, 214, 102))


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


class SpectraBoardModel(QtCore.QObject):
    """The session's :class:`~dynamix.model.spectra_board.SpectraBoard`: every change emits
    ``boardChanged``; a send also emits ``sent`` with the status line naming the spectrum."""

    boardChanged = QtCore.Signal()
    sent = QtCore.Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.board = SpectraBoard()

    def __len__(self) -> int:
        return len(self.board)

    def entries(self) -> list:
        return self.board.entries()

    def get(self, key: str) -> SentSpectrum:
        return self.board.get(key)

    def next_colour(self) -> str:
        """The first hue no entry wears, else the palette again in turn."""
        used = {e.colour for e in self.board.entries()}
        for rgb in BOARD_HUES:
            if _hex(rgb) not in used:
                return _hex(rgb)
        return _hex(BOARD_HUES[len(self.board) % len(BOARD_HUES)])

    def add(self, spectrum: SentSpectrum) -> str:
        if not spectrum.colour:
            spectrum.colour = self.next_colour()
        replaced = any(e.same_as(spectrum) for e in self.board.entries())
        key = self.board.add(spectrum)
        self.sent.emit(f"{'sent again' if replaced else 'sent'} to the D(h) comparison: "
                       f"{spectrum.full_name}")
        self.boardChanged.emit()
        return key

    def _changed(self, method, key, *args) -> None:
        method(key, *args)
        self.boardChanged.emit()

    def remove(self, key: str) -> None:
        self._changed(self.board.remove, key)

    def rename(self, key: str, name: str) -> None:
        self._changed(self.board.rename, key, name)

    def set_visible(self, key: str, on: bool) -> None:
        self._changed(self.board.set_visible, key, on)

    def set_show_errors(self, key: str, on: bool) -> None:
        self._changed(self.board.set_show_errors, key, on)

    def set_show_tangent(self, key: str, on: bool) -> None:
        self._changed(self.board.set_show_tangent, key, on)

    def set_colour(self, key: str, colour: str) -> None:
        self._changed(self.board.set_colour, key, colour)

    def move(self, key: str, step: int) -> None:
        self._changed(self.board.move, key, step)


def ancestor_with(widget, attr: str):
    """``widget`` or its nearest parent that has ``attr`` set, else None."""
    while widget is not None:
        if getattr(widget, attr, None) is not None:
            return widget
        widget = widget.parentWidget()
    return None


def board_of(widget) -> "SpectraBoardModel | None":
    """The board ``widget`` sends to: its own ``spectra_board``, else its nearest parent's."""
    holder = ancestor_with(widget, "spectra_board")
    return None if holder is None else holder.spectra_board


def say(widget, text: str) -> None:
    """``text`` on the main window's status line, when ``widget`` has one above it."""
    holder = ancestor_with(widget, "_notify")
    if holder is not None:
        holder._notify(text, "status")


def tool_identity(window) -> dict:
    """The producing result's tool identity as the main window stamped it on ``window``
    (``_opened_on``: the layer's id, the landed analysis key and the filters), with the layer's
    name; None in each field for a window opened on no landed result."""
    layer_id, recipe, filters = getattr(window, "_opened_on", None) or (None, None, None)
    holder = ancestor_with(window.parentWidget(), "_layer_by_id")
    layer = holder._layer_by_id.get(layer_id) if holder is not None else None
    return {"recipe": recipe, "layer_id": layer_id, "layer": getattr(layer, "name", None),
            "filters": filters}


def from_fit(window, fit: dict, kind: str, *, scale_window=None, fit_mode=None, table=None,
             settings=None, tangent=None, keep=None, origin=None) -> SentSpectrum:
    """A :class:`SentSpectrum` from a producer's fit dict: its ``name`` (the layer's name after it
    when the name lacks it) as the full and the starting display name, ``h`` and ``D`` with
    ``h_err`` and ``D_err`` as the bars when the fit has them, ``q`` (or ``q_list``) and ``tau``
    when the fit carries a τ(q) row per point; ``keep`` selects the points sent. The identity is
    the window's tool identity (or ``origin``, another result's), the scale window, the fit mode,
    the table and the settings."""
    origin = dict(origin) if origin is not None else tool_identity(window)
    name = str(fit.get("name") or kind)
    layer = origin.get("layer")
    if layer and layer not in name:
        name = f"{name} · {layer}"
    h = np.asarray(fit["h"], dtype=np.float64).ravel()
    keep = np.ones(h.size, bool) if keep is None else np.asarray(keep, bool)

    def column(key):
        value = fit.get(key)
        if value is None or np.ndim(value) != 1 or np.size(value) != h.size:
            return None
        return np.asarray(value, dtype=np.float64)[keep]

    q = column("q") if fit.get("q") is not None else column("q_list")
    tau = column("tau")
    if q is None or tau is None:
        q = tau = None
    identity = {"tool": origin,
                "scale_window": None if scale_window is None else [float(v) for v in scale_window],
                "fit_mode": fit_mode, "table": table, "settings": dict(settings or {})}
    return SentSpectrum(name=name, full_name=name, kind=kind, h=h[keep], D=column("D"),
                        dh=column("h_err"), dD=column("D_err"), q=q, tau=tau, tangent=tangent,
                        identity=identity)


def _grid(value):
    """A settings value as the identity keeps it: a grid as its ends and its length."""
    if np.ndim(value) == 1 and np.size(value) > 2:
        a = np.asarray(value, dtype=np.float64)
        return {"min": float(a.min()), "max": float(a.max()), "n": int(a.size)}
    return value


def generalized_entry(window, fit: dict, curve: str) -> "SentSpectrum | None":
    """The generalized spectrum window's ``curve`` from its fit: ``converged``, the estimate on
    the points the curve draws (the converged minima when the fit marks them, else the minima off
    the grid's boundary) with the leave-one-scale-out half-width as bars when the fit carries the
    band; ``legendre``, the Legendre spectrum of the same τ table at γ = 0, without bars. None
    when the fit holds no such curve."""
    h, D = np.asarray(fit["h"], dtype=np.float64), np.asarray(fit["D"], dtype=np.float64)
    dD = None
    if curve == "legendre":
        if fit.get("legendre") is None:
            return None
        D, name = np.asarray(fit["legendre"], dtype=np.float64), f"{fit['name']}, Legendre (γ = 0)"
    else:
        drawn = (np.asarray(fit["solid"], bool) if "solid" in fit
                 else ~np.asarray(fit["boundary"], bool) & np.isfinite(D))
        D, name = np.where(drawn, D, np.nan), f"{fit['name']}, converged"
        band = fit.get("band")
        dD = None if band is None else band["half_width"]
    settings = {"curve": curve, **{k: _grid(v) for k, v in (fit.get("settings") or {}).items()
                                   if k not in ("lo", "hi")}}
    return from_fit(window, {"name": name, "h": h, "D": D, "D_err": dD}, "generalized",
                    scale_window=fit.get("window"), settings=settings)


def send(window, spectrum: "SentSpectrum | None", nothing: str = "nothing to send") -> bool:
    """Put ``spectrum`` on the board ``window`` sends to (the board's ``sent`` names it on the
    status line); ``nothing`` goes to the status line instead when there is no spectrum."""
    board = board_of(window)
    if spectrum is None or not np.isfinite(spectrum.D).any():
        say(window, nothing)
        return False
    if board is None:
        return False
    board.add(spectrum)
    return True


class SpectraBoardInspector(QtWidgets.QWidget):
    """The board's inspector, one row per entry in board order: shown, its colour (a click opens
    the colour dialog), its display name (a double click edits it, Enter commits, Esc reverts,
    an empty name restores the full name), its error bars and its tangent where it has them,
    Up, Down and Remove. Every cell's tooltip is the full name. The owner sets ``board`` and
    calls :meth:`refresh` when the board changes; the rows are rebuilt on the event loop's next
    turn, so an edit or a button that changed the board never has its own cell replaced while
    its signal is being delivered."""

    SHOWN, COLOUR, NAME, ERRORS, TANGENT = range(5)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.board: "SpectraBoardModel | None" = None
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["shown", "", "name", "bars", "tangent", "", "", ""])
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
                                   | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.NAME, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        self._keys: list = []
        self._rebuild_timer = QtCore.QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(0)
        self._rebuild_timer.timeout.connect(self._rebuild)

    def _key(self, row: int) -> str:
        return self._keys[row]

    @staticmethod
    def _check(on: bool, enabled: bool = True) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem()
        flags = QtCore.Qt.ItemFlag.ItemIsUserCheckable
        if enabled:
            flags |= QtCore.Qt.ItemFlag.ItemIsEnabled
        item.setFlags(flags)
        item.setCheckState(QtCore.Qt.CheckState.Checked if on and enabled else QtCore.Qt.CheckState.Unchecked)
        return item

    def refresh(self) -> None:
        self._rebuild_timer.start()

    def _rebuild(self) -> None:
        entries = self.board.entries() if self.board is not None else []
        self.table.blockSignals(True)
        self._keys = [e.key for e in entries]
        self.table.setRowCount(len(entries))
        for row, e in enumerate(entries):
            swatch = QtWidgets.QTableWidgetItem()
            swatch.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            swatch.setBackground(QtGui.QColor(e.colour))
            name = QtWidgets.QTableWidgetItem(e.name)
            name.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsEditable)
            items = (self._check(e.visible), swatch, name, self._check(e.show_errors, e.has_errors),
                     self._check(e.show_tangent, e.tangent is not None))
            for col, item in enumerate(items):
                item.setToolTip(e.full_name)
                self.table.setItem(row, col, item)
            for col, (text, tip, step) in enumerate((("↑", "Up", -1), ("↓", "Down", 1), ("×", "Remove", 0)), 5):
                button = QtWidgets.QToolButton()
                button.setText(text)
                button.setToolTip(tip)
                board, key = self.board, e.key
                button.clicked.connect(lambda _c=False, board=board, key=key, step=step:
                                       board.move(key, step) if step else board.remove(key))
                self.table.setCellWidget(row, col, button)
        self.table.blockSignals(False)
        rows = sum(self.table.rowHeight(r) for r in range(min(len(entries), 5)))
        self.table.setFixedHeight(self.table.horizontalHeader().height() + rows + 2 * self.table.frameWidth())

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        key, col = self._key(item.row()), item.column()
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        if col == self.NAME:
            self.board.rename(key, item.text())
        elif col == self.SHOWN:
            self.board.set_visible(key, checked)
        elif col == self.ERRORS:
            self.board.set_show_errors(key, checked)
        elif col == self.TANGENT:
            self.board.set_show_tangent(key, checked)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col != self.COLOUR:
            return
        entry = self.board.get(self._key(row))
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(entry.colour), self, f"Colour of {entry.name}")
        if colour.isValid():
            self.board.set_colour(entry.key, colour.name())
