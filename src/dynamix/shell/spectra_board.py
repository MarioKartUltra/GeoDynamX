# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The D(h) comparison board on the shell side: the board as a QObject that says when it changes,
the path every fitting window sends a spectrum by, the comparison drawn on a D(h) panel and the
board's inspector.

The main window owns one :class:`SpectraBoardModel` (``MainWindow.spectra_board``); a window
finds it through its parents (:func:`board_of`). A sender builds a
:class:`~dynamix.model.spectra_board.SentSpectrum` from its fit (:func:`from_fit`) and hands it to
:func:`send`; the board's ``sent`` signal carries the status line naming what was sent and
``boardChanged`` redraws the comparison (:class:`BoardComparison`, the D(h) construction
window's)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.core import spectra
from dynamix.model.spectra_board import SentSpectrum, SpectraBoard

__all__ = ["BOARD_HUES", "BoardComparison", "SpectraBoardInspector", "SpectraBoardModel", "ancestor_with",
           "board_of", "dh_view_range", "from_fit", "generalized_entry", "kind_of_method", "landed_dialog",
           "say", "send", "send_landed", "tool_identity"]

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


def kind_of_method(method: str) -> str:
    """What a spectrum fitted over ``method``'s partition tables is on the board: the DWT or CWT
    leaders' table, else the canonical fit."""
    return {"wavelet_leaders": "dwt_leaders", "cwt_leaders": "cwt_leaders"}.get(str(method or ""), "canonical")


def landed_dialog(parent, landed) -> "tuple[QtWidgets.QDialog, QtWidgets.QListWidget]":
    """The list of the other layers' landed results, one unchecked row each."""
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Send landed results")
    box = QtWidgets.QVBoxLayout(dialog)
    box.addWidget(QtWidgets.QLabel("Fitted under this window's scale window, fit mode, η and frame:"))
    rows = QtWidgets.QListWidget()
    for source in landed:
        item = QtWidgets.QListWidgetItem(source["name"])
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        rows.addItem(item)
    box.addWidget(rows)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                         | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    box.addWidget(buttons)
    return dialog, rows


def send_landed(window, landed, names=None, *, lo: float, hi: float, mode: str, eta: float, frame: str,
                log2_L=None, table: str = "std", settings=None) -> int:
    """Send the landed results named in ``names`` (asked in :func:`landed_dialog` when None) from
    ``window``: a result's ``table`` (``std`` or ``cmax``, its only one when it has one) fitted
    over ``(lo, hi)`` under ``mode``, ``eta`` and ``frame``, its histogram construction as drawn.
    Returns how many spectra were sent."""
    if names is None:
        dialog, rows = landed_dialog(window, landed)
        accepted = dialog.exec()
        names = [rows.item(i).text() for i in range(rows.count())
                 if rows.item(i).checkState() == QtCore.Qt.CheckState.Checked] if accepted else []
        dialog.deleteLater()
    sent = 0
    for source in (s for s in landed if s["name"] in names):
        origin = source.get("identity") or {}
        hd = (source.get("hd_cmax" if table == "cmax" else "hd_std") or source.get("hd_std")
              or source.get("hd_cmax"))
        if hd is not None:
            fit = spectra.fit_spectra(hd, float(lo), float(hi), eta=eta, frame=frame, fit_mode=mode,
                                      log2_L=log2_L if mode != "naive" else None)
            one = source.get("hd_cmax") is None or source.get("hd_std") is source.get("hd_cmax")
            entry = from_fit(window, {**fit, "name": source["name"]}, kind_of_method(source.get("method")),
                             scale_window=(lo, hi), fit_mode=mode, table=None if one else table,
                             settings=settings, origin=origin)
            sent += send(window, entry, f"{source['name']}: no finite D(h) to send")
        if source.get("dh_hist") is not None:
            h, D, dD = source["dh_hist"]
            entry = from_fit(window, {"name": f"{source['name']} (histogram)", "h": h, "D": D, "D_err": dD},
                             "histogram", settings={"pin_to_D0": False}, origin=origin)
            sent += send(window, entry, f"{source['name']}: the histogram has no finite D(h)")
    return sent


def dh_view_range(plot) -> dict:
    """The D(h) range ``plot`` shows, as a figure record names it."""
    (x0, x1), (y0, y1) = plot.getViewBox().viewRange()
    return {"x": [x0, x1], "y": [y0, y1], "units": "h, D(h)"}


def _extent(entry: SentSpectrum) -> "tuple | None":
    """``(h min, h max, D min, D max)`` over ``entry``'s finite points and the bars it draws."""
    finite = np.isfinite(entry.h) & np.isfinite(entry.D)
    if not finite.any():
        return None
    h, D = entry.h[finite], entry.D[finite]
    dh = dD = 0.0
    if entry.show_errors:
        dh = 0.0 if entry.dh is None else np.nan_to_num(entry.dh[finite])
        dD = 0.0 if entry.dD is None else np.nan_to_num(entry.dD[finite])
    return (float(np.min(h - dh)), float(np.max(h + dh)), float(np.min(D - dD)), float(np.max(D + dD)))


class BoardComparison:
    """The D(h) comparison on a window's D(h) panel: every visible board entry in board order
    beside the window's own spectrum, in its colour (a construction as points, a fit as a line
    through its points), its error bars while shown, its tangent dashed across the view while
    shown, its τ(q) on the τ panel when it carries one; the board's inspector under the panels
    while the board holds entries. The window redraws on every board change while shown and
    catches up when shown again. A newly drawn entry outside the D(h) view widens it: the panel
    ranges again over the window's own drawing and the entries' points and bars.

    The window calls :meth:`_build_board` with its D(h) and τ(q) plots once they exist (and lays
    out the inspector it returns), :meth:`attach_board` once built, and redraws its legend in
    ``_on_board_drawn`` (``_board_legend`` lists ``(curve, display name, full name)``). Mixed in
    before the Qt base class; no ``__init__`` of its own."""

    spectra_board: "SpectraBoardModel | None" = None
    _board_connected: bool = False
    _board_items: tuple = ()
    _board_legend: tuple = ()
    _board_tangents: tuple = ()
    _board_drawn: tuple = ()

    def _build_board(self, d_plot, tau_plot) -> "SpectraBoardInspector":
        """The plots the comparison draws on and its inspector, hidden while the board is empty;
        the carried tangents follow the D(h) view."""
        self._board_plots = (d_plot, tau_plot)
        self.board_inspector = SpectraBoardInspector(self)
        self.board_inspector.setVisible(False)
        d_plot.getViewBox().sigXRangeChanged.connect(lambda *_a: self._update_board_tangents())
        return self.board_inspector

    def attach_board(self, board) -> None:
        """Draw ``board``'s visible entries, again on every change."""
        if board is None:
            return
        self._detach_board()
        self.spectra_board = self.board_inspector.board = board
        board.boardChanged.connect(self._draw_board)
        self._board_connected = True
        self._draw_board()

    def _detach_board(self) -> None:
        if self._board_connected:
            self.spectra_board.boardChanged.disconnect(self._draw_board)
            self._board_connected = False

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        if event.isAccepted():
            self._detach_board()

    def hideEvent(self, event) -> None:
        self._detach_board()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        if self.spectra_board is not None and not self._board_connected:
            self.attach_board(self.spectra_board)
        super().showEvent(event)

    def _add_board_item(self, plot, item, bounds: bool = False) -> None:
        # A tangent spans the view and a τ row shares its panel with the window's own: neither
        # sets the extent. The D(h) points and bars do (see _widen_to_board).
        plot.addItem(item, ignoreBounds=not bounds)
        self._board_items.append((plot, item))

    def _draw_board(self) -> None:
        d_plot, tau_plot = self._board_plots
        for plot, item in self._board_items:
            plot.removeItem(item)
        self._board_items, self._board_legend, self._board_tangents = [], [], []
        before, drawn = self._board_drawn, []
        for entry in self.spectra_board.entries() if self._board_connected else []:
            if not entry.visible:
                continue
            drawn.append(entry)
            colour = pg.mkColor(entry.colour)
            pen = None if entry.kind in ("histogram", "boxcount") else pg.mkPen(colour, width=1.4)
            curve = pg.PlotDataItem(entry.h, entry.D, pen=pen, connect="finite", symbol="o",
                                    symbolSize=4, symbolPen=None, symbolBrush=colour)
            self._add_board_item(d_plot, curve, bounds=True)
            finite = np.isfinite(entry.h) & np.isfinite(entry.D)
            if entry.show_errors and entry.has_errors and finite.any():
                bars = {"x": entry.h[finite], "y": entry.D[finite]}
                for key, err in (("height", entry.dD), ("width", entry.dh)):
                    if err is not None:
                        bars[key] = np.where(np.isfinite(err[finite]), 2.0 * err[finite], 0.0)
                self._add_board_item(d_plot, pg.ErrorBarItem(pen=pg.mkPen(colour), beam=0.0, **bars), bounds=True)
            if entry.show_tangent and entry.tangent is not None:
                line = pg.PlotDataItem(pen=pg.mkPen(colour, width=1.2, style=QtCore.Qt.DashLine))
                self._add_board_item(d_plot, line)
                self._board_tangents.append((line, entry.tangent))
            if entry.tau is not None:
                self._add_board_item(tau_plot, pg.PlotDataItem(entry.q, entry.tau, pen=pen, connect="finite",
                                                               symbol="o", symbolSize=3, symbolPen=None,
                                                               symbolBrush=colour))
            self._board_legend.append((curve, entry.name, entry.full_name))
        self._board_drawn = tuple(drawn)
        self._widen_to_board([e for e in drawn if not any(e is old for old in before)])
        self._update_board_tangents()
        self._on_board_drawn()
        self.board_inspector.setVisible(self._board_connected and len(self.spectra_board) > 0)
        self.board_inspector.refresh()

    def _widen_to_board(self, new) -> None:
        """When an entry drawn anew (sent, sent again or shown again) lies outside the D(h) view,
        the panel ranges again over the window's own drawing and the board's points and bars;
        a rename, a colour or an entry already in view leaves the view as it is."""
        boxes = [b for b in map(_extent, new) if b is not None]
        if not boxes:
            return
        view = self._board_plots[0].getViewBox()
        (x0, x1), (y0, y1) = view.viewRange()
        if all(x0 <= b[0] and b[1] <= x1 and y0 <= b[2] and b[3] <= y1 for b in boxes):
            return
        view.enableAutoRange()
        view.updateAutoRange()

    def _update_board_tangents(self) -> None:
        """The carried tangents across the D(h) view."""
        x = np.asarray(self._board_plots[0].getViewBox().viewRange()[0], dtype=np.float64)
        for line, tangent in self._board_tangents:
            line.setData(x, tangent["intercept"] + tangent["slope"] * x)

    def drawn_spectra(self) -> list:
        """What D(h) draws from the comparison, in legend order, as a figure record lists it."""
        return [entry.record() for entry in self._board_drawn] if self._board_connected else []


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
        title = QtWidgets.QLabel("D(h) comparison: the spectra sent here")
        title.setToolTip("Each window's Send to D(h) comparison puts its spectrum here; rename, recolour, "
                         "reorder or remove it in this table")
        layout.addWidget(title)
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
