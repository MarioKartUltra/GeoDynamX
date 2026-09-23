# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""point_import: CSV point-catalogue import for the shell -- the suffix glue plus the column-
mapping dialog ``dynamix.core.pointset`` never has to know about.

That module stays headless (nothing outside ``dynamix/shell/`` imports PySide6) by
only ever RAISING ``ValueError`` when lon/lat can't be identified, naming the header it actually
saw. :func:`load_points` is what turns that raise into the dialog: it tries auto-detection first
(:func:`~dynamix.core.pointset.resolve_columns`, no dialog touched -- the common case for a
catalogue with ordinary column names) and only opens :func:`_ask_mapping` when that fails.

**``_ask_mapping`` is the ONE seam this module touches ``QDialog`` through** -- monkeypatched
directly in every test (never opened for real there; a real modal dialog would hang an offscreen
run), the same discipline ``ViewDialog._pick_color``/``RightPanel._pick_color`` already use for
``QColorDialog``. It is a module-level function, not a method, because this module owns no
persistent widget of its own to hang a seam method off of -- ``load_points`` is a plain function
that takes the window as its first argument (``main_window.open_path``'s ``.csv`` branch hands
``self`` straight to it) rather than a method on some class this module would otherwise have no
reason to define.

**The accepted mapping is persisted, not re-detected on reopen.** Whatever mapping actually
resolved the columns -- auto-detected or picked in the dialog -- is stored verbatim as JSON in
``layer.tags["points.mapping"]``. ``main_window._open_project_path`` reads it straight back and
hands it to ``read_csv_points`` directly, never through this module's ``load_points`` -- so a
reopen can never re-open the dialog, even for a file whose auto-detection would now be ambiguous
(a header that changed, say). See that method's own comment for why going through this module at
all on reopen would be the wrong call site.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PySide6 import QtWidgets

from dynamix.core.pointset import read_csv_points, resolve_columns

__all__ = ["load_points"]

#: The dialog's "no such column" choice for the two OPTIONAL axes (depth, magnitude) -- lon/lat
#: get no such entry; every catalogue needs a location.
_NONE_CHOICE = "(none)"


def _csv_headers(path: str) -> list[str]:
    """The file's header row, stdlib ``csv`` only -- same reader ``pointset.read_csv_points``
    itself uses for the header, so a dialog built from this always lists exactly what auto-
    detection saw."""
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f), [])


def load_points(window, path: str) -> None:
    """Load a CSV point catalogue at ``path`` as a new layer on ``window``.

    Tries auto-detection first; on a ``ValueError`` (lon/lat unmappable) opens the mapping dialog
    -- a cancelled dialog leaves ``window`` untouched, no source or layer added. Either way, the
    mapping that actually worked is what gets persisted to the new layer's
    ``tags["points.mapping"]``, and what the new source is registered under (``kind="points"``).

    The loaded :class:`~dynamix.core.pointset.PointSet` is handed to ``window.add_layer_row``
    exactly as a raster field is (the "it IS the layer's field for engine purposes") and the
    new layer becomes the selected one -- drawing it is a separate step; this only has to make sure selecting it is safe (``MainWindow._select_layer``'s point-layer tolerance).
    """
    headers = _csv_headers(path)
    try:
        mapping = resolve_columns(headers)
    except ValueError:
        mapping = _ask_mapping(headers)
        if mapping is None:
            return
    pset = read_csv_points(path, mapping=mapping)

    source = window.project.add_source(path, kind="points")
    layer = window.project.add_layer(Path(path).stem or path, source.source_id)
    layer.tags["points.mapping"] = json.dumps(mapping)
    window.add_layer_row(layer, pset)
    window.layer_list.select_layer(layer.layer_id)


def _ask_mapping(headers: list[str]) -> dict[str, str] | None:
    """The CSV column-mapping dialog -- opened ONLY when auto-detection couldn't
    identify lon/lat on its own. One combo per axis, every header listed verbatim; lon/lat are
    required, depth/mag carry a leading :data:`_NONE_CHOICE` entry to opt the axis out entirely.
    Returns the mapping :func:`load_points` hands straight to ``read_csv_points``, or ``None`` if
    the user cancelled.
    """
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Map CSV columns")
    form = QtWidgets.QFormLayout(dialog)

    def _combo(optional: bool) -> QtWidgets.QComboBox:
        box = QtWidgets.QComboBox()
        if optional:
            box.addItem(_NONE_CHOICE)
        box.addItems(list(headers))
        return box

    lon_box, lat_box = _combo(False), _combo(False)
    depth_box, mag_box = _combo(True), _combo(True)
    form.addRow("Longitude", lon_box)
    form.addRow("Latitude", lat_box)
    form.addRow("Depth", depth_box)
    form.addRow("Magnitude", mag_box)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None

    mapping = {"lon": lon_box.currentText(), "lat": lat_box.currentText()}
    if depth_box.currentText() != _NONE_CHOICE:
        mapping["depth"] = depth_box.currentText()
    if mag_box.currentText() != _NONE_CHOICE:
        mapping["mag"] = mag_box.currentText()
    return mapping
