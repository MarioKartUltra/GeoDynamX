# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Importing from multi-grid containers: sensor grouping, the checkbox dialog, and the
window flow that lands each group as its own (multiband) dataset."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets  # noqa: F401

from dynamix.shell.import_dialog import ImportGridsDialog, group_container_grids

# The real ASTER shape of the problem: three telescopes at three resolutions, a backsight
# on its own grid, and ancillary lattices/tables that must not masquerade as image bands.
ASTER_SUBS = [
    ("VNIR_Band3B/ImageData", "VNIR_Band3B/ImageData (5400, 5000)"),
    ("VNIR_Band1/ImageData", "VNIR_Band1/ImageData (4200, 4100)"),
    ("VNIR_Band2/ImageData", "VNIR_Band2/ImageData (4200, 4100)"),
    ("SWIR_Band4/ImageData", "SWIR_Band4/ImageData (2100, 2048)"),
    ("SWIR_Band5/ImageData", "SWIR_Band5/ImageData (2100, 2048)"),
    ("TIR_Band10/ImageData", "TIR_Band10/ImageData (700, 700)"),
    ("VNIR/VNIR_Supplement", "VNIR/VNIR_Supplement (9602, 58)"),
    ("SWIR_Band4/Latitude", "SWIR_Band4/Latitude (107, 104)"),
    ("Cloud_Coverage_Table", "Cloud_Coverage_Table (106, 104)"),
]
ASTER_DIMS = {
    "VNIR_Band3B/ImageData": (5400, 5000),
    "VNIR_Band1/ImageData": (4200, 4100),
    "VNIR_Band2/ImageData": (4200, 4100),
    "SWIR_Band4/ImageData": (2100, 2048),
    "SWIR_Band5/ImageData": (2100, 2048),
    "TIR_Band10/ImageData": (700, 700),
    "VNIR/VNIR_Supplement": (9602, 58),
    "SWIR_Band4/Latitude": (107, 104),
    "Cloud_Coverage_Table": (106, 104),
}


def test_image_grids_group_by_sensor_and_resolution():
    groups = group_container_grids(ASTER_SUBS, ASTER_DIMS)
    by_label = {label: (ids, img) for label, ids, img in groups}
    assert by_label["VNIR (4200, 4100)"] == (
        ["VNIR_Band1/ImageData", "VNIR_Band2/ImageData"], True)
    assert by_label["VNIR (5400, 5000)"] == (["VNIR_Band3B/ImageData"], True)  # 3B alone
    assert by_label["SWIR (2100, 2048)"][0] == ["SWIR_Band4/ImageData",
                                                "SWIR_Band5/ImageData"]
    assert by_label["TIR (700, 700)"] == (["TIR_Band10/ImageData"], True)
    # lattices, supplements and tables are never image groups
    for sid in ("VNIR/VNIR_Supplement", "SWIR_Band4/Latitude", "Cloud_Coverage_Table"):
        assert by_label[sid] == ([sid], False)


def test_dialog_returns_only_the_ticked_bands_grouped(qtbot):
    dlg = ImportGridsDialog("x.hdf", ASTER_SUBS, ASTER_DIMS)
    qtbot.addWidget(dlg)
    assert dlg.groups() == []                       # nothing ticked by default
    ticked = 0
    for i in range(dlg.tree.topLevelItemCount()):
        top = dlg.tree.topLevelItem(i)
        if top.text(0).startswith("VNIR (4200"):
            for j in range(top.childCount()):
                top.child(j).setCheckState(0, QtCore.Qt.Checked)
                ticked += 1
        if top.text(0).startswith("TIR"):
            top.child(0).setCheckState(0, QtCore.Qt.Checked)
    assert ticked == 2
    assert dlg.groups() == [
        ("VNIR (4200, 4100)", ["VNIR_Band1/ImageData", "VNIR_Band2/ImageData"]),
        ("TIR (700, 700)", ["TIR_Band10/ImageData"]),
    ]


from tests.test_ingest import _aster_like  # noqa: E402
from tests.test_shell_window import stub_devices, window  # noqa: E402,F401


def test_each_imported_group_lands_as_its_own_dataset_row(window, qtbot, tmp_path):
    pytest.importorskip("pyhdf")
    try:
        path = _aster_like(tmp_path)
    except Exception as exc:                          # pragma: no cover - env guard
        pytest.skip(f"pyhdf cannot create HDF4 groups here: {exc}")
    win = window
    win._start_worker = lambda: None
    n_before = len(win.project.layers)
    win._open_container_groups(str(path), [
        ("VNIR (12, 10)", ["VNIR_Band1/ImageData", "VNIR_Band2/ImageData"]),
        ("SWIR_Band4/ImageData", ["SWIR_Band4/ImageData"]),
    ])
    new = [l for l in win.project.layers[n_before:] if l.parent_id is None]
    assert [l.name for l in new] == ["aster_like · VNIR (12, 10)",
                                     "aster_like · SWIR_Band4/ImageData"]
    fields = [win._fields[l.layer_id] for l in new]
    assert fields[0].values.shape == (12, 10, 2)
    assert fields[1].values.shape == (6, 8)
    assert fields[0].provenance["bands"] == ["VNIR_Band1/ImageData",
                                             "VNIR_Band2/ImageData"]
    assert fields[0].provenance["source"] == str(path)   # ROI reads know their file
    assert "imported" in win.statusBar().currentMessage()


def test_open_path_routes_a_container_through_the_import_dialog(window, qtbot, tmp_path,
                                                                monkeypatch):
    pytest.importorskip("pyhdf")
    import dynamix.shell.main_window as mw

    try:
        path = _aster_like(tmp_path)
    except Exception as exc:                          # pragma: no cover - env guard
        pytest.skip(f"pyhdf cannot create HDF4 groups here: {exc}")

    class _Dialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return 1

        def groups(self):
            return [("VNIR (12, 10)", ["VNIR_Band1/ImageData", "VNIR_Band2/ImageData"])]

    monkeypatch.setattr(mw, "ImportGridsDialog", _Dialog)
    win = window
    win._start_worker = lambda: None
    n_before = len(win.project.layers)
    win.open_path(str(path))
    new = [l for l in win.project.layers[n_before:] if l.parent_id is None]
    assert len(new) == 1 and win._fields[new[0].layer_id].values.shape == (12, 10, 2)
