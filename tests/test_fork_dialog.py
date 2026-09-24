# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ForkDialog: what a derivative dataset takes (raster bands and/or vectors), where its row sits,
and whether it is temporary or saved to a file."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.shell.fork_dialog import ForkDialog


def _dialog(qtbot, labels=("as shown (C2)", "C1", "C2", "residual"), counts=(0, 0)):
    dlg = ForkDialog(name="dem · ssa2d · C2", raster_labels=list(labels), vector_counts=counts,
                     parent_name="dem")
    qtbot.addWidget(dlg)
    return dlg


def _ok(dlg):
    return dlg._buttons.button(QtWidgets.QDialogButtonBox.Ok)


def test_the_defaults_take_the_shown_raster_into_a_file_of_its_own(qtbot):
    dlg = _dialog(qtbot)
    c = dlg.choices()
    assert c == {"name": "dem · ssa2d · C2", "bands": [0], "vectors": False, "nest": False,
                 "temporary": False}
    assert not dlg._vector_box.isEnabled()               # this result has no vectors
    assert _ok(dlg).isEnabled()


def test_several_bands_the_vectors_nesting_and_temporary(qtbot):
    dlg = _dialog(qtbot, counts=(120, 7))
    assert "120 extrema points" in dlg._vector_box.title() and "7 maxima lines" in \
        dlg._vector_box.title()
    dlg._bands.item(1).setCheckState(QtCore.Qt.Checked)
    dlg._bands.item(3).setCheckState(QtCore.Qt.Checked)
    dlg._vector_box.setChecked(True)
    dlg._nest.setChecked(True)
    dlg._temporary.setChecked(True)
    dlg._name.setText("stack")
    assert dlg.choices() == {"name": "stack", "bands": [0, 1, 3], "vectors": True, "nest": True,
                             "temporary": True}
    assert "inside dem" in dlg._nest.text()


def test_a_vector_only_result_forks_its_vectors(qtbot):
    dlg = _dialog(qtbot, labels=(), counts=(50, 3))
    assert not dlg._raster_box.isEnabled()
    c = dlg.choices()
    assert c["vectors"] is True and c["bands"] == []


def test_nothing_picked_or_no_name_disables_ok(qtbot):
    dlg = _dialog(qtbot, counts=(5, 0))
    dlg._bands.item(0).setCheckState(QtCore.Qt.Unchecked)
    assert not _ok(dlg).isEnabled()
    dlg._vector_box.setChecked(True)
    assert _ok(dlg).isEnabled()
    dlg._raster_box.setChecked(False)
    dlg._bands.item(0).setCheckState(QtCore.Qt.Checked)      # unchecked box: bands don't count
    dlg._vector_box.setChecked(False)
    assert not _ok(dlg).isEnabled()
    dlg._raster_box.setChecked(True)
    assert _ok(dlg).isEnabled()
    dlg._name.setText("   ")
    assert not _ok(dlg).isEnabled()
