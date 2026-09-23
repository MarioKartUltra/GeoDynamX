# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.profile_dialog.ProfileDialog -- pure Qt + matplotlib, offscreen (the
runner sets QT_QPA_PLATFORM=offscreen; matplotlib's QtAgg backend builds fine there, the same
precedent tests/test_skeleton_dialog.py already establishes for this shell)."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.transect import smooth_profile
from dynamix.shell.profile_dialog import ProfileDialog

DIST = np.linspace(0.0, 10.0, 21)
Z = np.sin(DIST) + np.linspace(0.0, 1.0, 21)          # something a smoothing pass visibly changes


def _dialog(qtbot, dist=DIST, z=Z, **kw):
    d = ProfileDialog(dist, z, **kw)
    qtbot.addWidget(d)
    return d


# --------------------------------------------------------------------------- construction


def test_dialog_builds_headless_with_both_curves(qtbot):
    d = _dialog(qtbot)
    np.testing.assert_allclose(d._raw_line.get_xdata(), DIST)
    np.testing.assert_allclose(d._raw_line.get_ydata(), Z)
    np.testing.assert_allclose(d._smooth_line.get_xdata(), DIST)
    # starts on "none" -- the smoothed curve equals the raw one until a kind is chosen
    np.testing.assert_allclose(d._smooth_line.get_ydata(), Z)


def test_title_names_the_label(qtbot):
    d = _dialog(qtbot, label="T3")
    assert "T3" in d.windowTitle()


def test_title_is_generic_with_no_label(qtbot):
    d = _dialog(qtbot)
    assert d.windowTitle() == "Transect profile"


def test_smoothing_combo_starts_at_none(qtbot):
    d = _dialog(qtbot)
    assert d._smoothing_combo.currentText() == "none"


def test_sigma_and_window_spin_defaults(qtbot):
    d = _dialog(qtbot)
    assert d._sigma_spin.value() == pytest.approx(3.0)
    assert d._sigma_spin.minimum() == pytest.approx(0.5)
    assert d._sigma_spin.maximum() == pytest.approx(50.0)
    assert d._window_spin.value() == 11
    assert d._window_spin.minimum() == 3
    assert d._window_spin.maximum() == 201


# --------------------------------------------------------------------------- live smoothing


def test_changing_smoothing_updates_the_smoothed_curve(qtbot):
    d = _dialog(qtbot)
    d._smoothing_combo.setCurrentText("gaussian")

    expected = smooth_profile(Z, "gaussian", d._sigma_spin.value(), d._window_spin.value())
    np.testing.assert_allclose(d._smooth_line.get_ydata(), expected)
    # the raw curve never changes
    np.testing.assert_allclose(d._raw_line.get_ydata(), Z)


def test_changing_sigma_redraws_the_smoothed_curve(qtbot):
    d = _dialog(qtbot)
    d._smoothing_combo.setCurrentText("gaussian")
    before = np.array(d._smooth_line.get_ydata())

    d._sigma_spin.setValue(10.0)

    after = np.array(d._smooth_line.get_ydata())
    assert not np.allclose(before, after)
    expected = smooth_profile(Z, "gaussian", 10.0, d._window_spin.value())
    np.testing.assert_allclose(after, expected)


def test_changing_window_redraws_the_smoothed_curve(qtbot):
    d = _dialog(qtbot)
    d._smoothing_combo.setCurrentText("median")

    d._window_spin.setValue(7)

    expected = smooth_profile(Z, "median", d._sigma_spin.value(), 7)
    np.testing.assert_allclose(d._smooth_line.get_ydata(), expected)


def test_window_spin_forces_odd_values(qtbot):
    d = _dialog(qtbot)
    d._window_spin.setValue(12)
    assert d._window_spin.value() == 13


# --------------------------------------------------------------------------- equal axes


def test_equal_axes_checkbox_sets_and_clears_aspect(qtbot):
    d = _dialog(qtbot)
    d._equal_axes_check.setChecked(True)
    assert d._ax.get_aspect() == 1.0

    d._equal_axes_check.setChecked(False)
    assert d._ax.get_aspect() == "auto"


# --------------------------------------------------------------------------- save


def test_save_png_writes_a_file(qtbot, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    d = _dialog(qtbot)
    out = tmp_path / "profile.png"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))

    d._on_save_png_clicked()

    assert out.exists() and out.stat().st_size > 0


def test_save_png_cancelled_writes_nothing(qtbot, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    d = _dialog(qtbot)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

    d._on_save_png_clicked()          # must not raise

    assert list(tmp_path.glob("*.png")) == []


def test_save_npz_writes_dist_z_and_the_current_smoothed_curve(qtbot, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    d = _dialog(qtbot)
    d._smoothing_combo.setCurrentText("gaussian")
    out = tmp_path / "profile.npz"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))

    d._on_save_npz_clicked()

    data = np.load(out)
    np.testing.assert_allclose(data["dist"], DIST)
    np.testing.assert_allclose(data["z"], Z)
    expected = smooth_profile(Z, "gaussian", d._sigma_spin.value(), d._window_spin.value())
    np.testing.assert_allclose(data["z_filt"], expected)
    assert data["smoothing"].item() == "gaussian"
