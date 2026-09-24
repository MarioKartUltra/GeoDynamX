# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ComponentsWindow: the grouping aids (thumbnails, shares, w-correlations) and the group it
picks -- the Group knob's text, both ways."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from dynamix.shell.components_window import THUMB_PX, ComponentsWindow, thumbnail


def _decomposition(n=6, ny=20, nx=24, L=4):
    rng = np.random.default_rng(1)
    comps = rng.standard_normal((n, ny, nx))
    shares = np.sort(rng.random(n))[::-1]
    shares /= shares.sum() * 1.05
    eig = rng.standard_normal((n, L, L))
    w = np.abs(rng.uniform(-0.3, 0.3, (n, n)))
    w = (w + w.T) / 2
    np.fill_diagonal(w, 1.0)
    return comps, shares, eig, w


def _window(qtbot, **kw):
    comps, shares, eig, w = _decomposition()
    win = ComponentsWindow(comps, shares, eigenarrays=eig, w_correlation=w, **kw)
    qtbot.addWidget(win)
    return win


def test_thumbnails_are_square_icons_and_survive_nan_and_constant_arrays(qtbot):
    a = np.arange(300.0).reshape(12, 25)
    a[3, 4] = np.nan
    pm = thumbnail(a)
    assert max(pm.width(), pm.height()) == THUMB_PX
    assert not thumbnail(np.zeros((5, 5))).isNull()
    assert not thumbnail(np.full((3, 3), np.nan)).isNull()
    assert max(thumbnail(np.ones((2000, 1500))).width(), 1) <= THUMB_PX


def test_the_window_shows_one_thumbnail_per_component_and_the_w_correlations(qtbot):
    win = _window(qtbot)
    assert win._list.count() == 6
    assert win._list.item(0).text().startswith("C1 · ")
    assert win._wplot.isVisible() and not win._wreading.isHidden()
    assert not win._kind_combo.isHidden()
    win._kind_combo.setCurrentIndex(1)                         # component images
    assert win._list.count() == 6


def test_without_w_correlations_or_eigenarrays_those_aids_hide(qtbot):
    comps, shares, _, _ = _decomposition()
    win = ComponentsWindow(comps, shares)
    qtbot.addWidget(win)
    assert win._kind_combo.isHidden()
    assert not win._wplot.isVisible() and win._wreading.isHidden()
    assert win._list.count() == 6


def test_set_group_picks_the_thumbnails_without_emitting(qtbot):
    win = _window(qtbot, group="1-2, 4")
    seen = []
    win.groupChanged.connect(seen.append)
    assert win._selected() == [0, 1, 3]
    win.set_group("all")
    assert win._selected() == list(range(6)) and win.group_text() == "all"
    win.set_group("5")
    assert win._selected() == [4] and seen == []


def test_picking_thumbnails_emits_the_group_text(qtbot):
    win = _window(qtbot)
    seen = []
    win.groupChanged.connect(seen.append)
    win._list.clearSelection()                                 # empty: not a group, no emit
    assert seen == []
    win._list.item(0).setSelected(True)
    assert seen[-1] == "1"
    win._list.item(2).setSelected(True)
    win._list.item(1).setSelected(True)
    assert seen[-1] == "1-3"
    for k in range(6):
        win._list.item(k).setSelected(True)
    assert seen[-1] == "all"
    assert "all 6" in win._group_label.text()


def test_the_all_button_takes_every_component(qtbot):
    win = _window(qtbot, group="2")
    seen = []
    win.groupChanged.connect(seen.append)
    qtbot.mouseClick(win._all_button, QtCore.Qt.LeftButton)
    assert seen == ["all"] and win._selected() == list(range(6))


def test_the_hint_says_when_the_group_is_not_on_screen(qtbot):
    win = _window(qtbot, show="component")
    assert "'component'" in win._hint.text()
    win.set_show("residual")
    assert "residual = the data minus it" in win._hint.text()


def test_reloading_a_decomposition_keeps_the_group(qtbot):
    win = _window(qtbot, group="2-3")
    comps, shares, eig, w = _decomposition(n=4)
    win.set_decomposition(comps, shares, eigenarrays=eig, w_correlation=w)
    assert win._list.count() == 4
    assert win._selected() == [1, 2]
    assert win.components is comps
