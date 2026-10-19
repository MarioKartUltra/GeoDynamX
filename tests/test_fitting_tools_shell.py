# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The main window's fitting tools over a result's sample sets: the two buttons follow the
Samples contract, each click opens that tool on the active result, every tool's fit is kept by
name and handed to the construction window, where the methods are compared."""
from __future__ import annotations

import numpy as np
import pytest

from test_structure_functions_window import _samples


@pytest.fixture
def win(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    window = MainWindow(steps=())
    qtbot.addWidget(window)
    return window


def _leaders_result():
    s = _samples()
    return {"samples": s, "scales": s["scales"], "h_map": np.full((64, 64), 0.5, dtype=np.float32),
            "params": {"wavelet": "sym4", "gamma": 0.0, "method": "leaders"},
            "record": {"levels": {"j_min": 2, "J": 5}}, "chains": [], "extrema": [], "_shape": (64, 64)}


def test_the_buttons_follow_the_samples_contract(win):
    assert not win._structure_button.isEnabled() and not win._cumulants_button.isEnabled()
    assert "Wavelet leaders" in win._structure_button.toolTip()
    win._active_result = _leaders_result()
    win._refresh_samples_buttons()
    assert win._structure_button.isEnabled() and win._cumulants_button.isEnabled()
    win._active_result = {"h_map": np.zeros((4, 4))}
    win._refresh_samples_buttons()
    assert not win._structure_button.isEnabled()


def test_each_click_opens_the_tool_on_the_active_result_and_its_fit_reaches_the_construction_window(win, qtbot):
    win._active_result = _leaders_result()
    win._refresh_samples_buttons()
    win._refresh_dh_button()
    win._on_structure_button_clicked()
    sf = win._structure_window
    qtbot.addWidget(sf)
    assert sf.windowTitle() == "Structure functions — layer · DWT sym4" and tuple(sf._region.getRegion()) == (2.0, 5.0)
    sf_name = "layer · DWT sym4 · leaders, structure functions"
    assert list(win._published_fits) == [sf_name]
    win._on_cumulants_button_clicked()
    lc = win._cumulants_window
    qtbot.addWidget(lc)
    assert lc.windowTitle() == "Log-cumulants — layer · DWT sym4" and lc.ranges() == [(2.0, 5.0)] * 4
    lc_name = "layer · DWT sym4 · leaders, log-cumulants"
    assert list(win._published_fits) == [sf_name, lc_name]                  # one entry per tool and set
    sf.set_combo.setCurrentText("WSE leaders")
    wse_name = "layer · DWT sym4 · WSE leaders, structure functions"
    assert list(win._published_fits) == [sf_name, lc_name, wse_name]
    sf.set_combo.setCurrentText("leaders")                                   # the same name: the latest wins
    assert list(win._published_fits) == [sf_name, lc_name, wse_name]
    win._on_dh_button_clicked()                                              # the construction window opens with them
    cw = win._spectrum_construction_window
    qtbot.addWidget(cw)
    assert set(cw._fit_items) == set(win._published_fits)
    cw.show()
    lc.set_combo.setCurrentText("WSE leaders")                               # a live window follows a new fit
    new_name = "layer · DWT sym4 · WSE leaders, log-cumulants"
    np.testing.assert_allclose(cw._fit_items[new_name]["tau"].getData()[1], lc.current_fit()["tau"])
    win._on_structure_button_clicked()                                       # a fresh window replaces the old
    assert win._structure_window is not sf
