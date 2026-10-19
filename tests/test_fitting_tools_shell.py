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
    assert win._structure_button.isEnabled() and win._cumulants_button.isEnabled() and win._ld_button.isEnabled()
    win._active_result = {"h_map": np.zeros((4, 4))}
    win._refresh_samples_buttons()
    assert not win._structure_button.isEnabled() and not win._cumulants_button.isEnabled()
    assert not win._ld_button.isEnabled()


def test_the_large_deviations_tool_opens_on_the_wtmm_s_maxima_too(win, qtbot):
    from test_samples import _chains

    scales = np.array([2.0, 4.0, 8.0, 16.0])
    extrema = [{"mod": np.array([1.0, 2.0, 3.0])} for _ in scales]
    win._active_result = {"scales": scales, "chains": _chains(scales), "extrema": extrema, "params": {},
                          "_shape": (128, 128)}
    win._refresh_samples_buttons()
    assert win._ld_button.isEnabled() and win._cumulants_button.isEnabled() and not win._structure_button.isEnabled()
    win._on_cumulants_button_clicked()
    lc = win._cumulants_window
    qtbot.addWidget(lc)
    assert lc.windowTitle() == "Log-cumulants — layer" and lc.current_fit()["name"] == "layer · maxima lines, log-cumulants"
    win._on_ld_button_clicked()
    ldw = win._ld_window
    qtbot.addWidget(ldw)
    assert ldw.windowTitle() == "Large deviations — layer"
    assert [ldw.source_combo.itemText(i) for i in range(2)] == ["maxima lines", "every maximum"]
    assert ldw._log2_L == 7.0
    fit = ldw.current_fit()
    assert fit is not None and fit["name"] == "layer · maxima lines, large deviations" and "q" in fit
    assert list(win._published_fits) == ["layer · maxima lines, log-cumulants", fit["name"]]


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
