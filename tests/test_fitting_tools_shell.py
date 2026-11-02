# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The main window's fitting tools over a result's sample sets: the two buttons follow the
Samples contract, each click opens that tool on the active result, every tool's fit is kept by
name, and a fit sent to the D(h) comparison is drawn in the construction window, where the methods
are compared."""
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
    assert lc.windowTitle() == "Log-cumulants — layer" and lc.current_fit()["name"] == "layer · WTMMM (chains; no sup), log-cumulants"
    win._on_ld_button_clicked()
    ldw = win._ld_window
    qtbot.addWidget(ldw)
    assert ldw.windowTitle() == "Large deviations — layer"
    assert [ldw.source_combo.itemText(i) for i in range(3)] == ["WTMMM (chains; no sup)", "WTMMM (chains w/sup)", "WTMM (complete h-chains) (experimental)"]
    assert ldw._log2_L == 7.0
    fit = ldw.current_fit()
    assert fit is not None and fit["name"] == "layer · WTMMM (chains; no sup), large deviations" and "q" in fit
    assert list(win._published_fits) == ["layer · WTMMM (chains; no sup), log-cumulants", fit["name"]]


def test_every_sample_set_chooser_says_what_each_set_holds(win, qtbot):
    """The Generalized, Large deviations, Log-cumulants and Structure functions choosers: each
    item's tooltip is its set's one-sentence description, and the chooser's own follows the set
    on show."""
    from PySide6 import QtCore
    from test_samples import _chains, _line

    from dynamix.model.samples import DESCRIPTIONS

    role = QtCore.Qt.ItemDataRole.ToolTipRole
    scales = np.array([2.0, 4.0, 8.0, 16.0])
    win._active_result = {"scales": scales, "chains": _chains(scales), "params": {}, "_shape": (128, 128),
                          "extrema": [_line([1.0, 3.0, 2.0, 2.5, 1.0, 0.5]) for _ in scales]}
    win._refresh_samples_buttons()
    win._on_ld_button_clicked()
    win._on_cumulants_button_clicked()
    win._on_generalized_button_clicked()
    choosers = [win._ld_window.source_combo, win._cumulants_window.set_combo, win._generalized_window.source_combo]
    win._active_result = _leaders_result()
    win._refresh_samples_buttons()
    win._on_structure_button_clicked()
    choosers.append(win._structure_window.set_combo)
    for w in (win._ld_window, win._cumulants_window, win._generalized_window, win._structure_window):
        qtbot.addWidget(w)
    for combo in choosers:
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert texts and all(combo.itemData(i, role) == DESCRIPTIONS[t] for i, t in enumerate(texts)), texts
        combo.setCurrentIndex(combo.count() - 1)
        assert combo.toolTip() == DESCRIPTIONS[texts[-1]]
    assert [choosers[0].itemText(i) for i in range(4)] == \
        ["WTMMM (chains; no sup)", "WTMMM (chains w/sup)", "WTMMM (raw; no chains)",
         "WTMM (complete h-chains) (experimental)"]
    assert "Z(q, a)" in choosers[0].itemData(0, role) and "unchained" in choosers[0].itemData(2, role)


def test_each_click_opens_the_tool_on_the_active_result_and_a_sent_fit_reaches_the_construction_window(win, qtbot):
    win._active_result = _leaders_result()
    win._refresh_samples_buttons()
    win._refresh_dh_button()
    win._on_structure_button_clicked()
    sf = win._structure_window
    qtbot.addWidget(sf)
    assert sf.windowTitle() == "Wavelet leader structure functions — layer · DWT sym4" and tuple(sf._region.getRegion()) == (2.0, 5.0)
    assert sf._log2_L == 6.0 and sf.fit_combo.isEnabled()                   # the field extent reaches the focus modes
    sf_name = "layer · DWT sym4 · leaders, WLSF"
    assert list(win._published_fits) == [sf_name]
    win._on_cumulants_button_clicked()
    lc = win._cumulants_window
    qtbot.addWidget(lc)
    assert lc.windowTitle() == "Log-cumulants — layer · DWT sym4" and lc.ranges() == [(2.0, 5.0)] * 4
    lc_name = "layer · DWT sym4 · leaders, log-cumulants"
    assert list(win._published_fits) == [sf_name, lc_name]                  # one entry per tool and set
    sf.set_combo.setCurrentText("WSE leaders")
    wse_name = "layer · DWT sym4 · WSE leaders, WSE-leader structure functions"
    assert list(win._published_fits) == [sf_name, lc_name, wse_name]
    sf.set_combo.setCurrentText("leaders")                                   # the same name: the latest wins
    assert list(win._published_fits) == [sf_name, lc_name, wse_name]
    win._on_dh_button_clicked()                                              # nothing arrives by itself
    cw = win._spectrum_construction_window
    qtbot.addWidget(cw)
    cw.show()
    assert cw._fit_items == {} and cw._board_legend == [] and not cw.show_button.isVisibleTo(cw)
    assert len(win.spectra_board) == 0                                       # published is not sent
    lc.send_button.click()                                                   # the tool's Send: drawn
    window = " · log2 a 2–5"
    assert [(n, full) for _c, n, full in cw._board_legend] == [("LC · WL (dwt) · a 2–5", lc_name + window)]
    lc.set_combo.setCurrentText("WSE leaders")                               # a new fit stays in its tool
    assert [full for _c, _n, full in cw._board_legend] == [lc_name + window]
    lc.send_button.click()                                                   # until it is sent too
    new_name = "layer · DWT sym4 · WSE leaders, log-cumulants"
    assert [(n, full) for _c, n, full in cw._board_legend] == \
        [("LC · WL (dwt) · a 2–5", lc_name + window), ("LC · wse-L (dwt) · a 2–5", new_name + window)]
    np.testing.assert_allclose(win.spectra_board.entries()[-1].tau, lc.current_fit()["tau"])
    sf.send_button.click()                                                   # another tool's Send: a third entry
    sf_tip = [tip for _c, _n, tip in cw._board_legend][-1]                  # the full name, then its bars
    assert sf_tip.splitlines()[:2] == [sf_name + window, "bars ±1σ: fit SE (propagated)"]
    win._on_dh_button_clicked()                                              # opened again: the board as it stands
    fresh = win._spectrum_construction_window
    qtbot.addWidget(fresh)
    assert fresh._fit_items == {}
    assert [tip.splitlines()[0] for _c, _n, tip in fresh._board_legend] == \
        [lc_name + window, new_name + window, sf_name + window]
    win._on_structure_button_clicked()                                       # a fresh window replaces the old
    assert win._structure_window is not sf


def test_a_leaders_result_opens_the_construction_window_with_the_gmwlp_label(win, qtbot):
    result = _leaders_result()
    result["record"]["method"] = "wavelet_leaders"
    win._active_result = result
    win._refresh_samples_buttons()
    win._refresh_dh_button()
    win._on_dh_button_clicked()
    cw = win._spectrum_construction_window
    qtbot.addWidget(cw)
    assert cw._hist_label.startswith("GMWLP histogram")


def test_a_landing_forgets_the_layer_s_published_fits_and_closes_its_stale_tools(win, qtbot):
    """A fit is published under the layer it was fitted on; when that layer lands anew the fit
    is forgotten and the tool windows opened on the previous result close, so a construction
    opened next shows the new result only."""
    win._active_result = _leaders_result()
    win._refresh_samples_buttons()
    win._on_structure_button_clicked()
    qtbot.addWidget(win._structure_window)
    name = "layer · DWT sym4 · leaders, WLSF"
    assert list(win._published_fits) == [name] and win._published_fits[name]["layer_id"] is None
    win._published_fits["other"] = {"name": "other", "layer_id": 7}
    win._forget_layer_tools(None)
    assert list(win._published_fits) == ["other"] and win._structure_window is None


def test_replaced_construction_stops_following_partition_range(win, qtbot, monkeypatch):
    from test_multifractal_window import _pair
    std, cmax = _pair()
    win._active_result = {**_leaders_result(), "hd_std": std, "hd_cmax": cmax}
    win._on_spectrum_button_clicked()
    partition = win._multifractal_window
    win._on_dh_button_clicked()
    old = win._spectrum_construction_window
    win._on_dh_button_clicked()
    fresh = win._spectrum_construction_window
    for window in (partition, old, fresh):
        qtbot.addWidget(window)
    calls = []
    original = old._recompute
    def counted():
        calls.append(True)
        original()
    monkeypatch.setattr(old, "_recompute", counted)
    partition.set_scale_window(2., 4.)
    assert calls == []
    assert (fresh._lo_spin.value(), fresh._hi_spin.value()) == (2., 4.)


def test_closed_partition_cannot_republish_an_old_table(win, qtbot):
    from test_multifractal_window import _pair
    std, cmax = _pair()
    win._active_result = {**_leaders_result(), "hd_std": std, "hd_cmax": cmax}
    win._on_spectrum_button_clicked()
    old = win._multifractal_window
    old_name = old.published_fit()["name"]
    win._on_spectrum_button_clicked()
    fresh = win._multifractal_window
    for window in (old, fresh):
        qtbot.addWidget(window)
    fresh._table_combo.setCurrentIndex(1 - fresh._table_combo.currentIndex())
    current_name = fresh.published_fit()["name"]
    assert current_name != old_name
    for lo in (1., 1.5, 2.):
        fresh.set_scale_window(lo, 4.)
        assert list(win._published_fits) == [current_name]


@pytest.mark.parametrize("opener, attr", [
    ("_on_structure_button_clicked", "_structure_window"),
    ("_on_cumulants_button_clicked", "_cumulants_window"),
    ("_on_ld_button_clicked", "_ld_window"),
    ("_on_generalized_button_clicked", "_generalized_window"),
    ("_on_spectrum_button_clicked", "_multifractal_window"),
    ("_on_dh_button_clicked", "_spectrum_construction_window"),
])
def test_replaced_fitting_widgets_are_released(win, opener, attr):
    import gc
    import weakref

    from PySide6 import QtCore
    from shiboken6 import isValid
    from test_multifractal_window import _pair
    std, cmax = _pair()
    win._active_result = {**_leaders_result(), "hd_std": std, "hd_cmax": cmax}
    getattr(win, opener)()
    previous = getattr(win, attr)
    gone = weakref.ref(previous)
    getattr(win, opener)()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert not isValid(previous)
    assert isValid(getattr(win, attr))
    del previous
    gc.collect()
    assert gone() is None                    # nothing outside the window holds its Python side


def test_a_sent_adaptive_large_deviation_fit_reaches_an_open_construction_window(win, qtbot):
    """The adaptive route publishes its tilts q beside h and D, and no τ: sent, the construction
    window draws its D(h) and leaves the τ(q) panel to the methods that have one."""
    from test_multifractal_window import _pair
    from test_samples import _chains

    std, cmax = _pair()
    scales = np.array([2.0, 4.0, 8.0, 16.0])
    extrema = [{"mod": np.array([1.0, 2.0, 3.0])} for _ in scales]
    win._active_result = {"scales": scales, "chains": _chains(scales), "extrema": extrema, "params": {},
                          "_shape": (128, 128), "hd_std": std, "hd_cmax": cmax}
    win._on_dh_button_clicked()
    cw = win._spectrum_construction_window
    qtbot.addWidget(cw)
    win._on_ld_button_clicked()
    qtbot.addWidget(win._ld_window)
    fit = win._ld_window.current_fit()
    assert fit.get("q") is not None and "tau" not in fit
    assert cw._fit_items == {} and cw._board_legend == [] and len(win.spectra_board) == 0   # published, not sent
    win._ld_window.send_button.click()                                       # the window's Send
    [(curve, name, full)] = cw._board_legend
    lo, hi = (round(float(v), 2) for v in fit["window"])
    assert full == f"{fit['name']} · log2 a {lo:g}–{hi:g}" and name.startswith("LD · WTMMM (chains; no sup) · a ")
    np.testing.assert_allclose(curve.getData()[1], fit["D"])
    assert not [i for p, i in cw._board_items if p is cw.method_plots["tau"]]
    [entry] = win.spectra_board.entries()
    assert entry.tau is None and entry.q is None                             # no τ(q) row to carry


def test_the_histogram_windows_count_only_the_full_chain_cells_of_a_leaders_h_map(win, qtbot):
    """The Partition functions and construction windows build the GMWLP histogram and the box
    counting over the cells whose chain spans every fitted level; the edge cells fitted over the
    finer levels alone are left out of both."""
    result = _leaders_result()
    result["record"]["method"] = "wavelet_leaders"
    levels = np.full((64, 64), 3.0, dtype=np.float32)
    levels[16:48, 16:48] = 4.0                                                # J − j_min + 1 = 4 levels
    result["levels_map"] = levels
    result["hd_std"] = result["hd_cmax"] = __import__("dynamix.core.leaders", fromlist=["x"]).structure_table(result["samples"], "leaders")
    result["hd_kind"] = "structure"
    win._active_result = result
    win._refresh_samples_buttons()
    win._refresh_dh_button()
    win._on_dh_button_clicked()
    cw = win._spectrum_construction_window
    qtbot.addWidget(cw)
    assert np.isfinite(cw._h_map).sum() == 32 * 32
    win._refresh_spectrum_button()
    win._on_spectrum_button_clicked()
    mw = win._multifractal_window
    qtbot.addWidget(mw)
    assert np.isfinite(mw._h_map).sum() == 32 * 32
