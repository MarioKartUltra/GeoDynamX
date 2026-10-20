# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.multifractal_window.MultifractalWindow -- the interactive canonical
(Arneodo WTMM) spectrum fitter over the hd partition tables.

Pure Qt + pyqtgraph, exercised entirely offscreen (``QT_QPA_PLATFORM=offscreen``) like
``test_skeleton_dialog.py``. The hd dicts are synthetic exact-line tables (the same ``_hd``
shape ``tests/test_spectra.py`` uses), so every fitted number the window shows has a closed-form
expectation -- and the window's ``current_fit`` is cross-checked against a direct
:func:`dynamix.core.spectra.fit_spectra` call, pinning the "engine truth" contract (the window
never fits anything the core function wouldn't).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import spectra
from dynamix.shell.multifractal_window import TABLE_CHOICES, MultifractalWindow


def _hd(slopes_tau, slopes_h, slopes_D, sigma=0.1, n_sc=8):
    """Exact-line tables over q = (-1, 0, 2); ``slopes_*[qi]`` is the log-log slope the fitter
    must recover -- same builder shape as tests/test_spectra.py's ``_hd``."""
    q = np.asarray([-1.0, 0.0, 2.0][: len(slopes_tau)], dtype=np.float64)
    log2_s = np.arange(n_sc, dtype=np.float64)

    def table(slopes, c):
        return np.asarray(slopes, dtype=np.float64)[:, None] * log2_s[None, :] + c

    sig = np.full((len(q), n_sc), float(sigma))
    return {
        "q_list": q, "scales": 2.0 ** log2_s, "log2_scales": log2_s,
        "N_a": np.full(n_sc, 100, dtype=np.int64),
        "tau_qa": table(slopes_tau, 1.0), "h_qa": table(slopes_h, -0.5),
        "D_qa": table(slopes_D, 2.0),
        "sigma_tau_qa": sig.copy(), "sigma_h_qa": sig.copy(), "sigma_D_qa": sig.copy(),
    }


def _pair():
    """(hd_std, hd_cmax) with DIFFERENT tau slopes so the table toggle is observable."""
    hd_std = _hd([1.0, 2.0, 3.0], [0.3, 0.5, 0.7], [2.0, 2.0, 1.4])
    hd_cmax = _hd([1.5, 2.5, 3.5], [0.4, 0.6, 0.8], [2.0, 1.9, 1.3])
    return hd_std, hd_cmax


# ------------------------------------------------------------------------------- construction

def test_window_builds_headless_and_is_a_nonmodal_canonical_window(qtbot):
    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    assert window.windowTitle() == "Partition functions — canonical (Arneodo WTMM)"
    assert window.isModal() is False
    # no-sup (std) is the default table: each line's modulus at the current scale, xsmurf's own
    # default, which keeps negative exponents (the sup flattens them toward 0).
    assert window._table_combo.currentText() == TABLE_CHOICES[0] == "no-sup (std)"


def test_region_defaults_to_the_full_scale_range(qtbot):
    hd_std, hd_cmax = _pair()
    window = MultifractalWindow(hd_std, hd_cmax)
    qtbot.addWidget(window)
    lo, hi = window._region.getRegion()
    assert lo == hd_cmax["log2_scales"].min()
    assert hi == hd_cmax["log2_scales"].max()


# ------------------------------------------------------------------------------- engine truth

def test_current_fit_is_exactly_the_core_fitter_on_the_selected_table(qtbot):
    hd_std, hd_cmax = _pair()
    window = MultifractalWindow(hd_std, hd_cmax)
    qtbot.addWidget(window)
    window._table_combo.setCurrentText("sup (cmax)")       # this test reads the sup table
    lo, hi = window._region.getRegion()
    expect = spectra.fit_spectra(hd_cmax, lo, hi)      # the sup table, picked above
    got = window.current_fit()
    for key in ("tau", "h", "D", "tau_err", "h_err", "D_err"):
        np.testing.assert_allclose(got[key], expect[key], rtol=1e-12, err_msg=key)


def test_region_change_refits_live(qtbot):
    """Piecewise-linear tau row (slope 1 early, slope 5 late): moving the region between the two
    halves changes the fitted slope accordingly -- the drag IS the re-fit."""
    hd = _hd([1.0], [0.5], [2.0])
    row = hd["tau_qa"][0].copy()
    row[4:] = row[3] + 5.0 * (hd["log2_scales"][4:] - hd["log2_scales"][3])
    hd["tau_qa"][0] = row
    window = MultifractalWindow(hd, hd)
    qtbot.addWidget(window)
    window.set_scale_window(-0.5, 3.5)
    np.testing.assert_allclose(window.current_fit()["tau"], [1.0], rtol=1e-12)
    window.set_scale_window(3.5, 7.5)
    np.testing.assert_allclose(window.current_fit()["tau"], [5.0], rtol=1e-12)


def test_table_toggle_switches_between_cmax_and_std(qtbot):
    hd_std, hd_cmax = _pair()
    window = MultifractalWindow(hd_std, hd_cmax)
    qtbot.addWidget(window)
    np.testing.assert_allclose(window.current_fit()["tau"], [1.0, 2.0, 3.0], rtol=1e-12)
    window._table_combo.setCurrentText("sup (cmax)")
    np.testing.assert_allclose(window.current_fit()["tau"], [1.5, 2.5, 3.5], rtol=1e-12)


# ------------------------------------------------------------------------------- eta / frame

def test_eta_seed_flows_into_the_control_and_the_fit(qtbot):
    """The caller-derived seed lands in the spin AND the fit; default frame 'original' applies
    the shift-back (tau - eta*q, h - eta), and the forward note is displayed verbatim."""
    hd_std, hd_cmax = _pair()
    note = "forward lift: a^1 on WT derivs (tensor)"
    window = MultifractalWindow(hd_std, hd_cmax, eta_seed=1.0, forward_note=note)
    qtbot.addWidget(window)
    window._table_combo.setCurrentText("sup (cmax)")       # this test reads the sup table
    assert window._eta_spin.value() == 1.0
    assert window._forward_label.text() == note
    fit = window.current_fit()
    assert fit["eta"] == 1.0 and fit["frame"] == "original"
    q = hd_cmax["q_list"]
    np.testing.assert_allclose(fit["tau"], np.asarray([1.5, 2.5, 3.5]) - 1.0 * q, rtol=1e-12)
    window._frame_combo.setCurrentText("integrated")
    np.testing.assert_allclose(window.current_fit()["tau"], [1.5, 2.5, 3.5], rtol=1e-12)


# ------------------------------------------------------------------------------- q subrange

def test_q_subrange_masks_the_display_but_not_the_fit(qtbot):
    hd_std, hd_cmax = _pair()
    window = MultifractalWindow(hd_std, hd_cmax)
    qtbot.addWidget(window)
    assert [it.isVisible() for it in window._family_items] == [True, True, True]
    window._qmin_spin.setValue(0.0)          # drop q = -1
    assert [it.isVisible() for it in window._family_items] == [False, True, True]
    # the fit stays full-q (engine truth -- module docstring)
    assert window.current_fit()["tau"].shape == (3,)
    # displayed tau curve holds only the masked points
    x, _y = window._spectrum_curves["tau"].getData()
    np.testing.assert_array_equal(x, [0.0, 2.0])


# --------------------------------------- three families + phase transition

def _hd_q(q, slopes_tau, n_sc=8):
    """Exact-line tables over an arbitrary q grid; tau slope per q is ``slopes_tau``."""
    q = np.asarray(q, dtype=np.float64)
    log2_s = np.arange(n_sc, dtype=np.float64)
    slopes = np.asarray(slopes_tau, dtype=np.float64)
    sig = np.full((q.size, n_sc), 0.1)
    table = slopes[:, None] * log2_s[None, :]
    return {"q_list": q, "scales": 2.0 ** log2_s, "log2_scales": log2_s,
            "N_a": np.full(n_sc, 100, dtype=np.int64),
            "tau_qa": table, "h_qa": table * 0.3, "D_qa": table * 0.1,
            "sigma_tau_qa": sig.copy(), "sigma_h_qa": sig.copy(), "sigma_D_qa": sig.copy()}


def test_all_three_partition_families_are_drawn_with_mirrored_regions(qtbot):
    from dynamix.shell.multifractal_window import FAMILY_TABLES

    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    window._table_combo.setCurrentText("sup (cmax)")       # this test reads the sup table
    assert [k for k, _ in FAMILY_TABLES] == ["tau_qa", "h_qa", "D_qa"]
    for key, _ in FAMILY_TABLES:
        assert len(window._family_items_by_table[key]) == 3      # one item per q
    # dragging ANY panel's region moves them all and re-fits
    window._regions[1].setRegion((1.0, 5.0))
    for region in window._regions:
        lo, hi = region.getRegion()
        assert (lo, hi) == (1.0, 5.0)
    expect = spectra.fit_spectra(_pair()[1], 1.0, 5.0)
    np.testing.assert_allclose(window.current_fit()["tau"], expect["tau"], rtol=1e-12)


def test_phase_transition_readout_matches_the_core_fitter(qtbot):
    """tau(q) linear in ln q with a slope break at q* = 2 (slopes 2 -> 5): the window's
    two-segment fit and scan must be exactly the core functions' numbers."""
    q = np.concatenate([[-1.0, 0.0], np.geomspace(0.25, 16.0, 25)])
    lnq = np.log(np.maximum(q, 1e-9))
    ln2 = np.log(2.0)
    slopes = np.where(q <= 0, 0.0, np.where(lnq <= ln2, 2.0 * lnq,
                                            2.0 * ln2 + 5.0 * (lnq - ln2)))
    hd = _hd_q(q, slopes)
    window = MultifractalWindow(hd, hd)
    qtbot.addWidget(window)
    window._qstar_spin.setValue(2.0)
    fit = window.current_fit()
    expect = spectra.phase_transition_fit(fit["q_list"], fit["tau"], 2.0)
    np.testing.assert_allclose(window._pt_fit["slope_L"], expect["slope_L"], rtol=1e-9)
    np.testing.assert_allclose(window._pt_fit["slope_R"], expect["slope_R"], rtol=1e-9)
    np.testing.assert_allclose(window._pt_fit["slope_L"], 2.0, atol=1e-9)
    np.testing.assert_allclose(window._pt_fit["slope_R"], 5.0, atol=1e-9)
    assert abs(window._pt_scan["best_q"] - 2.0) < 0.5
    assert "Δs=" in window._pt_label.text()


def _broken_pair():
    """cmax loses concavity on the negative branch below q = -3; std stays concave."""
    q = np.arange(-5.0, 5.01, 0.5)
    concave = 0.6 * q - 0.1 * q * q
    broken = concave.copy()
    bad = q < -3.0
    broken[bad] = concave[q == -3.0][0] + (0.6 + 0.6) * (q[bad] + 3.0) \
        + 0.3 * (q[bad] + 3.0) ** 2
    return _hd_q(q, concave), _hd_q(q, broken)


def test_the_phase_transition_tangent_on_D_h_tracks_the_negative_branch(qtbot):
    window = MultifractalWindow(*_broken_pair())
    qtbot.addWidget(window)
    window._table_combo.setCurrentText("sup (cmax)")       # this test reads the sup table
    fit = window.current_fit()                                      # cmax is broken
    expect = spectra.phase_transition_tangent(fit["q_list"], fit["tau"], fit["h"], fit["D"])
    assert expect is not None
    line = window._tangent_line
    assert line.isVisible()
    x, y = line.xData, line.yData
    view_x = window._spectrum_plots["D"].getViewBox().viewRange()[0]
    assert list(x) == pytest.approx(view_x)                          # across the whole view
    assert (y[-1] - y[0]) / (x[-1] - x[0]) == pytest.approx(expect["q_c"])  # slope q_c
    assert y[0] + expect["q_c"] * (0.0 - x[0]) == pytest.approx(expect["intercept"])  # through (0, intercept)
    assert f"{expect['intercept']:.3f}" in window._tangent_label.textItem.toPlainText()
    window._spectrum_plots["D"].setXRange(-1.0, 3.0, padding=0.0)   # the view moves: so does the line
    assert list(window._tangent_line.xData) == pytest.approx([-1.0, 3.0])
    assert not hasattr(window, "_convexity_line")                   # the tau(q) marker is gone
    window._table_combo.setCurrentText("no-sup (std)")
    assert not line.isVisible()                                     # std is clean


def test_the_D_h_legend_is_a_row_under_the_panel_with_short_names(qtbot):
    hd_std, hd_cmax = _pair()
    long_name = "moments over the projections (Turiel 2006, M method) · holder_measure @A"
    window = MultifractalWindow(hd_std, hd_cmax,
                                sources=[{"name": long_name, "hd_std": hd_std, "hd_cmax": hd_cmax}])
    qtbot.addWidget(window)
    assert window._glw.ci.rows[2][0] is window._dh_legend_row     # the strip under the spectra
    names = window._dh_legend_names()
    assert "holder_measure @A" in names and long_name not in names
    assert "canonical (Arneodo WTMM)" in names
    assert "c(q) fit" in names                                     # drawn after the fit
    registered = {name_of() for _item, name_of in window._dh_legend_entries}
    assert {"c(q) fit", "cumulant expansion", "log-normal c₂q²"} <= registered
    assert not window._cq_plot.titleLabel.isVisible()               # the caption moved here


def test_axes_read_plain_decimals_and_the_physical_axis_ticks_once_per_octave(qtbot):
    from dynamix.shell.multifractal_window import PhysicalAxis

    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    assert window._spectrum_plots["D"].getAxis("bottom").autoSIPrefix is False
    assert window._spectrum_plots["h"].getAxis("left").autoSIPrefix is False
    axis = PhysicalAxis(lambda v: (2.0 ** v, "px"))
    assert axis.tickValues(2.3, 5.6, 600) == [(1.0, [3.0, 4.0, 5.0])]
    layout = window._glw.ci.layout
    assert (layout.rowStretchFactor(0), layout.rowStretchFactor(1)) == (3, 0)


def test_scale_window_changes_are_broadcast(qtbot):
    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    seen = []
    window.scaleWindowChanged.connect(lambda lo, hi: seen.append((lo, hi)))
    window.set_scale_window(2.0, 6.0)
    assert seen and seen[-1] == (2.0, 6.0)


def test_a_physical_top_axis_and_the_range_readout(qtbot):
    hd_std, hd_cmax = _pair()
    window = MultifractalWindow(hd_std, hd_cmax,
                                physical_of_log2a=lambda v: (2.0 ** v * 14.1, "m"))
    qtbot.addWidget(window)
    axis = window._family_plots["tau_qa"].getAxis("top")
    assert axis.isVisible() and axis.tickStrings([3.0], 1.0, 1.0) == ["113 m"]
    window.set_scale_window(2.0, 4.0)
    assert "56.4 m" in window.range_label.text() and "226 m" in window.range_label.text()


def test_the_fitter_owns_the_fit_mode_and_the_construction_window_follows(qtbot):
    from dynamix.shell.spectrum_window import SpectrumWindow

    hd_std, hd_cmax = _pair()
    fitter = MultifractalWindow(hd_std, hd_cmax, log2_L=8.0)
    qtbot.addWidget(fitter)
    assert fitter._fitmode_combo.isEnabled()
    assert [fitter._fitmode_combo.itemText(i) for i in range(3)] == ["naive", "fixed focus", "focus"]
    construction = SpectrumWindow(hd_std, hd_cmax, log2_L=8.0)
    qtbot.addWidget(construction)
    fitter.fitModeChanged.connect(construction.set_fit_mode)
    with qtbot.waitSignal(fitter.fitModeChanged) as sig:
        fitter._fitmode_combo.setCurrentText("focus")
    assert sig.args == ["focus"]
    assert fitter.current_fit().get("focus") is not None           # the fitter fits through the focus
    assert construction._fitmode_combo.currentText() == "focus"
    blind = MultifractalWindow(hd_std, hd_cmax)                     # no extent: no focus modes
    qtbot.addWidget(blind)
    assert not blind._fitmode_combo.isEnabled()


def test_filtered_tables_carry_their_note_and_a_switch_to_all_chains(qtbot):
    hd_std, hd_cmax = _pair()
    all_std, all_cmax = _pair()
    w = MultifractalWindow(hd_std, hd_cmax, all_tables=(all_std, all_cmax),
                           filter_note="filtered: 12 of 40 chains")
    qtbot.addWidget(w)
    assert "filtered: 12 of 40 chains" in w.windowTitle()
    assert w.all_chains_check.isVisible() or not w.all_chains_check.isChecked()
    assert w._current_hd() is hd_cmax or w._current_hd() is hd_std
    w.all_chains_check.setChecked(True)
    assert w._current_hd() is all_cmax or w._current_hd() is all_std
    plain = MultifractalWindow(hd_std, hd_cmax)
    qtbot.addWidget(plain)
    assert "filtered" not in plain.windowTitle() and plain.all_chains_check is None


def test_the_specific_heat_panel_follows_the_fit_and_the_cascade_cumulants(qtbot):
    """A log-normal tau(q) = c1 q - c2 q^2/2 has c(q) = c2 q^2; the cumulant curve comes from
    the maxima moduli (ln|T| ~ N(c1 ln a, v0 - c2 ln a)) inside the fit window, with the
    log-normal reference c2 q^2 beside it; without moduli only the fit's own curve draws."""
    q = np.linspace(-3.0, 3.0, 25)
    hd = _hd_q(q, 0.7 * q - 0.3 * q ** 2 / 2)
    rng = np.random.default_rng(3)
    scales = hd["scales"]
    mods = [np.exp(rng.normal(0.7 * np.log(a), np.sqrt(3.0 - 0.3 * np.log(a)), 20000))
            for a in scales]
    window = MultifractalWindow(hd, hd, extrema_mods=mods, scales=scales)
    qtbot.addWidget(window)
    x, y = window._cq_curve.getData()
    np.testing.assert_allclose(y, 0.3 * x ** 2, atol=1e-6)
    assert window.current_cumulants() is None
    window.cumulant_check.setChecked(True)
    c = window.current_cumulants()
    np.testing.assert_allclose(c["c1"], 0.7, atol=0.03)
    np.testing.assert_allclose(c["c2"], 0.3, atol=0.03)
    xl, yl = window._cq_ln_curve.getData()
    np.testing.assert_allclose(yl, c["c2"] * xl ** 2)
    xc, yc = window._cq_cum_curve.getData()
    np.testing.assert_allclose(yc, spectra.cumulant_heat(c, xc))
    assert "c₂" in window._cq_label.text()
    window._region.setRegion((0.0, 0.5))                       # one scale inside: no fit
    assert np.isnan(window.current_cumulants()["c2"])
    bare = MultifractalWindow(hd, hd)
    qtbot.addWidget(bare)
    assert bare.current_cumulants() is None
    xb, _ = bare._cq_cum_curve.getData()
    assert xb is None or len(xb) == 0


# ------------------------------------------------------------------------------- the grid layout

def test_the_layout_is_partition_functions_over_a_row_of_four_square_spectra_with_bershadskii_optional(qtbot):
    """Three families on top, the four spectra τ(q), D(h), h(q), c(q) in one row of squares below
    (the row's height follows a quarter of the grid's width), the Bershadskii row optional; D(h)
    opens at h ∈ [−2, 2], D ∈ [−½, 1½]; the error bars are whiskers without caps."""
    from dynamix.shell.multifractal_window import DH_RANGE

    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)                                   # the deferred squaring
    ci = window._glw.ci
    assert ci.rows[0][0] is window._family_plots["tau_qa"] and ci.rows[0][8] is window._family_plots["D_qa"]
    assert [ci.rows[1][c] for c in (0, 3, 6, 9)] == [window._spectrum_plots["tau"], window._spectrum_plots["D"],
                                                    window._spectrum_plots["h"], window._cq_plot]
    assert [ci.layout.columnStretchFactor(c) for c in range(12)] == [1] * 12
    assert (ci.layout.rowStretchFactor(0), ci.layout.rowStretchFactor(1)) == (3, 0)
    sizes = [window._spectrum_plots[k].size() for k in ("tau", "D", "h")] + [window._cq_plot.size()]
    assert all(abs(sz.width() - sz.height()) <= 0.1 * sz.width() for sz in sizes), [(sz.width(), sz.height()) for sz in sizes]
    assert all(abs(sz.width() - sizes[0].width()) <= 2 for sz in sizes)
    vb = window._spectrum_plots["D"].getViewBox()
    (x0, x1), (y0, y1) = vb.viewRange()
    assert (x0, x1) == pytest.approx(DH_RANGE[0]) and (y0, y1) == pytest.approx(DH_RANGE[1])
    assert all(err.opts["beam"] == 0.0 for err in window._spectrum_errbars.values())
    assert not window.bershadskii_check.isChecked()
    assert window._pt_plot not in ci.items and not window._pt_label.isVisible()
    assert ci.rows[2][0] is window._dh_legend_row                # the legend strip
    window.bershadskii_check.setChecked(True)
    assert ci.rows[3][0] is window._pt_plot and ci.rows[3][8] is window._pt_scan_plot
    assert ci.layout.rowStretchFactor(0) == 3
    window.bershadskii_check.setChecked(False)
    assert window._pt_plot not in ci.items and ci.layout.rowStretchFactor(0) == 3


def test_the_wheel_zooms_x_and_with_shift_y(qtbot):
    from PySide6 import QtCore

    class _Wheel:
        def __init__(self, mods):
            self._mods = mods

        def delta(self):
            return 120

        def modifiers(self):
            return self._mods

        def pos(self):
            return QtCore.QPointF(0.0, 0.0)

        def accept(self):
            pass

    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    vb = window._spectrum_plots["tau"].getViewBox()
    vb.setRange(xRange=(-3.0, 3.0), yRange=(-4.0, 1.0), padding=0)   # a settled view
    (x0, x1), (y0, y1) = vb.viewRange()
    vb.wheelEvent(_Wheel(QtCore.Qt.NoModifier))
    (x0b, x1b), (y0b, y1b) = vb.viewRange()
    assert (x0b, x1b) != (x0, x1) and (y0b, y1b) == (y0, y1)
    vb.wheelEvent(_Wheel(QtCore.Qt.ShiftModifier))
    (x0c, x1c), (y0c, y1c) = vb.viewRange()
    assert (x0c, x1c) == (x0b, x1b) and (y0c, y1c) != (y0b, y1b)


def test_fit_lines_lie_on_the_core_lines_in_naive_mode(qtbot):
    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    lo, hi = window._region.getRegion()
    lines = spectra.partition_fit_lines(window._current_hd(), float(lo), float(hi))
    for key in ("tau_qa", "h_qa", "D_qa"):
        items = window._family_lines_by_table[key]
        assert len(items) == 3
        for i, item in enumerate(items):
            x, y = item.getData()
            expect = lines["lines"][key]["slope"][i] * x + lines["lines"][key]["intercept"][i]
            np.testing.assert_allclose(y, expect, atol=1e-9)
    assert not window._focus_markers["tau_qa"].isVisible()


def test_a_focus_mode_shows_the_power_means_their_lines_meeting_at_the_marked_focus(qtbot):
    from test_spectra import _focal_hd

    hd, h_q, q = _focal_hd(x0=9.0, log2_N=6.0)
    window = MultifractalWindow(hd, hd, log2_L=9.0)
    qtbot.addWidget(window)
    window._fitmode_combo.setCurrentText("fixed focus")
    lines = spectra.partition_fit_lines(hd, *map(float, window._region.getRegion()),
                                        fit_mode="fixed_focus", log2_L=9.0)
    x0, y0 = lines["focus"]
    nz = np.flatnonzero(q != 0)
    rows = window._family_items_by_table["tau_qa"]
    np.testing.assert_allclose(rows[nz[0]].yData, (hd["tau_qa"][nz[0]] - 6.0) / q[nz[0]], atol=1e-12)
    assert not rows[int(np.flatnonzero(q == 0)[0])].isVisible()          # no power mean at q = 0
    for j, i in enumerate(nz):
        x, y = window._family_lines_by_table["tau_qa"][i].getData()
        assert x[-1] == pytest.approx(x0)
        assert y[-1] == pytest.approx(y0, abs=1e-8)
    assert all(window._focus_markers[k].isVisible() for k in ("tau_qa", "h_qa", "D_qa"))
    fx, fy = window._focus_markers["tau_qa"].getData()
    assert (fx[0], fy[0]) == pytest.approx((x0, y0))
    assert "/q" in window._family_plots["tau_qa"].getAxis("left").labelText
    assert window._L_lines["tau_qa"].isVisible() and window._L_lines["tau_qa"].value() == 9.0
    window._fitmode_combo.setCurrentText("naive")
    rows = window._family_items_by_table["tau_qa"]                      # rebuilt with the mode
    np.testing.assert_allclose(rows[nz[0]].yData, hd["tau_qa"][nz[0]], atol=1e-12)
    assert not window._focus_markers["tau_qa"].isVisible() and not window._L_lines["tau_qa"].isVisible()


def test_the_histogram_construction_rides_on_the_dh_panel_and_the_source_titles(qtbot):
    h = np.linspace(0.0, 1.0, 20)
    D = 2.0 - 8.0 * (h - 0.5) ** 2
    dD = np.full(20, 0.05)
    window = MultifractalWindow(*_pair(), dh_hist=(h, D, dD), source="moments over the projections")
    qtbot.addWidget(window)
    x, y = window._dh_hist_curve.getData()
    np.testing.assert_allclose(x, h)
    np.testing.assert_allclose(y, D)
    assert "moments over the projections" in window.windowTitle()
    bare = MultifractalWindow(*_pair())
    qtbot.addWidget(bare)
    xb, _ = bare._dh_hist_curve.getData()
    assert xb is None or xb.size == 0
    assert "canonical" in bare.windowTitle()


def test_the_family_panels_can_read_log2_of_the_physical_length_on_the_bottom_axis(qtbot):
    window = MultifractalWindow(*_pair(), physical_of_log2a=lambda v: (2.0 ** v * 3.0, "km"))
    qtbot.addWidget(window)
    plot = window._family_plots["tau_qa"]
    assert window._xaxis_combo.currentText() == "log₂ a (px)"
    assert plot.getAxis("bottom").tickStrings([1.0], 1.0, 1.0) == ["1"]
    window._xaxis_combo.setCurrentText("log₂ λ* (unit)")
    axis = plot.getAxis("bottom")
    assert axis.tickStrings([1.0], 1.0, 1.0) == [f"{np.log2(6.0):.2f}"]
    assert "km" in axis.labelText
    window._xaxis_combo.setCurrentText("log₂ a (px)")
    assert plot.getAxis("bottom").tickStrings([1.0], 1.0, 1.0) == ["1"]
    window.refresh_physical_axes()                                   # a units change re-reads


def test_other_methods_ride_as_sources_with_a_show_menu(qtbot):
    """Every other method's tables draw beside the primary's -- their rows on the family
    panels, their fitted tau/D/h under the same window and fit mode -- and the Show menu
    toggles each source, the primary included."""
    hd_a, hd_b = _pair()
    window = MultifractalWindow(hd_a, hd_a,
                                sources=[{"name": "moments · B", "hd_std": hd_b, "hd_cmax": hd_b}])
    qtbot.addWidget(window)
    assert [a.text() for a in window.show_menu.actions()] == ["canonical (Arneodo WTMM)", "moments · B"]
    assert all(a.isChecked() for a in window.show_menu.actions())
    items = window._source_items["moments · B"]
    rows = items["rows"]["tau_qa"]
    assert len(rows) == 3 and all(r.isVisible() for r in rows)
    np.testing.assert_allclose(rows[1].yData, hd_b["tau_qa"][1])
    lo, hi = window._region.getRegion()
    fit = spectra.fit_spectra(hd_b, float(lo), float(hi))
    x, y = items["curves"]["tau"].getData()
    np.testing.assert_allclose(x, fit["q_list"])
    np.testing.assert_allclose(y, fit["tau"])
    xd, yd = items["curves"]["D"].getData()
    np.testing.assert_allclose(xd, fit["h"])
    np.testing.assert_allclose(yd, fit["D"])
    window._region.setRegion((1.0, 5.0))
    np.testing.assert_allclose(items["curves"]["tau"].getData()[1],
                               spectra.fit_spectra(hd_b, 1.0, 5.0)["tau"])
    window.show_menu.actions()[1].setChecked(False)
    assert not items["curves"]["tau"].isVisible() and not rows[0].isVisible()
    window.show_menu.actions()[0].setChecked(False)
    assert not window._spectrum_curves["tau"].isVisible()
    assert not window._family_items_by_table["tau_qa"][0].isVisible()
    window.show_menu.actions()[0].setChecked(True)
    assert window._spectrum_curves["tau"].isVisible()
    assert window._family_items_by_table["tau_qa"][0].isVisible()
    hist_only = MultifractalWindow(hd_a, hd_a, sources=[
        {"name": "histogram · C", "dh_hist": (np.linspace(0, 1, 5), np.full(5, 1.5), np.zeros(5))}])
    qtbot.addWidget(hist_only)
    assert "histogram · C" in hist_only._source_items
    assert hist_only._source_items["histogram · C"]["curves"]["dh"].getData()[0].size == 5


def test_the_family_rows_are_markers_and_the_fit_lines_solid(qtbot):
    """The EBSD figures' look: the data as points, the per-q best-fit line the only line, so
    the fit is visible over real data."""
    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    row = window._family_items_by_table["tau_qa"][1]
    assert row.opts["symbol"] == "o" and row.opts["pen"] is None
    line = window._family_lines_by_table["tau_qa"][1]
    assert line.opts["pen"].color().alpha() == 255 and line.opts["pen"].widthF() >= 1.2


def test_the_box_counting_curve_and_the_pin_to_d0_ride_on_the_fitters_dh_panel(qtbot):
    h = np.linspace(0.0, 1.0, 20)
    D = 2.0 - 8.0 * (h - 0.5) ** 2
    dD = np.full(20, 0.05)
    box = (h, D - 0.3, dD)
    window = MultifractalWindow(*_pair(), dh_hist=(h, D, dD), dh_box=box)
    qtbot.addWidget(window)
    np.testing.assert_allclose(window._dh_box_curve.getData()[1], D - 0.3)
    assert not window.pin_check.isChecked()
    window.pin_check.setChecked(True)
    fit = window.current_fit()
    d0 = -float(np.asarray(fit["tau"])[list(fit["q_list"]).index(0.0)])
    np.testing.assert_allclose(window._dh_hist_curve.getData()[1], D + (d0 - 2.0))
    window._region.setRegion((1.0, 5.0))                        # the pin follows the fit
    fit2 = window.current_fit()
    d0b = -float(np.asarray(fit2["tau"])[list(fit2["q_list"]).index(0.0)])
    np.testing.assert_allclose(window._dh_hist_curve.getData()[1], D + (d0b - 2.0))
    window.pin_check.setChecked(False)
    np.testing.assert_allclose(window._dh_hist_curve.getData()[1], D)


def _names(window):
    return window._dh_legend_names()


def test_the_dh_panel_carries_the_legend_and_the_histogram_and_box_counting_as_points(qtbot):
    rng = np.random.default_rng(2)
    h_map = rng.normal(0.4, 0.1, (64, 64))
    h = np.linspace(0.0, 1.0, 20)
    D = 2.0 - 8.0 * (h - 0.5) ** 2
    window = MultifractalWindow(*_pair(), dh_hist=(h, D, np.full(20, 0.05)), h_map=h_map,
                                sources=[{"name": "moments · B", "hd_std": _pair()[1], "hd_cmax": _pair()[1]}])
    qtbot.addWidget(window)
    names = _names(window)
    assert names[0] == "canonical (Arneodo WTMM)" and "microcanonical histogram" in names
    assert "B" in names and "box-counting across r" not in names       # the layer part of "moments · B"
    assert window._dh_hist_curve.opts["symbol"] == "o" and window._dh_hist_curve.opts["pen"] is None
    assert not window.boxcount_check.isChecked()
    window.boxcount_check.setChecked(True)
    assert window._dh_box_curve.getData()[0].size > 0 and "box-counting across r" in _names(window)
    window.show_menu.actions()[1].setChecked(False)
    assert "B" not in _names(window)
    assert getattr(window._cq_plot, "legend", None) is None or len(window._cq_plot.legend.items) == 0


def test_the_cumulant_spectrum_overlay_follows_the_fitted_cumulants(qtbot):
    from test_spectra import _focal_hd

    hd, h_q, q = _focal_hd(x0=9.0, log2_N=6.0)
    rng = np.random.default_rng(3)
    scales = hd["scales"]
    mods = [np.exp(rng.normal(0.7 * np.log(a), np.sqrt(3.0 - 0.3 * np.log(a)), 20000)) for a in scales]
    window = MultifractalWindow(hd, hd, extrema_mods=mods, scales=scales)
    qtbot.addWidget(window)
    assert not window.cumulant_check.isChecked() and window._cum_curves["tau"].getData()[0] is None
    window.cumulant_check.setChecked(True)
    c = window.current_cumulants()
    fit = window.current_fit()
    i0 = list(fit["q_list"]).index(0.0)
    c0 = -float(np.asarray(fit["tau"])[i0])
    xq, tau_c = window._cum_curves["tau"].getData()
    np.testing.assert_allclose(tau_c, spectra.cumulant_tau(c, xq, c0=c0))
    xh, h_c = window._cum_curves["h"].getData()
    np.testing.assert_allclose(h_c, np.gradient(spectra.cumulant_tau(c, xh, c0=c0), xh, edge_order=2), atol=1e-6)
    xd, D_c = window._cum_curves["D"].getData()
    np.testing.assert_allclose(D_c, xq * h_c - tau_c, atol=1e-6)
    assert "cumulant spectrum" in _names(window)
    # the moment route brings its cumulants in the table
    hd2 = dict(hd); hd2["lnT_moments"] = spectra.ln_moments(mods)
    table_window = MultifractalWindow(hd2, hd2, scales=scales)
    qtbot.addWidget(table_window)
    assert table_window.current_cumulants() is None
    table_window.cumulant_check.setChecked(True)
    assert table_window.current_cumulants()["c2"] == pytest.approx(c["c2"], rel=1e-9)


# ------------------------------------------------------------------ the window never fights the screen

def _width_report(window):
    """The style, the font and the widest controls beside a width assertion: a control's
    minimum width is what a row adds up."""
    from PySide6 import QtWidgets

    from tests.test_shell_docks import _font_report

    widest = sorted(((w.minimumSizeHint().width(), type(w).__name__, w.objectName())
                     for w in window.findChildren(QtWidgets.QWidget) if w.isVisibleTo(window)),
                    reverse=True)[:8]
    return f"{_font_report()}, widest {widest}"


def test_the_control_strip_never_sets_the_window_width(qtbot):
    """The strip's labels change text with every fit (the focus readout, the two-segment
    readout); a one-row strip let them raise the window's minimum width past the screen and
    Qt widened the window on its own. The strip wraps, the readouts clip, and a window the
    user shrank stays shrunk through fit-mode switches and region drags."""
    hd_std, hd_cmax = _pair()
    window = MultifractalWindow(hd_std, hd_cmax, log2_L=9.0)
    qtbot.addWidget(window)
    window.show()
    # The bound is a small laptop's usable width, not a pixel count: the strip's widgets grow
    # with the app theme's font (874 px with it, 760 without), and the one-row strip was 1361.
    assert window.minimumSizeHint().width() < 1000, _width_report(window)
    window.resize(1000, 700)
    qtbot.wait(10)
    assert window.size().width() == 1000
    for mode in ("fixed focus", "focus", "naive"):
        window._fitmode_combo.setCurrentText(mode)
        qtbot.wait(10)
        assert (window.size().width(), window.size().height()) == (1000, 700), mode
    lo, hi = window._region.getRegion()
    window._region.setRegion((lo + 0.3 * (hi - lo), hi - 0.2 * (hi - lo)))
    window.bershadskii_check.setChecked(True)
    window.cumulant_check.setChecked(True)
    qtbot.wait(10)
    assert (window.size().width(), window.size().height()) == (1000, 700)
    assert window.minimumSizeHint().width() < 1000


def test_the_fitter_opens_inside_its_screen(qtbot):
    from PySide6 import QtGui

    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(10)
    avail = (window.screen() or QtGui.QGuiApplication.primaryScreen()).availableGeometry()
    assert window.size().width() <= avail.width() and window.size().height() <= avail.height(), \
        _width_report(window)


def test_the_cumulants_can_take_their_own_scale_range(qtbot):
    """"cumulant range" shows a second window on the three log₂ a panels; dragging it on
    any panel moves it on all three and refits the cumulants only, the spectrum fit stays; off,
    the cumulants follow the fit window again."""
    q = np.linspace(-3.0, 3.0, 25)
    hd = _hd_q(q, 0.7 * q - 0.3 * q ** 2 / 2)
    rng = np.random.default_rng(3)
    scales = hd["scales"]
    mods = [np.exp(rng.normal(0.7 * np.log(a), np.sqrt(3.0 - 0.3 * np.log(a)), 20000))
            for a in scales]
    window = MultifractalWindow(hd, hd, extrema_mods=mods, scales=scales)
    qtbot.addWidget(window)
    fit_before = np.asarray(window._cq_curve.getData()[1]).copy()
    assert len(window._cum_regions) == 3 and not any(r.isVisible() for r in window._cum_regions)
    window.cumulant_range_check.setChecked(True)
    assert all(r.isVisible() for r in window._cum_regions)
    assert window._cum_regions[0].getRegion() == pytest.approx(window._region.getRegion())
    window._cum_regions[1].setRegion((0.0, 0.5))                # one scale inside: no fit
    assert window._cum_regions[0].getRegion() == pytest.approx((0.0, 0.5))
    assert np.isnan(window.current_cumulants()["c2"])
    np.testing.assert_allclose(window._cq_curve.getData()[1], fit_before)
    assert "log₂ a" in window._cq_label.text()
    window.cumulant_range_check.setChecked(False)
    assert not any(r.isVisible() for r in window._cum_regions)
    assert window.current_cumulants() is None
    window.cumulant_check.setChecked(True)
    np.testing.assert_allclose(window.current_cumulants()["c2"], 0.3, atol=0.03)


def test_the_kink_check_marks_q_star_and_the_bridged_h_interval(qtbot):
    """Spec 15, off by default: a τ(q) with a first-order transition gets a line at q* on τ(q)
    and the h interval the Legendre spectrum bridges shaded on D(h); a smooth τ gets neither."""
    q = np.linspace(-3.0, 3.0, 61)
    window = MultifractalWindow(*(lambda hd: (hd, hd))(_hd_q(q, np.minimum(0.3 * q, 0.7 * q))))
    qtbot.addWidget(window)
    assert not window.kink_check.isChecked() and not window._kink_lines
    window.kink_check.setChecked(True)
    (line,), (band,) = window._kink_lines, window._kink_bands
    assert line.value() == pytest.approx(0.0, abs=0.1)
    assert band.getRegion() == pytest.approx((0.3, 0.7), abs=1e-6)
    assert "q*" in window._kink_label.text()
    window.kink_check.setChecked(False)
    assert not window._kink_lines and window._kink_label.text() == ""
    smooth = MultifractalWindow(*(lambda hd: (hd, hd))(_hd_q(q, 0.5 * q - 0.05 * q ** 2)))
    qtbot.addWidget(smooth)
    smooth.kink_check.setChecked(True)
    assert not smooth._kink_lines


def test_the_gaussian_ensemble_overlays_d_h_points_from_the_maxima_moduli(qtbot):
    q = np.linspace(-3.0, 3.0, 25)
    hd = _hd_q(q, 0.7 * q - 0.3 * q ** 2 / 2)
    rng = np.random.default_rng(3)
    scales = hd["scales"]
    mods = [np.exp(rng.normal(0.7 * np.log(a), np.sqrt(3.0 - 0.3 * np.log(a)), 4000))
            for a in scales]
    window = MultifractalWindow(hd, hd, extrema_mods=mods, scales=scales, log2_L=12.0)
    qtbot.addWidget(window)
    assert window.gaussian_check.isEnabled() and not window.gaussian_check.isChecked()
    window.gaussian_check.setChecked(True)
    x, y = window._gaussian_points.getData()
    assert x is not None and len(x) > 10 and np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    window.gaussian_check.setChecked(False)
    assert not window._gaussian_points.isVisible()
    bare = MultifractalWindow(hd, hd)
    qtbot.addWidget(bare)
    assert not bare.gaussian_check.isEnabled()

def test_the_cumulant_range_is_disabled_without_moduli_or_moments(qtbot):
    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    assert not window.cumulant_range_check.isEnabled()


def test_a_q_picked_on_the_c_q_panel_sets_the_d_h_tangent(qtbot):
    """A vertical line on c(q) picks the q of the D(h) tangent (slope q, through the fitted
    (h(q), D(q))), snapped to the fitted q grid; before any pick it sits at the automatic
    convexity-loss q_c, and a refit keeps the picked q."""
    window = MultifractalWindow(*_broken_pair())
    qtbot.addWidget(window)
    pick = window._q_pick_line
    assert pick.isVisible() and not window._tangent_line.isVisible()     # std: concave
    pick.setValue(-1.3)                                                 # a drag
    pick.sigPositionChangeFinished.emit(pick)
    assert pick.value() == pytest.approx(-1.5)                          # the q grid steps 0.5
    fit = window.current_fit()
    i = int(np.argmin(np.abs(np.asarray(fit["q_list"]) + 1.5)))
    line = window._tangent_line
    assert line.isVisible()
    x, y = line.xData, line.yData
    assert (y[-1] - y[0]) / (x[-1] - x[0]) == pytest.approx(-1.5)
    at_zero = y[0] - 1.5 * (0.0 - x[0])                            # the line spans the view
    assert at_zero == pytest.approx(fit["D"][i] + 1.5 * fit["h"][i])
    assert "q = -1.5" in window._tangent_label.textItem.toPlainText()
    window._table_combo.setCurrentText("sup (cmax)")
    assert window._tangent_line.isVisible() and pick.value() == pytest.approx(-1.5)


def test_the_window_names_structure_functions_for_the_moment_route_and_hides_a_lone_table(qtbot):
    """hd_kind = "structure" (moments over the projections, leaders): the title and the first
    family read structure functions; with one table under both names the Table choice is hidden.
    The default stays partition functions with the choice visible."""
    from dynamix.shell.multifractal_window import vocabulary

    assert vocabulary("structure") == ("Structure functions", "log₂ S(q,a)")
    assert vocabulary(None) == vocabulary("partition") == ("Partition functions", "log₂ Z(q,a)")
    hd_std, hd_cmax = _pair()
    window = MultifractalWindow(hd_std, hd_std, kind="structure",
                                source="moments over the projections (Turiel 2006, M method)")
    qtbot.addWidget(window)
    assert window.windowTitle().startswith("Structure functions — moments over")
    assert window._family_plots["tau_qa"].getAxis("left").labelText == "log₂ S(q,a)"
    assert window._table_combo.isHidden() and window._table_label.isHidden()
    plain = MultifractalWindow(hd_std, hd_cmax)
    qtbot.addWidget(plain)
    assert plain.windowTitle().startswith("Partition functions — ")
    assert plain._family_plots["tau_qa"].getAxis("left").labelText == "log₂ Z(q,a)"
    assert not plain._table_combo.isHidden()


def test_the_main_windows_button_names_what_the_active_result_holds(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=())
    qtbot.addWidget(win)
    assert win._spectrum_button.text() == "Partition functions…" and not win._spectrum_button.isEnabled()
    hd_std, hd_cmax = _pair()
    win._active_result = {"hd_std": hd_std, "hd_cmax": hd_std, "hd_kind": "structure"}
    win._refresh_spectrum_button()
    assert win._spectrum_button.text() == "Structure functions…" and win._spectrum_button.isEnabled()
    win._active_result = {"hd_std": hd_std, "hd_cmax": hd_cmax}
    win._refresh_spectrum_button()
    assert win._spectrum_button.text() == "Partition functions…"


def test_named_tables_fill_the_table_choice_in_their_order(qtbot):
    """A result carrying named structure-function tables (the wavelet leaders' methods) lists
    them under the Table choice, the first the default; one named table hides the choice."""
    hd_std, hd_cmax = _pair()
    tables = {"p-leaders": hd_cmax, "leaders": hd_std, "wse": hd_std}
    window = MultifractalWindow(hd_cmax, hd_cmax, kind="structure", tables=tables)
    qtbot.addWidget(window)
    assert [window._table_combo.itemText(i) for i in range(window._table_combo.count())] == [
        "p-leaders", "leaders", "wse"]
    assert window._table_combo.isVisible() or not window._one_table
    assert window._current_hd() is hd_cmax
    window._table_combo.setCurrentText("leaders")
    assert window._current_hd() is hd_std
    fit = spectra.fit_spectra(hd_std, *window._region.getRegion())
    assert np.allclose(window.current_fit()["tau"], fit["tau"], equal_nan=True)
    lone = MultifractalWindow(hd_std, hd_std, kind="structure", tables={"leaders": hd_std})
    qtbot.addWidget(lone)
    assert lone._one_table and lone._current_hd() is hd_std


def test_the_fitter_publishes_its_fit_in_the_shared_schema_on_every_refit(qtbot):
    hd_std, hd_cmax = _pair()
    window = MultifractalWindow(hd_std, hd_cmax, source="canonical (Arneodo WTMM) · L1")
    qtbot.addWidget(window)
    published = []
    window.fitChanged.connect(published.append)
    window._region.setRegion((1.0, 5.0))
    fit = published[-1]
    assert fit["name"] == "canonical (Arneodo WTMM) · L1 · no-sup (std)" and fit["kind"] == "partition functions"
    expect = spectra.fit_spectra(hd_std, 1.0, 5.0)
    np.testing.assert_allclose(fit["tau"], expect["tau"])
    np.testing.assert_allclose(fit["h"], expect["h"]) and np.testing.assert_allclose(fit["D"], expect["D"])
    np.testing.assert_allclose(fit["hull"]["D"], spectra.legendre_dh(fit["q"], expect["tau"])["D"], equal_nan=True)
    np.testing.assert_allclose(fit["cq"], spectra.specific_heat(fit["q"], expect["tau"]), equal_nan=True)
    assert window.published_fit()["name"] == fit["name"]


def test_the_dh_legend_names_the_gmwlp_for_a_leaders_h_map(qtbot):
    h = np.linspace(0.2, 0.8, 7)
    D, dD = 2 - 5 * (h - 0.5) ** 2, np.full(7, 0.05)
    window = MultifractalWindow(*_pair(), dh_hist=(h, D, dD), h_map_method="wavelet_leaders")
    qtbot.addWidget(window)
    assert "GMWLP histogram (leaders)" in window._dh_legend_names()
    plain = MultifractalWindow(*_pair(), dh_hist=(h, D, dD))
    qtbot.addWidget(plain)
    assert "microcanonical histogram" in plain._dh_legend_names()
