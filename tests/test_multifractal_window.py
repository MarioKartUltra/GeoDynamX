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
    assert window.windowTitle() == "Multifractal spectrum — canonical (Arneodo WTMM)"
    assert window.isModal() is False
    # sup (cmax) is the default table -- the textbook WTMM convention.
    assert window._table_combo.currentText() == TABLE_CHOICES[0] == "sup (cmax)"


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
    lo, hi = window._region.getRegion()
    expect = spectra.fit_spectra(hd_cmax, lo, hi)      # default table is cmax
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
    np.testing.assert_allclose(window.current_fit()["tau"], [1.5, 2.5, 3.5], rtol=1e-12)
    window._table_combo.setCurrentText("no-sup (std)")
    np.testing.assert_allclose(window.current_fit()["tau"], [1.0, 2.0, 3.0], rtol=1e-12)


# ------------------------------------------------------------------------------- eta / frame

def test_eta_seed_flows_into_the_control_and_the_fit(qtbot):
    """The caller-derived seed lands in the spin AND the fit; default frame 'original' applies
    the shift-back (tau - eta*q, h - eta), and the forward note is displayed verbatim."""
    hd_std, hd_cmax = _pair()
    note = "forward lift: a^1 on WT derivs (tensor)"
    window = MultifractalWindow(hd_std, hd_cmax, eta_seed=1.0, forward_note=note)
    qtbot.addWidget(window)
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


def test_convexity_loss_marker_tracks_the_negative_branch(qtbot):
    q = np.arange(-5.0, 5.01, 0.5)
    concave = 0.6 * q - 0.1 * q * q
    broken = concave.copy()
    bad = q < -3.0
    broken[bad] = concave[q == -3.0][0] + (0.6 + 0.6) * (q[bad] + 3.0) \
        + 0.3 * (q[bad] + 3.0) ** 2
    window = MultifractalWindow(_hd_q(q, concave), _hd_q(q, broken))
    qtbot.addWidget(window)
    assert window._convexity_line.isVisible()                       # cmax (default) is broken
    window._table_combo.setCurrentText("no-sup (std)")
    assert not window._convexity_line.isVisible()                   # std is clean


def test_scale_window_changes_are_broadcast(qtbot):
    window = MultifractalWindow(*_pair())
    qtbot.addWidget(window)
    seen = []
    window.scaleWindowChanged.connect(lambda lo, hi: seen.append((lo, hi)))
    window.set_scale_window(2.0, 6.0)
    assert seen and seen[-1] == (2.0, 6.0)
