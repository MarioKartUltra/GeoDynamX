# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Structure functions window, the wavelet leaders' own fitter: its fit is the core's
structure_fit over the chosen set, moments, weights, frame and range; every change publishes."""
from __future__ import annotations

import numpy as np

from dynamix.core import leaders, spectra
from dynamix.shell.structure_functions_window import StructureFunctionsWindow


def _samples(H=0.5, J=5, n=64):
    """Exact power laws: every cell of level j holds 2^{jH}, the coefficients 2^{jH}/2, so
    ζ(q) = qH, τ(q) = qH − 2, h = H, D = 2 to machine precision; one cell per level invalid."""
    sets = {}
    for name, scale in (("leaders", 1.0), ("p-leaders", 1.0), ("wse", 1.0), ("coefficients", 0.5)):
        values, valid = [], []
        for j in range(1, J + 1):
            side = n // 2 ** j
            shape = (3, side, side) if name == "coefficients" else (side, side)
            values.append(np.full(shape, scale * 2.0 ** (j * H), dtype=np.float32))
            v = np.ones((side, side), bool)
            v[0, 0] = False
            valid.append(v)
        sets[name] = {"values": values, "valid": valid}
    sets["coefficients"]["positive_only"] = True
    return {"scales": 2.0 ** np.arange(1, J + 1), "cells": [(n // 2 ** j) ** 2 for j in range(1, J + 1)],
            "d": 2, "sets": sets}


def test_the_window_lists_the_sets_and_fits_the_core_over_the_seeded_range(qtbot):
    s = _samples()
    window = StructureFunctionsWindow(s, levels=(2, 5), source="L1 · sym4")
    qtbot.addWidget(window)
    assert window.windowTitle() == "Structure functions — L1 · sym4"
    assert [window.set_combo.itemText(i) for i in range(4)] == ["leaders", "p-leaders", "WSE leaders", "coefficients"]
    assert tuple(window._region.getRegion()) == (2.0, 5.0) and (window.lo_spin.value(), window.hi_spin.value()) == (2.0, 5.0)
    fit = window.current_fit()
    q = fit["q"]
    assert list(q) == list(np.arange(-3.0, 6.01, 0.5)) and fit["name"] == "L1 · sym4 · leaders, structure functions"
    expect = leaders.structure_fit(leaders.tables(s, q)["leaders"], 2, 5)
    np.testing.assert_allclose(fit["tau"], expect["tau"])
    np.testing.assert_allclose(fit["tau"], 0.5 * q - 2, atol=1e-5)
    np.testing.assert_allclose(fit["h"], 0.5, atol=1e-5)
    np.testing.assert_allclose(fit["D"], 2.0, atol=1e-4)
    np.testing.assert_allclose(fit["hull"]["D"], spectra.legendre_dh(q, fit["tau"])["D"], equal_nan=True)
    assert fit["kind"] == "structure functions" and len(window._family_items) == q.size == len(window._fit_items)
    xd, yd = window._curves["D"].getData()
    np.testing.assert_allclose(xd, fit["h"]) and np.testing.assert_allclose(yd, fit["D"])
    assert "ζ(2) = 1.000" in window.readout.text() and "h(1) = 0.500" in window.readout.text()
    assert "4 levels" in window.readout.text() and "uniform weights" in window.readout.text()


def test_the_region_the_set_the_weights_and_the_moments_drive_the_fit_and_publish(qtbot):
    s = _samples()
    window = StructureFunctionsWindow(s, levels=(2, 5))
    qtbot.addWidget(window)
    published = []
    window.fitChanged.connect(published.append)
    window._region.setRegion((1.0, 4.0))
    assert (window.lo_spin.value(), window.hi_spin.value()) == (1.0, 4.0)
    assert list(window.current_fit()["levels"]) == [1, 2, 3, 4] and published[-1] is window.current_fit()
    window.lo_spin.setValue(3.0)                                            # the spins drive the region
    assert tuple(window._region.getRegion()) == (3.0, 4.0) and np.all(np.isnan(window.current_fit()["tau"]))
    window.set_scale_window(2, 5)
    window.set_combo.setCurrentText("coefficients")
    fit = window.current_fit()
    assert fit["set"] == "coefficients" and np.all(np.isnan(fit["tau"][fit["q"] <= 0]))
    np.testing.assert_allclose(fit["tau"][fit["q"] > 0], (0.5 * fit["q"] - 2)[fit["q"] > 0], atol=1e-5)
    window.set_combo.setCurrentText("WSE leaders")
    window.weights_combo.setCurrentIndex(1)
    fit = window.current_fit()
    table = leaders.tables(s, fit["q"])["wse"]
    np.testing.assert_allclose(fit["tau"], leaders.structure_fit(table, 2, 5, weights=table["N_a"])["tau"])
    assert fit["weights"] == "counts" and "counts weights" in window.readout.text()
    window.qmin_spin.setValue(0.0)
    window.qmax_spin.setValue(4.0)
    window.qstep_spin.setValue(1.0)
    fit = window.current_fit()
    assert list(fit["q"]) == [0.0, 1.0, 2.0, 3.0, 4.0] and len(window._family_items) == 5
    assert len(published) >= 7


def test_gamma_enables_the_frame_choice_and_the_original_frame_takes_it_out(qtbot):
    s = _samples(H=1.5)                                                     # a lifted field: H + γ
    window = StructureFunctionsWindow(s, gamma=1.0, levels=(2, 5))
    qtbot.addWidget(window)
    assert window.frame_combo.isEnabled() and window.frame_combo.currentText() == "original"
    fit = window.current_fit()
    np.testing.assert_allclose(fit["h"], 0.5, atol=1e-5)
    np.testing.assert_allclose(fit["tau"], 0.5 * fit["q"] - 2, atol=1e-5)
    assert "γ = 1, original frame" in window.readout.text()
    window.frame_combo.setCurrentText("integrated")
    np.testing.assert_allclose(window.current_fit()["h"], 1.5, atol=1e-5)
    plain = StructureFunctionsWindow(_samples(), levels=(2, 5))
    qtbot.addWidget(plain)
    assert not plain.frame_combo.isEnabled()


def test_the_readout_reports_jaffards_sobolev_condition_on_the_fitted_zeta(qtbot):
    """Jaffard 1997 Part I (3.6), the consistency check every fitted ζ(q) must pass, is read
    beside the fit: exact power laws hold it with the margin 2/p at the largest fitted p."""
    window = StructureFunctionsWindow(_samples(), levels=(2, 5))
    qtbot.addWidget(window)
    fit = window.current_fit()
    assert fit["sobolev"]["holds"]
    assert "(3.6) holds" in window.readout.text()


def test_the_family_panel_draws_each_scale_as_a_dot_with_the_fit_line_through_them(qtbot):
    window = StructureFunctionsWindow(_samples(), levels=(2, 5))
    qtbot.addWidget(window)
    assert window._family_items
    assert all(it.opts["symbol"] == "o" and it.opts["pen"] is None for it in window._family_items)
    assert window._fit_items and all(it.opts["pen"] is not None for it in window._fit_items)


def test_the_x_axis_switches_to_the_physical_length_when_a_conversion_is_known(qtbot):
    from dynamix.shell.multifractal_window import PhysicalLog2Axis
    window = StructureFunctionsWindow(_samples(), levels=(2, 5), physical_of_log2a=lambda v: (2.0 ** v * 3.0, "m"))
    qtbot.addWidget(window)
    assert window._xaxis_combo.isEnabled()
    window._xaxis_combo.setCurrentText("log₂ λ* (unit)")
    assert isinstance(window.family_plot.getAxis("bottom"), PhysicalLog2Axis)
    assert "(m)" in window.family_plot.getAxis("bottom").labelText
    window._xaxis_combo.setCurrentText("log₂ a (px)")
    assert window.family_plot.getAxis("bottom").labelText == "log₂ a"
    plain = StructureFunctionsWindow(_samples(), levels=(2, 5))
    qtbot.addWidget(plain)
    assert not plain._xaxis_combo.isEnabled()


def test_a_dyadic_ladder_reads_j_on_the_scale_axis(qtbot):
    window = StructureFunctionsWindow({**_samples(), "dyadic": True}, levels=(2, 5))
    qtbot.addWidget(window)
    assert window.family_plot.getAxis("bottom").labelText == "j"
    assert window._scale_label.text() == "j"


def test_the_fit_carries_the_local_cubic_legendre_beside_the_hull(qtbot):
    window = StructureFunctionsWindow(_samples(), levels=(2, 5))
    qtbot.addWidget(window)
    local = window._fit["legendre_local"]
    assert local["h"].size == window._fit["q_list"].size - 4
    np.testing.assert_allclose(window._local.getData()[0], local["h"])


def test_the_fit_modes_run_the_focus_regression_on_the_same_table(qtbot):
    """naive, fixed focus and focus as in the Partition functions window: a focus mode replaces
    the per-q slopes by the focus regression anchored at the field extent, ζ = τ + d and the
    (3.6) check follow, the fit lines pass through the focus, and without the extent the
    switch is off."""
    from dynamix.core import spectra
    window = StructureFunctionsWindow(_samples(), levels=(1, 5), log2_L=6.0)
    qtbot.addWidget(window)
    assert window.fit_combo.isEnabled() and window._fit.get("focus") is None
    naive_tau = np.array(window._fit["tau"])
    window.fit_combo.setCurrentText("fixed focus")
    fit = window._fit
    table = window._table()
    expected = spectra.fit_spectra(table, 1.0, 5.0, eta=0.0, frame="original", fit_mode="fixed_focus", log2_L=6.0)
    assert fit["focus"]["fit_mode"] == "fixed_focus" and fit["focus"]["branch"] != "unavailable"
    np.testing.assert_allclose(fit["tau"], expected["tau"])
    np.testing.assert_allclose(fit["zeta"], fit["tau"] + 2.0)
    assert "holds" in fit["sobolev"] and not window.weights_combo.isEnabled()
    assert not np.allclose(fit["tau"], naive_tau)
    # The lines through the focus: at x0 every line reads q y0 + log2 N(x0).
    q, x0, y0 = fit["q_list"], fit["focus"]["x0"], fit["focus"]["y0"]
    gamma_q = 0.0
    at_x0 = fit["tau_0"] + (fit["tau"] + gamma_q) * x0
    x = np.asarray(table["log2_scales"], float)
    n = np.asarray(table["N_a"], float)
    alive = (x >= 1) & (x <= 5) & (n > 0)
    s_n = fit["focus"]["s_N"]
    log2n_x0 = s_n * x0 + (np.log2(n[alive]).mean() - s_n * x[alive].mean())
    np.testing.assert_allclose(at_x0, q * y0 + log2n_x0, atol=1e-9)
    assert len(window._fit_items) == np.isfinite(fit["tau"]).sum()
    window.fit_combo.setCurrentText("naive")
    assert window._fit.get("focus") is None and window.weights_combo.isEnabled()
    plain = StructureFunctionsWindow(_samples(), levels=(1, 5))
    qtbot.addWidget(plain)
    assert not plain.fit_combo.isEnabled()


def test_gamma_and_delta_turn_the_table_into_the_generalized_structure_functions(qtbot):
    """γ ≠ 0 swaps the table for Leonarduzzi's generalized one at (γ, δ) with the field extent as
    L, the fit is named for them, the Legendre routes carry +γ(h − δ)² back, and γ = 0 is the
    ordinary table again."""
    from dynamix.core import leaders, spectra
    window = StructureFunctionsWindow(_samples(), levels=(1, 5), log2_L=6.0)
    qtbot.addWidget(window)
    plain = window._table()
    window.gamma_spin.setValue(2.0)
    window.delta_spin.setValue(0.5)
    fit = window._fit
    assert fit["generalized"] == {"gamma": 2.0, "delta": 0.5} and fit["name"].endswith("structure functions, γ = 2 δ = 0.5")
    expected = leaders.generalized_table(_samples(), "leaders", window._q_list(), 2.0, 0.5, length_px=64.0)
    np.testing.assert_allclose(window._table()["tau_qa"], expected["tau_qa"], equal_nan=True)
    hull = spectra.legendre_dh(fit["q_list"], fit["tau"])
    np.testing.assert_allclose(fit["hull"]["D"], hull["D"] + 2.0 * (hull["h"] - 0.5) ** 2)
    assert "generalized γ = 2" in window.readout.text() if hasattr(window, "readout") else True
    window.gamma_spin.setValue(0.0)
    assert window._table() is plain and window._fit["generalized"] is None
