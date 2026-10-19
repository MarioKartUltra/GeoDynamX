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
