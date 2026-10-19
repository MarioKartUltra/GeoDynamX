# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np

from dynamix.shell.generalized_spectrum_window import GeneralizedSpectrumWindow


def test_generalized_fit_runs_off_thread_and_retracts_on_pending_edits(qtbot):
    scales = 2.**np.arange(1, 6)
    sets = {"leaders": {"scales": scales, "mods": [np.array([a**.4, a**.8]) for a in scales]}}
    w = GeneralizedSpectrumWindow(sets, log2_L=7.)
    qtbot.addWidget(w)
    w.spins["q min"].setValue(-3)
    w.spins["q max"].setValue(3)
    w.spins["γ max"].setValue(2)
    with qtbot.waitSignal(w.fitChanged, timeout=10000):
        w.apply_button.click()
    fit = w.current_fit()
    assert fit["valid"] and fit["ensemble"]["n_scales"] == 5
    assert fit["settings"]["amplitude"] == 1.
    with qtbot.waitSignal(w.fitChanged) as emitted:
        w.spins["γ max"].setValue(4)
    assert emitted.args[0]["valid"] is False and w.current_fit() is None


def test_generalized_invalid_grid_fails_without_dispatch(qtbot):
    sets = {"leaders": {"scales": [2., 4., 8.], "mods": [np.ones(3)]*3}}
    w = GeneralizedSpectrumWindow(sets)
    qtbot.addWidget(w)
    w.spins["q min"].setValue(5)
    w.spins["q max"].setValue(-5)
    w.apply_button.click()
    assert "ascending" in w.readout.text() and not w._jobs
