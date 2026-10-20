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
    w.auto_amplitude_check.setChecked(False)                    # an explicit reference of 1
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


def test_the_shift_and_the_automatic_reference_are_on_the_box_with_the_narrow_q_default(qtbot):
    """Leonarduzzi et al. 2019: the shifted family g = −γ(h − δ)² with a narrow q and a δ sweep
    over the spectrum's support (§III-C3) and the amplitude reference c₁,₀ (18) are the
    defaults; the inversion reports the δ that won at each h."""
    scales = 2.**np.arange(1, 6)
    sets = {"leaders": {"scales": scales, "mods": [np.array([a**.4, a**.8]) for a in scales]}}
    w = GeneralizedSpectrumWindow(sets, log2_L=7.)
    qtbot.addWidget(w)
    assert (w.spins["q min"].value(), w.spins["q max"].value()) == (-4., 4.)
    assert (w.spins["δ min"].value(), w.spins["δ max"].value()) == (w.spins["h min"].value(), w.spins["h max"].value())
    assert w.auto_amplitude_check.isChecked() and not w.spins["log₂ amplitude reference"].isEnabled()
    w.spins["Δδ"].setValue(0.5)
    with qtbot.waitSignal(w.fitChanged, timeout=20000):
        w.apply_button.click()
    fit = w.current_fit()
    assert fit["valid"] and fit["settings"]["amplitude"] == "auto"
    assert fit["delta_star"].shape == fit["h"].shape and fit["ensemble"]["tau"].ndim == 3
    assert "c₁,₀" in w.readout.text() and "δ" in w.readout.text()
    w.auto_amplitude_check.setChecked(False)
    assert w.spins["log₂ amplitude reference"].isEnabled()
