# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Large deviations window: the microcanonical route's own tool over a per-level sample
set, under both bin rules, publishing its curve."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import large_deviation as ld
from dynamix.shell.large_deviations_window import H_GRID, LD_RULES, LargeDeviationsWindow
from test_spectrum_window import _per_scale_set


def test_the_window_counts_the_set_under_both_rules_and_publishes(qtbot):
    sets, h, D = _per_scale_set()
    window = LargeDeviationsWindow(sets, log2_L=7.0, source="L1")
    qtbot.addWidget(window)
    assert window.windowTitle() == "Large deviations — L1"
    assert (window.lo_spin.value(), window.hi_spin.value()) == (1.0, 5.0)        # the set's own range
    published = []
    window.fitChanged.connect(published.append)
    mods, scales = sets["synthetic"]["mods"], sets["synthetic"]["scales"]
    fit = window.current_fit()
    q = np.arange(-3.0, 6.01, 0.5)
    truth = ld.adaptive(mods, scales, q_list=q, log2_a_min=1.0, log2_a_max=5.0, length_px=128.0)
    ok = np.isfinite(truth["h"]) & np.isfinite(truth["D"])
    np.testing.assert_allclose(fit["h"], truth["h"][ok]) and np.testing.assert_allclose(fit["D"], truth["D"][ok])
    mid = np.abs(fit["q"]) <= 2.0                                                # the tails carry the finite-bin excess
    np.testing.assert_allclose(fit["D"][mid], (2.0 - 5.0 * (fit["h"] - 0.5) ** 2)[mid], atol=0.15)
    assert fit["name"] == "L1 · synthetic, large deviations" and fit["kind"] == "large deviations"
    assert window.readout.text().startswith(f"synthetic · log₂a 1–5 (5 scales), L = 128 px · {int(ok.sum())} of {q.size} q")
    assert len(window._hist_items) == 5 and window.curves[-1].xData.size == ok.sum()
    window.rule_combo.setCurrentText(LD_RULES[1])
    fixed = ld.fixed(mods, scales, h_grid=H_GRID, log2_a_min=1.0, log2_a_max=5.0, length_px=128.0)
    for curve, Dk in zip(window.curves, fixed["D"]):                     # each trimmed to its finite span
        span = np.flatnonzero(np.isfinite(Dk))
        np.testing.assert_allclose(curve.xData, H_GRID[span[0]:span[-1] + 1])
    fit = window.current_fit()
    finite = np.isfinite(fixed["D"][-1])
    np.testing.assert_allclose(fit["h"], H_GRID[finite]) and np.testing.assert_allclose(fit["D"], fixed["D"][-1][finite])
    assert 0.25 < fit["h"].min() and fit["h"].max() < 0.75                      # the set's own span
    np.testing.assert_allclose(fit["D"], 2.0 - 5.0 * (fit["h"] - 0.5) ** 2, atol=0.1)
    assert "ε 0.1, 0.05, 0.025 · spread across ε ≤" in window.readout.text()
    assert len(published) == 1 and published[-1] is fit
    window.set_scale_window(2.0, 4.0)
    assert window.current_fit()["window"] == (2.0, 4.0) and len(window._hist_items) == 3
    window.set_scale_window(5.0, 5.0)                                            # one scale: no slope
    fit = window.current_fit()
    assert fit is None or not np.isfinite(fit["D"]).any()


def test_a_set_whose_scales_reach_the_extent_is_refused_in_the_readout_not_a_crash(qtbot):
    sets, _h, _D = _per_scale_set(L=32.0)
    window = LargeDeviationsWindow(sets, log2_L=5.0)
    qtbot.addWidget(window)
    assert window.current_fit() is None and "below the field extent" in window.readout.text()
    with pytest.raises(ValueError):
        LargeDeviationsWindow({"empty": {"mods": [], "scales": None}})


def test_fixed_grid_controls_apply_and_invalid_settings_retract_the_fit(qtbot):
    sets, _, _ = _per_scale_set()
    window = LargeDeviationsWindow(sets, log2_L=7.)
    qtbot.addWidget(window)
    assert not window.hstep_spin.isEnabled() and window.qstep_spin.isEnabled()
    window.rule_combo.setCurrentText(LD_RULES[1])
    assert window.hstep_spin.isEnabled() and not window.qstep_spin.isEnabled()
    published = []
    window.fitChanged.connect(published.append)
    window.hmin_spin.setValue(.2)
    window.hmax_spin.setValue(.8)
    window.hstep_spin.setValue(.05)
    window.apply_grid_button.click()
    np.testing.assert_allclose(window.current_fit()["h_grid"], np.arange(.2, .801, .05))
    original = window.current_fit()["D"].copy()
    for spin, eps in zip(window.eps_spins, (.025, .1, .05)):
        spin.setValue(eps)
    window.apply_grid_button.click()
    np.testing.assert_allclose(window.current_fit()["D"], original)
    window.eps_spins[1].setValue(window.eps_spins[0].value())
    window.apply_grid_button.click()
    assert window.current_fit() is None
    assert published[-1]["valid"] is False
    assert "distinct" in window.readout.text()
    assert all(curve.xData is None or len(curve.xData) == 0 for curve in window.curves)
