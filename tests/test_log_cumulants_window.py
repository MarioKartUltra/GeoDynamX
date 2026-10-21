# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Log-cumulants window: four regressions, one per cumulant of ln ℓ, each on its own range
unless the ranges are synced; the published expansion."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import spectra
from dynamix.model.samples import sample_sets
from dynamix.shell.log_cumulants_window import LogCumulantsWindow


def _samples(c1=0.5, c2=0.05, J=5, n=1024, seed=0):
    """A log-normal cascade in ln ℓ: at level j the mean is c1 j ln 2 and the variance
    1 − c2 j ln 2 (the variance shrinks towards coarse scales; its slope against ln a, c₂ in the
    leaders literature's sign, is −c2)."""
    rng = np.random.default_rng(seed)
    values, valid = [], []
    for j in range(1, J + 1):
        side = n // 2 ** j
        x = rng.standard_normal((side, side)) * np.sqrt(1.0 - c2 * j * np.log(2.0)) + c1 * j * np.log(2.0)
        values.append(np.exp(x))
        valid.append(np.ones((side, side), bool))
    sets = {"leaders": {"values": values, "valid": valid},
            "wse": {"values": [v * 2.0 for v in values], "valid": valid}}
    samples = {"scales": 2.0 ** np.arange(1, J + 1), "cells": [(n // 2 ** j) ** 2 for j in range(1, J + 1)],
               "d": 2, "sets": sets}
    return sample_sets({"samples": samples, "scales": samples["scales"]})


def test_four_panels_each_regress_their_cumulant_over_the_seeded_range(qtbot):
    s = _samples()
    window = LogCumulantsWindow(s, levels=(1, 5), source="L1 · sym4")
    qtbot.addWidget(window)
    assert window.windowTitle() == "Log-cumulants — L1 · sym4" and window.sync_check.isChecked()
    assert window.ranges() == [(1.0, 5.0)] * 4
    fit = window.current_fit()
    C = spectra.ln_moments(s["leaders"]["mods"])
    for m in (1, 2, 3, 4):
        reg = spectra.cumulant_regression(C, s["leaders"]["scales"], 1, 5, m)
        assert fit["cumulants"][f"c{m}"] == pytest.approx(reg["c"])
        assert fit["cumulants"]["n"][m] == 5 and fit["cumulants"]["r2"][m] == pytest.approx(reg["r2"])
        text = window.labels[m].text
        assert text.startswith(f"c{'₁₂₃₄'[m - 1]} = ") and "R²" in text and "N 5" in text
        xs, ys = window._points[m].getData()
        np.testing.assert_allclose(xs, np.arange(1, 6)) and np.testing.assert_allclose(ys, C[m - 1])
    assert abs(fit["cumulants"]["c1"] - 0.5) < 0.05 and abs(fit["cumulants"]["c2"] + 0.05) < 0.03
    assert fit["kind"] == "log-cumulants" and fit["name"] == "L1 · sym4 · leaders, log-cumulants"
    np.testing.assert_allclose(fit["tau"], spectra.cumulant_tau(fit["cumulants"], fit["q"], c0=2.0))
    np.testing.assert_allclose(fit["cq"], spectra.cumulant_heat(fit["cumulants"], fit["q"]))
    np.testing.assert_allclose(fit["D"], fit["q"] * fit["h"] - fit["tau"])
    assert "c₁ = +0.5" in window.readout.text() or "c₁ = +0.4" in window.readout.text()


def test_ranges_move_together_when_synced_and_alone_when_not_and_every_change_publishes(qtbot):
    s = _samples()
    window = LogCumulantsWindow(s, levels=(1, 5))
    qtbot.addWidget(window)
    published = []
    window.fitChanged.connect(published.append)
    window.regions[2].setRegion((2.0, 5.0))
    assert window.ranges() == [(2.0, 5.0)] * 4
    assert all(window.current_fit()["cumulants"]["n"][m] == 4 for m in (1, 2, 3, 4))
    window.sync_check.setChecked(False)
    window.regions[0].setRegion((3.0, 5.0))
    assert window.ranges() == [(3.0, 5.0), (2.0, 5.0), (2.0, 5.0), (2.0, 5.0)]
    cums = window.current_fit()["cumulants"]
    assert cums["n"][1] == 3 and cums["n"][2] == 4 and cums["range"][1] == (3.0, 5.0)
    window.sync_check.setChecked(True)                                     # the first panel's range wins
    assert window.ranges() == [(3.0, 5.0)] * 4
    window.regions[1].setRegion((1.0, 1.5))                                # one level: no regression
    assert np.isnan(window.current_fit()["cumulants"]["c2"])
    assert window.labels[2].text == "c₂: fewer than two levels in range"
    assert np.all(np.isnan(window.current_fit()["tau"]))                   # no expansion without every order
    assert len(published) == 4 and published[-1] is window.current_fit()   # one per gesture, the sync one included
    window.set_combo.setCurrentText("WSE leaders")
    assert window.current_fit()["set"] == "WSE leaders" and window.current_fit()["name"] == "WSE leaders, log-cumulants"


def test_the_original_frame_takes_gamma_out_of_c1_only(qtbot):
    s = _samples(c1=1.5)
    window = LogCumulantsWindow(s, gamma=1.0, levels=(1, 5))
    qtbot.addWidget(window)
    assert window.frame_combo.isEnabled()
    original = dict(window.current_fit()["cumulants"])
    window.frame_combo.setCurrentText("integrated")
    integrated = window.current_fit()["cumulants"]
    assert integrated["c1"] == pytest.approx(original["c1"] + 1.0) and integrated["c2"] == original["c2"]
    assert "γ = 1, integrated frame" in window.readout.text()


def test_the_wtmm_s_maxima_lines_feed_the_same_tool(qtbot):
    from test_spectrum_window import _per_scale_set

    sets, _h, _D = _per_scale_set()
    window = LogCumulantsWindow(sets, source="L1")
    qtbot.addWidget(window)
    assert window.ranges() == [(1.0, 5.0)] * 4 and window.current_fit()["name"] == "L1 · synthetic, log-cumulants"
    assert not window.frame_combo.isEnabled()
    with pytest.raises(ValueError):
        LogCumulantsWindow({"empty": {"mods": [], "scales": None}})
