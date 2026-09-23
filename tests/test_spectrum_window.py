# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.spectrum_window.SpectrumWindow -- the singularity-spectrum
construction window.
Offscreen Qt, engine truth pinned against the core functions exactly as
tests/test_multifractal_window.py pins the fitter window."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import chain_groups, spectra
from dynamix.shell.spectrum_window import SpectrumWindow
from test_multifractal_window import _hd, _pair


def _h_map():
    rng = np.random.default_rng(5)
    return rng.normal(0.5, 0.2, (64, 64)).astype(np.float32)


def _chains(scales):
    def chain(mods):
        mod = np.asarray(mods, dtype=np.float64)
        k = mod.size
        with np.errstate(divide="ignore"):
            lm = np.log2(np.abs(mod))
        return {"x": np.zeros(k, dtype=np.int64), "y": np.arange(k, dtype=np.int64),
                "mod": mod, "log2_mod": lm,
                "log2_scales": np.log2(scales)[:k]}
    return [chain([8.0, 6.0, 9.0, 7.0]), chain([0.5, 0.4, 0.3, 0.2]),
            chain([2.0, 1.5]), chain([3.0, 2.5, 2.0, 1.5])]


# ------------------------------------------------------------------------------ construction

def test_builds_with_tables_only_h_map_only_and_refuses_neither(qtbot):
    window = SpectrumWindow(*_pair())
    qtbot.addWidget(window)
    assert window.windowTitle() == "Singularity spectrum — construction"
    assert window.current_fit() is not None

    hist_only = SpectrumWindow(h_map=_h_map())
    qtbot.addWidget(hist_only)
    assert hist_only.current_fit() is None
    assert not hist_only._table_combo.isEnabled()          # no tables -> canonical controls off

    with pytest.raises(ValueError):
        SpectrumWindow()


# ------------------------------------------------------------------------------ engine truth

def test_canonical_points_are_exactly_the_core_fit(qtbot):
    hd_std, hd_cmax = _pair()
    window = SpectrumWindow(hd_std, hd_cmax)
    qtbot.addWidget(window)
    lo, hi = window._lo_spin.value(), window._hi_spin.value()
    expect = spectra.fit_spectra(hd_cmax, lo, hi)          # sup is the default table
    x, y = window._canon_curve.getData()
    finite = np.isfinite(expect["h"]) & np.isfinite(expect["D"])
    np.testing.assert_allclose(x, np.asarray(expect["h"])[finite], rtol=1e-12)
    np.testing.assert_allclose(y, np.asarray(expect["D"])[finite], rtol=1e-12)


def test_hull_overlay_is_the_core_legendre_transform_and_is_optional(qtbot):
    window = SpectrumWindow(*_pair())
    qtbot.addWidget(window)
    x, _ = window._hull_curve.getData()
    assert x is None or len(x) == 0                        # off by default -- an OPTION
    window._hull_check.setChecked(True)
    fit = window.current_fit()
    hull = spectra.legendre_dh(fit["q_list"], fit["tau"])
    x, y = window._hull_curve.getData()
    np.testing.assert_allclose(x, hull["h"], rtol=1e-12)
    np.testing.assert_allclose(y, hull["D"], rtol=1e-12)


def test_microcanonical_histogram_rides_with_the_pont_caveat(qtbot):
    from dynamix.core.microcanonical import dh_histogram

    h_map = _h_map()
    window = SpectrumWindow(h_map=h_map, h_map_estimator="punctual")
    qtbot.addWidget(window)
    h_ref, D_ref = dh_histogram(h_map)
    np.testing.assert_allclose(window._hist[0], h_ref)
    np.testing.assert_allclose(window._hist[1], D_ref, equal_nan=True)
    assert "right limb unreliable (punctual estimator)" in window._hist_label

    plain = SpectrumWindow(h_map=h_map, h_map_estimator="regression")
    qtbot.addWidget(plain)
    assert "unreliable" not in plain._hist_label


def test_scale_window_coupling_channel_refits(qtbot):
    hd = _hd([1.0, 2.0, 3.0], [0.3, 0.5, 0.7], [2.0, 2.0, 1.4])
    row = hd["tau_qa"][0].copy()
    row[4:] = row[3] + 5.0 * (hd["log2_scales"][4:] - hd["log2_scales"][3])
    hd["tau_qa"][0] = row
    window = SpectrumWindow(hd, hd)
    qtbot.addWidget(window)
    window.set_scale_window(-0.5, 3.5)
    np.testing.assert_allclose(window.current_fit()["tau"][0], 1.0, rtol=1e-12)
    window.set_scale_window(3.5, 7.5)
    np.testing.assert_allclose(window.current_fit()["tau"][0], 5.0, rtol=1e-12)


# --------------------------------------------------------------------------------- grouping

def test_grouping_classifies_and_broadcasts(qtbot):
    hd_std, hd_cmax = _pair()
    scales = np.asarray(hd_std["scales"], dtype=np.float64)[:4]
    chains = _chains(scales)
    window = SpectrumWindow(hd_std, hd_cmax, chains=chains, scales=scales)
    qtbot.addWidget(window)
    dom = window.current_grouping()
    assert dom is not None and dom.size == len(chains)
    expect = chain_groups.classify_chains(
        chain_groups.grouping_payload(chains, scales.size),
        scale_idx=int(window._gscale_spin.value()), q=2.0, mode="percentile",
        dom_percentile=20.0, min_len=3)
    np.testing.assert_array_equal(dom, expect)
    assert f"{int(dom.sum())} dominant / {dom.size}" in window._gcount_label.text()

    seen = []
    window.groupingChanged.connect(lambda m: seen.append(np.asarray(m).copy()))
    window._gq_spin.setValue(-3.0)                          # negative q flips dominance
    assert seen and not np.array_equal(seen[-1], dom)


def test_group_spectra_overlay_is_gated_and_matches_the_core_subsets(qtbot):
    pytest.importorskip("dynamix._vendor.wtmm_ebsd")
    hd_std, hd_cmax = _pair()
    scales = np.asarray(hd_std["scales"], dtype=np.float64)[:4]
    chains = _chains(scales)
    window = SpectrumWindow(hd_std, hd_cmax, chains=chains, scales=scales)
    qtbot.addWidget(window)
    x, _ = window._dom_curve.getData()
    assert x is None or len(x) == 0                         # gated off by default

    window._gminlen_spin.setValue(1)
    window._gspectra_check.setChecked(True)
    dom = window.current_grouping()
    assert dom.any() and (~dom).any()
    q_list = np.asarray(hd_cmax["q_list"], dtype=np.float64)
    lo, hi = window._lo_spin.value(), window._hi_spin.value()
    tables = chain_groups.subset_hd(chains, scales, q_list, dom, min_chain_len=2)
    expect = spectra.fit_spectra(tables[1], lo, hi)         # sup table, the default
    x, y = window._dom_curve.getData()
    finite = np.isfinite(expect["h"]) & np.isfinite(expect["D"])
    np.testing.assert_allclose(x, np.asarray(expect["h"])[finite], rtol=1e-12)
    np.testing.assert_allclose(y, np.asarray(expect["D"])[finite], rtol=1e-12)


def test_focus_fit_modes_and_the_honesty_readout(qtbot):
    """2026-09-22: the Fit combo (naive / fixed focus / focus) drives fit_spectra's
    fit_mode; the readout shows branch, x0 and the Delta-h naive->focus pair side by
    side (the price display). Without log2_L the combo is disabled."""
    hd_std, hd_cmax = _pair()
    window = SpectrumWindow(hd_std, hd_cmax, log2_L=8.0)
    qtbot.addWidget(window)
    assert [window._fitmode_combo.itemText(i) for i in range(3)] == \
        ["naive", "fixed focus", "focus"]
    assert window._focus_readout.text() == ""
    window._fitmode_combo.setCurrentText("focus")
    assert "focus" in window._fit
    assert window._focus_readout.text() != ""
    window._fitmode_combo.setCurrentText("naive")
    assert window._focus_readout.text() == ""

    bare = SpectrumWindow(*_pair())
    qtbot.addWidget(bare)
    assert not bare._fitmode_combo.isEnabled()
