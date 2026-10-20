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
    window._table_combo.setCurrentText("sup (cmax)")       # this test reads the sup table
    lo, hi = window._lo_spin.value(), window._hi_spin.value()
    expect = spectra.fit_spectra(hd_cmax, lo, hi)          # the sup table, picked above
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
    window._table_combo.setCurrentText("sup (cmax)")       # this test reads the sup table
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
    """The Fit combo (naive / fixed focus / focus) drives fit_spectra's fit_mode; the
    readout shows branch, x0 and the Delta-h naive->focus pair side by side (the price
    display). Without log2_L the combo is disabled."""
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


def test_filtered_tables_carry_their_note_and_a_switch_to_all_chains(qtbot):
    hd_std, hd_cmax = _pair()
    all_std, all_cmax = _pair()
    w = SpectrumWindow(hd_std, hd_cmax, all_tables=(all_std, all_cmax),
                       filter_note="filtered: 12 of 40 chains")
    qtbot.addWidget(w)
    assert "filtered: 12 of 40 chains" in w.windowTitle()
    assert w._current_hd() is hd_cmax or w._current_hd() is hd_std
    w.all_chains_check.setChecked(True)
    assert w._current_hd() is all_cmax or w._current_hd() is all_std
    plain = SpectrumWindow(hd_std, hd_cmax)
    qtbot.addWidget(plain)
    assert "filtered" not in plain.windowTitle() and plain.all_chains_check is None


def test_the_histogram_construction_has_bars_bins_resolution_per_scale_curves_and_gh(qtbot):
    from dynamix.core import microcanonical as mc

    rng = np.random.default_rng(0)
    h_map = rng.normal(0.4, 0.1, (64, 64))
    field = np.cumsum(np.cumsum(rng.normal(size=(64, 64)), axis=0), axis=1)
    scales = np.array([1.0, 2.0, 4.0])
    T = np.stack([np.exp(rng.normal(0.0, 0.3, (64, 64))) * r ** 0.5 for r in scales])
    curves = mc.per_scale_histograms(T, scales, n_bins=30)
    window = SpectrumWindow(h_map=h_map, scales=scales, dh_curves=curves, gh_field=field)
    qtbot.addWidget(window)
    assert window.bins_spin.value() == 80 and window.r0_combo.currentText() == "1/√N"
    h, D, dD, _counts = mc.dh_histogram_errors(h_map, n_bins=80)
    x, y = window._hist_curve.getData()
    np.testing.assert_allclose(x, h)
    np.testing.assert_allclose(y, D, equal_nan=True)
    np.testing.assert_allclose(window._hist_err.opts["height"], np.where(np.isfinite(dD), 2 * dD, 0.0))
    window.bins_spin.setValue(40)
    assert window._hist_curve.getData()[0].size == 40
    window.r0_combo.setCurrentText("r_min/L")
    _h2, D2, _dD2, _ = mc.dh_histogram_errors(h_map, n_bins=40,
                                              r0_rel=mc.relative_scale(1.0, (64, 64)))
    np.testing.assert_allclose(window._hist_curve.getData()[1], D2, equal_nan=True)
    assert not window.per_scale_check.isChecked() and len(window._scale_curves) == 0
    window.per_scale_check.setChecked(True)
    assert len(window._scale_curves) == 3 and all(c.isVisible() for c in window._scale_curves)
    window.per_scale_check.setChecked(False)
    assert not any(c.isVisible() for c in window._scale_curves)
    assert not window.gh_check.isChecked()
    window.gh_check.setChecked(True)
    _gh_h, gh_D, _gh_dD = mc.gradient_histogram_spectrum(field, n_bins=40)
    np.testing.assert_allclose(window._gh_curve.getData()[1], gh_D, equal_nan=True)


def test_box_counting_and_pin_to_d0_in_the_construction_window(qtbot):
    from dynamix.core import microcanonical as mc
    from dynamix.core import spectra

    rng = np.random.default_rng(1)
    h_map = rng.normal(0.4, 0.1, (64, 64))
    hd_std, hd_cmax = _pair()
    window = SpectrumWindow(hd_std, hd_cmax, h_map=h_map, log2_L=6.0)
    qtbot.addWidget(window)
    assert not window.boxcount_check.isChecked() and window._box_curve.getData()[0] is None
    window.boxcount_check.setChecked(True)
    hb, Db, _dDb = mc.box_counting_spectrum(h_map, n_bins=20)       # coarser bins than the histogram
    x, y = window._box_curve.getData()
    np.testing.assert_allclose(x, hb)
    np.testing.assert_allclose(y, Db, equal_nan=True)
    h, D, _dD, _c = mc.dh_histogram_errors(h_map, n_bins=80)
    assert not window.pin_check.isChecked()
    window.pin_check.setChecked(True)
    fit = window.current_fit()
    d0 = -float(np.asarray(fit["tau"])[list(fit["q_list"]).index(0.0)])
    np.testing.assert_allclose(window._hist_curve.getData()[1], D + (d0 - 2.0), equal_nan=True)
    assert any("D₀" in label.text for _sample, label in window._plot.legend.items)
    hist_only = SpectrumWindow(h_map=h_map)
    qtbot.addWidget(hist_only)
    assert not hist_only.pin_check.isEnabled()


def _legend_names(plot):
    return [label.text for _sample, label in plot.legend.items]


def test_the_legend_lists_only_what_is_drawn_and_names_the_source(qtbot):
    from dynamix.core import microcanonical as mc

    rng = np.random.default_rng(1)
    h_map = rng.normal(0.4, 0.1, (64, 64))
    field = np.cumsum(np.cumsum(rng.normal(size=(64, 64)), axis=0), axis=1)
    hd_std, hd_cmax = _pair()
    window = SpectrumWindow(hd_std, hd_cmax, h_map=h_map, gh_field=field,
                            source="moments over the projections (Turiel 2006, M method)")
    qtbot.addWidget(window)
    names = _legend_names(window._plot)
    assert names[0] == "moments over the projections (Turiel 2006, M method)"
    assert "microcanonical histogram (eq. 21)" in names
    for absent in ("gradient histogram (GH)", "box-counting across r (eq. 20)",
                   "dominant chains", "non-dominant chains", "D**(h): concave envelope of fitted τ(q)"):
        assert absent not in names
    window.gh_check.setChecked(True)
    assert "gradient histogram (GH)" in _legend_names(window._plot)
    window.gh_check.setChecked(False)
    assert "gradient histogram (GH)" not in _legend_names(window._plot)
    window._hull_check.setChecked(True)
    assert "D**(h): concave envelope of fitted τ(q)" in _legend_names(window._plot)
    # the histogram as points, the way the paper draws it
    assert window._hist_curve.opts["symbol"] == "o" and window._hist_curve.opts["pen"] is None


def test_the_construction_plot_zooms_per_axis_has_zero_lines_and_draws_the_tangent_across_the_view(qtbot):
    """The wheel zooms h, with Shift D(h) (the Partition functions window's view box); thin lines
    sit at h = 0 and D = 0; the tangent at the phase transition off the current fit spans the
    view and follows it, hidden while the negative branch is concave."""
    from pyqtgraph import InfiniteLine

    from dynamix.shell.multifractal_window import WheelAxisViewBox
    from test_multifractal_window import _broken_pair

    window = SpectrumWindow(*_pair())
    qtbot.addWidget(window)
    assert isinstance(window._plot.getViewBox(), WheelAxisViewBox)
    zeros = [it for it in window._plot.items if isinstance(it, InfiniteLine) and it.value() == 0.0
             and it.angle in (0, 90)]
    assert len(zeros) == 2
    assert not window._tangent_line.isVisible()                      # the clean pair: concave
    broken = SpectrumWindow(*_broken_pair())
    qtbot.addWidget(broken)
    broken._table_combo.setCurrentText("sup (cmax)")
    fit = broken.current_fit()
    expect = spectra.phase_transition_tangent(fit["q_list"], fit["tau"], fit["h"], fit["D"])
    assert expect is not None and broken._tangent_line.isVisible()
    x, y = broken._tangent_line.xData, broken._tangent_line.yData
    assert list(x) == pytest.approx(broken._plot.getViewBox().viewRange()[0])
    assert (y[-1] - y[0]) / (x[-1] - x[0]) == pytest.approx(expect["q_c"])
    broken._plot.setXRange(-0.5, 2.5, padding=0.0)
    assert list(broken._tangent_line.xData) == pytest.approx([-0.5, 2.5])
    assert f"q_c = {expect['q_c']:g}" in broken._tangent_label.textItem.toPlainText()


def test_the_construction_window_hides_the_table_choice_for_one_table(qtbot):
    hd_std, hd_cmax = _pair()
    one = SpectrumWindow(hd_std, hd_std, kind="structure")
    qtbot.addWidget(one)
    assert one._table_combo.isHidden() and one._table_label.isHidden() and one.current_fit() is not None
    two = SpectrumWindow(hd_std, hd_cmax)
    qtbot.addWidget(two)
    assert not two._table_combo.isHidden()


# ------------------------------------------------- beyond the hull: large deviations, γ, kinks

def _per_scale_set(L=128.0, scales=(2.0, 4.0, 8.0, 16.0, 32.0)):
    """An exact per-scale set: at each scale, cells of exponent h on a grid, as many as
    (a/L)^{-D(h)} with D(h) = 2 - 5 (h - ½)², each modulus (a/L)^h."""
    h = np.linspace(0.3, 0.7, 41)
    D = 2.0 - 5.0 * (h - 0.5) ** 2
    mods = []
    for a in scales:
        n = np.maximum(np.round((a / L) ** (-D)), 1).astype(int)
        mods.append(np.repeat((a / L) ** h, n))
    return {"synthetic": {"mods": mods, "scales": np.asarray(scales), "weights": None}}, h, D


def test_the_gaussian_ensemble_has_a_gamma_knob_and_a_per_point_readout(qtbot):
    from dynamix.core.nonconcave import gaussian_ensemble

    sets, _h, _D = _per_scale_set()
    hd_std, hd_cmax = _pair()
    window = SpectrumWindow(hd_std, hd_cmax, log2_L=7.0, per_scale=sets)
    qtbot.addWidget(window)
    assert window.gaussian_check.isEnabled() and not window.gaussian_check.isChecked()
    assert window.gamma_max_spin.value() == 8.0 and window.gamma_steps_spin.value() == 5
    assert list(window._gammas()) == [0.5, 1.0, 2.0, 4.0, 8.0]
    window.gaussian_check.setChecked(True)
    pts = window._gaussian_points
    assert len(pts.data) > 0 and "Ensemble candidates (synthetic, γ ≤ 8)" in _legend_names(window._plot)
    assert window._gamma_readout.text() == f"{len(pts.data)} points · click or hover one for its γ"
    q = hd_std["q_list"]
    lo, hi = window._scale_window()
    grid = np.linspace(q.min() - 16.0, q.max() + 16.0, 81)
    g = gaussian_ensemble(sets["synthetic"]["mods"], sets["synthetic"]["scales"], grid,
                          window._gammas(), lo, hi, length_px=128.0)
    keep = np.isfinite(g["h"]) & np.isfinite(g["D"]) & (g["D"] >= 0.0)
    assert len(pts.data) == int(keep.sum())
    assert all(isinstance(spot.data(), tuple) and len(spot.data()) == 2 for spot in pts.points()[:5])
    window.gamma_max_spin.setValue(4.0)
    window.gamma_steps_spin.setValue(3)
    assert list(window._gammas()) == [1.0, 2.0, 4.0]
    assert "Ensemble candidates (synthetic, γ ≤ 4)" in _legend_names(window._plot)
    point = pts.points()[0]
    window._on_gaussian_point(pts, [point])
    q0, gamma0 = point.data()
    assert window._gamma_readout.text().startswith(f"γ = {gamma0:g}, q = {q0:.2f}: h = ")
    window.gaussian_check.setChecked(False)
    assert len(pts.data) == 0 and window._gamma_readout.text() == ""


def test_kink_bands_shade_the_jump_in_h_and_clear(qtbot):
    from dynamix.core.nonconcave import kink_flags

    hd_std, hd_cmax = _pair()
    window = SpectrumWindow(hd_std, hd_cmax)
    qtbot.addWidget(window)
    assert window.kink_check is not None and window._kink_bands == []
    window.kink_check.setChecked(True)
    flags = kink_flags(window._fit["q_list"], window._fit["tau"])
    assert len(window._kink_bands) == len(flags)
    for band, k in zip(window._kink_bands, flags):
        assert band.getRegion() == pytest.approx((k["h_lo"], k["h_hi"]))
    window.kink_check.setChecked(False)
    assert window._kink_bands == []
    bare = SpectrumWindow(h_map=_h_map())                   # no tables: no kinks, no ensemble
    qtbot.addWidget(bare)
    assert bare.kink_check is None and bare.gaussian_check is None


def test_the_main_window_seeds_gamma_from_a_leaders_result():
    from dynamix.shell.main_window import _eta_note_of

    leaders = {"params": {"gamma": 1.0}, "samples": {}}                    # the Samples contract names the tool
    assert _eta_note_of(leaders) == (1.0, "forward lift: 2^(γj) on the DWT coefficients, γ = 1")
    assert _eta_note_of({"params": {"gamma": 1.0}}) == (0.0, "forward lift: none (η = 0)")
    assert _eta_note_of({"params": {"fracint_alpha": 0.0}}) == (0.0, "forward lift: none (η = 0)")


def test_named_tables_fill_the_construction_windows_choice(qtbot):
    hd_std, hd_cmax = _pair()
    window = SpectrumWindow(hd_std, hd_std, tables={"wse": hd_cmax, "leaders": hd_std})
    qtbot.addWidget(window)
    assert [window._table_combo.itemText(i) for i in range(2)] == ["wse", "leaders"]
    assert window._current_hd() is hd_cmax and not window._one_table
    lone = SpectrumWindow(hd_std, hd_std, tables={"leaders": hd_std})
    qtbot.addWidget(lone)
    assert lone._one_table


def test_the_methods_published_fits_are_drawn_on_the_four_panels_and_named(qtbot):
    """A fit a method's tool publishes lands on τ(q), h(q), c(q) and D(h) (points and the
    Legendre hull) in its own hue, named in the legend; a redraw replaces it, dropping the name
    removes it."""
    q = np.arange(-2.0, 4.01, 0.5)
    fit = {"name": "L1 · leaders", "kind": "structure functions", "q": q, "tau": 0.5 * q - 2,
           "h": np.full(q.size, 0.5), "D": np.full(q.size, 2.0), "cq": np.zeros(q.size),
           "hull": {"h": np.array([0.5, 0.5]), "D": np.array([2.0, 2.0])}}
    window = SpectrumWindow(h_map=_h_map(), fits=[fit])
    qtbot.addWidget(window)
    assert set(window.method_plots) == {"tau", "h", "cq"}
    items = window._fit_items["L1 · leaders"]
    np.testing.assert_allclose(items["tau"].getData()[1], 0.5 * q - 2)
    np.testing.assert_allclose(items["D"].getData()[0], 0.5)
    assert items["tau"] in window.method_plots["tau"].listDataItems() and items["D"] in window._plot.listDataItems()
    names = [name for _item, name in window._in_legend["order"]]
    assert "L1 · leaders" in names and "L1 · leaders (D**, envelope)" not in names   # hulls wait for the switch
    window._hull_check.setChecked(True)
    assert "L1 · leaders (D**, envelope)" in [name for _item, name in window._in_legend["order"]]
    other = {**fit, "name": "L1 · log-cumulants", "tau": 0.4 * q - 2, "hull": None}
    window.set_fits([{**fit, "tau": 0.6 * q - 2}, other])
    np.testing.assert_allclose(window._fit_items["L1 · leaders"]["tau"].getData()[1], 0.6 * q - 2)
    assert len(window._fit_items) == 2 and window._fit_items["L1 · log-cumulants"]["hull"].getData()[0] is None \
        or len(window._fit_items["L1 · log-cumulants"]["hull"].getData()[0]) == 0
    window.set_fits([other])
    assert list(window._fit_items) == ["L1 · log-cumulants"]
    assert "L1 · leaders" not in [name for _item, name in window._in_legend["order"]]


def test_a_leaders_h_map_is_the_gmwlp_and_its_histogram_says_so(qtbot):
    """The eq-21 histogram of a leaders device's h-map is Cardo & Corvalán's GMWLP, Turiel's
    single-resolution density formula on extrapolated exponents: the label names it and its r₀;
    a projection device's h-map keeps the microcanonical label."""
    h_map = np.random.default_rng(0).normal(0.5, 0.1, (64, 64)).astype(np.float32)
    window = SpectrumWindow(h_map=h_map, h_map_method="cwt_leaders")
    qtbot.addWidget(window)
    assert window._hist_label.startswith("GMWLP histogram (Cardo & Corvalán") and "r₀" in window._hist_label
    plain = SpectrumWindow(h_map=h_map, h_map_method="holder_map")
    qtbot.addWidget(plain)
    assert plain._hist_label.startswith("microcanonical histogram")


def test_published_hulls_follow_the_hull_checkbox(qtbot):
    """A published fit's Legendre hull draws only while the construction's own hull is asked
    for: the checkbox rules every hull on the panel."""
    from dynamix.core import spectra
    window = SpectrumWindow(*_pair())
    qtbot.addWidget(window)
    fit = window.current_fit()
    q = np.asarray(fit["q_list"], dtype=np.float64)
    window.set_fits([{"name": "other · std", "q": q, "tau": fit["tau"], "h": fit["h"], "D": fit["D"],
                      "cq": spectra.specific_heat(q, fit["tau"]), "hull": spectra.legendre_dh(q, fit["tau"])}])
    item = window._fit_items["other · std"]["hull"]
    assert not item.isVisible()
    window._hull_check.setChecked(True)
    assert item.isVisible()
    window._hull_check.setChecked(False)
    assert not item.isVisible()


def test_the_four_panels_sit_in_a_two_by_two_grid(qtbot):
    """τ(q) top left, D(h) top right, h(q) bottom left, c(q) bottom right."""
    window = SpectrumWindow(*_pair())
    qtbot.addWidget(window)
    grid = window._glw.ci
    assert grid.getItem(0, 0) is window.method_plots["tau"] and grid.getItem(0, 1) is window._plot
    assert grid.getItem(1, 0) is window.method_plots["h"] and grid.getItem(1, 1) is window.method_plots["cq"]


def test_the_frame_switch_reads_the_exponents_in_the_integrated_frame(qtbot):
    """With η on, the integrated frame keeps the lift: τ differs from the original frame's by η q."""
    window = SpectrumWindow(*_pair())
    qtbot.addWidget(window)
    window._eta_spin.setValue(0.5)
    original = window.current_fit()
    window._frame_combo.setCurrentText("integrated")
    integrated = window.current_fit()
    assert original["frame"] == "original" and integrated["frame"] == "integrated"
    q = np.asarray(original["q_list"], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(integrated["tau"]) - np.asarray(original["tau"]), 0.5 * q, atol=1e-9)
    window.set_frame("original")
    assert window._frame_combo.currentText() == "original"


def test_published_fits_never_set_the_panels_extent(qtbot):
    """A method's published fit (a log-cumulant extrapolation past a phase transition shoots far
    out) draws without widening the view: the window's own curves set the range."""
    window = SpectrumWindow(*_pair())
    qtbot.addWidget(window)
    q = np.arange(-2.0, 4.01, 0.5)
    window.set_fits([{"name": "far", "q": q, "tau": 1e3 * q, "h": 1e3 * np.ones(q.size), "D": 1e3 * np.ones(q.size),
                      "cq": 1e3 * np.ones(q.size), "hull": None}])
    for plot in (window._plot, *window.method_plots.values()):
        bounds = plot.getViewBox().childrenBounds()
        assert bounds[1] is None or bounds[1][1] < 100


def test_the_show_menu_hides_a_published_spectrum(qtbot):
    window = SpectrumWindow(*_pair())
    qtbot.addWidget(window)
    q = np.arange(-2.0, 4.01, 0.5)
    fit = {"name": "L1 · leaders", "q": q, "tau": 0.5 * q - 2, "h": np.full(q.size, 0.5), "D": np.full(q.size, 2.0),
           "cq": np.zeros(q.size), "hull": {"h": np.array([0.4, 0.6]), "D": np.array([2.0, 2.0])}}
    window.set_fits([fit])
    window._hull_check.setChecked(True)
    action = next(a for a in window.show_menu.actions() if a.text() == "L1 · leaders")
    items = window._fit_items["L1 · leaders"]
    assert action.isChecked() and all(items[k].isVisible() for k in ("tau", "h", "cq", "D", "hull"))
    action.setChecked(False)
    assert not any(items[k].isVisible() for k in ("tau", "h", "cq", "D", "hull"))
    assert "L1 · leaders" not in [name for _item, name in window._in_legend["order"]]
    action.setChecked(True)
    assert items["D"].isVisible() and items["hull"].isVisible()
    window.set_fits([])
    assert not [a for a in window.show_menu.actions() if a.text() == "L1 · leaders"]


def test_the_local_cubic_legendre_draws_on_its_switch(qtbot):
    from dynamix.core import spectra
    from test_multifractal_window import _hd_q
    q = np.arange(-2.0, 2.01, 0.5)                        # nine orders: five-point windows exist
    hd = _hd_q(q, 0.5 * q - 0.1 * q * q)
    window = SpectrumWindow(hd, hd)
    qtbot.addWidget(window)
    assert window._local_curve.getData()[0] is None
    window._local_check.setChecked(True)
    fit = window.current_fit()
    local = spectra.legendre_local(fit["q_list"], fit["tau"])
    np.testing.assert_allclose(window._local_curve.getData()[0], local["h"])
    np.testing.assert_allclose(window._local_curve.getData()[1], local["D"])
    assert any("local cubic" in name for _item, name in window._in_legend["order"])
    window._local_check.setChecked(False)
    assert window._local_curve.getData()[0] is None or window._local_curve.getData()[0].size == 0


def test_the_kink_geometry_draws_the_tangents_and_the_chord_and_reads_the_departure(qtbot):
    """Touchette & Beck 2006 Figs. 2 and 3 on the construction: the window's own τ(q), h(q) and
    c(q) sit on the method panels, the two one-sided tangents at q_c on τ(q), the envelope's
    chord across (h_l, h_h) on D(h) once D** is asked for, and a published generalized spectrum
    is read against the envelope for where it leaves it."""
    from test_multifractal_window import _hd_q
    q = np.arange(-3.0, 3.01, 0.5)
    hd = _hd_q(q, np.where(q < 0, 0.8 * q, 0.3 * q))
    window = SpectrumWindow(hd, hd)
    qtbot.addWidget(window)
    fit = window.current_fit()
    np.testing.assert_allclose(window._own_items["tau"].getData()[1], fit["tau"])
    assert window._tangent_left.isVisible() and window._tangent_right.isVisible()
    xl, yl = window._tangent_left.getData()
    assert abs((yl[1] - yl[0]) / (xl[1] - xl[0]) - 0.8) < 1e-6
    xr, yr = window._tangent_right.getData()
    assert abs((yr[1] - yr[0]) / (xr[1] - xr[0]) - 0.3) < 1e-6
    assert not window._chord.isVisible()
    window._hull_check.setChecked(True)
    assert window._chord.isVisible()
    np.testing.assert_allclose(window._chord.getData()[0], [0.3, 0.8], atol=1e-6)
    assert "h_l = 0.300" in window._kink_label.toPlainText() and "h_h = 0.800" in window._kink_label.toPlainText()
    h = np.linspace(0.2, 0.9, 71)
    window.set_fits([{"name": "gen", "kind": "generalized spectrum", "h": h,
                      "D": np.where((h >= 0.4) & (h <= 0.7), -5.0, 5.0)}])
    assert "leaves D** on (0.40, 0.70)" in window._kink_label.toPlainText()
