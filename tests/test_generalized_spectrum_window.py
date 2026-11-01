# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtWidgets

from dynamix.shell import generalized_spectrum_window as gsw
from dynamix.shell.generalized_spectrum_window import GeneralizedSpectrumWindow, _diagnose


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


def test_the_log_scale_diagrams_and_tau_sit_beside_the_spectrum_for_a_chosen_gamma_and_delta(qtbot):
    """What the inf is built from: for the γ and δ picked from the grids, log₂ Z_g(q, a) against
    log₂ a per q with the fitted slope τ_g(q, γ, δ) through the points, and τ_g(q) for every γ
    at that δ, the picked one bold; picking another γ redraws both."""
    scales = 2.**np.arange(1, 6)
    sets = {"leaders": {"scales": scales, "mods": [np.array([a**.4, a**.8, a**.6]) for a in scales]}}
    w = GeneralizedSpectrumWindow(sets, log2_L=7.)
    qtbot.addWidget(w)
    w.spins["q min"].setValue(-2)
    w.spins["q max"].setValue(2)
    w.spins["Δq"].setValue(1.0)
    w.spins["γ max"].setValue(2)
    w.spins["Δδ"].setValue(1.0)
    with qtbot.waitSignal(w.fitChanged, timeout=20000):
        w.apply_button.click()
    g = w.current_fit()["ensemble"]
    n_q, n_g = g["q"].size, g["gamma"].size
    assert w.gamma_combo.count() == n_g and w.delta_combo.count() == g["delta"].size
    assert w.gamma_combo.currentText() == "0" and w.delta_combo.currentText() == "0"
    dots = [it for it in w.diagram_plot.listDataItems() if it.opts.get("symbol")]
    lines = [it for it in w.diagram_plot.listDataItems() if not it.opts.get("symbol")]
    assert len(dots) == n_q and len(lines) == n_q
    di = list(g["delta"]).index(0.0)                             # the picker sits on δ = 0
    np.testing.assert_allclose(dots[0].getData()[0], np.log2(g["scales"]))
    np.testing.assert_allclose(dots[0].getData()[1], g["log_partition"][0, 0, di, :] / np.log(2.0))
    assert len(w.tau_plot.listDataItems()) == n_g
    w.gamma_combo.setCurrentIndex(n_g - 1)
    dots = [it for it in w.diagram_plot.listDataItems() if it.opts.get("symbol")]
    np.testing.assert_allclose(dots[0].getData()[1], g["log_partition"][0, n_g - 1, di, :] / np.log(2.0))
    assert w.diagram_plot.getAxis("left").labelText.startswith("log₂ Z")


def test_the_spectrum_line_stops_at_the_grid_boundary_and_leaves_those_points_as_dots(qtbot):
    """A minimum that sits on the q, γ or δ grid boundary is a bound, never a value: the drawn
    spectrum connects the interior minima only, and the boundary ones stay as loose dots (the
    drawing contract, on a landed result built by hand)."""
    scales = 2.**np.arange(1, 6)
    sets = {"leaders": {"scales": scales, "mods": [np.array([a**.4, a**.8, a**.6]) for a in scales]}}
    w = GeneralizedSpectrumWindow(sets, log2_L=7.)
    qtbot.addWidget(w)
    h = np.linspace(0.0, 1.0, 5)
    D = np.array([-1.0, 0.5, 1.0, 0.5, -1.0])
    boundary = np.array([True, False, False, False, True])
    ensemble = {"q": np.array([-1.0, 0.0, 1.0]), "gamma": np.array([0.0, 1.0]), "delta": np.array([0.0]),
                "scales": scales[:3], "n_scales": 3, "tau_r2": np.ones((3, 2, 1)), "reference_amplitude": 1.0,
                "tau": np.zeros((3, 2, 1)), "log_partition": np.zeros((3, 2, 1, 3))}
    result = {"h": h, "D": D, "boundary": boundary, "q_star": np.zeros(5), "gamma_star": np.zeros(5),
              "delta_star": np.zeros(5), "settings": {"lo": 1.0, "hi": 3.0, "amplitude": 1.0}, "ensemble": ensemble}
    w._completed(w._generation, _diagnose(result))
    _x, y = w.curve.getData()
    assert np.isnan(y[boundary]).all() and np.isfinite(y[~boundary]).all()
    dots_x, _dots_y = w.boundary_points.getData()
    assert dots_x.size == 2 and w.curve.opts["connect"] == "finite"


def _landed(qtbot):
    """A window with a hand-built landed result: two boundary minima at the ends, three inside."""
    scales = 2.**np.arange(1, 6)
    sets = {"leaders": {"scales": scales, "mods": [np.array([a**.4, a**.8, a**.6]) for a in scales]}}
    w = GeneralizedSpectrumWindow(sets, log2_L=7.)
    qtbot.addWidget(w)
    h = np.linspace(0.0, 1.0, 5)
    D = np.array([-1.0, 0.5, 1.0, 0.5, -1.0])
    boundary = np.array([True, False, False, False, True])
    q = np.array([-1.0, 0.0, 1.0])
    tau = np.stack([0.5 * q, 0.4 * q + 0.1 * q * q], axis=1)[:, :, None]      # (n_q, n_γ, n_δ)
    ensemble = {"q": q, "gamma": np.array([0.0, 1.0]), "delta": np.array([0.0]),
                "scales": scales[:3], "n_scales": 3, "tau_r2": np.ones((3, 2, 1)), "reference_amplitude": 1.0,
                "tau": tau, "log_partition": np.zeros((3, 2, 1, 3))}
    result = {"h": h, "D": D, "boundary": boundary, "q_star": np.zeros(5), "gamma_star": np.zeros(5),
              "delta_star": np.zeros(5), "settings": {"lo": 1.0, "hi": 3.0, "amplitude": 1.0}, "ensemble": ensemble}
    w._completed(w._generation, _diagnose(result))
    return w


def test_the_four_panels_sit_in_a_two_by_two_grid_with_the_specific_heat(qtbot):
    from dynamix.core import spectra
    w = _landed(qtbot)
    grid = w._glw.ci
    assert grid.getItem(0, 0) is w.plot and grid.getItem(0, 1) is w.diagram_plot
    assert grid.getItem(1, 0) is w.tau_plot and grid.getItem(1, 1) is w.cq_plot
    items = w.cq_plot.listDataItems()
    assert len(items) == 2                                                   # one c_γ(q) per γ
    g = w.current_fit()["ensemble"]
    np.testing.assert_allclose(items[1].getData()[1], spectra.specific_heat(g["q"], g["tau"][:, 1, 0]), equal_nan=True)


def test_the_spectrum_opens_on_a_fixed_range_with_the_grid_bracketed_range_marked(qtbot):
    """The dashed lines bound the h range whose minima lie inside the (q, γ, δ) grid; they carry
    no labels of their own, one "grid-bracketed" caption sits between them, and the readout
    names the range in words."""
    w = _landed(qtbot)
    (x0, x1), (y0, y1) = w.plot.viewRange()
    assert abs(x0 + 1.0) < 1e-6 and abs(x1 - 3.0) < 1e-6 and abs(y0 + 0.5) < 1e-6 and abs(y1 - 2.5) < 1e-6
    lo, hi = w.bracket_lines
    assert lo.isVisible() and hi.isVisible() and not hasattr(lo, "label") and not hasattr(hi, "label")
    assert abs(lo.value() - 0.25) < 1e-9 and abs(hi.value() - 0.75) < 1e-9
    assert w.bracket_caption.isVisible() and abs(w.bracket_caption.value() - 0.5) < 1e-9
    assert w.bracket_caption.label.toPlainText() == "grid-bracketed"
    assert "minima inside the grid for h in [0.25, 0.75]; outside, the drawn values are upper bounds" in w.readout.text()
    assert w.current_fit()["grid_bracketed"] == (0.25, 0.75)


def test_the_range_sliders_on_the_partition_panel_follow_and_drive_the_scale_spins(qtbot):
    w = _landed(qtbot)
    assert w._region in w.diagram_plot.items                                   # it survives a redraw
    w._region.setRegion((1.5, 2.5))
    assert (w.spins["log₂a min"].value(), w.spins["log₂a max"].value()) == (1.5, 2.5)
    w.spins["log₂a min"].setValue(1.0)
    assert tuple(round(v, 6) for v in w._region.getRegion()) == (1.0, 2.5)
    w.set_scale_window(2.0, 4.0)
    assert tuple(round(v, 6) for v in w._region.getRegion()) == (2.0, 4.0)


def _diagram_lines(w):
    dots = [it for it in w.diagram_plot.listDataItems() if it.opts.get("symbol")]
    lines = [it for it in w.diagram_plot.listDataItems() if not it.opts.get("symbol")]
    return dots, lines


def _slope(item):
    x, y = item.getData()
    return (y[-1] - y[0]) / (x[-1] - x[0])


def test_moving_the_scale_range_refits_the_partition_lines_at_once(qtbot):
    """The dashed lines on the log₂ Z_g panel follow the range sliders as they move: a per-q
    least squares over the stored per-scale log₂ Z inside the window, drawn across the window's
    scales; the spectrum itself waits for Compute (the fit is retracted meanwhile)."""
    scales = 2.**np.arange(1, 6)
    sets = {"leaders": {"scales": scales, "mods": [np.array([a**.4, a**.8, a**.6]) for a in scales]}}
    w = GeneralizedSpectrumWindow(sets, log2_L=7.)
    qtbot.addWidget(w)
    w.spins["q min"].setValue(-2)
    w.spins["q max"].setValue(2)
    w.spins["Δq"].setValue(1.0)
    w.spins["Δδ"].setValue(1.0)
    with qtbot.waitSignal(w.fitChanged, timeout=20000):
        w.apply_button.click()
    dots, lines = _diagram_lines(w)
    x, y = dots[0].getData()
    assert _slope(lines[0]) == pytest.approx(np.polyfit(x, y, 1)[0])
    w._region.setRegion((2.0, 4.0))                              # three of the five scales
    assert w.current_fit() is None                               # D(h) waits for Compute
    dots, lines = _diagram_lines(w)
    assert len(dots) == len(lines) == 5
    x, y = dots[0].getData()
    inside = (x >= 2.0) & (x <= 4.0)
    assert _slope(lines[0]) == pytest.approx(np.polyfit(x[inside], y[inside], 1)[0])
    np.testing.assert_allclose(lines[0].getData()[0], [2.0, 4.0])
    w.spins["log₂a max"].setValue(3.0)                           # the spins drive the lines too
    _, lines = _diagram_lines(w)
    inside = (x >= 2.0) & (x <= 3.0)
    assert _slope(lines[0]) == pytest.approx(np.polyfit(x[inside], y[inside], 1)[0])
    w.spins["log₂a max"].setValue(2.5)                           # one scale: no line to draw
    assert _diagram_lines(w)[1] == []


def test_a_recompute_over_a_narrower_window_keeps_every_scale_on_the_partition_panel(qtbot):
    """The per-scale log₂ Z does not depend on the window, so a narrower Compute still shows
    every scale as a dot and fits the lines over the window's scales alone."""
    scales = 2.**np.arange(1, 6)
    sets = {"leaders": {"scales": scales, "mods": [np.array([a**.4, a**.8, a**.6]) for a in scales]}}
    w = GeneralizedSpectrumWindow(sets, log2_L=7.)
    qtbot.addWidget(w)
    w.spins["q min"].setValue(-2)
    w.spins["q max"].setValue(2)
    w.spins["Δq"].setValue(1.0)
    w.spins["Δδ"].setValue(1.0)
    w.set_scale_window(2.0, 4.0)
    with qtbot.waitSignal(w.fitChanged, timeout=20000):
        w.apply_button.click()
    g = w.current_fit()["ensemble"]
    assert g["n_scales"] == 3 and g["scales"].size == 5
    dots, lines = _diagram_lines(w)
    np.testing.assert_allclose(dots[0].getData()[0], np.log2(scales))
    x, y = dots[0].getData()
    inside = (x >= 2.0) & (x <= 4.0)
    assert _slope(lines[0]) == pytest.approx(np.polyfit(x[inside], y[inside], 1)[0])
    assert _slope(lines[0]) == pytest.approx(g["tau"][0, 0, list(g["delta"]).index(0.0)] )
    assert "3 of 5 scales" in w.readout.text()


# ------------------------------------------------- diagnostics drawn with the estimate

def _points(item) -> int:
    x, _ = item.getData()
    return 0 if x is None else x.size


def test_a_gamma_capped_minimum_joins_the_curve_once_it_stops_falling(qtbot):
    """τ = ½q − 1 − 0.05q² + s γ on q ∈ [−2, 2], γ ∈ {0..3}: at h = 0.45 and 0.5 the profile
    falls with γ (s = 0.28) and the minimum sits on the largest γ, so those are open circles; at
    h = 0.6 it rises and the γ = 0 minimum is on the curve. With s = 0.255 the fall at h = 0.5
    is 0.005 per step, under the tolerance: the capped minimum is converged and drawn on the
    curve. No point sits on a q or δ limit, so there are no loose dots."""
    from dynamix.core.nonconcave import generalized_legendre
    scales = 2.**np.arange(1, 6)
    sets = {"leaders": {"scales": scales, "mods": [np.array([a**.4, a**.8, a**.6]) for a in scales]}}
    q, g, h = np.arange(-2.0, 2.001, 0.5), np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.45, 0.5, 0.6])
    for s, falling in ((0.28, [True, True, False]), (0.255, [True, False, False])):
        w = GeneralizedSpectrumWindow(sets, log2_L=7.)
        qtbot.addWidget(w)
        tau = ((0.5 * q - 1.0 - 0.05 * q * q)[:, None] + s * g[None, :])[:, :, None]      # one δ
        fit = generalized_legendre(tau, q, g, h, delta_list=[0.0])
        ensemble = {"q": q, "gamma": g, "delta": np.array([0.0]), "scales": scales[:3], "n_scales": 3,
                    "tau_r2": np.ones(tau.shape), "reference_amplitude": 1.0, "tau": tau,
                    "log_partition": np.zeros(tau.shape + (3,))}
        w._completed(w._generation, _diagnose({**fit, "ensemble": ensemble,
                                               "settings": {"lo": 1.0, "hi": 3.0, "amplitude": 1.0}}))
        assert fit["gamma_cap"][1] and not fit["boundary"][2]
        x_open, _ = w.open_points.getData()
        np.testing.assert_allclose(x_open, h[falling])
        _, y = w.curve.getData()
        assert np.isfinite(y).tolist() == [not f for f in falling]
        assert _points(w.boundary_points) == 0
        assert f"{3 - sum(falling)} of 3 h converged in γ" in w.readout.text()
        assert w.current_fit()["gamma_converged"]["n"] == 3 - sum(falling)


SCALES = 2.0 ** np.arange(1, 6.01, 0.5)                 # eleven scales, 2 to 64 px
LOG2_L = 8.0


def _cascade(rng, h0, n0, lam2=0.04, dmax=2.0):
    """Log-normal cascade moduli: at scale a about n0 (a/L)^−dmax of them, (a/L)^h with
    h ~ N(h0, lam2 / |ln(a/L)|), so D(h) = dmax − (h − h0)²/(2 lam2), concave."""
    L = 2.0 ** LOG2_L
    out = []
    for a in SCALES:
        x = np.log(a / L)
        out.append(np.exp(rng.normal(h0, np.sqrt(lam2 / -x), max(int(n0 * (a / L) ** -dmax), 20)) * x))
    return out


def _concave_sets():
    return {"cascade": {"scales": SCALES, "mods": _cascade(np.random.default_rng(0), 0.65, 0.5), "weights": None}}


def _two_population_sets():
    """Two cascades with exponents 0.35 and 0.95 mixed at every scale: D(h) is the larger of two
    caps, nonconcave between them, a dip of about 0.5 below the hull at h ≈ 0.65."""
    rng = np.random.default_rng(0)
    mods = [np.concatenate(pair) for pair in zip(_cascade(rng, 0.35, 0.25), _cascade(rng, 0.95, 0.25))]
    return {"two populations": {"scales": SCALES, "mods": mods, "weights": None}}


def _computed(qtbot, sets, window=None, q_max=8.0):
    """Computed over q in [-q_max, q_max]; None keeps the window's default q grid."""
    w = GeneralizedSpectrumWindow(sets, log2_L=LOG2_L)
    qtbot.addWidget(w)
    if q_max is not None:
        w.spins["q min"].setValue(-q_max)
        w.spins["q max"].setValue(q_max)
    if window is not None:
        w.set_scale_window(*window)
    with qtbot.waitSignal(w.fitChanged, timeout=60000):
        w.apply_button.click()
    w.show()
    QtWidgets.QApplication.processEvents()
    return w


def _labels(w):
    """The visible text on the spectrum panel, as scene rectangles: the lines' labels, any text
    item in the view and the title row."""
    texts = [it.label for it in w.plot.items if isinstance(it, pg.InfiniteLine) and hasattr(it, "label")
             and it.isVisible()]
    texts += [it for it in w.plot.items if isinstance(it, pg.TextItem) and it.isVisible()]
    if w.plot.titleLabel.isVisible() and w.plot.titleLabel.item.toPlainText():
        texts.append(w.plot.titleLabel.item)
    return [(t.toPlainText(), t.sceneBoundingRect()) for t in texts]


def _settle():
    """After a resize or a retitle the panels take their new layout over a few passes of the event loop."""
    for _ in range(5):
        QtWidgets.QApplication.processEvents()


def _no_overlap(labels):
    return all(not a.intersects(b) for i, (_, a) in enumerate(labels) for _, b in labels[i + 1:])


def test_a_concave_cascade_has_no_hollow_and_a_narrow_bracket_has_one_label(qtbot, monkeypatch):
    """A log-normal cascade: no hollow beyond the jackknife band, the Legendre curve beneath the
    estimate, a band from eleven refits. A bracketed range narrower than the labels (1.50 to
    1.56) still shows one caption, between the lines, overlapping nothing."""
    monkeypatch.setattr(gsw, "_grid_bracketed", lambda h, D, boundary: (1.50, 1.56))
    w = _computed(qtbot, _concave_sets())
    fit = w.current_fit()
    assert fit["hollows"] == [] and "no hollow beyond the jackknife band" in w.readout.text()
    assert fit["jackknife"]["n_refits"] == 11 and fit["jackknife"]["max_half_width"] > 0
    assert _points(w.band_lo) > 0 and w.band in w.plot.items
    x, y = w.legendre_curve.getData()
    assert x.size == fit["h"].size and np.isfinite(y).sum() > 10
    assert fit["gamma_converged"]["of"] == np.isfinite(fit["D"]).sum()
    assert f"{fit['gamma_converged']['n']} of {fit['gamma_converged']['of']} h converged in γ" in w.readout.text()
    assert "minima inside the grid for h in [1.5, 1.56]" in w.readout.text() and fit["grid_bracketed"] == (1.50, 1.56)
    labels = _labels(w)
    assert [text for text, _ in labels] == ["grid-bracketed", "no hollow beyond the jackknife band"]
    assert not any(hasattr(line, "label") for line in w.bracket_lines)
    assert abs(w.bracket_caption.value() - 1.53) < 1e-9 and _no_overlap(labels)


def test_two_populations_leave_a_hollow_that_survives_the_jackknife(qtbot):
    """Two cascades mixed: the generalized estimate enters the dip between them where the
    Legendre spectrum (γ = 0) draws the flat envelope, and the gap below the hull clears twice
    the leave-one-scale-out half-width over a stretch between the two exponents: one hollow,
    shaded, captioned, recorded; its caption and the bracket's do not overlap."""
    w = _computed(qtbot, _two_population_sets())
    fit = w.current_fit()
    [hol] = [x for x in fit["hollows"] if x["survives"]]
    assert 0.4 < hol["h_lo"] < 0.65 < hol["h_hi"] < 0.9 and hol["depth"] > 0.2
    assert f"hollow: h ∈ [{hol['h_lo']:.3g}, {hol['h_hi']:.3g}], depth {hol['depth']:.2f}, survives the jackknife" \
        in w.readout.text()
    regions = [it for it in w.plot.items if isinstance(it, pg.LinearRegionItem)]
    assert [tuple(round(v, 9) for v in r.getRegion()) for r in regions] == \
        [(round(x["h_lo"], 9), round(x["h_hi"], 9)) for x in fit["hollows"]]
    h, D = fit["h"], fit["D"]
    legendre = w.legendre_curve.getData()[1]
    inside = (h > hol["h_lo"]) & (h < hol["h_hi"]) & np.isfinite(D)
    assert np.all(D[np.isfinite(D)] <= legendre[np.isfinite(D)] + 1e-9)       # the infimum over more γ
    assert np.max(legendre[inside] - D[inside]) > 0.2
    lo, hi = w.band_lo.getData()[1], w.band_hi.getData()[1]
    assert lo.size > 0 and np.all(hi >= lo)
    assert "leave-one-scale-out" in [label.text for _, label in w.legend.items]
    texts = [text for text, _ in _labels(w)]
    assert "grid-bracketed" in texts and any(t.startswith("hollow: h ∈") and "survives" in t for t in texts)
    assert _no_overlap(_labels(w))


def test_on_the_default_q_grid_two_hollow_captions_overlap_nothing_and_stand_clear_of_the_data(qtbot):
    """On the default q grid (±4) a q-limit dot splits the two-population dip into two hollows.
    Their captions overlap neither each other nor the "grid-bracketed" caption, none of them
    lies over the plotted data, and their length does not move the panels' column split."""
    w = _computed(qtbot, _two_population_sets(), q_max=None)
    w.resize(1400, 900)
    _settle()
    fit = w.current_fit()
    assert w.spins["q min"].value() == -4 and w.spins["q max"].value() == 4 and len(fit["hollows"]) >= 2
    labels = _labels(w)
    assert _no_overlap(labels)
    lines = [line for text, _ in labels for line in text.splitlines() if line.startswith("hollow")]
    assert lines == [gsw._hollow_text(hol) for hol in fit["hollows"]]
    view = w.plot.vb.sceneBoundingRect()
    assert not any(rect.intersects(view) for text, rect in labels if text.startswith("hollow"))
    w._retract()
    _settle()
    assert abs(w.plot.vb.sceneBoundingRect().width() - view.width()) < 1.0


def test_the_jackknife_is_skipped_below_five_scales_and_says_so(qtbot):
    w = _computed(qtbot, _concave_sets(), window=(1.0, 2.5))
    fit = w.current_fit()
    assert fit["ensemble"]["n_scales"] == 4 and fit["jackknife"] is None
    assert "leave-one-scale-out skipped: 4 scales in the window, 5 needed" in w.readout.text()
    assert _points(w.band_lo) == 0
    assert "leave-one-scale-out" not in [label.text for _, label in w.legend.items]
