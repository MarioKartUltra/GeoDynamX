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
    w._completed(w._generation, result)
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
    w._completed(w._generation, result)
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


def test_the_spectrum_opens_on_a_fixed_range_with_h_min_and_h_max_marked(qtbot):
    w = _landed(qtbot)
    (x0, x1), (y0, y1) = w.plot.viewRange()
    assert abs(x0 + 1.0) < 1e-6 and abs(x1 - 3.0) < 1e-6 and abs(y0 + 0.5) < 1e-6 and abs(y1 - 2.5) < 1e-6
    assert w.h_min_line.isVisible() and w.h_max_line.isVisible()
    assert abs(w.h_min_line.value() - 0.25) < 1e-9 and abs(w.h_max_line.value() - 0.75) < 1e-9


def test_the_range_sliders_on_the_partition_panel_follow_and_drive_the_scale_spins(qtbot):
    w = _landed(qtbot)
    assert w._region in w.diagram_plot.items                                   # it survives a redraw
    w._region.setRegion((1.5, 2.5))
    assert (w.spins["log₂a min"].value(), w.spins["log₂a max"].value()) == (1.5, 2.5)
    w.spins["log₂a min"].setValue(1.0)
    assert tuple(round(v, 6) for v in w._region.getRegion()) == (1.0, 2.5)
    w.set_scale_window(2.0, 4.0)
    assert tuple(round(v, 6) for v in w._region.getRegion()) == (2.0, 4.0)
