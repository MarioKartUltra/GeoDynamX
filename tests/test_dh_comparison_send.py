# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Send to D(h) comparison (spec 45): every window that produces a spectrum sends it to the main
window's board; the D(h) construction window draws its own construction beside what was sent,
each entry in its colour with its error bars and tangent, names it in the legend by its display
name, and edits the board in its inspector; the Partition functions window sends and draws no
comparison of its own."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from dynamix.core import spectra
from dynamix.shell.multifractal_window import MultifractalWindow
from dynamix.shell.spectra_board import SpectraBoardModel, landed_dialog
from dynamix.shell.spectrum_window import SpectrumWindow
from tests.test_multifractal_window import _broken_pair, _pair
from tests.test_shell_window import stub_devices, window  # noqa: F401


class _Host(QtWidgets.QWidget):
    """What a fitting window finds above it in the app: the board, the status line and the layers."""

    def __init__(self):
        super().__init__()
        self.spectra_board = SpectraBoardModel(self)
        self.spectra_board.sent.connect(lambda text: self.status.append(text))
        self.status: list = []
        self._layer_by_id: dict = {}

    def _notify(self, text, tier="status"):
        self.status.append(text)


@pytest.fixture
def host(qtbot):
    widget = _Host()
    qtbot.addWidget(widget)
    return widget


def _hist():
    h = np.linspace(0.0, 1.0, 20)
    return h, 2.0 - 8.0 * (h - 0.5) ** 2, np.full(20, 0.05)


def _fitter(qtbot, host, *tables, **kw):
    window = MultifractalWindow(*(tables or _broken_pair()), parent=host, **kw)
    qtbot.addWidget(window)
    return window


def _construction(qtbot, host, *tables, **kw):
    """The D(h) construction window over ``tables`` (the clean pair: concave, no tangent)."""
    window = SpectrumWindow(*(tables or _pair()), parent=host, **kw)
    qtbot.addWidget(window)
    return window


def _landed():
    hd_a, hd_b = _pair()
    return [{"name": "moments · B", "hd_std": hd_b, "hd_cmax": hd_b, "method": "holder_measure",
             "identity": {"recipe": "r", "layer_id": 7, "layer": "B", "filters": "None",
                          "method": {"device": "holder_measure"}}},
            {"name": "dyadic leaders · C", "hd_std": hd_a, "hd_cmax": hd_a, "method": "wavelet_leaders",
             "identity": {"recipe": "s", "layer_id": 8, "layer": "C", "filters": "None",
                          "method": {"device": "wavelet_leaders", "set": "leaders"}}}]


def _choose(row):
    """A QDialog.exec that ticks ``row`` of the list and accepts."""
    def exec_(dialog):
        rows = dialog.findChild(QtWidgets.QListWidget)
        rows.item(row).setCheckState(rows.item(row).checkState().Checked)
        return 1
    return exec_


# ------------------------------------------------------------------- the short method names

@pytest.mark.parametrize("kind, method, table, settings, short", [
    ("canonical", {"device": "wtmm2d"}, "cmax", {}, "WTMM (sup)"),
    ("canonical", {"device": "wtmm2d"}, "std", {}, "WTMM (no sup)"),
    ("canonical", {"device": None}, "cmax", {}, "WTMM (sup)"),
    ("canonical", {"device": "wtmm2d_roi"}, "std", {}, "WTMM (no sup)"),
    ("canonical", {"device": "holder_measure"}, None, {}, "GMWP – measure (moments)"),
    ("canonical", {"device": "holder_multiaffine"}, None, {}, "GMWP – multiaffine (moments)"),
    ("canonical", {"device": "mz_edges"}, "std", {}, None),
    ("histogram", {"device": "wtmm2d"}, None, {}, "WTMM (histogram)"),
    ("boxcount", {"device": "wtmm2d"}, None, {}, "WTMM (box)"),
    ("histogram", {"device": "wavelet_leaders"}, None, {}, "GMWLP"),
    ("boxcount", {"device": "wavelet_leaders"}, None, {}, "GMWLP (box)"),
    ("histogram", {"device": "cwt_leaders", "measure": "leaders"}, None, {}, "GMWLP (cwt)"),
    ("boxcount", {"device": "cwt_leaders", "measure": "wtmm"}, None, {}, "GMWLP (wtmm, box)"),
    ("histogram", {"device": "holder_measure"}, None, {}, "GMWP – measure"),
    ("histogram", {"device": "holder_multiaffine"}, None, {}, "GMWP – multiaffine"),
    ("boxcount", {"device": "holder_measure"}, None, {}, "GMWP – measure (box)"),
    ("dwt_leaders", {"device": "wavelet_leaders", "set": "leaders"}, None, {}, "WL (dwt)"),
    ("dwt_leaders", {"device": "wavelet_leaders", "set": "p-leaders"}, None, {}, "p-L (dwt)"),
    ("dwt_leaders", {"device": "wavelet_leaders", "set": "wse"}, None, {}, "wse-L (dwt)"),
    ("cwt_leaders", {"device": "cwt_leaders", "measure": "leaders"}, None, {}, "WL (cwt)"),
    ("cwt_leaders", {"device": "cwt_leaders", "measure": "wtmm"}, None, {}, "WL (wtmm)"),
    ("generalized", {"device": "wavelet_leaders"}, None, {"set": "p-leaders", "curve": "converged"}, "p-L (dwt) (gen)"),
    ("generalized", {"device": "cwt_leaders", "measure": "leaders"}, None, {"set": "CWT p-leaders", "curve": "converged"},
     "p-L (cwt) (gen)"),
    ("generalized", {"device": "wtmm2d"}, None, {"set": "maxima lines", "curve": "legendre"}, "WTMM (lines) (Legendre)"),
    ("large_deviations", {"device": "wavelet_leaders"}, "WSE leaders", {}, "LD · wse-L (dwt)"),
    ("large_deviations", {"device": "cwt_leaders", "measure": "wtmm"}, "CWT leaders", {}, "LD · WL (wtmm)"),
    ("cumulant", {"device": "wavelet_leaders"}, "leaders", {}, "LC · WL (dwt)"),
    ("structure_functions", {"device": "wavelet_leaders"}, "p-leaders", {}, "WLSF · p-L (dwt)"),
    ("structure_functions", {"device": "wavelet_leaders"}, "coefficients", {}, "SF · coef (dwt)"),
    ("structure_functions", {"device": "cwt_leaders"}, "cwt-leaders", {}, "WLSF · WL (cwt)"),
])
def test_each_method_has_its_short_name(kind, method, table, settings, short):
    from dynamix.shell.spectra_board import short_name

    assert short_name(kind, {"tool": {"method": method}, "table": table, "settings": settings}) == short


def test_the_display_name_is_the_short_name_with_the_scale_window():
    """The short name and `· a lo–hi` (log2 a as an integer or one decimal); the full name keeps
    the producer's name and `log2 a lo–hi`; a method without a short name shows its full name;
    a spectrum sent without a scale window shows the short name alone."""
    from types import SimpleNamespace

    from dynamix.shell.spectra_board import from_fit

    h = np.linspace(0.1, 0.9, 5)
    fit = {"name": "canonical (Arneodo WTMM) · sup (cmax)", "h": h, "D": 1.0 + 0 * h}
    window = SimpleNamespace(parentWidget=lambda: None, _method={"device": "wtmm2d"})
    e = from_fit(window, fit, "canonical", scale_window=(2.0, 5.0), table="cmax")
    assert (e.name, e.full_name) == ("WTMM (sup) · a 2–5", "canonical (Arneodo WTMM) · sup (cmax) · log2 a 2–5")
    assert from_fit(window, fit, "canonical", scale_window=(1.96, 5.25), table="std").name == "WTMM (no sup) · a 2–5.2"
    assert from_fit(window, fit, "canonical", scale_window=(1.44, 4.5), table="std").name == "WTMM (no sup) · a 1.4–4.5"
    assert from_fit(window, {**fit, "name": "hist"}, "histogram").name == "WTMM (histogram)"
    odd = SimpleNamespace(parentWidget=lambda: None, _method={"device": "mz_edges"})
    e = from_fit(odd, fit, "canonical", scale_window=(2.0, 5.0), table="std")
    assert e.name == e.full_name == "canonical (Arneodo WTMM) · sup (cmax) · log2 a 2–5"
    plain = SimpleNamespace(parentWidget=lambda: None, _h_map_method="cwt_leaders")
    assert from_fit(plain, fit, "cwt_leaders", scale_window=(2, 5)).name == "WL (cwt) · a 2–5"


def test_the_method_is_read_off_the_result():
    from dynamix.shell.spectra_board import method_of

    assert method_of({"_execution": [{"device": "dwt2d"}, {"device": "wtmm2d"}, {"device": "chain_filters"}]}) == \
        {"device": "wtmm2d"}
    assert method_of({"record": {"method": "wavelet_leaders"}, "hd_source": "dyadic p-leaders"}) == \
        {"device": "wavelet_leaders", "set": "p-leaders"}
    assert method_of({"record": {"method": "cwt_leaders", "measure": "wtmm"}}) == \
        {"device": "cwt_leaders", "measure": "wtmm"}
    assert method_of({"_execution": [{"device": "holder_multiaffine"}]}) == {"device": "holder_multiaffine"}
    assert method_of({}) == {"device": None} and method_of(None) == {"device": None}


# ---------------------------------------------------------------------------- the senders

def test_the_canonical_fit_carries_the_bars_d_h_draws_and_its_tau(qtbot, host):
    window = _fitter(qtbot, host)
    window._region.setRegion((1.0, 6.0))
    assert window.send_button.text() == "Send to D(h) comparison"
    assert window.send_primary()
    [entry] = host.spectra_board.entries()
    fit = window.current_fit()
    assert entry.kind == "canonical" and entry.h.shape == entry.D.shape == fit["q_list"].shape
    np.testing.assert_allclose(entry.dD, fit["D_err"])          # the fit's standard errors
    np.testing.assert_allclose(entry.dh, fit["h_err"])          # over the window alone
    np.testing.assert_allclose(entry.tau, fit["tau"])
    assert entry.identity["scale_window"] == [1.0, 6.0]
    assert (entry.identity["fit_mode"], entry.identity["table"]) == ("naive", "std")
    assert entry.tangent is None                                  # "with tangent" is off by default
    assert entry.name == "WTMM (no sup) · a 1–6"                   # the short name with the window
    assert entry.full_name == "canonical (Arneodo WTMM) · no-sup (std) · log2 a 1–6"
    assert host.status[-1] == "sent to the D(h) comparison: canonical (Arneodo WTMM) · no-sup (std) · log2 a 1–6"
    window.send_primary()                                         # the same state: replaced
    assert len(host.spectra_board) == 1 and host.status[-1].startswith("sent again")
    window._region.setRegion((2.0, 6.0))                          # a new window: a new entry
    window.send_primary()
    first, second = host.spectra_board.entries()
    assert (first.name, second.name) == ("WTMM (no sup) · a 1–6", "WTMM (no sup) · a 2–6")
    assert second.full_name == "canonical (Arneodo WTMM) · no-sup (std) · log2 a 2–6"


def test_the_displayed_q_rows_are_the_ones_sent(qtbot, host):
    window = _fitter(qtbot, host)
    window._qmin_spin.setValue(-2.0)
    window.send_primary()
    entry = host.spectra_board.entries()[0]
    q = np.asarray(window.current_fit()["q_list"])
    np.testing.assert_allclose(entry.q, q[q >= -2.0])
    assert entry.identity["settings"]["q_range"][0] == -2.0


def test_the_tangent_travels_only_when_ticked(qtbot, host):
    window = _fitter(qtbot, host)
    window._table_combo.setCurrentText("sup (cmax)")              # the broken negative branch
    fit = window.current_fit()
    expect = spectra.phase_transition_tangent(fit["q_list"], fit["tau"], fit["h"], fit["D"])
    window.send_primary()
    assert host.spectra_board.entries()[0].tangent is None
    assert not host.spectra_board.entries()[0].full_name.endswith("tangent")
    window.with_tangent_check.setChecked(True)
    window.send_primary()
    [entry] = host.spectra_board.entries()                        # one identity: replaced
    lo, hi = (round(v, 2) for v in window._region.getRegion())
    lo1, hi1 = (round(v, 1) for v in window._region.getRegion())
    assert entry.name == f"WTMM (sup) · a {lo1:g}–{hi1:g}"
    assert entry.full_name == f"canonical (Arneodo WTMM) · sup (cmax) · log2 a {lo:g}–{hi:g}, tangent"
    assert entry.tangent == pytest.approx({"q_c": expect["q_c"], "h_c": expect["h_c"],
                                           "D_c": expect["D_c"], "slope": expect["q_c"],
                                           "intercept": expect["intercept"]})


def test_the_tangent_mode_off_draws_none_and_sends_none(qtbot, host):
    """Off: no tangent on D(h), no pick line on c(q), and a send with "with tangent" ticked
    carries none; at the kink it rests on the automatic q_c; a drag of the c(q) line turns the
    mode to the dragged q."""
    from dynamix.shell.multifractal_window import TANGENT_MODES

    window = _fitter(qtbot, host)
    window._table_combo.setCurrentText("sup (cmax)")
    fit = window.current_fit()
    auto = spectra.phase_transition_tangent(fit["q_list"], fit["tau"], fit["h"], fit["D"])
    assert window.tangent_combo.currentText() == "at the kink" == TANGENT_MODES[1]
    assert window._tangent_line.isVisible() and window._q_pick_line.isVisible()
    window.tangent_combo.setCurrentText("off")
    assert not window._tangent_line.isVisible() and not window._tangent_marker.isVisible()
    assert not window._q_pick_line.isVisible()
    window.with_tangent_check.setChecked(True)
    window.send_primary()
    assert host.spectra_board.entries()[0].tangent is None
    window._region.setRegion((1.0, 6.0))                         # a refit keeps it off
    assert not window._tangent_line.isVisible()
    window.tangent_combo.setCurrentText("at the kink")
    window._q_pick_line.setValue(-1.3)                           # a drag
    assert window.tangent_combo.currentText() == "at the dragged q"
    assert window._tangent_now()["q_c"] == pytest.approx(-1.5)
    window.tangent_combo.setCurrentText("at the kink")
    fit = window.current_fit()
    auto = spectra.phase_transition_tangent(fit["q_list"], fit["tau"], fit["h"], fit["D"])
    assert window._tangent_now()["q_c"] == pytest.approx(auto["q_c"])
    assert window._q_pick_line.value() == pytest.approx(auto["q_c"])


def test_the_histogram_and_box_counting_constructions_send_with_their_bars(qtbot, host):
    rng = np.random.default_rng(2)
    h, D, dD = _hist()
    window = _fitter(qtbot, host, *_pair(), dh_hist=(h, D, dD), h_map=rng.normal(0.4, 0.1, (64, 64)))
    assert window.send_hist_button.isEnabled() and window.send_box_button.isEnabled()
    window.send_hist_button.click()
    window.send_box_button.click()
    hist, box = host.spectra_board.entries()
    assert (hist.kind, box.kind) == ("histogram", "boxcount")
    np.testing.assert_allclose(hist.D, D)
    np.testing.assert_allclose(hist.dD, dD)
    assert hist.full_name == "microcanonical histogram" and hist.identity["scale_window"] is None
    assert box.h.shape == box.D.shape == box.dD.shape and np.isfinite(box.D).any()
    window.pin_check.setChecked(True)                              # pinned: drawn shifted, a new entry
    window.send_histogram()
    pinned = host.spectra_board.entries()[-1]
    assert len(host.spectra_board) == 3 and pinned.identity["settings"]["pin_to_D0"]
    np.testing.assert_allclose(pinned.D, window._dh_hist_curve.getData()[1])


def test_without_a_construction_its_send_is_greyed_and_says_so(qtbot, host):
    window = _fitter(qtbot, host, *_pair())
    assert not window.send_hist_button.isEnabled() and not window.send_box_button.isEnabled()
    assert not window.send_landed_button.isEnabled()
    assert not window.send_histogram() and host.status[-1] == "no histogram construction to send"
    assert len(host.spectra_board) == 0


def test_the_leaders_table_sends_as_its_leaders(qtbot, host):
    hd_std, _ = _pair()
    for method, kind, short in (("wavelet_leaders", "dwt_leaders", "WL (dwt)"),
                                ("cwt_leaders", "cwt_leaders", "WL (cwt)")):
        window = _fitter(qtbot, host, hd_std, hd_std, h_map_method=method, kind="structure",
                         source=f"L · {method}")
        window._region.setRegion((2.0, 5.0))
        window.send_primary()
        entry = host.spectra_board.entries()[-1]
        assert entry.kind == kind and entry.name == f"{short} · a 2–5"
        assert entry.full_name == f"L · {method}, partition functions · log2 a 2–5"


def test_send_landed_results_lists_the_other_layers_and_sends_the_chosen(qtbot, host, monkeypatch):
    sources = _landed()
    hd_a = sources[1]["hd_std"]
    window = _fitter(qtbot, host, hd_a, hd_a, sources=sources)
    dialog, rows = landed_dialog(window, window._landed)
    assert [rows.item(i).text() for i in range(rows.count())] == ["moments · B", "dyadic leaders · C"]
    dialog.deleteLater()
    monkeypatch.setattr(QtWidgets.QDialog, "exec", _choose(1))
    window._region.setRegion((2.0, 5.0))
    window.send_landed_button.click()
    [entry] = host.spectra_board.entries()
    assert entry.kind == "dwt_leaders" and entry.name == "WL (dwt) · a 2–5"
    assert entry.full_name == "dyadic leaders · C · log2 a 2–5"
    assert entry.identity["tool"] == sources[1]["identity"]
    assert entry.dD is not None and entry.tau is not None


def test_the_other_windows_send_their_spectra(qtbot, host):
    from dynamix.shell.large_deviations_window import LargeDeviationsWindow
    from dynamix.shell.log_cumulants_window import LogCumulantsWindow
    from dynamix.shell.structure_functions_window import StructureFunctionsWindow
    from tests.test_log_cumulants_window import _samples as cascade
    from tests.test_spectrum_window import _per_scale_set
    from tests.test_structure_functions_window import _samples as power_laws

    ld = LargeDeviationsWindow(_per_scale_set()[0], log2_L=7.0, source="L1", parent=host)
    lc = LogCumulantsWindow(cascade(), levels=(1, 5), source="L1 · sym4", parent=host)
    sf = StructureFunctionsWindow(power_laws(), levels=(2, 5), source="L1 · sym4", parent=host)
    for window in (ld, lc, sf):
        qtbot.addWidget(window)
        window.send_button.click()
    got = {e.kind: e for e in host.spectra_board.entries()}
    assert set(got) == {"large_deviations", "cumulant", "structure_functions"}
    for entry in got.values():
        assert entry.h.shape == entry.D.shape and np.isfinite(entry.D).any()
    assert got["large_deviations"].dD is None and got["large_deviations"].tau is None   # no bars
    assert got["large_deviations"].identity["table"] == "synthetic"
    assert got["cumulant"].dD is None and got["cumulant"].tau is not None
    assert got["cumulant"].identity["scale_window"] == [1.0, 5.0]
    sf_fit = sf.current_fit()
    np.testing.assert_allclose(got["structure_functions"].dD, sf_fit["D_err"])
    assert got["structure_functions"].identity["fit_mode"] == "naive"
    assert got["structure_functions"].identity["table"] == "leaders"
    assert any("large deviations" in s for s in host.status)


def _generalized_fit(name="L1 · synthetic, generalized spectrum"):
    """A fit as the Generalized spectrum window keeps it after a Compute: the minima on the h
    grid, those on the grid's boundary flagged, the scale window and the settings."""
    h = np.linspace(0.0, 1.0, 21)
    return {"name": name, "kind": "generalized spectrum", "valid": True, "h": h,
            "D": 2.0 - 5.0 * (h - 0.5) ** 2, "boundary": (h < 0.2) | (h > 0.8), "window": (1.0, 5.0),
            "settings": {"q": np.arange(-4.0, 4.01, 0.5), "gamma": np.arange(0.0, 16.01), "h": h,
                         "delta": np.arange(-1.0, 2.01, 0.25), "lo": 1.0, "hi": 5.0, "L": 128.0,
                         "amplitude": "auto", "set": "leaders"}}


def test_the_generalized_window_sends_the_converged_estimate_and_the_legendre_reference(qtbot, host):
    from dynamix.shell.generalized_spectrum_window import GeneralizedSpectrumWindow
    from tests.test_spectrum_window import _per_scale_set

    w = GeneralizedSpectrumWindow(_per_scale_set()[0], log2_L=7., source="L1", parent=host)
    qtbot.addWidget(w)
    assert not w.send_to_board() and "Compute" in host.status[-1]
    w._fit = fit = _generalized_fit()
    buttons = {b.text(): b for b in w.findChildren(QtWidgets.QPushButton)}
    buttons["Send converged"].click()
    [entry] = host.spectra_board.entries()
    assert entry.kind == "generalized" and entry.h.shape == entry.D.shape == fit["h"].shape
    assert entry.name == "WL (dwt) (gen) · a 1–5"
    assert entry.full_name == "L1 · synthetic, generalized spectrum, converged · log2 a 1–5"
    assert np.isnan(entry.D[fit["boundary"]]).all()                  # bounds stay off the curve
    np.testing.assert_allclose(entry.D[~fit["boundary"]], fit["D"][~fit["boundary"]])
    assert entry.dD is None                                          # no leave-one-scale-out band
    assert entry.identity["scale_window"] == [1.0, 5.0]
    assert entry.identity["settings"]["delta"] == {"min": -1.0, "max": 2.0, "n": 13}
    buttons["Send Legendre (γ = 0)"].click()
    assert len(host.spectra_board) == 1 and "no legendre" in host.status[-1]
    half = np.full(21, 0.03)
    w._fit = {**fit, "legendre": fit["D"] + 0.1, "solid": ~fit["boundary"],
              "band": {"half_width": half}}                          # what the diagnostics add
    w.send_to_board("legendre")
    w.send_to_board("converged")
    converged, legendre = host.spectra_board.entries()
    assert legendre.name == "WL (dwt) (Legendre) · a 1–5" and legendre.dD is None
    assert legendre.full_name.startswith("L1 · synthetic, generalized spectrum, Legendre (γ = 0)")
    np.testing.assert_allclose(legendre.D, fit["D"] + 0.1)
    np.testing.assert_allclose(converged.dD, half)                   # sent again: replaced in place


def test_a_computed_generalized_spectrum_sends_its_converged_points_with_their_bars(qtbot, host):
    """After a Compute the converged estimate is sent as the solid curve draws it, with the
    leave-one-scale-out half-width as its bars."""
    from dynamix.shell.generalized_spectrum_window import GeneralizedSpectrumWindow
    from tests.test_generalized_spectrum_window import LOG2_L, _two_population_sets

    w = GeneralizedSpectrumWindow(_two_population_sets(), log2_L=LOG2_L, parent=host)
    qtbot.addWidget(w)
    w.spins["q min"].setValue(-8.0)
    w.spins["q max"].setValue(8.0)
    with qtbot.waitSignal(w.fitChanged, timeout=60000):
        w.apply_button.click()
    assert w.send_to_board("converged")
    [entry] = host.spectra_board.entries()
    fit = w.current_fit()
    solid = np.asarray(fit["solid"], bool)
    assert solid.sum() > 3 and np.isfinite(entry.D[solid]).all() and np.isnan(entry.D[~solid]).all()
    np.testing.assert_allclose(entry.D[solid], fit["D"][solid])
    np.testing.assert_allclose(entry.dD, fit["band"]["half_width"])


def test_the_kink_readout_reads_the_converged_estimate_never_the_legendre_reference(qtbot, host):
    """The construction's kink reading names where the generalized spectrum leaves D**: the
    converged estimate on the board, never the Legendre (γ = 0) reference sent beside it."""
    from dynamix.shell.generalized_spectrum_window import GeneralizedSpectrumWindow
    from tests.test_multifractal_window import _hd_q
    from tests.test_spectrum_window import _per_scale_set

    q = np.arange(-3.0, 3.01, 0.5)
    hd = _hd_q(q, np.where(q < 0, 0.8 * q, 0.3 * q))
    construction = _construction(qtbot, host, hd, hd)
    construction._hull_check.setChecked(True)
    sender = GeneralizedSpectrumWindow(_per_scale_set()[0], log2_L=7., source="L1", parent=host)
    qtbot.addWidget(sender)
    h = np.linspace(0.2, 0.9, 71)
    sender._fit = {**_generalized_fit(), "h": h, "boundary": np.zeros(h.size, bool),
                   "D": np.where((h >= 0.4) & (h <= 0.7), -5.0, 5.0),
                   "legendre": np.where((h >= 0.3) & (h <= 0.5), -5.0, 5.0)}
    readout = construction._kink_label.toPlainText
    assert sender.send_to_board("legendre")
    assert "leaves D**" not in readout()                      # the reference alone: no departure
    assert sender.send_to_board("converged")
    assert "the generalized spectrum leaves D** on (0.40, 0.70)" in readout()


# --------------------------------------------------------------------------- the comparison

def _entry(name, kind="large_deviations", shift=0.0, **kw):
    from dynamix.model.spectra_board import SentSpectrum

    h = np.linspace(0.1, 0.9, 9)
    return SentSpectrum(name=name, full_name=f"{name} · a long provenance name", kind=kind, h=h,
                        D=2.0 - 5.0 * (h - 0.5) ** 2 + shift, identity={"n": name}, **kw)


def _drawn_curves(plot):
    import pyqtgraph as pg

    return [i for i in plot.listDataItems() if isinstance(i, pg.PlotDataItem) and i.isVisible()
            and i.xData is not None and len(i.xData)]


def _legend(window):
    """The construction's legend: ``(name, tooltip)`` per entry, in order."""
    return [(label.text, label.toolTip()) for _sample, label in window._plot.legend.items]


TANGENT = {"q_c": -1.0, "h_c": 0.9, "D_c": 1.2, "slope": -1.0, "intercept": 2.1}


def test_the_construction_draws_its_own_points_and_exactly_the_visible_entries(qtbot, host):
    window = _construction(qtbot, host)
    d_plot, tau_plot = window._plot, window.method_plots["tau"]
    assert _drawn_curves(d_plot) == [window._canon_curve]        # empty until something is sent
    assert len(_drawn_curves(tau_plot)) == 1
    board = host.spectra_board
    a = board.add(_entry("A"))
    b = board.add(_entry("B", shift=-0.2, q=np.arange(9.0), tau=np.arange(9.0) - 2.0))
    assert len(_drawn_curves(d_plot)) == 3
    assert len(_drawn_curves(tau_plot)) == 2                     # the construction's τ and B's
    board.set_visible(a, False)
    assert len(_drawn_curves(d_plot)) == 2
    curve = window._board_legend[0][0]
    np.testing.assert_allclose(curve.yData, board.get(b).D)
    assert curve.opts["pen"].color().name() == board.get(b).colour
    assert window._canon_err.opts["height"] is not None          # the primary keeps its bars
    board.remove(b)
    board.set_visible(a, True)
    assert len(_drawn_curves(d_plot)) == 2 and window._board_legend[0][1] == "A"
    board.remove(a)
    assert _drawn_curves(d_plot) == [window._canon_curve]


def test_a_construction_draws_as_points_and_a_fit_as_a_line(qtbot, host):
    window = _construction(qtbot, host)
    host.spectra_board.add(_entry("hist", kind="histogram"))
    host.spectra_board.add(_entry("fit", kind="canonical", shift=-0.1))
    (hist, _n, _f), (fit, _m, _g) = window._board_legend
    assert hist.opts["pen"] is None and hist.opts["symbol"] == "o"
    assert fit.opts["pen"] is not None


def test_an_entry_draws_its_bars_and_its_tangent_in_its_colour_while_shown(qtbot, host):
    import pyqtgraph as pg

    window = _construction(qtbot, host)
    d_plot = window._plot
    key = host.spectra_board.add(_entry("A", kind="canonical", dD=np.full(9, 0.05), tangent=TANGENT))
    colour = host.spectra_board.get(key).colour
    bars = [i for _p, i in window._board_items if isinstance(i, pg.ErrorBarItem)]
    assert len(bars) == 1 and bars[0].opts["pen"].color().name() == colour
    np.testing.assert_allclose(bars[0].opts["height"], 0.1)
    [(line, carried)] = window._board_tangents
    assert line.opts["pen"].style() == pg.QtCore.Qt.PenStyle.DashLine
    assert line.opts["pen"].color().name() == colour
    x, y = line.getData()
    np.testing.assert_allclose(x, d_plot.getViewBox().viewRange()[0])
    np.testing.assert_allclose(y, 2.1 - x)
    d_plot.setXRange(-1.0, 3.0, padding=0.0)                    # the view moves: so does the line
    np.testing.assert_allclose(line.getData()[0], [-1.0, 3.0])
    host.spectra_board.set_show_errors(key, False)
    host.spectra_board.set_show_tangent(key, False)              # the tangent off: none drawn
    assert not any(isinstance(i, pg.ErrorBarItem) for _p, i in window._board_items)
    assert window._board_tangents == []
    dashed = [i for i in _drawn_curves(d_plot) if i.opts["pen"] is not None
              and pg.mkPen(i.opts["pen"]).color().name() == colour and pg.mkPen(i.opts["pen"]).style()
              == pg.QtCore.Qt.PenStyle.DashLine]
    assert dashed == []


def test_the_construction_s_own_tangent_can_be_switched_off(qtbot, host):
    """The tangent at the phase transition on the construction's own points draws while
    "tangent at q_c" is ticked (the default); unticked, none is drawn and the record names none."""
    window = _construction(qtbot, host, *_broken_pair())
    window._table_combo.setCurrentText("sup (cmax)")
    assert window.tangent_check.isChecked() and window._tangent_line.isVisible()
    assert window.drawn_fit()["tangent"]["q_c"] == pytest.approx(window._phase_tangent()["q_c"])
    window.tangent_check.setChecked(False)
    assert not any(i.isVisible() for i in (window._tangent_line, window._tangent_marker, window._tangent_label))
    assert window.drawn_fit()["tangent"] is None
    window._lo_spin.setValue(window._lo_spin.value() + 0.25)      # a refit keeps it off
    assert not window._tangent_line.isVisible()
    window.tangent_check.setChecked(True)
    assert window._tangent_line.isVisible()


def test_a_sent_entry_outside_the_view_widens_it_and_the_construction_stays_in_view(qtbot, host):
    """An entry drawn anew outside the D(h) view widens it to the construction and the board's
    points and bars, the opening view (the construction's own range) the least it shows; a rename
    leaves a zoomed view alone; with the far entries gone the view returns to the construction."""
    from dynamix.model.spectra_board import SentSpectrum

    window = _construction(qtbot, host)
    window.show()
    qtbot.waitExposed(window)
    view = window._plot.getViewBox()
    QtWidgets.QApplication.processEvents()
    (x0, x1), (y0, y1) = opening = view.viewRange()
    assert x1 < 2.5 and y1 < 4.0                                 # the high entry starts out of view
    h = np.linspace(2.5, 3.5, 9)
    far = host.spectra_board.add(SentSpectrum(name="far", full_name="far", kind="canonical", h=h,
                                              D=6.0 - (h - 3.0) ** 2, dD=np.full(9, 0.2), identity={"n": "far"}))
    (a0, a1), (b0, b1) = view.viewRange()
    assert a0 <= x0 and a1 >= 3.5 and b0 <= y0 and b1 >= 6.2   # the entry with its bars, and the construction
    window._plot.setXRange(0.3, 0.6, padding=0.0)                # the user zooms in
    window._plot.setYRange(1.0, 2.0, padding=0.0)
    host.spectra_board.rename(far, "renamed")                    # nothing new drawn: the zoom stays
    host.spectra_board.set_colour(far, "#123456")
    assert view.viewRange()[0] == pytest.approx([0.3, 0.6])
    low = host.spectra_board.add(_entry("low", shift=-6.0))      # D around −4: out of the zoomed view
    (a0, a1), (b0, b1) = view.viewRange()
    assert a0 <= x0 and a1 >= 3.5 and b0 <= -4.0 and b1 >= 6.2
    host.spectra_board.remove(far)
    host.spectra_board.remove(low)
    QtWidgets.QApplication.processEvents()
    # the peak's text label spans a few pixels, a width in h and D that follows the zoom
    np.testing.assert_allclose(view.viewRange(), opening, atol=5e-3)


def test_the_legend_shows_the_display_names_with_the_full_names_as_tooltips(qtbot, host):
    window = _construction(qtbot, host)
    key = host.spectra_board.add(_entry("A"))
    legend = _legend(window)
    assert legend[0][0] == "canonical (Arneodo direct)" and legend[-1] == ("A", "A · a long provenance name")
    host.spectra_board.rename(key, "mine")
    assert _legend(window)[-1] == ("mine", "A · a long provenance name")
    host.spectra_board.remove(key)
    assert [name for name, _tip in _legend(window)] == ["canonical (Arneodo direct)"]


def test_a_closed_window_stops_drawing_and_a_reopened_one_catches_up(qtbot, host):
    window = _construction(qtbot, host)
    window.show()
    window.hide()
    host.spectra_board.add(_entry("A"))
    assert window._board_legend == []
    window.show()
    assert [n for _c, n, _f in window._board_legend] == ["A"]


def test_the_construction_sends_its_own_points_histogram_and_box_counting(qtbot, host):
    """The construction window freezes its own state on the board: the canonical points with
    the bars D(h) draws (and the tangent at q_c when "with tangent" is ticked), the histogram
    as drawn with its eq. 39 bars, the box-counting construction with its slopes' errors; the
    canonical points under a second scale window are a second entry."""
    from dynamix.core.microcanonical import box_counting_spectrum

    rng = np.random.default_rng(3)
    window = _construction(qtbot, host, *_broken_pair(), h_map=rng.normal(0.4, 0.1, (64, 64)))
    window._table_combo.setCurrentText("sup (cmax)")              # the broken negative branch
    assert window.send_button.text() == "Send to D(h) comparison"
    window.send_button.click()
    window.send_hist_button.click()
    window.send_box_button.click()
    canon, hist, box = host.spectra_board.entries()
    assert (canon.kind, hist.kind, box.kind) == ("canonical", "histogram", "boxcount")
    fit = window.current_fit()
    for got, key in ((canon.h, "h"), (canon.D, "D"), (canon.dh, "h_err"), (canon.dD, "D_err"), (canon.tau, "tau")):
        np.testing.assert_allclose(got, fit[key])
    assert canon.identity["scale_window"] == list(window._scale_window()) and canon.identity["table"] == "cmax"
    assert canon.tangent is None                                   # "with tangent" is off by default
    h, D = window._hist_curve.getData()
    np.testing.assert_allclose(hist.h, h)
    np.testing.assert_allclose(hist.D, D)
    np.testing.assert_allclose(hist.dD, window._hist_dD)
    expect = box_counting_spectrum(window._h_map, n_bins=max(8, window._bins // 4))
    for got, want in zip((box.h, box.D, box.dD), expect):
        np.testing.assert_allclose(got, want)

    window.with_tangent_check.setChecked(True)
    window.send_primary()
    assert len(host.spectra_board) == 3                            # one identity: replaced
    assert window._tangent_now() is not None
    assert host.spectra_board.entries()[0].tangent == pytest.approx(window._tangent_now())
    window.tangent_check.setChecked(False)                         # the tangent off: none to send
    assert not window.with_tangent_check.isEnabled()
    window.send_primary()
    assert host.spectra_board.entries()[0].tangent is None

    window._lo_spin.setValue(window._lo_spin.value() + 1.0)        # a second scale window
    window.send_primary()
    first, second = [e for e in host.spectra_board.entries() if e.kind == "canonical"]
    assert first.identity["scale_window"] != second.identity["scale_window"]
    assert first.name != second.name and first.name.startswith("WTMM (sup) · a ")
    assert second.name.startswith("WTMM (sup) · a ")
    assert first.full_name != second.full_name
    assert second.full_name.startswith("canonical (Arneodo direct) · sup (cmax) · log2 a ")
    np.testing.assert_allclose(first.D, fit["D"])
    np.testing.assert_allclose(second.D, window.current_fit()["D"])


def test_a_construction_without_tables_or_an_h_map_has_nothing_of_that_to_send(qtbot, host):
    tables_only = _construction(qtbot, host)
    assert tables_only.send_hist_button is None and tables_only.send_box_button is None
    assert not tables_only.send_histogram() and host.status[-1] == "no histogram construction to send"
    rng = np.random.default_rng(4)
    hist_only = SpectrumWindow(h_map=rng.normal(0.4, 0.1, (64, 64)), parent=host)
    qtbot.addWidget(hist_only)
    assert not hist_only.send_button.isEnabled() and not hist_only.with_tangent_check.isEnabled()
    assert not hist_only.send_primary() and host.status[-1] == "no canonical points to send"
    assert len(host.spectra_board) == 0


def test_the_construction_window_sends_the_landed_results_under_its_own_fit(qtbot, host, monkeypatch):
    """"Send landed results…" lists the other layers' landed results; the chosen one is fitted
    under the construction's scale window, fit mode, η and frame, sent, and drawn once sent."""
    sources = _landed()
    hd_a, hd_b = sources[1]["hd_std"], sources[0]["hd_std"]
    assert not _construction(qtbot, host, hd_a, hd_a).send_landed_button.isEnabled()
    window = _construction(qtbot, host, hd_a, hd_a, sources=sources)
    window._lo_spin.setValue(1.0)
    window._hi_spin.setValue(5.0)
    assert window.send_landed_button.isEnabled() and window._board_legend == []
    monkeypatch.setattr(QtWidgets.QDialog, "exec", _choose(0))
    window.send_landed_button.click()
    [entry] = host.spectra_board.entries()
    fit = spectra.fit_spectra(hd_b, 1.0, 5.0)
    assert entry.kind == "canonical" and entry.name == "GMWP – measure (moments) · a 1–5"
    assert entry.full_name == "moments · B · log2 a 1–5"
    np.testing.assert_allclose(entry.D, fit["D"])
    np.testing.assert_allclose(entry.dD, fit["D_err"])
    assert entry.identity["tool"] == sources[0]["identity"] and entry.identity["scale_window"] == [1.0, 5.0]
    assert [n for _c, n, _f in window._board_legend] == ["GMWP – measure (moments) · a 1–5"]
    assert window.send_landed_results(["dyadic leaders · C"]) == 1
    assert host.spectra_board.entries()[-1].kind == "dwt_leaders"


# ---------------------------------------------------------------------------- the inspector

def _rows(window, qtbot):
    inspector = window.board_inspector
    qtbot.waitUntil(lambda: inspector.table.rowCount() == len(window.spectra_board))
    return inspector.table


def test_the_inspector_lists_the_entries_and_edits_them(qtbot, host, monkeypatch):
    from PySide6 import QtCore, QtGui

    window = _construction(qtbot, host)
    I = window.board_inspector
    assert I is not window.spectra_inspector                     # the published fits' list is another
    assert not I.isVisibleTo(window)                             # empty: nothing to inspect
    board = host.spectra_board
    a = board.add(_entry("A", kind="canonical", dD=np.full(9, 0.05), tangent=TANGENT))
    b = board.add(_entry("B"))
    table = _rows(window, qtbot)
    assert I.isVisibleTo(window)
    assert window.layout().indexOf(I) == window.layout().indexOf(window._glw) + 1   # under the panels
    assert [table.item(r, I.NAME).text() for r in range(2)] == ["A", "B"]
    assert all(table.item(1, c).toolTip() == "B · a long provenance name" for c in range(5))
    flags = QtCore.Qt.ItemFlag.ItemIsEnabled
    assert table.item(0, I.ERRORS).flags() & flags and table.item(0, I.TANGENT).flags() & flags
    assert not table.item(1, I.ERRORS).flags() & flags            # B has no bars, no tangent
    assert not table.item(1, I.TANGENT).flags() & flags
    table.item(0, I.SHOWN).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not board.get(a).visible and [n for _c, n, _f in window._board_legend] == ["B"]
    table.item(0, I.SHOWN).setCheckState(QtCore.Qt.CheckState.Checked)
    table.item(0, I.ERRORS).setCheckState(QtCore.Qt.CheckState.Unchecked)
    table.item(0, I.TANGENT).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not board.get(a).show_errors and not board.get(a).show_tangent
    assert window._board_tangents == []
    table.item(1, I.NAME).setText("mine")                         # an edit committed
    assert board.get(b).name == "mine" and board.get(b).full_name == "B · a long provenance name"
    assert _legend(window)[-1] == ("mine", "B · a long provenance name")
    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", lambda *a, **k: QtGui.QColor("#102030"))
    table.cellClicked.emit(0, I.COLOUR)
    assert board.get(a).colour == "#102030"
    qtbot.waitUntil(lambda: table.item(0, I.COLOUR).background().color().name() == "#102030")
    table.cellWidget(1, 5).click()                                # B up
    assert [e.key for e in board.entries()] == [b, a]
    assert [n for _c, n, _f in window._board_legend] == ["mine", "A"]
    table = _rows(window, qtbot)
    qtbot.waitUntil(lambda: table.item(0, I.NAME).text() == "mine")
    table.cellWidget(0, 6).click()                                # and down again
    assert [e.key for e in board.entries()] == [a, b]
    qtbot.waitUntil(lambda: table.item(0, I.NAME).text() == "A")
    table.cellWidget(0, 7).click()                                # remove A
    assert [e.key for e in board.entries()] == [b]
    assert _rows(window, qtbot).rowCount() == 1
    assert [n for _c, n, _f in window._board_legend] == ["mine"]
    table.cellWidget(0, 7).click()                                # and B: the inspector goes
    assert not I.isVisibleTo(window)


def test_a_name_edit_commits_on_enter_and_reverts_on_escape(qtbot, host):
    from PySide6 import QtCore

    window = _construction(qtbot, host)
    window.show()
    qtbot.waitExposed(window)
    key = host.spectra_board.add(_entry("A"))
    table = _rows(window, qtbot)
    item = table.item(0, window.board_inspector.NAME)

    def edit(text, key_):
        table.editItem(item)
        [editor] = [e for e in table.viewport().findChildren(QtWidgets.QLineEdit) if e.isVisible()]
        editor.setText(text)
        qtbot.keyClick(editor, key_)
        qtbot.wait(10)                                            # the closed editor goes

    edit("dropped", QtCore.Qt.Key.Key_Escape)
    assert host.spectra_board.get(key).name == "A"
    edit("kept", QtCore.Qt.Key.Key_Return)
    assert host.spectra_board.get(key).name == "kept"


def test_r_and_the_context_menu_rename_the_selected_row_and_the_legend_follows(qtbot, host):
    """R on the selected row opens its name for editing, as the row's context-menu "Rename…"
    does; the committed name shows in the legend; an empty name restores the producer's."""
    from PySide6 import QtCore

    window = _construction(qtbot, host)
    window.show()
    qtbot.waitExposed(window)
    board = host.spectra_board
    a, b = board.add(_entry("A")), board.add(_entry("B"))
    table = _rows(window, qtbot)
    I = window.board_inspector

    def commit(text):
        [editor] = [e for e in table.viewport().findChildren(QtWidgets.QLineEdit) if e.isVisible()]
        editor.setText(text)
        qtbot.keyClick(editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait(10)

    table.selectRow(1)
    table.setFocus()
    qtbot.keyClick(table, QtCore.Qt.Key.Key_R)
    assert table.state() == QtWidgets.QAbstractItemView.State.EditingState
    assert table.currentRow() == 1 and table.currentColumn() == I.NAME
    commit("by R")
    assert board.get(b).name == "by R" and board.get(b).renamed
    assert _legend(window)[-1] == ("by R", "B · a long provenance name")

    table = _rows(window, qtbot)
    assert table.rowAt(table.visualItemRect(table.item(0, I.NAME)).center().y()) == 0
    [action] = [x for x in I.row_menu(0).actions() if x.text() == "Rename…"]
    action.trigger()
    assert table.state() == QtWidgets.QAbstractItemView.State.EditingState and table.currentRow() == 0
    commit("by menu")
    assert board.get(a).name == "by menu"
    assert [label for label, _tip in _legend(window)][-2:] == ["by menu", "by R"]
    table = _rows(window, qtbot)
    I.rename_row(1)
    commit("")                                                    # empty: the producer's name again
    assert board.get(b).name == "B" and not board.get(b).renamed
    assert _legend(window)[-1] == ("B", "B · a long provenance name")


def test_the_main_window_owns_the_board_and_names_a_send_on_its_status_line(qtbot, window):
    main = window
    assert isinstance(main.spectra_board, SpectraBoardModel)
    fitter = MultifractalWindow(*_pair(), parent=main)
    qtbot.addWidget(fitter)
    fitter.send_primary()
    assert main._status_message.startswith("sent to the D(h) comparison: canonical (Arneodo WTMM)")
