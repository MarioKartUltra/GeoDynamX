# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Send to D(h) comparison (spec 45): every window that produces a spectrum sends it to the main
window's board; the partition fitter draws its own fit beside what was sent, with each entry's
colour, error bars and tangent, names it in the legend by its display name, and edits the board
in its inspector."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from dynamix.core import spectra
from dynamix.shell.multifractal_window import MultifractalWindow
from dynamix.shell.spectra_board import SpectraBoardModel
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
    assert host.status[-1] == "sent to the D(h) comparison: canonical (Arneodo WTMM) · no-sup (std)"
    window.send_primary()                                         # the same state: replaced
    assert len(host.spectra_board) == 1 and host.status[-1].startswith("sent again")
    window._region.setRegion((2.0, 6.0))                          # a new window: a new entry
    window.send_primary()
    assert len(host.spectra_board) == 2


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
    window.with_tangent_check.setChecked(True)
    window.send_primary()
    [entry] = host.spectra_board.entries()                        # one identity: replaced
    assert entry.tangent == pytest.approx({"q_c": expect["q_c"], "h_c": expect["h_c"],
                                           "D_c": expect["D_c"], "slope": expect["q_c"],
                                           "intercept": expect["intercept"]})


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
    for method, kind in (("wavelet_leaders", "dwt_leaders"), ("cwt_leaders", "cwt_leaders")):
        window = _fitter(qtbot, host, hd_std, hd_std, h_map_method=method, kind="structure",
                         source=f"L · {method}")
        window.send_primary()
        entry = host.spectra_board.entries()[-1]
        assert entry.kind == kind and entry.full_name == f"L · {method}, partition functions"


def test_send_landed_results_lists_the_other_layers_and_sends_the_chosen(qtbot, host, monkeypatch):
    hd_a, hd_b = _pair()
    sources = [{"name": "moments · B", "hd_std": hd_b, "hd_cmax": hd_b, "method": "holder_measure",
                "identity": {"recipe": "r", "layer_id": 7, "layer": "B", "filters": "None"}},
               {"name": "dyadic leaders · C", "hd_std": hd_a, "hd_cmax": hd_a, "method": "wavelet_leaders",
                "identity": {"recipe": "s", "layer_id": 8, "layer": "C", "filters": "None"}}]
    window = _fitter(qtbot, host, hd_a, hd_a, sources=sources)
    dialog, rows = window.landed_dialog()
    assert [rows.item(i).text() for i in range(rows.count())] == ["moments · B", "dyadic leaders · C"]
    dialog.deleteLater()

    def choose_second(dialog):
        rows = dialog.findChild(QtWidgets.QListWidget)
        rows.item(1).setCheckState(rows.item(1).checkState().Checked)
        return 1

    monkeypatch.setattr(QtWidgets.QDialog, "exec", choose_second)
    window.send_landed_button.click()
    [entry] = host.spectra_board.entries()
    assert entry.kind == "dwt_leaders" and entry.full_name == "dyadic leaders · C"
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
                         "amplitude": "auto"}}


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
    assert entry.full_name == "L1 · synthetic, generalized spectrum, converged"
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
    assert legendre.full_name.endswith("Legendre (γ = 0)") and legendre.dD is None
    np.testing.assert_allclose(legendre.D, fit["D"] + 0.1)
    np.testing.assert_allclose(converged.dD, half)                   # sent again: replaced in place


def test_the_main_window_owns_the_board_and_names_a_send_on_its_status_line(qtbot, window):
    main = window
    assert isinstance(main.spectra_board, SpectraBoardModel)
    fitter = MultifractalWindow(*_pair(), parent=main)
    qtbot.addWidget(fitter)
    fitter.send_primary()
    assert main._status_message.startswith("sent to the D(h) comparison: canonical (Arneodo WTMM)")
