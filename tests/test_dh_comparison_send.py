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


# --------------------------------------------------------------------------- the comparison

def _entry(name, kind="large_deviations", shift=0.0, **kw):
    from dynamix.model.spectra_board import SentSpectrum

    h = np.linspace(0.1, 0.9, 9)
    return SentSpectrum(name=name, full_name=f"{name} · a long provenance name", kind=kind, h=h,
                        D=2.0 - 5.0 * (h - 0.5) ** 2 + shift, identity={"n": name}, **kw)


def _drawn_curves(plot):
    return [i for i in plot.listDataItems() if i.isVisible() and i.xData is not None and len(i.xData)]


def test_the_comparison_draws_the_fit_and_exactly_the_visible_entries(qtbot, host):
    window = _fitter(qtbot, host, *_pair())                       # std: concave, no tangent
    d_plot, tau_plot = window._spectrum_plots["D"], window._spectrum_plots["tau"]
    assert len(_drawn_curves(d_plot)) == 1                       # empty until something is sent
    board = host.spectra_board
    a = board.add(_entry("A"))
    b = board.add(_entry("B", shift=-0.2, q=np.arange(9.0), tau=np.arange(9.0) - 2.0))
    assert len(_drawn_curves(d_plot)) == 3
    assert len(_drawn_curves(tau_plot)) == 2                     # the fit's τ and B's
    board.set_visible(a, False)
    assert len(_drawn_curves(d_plot)) == 2
    curve = window._board_legend[0][0]
    np.testing.assert_allclose(curve.yData, board.get(b).D)
    assert curve.opts["pen"].color().name() == board.get(b).colour
    board.remove(b)
    board.set_visible(a, True)
    assert len(_drawn_curves(d_plot)) == 2 and window._board_legend[0][1] == "A"


def test_an_entry_draws_its_bars_and_its_tangent_in_its_colour_while_shown(qtbot, host):
    import pyqtgraph as pg

    window = _fitter(qtbot, host, *_pair())
    d_plot = window._spectrum_plots["D"]
    tangent = {"q_c": -1.0, "h_c": 0.9, "D_c": 1.2, "slope": -1.0, "intercept": 2.1}
    key = host.spectra_board.add(_entry("A", kind="canonical", dD=np.full(9, 0.05), tangent=tangent))
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
    host.spectra_board.set_show_tangent(key, False)
    assert not any(isinstance(i, pg.ErrorBarItem) for _p, i in window._board_items)
    assert window._board_tangents == []


def test_the_legend_shows_the_display_names_with_the_full_names_as_tooltips(qtbot, host):
    window = _fitter(qtbot, host, *_pair())
    key = host.spectra_board.add(_entry("A"))
    names = window._dh_legend_names()
    assert names[0] == "canonical (Arneodo WTMM)" and names[-1] == "A"
    labels = [i for i in window._dh_legend_row.items if hasattr(i, "text") and i.text == "A"]
    assert labels and labels[0].toolTip() == "A · a long provenance name"
    host.spectra_board.rename(key, "mine")
    assert window._dh_legend_names()[-1] == "mine"


def test_a_closed_window_stops_drawing_and_a_reopened_one_catches_up(qtbot, host):
    window = _fitter(qtbot, host, *_pair())
    window.show()
    window.hide()
    host.spectra_board.add(_entry("A"))
    assert window._board_legend == []
    window.show()
    assert [n for _c, n, _f in window._board_legend] == ["A"]


# ---------------------------------------------------------------------------- the inspector

def _rows(window, qtbot):
    inspector = window.inspector
    qtbot.waitUntil(lambda: inspector.table.rowCount() == len(window.spectra_board))
    return inspector.table


def test_the_inspector_lists_the_entries_and_edits_them(qtbot, host, monkeypatch):
    from PySide6 import QtCore, QtGui

    window = _fitter(qtbot, host, *_pair())
    assert not window.inspector.isVisibleTo(window)              # empty: nothing to inspect
    board = host.spectra_board
    tangent = {"q_c": -1.0, "h_c": 0.9, "D_c": 1.2, "slope": -1.0, "intercept": 2.1}
    a = board.add(_entry("A", kind="canonical", dD=np.full(9, 0.05), tangent=tangent))
    b = board.add(_entry("B"))
    table = _rows(window, qtbot)
    assert window.inspector.isVisibleTo(window)
    I = window.inspector
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
    table.item(1, I.NAME).setText("mine")                         # an edit committed
    assert board.get(b).name == "mine" and board.get(b).full_name == "B · a long provenance name"
    assert window._dh_legend_names()[-1] == "mine"
    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", lambda *a, **k: QtGui.QColor("#102030"))
    table.cellClicked.emit(0, I.COLOUR)
    assert board.get(a).colour == "#102030"
    qtbot.waitUntil(lambda: table.item(0, I.COLOUR).background().color().name() == "#102030")
    table.cellWidget(1, 5).click()                                # B up
    assert [e.key for e in board.entries()] == [b, a]
    table = _rows(window, qtbot)
    qtbot.waitUntil(lambda: table.item(0, I.NAME).text() == "mine")
    table.cellWidget(0, 6).click()                                # and down again
    assert [e.key for e in board.entries()] == [a, b]
    qtbot.waitUntil(lambda: table.item(0, I.NAME).text() == "A")
    table.cellWidget(0, 7).click()                                # remove A
    assert [e.key for e in board.entries()] == [b]
    assert _rows(window, qtbot).rowCount() == 1


def test_a_name_edit_commits_on_enter_and_reverts_on_escape(qtbot, host):
    from PySide6 import QtCore

    window = _fitter(qtbot, host, *_pair())
    window.show()
    qtbot.waitExposed(window)
    key = host.spectra_board.add(_entry("A"))
    table = _rows(window, qtbot)
    item = table.item(0, window.inspector.NAME)

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


def test_the_main_window_owns_the_board_and_names_a_send_on_its_status_line(qtbot, window):
    main = window
    assert isinstance(main.spectra_board, SpectraBoardModel)
    fitter = MultifractalWindow(*_pair(), parent=main)
    qtbot.addWidget(fitter)
    fitter.send_primary()
    assert main._status_message.startswith("sent to the D(h) comparison: canonical (Arneodo WTMM)")
