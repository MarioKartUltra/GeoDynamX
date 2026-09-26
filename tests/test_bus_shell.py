# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Band buses in the shell: the routing dialog, the row menus, and the window flow -- a bus
child whose rack starts with the bus, shown as its own stack, analysed by the tool after it."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets  # noqa: F401

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.shell.bus_dialog import BusDialog


def _send(path, label, band=None):
    return {"path": path, "subdataset": None, "band": band, "label": label}


def _check(dlg, text):
    it = QtWidgets.QTreeWidgetItemIterator(dlg.tree)
    while it.value():
        if it.value().text(0) == text:
            it.value().setCheckState(0, QtCore.Qt.Checked)
            return it.value()
        it += 1
    raise AssertionError(f"no row {text!r}")


def test_ticking_appends_in_order_and_an_off_grid_dataset_cannot_send(qtbot):
    cands = [{"label": "A", "sends": [_send("/a.npz", "A · b1")], "reason": None},
             {"label": "B", "sends": [_send("/b.npz", "B · b1")], "reason": None},
             {"label": "C", "sends": [_send("/c.npz", "C · b1")],
              "reason": "another grid: a 5 x 5 grid, not 4 x 4"}]
    dlg = BusDialog(cands)
    qtbot.addWidget(dlg)
    _check(dlg, "B · b1")
    _check(dlg, "A · b1")
    assert [s["label"] for s in dlg.sends()] == ["B · b1", "A · b1"]
    top_c = dlg.tree.topLevelItem(2)
    assert not (top_c.flags() & QtCore.Qt.ItemIsEnabled)
    assert not (top_c.child(0).flags() & QtCore.Qt.ItemIsUserCheckable)
    # drag-reorder is the list's own move; the send order follows it
    dlg.order.insertItem(0, dlg.order.takeItem(1))
    assert [s["label"] for s in dlg.sends()] == ["A · b1", "B · b1"]
    # unticking removes the send
    dlg.tree.topLevelItem(0).child(0).setCheckState(0, QtCore.Qt.Unchecked)
    assert [s["label"] for s in dlg.sends()] == ["B · b1"]


def test_existing_sends_come_back_ticked_in_their_own_order(qtbot):
    a, b = _send("/a.npz", "A · b1"), _send("/b.npz", "B · b1")
    cands = [{"label": "A", "sends": [a], "reason": None},
             {"label": "B", "sends": [b], "reason": None}]
    dlg = BusDialog(cands, [b, a])
    qtbot.addWidget(dlg)
    assert [s["label"] for s in dlg.sends()] == ["B · b1", "A · b1"]
    assert dlg.tree.topLevelItem(0).child(0).checkState(0) == QtCore.Qt.Checked


from tests.test_shell_window import stub_devices, window  # noqa: E402,F401


def _npz(tmp_path, name, values, x0=500_000.0):
    ny, nx = values.shape[:2]
    f = RasterField(name=name, values=np.asarray(values, dtype=np.float64),
                    frame=LocalFrame(x0=x0, y0=4_000_000.0, dx=30.0, dy=30.0, units="metre"),
                    x_axis=x0 + 30.0 * np.arange(nx) + 15.0,
                    y_axis=4_000_000.0 - 30.0 * np.arange(ny) - 15.0,
                    units="metre", provenance={"crs": "EPSG:32611"})
    p = tmp_path / f"{name}.npz"
    f.save_npz(p)
    return RasterField.from_file(p), str(p)


class _Route:
    """Stands in for BusDialog: records the candidates offered, returns fixed sends."""

    def __init__(self, pick):
        self.pick, self.offered = pick, None

    def __call__(self, candidates, sends=(), **kw):
        self.offered, self.current = candidates, list(sends)
        return self

    def exec(self):
        return 1

    def sends(self):
        return self.pick(self.offered)


@pytest.fixture
def three_datasets(window, qtbot, tmp_path):
    """Two same-grid datasets (the host flat, the other textured) and one off-grid."""
    win = window
    rng = np.random.default_rng(5)
    host, ph = _npz(tmp_path, "host", np.zeros((36, 36)))
    tex, pt = _npz(tmp_path, "tex", rng.normal(size=(36, 36)).cumsum(0))
    far, pf = _npz(tmp_path, "far", np.ones((36, 36)), x0=900_000.0)
    for f, p in ((host, ph), (tex, pt), (far, pf)):
        with qtbot.waitSignal(win.resolved, timeout=10000):
            win.load_field(f, p)
    return win, [l for l in win.project.layers if l.parent_id is None]


def test_routing_builds_a_bus_child_that_shows_its_sends_and_feeds_a_tool(
        three_datasets, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw
    from dynamix.model.device import defaults_for, get_device

    win, (host, tex, far) = three_datasets
    route = _Route(lambda cands: [s for c in cands if c["label"] in ("tex", "host")
                                  for s in c["sends"]][::-1])
    monkeypatch.setattr(mw, "BusDialog", route)
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win._on_bus_requested(host.layer_id)
    reasons = {c["label"]: c["reason"] for c in route.offered}
    assert reasons["host"] is None and reasons["tex"] is None
    assert "anchor" in reasons["far"]                          # same size, somewhere else
    bus = win.layer
    assert bus.parent_id == host.layer_id and bus.chain.steps[0].device == "bus"
    # the bus row shows its OWN stack, with the mixer over its sends
    assert np.asarray(win.canvas._field.values).shape == (36, 36, 2)
    assert win._composite_frame.isVisibleTo(win)
    assert [r[0] for r in win.composite_panel._rows] and len(win.composite_panel._rows) == 2
    # a tool dropped after the bus analyses the sends, not the flat host
    sends_text = bus.chain.steps[0].params["_sends"]
    desc = [{"device": "bus", "params": {"_sends": sends_text}},
            {"device": "pca", "params": defaults_for(get_device("pca"))}]
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    assert win.layer is bus and [s.device for s in bus.chain.steps] == ["bus", "pca"]
    assert len(np.asarray(win._active_result["explained_var_ratio"])) == 2


def test_editing_sends_reroutes_through_the_parameter_path(three_datasets, qtbot,
                                                             monkeypatch):
    import dynamix.shell.main_window as mw

    win, (host, tex, _far) = three_datasets
    both = lambda cands: [s for c in cands if c["label"] in ("host", "tex")  # noqa: E731
                          for s in c["sends"]]
    monkeypatch.setattr(mw, "BusDialog", _Route(both))
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win._on_bus_requested(host.layer_id)
    bus = win.layer
    edit = _Route(lambda cands: list(reversed(edit.current)))
    monkeypatch.setattr(mw, "BusDialog", edit)
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win._on_bus_edit_requested(bus.layer_id)
    assert [s["label"] for s in edit.current] == ["host · band 1", "tex · band 1"]
    stored = json.loads(bus.chain.steps[0].params["_sends"])
    assert [s["label"] for s in stored] == ["tex · band 1", "host · band 1"]


def test_saving_a_temporary_derivative_repoints_the_buses_that_send_from_it(
        three_datasets, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw

    win, (host, tex, _far) = three_datasets
    monkeypatch.setattr(mw, "BusDialog", _Route(
        lambda cands: [s for c in cands if c["label"] == "tex" for s in c["sends"]]))
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win._on_bus_requested(host.layer_id)
    bus = win.layer
    old = json.loads(bus.chain.steps[0].params["_sends"])[0]["path"]
    win._repoint_bus_sends(old, "/kept/tex.npz")
    assert json.loads(bus.chain.steps[0].params["_sends"])[0]["path"] == "/kept/tex.npz"
    assert json.loads(win._params[0]["_sends"])[0]["path"] == "/kept/tex.npz"


def test_the_row_menus_offer_routing_and_a_bus_row_offers_editing(three_datasets,
                                                                   monkeypatch):
    import dynamix.shell.main_window as mw

    win, (host, tex, _far) = three_datasets
    got = []
    win.layer_list.busRequested.disconnect()
    win.layer_list.busRequested.connect(lambda i: got.append(("route", i)))
    menu = win.layer_list._build_source_context_menu(host.source_id)
    next(a for a in menu.actions() if a.text().startswith("Route bands")).trigger()
    assert got == [("route", host.layer_id)]
    item = win.layer_list._layer_items[host.layer_id]
    texts = [a.text() for a in win.layer_list._build_context_menu(item, host.layer_id).actions()]
    assert "Route bands to a new bus…" in texts and "Edit bus sends…" not in texts
