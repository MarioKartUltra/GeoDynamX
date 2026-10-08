# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The inpainting tools in the shell: sends with roles on any step that declares them, the
Mask mode, the output rows and Show, the histograms window, the readings and the
reconstruction note."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets  # noqa: F401

pytest.importorskip("numba")
pytest.importorskip("pywt")

from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402
from dynamix.model.device import defaults_for, get_device  # noqa: E402

FAST = {"n_iter": 3, "heldout_blocks": 2, "heldout_side": 4}


def _utm(values, x0=500_000.0, y0=4_000_000.0, px=30.0, name="f"):
    ny, nx = values.shape[:2]
    return RasterField(name=name, values=np.asarray(values, dtype=np.float64),
                       frame=LocalFrame(x0=x0 + px / 2, y0=y0 - px / 2, dx=px, dy=px, units="m"),
                       x_axis=x0 + px * (np.arange(nx) + 0.5), y_axis=y0 - px * (np.arange(ny) + 0.5),
                       units="m", provenance={"crs": "EPSG:32615"})


def _npz(tmp_path, name, values, **kw):
    f = _utm(values, name=name, **kw)
    p = tmp_path / f"{name}.npz"
    f.save_npz(p)
    return RasterField.from_file(p), str(p)


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def win(qtbot, builtins, tmp_path):
    """A holed host dataset and a 2x coarser prior dataset, both loaded from files."""
    from dynamix.shell.main_window import MainWindow

    rng = np.random.default_rng(4)
    truth = rng.standard_normal((48, 56)).cumsum(0).cumsum(1)
    holed = truth.copy()
    holed[16:28, 20:36] = np.nan
    host, ph = _npz(tmp_path, "host", holed)
    prior, pp = _npz(tmp_path, "prior", truth.reshape(24, 2, 28, 2).mean(axis=(1, 3)), px=60.0)
    w = MainWindow(steps=())
    qtbot.addWidget(w)
    for f, p in ((prior, pp), (host, ph)):
        with qtbot.waitSignal(w.resolved, timeout=20000):
            w.load_field(f, p)
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    w._truth = truth
    return w


def _drop(win, qtbot, device, **params):
    desc = [{"device": device, "params": {**defaults_for(get_device(device)), **params}}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def _layer_named(win, name):
    return next(l for l in win.project.layers if l.name == name)


class _Pick:
    """Stands in for SendsDialog: records what was offered, returns the chosen sends."""

    def __init__(self, choose):
        self.choose, self.offered, self.relation_of = choose, None, None

    def __call__(self, candidates, sends=(), *, relation_of=None, **kw):
        self.offered, self.current, self.relation_of = candidates, list(sends), relation_of
        return self

    def exec(self):
        return 1

    def sends(self):
        return self.choose(self.offered)


# ---------------------------------------------------------------- 7.1 sends

def test_the_sends_dialog_carries_roles_bindings_and_relations(qtbot):
    from dynamix.shell.sends_dialog import SendsDialog

    a = {"path": "/a.tif", "subdataset": None, "band": None, "label": "A · b1"}
    b = {"path": "/b.tif", "subdataset": None, "band": None, "label": "B · b1"}
    cands = [{"label": "A", "sends": [a], "reason": None},
             {"label": "B", "sends": [b], "reason": None}]
    dlg = SendsDialog(cands, [dict(a, role="value")],
                      relation_of=lambda s: "nested x2" if s["path"] == "/b.tif" else "same")
    qtbot.addWidget(dlg)
    assert "same" in dlg.tree.topLevelItem(0).child(0).text(0)
    assert "nested x2" in dlg.tree.topLevelItem(1).child(0).text(0)
    dlg.tree.topLevelItem(1).child(0).setCheckState(0, QtCore.Qt.Checked)
    rows = dlg.rows()
    assert [r.label for r in rows] == ["A · b1", "B · b1"]
    assert rows[0].role.currentText() == "value"
    rows[1].role.setCurrentText("weights")
    assert rows[1].target.isEnabled() and [rows[1].target.itemText(i) for i in range(rows[1].target.count())] == ["target", "A · b1"]
    rows[1].target.setCurrentText("A · b1")
    rows[0].shade.setChecked(True)
    out = dlg.sends()
    assert out[0]["role"] == "value" and out[0]["shade"] is True
    assert out[1]["role"] == "weights" and out[1]["for"] == "A · b1"
    rows[0].role.setCurrentText("structure")
    assert not rows[0].target.isEnabled()
    # re-opening with the weights send listed BEFORE the send it weights keeps the binding
    again = SendsDialog(cands, [dict(b, role="weights", **{"for": "A · b1"}), dict(a, role="value")])
    qtbot.addWidget(again)
    out = again.sends()
    assert out[0]["role"] == "weights" and out[0]["for"] == "A · b1" and out[1]["role"] == "value"


def test_editing_sends_on_an_inpaint_step_writes_the_param_and_the_run_reads_the_prior(
        win, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw

    host = _layer_named(win, "host")
    win.layer_list.select_layer(host.layer_id)
    child = _drop(win, qtbot, "inpaint", **FAST)
    assert win._send_steps(child) == [(0, child.chain.steps[0])]
    pick = _Pick(lambda cands: [dict(s, role="value") for c in cands if c["label"] == "prior"
                                for s in c["sends"]])
    monkeypatch.setattr(mw, "SendsDialog", pick)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_sends_edit_requested(child.layer_id)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    labels = {c["label"] for c in pick.offered}
    assert {"prior", "host"} <= labels
    assert pick.relation_of(next(s for c in pick.offered if c["label"] == "prior"
                                 for s in c["sends"])) == "nested x2"
    sends = json.loads(child.chain.steps[0].params["_sends"])
    assert sends[0]["role"] == "value" and sends[0]["path"].endswith("prior.npz")
    assert sends[0]["label"] == "prior · band 1"              # sends_for_field's band label
    assert win._active_result["_grid_relations"]["prior · band 1"]["kind"] == "nested"
    assert "prior · band 1 nested x2" in win.strips.strip(0).reading_label.text()


def test_sends_land_in_the_right_slot_when_a_bypassed_step_precedes_the_inpaint(win, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw

    host = _layer_named(win, "host")
    win.layer_list.select_layer(host.layer_id)
    child = _drop(win, qtbot, "inpaint", **FAST)
    desc = [{"device": "noise", "params": defaults_for(get_device("noise")), "bypassed": True},
            {"device": "inpaint", "params": {**defaults_for(get_device("inpaint")), **FAST}}]
    win.strips.set_steps(desc, field=win.field)
    win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert [s.device for s in child.chain.steps] == ["inpaint"] and win._names == ["noise", "inpaint"]
    pick = _Pick(lambda cands: [dict(s, role="value") for c in cands if c["label"] == "prior"
                                for s in c["sends"]])
    monkeypatch.setattr(mw, "SendsDialog", pick)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_sends_edit_requested(child.layer_id)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert "_sends" not in win._params[0] and json.loads(win._params[1]["_sends"])[0]["role"] == "value"
    assert json.loads(child.chain.steps[0].params["_sends"])[0]["label"] == "prior · band 1"


def test_a_layer_send_on_an_inpaint_step_gets_the_prelude_and_resolves(win, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw

    host = _layer_named(win, "host")
    prior_layer = _layer_named(win, "prior")
    win.layer_list.select_layer(host.layer_id)
    child = _drop(win, qtbot, "inpaint", **FAST)
    pick = _Pick(lambda cands: [dict(s, role="value") for c in cands
                                if c["label"] == "prior (layer)" for s in c["sends"]])
    monkeypatch.setattr(mw, "SendsDialog", pick)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_sends_edit_requested(child.layer_id)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    sends = json.loads(child.chain.steps[0].params["_sends"])
    assert sends[0]["layer"] == prior_layer.layer_id and sends[0]["stamp"]
    assert win._active_result["_grid_relations"]["prior (layer)"]["kind"] == "nested"
    plan = win._bus_plan(child)
    assert plan == [] or plan[0][0] is prior_layer           # filed already, or planned
    from dynamix.core.bus import _PLANES

    _PLANES.clear()
    assert win._bus_plan(child)[0][0] is prior_layer           # the prelude re-files it


def test_the_row_menu_offers_edit_sends_on_an_inpaint_layer(win, qtbot):
    host = _layer_named(win, "host")
    win.layer_list.select_layer(host.layer_id)
    child = _drop(win, qtbot, "inpaint", **FAST)
    item = win.layer_list._layer_items[child.layer_id]
    menu = win.layer_list._build_context_menu(item, child.layer_id)
    texts = [a.text() for a in menu.actions()]
    assert "Edit sends…" in texts and "Edit bus sends…" not in texts
    menu_host = win.layer_list._build_context_menu(win.layer_list._layer_items[host.layer_id],
                                                   host.layer_id)
    assert "Edit sends…" not in [a.text() for a in menu_host.actions()]
