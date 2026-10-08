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


def test_the_sends_dialog_saves_the_binding_it_shows(qtbot):
    from dynamix.shell.sends_dialog import SendsDialog

    a = {"path": "/a.tif", "subdataset": None, "band": None, "label": "A · b1"}
    b = {"path": "/b.tif", "subdataset": None, "band": None, "label": "B · b1"}
    cands = [{"label": "A", "sends": [a], "reason": None},
             {"label": "B", "sends": [b], "reason": None}]
    dlg = SendsDialog(cands, [dict(b, role="weights", **{"for": "A · b1"}), dict(a, role="value")])
    qtbot.addWidget(dlg)
    assert dlg.sends()[0]["for"] == "A · b1"
    dlg.tree.topLevelItem(0).child(0).setCheckState(0, QtCore.Qt.Unchecked)
    out = dlg.sends()
    assert [s["label"] for s in out] == ["B · b1"] and out[0]["for"] == "target"
    dlg.tree.topLevelItem(0).child(0).setCheckState(0, QtCore.Qt.Checked)
    row_b = next(r for r in dlg.rows() if r.label == "B · b1")
    assert [row_b.target.itemText(i) for i in range(row_b.target.count())] == ["target", "A · b1"]
    assert dlg.sends()[0]["for"] == "target"
    row_b.target.setCurrentText("A · b1")
    assert next(s for s in dlg.sends() if s["label"] == "B · b1")["for"] == "A · b1"


def test_a_layer_candidate_shows_its_relation_not_an_error(win):
    host = _layer_named(win, "host")
    cands = win._sends_candidates(win._fields[host.layer_id], exclude_layer_id=host.layer_id)
    send = next(s for c in cands if c["label"] == "prior (layer)" for s in c["sends"])
    text = win._relation_text(win._fields[host.layer_id], send)
    assert text.split()[0] in ("same", "nested", "foreign") and "unreadable" not in text


# ---------------------------------------------------------------- 7.2 mask mode

from tests.test_shell_roi_flow import _data_from_geometry, _move, _press, _release, canvas  # noqa: E402,F401


def test_mask_mode_drag_emits_the_polygon_in_file_pixels(qtbot, canvas):
    canvas.set_selection_mode("mask")
    screen = [(120.0, 100.0), (300.0, 100.0), (300.0, 260.0), (120.0, 260.0)]
    none = QtCore.Qt.KeyboardModifier.NoModifier
    with qtbot.waitSignal(canvas.maskDrawn, timeout=1000) as sig:
        canvas.mousePressEvent(_press(screen[0], none))
        for pt in screen[1:]:
            canvas.mouseMoveEvent(_move(pt, none))
        canvas.mouseReleaseEvent(_release(screen[-1], none))
    (points,) = sig.args
    assert len(points) == 4 and all(len(p) == 2 for p in points)
    x0, y0 = _data_from_geometry(canvas, *screen[0])
    assert abs(points[0][0] - x0) < 0.6 and abs(points[0][1] - y0) < 0.6
    assert canvas.mask_item.xData is None or len(canvas.mask_item.xData) == 0   # the live line is gone
    canvas.set_mask_polygons([{"kind": "gap", "points": points}])
    assert len(canvas.mask_outline_item.xData) == 6                            # a closed ring + NaN


def test_a_two_point_drag_in_mask_mode_emits_nothing(qtbot, canvas):
    canvas.set_selection_mode("mask")
    none = QtCore.Qt.KeyboardModifier.NoModifier
    fired = []
    canvas.maskDrawn.connect(lambda pts: fired.append(pts))
    canvas.mousePressEvent(_press((100.0, 100.0), none))
    canvas.mouseReleaseEvent(_release((140.0, 100.0), none))
    assert fired == []


def test_the_mask_mode_row_writes_the_regions_into_the_inpaint_step_and_the_run_fills_them(win, qtbot):
    host = _layer_named(win, "host")
    win.layer_list.select_layer(host.layer_id)
    child = _drop(win, qtbot, "inpaint", veil_model="offset", **FAST)   # the host is signed: no gain veil
    assert "mask" in win._selection_mode_buttons and not win._mask_kind_combo.isVisibleTo(win)
    win.set_selection_mode("mask")
    assert win.canvas._selection_mode == "mask" and win._mask_kind_combo.isVisibleTo(win)
    win._mask_kind_combo.setCurrentText("veil")
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.canvas.maskDrawn.emit([[4.0, 4.0], [14.0, 4.0], [14.0, 12.0], [4.0, 12.0]])
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    step = win.layer.chain.steps[0]
    polys = json.loads(step.params["_mask_polygons"])
    assert polys == [{"kind": "veil", "points": [[4.0, 4.0], [14.0, 4.0], [14.0, 12.0], [4.0, 12.0]]}]
    assert step.params["mask_source"] == "region"           # nodata stays a gap under every source
    mask = win._active_result["fill_mask"]
    assert mask[8, 8] > 0 and mask[20, 28] > 0                 # the drawn veil and the nodata gap
    assert len(win.canvas.mask_outline_item.xData) == 6
    win._mask_clear_button.click()
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert win.layer.chain.steps[0].params["_mask_polygons"] == "[]"
    win.set_selection_mode("click")
    assert not win._mask_kind_combo.isVisibleTo(win)


def test_drawing_a_mask_without_an_inpaint_step_only_notifies(win, qtbot):
    host = _layer_named(win, "host")
    win.layer_list.select_layer(host.layer_id)
    win.set_selection_mode("mask")
    win.canvas.maskDrawn.emit([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0]])
    assert "inpaint" in win.statusBar().currentMessage() or "fill" in win.statusBar().currentMessage()


# ------------------------------------------- 7.2 fix: descriptor, dispatch count, file pixels

_SQUARE = [[4.0, 4.0], [14.0, 4.0], [14.0, 12.0], [4.0, 12.0]]


def _with_noise_step(win, qtbot, **params):
    """The host layer with a ``noise`` step before the inpaint step, so a bypass toggle on a
    step other than the mask step is available; returns the inpaint slot."""
    host = _layer_named(win, "host")
    win.layer_list.select_layer(host.layer_id)
    desc = [{"device": "noise", "params": defaults_for(get_device("noise"))},
            {"device": "inpaint", "params": {**defaults_for(get_device("inpaint")),
                                             "veil_model": "offset", **FAST, **params}}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win._mask_step_index()


def _toggle_bypass(win, qtbot, slot, value):
    win.strips.strip(slot).bypassToggled.emit(value)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)


def test_a_drawn_region_survives_a_bypass_toggle_on_another_step(win, qtbot):
    idx = _with_noise_step(win, qtbot)
    win.set_selection_mode("mask")
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.canvas.maskDrawn.emit(_SQUARE)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    _toggle_bypass(win, qtbot, 0, True)
    idx = win._mask_step_index()
    polys = json.loads(win._params[idx]["_mask_polygons"])
    assert polys == [{"kind": "gap", "points": _SQUARE}]
    assert win.strips._descriptors[idx]["params"]["_mask_polygons"] == win._params[idx]["_mask_polygons"]
    assert win._params[idx]["mask_source"] == "region"
    assert len(win.canvas.mask_outline_item.xData) == 6


def test_a_clear_survives_a_bypass_toggle_on_another_step(win, qtbot):
    _with_noise_step(win, qtbot, mask_source="region",
                     _mask_polygons=json.dumps([{"kind": "gap", "points": _SQUARE}]))
    win._mask_clear_button.click()
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    _toggle_bypass(win, qtbot, 0, True)
    idx = win._mask_step_index()
    assert win._params[idx]["_mask_polygons"] == "[]"
    assert win.strips._descriptors[idx]["params"]["_mask_polygons"] == "[]"


def test_edited_sends_survive_a_bypass_toggle_on_another_step(win, qtbot):
    idx = _with_noise_step(win, qtbot)
    text = json.dumps([{"label": "x", "role": "guide"}])
    win._rewrite_sends(win.layer, 1, text)           # chain index 1: the inpaint step
    assert win.strips._descriptors[idx]["params"]["_sends"] == text
    _toggle_bypass(win, qtbot, 0, True)
    assert win._params[win._mask_step_index()]["_sends"] == text


def test_the_first_drawn_region_starts_exactly_one_worker(win, qtbot, monkeypatch):
    _with_noise_step(win, qtbot)
    calls = []
    real = win._start_worker
    monkeypatch.setattr(win, "_start_worker", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.canvas.maskDrawn.emit(_SQUARE)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert len(calls) == 1


def test_clear_with_nothing_drawn_dispatches_nothing(win, qtbot, monkeypatch):
    _with_noise_step(win, qtbot)
    calls = []
    monkeypatch.setattr(win, "_start_worker", lambda *a, **k: calls.append(1))
    win._mask_clear_button.click()
    assert calls == []


def test_a_region_drawn_on_a_windowed_field_is_in_file_pixels(qtbot, canvas):
    from dynamix.core.inpaint.engine import regions_from

    canvas.set_selection_mode("mask")
    none = QtCore.Qt.KeyboardModifier.NoModifier
    screen = [(100.0, 100.0), (300.0, 100.0), (300.0, 170.0), (100.0, 170.0)]
    with qtbot.waitSignal(canvas.maskDrawn, timeout=1000) as sig:
        canvas.mousePressEvent(_press(screen[0], none))
        for pt in screen[1:]:
            canvas.mouseMoveEvent(_move(pt, none))
        canvas.mouseReleaseEvent(_release(screen[-1], none))
    (points,) = sig.args
    for p, s in zip(points, screen):
        x, y = _data_from_geometry(canvas, *s)
        assert abs(p[0] - x) < 0.6 and abs(p[1] - y) < 0.6        # file pixels, no extra offset
    off = (int(min(p[1] for p in points)) - 3, int(min(p[0] for p in points)) - 3)
    text = json.dumps([{"kind": "gap", "points": points}])
    ids, _ = regions_from(np.zeros((2000, 2000)), {"mask_source": "region", "_mask_polygons": text},
                          window_offset=off)
    cy = np.mean([p[1] for p in points]) - off[0]
    cx = np.mean([p[0] for p in points]) - off[1]
    assert ids[int(cy), int(cx)] > 0
    assert ids[int(cx), int(cy)] == 0                 # the transposed point is outside
