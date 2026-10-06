# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Racks in the shell: the browser's rack rows (user racks beside the built-in ones, greyed with a
reason), the workflow zone's split-and-merge rack drop, and the window's Save as rack, rack drop
on a band row, and rack fork.

Offscreen Qt; drops call ``dropEvent`` with a stub event, as tests/test_workflow_zone.py does."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore

from dynamix.devices import register_builtin_devices
from dynamix.model.chain import Chain, DeviceRef
from tests.test_shell_window import stub_devices, window  # noqa: F401


@pytest.fixture
def registered_builtins(clean_registry):
    register_builtin_devices()
    return clean_registry


def _rack(name, *steps):
    from dynamix.model.racks import Rack, RackStep

    return Rack(name, tuple(RackStep(d, dict(p)) for d, p in steps))


def _rows(browser):
    racks = browser._categories["Racks"]
    return {racks.child(i).text(0): racks.child(i) for i in range(racks.childCount())}


def _greyed(browser, row) -> bool:
    """Greyed as Phase 3 greys a device row: the muted ink in column 0."""
    return (row.data(0, QtCore.Qt.ForegroundRole) is not None
            and row.foreground(0).color() == browser._muted.color())


# --------------------------------------------------------------------------- browser rows


def test_user_racks_join_the_builtin_ones_under_racks(qtbot, registered_builtins, tmp_path):
    from dynamix.model.racks import RackEntry, builtin_racks
    from dynamix.shell.browser import PRESET_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    browser.set_presets({"WTMM standard": (("wtmm2d", {}),)})
    mine = _rack("Mine", ("wtmm2d", {}), ("chain_holder", {}))
    browser.set_racks([RackEntry(r, None) for r in builtin_racks()]
                      + [RackEntry(mine, tmp_path / "mine.json")])
    rows = _rows(browser)
    assert list(rows) == [r.name for r in builtin_racks()] + ["Mine"]   # the preset row replaced
    mime = browser.mimeData([rows["Mine"]])
    assert bytes(mime.data(PRESET_MIME).data()) == b"Mine"
    assert rows["Mine"].toolTip(0) == str(tmp_path / "mine.json")
    assert rows["WTMM + Hölder"].toolTip(0) == "built-in rack"


def test_a_rack_with_a_problem_is_listed_disabled_with_the_reason(qtbot, registered_builtins,
                                                                 tmp_path):
    from dynamix.model.racks import RackEntry
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    browser.set_racks([
        RackEntry(_rack("Old", ("gone_device", {})), tmp_path / "old.json",
                  "unknown device 'gone_device'"),
        RackEntry(None, tmp_path / "broken.json", "broken.json: not a geodynamix-rack file")])
    rows = _rows(browser)
    assert rows["Old"].isDisabled() and rows["Old"].toolTip(0) == "unknown device 'gone_device'"
    assert rows["broken"].isDisabled()
    assert rows["broken"].toolTip(0) == "broken.json: not a geodynamix-rack file"


def test_a_rack_that_cannot_head_a_drop_on_the_active_layer_is_greyed(qtbot,
                                                                      registered_builtins):
    from dynamix.model.contracts import FIELD, STACK
    from dynamix.model.racks import RackEntry
    from dynamix.shell.browser import PRESET_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    browser.set_racks([RackEntry(_rack("W", ("wtmm2d", {}), ("chain_holder", {})), None),
                       RackEntry(_rack("Cuts", ("chain_holder", {})), None),
                       RackEntry(_rack("P", ("pca", {})), None),
                       RackEntry(_rack("T", ("tucker_HOOI_HOSVD", {})), None)])
    rows = _rows(browser)
    assert not any(_greyed(browser, row) for row in rows.values())   # no active layer yet
    browser.set_rack_context((), STACK, master=True)                  # a stack dataset row
    assert _greyed(browser, rows["W"]) and "band stack" in rows["W"].toolTip(0)
    assert not rows["W"].isDisabled()
    assert browser.mimeData([rows["W"]]).hasFormat(PRESET_MIME)       # greyed, still draggable
    assert not _greyed(browser, rows["P"])                            # a band tool runs on it
    band = DeviceRef("band_select", {"band": 1})
    browser.set_rack_context((band,), STACK)                          # a band row
    assert not _greyed(browser, rows["W"])
    assert rows["W"].data(0, QtCore.Qt.ForegroundRole) is None
    assert _greyed(browser, rows["P"]) and "band stack" in rows["P"].toolTip(0)
    browser.set_rack_context((band, DeviceRef("wtmm2d", {})), STACK)  # an analysed band row
    assert not _greyed(browser, rows["W"]) and _greyed(browser, rows["P"])
    browser.set_rack_context((DeviceRef("bus", {}), DeviceRef("pca", {})), FIELD)   # a bus row
    assert not _greyed(browser, rows["T"]) and _greyed(browser, rows["W"])
    browser.set_rack_context((DeviceRef("mz_edges", {}),), FIELD)
    assert not _greyed(browser, rows["W"])                            # forks a sibling there
    assert _greyed(browser, rows["Cuts"]) and "needs" in rows["Cuts"].toolTip(0)


def test_double_clicking_a_rack_row_asks_for_the_editor(qtbot, registered_builtins):
    from dynamix.model.contracts import STACK
    from dynamix.model.racks import RackEntry
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    browser.set_racks([RackEntry(_rack("W", ("wtmm2d", {})), None)])
    browser.set_rack_context((), STACK, master=True)                  # greyed: still opens
    with qtbot.waitSignal(browser.rackEditRequested) as sig:
        browser.itemDoubleClicked.emit(_rows(browser)["W"], 0)
    assert sig.args == ["W"]


# --------------------------------------------------------------------------- zone drop


class _StubDropEvent:
    def __init__(self, mime: QtCore.QMimeData):
        self._mime = mime
        self.accepted: bool | None = None

    def position(self):
        return QtCore.QPointF(0, 0)

    def mimeData(self):
        return self._mime

    def proposedAction(self):
        return QtCore.Qt.MoveAction

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def _rack_drop(name: str) -> _StubDropEvent:
    from dynamix.shell.browser import PRESET_MIME

    mime = QtCore.QMimeData()
    mime.setData(PRESET_MIME, name.encode("utf-8"))
    return _StubDropEvent(mime)


def _descriptors(*names):
    return [{"device": n, "params": {}, "bypassed": False, "rack": None} for n in names]


def test_a_rack_drop_splits_and_merges_onto_a_chain_with_filters(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import RackBox, WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors("wtmm2d", "modulus_threshold", "hline_length"))
    zone.set_racks([_rack("Hölder chain", ("wtmm2d", {}), ("chain_topology", {}),
                          ("chain_holder", {}))])
    with qtbot.waitSignal(zone.chainEdited) as sig:
        zone.dropEvent(_rack_drop("Hölder chain"))
    # The chain's own wtmm2d stays the analysis; the rack's topology and filter join it.
    assert [d["device"] for d in sig.args[0]] == ["wtmm2d", "chain_topology", "modulus_threshold",
                                                  "hline_length", "chain_holder"]
    assert [d["rack"] for d in sig.args[0]] == [None, "Hölder chain", None, None,
                                                "Hölder chain (2)"]
    assert sorted(r.title for r in zone.findChildren(RackBox)) == ["Hölder chain",
                                                                   "Hölder chain (2)"]


def test_a_preset_drop_onto_a_chain_with_filters_now_lands(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    zone.set_steps(_descriptors("wtmm2d", "modulus_threshold"))
    zone.set_presets({"Edges": (("mz_edges", {"n_levels": 3}), ("hline_length", {}))})
    with qtbot.waitSignal(zone.chainEdited) as sig:
        zone.dropEvent(_rack_drop("Edges"))
    assert [d["device"] for d in sig.args[0]] == ["wtmm2d", "mz_edges", "modulus_threshold",
                                                  "hline_length"]
    assert sig.args[0][1]["params"] == {"n_levels": 3}


def test_the_strip_offers_save_as_rack(qtbot, registered_builtins):
    from dynamix.shell.workflow_zone import WorkflowZone

    zone = WorkflowZone()
    qtbot.addWidget(zone)
    with qtbot.waitSignal(zone.saveRackRequested):
        zone.source_box.save_rack_button.click()


def test_a_rack_headed_by_a_different_analyzer_forks_a_sibling(qtbot, window):
    from dynamix.model.presets import PRESETS

    win = window
    win._start_worker = lambda: None
    field = np.linspace(0.0, 1.0, 256).reshape(16, 16)
    win.load_field(field, "mem:fork")
    master = win.layer
    child = win.project.add_layer(
        "mz", master.source_id,
        Chain((DeviceRef("mz_edges", {}), DeviceRef("hline_length", {}))).materialized(),
        parent_id=master.layer_id)
    win.add_layer_row(child, field)
    win.layer_list.select_layer(child.layer_id)
    before = len(win.project.layers)
    win.strips.dropEvent(_rack_drop("WTMM + Hölder"))
    qtbot.waitUntil(lambda: len(win.project.layers) == before + 1, timeout=5000)
    sibling = win.project.layers[-1]
    assert sibling.parent_id == child.layer_id and win.layer is sibling
    assert [s.device for s in sibling.chain.steps] == [d for d, _ in PRESETS["WTMM + Hölder"]]
    assert [s.device for s in child.chain.steps] == ["mz_edges", "hline_length"]
