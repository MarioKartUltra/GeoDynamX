# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Racks in the shell: the browser's rack rows (user racks beside the built-in ones, greyed with a
reason), the workflow zone's split-and-merge rack drop, and the window's Save as rack, rack drop
on a band row, and rack fork.

Offscreen Qt; drops call ``dropEvent`` with a stub event, as tests/test_workflow_zone.py does."""
from __future__ import annotations

import pytest
from PySide6 import QtCore

from dynamix.devices import register_builtin_devices
from dynamix.model.chain import DeviceRef


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
