# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.chain_strip -- device strips with generated controls and honesty
readings.

Offscreen Qt (see conftest / the module-level env var the test runner sets), stub-free: exercises
the real registered devices (``wtmm2d`` -- a Transform with 5 params incl. a CHOICE -- and
``min_vchains`` -- a Filter with 1 param) so the strip's generated-control path is proven against
the real ``Param``/``ControlSpec``/``make_control`` chain, not a hand-rolled stand-in.
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore

from dynamix.devices import register_builtin_devices
from dynamix.model.chain import DeviceRef
from dynamix.model.device import get_device


@pytest.fixture
def registered_builtins(clean_registry):
    """The real device registry, mirroring tests/test_projectfile.py's ``_registry`` pattern."""
    register_builtin_devices()
    return clean_registry


def test_strip_generates_one_control_per_param(qtbot, registered_builtins):
    from dynamix.model.device import get_device
    from dynamix.shell.chain_strip import DeviceStrip
    dev = get_device("wtmm2d")
    s = DeviceStrip(0, dev, {p.name: p.default for p in dev.params})
    qtbot.addWidget(s)
    assert len(s.controls) == len(dev.params)      # 5 for wtmm2d, incl. the CHOICE wavelet


def test_param_change_signal_carries_name_and_value(qtbot, registered_builtins):
    from dynamix.shell.chain_strip import DeviceStrip
    dev = get_device("wtmm2d")
    s = DeviceStrip(0, dev, {p.name: p.default for p in dev.params})
    qtbot.addWidget(s)
    with qtbot.waitSignal(s.paramChanged) as sig:
        s.controls["n_oct"].valueChanged.emit(4)      # as the widget itself would propose
    assert sig.args == ["n_oct", 4]


def test_filter_strip_has_no_state_dot_and_transform_strip_has(qtbot, registered_builtins):
    from dynamix.shell.chain_strip import DeviceStrip

    filter_dev = get_device("min_vchains")
    fs = DeviceStrip(0, filter_dev, {p.name: p.default for p in filter_dev.params})
    qtbot.addWidget(fs)
    assert fs.state_dot is None

    transform_dev = get_device("wtmm2d")
    ts = DeviceStrip(0, transform_dev, {p.name: p.default for p in transform_dev.params})
    qtbot.addWidget(ts)
    assert ts.state_dot is not None


def test_zone_builds_strips_and_single_selection(qtbot, registered_builtins):
    from dynamix.shell.chain_strip import ChainStripZone

    steps = [DeviceRef("wtmm2d", {}), DeviceRef("min_vchains", {})]
    zone = ChainStripZone(steps)
    qtbot.addWidget(zone)
    assert len(zone.strip(0).controls) == len(get_device("wtmm2d").params)
    assert len(zone.strip(1).controls) == len(get_device("min_vchains").params)

    # Click near the strip's top-left corner -- inside DeviceStrip's own content margins, so the
    # click always lands on the frame's own background rather than a generated child control
    # (whose exact position depends on layout stretch and must not matter to this test).
    qtbot.mouseClick(zone.strip(1), QtCore.Qt.LeftButton, pos=QtCore.QPoint(2, 2))
    assert zone.strip(1).property("selected") == "true"
    assert zone.strip(0).property("selected") != "true"


def test_honesty_reading(qtbot, registered_builtins):
    from dynamix.shell.chain_strip import DeviceStrip
    dev = get_device("min_vchains")
    s = DeviceStrip(0, dev, {p.name: p.default for p in dev.params})
    qtbot.addWidget(s)
    s.set_reading("dropped 12")
    assert s.reading_label.text() == "dropped 12"
    assert s.reading_label.property("reading") == "true"


def test_paramchanged_reemit_closes_the_dragvalue_confirm_loop(qtbot, registered_builtins):
    """CRITICAL carry-forward from the review: DragValue is a CONTROLLED component -- its
    nudge/drag/edit gestures PROPOSE a value via ``valueChanged`` but never update their own
    displayed text; only ``set_value()`` does that. DeviceStrip is the layer that closes this
    propose -> confirm loop for slice 1 (the window only updates strips wholesale on
    recompute), so its ``paramChanged`` re-emit path must call the originating control's
    ``set_value(value)`` with the applied value -- even when that value is identical to the raw
    proposal, as it is here, with nothing else in the pipeline yet to requantize it."""
    from dynamix.shell.chain_strip import DeviceStrip
    dev = get_device("wtmm2d")
    s = DeviceStrip(0, dev, {p.name: p.default for p in dev.params})
    qtbot.addWidget(s)
    control = s.controls["n_oct"]
    assert control.text() == "3"          # wtmm2d's declared default

    control.valueChanged.emit(4)
    assert control.text() == "4"          # updated ONLY because the strip called set_value(4)
