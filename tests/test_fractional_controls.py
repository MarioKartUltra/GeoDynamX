# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
from dynamix.devices.wtmm import WTMM2D
from dynamix.devices.wtmm_roi import WTMM2DROI
from dynamix.model.device import defaults_for
from dynamix.shell.workflow_zone import DeviceBox
from dynamix.shell.main_window import _eta_note_of


def test_coefficient_lift_is_visible_and_combined_shift_is_seeded_once(qtbot):
    for dev in (WTMM2D(), WTMM2DROI()):
        box = DeviceBox(0, dev, defaults_for(dev))
        qtbot.addWidget(box)
        assert "fracint_alpha" in box.controls
    dev = WTMM2D()
    box = DeviceBox(0, dev, defaults_for(dev))
    qtbot.addWidget(box)
    assert "integration_order" in box.controls
    seed, note = _eta_note_of({"params": {"fracint_alpha": .3}, "_integration_order": .4})
    assert abs(seed-.7) < 1e-12 and "total shift" in note
    seed, _ = _eta_note_of({"params": {"gamma": .2}, "samples": {}, "_integration_order": .4})
    assert abs(seed-.6) < 1e-12
