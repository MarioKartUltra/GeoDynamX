# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
from dynamix.devices.wtmm import WTMM2D
from dynamix.model.device import defaults_for
from dynamix.shell.right_panel import DevicePanel


def test_downstream_wtmm_controls_live_in_the_inspector_and_keep_scientific_keys(qtbot):
    device = WTMM2D()
    by_name = {p.name: p for p in device.params}
    for name in device.inspector_actions:
        assert by_name[name].level == ("hidden" if name in ("border_policy", "coi_k") else "advanced")
        assert not by_name[name].view
    panel = DevicePanel()
    qtbot.addWidget(panel)
    panel.set_device([p for p in device.params if p.level == "advanced"], defaults_for(device))
    panel.set_actions(device.inspector_actions)
    panel.set_pending(False)
    assert "q_step" in panel.controls and not panel.apply_button.isEnabled()
    assert panel.controls["q_step"].toolTip() == "Repartition"
    panel.set_pending(True)
    with qtbot.waitSignal(panel.applyRequested):
        panel.apply_button.click()
