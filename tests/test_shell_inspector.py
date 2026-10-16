# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The inspector follows the selection: a chain step's knobs, basic open and advanced collapsed."""
from __future__ import annotations

from PySide6 import QtWidgets

from dynamix.model.param import Param, ParamKind
from dynamix.shell.right_panel import DevicePanel
from tests.test_shell_window import _FIELD, loaded, stub_devices, window  # noqa: F401


def _val(control):
    return control.value() if callable(getattr(control, "value", None)) else control._value


def test_device_panel_keeps_advanced_knobs_collapsed_until_asked(qtbot):
    panel = DevicePanel()
    qtbot.addWidget(panel)
    panel.show()
    params = (Param("a", ParamKind.INT, default=1, min=0, max=9),
              Param("b", ParamKind.INT, default=2, min=0, max=9, level="advanced"))
    assert panel.set_device(params, {"a": 1, "b": 2})
    assert panel.controls["a"].isVisible() and not panel.controls["b"].isVisible()
    panel.advanced_button.setChecked(True)
    assert panel.controls["b"].isVisible()
    assert not panel.set_device(params, {"a": 3, "b": 4})       # same knobs: values only
    assert _val(panel.controls["a"]) == 3


def test_selecting_a_step_shows_its_knobs_and_a_layer_hides_them(qtbot, loaded):
    w = loaded
    section = w.right_panel._sections["Device"]
    assert section.isHidden()                       # the test window is never shown: hidden state
    w.strips.strip(0).selected.emit()
    assert not section.isHidden() and w._inspected_step == 0
    assert section.isAncestorOf(w._device_panel)
    assert "n_scales" in w._device_panel.controls
    assert w.right_panel._headings["Device"].text().startswith("Device")
    w.layer_list.layerSelected.emit(w.layer.layer_id)
    assert section.isHidden() and w._inspected_step is None


def test_a_chain_rebuild_keeps_the_device_section_on_the_surviving_step(qtbot, loaded):
    """A rebuild (a band pick, a drop, a revert) that keeps the selected step keeps the inspector
    on it; one that drops the step closes the section."""
    w = loaded
    section = w.right_panel._sections["Device"]
    w.strips.strip(0).selected.emit()
    assert w._inspected_step == 0
    descriptors = [{"device": ref.device, "params": dict(ref.params)} for ref in w.layer.chain.steps]
    w.strips.set_steps(descriptors, field=w.field)
    assert not section.isHidden() and w._inspected_step == 0
    assert w.strips.strip(0).property("selected") == "true"
    w.strips.set_steps([], field=w.field)
    assert section.isHidden() and w._inspected_step is None


def test_an_inspector_edit_takes_the_box_edit_path_and_mirrors_into_the_box(qtbot, loaded):
    w = loaded
    w.strips.strip(0).selected.emit()
    w._device_panel.paramChanged.emit("n_scales", 5)
    assert w._params[0]["n_scales"] == 5
    assert _val(w.strips.strip(0).controls["n_scales"]) == 5
    w.strips.strip(0).controls["n_scales"].valueChanged.emit(6)       # a box edit
    assert _val(w._device_panel.controls["n_scales"]) == 6


def test_a_refused_inspector_edit_shows_in_neither_control(qtbot, loaded):
    """A locked layer refuses the edit; the box and the inspector both keep the old value (a control
    only proposes a value, the owner shows it once accepted)."""
    w = loaded
    w.strips.strip(0).selected.emit()
    before = _val(w.strips.strip(0).controls["n_scales"])
    w.layer.tags["ui.lock"] = "1"
    knob = w._device_panel.controls["n_scales"]
    knob.set_value(before + 1)                        # the user turned the inspector knob...
    knob.valueChanged.emit(before + 1)                # ...and the panel proposed the value
    assert w._params[0].get("n_scales", before) == before
    assert _val(w.strips.strip(0).controls["n_scales"]) == before
    assert _val(knob) == before
    assert "locked" in w.strips.reading_label.text()          # refused for that reason, and said so
