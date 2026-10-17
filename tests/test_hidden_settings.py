# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Superseded settings hidden (spec 25 §3.6): a hidden choice is still a valid stored value but the
cycle button never offers it, and a hidden parameter is drawn only when a project holds a value other
than its default, labelled "(superseded)"."""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets

from dynamix.model.param import Param, ParamKind

EDGE = Param("edge", ParamKind.CHOICE, default="auto", choices=("auto", "reflect", "cut"),
             hidden_choices=("cut",))
OLD = Param("border_policy", ParamKind.CHOICE, default="none", choices=("none", "coi_trim"),
            level="hidden")


def test_a_hidden_choice_stays_valid_and_must_be_one_of_the_choices():
    assert EDGE.validate("cut") == "cut"
    with pytest.raises(ValueError, match="hidden"):
        Param("e", ParamKind.CHOICE, default="a", choices=("a", "b"), hidden_choices=("z",))
    with pytest.raises(ValueError, match="hidden"):
        Param("e", ParamKind.CHOICE, default="a", choices=("a", "b"), hidden_choices=("a",))
    assert OLD.level == "hidden"


def test_the_cycle_button_skips_a_hidden_choice_and_labels_one_held(qtbot):
    from dynamix.shell.knob_widgets import make_control
    from dynamix.shell.knobs import control_spec

    button = make_control(control_spec(EDGE), "reflect")
    qtbot.addWidget(button)
    seen = []
    button.valueChanged.connect(seen.append)
    button.click()
    assert seen == ["auto"]                      # reflect -> auto, never through "cut"
    held = make_control(control_spec(EDGE), "cut")
    qtbot.addWidget(held)
    assert held.text().endswith("(superseded)")
    held.click()
    assert held.text() == "auto"


def test_a_hidden_parameter_shows_only_when_a_project_holds_another_value(qtbot):
    from dynamix.shell.right_panel import DevicePanel

    panel = DevicePanel()
    qtbot.addWidget(panel)
    panel.set_device((EDGE, OLD), {"edge": "auto", "border_policy": "none"})
    assert "border_policy" not in panel.controls
    panel.set_device((EDGE, OLD), {"edge": "auto", "border_policy": "coi_trim"})
    assert "border_policy" in panel.controls
    labels = [w.text() for w in panel.findChildren(QtWidgets.QLabel)]
    assert any("(superseded)" in t for t in labels)


def test_the_strip_never_draws_a_hidden_parameter():
    from dynamix.shell.workflow_zone import _grouped_params

    class Dev:
        params = (EDGE, OLD)

    names = [p.name for _title, group in _grouped_params(Dev()) for p in group]
    assert names == ["edge"]
