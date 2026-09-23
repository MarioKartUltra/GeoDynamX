# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The ROI panel's saved-ROI half: Save ROI, a list of saved ROIs, the
active one; the child-dataset crop button is unhooked (its handler stays in code)."""
from __future__ import annotations

import pytest

from dynamix.shell.roi_panel import RoiPanel


@pytest.fixture
def panel(qtbot):
    p = RoiPanel()
    qtbot.addWidget(p)
    return p


def test_save_emits_the_panels_numbers(qtbot, panel):
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    with qtbot.waitSignal(panel.saveRequested, timeout=1000) as sig:
        panel.save_button.click()
    assert (sig.args[0]["roi_row"], sig.args[0]["roi_col"]) == (16, 24)
    assert (sig.args[0]["roi_h"], sig.args[0]["roi_w"]) == (32, 40)


def test_save_needs_a_legal_box_but_not_an_unblocked_target(panel):
    """Saving a region is legal on any layer (it is saved on the SOURCE); the nested-wtmm2d_roi
    block only ever concerned the legacy create path."""
    panel.show_roi(16, 24, 32, 40, (None, "px"), blocked="already an ROI")
    assert panel.save_button.isEnabled() is True
    panel.edit("roi_h").setText("2")                     # a sliver: illegal
    assert panel.save_button.isEnabled() is False


def test_the_child_dataset_crop_is_unhooked_and_create_reads_save_and_run(panel):
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    assert panel.child_button.isHidden()
    assert panel.create_button.text() == "Save + run WTMM"


def test_saved_rois_list_shows_the_active_one_and_activation_emits_its_id(qtbot, panel):
    panel.set_saved([("roi0", "A"), ("roi1", "B")], active="roi1")
    # Visibility stays the window's call (a layer switch hides the panel so a stale box can
    # never act on the new target); the list is reached through the ROI tool or a draw.
    assert panel.isHidden()
    assert panel.saved_list.count() == 2
    assert panel.saved_list.currentRow() == 1
    with qtbot.waitSignal(panel.roiActivated, timeout=1000) as sig:
        panel.saved_list.setCurrentRow(0)
    assert sig.args[0] == "roi0"


def test_setting_the_list_does_not_echo_an_activation(qtbot, panel):
    got = []
    panel.roiActivated.connect(got.append)
    panel.set_saved([("roi0", "A")], active="roi0")
    assert got == []


def test_deselect_emits_an_empty_activation(qtbot, panel):
    panel.set_saved([("roi0", "A")], active="roi0")
    with qtbot.waitSignal(panel.roiActivated, timeout=1000) as sig:
        panel.deselect_button.click()
    assert sig.args[0] == ""
    assert panel.saved_list.currentRow() == -1
