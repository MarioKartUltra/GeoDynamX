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


@pytest.mark.parametrize('origin', [('240', '240'), ('', ''), ('-4', '-8')])
def test_place_box_uses_size_independently_of_typed_origin(panel, qtbot, origin):
    panel.show_roi(0, 0, 64, 64, (None, 'px'), dims=(256, 256))
    panel.edit('roi_row').setText(origin[0])
    panel.edit('roi_col').setText(origin[1])
    assert panel.values() is None
    assert not panel.save_button.isEnabled()
    assert panel.place_button.isEnabled()
    with qtbot.waitSignal(panel.placeRequested, timeout=1000) as sig:
        panel.place_button.click()
    assert sig.args == [64, 64]


@pytest.mark.parametrize('height,width', [('2', '64'), ('257', '64'), ('64', '257'), ('', '64')])
def test_place_box_requires_a_size_that_fits_the_dataset(panel, height, width):
    panel.show_roi(0, 0, 64, 64, (None, 'px'), dims=(256, 256))
    panel.edit('roi_h').setText(height)
    panel.edit('roi_w').setText(width)
    assert not panel.place_button.isEnabled()


def test_half_and_double_scale_the_box_about_its_centre_inside_the_data(qtbot):
    from dynamix.shell.roi_panel import RoiPanel

    panel = RoiPanel()
    qtbot.addWidget(panel)
    panel.show_roi(10, 20, 16, 20, (None, "px"), dims=(40, 50))

    def box():
        v = panel.values()
        return v["roi_row"], v["roi_col"], v["roi_h"], v["roi_w"]

    panel.double_button.click()
    assert box() == (2, 10, 32, 40)
    panel.double_button.click()                       # 64 x 80 does not fit: the whole data
    assert box() == (0, 0, 40, 50)
    panel.half_button.click()
    assert box() == (10, 12, 20, 25)
    for _ in range(4):
        panel.half_button.click()                     # never below the engine's minimum side
    assert min(box()[2:]) >= 8
