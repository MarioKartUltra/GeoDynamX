# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The user's own colour ramps: kept in the settings file, resolved wherever a colormap name is."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.test_shell_window import loaded, stub_devices, window  # noqa: F401


def test_a_saved_ramp_resolves_lists_and_deletes():
    from dynamix.shell import ramps
    from dynamix.shell.settings import load_settings

    ramps.save_ramp("zebra h", [(0.0, "#000000"), (0.5, "#ff0000"), (1.0, "#ffffff")])
    assert ramps.ramp_names() == ["zebra h"]
    assert load_settings().ramps["zebra h"] == [[0.0, "#000000"], [0.5, "#ff0000"], [1.0, "#ffffff"]]
    lut = ramps.ramp_colormap("zebra h").getLookupTable(nPts=3, alpha=False)
    assert lut.tolist() == [[0, 0, 0], [255, 0, 0], [255, 255, 255]]
    ramps.delete_ramp("zebra h")
    assert ramps.ramp_names() == [] and ramps.ramp_colormap("zebra h") is None


def test_a_ramp_needs_two_stops_and_a_name_no_builtin_colormap_has():
    from dynamix.shell import ramps

    with pytest.raises(ValueError):
        ramps.save_ramp("viridis", [(0.0, "#000000"), (1.0, "#ffffff")])
    with pytest.raises(ValueError):
        ramps.save_ramp("one stop", [(0.0, "#000000")])
    with pytest.raises(ValueError):
        ramps.save_ramp(" ", [(0.0, "#000000"), (1.0, "#ffffff")])


def test_garbled_ramps_in_the_settings_file_are_left_out():
    from dynamix.shell.settings import load_settings

    path = os.environ["DYNAMIX_SETTINGS_PATH"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ramps": {"ok": [[0, "#000000"], [1, "#ffffff"]], "short": [[0, "#000000"]],
                             "words": [[0, "blue"], [1, "red"]], "far": [[0, "#000000"], [2, "#ffffff"]]}}, f)
    assert load_settings().ramps == {"ok": [[0.0, "#000000"], [1.0, "#ffffff"]]}


def test_the_canvas_and_the_colormap_list_use_the_library(qtbot):
    from dynamix.shell import ramps
    from dynamix.shell.canvas import Canvas
    from dynamix.shell.right_panel import _all_cmap_names

    ramps.save_ramp("mine", [(0.0, "#00ff00"), (1.0, "#0000ff")])
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.linspace(0.0, 1.0, 64).reshape(8, 8))
    canvas.set_colormap("mine")
    lut = np.asarray(canvas.image_item.lut)
    assert lut[0][:3].tolist() == [0, 255, 0] and lut[-1][:3].tolist() == [0, 0, 255]
    assert _all_cmap_names()[:2] == ("mine", None)


def test_the_editor_starts_from_a_colormap_and_saves_a_ramp(qtbot):
    from dynamix.shell import ramps
    from dynamix.shell.ramp_editor import RampEditorDialog

    dialog = RampEditorDialog("binary_r")
    qtbot.addWidget(dialog)
    stops = dialog.stops()
    assert stops[0] == (0.0, "#000000") and stops[-1] == (1.0, "#ffffff")
    dialog.name_edit.setText("viridis")
    assert not dialog.save() and "viridis" in dialog.message.text()
    dialog.name_edit.setText("grey bands")
    assert dialog.save() and ramps.ramp_colormap("grey bands") is not None


def test_the_window_opens_the_editor_lists_the_saved_ramp_and_falls_back_for_a_missing_one(
        loaded, monkeypatch):
    import pyqtgraph as pg

    import dynamix.shell.ramp_editor as editor
    from dynamix.shell.canvas import DEFAULT_COLORMAP
    from dynamix.shell.main_window import _display_style_of

    class _Saves(editor.RampEditorDialog):
        def exec(self):
            self.name_edit.setText("ridge bands")
            return int(self.save())

    monkeypatch.setattr(editor, "RampEditorDialog", _Saves)
    loaded.right_panel.ramp_button.click()
    assert loaded.layer.tags["ui.colormap"] == "ridge bands"
    combo = loaded.right_panel.cmap_combo
    assert combo.itemText(0) == "ridge bands" and combo.currentText() == "ridge bands"
    notes = []
    loaded._notify = lambda msg, *a, **k: notes.append(msg)
    loaded.layer.tags["ui.colormap"] = "someone else's ramp"
    loaded._apply_display_style(_display_style_of(loaded.layer))
    expected = pg.colormap.get(DEFAULT_COLORMAP, source="matplotlib").getLookupTable(nPts=256)
    assert np.array_equal(np.asarray(loaded.canvas.image_item.lut), expected)
    assert len(notes) == 1 and "not in your ramp library" in notes[0]



def test_a_slice_seeds_its_class_colours_from_the_ramp(qtbot):
    from dynamix.shell import ramps
    from dynamix.shell.levels_dialog import LevelsDialog

    ramps.save_ramp("two tone", [(0.0, "#ff0000"), (1.0, "#0000ff")])
    dialog = LevelsDialog(np.linspace(0.0, 1.0, 100), colormap_name="two tone")
    qtbot.addWidget(dialog)
    first, last = dialog._default_color(0, 2), dialog._default_color(1, 2)
    assert first[0] > first[2] and last[2] > last[0]
