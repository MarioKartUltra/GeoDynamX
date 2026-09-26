# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Offscreen tests for the two histogram dialogs (levels_dialog.py):
LevelsDialog = display coloring only; BandDialog = the h-band reconstruction mask only."""
from __future__ import annotations

import numpy as np
import pytest
import pyqtgraph as pg

from dynamix.shell.levels_dialog import BandDialog, LevelsDialog


def _values():
    rng = np.random.default_rng(0)
    v = rng.normal(0.3, 0.4, (128, 128))
    v[0, 0] = np.nan
    return v


# --------------------------------------------------------------------- LevelsDialog (display)

def test_levels_builds_and_emits_the_tag_texts(qtbot):
    dialog = LevelsDialog(_values(), levels_text="", colors_text="")
    qtbot.addWidget(dialog)
    assert len(dialog._lines) == 4                 # default 5 classes -> 4 quantile bounds
    assert len(dialog._colors) == 5
    got = {}
    dialog.levelsApplied.connect(lambda b, c, s: got.update(breaks=b, colors=c, sieve=s))
    dialog._emit()
    assert got["sieve"] == 0
    breaks = [float(x) for x in got["breaks"].split(",")]
    assert breaks == sorted(breaks) and len(breaks) == 4
    assert len(got["colors"].split(",")) == 5
    assert all(c.startswith("#") and len(c) == 7 for c in got["colors"].split(","))


def test_levels_seeds_rgba_and_round_trips_transparent_classes(qtbot):
    """"none" classes (the set display) survive seed -> emit unchanged, and their tint region
    carries alpha 0."""
    dialog = LevelsDialog(_values(), levels_text="-0.1, 0.5",
                          colors_text="none,#ff8800,none")
    qtbot.addWidget(dialog)
    assert dialog.bounds() == [-0.1, 0.5]
    assert dialog._colors == [(0, 0, 0, 0), (255, 136, 0, 255), (0, 0, 0, 0)]
    assert dialog._regions[0].brush.color().alpha() == 0
    assert dialog._regions[1].brush.color().alpha() > 0
    got = {}
    dialog.levelsApplied.connect(lambda b, c, s: got.update(breaks=b, colors=c))
    dialog._emit()
    assert got["colors"] == "none,#ff8800,none"


def test_levels_add_remove_bounds_track_classes(qtbot):
    dialog = LevelsDialog(_values(), levels_text="0.0", colors_text="")
    qtbot.addWidget(dialog)
    assert len(dialog._lines) == 1 and len(dialog._colors) == 2
    dialog._on_add_bound()
    assert len(dialog._lines) == 2 and len(dialog._colors) == 3
    dialog._on_remove_bound()
    dialog._on_remove_bound()                       # floor at one bound
    assert len(dialog._lines) == 1 and len(dialog._colors) == 2


def test_levels_has_no_reconstruction_surface(qtbot):
    """The split's contract: the display dialog cannot drive band_recon."""
    dialog = LevelsDialog(_values())
    qtbot.addWidget(dialog)
    assert not hasattr(dialog, "reconstructRequested")
    assert not hasattr(dialog, "bandPreviewRequested")


# --------------------------------------------------------------------- BandDialog (analysis)

def test_band_seeds_from_knobs_and_emits_both_signals(qtbot):
    dialog = BandDialog(_values(), h_lo=-0.1, h_hi=0.5)
    qtbot.addWidget(dialog)
    assert dialog.band() == (-0.1, 0.5)
    got = {}
    dialog.reconstructRequested.connect(lambda lo, hi, s: got.update(commit=(lo, hi, s)))
    dialog.bandPreviewRequested.connect(
        lambda lo, hi, mode, s: got.update(preview=(lo, hi, mode, s)))
    dialog._on_reconstruct()
    assert got["commit"] == (-0.1, 0.5, 0)
    dialog._band_region.setRegion((0.0, 0.4))
    qtbot.wait(120)                                 # the 50 ms coalescer
    assert got["preview"] == (0.0, 0.4, "mask", 0)  # binary set, unsieved mid-drag
    # the toggle: switching preview mode re-emits with the new mode, same band
    dialog._preview_combo.setCurrentText("reconstruction")
    qtbot.wait(120)
    assert got["preview"] == (0.0, 0.4, "reconstruction", 0)


def test_band_settle_fires_one_sieved_preview(qtbot):
    """The lazy contract: ticks carry sieve 0; ~400 ms after the last gesture ONE sieved
    preview fires with the spin's value."""
    dialog = BandDialog(_values(), h_lo=-0.1, h_hi=0.5)
    qtbot.addWidget(dialog)
    got = []
    dialog.bandPreviewRequested.connect(lambda lo, hi, mode, s: got.append(s))
    dialog._island_spin.setValue(25)
    dialog._band_region.setRegion((0.0, 0.4))
    qtbot.wait(120)
    assert got and all(s == 0 for s in got)         # nothing sieved mid-drag
    qtbot.wait(450)
    assert got[-1] == 25                            # the settle pass


def test_band_writes_no_display_tags(qtbot):
    """The other half of the contract: no levelsApplied on the mask dialog."""
    dialog = BandDialog(_values())
    qtbot.addWidget(dialog)
    assert not hasattr(dialog, "levelsApplied")


# --------------------------------------------------------------------- shared histogram base

def test_histogram_zoom_is_horizontal_only(qtbot):
    for dialog in (LevelsDialog(_values()), BandDialog(_values())):
        qtbot.addWidget(dialog)
        vb = dialog._plot.getViewBox()
        assert vb.state["mouseEnabled"] == [True, False]
        assert dialog._bin_spin.value() > 0         # bin control present on both


def test_levels_live_preview_coalesces_bound_drags(qtbot):
    """Live delineation: dragging a bound / picking a color emits the coalesced
    PREVIEW signal (same payload as Apply, no tag semantics); unchecking Live silences it."""
    dialog = LevelsDialog(_values(), levels_text="0.0", colors_text="#ff0000,#00ff00")
    qtbot.addWidget(dialog)
    got = []
    dialog.levelsPreviewRequested.connect(lambda b, c, s: got.append((b, c, s)))
    dialog._lines[0].setValue(0.25)
    qtbot.wait(140)
    assert got and got[-1][0] == "0.25"
    assert got[-1][1] == "#ff0000,#00ff00"
    assert got[-1][2] == 0                      # drag ticks never sieve (the lazy contract)
    dialog._live_check.setChecked(False)
    n = len(got)
    dialog._lines[0].setValue(0.4)
    qtbot.wait(140)
    assert len(got) == n


def test_dragging_a_class_span_moves_both_bounds_together(qtbot):
    """Grabbing the middle of a class span shifts BOTH bounding lines by the drag delta (the
    recon dialog's gesture); an outermost span moves its one real bound."""
    dialog = LevelsDialog(_values(), levels_text="0.0, 0.5", colors_text="")
    qtbot.addWidget(dialog)
    assert dialog.bounds() == [0.0, 0.5]
    # region 1 spans the two real bounds: drag it by +0.1
    lo, hi = dialog._regions[1].getRegion()
    dialog._regions[1].setRegion((lo + 0.1, hi + 0.1))
    assert dialog.bounds()[0] == pytest.approx(0.1)
    assert dialog.bounds()[1] == pytest.approx(0.6)
    # outermost span (region 0): only the real right bound moves
    lo0, hi0 = dialog._regions[0].getRegion()
    dialog._regions[0].setRegion((lo0 - 0.05, hi0 - 0.05))
    assert dialog.bounds()[0] == pytest.approx(0.05)
    assert dialog.bounds()[1] == pytest.approx(0.6)
