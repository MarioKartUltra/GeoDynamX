# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Layers spawned FROM a child dataset must inherit its ``roi.window`` provenance tag.

2026-09-22 (user: analyzer drop on an analyzed ROI child "spawns a sub child but both child
layers have chains and the og ROI child can no longer use the apps controls"): the analyzer
fork, refined-run and band-commit spawn paths all resolve the new layer against the parent's
FIELD (the crop) but minted the layer with EMPTY tags -- so its ``engine.resolve.
source_identity`` lost the ``|win:`` fold and its cache lines collided with a same-chain run
on the FULL dataset (or any other untagged fork of any other window). First writer wins; the
other layer silently displays the wrong result -- the exact collision class the window fold
exists to prevent. ``_on_roi_create`` is unaffected (its window rides in the chain params).
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices


@pytest.fixture
def registered_builtins(clean_registry):
    register_builtin_devices()
    return clean_registry


@pytest.fixture
def win(qtbot, registered_builtins):
    from dynamix.shell.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    n = 64
    f = np.zeros((n, n)); f[:, n // 2:] = 1.0
    field = RasterField(name="dem", values=f, frame=LocalFrame(),
                        x_axis=np.arange(n, dtype=float), y_axis=np.arange(n, dtype=float))
    with qtbot.waitSignal(w.resolved, timeout=15000):
        w.load_field(field, "mem:spawn-tags")
    w._on_roi_child_create({"roi_row": 8, "roi_col": 8, "roi_h": 32, "roi_w": 32})
    for _ in range(100):
        qtbot.wait(10)
        if not w.is_computing:
            break
    assert w.layer.tags.get("roi.window") == "8,8,32,32"
    return w


def test_roi_holder_child_replaces_its_analyzer_in_place(win, qtbot):
    """Contract change 2026-09-22: an
    ROI-holder child NEVER forks -- a different primary analyzer replaces the one in its
    chain, in place, keeping the filter tail and the window tag. (The analyzer fork stays
    the behavior for ordinary analyzed layers; the old fork-with-tag-inheritance pin this
    test replaces recorded the pre-change contract.)"""
    one = [{"device": "mz_edges", "params": {}, "bypassed": False, "rack": None},
           {"device": "scale_select", "params": {}, "bypassed": False, "rack": None}]
    win.strips.set_steps(one, field=win.field)
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win._on_chain_edited(one)
    child_id = win.layer.layer_id
    n_layers = len(win.project.layers)
    two = one + [{"device": "cdf_edges", "params": {}, "bypassed": False, "rack": None}]
    win.strips.set_steps(two, field=win.field)
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win._on_chain_edited(two)
    assert len(win.project.layers) == n_layers          # nothing spawned
    assert win.layer.layer_id == child_id               # same child, edited in place
    devs = [ref.device for ref in win.layer.chain.steps]
    assert devs == ["cdf_edges", "scale_select"]        # replaced head, filter kept
    assert win.layer.tags.get("roi.window") == "8,8,32,32"


def test_refined_run_inherits_the_window_tag(win, qtbot):
    one = [{"device": "mz_edges", "params": {}, "bypassed": False, "rack": None}]
    win.strips.set_steps(one, field=win.field)
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win._on_chain_edited(one)
    child_id = win.layer.layer_id
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win._on_refined_run(child_id)
    refined = win.layer
    assert refined.parent_id == child_id
    assert refined.tags.get("roi.window") == "8,8,32,32"
    assert refined.tags.get("fed_by") == str(child_id)     # the pre-existing tag survives


def test_child_from_an_overview_reads_native_pixels_from_disk(qtbot, registered_builtins,
                                                              tmp_path):
    """2026-09-22: a child drawn on
    a decimated whole-extent overview cuts NATIVE pixels by reading the window straight off
    the source file (the wtmm2d_roi coordinate convention: native = display*ov + off),
    instead of refusing."""
    import rasterio
    from rasterio.transform import from_origin
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.opening import open_field

    tif = tmp_path / "big.tif"
    vals = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
    with rasterio.open(tif, "w", driver="GTiff", height=64, width=64, count=1,
                       dtype="float32", crs="EPSG:32615",
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(vals, 1)

    field = open_field(str(tif), max_pixels=256, mode="overview")   # stride 4 -> 16x16 (legacy path, kept)
    assert int(field.provenance["overview"]) == 4

    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.resolved, timeout=15000):
        w.load_field(field, str(tif))
    w._on_roi_child_create({"roi_row": 2, "roi_col": 3, "roi_h": 4, "roi_w": 5})
    child = w.layer
    assert child.parent_id is not None
    assert child.tags.get("roi.window") == "8,12,16,20"   # display*4 -> native
    cf = w._fields[child.layer_id]
    np.testing.assert_allclose(np.asarray(cf.values), vals[8:24, 12:32])


def test_source_hide_spares_a_child_datasets_own_crop(win, qtbot):
    """2026-09-22: the dataset header's H hides rasters PER SOURCE, and a child shares its
    parent's source by design -- so the crop vanished with the big raster. A layer carrying
    ``roi.window`` is exempt: its crop is its own product, not 'the dataset'."""
    child = win.layer                                  # the fixture leaves the child active
    assert child.tags.get("roi.window")
    win._on_source_hide_toggled(child.source_id, True)
    assert win.canvas.image_item.isVisible()           # the child's own crop stays up
    # ...while the PARENT (no roi.window) hides with its source, as before
    parent = next(l for l in win.project.layers if l.parent_id is None)
    win.layer_list.select_layer(parent.layer_id)
    for _ in range(200):
        qtbot.wait(10)
        if not win.is_computing:
            break
    assert not win.canvas.image_item.isVisible()
