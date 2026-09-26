# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Stack datasets in the layer tree: each imported group its own dataset, a "Bands" row
listing its bands, and removing a band from the project (never the file) with its identity."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets  # noqa: F401

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine import source_identity
from tests.test_ingest import _aster_like
from tests.test_shell_window import stub_devices, window  # noqa: F401


def test_composite_spec_forgets_a_removed_band_and_shifts_the_rest():
    from dynamix.shell.main_window import _composite_without_band

    raw = json.dumps({"r": 0, "g": 1, "b": 2, "solo": [1, 3], "mute": [2],
                      "stretch_pct": 2.0})
    out = _composite_without_band(raw, 1)
    assert (out["r"], out["g"], out["b"]) == (0, None, 1)
    assert out["solo"] == [2] and out["mute"] == [1] and out["stretch_pct"] == 2.0


@pytest.fixture
def aster(window, tmp_path):
    pytest.importorskip("pyhdf")
    try:
        path = _aster_like(tmp_path)
    except Exception as exc:                          # pragma: no cover - env guard
        pytest.skip(f"pyhdf cannot create HDF4 groups here: {exc}")
    window._start_worker = lambda: None
    window._open_container_groups(str(path), [
        ("VNIR (12, 10)", ["VNIR_Band1/ImageData", "VNIR_Band2/ImageData"]),
        ("SWIR (6, 8)", ["SWIR_Band4/ImageData"])])
    return window, path


def _masters(win):
    return [l for l in win.project.layers if l.parent_id is None]


def test_each_group_from_one_file_is_its_own_dataset_row_and_identity(aster):
    win, path = aster
    vnir, swir = _masters(win)
    assert vnir.source_id != swir.source_id
    assert source_identity(vnir) != source_identity(swir)       # no shared cache lines
    src = win.project.sources[vnir.source_id]
    assert src.path == str(path) and src.bands == ["VNIR_Band1/ImageData",
                                                   "VNIR_Band2/ImageData"]
    headers = win.layer_list._source_headers
    assert headers[vnir.source_id] is not headers[swir.source_id]
    assert headers[vnir.source_id].text(0).endswith("VNIR (12, 10)")


def _band_layers(win, master):
    return [l for l in win.project.layers
            if l.parent_id == master.layer_id and l.tags.get("band.id")]


def test_a_stack_datasets_bands_are_child_rows_under_a_collapsed_bands_row(aster):
    win, _path = aster
    vnir, swir = _masters(win)
    bands = _band_layers(win, vnir)
    assert [l.name for l in bands] == ["VNIR_Band1", "VNIR_Band2"]
    assert all(l.chain.steps[0].device == "band_select" for l in bands)
    group = win.layer_list._band_groups[vnir.source_id]
    assert group.text(0) == "Bands (2)" and not group.isExpanded()
    assert win.layer_list._layer_items[bands[0].layer_id].parent() is group
    assert _band_layers(win, swir) == []                     # one band: no band rows


def test_selecting_a_band_row_shows_that_band_and_noise_on_it_stays_on_it(window, qtbot,
                                                                         tmp_path):
    pytest.importorskip("pyhdf")
    try:
        path = _aster_like(tmp_path)
    except Exception as exc:                          # pragma: no cover - env guard
        pytest.skip(f"pyhdf cannot create HDF4 groups here: {exc}")
    win = window
    win._open_container_groups(str(path), [
        ("VNIR (12, 10)", ["VNIR_Band1/ImageData", "VNIR_Band2/ImageData"])])
    (master,) = _masters(win)
    b2 = _band_layers(win, master)[1]
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win.layer_list.select_layer(b2.layer_id)
    np.testing.assert_array_equal(win.canvas._field.values,
                                  win._fields[master.layer_id].values[..., 1])
    assert not win._composite_frame.isVisibleTo(win)         # one band: a ramp, no mixer
    desc = [{"device": "band_select", "params": dict(b2.chain.steps[0].params)},
            {"device": "noise", "params": {"amplitude": 5.0}}]
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    assert win.layer is b2 and [s.device for s in b2.chain.steps] == ["band_select", "noise"]
    shown = np.asarray(win.canvas._field.values)
    assert shown.shape == (12, 10)
    assert not np.array_equal(shown, win._fields[master.layer_id].values[..., 1])


def test_removing_a_band_row_removes_the_band_and_keeps_the_others_by_name(aster, qtbot):
    win, path = aster
    vnir, _swir = _masters(win)
    before = source_identity(vnir)
    b1, b2 = _band_layers(win, vnir)
    notes = []
    win._notify = lambda msg, *a, **k: notes.append(msg)
    menu = win.layer_list._build_context_menu(win.layer_list._layer_items[b1.layer_id],
                                              b1.layer_id)
    assert [a.text() for a in menu.actions()][0] == "Remove band from project"
    menu.actions()[0].trigger()
    assert b1 not in win.project.layers and b2 in win.project.layers
    src = win.project.sources[vnir.source_id]
    assert src.bands == ["VNIR_Band2/ImageData"]
    field = win._fields[vnir.layer_id]
    assert field.values.shape == (12, 10) and field.values[0, 0] == 500.0   # band 2 left
    assert source_identity(vnir) != before
    # the surviving row still resolves ITS band, by name, on the reduced dataset
    from dynamix.model.device import get_device

    out = get_device("band_select").compute(field, dict(b2.chain.steps[0].params))
    np.testing.assert_array_equal(out.values, field.values)
    assert win.layer_list._band_groups[vnir.source_id].text(0) == "Bands (1)"
    assert path.exists() and any("untouched" in n for n in notes)
    # the last band cannot go
    win._on_remove_band_layer(b2.layer_id)
    assert b2 in win.project.layers and any("last band" in n for n in notes)


def test_delete_key_on_a_band_row_asks_to_remove_its_band(aster, qtbot):
    from PySide6 import QtGui

    win, _path = aster
    vnir, _swir = _masters(win)
    b2 = _band_layers(win, vnir)[1]
    win.layer_list.setCurrentItem(win.layer_list._layer_items[b2.layer_id])
    with qtbot.waitSignal(win.layer_list.removeBandRequested) as blocker:
        win.layer_list.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                                     QtCore.Qt.Key_Delete,
                                                     QtCore.Qt.NoModifier))
    assert blocker.args == [b2.layer_id]


def test_a_band_removed_from_a_multiband_file_reopens_as_its_subset(window, tmp_path):
    win = window
    win._start_worker = lambda: None
    v = np.stack([np.full((6, 7), float(k)) for k in range(3)], axis=-1)
    p = tmp_path / "stack.npz"
    RasterField(name="stack", values=v, frame=LocalFrame(), x_axis=np.arange(7.0),
                y_axis=np.arange(6.0)).save_npz(p)
    win.load_field(RasterField.from_file(p), str(p))
    master = win.layer
    bands = _band_layers(win, master)
    assert [l.tags["band.id"] for l in bands] == ["#0", "#1", "#2"]
    win._on_remove_band_layer(bands[1].layer_id)
    src = win.project.sources[master.source_id]
    assert src.bands == ["#0", "#2"]
    from dynamix.core.ingest import load_grid_stack

    again = load_grid_stack(str(p), src.bands)
    np.testing.assert_array_equal(again.values, win._fields[master.layer_id].values)


def test_a_devloop_rebuild_keeps_every_dataset_with_its_own_field(aster, qtbot):
    from dynamix import devloop

    win, _path = aster
    vnir, swir = _masters(win)
    win.layer_list.select_layer(swir.layer_id)
    saved = {k: devloop.SESSION.get(k) for k in devloop.SESSION}
    try:
        devloop._capture(win)
        app = QtWidgets.QApplication.instance()
        rebuilt = devloop._build_window(app)
        qtbot.addWidget(rebuilt)
        assert vnir.layer_id in rebuilt._layer_by_id and swir.layer_id in rebuilt._layer_by_id
        assert rebuilt._fields[vnir.layer_id].values.shape == (12, 10, 2)
        assert rebuilt._fields[swir.layer_id].values.shape == (6, 8)
        assert len({l.source_id for l in rebuilt.project.layers}) == 2   # no stray source
        assert rebuilt.layer is swir
    finally:
        devloop.SESSION.update(saved)
