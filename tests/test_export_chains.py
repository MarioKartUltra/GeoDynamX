# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""File > Export Chains -- the active chain product persisted as the v4 npz.

The dialog-free halves (``_chain_export_blocker`` / ``_export_chain_product``) carry the whole
behavior, so the tests exercise them directly; the click handler is only a file picker around
them. Loadability through the VERBATIM EQSelect loader is the interop gate, same as
tests/test_chain_product.py.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


def _prod_result():
    from dynamix.core.chain_product import attach_chain_product

    n = 6
    l0 = {"x": np.arange(n, dtype=np.int64), "y": np.zeros(n, dtype=np.int64),
          "mod": np.linspace(1.0, 0.4, n), "arg": np.zeros(n),
          "line_id": np.array([0, 0, 0, 1, 1, -1], np.int64)}
    return attach_chain_product({"extrema": [l0], "chains": [],
                                 "scales": np.asarray([1.0]), "_shape": (8, 64)})


def test_export_writes_the_active_product(qtbot, clean_registry, tmp_path):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import load_chains_npz
    from dynamix.shell.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.field = RasterField(name="f", values=np.zeros((8, 64)), frame=LocalFrame(units="px"),
                            x_axis=np.arange(64.0), y_axis=np.arange(8.0))
    win._active_result = _prod_result()
    assert win._chain_export_blocker() is None

    path = tmp_path / "chains.npz"
    msg = win._export_chain_product(str(path))

    assert "2 H" in msg
    te = load_chains_npz(str(path))                    # the verbatim EQSelect loader
    assert len(te["h_segments"]) == 2


def test_export_refuses_without_a_product_and_for_roi_results(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win._active_result = None
    assert "no computed chain product" in win._chain_export_blocker()

    roi = _prod_result()
    roi["_roi"] = {"roi": (0, 0, 8, 8)}
    win._active_result = roi
    win.field = object()
    assert "ROI results" in win._chain_export_blocker()


def test_an_export_writes_its_record_beside_it_and_the_command_writes_it_again(
        qtbot, clean_registry, tmp_path):
    """The chain export writes ``<file>.gprv`` from the active layer; File > Write Provenance
    Record writes it again (the earlier record moves aside); without a layer the export still
    lands and says why it has no record."""
    pytest.importorskip("gprv")
    import json

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    win = MainWindow()
    qtbot.addWidget(win)
    path = tmp_path / "chains.npz"
    win.field = RasterField(name="f", values=np.zeros((8, 64)), frame=LocalFrame(units="px"),
                            x_axis=np.arange(64.0), y_axis=np.arange(8.0))
    win._active_result = _prod_result()
    assert "no provenance record (no active layer)" in win._export_chain_product(str(path))

    values = np.random.default_rng(0).standard_normal((48, 48)).cumsum(0).cumsum(1)
    win.load_field(RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                               x_axis=np.arange(48.0), y_axis=np.arange(48.0)),
                   str(tmp_path / "field.npz"), inert=True)
    from dynamix.engine.resolve import resolve, shown_level
    from dynamix.model.chain import Chain, DeviceRef
    win.layer.chain = Chain((DeviceRef("wtmm2d", {"n_oct": 1, "n_voice": 2, "precision": "f64"}),))
    win._active_result = resolve(win.layer, win.field, win.cache, level=shown_level(win.layer)).result
    msg = win._export_chain_product(str(path))
    assert "provenance chains.npz.gprv" in msg
    record = json.loads((tmp_path / "chains.npz.gprv").read_text(encoding="utf-8"))
    assert record["processStep"][-1]["processingInformation"]["identifier"]["code"] == \
        "export/chains_npz"
    win._on_write_provenance()
    assert "provenance chains.npz.gprv" in win._status_message
    assert len(list(tmp_path.glob("chains.npz.*.gprv"))) == 1


def test_a_preview_is_not_exported_and_gets_no_record(qtbot, clean_registry, tmp_path):
    """While only the finest-scale preview is on show, the export waits for the full stack: a
    record of the full recipe beside preview chains would describe what was not written."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.shell.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.field = RasterField(name="f", values=np.zeros((8, 64)), frame=LocalFrame(units="px"),
                            x_axis=np.arange(64.0), y_axis=np.arange(8.0))
    win._active_result = {**_prod_result(), "_preview": True}
    assert "still computing" in win._chain_export_blocker()
    assert "preview" in win._write_provenance(tmp_path / "x.npz", "chains_npz")
