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
