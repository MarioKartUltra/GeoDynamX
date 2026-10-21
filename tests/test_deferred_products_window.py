# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The window's landings of the demo chain build no partition tables, chain product or chain
graph; opening the partition-function window builds the tables, once."""
from __future__ import annotations

import collections

import numpy as np

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices


def test_window_landings_defer_the_products_until_a_fitting_window_reads_them(
        qtbot, fbm64, clean_registry, monkeypatch):
    from dynamix.core import chain_product, wtmm_backend
    from dynamix.shell.main_window import MainWindow
    from dynamix.topology import wtmm as topology_wtmm

    register_builtin_devices()
    counts = collections.Counter()
    for owner, name in ((wtmm_backend.PythonWTMMBackend, "partition2d"),
                        (chain_product, "build_chain_product"),
                        (topology_wtmm, "build_chain_topology")):
        def counted(*args, _name=name, _original=getattr(owner, name), **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, name, counted)

    field = RasterField(name="fbm", values=fbm64, frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    win = MainWindow()
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000) as sig:
        win.load_field(field, "mem:fbm")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.transport.slider.setValue(2)
    result = sig.args[0].result
    assert "topology" in result and win.canvas.extrema_item.getData()[0] is not None
    assert counts == {}
    assert win._spectrum_button.isEnabled()

    win._on_spectrum_button_clicked()
    assert counts == {"partition2d": 1}
    std, _cmax = wtmm_backend.PythonWTMMBackend().partition2d(
        result["chains"], result["scales"], q_list=result["params"]["q_list"],
        min_chain_len=result["params"]["min_chain_len"])
    np.testing.assert_array_equal(win._active_result["hd_std"]["tau_qa"], std["tau_qa"])
    win._multifractal_window.close()
