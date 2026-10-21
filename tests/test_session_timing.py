# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Session diagnostics preserve worker delivery and remain opt-in."""
import json

import pytest

from dynamix.shell.main_window import MainWindow


@pytest.mark.parametrize("enabled", [False, True])
def test_landing_diagnostics_are_opt_in(qtbot, clean_registry, monkeypatch, capsys, enabled):
    from dynamix.devices import register_builtin_devices
    from dynamix.core.rasterfield import RasterField
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    monkeypatch.setenv("DYNAMIX_TIMING", "1" if enabled else "0")
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_field(RasterField.load_npz("tests/fixtures/kam_64.npz"), "/nonexistent/kam.npz")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    steps = [{"device": "wtmm2d", "params": {
        **defaults_for(get_device("wtmm2d")), "n_oct": 2, "n_voice": 2}}]
    win.strips.set_steps(steps, field=win.field)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_chain_edited(steps)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    rows = [json.loads(line.removeprefix("DYNAMIX_TIMING "))
            for line in capsys.readouterr().err.splitlines() if line.startswith("DYNAMIX_TIMING ")]
    if not enabled:
        assert rows == []
        assert "_session_timing" not in win.__dict__
        assert "_session_timing" not in win.canvas.__dict__
        return
    landing = next(row for row in rows if row["event"] == "landing_ms" and row["engine_ms"] > 0)
    assert landing["engine_ms"] > 0
    assert landing["apply_ms"] > 0
    assert landing["windows_fed"] >= 1
    assert landing["cache_entries"] > 0
    assert landing["python_objects"] > 0
    assert landing["object_types"]
    assert landing["rss_bytes"] > 0
    win.show()
    qtbot.waitExposed(win)
    win.canvas.grab()
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.transport._set_index(1)
    ticks = [json.loads(line.removeprefix("DYNAMIX_TIMING "))
             for line in capsys.readouterr().err.splitlines() if line.startswith("DYNAMIX_TIMING ")]
    assert any(row["event"] == "scale_tick_ms" for row in ticks)
    assert any(row["paint_count"] > 0 and row["paint_ms"] > 0 for row in ticks)


def test_cache_count_survives_a_background_insertion(qtbot, monkeypatch, capsys):
    from types import SimpleNamespace
    import numpy as np
    from dynamix.engine import cache as cache_module
    from dynamix.shell import session_timing

    cache = cache_module.Cache()
    cache.put("first", np.ones(4))
    window = SimpleNamespace(cache=cache)
    original = cache_module.stage_nbytes
    inserted = False

    def count_with_insertion(value):
        nonlocal inserted
        if not inserted:
            inserted = True
            cache.put("background", np.ones(8))
        return original(value)

    monkeypatch.setattr(cache_module, "stage_nbytes", count_with_insertion)
    session_timing.emit(window, "landing_ms")
    row = json.loads(capsys.readouterr().err.removeprefix("DYNAMIX_TIMING "))
    assert row["cache_entries"] == 1
    assert row["cache_bytes"] == 32
    assert len(cache) == 2
