# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A figure's result too large for memory lives in the scratch folder beside the settings file:
the signed scalar CWT stack is written scale by scale into a memory-mapped file there, the window
removes its file when the result is replaced or the window closes, and the app sweeps the folder
at start for whatever a crash left behind."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from test_directional_figures import source


def test_a_signed_stack_over_the_memory_budget_lands_in_the_scratch_folder(qtbot, tmp_path,
                                                                         monkeypatch):
    from dynamix.core.wtmm_backend import cwt2d_scalar
    from dynamix.shell import directional_figures as figures
    from dynamix.shell.directional_figures import DirectionalFigure
    from dynamix.shell.settings import scratch_dir

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(figures, "SIGNED_STACK_BUDGET", 1)          # every stack is over budget
    field, result = source()
    w = DirectionalFigure("scalar")
    qtbot.addWidget(w)
    w.set_result(result, field)
    w.compute.click()
    qtbot.waitUntil(lambda: not w._busy, timeout=30000)
    signed = w._computed["signed"]
    assert isinstance(signed, np.memmap)
    path = Path(signed.filename)
    assert path.parent == scratch_dir() and path.exists()
    expected = cwt2d_scalar(np.asarray(field.values), w._computed["scales"],
                            wavelet=w.wavelet.currentText(), signed=True)["signed"]
    np.testing.assert_allclose(np.asarray(signed), expected, rtol=1e-5, atol=1e-6)
    w.close()
    assert not path.exists()


def test_the_app_sweeps_the_scratch_folder_at_start(tmp_path, monkeypatch):
    from dynamix.shell.settings import scratch_dir, sweep_scratch

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "settings.json"))
    stray = scratch_dir() / "signed-deadbeef.npy"
    stray.write_bytes(b"\0" * 16)
    sweep_scratch()
    assert scratch_dir().is_dir() and not stray.exists()
