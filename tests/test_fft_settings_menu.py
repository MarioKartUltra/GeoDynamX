# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Settings > Compute engine / Precision (2026-09-22): persisted, applied at STARTUP through
fft_policy.configure (changes apply after a restart)."""
from __future__ import annotations

import json

import pytest

from dynamix.core import fft_policy


@pytest.fixture
def registered_builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _window(qtbot):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=())
    qtbot.addWidget(win)
    return win


def test_startup_configures_the_fft_policy_from_settings(qtbot, registered_builtins,
                                                         tmp_path, monkeypatch):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"compute_engine": "fftw", "compute_precision": 64}))
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(path))
    _window(qtbot)
    be = fft_policy.active()
    assert (be.name, be.precision) == ("pyfftw", 64)


def test_the_menu_offers_auto_mlx_fftw_and_32_64_and_never_numpy(qtbot, registered_builtins):
    win = _window(qtbot)
    assert set(win._engine_actions) == {"auto", "mlx", "fftw"}
    assert set(win._precision_actions) == {32, 64}
    assert win._precision_actions[32].isChecked()


def test_a_menu_choice_persists_and_says_restart(qtbot, registered_builtins, tmp_path,
                                                 monkeypatch):
    from dynamix.shell.settings import load_settings

    win = _window(qtbot)
    notes = []
    win._notify = lambda msg, *a, **k: notes.append(msg)
    win._precision_actions[64].trigger()
    assert load_settings().compute_precision == 64
    win._engine_actions["fftw"].trigger()
    assert load_settings().compute_engine == "fftw"
    assert all("restart" in n.lower() for n in notes) and len(notes) == 2
