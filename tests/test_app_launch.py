# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""``dynamix.shell.app`` / ``dynamix.devloop`` launch defaults: which raster opens at startup.

Qt-only offscreen subset (imports the shell through ``app``; ``QtWidgets`` is imported so the
grep-based subset split files this correctly)."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets  # noqa: F401

from dynamix.shell import app, settings
from tests.test_shell_window import stub_devices, window  # noqa: F401


def test_default_raster_honours_open_on_launch_when_the_file_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    raster = tmp_path / "bathy.tif"
    raster.write_bytes(b"")
    settings.update_settings(open_on_launch=str(raster))
    assert app._default_raster() == str(raster)


def test_default_raster_falls_back_when_open_on_launch_is_missing_or_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    fallback = app._default_raster()
    assert Path(fallback).exists()
    settings.update_settings(open_on_launch=str(tmp_path / "gone.tif"))
    assert app._default_raster() == fallback


def test_open_path_reads_a_geotiff_window_of_the_configured_size(tmp_path, monkeypatch, window):
    """A big GeoTIFF opens as a centred window; ``Settings.open_window_px`` sets its edge ("a mini bathymetry window" for fast shell testing)."""
    import dynamix.shell.main_window as mw
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    settings.update_settings(open_window_px=1024)
    seen = {}

    def fake_open_field(path, **kw):
        seen.update(kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr(mw, "open_field", fake_open_field)
    try:
        window.open_path(str(tmp_path / "big.tif"))
    except RuntimeError:
        pass
    assert seen["window_size"] == 1024
