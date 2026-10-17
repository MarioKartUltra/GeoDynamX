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


def test_the_map_view_can_be_the_first_view_before_any_landing(qtbot, clean_registry):
    """A launch that restores the Map as the centre view builds the map view before any
    result has landed; the bars' state must exist from construction (the Dock launch at
    7213f4c died here with an AttributeError on ``_bar_device``)."""
    from dynamix.shell.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert window._canvas_bar_state() == (None, "")
    window._ensure_map_view()
    window._set_center_view("map")
    assert window.map_view is not None


def test_an_unknown_theme_name_falls_back_to_restrained_dark(capsys):
    import pytest

    from dynamix.shell.app import _theme_for
    from dynamix.shell.theme import RESTRAINED_DARK, load_theme

    with pytest.raises(KeyError):
        load_theme("nope")
    assert _theme_for("nope") == RESTRAINED_DARK
    assert "nope" in capsys.readouterr().err
    assert _theme_for("light").name == "light"
