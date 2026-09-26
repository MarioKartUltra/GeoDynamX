# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The footprint browser in the shell: scan a folder, outline every raster on the
world, right-click a footprint -> Import. Qt-only offscreen (no pyvista needed: the arrangement
view is never built here; the MainWindow half is tested through its own methods)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from dynamix.geo.footprints import Footprint  # noqa: E402
from dynamix.shell.settings import load_settings, update_settings  # noqa: E402
from tests.test_shell_window import stub_devices, window  # noqa: E402,F401


def _tile(path, west, north, n=8, dtype="int16"):
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype=dtype,
                       crs=CRS.from_epsg(4326), transform=from_origin(west, north, 1.0 / n, 1.0 / n)) as dst:
        dst.write(np.zeros((n, n), dtype=dtype), 1)


def _fp(name, west, south, east, north, dtype="int16"):
    return Footprint(path=f"/data/{name}.tif", name=name, width=3601, height=3601, count=1,
                     dtype=dtype, crs="EPSG:4326", bounds=(west, south, east, north),
                     corners=((west, south), (east, south), (east, north), (west, north)))


def test_scan_action_registers_footprints_persists_the_folder_and_reports(window, tmp_path, monkeypatch):
    _tile(tmp_path / "ASTGTMV003_S22E119_dem.tif", 119.0, -21.0)
    _tile(tmp_path / "ASTGTMV003_S22E119_num.tif", 119.0, -21.0, dtype="uint8")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    window._on_scan_footprints_clicked()
    assert [f.name for f in window._footprints] == ["ASTGTMV003_S22E119_dem", "ASTGTMV003_S22E119_num"]
    assert load_settings().footprint_folders == [str(tmp_path)]
    assert "2 footprints" in window.statusBar().currentMessage()


def test_scanning_the_same_folder_twice_does_not_duplicate_footprints(window, tmp_path, monkeypatch):
    _tile(tmp_path / "a.tif", 119.0, -21.0)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    window._on_scan_footprints_clicked()
    window._on_scan_footprints_clicked()
    assert len(window._footprints) == 1 and load_settings().footprint_folders == [str(tmp_path)]


def test_cancelled_folder_dialog_changes_nothing(window, monkeypatch):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    window._on_scan_footprints_clicked()
    assert window._footprints == [] and load_settings().footprint_folders == []


def test_footprint_folders_are_rescanned_when_a_window_opens(tmp_path, qtbot, stub_devices):  # noqa: F811
    _tile(tmp_path / "a.tif", 119.0, -21.0)
    update_settings(footprint_folders=[str(tmp_path), str(tmp_path / "gone")])
    from dynamix.shell.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    assert [f.name for f in win._footprints] == ["a"]


def test_footprint_menu_one_import_per_raster_and_a_submenu_for_stacked_variants(window, monkeypatch):
    dem = _fp("ASTGTMV003_S22E119_dem", 119, -22, 120, -21)
    num = _fp("ASTGTMV003_S22E119_num", 119, -22, 120, -21, dtype="uint8")
    east = _fp("ASTGTMV003_S22E120_dem", 120, -22, 121, -21)
    opened = []
    monkeypatch.setattr(window, "open_path", lambda p: opened.append(p))
    menu = window._footprint_menu([dem, num, east])
    top = menu.actions()
    assert [a.text() for a in top] == ["ASTGTMV003_S22E119 ▸", "Import ASTGTMV003_S22E120_dem",
                                       "Preview ASTGTMV003_S22E120_dem"]
    sub = top[0].menu()
    assert sub is not None
    assert [a.text() for a in sub.actions()] == ["Preview scene (dem)", "", "dem  (int16, 3601×3601)",
                                                 "num  (uint8, 3601×3601)"]
    sub.actions()[3].trigger()
    top[1].trigger()
    assert opened == ["/data/ASTGTMV003_S22E119_num.tif", "/data/ASTGTMV003_S22E120_dem.tif"]


def test_footprint_menu_names_aster_granules_by_date_newest_first_with_band_entries(window):
    def granule(gid, west):
        base = f"AST_07XT_{gid}_20250809164339_SRF_VNIR"
        return [_fp(f"{base}_{b}", west, -22, west + 0.6, -21.4) for b in ("B01", "B02", "B03N")]
    old = granule("00411282015022621", 119.0)      # 2015-11-28
    new = granule("00410122018022055", 119.01)     # 2018-10-12 (slightly offset footprint)
    menu = window._footprint_menu(old + new)
    top = menu.actions()
    assert [a.text() for a in top] == ["AST_07XT · 2018-10-12 02:20 ▸", "AST_07XT · 2015-11-28 02:26 ▸"]
    assert [a.text() for a in top[1].menu().actions()] == ["Preview scene (B03N)", "", "B01  (int16, 3601×3601)",
                                                            "B02  (int16, 3601×3601)", "B03N  (int16, 3601×3601)"]


def test_footprint_menu_for_no_hits_is_empty(window):
    assert window._footprint_menu([]).actions() == []


def test_footprint_menu_keeps_a_granules_vnir_and_swir_bands_in_one_submenu(window):
    base = "AST_07XT_00410302004021233_20250402184734_SRF"
    vnir = _fp(f"{base}_VNIR_B01", 119.0, -22.0, 119.6, -21.4)
    swir = Footprint(path=f"/data/{base}_SWIR_B04.tif", name=f"{base}_SWIR_B04", width=2490, height=2100,
                     count=1, dtype="int16", crs="EPSG:32750", bounds=(0, 0, 1, 1),
                     corners=((119.00005, -22.0001), (119.60005, -22.0001), (119.60005, -21.4001), (119.00005, -21.4001)))
    menu = window._footprint_menu([vnir, swir])
    top = menu.actions()
    assert [a.text() for a in top] == ["AST_07XT · 2004-10-30 02:12 ▸"]
    assert [a.text() for a in top[0].menu().actions()] == ["Preview scene (VNIR_B01)", "",
                                                            "VNIR_B01  (int16, 3601×3601)",
                                                            "SWIR_B04  (int16, 2490×2100)"]


# ------------------------------------------------------------------- previews

def test_footprint_menu_offers_preview_per_scene_and_clear_when_previews_exist(window, monkeypatch):
    import dynamix.shell.main_window as mw
    base = "AST_07XT_00410302004021233_20250402184734_SRF"
    b01 = _fp(f"{base}_VNIR_B01", 119.0, -22.0, 119.6, -21.4)
    b3n = _fp(f"{base}_VNIR_B03N", 119.0, -22.0, 119.6, -21.4)
    dem = _fp("ASTGTMV003_S22E119_dem", 119, -22, 120, -21)
    built = []
    monkeypatch.setattr(mw, "overview_field", lambda path, max_px=400: built.append(path) or object())
    menu = window._footprint_menu([b01, b3n, dem])
    actions = menu.actions()          # hold the QAction wrappers: a temporary one takes its submenu with it
    top = [a.text() for a in actions]
    assert top[:2] == ["AST_07XT · 2004-10-30 02:12 ▸", "Import ASTGTMV003_S22E119_dem"]
    sub = actions[0].menu()
    sub_texts = [a.text() for a in sub.actions()]
    assert sub_texts[0] == "Preview scene (B03N)" and sub_texts[1] == ""  # separator, then bands
    assert "Clear previews" not in top
    sub.actions()[0].trigger()
    assert built == ["/data/AST_07XT_00410302004021233_20250402184734_SRF_VNIR_B03N.tif"]
    assert len(window._previews) == 1
    menu2 = window._footprint_menu([dem])
    actions2 = menu2.actions()
    texts = [a.text() for a in actions2]
    assert texts[-1] == "Clear previews" and "Preview ASTGTMV003_S22E119_dem" in texts
    actions2[-1].trigger()
    assert window._previews == {}


def test_previewing_a_scene_twice_replaces_not_duplicates(window, monkeypatch):
    import dynamix.shell.main_window as mw
    monkeypatch.setattr(mw, "overview_field", lambda path, max_px=400: object())
    dem = _fp("ASTGTMV003_S22E119_dem", 119, -22, 120, -21)
    window._preview_footprints([dem]); window._preview_footprints([dem])
    assert len(window._previews) == 1
