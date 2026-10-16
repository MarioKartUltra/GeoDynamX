# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Map's distance bar follows its units menu: every SI choice, "px" (the selected layer's
pixels, counted at its ground pixel size) and "native" (the map's own unit: metres on a
projected map, degrees on a geographic one); a raster of unknown scale reads pixels whatever
the choice."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
rasterio = pytest.importorskip("rasterio")


def _tif(path, crs, transform):
    v = np.random.default_rng(0).standard_normal((64, 64)).cumsum(0).cumsum(1).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32",
                       crs=crs, transform=transform) as ds:
        ds.write(v, 1)
    return str(path)


def _map(qtbot, path):
    from dynamix.devices import register_builtin_devices
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1200, 800)
    win.show()
    from dynamix.shell.opening import open_field

    win.load_field(open_field(path), path, inert=True)     # no worker outlives the test
    win._set_center_view("map")
    return win, win.map_view


def _pick(mv, choice):
    mv.scale_bar._change_style({"units": choice})
    return mv.scale_bar.text.toPlainText()


def test_px_counts_the_layers_pixels_on_a_projected_map(qtbot, clean_registry, tmp_path):
    from rasterio.transform import from_origin

    win, mv = _map(qtbot, _tif(tmp_path / "utm.tif", "EPSG:32611",
                               from_origin(500000, 4000000, 30, 30)))
    text = _pick(mv, "px")
    assert text.endswith(" px")
    count = float(text.split()[0])
    assert count * 30.0 == pytest.approx(mv.scale_bar.size)     # map units are metres here
    assert _pick(mv, "km").endswith(" km") and _pick(mv, "native").endswith(" m")


def test_native_reads_degrees_on_a_geographic_map(qtbot, clean_registry, tmp_path):
    from rasterio.transform import from_origin

    win, mv = _map(qtbot, _tif(tmp_path / "geo.tif", "EPSG:4326",
                               from_origin(-120.0, 40.0, 0.001, 0.001)))
    assert _pick(mv, "native").endswith("°")
    assert _pick(mv, "m").endswith(" m")


def test_unknown_scale_reads_pixels_whatever_the_choice(qtbot, clean_registry, tmp_path):
    import warnings

    from rasterio.transform import from_origin

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = _tif(tmp_path / "bare.tif", None, from_origin(0, 640, 10, 10))
    win, mv = _map(qtbot, path)
    for choice in ("auto", "km", "native", "px"):
        assert _pick(mv, choice).endswith(" px"), choice
