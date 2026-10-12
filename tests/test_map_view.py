# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Map view draws the stack's raster rows by georeference in one map CRS."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.layer_stack import Row
from dynamix.shell.map_view import MapView, RowContent

STYLE = {"colormap": "viridis", "stretch": "linear", "stretch_pct": 2.0, "hillshade": False,
         "sun_azimuth": 315.0, "sun_altitude": 45.0, "z_factor": 1.0, "levels": "",
         "levels_colors": "", "levels_sieve": 0}


def _utm(ny=40, nx=50, x0=500005.0, y0=4399995.0, d=10.0, crs="EPSG:32613"):
    f = RasterField(name="f", values=np.random.default_rng(0).standard_normal((ny, nx)),
                    frame=LocalFrame(x0=x0, y0=y0, dx=d, dy=d, units="metre"),
                    x_axis=x0 + d * np.arange(nx), y_axis=y0 - d * np.arange(ny))
    if crs:
        f.provenance["crs"] = crs
    return f


def _view(qtbot, crs="EPSG:32613"):
    v = MapView()
    qtbot.addWidget(v)
    v.set_map_crs(crs)
    return v


def test_a_same_crs_raster_sits_on_its_georeferenced_extent(qtbot):
    v = _view(qtbot)
    v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                 lambda rid: RowContent("raster", field=_utm(), style=STYLE))
    r = v.item_rect("src:a")
    assert (r.left(), r.right()) == pytest.approx((500000.0, 500500.0))
    assert (min(r.top(), r.bottom()), max(r.top(), r.bottom())) == pytest.approx(
        (4399600.0, 4400000.0))


def test_a_fork_lands_inside_its_parents_roi(qtbot):
    parent = _utm()
    fork = _utm(ny=10, nx=12, x0=500005.0 + 10 * 20, y0=4399995.0 - 10 * 8)   # ROI (8, 20, 10, 12)
    v = _view(qtbot)
    v.show_stack([(Row("src:a", "raster"), True, 1.0), (Row("src:b", "raster"), True, 1.0)],
                 lambda rid: RowContent("raster", field=parent if rid == "src:a" else fork,
                                        style=STYLE))
    r = v.item_rect("src:b")
    assert (r.left(), r.right()) == pytest.approx((500200.0, 500320.0))
    assert max(r.top(), r.bottom()) == pytest.approx(4400000.0 - 80.0)


def test_rows_it_cannot_place_are_greyed_with_a_reason(qtbot):
    v = _view(qtbot)
    reasons = v.show_stack(
        [(Row("src:a", "raster"), True, 1.0), (Row("src:b", "raster"), True, 1.0),
         (Row("prod:1", "product"), True, 1.0)],
        lambda rid: {"src:a": RowContent("raster", field=_utm(crs=None), style=STYLE),
                     "src:b": RowContent("raster", field=_utm(), style=STYLE),
                     "prod:1": RowContent("product", reason="not run")}[rid])
    assert reasons == {"src:a": "no CRS", "prod:1": "not run"}


def test_another_crs_is_reprojected_for_display(qtbot):
    v = _view(qtbot, crs="EPSG:4326")
    v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                 lambda rid: RowContent("raster", field=_utm(), style=STYLE))
    r = v.item_rect("src:a")
    assert -105.1 < r.left() < r.right() < -104.9


def test_a_huge_picture_paints_at_most_max_dim(qtbot):
    from dynamix.shell import map_view

    big = _utm(ny=3000, nx=5000)
    v = _view(qtbot)
    v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                 lambda rid: RowContent("raster", field=big, style=STYLE))
    assert max(v._items["src:a"].image.shape[:2]) <= map_view.MAX_DIM


def test_a_failed_reprojection_greys_only_that_row(qtbot, monkeypatch):
    from dynamix.shell import map_view

    def boom(*a, **k):
        raise ValueError("bad grid")

    monkeypatch.setattr(map_view, "reproject_for_display", boom)
    v = _view(qtbot, crs="EPSG:4326")
    reasons = v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                           lambda rid: RowContent("raster", field=_utm(), style=STYLE))
    assert reasons["src:a"].startswith("reprojection failed")


def test_no_project_crs_places_by_the_first_frame(qtbot):
    v = _view(qtbot, crs=None)
    a, b = _utm(crs=None), _utm(crs=None, d=5.0)
    reasons = v.show_stack([(Row("src:a", "raster"), True, 1.0), (Row("src:b", "raster"), True, 1.0)],
                           lambda rid: RowContent("raster", field=a if rid == "src:a" else b,
                                                  style=STYLE))
    assert reasons == {"src:b": "different frame"}
