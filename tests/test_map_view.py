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


def _ext_result():
    ext = {"x": np.array([10, 11, 30]), "y": np.array([5, 5, 20]),
           "mod": np.array([1.0, 2.0, 4.0]), "arg": np.array([0.0, 0.0, np.pi / 2]),
           "line_id": np.array([0, 0, -1])}
    chains = [{"x": np.array([10, 10]), "y": np.array([5, 6]), "mod": np.array([1.0, 1.0]),
               "tags": []}]
    return {"extrema": [ext], "chains": chains, "_shape": (40, 50)}


def test_extrema_are_unconnected_blocks_with_xsmurf_ticks(qtbot):
    v = _view(qtbot)
    field = _utm()
    v.show_stack([(Row("ext:1", "extrema"), True, 1.0)],
                 lambda rid: RowContent("extrema", result=_ext_result(), scale_idx=0,
                                        grid_field=field))
    blocks, ticks = v._vectors["ext:1"]
    xs, ys = blocks.getData()
    assert len(xs) == 3                                          # one block per maximum
    tx, ty = ticks.getData()
    # third maximum: arg = pi/2 (down the image = south), length 3 + 30 * 4/4 = 33 px = 330 m
    assert ty[5] == pytest.approx(ty[4] - 330.0) and tx[5] == pytest.approx(tx[4])


def test_chains_are_trails_and_both_hide_with_their_row(qtbot):
    v = _view(qtbot)
    content = lambda rid: RowContent(rid.split(":")[0] == "ext" and "extrema" or "chains",
                                     result=_ext_result(), scale_idx=0, grid_field=_utm())
    v.show_stack([(Row("chn:1", "chains"), True, 1.0), (Row("ext:1", "extrema"), True, 1.0)],
                 content)
    assert v._vectors["chn:1"][0].isVisible() and v._vectors["ext:1"][0].isVisible()
    v.show_stack([(Row("chn:1", "chains"), True, 1.0), (Row("ext:1", "extrema"), False, 1.0)],
                 content)
    assert v._vectors["chn:1"][0].isVisible() and not v._vectors["ext:1"][0].isVisible()


def test_excluded_chains_draw_as_ghosts_and_the_picked_chain_highlights(qtbot):
    v = _view(qtbot)
    res = _ext_result()
    res["chains_excluded"] = [{"x": np.array([20, 21]), "y": np.array([9, 9]), "tags": []}]
    content = lambda rid: RowContent("chains", result=res, scale_idx=0, grid_field=_utm())
    v.show_stack([(Row("chn:1", "chains"), True, 1.0)], content)
    trail, ghosts, picked = v._vectors["chn:1"]
    assert ghosts.isVisible() and np.isfinite(ghosts.getData()[0]).sum() == 2
    v.set_highlight("chn:1", [0])
    assert picked.isVisible() and np.isfinite(picked.getData()[0]).sum() == 2
    res["_show_ghosts"] = False
    v.show_stack([(Row("chn:1", "chains"), True, 1.0)], content)
    assert not v._vectors["chn:1"][1].isVisible()


def test_the_readout_names_map_coordinates_and_the_selected_value(qtbot):
    v = _view(qtbot)
    v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                 lambda rid: RowContent("raster", field=_utm(), style=STYLE))
    v.set_selected_row("src:a")
    text = v._readout_at(500005.0, 4399995.0)
    assert "500005" in text and "4399995" in text and "=" in text
    assert v._row_at(500005.0, 4399995.0) == "src:a"


def test_a_bare_array_without_coordinates_is_greyed(qtbot):
    v = _view(qtbot, crs=None)
    reasons = v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                           lambda rid: RowContent("raster", field=np.zeros((4, 5)), style=STYLE))
    assert reasons == {"src:a": "no coordinates"}


def test_extrema_on_a_bare_array_are_greyed(qtbot):
    v = _view(qtbot, crs=None)
    reasons = v.show_stack(
        [(Row("ext:1", "extrema"), True, 1.0), (Row("chn:1", "chains"), True, 1.0)],
        lambda rid: RowContent("extrema" if rid.startswith("ext") else "chains",
                               result=_ext_result(), grid_field=np.zeros((40, 50))))
    assert reasons == {"ext:1": "no coordinates", "chn:1": "no coordinates"}
