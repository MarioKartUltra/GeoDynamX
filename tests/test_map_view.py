# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Map view draws the stack's raster rows by georeference in one map CRS."""
from __future__ import annotations

import dataclasses

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


@pytest.mark.skip(reason="extrema draw as one image on their grid, a pixel per maximum, as the Raster tab draws them (test_extrema_are_one_image_*)")
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


def _bar(v):
    return v.scale_bar.size, v.scale_bar.text.toPlainText()


def test_the_scale_bar_reads_ground_distance_on_a_geographic_map(qtbot):
    """Degrees convert to metres at the view centre's latitude: at 60 deg a 20-degree-wide view
    is about 1113 km across, so a quarter of it rounds to 200 km = 3.593 degrees of longitude."""
    v = _view(qtbot, crs="EPSG:4326")
    v.view.setRange(xRange=(-10.0, 10.0), yRange=(55.0, 65.0), padding=0)
    v._update_scale_bar()
    size, text = _bar(v)
    width = v.view.viewRange()[0][1] - v.view.viewRange()[0][0]
    lat = sum(v.view.viewRange()[1]) / 2
    metres = width * np.pi * 6378137.0 / 180.0 * np.cos(np.radians(lat))
    from dynamix.shell.canvas import nice_round_scalebar
    _label, want = nice_round_scalebar(metres)
    assert text == (f"{want / 1000:g} km" if want >= 1000 else f"{want:g} m")
    assert size == pytest.approx(want / (np.pi * 6378137.0 / 180.0 * np.cos(np.radians(lat))))


def test_the_scale_bar_reads_metres_on_a_projected_map(qtbot):
    v = _view(qtbot)
    v.view.setRange(xRange=(500000.0, 505000.0), yRange=(4395000.0, 4400000.0), padding=0)
    v._update_scale_bar()
    size, text = _bar(v)
    from dynamix.shell.canvas import nice_round_scalebar
    width = v.view.viewRange()[0][1] - v.view.viewRange()[0][0]
    _label, want = nice_round_scalebar(width)
    assert size == pytest.approx(want)
    assert text == (f"{want / 1000:g} km" if want >= 1000 else f"{want:g} m")


def test_without_a_map_crs_the_scale_bar_follows_the_first_frame(qtbot):
    from dynamix.core.frames import GeographicFrame

    f = RasterField(name="g", values=np.zeros((20, 30)), frame=GeographicFrame(),
                    x_axis=np.linspace(-30.0, 12.0, 30), y_axis=np.linspace(-23.0, 19.0, 20))
    v = _view(qtbot, crs=None)
    v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                 lambda rid: RowContent("raster", field=f, style=STYLE))
    v.view.setRange(xRange=(-30.0, 12.0), yRange=(-23.0, 19.0), padding=0)
    v._update_scale_bar()
    assert _bar(v)[1].endswith(" km")


def test_an_unchanged_row_is_not_repainted(qtbot, monkeypatch):
    """A live preview redraws the selected layer's rows on every tick: a row whose content and
    style did not move keeps its image."""
    from dynamix.shell import map_view

    painted = []
    real = map_view._RowPainter._paint
    monkeypatch.setattr(map_view._RowPainter, "_paint",
                        lambda self, item, field, s, **kw: (painted.append(1),
                                                            real(self, item, field, s, **kw)))
    v = _view(qtbot)
    field = _utm()
    content = lambda rid: RowContent("raster", field=field, style=STYLE)
    v.show_stack([(Row("src:a", "raster"), True, 1.0)], content)
    v.update_row("src:a", content("src:a"), opacity=0.5, z=0)
    assert painted == [1] and v._items["src:a"].opacity() == 0.5
    v.update_row("src:a", RowContent("raster", field=field, style={**STYLE, "colormap": "magma"}))
    assert painted == [1, 1]


@pytest.mark.skip(reason="extrema draw as one image on their grid, a pixel per maximum, as the Raster tab draws them (test_extrema_are_one_image_*)")
def test_a_copy_of_the_result_with_the_same_extrema_is_not_redrawn(qtbot):
    """A lazy output landing hands over a copy of the result around the same extrema."""
    v = _view(qtbot)
    field, res = _utm(), _ext_result()
    v.show_stack([(Row("ext:1", "extrema"), True, 1.0), (Row("chn:1", "chains"), True, 1.0)],
                 lambda rid: RowContent("extrema" if rid.startswith("ext") else "chains",
                                        result=res, grid_field=field))
    blocks, trail = v._vectors["ext:1"][0], v._vectors["chn:1"][0]
    calls = []
    blocks.setData = lambda *a, **k: calls.append("ext")
    trail.setData = lambda *a, **k: calls.append("chn")
    copy = {**res, "raster_out": np.zeros((40, 50))}
    v.update_row("ext:1", RowContent("extrema", result=copy, grid_field=field))
    v.update_row("chn:1", RowContent("chains", result=copy, grid_field=field))
    assert calls == []


@pytest.mark.skip(reason="extrema draw as one image on their grid, a pixel per maximum, as the Raster tab draws them (test_extrema_are_one_image_*)")
def test_group_colours_and_modulus_colours_mix_per_point(qtbot):
    v = _view(qtbot)
    res = _ext_result()
    v.extrema_style = {**v.extrema_style, "colour": "group"}
    res["chains"] = [{"x": np.array([10]), "y": np.array([5]), "mod": np.array([1.0]),
                      "tags": [], "group_color": (255, 0, 0)}]
    v.show_stack([(Row("ext:1", "extrema"), True, 1.0)],
                 lambda rid: RowContent("extrema", result=res, grid_field=_utm()))
    brushes = v._vectors["ext:1"][0].data["brush"]
    assert len(brushes) == 3


def test_extrema_are_one_image_a_pixel_per_maximum_with_xsmurf_ticks(qtbot):
    """Panning stays as quick as on the Raster tab: the maxima are one image on their grid."""
    v = _view(qtbot)
    field = _utm()
    v.show_stack([(Row("ext:1", "extrema"), True, 1.0)],
                 lambda rid: RowContent("extrema", result=_ext_result(), scale_idx=0,
                                        grid_field=field))
    blocks, ticks = v._vectors["ext:1"]
    lit = np.argwhere(blocks.image[..., 3] > 0)            # image axes are (col, row)
    assert sorted(map(tuple, lit)) == [(10, 5), (11, 5), (30, 20)]
    r = blocks.mapRectToView(blocks.boundingRect())
    assert (r.left(), r.right()) == pytest.approx((500000.0, 500500.0))
    tx, ty = ticks.getData()
    # third maximum: arg = pi/2 (down the image = south), length 3 + 30 * 4/4 = 33 px = 330 m
    assert ty[5] == pytest.approx(ty[4] - 330.0) and tx[5] == pytest.approx(tx[4])


def test_extrema_images_colour_by_group_over_modulus(qtbot):
    v = _view(qtbot)
    v.extrema_style = {**v.extrema_style, "colour": "group"}
    res = _ext_result()
    res["chains"] = [{"x": np.array([10]), "y": np.array([5]), "mod": np.array([1.0]),
                      "tags": [], "group_color": (255, 0, 0)}]
    from dynamix.geo.extrema_draw import group_rgb

    v.show_stack([(Row("ext:1", "extrema"), True, 1.0)],
                 lambda rid: RowContent("extrema", result=res, grid_field=_utm()))
    img = v._vectors["ext:1"][0].image
    want = group_rgb(res["chains"], 0, np.array([10, 11, 30]), np.array([5, 5, 20]))
    for (c, r), g in zip([(10, 5), (11, 5), (30, 20)], want):
        if g[0] >= 0:
            assert tuple(img[c, r, :3]) == tuple(g)
        assert img[c, r, 3] == 255


def test_extrema_images_are_not_redrawn_for_a_copy_of_the_result(qtbot):
    v = _view(qtbot)
    field, res = _utm(), _ext_result()
    v.show_stack([(Row("ext:1", "extrema"), True, 1.0), (Row("chn:1", "chains"), True, 1.0)],
                 lambda rid: RowContent("extrema" if rid.startswith("ext") else "chains",
                                        result=res, grid_field=field))
    blocks, trail = v._vectors["ext:1"][0], v._vectors["chn:1"][0]
    calls = []
    blocks.setImage = lambda *a, **k: calls.append("ext")
    trail.setData = lambda *a, **k: calls.append("chn")
    copy = {**res, "raster_out": np.zeros((40, 50))}
    v.update_row("ext:1", RowContent("extrema", result=copy, grid_field=field))
    v.update_row("chn:1", RowContent("chains", result=copy, grid_field=field))
    assert calls == []


def test_subpixel_extrema_keep_their_squares(qtbot):
    v = _view(qtbot)
    v.extrema_style = {**v.extrema_style, "subpixel": True}
    v.show_stack([(Row("ext:1", "extrema"), True, 1.0)],
                 lambda rid: RowContent("extrema", result=_ext_result(), grid_field=_utm()))
    xs, _ys = v._vectors["ext:1"][0].getData()
    assert len(xs) == 3


def _three_band(crs="EPSG:32613"):
    f = _utm(crs=crs)
    return dataclasses.replace(f, values=np.stack([f.values, 2 * f.values, 3 * f.values], -1))


def test_a_reprojected_composite_draws_as_rgba(qtbot):
    v = _view(qtbot, crs="EPSG:4326")
    v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                 lambda rid: RowContent("raster", field=_three_band(), style=STYLE,
                                        composite={"r": 0, "g": 1, "b": 2}))
    img = v._items["src:a"].image
    assert img.ndim == 3 and img.shape[-1] == 4


def test_a_reprojected_hillshade_keeps_its_native_spacing(qtbot):
    """A plane rising 0.5 per 10 m pixel shades the same whether or not the Map reprojects it."""
    f = _utm()
    f = dataclasses.replace(f, values=np.tile(0.5 * np.arange(f.values.shape[1]),
                                              (f.values.shape[0], 1)))
    style = {**STYLE, "hillshade": True}
    shades = []
    for crs in ("EPSG:32613", "EPSG:4326"):
        v = _view(qtbot, crs=crs)
        v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                     lambda rid: RowContent("raster", field=f, style=style))
        img = v._items["src:a"].image.astype(np.float64)
        shades.append(img[..., 0][img[..., 3] > 0].mean())
    assert abs(shades[0] - shades[1]) < 6.0


def test_the_warp_cache_holds_one_entry_per_row_and_prunes_with_the_row(qtbot):
    v = _view(qtbot, crs="EPSG:4326")
    a, b = _utm(), _utm()
    for f in (a, b):
        v.update_row("src:a", RowContent("raster", field=f, style=STYLE), 1.0, 0)
    assert list(v._warp_cache) == ["src:a"] and v._warp_cache["src:a"][0] is b.values
    v.show_stack([(Row("chn:1", "chains"), True, 1.0)],
                 lambda rid: RowContent("chains", result=_ext_result(), grid_field=_utm()))
    assert not v._warp_cache and "src:a" not in v._painters and "chn:1" in v._chain_rows
    v.show_stack([], lambda rid: None)
    assert not v._chain_rows


def test_the_readout_falls_back_to_the_dataset_row(qtbot):
    v = _view(qtbot)
    v.show_stack([(Row("src:a", "raster"), True, 1.0)],
                 lambda rid: RowContent("raster", field=_utm(), style=STYLE))
    v.set_selected_row("prod:1", fallback="src:a")
    assert "src:a = " in v._readout_at(500105.0, 4399905.0)


def test_an_unresolvable_map_crs_does_not_raise(qtbot):
    v = MapView()
    qtbot.addWidget(v)
    v.set_map_crs("not a crs")
    assert v._unit_metres == 1.0
