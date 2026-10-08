# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A result's own raster draws on the canvas's product layer, above the dataset.

The product (an h-map, a reconstruction, a filtered field, the live band preview) goes on
``Canvas.product_item`` with the registration :meth:`Canvas.set_field` gives a field (window
offset, display stride and anchor); the dataset stays on ``image_item`` beneath, with its own
LUT and levels, and follows its own H. Non-finite product pixels draw transparent, so where the
product has no support the dataset shows instead of the viewport background.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


def _field(ny=12, nx=16, seed=0, name="dataset"):
    v = np.random.default_rng(seed).standard_normal((ny, nx)).cumsum(0)
    return RasterField(name=name, values=v, frame=LocalFrame(),
                       x_axis=np.arange(nx, dtype=np.float64),
                       y_axis=np.arange(ny, dtype=np.float64))


def _holed(shape, seed=1):
    """A product with no support over a block and a border strip."""
    h = np.random.default_rng(seed).uniform(0.1, 0.9, shape)
    h[2:5, 3:9] = np.nan
    h[:, -2:] = np.nan
    return h


def _rect(item):
    r = item.mapRectToView(item.boundingRect())
    return r.x(), r.y(), r.width(), r.height()


def _alpha(item):
    """The rendered alpha of an image item, (rows, cols) like the values it shows."""
    import pyqtgraph as pg

    item.render()
    return pg.functions.ndarray_from_qimage(item.qimage)[..., 3]


def _shown(item):
    """The values an image item holds, (rows, cols) -- items store the transpose."""
    return np.asarray(item.image).T


# ---------------------------------------------------------------------------- the canvas

def test_a_product_draws_above_the_dataset_with_its_holes_clear(qtbot):
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    data = _field()
    canvas.set_field(data)
    dataset_rect = _rect(canvas.image_item)
    h = _holed(data.values.shape)
    canvas.set_product(RasterField(name="h", values=h, frame=data.frame, x_axis=data.x_axis,
                                   y_axis=data.y_axis))
    assert canvas.image_item.isVisible()
    np.testing.assert_array_equal(_shown(canvas.image_item), data.values)
    assert canvas.product_item.isVisible()
    np.testing.assert_array_equal(_shown(canvas.product_item), h)
    alpha = _alpha(canvas.product_item)
    assert (alpha[~np.isfinite(h)] == 0).all()
    assert (alpha[np.isfinite(h)] == 255).all()
    assert _rect(canvas.product_item) == dataset_rect == (-0.5, -0.5, 16.0, 12.0)
    added = canvas.view.addedItems
    assert canvas.product_item.zValue() >= canvas.image_item.zValue()
    assert added.index(canvas.product_item) > added.index(canvas.image_item)
    assert canvas._field is data                     # gestures and overlays stay the dataset's


def test_without_a_product_the_product_layer_is_empty_and_the_dataset_is_as_before(qtbot):
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    data = _field()
    canvas.set_field(data)
    assert not canvas.product_item.isVisible() and canvas.product_item.image is None
    np.testing.assert_array_equal(_shown(canvas.image_item), data.values)
    assert _rect(canvas.image_item) == (-0.5, -0.5, 16.0, 12.0)
    canvas.set_product(_holed(data.values.shape))
    canvas.clear_product()
    assert not canvas.product_item.isVisible() and canvas.product_item.image is None
    np.testing.assert_array_equal(_shown(canvas.image_item), data.values)
    canvas.set_product(_holed(data.values.shape))
    other = _field(seed=3, name="other")
    canvas.set_field(other)                          # a new dataset: the old product is gone
    assert not canvas.product_item.isVisible() and canvas.product_item.image is None
    np.testing.assert_array_equal(_shown(canvas.image_item), other.values)


def test_a_windowed_strided_product_is_pinned_like_a_field(qtbot):
    """The product layer reuses the field registration: an ROI's window offset, a strided
    sample's anchor -- the rect a field with the same stamps gets on ``image_item``."""
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_field(ny=40, nx=50))
    prod = _field(ny=6, nx=7, seed=5, name="preview")
    prod.provenance.update({"window": {"row_off": 4, "col_off": 5}, "display_stride": 3,
                            "display_anchor": "sample", "full_dims": (16, 20)})
    canvas.set_product(prod)
    assert _rect(canvas.product_item) == (5 - 1.5, 4 - 1.5, 21.0, 18.0)
    reference = Canvas()
    qtbot.addWidget(reference)
    reference.set_field(prod)
    assert _rect(canvas.product_item) == _rect(reference.image_item)
    assert _rect(canvas.image_item) == (-0.5, -0.5, 50.0, 40.0)


def test_a_density_slice_belongs_to_the_product_while_one_is_up(qtbot):
    """The Slice editor reads the displayed product, so its classes paint the product; the
    dataset beneath keeps its continuous ramp, and gets the slice back when the product goes."""
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    data = _field()
    canvas.set_field(data)
    canvas.set_levels("3")
    sliced_dataset = _shown(canvas.image_item).copy()
    assert not np.array_equal(sliced_dataset, data.values)
    h = _holed(data.values.shape)
    canvas.set_product(h)
    np.testing.assert_array_equal(_shown(canvas.image_item), data.values)
    assert not np.array_equal(_shown(canvas.product_item), h, equal_nan=True)
    canvas.clear_product()
    np.testing.assert_array_equal(_shown(canvas.image_item), sliced_dataset)


# ---------------------------------------------------------------------------- the window

@pytest.fixture
def registered_builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def plain_window(qtbot, registered_builtins):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(_field(ny=40, nx=50, seed=4, name="plain"), "/nonexistent/plain.npz")
    return win


@pytest.fixture
def real_pic_window(qtbot, registered_builtins, tmp_path):
    """A picture of a real GeoTIFF, so the worker's ROI runner can read native pixels."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from dynamix.roi.picture import read_picture
    from dynamix.shell.main_window import MainWindow

    v = np.random.default_rng(2).standard_normal((64, 80)).cumsum(0).cumsum(1)
    path = tmp_path / "r.tif"
    with rasterio.open(path, "w", driver="GTiff", height=64, width=80, count=1,
                       dtype="float32", crs="EPSG:32615",
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as d:
        d.write(v.astype(np.float32), 1)
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(read_picture(path, max_dim=16), str(path))
    return win


def _drop(win, qtbot, device):
    from dynamix.model.device import defaults_for, get_device

    desc = [{"device": device, "params": defaults_for(get_device(device))}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
        qtbot.wait(30)                                # the fork spawns one tick later
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def test_an_hmap_with_holes_shows_the_dataset_through_them(plain_window):
    win = plain_window
    dataset_rect = _rect(win.canvas.image_item)
    h = _holed(win.field.values.shape)
    win._sync_holder_raster({"h_map": h})
    canvas = win.canvas
    assert canvas.image_item.isVisible()
    np.testing.assert_array_equal(_shown(canvas.image_item), win.field.values)
    assert canvas.product_item.isVisible()
    np.testing.assert_array_equal(_shown(canvas.product_item), h)
    assert (_alpha(canvas.product_item)[~np.isfinite(h)] == 0).all()
    assert _rect(canvas.product_item) == dataset_rect == (-0.5, -0.5, 50.0, 40.0)
    win._sync_holder_raster({})                      # the product goes, the dataset stays
    assert not canvas.product_item.isVisible() and canvas.product_item.image is None
    assert canvas.image_item.isVisible()
    np.testing.assert_array_equal(_shown(canvas.image_item), win.field.values)


def test_the_datasets_h_hides_the_dataset_under_a_product_not_the_product(plain_window):
    win = plain_window
    win._sync_holder_raster({"h_map": _holed(win.field.values.shape)})
    win._on_source_hide_toggled(win.layer.source_id, True)
    assert not win.canvas.image_item.isVisible()
    assert win.canvas.product_item.isVisible()
    win._on_source_hide_toggled(win.layer.source_id, False)
    assert win.canvas.image_item.isVisible() and win.canvas.product_item.isVisible()


def test_switching_layers_takes_the_product_off(plain_window, qtbot):
    win = plain_window
    master = win.layer
    child = _drop(win, qtbot, "holder_measure")
    assert child is not master and win.canvas.product_item.isVisible()
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.layer_list.select_layer(master.layer_id)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert not win.canvas.product_item.isVisible() and win.canvas.product_item.image is None
    np.testing.assert_array_equal(_shown(win.canvas.image_item), win.field.values)


def test_an_roi_hmap_sits_on_its_roi_over_the_dataset(real_pic_window, qtbot):
    win = real_pic_window
    dataset = win.field
    dataset_rect = _rect(win.canvas.image_item)
    win._on_roi_save({"roi_row": 20, "roi_col": 24, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    _drop(win, qtbot, "holder_measure")
    canvas = win.canvas
    assert _rect(canvas.product_item) == (23.5, 19.5, 20.0, 16.0)
    assert canvas._product.values.shape == (16, 20)
    assert canvas.product_item.isVisible()
    assert canvas._field is dataset and canvas.image_item.isVisible()
    assert _rect(canvas.image_item) == dataset_rect
    np.testing.assert_array_equal(_shown(canvas.image_item), dataset.values)


@pytest.mark.parametrize("where", ["whole", "roi"])
@pytest.mark.parametrize("mode", ["mask", "reconstruction"])
def test_a_strided_band_preview_sits_over_the_dataset(plain_window, qtbot, monkeypatch,
                                                      where, mode):
    import dynamix.shell.main_window as mw
    from dynamix.roi.picture import display_stride

    win = plain_window
    dataset = win.field
    if where == "roi":
        win._on_roi_save({"roi_row": 4, "roi_col": 5, "roi_h": 16, "roi_w": 20,
                          "boundary": "auto"})
        _drop(win, qtbot, "holder_measure")
        row_off, col_off, h, w = 4, 5, 16, 20
    else:
        win.layer_list.select_layer(win.project.layers[0].layer_id)
        _drop(win, qtbot, "holder_measure")
        row_off, col_off, h, w = 0, 0, 40, 50
    monkeypatch.setattr(mw, "_PREVIEW_MAX_DIM", 8)
    s = -(-max(h, w) // 8)
    hm = np.asarray(win._active_result["h_map"])
    win._on_band_preview_requested(float(np.nanmin(hm)), float(np.nanmax(hm)) + 1.0, mode)
    canvas = win.canvas
    shown = canvas._product
    assert shown.values.shape == (-(-h // s), -(-w // s))
    assert display_stride(shown) == s
    r = canvas.product_item.mapRectToView(canvas.product_item.boundingRect())
    assert (r.x(), r.y()) == (col_off - s / 2, row_off - s / 2)
    assert abs(r.width() - w) < s and abs(r.height() - h) < s
    assert canvas.product_item.isVisible()
    assert canvas._field is dataset and canvas.image_item.isVisible()
    np.testing.assert_array_equal(_shown(canvas.image_item), dataset.values)
    assert _rect(canvas.image_item) == (-0.5, -0.5, 50.0, 40.0)
