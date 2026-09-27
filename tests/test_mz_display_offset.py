# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Where the LastWave M–Z maxima and coarse channel draw.

A result carrying ``_display_offset = (dx, dy)`` draws its maxima (the pixel overlay, the dots,
the H-lines, the gradient arrows) that many pixels off the index they are stored at; an output
carrying ``display_offset`` draws its raster that far off the field's grid, on the canvas and as
the Vector/Geo drape. The LastWave engine registers both half a pixel west and north. A result or
an output without the key (wtmm2d, the printed algorithm, the reconstructions) draws where it
always did."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.resolve import output_key, resolve
from dynamix.model.device import declared_outputs, defaults_for, get_device

N = 64
J = 3
LASTWAVE = (-0.5, -0.5)


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(n=N):
    values = np.random.default_rng(11).standard_normal((n, n)).cumsum(0).cumsum(1)
    return RasterField(name="fbm", values=values, frame=LocalFrame(),
                       x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))


# --------------------------------------------------------------------------- the canvas alone
def _canvas(qtbot, n=16):
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((n, n), dtype=np.float32))
    canvas.set_arrow_mode("all")
    return canvas


def _result(offset=None):
    ext = {"x": np.array([3, 4, 5, 9], dtype=np.int64),
           "y": np.array([7, 7, 7, 2], dtype=np.int64),
           "mod": np.ones(4), "arg": np.zeros(4),
           "line_id": np.array([0, 0, 0, -1], dtype=np.int64)}
    res = {"extrema": [ext], "chains": [], "_shape": (16, 16), "scales": np.array([2.0])}
    if offset is not None:
        res["_display_offset"] = offset
    return res


def _pixel_centre(canvas, x, y):
    """Where the maxima overlay draws the centre of its pixel (x, y), in view coordinates."""
    from PySide6 import QtCore

    c = canvas.extrema_raster_item.mapToView(QtCore.QPointF(x + 0.5, y + 0.5))
    return c.x(), c.y()


def _arrow_starts(canvas):
    """Each arrow's foot (every arrow is six pairs-connected vertices, the foot first)."""
    ax, ay = canvas.arrow_item.getData()
    return sorted(zip(np.asarray(ax)[0::6].tolist(), np.asarray(ay)[0::6].tolist()))


@pytest.mark.parametrize("offset", [LASTWAVE, None])
def test_the_maxima_draw_at_the_results_registered_offset(qtbot, offset):
    canvas = _canvas(qtbot)
    res = _result(offset)
    canvas.set_result(res, 0)
    dx, dy = offset or (0.0, 0.0)
    ext = res["extrema"][0]
    for x, y in zip(ext["x"], ext["y"]):
        assert _pixel_centre(canvas, x, y) == (x + dx, y + dy)
    px, py = canvas.extrema_item.getData()                      # the isolated maximum's dot
    assert (px.tolist(), py.tolist()) == ([9 + dx], [2 + dy])
    hx, hy = canvas.hchain_item.getData()
    assert sorted(zip(hx.tolist(), hy.tolist())) == [(3 + dx, 7 + dy), (4 + dx, 7 + dy),
                                                    (5 + dx, 7 + dy)]
    assert _arrow_starts(canvas) == sorted((float(x) + dx, float(y) + dy)
                                           for x, y in zip(ext["x"], ext["y"]))


def test_a_wtmm2d_result_draws_on_its_pixel_index(qtbot, builtins):
    field = _field(48)
    device = get_device("wtmm2d")
    res = device.compute(field, defaults_for(device))
    assert "_display_offset" not in res
    canvas = _canvas(qtbot, 48)
    canvas.set_result(res, 0)
    ext = res["extrema"][0]
    x, y = int(ext["x"][0]), int(ext["y"][0])
    assert _pixel_centre(canvas, x, y) == (x, y)
    starts = _arrow_starts(canvas)
    assert starts == sorted(zip(np.asarray(ext.get("x_sub", ext["x"]), float).tolist(),
                                np.asarray(ext.get("y_sub", ext["y"]), float).tolist()))


# --------------------------------------------------------------------------- in the window
@pytest.fixture
def win(qtbot, builtins):
    from dynamix.shell.main_window import MainWindow

    w = MainWindow(steps=())
    qtbot.addWidget(w)
    w.load_field(_field(), "/nonexistent/fbm.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


def _mz_layer(win, qtbot, **params):
    params = {**defaults_for(get_device("mz_edges")), "n_levels": J, **params}
    desc = [{"device": "mz_edges", "params": params}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def _key(win, layer, name):
    r = resolve(layer, win._fields[layer.layer_id], win.cache, source_id=layer.source_id)
    out = next(o for o in declared_outputs(get_device("mz_edges")) if o.name == name)
    return output_key(r.analysis_device, out, r.analysis_params, r.analysis_key)


def _show(win, qtbot, layer, name, wait_for=None):
    """Un-hide ``name``'s row and wait until what it draws is cached and the window is idle."""
    win.layer_list.outputHideToggled.emit(layer.layer_id, name, False)
    key = _key(win, layer, wait_for or name)
    qtbot.waitUntil(lambda: key in win.cache and not win.is_computing
                    and win._out_request is None, timeout=60000)
    return key


def _hide(win, qtbot, layer, name):
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(layer.layer_id, name, True)


def _rect(canvas):
    item = canvas.image_item
    r = item.mapRectToView(item.boundingRect())
    return r.x(), r.y(), r.width(), r.height()


def _first_maximum(win):
    ext = win._active_result["extrema"][0]
    return int(ext["x"][0]), int(ext["y"][0])


def test_lastwave_maxima_draw_half_a_pixel_west_and_north(win, qtbot):
    _mz_layer(win, qtbot)
    assert win._active_result["_display_offset"] == LASTWAVE
    x, y = _first_maximum(win)
    assert _pixel_centre(win.canvas, x, y) == (x - 0.5, y - 0.5)


def test_printed_maxima_draw_on_their_pixel_index(win, qtbot):
    _mz_layer(win, qtbot, algorithm="printed", iterations=2)
    assert "_display_offset" not in win._active_result
    x, y = _first_maximum(win)
    assert _pixel_centre(win.canvas, x, y) == (x, y)


def test_the_lastwave_coarse_row_draws_half_a_pixel_west_and_north(win, qtbot, tmp_path):
    layer = _mz_layer(win, qtbot)
    raw = _rect(win.canvas)
    assert raw == (-0.5, -0.5, N, N)
    key = _show(win, qtbot, layer, "coarse")
    assert win.cache.get(key)["display_offset"] == LASTWAVE
    x0, y0, w, h = _rect(win.canvas)
    assert (x0 - raw[0], y0 - raw[1], w, h) == (-0.5, -0.5, N, N)
    assert "display_offset" not in (win.field.provenance or {})     # the field is untouched
    derived = win._derived_fields[layer.layer_id]
    assert "display_offset" not in (derived.provenance or {})
    # The dataset (the export, a surface source) sits where its samples register.
    np.testing.assert_array_equal(derived.values, win.cache.get(key)["raster"])
    np.testing.assert_allclose(derived.x_axis, np.asarray(win.field.x_axis) - 0.5)
    np.testing.assert_allclose(derived.y_axis, np.asarray(win.field.y_axis) - 0.5)
    derived.save_npz(tmp_path / "coarse.npz")
    reloaded = RasterField.load_npz(tmp_path / "coarse.npz")
    np.testing.assert_allclose(reloaded.x_axis, derived.x_axis)
    np.testing.assert_allclose(reloaded.y_axis, derived.y_axis)
    x, y = _first_maximum(win)                                      # the maxima stay over it
    assert _pixel_centre(win.canvas, x, y) == (x - 0.5, y - 0.5)
    _hide(win, qtbot, layer, "coarse")
    assert _rect(win.canvas) == raw


def test_the_lastwave_thumbnail_centres_each_block_on_its_registered_sample(win, qtbot):
    """The thumbnail is the coarse channel sampled every 2^J pixels, so block k draws centred
    where the coarse row draws pixel k * 2^J: half a pixel west and north of that pixel."""
    layer = _mz_layer(win, qtbot)
    coarse = win.cache.get(_show(win, qtbot, layer, "coarse"))["raster"]
    cx0, cy0, _w, _h = _rect(win.canvas)                # coarse pixel i centres at cx0 + i + 1/2
    thumb = win.cache.get(_show(win, qtbot, layer, "thumbnail"))["raster"]
    S = 2 ** J
    m = math.ceil(N / S)
    np.testing.assert_array_equal(thumb, coarse[::S, ::S])   # sample k is coarse pixel k * S
    shown = win.canvas._field
    assert shown.provenance["display_anchor"] == "sample"
    assert "display_anchor" not in (win.field.provenance or {})     # the field is untouched
    x0, y0, w, h = _rect(win.canvas)
    assert (w, h) == (m * S, m * S)
    assert (x0 + S / 2, y0 + S / 2) == (cx0 + 0.5, cy0 + 0.5)       # block 0 over pixel 0
    assert (x0, y0) == (-S / 2 - 0.5, -S / 2 - 0.5)
    for axis, full in ((shown.x_axis, win.field.x_axis), (shown.y_axis, win.field.y_axis)):
        np.testing.assert_allclose(axis, np.asarray(full)[::S][:m] - 0.5)


def test_the_recon_row_draws_on_the_input_grid(win, qtbot):
    layer = _mz_layer(win, qtbot)
    _show(win, qtbot, layer, "recon", wait_for="recon_preview")
    assert _rect(win.canvas) == (-0.5, -0.5, N, N)


def test_the_printed_coarse_row_draws_on_the_input_grid(win, qtbot):
    layer = _mz_layer(win, qtbot, algorithm="printed", iterations=2)
    _show(win, qtbot, layer, "coarse")
    assert _rect(win.canvas) == (-0.5, -0.5, N, N)


def test_the_vector_drape_carries_the_outputs_offset(win, qtbot):
    layer = _mz_layer(win, qtbot)
    seen = []

    class _Arrangement:
        def set_layers(self, entries):
            seen.append({e["layer"].layer_id: (e.get("drape"), e.get("drape_offset"))
                         for e in entries if e["status"] == "ok"})

    key = _show(win, qtbot, layer, "coarse")
    win._arrangement = _Arrangement()
    win._sync_arrangement(frame_mode=True)
    drape, offset = seen[-1][layer.layer_id]
    assert drape is win.cache.get(key)["raster"] and offset == LASTWAVE
    _show(win, qtbot, layer, "recon", wait_for="recon_preview")
    win._sync_arrangement(frame_mode=True)
    drape, offset = seen[-1][layer.layer_id]
    assert drape is not None and offset is None
