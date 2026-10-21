# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The dyadic grid's place on a field whose extent is not a multiple of the coarsest cell: the
remainder is split between the two edges, so the border the bank costs is the same width on
every side, and a region's cells are the file's cells wherever the box is drawn."""
from __future__ import annotations

import sys

import numpy as np

from dynamix.engine.resolve import output_raster  # noqa: E402
import pytest

sys.path.insert(0, "tests")
from conftest import fbm2d  # noqa: E402

from dynamix.core import leaders as L  # noqa: E402
from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices
    register_builtin_devices()
    return clean_registry


def _bands(valid, cell: int, n: int):
    """The excluded pixel bands at the start and the end of a valid-cell mask's first axis,
    once the mask is painted at ``cell`` px and cropped to ``n`` px by the run's grid offset."""
    rows = np.where(valid.any(axis=1))[0]
    return rows.min() * cell, n - (rows.max() + 1) * cell


def test_a_field_with_a_remainder_loses_the_same_band_on_both_sides():
    """400 px with 32-px cells: a remainder of 16 px. The first full cell starts 8 px in, the
    last ends 8 px before the far edge, and the valid region at the coarsest level sits at
    border + 8 px from both edges."""
    x = fbm2d(512, 0.5, seed=7)[:400, :400]
    run = L.run_leaders(x, scale_min=4, scale_max=32)
    assert tuple(run["pad"]) == (24, 24)                       # 32 - 16 // 2: the data starts here in the working image
    J, e = run["levels"], run["border_cells"]
    v = run["valid"]["leaders"][J - 1]
    first = np.where(v.any(axis=1))[0].min()
    last = np.where(v.any(axis=1))[0].max() + 1
    top = first * 2 ** J - run["pad"][0]
    bottom = 400 - (last * 2 ** J - run["pad"][0])
    assert top == bottom == e * 32 + 8
    painted = L.paint(run["local"]["leaders"][0], run["j_min"], (400, 400), pad=run["pad"])
    rows = np.where(np.isfinite(painted).any(axis=1))[0]
    assert rows.min() == 400 - rows.max() - 1                  # the h-map's finite band is centred too


def test_a_field_that_is_a_multiple_of_the_cell_is_unchanged():
    x = fbm2d(256, 0.5, seed=8)
    run = L.run_leaders(x, scale_min=4, scale_max=32)
    assert tuple(run["pad"]) == (0, 0)
    before = L.run_leaders(x, scale_min=4, scale_max=32, anchor=(0, 0))
    for a, b in zip(run["quantity"]["leaders"], before["quantity"]["leaders"]):
        np.testing.assert_array_equal(a, b)


def test_an_anchor_places_the_grid_where_the_caller_says():
    """A region of a file keeps the file's cells: the anchor names the data pixel on which a
    coarsest-cell boundary falls."""
    x = fbm2d(512, 0.5, seed=9)[:400, :400]
    run = L.run_leaders(x, scale_min=4, scale_max=32, anchor=(5, 21))
    assert tuple(run["pad"]) == (27, 11)                       # (32 - 5) % 32, (32 - 21) % 32
    J = run["levels"]
    lo, hi = run["block"][J - 1][0]
    assert lo * 32 - run["pad"][0] == 5                        # the first full cell starts on the anchor's row


def test_a_region_drawn_anywhere_has_the_cells_of_the_whole_field(builtins):
    """The device on a processing window of a file: its h-map and its coarsest leaders equal the
    whole field's at that place, box origin and extent being no multiples of the cell."""
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.device import defaults_for, get_device
    from dynamix.roi.runner import run_on_region
    from conftest import make_layer

    n = 384
    f = RasterField(name="fbm", values=fbm2d(n, 0.6, seed=3).astype(np.float32), frame=LocalFrame(),
                    x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))
    f.provenance["window"] = {"row_off": 0, "col_off": 0}
    dev = get_device("wavelet_leaders")
    params = {**defaults_for(dev), "scale_max": 16}
    full = resolve(make_layer(("wavelet_leaders", {"scale_max": 16})), f, Cache()).result
    r0, c0, h, w = rect = (137, 101, 150, 123)
    roi = run_on_region([(dev, params)], f, rect)
    # The same cells of the same pixels through two paddings of the transform: float32 parity.
    np.testing.assert_allclose(roi["h_map"], full["h_map"][r0:r0 + h, c0:c0 + w], rtol=1e-5, atol=1e-5)
    assert np.isfinite(roi["h_map"]).all()
    level = 3
    top = output_raster(dev.compute_output("leader", f.values, full, params, fetch=None, level=level))
    inner = output_raster(dev.compute_output("leader", f.values[r0:r0 + h, c0:c0 + w], roi, params, fetch=None, level=level))
    np.testing.assert_allclose(inner, top[r0:r0 + h, c0:c0 + w], rtol=1e-5, atol=1e-5)
    assert np.isfinite(inner).all()
