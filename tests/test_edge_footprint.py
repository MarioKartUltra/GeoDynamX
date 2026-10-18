# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The footprint the Raster view draws while the edge knobs move (spec 25 §3.4): the padding's
outline and content outside the raster, and the band the Border cut (or the taper) removes inside
it, all from the numbers the run itself uses."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model.device import defaults_for, get_device
from tests.conftest import make_field
from tests.test_periodic_extension import _smooth


@pytest.fixture
def wtmm(clean_registry):
    register_builtin_devices()
    return get_device("wtmm2d")


@pytest.fixture
def field():
    return make_field(_smooth(128, seed=3))


def _params(dev, **over):
    return dict(defaults_for(dev), n_oct=2, n_voice=4, wavelet="gaussian", **over)


def test_the_footprint_is_the_runs_padding_and_border(wtmm, field):
    params = _params(wtmm, border_cut="measured_reach")
    fp = wtmm.footprint(field, params, 0)
    res = wtmm.compute(field, params)
    p = res["_padding"]["pad"]
    assert fp["edge"] == "reflect" and fp["pad"] == fp["shown"] == p > 0
    assert fp["band"] == "border_cut" and fp["border"] == res["_border_cut"]["borders"][0] > 0
    v = np.asarray(field.values)
    assert fp["padded"].shape == (128 + 2 * p, 128 + 2 * p) and fp["stride"] == 1
    np.testing.assert_array_equal(fp["padded"][p:-p, p:-p], v)
    np.testing.assert_array_equal(fp["padded"][p - 1, p:-p], v[1])           # the mirror


@pytest.mark.parametrize("edge", ["edge_pad", "zero_pad", "periodic"])
def test_each_padded_mode_shows_its_own_content(wtmm, field, edge):
    fp = wtmm.footprint(field, _params(wtmm, edge=edge), 0)
    v, s = np.asarray(field.values), fp["shown"]
    outside = fp["padded"][0, s:-s]
    if edge == "edge_pad":
        np.testing.assert_array_equal(outside, v[0])
    elif edge == "zero_pad":
        assert not outside.any()
    else:                                        # no padding: the opposite edge wrapped round
        assert fp["pad"] == 0 and s > 0
        np.testing.assert_array_equal(fp["padded"][s - 1, s:-s], v[-1])


def test_the_border_follows_the_shown_scale_under_per_scale(wtmm, field):
    params = _params(wtmm, border_cut="measured_reach", border_scope="per_scale")
    first, last = wtmm.footprint(field, params, 0), wtmm.footprint(field, params, 7)
    assert 0 < first["border"] < last["border"]
    largest = _params(wtmm, border_cut="measured_reach")
    assert wtmm.footprint(field, largest, 0)["border"] == last["border"]
    assert wtmm.footprint(field, _params(wtmm, border_cut="none"), 0)["border"] == 0


def test_the_taper_shows_its_band_and_a_periodic_extension_output_shows_nothing_cut(wtmm, field):
    taper = wtmm.footprint(field, _params(wtmm, edge="gaussian_taper", border_cut="none"), 0)
    assert taper["band"] == "taper" and taper["border"] > 0 and taper["pad"] == 32
    from dynamix.devices.periodic_extension import PeriodicExtension

    ext = PeriodicExtension().compute(field, defaults_for(PeriodicExtension()))
    fp = wtmm.footprint(ext, _params(wtmm), 0)
    assert fp["edge"] == "periodic" and fp["pad"] == 0 and fp["border"] == 0


def test_a_large_raster_is_drawn_decimated_then_padded(wtmm, field):
    """The content of a decimated display: every 4th pixel padded by the pad in samples, its first
    sample ``offset`` px before the raster's first."""
    fp = wtmm.footprint(field, _params(wtmm), 0, stride=4)
    q = -(-fp["pad"] // 4)
    assert fp["stride"] == 4 and fp["padded"].shape == (32 + 2 * q,) * 2 and fp["offset"] == -4 * q
    assert wtmm.footprint(field, _params(wtmm), 0)["offset"] == -fp["pad"]


def test_the_superseded_modes_draw_no_footprint(wtmm, field):
    assert wtmm.footprint(field, _params(wtmm, edge="cut"), 0) is None


def test_the_field_reaching_a_step_is_read_from_the_cache_alone(wtmm, field):
    """The footprint of a WTMM after an in-chain Periodic extension reads the extension the last
    run cached; before any run there is nothing to read and nothing is computed."""
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import cached_input, resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    prior = [DeviceRef("periodic_extension", {})]
    layer = Layer(layer_id=1, name="L", source_id="s",
                  chain=Chain((*prior, DeviceRef("wtmm2d", _params(wtmm)))).materialized())
    cache = Cache()
    assert cached_input(layer, field, cache, []) is field
    assert cached_input(layer, field, cache, prior) is None and len(cache) == 0
    resolve(layer, field, cache)
    extended = cached_input(layer, field, cache, prior)
    assert "edges.filled" in extended.provenance
    assert wtmm.footprint(extended, _params(wtmm), 0)["edge"] == "periodic"


def test_on_an_roi_the_footprint_is_its_halo_mirrored_past_the_dataset(wtmm, field):
    """An ROI's run reads each scale's halo off the dataset and mirrors only past the dataset's
    edges (spec 22): the footprint sits on the ROI, as wide as the shown scale's halo, its content
    the dataset there and the mirror beyond, with no Border cut (that runs on whole rasters)."""
    params = _params(wtmm)
    rect = (3, 20, 40, 40)
    fp = wtmm.roi_footprint(field, params, rect, 0)
    m = wtmm._halo_plan(params, parent_shape=(128, 128))[0].margin
    assert fp["pad"] == fp["shown"] == m > 3 and fp["offset"] == -m
    assert fp["origin"] == (3, 20) and fp["shape"] == (40, 40) and fp["border"] == 0
    v = np.asarray(field.values)
    assert fp["padded"].shape == (40 + 2 * m, 40 + 2 * m)
    np.testing.assert_array_equal(fp["padded"][m:m + 40, m:m + 40], v[3:43, 20:60])
    np.testing.assert_array_equal(fp["padded"][m - 4, m:m + 40], v[1, 20:60])   # row -1 mirrors 1
    coarse = wtmm.roi_footprint(field, params, rect, 7)
    assert coarse["pad"] > m                                                    # its own halo
