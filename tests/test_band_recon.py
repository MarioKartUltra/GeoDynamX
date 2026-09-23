# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the band_recon Transform -- reconstruction from an arbitrary h band."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import fbm2d
from dynamix.core import microcanonical as mc
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.band_recon import BandRecon
from dynamix.devices.holder_map import HolderMap, holder_arrays
from dynamix.model.device import defaults_for


def _field(n=64, H=0.8, seed=2):
    vals = fbm2d(n, H, seed=seed)
    return RasterField(name=f"f{n}", values=vals, frame=LocalFrame(),
                       x_axis=np.arange(n, dtype=np.float64),
                       y_axis=np.arange(n, dtype=np.float64))


def test_registered_as_a_builtin(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES

    register_builtin_devices()
    assert "band_recon" in DEVICES


def test_engine_params_are_holder_maps_verbatim():
    """One mental model: the h(x) engine knobs are HolderMap's tuple, by identity, with only
    the band appended."""
    assert BandRecon.params[: len(HolderMap.params)] == HolderMap.params
    assert [p.name for p in BandRecon.params[len(HolderMap.params):]] == [
        "h_lo", "h_hi", "min_island", "max_island", "connectivity"]


def test_compute_is_exactly_the_core_band_pipeline():
    field = _field()
    params = dict(defaults_for(BandRecon()), h_lo=-1.0, h_hi=0.5)
    res = BandRecon().compute(field, params)

    vals = np.asarray(field.values, dtype=np.float64)
    h, _r2, scales = holder_arrays(vals, params)
    mask = mc.band_mask(h, -1.0, 0.5)
    recon, psnr, rel = mc.reconstruct_from_msc(vals, mask)
    np.testing.assert_array_equal(res["h_map"], h)
    np.testing.assert_array_equal(res["band_mask"], mask)
    np.testing.assert_array_equal(res["raster_out"], recon)
    assert res["psnr_db"] == psnr and res["rel_err"] == rel
    assert res["band_density"] == pytest.approx(mask.mean())
    assert res["raster_out"].shape == field.values.shape
    assert res["chains"] == [] and res["extrema"] == []


def test_band_knobs_change_the_cache_key_and_the_result():
    dev = BandRecon()
    base = dict(defaults_for(dev), h_lo=-1.0, h_hi=0.0)
    assert dev.cache_key("s", base) != dev.cache_key("s", dict(base, h_hi=0.5))
    assert dev.cache_key("s", base) != dev.cache_key("s", dict(base, h_lo=-0.5))
    field = _field()
    a = dev.compute(field, base)
    b = dev.compute(field, dict(base, h_hi=1.0))
    assert not np.array_equal(a["band_mask"], b["band_mask"])


def test_refuses_an_upstream_result_dict():
    with pytest.raises(ValueError, match="FIRST in its chain"):
        BandRecon().compute({"chains": []}, defaults_for(BandRecon()))


# ------------------------------------------------------- the source split (2026-09-19, §6.3)

def test_split_variants_are_registered(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES

    register_builtin_devices()
    assert "band_recon_measure" in DEVICES
    assert "band_recon_multiaffine" in DEVICES
    assert "band_recon" in DEVICES        # superseded, never deregistered (saved projects)


def test_variant_engine_params_are_their_holder_siblings_verbatim():
    """The split keeps the shared-by-identity law: each variant's engine knobs are its h-map
    device's tuple with only BandRecon's own band knobs appended -- the h-map a band is
    picked from is the h-map the reconstruction uses."""
    from dynamix.devices.band_recon import BandReconMeasure, BandReconMultiaffine
    from dynamix.devices.holder_methods import HolderMeasure, HolderMultiaffine

    band_knobs = BandRecon.params[len(HolderMap.params):]
    assert BandReconMeasure.params == HolderMeasure.params + band_knobs
    assert BandReconMultiaffine.params == HolderMultiaffine.params + band_knobs


def test_variant_compute_is_exactly_the_split_band_pipeline():
    from dynamix.devices.band_recon import BandReconMeasure, BandReconMultiaffine
    from dynamix.devices.holder_methods import method_arrays

    field = _field()
    vals = np.asarray(field.values, dtype=np.float64)
    for dev, method in ((BandReconMeasure(), "measure"),
                        (BandReconMultiaffine(), "multiaffine")):
        params = dict(defaults_for(dev), h_lo=-1.0, h_hi=0.5)
        res = dev.compute(field, params)
        h, _r2, scales = method_arrays(vals, method, params)
        mask = mc.band_mask(h, -1.0, 0.5)
        recon, psnr, rel = mc.reconstruct_from_msc(vals, mask)
        np.testing.assert_array_equal(res["h_map"], h)
        np.testing.assert_array_equal(res["band_mask"], mask)
        np.testing.assert_array_equal(res["raster_out"], recon)
        assert res["psnr_db"] == psnr and res["rel_err"] == rel
        assert res["chains"] == [] and res["extrema"] == []


def test_variant_punctual_band_is_the_msc_reconstruction_path():
    """A band over the PUNCTUAL h-map -- the natural most-singular-set reconstruction (the
    left-limb extractor feeding the inversion)."""
    from dynamix.devices.band_recon import BandReconMultiaffine
    from dynamix.devices.holder_methods import method_arrays

    field = _field()
    vals = np.asarray(field.values, dtype=np.float64)
    params = dict(defaults_for(BandReconMultiaffine()), estimator="punctual",
                  h_lo=-3.0, h_hi=0.0)
    res = BandReconMultiaffine().compute(field, params)
    h, _r2, _sc = method_arrays(vals, "multiaffine", params)
    np.testing.assert_array_equal(res["h_map"], h)
    np.testing.assert_array_equal(res["band_mask"], mc.band_mask(h, -3.0, 0.0))
