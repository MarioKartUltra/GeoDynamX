# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Periodic extension (DAF), spec 25 §3.1: the field plus a DAF-filled gap after its last row and
column, so that the whole is periodic; the real block is stamped so a scale-space tool after it
keeps no maxima in the gap; below the DAF's band limit the field passes through with the reason."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model.device import defaults_for, get_device
from tests.conftest import make_field


def _device():
    register_builtin_devices()
    return get_device("periodic_extension")


def _smooth(n=192, seed=1):
    """A band-limited field large enough for the DAF stencil and its gap (spec 22 §3.4): a random
    spectrum low-passed and transformed through the FFT policy, then cut out of the larger result so
    that it is not periodic (an FFT-built field alone would be)."""
    from dynamix.core import fft_policy

    big = 2 * n
    rng = np.random.default_rng(seed)
    k = np.hypot(np.fft.fftfreq(big)[:, None], np.fft.rfftfreq(big)[None, :])
    spectrum = (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) * np.exp(-(k / 0.1) ** 2)
    smooth = fft_policy.double().irfft2(spectrum, s=(big, big))
    smooth = smooth[n // 3:n // 3 + n, n // 5:n // 5 + n]
    return (smooth - smooth.mean()) / smooth.std()


def _origin(dev) -> dict:
    """The extension's params with the data at the origin: the fill tests read the block there;
    the default layout (centred) has its own tests below."""
    return dict(defaults_for(dev), layout="origin")


def test_the_output_is_the_field_plus_a_filled_gap_with_its_real_block_stamped(clean_registry):
    dev = _device()
    f = make_field(_smooth())
    out = dev.compute(f, _origin(dev))
    H, W = f.values.shape
    gy, gx = out.values.shape[0] - H, out.values.shape[1] - W
    assert gy > 0 and gx > 0
    np.testing.assert_array_equal(out.values[:H, :W], f.values)
    assert out.provenance["edges.filled"] == {"rows": H, "cols": W}
    assert len(out.x_axis) == W + gx and len(out.y_axis) == H + gy
    assert np.allclose(np.diff(out.x_axis), np.diff(f.x_axis)[0])


def test_the_extended_field_is_periodic_across_its_wrap(clean_registry):
    dev = _device()
    f = make_field(_smooth())
    v = dev.compute(f, defaults_for(dev)).values
    neighbour = np.sqrt(np.mean(np.diff(v, axis=0) ** 2) + np.mean(np.diff(v, axis=1) ** 2))
    wrap = np.sqrt(np.mean((v[0] - v[-1]) ** 2) + np.mean((v[:, 0] - v[:, -1]) ** 2))
    plain = f.values
    seam = np.sqrt(np.mean((plain[0] - plain[-1]) ** 2) + np.mean((plain[:, 0] - plain[:, -1]) ** 2))
    assert wrap <= 2.0 * neighbour and seam > 3.0 * wrap


def test_below_the_band_limit_the_tool_extends_anyway_and_warns(clean_registry):
    """White noise has no band limit worth the name (every pixel is a wavelength): the tool still
    extends, the ⚠ under the band knob says the fill is rough, and the record carries the points
    per wavelength. Only a stencil that cannot fit the field passes it through."""
    dev = _device()
    noise = make_field(np.random.default_rng(0).standard_normal((64, 64)))
    out = dev.compute(noise, _origin(dev))
    assert out.provenance["edges.filled"] == {"rows": 64, "cols": 64}
    assert out.values.shape[0] > 64 and out.provenance["periodic_extension"]["band_limited"] is False
    warnings = dev.knob_warnings(noise, defaults_for(dev))
    assert set(warnings) == {"band"} and "points per wavelength" in warnings["band"]
    assert "rough" in warnings["band"] and "reflection" not in warnings["band"]
    assert dev.knob_warnings(make_field(_smooth()), defaults_for(dev)) == {}
    tiny = make_field(_smooth()[:12, :12])
    unfit = dev.compute(tiny, _origin(dev))
    np.testing.assert_array_equal(unfit.values, tiny.values)
    assert "edges.filled" not in unfit.provenance
    assert "does not fit" in dev.knob_warnings(tiny, defaults_for(dev))["band"]


def test_a_given_band_wavelength_sets_the_gap_and_the_reading_says_what_ran(clean_registry):
    dev = _device()
    field = make_field(_smooth())
    p = {**defaults_for(dev), "band": "given", "lambda_b": 32.0}
    out = dev.compute(field, p)
    rec = out.provenance["periodic_extension"]
    assert rec["band"] == "given" and rec["f_b"] == pytest.approx(1 / 32) and rec["gap"] == 64
    assert out.values.shape == (192 + 64, 192 + 64)
    assert dev.derived_reading("band", "given", field, p) == f"λ_b 32.0 px · gap 64 px · σ {rec['sigma']:.1f} px"
    assert dev.derived_reading("M", 12, field, p) is None


def test_it_sits_in_data_preparation(clean_registry):
    from dynamix.model.families import families_of

    dev = _device()
    assert dev.label == "Periodic extension (DAF)" and families_of(dev) == ("Data preparation",)


def _roi_layer(*steps, window="20,30,96,96"):
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    layer = Layer(layer_id=1, name="roi", source_id="s",
                  chain=Chain(tuple(DeviceRef(d, p) for d, p in steps)).materialized())
    layer.tags["roi.window"] = window
    return layer


def test_on_an_roi_it_extends_the_rois_own_pixels(clean_registry):
    """An ROI layer whose chain opens with Periodic extension: the region runner hands the tool the
    ROI's own pixels, and its field (the ROI and the filled gap, stamped, placed at the ROI) is the
    result."""
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve

    register_builtin_devices()
    field = make_field(_smooth(192, seed=1))
    out = resolve(_roi_layer(("periodic_extension", {"layout": "origin"})), field, Cache()).result
    assert out.provenance["edges.filled"] == {"rows": 96, "cols": 96}
    assert out.provenance["window"] == {"row_off": 20, "col_off": 30}
    np.testing.assert_array_equal(np.asarray(out.values)[:96, :96],
                                  np.asarray(field.values)[20:116, 30:126])
    assert np.asarray(out.values).shape[0] > 96


def test_wtmm_after_it_on_an_roi_runs_periodic_on_the_extended_roi(clean_registry):
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve

    register_builtin_devices()
    field = make_field(_smooth(192, seed=1))
    res = resolve(_roi_layer(("periodic_extension", {"layout": "origin"}), ("wtmm2d", {"n_oct": 2, "n_voice": 2})),
                  field, Cache()).result
    assert res["params"]["pad_mode"] == "periodic" and res["params"]["real_block"] == (96, 96)
    assert tuple(res["_roi"]["roi"]) == (20, 30, 96, 96)          # drawn at the ROI
    assert any(len(e["x"]) for e in res["extrema"])
    for e in res["extrema"]:
        assert (np.asarray(e["x"]) < 96).all() and (np.asarray(e["y"]) < 96).all()


# ------------------------------------------- the stage frame: a result on the extended raster

def test_the_layout_is_centred_by_default_and_the_stamp_carries_the_offset(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device
    from tests.conftest import fbm2d, make_field

    register_builtin_devices()
    dev = get_device("periodic_extension")
    assert defaults_for(dev)["layout"] == "centred"
    out = dev.compute(make_field(fbm2d(128, 0.5, seed=1)), defaults_for(dev))
    stamp = out.provenance["edges.filled"]
    gap_r, gap_c = out.values.shape[0] - 128, out.values.shape[1] - 128
    assert (stamp["rows"], stamp["cols"]) == (128, 128) and gap_r > 0
    assert (stamp["row0"], stamp["col0"]) == (gap_r // 2, gap_c // 2)
    assert np.allclose(out.values[stamp["row0"]:stamp["row0"] + 128, stamp["col0"]:stamp["col0"] + 128],
                       fbm2d(128, 0.5, seed=1))


def test_a_result_on_the_extended_raster_is_drawn_back_on_the_data(clean_registry):
    """The chain extension → wtmm2d: the engine stamps the stage frame, the canvas offset moves
    the maxima by the block's origin on the data's own field and not on the extended one, and a
    result raster of the extended shape crops to the block."""
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.contracts import stage_of
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.canvas import display_offset
    from dynamix.shell.main_window import _display_raster_of
    from tests.conftest import fbm2d, make_field, make_layer

    register_builtin_devices()
    field = make_field(fbm2d(256, 0.5, seed=1))
    extended = get_device("periodic_extension").compute(field, defaults_for(get_device("periodic_extension")))
    stage = stage_of(extended)
    r0, c0 = stage["origin"]
    assert stage["block"] == (256, 256) and stage["shape"] == extended.values.shape and r0 > 0 and c0 > 0
    assert stage_of(field) is None and stage_of({"_stage": stage}) is stage
    res = resolve(make_layer(("periodic_extension", {}), ("wtmm2d", {"n_oct": 2})), field, Cache()).result
    assert res["_stage"] == stage and res["_padding"]["edge"] == "periodic" and res["_padding"]["pad"] == 0
    assert tuple(res["_shape"]) == extended.values.shape
    assert display_offset(res, field) == (-r0, -c0)                 # on the data
    assert display_offset(res, extended) == (0, 0)                  # on the extended raster itself
    for ext in res["extrema"]:
        assert np.all(np.asarray(ext["x"]) - c0 >= 0) and np.all(np.asarray(ext["x"]) - c0 < 256)
        assert np.all(np.asarray(ext["y"]) - r0 >= 0) and np.all(np.asarray(ext["y"]) - r0 < 256)
    big = np.arange(float(np.prod(extended.values.shape))).reshape(extended.values.shape)
    cropped = _display_raster_of({**res, "raster_out": big})
    assert cropped.shape == (256, 256) and cropped[0, 0] == big[r0, c0]
    small = np.zeros((256, 256))
    assert _display_raster_of({**res, "raster_out": small}) is small
    plain = {"raster_out": big}
    assert _display_raster_of(plain) is big
