# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The chaining band per step or per octave, the chainable range of h it implies, the analyzing
wavelet's vanishing moments, and the reading beside the similitude knob that shows both limits."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.core.smoothers import smoother
from dynamix.core.wtmm_backend import chainable_h, compute_scales2d, get_backend, run_wtmm2d
from dynamix.devices.wtmm import WTMM2D, chaining_reading
from dynamix.model.device import defaults_for
from tests.conftest import fbm2d


def _cusp(h: float, n: int = 128) -> RasterField:
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    values = -(np.hypot(x - (n / 2 - 0.5), y - (n / 2 - 0.5)) ** h)
    return RasterField(name="cusp", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))


@pytest.mark.parametrize("name, params, n_psi", [
    ("gaussian", None, 1), ("mexican", None, 3), ("frac_bspline", {"alpha": 3.0}, 1),
    ("q_gaussian", {"q": 1.5}, 1), ("q_mexican", {"q": 1.5}, 3), ("q_meyer", {"q": 1.0}, 1)])
def test_each_smoother_knows_its_vanishing_moments(name, params, n_psi):
    assert smoother(name, params).vanishing_moments == n_psi


def test_the_chainable_range_per_step_and_per_octave():
    w = -np.log2(0.8)
    assert chainable_h(0.8, 4) == pytest.approx((-4 * w, 4 * w))
    assert chainable_h(0.8, 8) == pytest.approx((-8 * w, 8 * w))
    assert chainable_h(0.8, 4, band_per="octave") == pytest.approx((-w, w))
    assert chainable_h(0.8, 8, band_per="octave") == chainable_h(0.8, 4, band_per="octave")
    assert chainable_h(0.8, 4, eta=1.0) == pytest.approx((-4 * w - 1, 4 * w - 1))
    assert chainable_h(0.0, 4) is None


def test_the_octave_band_is_the_step_band_with_the_steps_share():
    backend = get_backend("python")
    scales = compute_scales2d(2, 4)
    cwt = backend.cwt2d(fbm2d(64, 0.6, seed=3), scales)
    extrema = backend.extrema2d(cwt, scales, thresh=1e-3)
    share = 0.6 ** float(np.log2(scales[1] / scales[0]))
    a = backend.chains2d(extrema, scales, similitude=0.6, band_per="octave")
    b = backend.chains2d(extrema, scales, similitude=share)
    assert len(a) == len(b) > 0
    assert all(np.array_equal(p["x"], q["x"]) and np.array_equal(p["mod"], q["mod"])
               for p, q in zip(a, b))
    with pytest.raises(ValueError, match="band_per"):
        backend.chains2d(extrema, scales, band_per="decade")


def _chains_through_the_cusp(h, voices, similitude, band_per):
    res = run_wtmm2d(_cusp(h), {"n_oct": 3, "n_voice": voices, "similitude": similitude,
                                "band_per": band_per, "fracint_alpha": 0.0})
    return max((len(c["mod"]) for c in res["chains"]), default=0)


def test_per_octave_the_voices_no_longer_decide_which_h_chains():
    """A cusp of h = 0.5: per octave, similitude 0.6 admits h < 0.74 and 0.8 admits h < 0.32, at
    four voices and at eight alike; per step, 0.8 admits it at four voices (h < 1.29)."""
    for voices in (4, 8):
        assert _chains_through_the_cusp(0.5, voices, 0.6, "octave") == 3 * voices
        assert _chains_through_the_cusp(0.5, voices, 0.8, "octave") < 3 * voices
    assert _chains_through_the_cusp(0.5, 4, 0.8, "step") == 12


def test_the_reading_beside_similitude_shows_both_limits():
    device = WTMM2D()
    params = dict(defaults_for(device), wavelet="gaussian", n_voice=4, fracint_alpha=0.0)
    text = device.derived_reading("similitude", 0.8, None, params)
    assert text == chaining_reading(0.8, params)
    assert text == "chains h<1.29\nψ resolves h<1"
    octave = device.derived_reading("similitude", 0.8, None, dict(params, band_per="octave"))
    assert octave.startswith("chains h<0.32\n")
    lifted = device.derived_reading("similitude", 0.8, None, dict(params, fracint_alpha=1.0))
    assert lifted.startswith("chains h<0.29\n")
    assert device.derived_reading("similitude", 0.8, None,
                                  dict(params, wavelet="mexican")).endswith("ψ resolves h<3")
    assert device.derived_reading("similitude", 0.0, None, params).startswith("every h chains")


def test_the_step_band_keeps_the_chains_stage_key():
    """xsmurf's band is the default and leaves the chains stage key as it was; the octave band
    enters it."""
    from dynamix.core import wtmm_backend

    field = RasterField(name="f", values=fbm2d(64, 0.6, seed=3), frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    base = {"n_oct": 2, "n_voice": 2}
    a = run_wtmm2d(field, base)
    b = run_wtmm2d(field, dict(base, band_per="step"))
    assert len(a["chains"]) == len(b["chains"])
    assert wtmm_backend._WTMM2D_PARAM_DEFAULTS["band_per"] == "step"
