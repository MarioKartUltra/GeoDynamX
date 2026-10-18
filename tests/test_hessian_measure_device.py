# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Hessian measure tool (spec 19): the device over ``dynamix.core.disorder`` — μ(β) of one
field, no planform needed, the derivative scheme recorded and the density as the raster."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import disorder
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.contracts import CURVE, FIELD
from dynamix.model.device import get_device, keyed_params, validate_params


@pytest.fixture
def dev(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return get_device("hessian_measure")


def _field(values):
    values = np.asarray(values, dtype=np.float64)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(values.shape[1])),
                       y_axis=np.arange(float(values.shape[0])))


def test_the_device_is_registered_under_measures_with_no_planform_knobs(dev):
    assert dev.label == "Hessian measure (Nathan–Gunaratne)" and dev.families == ("Measures",)
    assert dev.consumes == ((FIELD,),) and CURVE in dev.produces
    assert {p.name for p in dev.params} == {"deriv", "daf_M", "daf_sigma", "sigma_d", "periodic",
                                            "detrend", "hessian_norm", "beta_floor"}
    p = validate_params(dev, {})
    assert (p["deriv"], p["detrend"], p["hessian_norm"]) == ("daf", "mean", "variance")
    assert {"deriv", "detrend", "hessian_norm"} <= set(keyed_params(dev, p))


def test_a_paraboloid_reads_its_closed_form_and_a_ruled_surface_zero(dev):
    """u = a x² + b y²: |Λ| = 4|ab| / Var(u) everywhere, so μ(β) = (4|ab| / Var u)^{1/4} for every
    β (PRE 71:035101 Eq. 1), read away from the reflected edges; u = f(x) has det H = 0 and μ = 0."""
    n = 96
    y, x = np.mgrid[0:n, 0:n].astype(np.float64) - n / 2
    u = 0.5 * x * x + 0.25 * y * y
    res = dev.compute(_field(u), validate_params(dev, {"detrend": "none"}))
    assert np.allclose(res["hessian_density"][20:-20, 20:-20], 4 * 0.5 * 0.25 / np.var(u), rtol=1e-2)
    assert np.array_equal(res["beta_mu"], disorder.MU_BETAS)
    assert res["raster_out"] is res["hessian_density"] and res["_shape"] == (n, n)
    ruled = dev.compute(_field(np.cos(2 * np.pi * 6 * x / n)), validate_params(dev, {"periodic": True}))
    assert np.all(ruled["hessian_measure"] == 0.0)
    assert ruled["record"]["deriv"]["periodized_by"] == "declared periodic"
    assert ruled["record"]["hessian_norm"] == "variance" and ruled["record"]["valid_pixels"] == n * n


def test_the_reading_and_the_roi_margin(dev):
    n = 64
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    res = dev.compute(_field(np.sin(x / 5) * np.cos(y / 7)), validate_params(dev, {}))
    assert "μ(1)" in dev.reading(res, {}) and "px⁻¹" in dev.reading(res, {})
    assert dev.roi_margin(validate_params(dev, {})) == disorder.reach("daf")
    assert dev.roi_margin(validate_params(dev, {"deriv": "gaussian", "sigma_d": 2.0})) == disorder.reach("gaussian", 2.0)
