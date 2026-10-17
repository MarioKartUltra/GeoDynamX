# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The lineal path L(r) of a two-phase map: the run-length kernel equals a brute-force count of
segments to 1e-12, L(0) is the phase fraction, L never increases, and the device declares the
Curve contract under Microstructure."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices
from dynamix.devices.lineal_path import LinealPath, lineal_path
from dynamix.model.contracts import CURVE, produces_of
from dynamix.model.device import validate_params
from dynamix.model.families import FAMILY_ORDER, families_of


def _disks(n=96, seed=0, k=40, radius=6.0) -> np.ndarray:
    """A random packing of disks (phase 1 inside) on an n × n grid."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    phase = np.zeros((n, n), dtype=bool)
    for cx, cy in rng.uniform(0, n, size=(k, 2)):
        phase |= (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    return phase


def _brute_force(phase: np.ndarray, r_max: int, direction: str) -> np.ndarray:
    """Every segment of r + 1 pixels along the rows (or the columns), counted one by one."""
    arr = phase if direction == "horizontal" else phase.T
    ny, nx = arr.shape
    out = np.full(r_max + 1, np.nan)
    for r in range(min(r_max, nx - 1) + 1):
        hits = 0
        for i in range(ny):
            for j in range(nx - r):
                hits += bool(arr[i, j:j + r + 1].all())
        out[r] = hits / (ny * (nx - r))
    return out


@pytest.mark.parametrize("direction", ["horizontal", "vertical"])
def test_the_kernel_equals_the_brute_force_count(direction):
    phase = _disks()
    got = lineal_path(phase, 40, direction)
    want = _brute_force(phase, 40, direction)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_both_directions_average_and_the_curve_starts_at_the_phase_fraction_and_never_rises():
    phase = _disks(seed=3)
    both = lineal_path(phase, 30, "both")
    h, v = lineal_path(phase, 30, "horizontal"), lineal_path(phase, 30, "vertical")
    np.testing.assert_allclose(both, (h + v) / 2, rtol=1e-12)
    assert both[0] == pytest.approx(phase.mean(), rel=1e-12)
    assert np.all(np.diff(both) <= 1e-15)
    assert np.isnan(lineal_path(phase, 200, "horizontal")[96:]).all()   # beyond the width: no segment


def test_the_device_runs_on_a_field_and_declares_the_curve(clean_registry):
    register_builtin_devices()
    device = clean_registry["lineal_path"]
    assert isinstance(device, LinealPath) and families_of(device) == ("Microstructure",)
    params = validate_params(device, {})
    assert "Microstructure" in FAMILY_ORDER and CURVE in produces_of(device, params, frozenset({"Field"}))
    phase = _disks(seed=1)
    field = RasterField(name="disks", values=phase.astype(np.float64), frame=LocalFrame(units="px"),
                        x_axis=np.arange(96.0), y_axis=np.arange(96.0))
    params = validate_params(device, {"r_max": 20, "threshold": 0.5})
    res = device.compute(field, params)
    assert res["L"].shape == (21,) and res["r"][-1] == 20 and res["phase_fraction"] == phase.mean()
    np.testing.assert_allclose(res["L"], lineal_path(phase, 20, "both"))
    assert "phase fraction" in device.reading(res, params)
    assert device.cache_key(params) != device.cache_key({**params, "direction": "vertical"})
