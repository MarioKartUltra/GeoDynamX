# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""What the FFT torus holds beyond the real data (spec 25 §3.2): every convolution is circular, and
the edge mode pads the field before the transform -- mirrored, the edge values repeated, zeros -- or
not at all, ``periodic``, where the raster itself is the torus."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.wtmm_backend import PythonWTMMBackend

SCALES = [1.0, 2.0, 3.0]
ENGINES = ["fftw"] + (["mlx"] if __import__("importlib").util.find_spec("mlx") else [])
NP_MODE = {"reflect": "reflect", "edge_pad": "edge", "zero_pad": "constant"}


def _field(n=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)).cumsum(0).cumsum(1) / n


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mode", sorted(NP_MODE))
def test_a_padded_mode_is_the_periodic_transform_of_the_padded_field(engine, mode):
    f, pad = _field(), 20
    be = PythonWTMMBackend()
    got = be.cwt2d(f, SCALES, pad=pad, pad_mode=mode, _engine=engine)
    ref = be.cwt2d(np.pad(f, pad, mode=NP_MODE[mode]), SCALES, pad=0, pad_mode="periodic",
                   _engine=engine)
    crop = (slice(None), slice(pad, pad + f.shape[0]), slice(pad, pad + f.shape[1]))
    scale = float(np.abs(ref["mod"]).max())
    np.testing.assert_allclose(got["mod"], ref["mod"][crop], rtol=1e-5, atol=1e-5 * scale)


@pytest.mark.parametrize("engine", ENGINES)
def test_periodic_is_the_centre_tile_of_the_tiled_field(engine):
    n = 48
    y, x = np.mgrid[0:n, 0:n] / n
    f = np.sin(2 * np.pi * 3 * x) + 0.5 * np.cos(2 * np.pi * 2 * y + 1.0) + 0.3 * np.sin(2 * np.pi * (x + y))
    be = PythonWTMMBackend()
    got = be.cwt2d(f, SCALES, pad=0, pad_mode="periodic", _engine=engine)
    tiled = be.cwt2d(np.tile(f, (3, 3)), SCALES, pad=0, pad_mode="periodic", _engine=engine)
    centre = (slice(None), slice(n, 2 * n), slice(n, 2 * n))
    scale = float(np.abs(tiled["mod"]).max())
    np.testing.assert_allclose(got["mod"], tiled["mod"][centre], rtol=1e-5, atol=1e-5 * scale)


def test_an_unknown_pad_mode_is_refused():
    with pytest.raises(ValueError, match="pad_mode"):
        PythonWTMMBackend().cwt2d(_field(16), [1.0], pad=4, pad_mode="toroidal", _engine="fftw")
