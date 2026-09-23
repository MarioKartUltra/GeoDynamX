# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""WTMM2D's dither option (2026-08-30): the M-Z devices' seeded half-LSB dither, on the main
WTMM path -- so processing noise is broken up before chaining.
Same core helpers (dynamix.core.mz_edges.measure_lsb/_dithered), same fail-toward-no-op rule."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dynamix.core.rasterfield import RasterField
from dynamix.devices.wtmm import WTMM2D
from dynamix.model.device import defaults_for

_SMALL = {"n_oct": 2, "n_voice": 2, "a_min": 1.0, "wavelet": "mexican", "min_chain_len": 2,
          "smooth": False, "thresh": 1e-3, "dist2_max": 50.0, "box_ratio": 1.0, "similitude": 0.8}


def _quantized_field(name="dq.npy"):
    rng = np.random.default_rng(7)
    vals = np.round(rng.random((48, 48)) * 20.0) * 0.5          # a coarse 0.5 lattice
    return RasterField._from_bare_array(vals, Path(name))


def _params(**over):
    d = defaults_for(WTMM2D())
    d.update(_SMALL)
    d.update(over)
    return d


def test_dither_defaults_off_and_changes_the_result_when_on():
    field = _quantized_field()
    d = defaults_for(WTMM2D())
    assert d["dither"] is False
    raw = WTMM2D().compute(field, _params())
    dit = WTMM2D().compute(field, _params(dither=True))
    raw_mod = np.concatenate([np.asarray(l["mod"]) for l in raw["extrema"]])
    dit_mod = np.concatenate([np.asarray(l["mod"]) for l in dit["extrema"]])
    assert raw_mod.shape != dit_mod.shape or not np.allclose(raw_mod, dit_mod)


def test_dither_is_deterministic_across_runs():
    field = _quantized_field()
    a = WTMM2D().compute(field, _params(dither=True))
    b = WTMM2D().compute(field, _params(dither=True))
    np.testing.assert_array_equal(a["extrema"][0]["mod"], b["extrema"][0]["mod"])
