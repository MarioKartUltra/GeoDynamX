# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Noise: seeded dither as a CHAIN STEP, ahead of the analysing transform.

Distinct from ``wtmm2d``'s own Dither checkbox (auto half-LSB, no knobs): this is a Transform
that hands the NEXT transform a noised clone of the field, so it composes in the chain, caches
by its own params (the engine threads the lineage into every downstream cache key), and offers
the manual amplitude a stitched mosaic needs -- ``measure_lsb`` is honest only when ONE value
lattice exists; a mosaic of several quantisations has no single LSB to measure.

Conventions:

* ``amplitude`` is the HALF-WIDTH for uniform noise (``U(-a, +a)``; the classic ±half-LSB dither
  is ``a = lsb/2``) and the SIGMA for gaussian. ``0`` means auto: half the measured LSB
  (:func:`dynamix.core.mz_edges.measure_lsb`), and a field with no measurable lattice passes
  through UNCHANGED -- fails toward no-op, never corrupts (the M-Z rule).
* ``seed`` makes the noise deterministic -- same seed, same noise, byte for byte -- so a run is
  reproducible and an A/B is honest. ``sign="subtract"`` applies the same seeded noise negated.
* The clone's NAME encodes the params: ``wtmm_backend``'s stage cache keys on ``field.name``,
  and a noised run must never reload a raw run's stages (or another seed's).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamix.model.param import Param, ParamKind


class Noise:
    """Add (or subtract) seeded noise to the field before the analysing transform."""

    name = "noise"
    #: A FIELD STAGE (field -> field) -- on an ROI layer it runs over the
    #: whole processing window before the analyzer (noise covers the ROI and its surrounding pixels before processing). Pointwise: no margin.
    field_stage = True

    def roi_margin(self, params: dict) -> int:
        return 0

    params = (
        Param("dist", ParamKind.CHOICE, default="uniform",
              choices=("uniform", "gaussian"), label="Distribution"),
        Param("amplitude", ParamKind.FLOAT, default=0.0, min=0.0, max=1e9,
              soft_min=0.0, soft_max=1.0,
              label="Amplitude (0 = auto ½ LSB; uniform: half-width, gaussian: σ)"),
        Param("seed", ParamKind.INT, default=0, min=0, max=2**31 - 1,
              soft_min=0, soft_max=64, label="Seed"),
        Param("sign", ParamKind.CHOICE, default="add", choices=("add", "subtract"),
              label="Apply"),
    )

    def compute(self, field, params: dict, *, progress=None):
        if isinstance(field, dict):
            raise ValueError("noise runs on the FIELD, before the analysing transform — "
                             "drag it ahead of wtmm2d in the chain")
        from dynamix.core.mz_edges import measure_lsb

        values = np.asarray(field.values, dtype=np.float64)
        amp = float(params["amplitude"])
        if amp <= 0.0:
            lsb = measure_lsb(values)
            if not lsb:
                return field                       # no lattice, nothing honest to add: no-op
            amp = lsb / 2.0
        rng = np.random.default_rng(int(params["seed"]))
        if params["dist"] == "gaussian":
            noise = rng.normal(0.0, amp, values.shape)
        else:
            noise = rng.uniform(-amp, amp, values.shape)
        if params["sign"] == "subtract":
            noise = -noise
        tag = f"{params['dist'][0]}{amp:g}s{int(params['seed'])}{'-' if params['sign'] == 'subtract' else '+'}"
        return dataclasses.replace(field, values=values + noise,
                                   name=f"{field.name}+noise{tag}")

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
