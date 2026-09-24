# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""holder_map — the per-pixel Hölder/singularity-exponent raster as a Transform.

Wraps :mod:`dynamix.core.microcanonical`'s per-pixel estimators: the MULTIAFFINE Ricker route
(``ricker_projections`` -- the default, the estimator for smooth/function-class fields, with
the Turiel 2008 fig-2 minimum-scale convention: scales are zero-crossing radii, ``r_min = 1``
means 2 samples between the Ricker's zero crossings) and the gradient-MEASURE route
(``measure_projections`` over ``||grad s||`` -- for measure-like/edge-sparse fields; on smooth
fields it is mean-dominated, Pont 2006). Per-pixel h from the vectorized log-log regression
with ``r2_min = 0`` -- the R² gate is a selection bias on both routes (see the core module and
the benchmark report), so no gating here; ``r2_map`` rides the result for anyone who wants to
mask deliberately.

The result's ``"h_map"`` key is this device's display contract (the ``backproject``
``points_px`` precedent -- no other device stamps it): ``main_window`` shows the raster on the
canvas IN PLACE of the raw field through the ordinary ``set_field`` path, so stretch/hillshade
apply to the exponent field like any raster. ``chains``/``extrema`` are stamped empty
deliberately (the ``mz_edges`` precedent) -- this transform has no line skeleton.

This transform analyses the LAYER'S FIELD, so it must head its own chain: a Transform placed
after another receives the upstream RESULT dict (the ``chain_topology`` contract), which is
refused here with a pointed message rather than mis-analysed.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind


def field_values_2d(field, device_name: str):
    """The layer field's scalar 2-D values, with the shared refusals (a Transform placed
    after another receives that transform's RESULT dict -- the chain_topology contract; note
    a plain ``hasattr(field, "values")`` guard would pass a dict, ``dict.values`` the METHOD)."""
    import numpy as np

    if isinstance(field, dict) or not hasattr(field, "values"):
        raise ValueError(
            f"{device_name} analyses the layer's own field -- place it FIRST in its chain "
            "(a Transform placed after another receives that transform's result)")
    vals = np.asarray(field.values, dtype=np.float64)
    if vals.ndim != 2:
        raise ValueError(f"{device_name} needs a scalar 2-D field; got shape {vals.shape}")
    return vals


def holder_arrays(vals, params: dict, progress=None):
    """``(h_map, r2_map, scales)`` for the shared estimator/wavelet/scale params -- the one
    engine both ``holder_map`` and ``band_recon`` run (2026-09-16)."""
    import numpy as np

    from dynamix.core import microcanonical as mc

    scales = np.geomspace(params["r_min"], params["r_min"] * params["kappa"],
                          int(params["n_scales"]))
    if progress is not None:
        progress("holder projections", 0.0)
    # Projections are the slow part: reported per scale, over 0..70 %.
    per_scale = (None if progress is None
                 else (lambda stage, frac: progress(stage, 0.7 * float(frac))))
    if params["estimator"] == "multiaffine":
        T = mc.ricker_projections(vals, scales, wavelet=params["wavelet"],
                                  beta=params["beta"], q_tsallis=params["q_tsallis"],
                                  progress=per_scale)
    else:
        T = mc.measure_projections(mc.gradient_measure(vals), scales,
                                   wavelet=params["wavelet"], beta=params["beta"],
                                   q_tsallis=params["q_tsallis"], progress=per_scale)
    if progress is not None:
        progress("holder regression", 0.7)
    h_map, r2_map = mc.singularity_map_regression(T, scales, r2_min=0.0)
    if progress is not None:
        progress("holder regression", 1.0)
    # NaN in, NaN out (2026-09-22): a nodata pixel never gets an exponent, whatever the
    # kernel -- heavy tails used to report their own tail exponent (2beta - d) there, and even
    # a Gaussian reports one near the nodata edge (its reach crosses to real data).
    nodata = ~np.isfinite(np.asarray(vals, dtype=np.float64))
    if nodata.any():
        h_map = np.asarray(h_map).copy()
        r2_map = np.asarray(r2_map).copy()
        h_map[nodata] = np.nan
        r2_map[nodata] = np.nan
    return h_map, r2_map, scales


class HolderMap:
    name = "holder_map"
    params = (
        # Turiel 2008's own dichotomy, named HIS way (renamed from "ricker" 2026-09-16 --
        # user caught the conflation: the Ricker is a WAVELET, the estimator is the FUNCTIONAL):
        # multiaffine = |T_psi s|, the SIGNAL projected with a zero-mean wavelet (SS4.2.1;
        # smooth/function-class fields, slope = gamma); measure = positive-kernel projections
        # of ||grad s|| (SS4.2.2; measure-like fields, slope = h). Frames differ by -1
        # (eq 29-31: h_measure = gamma - 1); no silent conversion is applied. The Wavelet knob
        # below picks the ENVELOPE for either route (zero-mean Marr form for multiaffine, the
        # positive kernel for measure).
        Param("estimator", ParamKind.CHOICE, default="multiaffine",
              choices=("multiaffine", "measure"), label="Estimator"),
        # Defaults are Turiel 2009's image practice: "a range of scales typically going from
        # 1 to 8 pixels non uniformly sampled" -- r1=1 (the fig-2 zero-crossing minimum),
        # kappa=8, geometric. MEASURED (2026-09-16, fBm 256^2): the sample COUNT within a range
        # is nearly irrelevant (dyadic 4-pt vs geometric 6-pt over kappa=8: r=0.95) while the
        # RANGE dominates (kappa=32 vs 8: r=0.51; the prototype notebook's 2..50 grid: r=-0.06,
        # a different, coarse-dominated field) -- so n_scales stays small for scrub speed and
        # kappa is the knob that actually changes the physics.
        Param("r_min", ParamKind.FLOAT, default=1.0, min=0.1, max=32.0,
              soft_min=1.0, soft_max=4.0, units="px", label="r₁"),
        Param("kappa", ParamKind.FLOAT, default=8.0, min=1.1, max=200.0,
              soft_min=4.0, soft_max=12.0, units="", label="κ = r₂/r₁"),
        Param("n_scales", ParamKind.INT, default=6, min=2, max=64,
              soft_min=4, soft_max=10, label="Scales"),
        # Envelope family, shared by BOTH estimators (positive kernel for the measure route,
        # its exact 2-D Laplacian -- the generalized Mexican hat -- for the ricker route). All
        # three share the zero-crossing-radius scale convention. Heavy tails (q > 1, low beta)
        # trade calibration for localization: measured on fBm H=0.8, gamma medians 0.71 /
        # 0.43 / 0.29 for gaussian / q=1.5 / L1 -- values are NOT cross-comparable between
        # wavelets. q_tsallis binds only for q_gaussian, beta only for lorentzian; q = 1.5
        # coincides exactly with lorentzian beta = 2 (Student-t family identity).
        Param("wavelet", ParamKind.CHOICE, default="gaussian",
              choices=("gaussian", "q_gaussian", "lorentzian"), label="Wavelet"),
        Param("q_tsallis", ParamKind.FLOAT, default=1.5, min=1.0, max=3.0,
              soft_min=1.1, soft_max=2.0, units="", label="q"),
        Param("beta", ParamKind.FLOAT, default=1.0, min=0.1, max=8.0,
              soft_min=0.5, soft_max=2.0, units="", label="β"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        vals = field_values_2d(field, self.name)
        h_map, r2_map, scales = holder_arrays(vals, params, progress)
        return {
            "h_map": h_map, "r2_map": r2_map, "chains": [], "extrema": [],
            "scales": scales, "params": dict(params),
            "_frame": getattr(field, "frame", None), "_shape": vals.shape,
        }

    def roi_margin(self, params: dict) -> int:
        """The largest kernel's measured 2-D reach, on the route its
        ``estimator`` selects (:func:`dynamix.devices.holder_methods.holder_roi_margin`)."""
        from dynamix.devices.holder_methods import holder_roi_margin

        method = "multiaffine" if params["estimator"] == "multiaffine" else "measure"
        return holder_roi_margin(method, params)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
