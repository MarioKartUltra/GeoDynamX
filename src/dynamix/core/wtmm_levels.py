# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""One scale of the scalar WTMM pipeline's cwt stage: what the per-scale raster outputs show.

The backend (:mod:`dynamix.core.wtmm_backend`) is a tracked EQSelect copy and stays as it is. This
module reproduces one scale of its cwt stage (``_run_wtmm2d_scalar``'s ``compute_cwt`` and the
lift after it) from the backend's own pieces: the typo guard ``_resolve_wtmm2d_params``, the
scale ladder ``compute_scales2d``, ``PythonWTMMBackend.cwt2d`` called as the stage calls it for
the run's detector, and the ``_apply_fracint2d`` lift. Like :mod:`dynamix.core.hlines` it
imports private backend names; at module level it pulls in nothing but numpy and the backend,
which itself imports only numpy and the standard library.
"""
from __future__ import annotations

import numpy as np

from dynamix.core.wtmm_backend import (_apply_fracint2d, _resolve_wtmm2d_params,
                                       compute_scales2d, get_backend)

__all__ = ["run_wtmm2d_level"]


def run_wtmm2d_level(values, params: dict, level: int, *, backend=None) -> dict:
    """Scale ``level`` (0-based, finest first) of the scalar pipeline's cwt stage, lifted.

    The stage's own ``backend.cwt2d`` call on the one-scale list ``scales[level:level + 1]``:
    ``derivs="first"`` under the NMS detector (the stage's all-scales call, of which each
    scale's filter reads only the padded field's FFT and that scale, so a one-scale call gives
    the same values), ``derivs="all"`` under the follow detector (the stage's own per-scale
    call). Then the same :func:`~dynamix.core.wtmm_backend._apply_fracint2d` lift. So the result
    is value-identical to that scale of :func:`~dynamix.core.wtmm_backend.run_wtmm2d`, as
    ``run_wtmm2d_preview`` is for the finest one. ``values`` is the 2-D field as the analysis
    saw it (a dither is the caller's, as for ``run_wtmm2d``); ``params`` resolves through the
    backend's typo guard.

    Returns ``{"mod": (ny, nx) float32, "arg": (ny, nx) float32, "scale": float}``: the lifted
    modulus the extrema stage detects on, its angle (rad), and the normalised scale in px.
    Uncached: the output's cache entry is the caller's.
    """
    resolved = _resolve_wtmm2d_params(params)
    if resolved["mode"] != "scalar":
        raise ValueError("run_wtmm2d_level: scalar mode only (one scale of the cwt stage)")
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"run_wtmm2d_level: needs a 2-D field; got shape {values.shape}")
    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    level = int(level)
    if not 0 <= level < len(scales):
        raise ValueError(f"run_wtmm2d_level: level {level} is outside 0..{len(scales) - 1}")
    backend = backend if backend is not None else get_backend("python")
    one = scales[level:level + 1]
    derivs = "all" if resolved["detector"] == "follow" else "first"
    cwt = backend.cwt2d(values, one, wavelet=resolved["wavelet"], pad=resolved["pad"],
                        pad_mode=resolved["pad_mode"], derivs=derivs,
                        smoother_params=resolved["smoother_params"])
    cwt = _apply_fracint2d({"mod": cwt["mod"], "arg": cwt["arg"]}, one,
                           resolved["fracint_alpha"])
    return {"mod": cwt["mod"][0], "arg": cwt["arg"][0], "scale": float(one[0])}
