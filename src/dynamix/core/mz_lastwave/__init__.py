# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The authors' 2-D multiscale-edge pipeline, ported exactly from LastWave's ``dwtrans2d``.

``dwt2d`` (the dyadic wavelet transform with the p3.1 / p3.2 filters, computed by direct FIR
convolution), ``extrema2`` (the multiscale edge detection) and ``e2recons`` (reconstruction from the
edges by alternating projections), bit for bit with the authors' C on square fields; numba is
required and imported lazily.
"""
from dynamix.core.mz_lastwave import _kernels
from dynamix.core.mz_lastwave._kernels import kernels
from dynamix.core.mz_lastwave.transform import (
    FACT1, FILTERS, Transform, _symmetry_centre, dwt2d, fact, idwt2d, polar,
)

__all__ = ["FACT1", "FILTERS", "Transform", "dwt2d", "fact", "idwt2d", "kernels", "polar"]
