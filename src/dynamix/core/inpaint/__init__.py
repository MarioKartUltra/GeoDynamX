# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The inpainting and fusion engine: fill declared regions of a raster from its own cascade,
from cycle-spun wavelet sparsity, and from co-registered guides on their own grids.

Modules: ``grid`` (how two rasters relate), ``forward`` (measurement operators and terms),
``priors`` (the sparsity prox, the cascade inference, the abacus), ``correct`` (the pre-fill
and the known-pixel correction), ``readouts`` (held-out blocks, errors, histograms),
``solver`` (the two loops), ``sends`` (guides resolved each on its own grid), ``engine`` (one
run), ``_kernels`` (the numba loops). Nothing here imports Qt; numba and PyWavelets are
imported lazily.
"""
