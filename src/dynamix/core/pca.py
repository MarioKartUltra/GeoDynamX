# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Principal component analysis over multi-component rasters -- the covariance-trick route.

Ported in SPIRIT from the author's ASTER notebook (``aster_him.ipynb`` cell 1: sklearn PCA over
flattened SWIR band stacks), reimplemented for the app with the efficiency mandate:

- A band stack is TALL-SKINNY -- (n_pixels, n_bands) with n_bands ~ 3..15 -- so the optimal
  algorithm is the covariance trick: one BLAS GEMM for the (n_bands, n_bands) Gram, one tiny
  ``eigh``, one GEMM to project. Accelerate/AMX parallelizes the GEMMs; sklearn (which SVDs the
  tall matrix), numba, and mlx would all be slower or pointless at this shape -- vectorized
  BLAS IS the right tool here, and that is a decision, not an omission.
- NaN-aware: a pixel with ANY non-finite band is excluded from the fit and comes back NaN in
  every component image (the nodata law); accumulation in float64 regardless of input dtype.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fit_pca"]


def fit_pca(stack, n_components: int, *, standardize: bool = False) -> dict:
    """PCA of an ``(ny, nx, nc)`` band stack (``nc >= 2``).

    Returns ``{"images" (k, ny, nx) float32 -- component scores as rasters, NaN at masked
    pixels; "components" (nc, k) -- the loading vectors (columns, unit norm);
    "explained_var_ratio" (k,); "mean" (nc,); "scale" (nc,) -- ones unless standardized}``.

    ``standardize=True`` divides each centered band by its standard deviation (correlation-PCA)
    -- the right choice when bands carry incommensurate units; reflectance stacks usually want
    the covariance default.
    """
    a = np.asarray(stack)
    if a.ndim != 3 or a.shape[2] < 2:
        raise ValueError(f"fit_pca needs an (ny, nx, nc>=2) stack, got shape {a.shape}")
    ny, nx, nc = a.shape
    k = int(min(n_components, nc))
    X = a.reshape(-1, nc).astype(np.float64, copy=False)
    valid = np.all(np.isfinite(X), axis=1)
    Xv = X[valid]
    if Xv.shape[0] < nc:
        raise ValueError(f"fit_pca: only {Xv.shape[0]} finite pixels for {nc} bands")
    mean = Xv.mean(axis=0)
    Xc = Xv - mean
    scale = np.ones(nc)
    if standardize:
        scale = Xc.std(axis=0)
        scale[scale == 0] = 1.0
        Xc = Xc / scale
    # The covariance trick: (nc, nc) Gram via one GEMM; eigh ascending -> take the top k.
    C = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    components = evecs[:, order[:k]]                     # (nc, k), unit columns
    # Sign convention: make each component's largest-|loading| entry positive, so repeated
    # runs (and float noise) cannot flip a displayed image's polarity silently.
    flip = np.sign(components[np.argmax(np.abs(components), axis=0), np.arange(k)])
    flip[flip == 0] = 1.0
    components = components * flip
    scores = Xc @ components                             # (n_valid, k) -- one GEMM
    total = float(evals.sum())
    images = np.full((k, ny * nx), np.nan, dtype=np.float32)
    images[:, valid] = scores.T.astype(np.float32)
    return {
        "images": images.reshape(k, ny, nx),
        "components": components,
        "explained_var_ratio": (evals[:k] / total if total > 0 else np.zeros(k)),
        "mean": mean,
        "scale": scale,
    }
