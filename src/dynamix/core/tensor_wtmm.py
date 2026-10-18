# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The tensor WTMM: the alpha-Jacobian pipeline over a three-component field.

The app's own port (spec 09, Phase B) of the vendored ``wtmm_ebsd.twtmm.alpha_jacobian_twtmm``,
checked against its golden (``tests/golden/vendor/tensor.npz``). Three component CWTs give the
Jacobian ``kappa_ij = d g_i / d x_j`` per scale; the Pantleon alpha components (zero elastic
strain, Eqs. 11 and 13) and their derivatives (the wavelet Hessian, or a pixel gradient) give
the 6 × 2 alpha-Jacobian, whose singular values and leading direction come from the 2 × 2
closed form of JᵀJ; the 3 × 2 kappa Jacobian gives the sigma, modL and modT diagnostics. Each
chosen mode's modulus and argument then run the scalar WTMM: per-scale maxima through the
injected ``xsm`` shim, chains through the injected ``chain_and_extract``, per-chain Hölder
exponents through ``xsm.compute_holder_exponents``. The injected callables are the backend's
own (``dynamix.core.wtmm_backend``), as the vendored driver took them.
"""
from __future__ import annotations

import numpy as np

#: Every mode the driver builds, and the argument field each one runs the WTMM with.
ARG_FOR_MODE = {"sigma_max": "arg", "sigma_min": "arg", "modL": "arg", "modT": "arg",
                "alpha_sigma_max": "alpha_arg", "alpha_sigma_min": "alpha_arg"}


def _svd_3x2(k11, k12, k21, k22, k31, k32):
    """Per pixel, the singular values and the leading right singular direction of the 3 × 2
    Jacobian, from the 2 × 2 closed form of JᵀJ (the half-angle form of the direction)."""
    A = k11 * k11 + k21 * k21 + k31 * k31
    C = k12 * k12 + k22 * k22 + k32 * k32
    B = k11 * k12 + k21 * k22 + k31 * k32
    return _eig_2x2(A, B, C)


def _eig_2x2(A, B, C):
    tr = A + C
    df = A - C
    disc = np.sqrt(df * df + 4.0 * B * B)
    lam_max = 0.5 * (tr + disc)
    lam_min = np.maximum(0.5 * (tr - disc), 0.0)
    return (np.sqrt(lam_max).astype(np.float32), np.sqrt(lam_min).astype(np.float32),
            (0.5 * np.arctan2(2.0 * B, df)).astype(np.float32))


def alpha_jacobian_twtmm(field3: np.ndarray, scales, *, cwt_2d_f32, xsm, chain_and_extract,
                         fracint_alpha: float = 1.0, pad: int = 32, wavelet: str = "gaussian",
                         wavelet_hessian: str = "gaussian", use_wavelet_hessian: bool = True,
                         similitude: float = 0.8, smooth_chain: bool = False,
                         thresh: float = 1e-3, svd_modes: tuple = ("sigma_max",),
                         verbose: bool = False) -> dict:
    """``{mode: {"chains", "holders", "mod", "ext_images", "scales"}}`` for every mode of
    ``svd_modes`` (names in :data:`ARG_FOR_MODE`). ``field3`` is ``(ny, nx, 3)``; ``scales`` in
    pixels; ``fracint_alpha`` scales every derivative at scale ``a`` by ``a**fracint_alpha``."""
    field3 = np.asarray(field3)
    if field3.ndim != 3 or field3.shape[-1] != 3:
        raise ValueError(f"field3 must be (ny, nx, 3); got {field3.shape}")
    scales = np.asarray(scales, dtype=np.float64)
    n_sc = len(scales)
    ny, nx = field3.shape[:2]

    derivs = [cwt_2d_f32(field3[..., c], scales, pad=pad, derivs="first", wavelet=wavelet,
                         verbose=False) for c in range(3)]
    hess = ([cwt_2d_f32(field3[..., c], scales, pad=pad, derivs="all", wavelet=wavelet_hessian,
                        verbose=False) for c in range(3)] if use_wavelet_hessian else None)
    if fracint_alpha != 0:
        for c in range(3):
            for si, scale in enumerate(scales):
                f = np.float32(scale ** fracint_alpha)
                derivs[c]["dx"][si] *= f
                derivs[c]["dy"][si] *= f
                if hess is not None:
                    for name in ("dxx", "dxy", "dyy"):
                        hess[c][name][si] *= f

    fields = {name: np.zeros((n_sc, ny, nx), dtype=np.float32)
              for name in ("sigma_max", "sigma_min", "modL", "modT", "arg",
                           "alpha_sigma_max", "alpha_sigma_min", "alpha_arg")}
    for si in range(n_sc):
        k11, k12 = derivs[0]["dx"][si], derivs[0]["dy"][si]
        k21, k22 = derivs[1]["dx"][si], derivs[1]["dy"][si]
        k31, k32 = derivs[2]["dx"][si], derivs[2]["dy"][si]
        smax, smin, karg = _svd_3x2(k11, k12, k21, k22, k31, k32)
        fields["sigma_max"][si], fields["sigma_min"][si], fields["arg"][si] = smax, smin, karg
        fields["modL"][si] = np.sqrt(k11 ** 2 + k22 ** 2 + 0.25 * (k21 + k12) ** 2
                                     + k31 ** 2 + k32 ** 2).astype(np.float32)
        fields["modT"][si] = (0.5 * np.abs(k21 - k12)).astype(np.float32)
        # the six alpha components' derivatives: the wavelet Hessian, or a pixel gradient
        if hess is not None:
            g1, g2, g3 = (hess[c] for c in range(3))
            g1_xx, g1_xy, g1_yy = g1["dxx"][si], g1["dxy"][si], g1["dyy"][si]
            g2_xx, g2_xy, g2_yy = g2["dxx"][si], g2["dxy"][si], g2["dyy"][si]
            g3_xx, g3_xy, g3_yy = g3["dxx"][si], g3["dxy"][si], g3["dyy"][si]
            da_x = [g2_xx, g3_xx, g1_xy, g3_xy, -(g1_xx + g2_xy), (g1_xx - g2_xy)]
            da_y = [g2_xy, g3_xy, g1_yy, g3_yy, -(g1_xy + g2_yy), (g1_xy - g2_yy)]
        else:
            alpha = (k21, k31, k12, k32, -(k11 + k22), k11 - k22)
            da_x = [np.gradient(a, axis=1).astype(np.float32) for a in alpha]
            da_y = [np.gradient(a, axis=0).astype(np.float32) for a in alpha]
        A = np.zeros((ny, nx), dtype=np.float32)
        B = np.zeros((ny, nx), dtype=np.float32)
        C = np.zeros((ny, nx), dtype=np.float32)
        for dx, dy in zip(da_x, da_y):
            A += (dx * dx).astype(np.float32)
            B += (dx * dy).astype(np.float32)
            C += (dy * dy).astype(np.float32)
        a_smax, a_smin, a_arg = _eig_2x2(A, B, C)
        fields["alpha_sigma_max"][si] = a_smax
        fields["alpha_sigma_min"][si] = a_smin
        fields["alpha_arg"][si] = a_arg

    out: dict = {}
    for mode in svd_modes:
        if mode not in ARG_FOR_MODE:
            raise ValueError(f"unknown tensor WTMM mode {mode!r}; known: {sorted(ARG_FOR_MODE)}")
        mod, arg = fields[mode], fields[ARG_FOR_MODE[mode]]
        ext_images = []
        for si, scale in enumerate(scales):
            dx_r = (mod[si] * np.cos(arg[si])).astype(np.float32)
            dy_r = (mod[si] * np.sin(arg[si])).astype(np.float32)
            ext, _, _ = xsm.wtmm2d(xsm.XImage.from_numpy(dx_r), xsm.XImage.from_numpy(dy_r),
                                   scale, thresh=thresh)
            ext_images.append(ext)
        chains = chain_and_extract(ext_images, scales, similitude=similitude,
                                   smooth=smooth_chain)
        out[mode] = {"chains": chains, "holders": xsm.compute_holder_exponents(chains),
                     "mod": mod, "ext_images": ext_images, "scales": scales.copy()}
    return out
