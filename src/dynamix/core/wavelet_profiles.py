# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every wavelet's section at the level on show, and the one physical length that labels it:
λ*, the bandpass peak wavelength (1/k*, k* where |ψ̂| peaks along the analysing direction, k in
cycles per pixel). Under the app's normalisation ψ̂_a(k) = ψ̂(ak), so λ* is a constant per
family times the device's own scale parameter: log-log slopes are untouched and "matched
scales" means the same length in every family. The design and the per-family constants:
``docs/specs/2026-10-13-wavelet-scale-bar-design.md`` §3.

A wavelet-bearing device implements ``wavelet_profile(params, scale_px, level) -> Profile``;
:func:`profile_for` dispatches to it. λ* is never read off an FFT of the drawn 1-D section (by
the projection-slice theorem that transform is the 2-D filter integrated over k_y, not its
slice); an isotropic kernel takes the k_x slice of its own 2-D Fourier form, a separable 1-D
filter its own cascade. Pure numpy/scipy/pywt: no Qt.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Profile", "format_length", "lambda_px", "profile_for", "wtmm_profile"]


@dataclasses.dataclass(frozen=True)
class Profile:
    x_px: np.ndarray          # pixel offsets along the section, symmetric about 0
    psi: np.ndarray           # amplitude, max |psi| = 1
    scale_px: float           # the device's own scale parameter at this level
    lambda_px: float          # λ* in pixels; nan when the kernel has no band
    family: str
    off_ladder: bool = False  # a fine dyadic level whose own constant is off the asymptotic one
    broad: bool = False       # half-max band wider than 4 octaves
    extra: dict = dataclasses.field(default_factory=dict)


def format_length(value: float, unit: str) -> str:
    """The notebook's three tiers: 480 m, 52.0 μm, 3.46 m."""
    v = float(value)
    text = f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")
    return f"{text} {unit}"


def _normalised(x, psi) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    peak = float(np.max(np.abs(psi))) if psi.size else 0.0
    return x, psi / peak if peak > 0 else psi


def wtmm_profile(smoothing: str, a: float) -> Profile:
    """The WTMM analysing wavelet ψ (``deriv_order`` 1) at nominal scale ``a``: the true 2-D
    section from :mod:`dynamix.core.scale_units`, and its λ* = 2πσ/√m."""
    from dynamix.core.scale_units import (analyzing_names, kernel_section, lambda_peak_px,
                                          sigma_px)

    x, psi = _normalised(*kernel_section(float(a), smoothing, 1))
    theta_name, psi_name = analyzing_names(smoothing)
    return Profile(x, psi, float(a), lambda_peak_px(float(a), smoothing, 1),
                   f"ψ {psi_name} (WTMM {smoothing})",
                   extra={"a": float(a), "sigma": sigma_px(float(a)), "theta": theta_name})


def profile_for(device_name: str, params: dict, result: dict, level: int):
    """The profile of ``device_name`` at the level on show, or ``None`` when the device has no
    wavelet, the result no scales, or the level no entry. ``result["_scale_idx"]`` (the
    ScaleSelect stamp) is the scale axis's own index; ``level`` the fallback."""
    from dynamix.model.device import get_device

    try:
        device = get_device(device_name)
    except Exception:                                                # noqa: BLE001
        return None
    hook = getattr(device, "wavelet_profile", None)
    scales = result.get("scales") if isinstance(result, dict) else None
    if hook is None or scales is None:
        return None
    scales = np.asarray(scales, dtype=np.float64)
    idx = int(result.get("_scale_idx", level))
    if not 0 <= idx < scales.size:
        return None
    return hook(dict(params or {}), float(scales[idx]), idx)


def lambda_px(device_name: str, params: dict, scale_px: float, level: int = 0) -> float:
    """λ* alone, for an axis: the device's constant times ``scale_px``; nan without a band."""
    from dynamix.model.device import get_device

    hook = getattr(get_device(device_name), "wavelet_profile", None)
    if hook is None:
        return float("nan")
    return float(hook(dict(params or {}), float(scale_px), int(level)).lambda_px)
