# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""tests/test_scale_units.py — the per-wavelet scale registry and its inverse-FFT oracle."""
import math

import numpy as np
import pytest

from dynamix.core.scale_units import (SIGMA_PER_SCALE_EXACT, analyzing_names,
                                      kernel_section, lambda_peak_px, outer_extremum_sigma,
                                      sigma_px)
from dynamix.core.wtmm_backend import _build_wavelet_filters_numpy, compute_scales2d


def test_sigma_exact_constant():
    assert SIGMA_PER_SCALE_EXACT == pytest.approx(1.0 / (math.pi * math.sqrt(2.0)))


def test_sigma_matches_halo_ladder():
    # halo.py's rounded ladder: sigma(finest normalized scale) ~ 1.57 px for a_min = 1.
    finest = compute_scales2d(1, 1, a_min=1.0)[0]          # 6.0 / 0.86
    assert sigma_px(finest) == pytest.approx(1.57, abs=5e-3)


def test_analyzing_names():
    assert analyzing_names("gaussian") == ("g0", "g1")
    assert analyzing_names("mexican") == ("g2", "g3")
    with pytest.raises(ValueError):
        analyzing_names("haar")


@pytest.mark.parametrize("smoothing,order,expected", [
    ("gaussian", 0, 0.0),
    ("gaussian", 1, 1.0),
    ("mexican", 0, 2.0),
    ("mexican", 1, math.sqrt((7.0 + math.sqrt(33.0)) / 2.0)),
])
def test_outer_extrema_constants(smoothing, order, expected):
    assert outer_extremum_sigma(smoothing, order) == pytest.approx(expected, rel=1e-6)


def test_lambda_peak_constants_and_ordering():
    s = 40.0
    lam_g = lambda_peak_px(s, "gaussian", 1)                 # m = 1
    lam_mt = lambda_peak_px(s, "mexican", 0)                 # m = 2
    lam_mp = lambda_peak_px(s, "mexican", 1)                 # m = 3
    assert lam_g == pytest.approx(2 * math.pi * sigma_px(s))
    assert lam_mt == pytest.approx(2 * math.pi * sigma_px(s) / math.sqrt(2))
    assert lam_mp == pytest.approx(2 * math.pi * sigma_px(s) / math.sqrt(3))
    assert lam_mp < lam_g                # mexican probes FINER structure at the same scale
    assert lambda_peak_px(2 * s, "mexican", 1) == pytest.approx(2 * lam_mp)
    with pytest.raises(ValueError):
        lambda_peak_px(s, "gaussian", 0)                     # pure lowpass: no bandpass


@pytest.mark.parametrize("wavelet,order,m", [
    ("gaussian", 1, 1), ("mexican", 0, 2), ("mexican", 1, 3),
])
def test_lambda_peak_against_filter_argmax(wavelet, order, m):
    """Fourier-side oracle: the radial profile |k|^m exp(-s^2 k^2) peaks at k = sqrt(m/2)/s,
    i.e. lambda = 1/k_peak must equal lambda_peak_px."""
    s = 40.0
    k = np.linspace(1e-6, 0.5, 200001)
    profile = (k ** m) * np.exp(-((s * k) ** 2))
    k_peak = k[int(np.argmax(profile))]
    assert lambda_peak_px(s, wavelet, order) == pytest.approx(1.0 / k_peak, rel=1e-4)


def test_kernel_section_extrema_self_consistent():
    # The registered outer extremum must be a stationary point of the sampled section.
    s = 40.0
    x, y = kernel_section(s, "mexican", 1, n_points=4001)
    outer = outer_extremum_sigma("mexican", 1) * sigma_px(s)
    dy = np.gradient(y, x)
    i = int(np.argmin(np.abs(x - outer)))
    assert abs(dy[i]) < np.max(np.abs(dy)) * 1e-3


def _center_row_of_ifft(filt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real-space kernel's y=0 cross-section from a Fourier-domain filter, centered."""
    kern = np.fft.ifft2(filt).real
    kern = np.fft.fftshift(kern)
    n = kern.shape[0]
    row = kern[n // 2, :]
    x = np.arange(n, dtype=np.float64) - n // 2
    return x, row


def _subpixel_extremum(x, y, i):
    """Quadratic interpolation around sample i -> sub-pixel extremum position."""
    denom = y[i - 1] - 2 * y[i] + y[i + 1]
    if denom == 0:
        return x[i]
    return x[i] + 0.5 * (y[i - 1] - y[i + 1]) / denom


@pytest.mark.parametrize("wavelet", ["gaussian", "mexican"])
def test_oracle_psi_section_matches_pipeline_filter(wavelet):
    """THE oracle: ifft the actual pipeline 'dx' filter, compare against kernel_section."""
    n, s = 1024, 40.0                                        # sigma ~ 9 px on a 1024 grid
    k = np.fft.fftfreq(n).astype(np.float32)
    kx, ky = np.meshgrid(k, k)
    filt = _build_wavelet_filters_numpy(kx, ky, s, wavelet=wavelet)["dx"]
    x, row = _center_row_of_ifft(filt)

    # (a) outermost extremum position matches the registry constant.
    sig = sigma_px(s)
    predicted = outer_extremum_sigma(wavelet, 1) * sig
    # scan outward-in for the outermost local extremum of the (odd) section's positive side
    pos = row[x > 0]
    xp = x[x > 0]
    idx = [i for i in range(1, len(pos) - 1)
           if (pos[i] - pos[i - 1]) * (pos[i + 1] - pos[i]) <= 0
           and abs(pos[i]) > 1e-4 * np.max(np.abs(pos))]
    outermost = max(_subpixel_extremum(xp, pos, i) for i in idx)
    assert outermost == pytest.approx(predicted, rel=0.02)

    # (b) full shape agreement over +/-3 sigma (float32 filters -> loose atol).
    # Reverse interpolation direction: evaluate fine analytic grid at measured integer pixels
    # to avoid ~0.5% loss from interpolating coarse measurement onto fine analytic grid.
    xs, ys = kernel_section(s, wavelet, 1, n_points=8001)  # Fine grid: ~0.03 px spacing
    keep = np.abs(x) <= 3 * sig  # Keep only measured points within ±3σ
    pred = np.interp(x[keep], xs, ys)  # Analytic evaluated at integer pixel positions
    row_keep = row[keep]
    # Normalize each by its own max(abs) over the same window
    ys_n = pred / np.max(np.abs(pred))
    sampled_n = row_keep / np.max(np.abs(row_keep))
    assert np.allclose(ys_n, sampled_n, atol=5e-3)


@pytest.mark.parametrize("wavelet", ["gaussian", "mexican"])
def test_oracle_theta_radial_profile(wavelet):
    """The smoother's radial profile: build the plain smoother filter the same way the
    pipeline does (gauss, or |k|^2*gauss) and compare its section against deriv_order=0."""
    n, s = 1024, 40.0
    k = np.fft.fftfreq(n).astype(np.float32)
    kx, ky = np.meshgrid(k, k)
    sx, sy = kx * s, ky * s
    gauss = np.exp(-(sx * sx + sy * sy)).astype(np.float32)
    filt = gauss if wavelet == "gaussian" else ((sx * sx + sy * sy) * gauss)
    x, row = _center_row_of_ifft(filt.astype(np.complex64))
    sig = sigma_px(s)
    # Reverse interpolation direction: evaluate fine analytic grid at measured integer pixels.
    xs, ys = kernel_section(s, wavelet, 0, n_points=8001)  # Fine grid: ~0.03 px spacing
    keep = np.abs(x) <= 3 * sig  # Keep only measured points within ±3σ
    pred = np.interp(x[keep], xs, ys)  # Analytic evaluated at integer pixel positions
    row_keep = row[keep]
    # Normalize each by its own max(abs) over the same window
    ys_n = pred / np.max(np.abs(pred))
    sampled_n = row_keep / np.max(np.abs(row_keep))
    assert np.allclose(ys_n, sampled_n, atol=5e-3)


def test_no_bare_scale_times_pixel_size_in_shell():
    """Nominal wavelet scales must reach physical units only through scale_units.
    Guard: the two historical offender EXPRESSIONS stay dead in the shell sources.
    Word boundaries ensure we catch bare variables (px, a) but not embedded names
    (bar_px, sigma, lam, scale_px, m_per_px) which are legitimate sites."""
    import pathlib
    import re

    _OFFENDER_PATTERNS = (
        re.compile(r"\bpx \* self\._px_to_unit\b"),   # transport's old nominal-scale-as-px
        re.compile(r"\ba \* m_per_px\b"),             # the bar's old bare-scale conversion
    )

    shell = pathlib.Path(__file__).resolve().parent.parent / "src" / "dynamix" / "shell"
    offenders = []
    for py in shell.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if any(p.search(text) for p in _OFFENDER_PATTERNS):
            offenders.append(py.name)
    assert offenders == []


# -- the registry's smoothers through the same three helpers -----------------------------------

def test_registry_smoothers_have_sections_names_and_lambda_through_scale_units():
    s = 8.0
    lam_g = lambda_peak_px(s, "gaussian", 1)
    for name, params in (("frac_bspline", {"alpha": 3.0}), ("q_gaussian", {"q": 1.5}),
                         ("q_mexican", {"q": 1.5}), ("q_meyer", {"q": 1.0})):
        x, y = kernel_section(s, name, 1, n_points=1001, params=params)
        assert x[0] < 0 < x[-1] and np.allclose(y, -y[::-1], atol=1e-9)        # ψ_x is odd
        lam = lambda_peak_px(s, name, 1, params=params)
        assert lambda_peak_px(2 * s, name, 1, params=params) == pytest.approx(2 * lam, rel=1e-3)
    # Matched σ: the fractional B-spline at α = 3 reads the Gaussian's λ* within 5 %. The
    # q-Gaussian's tails order its band: heavier than the Gaussian's above q = 1 (longer λ*),
    # compact below (shorter).
    assert lambda_peak_px(s, "frac_bspline", 1, params={"alpha": 3.0}) == pytest.approx(lam_g, rel=0.05)
    assert lambda_peak_px(s, "q_gaussian", 1, params={"q": 1.5}) > 1.2 * lam_g
    assert lambda_peak_px(s, "q_gaussian", 1, params={"q": 0.5}) < 0.85 * lam_g
    assert analyzing_names("q_gaussian", {"q": 1.0}) == ("q-Gaussian q = 1", "∂θ_q")
    assert analyzing_names("frac_bspline", {"alpha": 3.0})[1] == "∂θ_α"
    assert analyzing_names("q_mexican", {"q": 1.0})[1] == "q-g3"
    assert lambda_peak_px(s, "q_mexican", 1, params={"q": 1.0}) == pytest.approx(
        lambda_peak_px(s, "mexican", 1), rel=1e-3)                            # q → 1 is the Mexican
    assert lambda_peak_px(s, "q_meyer", 1, params={"q": 1.0}) == pytest.approx(lam_g, rel=2e-3)
    assert lambda_peak_px(s, "q_gaussian", 1, params={"q": 1.0}) == pytest.approx(
        lambda_peak_px(s, "gaussian", 1), rel=1e-3)                            # q → 1 is the Gaussian
    with pytest.raises(ValueError):
        kernel_section(s, "frac_bspline", 1, params={"alpha": 0.5})
