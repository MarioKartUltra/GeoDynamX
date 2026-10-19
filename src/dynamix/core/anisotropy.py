# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Anisotropy of a WTMM skeleton -- Arnéodo, Decoster & Roux 2000 (Eur. Phys. J. B 15:567,
§6 and Figs 24–26) and the directional roughness of Khalil, Joncas, Nekka, Kestener & Arnéodo
2006 (ApJS 165:512, §§3.3.2, 4.2.4, 5.2.4).

At each scale ``a`` the WTMMM (the maxima of the modulus along each maxima line: xsmurf's
``ssm``) carry a modulus M and an argument A. Their joint pdf P_a(M, A) says:

- P_a(A) FLAT -> isotropic scale invariance; non-uniform -> anisotropy, and its change across
  scales says whether the anisotropy strengthens or weakens (Fig. 24b);
- the WTMMM in the gradient plane (T_ψ1, T_ψ2) = (M cos A, M sin A): radial symmetry ->
  isotropy (Fig. 25);
- the pdf of M conditioned on A mod π in 4 sectors: shape invariance -> M and A independent,
  so the singularity spectrum carries no directional information (Fig. 26).

Khalil et al. make the anisotropy a pair of numbers: the two components of the WTMMM scale on
their own, T_ψ1 ~ a^{H_x} and T_ψ2 ~ a^{H_y} (their eqs. 33–36), so the pdf of each component
collapses onto one curve once rescaled by a^{H}; the peaks of P_a(A) sharpen towards fine
scales as tan A ~ a^{H_y − H_x} (eq. 38); the modulus follows whichever component dominates over
the scales at hand (their §5.2.4); the one-dimensional cuts' power spectra read β_x = 2H_x + 1
and β_y = 2H_y + 1 (their Fig. 9); and a^{−qH} M^q P_a(M) against M/a^{H} collapses only for the
q whose partition function has enough statistics (their eq. 32 and the 10 % area rule).

Pure numpy, the FFT on the policy: angles come in as an
:class:`~dynamix.core.orientation.Orientation`, so the same statistics run in the pixel frame
(the paper's A) or as bearings.
"""
from __future__ import annotations

import numpy as np

from dynamix.core.orientation import Orientation, axial_sector

__all__ = ["SECTOR_LABELS", "angle_pdf", "anisotropy_vs_scale", "component_pdfs", "cut_spectra",
           "directional_hurst", "gradient_plane", "modulus_collapse", "modulus_hurst", "modulus_pdfs",
           "sector_modulus_pdfs", "wtmmm_by_scale"]

#: Fig. 26's sectors (orientation mod 180°, ±22.5°), and their compass reading as bearings.
SECTOR_LABELS = {"pixel": ("0°", "45°", "90°", "135°"),
                 "azimuth": ("N–S", "NE–SW", "E–W", "SE–NW")}


def wtmmm_by_scale(extrema: list) -> list:
    """The WTMMM at every scale (xsmurf ``ssm``, the exact port in the WTMM backend): one
    ``{"x", "y", "mod", "arg", "line_id"}`` dict per scale."""
    from dynamix.core.wtmm_backend import PythonWTMMBackend

    return PythonWTMMBackend().single_maxima2d(list(extrema))


def angle_pdf(o: Orientation, bins: int = 36) -> tuple:
    """``(centres_deg, pdf)`` of the angles over one full turn, normalised so the pdf
    integrates to 1 over DEGREES -- a flat (isotropic) pdf reads ``1/360``. PIXEL angles are
    binned over (−180, 180]; azimuths over [0, 360)."""
    d = np.asarray(o.degrees, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    lo = -180.0 if o.frame.value == "pixel" else 0.0
    edges = np.linspace(lo, lo + 360.0, int(bins) + 1)
    counts, _ = np.histogram(np.mod(d - lo, 360.0) + lo, bins=edges)
    width = 360.0 / int(bins)
    pdf = counts / (max(d.size, 1) * width)
    return (edges[:-1] + edges[1:]) / 2.0, pdf


def gradient_plane(mod, arg_rad) -> tuple:
    """``(T_ψ1, T_ψ2) = (M cos A, M sin A)`` -- the wavelet gradient of each WTMMM (PIXEL A)."""
    m = np.asarray(mod, dtype=np.float64)
    a = np.asarray(arg_rad, dtype=np.float64)
    return m * np.cos(a), m * np.sin(a)


def sector_modulus_pdfs(mod, o: Orientation, *, n: int = 4, bins: int = 30,
                        log: bool = True) -> list:
    """``[(sector, centres, pdf, count)]``: the pdf of M (of log2 M when ``log``) within each
    axial sector of the angles, on SHARED bin edges so the curves compare directly. A sector
    holding nothing gives an all-zero pdf and ``count = 0``."""
    m = np.asarray(mod, dtype=np.float64).ravel()
    ok = np.isfinite(m) & (m > 0 if log else np.ones(m.shape, bool))
    val = np.log2(m[ok]) if log else m[ok]
    sec = axial_sector(Orientation(np.asarray(o.degrees).ravel()[ok], o.frame), n)
    if val.size == 0:
        return [(k, np.zeros(bins), np.zeros(bins), 0) for k in range(n)]
    edges = np.linspace(val.min(), val.max() if val.max() > val.min() else val.min() + 1.0,
                        int(bins) + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    out = []
    for k in range(int(n)):
        v = val[sec == k]
        counts, _ = np.histogram(v, bins=edges)
        pdf = counts / (max(v.size, 1) * (edges[1] - edges[0]))
        out.append((k, centres, pdf, int(v.size)))
    return out


# ------------------------------------------------- Khalil et al. 2006: the directional roughness

def _shared_edges(samples, bins: int, *, lo=None) -> np.ndarray:
    pooled = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in samples if np.size(v)])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return np.linspace(0.0, 1.0, int(bins) + 1)
    a, b = (float(pooled.min()) if lo is None else float(lo)), float(pooled.max())
    if b <= a:
        b = a + 1.0
    return np.linspace(a, b, int(bins) + 1)


def _ln_density(values, edges) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    with np.errstate(divide="ignore"):
        dens = counts / (max(int(counts.sum()), 1) * (edges[1] - edges[0]))
        return np.where(dens > 0, np.log(np.where(dens > 0, dens, 1.0)), np.nan)


def component_pdfs(components, scales, *, bins: int = 60, rescale=None) -> dict:
    """ln P_a(T) of the two WTMMM components at every scale on edges shared across the scales
    (Khalil et al. Figs. 12–13, 24): ``components[k] = (T_1, T_2)`` at scale ``scales[k]``.
    With ``rescale = (H_1, H_2)`` the values are T / a^{H} (their eqs. 35–36: one curve for a
    monofractal). ``{"centres": (2, bins), "ln_pdf": (2, n_scales, bins), "counts": (2, n_scales)}``."""
    a = np.asarray(scales, dtype=np.float64)
    out_c, out_p, out_n = [], [], []
    for i in range(2):
        vals = [np.asarray(c[i], dtype=np.float64) / (a[k] ** rescale[i] if rescale is not None else 1.0)
                for k, c in enumerate(components)]
        edges = _shared_edges(vals, bins)
        out_c.append((edges[:-1] + edges[1:]) / 2.0)
        out_p.append(np.stack([_ln_density(v, edges) for v in vals]))
        out_n.append(np.array([v.size for v in vals]))
    return {"centres": np.stack(out_c), "ln_pdf": np.stack(out_p), "counts": np.stack(out_n)}


def _ols(x, y) -> tuple:
    """Slope and its standard error of ``y`` on ``x``; NaN below three points."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    x, y = x[ok], y[ok]
    slope, intercept = np.polyfit(x, y, 1)
    res = y - (slope * x + intercept)
    xm = x - x.mean()
    se = np.sqrt((res @ res) / (x.size - 2) / float(xm @ xm))
    return float(slope), float(se)


def _collapse_h(values, a, grid) -> tuple:
    """The H minimising the spread across scales of the quantile curves of |T| / a^{H}, and the
    half-width of the interval where the spread stays within twice its minimum."""
    qs = np.linspace(0.1, 0.9, 17)
    spread = np.array([np.mean(np.std(np.log(np.array(
        [np.quantile(np.abs(v) / (s ** H), qs) for v, s in zip(values, a)]) + 1e-300), axis=0))
        for H in grid])
    i = int(np.argmin(spread))
    within = np.flatnonzero(spread <= 2.0 * spread[i])
    return float(grid[i]), float(0.5 * (grid[within.max()] - grid[within.min()]))


def directional_hurst(components, scales, log2_a_min: float, log2_a_max: float, *,
                      grid=None) -> dict:
    """The directional Hurst exponents of the two WTMMM components over the scales inside the
    window (Khalil et al. eqs. 33–36): by the slope against log2 a of the median of |T| and of
    the standard deviation of T (each with the slope's standard error), and by the collapse of
    the rescaled quantile curves (their criterion; σ the half-width of the near-minimum). Also
    which component carries the larger spread over the window (``dominant``: the one the
    modulus follows, their §5.2.4) and the scales used. Every estimate is NaN below three
    scales."""
    a = np.asarray(scales, dtype=np.float64)
    keep = [k for k in range(a.size) if log2_a_min - 1e-9 <= np.log2(a[k]) <= log2_a_max + 1e-9
            and np.size(components[k][0]) > 0]
    grid = np.arange(-0.5, 1.5001, 0.01) if grid is None else np.asarray(grid, dtype=np.float64)
    out = {"scales": a[keep], "median": [], "std": [], "collapse": [], "spread": []}
    for i in range(2):
        vals = [np.asarray(components[k][i], dtype=np.float64) for k in keep]
        x = np.log2(a[keep])
        med = _ols(x, [np.log2(np.median(np.abs(v))) if v.size else np.nan for v in vals])
        sd = _ols(x, [np.log2(np.std(v)) if v.size > 1 else np.nan for v in vals])
        col = _collapse_h(vals, a[keep], grid) if len(keep) >= 3 else (float("nan"), float("nan"))
        out["median"].append(med)
        out["std"].append(sd)
        out["collapse"].append(col)
        out["spread"].append(float(np.mean([np.median(np.abs(v)) for v in vals if v.size])) if vals else np.nan)
    out["dominant"] = int(np.nanargmax(out["spread"])) if np.isfinite(out["spread"]).any() else 0
    return out


def anisotropy_vs_scale(orientations, scales, *, bins: int = 36) -> dict:
    """Per scale, the direction of the highest peak of P_a(A) taken modulo 180° (a gradient and
    its opposite point the same way), given in (−90°, 90°] so the first axis reads 0 and does
    not flip between 0 and 180, and that peak's height relative to the isotropic level (1/180°
    on the folded pdf), Khalil et al. eq. 38's sharpening and their "angle of anisotropy":
    ``{"angle_deg", "relative_height", "counts"}``, NaN for an empty scale."""
    edges = np.linspace(0.0, 180.0, int(bins) + 1)
    angle, height, counts = [], [], []
    for o in orientations:
        d = np.asarray(o.degrees, dtype=np.float64).ravel()
        d = np.mod(d[np.isfinite(d)], 180.0)
        counts.append(int(d.size))
        if d.size == 0:
            angle.append(np.nan)
            height.append(np.nan)
            continue
        c, _ = np.histogram(d, bins=edges)
        pdf = c / (d.size * (180.0 / int(bins)))
        i = int(np.argmax(c))
        centre = (edges[i] + edges[i + 1]) / 2.0
        angle.append(float(centre - 180.0 if centre > 90.0 else centre))
        height.append(float(pdf[i] * 180.0))
    return {"scales": np.asarray(scales, dtype=np.float64), "angle_deg": np.array(angle),
            "relative_height": np.array(height), "counts": np.array(counts)}


def modulus_hurst(mods, scales, log2_a_min: float, log2_a_max: float) -> tuple:
    """The global H the modulus reads over the window: the slope of log2 of the median of M
    against log2 a, with its standard error (Khalil et al. eq. 37: M follows the dominant
    component over the scales at hand). NaN below three scales."""
    a = np.asarray(scales, dtype=np.float64)
    keep = [k for k in range(a.size) if log2_a_min - 1e-9 <= np.log2(a[k]) <= log2_a_max + 1e-9
            and np.size(mods[k]) > 0]
    return _ols(np.log2(a[keep]), [np.log2(np.median(np.asarray(mods[k], dtype=np.float64))) for k in keep])


def modulus_pdfs(mods, scales, *, bins: int = 60) -> dict:
    """P_a(M) per scale on shared edges from 0 (Khalil et al. Figs. 7a, c): ``{"centres",
    "pdf": (n_scales, bins), "counts"}``."""
    vals = [np.asarray(m, dtype=np.float64).ravel() for m in mods]
    edges = _shared_edges(vals, bins, lo=0.0)
    pdf = []
    for v in vals:
        c, _ = np.histogram(v[np.isfinite(v)], bins=edges)
        pdf.append(c / (max(int(c.sum()), 1) * (edges[1] - edges[0])))
    return {"scales": np.asarray(scales, dtype=np.float64), "centres": (edges[:-1] + edges[1:]) / 2.0,
            "pdf": np.stack(pdf), "counts": np.array([v.size for v in vals])}


def modulus_collapse(mods, scales, H: float, q_list, log2_a_min: float, log2_a_max: float, *,
                     bins: int = 60, tolerance: float = 0.10) -> dict:
    """a^{−qH} M^q P_a(M) against x = M / a^{H} (Khalil et al. eq. 32, Fig. 8): per q and
    scale the curve x^q P̃_a(x) with P̃_a the density of the rescaled modulus, on shared edges
    over the scales inside the window. Its area is the q-th moment of the rescaled modulus,
    the same at every scale when the modulus scales as a^{H}: the relative spread of the areas
    across the scales (their 10 % rule, ``tolerance``) says which q have the statistics their
    partition function needs. ``{"q", "x", "curves": (n_q, n_kept, bins), "area_spread": (n_q,),
    "usable": (n_q,) bool, "scales"}``."""
    a = np.asarray(scales, dtype=np.float64)
    q = np.asarray(q_list, dtype=np.float64)
    keep = [k for k in range(a.size) if log2_a_min - 1e-9 <= np.log2(a[k]) <= log2_a_max + 1e-9
            and np.size(mods[k]) > 0]
    vals = [np.asarray(mods[k], dtype=np.float64).ravel() / (a[k] ** float(H)) for k in keep]
    vals = [v[np.isfinite(v) & (v > 0)] for v in vals]
    edges = _shared_edges(vals, bins, lo=0.0)
    x = (edges[:-1] + edges[1:]) / 2.0
    dx = edges[1] - edges[0]
    curves = np.full((q.size, len(keep), x.size), np.nan)
    for k, v in enumerate(vals):
        c, _ = np.histogram(v, bins=edges)
        dens = c / (max(int(c.sum()), 1) * dx)
        with np.errstate(divide="ignore", invalid="ignore"):
            curves[:, k, :] = np.where(x > 0, x[None, :] ** q[:, None], 0.0) * dens[None, :]
    area = np.nansum(curves, axis=2) * dx                      # (n_q, n_kept)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = ((area.max(axis=1) - area.min(axis=1)) / np.where(area.mean(axis=1) > 0, area.mean(axis=1), np.nan)
                  if len(keep) else np.full(q.size, np.nan))
    return {"q": q, "x": x, "curves": curves, "area_spread": spread,
            "usable": np.isfinite(spread) & (spread <= float(tolerance)), "scales": a[keep]}


def cut_spectra(values, *, low_modes: int = 2, top_fraction: float = 0.25) -> dict:
    """The directional power spectra of a field's one-dimensional cuts (Khalil et al. Fig. 9):
    the Hann-windowed power of every row averaged (the x-cuts, against k_x) and of every column
    (the y-cuts, against k_y), each with β = −slope of log2 S against log2 k over the modes from
    ``low_modes`` up to (1 − ``top_fraction``) of Nyquist (the standard error beside it), the
    radial spectrum S(|k|) with its β likewise, and the field's log2 power on the shifted 2-D
    grid for a contour. On a fractional Brownian sheet β_x = 2H_x + 1, β_y = 2H_y + 1 and the
    radial β = 2 min(H_x, H_y) + 2 (their eqs. 17–18, 31). k in cycles per pixel."""
    from dynamix.core.daf import radial_spectrum
    from dynamix.core.fft_policy import double

    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    v = np.where(finite, v, v[finite].mean() if finite.any() else 0.0) - (v[finite].mean() if finite.any() else 0.0)
    fft = double()

    def cuts(arr):
        n = arr.shape[1]
        win = np.hanning(n)[None, :]
        power = np.abs(np.asarray(fft.rfft(arr * win, axis=1))) ** 2
        power = power.mean(axis=0) / (win ** 2).sum()
        k = np.fft.rfftfreq(n)
        lo, hi = int(low_modes), int(np.floor((1.0 - float(top_fraction)) * (k.size - 1)))
        sel = slice(lo, max(hi, lo + 3))
        beta, se = _ols(np.log2(k[sel]), np.log2(power[sel]))
        return k[1:], power[1:], -beta, se, (float(k[sel][0]), float(k[sel][-1]))

    kx, sx, bx, sex, win_x = cuts(v)
    ky, sy, by, sey, win_y = cuts(v.T)
    kr, sr, _frac = radial_spectrum(v)
    pos = kr > 0
    kr, sr = kr[pos], sr[pos]
    # The radial fit stays inside the inscribed circle: beyond k = ½ an annulus samples the
    # corners alone, the diagonal directions only.
    inner = np.flatnonzero(kr <= 0.5 * (1.0 - float(top_fraction)))
    lo_r = min(int(low_modes), max(inner.size - 3, 0))
    sel_r = inner[lo_r:] if inner.size - lo_r >= 3 else inner
    br, ser = _ols(np.log2(kr[sel_r]), np.log2(sr[sel_r]))
    with np.errstate(divide="ignore"):
        log2_power = np.log2(np.fft.fftshift(np.abs(np.asarray(fft.fft2(v))) ** 2) + 1e-300)
    return {"k_x": kx, "S_x": sx, "beta_x": bx, "beta_x_se": sex, "window_x": win_x,
            "k_y": ky, "S_y": sy, "beta_y": by, "beta_y_se": sey, "window_y": win_y,
            "k_r": kr, "S_r": sr, "beta_r": -br, "beta_r_se": ser,
            "log2_power": log2_power}
