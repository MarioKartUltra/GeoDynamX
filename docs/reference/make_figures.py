# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Kernel figures for devices_reference.tex -- drawn from the app's OWN kernel code.

    python docs/reference/make_figures.py        # writes docs/reference/figures/*.pdf

Every curve is sampled from the function the analysis actually runs (``microcanonical``'s
kernel builders, ``scale_units.kernel_section`` for the WTMM pair), except the fractional
Mallat-Zhong pair, which is the analytic limit its filter cascade is pinned to in
tests/test_mz_frac.py. So a figure can never drift from what a tool computes. Profiles are
radial sections in units of the scale r (the zero-crossing radius for the Mexican hats, the
kernel's own width parameter for the positive kernels), scaled to 1 at the centre for
comparison -- the app itself normalizes every kernel to unit L1 mass at every scale.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from dynamix.core import microcanonical as mc  # noqa: E402
from dynamix.core.scale_units import kernel_section, sigma_px  # noqa: E402

OUT = HERE / "figures"
R = 200.0                                   # samples per unit radius: smooth curves
X = np.linspace(-3.0, 3.0, 2401)            # rho / r


def _profile_positive(wavelet, **kw):
    """A positive (measure-route) kernel's radial section, peak 1."""
    r = 1.0 / (X[1] - X[0])                 # the kernel's r, in samples: rho/r == X
    k = np.fft.fftshift(mc._radial_kernel((1, X.size), r, wavelet, kw.get("beta", 1.0),
                                          kw.get("q", 1.5), kw.get("frac_n", 2.0)))[0]
    return k / k.max()


def _profile_marr(wavelet, q=1.0, frac_n=2.0):
    """A multiaffine-route Mexican hat's radial section (zero crossing at rho = r), 1 at 0."""
    rho2 = (X * R) ** 2
    u0 = mc._frac_u0(frac_n) if wavelet == "frac_gaussian" else None
    k = mc._marr_kernel(rho2, R, wavelet, 1.0, q, frac_n, u0)
    return k / k[X.size // 2]


def _family(ax, values, curve, cmap, label, ylim=None):
    cm = plt.get_cmap(cmap)
    lo, hi = min(values), max(values)
    for v in values:
        ax.plot(X, curve(v), color=cm((v - lo) / (hi - lo)), lw=1.1)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(lo, hi))
    ax.figure.colorbar(sm, ax=ax, label=label, pad=0.02)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$\rho / r$")
    if ylim:
        ax.set_ylim(*ylim)


def fig_q_family():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    qs = np.round(np.arange(-1.0, 3.01, 0.25), 2)
    _family(a, qs, lambda q: _profile_positive("q_gaussian", q=q), "rainbow", "$q_T$")
    a.set_title(r"holder_measure: positive $q$-Gaussian, $-1 \leq q_T \leq 3$")
    a.set_ylabel("kernel (peak = 1)")
    qb = [q for q in np.round(np.arange(-1.0, 1.96, 0.25), 2)] + [1.95]
    _family(b, qb, lambda q: _profile_marr("q_gaussian", q=mc._borges_q(q)), "rainbow",
            "$q_T$ (Borges et al. 2004)", ylim=(-1.2, 1.6))
    b.set_title(r"holder_multiaffine: $q$-Mexican hat, $-1 \leq q_T < 2$")
    b.set_ylabel(r"$\psi_q(\rho)\,/\,\psi_q(0)$")
    fig.savefig(OUT / "q_family.pdf")


def fig_frac_gaussian():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    ns = np.round(np.arange(0.5, 6.01, 0.5), 2)
    _family(a, ns, lambda n: _profile_positive("frac_gaussian", frac_n=n), "rainbow", "$n$")
    a.set_title(r"holder_measure: envelope $e^{-\rho^n/2}$")
    a.set_ylabel("kernel (peak = 1)")
    _family(b, ns, lambda n: _profile_marr("frac_gaussian", frac_n=n), "rainbow", "$n$",
            ylim=(-1.0, 1.1))
    b.set_title(r"holder_multiaffine: ${}_1F_1(\frac{n+2}{2};1;-\rho^2/2\sigma^2)$")
    b.set_ylabel(r"$\psi_n(\rho)\,/\,\psi_n(0)$")
    fig.savefig(OUT / "frac_gaussian.pdf")


def _mz_frac_pair(alpha, x):
    """theta_alpha and psi_alpha = theta' in real space: inverse FT of |sinc(w/4)|^(alpha+1)
    and i w |sinc(w/4)|^(alpha+1) (the cascade's pinned limit). theta is even and psi odd, so
    theta = (1/pi) int_0^inf th cos(wx) dw and psi = -(1/pi) int_0^inf w th sin(wx) dw."""
    w = np.linspace(0.0, 400.0, 2 ** 15)
    th = np.abs(np.sinc(w / 4.0 / np.pi)) ** (alpha + 1.0)
    th = th * np.sinc(w / w[-1])            # Lanczos sigma factor: no Gibbs ringing at alpha=1
    dw = w[1] - w[0]
    wx = np.outer(x, w)
    theta = np.cos(wx) @ th * dw / np.pi
    psi = -(np.sin(wx) @ (w * th)) * dw / np.pi
    return theta, psi


def _mz_frac_pair_exact(alpha, x):
    """theta_alpha(x) = 2 beta_*(2x) and psi_alpha = theta' = 4 beta_*'(2x), from the exact
    time-domain series (dynamix.core.frac_bspline_exact; Unser & Blu 1999 eq. 11) -- no Fourier
    inversion, so the alpha = 1 box pair has clean jumps (the truncated inverse transform of
    ``_mz_frac_pair`` rang there), and the small tail lobes of non-odd orders are the real
    ones."""
    from dynamix.core.frac_bspline_exact import beta_star, beta_star_derivative

    return 2.0 * beta_star(2.0 * x, alpha), 4.0 * beta_star_derivative(2.0 * x, alpha)


def fig_frac_bspline():
    x = np.linspace(-2.5, 2.5, 801)          # compact only at odd integer alpha; tails small here
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    pairs = {a_: _mz_frac_pair_exact(a_, x) for a_ in alphas}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    cm = plt.get_cmap("rainbow")
    for al in alphas:
        c = cm((al - alphas[0]) / (alphas[-1] - alphas[0]))
        th, ps = pairs[al]
        a.plot(x, th / th.max(), color=c, lw=1.1)
        b.plot(x, ps / np.abs(ps).max(), color=c, lw=1.1)
    for ax, title in ((a, r"smoothing $\theta_\alpha$ (fractional B-spline)"),
                      (b, r"wavelet $\psi_\alpha = \theta_\alpha'$ (mz_edges, frac_bspline)")):
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(alphas[0], alphas[-1]))
        fig.colorbar(sm, ax=ax, label=r"$\alpha$", pad=0.02)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("$x$")
    a.set_ylabel("peak = 1")
    fig.savefig(OUT / "frac_bspline.pdf")


def fig_wtmm_pair():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for ax, smoothing, (n0, n1) in zip(axes, ("gaussian", "mexican"),
                                       (("g0", "g1"), ("g2", "g3"))):
        for order, name, style in ((0, n0, "-"), (1, n1, "--")):
            x, y = kernel_section(1.0, smoothing, order, n_points=801)
            ax.plot(x / sigma_px(1.0), y / np.abs(y).max(), style, lw=1.4,
                    label=(rf"smoothing $\theta$ = {name}" if order == 0
                           else rf"analyzing $\psi = \partial\theta$ = {name}"))
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_title(f"wtmm2d, wavelet = {smoothing}")
        ax.set_xlabel(r"$x / \sigma$ (section through the 2-D kernel)")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("max |.| = 1")
    fig.savefig(OUT / "wtmm_pair.pdf")


def main():
    OUT.mkdir(exist_ok=True)
    for f in (fig_q_family, fig_frac_gaussian, fig_frac_bspline, fig_wtmm_pair):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
