# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The wavelet atlas of devices_reference.tex -- every analyzing wavelet and smoothing function the
app runs, drawn from its own kernel code.

    python docs/reference/make_wavelet_figures.py   # writes docs/reference/figures/atlas_*.pdf

Each figure samples the functions a device computes with: ``core.smoothers`` (the WTMM smoothers,
their k-space forms and the derivative filters the transform builds), ``core.wavelet_profiles``
(the microcanonical kernels, the dyadic and edge-detector banks), ``core.hyperbolic_preview`` (the
DWT banks' cascades), ``core.interpolating``, ``core.daf``, ``core.analytic_cwt`` (the 2-D
q-Morlet) and ``core.wavelet_skeleton``. Sections are scaled for comparison (the app itself
normalizes every kernel to unit L1 mass at every scale). Figures A1--A8 are the appendix of the
tuneable-wavelets note of 14 October 2026, redrawn from the same code.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from dynamix.core import fft_policy  # noqa: E402
from dynamix.core import smoothers as sm  # noqa: E402
from dynamix.core import wavelet_profiles as wp  # noqa: E402

OUT = HERE / "figures"
LW = 0.6
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 5.5, "figure.titlesize": 8})


# -- shared drawing -----------------------------------------------------------------------------

def _norm(y):
    y = np.asarray(y, dtype=float)
    m = np.abs(y).max()
    return y / m if m > 0 else y


def _ramp(values, cmap, log=False):
    return plt.get_cmap(cmap), (LogNorm if log else Normalize)(vmin=min(values), vmax=max(values))


def _colorbar(fig, ax, cm, nm, label):
    cb = fig.colorbar(ScalarMappable(norm=nm, cmap=cm), ax=ax, pad=0.01, fraction=0.05)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=5)


def _sweep(ax, values, curve, cmap, label, log=False, ref=None):
    """Hairlines colored along the ramp, ``curve(v) -> (x, y)``, with the parameter's color bar;
    ``ref`` is drawn dashed (the title names it)."""
    cm, nm = _ramp(values, cmap, log)
    for v in values:
        ax.plot(*curve(v), color=cm(nm(v)), lw=LW)
    if ref is not None:
        ax.plot(*ref, "k--", lw=0.8)
    _colorbar(ax.figure, ax, cm, nm, label)
    ax.axhline(0.0, color="0.85", lw=0.5)


def _waterfall(ax, values, curve, cmap, label, step=0.35, log=False):
    """The sections stacked up the parameter axis, each normalized, colored along the ramp."""
    cm, nm = _ramp(values, cmap, log)
    for i, v in enumerate(values):
        x, y = curve(v)
        y = 0.5 * _norm(y) + i * step
        ax.plot(x, y, color=cm(nm(v)), lw=LW)
        ax.fill_between(x, i * step, y, color=cm(nm(v)), alpha=0.08, lw=0)
    ax.set_yticks([0.0, (len(values) - 1) * step])
    ax.set_yticklabels([f"{label} = {min(values):g}", f"{label} = {max(values):g}"])
    ax.spines[["top", "right"]].set_visible(False)


def _legend_below(ax, ncol=2, **kw):
    """The panel's legend under its x axis, clear of the curves."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, frameon=False, **kw)


def _surface(ax, x, y, z, title, lines: int = 34):
    """A wireframe-shaded surface in the manner of Arneodo, Decoster and Roux (2000), Fig. 1."""
    gx, gy = np.meshgrid(x, y)
    rs, cs = max(1, y.size // lines), max(1, x.size // lines)
    ax.plot_surface(gx, gy, z, rstride=rs, cstride=cs, color="0.93", edgecolor="0.1",
                    linewidth=0.2, shade=True, antialiased=True, rasterized=True)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect((1.0, float(np.ptp(y) / np.ptp(x)), 0.5), zoom=1.3)
    ax.view_init(elev=30, azim=-58)
    ax.set_axis_off()
    ax.set_title(title, fontsize=6.5, pad=-6)


_SYMBOL = {"alpha": "α", "beta": "β", "q": "q"}


def _setting(params: dict) -> str:
    return ", ".join(f"{_SYMBOL.get(k, k)} = {v:g}" for k, v in params.items() if not isinstance(v, str))


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=200)
    plt.close(fig)


# -- 2-D kernels from the app's own filters -----------------------------------------------------

def _kernel_2d(smoother, order: "str | None", span: float, du: float = 1.0 / 16.0):
    """``(u, field)``: θ (``order`` None) or the derivative kernel (i s)^order θ̂ of ``smoother`` in
    real space on |u| ≤ ``span`` σ, from θ̂ and the derivative filters the WTMM transform builds,
    inverted on a grid of step ``du`` σ; max |field| = 1."""
    n = int(2 ** np.ceil(np.log2(4.0 * span / du)))
    kappa = np.fft.fftfreq(n, d=du)                              # cycles per σ
    kx, ky = np.meshgrid(kappa, kappa)
    sx, sy = np.pi * np.sqrt(2.0) * kx, np.pi * np.sqrt(2.0) * ky    # s = k a = κ (a/σ)
    theta_hat = smoother.theta_hat(sx, sy)
    filt = (theta_hat if order is None else
            sm.derivative_filters(sx, sy, theta_hat, (order,), dtype=np.float64)[order])
    field = np.fft.fftshift(np.real(fft_policy.double().ifft2(filt)))
    u = (np.arange(n) - n // 2) * du
    keep = np.abs(u) <= span
    field = field[np.ix_(keep, keep)]
    return u[keep], field / np.abs(field).max()


def _radial(profile, span_px, n=121):
    """A radial kernel's surface from its section (the microcanonical kernels are radial)."""
    pos = profile.x_px >= 0
    x, y = profile.x_px[pos], profile.psi[pos]
    u = np.linspace(-span_px, span_px, n)
    gx, gy = np.meshgrid(u, u)
    return u, np.interp(np.hypot(gx, gy), x, y, right=y[-1])


# -- the 3-D atlas --------------------------------------------------------------------------------

#: The smoothers' names as the WTMM menu shows them.
_SMOOTHER_NAMES = {"gaussian": "Gaussian", "mexican": "Mexican hat", "frac_bspline": "polyharmonic B-spline",
                   "q_gaussian": "q-Gaussian", "q_mexican": "q-Mexican hat", "q_meyer": "q-Meyer"}


def _polyharmonic_2d(alpha, order, span, n=129):
    """θ, ∂θ/∂x or ∂θ/∂y of the isotropic polyharmonic B-spline on |u| ≤ ``span`` σ, evaluated in
    the space domain (``dynamix.core.polyharmonic_exact``); max |field| = 1."""
    from dynamix.core.polyharmonic_exact import evaluate

    gamma = alpha + 1.0
    h = np.sqrt(12.0 / gamma)                     # the knot spacing in σ
    u = np.linspace(-span, span, n)
    gx, gy = np.meshgrid(u, u)
    if order == "dy":
        field = evaluate(gy / h, gx / h, gamma, 1)
    else:
        field = evaluate(gx / h, gy / h, gamma, 0 if order is None else 1)
    return u, field / np.abs(field).max()


def fig_wtmm_3d():
    """θ, ψ₁ = ∂θ/∂x and ψ₂ = ∂θ/∂y of the six WTMM smoothers at their default settings."""
    cases = [("gaussian", 4.0), ("mexican", 4.5), ("frac_bspline", 4.0),
             ("q_gaussian", 5.0), ("q_mexican", 4.5), ("q_meyer", 9.0)]
    parts = ((None, r"\theta"), ("dx", r"\psi_1=\partial\theta/\partial x"),
             ("dy", r"\psi_2=\partial\theta/\partial y"))
    fig = plt.figure(figsize=(6.5, 9.0))
    for i, (name, span) in enumerate(cases):
        s = sm.smoother(name)
        label = _SMOOTHER_NAMES[name]
        head = f"{label} ({_setting(s.params)})" if _setting(s.params) else label
        for j, (order, comp) in enumerate(parts):
            ax = fig.add_subplot(len(cases), 3, 3 * i + j + 1, projection="3d")
            u, z = (_polyharmonic_2d(s.params["alpha"], order, span) if name == "frac_bspline"
                    else _kernel_2d(s, order, span))
            _surface(ax, u, u, z, f"{head}: ${comp}$")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.98, wspace=0.0, hspace=0.08)
    _save(fig, "atlas_wtmm_3d")


def fig_microcanonical():
    """The measure route's positive kernels and the multiaffine route's wavelets at r = 8 px."""
    r = 8.0
    measure = ("gaussian", "q_gaussian", "lorentzian", "frac_gaussian")
    multiaffine = ("g1", "g2", "g3", "q_mexican", "lorentzian_marr", "frac_gaussian")
    kw = {"beta": 2.0, "q_tsallis": 1.5, "frac_n": 2.0}
    fig = plt.figure(figsize=(6.5, 8.6))
    gs = fig.add_gridspec(4, 12, height_ratios=(1.15, 1.0, 1.0, 1.0))
    for c, (names, prof, title) in enumerate(((measure, wp.measure_profile, "measure route: positive kernels"),
                                              (multiaffine, wp.multiaffine_profile,
                                               "multiaffine route: wavelets"))):
        ax = fig.add_subplot(gs[0, 6 * c:6 * c + 6])
        for k, name in enumerate(names):
            p = prof(name, r, **kw)
            ax.plot(p.x_px / r, p.psi, color=plt.get_cmap("tab10")(k), lw=0.9, label=name)
        ax.axhline(0.0, color="0.85", lw=0.5)
        ax.set_xlim(-4.0, 4.0)
        ax.set_xlabel(r"$\rho / r$")
        ax.set_title(title)
        ax.legend(loc="upper right")
    for k, name in enumerate(measure):
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3], projection="3d")
        _surface(ax, *_radial3(wp.measure_profile(name, r, **kw), 3.5 * r), f"measure: {name}")
    for k, name in enumerate(multiaffine):
        ax = fig.add_subplot(gs[2 + k // 3, 4 * (k % 3):4 * (k % 3) + 4], projection="3d")
        _surface(ax, *_radial3(wp.multiaffine_profile(name, r, **kw), 3.0 * r), f"multiaffine: {name}")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.01, top=0.98, wspace=0.6, hspace=0.25)
    _save(fig, "atlas_microcanonical")


def _radial3(profile, span_px):
    u, z = _radial(profile, span_px)
    return u, u, z


def _axis_atom(bank, index, depth):
    from dynamix.core.hyperbolic_preview import axis_preview

    p = axis_preview(bank, index, depth)
    return p["x"], p["profile"]


def fig_dwt_banks():
    """φ and ψ of every DWT bank at level 5: the cascades the pyramid applies."""
    from dynamix.core.dwt import BANKS

    cols = 4
    rows = int(np.ceil(len(BANKS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5, 1.45 * rows), constrained_layout=True)
    for ax, bank in zip(axes.ravel(), BANKS):
        x0, phi = _axis_atom(bank, 0, 5)
        x1, psi = _axis_atom(bank, 5, 5)
        ax.plot(x0, phi, color="0.55", lw=0.8, label=r"$\varphi$")
        ax.plot(x1, psi, color="k", lw=0.8, label=r"$\psi$")
        ax.axhline(0.0, color="0.85", lw=0.5)
        ax.set_title(bank)
        ax.set_xlim(-60, 60)
    for ax in axes.ravel()[len(BANKS):]:
        ax.set_axis_off()
    axes[0, 0].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("x (px, level 5)")
    _save(fig, "atlas_dwt_banks")


def fig_dwt_3d():
    """The separable 2-D atoms of sym4 at level 3 and two hyperbolic atoms ψ_{j₁}(x) ψ_{j₂}(y)."""
    bank = "sym4"
    phi3, psi3 = _axis_atom(bank, 0, 3), _axis_atom(bank, 3, 3)
    psi2, psi4 = _axis_atom(bank, 2, 4), _axis_atom(bank, 4, 4)
    atoms = [(phi3, phi3, r"$\varphi_3(x)\,\varphi_3(y)$"), (psi3, phi3, r"$\psi_3(x)\,\varphi_3(y)$"),
             (phi3, psi3, r"$\varphi_3(x)\,\psi_3(y)$"), (psi3, psi3, r"$\psi_3(x)\,\psi_3(y)$"),
             (psi2, psi4, r"hyperbolic $\psi_2(x)\,\psi_4(y)$"), (psi4, psi2, r"hyperbolic $\psi_4(x)\,\psi_2(y)$")]
    fig = plt.figure(figsize=(6.5, 4.6))
    extent = lambda x, f: float(np.abs(x[np.abs(f) > 0.02]).max())    # noqa: E731
    for k, ((xx, fx), (yy, fy), title) in enumerate(atoms):
        u = np.linspace(-1.0, 1.0, 121) * max(extent(xx, fx), extent(yy, fy))
        z = np.outer(np.interp(u, yy, fy, left=0.0, right=0.0), np.interp(u, xx, fx, left=0.0, right=0.0))
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        _surface(ax, u, u, z / np.abs(z).max(), f"{bank}: {title}")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.96, wspace=0.0, hspace=0.1)
    _save(fig, "atlas_dwt_3d")


def _lifting_atom(band, levels=4, n=256, **bank):
    """The interpolating transform's atom: one coefficient at the center of ``band`` (``low``,
    ``horizontal`` = ψ(x)φ(y), ``vertical`` = φ(x)ψ(y), ``diagonal``) at the coarsest level,
    reconstructed by the device's own inverse."""
    from dynamix.core import interpolating as it

    c = it.pyramid(np.zeros((n, n)), levels, **bank)
    target = c["low"] if band == "low" else c["bands"][levels - 1][("horizontal", "vertical", "diagonal").index(band)]
    target[target.shape[0] // 2, target.shape[1] // 2] = 1.0
    z = it.inverse(c, keep=(levels, band))
    iy, ix = np.unravel_index(np.argmax(np.abs(z)), z.shape)
    x = np.arange(n) - ix
    return x, z[iy], z


def fig_interpolating():
    """DD4 and GLDAF lifting atoms (the interpolating-wavelet tools) and the Hermite DAF."""
    from dynamix.core.daf import daf_gain, hermite_daf

    banks = [("DD4", {"bank": "dd4"}), ("GLDAF M = 3, σ = 1", {"bank": "gldaf", "order": 3, "sigma": 1.0})]
    fig = plt.figure(figsize=(6.5, 7.4))
    gs = fig.add_gridspec(3, 12, height_ratios=(1.0, 1.15, 1.15))
    a_phi, a_psi = fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])
    for k, (label, bank) in enumerate(banks):
        x, row, _ = _lifting_atom("low", **bank)
        a_phi.plot(x / 16.0, _norm(row), lw=0.9, color=plt.get_cmap("tab10")(k), label=label)
        x, row, _ = _lifting_atom("horizontal", **bank)
        a_psi.plot(x / 16.0, _norm(row), lw=0.9, color=plt.get_cmap("tab10")(k), label=label)
    for ax, title in ((a_phi, r"scaling function $\varphi$ (level 4)"), (a_psi, r"wavelet $\psi$ (level 4)")):
        ax.set_xlim(-6, 6)
        ax.axhline(0.0, color="0.85", lw=0.5)
        ax.set_xlabel("x (coarse cells)")
        ax.set_title(title)
    _legend_below(a_phi, ncol=1, fontsize=5)
    a_taps, a_gain = fig.add_subplot(gs[0, 6:9]), fig.add_subplot(gs[0, 9:12])
    f = np.linspace(0.0, 0.5, 501)
    for k, M in enumerate((0, 4, 12, 24)):
        taps = hermite_daf(3.0, M)
        j = np.arange(taps.size) - taps.size // 2
        color = plt.get_cmap("viridis")(k / 3)
        a_taps.plot(j, taps / taps.max(), lw=0.9, color=color, label=f"M = {M}")
        a_gain.plot(f, daf_gain(2 * np.pi * f, 3.0, M), lw=0.9, color=color, label=f"M = {M}")
    a_taps.set_xlim(-12, 12)
    a_taps.axhline(0.0, color="0.85", lw=0.5)
    a_taps.set_xlabel("j (px)")
    a_taps.set_title(r"Hermite DAF $\delta_M$, $\sigma = 3$ px")
    _legend_below(a_taps, ncol=2, fontsize=5)
    a_gain.set_xlabel("frequency (cycles/px)")
    a_gain.set_title("DAF low-pass gain")
    for r, (label, bank) in enumerate(banks):
        for c, band in enumerate(("low", "horizontal", "diagonal")):
            x, _row, z = _lifting_atom(band, **bank)
            center = np.unravel_index(np.argmax(np.abs(z)), z.shape)
            keep = slice(center[0] - 64, center[0] + 64), slice(center[1] - 64, center[1] + 64)
            u = np.arange(-64, 64) / 16.0
            ax = fig.add_subplot(gs[1 + r, 4 * c:4 * c + 4], projection="3d")
            factor = {"low": r"$\varphi(x)\varphi(y)$", "horizontal": r"$\psi(x)\varphi(y)$",
                      "diagonal": r"$\psi(x)\psi(y)$"}[band]
            _surface(ax, u, u, _norm(z[keep]), f"{label}: {factor}")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.0, top=0.97, wspace=1.2, hspace=0.3)
    _save(fig, "atlas_interpolating")


def fig_qmorlet_3d():
    """The 2-D analytic q-Morlet of the wavelet ridge: Re ψ, Im ψ and |ψ| for three q."""
    from dynamix.core import analytic_cwt as ac

    n, half = 96, 24
    kx, ky = ac.grid((n, n))
    kappa = (2.0 * np.pi / 12.0, 0.0)
    fig = plt.figure(figsize=(6.5, 6.6))
    u = np.arange(-half, half)
    keep = slice(n // 2 - half, n // 2 + half)
    for i, q in enumerate((0.7, 1.0, 1.5)):
        hat = ac.wavelet_hat(kx, ky, kappa, q=q, zero_mean=True)
        psi = np.fft.fftshift(fft_policy.double().ifft2(hat))[keep, keep]
        psi = psi / np.abs(psi).max()
        for j, (part, label) in enumerate(((psi.real, r"Re $\psi_q$"), (psi.imag, r"Im $\psi_q$"),
                                           (np.abs(psi), r"$|\psi_q|$"))):
            ax = fig.add_subplot(3, 3, 3 * i + j + 1, projection="3d")
            _surface(ax, u, u, part, f"q = {q:g}: {label}", lines=40)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.97, wspace=0.0, hspace=0.1)
    _save(fig, "atlas_qmorlet_3d")


def fig_edges():
    """The edge detectors' kernels: CDF's equivalent kernels and the skeleton's smoothers."""
    from dynamix.core.wavelet_skeleton import theta_kernel

    kernels = ("tang_you", "gaussian", "bspline2")
    fig = plt.figure(figsize=(6.5, 6.4))
    gs = fig.add_gridspec(3, 6, height_ratios=(1.0, 1.1, 1.1))
    a = fig.add_subplot(gs[0, 0:3])
    channels = (("mz", "-", "Re u, Canny (mz)"), ("marr", "--", "Im v, Marr (marr)"))
    for k, (mode, ls, part) in enumerate(channels):
        for c, sigma in enumerate((2.0, 4.0, 8.0)):
            p = wp.cdf_profile(mode, sigma)
            a.plot(p.x_px, p.psi, ls, lw=0.9, color=plt.get_cmap("viridis")(c / 2),
                   label=f"{part}, σ = {sigma:g}")
    a.set_xlim(-30, 30)
    a.set_title("cdf_edges: ∇G_σ behind Re u, θtΔG_σ behind Im v")
    a.set_xlabel("x (px)")
    _legend_below(a, ncol=2, fontsize=5)
    b = fig.add_subplot(gs[0, 3:6])
    for k, kernel in enumerate(kernels):
        p = wp.tang_you_profile(kernel, 8.0)
        b.plot(p.x_px, p.psi, lw=0.9, color=plt.get_cmap("tab10")(k), label=kernel)
    b.set_xlim(-14, 14)
    b.set_title(r"wavelet_skeleton: $\partial\theta/\partial x$ at s = 8 px")
    b.set_xlabel("x (px)")
    _legend_below(b, ncol=3, fontsize=5)
    for ax in (a, b):
        ax.axhline(0.0, color="0.85", lw=0.5)
    n, s = 64, 20.0
    u = np.arange(n) - n // 2
    for k, kernel in enumerate(kernels):
        theta = np.fft.fftshift(theta_kernel((n, n), s, kernel))
        for r, (z, label) in enumerate(((theta, r"$\theta$"), (np.gradient(theta, axis=1), r"$\partial\theta/\partial x$"))):
            ax = fig.add_subplot(gs[1 + r, 2 * k:2 * k + 2], projection="3d")
            _surface(ax, u, u, _norm(z), f"{kernel}: {label}")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.0, top=0.97, wspace=0.5, hspace=0.25)
    _save(fig, "atlas_edges")


# -- A1--A8: the appendix of the tuneable-wavelets note ------------------------------------------

def _smoother_page(fams, name):
    """A1's columns: θ, ψ = ∂θ and θ̂ swept over the parameter, and ψ stacked up it, at a = 8 px."""
    a = 8.0
    sigma = a * sm._SIGMA_PER_SCALE
    s_grid = np.linspace(0.0, 3.0, 600)
    zero = np.zeros_like(s_grid)
    fig, ax = plt.subplots(4, len(fams), figsize=(6.5, 8.4), constrained_layout=True)
    for j, (title, smoother, key, lab, vals, refname, cmap, span) in enumerate(fams):
        ref = sm.smoother(refname)

        def sec(v, order, smoother=smoother, key=key):
            x, y = sm.smoother(smoother, {key: v}).section(a, order)
            return x / sigma, _norm(y)

        def ref_sec(order, ref=ref):
            x, y = ref.section(a, order)
            return x / sigma, _norm(y)

        _sweep(ax[0, j], vals, lambda v, sec=sec: sec(v, 0), cmap, lab, ref=ref_sec(0))
        _sweep(ax[1, j], vals, lambda v, sec=sec: sec(v, 1), cmap, lab, ref=ref_sec(1))
        _sweep(ax[2, j], vals,
               lambda v, smoother=smoother, key=key: (s_grid, sm.smoother(smoother, {key: v}).theta_hat(s_grid, zero)),
               cmap, lab, ref=(s_grid, ref.theta_hat(s_grid, zero)))
        _waterfall(ax[3, j], vals, lambda v, sec=sec: sec(v, 1), cmap, lab)
        ax[0, j].set_title(f"{title}: {lab} from {vals[0]:g} to {vals[-1]:g}\n(dashed: the {refname} pair)")
        for r in (0, 1, 3):
            ax[r, j].set_xlim(-span, span)
        ax[3, j].set_xlabel("x/σ")
        ax[2, j].set_xlabel("s = a·k")
    for r, yl in enumerate(("θ (x/σ)", "ψ = ∂θ (x/σ)", "θ̂(s)", "ψ up the parameter")):
        ax[r, 0].set_ylabel(yl)
    _save(fig, name)


def fig_a1_smoothers():
    """A1. The q-Gaussian and q-Mexican hat smoothers morphing with q at a = 8 px."""
    _smoother_page([("q-Gaussian (σ matched)", "q_gaussian", "q", "q", np.linspace(0.3, 1.95, 26), "gaussian",
                     "rainbow", 6.0),
                    ("q-Mexican hat", "q_mexican", "q", "q", np.linspace(0.7, 1.95, 26), "mexican", "brg", 6.0)],
                   "atlas_a1_smoothers")


def fig_a1b_smoothers():
    """A1, continued. The q-Meyer and the isotropic polyharmonic B-spline morphing with their parameter."""
    _smoother_page([("q-Meyer", "q_meyer", "q", "q", np.linspace(0.1, 1.95, 26), "gaussian", "rainbow", 12.0),
                    ("isotropic polyharmonic B-spline", "frac_bspline", "alpha", "α", np.linspace(1.0, 6.0, 26), "gaussian",
                     "brg", 6.0)],
                   "atlas_a1b_smoothers")


def fig_a2_turiel():
    """A2. Turiel's kernels at r = 8 px under three sweeps."""
    r = 8.0
    qs = np.linspace(0.3, 1.95, 26)
    betas = np.geomspace(0.1, 4.0, 26)
    routes = [("measure route, q-Gaussian", wp.measure_profile, "q_gaussian"),
              ("multiaffine route, q-Mexican hat", wp.multiaffine_profile, "q_mexican")]
    fig, ax = plt.subplots(4, 3, figsize=(6.5, 8.4), constrained_layout=True)
    for i, (route, prof, qname) in enumerate(routes):
        def P(prof=prof, qname=qname, **kw):
            p = prof(qname, r, **kw)
            return p.x_px / r, p.psi

        sweeps = [("q varies, β fixed at ½", qs, lambda q, P=P: P(q_tsallis=q, q_pairing="fixed", q_beta=0.5),
                   "q", "rainbow", False),
                  ("β varies, q = 1.5 (a dilation)", betas,
                   lambda b, P=P: P(q_tsallis=1.5, q_pairing="fixed", q_beta=b), "β", "brg", True),
                  ("q varies, β(q) = 1/(2(2−q))", qs, lambda q, P=P: P(q_tsallis=q, q_pairing="q-paired", q_beta=0.5),
                   "q", "rainbow", False)]
        for j, (title, vals, curve, lab, cmap, log) in enumerate(sweeps):
            _sweep(ax[2 * i, j], vals, curve, cmap, lab, log=log)
            _waterfall(ax[2 * i + 1, j], vals, curve, cmap, lab, log=log)
            ax[2 * i, j].set_title(f"{route}\n{title}")
            ax[2 * i, j].set_xlim(-4, 4)
            ax[2 * i + 1, j].set_xlim(-4, 4)
            ax[2 * i + 1, j].set_xlabel("x / r")
    _save(fig, "atlas_a2_turiel")


def fig_a3_dyadic():
    """A3. Dyadic banks and edge detectors across order, scale and support."""
    from dynamix.core.dwt import BANKS

    fig, ax = plt.subplots(2, 3, figsize=(6.5, 5.6), constrained_layout=True,
                           gridspec_kw={"width_ratios": (1.0, 1.0, 1.15)})
    alphas = np.linspace(1.0, 5.0, 26)

    def mz(al):
        p = wp.mz_profile(3, alpha=al)
        return p.x_px, p.psi

    _sweep(ax[0, 0], alphas, mz, "brg", "α")
    ax[0, 0].set_title("Mallat–Zhong, level 3:\nα from 1 (box pair) to 5")
    _waterfall(ax[1, 0], alphas, mz, "brg", "α")
    ax[1, 0].set_xlabel("x (px)")
    sigmas = np.geomspace(1.0, 16.0, 26)
    _sweep(ax[0, 1], sigmas, lambda sg: (lambda p: (p.x_px, p.psi))(wp.cdf_profile("mz", sg)), "rainbow", "σ", log=True)
    ax[0, 1].set_title("CDF edges, Re u (Canny):\nσ from 1 to 16 px")
    ss = np.geomspace(2.0, 32.0, 26)
    _sweep(ax[1, 1], ss, lambda s_: (lambda p: (p.x_px, p.psi))(wp.tang_you_profile("tang_you", s_)), "rainbow", "s",
           log=True)
    ax[1, 1].set_title("Tang–You kernel:\nsupport radius s from 2 to 32 px")
    ax[1, 1].set_xlabel("x (px)")
    cm = plt.get_cmap("gist_rainbow")
    for k, b in enumerate(BANKS):
        p = wp.dwt_profile(b, 3)
        ax[0, 2].plot(p.x_px, p.psi, color=cm(k / (len(BANKS) - 1)), lw=LW, label=b)
    _legend_below(ax[0, 2], ncol=4, fontsize=4.5, handlelength=1.0, columnspacing=0.8)
    ax[0, 2].set_title(f"the {len(BANKS)} DWT banks at level 3")
    ax[0, 2].set_xlim(-40, 40)
    for k, lv in enumerate((1, 2, 3, 4, 5)):
        p = wp.mz_profile(lv, 3.0)
        ax[1, 2].plot(p.x_px, p.psi, color=plt.get_cmap("brg")(k / 4), lw=LW,
                      label=f"level {lv}" + (" (off ladder)" if p.off_ladder else ""))
    _legend_below(ax[1, 2], ncol=2, fontsize=5)
    ax[1, 2].set_title("Mallat–Zhong, α = 3: levels 1–5")
    ax[1, 2].set_xlabel("x (px)")
    for a_ in ax.ravel():
        a_.axhline(0.0, color="0.85", lw=0.5)
    _save(fig, "atlas_a3_dyadic")


def fig_a4_q_range():
    """A4. The q family over its whole range from the app's own e_q."""
    from dynamix.core.microcanonical import paired_q_beta

    segments = [("q → −∞ … −1", np.array([-100.0, -30.0, -10.0, -5.0, -3.0, -2.0, -1.5]), "Blues", (-100, -1)),
                ("−1 … 0", np.linspace(-1.0, 0.0, 8), "Greens", (-1, 0)),
                ("0 … 1", np.linspace(0.0, 1.0, 9)[1:-1], "Oranges", (0, 1)),
                ("1 … 2", np.linspace(1.0, 1.95, 9)[1:], "Reds", (1, 2))]
    land = [(-100.0, "navy", "q = −100 (→ −∞: the box)"), (-1.0, "teal", "q = −1 (semicircle)"),
            (0.0, "purple", "q = 0 (parabola)"), (1.0, "k", "q = 1 (Gaussian)")]

    def beta_of(q, paired):
        return paired_q_beta(q) if paired else 0.5

    def inside(u, q, b):
        return np.ones_like(u, dtype=bool) if q >= 1.0 else (1.0 - (1.0 - q) * b * u * u) > 0

    def theta(u, q, paired):
        return sm._e_q(-beta_of(q, paired) * u * u, q)

    def psi(u, q, paired):                     # d/du e_q(−βu²) = −2βu e_q^q inside; slope −1 at 0
        b = beta_of(q, paired)
        m = inside(u, q, b)
        e = np.where(m, sm._e_q(-b * u * u, q), 1.0)
        return np.where(m, -2 * b * u * e ** q, 0.0) / (2 * b)

    def marr(u, q, paired):                    # −Δ in 2-D of e_q(−βρ²); value 1 at the origin
        b = beta_of(q, paired)
        m = inside(u, q, b)
        e = np.where(m, sm._e_q(-b * u * u, q), 1.0)
        return np.where(m, 4 * b * e ** q - 4 * b * b * q * u * u * e ** (2 * q - 1), 0.0) / (4 * b)

    def closed(kind, u):                       # the closed forms with β responding
        if kind == "theta":
            return {"box": (np.abs(u) < np.sqrt(2.0)).astype(float),
                    "semicircle": np.sqrt(np.clip(1 - u * u / 3.0, 0, None)),
                    "parabola": np.clip(1 - u * u / 4.0, 0, None), "gaussian": np.exp(-u * u / 2.0)}
        if kind == "psi":
            return {"semicircle": -u / np.sqrt(np.clip(1 - u * u / 3.0, 1e-9, None)) * (np.abs(u) < np.sqrt(3)),
                    "parabola": -u * (np.abs(u) < 2.0), "gaussian": -u * np.exp(-u * u / 2.0)}
        return {"gaussian": (1 - u * u / 2.0) * np.exp(-u * u / 2.0)}

    u = np.linspace(-5, 5, 4001)
    rows = [("θ = e_q(−βu²), max 1", theta, "theta"), ("ψ = ∂θ (q-g1), slope −1 at 0", psi, "psi"),
            ("−Δθ (q-Mexican hat), value 1 at 0; q > ½", marr, "marr")]
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 8.0))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for i, (title, fn, kind) in enumerate(rows):
            for j, paired in enumerate((False, True)):
                ax = axes[i, j]
                scale = (lambda y: y / np.nanmax(np.abs(y))) if kind == "theta" else (lambda y: y)
                for _name, qs, cmap, (lo, hi) in segments:
                    cm, nm = plt.get_cmap(cmap), Normalize(vmin=lo - 0.3 * (hi - lo), vmax=hi)
                    for q in qs:
                        if q in (-100.0, -1.0, 0.0, 1.0) or (kind == "marr" and q <= 0.5):
                            continue
                        ax.plot(u, scale(fn(u, q, paired)), color=cm(nm(q)), lw=LW)
                for q, c, lab in land:
                    if kind == "marr" and q <= 0.5:
                        continue
                    ax.plot(u, scale(fn(u, q, paired)), color=c, lw=1.6 if q == 1.0 else 1.1, label=lab, zorder=5)
                if paired:
                    for key, y in closed(kind, u).items():
                        ax.plot(u, scale(y) if kind == "theta" else y, "k:", lw=0.8, zorder=6,
                                label="closed forms" if key == "gaussian" else None)
                ax.set_ylim({"theta": (-0.05, 1.08), "psi": (-2.2, 2.2), "marr": (-0.5, 1.2)}[kind])
                ax.set_title(f"{title}\n" + ("β(q) = 1/(2(2−q)): escort variance held" if paired
                                             else "β fixed at ½: the support shrinks as q → −∞"))
                ax.axhline(0.0, color="0.85", lw=0.5)
                ax.set_xlim(-5, 5)
                if i == 2:
                    ax.set_xlabel("u = x/σ (σ of the fixed-β Gaussian)")
    for k, (name, _qs, cmap, (lo, hi)) in enumerate(segments):
        cax = fig.add_axes([0.9, 0.78 - 0.21 * k, 0.012, 0.16])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap=cmap), cax=cax)
        cb.set_label(f"q: {name}")
        cb.ax.tick_params(labelsize=5)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.47, 0.0))
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.1, top=0.95, hspace=0.45, wspace=0.18)
    _save(fig, "atlas_a4_q_range")


def fig_a5_meyer():
    """A5. The q-Meyer against the Gaussian, matched by band peak and by real-space lobe."""
    a = 8.0
    sigma = a * sm._SIGMA_PER_SCALE
    qs = np.linspace(0.1, 1.95, 26)
    cm, nm = _ramp(qs, "rainbow")
    g = sm.smoother("gaussian")
    s_grid = np.linspace(0.0, 3.5, 700)
    zero = np.zeros_like(s_grid)

    def sections(q):
        """θ and ψ out to ±26 σ (the registry's own section stops at ±12 σ, which the dilated
        column would shrink to ±6.4 σ), from the same inverse transform of θ̂."""
        s = sm.smoother("q_meyer", {"q": q})
        x0, th = sm._central_row(s.theta_hat, 0, n=4096, du=1.0 / 32.0, span=26.0)
        x1, ps = sm._central_row(s.theta_hat, 1, n=4096, du=1.0 / 32.0, span=26.0)
        return x0, th, x1, ps, s

    x1p, psi1 = sm.smoother("q_meyer", {"q": 1.0}).section(a, 1)
    dil = abs(x1p[np.argmax(psi1)]) / sigma       # the q = 1 Meyer's ψ peak in σ, the registry's section
    fig, ax = plt.subplots(3, 2, figsize=(6.5, 7.2), constrained_layout=True)
    modes = (("as the registry draws it: band peak λ* matched to the Gaussian's", 1.0),
             (f"dilated by 1/{dil:.2f}: ψ peaks at ±σ like the Gaussian's", dil))
    for j, (mode, factor) in enumerate(modes):
        for q in qs:
            x0, th, x1, ps, s = sections(q)
            ax[0, j].plot(x0 / factor, th, color=cm(nm(q)), lw=LW)
            ax[1, j].plot(x1 / factor, ps, color=cm(nm(q)), lw=LW)
            ax[2, j].plot(s_grid, s.theta_hat(s_grid / factor, zero), color=cm(nm(q)), lw=LW)
        x0, th, x1, ps, s = sections(1.0)
        ax[0, j].plot(x0 / factor, th, "k", lw=1.4, label="Meyer, q = 1 (the reference)")
        ax[1, j].plot(x1 / factor, ps, "k", lw=1.4)
        ax[2, j].plot(s_grid, s.theta_hat(s_grid / factor, zero), "k", lw=1.4)
        (gx0, gth), (gx1, gps) = g.section(a, 0), g.section(a, 1)
        ax[0, j].plot(gx0 / sigma, _norm(gth), "k--", lw=0.8, label="Gaussian pair (g0, g1)")
        ax[1, j].plot(gx1 / sigma, _norm(gps), "k--", lw=0.8)
        ax[2, j].plot(s_grid, g.theta_hat(s_grid, zero), "k--", lw=0.8)
        ax[0, j].set_title(mode)
        for i in (0, 1):
            ax[i, j].set_xlim(-12, 12)
            ax[i, j].axhline(0.0, color="0.85", lw=0.5)
        for i in range(3):
            _colorbar(fig, ax[i, j], cm, nm, "q")
        ax[2, j].set_xlabel("s = a·k")
        ax[1, j].set_xlabel("x/σ")
    ax[0, 0].set_ylabel("θ")
    ax[1, 0].set_ylabel("ψ = ∂θ")
    ax[2, 0].set_ylabel("θ̂(s)")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    _save(fig, "atlas_a5_meyer")


def fig_a6_meyer_rings():
    """A6. Why a bandlimited analyzer looks wide: rings, their ρ⁻⁴ decay, the narrow band."""
    span = 40.0
    curves = [("q-Meyer ψ, q = 1.0", sm.smoother("q_meyer", {"q": 1.0}), "k"),
              ("q-Meyer ψ, q = 1.5", sm.smoother("q_meyer", {"q": 1.5}), "tab:red"),
              ("q-Meyer ψ, q = 0.5", sm.smoother("q_meyer", {"q": 0.5}), "tab:blue")]
    g = sm.smoother("gaussian")
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 3.1), constrained_layout=True)
    s_grid = np.linspace(0.0, 2.0, 800)
    zero = np.zeros_like(s_grid)
    for label, s, c in curves:
        u, y = sm._central_row(s.theta_hat, 1, n=4096, du=1.0 / 32.0, span=span)
        ax[0].plot(u, y, color=c, lw=0.7, label=label)
        pos = u > 0
        ax[1].semilogy(u[pos], np.abs(y[pos]), color=c, lw=0.6, label=f"|ψ|, q = {s.params['q']:g}")
        ax[2].plot(s_grid, _norm(s_grid * s.theta_hat(s_grid, zero)), color=c, lw=0.9,
                   label=f"q-Meyer s·θ̂, q = {s.params['q']:g}")
    u, y = sm._central_row(g.theta_hat, 1, n=4096, du=1.0 / 32.0, span=span)
    ax[0].plot(u, y, "k--", lw=0.7, label="Gaussian ψ")
    rho = np.linspace(1.0, span, 200)
    ax[1].semilogy(rho, 50.0 * rho ** -4.0, ":", color="0.4", lw=0.8, label=r"$\rho^{-4}$ (a C³ taper's decay)")
    ax[1].set_ylim(1e-7, 2.0)
    ax[2].plot(s_grid, _norm(s_grid * g.theta_hat(s_grid, zero)), "k--", lw=0.8, label="Gaussian s·θ̂ (g1 band)")
    ax[2].axvline(1 / np.sqrt(2.0), color="0.8", lw=0.6)
    ax[0].set_title("q-Meyer ψ over ±40 σ")
    ax[1].set_title("|ψ| envelope")
    ax[2].set_title("the analyzer bands")
    ax[0].set_xlabel("x/σ")
    ax[1].set_xlabel("x/σ")
    ax[2].set_xlabel("s = a·k")
    for a_ in ax:
        _legend_below(a_, ncol=1, fontsize=5)
    _save(fig, "atlas_a6_meyer_rings")


def fig_a7_frac_grids():
    """A7. The isotropic polyharmonic B-spline below α = 3 on three inverse-FFT grids, against its
    space-domain evaluation."""
    from dynamix.core.polyharmonic_exact import evaluate

    grids = [(1.0 / 16.0, 1024, "tab:blue"), (1.0 / 64.0, 4096, "tab:orange"), (1.0 / 128.0, 8192, "k")]
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.7), constrained_layout=True)
    for a_, alpha in zip(ax, (1.5, 2.0, 2.5)):
        h = np.sqrt(12.0 / (alpha + 1.0))         # the knot spacing in σ (the matched-σ spline)
        def main_lobe(u, y, alpha=alpha):
            """Unbounded at the knots (α < 2): scaled by the largest value a tenth of a knot
            spacing from every knot and clipped; bounded: scaled by its own maximum."""
            if alpha >= 2.0:
                return y / np.abs(y).max()
            knot = u / h
            return np.clip(y / np.abs(y[np.abs(knot - np.round(knot)) > 0.1]).max(), -1.0, 1.0)

        for du, n, c in grids:
            u, y = sm._central_row(sm._frac_hat_factory(alpha), 1, n=n, du=du, span=4.0)
            a_.plot(u, main_lobe(u, y), color=c, lw=0.6, alpha=0.85,
                    label=f"grid σ/{int(round(1 / du))}, Nyquist s = {np.pi * np.sqrt(2.0) / (2 * du):.0f}")
        u = (np.arange(-512, 512) + 0.5) / 128.0
        a_.plot(u, main_lobe(u, evaluate(u / h, 0.0, alpha + 1.0, 1)), color="tab:red", lw=0.9,
                label="space domain (exact)")
        for k in (-2, -1, 1, 2):
            a_.axvline(k * h, color="0.85", lw=0.5, ls=":")
        a_.set_xlim(-4, 4)
        a_.set_ylim(-1.15, 1.15)
        a_.set_title(f"α = {alpha:g}: knots at multiples of {h:.2f} σ")
        a_.set_xlabel("x/σ")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    _save(fig, "atlas_a7_frac_grids")


def _k_q(q, bw=0.5):
    """The q-Morlet's carrier, Borges et al. (2004) eq. 29: the envelope at one half after a period."""
    if q == 1.0:
        return 2 * np.pi * np.sqrt(bw / np.log(2.0))
    return 2 * np.pi * np.sqrt((1 - q) * bw / (1 - 2 ** (q - 1)))


def _q_morlet_1d(u, q, bw=0.5):
    """The 1-D q-Morlet (Borges et al. 2004 eq. 23) with the exact zero-mean correction."""
    env = sm._e_q(-bw * u * u, q)
    carrier = np.exp(1j * _k_q(q, bw) * u)
    return env * (carrier - (env * carrier).sum() / env.sum())


def fig_a8_qmorlet_chirp():
    """A8. A single oscillating singularity x₊^h sin(x^−β) under the 1-D q-Morlet (L¹)."""
    h, beta = 0.6, 0.5
    n, half = 1 << 19, 2.0
    x = (np.arange(n) - n // 2) * (2 * half / n)
    dx = x[1] - x[0]
    with np.errstate(all="ignore"):
        f = np.where(x > 0, np.abs(x) ** h * np.sin(np.abs(x) ** -beta), 0.0)
    f[~np.isfinite(f)] = 0.0
    fft = fft_policy.double()
    f_hat = fft.fft(f)
    sel = (x > 5 * (beta * dx) ** (1 / (1 + beta))) & (x < 1.0)
    idx = np.flatnonzero(sel)
    scales = np.geomspace(0.002, 0.2, 40)

    def cwt(a, q):
        psi = _q_morlet_1d(x / a, q) / a
        return fft.ifft(f_hat * fft.fft(np.fft.ifftshift(np.conj(psi)))) * dx

    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.6), constrained_layout=True)
    xs_log = np.geomspace(1e-3, 0.3, 300)
    cols = np.searchsorted(x, xs_log)
    image, ridge = [], {}
    colors = {1.0: "k", 1.5: "tab:red", 0.7: "tab:blue"}
    for q in (1.0, 1.5, 0.7):
        pos, mod, re, im = [], [], [], []
        for a in scales:
            T = cwt(a, q)
            i = idx[np.argmax(np.abs(T[sel]))]
            pos.append(x[i]); mod.append(abs(T[i])); re.append(abs(T[i].real)); im.append(abs(T[i].imag))
            if q == 1.0:
                row = np.abs(T[cols])
                image.append(row / row.max())
        ridge[q] = tuple(np.asarray(v) for v in (pos, mod, re, im))
    ax[0, 0].pcolormesh(xs_log, scales, np.asarray(image), cmap="magma", shading="auto")
    ax[0, 0].plot(ridge[1.0][0], scales, "o", ms=2.5, mfc="c", mec="k", mew=0.3, label="modulus ridge")
    ax[0, 0].plot((beta * scales / _k_q(1.0)) ** (1 / (1 + beta)), scales, "w--", lw=0.8,
                  label=r"law $x_a=(\beta a/k_q)^{1/(1+\beta)}$")
    ax[0, 0].set(xscale="log", yscale="log", xlabel="x", ylabel="scale a",
                 title=f"|T(x,a)| per scale, q = 1, chirp h = {h:g}, β = {beta:g}")
    ax[0, 0].legend(loc="lower right")
    slope_amp = (h - beta / 2) / (1 + beta)
    for q, (pos, mod, re, im) in ridge.items():
        ax[0, 1].loglog(scales, mod, color=colors[q], lw=1.0, label=f"|T| at the ridge, q = {q:g}")
        s = np.polyfit(np.log(scales), np.log(pos), 1)[0]
        ax[1, 0].loglog(scales, pos, "o", ms=2.0, color=colors[q], label=f"ridge, q = {q:g} (slope {s:.3f})")
        ax[1, 0].loglog(scales, (beta * scales / _k_q(q)) ** (1 / (1 + beta)), "--", color=colors[q], lw=0.6)
    pos, mod, re, im = ridge[1.0]
    ax[0, 1].loglog(scales, re, ".", ms=2.5, color="tab:green", label="|Re T| at the ridge, q = 1")
    ax[0, 1].loglog(scales, im, "x", ms=2.5, color="tab:orange", label="|Im T| at the ridge, q = 1")
    ax[0, 1].loglog(scales, 1.5 * mod[-1] * (scales / scales[-1]) ** slope_amp, "k:", lw=0.8,
                    label=f"law $a^{{(h-\\beta/2)/(1+\\beta)}} = a^{{{slope_amp:.3f}}}$")
    ax[0, 1].set(xlabel="scale a", ylabel="ridge amplitude", title="the modulus reads h; the parts oscillate")
    ax[0, 1].legend(loc="lower right")
    ax[1, 0].set(xlabel="scale a", ylabel=r"ridge position $x_a$",
                 title=rf"$x_a \propto a^{{1/(1+\beta)}}$: slope {1 / (1 + beta):.3f} gives β (dashed: the law)")
    ax[1, 0].legend(loc="upper left")
    u = np.linspace(-6, 6, 1201)
    for q in (0.7, 1.0, 1.5):
        w = _q_morlet_1d(u, q)
        ax[1, 1].plot(u, w.real / np.abs(w).max(), color=colors[q], lw=0.9,
                      label=f"Re ψ, q = {q:g}, $k_q$ = {_k_q(q):.2f}")
        ax[1, 1].plot(u, sm._e_q(-0.5 * u * u, q), ":", color=colors[q], lw=0.7)
    ax[1, 1].set(xlabel="u", title="the q-Morlet: Re ψ and envelope (Borges eqs. 23, 29)")
    ax[1, 1].legend(loc="lower right")
    _save(fig, "atlas_a8_qmorlet_chirp")


FIGURES = (fig_wtmm_3d, fig_microcanonical, fig_dwt_banks, fig_dwt_3d, fig_interpolating, fig_qmorlet_3d,
           fig_edges, fig_a1_smoothers, fig_a1b_smoothers, fig_a2_turiel, fig_a3_dyadic, fig_a4_q_range, fig_a5_meyer,
           fig_a6_meyer_rings, fig_a7_frac_grids, fig_a8_qmorlet_chirp)


def main(names=()):
    OUT.mkdir(exist_ok=True)
    for f in FIGURES:
        if names and f.__name__ not in names:
            continue
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main(sys.argv[1:])
