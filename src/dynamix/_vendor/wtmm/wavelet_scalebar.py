"""Render the analyzing wavelet kernel as an inset scalebar.

In xsmurf's convention, you pick a SMOOTHING function (`exp(-x^2-y^2)` =
Gaussian = g0, or the Mexican hat = g2 = second derivative of Gaussian),
and the CWT then differentiates it once or twice (`derivs='first'` or
`'all'`).  The actual ANALYZING WAVELET that convolves with the field is
the SMOOTHER differentiated `deriv_order` times:

    smoothing='gaussian' + deriv_order=1  ->  psi = g1   (first deriv of Gaussian)
    smoothing='gaussian' + deriv_order=2  ->  psi = g2   (= Mexican hat)
    smoothing='mexican'  + deriv_order=1  ->  psi = g3   (first deriv of Mexican)
    smoothing='mexican'  + deriv_order=2  ->  psi = g4

The scalebar inset shows the ANALYZING WAVELET (psi), not the smoother,
because that's what's actually convolved against your field.

Direct override: pass `wavelet_name='gN'` to render any registered
Gaussian-derivative directly (or any registered B-spline / q-Gaussian),
bypassing the smoother+order resolution.

Usage
-----
    from wtmm.wavelet_scalebar import draw_wavelet_scalebar
    fig, ax = plt.subplots(...)
    # main plot ...
    draw_wavelet_scalebar(ax, scale_px=8.0,
                           smoothing='gaussian', deriv_order=1,
                           px_um=0.5)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .wavelets import WAVELETS, wavelet_direct, wavelet_support


# Smoothing-function name -> base index in the Gaussian-derivative family.
# 'gaussian' is the unmodified Gaussian (g0).  'mexican' is the Mexican hat
# (g2 = second derivative of Gaussian); since it's already a 2nd-order
# derivative, applying derivative order N to it yields g_{2+N}.
_SMOOTHING_BASE = {
    'gaussian':    0,
    'g0':          0,
    'mexican':     2,
    'mexican_hat': 2,
    'mh':          2,
    'g2':          2,    # legacy alias: if you typed 'g2' as the smoother,
                          # we treat it as Mexican hat smoother
}


def resolve_analyzing_wavelet(
    smoothing: str = 'gaussian',
    deriv_order: int = 1,
) -> str:
    """Map (smoothing, deriv_order) -> the canonical analyzing wavelet name.

    Returns one of {'g0', 'g1', 'g2', 'g3', 'g4'} per
    `wtmm.wavelets.WAVELETS`. Raises if the resulting index is out of range
    (only g0..g4 are registered for the standard Gaussian-derivative family).
    """
    sm = str(smoothing).strip().lower()
    if sm not in _SMOOTHING_BASE:
        raise ValueError(
            f"unknown smoothing function {smoothing!r}; "
            f"expected one of {sorted(_SMOOTHING_BASE.keys())}"
        )
    base = _SMOOTHING_BASE[sm]
    n = base + int(deriv_order)
    name = f'g{n}'
    if name not in WAVELETS:
        raise ValueError(
            f"smoothing={smoothing!r} + deriv_order={deriv_order} requires "
            f"wavelet {name!r}, which is not registered. Registered: "
            f"{[k for k in WAVELETS.keys() if k.startswith('g')]}"
        )
    return name


def render_wavelet_kernel(
    scale_px: float,
    *,
    wavelet_name: Optional[str] = None,
    smoothing: str = 'gaussian',
    deriv_order: int = 1,
    n_points: int = 256,
    pad_factor: float = 1.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the analyzing wavelet at scale `scale_px`, in pixel units.

    Two ways to specify the wavelet:
      (a) `wavelet_name='gN'` (or any registered name)
          -> direct, bypasses smoother/deriv_order.
      (b) `smoothing=...` and `deriv_order=...`
          -> resolves to the actual analyzing wavelet (recommended).

    Parameters
    ----------
    scale_px : float          wavelet scale in pixel units
    wavelet_name : str | None direct override
    smoothing : str           smoothing-function name ('gaussian' | 'mexican' | ...)
    deriv_order : int         derivative order (1 for first-derivative WTMM,
                               2 for the Hessian path)
    n_points : int
    pad_factor : float

    Returns
    -------
    x : (n_points,) float64    in pixel units
    psi : (n_points,) float64  analyzing wavelet amplitude (unnormalized)
    """
    if wavelet_name is None:
        name = resolve_analyzing_wavelet(smoothing=smoothing,
                                          deriv_order=deriv_order)
    else:
        name = str(wavelet_name).strip().lower()
        if name not in WAVELETS:
            raise ValueError(
                f"unknown wavelet {wavelet_name!r}; registered: "
                f"{list(WAVELETS.keys())}"
            )
    x_min, x_max = wavelet_support(scale_px, wavelet_name=name)
    x_min *= pad_factor; x_max *= pad_factor
    x = np.linspace(x_min, x_max, n_points)
    psi = wavelet_direct(x, scale_px, wavelet_name=name)
    return x, psi


def draw_wavelet_scalebar(
    ax,
    scale_px: float,
    *,
    wavelet_name: Optional[str] = None,
    smoothing: str = 'gaussian',
    deriv_order: int = 1,
    px_um: Optional[float] = None,
    inset_loc: str = 'upper right',
    inset_size: str = '20%',
    color: str = 'black',
    label: Optional[str] = None,
    label_size: int = 8,
    show_zero_line: bool = True,
):
    """Add an inset showing the ANALYZING WAVELET kernel at scale_px.

    Parameters
    ----------
    ax : matplotlib Axes
    scale_px : float
        Wavelet scale in pixel units (the same `a` you passed to the CWT).
    wavelet_name : str | None
        Direct wavelet name (e.g. 'g2') to bypass smoother resolution.
    smoothing : str
        Smoothing function: 'gaussian' or 'mexican'.  Used together with
        `deriv_order` to resolve the analyzing wavelet.
    deriv_order : int
        Derivative order applied to the smoother by the CWT (1 or 2).
        v1 calls `cwt_2d_f32(..., derivs='first')` -> deriv_order=1.
        The Hessian path (`derivs='all'`) uses deriv_order=2.
    px_um : float, optional
        Pixel size in micrometers; if supplied, the auto-label includes
        the physical scale.
    inset_loc, inset_size, color, label, label_size, show_zero_line :
        Standard mpl style kwargs.

    Returns
    -------
    ax_inset : matplotlib Axes
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    x, psi = render_wavelet_kernel(scale_px, wavelet_name=wavelet_name,
                                     smoothing=smoothing,
                                     deriv_order=deriv_order)
    if wavelet_name is not None:
        analyzing_name = str(wavelet_name).strip().lower()
        analyzing_long = analyzing_name
    else:
        analyzing_name = resolve_analyzing_wavelet(smoothing=smoothing,
                                                    deriv_order=deriv_order)
        analyzing_long = (f'{smoothing} + d^{deriv_order} -> {analyzing_name}'
                          if deriv_order > 1
                          else f'{smoothing} + d -> {analyzing_name}')

    ax_inset = inset_axes(ax, width=inset_size, height=inset_size, loc=inset_loc,
                            borderpad=0.5)
    ax_inset.plot(x, psi, color=color, lw=1.0)
    if show_zero_line:
        ax_inset.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.5)

    if label is None:
        if px_um is not None:
            label = (f'{analyzing_long}, a={scale_px:.1f} px '
                     f'({scale_px*px_um:.2f} μm)')
        else:
            label = f'{analyzing_long}, a={scale_px:.1f} px'
    ax_inset.set_title(label, fontsize=label_size, pad=2)
    return ax_inset


# ---- backwards compatibility -------------------------------------------
# Earlier draft of this module exported `_canonical_wavelet_name` mapping
# 'gaussian' -> 'g0' (smoothing function).  That convention was wrong for
# the WTMM use case, where the analyzing wavelet is the SMOOTHER's
# DERIVATIVE.  Keep the alias around but redirect callers to the
# correct resolver.
def _canonical_wavelet_name(name: str) -> str:
    """Deprecated.  Returns the analyzing wavelet for the standard
    first-derivative WTMM convention (smoothing + d^1).

    This was the older API and is kept for any external callers; new code
    should use `resolve_analyzing_wavelet(smoothing, deriv_order)` instead.
    """
    n = str(name).strip().lower()
    if n in WAVELETS:
        return n
    return resolve_analyzing_wavelet(smoothing=n, deriv_order=1)
