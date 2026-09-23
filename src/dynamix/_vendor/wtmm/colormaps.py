"""Colormaps for WTMM scalogram visualization, ported from LastWave."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap


def lastwave_colormap(n=256, name='lastwave'):
    """Build LastWave's default 'color' colormap from scripts/misc/color _CMInit.

    HSV construction:
        hue        = (1 - i/N) * 270   # blue (270°) → red (0°)
        saturation = 1.0
        value      = (i/N)^0.3          # power-law brightness
    """
    t = np.linspace(0, 1, n)
    h = (1 - t) * 270 / 360       # matplotlib wants [0,1] not degrees
    s = np.ones(n)
    v = t ** 0.3
    hsv = np.column_stack([h, s, v])
    rgb = np.array([mc.hsv_to_rgb(c) for c in hsv])
    return ListedColormap(rgb, name=name)


def lastwave_grey_colormap(n=256, name='lastwave_grey'):
    """Build LastWave's 'grey' colormap: linear black→white."""
    t = np.linspace(0, 1, n)
    rgb = np.column_stack([t, t, t])
    return ListedColormap(rgb, name=name)


def lastwave_b2r_colormap(n=256, name='lastwave_b2r'):
    """Build LastWave's 'blue2red' signed colormap from _CMInit blue2red branch.

    Blue→black→red with sqrt brightness ramp on each side.
    """
    half = n // 2
    extra = n % 2
    # blue side: bright blue → black
    t_blue = np.linspace(1, 0, half + extra) ** 0.5
    blue_rgb = np.column_stack([np.zeros(half + extra), np.zeros(half + extra), t_blue * 1.0])
    # red side: black → bright red
    t_red = np.linspace(0, 1, half) ** 0.5
    red_rgb = np.column_stack([t_red * 1.0, np.zeros(half), np.zeros(half)])
    rgb = np.vstack([blue_rgb, red_rgb])
    return ListedColormap(rgb, name=name)


def scalogram_lastwave(cwt_matrix, scales=None, x=None, ax=None,
                        norm_mode='lglobal', cmap=None, n_colors=256,
                        signed=False, vlim=None, causal_ranges=None,
                        interpolation='nearest',
                        dx=None, units='', wavelet=None, c_psi=None,
                        orient='horizontal', x_label=None,
                        **imshow_kw):
    """Plot a CWT scalogram using LastWave's rendering approach.

    Parameters
    ----------
    cwt_matrix : 2D array, shape (n_scales, n_samples)
        CWT coefficients (raw, not log-transformed).
    scales : 1D array, optional
        Scale values for y-axis. If None, uses row indices.
    x : 1D array, optional
        Sample positions for x-axis. If None, uses column indices.
    ax : matplotlib Axes, optional
    norm_mode : str
        'lglobal'  — per-scale-line normalization (default, LastWave default)
        'global'   — single global normalization across all scales
        'signedglobal' — signed mode, preserves sign, symmetric around 0
        'log2abs'  — log2(|W|+epsilon), global colorscale. Shows absolute
                     scaling structure across all scales simultaneously.
    cmap : colormap, optional
        If None, uses lastwave_colormap() for unsigned or lastwave_b2r for signed.
    n_colors : int
        Number of discrete color levels.
    signed : bool
        If True, map signed values (not absolute). Forces signedglobal norm.
    vlim : (vmin, vmax), optional
        Override normalization bounds for global/signedglobal modes.
    causal_ranges : list of (first, last) tuples, optional
        Per-scale valid sample ranges (from CWT border effects).
        Samples outside range are drawn transparent.
    interpolation : str
        imshow interpolation method.

    orient : str
        'horizontal' (default) — sample/depth on x, scale on y.
        'vertical' — scale on x, sample/depth on y (downward).
    x_label : str, optional
        Label for the sample axis (default 'Sample').

    Returns
    -------
    im : AxesImage
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    n_scales, n_samples = cwt_matrix.shape

    if signed or norm_mode == 'signedglobal':
        norm_mode = 'signedglobal'
        data = cwt_matrix.copy().astype(float)
    else:
        data = np.abs(cwt_matrix).astype(float)

    if cmap is None:
        if norm_mode == 'signedglobal':
            cmap = lastwave_b2r_colormap(n_colors)
        else:
            cmap = lastwave_colormap(n_colors)

    # Build the normalized image row by row, matching _DrawGraphWtrans
    img = np.zeros_like(data)

    if norm_mode == 'lglobal':
        # Per-scale normalization: each row independently
        for i in range(n_scales):
            row = data[i]
            if causal_ranges is not None:
                fp, lp = causal_ranges[i]
                valid = row[fp:lp+1]
            else:
                valid = row
            vmax = np.max(np.abs(valid)) if len(valid) > 0 else 1.0
            if vmax == 0:
                vmax = 1.0
            # LastWave: symmetric around 0, then takes |val|/valMax
            img[i] = row / vmax

    elif norm_mode == 'global':
        if vlim is not None:
            vmax = vlim[1]
        else:
            vmax = np.max(data)
        if vmax == 0:
            vmax = 1.0
        img = data / vmax

    elif norm_mode == 'signedglobal':
        if vlim is not None:
            vmin_s, vmax_s = vlim
        else:
            vmax_s = np.max(np.abs(data))
            vmin_s = -vmax_s
        if vmax_s == vmin_s:
            vmax_s = vmin_s + 1.0
        img = (data - vmin_s) / (vmax_s - vmin_s)

    elif norm_mode == 'log2abs':
        # Replace zeros with NaN, take log2 of positive values only
        pos_mask = data > 0
        log2data = np.full_like(data, np.nan)
        log2data[pos_mask] = np.log2(data[pos_mask])
        if vlim is not None:
            vmin_l, vmax_l = vlim
        else:
            finite = log2data[np.isfinite(log2data)]
            vmin_l = np.percentile(finite, 1)
            vmax_l = np.percentile(finite, 99)
        if vmax_l == vmin_l:
            vmax_l = vmin_l + 1.0
        img = (log2data - vmin_l) / (vmax_l - vmin_l)
        img = np.nan_to_num(img, nan=0.0)
        # Store for colorbar reference
        _log2_vlim = (vmin_l, vmax_l)

    # Mask out-of-causal-range samples
    if causal_ranges is not None:
        mask = np.ones_like(img, dtype=bool)
        for i in range(n_scales):
            fp, lp = causal_ranges[i]
            mask[i, fp:lp+1] = False
        img = np.ma.array(img, mask=mask)
        cmap.set_bad(alpha=0)

    # Clamp to [0,1]
    img = np.clip(img, 0, 1)

    # Build extent
    if x is not None:
        x0, x1 = x[0], x[-1]
    else:
        x0, x1 = 0, n_samples
    if scales is not None:
        log2s = np.log2(scales)
        if len(log2s) > 1:
            dy = log2s[1] - log2s[0]  # voice spacing in log2
        else:
            dy = 1.0
        y0 = log2s[0] - dy / 2
        y1 = log2s[-1] + dy / 2
    else:
        y0, y1 = 0, n_scales

    if x_label is None:
        x_label = 'Sample'

    if orient == 'vertical':
        # Transpose: scale on x-axis, sample/depth on y-axis (downward)
        img = img.T[:, ::-1]  # (n_samples, n_scales) then flip columns: coarse left
        extent = [y1, y0, x1, x0]  # [scale_hi, scale_lo, depth_hi, depth_lo]
        im = ax.imshow(img, aspect='auto', origin='upper', cmap=cmap,
                       extent=extent, interpolation=interpolation,
                       vmin=0, vmax=1, **imshow_kw)
        ax.set_xlim(y1, y0)  # coarse on left, fine on right
        ax.set_ylim(x1, x0)  # depth increasing downward

        if scales is not None:
            ax.set_xlabel('log₂(a)')
            ax.set_xticks(range(int(np.ceil(y0)), int(np.floor(y1)) + 1))
        ax.set_ylabel(x_label)

        # Physical unit secondary axis on top
        if dx is not None and scales is not None:
            from .spectra import _resolve_c_psi, _add_physical_scale_secondary
            c = _resolve_c_psi(wavelet, c_psi)
            _add_physical_scale_secondary(ax, dx, c, units, axis='x')
    else:
        # Default horizontal orientation
        extent = [x0, x1, y0, y1]
        im = ax.imshow(img, aspect='auto', origin='lower', cmap=cmap,
                       extent=extent, interpolation=interpolation,
                       vmin=0, vmax=1, **imshow_kw)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        if scales is not None:
            ax.set_ylabel('log₂(a)')
            ax.set_yticks(range(int(np.ceil(y0)), int(np.floor(y1)) + 1))
        ax.set_xlabel(x_label)

        # Physical unit secondary y-axis on the right side
        if dx is not None and scales is not None:
            from .spectra import _resolve_c_psi, _add_physical_scale_secondary
            c = _resolve_c_psi(wavelet, c_psi)
            _add_physical_scale_secondary(ax, dx, c, units, axis='y')

    # For log2abs mode, attach vlim so caller can build a meaningful colorbar
    if norm_mode == 'log2abs':
        im._log2_vlim = _log2_vlim

    return im


# Register colormaps globally so they can be used by name
_lw_cmap = lastwave_colormap(256)
_lw_grey = lastwave_grey_colormap(256)
_lw_b2r = lastwave_b2r_colormap(256)
try:
    plt.colormaps.register(_lw_cmap)
    plt.colormaps.register(_lw_grey)
    plt.colormaps.register(_lw_b2r)
except ValueError:
    pass  # already registered
