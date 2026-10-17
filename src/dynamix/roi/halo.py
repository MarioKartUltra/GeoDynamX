# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The halo engine: analyse a small ROI of a huge raster without lying at its edges.

A window cut out of a raster has fabricated edges. Convolving it produces boundary artefacts that
look exactly like structure, and the WTMM's whole business is finding structure -- so a naive ROI
crop reports fiction. The fix is to give each SCALE its own halo of REAL parent data, wide enough
that the kernel at that scale is numerically dead before it reaches the window's fabricated edge,
then throw the halo away and keep the ROI. Coarse scales read a wide window, fine scales a narrow
one; every one of them sees only real data over the ROI.

``tests/test_roi_halo.py`` is the proof (THE ORACLE): each per-scale window is bit-identical to
the corresponding sub-rect of a single union window, and the resulting extrema match a whole-
union-window ``run_wtmm2d`` exactly in position and to the float32 FFT floor in value.

The unit ladder (read this before touching a constant)
------------------------------------------------------
Three different numbers in this codebase are all called "a". They are related by fixed factors,
and confusing them is a 6.977x error in a physical reading:

===========================  =============================================  ==================
quantity                     what it is                                     sigma in px
===========================  =============================================  ==================
``a_min``                    the user-facing MULTIPLIER on the device knob   ``1.57 * a_min``
``scale`` (normalized ``a``)  ``a_min * 6.977 * 2**(oct + voice/n_voice)``,   ``0.225 * scale``
                             straight out of ``compute_scales2d``
``sigma``                    the real-space Gaussian width the kernel has    --
===========================  =============================================  ==================

``6.977 = 6.0/0.86`` is xsmurf's normalization, applied inside ``compute_scales2d``. The Fourier
filter is ``exp(-(scale*k)**2)`` with ``k`` in cycles/px, so ``sigma = scale/(pi*sqrt(2)) =
0.225*scale``; feeding ``a_min`` instead gives ``1.57*a_min``, the same number by another route.
:func:`sigma_px_amin` and :func:`sigma_px_scale` are the two conversions, named for their input so
neither can be called with the wrong one by accident. :func:`halo_margin` deliberately uses
NEITHER -- it has its own constant, in normalized-scale units, for the reason below.

:mod:`dynamix.core.scale_units` is this ladder's per-wavelet generalization (true-2D kernel
sections, oracle-pinned); these Gaussian-path constants remain correct and remain in use.

Margins
-------
``m(a) = ceil(MARGIN_PER_SCALE * a)`` with ``MARGIN_PER_SCALE = 2.5`` and ``a`` the NORMALIZED
scale -- about 11 times the kernel's real-space sigma, not the 3-4 sigma a Gaussian's tail would
suggest.

That generosity is measured, not inherited. The discrete kernel's tail is NOT the continuous
Gaussian's: at the finest usable scales sigma is barely a pixel, the filter is close to all-pass
at Nyquist, and the SAMPLED kernel rings rather than decaying. Measured on synthetic fields, the
accuracy cliff sits at roughly 4.5 sigma_true (a margin of 1.0*a); ``2.5*a`` clears it by 2.5x for
field variety, at about a third of the I/O the first (28-sigma) rule cost. The oracle and its
residual-ceiling test are the enforcement -- if a field ever needs more, they go red rather than
quietly returning contaminated numbers.

Separately, and not something a wider margin buys away: ``extrema2d``'s directional NMS can seat
a ridge point on either of two adjacent pixels once its gradient angle moves by a float32 ulp, a
rare knife-edge suppression tie of at most a couple points per scale. Measured over a margin sweep
from ``1.0*a`` to ``6.28*a``, it occurs at ANY margin in that range -- so the binding constraint on
margin size is the accuracy cliff above, not a clean cliff at which knife-edges also vanish. See
``tests/test_roi_halo.py``'s ``MAX_NMS_TIES_PER_SCALE`` for the tolerance this earns the oracle.

The reach widens the halo. :func:`halo_plan` measures each scale's reach
(:mod:`dynamix.core.reach`, the |ψ| mass radius to ``edge_eps``, converged when two grids agree) and
takes ``max(m(a), ceil(reach))``, so the 2.5 a budget stays a floor and a heavy-tailed smoother (the
q-Gaussian, the q-Meyer taper) gets the room its kernel needs. Where a scale's reach does not
converge (the sampled kernel rings at Nyquist and its mass radius grows with the grid) no finite halo
is honest: that scale reads the whole parent when the parent fits :data:`WHOLE_PARENT_MAX_PIXELS`,
and otherwise keeps m(a) and the run carries :func:`honest_halo_warning` in ``_warnings``.

Boundaries
----------
``boundary="auto"`` clamps ``ROI + m(a)`` to the parent and reflect-pads only the deficit, per
edge, so a ROI in the middle of a raster gets all-real halos and one at the corner gets real data
on the inward sides and reflection outward. ``boundary="reflective"`` restricts the parent view to
the ROI itself, so nothing outside the box is ever read. Either way ``result["_roi_margins"]``
records what actually happened, per scale, for the strip to report honestly.

Missing data
------------
Nodata (read from the file, never assumed) becomes 0.0 BEFORE the transform and before padding --
no interpolation, no extrema dropping. Zero-filled cells do NOT count towards ``real_frac``: they
are as fabricated as a reflection, and the honesty reading exists to say so. The ROI's missing
mask rides in the result with the per-scale halo radii so the canvas can outline the contaminated
region for the displayed scale.

Pure numpy at module level; ``rasterio`` and the WTMM backend are imported lazily inside the
functions that need them.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["MARGIN_PER_SCALE", "SIGMA_PER_A_MIN", "SIGMA_PER_SCALE", "ScaleHalo",
           "WHOLE_PARENT_MAX_PIXELS", "halo_margin", "halo_plan", "honest_halo_warning",
           "read_halo_window", "run_whole_through_core", "run_wtmm2d_roi",
           "run_wtmm2d_roi_level", "run_wtmm2d_roi_on_window", "sigma_px_amin", "sigma_px_scale",
           "taper_fallback"]

#: Pixels of real-space Gaussian sigma per unit of the ``a_min`` MULTIPLIER. The one home of this
#: number: the device readings and ``dynamix.shell.units`` import it from here.
SIGMA_PER_A_MIN = 1.57

#: Pixels of real-space Gaussian sigma per unit of NORMALIZED scale (``1.57 / 6.977``, i.e.
#: ``1/(pi*sqrt(2))``). Use this one for anything coming out of ``compute_scales2d``.
SIGMA_PER_SCALE = 0.225

#: Halo pixels per unit of NORMALIZED scale. Its own constant, in its own units, on purpose --
#: see the module docstring; it is an empirical accuracy budget, not a multiple of sigma.
MARGIN_PER_SCALE = 2.5

#: Both ROI sides must reach this. A one-pixel-tall box carries no multi-scale structure, and the
#: drag gesture will produce slivers by accident.
MIN_ROI_SIDE = 8

#: ``a_min`` below this makes the Mexican hat's sampled kernel non-local. The halo itself follows
#: each smoother's measured reach (:func:`halo_plan`); the knob's floor and readings quote this.
MIN_A_MIN = 1.0

#: A scale whose reach does not converge reads the whole parent up to this many pixels (the budget
#: ``shell.opening.open_field`` holds whole); a larger parent keeps the halo and the warning.
WHOLE_PARENT_MAX_PIXELS = 64_000_000

#: The edge parameters the devices carry; split off before the backend's typo guard.
_EDGE_KEYS = ("edge", "cut_scope", "edge_eps")


@dataclasses.dataclass(frozen=True)
class ScaleHalo:
    """One scale's halo: ``margin`` px of parent around the ROI, or the ``whole`` parent. ``a`` is
    the scale array's own element (a numpy float64): the transform builds its filter from it, and
    a Python float takes another arithmetic path there."""

    a: float
    margin: int
    whole: bool
    reach: object                                # dynamix.core.reach.Reach


def halo_plan(scales, wavelet: str, smoother_params, eps: float, parent_shape, *,
              until_converged: bool = False) -> list:
    """Per scale (finest first): the halo the reach asks for, never narrower than
    :func:`halo_margin`; the whole parent where the reach does not converge and ``parent_shape``
    (``(H, W)``, None when unknown) fits :data:`WHOLE_PARENT_MAX_PIXELS`. ``until_converged`` stops
    after the first scale whose reach converges (the finer scales are the ones that do not)."""
    from dynamix.core.reach import reach

    fits = (parent_shape is not None
            and int(parent_shape[0]) * int(parent_shape[1]) <= WHOLE_PARENT_MAX_PIXELS)
    out = []
    for a in scales:
        r = reach(wavelet, smoother_params, float(a), eps)
        m = halo_margin(a)
        if r.converged:
            out.append(ScaleHalo(a, max(m, int(math.ceil(r.radius))), False, r))
            if until_converged:
                break
        else:
            out.append(ScaleHalo(a, m, fits, r))
    return out


def honest_halo_warning(plan) -> "str | None":
    """The warning a run carries where a scale's reach does not converge and the parent was too
    large (or unknown) to read whole, or None."""
    bad = [sh for sh in plan if not sh.reach.converged and not sh.whole]
    if not bad:
        return None
    r = bad[0].reach
    span = f"{bad[0].a:.2f} px" if len(bad) == 1 else f"{bad[0].a:.2f}–{bad[-1].a:.2f} px"
    measured = ("" if r.measured is None else
                f" (its |ψ| mass radius is {r.measured:.0f} px on a {r.grids[1]} grid, "
                f"{r.ratio:.2f}× the {r.grids[0]} grid's)")
    return (f"{len(bad)} scale(s) at {span}: the wavelet's reach does not converge{measured}, so "
            "no halo makes this ROI match a whole-raster run there, and the parent is too large "
            "to read whole. Analyse the whole raster for a reference.")


def _run_warnings(plan, roi_margins, mode: str) -> list:
    """A run's edge warnings: the honest-halo limit, then the reflection at the raster's edges."""
    from dynamix.core.edges import corner_warning

    rows = [(m["a"], m["synthetic_px"]) for m in roi_margins if not m["whole"]]
    return [w for w in (honest_halo_warning(plan), corner_warning(rows, mode)) if w]


def _taper_fill(roi_values, parent_shape, plan):
    """``(fill, extras)`` for ``edge = "gaussian_taper"``: the taper (``{"mu", "w_t"}``, the ROI's
    mean and the band width) or None where it would not fit, and what the record says of it (the
    width, the field's dominant wavelength, the fallback's reason)."""
    from dynamix.core.edges import dominant_wavelength, taper_check

    v = np.asarray(roi_values, dtype=np.float64)
    finite = np.isfinite(v)
    mu = float(v[finite].mean()) if finite.any() else 0.0
    top = plan[-1].reach
    dominant = dominant_wavelength(v)
    w_t, reason = taper_check(parent_shape, dominant, top.radius if top.converged else top.measured)
    extras = {"requested": "gaussian_taper", "taper_px": w_t, "dominant_px": dominant}
    if reason:
        extras["fallback"] = reason
        return None, extras
    return {"mu": mu, "w_t": w_t}, extras


def _cut_of(plan, params: dict, offset, parent_shape, fill=None) -> "dict | None":
    """The core's ``cut``: each scale's border under ``edge = "cut"``, the taper band under the
    Gaussian taper; else None."""
    if fill is not None:
        borders, scope = [int(math.ceil(fill["w_t"]))] * len(plan), None
    elif params.get("edge", "reflect") == "cut":
        from dynamix.core.edges import cut_borders

        scope = params.get("cut_scope", "largest")
        borders = cut_borders(plan, scope)
    else:
        return None
    return {"borders": borders, "scope": scope,
            "offset": tuple(int(v) for v in offset),
            "parent_shape": tuple(int(v) for v in parent_shape[:2])}


def _edge_stamp(device: str, plan, roi_margins, parent_shape, eps: float, mode: str,
                cut=None) -> dict:
    """The ``_edge`` record of a run: the halo each scale used, its reach and where the
    halo's pixels came from, per side; under a cut, each scale's border."""
    stamp = {"device": device, "mode": mode, "eps": float(eps),
            "parent_shape": None if parent_shape is None else [int(v) for v in parent_shape[:2]],
            "budget_px": WHOLE_PARENT_MAX_PIXELS,
            "scales": [{"a": float(sh.a), "margin": sh.margin, "whole": sh.whole,
                        "radius": sh.reach.radius, "measured": sh.reach.measured,
                        "ratio": sh.reach.ratio, "grids": list(sh.reach.grids),
                        "converged": sh.reach.converged,
                        "real_px": dict(m["real_px"]), "synthetic_px": dict(m["synthetic_px"])}
                       for sh, m in zip(plan, roi_margins)]}
    if cut is not None:
        stamp["cut_scope"] = cut.get("scope")
        for row, b in zip(stamp["scales"], cut["borders"]):
            row["border"] = int(b)
    return stamp

#: Reported edge names, in this order.
_EDGES = ("N", "S", "E", "W")

_BOUNDARIES = ("auto", "reflective")


def sigma_px_amin(a_min) -> float:
    """Real-space Gaussian sigma in px for an ``a_min`` MULTIPLIER -- ``1.57 * a_min``.

    ``a_min`` is the unitless knob on the WTMM device, not a scale in pixels: the finest scale
    actually analysed is ``a_min * 6.977``. This is the conversion the device's derived reading
    wants (``a_min = 2`` -> ``sigma ~ 3.1 px``).
    """
    return SIGMA_PER_A_MIN * float(a_min)


def sigma_px_scale(scale) -> float:
    """Real-space Gaussian sigma in px for a NORMALIZED scale -- ``0.225 * scale``.

    ``scale`` is an element of :func:`~dynamix.core.wtmm_backend.compute_scales2d`'s output,
    i.e. ``a_min * 6.977 * 2**(octave + voice/n_voice)``. This is the conversion the scale sweep
    and the wavelet bar want.
    """
    return SIGMA_PER_SCALE * float(scale)


def halo_margin(a, margin_per_scale: float = MARGIN_PER_SCALE) -> int:
    """Halo width in whole pixels for NORMALIZED scale ``a`` -- ``ceil(2.5 * a)``.

    Not derived from :func:`sigma_px_scale`: the multiplier is an accuracy budget measured against
    the oracle, not a count of sigmas. See the module docstring.
    """
    return int(math.ceil(margin_per_scale * float(a)))


def _reflect_pad(arr: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """``np.pad(mode="reflect")`` that also works when a pad is wider than the array.

    numpy rejects a reflect pad of ``>= dim``; a coarse scale on a small parent asks for exactly
    that. Padding in repeated capped passes is the same reflection, just spelled out.

    A size-1 axis is the degenerate case and is handled first: it has nothing to mirror, so each
    capped pass would add ``dim - 1 == 0`` pixels and the loop would spin forever (it did).
    Reflecting a single row about itself IS replicating it, so that axis is replicated outright.
    """
    out = np.asarray(arr)
    remaining = [[int(top), int(bottom)], [int(left), int(right)]]
    for axis in (0, 1):
        if out.shape[axis] == 1 and any(v > 0 for v in remaining[axis]):
            width = [(0, 0), (0, 0)]
            width[axis] = (remaining[axis][0], remaining[axis][1])
            out = np.pad(out, width, mode="edge")
            remaining[axis] = [0, 0]
    while any(v > 0 for axis in remaining for v in axis):
        step = [[min(v, out.shape[axis] - 1) for v in remaining[axis]] for axis in (0, 1)]
        out = np.pad(out, ((step[0][0], step[0][1]), (step[1][0], step[1][1])), mode="reflect")
        remaining = [[remaining[a][s] - step[a][s] for s in (0, 1)] for a in (0, 1)]
    return out


def _read_halo(source, parent_rect, roi, margin, nodata, whole=False, fill=None):
    """``roi + margin``, clamped to ``parent_rect``, nodata zero-filled, deficit reflect-padded;
    with ``whole``, the parent rect itself and nothing padded. ``fill`` (the Gaussian taper's
    ``{"mu", "w_t"}``) rolls the parent's data off near its edges and puts its mean beyond them
    (``dynamix.core.edges.tapered``) in place of the reflection.

    ``parent_rect`` is ``(row_off, col_off, height, width)`` of the sub-rect of ``source`` that
    counts as "the parent": the whole file for ``boundary="auto"``, the ROI itself for
    ``"reflective"``. ``roi`` is in the file's own row/col coordinates.
    """
    pr0, pc0, ph, pw = (int(v) for v in parent_rect)
    r0, c0, h, w = (int(v) for v in roi)
    if whole:
        want_r0, want_c0, win_h, win_w = pr0, pc0, ph, pw
    else:
        want_r0, want_c0 = r0 - margin, c0 - margin
        win_h, win_w = h + 2 * margin, w + 2 * margin

    read_r0, read_c0 = max(pr0, want_r0), max(pc0, want_c0)
    read_r1 = min(pr0 + ph, want_r0 + win_h)
    read_c1 = min(pc0 + pw, want_c0 + win_w)
    top, left = read_r0 - want_r0, read_c0 - want_c0
    bottom, right = (want_r0 + win_h) - read_r1, (want_c0 + win_w) - read_c1

    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:                                    # pragma: no cover - env guard
        raise ValueError(f"{source}: windowed reads need rasterio ({exc})") from exc
    with rasterio.open(source) as src:
        values = np.asarray(
            src.read(1, window=Window(read_c0, read_r0, read_c1 - read_c0, read_r1 - read_r0)),
            dtype=np.float64)
        file_nodata = src.nodata

    sentinel = file_nodata if nodata is None else nodata
    missing = ~np.isfinite(values)
    if sentinel is not None:
        missing |= values == sentinel
    # BOEM-style undeclared fill: the tifs declare nodata = 0.0 but fill with
    # float32-lowest -- |v| >= 3e38 is nodata whatever the header says (the rule both other
    # read paths, from_geotiff_window and the picture, apply at read).
    missing |= np.abs(values) >= 3e38
    n_missing = int(missing.sum())
    if n_missing:
        values = np.where(missing, 0.0, values)           # zero-fill BEFORE the padding reflects it

    window = _reflect_pad(values, top, bottom, left, right)
    missing = _reflect_pad(missing.astype(np.uint8), top, bottom, left, right).astype(bool)
    if fill is not None:
        from dynamix.core.edges import tapered

        window = tapered(window, want_r0 - pr0 + np.arange(win_h), want_c0 - pc0 + np.arange(win_w),
                         (ph, pw), fill["mu"], fill["w_t"])
    # Zero-filled cells are fabricated too -- they are not "real data" for the honesty reading.
    real = (read_r1 - read_r0) * (read_c1 - read_c0) - n_missing
    reflected = {"N": top > 0, "S": bottom > 0, "W": left > 0, "E": right > 0}
    oy, ox = r0 - want_r0, c0 - want_c0
    info = {
        "real_frac": real / float(win_h * win_w),
        "reflected_edges": tuple(e for e in _EDGES if reflected[e]),
        "offset_in_window": (oy, ox),
        "missing": missing,                               # extra: the ROI's mask comes from here
        "real_px": {"N": oy - top, "S": win_h - oy - h - bottom, "E": win_w - ox - w - right,
                    "W": ox - left},
        "synthetic_px": {"N": top, "S": bottom, "E": right, "W": left},
    }
    return window, info


def read_halo_window(source, full_dims, roi, margin, *, nodata=None):
    """Read ``roi + margin`` from ``source``, real data where the parent has it.

    Parameters
    ----------
    source : str
        Anything rasterio opens, including a ``zip:/abs/path.zip!member.tif`` URL.
    full_dims : (height, width)
        The parent raster's full grid.
    roi : (row_off, col_off, height, width)
        The region of interest, in the parent's row/col coordinates.
    margin : int
        Halo width, typically :func:`halo_margin` of the scale being analysed.
    nodata : float or None
        Overrides the file's own nodata declaration. ``None`` uses the file's.

    Returns
    -------
    (window, info)
        ``window`` is float64, always exactly ``(h + 2*margin, w + 2*margin)`` -- clamped to the
        parent and reflect-padded per edge for whatever falls outside. ``info`` carries
        ``real_frac`` (fraction of the window that is genuine parent data: neither reflected NOR
        zero-filled nodata), ``reflected_edges`` (subset of ``("N", "S", "E", "W")``),
        ``offset_in_window`` (the ROI's origin inside the window) and ``missing`` (bool,
        window-shaped, the nodata that was zero-filled).
    """
    return _read_halo(source, (0, 0, int(full_dims[0]), int(full_dims[1])), roi, int(margin),
                      nodata)


def _crop_to_roi(extrema: dict, oy: int, ox: int, h: int, w: int) -> dict:
    """Keep the extrema inside the ROI sub-rect at row ``oy``, column ``ox`` of the window; shift
    their positions onto the ROI grid.

    The subpixel channels (``x_sub``/``y_sub``, present when the ``interpolate`` knob is on)
    shift by the same offset -- membership is decided by the INTEGER support, so the float
    position of a kept point may legitimately sit up to half a pixel outside the ROI edge."""
    x, y = extrema["x"], extrema["y"]
    keep = (x >= ox) & (x < ox + w) & (y >= oy) & (y < oy + h)
    out = {"x": x[keep] - ox, "y": y[keep] - oy, "mod": extrema["mod"][keep],
           "arg": extrema["arg"][keep], "line_id": extrema["line_id"][keep]}
    if "x_sub" in extrema:
        out["x_sub"] = extrema["x_sub"][keep] - ox
        out["y_sub"] = extrema["y_sub"][keep] - oy
    return out


def run_wtmm2d_roi(source, full_dims, roi, params: dict, *, boundary: str = "auto",
                   nodata=None, progress=None) -> dict:
    """Scalar 2D WTMM over ``roi`` of ``source``, one real-data halo per scale.

    Mirrors :func:`~dynamix.core.wtmm_backend._run_wtmm2d_scalar`'s stage composition and result
    keys, with the CWT and extrema stages run per scale on that scale's own window instead of once
    on a whole field. Nothing is cached: the windows differ per scale, so the backend's staged
    cache (keyed on one field hash) has nothing to key on -- ``cache_hits`` comes back empty and
    ``npz_path`` ``None``. The device above this owns the run-level cache.

    Parameters
    ----------
    source : str
        Path or GDAL URL of the parent raster.
    full_dims : (height, width)
        The parent's full grid.
    roi : (row_off, col_off, height, width)
        Both sides must be at least :data:`MIN_ROI_SIDE` px.
    params : dict
        The WTMM2D param contract, resolved by the backend's own typo guard, plus the edge keys
        (``edge_eps``, the reach's mass fraction). Where a scale's reach does not converge and the
        parent is too large to read whole, the run carries :func:`honest_halo_warning`.
    boundary : {"auto", "reflective"}
        ``"auto"`` takes real parent data wherever the halo has it; ``"reflective"`` reads only
        the ROI and reflects it, ignoring the parent entirely.
    nodata : float or None
        Overrides the file's own nodata declaration, forwarded to :func:`read_halo_window`.
    progress : callable(stage, frac) or None
        Called ``("roi scale i/n", 0.0)`` and ``(..., 1.0)`` around each scale. Per-scale is the
        honest granularity here -- the ROI path has no whole-field stages to report.

    Returns
    -------
    dict
        The canonical ``run_wtmm2d`` keys (``chains``, ``extrema``, ``scales``, ``hd_std``,
        ``hd_cmax``, ``npz_path``, ``params``, ``cache_hits``) plus ``_shape`` (the ROI's
        ``(h, w)``), ``_roi`` (source/roi/boundary), ``_roi_margins`` (per scale: ``a``,
        ``margin``, ``real_frac``, ``reflected_edges``), ``_missing_mask`` (bool, ROI-shaped) and
        ``_coi_radii`` (per-scale halo widths, for the canvas's contamination outlines) and
        ``_edge`` (per scale: the halo, whether the whole parent was read, the reach and its
        grids and ratio, the real and synthetic px per side).
    """
    from dynamix.core.wtmm_backend import _resolve_wtmm2d_params, compute_scales2d

    if boundary not in _BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; expected one of {list(_BOUNDARIES)}")
    r0, c0, h, w = (int(v) for v in roi)
    if h < MIN_ROI_SIDE or w < MIN_ROI_SIDE:
        raise ValueError(
            f"run_wtmm2d_roi: ROI is {h}x{w} px; both sides must be at least {MIN_ROI_SIDE} px. "
            "A sliver carries no multi-scale structure to measure, and its halo is almost "
            "entirely fabricated.")
    fh, fw = int(full_dims[0]), int(full_dims[1])
    if r0 < 0 or c0 < 0 or r0 + h > fh or c0 + w > fw:
        raise ValueError(
            f"run_wtmm2d_roi: roi (row={r0}, col={c0}, h={h}, w={w}) does not fit inside the "
            f"{fh}x{fw} px raster {source!r}. row/col must be >= 0 and row+h/col+w must not "
            "exceed the raster's own dimensions -- unchecked, this reaches rasterio's own window "
            "read once the clamped region goes empty and dies on its internal message instead.")

    eps = float(params.get("edge_eps", 1e-3))
    resolved = _resolve_wtmm2d_params({k: v for k, v in params.items() if k not in _EDGE_KEYS})
    if resolved["mode"] != "scalar":
        raise ValueError(
            f"run_wtmm2d_roi: mode={resolved['mode']!r} is not supported; the ROI halo path is "
            "scalar-only (the tensor pipeline is monolithic and has no per-scale seam)")

    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    # "reflective" IS "auto" against a parent that stops at the ROI's own edges.
    parent_rect = ((0, 0, int(full_dims[0]), int(full_dims[1])) if boundary == "auto"
                   else (r0, c0, h, w))
    plan = halo_plan(scales, resolved["wavelet"], resolved["smoother_params"], eps,
                     parent_rect[2:])
    mode, fill, extras = params.get("edge", "reflect"), None, {}
    if mode == "gaussian_taper":
        block, info0 = _read_halo(source, parent_rect, (r0, c0, h, w), 0, nodata)
        fill, extras = _taper_fill(np.where(info0["missing"], np.nan, block), parent_rect[2:],
                                   plan)
        mode = "gaussian_taper" if fill else "reflect"

    def _window_for(sh):
        return _read_halo(source, parent_rect, (r0, c0, h, w), sh.margin, nodata, whole=sh.whole,
                          fill=fill)

    cut = _cut_of(plan, params, (r0 - parent_rect[0], c0 - parent_rect[1]), parent_rect[2:],
                  fill)
    (extrema, chains, hd_std, hd_cmax, roi_margins, missing_mask,
     coi_radii) = _wtmm2d_roi_core(_window_for, h, w, resolved, plan, progress, cut)

    out = {
        "chains": chains, "extrema": extrema, "scales": scales,
        "hd_std": hd_std, "hd_cmax": hd_cmax, "npz_path": None,
        "params": resolved, "cache_hits": set(),
        "_shape": (h, w),
        "_roi": {"source": str(source), "roi": (r0, c0, h, w), "boundary": boundary},
        "_roi_margins": roi_margins,
        "_missing_mask": missing_mask,
        "_coi_radii": coi_radii,
        "_edge": {**_edge_stamp("wtmm2d_roi", plan, roi_margins, parent_rect[2:], eps, mode, cut),
                  **extras},
    }
    warnings = _run_warnings(plan, roi_margins, mode)
    if warnings:
        out["_warnings"] = warnings
    return out


def run_wtmm2d_roi_level(source, full_dims, roi, params: dict, level: int, *,
                         boundary: str = "auto", nodata=None) -> dict:
    """One scale of :func:`run_wtmm2d_roi`: the lifted modulus and the argument of scale
    ``level`` (0-based, finest first) over ``roi``, computed as :func:`_wtmm2d_roi_core`
    computes that scale -- the scale's own halo read off ``source`` (``boundary`` and
    ``nodata`` as the analysis ran), the backend's one-scale ``cwt2d`` with the derivatives the
    run's detector asks for, the ``fracint_alpha`` lift -- then cropped to the ROI.

    Returns ``{"mod": (h, w) float32, "arg": (h, w) float32, "scale": float, "margin": int}``.
    """
    from dynamix.core.wtmm_backend import (_apply_fracint2d, _resolve_wtmm2d_params,
                                           compute_scales2d, get_backend)

    if boundary not in _BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; expected one of {list(_BOUNDARIES)}")
    r0, c0, h, w = (int(v) for v in roi)
    eps = float(params.get("edge_eps", 1e-3))
    resolved = _resolve_wtmm2d_params({k: v for k, v in params.items() if k not in _EDGE_KEYS})
    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    level = int(level)
    if not 0 <= level < len(scales):
        raise ValueError(f"run_wtmm2d_roi_level: level {level} is outside 0..{len(scales) - 1}")
    a = scales[level]
    parent_rect = ((0, 0, int(full_dims[0]), int(full_dims[1])) if boundary == "auto"
                   else (r0, c0, h, w))
    sh = halo_plan(scales, resolved["wavelet"], resolved["smoother_params"], eps,
                   parent_rect[2:])[level]
    margin = sh.margin
    window, info = _read_halo(source, parent_rect, (r0, c0, h, w), margin, nodata,
                              whole=sh.whole)
    oy, ox = info["offset_in_window"]
    follow = resolved["detector"] == "follow"
    cwt = get_backend("python").cwt2d(window, [a], wavelet=resolved["wavelet"],
                                      pad=resolved["pad"], derivs="all" if follow else "first",
                                      smoother_params=resolved["smoother_params"])
    cwt = _apply_fracint2d({"mod": cwt["mod"], "arg": cwt["arg"]}, [a],
                           resolved["fracint_alpha"])
    crop = (slice(oy, oy + h), slice(ox, ox + w))
    return {"mod": cwt["mod"][0][crop], "arg": cwt["arg"][0][crop], "scale": float(a),
            "margin": margin}


def _wtmm2d_roi_core(window_for, h, w, resolved, plan, progress=None, cut=None):
    """The per-scale halo loop shared by :func:`run_wtmm2d_roi` (halos read off the FILE) and
    :func:`run_wtmm2d_roi_on_window` (halos sliced from an in-memory window): for each scale of
    ``plan`` (:func:`halo_plan`), ``window_for(scale_halo) -> (window, info)`` supplies ROI + that
    scale's halo, or the whole parent (zero-filled nodata, reflect-padded deficit) with ``info`` =
    ``real_frac`` / ``reflected_edges`` / ``missing`` / ``offset_in_window`` (the ROI's row and
    column in the window) / ``real_px`` / ``synthetic_px``; the CWT + extrema run on it, the
    extrema are cropped to the ROI, and only THEN are they chained and partitioned -- chaining
    happens inside the ROI. ``cut`` (``edge = "cut"``: ``borders`` per scale, the ROI's
    ``offset`` in the parent, the ``parent_shape``) trims the chains at each scale's cone of
    influence (spec 14's ``coi_trim``, or ``coi_drop`` when that is the border policy) and drops
    the maxima inside it, before the partition functions are built.

    Extracted verbatim from :func:`run_wtmm2d_roi` so both
    entries run one implementation; ``tests/test_roi_halo.py`` is the file path's oracle,
    ``tests/test_roi_wtmm_window.py`` the in-memory path's."""
    from dynamix.core.wtmm_backend import _apply_fracint2d, get_backend

    backend = get_backend("python")

    scales = np.asarray([sh.a for sh in plan])
    n_scales = len(plan)
    extrema, roi_margins, coi_radii = [], [], []
    missing_mask = None
    for i, sh in enumerate(plan):
        a = sh.a
        if progress is not None:
            progress(f"roi scale {i + 1}/{n_scales}", 0.0)
        margin = sh.margin
        window, info = window_for(sh)
        oy, ox = info["offset_in_window"]
        if missing_mask is None:
            missing_mask = info["missing"][oy:oy + h, ox:ox + w].copy()
        follow = resolved["detector"] == "follow"
        cwt = backend.cwt2d(window, [a], wavelet=resolved["wavelet"], pad=resolved["pad"],
                            derivs="all" if follow else "first",
                            smoother_params=resolved["smoother_params"])
        if follow:
            from dynamix.core.follow2d import kapa_fields
            kapa, kapap = kapa_fields(cwt, 0)
            cwt = {"mod": cwt["mod"], "arg": cwt["arg"]}
        # Same per-scale lift as the full run (_run_wtmm2d_scalar) -- an ROI recompute must
        # agree with the main result's moduli, chaining band and partition tables.
        cwt = _apply_fracint2d(cwt, [a], resolved["fracint_alpha"])
        if follow:
            # Same detection as the full run's extrema stage, on the halo window BEFORE the
            # crop (kappa's probes need the halo pixels; the crop shifts x_sub/y_sub). No
            # invalid mask here, matching the NMS ROI branch exactly: _read_halo zero-fills
            # nodata BEFORE the transform, and the honesty channel for fabricated data is
            # _missing_mask/real_frac reporting, never extrema dropping (the whole-field
            # pipeline's NaN-distrust contract does not apply to the windowed read).
            # The exact xsmurf port, matching the full run's extrema stage
            # (dynamix.core.xsmurf_follow, parity-proven -- see wtmm_backend's own use of it).
            from dynamix.core.xsmurf_follow import follow_extrema_scale_exact
            scale_extrema = follow_extrema_scale_exact(
                cwt["mod"][0], cwt["arg"][0], kapa, kapap,
                thresh=resolved["thresh"])
            scale_extrema.pop("_xs_runs", None)
            scale_extrema.pop("_xs_closed", None)
        else:
            scale_extrema = backend.extrema2d(cwt, [a], thresh=resolved["thresh"],
                                              field=window)[0]
            if resolved["interpolate"]:
                # Same refinement as the full run's extrema stage, on the window's own lifted
                # mod raster BEFORE the crop (the parabola's probes need the halo pixels).
                from dynamix.core.subpixel import refine_extrema_stack
                scale_extrema = refine_extrema_stack([scale_extrema], cwt["mod"])[0]
        extrema.append(_crop_to_roi(scale_extrema, oy, ox, h, w))
        roi_margins.append({"a": float(a), "margin": margin, "real_frac": info["real_frac"],
                            "reflected_edges": info["reflected_edges"], "whole": sh.whole,
                            "real_px": info["real_px"], "synthetic_px": info["synthetic_px"]})
        coi_radii.append(margin)
        if progress is not None:
            progress(f"roi scale {i + 1}/{n_scales}", 1.0)

    from dynamix.core.chaining import link

    chains = link(extrema, scales, resolved, backend)
    if cut is not None:
        from dynamix.core.chaining import border_trimmed
        from dynamix.core.edges import chain_reach, drop_in_cone

        policy = "coi_drop" if resolved.get("border_policy") == "coi_drop" else "coi_trim"
        chains = border_trimmed(chains, cut["parent_shape"], policy, 0.0,
                                int(resolved["min_chain_len"]),
                                reach_at=chain_reach(scales, cut["borders"]),
                                offset=cut["offset"])
        extrema = drop_in_cone(extrema, cut["borders"], cut["offset"], cut["parent_shape"])
    hd_std, hd_cmax = backend.partition2d(chains, scales, q_list=resolved["q_list"],
                                          min_chain_len=resolved["min_chain_len"])

    return extrema, chains, hd_std, hd_cmax, roi_margins, missing_mask, coi_radii


def run_wtmm2d_roi_on_window(window_values, real, offset: int, h: int, w: int, params: dict,
                             *, progress=None, parent_shape=None, roi_origin=None) -> dict:
    """:func:`run_wtmm2d_roi` over an IN-MEMORY processing window.

    ``window_values`` is ROI + ``offset`` px on every side, already read by
    :func:`dynamix.roi.runner.read_processing_window` (real data where the file has it, the
    deficit reflect-padded, nodata as NaN); ``real`` marks the genuine pixels. Each scale's
    halo is the centred sub-rect ``[offset - m, offset + h + m)`` -- bit-identical to what
    ``_read_halo`` reads off the file (the union-window claim ``test_roi_halo`` proves) -- with
    NaN zero-filled BEFORE the transform (this device's convention, reported in
    ``_missing_mask``/``real_frac``, never interpolated). ``offset`` must cover the widest halo
    that is not the whole parent. A scale that reads the whole parent slices the real rect of the
    window, allowed only when that rect is the whole dataset, ``parent_shape`` ``(H, W)`` (the
    runner reads it whole when a step asks and the budget allows). roi_origin is the ROI's
    (row, col) in that dataset, for a cut at its edges (the window's real rect stands in
    without it). Returns the same keys as
    :func:`run_wtmm2d_roi` (``_roi`` without a source: the caller stamps the region)."""
    from dynamix.core.wtmm_backend import _resolve_wtmm2d_params, compute_scales2d

    if h < MIN_ROI_SIDE or w < MIN_ROI_SIDE:
        raise ValueError(
            f"wtmm2d on an ROI: the ROI is {h}x{w} px; both sides must be at least "
            f"{MIN_ROI_SIDE} px (a sliver carries no multi-scale structure).")
    eps = float(params.get("edge_eps", 1e-3))
    resolved = _resolve_wtmm2d_params({k: v for k, v in params.items() if k not in _EDGE_KEYS})
    if resolved["mode"] != "scalar":
        raise ValueError(
            f"wtmm2d on an ROI: mode={resolved['mode']!r} is not supported; the per-scale halo "
            "path is scalar-only (the tensor pipeline has no per-scale seam)")
    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    values = np.asarray(window_values, dtype=np.float64)
    real = np.asarray(real, dtype=bool)
    o = int(offset)
    rows, cols = np.flatnonzero(real.any(axis=1)), np.flatnonzero(real.any(axis=0))
    pr0, pr1, pc0, pc1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    whole_ok = (parent_shape is not None
                and (pr1 - pr0, pc1 - pc0) == (int(parent_shape[0]), int(parent_shape[1])))
    plan = halo_plan(scales, resolved["wavelet"], resolved["smoother_params"], eps,
                     parent_shape if whole_ok else None)
    if roi_origin is not None and parent_shape is not None:
        origin, pshape = tuple(roi_origin), tuple(parent_shape)
    else:
        origin, pshape = (o - pr0, o - pc0), (pr1 - pr0, pc1 - pc0)
    mode, fill, extras = params.get("edge", "reflect"), None, {}
    if mode == "gaussian_taper":
        fill, extras = _taper_fill(values[o:o + h, o:o + w], pshape, plan)
        mode = "gaussian_taper" if fill else "reflect"
    need = max([sh.margin for sh in plan if not sh.whole], default=0)
    if o < need:
        raise ValueError(f"window margin {offset} px < the widest halo {need} px")

    def _filled(win, y0, x0):
        """The window under the taper, ``y0``/``x0`` its first row and column in the window."""
        if fill is None:
            return win
        from dynamix.core.edges import tapered

        rows = origin[0] + y0 - o + np.arange(win.shape[0])
        cols = origin[1] + x0 - o + np.arange(win.shape[1])
        return tapered(win, rows, cols, pshape, fill["mu"], fill["w_t"])

    def _window_for(sh):
        if sh.whole:
            sub = values[pr0:pr1, pc0:pc1]
            missing = ~np.isfinite(sub)
            win = _filled(np.where(missing, 0.0, sub) if missing.any() else sub.copy(), pr0, pc0)
            oy, ox = o - pr0, o - pc0
            return win, {"real_frac": float((~missing).mean()), "reflected_edges": (),
                         "missing": missing, "offset_in_window": (oy, ox),
                         "real_px": {"N": oy, "S": pr1 - pr0 - oy - h, "E": pc1 - pc0 - ox - w,
                                     "W": ox},
                         "synthetic_px": {"N": 0, "S": 0, "E": 0, "W": 0}}
        m = sh.margin
        sl = (slice(o - m, o + h + m), slice(o - m, o + w + m))
        sub, rsub = values[sl], real[sl]
        missing = ~np.isfinite(sub)
        win = _filled(np.where(missing, 0.0, sub) if missing.any() else sub.copy(), o - m, o - m)
        edges = {"N": not rsub[0].any(), "S": not rsub[-1].any(),
                 "W": not rsub[:, 0].any(), "E": not rsub[:, -1].any()}
        r_any, c_any = rsub.any(axis=1), rsub.any(axis=0)
        synth = {"N": int(np.argmax(r_any)), "S": int(np.argmax(r_any[::-1])),
                 "W": int(np.argmax(c_any)), "E": int(np.argmax(c_any[::-1]))}
        return win, {"real_frac": float((rsub & ~missing).mean()),
                     "reflected_edges": tuple(e for e in _EDGES if edges[e]),
                     "missing": missing, "offset_in_window": (m, m),
                     "real_px": {k: m - v for k, v in synth.items()}, "synthetic_px": synth}

    cut = _cut_of(plan, params, origin, pshape, fill)
    (extrema, chains, hd_std, hd_cmax, roi_margins, missing_mask,
     coi_radii) = _wtmm2d_roi_core(_window_for, h, w, resolved, plan, progress, cut)
    out = {
        "chains": chains, "extrema": extrema, "scales": scales,
        "hd_std": hd_std, "hd_cmax": hd_cmax, "npz_path": None,
        "params": resolved, "cache_hits": set(),
        "_shape": (h, w),
        "_roi": {"source": None, "roi": None, "boundary": "auto"},
        "_roi_margins": roi_margins,
        "_missing_mask": missing_mask,
        "_coi_radii": coi_radii,
        "_edge": {**_edge_stamp("wtmm2d", plan, roi_margins, parent_shape, eps, mode, cut),
                  **extras},
    }
    warnings = _run_warnings(plan, roi_margins, mode)
    if warnings:
        out["_warnings"] = warnings
    return out


def _whole_plan(resolved: dict, eps: float) -> list:
    """Every scale reading the whole raster, with its reach (for the record and the borders)."""
    from dynamix.core.reach import reach
    from dynamix.core.wtmm_backend import compute_scales2d

    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    return [ScaleHalo(a, 0, True, reach(resolved["wavelet"], resolved["smoother_params"],
                                        float(a), eps)) for a in scales]


def run_whole_through_core(values, params: dict, *, progress=None):
    """``wtmm2d`` over a whole raster under an edge mode that changes its data (the Gaussian
    taper): the raster is its own ROI, every scale reads it whole, the mode fills it, and the
    maxima outside the band the mode made are chained and partitioned. Returns ``(result,
    extras)``, ``result`` None where the mode falls back to the backend's own run (``extras``
    then says why)."""
    from dynamix.core.wtmm_backend import _resolve_wtmm2d_params

    eps = float(params.get("edge_eps", 1e-3))
    resolved = _resolve_wtmm2d_params({k: v for k, v in params.items() if k not in _EDGE_KEYS})
    if resolved["mode"] != "scalar":
        raise ValueError(f"wtmm2d: edge = {params.get('edge')!r} runs scale by scale and is "
                         "scalar-only (the tensor pipeline has no per-scale seam)")
    v = np.asarray(values, dtype=np.float64)
    H, W = v.shape[:2]
    missing = ~np.isfinite(v)
    plan = _whole_plan(resolved, eps)
    fill, extras = _taper_fill(np.where(missing, np.nan, v), (H, W), plan)
    if fill is None:
        return None, extras
    from dynamix.core.edges import tapered

    window = tapered(np.where(missing, 0.0, v), np.arange(H), np.arange(W), (H, W), fill["mu"],
                     fill["w_t"])
    info = {"real_frac": float((~missing).mean()), "reflected_edges": (), "missing": missing,
            "offset_in_window": (0, 0), "real_px": {s: 0 for s in _EDGES},
            "synthetic_px": {s: 0 for s in _EDGES}}
    cut = _cut_of(plan, params, (0, 0), (H, W), fill)
    (extrema, chains, hd_std, hd_cmax, roi_margins, _missing,
     _radii) = _wtmm2d_roi_core(lambda _sh: (window, info), H, W, resolved, plan, progress, cut)
    return {"chains": chains, "extrema": extrema, "scales": np.asarray([sh.a for sh in plan]),
            "hd_std": hd_std, "hd_cmax": hd_cmax, "npz_path": None, "params": resolved,
            "cache_hits": set(),
            "_edge": {**_edge_stamp("wtmm2d", plan, roi_margins, (H, W), eps,
                                    "gaussian_taper", cut), **extras}}, extras


def whole_fallback_stamp(params: dict, shape, extras: dict) -> dict:
    """The ``_edge`` of a whole-raster run whose edge mode fell back to the backend's run."""
    from dynamix.core.wtmm_backend import _resolve_wtmm2d_params

    eps = float(params.get("edge_eps", 1e-3))
    resolved = _resolve_wtmm2d_params({k: v for k, v in params.items() if k not in _EDGE_KEYS})
    plan = _whole_plan(resolved, eps)
    return {"device": "wtmm2d", "mode": "reflect", "eps": eps,
            "parent_shape": [int(shape[0]), int(shape[1])], "budget_px": None,
            "scales": [{"a": float(sh.a), "radius": sh.reach.radius,
                        "measured": sh.reach.measured, "ratio": sh.reach.ratio,
                        "grids": list(sh.reach.grids), "converged": sh.reach.converged}
                       for sh in plan], **extras}


def taper_fallback(values, parent_shape, params: dict, smoother_params) -> "str | None":
    """Before a run: why the Gaussian taper would fall back to reflection on ``values`` (the ROI's
    or the raster's) in a parent of ``parent_shape``, or None; ``smoother_params`` as the device
    resolves them."""
    from dynamix.core.edges import dominant_wavelength, taper_check
    from dynamix.core.reach import reach
    from dynamix.core.wtmm_backend import compute_scales2d

    top = compute_scales2d(int(params.get("n_oct", 3)), int(params.get("n_voice", 4)),
                           float(params.get("a_min", 1.0)))[-1]
    r = reach(params.get("wavelet", "mexican"), smoother_params, float(top),
              float(params.get("edge_eps", 1e-3)))
    if r.measured is None:
        return None
    return taper_check(parent_shape, dominant_wavelength(values),
                       r.radius if r.converged else r.measured)[1]
