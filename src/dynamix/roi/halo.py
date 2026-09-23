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

``a_min < 1`` is refused outright (see :func:`run_wtmm2d_roi`): below it the kernel is so
under-sampled that no affordable margin is honest.

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

import math

import numpy as np

__all__ = ["MARGIN_PER_SCALE", "SIGMA_PER_A_MIN", "SIGMA_PER_SCALE", "halo_margin",
           "read_halo_window", "run_wtmm2d_roi", "run_wtmm2d_roi_on_window", "sigma_px_amin",
           "sigma_px_scale"]

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

#: ``a_min`` below this makes the sampled kernel non-local (see the module docstring).
MIN_A_MIN = 1.0

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


def _read_halo(source, parent_rect, roi, margin, nodata):
    """``roi + margin``, clamped to ``parent_rect``, nodata zero-filled, deficit reflect-padded.

    ``parent_rect`` is ``(row_off, col_off, height, width)`` of the sub-rect of ``source`` that
    counts as "the parent": the whole file for ``boundary="auto"``, the ROI itself for
    ``"reflective"``. ``roi`` is in the file's own row/col coordinates.
    """
    pr0, pc0, ph, pw = (int(v) for v in parent_rect)
    r0, c0, h, w = (int(v) for v in roi)
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
    # BOEM-style undeclared fill (2026-09-22): the tifs declare nodata = 0.0 but fill with
    # float32-lowest -- |v| >= 3e38 is nodata whatever the header says (the rule both other
    # read paths, from_geotiff_window and the picture, apply at read).
    missing |= np.abs(values) >= 3e38
    n_missing = int(missing.sum())
    if n_missing:
        values = np.where(missing, 0.0, values)           # zero-fill BEFORE the padding reflects it

    window = _reflect_pad(values, top, bottom, left, right)
    missing = _reflect_pad(missing.astype(np.uint8), top, bottom, left, right).astype(bool)
    # Zero-filled cells are fabricated too -- they are not "real data" for the honesty reading.
    real = (read_r1 - read_r0) * (read_c1 - read_c0) - n_missing
    reflected = {"N": top > 0, "S": bottom > 0, "W": left > 0, "E": right > 0}
    info = {
        "real_frac": real / float(win_h * win_w),
        "reflected_edges": tuple(e for e in _EDGES if reflected[e]),
        "offset_in_window": (margin, margin),
        "missing": missing,                               # extra: the ROI's mask comes from here
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


def _crop_to_roi(extrema: dict, offset: int, h: int, w: int) -> dict:
    """Keep the extrema inside the ROI sub-rect; shift their positions onto the ROI grid.

    The subpixel channels (``x_sub``/``y_sub``, present when the ``interpolate`` knob is on)
    shift by the same offset -- membership is decided by the INTEGER support, so the float
    position of a kept point may legitimately sit up to half a pixel outside the ROI edge."""
    x, y = extrema["x"], extrema["y"]
    keep = (x >= offset) & (x < offset + w) & (y >= offset) & (y < offset + h)
    out = {"x": x[keep] - offset, "y": y[keep] - offset, "mod": extrema["mod"][keep],
           "arg": extrema["arg"][keep], "line_id": extrema["line_id"][keep]}
    if "x_sub" in extrema:
        out["x_sub"] = extrema["x_sub"][keep] - offset
        out["y_sub"] = extrema["y_sub"][keep] - offset
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
        The WTMM2D param contract, resolved by the backend's own typo guard. ``a_min`` must be at
        least :data:`MIN_A_MIN`.
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
        ``_coi_radii`` (per-scale halo widths, for the canvas's contamination outlines).
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

    resolved = _resolve_wtmm2d_params(params)
    if resolved["mode"] != "scalar":
        raise ValueError(
            f"run_wtmm2d_roi: mode={resolved['mode']!r} is not supported; the ROI halo path is "
            "scalar-only (the tensor pipeline is monolithic and has no per-scale seam)")
    if resolved["a_min"] < MIN_A_MIN:
        raise ValueError(
            f"run_wtmm2d_roi: a_min={resolved['a_min']} is below {MIN_A_MIN}. The sampled wavelet "
            "at that scale is close to all-pass at Nyquist, so its influence reaches hundreds of "
            "pixels (measured: 368 px to the 1e-3 level at a_min=0.25) and no affordable halo "
            "makes the ROI's answer match a whole-raster one. Analyse the full raster instead.")

    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    # "reflective" IS "auto" against a parent that stops at the ROI's own edges.
    parent_rect = ((0, 0, int(full_dims[0]), int(full_dims[1])) if boundary == "auto"
                   else (r0, c0, h, w))

    def _window_for(margin):
        return _read_halo(source, parent_rect, (r0, c0, h, w), margin, nodata)

    (extrema, chains, hd_std, hd_cmax, roi_margins, missing_mask,
     coi_radii) = _wtmm2d_roi_core(_window_for, h, w, resolved, scales, progress)

    return {
        "chains": chains, "extrema": extrema, "scales": scales,
        "hd_std": hd_std, "hd_cmax": hd_cmax, "npz_path": None,
        "params": resolved, "cache_hits": set(),
        "_shape": (h, w),
        "_roi": {"source": str(source), "roi": (r0, c0, h, w), "boundary": boundary},
        "_roi_margins": roi_margins,
        "_missing_mask": missing_mask,
        "_coi_radii": coi_radii,
    }


def _wtmm2d_roi_core(window_for, h, w, resolved, scales, progress=None):
    """The per-scale halo loop shared by :func:`run_wtmm2d_roi` (halos read off the FILE) and
    :func:`run_wtmm2d_roi_on_window` (halos sliced from an in-memory window): for each scale,
    ``window_for(margin) -> (window, info)`` supplies ROI + that scale's halo (zero-filled
    nodata, reflect-padded deficit) with ``info`` = ``real_frac`` / ``reflected_edges`` /
    ``missing``; the CWT + extrema run on it, the extrema are cropped to the ROI, and only
    THEN are they chained and partitioned -- chaining happens inside the ROI.

    Extracted verbatim from :func:`run_wtmm2d_roi` so both
    entries run one implementation; ``tests/test_roi_halo.py`` is the file path's oracle,
    ``tests/test_roi_wtmm_window.py`` the in-memory path's."""
    from dynamix.core.wtmm_backend import _apply_fracint2d, get_backend

    backend = get_backend("python")

    n_scales = len(scales)
    extrema, roi_margins, coi_radii = [], [], []
    missing_mask = None
    for i, a in enumerate(scales):
        if progress is not None:
            progress(f"roi scale {i + 1}/{n_scales}", 0.0)
        margin = halo_margin(a)
        window, info = window_for(margin)
        if missing_mask is None:
            missing_mask = info["missing"][margin:margin + h, margin:margin + w].copy()
        follow = resolved["detector"] == "follow"
        cwt = backend.cwt2d(window, [a], wavelet=resolved["wavelet"], pad=resolved["pad"],
                            derivs="all" if follow else "first")
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
            # 2026-09-22 exact-port swap, matching the full run's extrema stage
            # (dynamix.core.xsmurf_follow, parity-proven -- see wtmm_backend's own swap).
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
        extrema.append(_crop_to_roi(scale_extrema, margin, h, w))
        roi_margins.append({"a": float(a), "margin": margin, "real_frac": info["real_frac"],
                            "reflected_edges": info["reflected_edges"]})
        coi_radii.append(margin)
        if progress is not None:
            progress(f"roi scale {i + 1}/{n_scales}", 1.0)

    chains = backend.chains2d(extrema, scales, similitude=resolved["similitude"],
                              box_ratio=resolved["box_ratio"], dist2_max=resolved["dist2_max"],
                              min_len=resolved["min_chain_len"], smooth=resolved["smooth"])
    hd_std, hd_cmax = backend.partition2d(chains, scales, q_list=resolved["q_list"],
                                          min_chain_len=resolved["min_chain_len"])

    return extrema, chains, hd_std, hd_cmax, roi_margins, missing_mask, coi_radii


def run_wtmm2d_roi_on_window(window_values, real, offset: int, h: int, w: int, params: dict,
                             *, progress=None) -> dict:
    """:func:`run_wtmm2d_roi` over an IN-MEMORY processing window.

    ``window_values`` is ROI + ``offset`` px on every side, already read by
    :func:`dynamix.roi.runner.read_processing_window` (real data where the file has it, the
    deficit reflect-padded, nodata as NaN); ``real`` marks the genuine pixels. Each scale's
    halo is the centred sub-rect ``[offset - m, offset + h + m)`` -- bit-identical to what
    ``_read_halo`` reads off the file (the union-window claim ``test_roi_halo`` proves) -- with
    NaN zero-filled BEFORE the transform (this device's convention, reported in
    ``_missing_mask``/``real_frac``, never interpolated). ``offset`` must cover the coarsest
    scale's halo. Returns the same keys as :func:`run_wtmm2d_roi` (``_roi`` without a source:
    the caller stamps the region)."""
    from dynamix.core.wtmm_backend import _resolve_wtmm2d_params, compute_scales2d

    if h < MIN_ROI_SIDE or w < MIN_ROI_SIDE:
        raise ValueError(
            f"wtmm2d on an ROI: the ROI is {h}x{w} px; both sides must be at least "
            f"{MIN_ROI_SIDE} px (a sliver carries no multi-scale structure).")
    resolved = _resolve_wtmm2d_params(params)
    if resolved["mode"] != "scalar":
        raise ValueError(
            f"wtmm2d on an ROI: mode={resolved['mode']!r} is not supported; the per-scale halo "
            "path is scalar-only (the tensor pipeline has no per-scale seam)")
    if resolved["a_min"] < MIN_A_MIN:
        raise ValueError(
            f"wtmm2d on an ROI: a_min={resolved['a_min']} is below {MIN_A_MIN} -- the sampled "
            "wavelet is close to all-pass at Nyquist and no affordable halo is honest. Raise "
            "a_min or analyse the whole raster.")
    scales = compute_scales2d(resolved["n_oct"], resolved["n_voice"], resolved["a_min"])
    need = halo_margin(max(scales))
    if int(offset) < need:
        raise ValueError(f"window margin {offset} px < the coarsest scale's halo {need} px")
    values = np.asarray(window_values, dtype=np.float64)
    real = np.asarray(real, dtype=bool)
    o = int(offset)

    def _window_for(margin):
        sl = (slice(o - margin, o + h + margin), slice(o - margin, o + w + margin))
        sub, rsub = values[sl], real[sl]
        missing = ~np.isfinite(sub)
        win = np.where(missing, 0.0, sub) if missing.any() else sub.copy()
        edges = {"N": not rsub[0].any(), "S": not rsub[-1].any(),
                 "W": not rsub[:, 0].any(), "E": not rsub[:, -1].any()}
        return win, {"real_frac": float((rsub & ~missing).mean()),
                     "reflected_edges": tuple(e for e in _EDGES if edges[e]),
                     "missing": missing}

    (extrema, chains, hd_std, hd_cmax, roi_margins, missing_mask,
     coi_radii) = _wtmm2d_roi_core(_window_for, h, w, resolved, scales, progress)
    return {
        "chains": chains, "extrema": extrema, "scales": scales,
        "hd_std": hd_std, "hd_cmax": hd_cmax, "npz_path": None,
        "params": resolved, "cache_hits": set(),
        "_shape": (h, w),
        "_roi": {"source": None, "roi": None, "boundary": "auto"},
        "_roi_margins": roi_margins,
        "_missing_mask": missing_mask,
        "_coi_radii": coi_radii,
    }
