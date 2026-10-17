# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The wtmm2d_roi transform: WTMM over one region of interest of a huge parent raster.

A sibling of ``wtmm2d`` (:mod:`dynamix.devices.wtmm`), not a replacement -- this device exists so
a layer can analyse a window of a raster too large to load whole, without lying at the window's
edges. It delegates every bit of the actual halo/crop reasoning to
:func:`dynamix.roi.halo.run_wtmm2d_roi`; see that module's docstring for the margin budget, the
boundary modes and the missing-data convention.

**Thresholding:** ``thresh`` is a FRACTION of the ANALYSED window's
own modulus max, not the parent raster's -- and the analysed window is ``ROI ⊕ halo``, a different
extent per scale. An ROI's extrema threshold is therefore always relative to its own window, never
to the parent's global maximum: the same field can report differently "loud" extrema depending on
where the box is drawn. This is inherent to relative thresholding over a windowed read, not a bug
to fix here.

**Where a scale's reach does not converge** (the sampled kernel rings at Nyquist; for the Mexican hat
below about ``a_min`` 1.0, for heavier-tailed smoothers higher), no finite halo is honest
(``dynamix.roi.halo``'s module docstring): the scale reads the whole parent when it fits the budget,
and otherwise the run carries ``halo.honest_halo_warning`` in ``_warnings``; the strip shows it on the
``a_min`` knob before the run (:meth:`WTMM2DROI.knob_warnings`), with the under-sampling warning
``wtmm2d`` raises. The knob reaches the whole-raster floor.
"""
from __future__ import annotations

from dynamix.model.contracts import EXTREMA, FIELD, STACK, VCHAINS
from dynamix.model.origin import XSMURF_DERIVED
from dynamix.model.device import Output, keyed_params
from dynamix.devices.wtmm import (BAND_PER_PARAM, CHAINING_OPTION_PARAMS, EDGE_PARAMS,
                                  SMOOTHER_PARAMS, chaining_reading, eta_warning,
                                  smoother_params, width_reading)
from dynamix.devices.precision import PRECISION_PARAM
from dynamix.model.param import Param, ParamKind
from dynamix.roi.halo import MIN_ROI_SIDE, halo_plan, honest_halo_warning


def _halo_warning(field, params: dict) -> "str | None":
    """The honest-halo warning for these params before the run: the finest scales up to the first
    whose reach converges (the coarser ones converge too), against the parent the run would read."""
    from dynamix.core.wtmm_backend import compute_scales2d

    a_min = float(params.get("a_min", 1.0))
    scales = compute_scales2d(int(params.get("n_oct", 3)), int(params.get("n_voice", 4)), a_min)
    full = (getattr(field, "provenance", None) or {}).get("full_dims")
    if params.get("boundary") == "reflective":
        full = (params.get("roi_h", 512), params.get("roi_w", 512))
    try:
        plan = halo_plan(scales, params.get("wavelet", "mexican"), smoother_params(params),
                         float(params.get("edge_eps", 1e-3)), full, until_converged=True)
    except ValueError:                           # a q the smoother refuses: the run says so
        return None
    return honest_halo_warning(plan)


def _taper_warning(field, params: dict) -> "str | None":
    """Why the Gaussian taper would fall back to reflection on this ROI, read off the source."""
    if params.get("edge") != "gaussian_taper":
        return None
    prov = getattr(field, "provenance", None) or {}
    source, full = prov.get("source"), prov.get("full_dims")
    if not source or full is None:
        return None
    import numpy as np

    from dynamix.roi.halo import _read_halo, taper_fallback

    roi = tuple(int(params.get(k, d)) for k, d in (("roi_row", 0), ("roi_col", 0),
                                                    ("roi_h", 512), ("roi_w", 512)))
    parent = ((0, 0) + tuple(full) if params.get("boundary", "auto") == "auto" else roi)
    try:
        block, info = _read_halo(source, parent, roi, 0, None)
    except (OSError, ValueError):
        return None
    return taper_fallback(np.where(info["missing"], np.nan, block), parent[2:], params,
                          smoother_params(params))


def _corner_warning(field, params: dict) -> "str | None":
    """The reflection warning before the run: the scales whose halo runs past the parent at this
    ROI (``dynamix.core.edges``). The coarsest halo is the widest; an ROI it fits needs no more."""
    from dynamix.core import edges
    from dynamix.core.wtmm_backend import compute_scales2d

    full = (getattr(field, "provenance", None) or {}).get("full_dims")
    roi = tuple(int(params.get(k, d)) for k, d in (("roi_row", 0), ("roi_col", 0),
                                                    ("roi_h", 512), ("roi_w", 512)))
    if params.get("boundary") == "reflective":
        full, roi = roi[2:], (0, 0) + roi[2:]
    if full is None or params.get("edge", "reflect") != "reflect":
        return None
    scales = compute_scales2d(int(params.get("n_oct", 3)), int(params.get("n_voice", 4)),
                              float(params.get("a_min", 1.0)))
    args = (params.get("wavelet", "mexican"), smoother_params(params),
            float(params.get("edge_eps", 1e-3)), full)
    try:
        top = halo_plan(scales[-1:], *args)[0]
        if not top.whole and not any(edges.deficits(roi, full, top.margin).values()):
            return None
        plan = halo_plan(scales, *args)
    except ValueError:                           # a q the smoother refuses: the run says so
        return None
    return edges.corner_warning([(sh.a, edges.deficits(roi, full, sh.margin))
                                 for sh in plan if not sh.whole], "reflect")


class WTMM2DROI:
    """Wavelet Transform Modulus Maxima over one ROI of a GeoTIFF, haloed per scale.

    Wraps :func:`dynamix.roi.halo.run_wtmm2d_roi`. The source raster is not the field passed to
    ``compute`` -- it is read straight from ``field.provenance["source"]``, because the ROI path
    reads its own per-scale windows off disk rather than operating on an already-materialised
    array. A field with no recorded source (e.g. loaded from a bare ``.npz``) cannot drive this
    device: there is nothing on disk to halo against.
    """

    name = "wtmm2d_roi"
    capabilities = frozenset({"OutputProducer", "SourceReader", "StripExtras", "Viewable"})
    chain_producer = True
    label = "WTMM 2D, ROI"
    origin = XSMURF_DERIVED
    families = ("Edge detection", "Regularity")
    #: It reads band 1 of its source file, whatever field it is handed, so it runs on a
    #: dataset row's band stack as well as on a single-band field.
    consumes = ((FIELD,), (STACK,))
    produces = (EXTREMA, VCHAINS)
    #: Reads its own pixels off ``field.provenance["source"]`` (never ``field.values``), so the
    #: engine lets it run over a display picture -- it is region-aware.
    reads_source = True
    #: Same split as ``WTMM2D.param_groups`` (dynamix/devices/wtmm.py), for the same reason --
    #: duplicated rather than imported, matching this file's existing "self-contained device
    #: declaration" convention. The ROI/boundary knobs are neither PRE nor CHAINING (they select a
    #: WINDOW, not a wavelet or a linking rule), so they land in the trailing unlabeled group
    #: ``workflow_zone.DeviceBox`` builds for params no entry names.
    param_groups = {
        "PRE": ("wavelet", "a_min", "alpha", "q_tsallis", "q_pairing", "q_beta", "n_oct",
                "n_voice", "smooth", "fracint_alpha", "interpolate", "detector"),
        "CHAINING": ("min_chain_len", "thresh", "dist2_max", "box_ratio", "similitude",
                     "band_per"),
    }
    params = (
        # The first four mirror wtmm2d's declarations exactly (dynamix/devices/wtmm.py) -- same
        # defaults, bounds and labels, duplicated rather than imported so this file stays a
        # self-contained device declaration.
        Param("n_oct", ParamKind.INT, default=3, min=1, max=12, label="Octaves"),
        Param("n_voice", ParamKind.INT, default=4, min=1, max=64, label="Voices/octave"),
        # a_min reaches wtmm2d's floor; below 1.0 the run carries the honest-halo warning (see
        # the module docstring).
        Param("a_min", ParamKind.FLOAT, default=1.0, min=0.01, max=16.0,
              soft_min=0.5, soft_max=4.0, units="", label="aₘᵢₙ"),
        # The smoother θ and its parameters: wtmm2d's own declarations (the registry's names
        # must match between parent and ROI child, so these are shared rather than mirrored).
        *SMOOTHER_PARAMS,
        # Mirrors wtmm2d's declaration: the ROI path APPLIES the lift (``run_wtmm2d_roi`` reads
        # ``resolved["fracint_alpha"]``), so the knob must be declared and carried here, or a
        # parent tuned off the default silently loses it on every ROI child. Default 0, as wtmm2d.
        Param("fracint_alpha", ParamKind.FLOAT, default=0.0, min=-6.0, max=6.0,
              soft_min=0.0, soft_max=2.0, units="", label="Frac. int. η"),
        # Mirrors wtmm2d's subpixel/parity knob -- an ROI child must run the same
        # refinement its parent ran, or the two are different analyses (the shared-params law).
        Param("interpolate", ParamKind.BOOL, default=False, label="Interpolate"),
        # Mirrors wtmm2d's detector knob (nms / follow) -- shared-params law.
        Param("detector", ParamKind.CHOICE, default="nms", choices=("nms", "follow"),
              label="Detector"),
        Param("min_chain_len", ParamKind.INT, default=2, min=1, max=1024,
              soft_min=2, soft_max=16, label="Min chain length"),
        # These five mirror wtmm2d's own declarations (see that file for the backend-default
        # verification, and for ``similitude``'s bound reasoning). ``run_wtmm2d_roi`` resolves
        # them through the identical ``_resolve_wtmm2d_params`` contract wtmm2d uses.
        Param("smooth", ParamKind.BOOL, default=True, label="Smooth"),
        Param("thresh", ParamKind.FLOAT, default=1e-3, min=0.0, max=1.0,
              soft_min=0.0, soft_max=0.05, units="", label="Thresh"),
        Param("dist2_max", ParamKind.FLOAT, default=50.0, min=0.0, max=1000000.0,
              soft_min=10.0, soft_max=150.0, units="", label="Dist² max"),
        Param("box_ratio", ParamKind.FLOAT, default=1.0, min=0.001, max=64.0,
              soft_min=0.5, soft_max=2.0, units="", label="Box ratio"),
        Param("similitude", ParamKind.FLOAT, default=0.8, min=0.0, max=1.0,
              soft_min=0.5, soft_max=0.95, units="", label="Similitude"),
        BAND_PER_PARAM,
        # The cross-scale chainer, as on wtmm2d (not keyed at its default).
        Param("chainer", ParamKind.CHOICE, default="xsmurf_bench",
              choices=("xsmurf_bench", "gradient_arg_horizontal", "vertical_experimental"),
              label="Chainer", keyed_at_default=False),
        *CHAINING_OPTION_PARAMS,
        *EDGE_PARAMS,
        # The ROI + boundary knobs. min for roi_h/roi_w mirrors MIN_ROI_SIDE, the same floor
        # run_wtmm2d_roi enforces -- a sliver carries no multi-scale structure to measure.
        Param("roi_row", ParamKind.INT, default=0, min=0, max=1_000_000, label="ROI row"),
        Param("roi_col", ParamKind.INT, default=0, min=0, max=1_000_000, label="ROI col"),
        Param("roi_h", ParamKind.INT, default=512, min=MIN_ROI_SIDE, max=1_000_000,
              label="ROI height"),
        Param("roi_w", ParamKind.INT, default=512, min=MIN_ROI_SIDE, max=1_000_000,
              label="ROI width"),
        Param("boundary", ParamKind.CHOICE, default="auto", choices=("auto", "reflective"),
              label="Boundary"),
        # Mirrors wtmm2d's Show: the raster drawn in place of the field. View-only.
        Param("show", ParamKind.CHOICE, default="edges",
              choices=("edges", "modulus", "argument"), label="Show", view=True),
        PRECISION_PARAM,
    )

    #: wtmm2d's per-scale output rows, on the ROI's own grid.
    outputs = (Output("modulus", "raster", lazy=True, label="modulus", per_level=True),
               Output("argument", "raster", lazy=True, label="argument", per_level=True))

    def wavelet_profile(self, params: dict, scale_px: float, level: int):
        """The analysing wavelet's section and λ* at ``scale_px`` (the wavelet scale bar)."""
        from dynamix.core.wavelet_profiles import wtmm_profile

        return wtmm_profile(params["wavelet"], scale_px, smoother_params(params))

    def compute(self, field, params: dict, *, progress=None) -> dict:
        from dynamix.core.rasterfield import geotiff_info
        from dynamix.roi.halo import run_wtmm2d_roi

        provenance = getattr(field, "provenance", None) or {}
        source = provenance.get("source")
        if not source:
            raise ValueError(
                "wtmm2d_roi: field.provenance has no 'source' to halo against. An npz-loaded "
                "field carries no path back to a parent raster, so there is nothing on disk to "
                "read a per-scale halo from -- load this layer from its GeoTIFF instead."
            )
        # dynamix.shell.opening stamps full_dims on every field it opens (both the whole-file and
        # windowed routes) -- consuming that here saves an info read per compute. A field that
        # reached this device some other way (no stamp) falls back to reading it off the file.
        stamped = provenance.get("full_dims")
        if stamped is not None:
            full_dims = (int(stamped[0]), int(stamped[1]))
        else:
            info = geotiff_info(source)
            full_dims = (info["height"], info["width"])
        roi = (params["roi_row"], params["roi_col"], params["roi_h"], params["roi_w"])
        wtmm_params = {
            "n_oct": params["n_oct"],
            "n_voice": params["n_voice"],
            "a_min": params["a_min"],
            "wavelet": params["wavelet"],
            "smoother_params": smoother_params(params),
            "fracint_alpha": params["fracint_alpha"],
            "interpolate": params["interpolate"],
            "detector": params["detector"],
            "min_chain_len": params["min_chain_len"],
            "smooth": params["smooth"],
            "thresh": params["thresh"],
            "dist2_max": params["dist2_max"],
            "box_ratio": params["box_ratio"],
            "similitude": params["similitude"],
            "band_per": params["band_per"],
            "chainer": params["chainer"],
            **{p.name: params[p.name] for p in CHAINING_OPTION_PARAMS},
            **{p.name: params[p.name] for p in EDGE_PARAMS},
        }
        res = run_wtmm2d_roi(source, full_dims, roi, wtmm_params,
                             boundary=params["boundary"], progress=progress)
        # Carry the field's own frame through, same as wtmm2d.compute -- filters and views convert
        # pixels to physical units without reaching back to the source. Unlike wtmm2d, _shape is
        # NOT reset here: run_wtmm2d_roi already set it to the ROI's own (h, w), and the parent
        # field's shape would be wrong (it is the whole raster, not the analysed window).
        res = dict(res)
        res["_frame"] = getattr(field, "frame", None)
        # The draw-ready CSR chain product + ordering runs, stamped
        # worker-side exactly as wtmm2d.compute does -- over the ROI's own _shape, which
        # run_wtmm2d_roi already set to the analysed window's (h, w).
        from dynamix.core.chain_product import attach_chain_product
        attach_chain_product(res)
        warning = eta_warning(params)
        if warning:
            res["_warnings"] = list(res.get("_warnings", [])) + [warning]
        return res

    def knob_warnings(self, field, params: dict) -> dict:
        """The ``a_min`` knob's warnings, for the strip's ⚠: the honest-halo limit where the
        finest scales' reach does not converge and the parent (``field.provenance["full_dims"]``,
        unknown without a field) is too large to read whole, and the under-sampling of the finest
        scale (``wtmm.eta_warning``)."""
        params = params or {}
        found = [w for w in (_halo_warning(field, params), _corner_warning(field, params),
                             eta_warning(params)) if w]
        out = {"a_min": "\n\n".join(found)} if found else {}
        reason = _taper_warning(field, params)
        if reason:
            out["edge"] = reason
        return out

    def view(self, result: dict, params: dict) -> dict:
        """``show`` picks a per-scale output, which the window computes and caches under its
        own key; the analysis arrays are shown as they are."""
        return result

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None, level=None) -> dict:
        """The per-scale output ``name`` (``modulus`` or ``argument``) of scale ``level``
        (0-based, finest first, clipped into the result's scales), read and transformed as the
        analysis read and transformed that scale (:func:`~dynamix.roi.halo.run_wtmm2d_roi_level`
        over the ROI and boundary the result records, the scale's own halo off the source
        file). ``values`` is not read: this device reads its pixels off the file. ``roi_axes``
        are the ROI's own ``(x_axis, y_axis)``, which place the raster where the region sits."""
        import numpy as np

        from dynamix.core.rasterfield import RasterField, geotiff_info
        from dynamix.core.wtmm_backend import ComputeCancelled
        from dynamix.roi.halo import run_wtmm2d_roi_level

        if name not in ("modulus", "argument"):
            raise ValueError(f"{self.name}: no lazy output named {name!r}")
        if cancel is not None and cancel():
            raise ComputeCancelled()
        roi = result["_roi"]
        source = roi["source"]
        r0, c0, h, w = (int(v) for v in roi["roi"])
        info = geotiff_info(source)
        n = len(result["scales"])
        idx = int(np.clip(0 if level is None else int(level), 0, n - 1))
        if progress is not None:
            progress(f"{name} scale {idx + 1}/{n}", 0.0)
        one = run_wtmm2d_roi_level(
            source, (info["height"], info["width"]), (r0, c0, h, w),
            dict({k: params[k] for k in ("n_oct", "n_voice", "a_min", "wavelet",
                                         "fracint_alpha", "detector")},
                 smoother_params=smoother_params(params)),
            idx, boundary=roi["boundary"])
        grid = RasterField.from_geotiff_window(source, row_off=r0, col_off=c0, height=h,
                                               width=w)
        if progress is not None:
            progress(f"{name} scale {idx + 1}/{n}", 1.0)
        raster = one["mod"] if name == "modulus" else one["arg"]
        return {"raster": np.asarray(raster, dtype=np.float32),
                "diag": {"level": idx, "scale_px": one["scale"]},
                "roi_axes": (np.asarray(grid.x_axis), np.asarray(grid.y_axis))}

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, keyed_params(self, params))

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        """The sigma line beside the ``aₘᵢₙ`` knob, now with the analyzing wavelet's bandpass
        wavelength too. Duplicated from ``WTMM2D.derived_reading`` (dynamix/devices/wtmm.py --
        see that docstring for the sigma/lambda reasoning) rather than imported -- it is a dozen
        lines with no shared state to drift out of sync, and importing a method off a sibling
        device for its side-effect-free body would be a stranger dependency than repeating it."""
        if name == "similitude":
            return chaining_reading(value, params or {})
        if name == "q_pairing":
            return width_reading(params or {})
        if name != "a_min":
            return None
        from dynamix.core.scale_units import lambda_peak_px, sigma_px
        from dynamix.core.wtmm_backend import compute_scales2d
        from dynamix.core.units import px_to_metres

        finest = compute_scales2d(1, 1, a_min=float(value))[0]
        sigma = sigma_px(finest)
        wavelet = (params or {}).get("wavelet", "mexican")     # this device's own param default
        lam = lambda_peak_px(finest, wavelet, 1, smoother_params(params or {}))
        px, unit = px_to_metres(field)
        if px is None or unit == "px":
            return f"σ ≈ {sigma:.1f} px · λ {lam:.1f} px"
        return (f"σ ≈ {sigma:.1f} px ≈ {sigma * px:.3g} {unit}"
                f" · λ {lam:.1f} px ≈ {lam * px:.3g} {unit}")
