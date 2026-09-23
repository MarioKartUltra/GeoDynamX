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

**The ``a_min`` floor:** the engine refuses ``a_min < 1.0`` outright (the
sampled kernel is close to all-pass at Nyquist below that, so no affordable halo margin is honest
-- see ``dynamix.roi.halo``'s module docstring). The Param below sets its hard ``min`` at that same
floor so the knob itself cannot ask for a value the engine would reject; ``wtmm2d``'s own ``a_min``
keeps its wider floor (0.25) because the whole-raster path has no halo to contaminate.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind
from dynamix.roi.halo import MIN_A_MIN, MIN_ROI_SIDE


class WTMM2DROI:
    """Wavelet Transform Modulus Maxima over one ROI of a GeoTIFF, haloed per scale.

    Wraps :func:`dynamix.roi.halo.run_wtmm2d_roi`. The source raster is not the field passed to
    ``compute`` -- it is read straight from ``field.provenance["source"]``, because the ROI path
    reads its own per-scale windows off disk rather than operating on an already-materialised
    array. A field with no recorded source (e.g. loaded from a bare ``.npz``) cannot drive this
    device: there is nothing on disk to halo against.
    """

    name = "wtmm2d_roi"
    #: Reads its own pixels off ``field.provenance["source"]`` (never ``field.values``), so the
    #: engine lets it run over a display picture -- it is region-aware.
    reads_source = True
    #: Same split as ``WTMM2D.param_groups`` (dynamix/devices/wtmm.py), for the same reason --
    #: duplicated rather than imported, matching this file's existing "self-contained device
    #: declaration" convention. The ROI/boundary knobs are neither PRE nor CHAINING (they select a
    #: WINDOW, not a wavelet or a linking rule), so they land in the trailing unlabeled group
    #: ``workflow_zone.DeviceBox`` builds for params no entry names.
    param_groups = {
        "PRE": ("wavelet", "a_min", "n_oct", "n_voice", "smooth", "fracint_alpha",
                "interpolate", "detector"),
        "CHAINING": ("min_chain_len", "thresh", "dist2_max", "box_ratio", "similitude"),
    }
    params = (
        # The first four mirror wtmm2d's declarations exactly (dynamix/devices/wtmm.py) -- same
        # defaults, bounds and labels, duplicated rather than imported so this file stays a
        # self-contained device declaration.
        Param("n_oct", ParamKind.INT, default=3, min=1, max=12, label="Octaves"),
        Param("n_voice", ParamKind.INT, default=4, min=1, max=64, label="Voices/octave"),
        # a_min differs from wtmm2d's: hard min raised to MIN_A_MIN (1.0), the floor
        # run_wtmm2d_roi enforces itself (contaminated-numbers regime -- see the module
        # docstring). soft_min follows it up so the declaration stays internally consistent
        # (Param.__post_init__ rejects a soft_min below the hard min).
        Param("a_min", ParamKind.FLOAT, default=1.0, min=MIN_A_MIN, max=16.0,
              soft_min=MIN_A_MIN, soft_max=4.0, units="", label="aₘᵢₙ"),
        Param("wavelet", ParamKind.CHOICE, default="mexican",
              choices=("mexican", "gaussian"), label="Wavelet"),
        # Mirrors wtmm2d's declaration (2026-09-20 fix): the ROI path always APPLIED the lift
        # (``run_wtmm2d_roi`` reads ``resolved["fracint_alpha"]``) but the knob was neither
        # declared nor carried, so a parent tuned off the default silently lost it on every ROI
        # child -- the exact class of bug an earlier fix closed for the other five params.
        Param("fracint_alpha", ParamKind.FLOAT, default=1.0, min=-6.0, max=6.0,
              soft_min=0.0, soft_max=2.0, units="", label="Frac. int. η"),
        # Mirrors wtmm2d's 2026-09-20 subpixel/parity knob -- an ROI child must run the same
        # refinement its parent ran, or the two are different analyses (the shared-params law).
        Param("interpolate", ParamKind.BOOL, default=False, label="Interpolate"),
        # Mirrors wtmm2d's 2026-09-21 detector knob (nms / follow) -- shared-params law.
        Param("detector", ParamKind.CHOICE, default="nms", choices=("nms", "follow"),
              label="Detector"),
        Param("min_chain_len", ParamKind.INT, default=2, min=1, max=1024,
              soft_min=2, soft_max=16, label="Min chain length"),
        # The five exposed here for the first time -- mirror wtmm2d's own declarations
        # (see that file for the backend-default verification, and for ``similitude``'s bound reasoning). ``run_wtmm2d_roi`` resolves them through the
        # identical ``_resolve_wtmm2d_params`` contract wtmm2d uses.
        Param("smooth", ParamKind.BOOL, default=True, label="Smooth"),
        Param("thresh", ParamKind.FLOAT, default=1e-3, min=0.0, max=1.0,
              soft_min=0.0, soft_max=0.05, units="", label="Thresh"),
        Param("dist2_max", ParamKind.FLOAT, default=50.0, min=0.0, max=1000000.0,
              soft_min=10.0, soft_max=150.0, units="", label="Dist² max"),
        Param("box_ratio", ParamKind.FLOAT, default=1.0, min=0.001, max=64.0,
              soft_min=0.5, soft_max=2.0, units="", label="Box ratio"),
        Param("similitude", ParamKind.FLOAT, default=0.8, min=0.0, max=1.0,
              soft_min=0.5, soft_max=0.95, units="", label="Similitude"),
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
    )

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
            "fracint_alpha": params["fracint_alpha"],
            "interpolate": params["interpolate"],
            "detector": params["detector"],
            "min_chain_len": params["min_chain_len"],
            "smooth": params["smooth"],
            "thresh": params["thresh"],
            "dist2_max": params["dist2_max"],
            "box_ratio": params["box_ratio"],
            "similitude": params["similitude"],
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
        return res

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        """The sigma line beside the ``aₘᵢₙ`` knob, now with the analyzing wavelet's bandpass
        wavelength too. Duplicated from ``WTMM2D.derived_reading`` (dynamix/devices/wtmm.py --
        see that docstring for the sigma/lambda reasoning) rather than imported -- it is a dozen
        lines with no shared state to drift out of sync, and importing a method off a sibling
        device for its side-effect-free body would be a stranger dependency than repeating it."""
        if name != "a_min":
            return None
        from dynamix.core.scale_units import lambda_peak_px, sigma_px
        from dynamix.core.wtmm_backend import compute_scales2d
        from dynamix.shell.units import px_to_metres

        finest = compute_scales2d(1, 1, a_min=float(value))[0]
        sigma = sigma_px(finest)
        wavelet = (params or {}).get("wavelet", "mexican")     # this device's own param default
        lam = lambda_peak_px(finest, wavelet, 1)
        px, unit = px_to_metres(field)
        if px is None or unit == "px":
            return f"σ ≈ {sigma:.1f} px · λ {lam:.1f} px"
        return (f"σ ≈ {sigma:.1f} px ≈ {sigma * px:.3g} {unit}"
                f" · λ {lam:.1f} px ≈ {lam * px:.3g} {unit}")
