# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The WTMM transform: the expensive one, and the reason everything downstream is cheap.

Runs the whole scale stack once and caches it. Every filter after this point is a lookup or a
predicate over the result, which is what makes the scale slider feel instant.
"""
from __future__ import annotations

import numpy as np

from dynamix.model.param import Param, ParamKind


class WTMM2D:
    """Wavelet Transform Modulus Maxima over a 2-D field.

    Wraps :func:`dynamix.core.wtmm_backend.run_wtmm2d`, which is copied verbatim from EQSelect.
    The scale stack it produces is ``n_oct * n_voice`` layers of extrema, each carrying ``x``,
    ``y``, ``mod`` and ``arg`` -- the argument being the orientation the wedge filter acts on.
    """

    name = "wtmm2d"
    #: Progressive compute: resolve() threads its ``cancel`` predicate into this device's
    #: ``compute`` because it opts in here. Reaches ``run_wtmm2d``'s per-stage check so a
    #: superseding edit kills the in-flight stack instead of waiting it out.
    wants_cancel = True
    #: PRE is everything that shapes the wavelet transform and its extrema
    #: (run once, expensive); CHAINING is everything that decides how extrema link into maxima
    #: lines ACROSS scales (cheap by comparison, but still part of this Transform -- xsmurf's own
    #: ``chains2d`` has no separate cache seam from the CWT/extrema stages the way a Filter would).
    #: ``workflow_zone.DeviceBox`` renders one labeled column-group per entry, in THIS param order
    #: -- not ``params`` tuple order -- so the pairing below (wavelet+a_min, n_oct+n_voice, smooth
    #: alone; min_chain_len+thresh, dist2_max+box_ratio, similitude alone) is a deliberate
    #: two-per-column layout, not an accident of dict iteration. ``similitude`` joins CHAINING:
    #: it is the third linking kwarg in the exact same ``chains2d`` call
    #: as ``box_ratio``/``dist2_max`` (dynamix/core/wtmm_backend.py), so a group named CHAINING that
    #: omitted it was incomplete, not merely conservative.
    param_groups = {
        "PRE": ("wavelet", "a_min", "n_oct", "n_voice", "smooth", "dither", "fracint_alpha",
                "interpolate", "detector"),
        "CHAINING": ("min_chain_len", "thresh", "dist2_max", "box_ratio", "similitude"),
    }
    params = (
        Param("n_oct", ParamKind.INT, default=3, min=1, max=12, label="Octaves"),
        Param("n_voice", ParamKind.INT, default=4, min=1, max=64, label="Voices/octave"),
        Param("a_min", ParamKind.FLOAT, default=1.0, min=0.01, max=64.0,
              soft_min=0.5, soft_max=4.0, units="", label="aₘᵢₙ"),
        Param("wavelet", ParamKind.CHOICE, default="mexican",
              choices=("mexican", "gaussian"), label="Wavelet"),
        # The M-Z devices' seeded half-LSB dither, on the WTMM path: it breaks up processing
        # noise before chaining. Applied by THIS device to a CLONE of the field (the backend is
        # a verbatim EQSelect copy and rejects unknown params); fails toward no-op when no
        # value lattice is measurable (core.mz_edges.measure_lsb's contract).
        Param("dither", ParamKind.BOOL, default=False, label="Dither (±½ LSB)"),
        # Pseudo-fractional integration order η (Wendt 2009; available
        # on the scalar path too, not only tensor) -- the per-scale a**η modulus lift applied
        # right after the cached cwt stage (``wtmm_backend._apply_fracint2d``). Default 1.0 is
        # the reference EBSD pipeline's own; 0 turns the lift off entirely. The spectrum window
        # seeds its η shift-back from this value, so what is lifted forward is undone at the fit.
        Param("fracint_alpha", ParamKind.FLOAT, default=1.0, min=-6.0, max=6.0,
              soft_min=0.0, soft_max=2.0, units="", label="Frac. int. η"),
        # Parabolic refinement of each NMS maximum along its gradient direction, for xsmurf
        # parity (dynamix.core.subpixel) -- the refined MODULUS replaces mod (the xsmurf
        # follow / LastWave-1D value channel, feeding chaining and Z(q,a)); float x_sub/y_sub
        # positions ride alongside the untouched integer support.
        # Default off = byte-identical pipeline.
        Param("interpolate", ParamKind.BOOL, default=False, label="Interpolate"),
        # The detection method. "nms" = bilinear non-maxima suppression (the historical DynamiX
        # behavior, xsmurf's wtmm2d/Malandain path); "follow" = kappa zero-crossing with the
        # kappa'<0 gate (xsmurf's scalar-2D historical default, dynamix.core.follow2d).
        # Follow natively stamps x_sub/y_sub + crossing-interpolated moduli, so the
        # interpolate knob above is inert under it (keyed regardless -- k_edge precedent).
        Param("detector", ParamKind.CHOICE, default="nms", choices=("nms", "follow"),
              label="Detector"),
        Param("min_chain_len", ParamKind.INT, default=2, min=1, max=1024,
              soft_min=2, soft_max=16, label="Min chain length"),
        # The five exposed here for the first time. Defaults/bounds verified against the
        # actual backend kwargs they thread into (dynamix/core/wtmm_backend.py):
        # ``_nms_extrema_scale``'s own ``thresh: float = 1e-3`` (~line 521) and
        # ``_make_chain_adapter``'s ``box_ratio: float = 1.0, dist2_max: float = 50.0`` (~line
        # 821) -- both match the plan's stated defaults exactly, so no override was needed.
        # ``similitude``: default 0.8 is ``_WTMM2D_PARAM_DEFAULTS``'s own
        # value. Bounds follow ``chains2d``'s own math (dynamix/core/wtmm_backend.py ~1231-1232):
        # the band it defines is ``similitude < ratio < 1/similitude``, which is only a genuine
        # band for ``similitude`` strictly between 0 (no constraint -- ``hi_band`` degenerates to
        # ``inf``) and 1 (empty band); soft bounds sit close around the 0.8 default, matching
        # ``a_min``/``box_ratio``'s own convention of a narrow default slider span within a wide
        # hard range.
        Param("smooth", ParamKind.BOOL, default=True, label="Smooth"),
        Param("thresh", ParamKind.FLOAT, default=1e-3, min=0.0, max=1.0,
              soft_min=0.0, soft_max=0.05, units="", label="Thresh"),
        Param("dist2_max", ParamKind.FLOAT, default=50.0, min=0.0, max=1000000.0,
              soft_min=10.0, soft_max=150.0, units="", label="Dist² max"),
        Param("box_ratio", ParamKind.FLOAT, default=1.0, min=0.001, max=64.0,
              soft_min=0.5, soft_max=2.0, units="", label="Box ratio"),
        Param("similitude", ParamKind.FLOAT, default=0.8, min=0.0, max=1.0,
              soft_min=0.5, soft_max=0.95, units="", label="Similitude"),
    )

    def compute(self, field, params: dict, *, progress=None, cancel=None) -> dict:
        from dynamix.core.wtmm_backend import run_wtmm2d

        if params.get("dither"):
            # Deterministic (fixed-seed) half-LSB uniform dither, the same helpers the M-Z
            # transform uses. The clone gets a distinct NAME because wtmm_backend's stage cache
            # keys on field.name -- a dithered run must never reload the raw run's stages. The
            # ROI device does NOT offer this: it re-reads per-scale halo windows straight off
            # the source file, which the device cannot dither without touching the halo core.
            import dataclasses

            from dynamix.core.mz_edges import _dithered, measure_lsb
            lsb = measure_lsb(field.values)
            if lsb:
                field = dataclasses.replace(field, values=_dithered(field.values, lsb),
                                            name=f"{field.name}+dither")
        res = run_wtmm2d(
            field,
            {
                "n_oct": params["n_oct"],
                "n_voice": params["n_voice"],
                "a_min": params["a_min"],
                "wavelet": params["wavelet"],
                "min_chain_len": params["min_chain_len"],
                "smooth": params["smooth"],
                "thresh": params["thresh"],
                "dist2_max": params["dist2_max"],
                "box_ratio": params["box_ratio"],
                "similitude": params["similitude"],
                "fracint_alpha": params["fracint_alpha"],
                "interpolate": params["interpolate"],
                "detector": params["detector"],
            },
            progress=progress,
            cancel=cancel,
        )
        # Carry the field's own axes through so filters and views can convert pixels to physical
        # units without reaching back to the source.
        res = dict(res)
        res["_frame"] = getattr(field, "frame", None)
        res["_shape"] = tuple(field.values.shape)
        # H-line ordering, paid HERE on the worker: run on the MAIN thread, the _order_lines
        # walk would run at every landing -- and with a noise step upstream every run mints a
        # fresh result, so it would run every time, freezing the GUI for the whole walk. Stamped
        # per scale, cached with the result; ScaleSelect hands the selected scale's runs to the
        # canvas and the scene, which then never walk at landing.
        from dynamix.core.hlines import hline_runs
        shape = res["_shape"][:2]
        res["_hline_runs"] = [hline_runs(layer, shape) for layer in res["extrema"]]
        # The draw-ready CSR chain product: EQSelect's representation
        # stamped as the result's native form -- filters narrow index selections over its metric
        # table, views draw from it by concatenation, and the npz export writes the SAME bundle.
        # Reuses the runs stamped just above; still worker-side.
        from dynamix.core.chain_product import attach_chain_product
        attach_chain_product(res)
        return res

    # -- ROI runner ----------------------
    _ROI_KEYS = ("n_oct", "n_voice", "a_min", "wavelet", "fracint_alpha", "interpolate",
                 "detector", "min_chain_len", "smooth", "thresh", "dist2_max", "box_ratio",
                 "similitude")

    def roi_margin(self, params: dict) -> int:
        """The coarsest scale's halo (``halo_margin``, the measured 2.5 x normalized scale) --
        every finer scale's halo is a centred sub-rect of it."""
        from dynamix.core.wtmm_backend import compute_scales2d
        from dynamix.roi.halo import halo_margin

        return halo_margin(max(compute_scales2d(params["n_oct"], params["n_voice"],
                                                params["a_min"])))

    def compute_roi(self, window_field, core, params: dict, *, info=None,
                    progress=None) -> dict:
        """wtmm on a region: each scale's halo sliced from the runner's window, extrema
        cropped to the ROI, THEN chained and partitioned inside it -- ``run_wtmm2d_roi``'s proven loop."""
        from dynamix.core.chain_product import attach_chain_product
        from dynamix.roi.halo import run_wtmm2d_roi_on_window

        r0, _c0, h, w = core
        values = np.asarray(window_field.values)
        real = info["real"] if info is not None else np.ones(values.shape[:2], dtype=bool)
        if params.get("dither"):
            # The whole-field path's deterministic half-LSB dither, applied to the window
            # (the knob was silently ignored on ROI runs). The lattice is
            # measured on the window's own pixels.
            from dynamix.core.mz_edges import _dithered, measure_lsb

            lsb = measure_lsb(values)
            if lsb:
                values = _dithered(values, lsb)
        res = run_wtmm2d_roi_on_window(values, real, int(r0),
                                       int(h), int(w), {k: params[k] for k in self._ROI_KEYS},
                                       progress=progress)
        res = dict(res)
        res["_frame"] = getattr(window_field, "frame", None)
        attach_chain_product(res)
        return res

    def preview(self, field, params: dict) -> dict:
        """Finest-scale-only result for the progressive preview: the wavelet transform +
        H-lines for the SINGLE finest scale, value-identical to this device's own scale-0
        layer, computed in ~1/n_scales the time. Skips ``dither`` (a preview is a
        provisional frame, not the analysed answer) and all chaining. ``engine.resolve.
        preview_resolve`` calls this then applies the chain's cheap filter steps on top."""
        from dynamix.core.wtmm_backend import run_wtmm2d_preview

        return run_wtmm2d_preview(field, {
            "n_oct": params["n_oct"], "n_voice": params["n_voice"], "a_min": params["a_min"],
            "wavelet": params["wavelet"], "thresh": params["thresh"],
            "min_chain_len": params["min_chain_len"], "smooth": params["smooth"],
            "dist2_max": params["dist2_max"], "box_ratio": params["box_ratio"],
            "similitude": params["similitude"], "fracint_alpha": params["fracint_alpha"],
            "interpolate": params["interpolate"], "detector": params["detector"],
        })

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        """The sigma line beside the ``aₘᵢₙ`` knob, now with the analyzing wavelet's bandpass
        wavelength too (``dynamix.core.scale_units``): sigma is smoother-sigma (wavelet-
        independent); lambda is per-wavelet -- the reason ``params`` joined the hook's signature.
        Lambda, not an outer-extremum footprint: the footprint metric orders the wavelets
        backwards (see ``scale_units``' docstring). ``None`` for every other param: a derived
        reading explains a knob whose face value is not itself physical, not a general per-param
        annotation mechanism."""
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
