# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The WTMM transform: the expensive one, and the reason everything downstream is cheap.

Runs the whole scale stack once and caches it. Every filter after this point is a lookup or a
predicate over the result, which is what makes the scale slider feel instant.
"""
from __future__ import annotations

import numpy as np

from dynamix.model.contracts import EXTREMA, FIELD, VCHAINS
from dynamix.model.device import Output, keyed_params
from dynamix.model.origin import XSMURF_DERIVED
from dynamix.devices import q_widths
from dynamix.devices.precision import PRECISION_PARAM
from dynamix.model.param import Param, ParamKind

#: The params a per-scale output reads: the scale grid, the wavelet, the lift and the detector
#: (which picks the cwt call the analysis made); the smoother's own parameters ride beside them
#: as the backend's ``smoother_params`` (:func:`smoother_params`).
_LEVEL_KEYS = ("n_oct", "n_voice", "a_min", "wavelet", "fracint_alpha", "detector")

#: The smoother knobs beside the ``wavelet`` choice: for each registry smoother outside the
#: Gaussian pair, (the device's param name, the registry's key).
_SMOOTHER_KNOBS = {"frac_bspline": ("alpha", "alpha"), "q_gaussian": ("q_tsallis", "q"),
                   "q_mexican": ("q_tsallis", "q"), "q_meyer": ("q_tsallis", "q")}

#: The q smoothers with a width β; the q-Meyer taper has none (q-paired, it holds its band peak).
_BETA_SMOOTHERS = ("q_gaussian", "q_mexican")

#: The smoother choice and its two parameters, shared with the ROI device. The choice names
#: the smoothing function θ; the analysing wavelets are its gradient components: g3 for the
#: Mexican hat, g1 for the Gaussian, ∂θ for the fractional B-spline (degree α) and the
#: q-Gaussian (Tsallis q), each parameter shown with its own smoother.
SMOOTHER_PARAMS = (
    Param("wavelet", ParamKind.CHOICE, default="mexican",
          choices=("mexican", "gaussian", "frac_bspline", "q_gaussian", "q_mexican", "q_meyer"),
          label="Smoother θ",
          choice_labels=("mexican (ψ = g3)", "gaussian (ψ = g1)",
                         "fractional B-spline (ψ = ∂θ_α)", "q-Gaussian (ψ = ∂θ_q)",
                         "q-Mexican hat (ψ = q-g3)", "q-Meyer (ψ = ∂θ_q)")),
    Param("alpha", ParamKind.FLOAT, default=3.0, min=1.0, max=8.0, soft_min=2.0, soft_max=4.0,
          units="", label="α (degree)", active_when=("wavelet", ("frac_bspline",))),
    # One q for the three q smoothers; the q-Mexican hat refuses q ≤ 2/3 at compute time (its
    # q-g3 would jump at the support's edge), the registry's own check.
    Param("q_tsallis", ParamKind.FLOAT, default=1.5, min=0.05, max=1.95, soft_min=0.5,
          soft_max=1.8, units="", label="q (Tsallis)",
          active_when=("wavelet", ("q_gaussian", "q_mexican", "q_meyer"))),
    # The q smoothers' two widths, fixed (default, today's kernels) or q-paired; out of the
    # key at their defaults, so every recipe keyed before them keeps its key.
    *q_widths.width_knobs(("q_gaussian", "q_mexican", "q_meyer"), beta_wavelets=_BETA_SMOOTHERS,
                          keyed_at_default=False),
)


def smoother_params(params: dict) -> dict | None:
    """The backend's ``smoother_params`` for the chosen θ: ``{"alpha": α}`` for the fractional
    B-spline, ``{"q": q}`` for the q smoothers, ``None`` for the Gaussian pair. A q width other
    than today's rides beside q: ``"beta"`` (the resolved β, when not ½) for the q-Gaussian and
    the q-Mexican hat, ``"pairing": "q-paired"`` for the q-Meyer taper (not at q = 1, where its
    band peak already sits there). Paired at q = 1 is fixed ½: the same dict, the same key."""
    wavelet = params.get("wavelet", "mexican")
    knob = _SMOOTHER_KNOBS.get(wavelet)
    if knob is None:
        return None
    name, key = knob
    default = next(p.default for p in SMOOTHER_PARAMS if p.name == name)
    out = {key: float(params.get(name, default))}
    if wavelet in _BETA_SMOOTHERS:
        resolved = dict(params, q_tsallis=out["q"])
        q_widths.q_width_check(resolved, _BETA_SMOOTHERS)
        beta = q_widths.q_width(resolved, _BETA_SMOOTHERS)
        if beta != 0.5:
            out["beta"] = beta
    elif wavelet == "q_meyer" and params.get("q_pairing") == "q-paired" and out["q"] != 1.0:
        out["pairing"] = "q-paired"
    return out


def eta_warning(params: dict) -> "str | None":
    """The ``a_min`` knob's under-sampling warning: the finest scale's η (the analysing wavelet's
    energy beyond Nyquist, :mod:`dynamix.core.undersampling`) against the configured level, or None
    while the grid resolves that scale."""
    from dynamix.core import undersampling
    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.core.wtmm_backend import compute_scales2d

    finest = compute_scales2d(1, 1, a_min=float(params.get("a_min", 1.0)))[0]
    wavelet = params.get("wavelet", "mexican")
    try:
        sp = smoother_params(params)
        eta = undersampling.eta(wavelet, sp, finest)
    except ValueError:                           # a q the smoother refuses: the run says so
        return None
    level = undersampling.eta_warn()
    if not eta > level:
        return None
    lam = lambda_peak_px(finest, wavelet, 1, sp)
    return (f"The finest scale's wavelet has η = {eta:.1e} of its energy beyond Nyquist "
            f"(λ* {lam:.1f} px): the grid does not resolve this scale, and its moduli carry the "
            f"pixel lattice (warning level {level:.0e}).")


def width_reading(params: dict) -> str:
    """The two lines beside the ``q width`` knob: the β the q smoother runs with and what the
    width holds -- fixed keeps β, so q moves the width with the shape; q-paired keeps the escort
    variance (the band peak for the q-Meyer taper), so q moves the shape alone. Empty, not None,
    while no q smoother is chosen, so the label exists and can light up later."""
    wavelet = params.get("wavelet", "mexican")
    paired = params.get("q_pairing", "fixed") == "q-paired"
    if wavelet == "q_meyer":
        return "band peak held" if paired else "q moves λ*"
    if wavelet not in _BETA_SMOOTHERS:
        return ""
    beta = smoother_params(params).get("beta", 0.5)
    return f"β {beta:.3g}\n{'escort variance held' if paired else 'q moves width too'}"


#: The chaining-band knob, shared by wtmm2d and wtmm2d_roi.
#: Spec 14's chaining options (``dynamix.core.chaining.OPTION_DEFAULTS``), off by default
#: (xsmurf's behaviour), in the inspector's advanced section, out of the key at their defaults.
CHAINING_OPTION_PARAMS = (
    Param("angle_test", ParamKind.BOOL, default=False, label="Angle test", level="advanced",
          keyed_at_default=False),
    Param("angle_threshold", ParamKind.FLOAT, default=0.8, min=0.0, max=1.0, soft_min=0.5,
          soft_max=0.95, label="Angle similarity", level="advanced", keyed_at_default=False,
          active_when=("angle_test", (True,))),
    Param("radius_rule", ParamKind.CHOICE, default="xsmurf",
          choices=("xsmurf", "scale_proportional"), choice_labels=("xsmurf box", "∝ scale"),
          label="Search radius", level="advanced", keyed_at_default=False),
    Param("radius_c", ParamKind.FLOAT, default=1.0, min=0.01, max=100.0, soft_min=0.5,
          soft_max=4.0, label="Radius c", level="advanced", keyed_at_default=False,
          active_when=("radius_rule", ("scale_proportional",))),
    Param("tie_break", ParamKind.CHOICE, default="distance", choices=("distance", "modulus"),
          choice_labels=("closer wins (xsmurf)", "modulus closer"), label="Tie-break",
          level="advanced", keyed_at_default=False),
    Param("border_policy", ParamKind.CHOICE, default="none",
          choices=("none", "coi_trim", "coi_drop"),
          choice_labels=("none (xsmurf)", "trim at COI", "drop at COI"), label="Border",
          level="advanced", keyed_at_default=False),
    Param("coi_k", ParamKind.FLOAT, default=2.0, min=0.1, max=10.0, soft_min=1.0, soft_max=4.0,
          label="COI k", level="advanced", keyed_at_default=False,
          active_when=("border_policy", ("coi_trim", "coi_drop"))),
)

BAND_PER_PARAM = Param("band_per", ParamKind.CHOICE, default="voice", choices=("voice", "octave"),
                       choice_labels=("per voice (xsmurf)", "per octave"), label="Similitude per")


def chaining_reading(similitude: float, params: dict) -> str:
    """The two lines beside the similitude knob: the largest Hölder exponent a chain can carry
    through the band at these voices, η and band mode (``chainable_h`` gives the lower limit
    too), and the largest h the analyzing wavelet resolves, h < n_ψ (its vanishing moments; the
    smoother knob names ψ). Two different limits: a chain may carry an h the wavelet cannot
    measure. Short enough for the strip's fixed column."""
    from dynamix.core.smoothers import smoother
    from dynamix.core.wtmm_backend import chainable_h

    wavelet = params.get("wavelet", "mexican")
    sm = smoother(wavelet, smoother_params(params))
    resolves = f"ψ resolves h<{sm.vanishing_moments}"
    span = chainable_h(float(similitude), int(params.get("n_voice", 4)),
                       float(params.get("fracint_alpha", 0.0)), params.get("band_per", "voice"))
    if span is None:
        return f"every h chains\n{resolves}"
    return f"chains h<{span[1]:.2f}\n{resolves}"


class WTMM2D:
    """Wavelet Transform Modulus Maxima over a 2-D field.

    Wraps :func:`dynamix.core.wtmm_backend.run_wtmm2d`, which is copied verbatim from EQSelect.
    The scale stack it produces is ``n_oct * n_voice`` layers of extrema, each carrying ``x``,
    ``y``, ``mod`` and ``arg`` -- the argument being the orientation the wedge filter acts on.
    """

    name = "wtmm2d"
    capabilities = frozenset({"Cancellable", "OutputProducer", "Previewable", "RegionAware", "StripExtras", "Viewable"})
    role = "primary"
    chain_producer = True
    label = "WTMM 2D"
    origin = XSMURF_DERIVED
    families = ("Edge detection", "Regularity")
    consumes = ((FIELD,),)
    produces = (EXTREMA, VCHAINS)
    #: Progressive compute: resolve() threads its ``cancel`` predicate into this device's
    #: ``compute`` because it opts in here. Reaches ``run_wtmm2d``'s per-stage check so a
    #: superseding edit kills the in-flight stack instead of waiting it out.
    wants_cancel = True
    #: PRE is everything that shapes the wavelet transform and its extrema
    #: (run once, expensive); CHAINING is everything that decides how extrema link into maxima
    #: lines ACROSS scales (cheap by comparison, but still part of this Transform -- xsmurf's own
    #: ``chains2d`` has no separate cache seam from the CWT/extrema stages the way a Filter would).
    #: ``workflow_zone.DeviceBox`` renders one labeled column-group per entry, in THIS param order
    #: -- not ``params`` tuple order -- so the pairing below (wavelet+a_min, alpha+q_tsallis (one
    #: shown at a time), q_pairing+q_beta, n_oct+n_voice, smooth alone; min_chain_len+thresh,
    #: dist2_max+box_ratio, similitude alone) is a deliberate two-per-column layout, not an
    #: accident of dict iteration. ``similitude`` joins CHAINING: it is the third linking kwarg in the exact same
    #: ``chains2d`` call as ``box_ratio``/``dist2_max`` (dynamix/core/wtmm_backend.py), so a
    #: group named CHAINING that omitted it was incomplete, not merely conservative.
    param_groups = {
        "PRE": ("wavelet", "a_min", "alpha", "q_tsallis", "q_pairing", "q_beta", "n_oct",
                "n_voice", "smooth", "dither", "fracint_alpha", "interpolate", "detector"),
        "CHAINING": ("min_chain_len", "thresh", "dist2_max", "box_ratio", "similitude",
                     "band_per"),
    }
    params = (
        Param("n_oct", ParamKind.INT, default=3, min=1, max=12, label="Octaves"),
        Param("n_voice", ParamKind.INT, default=4, min=1, max=64, label="Voices/octave"),
        Param("a_min", ParamKind.FLOAT, default=1.0, min=0.01, max=64.0,
              soft_min=0.5, soft_max=4.0, units="", label="aₘᵢₙ"),
        # The smoother θ and its parameters (module level, shared with the ROI device).
        *SMOOTHER_PARAMS,
        # The M-Z devices' seeded half-LSB dither, on the WTMM path: it breaks up processing
        # noise before chaining. Applied by THIS device to a CLONE of the field (the backend is
        # a verbatim EQSelect copy and rejects unknown params); fails toward no-op when no
        # value lattice is measurable (core.mz_edges.measure_lsb's contract).
        Param("dither", ParamKind.BOOL, default=False, label="Dither (±½ LSB)"),
        # Pseudo-fractional integration order η (Wendt 2009; available
        # on the scalar path too, not only tensor) -- the per-scale a**η modulus lift applied
        # right after the cached cwt stage (``wtmm_backend._apply_fracint2d``). Default 0: no
        # lift, so a chain's slope is its Hölder exponent; 1.0 is the reference EBSD pipeline's
        # value. The spectrum window seeds its η shift-back from this value, so what is lifted
        # forward is undone at the fit; the chain-Hölder filter reads the lifted slope (h + η).
        Param("fracint_alpha", ParamKind.FLOAT, default=0.0, min=-6.0, max=6.0,
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
        BAND_PER_PARAM,
        # The cross-scale chainer (dynamix.core.chaining): the benchmark, xsmurf's vchain port,
        # by default; not keyed at that default, so every recipe keyed before the knob existed
        # keeps its key.
        Param("chainer", ParamKind.CHOICE, default="xsmurf_bench",
              choices=("xsmurf_bench", "gradient_arg_horizontal", "vertical_experimental"),
              label="Chainer", keyed_at_default=False),
        *CHAINING_OPTION_PARAMS,
        # The raster drawn in place of the field: "edges" is the field itself under the
        # maxima; "modulus" and "argument" name the per-scale outputs below, at the level the
        # Transport shows. View-only, so the analysis key never sees it.
        Param("show", ParamKind.CHOICE, default="edges",
              choices=("edges", "modulus", "argument"), label="Show", view=True),
        PRECISION_PARAM,
    )

    #: The layer panel's output rows, a radio over ``show``, off by default: the lifted
    #: wavelet-gradient modulus the extrema are detected on, and its argument, one cache entry
    #: per level, computed on request.
    outputs = (Output("modulus", "raster", lazy=True, label="modulus", per_level=True),
               Output("argument", "raster", lazy=True, label="argument", per_level=True))

    def wavelet_profile(self, params: dict, scale_px: float, level: int):
        """The analysing wavelet's section and λ* at ``scale_px`` (the wavelet scale bar)."""
        from dynamix.core.wavelet_profiles import wtmm_profile

        return wtmm_profile(params["wavelet"], scale_px, smoother_params(params))

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
                "smoother_params": smoother_params(params),
                "min_chain_len": params["min_chain_len"],
                "smooth": params["smooth"],
                "thresh": params["thresh"],
                "dist2_max": params["dist2_max"],
                "box_ratio": params["box_ratio"],
                "similitude": params["similitude"],
                "fracint_alpha": params["fracint_alpha"],
                "interpolate": params["interpolate"],
                "detector": params["detector"],
                "band_per": params["band_per"],
                "chainer": params["chainer"],
                **{p.name: params[p.name] for p in CHAINING_OPTION_PARAMS},
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
        warning = eta_warning(params)
        if warning:
            res["_warnings"] = [warning]
        return res

    def knob_warnings(self, field, params: dict) -> dict:
        """The knobs whose value the grid cannot carry honestly, for the strip's ⚠: ``a_min`` by
        :func:`eta_warning`."""
        warning = eta_warning(params or {})
        return {"a_min": warning} if warning else {}

    def view(self, result: dict, params: dict) -> dict:
        """``show`` picks a per-scale output, which the window computes and caches under its
        own key; the analysis arrays are shown as they are."""
        return result

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None, level=None) -> dict:
        """The per-scale output ``name`` (``modulus`` or ``argument``) of scale ``level``
        (0-based, finest first, clipped into the result's scales as ``scale_select`` clips):
        the backend's own one-scale cwt and lift
        (:func:`~dynamix.core.wtmm_levels.run_wtmm2d_level`) over ``values`` dithered exactly
        as :meth:`compute` dithers them. ``{"raster": float32, "diag": {"level", "scale_px"}}``."""
        if name not in ("modulus", "argument"):
            raise ValueError(f"{self.name}: no lazy output named {name!r}")
        from dynamix.core.wtmm_backend import ComputeCancelled
        from dynamix.core.wtmm_levels import run_wtmm2d_level

        if cancel is not None and cancel():
            raise ComputeCancelled()
        values = np.asarray(values)
        if params.get("dither"):
            from dynamix.core.mz_edges import _dithered, measure_lsb

            lsb = measure_lsb(values)
            if lsb:
                values = _dithered(values, lsb)
        n = len(result["scales"])
        idx = int(np.clip(0 if level is None else int(level), 0, n - 1))
        if progress is not None:
            progress(f"{name} scale {idx + 1}/{n}", 0.0)
        one = run_wtmm2d_level(values, dict({k: params[k] for k in _LEVEL_KEYS},
                                            smoother_params=smoother_params(params)), idx)
        if progress is not None:
            progress(f"{name} scale {idx + 1}/{n}", 1.0)
        raster = one["mod"] if name == "modulus" else one["arg"]
        return {"raster": np.asarray(raster, dtype=np.float32),
                "diag": {"level": idx, "scale_px": one["scale"]}}

    # -- ROI runner ----------------------
    _ROI_KEYS = ("n_oct", "n_voice", "a_min", "wavelet", "fracint_alpha", "interpolate",
                 "detector", "min_chain_len", "smooth", "thresh", "dist2_max", "box_ratio",
                 "similitude", "band_per", "chainer")

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
        res = run_wtmm2d_roi_on_window(values, real, int(r0), int(h), int(w),
                                       dict({k: params[k] for k in self._ROI_KEYS},
                                            smoother_params=smoother_params(params)),
                                       progress=progress)
        res = dict(res)
        res["_frame"] = getattr(window_field, "frame", None)
        attach_chain_product(res)
        warning = eta_warning(params)
        if warning:
            res["_warnings"] = list(res.get("_warnings", [])) + [warning]
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
            "wavelet": params["wavelet"], "smoother_params": smoother_params(params),
            "thresh": params["thresh"],
            "min_chain_len": params["min_chain_len"], "smooth": params["smooth"],
            "dist2_max": params["dist2_max"], "box_ratio": params["box_ratio"],
            "similitude": params["similitude"], "fracint_alpha": params["fracint_alpha"],
            "interpolate": params["interpolate"], "detector": params["detector"],
        })

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, keyed_params(self, params))

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        """The sigma line beside the ``aₘᵢₙ`` knob, now with the analyzing wavelet's bandpass
        wavelength too (``dynamix.core.scale_units``): sigma is smoother-sigma (wavelet-
        independent); lambda is per-wavelet -- the reason ``params`` joined the hook's signature.
        Lambda, not an outer-extremum footprint: the footprint metric orders the wavelets
        backwards (see ``scale_units``' docstring). ``None`` for every other param: a derived
        reading explains a knob whose face value is not itself physical, not a general per-param
        annotation mechanism."""
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
