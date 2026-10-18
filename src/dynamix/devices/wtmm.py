# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The WTMM transform: the expensive one, and the reason everything downstream is cheap.

Runs the whole scale stack once and caches it. Every filter after this point is a lookup or a
predicate over the result, which is what makes the scale slider feel instant.
"""
from __future__ import annotations

import dataclasses
import math

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


def cut_whole(res: dict, params: dict, shape) -> dict:
    """``edge = "cut"`` on a whole raster: each scale's maxima inside its cone of influence (the
    measured reach, ``dynamix.core.edges.cut_borders``) dropped, the chains trimmed there (spec
    14's ``coi_trim``, or ``coi_drop`` when that is the border policy) and the partition functions
    rebuilt from them; ``_edge`` records the borders."""
    from dynamix.core import edges
    from dynamix.core.chaining import border_trimmed
    from dynamix.core.wtmm_backend import get_backend
    from dynamix.roi.halo import halo_plan

    scales = np.asarray(res["scales"])
    resolved = res["params"]
    pad = int(resolved.get("pad", 0))
    run_grid = max(int(shape[0]), int(shape[1])) + 2 * pad
    plan = halo_plan(scales, params.get("wavelet", "mexican"), smoother_params(params),
                     float(params.get("edge_eps", 1e-3)), None, run_grid=run_grid)
    scope = params.get("cut_scope", "largest")
    borders = edges.cut_borders(plan, scope)
    policy = "coi_drop" if params.get("border_policy") == "coi_drop" else "coi_trim"

    out = dict(res)
    out["chains"] = border_trimmed(res["chains"], shape, policy, 0.0,
                                   int(resolved["min_chain_len"]),
                                   reach_at=edges.chain_reach(scales, borders))
    out["extrema"] = edges.drop_in_cone(res["extrema"], borders, (0, 0), shape)
    out["hd_std"], out["hd_cmax"] = get_backend("python").partition2d(
        out["chains"], scales, q_list=resolved["q_list"], min_chain_len=resolved["min_chain_len"])
    out["_edge"] = {
        "device": "wtmm2d", "mode": "cut", "cut_scope": scope, "run_grid": run_grid,
        "eps": float(params.get("edge_eps", 1e-3)),
        "parent_shape": [int(shape[0]), int(shape[1])], "budget_px": None,
        "scales": [{"a": float(sh.a), "margin": None, "whole": True, "radius": sh.reach.radius,
                    "measured": sh.reach.measured, "ratio": sh.reach.ratio,
                    "grids": list(sh.reach.grids), "ratio_run": sh.reach.ratio_run,
                    "converged": sh.reach.converged,
                    "real_px": {s: 0 for s in "NSEW"}, "synthetic_px": {s: pad for s in "NSEW"},
                    "border": int(b)} for sh, b in zip(plan, borders)]}
    return out


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

#: Spec 22's edge knobs, shared by wtmm2d and wtmm2d_roi, out of the key at their defaults: what fills
#: a scale's reach where real data runs out (``dynamix.core.edges``), whose cone of influence a cut
#: uses, and the mass fraction the reach leaves outside (``dynamix.core.reach``).
EDGE_PARAMS = (
    Param("edge", ParamKind.CHOICE, default="reflect",
          choices=("reflect", "cut", "gaussian_taper", "daf_periodic"),
          choice_labels=("reflect", "cut at COI", "Gaussian taper", "DAF periodic"), label="Edge",
          keyed_at_default=False),
    Param("cut_scope", ParamKind.CHOICE, default="largest", choices=("largest", "per_scale"),
          choice_labels=("largest scale's COI", "each scale's COI"), label="Cut scope",
          keyed_at_default=False, active_when=("edge", ("cut",))),
    Param("edge_eps", ParamKind.FLOAT, default=1e-3, min=1e-6, max=0.1, soft_min=1e-4,
          soft_max=1e-2, label="Edge ε", level="advanced", keyed_at_default=False),
)

#: Spec 25's boundary and Border cut knobs for the whole-raster tool. Every convolution is circular:
#: ``edge`` says what the FFT torus holds beyond the field (``dynamix.core.wtmm_backend.PAD_MODES``),
#: ``pad_width`` how wide the padding is, and the Border cut, run between the H-chains and the
#: V-chains, which maxima go before the chaining (``dynamix.core.edges``). The Edge's ``auto`` reads
#: the field: on a Periodic extension output periodic with no padding, otherwise the mirror
#: (JPEG2000's symmetric extension), which the transform crops off before the maxima are found. The
#: Border cut is off by default; its ``auto`` cuts by the measured reach except on a Periodic
#: extension output and where that border would cover the raster. Spec 22's ``cut`` and
#: ``daf_periodic`` stay valid for saved recipes but are no longer offered; their own scope knob is
#: hidden with them.
BOUNDARY_PARAMS = (
    Param("edge", ParamKind.CHOICE, default="auto",
          choices=("auto", "reflect", "periodic", "edge_pad", "zero_pad", "gaussian_taper", "cut",
                   "daf_periodic"),
          choice_labels=("auto", "reflect", "periodic", "edge-value padding", "zero padding",
                         "Gaussian taper", "cut at COI", "DAF periodic"),
          hidden_choices=("cut", "daf_periodic"), label="Edge"),
    Param("pad_width", ParamKind.CHOICE, default="measured_reach",
          choices=("measured_reach", "fixed_px", "border_pct"),
          choice_labels=("measured reach", "fixed px", "border %"), label="Padding",
          active_when=("edge", ("auto", "reflect", "edge_pad", "zero_pad"))),
    Param("pad_px", ParamKind.INT, default=32, min=0, max=8192, soft_min=0, soft_max=256,
          label="Padding px", level="advanced", active_when=("pad_width", ("fixed_px",))),
    Param("pad_pct", ParamKind.FLOAT, default=0.72, min=0.0, max=10.0, soft_min=0.1, soft_max=2.0,
          label="Padding %", level="advanced", active_when=("pad_width", ("border_pct",))),
    Param("border_cut", ParamKind.CHOICE, default="none",
          choices=("none", "measured_reach", "border_pct", "k_scale", "auto"),
          choice_labels=("none", "measured reach", "border %", "k × scale", "auto"),
          label="Border cut"),
    Param("border_pct", ParamKind.FLOAT, default=0.72, min=0.0, max=10.0, soft_min=0.1,
          soft_max=2.0, label="Border %", level="advanced", active_when=("border_cut", ("border_pct",))),
    Param("border_k", ParamKind.FLOAT, default=2.0, min=0.1, max=10.0, soft_min=1.0, soft_max=4.0,
          label="Border k", level="advanced", active_when=("border_cut", ("k_scale",))),
    Param("border_scope", ParamKind.CHOICE, default="largest", choices=("largest", "per_scale"),
          choice_labels=("largest scale", "each scale"), label="Border scope", level="advanced"),
    Param("cut_scope", ParamKind.CHOICE, default="largest", choices=("largest", "per_scale"),
          choice_labels=("largest scale's COI", "each scale's COI"), label="Cut scope",
          level="hidden", keyed_at_default=False, active_when=("edge", ("cut",))),
    Param("edge_eps", ParamKind.FLOAT, default=1e-3, min=1e-6, max=0.1, soft_min=1e-4,
          soft_max=1e-2, label="Edge ε", level="advanced"),
)

#: The whole-raster tool's chaining options: the old chain trim is superseded by the Border cut.
WTMM2D_CHAINING_OPTION_PARAMS = tuple(
    dataclasses.replace(p, level="hidden") if p.name in ("border_policy", "coi_k") else p
    for p in CHAINING_OPTION_PARAMS)

#: Above this the periodic seam warns: the RMS difference across opposite edges over the RMS
#: difference between neighbours inside the field (spec 25 §3.2).
SEAM_LIMIT = 3.0


def filled_block(field):
    """A Periodic extension output's real block ``(rows, cols)``, else None."""
    stamp = (getattr(field, "provenance", None) or {}).get("edges.filled")
    return None if not stamp else (int(stamp["rows"]), int(stamp["cols"]))


def resolved_boundary(params: dict, field) -> tuple:
    """``(edge, border_cut)`` with ``auto`` read from ``field``: on a Periodic extension output
    ``("periodic", "none")``, otherwise ``("reflect", "measured_reach")``, the cut ``none`` where
    the measured-reach border would cover the raster and leave it no maxima."""
    stamped = filled_block(field) is not None
    edge = params.get("edge", "auto")
    border = params.get("border_cut", "none")
    if edge == "auto":
        edge = "periodic" if stamped else "reflect"
    if border == "auto":
        border = "none" if stamped else "measured_reach"
        values = getattr(field, "values", None)
        if border == "measured_reach" and values is not None and _covering(
                params, border, np.shape(values)[:2]) is not None:
            border = "none"
    return edge, border


def boundary_reading(name: str, params: dict, field) -> "str | None":
    """The line beside the Edge or Border cut knob: what ``auto`` resolved to, and the padding or
    the coarsest scale's border in px (spec 25 §3.2, §3.3)."""
    from dynamix.core.edges import border_widths, pad_widths
    from dynamix.core.wtmm_backend import compute_scales2d

    try:
        edge, border = resolved_boundary(params, field)
        scales = list(compute_scales2d(params["n_oct"], params["n_voice"], params["a_min"]))
        sp = smoother_params(params)
        eps = float(params.get("edge_eps", 1e-3))
        if name == "edge":
            head = f"→ {edge}" if params.get("edge", "auto") == "auto" else ""
            if edge == "periodic":
                return (head + " · no padding").lstrip(" ·")
            if edge in ("reflect", "edge_pad", "zero_pad"):
                widths, notes = pad_widths(scales, params.get("pad_width", "measured_reach"),
                                           wavelet=params["wavelet"], smoother_params=sp, eps=eps,
                                           pad_px=params.get("pad_px", 32),
                                           pad_pct=params.get("pad_pct", 0.72))
                tail = " · the finest scales take the widest converged reach" if notes else ""
                return (head + f" · pad {max(widths)} px" + tail).lstrip(" ·")
            return head or None
        head = f"→ {border.replace('_', ' ')}" if params.get("border_cut", "none") == "auto" else ""
        if border == "none":
            values = getattr(field, "values", None)
            if head and filled_block(field) is None and values is not None:
                return head + " · the measured reach would cover the raster"
            return head or None
        b = border_widths(scales, border, wavelet=params["wavelet"], smoother_params=sp, eps=eps,
                          pct=params.get("border_pct", 0.72), k=params.get("border_k", 2.0),
                          scope=params.get("border_scope", "largest"))
        return (head + f" · {b[-1]:.0f} px at the coarsest scale").lstrip(" ·")
    except (KeyError, ValueError):
        return None


def _covering(params: dict, border: str, shape: tuple) -> "tuple | None":
    """``(border px, scale px)`` at the first scale whose Border cut meets itself across a
    ``shape`` raster (twice the border reaches its shorter side), else None."""
    from dynamix.core.edges import border_widths
    from dynamix.core.wtmm_backend import compute_scales2d

    try:
        scales = list(compute_scales2d(params["n_oct"], params["n_voice"], params["a_min"]))
        b = border_widths(scales, border, wavelet=params["wavelet"],
                          smoother_params=smoother_params(params),
                          eps=float(params.get("edge_eps", 1e-3)), pct=params.get("border_pct", 0.72),
                          k=params.get("border_k", 2.0), scope=params.get("border_scope", "largest"),
                          run_grid=max(shape))
    except (KeyError, ValueError):
        return None
    first = next((i for i, w in enumerate(b) if 2 * w >= min(int(shape[0]), int(shape[1]))), None)
    return None if first is None else (b[first], scales[first])


def border_covering(params: dict, border: str, shape: tuple) -> "str | None":
    """The ⚠ beside the Border cut where its border covers a ``shape`` raster: a chosen cut keeps
    no maxima from that scale up; ``auto`` cuts nothing there and keeps the maxima near the edges."""
    asked = params.get("border_cut", "none")
    hit = _covering(params, "measured_reach" if asked == "auto" else border, shape)
    if hit is None:
        return None
    b, a = hit
    H, W = int(shape[0]), int(shape[1])
    if asked == "auto":
        return (f"auto cuts nothing on this raster: the measured-reach border ({b:.0f} px at scale "
                f"{a:.3g} px) would cover its {H} × {W} pixels, so the maxima near its edges are "
                "kept; border % or k × scale cut less")
    return (f"the border ({b:.0f} px at scale {a:.3g} px) covers this {H} × {W} raster: no maxima "
            "are kept from that scale up; a smaller border, fewer octaves, or none")


def seam_ratio(values) -> float:
    """The RMS difference across opposite edges over the RMS difference between neighbours."""
    v = np.asarray(values, dtype=np.float64)
    inside = math.sqrt(np.nanmean(np.diff(v, axis=0) ** 2) + np.nanmean(np.diff(v, axis=1) ** 2))
    across = math.sqrt(np.nanmean((v[0] - v[-1]) ** 2) + np.nanmean((v[:, 0] - v[:, -1]) ** 2))
    return across / inside if inside > 0 else (0.0 if across == 0 else math.inf)


#: The nms detector's tie rule (``dynamix.core.wtmm_backend.TIE_RULES``): xsmurf's asymmetric
#: test by default (one pixel of a tied pair), strict (neither), or both as the detector did before.
TIE_RULE_PARAM = Param("tie_rule", ParamKind.CHOICE, default="asymmetric",
                       choices=("asymmetric", "strict", "keep_both"),
                       choice_labels=("one of a tie (xsmurf)", "strict", "keep both (before)"),
                       label="Ties", level="advanced", active_when=("detector", ("nms",)))

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
        "EDGES": ("edge", "pad_width", "border_cut"),
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
        TIE_RULE_PARAM,
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
        *WTMM2D_CHAINING_OPTION_PARAMS,
        *BOUNDARY_PARAMS,
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
        backend_params = {
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
                "tie_rule": params["tie_rule"],
                "band_per": params["band_per"],
                "chainer": params["chainer"],
                **{p.name: params[p.name] for p in CHAINING_OPTION_PARAMS},
        }
        res, fell_back = None, None
        edge, border = resolved_boundary(params, field)
        params = dict(params, edge=edge, border_cut=border)
        padding = None
        if edge in ("reflect", "periodic", "edge_pad", "zero_pad"):
            backend_params, padding = self._boundary(field, params, backend_params)
        if params.get("edge", "reflect") in ("gaussian_taper", "daf_periodic"):
            # An edge mode that changes the data runs scale by scale over the whole raster as
            # its own ROI (dynamix.roi.halo); where it does not fit, the backend's run stands.
            from dynamix.roi.halo import run_whole_through_core

            res, fell_back = run_whole_through_core(
                field.values, dict(backend_params, **{p.name: params[p.name] for p in EDGE_PARAMS}),
                progress=progress)
        if res is None:
            res = run_wtmm2d(field, backend_params, progress=progress, cancel=cancel)
        # Carry the field's own axes through so filters and views can convert pixels to physical
        # units without reaching back to the source.
        res = dict(res)
        if fell_back and "_edge" not in res:
            from dynamix.roi.halo import whole_fallback_stamp

            res["_edge"] = whole_fallback_stamp(
                dict(backend_params, **{p.name: params[p.name] for p in EDGE_PARAMS}),
                tuple(field.values.shape)[:2], fell_back)
        if params.get("edge", "reflect") == "cut":
            res = cut_whole(res, params, tuple(field.values.shape)[:2])
        if padding is not None:
            res["_padding"] = padding
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

    @staticmethod
    def _boundary(field, params: dict, backend_params: dict) -> tuple:
        """``(backend params, padding record)`` for a padded or periodic Edge (spec 25 §3.2): the
        pad mode, the padding (the widest scale's width, since the transform pads once), the
        Border cut and, on a Periodic extension output, its real block."""
        from dynamix.core.edges import pad_widths
        from dynamix.core.wtmm_backend import compute_scales2d

        scales = list(compute_scales2d(params["n_oct"], params["n_voice"], params["a_min"]))
        shape = tuple(np.asarray(field.values).shape)[:2]
        edge = params["edge"]
        if edge == "periodic":
            widths, notes = [0] * len(scales), []
        else:
            widths, notes = pad_widths(scales, params["pad_width"], wavelet=params["wavelet"],
                                       smoother_params=smoother_params(params),
                                       eps=params["edge_eps"], pad_px=params["pad_px"],
                                       pad_pct=params["pad_pct"], run_grid=max(shape))
        out = dict(backend_params, pad=max(widths) if widths else 0, pad_mode=edge,
                   border_cut=params["border_cut"], border_pct=params["border_pct"],
                   border_k=params["border_k"], border_scope=params["border_scope"],
                   edge_eps=params["edge_eps"], real_block=filled_block(field))
        record = {"edge": edge, "pad_width": params["pad_width"] if edge != "periodic" else None,
                  "widths": widths, "pad": out["pad"], "notes": notes}
        return out, record

    def footprint(self, field, params: dict, level: int, stride: int = 1) -> "dict | None":
        """What the Raster view draws while the edge knobs move (spec 25 §3.4), from the numbers
        the run uses at scale ``level``: ``pad`` (the run's padding, px; 0 for periodic),
        ``shown`` (the outline's width outside the raster: the pad, or for periodic the measured
        reach the coarsest scale reads round the wrap), ``padded`` (the raster as the mode fills
        it, every ``stride``-th pixel, padded by ``shown`` in samples), ``offset`` (px from the
        raster's first pixel back to ``padded``'s first), and ``border`` (px of the band inside
        the raster ``band`` names: the Border cut at this scale, or the taper's). None for the
        superseded modes, which keep spec 22's own record."""
        from dynamix.core.edges import border_widths, pad_widths
        from dynamix.core.wtmm_backend import _WTMM2D_PARAM_DEFAULTS, _padded, compute_scales2d

        v = np.asarray(field.values, dtype=np.float64)
        v = v[..., 0] if v.ndim == 3 else v
        H, W = v.shape
        edge, border = resolved_boundary(params, field)
        params = dict(params, edge=edge, border_cut=border)
        scales = list(compute_scales2d(params["n_oct"], params["n_voice"], params["a_min"]))
        level = min(max(int(level), 0), len(scales) - 1)
        sp, eps = smoother_params(params), params["edge_eps"]
        content, mode, band, width = v, edge, "border_cut", 0
        if edge in ("reflect", "periodic", "edge_pad", "zero_pad"):
            backend, _record = self._boundary(field, params, {})
            pad = shown = int(backend["pad"])
            if edge == "periodic":
                widths, _ = pad_widths(scales, "measured_reach", wavelet=params["wavelet"],
                                       smoother_params=sp, eps=eps, run_grid=max(H, W))
                shown = int(max(widths))
            if border != "none":
                width = int(border_widths(scales, border, wavelet=params["wavelet"],
                                          smoother_params=sp, eps=eps, pct=params["border_pct"],
                                          k=params["border_k"], scope=params["border_scope"],
                                          run_grid=max(H, W) + 2 * pad)[level])
        elif edge == "gaussian_taper":
            # the taper's own path: the backend's fixed pad around the tapered raster, no cut
            from dynamix.core.edges import tapered
            from dynamix.core.wtmm_backend import _resolve_wtmm2d_params
            from dynamix.roi.halo import _taper_fill, _whole_plan

            pad = shown = int(_WTMM2D_PARAM_DEFAULTS["pad"])
            resolved = _resolve_wtmm2d_params({k: params[k] for k in _LEVEL_KEYS}
                                              | {"smoother_params": sp})
            fill, _extras = _taper_fill(v, (H, W), _whole_plan(resolved, eps, (H, W)))
            mode = "reflect"
            if fill is not None:
                content = tapered(np.where(np.isfinite(v), v, 0.0), np.arange(H), np.arange(W),
                                  (H, W), fill["mu"], fill["w_t"])
                band, width = "taper", int(fill["w_t"])
        else:
            return None
        s = max(1, int(stride))
        q = -(-shown // s)
        small = content[::s, ::s]
        padded = np.pad(small, q, mode="wrap") if mode == "periodic" else _padded(small, q, mode)[0]
        return {"edge": edge, "pad": pad, "shown": shown, "padded": padded, "stride": s,
                "offset": -q * s, "border": width, "band": band, "shape": (H, W)}

    def roi_footprint(self, field, params: dict, rect, level: int, stride: int = 1) -> dict:
        """The footprint of the run on an ROI ``rect = (row, col, h, w)`` (the field's own px):
        scale ``level``'s halo (spec 22's halo plan; the whole dataset where the reach does not
        converge), as the run reads it -- the dataset where it has pixels, mirrored about its
        edges past them (``_reflect_pad``'s rule) -- sampled every ``stride`` px. An ROI has no
        Border cut: that runs on whole rasters."""
        from dynamix.roi.picture import display_stride

        v = np.asarray(field.values, dtype=np.float64)
        v = v[..., 0] if v.ndim == 3 else v
        k = display_stride(field)                     # file px per stored sample (a picture)
        H, W = v.shape[0] * k, v.shape[1] * k
        r, c, h, w = (int(x) for x in rect)
        plan = self._halo_plan(params, parent_shape=(H, W))
        sh = plan[min(max(int(level), 0), len(plan) - 1)]
        m = max(r, c, H - r - h, W - c - w) if sh.whole else int(sh.margin)
        s = max(1, int(stride)) * k

        def axis(start, length, n):
            px = start - m + s * np.arange(-(-(length + 2 * m) // s))
            period = 2 * (n - 1)
            px = np.abs(px) % period if period > 0 else np.zeros_like(px)
            px = np.where(px >= n, period - px, px)
            return np.minimum(px // k, n // k - 1)

        return {"edge": "reflect", "pad": m, "shown": m, "padded": v[np.ix_(axis(r, h, H),
                                                                            axis(c, w, W))],
                "stride": s, "offset": -m, "border": 0, "band": "border_cut", "shape": (h, w),
                "origin": (r, c)}

    def knob_warnings(self, field, params: dict) -> dict:
        """The knobs whose value the grid cannot carry honestly, for the strip's ⚠: ``a_min`` by
        :func:`eta_warning`, ``edge`` where the Gaussian taper would not fit this field or where
        ``periodic`` meets a raster whose opposite edges do not match, ``border_cut`` where the
        border covers the raster. A scale whose reach does not converge pads by the widest
        converged one: the run records it and the Edge knob's reading says so."""
        params = params or {}
        out = {}
        warning = eta_warning(params)
        if warning:
            out["a_min"] = warning
        values = getattr(field, "values", None)
        edge, _border = resolved_boundary(params, field)
        if edge == "periodic" and values is not None and filled_block(field) is None:
            ratio = seam_ratio(np.asarray(values)[..., 0] if np.asarray(values).ndim == 3 else values)
            if ratio > SEAM_LIMIT:
                out["edge"] = (f"the raster is not periodic: its opposite edges differ {ratio:.1f}× "
                               f"its neighbours (above {SEAM_LIMIT:g}×), and periodic joins them; "
                               "Periodic extension (DAF) first, or a padded mode")
        if values is not None and filled_block(field) is None and (
                params.get("border_cut", "none") == "auto" or _border != "none"):
            covered = border_covering(params, _border, np.asarray(values).shape[:2])
            if covered:
                out["border_cut"] = covered
        if params.get("edge") in ("gaussian_taper", "daf_periodic") and values is not None:
            from dynamix.roi.halo import edge_fallback

            v = np.asarray(values)
            reason = edge_fallback(v, v.shape[:2], params, smoother_params(params),
                                   run_grid=max(v.shape[:2]) + 64)
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
        one = None
        if params.get("edge", "reflect") in ("gaussian_taper", "daf_periodic"):
            # the window the analysis transformed (tapered, or extended by the DAF)
            from dynamix.roi.halo import run_whole_level

            one = run_whole_level(values, dict({k: params[k] for k in _LEVEL_KEYS},
                                               smoother_params=smoother_params(params),
                                               **{p.name: params[p.name] for p in EDGE_PARAMS}),
                                  idx)
        if one is None:
            # The padding the analysis used (spec 25 §3.2), so a level's modulus is the one its
            # maxima were found on.
            ran = (result or {}).get("params") or {}
            padding = {k: ran[k] for k in ("pad", "pad_mode") if k in ran}
            one = run_wtmm2d_level(values, dict({k: params[k] for k in _LEVEL_KEYS},
                                                smoother_params=smoother_params(params), **padding),
                                   idx)
        if progress is not None:
            progress(f"{name} scale {idx + 1}/{n}", 1.0)
        raster = one["mod"] if name == "modulus" else one["arg"]
        return {"raster": np.asarray(raster, dtype=np.float32),
                "diag": {"level": idx, "scale_px": one["scale"]}}

    # -- ROI runner ----------------------
    _ROI_KEYS = ("n_oct", "n_voice", "a_min", "wavelet", "fracint_alpha", "interpolate",
                 "detector", "min_chain_len", "smooth", "thresh", "dist2_max", "box_ratio",
                 "similitude", "band_per", "chainer", "tie_rule")

    def _halo_plan(self, params: dict, parent_shape=None) -> list:
        from dynamix.core.wtmm_backend import compute_scales2d
        from dynamix.roi.halo import halo_plan

        scales = compute_scales2d(params["n_oct"], params["n_voice"], params["a_min"])
        return halo_plan(scales, params["wavelet"], smoother_params(params),
                         float(params.get("edge_eps", 1e-3)), parent_shape)

    def roi_margin(self, params: dict) -> int:
        """The widest halo of :func:`~dynamix.roi.halo.halo_plan` with the parent unknown (the
        2.5 x normalized scale budget, widened to each scale's measured reach) -- every scale's
        halo is a centred sub-rect of it."""
        return max(sh.margin for sh in self._halo_plan(params))

    def roi_whole_parent(self, params: dict) -> bool:
        """Whether some scale's reach does not converge, so that scale reads the whole parent
        when the runner can hold it."""
        return any(not sh.reach.converged for sh in self._halo_plan(params))

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
        prov = getattr(window_field, "provenance", None) or {}
        full = prov.get("full_dims")
        win = prov.get("window")
        origin = (None if win is None else
                  (int(win["row_off"]) + int(r0), int(win["col_off"]) + int(core[1])))
        # The halo reads the parent; spec 25's padded and periodic modes and the Border cut run on
        # whole rasters, so past the parent's edges an ROI reflects (spec 22's modes stand).
        edge = {p.name: params[p.name] for p in EDGE_PARAMS}
        asked = edge["edge"]
        if asked in ("auto", "periodic", "edge_pad", "zero_pad"):
            edge["edge"] = "reflect"
        res = run_wtmm2d_roi_on_window(values, real, int(r0), int(h), int(w),
                                       dict({k: params[k] for k in self._ROI_KEYS},
                                            smoother_params=smoother_params(params), **edge),
                                       progress=progress, parent_shape=full,
                                       roi_origin=origin)
        res = dict(res)
        if asked not in ("auto", edge["edge"]):
            res["_warnings"] = list(res.get("_warnings", [])) + [
                f"edge = {asked} runs on whole rasters; in an ROI the halo reads the parent and "
                "reflects past its edges"]
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
            "tie_rule": params["tie_rule"],
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
        if name in ("edge", "border_cut"):
            return boundary_reading(name, params or {}, field)
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
