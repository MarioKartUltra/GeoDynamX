# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""holder_measure / holder_multiaffine — the SPLIT microcanonical tools: one device per method,
each exposing ONLY the wavelet class its method admits, under the method's own names.

The smell these fix: the conflated ``holder_map`` crossed ``estimator`` with ``wavelet`` as
if they were orthogonal, but "gaussian" named a POSITIVE Gaussian under ``measure`` and the
LAPLACIAN-of-Gaussian under ``multiaffine`` — same word, two kernels, silently. The wavelet
family is a property of the METHOD:

* ``holder_measure`` — positive unit-mass kernels over ``||grad s||`` (Turiel 2008 §4.2.2;
  measure-like/edge-sparse fields): gaussian, q-Gaussian, Lorentzian, fractional Gaussian
  (the envelope fractionalized — see :func:`dynamix.core.microcanonical._radial_kernel`).
* ``holder_multiaffine`` — zero-mean wavelets over the signal (§4.2.1; smooth/function-class
  fields): g1/g2/g3 (the derivative-order axis; g2 IS the conflated device's default
  Ricker), q-Mexican, Lorentzian-Marr, fractional Gaussian (continuous order ``frac_n``).

Both expose the Pont 2006 estimator pair: ``regression`` (multiscale local-singularity —
resolves BOTH limbs of D(h); the default) and ``punctual``
(:func:`~dynamix.core.microcanonical.singularity_map_point` — single finest scale, the LEFT
limb / most-singular-set extractor with the sharpest spatial localization; the full scale
grid is still projected for the estimator-orthogonal no-support mask -- see
:func:`~dynamix.core.microcanonical.no_support_mask` -- but h reads from the finest scale
alone). ``r2_map`` is all-NaN under punctual —
no regression ran, and an honest absence beats a fabricated 1.0.

``holder_map`` (and its module) stays untouched and registered — saved projects name it,
and the never-delete rule holds; these two supersede it in the palette (the browser routes
superseded names to the collapsed Dev category). Display/result contract, chain-head
refusal, and cache keying are holder_map's exactly — everything downstream is
method-agnostic and consumes ``h_map``/``raster_out``.
"""
from __future__ import annotations

import dataclasses
import functools

from dynamix.devices.holder_map import field_values_2d
from dynamix.model.param import Param, ParamKind


def method_arrays(vals, method: str, params: dict, progress=None):
    """``(h_map, r2_map, scales)`` for one METHOD (``"multiaffine"`` | ``"measure"``) under
    the split params — the split tools' engine, shared verbatim by the band-reconstruction
    variants so the h-map a band is picked from is the h-map the reconstruction uses.
    ``holder_map.holder_arrays`` stays the conflated device's engine, untouched."""
    import numpy as np

    from dynamix.core import microcanonical as mc

    if method not in ("multiaffine", "measure"):
        raise ValueError(f"unknown method {method!r}; expected 'multiaffine' or 'measure'")
    punctual = params["estimator"] == "punctual"
    scales = np.geomspace(params["r_min"], params["r_min"] * params["kappa"],
                          int(params["n_scales"]))
    kw = dict(wavelet=params["wavelet"], beta=params["beta"],
              q_tsallis=params["q_tsallis"], frac_n=params["frac_n"],
              q_beta=q_width(method, params),
              construction=params.get("construction", "real"))
    if progress is not None:
        progress("holder projections", 0.0)
    # Projections are the slow part: reported per scale, over 0..70 %.
    per_scale = (None if progress is None
                 else (lambda stage, frac: progress(stage, 0.7 * float(frac))))
    if method == "multiaffine":
        T = mc.ricker_projections(vals, scales, progress=per_scale, **kw)
    else:
        T = mc.measure_projections(mc.gradient_measure(vals), scales, progress=per_scale, **kw)
    if progress is not None:
        progress("holder estimate", 0.7)
    if punctual:
        # h from the FINEST scale alone (Pont's estimator), but the no-support mask from
        # the whole stack: support is a data-quality property both estimators share (see
        # no_support_mask), so the full grid is still projected and kappa/n_scales shape only
        # the mask.
        h_map = mc.singularity_map_point(T[0], mc.relative_scale(scales[0], vals.shape))
        h_map[mc.no_support_mask(T).reshape(vals.shape)] = np.nan
        r2_map = np.full(vals.shape, np.nan, dtype=np.float32)
    else:
        h_map, r2_map = mc.singularity_map_regression(T, scales, r2_min=0.0)
    if progress is not None:
        progress("holder estimate", 1.0)
    # NaN in, NaN out: a nodata pixel never gets an exponent, whatever the kernel -- unmasked,
    # heavy tails report their own tail exponent (2beta - d) there, and even a Gaussian reports
    # one near the nodata edge (its reach crosses to real data).
    nodata = ~np.isfinite(np.asarray(vals, dtype=np.float64))
    if nodata.any():
        h_map = np.asarray(h_map).copy()
        r2_map = np.asarray(r2_map).copy()
        h_map[nodata] = np.nan
        r2_map[nodata] = np.nan
    return h_map, r2_map, scales


def q_width(method: str, params: dict) -> "float | None":
    """The width beta the q-family kernel runs with (None for every other kernel): the knob's
    value when fixed, ``microcanonical.paired_q_beta(q)`` when q-paired."""
    q_family = {"measure": "q_gaussian", "multiaffine": "q_mexican"}[method]
    if params.get("wavelet") != q_family:
        return None
    if params.get("q_pairing", "fixed") == "q-paired":
        from dynamix.core.microcanonical import paired_q_beta

        return paired_q_beta(float(params["q_tsallis"]))
    return float(params.get("q_beta", 0.5))


def q_width_check(method: str, params: dict) -> None:
    """A q-paired width needs q < 2 (the 2-D escort variance it holds at one is infinite from
    q = 2); the q-Mexican hat's own knob already stops at 1.95."""
    q_family = {"measure": "q_gaussian", "multiaffine": "q_mexican"}[method]
    if (params.get("wavelet") == q_family and params.get("q_pairing") == "q-paired"
            and float(params.get("q_tsallis", 1.5)) >= 2.0):
        raise ValueError(f"a q-paired width needs q < 2 (got q = {params['q_tsallis']}): the "
                         "escort variance it holds at one is infinite from q = 2 -- use a fixed "
                         "width there")


#: The multiaffine route's resolution floor (Turiel 2008, figure 2 and section 4.2.1): a
#: discretised wavelet must separate its positive and negative parts, so its zero crossings set
#: the minimum attainable resolution; Turiel 2009 puts the Mexican hat's at "several pixels".
#: Read here as every sign lobe -- between crossings, or from a crossing to the edge of a compact
#: support -- at least this many pixels across at r1; the central disc counts by its diameter,
#: so a wavelet with one crossing at r keeps the floor r1 = 1.
_LOBE_PX = 2.0


@functools.lru_cache(maxsize=64)
def _frac_crossings(n: float) -> tuple:
    """The fractional-Gaussian wavelet's zero-crossing radii in units of r (the first is 1 by
    its calibration): the sign changes of ``1F1((n+2)/2; 1; -u)``, with ``rho/r =
    sqrt(u / u0)``."""
    import numpy as np
    from scipy.optimize import brentq
    from scipy.special import hyp1f1

    from dynamix.core.microcanonical import _frac_u0

    a = (float(n) + 2.0) / 2.0
    u = np.linspace(0.0, 50.0 + 10.0 * float(n), 200_001)
    v = hyp1f1(a, 1.0, -u)
    v[np.abs(v) < 1e-12] = 0.0                    # the dust of an underflowing tail
    nz = np.flatnonzero(v)
    flips = nz[np.flatnonzero(np.diff(np.sign(v[nz])))]
    roots = [brentq(lambda x: hyp1f1(a, 1.0, -x), u[i], u[j])
             for i, j in zip(flips, nz[np.searchsorted(nz, flips) + 1])]
    u0 = _frac_u0(float(n))
    return tuple(float(np.sqrt(x / u0)) for x in roots)


def lobe_floor(method: str, params: dict) -> "float | None":
    """The smallest r1 (px) at which every sign lobe of the kernel is :data:`_LOBE_PX` wide.
    The q-Mexican hat's is closed-form: its crossing sits at ``r / sqrt(2 beta (2 - q))`` (r
    when q-paired), and below q = 1 its support ends at that radius times
    ``sqrt((2 - q)/(1 - q))``. On the measure route the positive kernels have no zero
    crossings; only the compact q-Gaussian (q < 1) gets a floor, its one lobe being the support
    disc of radius ``r / sqrt(beta (1 - q))`` -- the steepest decrease there is (Turiel 2008
    section 4.2.2). None for every other positive kernel."""
    import numpy as np

    wavelet = params.get("wavelet")
    if method == "measure":
        q = float(params.get("q_tsallis", 1.5))
        if wavelet != "q_gaussian" or q >= 1.0:
            return None
        return _LOBE_PX / (2.0 / np.sqrt(q_width(method, params) * (1.0 - q)))
    if wavelet == "q_mexican":
        q = float(params["q_tsallis"])
        rho0 = 1.0 / np.sqrt(2.0 * q_width(method, params) * (2.0 - q))
        widths = [2.0 * rho0]
        if q < 1.0:
            widths.append(rho0 * (np.sqrt((2.0 - q) / (1.0 - q)) - 1.0))
    elif wavelet in ("g1", "g3", "frac_gaussian"):
        n = {"g1": 1.0, "g3": 3.0}.get(wavelet, float(params.get("frac_n", 2.0)))
        radii = _frac_crossings(n)
        widths = [2.0 * radii[0]] + list(np.diff(radii))
    else:                                         # g2, lorentzian_marr: one crossing, at r
        widths = [2.0]
    return _LOBE_PX / min(widths)


def scale_warning(method: str, value, params: "dict | None") -> "str | None":
    """The r1 knob's live line: a warning below :func:`lobe_floor`, empty above it or where no
    floor applies -- empty, not None, so the label exists and can light up when a sibling knob
    (the kernel, q, the width) moves the floor."""
    floor = lobe_floor(method, params or {})
    if floor is None:
        return ""
    if float(value) < floor * (1.0 - 1e-9):
        return (f"⚠ below {floor:.2f} px a lobe of this wavelet is narrower than "
                f"{_LOBE_PX:g} px")
    return ""


def knob_warning(method: str, name: str, value, params: "dict | None") -> "str | None":
    """The holder tools' ``derived_reading``: r1's resolution floor (:func:`scale_warning`),
    q's clean range (:func:`q_warning`)."""
    if name == "r_min":
        return scale_warning(method, value, params)
    return q_warning(method, name, value, params)


def construction_check(method: str, params: dict) -> None:
    """A Fourier-built kernel needs a finite closed-form transform: the measure route refuses
    frac_gaussian, a q-Gaussian from q = 2 and a Lorentzian with beta <= 1 there
    (:func:`dynamix.core.microcanonical._check_construction`)."""
    from dynamix.core.microcanonical import _check_construction

    _check_construction(params.get("construction", "real"),
                        "measure" if method == "measure" else "marr", params.get("wavelet"),
                        float(params.get("beta", 1.0)), float(params.get("q_tsallis", 1.5)))


#: Where each kernel is built: "real" samples it on the pixel grid (Turiel's construction, the
#: default); "fourier" evaluates the same continuous kernel's closed-form 2-D transform,
#: band-limited instead of sampled, to compare the two (see
#: :data:`dynamix.core.microcanonical._CONSTRUCTIONS`).
_CONSTRUCTION = Param("construction", ParamKind.CHOICE, default="real",
                      choices=("real", "fourier"), label="Wavelet space")


#: The margin a Hölder estimate needs around an ROI is the
#: reach of its LARGEST kernel -- the radius holding all but ``_REACH_TOL`` of the kernel's 2-D
#: |mass| (the convolution error an ROI pixel can pick up from outside the margin), capped at
#: ``_REACH_CAP`` x that radius: a Lorentzian with beta <= 1 has a non-integrable 2-D tail
#: and never converges (Turiel 2008 appendix A -- its result is field-size dependent by
#: nature). Measured from the tools' OWN projection code (a delta through
#: measure_projections / ricker_projections), so no kernel formula is restated here.
_REACH_TOL = 1e-3
_REACH_CAP = 32


@functools.lru_cache(maxsize=128)
def _kernel_reach(method: str, wavelet: str, r: float, beta: float, q_tsallis: float,
                  frac_n: float, q_beta: "float | None" = None) -> int:
    import numpy as np

    from dynamix.core import microcanonical as mc

    L = int(np.ceil(_REACH_CAP * r))
    n = 2 * L + 1
    delta = np.zeros((n, n))
    delta[L, L] = 1.0
    kw = dict(wavelet=wavelet, beta=beta, q_tsallis=q_tsallis, frac_n=frac_n, pad=0,
              q_beta=q_beta)
    if method == "measure":
        K = np.abs(mc.measure_projections(delta, [r], **kw)[0])
    else:
        K = np.abs(mc.ricker_projections(delta, [r], **kw)[0])
    yy, xx = np.mgrid[0:n, 0:n]
    rho = np.hypot(yy - L, xx - L).ravel()
    order = np.argsort(rho)
    cum = np.cumsum(K.ravel()[order])
    cum /= cum[-1]
    idx = int(np.searchsorted(cum, 1.0 - _REACH_TOL))
    return int(np.ceil(rho[order][min(idx, rho.size - 1)]))


def holder_roi_margin(method: str, params: dict) -> int:
    """Pixels of real data a Hölder estimate needs around an ROI: the largest kernel's reach
    (r_max = r_min * kappa) plus the 1-px gradient stencil of the measure route."""
    r_max = float(params["r_min"]) * float(params["kappa"])
    reach = _kernel_reach(method, str(params["wavelet"]), r_max, float(params["beta"]),
                          float(params["q_tsallis"]), float(params.get("frac_n", 2.0)),
                          q_width(method, params))
    return reach + (1 if method == "measure" else 0)


#: Knobs shared by both methods, in holder_map's own order (one mental model): the Pont
#: estimator, then the Turiel scale grid. Defaults and bounds are holder_map's verbatim.
#: NOTE the r-knob's MEANING differs per method (Turiel 2008
#: eq 6-8): both tools use Turiel's pure-dilation convention Psi_r = r^-d Psi(x/r), but
#: the MEASURE tool's mother kernels are eq 7/8 verbatim -- so its r is the Gaussian's
#: STANDARD DEVIATION (positive kernels have no zero crossings; fig 2 imposes nothing
#: beyond pixel sampling, which is why this route localizes sharpest) -- while the
#: MULTIAFFINE tool's r is the ZERO-CROSSING radius (our concrete reading of the fig-2
#: rule: every wavelet in that menu is calibrated to change sign at radius r, so
#: "the discrete wavelet separates its +/- parts" reads as r >= 1 for one crossing, and
#: higher where more crossings or a compact edge make a lobe narrower -- lobe_floor). The same
#: r value is therefore NOT numerically comparable across the two tools.
_ESTIMATOR_AND_SCALES = (
    Param("estimator", ParamKind.CHOICE, default="regression",
          choices=("regression", "punctual"), label="Estimator"),
    Param("r_min", ParamKind.FLOAT, default=1.0, min=0.1, max=32.0,
          soft_min=1.0, soft_max=4.0, units="px", label="r₁"),
    Param("kappa", ParamKind.FLOAT, default=8.0, min=1.1, max=200.0,
          soft_min=4.0, soft_max=12.0, units="", label="κ = r₂/r₁"),
    Param("n_scales", ParamKind.INT, default=6, min=2, max=64,
          soft_min=4, soft_max=10, label="Scales"),
)

#: The family shape knobs: q binds only for the q-family, β only for the Lorentzians,
#: frac_n only for frac_gaussian (the inert-knob precedent holder_map set). q spans Tsallis's
#: whole [-1, 3] for the positive kernel (below 1 it is compactly supported: q = 0 the
#: Epanechnikov paraboloid, q = -1 a half-dome).
_FAMILY_KNOBS = (
    Param("q_tsallis", ParamKind.FLOAT, default=1.5, min=-1.0, max=3.0,
          soft_min=-1.0, soft_max=3.0, units="", label="Tsallis q"),
    Param("beta", ParamKind.FLOAT, default=1.0, min=0.1, max=8.0,
          soft_min=0.5, soft_max=2.0, units="", label="β"),
    Param("frac_n", ParamKind.FLOAT, default=2.0, min=0.05, max=8.0,
          soft_min=1.0, soft_max=3.0, units="", label="n"),
)

#: q_mexican is Borges et al. 2004's q-Mexican hat in 2-D (microcanonical._borges_q): a wavelet
#: for -1 <= q < 2 -- bounded for q >= 0, unbounded (still integrable) at its cutoff below 0,
#: and nonexistent from q = 2 up (the 2-D Laplacian vanishes at 2), so the knob stops at 1.95.
_MULTIAFFINE_KNOBS = (dataclasses.replace(_FAMILY_KNOBS[0], max=1.95, soft_max=1.95),) \
    + _FAMILY_KNOBS[1:]


def _width_knobs(q_wavelet: str) -> tuple:
    """The q-family width beta of Borges et al. 2004's ``e_q^(-beta x^2)``: FIXED (default 1/2 --
    sigma = r at q = 1 for the q-Gaussian, the zero crossing at r at q = 1 for the q-Mexican
    hat) or Q-PAIRED, ``beta(q) = 1/(2(2 - q))``: the escort (q-)variance held at one per
    component, the 2-D form of the paper's ``1/(3 - q)``, which also pins the q-Mexican hat's
    zero crossing at r for every q (its classic calibration). Both agree at q = 1. Greyed out
    unless the tool's q-kernel is chosen; the value knob also while paired."""
    return (
        Param("q_pairing", ParamKind.CHOICE, default="fixed", choices=("fixed", "q-paired"),
              label="q width", active_when=("wavelet", (q_wavelet,))),
        Param("q_beta", ParamKind.FLOAT, default=0.5, min=0.05, max=5.0, soft_min=0.1,
              soft_max=2.0, units="", label="q β",
              active_when=(("wavelet", (q_wavelet,)), ("q_pairing", ("fixed",)))),
    )


def q_warning(method: str, name: str, value, params: "dict | None") -> "str | None":
    """The q knob's live line (the ``derived_reading`` hook): empty while q is inside the chosen
    kernel's clean range -- empty, not None, so the label exists and can light up later -- and a
    warning where the kernel stops being well-behaved: the measure route's q-Gaussian at q >= 2
    (no finite 2-D mass, so the result depends on the field size -- the Lorentzian beta <= 1
    case), the multiaffine route's q_mexican below 0 (unbounded at its cutoff, still
    integrable). ``None`` for every other knob."""
    if name != "q_tsallis":
        return None
    wavelet = (params or {}).get("wavelet")
    if method == "measure" and wavelet == "q_gaussian" and float(value) >= 2.0:
        return "⚠ q ≥ 2: no finite 2-D mass — result depends on field size"
    if method == "multiaffine" and wavelet == "q_mexican" and float(value) < 0.0:
        return "⚠ q < 0: unbounded at its cutoff (still integrable)"
    return ""


def _compute(device, method: str, field, params: dict, progress=None) -> dict:
    vals = field_values_2d(field, device.name)
    h_map, r2_map, scales = method_arrays(vals, method, params, progress)
    return {
        "h_map": h_map, "r2_map": r2_map, "chains": [], "extrema": [],
        "scales": scales, "params": dict(params),
        "_frame": getattr(field, "frame", None), "_shape": vals.shape,
    }


class HolderMeasure:
    name = "holder_measure"
    params = _ESTIMATOR_AND_SCALES + (
        # Positive unit-mass kernels only (log T must exist) — Turiel 2008 appendix A's
        # resolution trade-off: Gaussian resolves every h, Lorentzian truncates at
        # h < 2β - d but localizes sharpest, q-Gaussian interpolates, frac_gaussian bends
        # the ENVELOPE (n < 2 fattens the tail, n > 2 flattens the top).
        Param("wavelet", ParamKind.CHOICE, default="gaussian",
              choices=("gaussian", "q_gaussian", "lorentzian", "frac_gaussian"),
              label="Wavelet"),
    ) + _FAMILY_KNOBS + _width_knobs("q_gaussian") + (_CONSTRUCTION,)

    def check(self, params: dict) -> None:
        q_width_check("measure", params)
        construction_check("measure", params)

    def compute(self, field, params: dict, *, progress=None) -> dict:
        return _compute(self, "measure", field, params, progress)

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        return knob_warning("measure", name, value, params)

    def roi_margin(self, params: dict) -> int:
        """The largest kernel's measured 2-D reach
        (:func:`holder_roi_margin`)."""
        return holder_roi_margin("measure", params)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)


class HolderMultiaffine:
    name = "holder_multiaffine"
    params = _ESTIMATOR_AND_SCALES + (
        # Zero-mean wavelets with n vanishing moments, named the method's way: the g-ladder
        # is the derivative-order axis (order n measures only γ < n; g2 = the Ricker, the
        # conflated device's default), frac_gaussian makes the order a REAL knob (frac_n).
        # All share the zero-crossing-radius scale convention (r_min = 1 is the fig-2
        # minimum for one crossing; r1's live line warns where a lobe is narrower).
        Param("wavelet", ParamKind.CHOICE, default="g2",
              choices=("g1", "g2", "g3", "q_mexican", "lorentzian_marr", "frac_gaussian"),
              label="Wavelet"),
    ) + _MULTIAFFINE_KNOBS + _width_knobs("q_mexican") + (_CONSTRUCTION,)

    def check(self, params: dict) -> None:
        q_width_check("multiaffine", params)
        construction_check("multiaffine", params)

    def compute(self, field, params: dict, *, progress=None) -> dict:
        return _compute(self, "multiaffine", field, params, progress)

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        return knob_warning("multiaffine", name, value, params)

    def roi_margin(self, params: dict) -> int:
        """The largest kernel's measured 2-D reach
        (:func:`holder_roi_margin`)."""
        return holder_roi_margin("multiaffine", params)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
