# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Per-wavelet scale -> spatial conversion: the one authority (spec branch 1).

The unit ladder in ``dynamix/roi/halo.py`` is correct for the Gaussian smoother and stays;
this module is its per-wavelet generalization (halo's docstring points here). A nominal 2D
CWT scale ``s`` (from ``compute_scales2d``, xsmurf ``norm = 6.0/0.86`` included) drives the
Fourier filter ``exp(-(s|k|)^2)`` with ``k`` in cycles/px, whose real-space Gaussian has

    sigma = s / (pi * sqrt(2))  ~  0.22508 * s   px.

The ANALYZING wavelet of the WTMM path is ``d/dx`` of the smoother -- a directional dipole,
not isotropic. What this module samples for display is therefore the kernel's TRUE 2D
sections:

- theta (smoother): the isotropic radial profile. Gaussian: ``exp(-u^2/2)``. Mexican
  (``|k|^2 * gauss`` in Fourier is ``-laplacian(G)`` in real space -- the (2 pi i k)^2 sign):
  ``(2 - u^2) exp(-u^2/2)``, the positive-center Mexican hat -- NOT the 1D mexican hat
  ``(u^2 - 1)e^{-u^2/2}``; the 2D profile's zero sits at sqrt(2) sigma.
- psi (analyzing): the along-gradient section (y = 0). Gaussian path: ``-u e^{-u^2/2}``
  (exactly g1). Mexican path: ``d/dx[-laplacian(G)]`` along x, the two sign flips cancelling
  into ``(u^3 - 4u) e^{-u^2/2}`` -- NOT the 1D g3 ``(u^3 - 3u)e^{-u^2/2}``; outer
  extrema at ``u^2 = (7 + sqrt(33))/2`` (+-2.5243 sigma) vs g3's +-2.3344 sigma.

The g3 label is not wrong -- it names a different object. For a feature invariant along y
(a straight edge, which is what a WTMM chain locally is) the response kernel is
``integral of psi_x over y``, and ``integral(g'') = 0`` kills the transverse term: the
EDGE-RESPONSE kernel is exactly the 1D g3 ``(u^3 - 3u)e^{-u^2/2}``. The section drawn here
and the edge response are both true; do not "correct" one into the other.

The per-wavelet reading is the bandpass peak wavelength ``lambda = 2 pi sigma / sqrt(m)``
(filter ``|k|^m exp(-s^2|k|^2)``; m = 1 gaussian-psi, 2 mexican-theta, 3 mexican-psi) --
NOT an outer-extremum footprint, whose vestigial outer lobe (12.7% of the inner, mexican
path) orders the wavelets backwards: mexican probes FINER structure at the same nominal
scale. Nyquist floor: lambda >= 2 px <=> s >= sqrt(2m) (1.41/2.00/2.45 for m = 1/2/3); the
finest ladder scale 6.977*a_min clears the strictest floor ~3x at a_min = 1 -- do not lower
a_min blind. Amplitude normalization: with f(x) = integral F(k) e^{+2 pi i k x} dk, the
pipeline's ``i*sx`` filter is ``s * d/dx / (2 pi)`` -- positive constant, cancels in every
log-log slope, but bites raw-amplitude comparisons against analytic or xsmurf numbers.

Display names keep the community's g-family labels (theta g0/g2, psi g1/g3); the curves
drawn are the 2D sections above. ``tests/test_scale_units.py`` pins every constant against
an inverse FFT of the actual pipeline filters (``_build_wavelet_filters_numpy``) -- the
oracle, not this docstring, is the ground truth.

Adding a wavelet is one ``_Kernel`` entry per (smoothing, deriv_order) pair: a profile
callable in ``u = x/sigma`` units and its outer-extremum position in sigma units.

xsmurf provenance (the primary tree ``xsmurf`` lacks ``tcl_library/``, so the Tcl
citations below are the byte-identical copy at ``xSmurfMetal/xsmurf``, confirmed
identical against ``xSmurfMacPorts/xsmurf``):

``6.0/0.86`` is xsmurf's own scale-ladder normalization, applied AFTER the octave/voice
power-of-2 ladder -- found as ``set scale [expr $scale*(6/0.86)]`` in
``tcl_library/wt.tcl:66`` (the ``iwt`` proc), and reproduced identically in
``tcl_library/imStudy.tcl:2082,2107,2113,2121`` (``imStudy::GetScale``),
``tcl_library/study.tcl:976-977,999-1000`` (``compute_wtmm``, gaussian) and
``tcl_library/study.tcl:1061-1062,1084-1085`` (``compute_mexican``) -- 30+ occurrences across
``wt.tcl``, ``study.tcl``, ``imStudy.tcl``, ``mmto.tcl``, ``to.tcl``, ``hpcal_proc.tcl``. Every
occurrence writes the SAME two bare literals as a ratio, ``(6/0.86)``, never pre-combined into
one decimal, and NONE is ever commented on -- no hit anywhere in the tree ties either number to
"support", "halo width", or "calibration". The pair reads as an inherited empirical
normalization from the original 1998 CRPP Bordeaux Tcl library (wt.tcl's own
copyright: "Written by Nicolas Decoster"), copied forward unchanged through Kestener's
1999-2007 extensions into every scale-generating script (always two literals, always the
same split -- consistent with two once-meaningful numbers whose individual meaning was never
written down, not with a single fitted constant). The numerological match
``0.86 ~ 12/(pi^2*sqrt(2)) = 0.85974`` has no support in how the literal is written:
a bare 2-decimal number ratioed against a bare integer, never expressed via pi or sqrt anywhere
in the tree -- reads as coincidence, not the origin.

Per-wavelet a0 in the TRANSFORM LADDER: none. ``compute_wtmm`` (gaussian,
``study.tcl:969-1052``) and ``compute_mexican`` (``study.tcl:1054-1135``) compute ``scale``
from byte-identical lines; ``imStudy::WtmmgCurrentScale`` (``imStudy.tcl:1054`` ff.) uses one
shared ``$scale`` for both the ``gaussian`` and ``mexican`` switch branches -- only the FILTER
EXPRESSION differs (``gaussianDef``/``mexicanDef`` arrays, ``imStudy.tcl:103-144``), never the
scale value. ``compute_scales2d``'s single ``norm`` for every wavelet is therefore not a
DynamiX simplification of something xsmurf varies per wavelet -- it is what xsmurf itself does.

Fourier convention: xsmurf's Gaussian is written in cycles/px, not angular
frequency -- confirmed two ways in the primary C++ tree. The built-in filter path,
``im_fourier_conv_`` (``image_cpp/wt2d.cpp:66-133``), builds frequency via ``_ImaCvlInit_``
(``wt2d.cpp:29-61``: ``_freq1_[i] = i/sizeX``, standard fftfreq-style cycles/px) and computes
``exp(-scale^2*(kx^2+ky^2))`` -- no ``1/2``. The user-expression path, ``im_mult_analog``
(``wt2d.cpp:356-437``, fed ``x = i*scale/lx``), is driven by the exact strings in
``imStudy.tcl:103-144``: ``gaussianDef(dx,i) = x*exp(-x*x-y*y)``,
``mexicanDef(dx,i) = x*(x*x+y*y)*exp(-x*x-y*y)`` -- ``x``/``y`` are ``scale*k`` in cycles/px,
the SAME exponent this module and ``_build_wavelet_filters_numpy`` build (``gauss =
exp(-(sx**2+sy**2))``, mexican = ``|k|^2 * gauss``). Because the convention already matches,
``6.0/0.86`` is not a units bridge between angular and cycles/px frequency (which would have
forced a ladder change) -- it is purely a scale NORMALIZATION, of unexplained
origin, that both this module and ``compute_scales2d`` inherit unchanged from xsmurf.

Pure numpy; no Qt, no wtmm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["SIGMA_PER_SCALE_EXACT", "analyzing_names", "kernel_section", "lambda_peak_px",
           "outer_extremum_sigma", "sigma_px"]

#: Exact form of halo.py's rounded 0.225: real-space Gaussian sigma per unit nominal scale.
SIGMA_PER_SCALE_EXACT = 1.0 / (math.pi * math.sqrt(2.0))

#: Community names for the smoother/analyzing pair, per smoothing function. Kept local --
#: this module must not import wtmm (see the module docstring's naming note).
_NAMES = {"gaussian": ("g0", "g1"), "mexican": ("g2", "g3")}


@dataclass(frozen=True)
class _Kernel:
    """One registered section: profile in u = x/sigma units, outer extremum in sigma units,
    and the Fourier radial order m (filter ∝ |k|^m exp(-s^2|k|^2); 0 = pure lowpass)."""
    profile: Callable[[np.ndarray], np.ndarray]
    outer_extremum_u: float
    bandpass_m: int


_KERNELS: dict[tuple[str, int], _Kernel] = {
    ("gaussian", 0): _Kernel(lambda u: np.exp(-u * u / 2.0), 0.0, 0),
    ("gaussian", 1): _Kernel(lambda u: -u * np.exp(-u * u / 2.0), 1.0, 1),
    ("mexican", 0): _Kernel(lambda u: (2.0 - u * u) * np.exp(-u * u / 2.0), 2.0, 2),
    ("mexican", 1): _Kernel(lambda u: (u ** 3 - 4.0 * u) * np.exp(-u * u / 2.0),
                            math.sqrt((7.0 + math.sqrt(33.0)) / 2.0), 3),
}


def sigma_px(scale: float) -> float:
    """Real-space Gaussian sigma in px for a nominal (normalized) 2D CWT scale."""
    return float(scale) * SIGMA_PER_SCALE_EXACT


def analyzing_names(smoothing: str) -> tuple[str, str]:
    """``(theta_name, psi_name)`` display names for a smoothing function."""
    try:
        return _NAMES[str(smoothing)]
    except KeyError:
        raise ValueError(f"unknown smoothing {smoothing!r}; expected one of {sorted(_NAMES)}")


def outer_extremum_sigma(smoothing: str, deriv_order: int) -> float:
    """Outermost extremum position of the registered section, in sigma units."""
    return _KERNELS[(str(smoothing), int(deriv_order))].outer_extremum_u


def lambda_peak_px(scale: float, smoothing: str, deriv_order: int = 1) -> float:
    """Bandpass peak wavelength in px: 2*pi*sigma/sqrt(m) for a |k|^m exp(-s^2|k|^2)
    filter -- the per-wavelet number the readings quote (an outer-extremum footprint orders
    the wavelets backwards; lambda is the metric with an operational meaning). Raises
    ValueError for a pure-lowpass entry (m = 0): a smoother has no bandpass wavelength."""
    m = _KERNELS[(str(smoothing), int(deriv_order))].bandpass_m
    if m == 0:
        raise ValueError(f"({smoothing!r}, {deriv_order}) is pure lowpass: no bandpass wavelength")
    return 2.0 * math.pi * sigma_px(scale) / math.sqrt(m)


def kernel_section(scale: float, smoothing: str, deriv_order: int,
                   n_points: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Sample the section over its own support: +-(outer + 3) sigma, at least +-4 sigma.

    Returns ``(x_px, y)`` -- x in true pixel units, amplitude unnormalized (the canvas
    normalizes display height itself; see ``wavelet_bar_scale_factor``)."""
    entry = _KERNELS[(str(smoothing), int(deriv_order))]
    span_u = max(entry.outer_extremum_u + 3.0, 4.0)
    sig = sigma_px(scale)
    u = np.linspace(-span_u, span_u, int(n_points))
    return u * sig, entry.profile(u)
