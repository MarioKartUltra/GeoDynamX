# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""wavelet_skeleton — the Tang-You medial-axis extractor as a peer primary analyzer.

The third representation next to ``wtmm2d`` and ``mz_edges`` (each method its own tool): the
CWT gradient-modulus machinery with the method's OWN wavelet, the Tang-You 2003 constructed
compact-support kernel (support radius = the scale; the width-invariance theorems hold for it
and for neither the Gaussian nor the spline), and You et al. 2006's modulus-MINIMA thinning —
Algorithm 1 over :mod:`dynamix.core.wavelet_skeleton`, deviations documented there.

Output speaks the app's per-scale extrema schema (x/y/mod/arg/line_id — the ``mz_edges``
precedent), so the shell renders skeletons through the existing display path unchanged:
``mod``/``arg`` are the STAGE-1 FIELD transform's values at the skeleton points
(physically meaningful — the mask-stage moduli are mask units), ``line_id`` labels
8-connected skeleton curves with the −1 isolated-point sentinel, ``scales`` is ``[s1]``
(the field-analysis scale). ``skeleton_mask``/``initial_mask`` ride along for any
downstream consumer (sieve, export). ``chains`` is stamped empty deliberately — cross-
scale linking is the papers' multiscale-processing refinement, reserved.

Standalone transform (heads its own chain): it analyses the LAYER'S FIELD, so a placement
after another Transform is refused with the shared pointed message.
"""
from __future__ import annotations

from dynamix.devices.holder_map import field_values_2d
from dynamix.model.param import Param, ParamKind


class WaveletSkeleton:
    name = "wavelet_skeleton"
    params = (
        # Scales are the theta support RADIUS in px (scales-in-pixels law; the paper's
        # maxima separation == s). The method's own hypothesis: s1 must COVER the ribbon
        # width — below it the medial signal honestly vanishes (core docstring).
        Param("s1", ParamKind.FLOAT, default=6.0, min=2.0, max=64.0,
              soft_min=4.0, soft_max=16.0, units="px", label="s₁"),
        Param("s2", ParamKind.FLOAT, default=6.0, min=2.0, max=64.0,
              soft_min=4.0, soft_max=16.0, units="px", label="s₂"),
        # Valley contrast: keep minima at most this fraction of their flanking ridge —
        # the paper's absolute threshold T made local and scale-free.
        Param("t_frac", ParamKind.FLOAT, default=0.5, min=0.1, max=0.9,
              soft_min=0.3, soft_max=0.7, units="", label="T/ridge"),
        # Significance: the flanking ridge PAIR must clear this fraction of the peak
        # modulus (peak-relative -- quantiles over mostly-flat rasters are
        # background-dominated; core docstring).
        Param("edge_frac", ParamKind.FLOAT, default=0.1, min=0.01, max=0.6,
              soft_min=0.05, soft_max=0.3, units="", label="Edge/peak"),
        Param("n_stages", ParamKind.INT, default=2, min=1, max=5,
              soft_min=2, soft_max=3, label="Stages"),
        # What the ribbon detector runs over (block-mountain DEM): the
        # medial-axis machinery needs a RIBBON -- a valley between an edge PAIR -- and a
        # fault scarp is a single EDGE, invisible to it on the raw signal. "gradient" runs
        # the skeleton over ||grad s|| (central differences, the eq-28 measure), where a
        # diffused scarp IS a ribbon of width ~ sqrt(2 kappa t) and its medial axis is the
        # fault trace. "signal" stays the papers' own stroke/ribbon reading.
        Param("input", ParamKind.CHOICE, default="signal",
              choices=("signal", "gradient"), label="Input"),
        # The papers' own comparison set (Tang-You 2003 benchmarks kappa against "Gaussian
        # function and quadratic spline"): the theorems hold for tang_you; the other two
        # are the paper's baselines, offered for exactly that comparison.
        Param("wavelet", ParamKind.CHOICE, default="tang_you",
              choices=("tang_you", "gaussian", "bspline2"), label="Wavelet"),
        # Pair dominance (formerly a constant, now a knob): a valley whose flanking
        # pair is under this fraction of the strongest reachable ridge is on someone
        # else's slope. 0 disables the gate.
        Param("pair_dom", ParamKind.FLOAT, default=0.5, min=0.0, max=0.9,
              soft_min=0.3, soft_max=0.7, units="", label="Pair dom."),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        import numpy as np

        from dynamix.core import wavelet_skeleton as wsk

        vals = field_values_2d(field, self.name)
        if params["input"] == "gradient":
            from dynamix.core.microcanonical import gradient_measure

            vals = gradient_measure(vals)
        if progress is not None:
            progress("wavelet skeleton", 0.0)
        core = wsk.skeletonize(vals, s1=params["s1"], s2=params["s2"],
                               t_frac=params["t_frac"],
                               edge_frac=params["edge_frac"],
                               n_stages=int(params["n_stages"]),
                               kernel=params["wavelet"],
                               pair_dom=params["pair_dom"],
                               progress=(None if progress is None else
                                         (lambda st, f: progress(st, 0.9 * float(f)))))
        if progress is not None:
            progress("wavelet skeleton", 0.9)
        ys, xs = np.where(core["skeleton"])
        if ys.size:
            from scipy.ndimage import label

            labels, _n = label(core["skeleton"], structure=np.ones((3, 3), dtype=bool))
            ids = labels[ys, xs].astype(np.int64) - 1          # dense, 0-based
            _uniq, inverse, counts = np.unique(ids, return_inverse=True,
                                               return_counts=True)
            line_id = np.where(counts[inverse] == 1, -1, ids).astype(np.int64)
        else:
            line_id = np.zeros(0, dtype=np.int64)
        extrema = [{
            "x": xs.astype(np.int64),
            "y": ys.astype(np.int64),
            "mod": core["mod"][ys, xs],
            "arg": core["arg"][ys, xs],
            "line_id": line_id,
        }]
        if progress is not None:
            progress("wavelet skeleton", 1.0)
        return {
            "extrema": extrema, "chains": [],
            "scales": np.asarray([float(params["s1"])]),
            "skeleton_mask": core["skeleton"], "initial_mask": core["initial"],
            "params": dict(params),
            "_frame": getattr(field, "frame", None), "_shape": vals.shape,
        }

    def roi_margin(self, params: dict) -> int:
        """Reach adds up per stage -- the s1 kernel's measured reach + the
        gradient (1) + the directional minima search (ceil s1) + its 1-px flank, then per
        refinement stage the s2 kernel + gradient + search + flank + the locality dilation
        (ceil s2/2); +1 when the input is the gradient measure. Oracle-pinned."""
        import numpy as np

        from dynamix.core.wavelet_skeleton import theta_reach

        kernel = str(params["wavelet"])
        s1, s2 = float(params["s1"]), float(params["s2"])
        m = theta_reach(s1, kernel) + 1 + int(np.ceil(s1)) + 1
        stage = theta_reach(s2, kernel) + 1 + int(np.ceil(s2)) + 1 + int(np.ceil(s2 / 2.0))
        m += (int(params["n_stages"]) - 1) * stage
        return m + (1 if params["input"] == "gradient" else 0)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
