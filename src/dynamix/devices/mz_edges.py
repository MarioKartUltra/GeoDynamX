# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""mz_edges — the Mallat-Zhong multiscale edges as a peer Transform device.

Two engines sit behind the Algorithm knob. ``"lastwave"`` (the default) is the authors' own 2-D
pipeline, ported exactly from LastWave's ``dwtrans2d`` (``core.mz_lastwave``): the dyadic
transform by direct FIR convolution, run on the field's 2N mirror (Border = mirror) or on the
field itself as a torus (periodic), ``extrema2``'s detection (level 1 optionally co-located) and
``e2recons``' alternating projections. ``"printed"`` is the algorithm as the paper prints it
(``core.mz_edges`` over ``mzlib``), kept selectable for comparison with the research scripts;
Dither, Interpolate and Mode are its own knobs. Both run the dyadic spline wavelet.

Both emit the app's extrema/line_id schema per dyadic scale 2**l (l = 1..J). Cross-scale V-chains
are Mallat-Hwang SVII machinery (reserved script 11) -- "chains" is stamped empty deliberately. A
LastWave result carries ``_display_offset = (-0.5, -0.5)``: its maxima and its coarse channel
register half a pixel west and north of the pixel index they are stored at.

The edges are the analysis; everything else the tool shows is a LAZY output computed from the
cached analysis on request (``compute_output``, through ``engine.resolve.resolve_output``): the
full-resolution coarse channel S_J, its 2^J thumbnail, the reconstruction from the multiscale
edges with S_J pinned or with S = 0, the residual (field minus reconstruction), and for the
LastWave engine the one-iteration preview a reconstruction continues from. The reconstruction
knobs (``section="reconstruction"``) and Show are view-only, so the analysis always runs with the
full coarse channel and flipping one of them never re-runs it.
"""
from __future__ import annotations

import numpy as np

from dynamix.model.device import Output, keyed_params
from dynamix.model.param import Param, ParamKind

_LASTWAVE = ("algorithm", ("lastwave",))
_PRINTED = ("algorithm", ("printed",))
_RECON = "reconstruction"

#: Every setting a reconstruction reads; each keys its own cache entry.
_RECON_PARAMS = ("kappa", "clip", "run_mode", "iterations", "tolerance", "coarse", "mode")

#: Where LastWave's maxima and coarse channel register relative to their pixel index (x, y).
_DISPLAY_OFFSET = (-0.5, -0.5)


class MZEdges:
    name = "mz_edges"
    params = (
        Param("n_levels", ParamKind.INT, default=4, min=1, max=10,
              soft_min=2, soft_max=6, label="Levels J"),
        Param("algorithm", ParamKind.CHOICE, default="lastwave",
              choices=("lastwave", "printed"), label="Algorithm"),
        # LastWave's transform is periodic: "mirror" runs it on the field's 2N mirror and crops,
        # "periodic" on the field as given.
        Param("border", ParamKind.CHOICE, default="mirror", choices=("mirror", "periodic"),
              label="Border", active_when=_LASTWAVE),
        # Detect level 1 on the 2-tap co-located gradient, which sits where the coarser levels
        # do; the stored values stay the raw level-1 gradient.
        Param("colocate_l1", ParamKind.BOOL, default=False, label="Co-locate level 1",
              active_when=_LASTWAVE),
        Param("dither", ParamKind.BOOL, default=False, label="Dither", active_when=_PRINTED),
        # The wtmm2d subpixel/value refinement, on the dyadic maxima. POCS input (mz_maxima)
        # stays integer-supported regardless -- reconstruction is pixel-exact by the papers'
        # own numerics.
        Param("interpolate", ParamKind.BOOL, default=False, label="Interpolate",
              active_when=_PRINTED),
        # The projection's decay a = exp(-kappa / 2**l): 1 is the published constant, 2 ln 5.8
        # (about 3.516) LastWave's, which the engine then computes with the C's own expression.
        Param("kappa", ParamKind.FLOAT, default=1.0, min=0.1, max=10.0,
              soft_min=0.5, soft_max=4.0, label="Decay κ", view=True, section=_RECON,
              active_when=_LASTWAVE),
        Param("clip", ParamKind.BOOL, default=False, label="Clip", view=True, section=_RECON,
              active_when=_LASTWAVE),
        # "converge" iterates until the relative change drops under Tolerance, capped at
        # Iterations; "fixed" runs Iterations. Iterations is the total in both.
        Param("run_mode", ParamKind.CHOICE, default="converge", choices=("converge", "fixed"),
              label="Run mode", view=True, section=_RECON, active_when=_LASTWAVE),
        Param("iterations", ParamKind.INT, default=20, min=1, max=500,
              soft_min=1, soft_max=50, label="Iterations", view=True, section=_RECON),
        Param("tolerance", ParamKind.FLOAT, default=1e-3, min=1e-6, max=1e-1,
              label="Tolerance", view=True, section=_RECON, active_when=_LASTWAVE),
        # Which coarse channel the reconstruction pins: S_J at full resolution, or the one
        # decoded from its 2^J thumbnail (which needs the grid divisible by 2^J).
        Param("coarse", ParamKind.CHOICE, default="full",
              choices=("full", "thumbnail"), label="Coarse", view=True, section=_RECON),
        # How the printed algorithm's P_Gamma interpolates between the maxima: the vectorized
        # separable operator, the same one row by row through mzlib (the reference), or the
        # maxima values alone. The choices are ``core.mz_edges.RECON_MODES``, spelled here so
        # the device module stays free of the core import.
        Param("mode", ParamKind.CHOICE, default="separable",
              choices=("separable", "separable_mzlib", "set_points"), label="Mode",
              view=True, section=_RECON, active_when=_PRINTED),
        # The raster output drawn in place of the field ("edges": the field itself, with the
        # maxima over it); each other choice names a lazy output below.
        Param("show", ParamKind.CHOICE, default="edges",
              choices=("edges", "coarse", "thumbnail", "recon", "recon_edges_only",
                       "residual"), label="Show", view=True),
    )

    outputs = (
        Output("edges", "vector", label="edges"),
        Output("coarse", "raster", lazy=True, label="coarse"),
        Output("thumbnail", "raster", lazy=True, label="thumbnail", grid="stride"),
        Output("recon", "raster", lazy=True, params=_RECON_PARAMS,
               label="recon (edges + coarse)"),
        Output("recon_edges_only", "raster", lazy=True,
               params=tuple(p for p in _RECON_PARAMS if p != "coarse"),
               label="recon (edges only)"),
        Output("residual", "raster", lazy=True, params=_RECON_PARAMS, label="residual"),
        # The LastWave initial pass and one iteration, which a reconstruction with the same
        # decay, clipping and coarse continues from; drawn on the recon row.
        Output("recon_preview", "raster", lazy=True, params=("kappa", "clip", "coarse"),
               label="recon preview", row=False),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        from dynamix.core import mz_edges

        values = np.asarray(field.values)
        mz_edges._require_finite(values, "M–Z edge detection")
        if params["algorithm"] == "lastwave":
            bundle = self._lastwave(values, params, progress)
        else:
            # Always the full coarse channel: the thumbnail is derived from it on request, and
            # the Coarse knob only chooses what a reconstruction pins.
            bundle = mz_edges.analyze(
                values,
                params["n_levels"],
                coarse="full",
                dither=params["dither"],
                interpolate=params["interpolate"],
                progress=progress,
            )
        bundle["algorithm"] = params["algorithm"]
        bundle["chains"] = []
        bundle["params"] = dict(params)
        bundle["_frame"] = field.frame
        bundle["_shape"] = values.shape
        return bundle

    @staticmethod
    def _lastwave(values, params: dict, progress) -> dict:
        """The LastWave analysis bundle: per level the extrema on the field's grid (row-major),
        ``mod`` the magnitude normalised as ``extrema2`` stores it; ``_lastwave`` is
        ``(S_J, extrep)``, the working field's fact-scaled coarse channel and its extrema, which
        is all the lazy outputs read. The rest of the transform (every level's S, Wx and Wy on
        the working field, the 2N mirror by default) is not kept with the cached result."""
        from dynamix.core import mz_lastwave as lw

        J = int(params["n_levels"])
        if progress is not None:
            progress("M–Z (LastWave) analysis", 0.0)
        t, ex = lw.analyze(values, J, border=params["border"],
                           colocate_l1=params["colocate_l1"])
        extrema = []
        for l in range(1, J + 1):
            mask, mag, arg = lw.primary_extrema(t, ex, l)
            y, x = np.nonzero(mask)
            extrema.append({"x": x.astype(np.int64), "y": y.astype(np.int64),
                            "mod": mag[mask], "arg": arg[mask],
                            "line_id": _line_ids(mask)})
        if progress is not None:
            progress("M–Z (LastWave) analysis", 1.0)
        return {
            "extrema": extrema,
            "scales": (2.0 ** np.arange(1, J + 1)).astype(np.float64),
            "_lastwave": (t.S_full[J], ex),
            "_display_offset": _DISPLAY_OFFSET,
        }

    def view(self, result: dict, params: dict) -> dict:
        """The view-only knobs pick or shape a lazy output, which the engine computes and caches
        under its own key; the analysis arrays are shown unchanged. The returned copy's
        ``params`` carry the current view-only knobs (the cached bundle keeps the ones it was
        first computed with), so a label read from ``params["show"]`` names what is shown."""
        current = {p.name: params[p.name] for p in self.params if p.view}
        return {**result, "params": {**result.get("params", {}), **current}}

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None) -> dict:
        """The lazy output ``name`` from the cached analysis ``result`` of ``values``:
        ``{"raster": float32 ndarray, "diag": dict}``, plus ``display_stride``/``full_dims``
        for the thumbnail, whose samples sit every 2^J pixels of the (ny, nx) field. The
        residual is ``values`` minus the recon ``fetch`` returns, so it reuses that one's work.
        On the LastWave engine the coarse and the thumbnail also carry ``display_offset``
        (``(dx, dy)`` in pixels, where their samples register), and ``recon``,
        ``recon_edges_only`` and ``recon_preview`` carry ``state``, the ``ReconState`` a
        further run continues from."""
        values = np.asarray(values)
        if name == "residual":
            recon = fetch("recon")
            return {"raster": (values - recon["raster"]).astype(np.float32),
                    "diag": recon["diag"]}
        if params["algorithm"] == "lastwave":
            return self._lastwave_output(name, values, result, params, fetch=fetch,
                                         progress=progress, cancel=cancel)
        from dynamix.core import mz_edges

        if name == "coarse":
            return {"raster": mz_edges.coarse_image(values, result).astype(np.float32),
                    "diag": {}}
        if name == "thumbnail":
            return {"raster": mz_edges.coarse_thumbnail(values, result), "diag": {},
                    "display_stride": 2 ** len(result["mz_maxima"]),
                    "full_dims": tuple(values.shape)}
        if name in ("recon", "recon_edges_only"):
            coarse = params["coarse"] if name == "recon" else "none"
            img, diag = mz_edges.reconstruct(values, result, n_iter=params["iterations"],
                                             mode=params["mode"], coarse=coarse,
                                             progress=progress, cancel=cancel)
            diag = {**diag, "iterations": diag["n_iter"], "stop": "fixed"}
            return {"raster": img.astype(np.float32), "diag": diag}
        if name == "recon_preview":
            raise ValueError(f"{self.name}: the one-iteration preview is the LastWave "
                             f"engine's; this result ran algorithm 'printed' (choose "
                             f"'lastwave')")
        raise ValueError(f"{self.name}: no lazy output named {name!r}")

    def _lastwave_output(self, name, values, result, params, *, fetch, progress, cancel):
        """The LastWave engine's lazy outputs. The coarse channel is S_J divided by the
        ``fact(J)`` the transform multiplied it by, back in the field's units; the
        reconstructions pin the scaled working-field S_J itself. ``recon`` continues from the
        ``recon_preview`` of the same decay, clipping and coarse, so the preview's iteration is
        never run twice: Iterations is the total, which a fixed run completes and a converge run
        caps."""
        from dynamix.core import mz_lastwave as lw

        S_J, ex = result["_lastwave"]
        J = ex.J
        ny, nx = values.shape
        if name in ("coarse", "thumbnail"):
            coarse = S_J[:ny, :nx] / lw.fact(J)
            if name == "coarse":
                return {"raster": coarse.astype(np.float32), "diag": {},
                        "display_offset": _DISPLAY_OFFSET}
            step = 2 ** J
            return {"raster": coarse[::step, ::step].astype(np.float32), "diag": {},
                    "display_stride": step, "full_dims": tuple(values.shape),
                    "display_offset": _DISPLAY_OFFSET}
        knobs = dict(kappa=params["kappa"], clip=params["clip"], border=params["border"],
                     progress=progress, cancel=cancel)
        if name == "recon_preview":
            img, diag, state = lw.e2recons(values, ex, S_J, J, k=1, mode="fixed",
                                           state=None, coarse=params["coarse"], **knobs)
        elif name in ("recon", "recon_edges_only"):
            total, mode = int(params["iterations"]), params["run_mode"]
            state, k, coarse = None, total, "none"
            if name == "recon":
                coarse = params["coarse"]
                state = fetch("recon_preview")["state"]
                if mode == "fixed":
                    k = max(0, total - state.iterations)
            img, diag, state = lw.e2recons(values, ex, S_J, J, k=k, mode=mode,
                                           tol=params["tolerance"], state=state,
                                           coarse=coarse, **knobs)
        else:
            raise ValueError(f"{self.name}: no lazy output named {name!r}")
        return {"raster": img.astype(np.float32), "diag": diag, "state": state}

    def roi_margin(self, params: dict) -> int:
        """How far outside an ROI the analysis reads, so the window's own border (the mirror
        seam or the periodic wrap the transform folds onto the WINDOW edge) never reaches it.

        LastWave: exact from the FIR filters. The level-l gradient reads the input through the
        H1 chain of levels 1..l-1 and the level's G1, each filter reaching its first-tap offset
        plus (taps - 2) dilations; detection then compares each point with its neighbours 1 px
        away, 2 px at level 1 (the gap-filling pass reads extrema decided 1 px further out), one
        more with Co-locate level 1 (the 2-tap average). Printed: the transform's own measured
        impulse reach at the coarsest level (``mz_edges.mz_impulse_reach``) + 2 px for the
        bilinear NMS probes and the subpixel refinement."""
        J = int(params["n_levels"])
        if params["algorithm"] == "lastwave":
            return _lastwave_margin(J, bool(params["colocate_l1"]))
        from dynamix.core.mz_edges import mz_impulse_reach

        return mz_impulse_reach(J) + 2

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, keyed_params(self, params))


def _line_ids(mask) -> np.ndarray:
    """``line_id`` of the points of ``mask`` in row-major order: the 8-connected component of the
    mask each belongs to (0-based), or -1 for a component of one point (not a line)."""
    from scipy import ndimage

    labels, _n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    ids = labels[mask]
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return np.where(counts[inverse] == 1, -1, ids - 1).astype(np.int64)


def _fir_reach(name: str, scale: int) -> int:
    """How far (px, on either side) one LastWave filter reads at dilation ``scale``."""
    from dynamix.core.mz_lastwave.transform import FILTERS, _l1r1

    size, shift, _sym, _taps = FILTERS[name]
    if size < 2:
        return 0
    l1, r1 = _l1r1(shift, scale)
    return max(l1, r1) + (size - 2) * scale


def _lastwave_margin(J: int, colocate_l1: bool) -> int:
    """The widest reach of any level's detection (see ``MZEdges.roi_margin``)."""
    margin, chain = 0, 0
    for l in range(1, J + 1):
        scale = 2 ** (l - 1)
        probe = 2 + int(colocate_l1) if l == 1 else 1
        margin = max(margin, chain + _fir_reach("G1", scale) + probe)
        chain += _fir_reach("H1", scale)
    return margin
