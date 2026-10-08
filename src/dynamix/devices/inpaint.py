# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""inpaint and fill -- the inpainting engine as a Reconstruction tool and as a field stage.

``inpaint`` fills the declared regions of the host band (or of ``target_band`` on a stack) from
the band's own cascade, from cycle-spun wavelet sparsity, or both, with co-registered guides
sent to it each on its own grid (``dynamix.core.inpaint``), and reports the error on held-out
blocks and whether the fill kept the field's cascade. ``fill`` is the same engine as a field
stage ahead of an analyzer: the clone carries the fill's stamps so a downstream reading says
that analysis inside the region reads a reconstruction.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from dynamix.core.dwt import BANKS
from dynamix.model.contracts import COMPONENTS, FIELD, RECEIVED, STACK
from dynamix.model.device import Output
from dynamix.model.param import Param, ParamKind

_SPARSE = ("route", ("sparsity", "both"))
_CASCADE = ("route", ("cascade", "both"))
_THRESHOLD = ("mask_source", ("threshold", "all"))

PARAMS = (
    Param("route", ParamKind.CHOICE, default="sparsity", choices=("sparsity", "cascade", "both"),
          label="Route"),
    Param("wavelet", ParamKind.CHOICE, default="sym7", choices=BANKS, label="Wavelet"),
    Param("level", ParamKind.INT, default=0, min=0, max=10, soft_min=0, soft_max=8,
          label="Levels (0 = the route's depth)"),
    Param("lam_rel", ParamKind.FLOAT, default=0.2, min=0.0, max=10.0, soft_min=0.0, soft_max=1.0,
          label="Lambda (x sigma sqrt 2 ln N)", active_when=_SPARSE),
    Param("n_iter", ParamKind.INT, default=0, min=0, max=5000, soft_min=0, soft_max=500,
          label="Iterations (0 = the route's default)"),
    Param("tol", ParamKind.FLOAT, default=1e-3, min=0.0, max=1.0, soft_min=0.0, soft_max=0.01,
          label="Stop: relative change", active_when=_SPARSE),
    Param("stop_var_err", ParamKind.FLOAT, default=0.01, min=0.0, max=1.0, soft_min=0.0,
          soft_max=0.1, label="Stop: variance error", active_when=_CASCADE),
    Param("seed", ParamKind.INT, default=0, min=0, max=2 ** 31 - 1, soft_min=0, soft_max=64,
          label="Seed"),
    Param("batch", ParamKind.FLOAT, default=1.0, min=0.05, max=1.0, soft_min=0.05, soft_max=1.0,
          label="Mini-batch", active_when=_SPARSE),
    Param("cascade_synthesis", ParamKind.BOOL, default=False,
          label="Synthesis pass (beyond the paper)", active_when=_CASCADE),
    Param("target_band", ParamKind.TEXT, default="", label="Target band", editable=True),
    Param("mask_source", ParamKind.CHOICE, default="nodata",
          choices=("nodata", "threshold", "region", "send", "all"), label="Mask"),
    Param("mask_band", ParamKind.TEXT, default="", label="Threshold band", editable=True,
          active_when=_THRESHOLD),
    Param("mask_op", ParamKind.CHOICE, default=">", choices=(">", "<"), label="Threshold",
          active_when=_THRESHOLD),
    Param("mask_value", ParamKind.FLOAT, default=0.0, min=-1e30, max=1e30, soft_min=0.0,
          soft_max=1.0, label="Threshold value", active_when=_THRESHOLD),
    Param("mask_min_px", ParamKind.INT, default=16, min=1, max=10 ** 6, soft_min=1,
          soft_max=1024, label="Smallest region (px)", active_when=_THRESHOLD),
    Param("threshold_kind", ParamKind.CHOICE, default="gap", choices=("gap", "veil"),
          label="Threshold regions are", active_when=_THRESHOLD),
    Param("region_kind", ParamKind.CHOICE, default="gap", choices=("gap", "veil"),
          label="Sent mask regions are", active_when=("mask_source", ("send", "all"))),
    Param("veil_model", ParamKind.CHOICE, default="gain", choices=("gain", "offset"),
          label="Veil model"),
    Param("veil_levels", ParamKind.INT, default=3, min=1, max=8, soft_min=1, soft_max=6,
          label="Veil levels"),
    Param("veil_passes", ParamKind.INT, default=2, min=1, max=8, soft_min=1, soft_max=4,
          label="Veil passes"),
    Param("sun_azimuth", ParamKind.ANGLE, default=0.0, wrap=360.0, label="Sun azimuth"),
    Param("sun_elevation", ParamKind.FLOAT, default=45.0, min=0.0, max=90.0, soft_min=0.0,
          soft_max=90.0, label="Sun elevation"),
    Param("prior_weight", ParamKind.FLOAT, default=0.3, min=0.0, max=10.0, soft_min=0.0,
          soft_max=1.0, label="Value prior weight"),
    Param("structure_weight", ParamKind.FLOAT, default=0.1, min=0.0, max=10.0, soft_min=0.0,
          soft_max=1.0, label="Structure weight"),
    Param("structure_levels", ParamKind.INT, default=3, min=1, max=8, soft_min=1, soft_max=6,
          label="Structure levels"),
    Param("abacus", ParamKind.CHOICE, default="self", choices=("self", "file"), label="Abacus",
          active_when=_CASCADE),
    Param("abacus_file", ParamKind.TEXT, default="", label="Abacus file", editable=True,
          active_when=(_CASCADE, ("abacus", ("file",)))),
    Param("abacus_min_pairs", ParamKind.INT, default=1000, min=100, max=10 ** 6, soft_min=100,
          soft_max=10000, label="Abacus: fewest pairs",
          active_when=(_CASCADE, ("abacus", ("self",)))),
    Param("correct", ParamKind.BOOL, default=True, label="Correct known pixels"),
    Param("heldout_blocks", ParamKind.INT, default=8, min=0, max=256, soft_min=0, soft_max=32,
          label="Held-out blocks"),
    Param("heldout_side", ParamKind.INT, default=16, min=4, max=256, soft_min=4, soft_max=64,
          label="Held-out side (px)"),
    Param("show", ParamKind.CHOICE, default="filled", choices=("filled", "residual", "heldout"),
          label="Show", view=True),
    # Written by the shell: the sends (role, source, weights) and the drawn regions.
    Param("_sends", ParamKind.TEXT, default="[]", label="Sends"),
    Param("_mask_polygons", ParamKind.TEXT, default="[]", label="Drawn regions"),
)


def _store(a):
    from dynamix.devices.ssa2d import _store as store

    return store(a)


def _target_band(field, params: dict, device: str):
    """``(values 2-D, band index or None)``: the field, or the stack band ``target_band`` names
    (a band id of ``provenance["bands"]``, else a 1-based index)."""
    vals = np.asarray(field.values, dtype=np.float64)
    if vals.ndim == 2:
        return vals, None
    bands = [str(b) for b in ((getattr(field, "provenance", None) or {}).get("bands") or ())]
    names = bands or [str(i + 1) for i in range(vals.shape[2])]
    tid = str(params.get("target_band", "") or "")
    if not tid:
        raise ValueError(f"{device}: a stack needs target_band -- one of {', '.join(names)}")
    if tid in names:
        k = names.index(tid)
    elif tid.isdigit() and 1 <= int(tid) <= vals.shape[2]:
        k = int(tid) - 1
    else:
        raise ValueError(f"{device}: no band {tid!r} in this stack (target_band is one of "
                         f"{', '.join(names)})")
    return vals[..., k], k


def _margin(params: dict) -> int:
    from dynamix.core.dwt import filter_length

    J = int(params.get("level", 0)) or 3
    return min(filter_length(params.get("wavelet", "sym7")) * 2 ** min(J, 4), 256)


def _run(device, field, values, params: dict, *, progress, cancel, stage_cache) -> dict:
    from dynamix.core.inpaint.engine import run_inpaint
    from dynamix.core.inpaint.sends import resolve_guides

    guides = resolve_guides(params.get("_sends", "[]") or "[]", field)
    params = _with_scene_sun(field, params)
    return run_inpaint(field, values, params, guides, stage_cache=stage_cache, cancel=cancel,
                       progress=progress)


def _with_scene_sun(field, params: dict) -> dict:
    """``params`` with the scene sun the loader stamped on the field when the sun knobs are
    still at their defaults; the stamped azimuth is measured from true north (``sun_north``)."""
    sun = (getattr(field, "provenance", None) or {}).get("sun")
    if (not sun or float(params.get("sun_azimuth", 0.0)) != 0.0
            or float(params.get("sun_elevation", 45.0)) != 45.0):
        return params
    return {**params, "sun_azimuth": float(sun["azimuth"]), "sun_elevation": float(sun["elevation"]),
            "sun_north": "true"}


def _relations_text(res: dict) -> str:
    rel = res.get("_grid_relations") or {}
    if not rel:
        return "no sends"
    parts = []
    for label, r in rel.items():
        kind = r["kind"] if isinstance(r, dict) else str(r)
        if kind == "nested" and isinstance(r, dict):
            kind = f"nested x{r['ratio']}"
        parts.append(f"{label} {kind}")
    return ", ".join(parts)


def inactive_for(params: dict, sends) -> set:
    """The parameters that do not apply given the sends: the gprv writer's second condition
    (``Param.active_when`` carries the first)."""
    roles = {str(s.get("role", "value")) for s in sends}
    off = set()
    if "value" not in roles:
        off.add("prior_weight")
    if "structure" not in roles:
        off |= {"structure_weight", "structure_levels"}
    try:
        polys = json.loads(params.get("_mask_polygons", "[]") or "[]")
    except ValueError:
        polys = []
    veils = (params.get("threshold_kind") == "veil" and params.get("mask_source") in ("threshold", "all")) \
        or (params.get("region_kind") == "veil" and params.get("mask_source") in ("send", "all")) \
        or any(p.get("kind") == "veil" for p in polys) or "veil_seed" in roles
    if not veils:
        off |= {"veil_model", "veil_levels", "veil_passes", "sun_azimuth", "sun_elevation"}
    return off


class Inpaint:
    """Fill the declared regions of the host band; the filled band is the result's raster."""

    name = "inpaint"
    label = "Inpaint"
    families = ("Reconstruction",)
    consumes = ((FIELD,), (STACK,))
    produces = (COMPONENTS,)
    wants_cancel = True
    uses_stage_cache = True
    #: The shell's sends dialog writes ``_sends``; the stamps prelude and the repoint follow it.
    declares_sends = True
    params = PARAMS
    outputs = (Output("residual", "raster", lazy=True, label="residual"),
               Output("heldout", "raster", lazy=True, label="held-out error"))

    def roi_margin(self, params: dict) -> int:
        return _margin(params)

    def compute(self, field, params: dict, *, progress=None, cancel=None, stage_cache=None) -> dict:
        from dynamix.devices.decompose import _field_values, _result

        _field_values(field, self.name)
        values, k = _target_band(field, params, self.name)
        res = _run(self, field, values, params, progress=progress, cancel=cancel,
                   stage_cache=stage_cache)
        extras = {key: v for key, v in res.items()
                  if key not in ("raster_out", "chains", "extrema", "params")}
        out = _result(field, _store(res["raster_out"]), extras, params)
        out["_shape"] = values.shape
        out["_band_index"] = k
        return out

    def compute_roi(self, window_field, core, params: dict, *, info=None, progress=None,
                    cancel=None, stage_cache=None) -> dict:
        """The whole processing window (the sends follow its axes), cut to the region."""
        full = self.compute(window_field, params, progress=progress, cancel=cancel,
                            stage_cache=stage_cache)
        r0, c0, h, w = (int(v) for v in core)
        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
        for key in ("raster_out", "fill_mask", "_estimate", "_heldout_mask"):
            full[key] = full[key][sl]
        full["_shape"] = (h, w)
        return full

    def view(self, result: dict, params: dict) -> dict:
        note = "no region declared" if result.get("_stop") == "no region" else "filled"
        return {**result, "_view_note": note}

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None) -> dict:
        if name not in ("residual", "heldout"):
            raise ValueError(f"{self.name}: no lazy output named {name!r}")
        y = np.asarray(values, dtype=np.float64)
        if y.ndim == 3:
            y = y[..., int(result.get("_band_index") or 0)]
        est = np.asarray(result["_estimate"], dtype=np.float64)
        if name == "residual":
            known = np.isfinite(y) & (np.asarray(result["fill_mask"]) == 0)
            raster = np.where(known, est - y, np.nan)
            diag = {"rms_rel": float(np.sqrt(np.nanmean(raster ** 2)) /
                                     max(np.nanmax(y) - np.nanmin(y), 1e-30))}
        else:
            held = np.asarray(result["_heldout_mask"], dtype=bool)
            raster = np.where(held, np.abs(est - y), np.nan)
            diag = {"rms_rel": float(result["_heldout"]["rms_rel"]),
                    "n_blocks": int(result["_heldout"]["n_blocks"])}
        return {"raster": raster.astype(np.float32), "diag": diag}

    def reading(self, result, params: dict) -> str:
        if not isinstance(result, dict) or "_route" not in result:
            return ""
        if result.get("_stop") == "no region":
            return "no region declared"
        held = result.get("_heldout") or {}
        regions = result.get("_regions") or {}
        counts = ", ".join(f"{regions[k]} {k}" for k in ("gap", "veil") if regions.get(k))
        text = (f"{result['_route']} · {counts or 'no region'} · {_relations_text(result)} · "
                f"{result.get('_iterations', 0)} it ({result.get('_stop', '')}) · "
                f"var err {float(result.get('_variance_error', 0.0)):.3g}")
        if held.get("n_blocks"):
            text += f" · held-out RMS {float(held['rms_rel']):.3g}"
        foreign = [label for label, r in (result.get("_grid_relations") or {}).items()
                   if isinstance(r, dict) and r.get("kind") == "foreign"]
        if foreign:
            text += f" · foreign send {', '.join(foreign)}: the cascade routes are closed"
        if result.get("_warnings"):
            text += " · " + "; ".join(str(w) for w in result["_warnings"])
        return text

    def inactive_params(self, params: dict, sends) -> set:
        return inactive_for(params, sends)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))


class Fill:
    """The engine as a field stage: the host band filled ahead of the analyzer, stamped."""

    name = "fill"
    label = "Fill"
    families = ("Data preparation",)
    consumes = ((FIELD,), (STACK,))
    produces = (RECEIVED,)
    field_stage = True
    wants_cancel = True
    uses_stage_cache = True
    declares_sends = True
    params = tuple(p for p in PARAMS if p.name not in ("show", "heldout_blocks", "heldout_side"))

    def roi_margin(self, params: dict) -> int:
        return _margin(params)

    def pass_all(self, params: dict) -> bool:
        """Decidable from the params alone: drawn regions chosen and none drawn."""
        try:
            polys = json.loads(params.get("_mask_polygons", "[]") or "[]")
        except ValueError:
            polys = []
        return params.get("mask_source") == "region" and not polys

    def compute(self, field, params: dict, *, progress=None, cancel=None, stage_cache=None):
        if isinstance(field, dict):
            raise ValueError("fill runs on the FIELD, before the analysing transform -- drag it "
                             "ahead of the analyzer in the chain")
        if self.pass_all(params):
            return field
        values, k = _target_band(field, params, self.name)
        run_params = {**params, "heldout_blocks": 0, "heldout_side": 16}
        res = _run(self, field, values, run_params, progress=progress, cancel=cancel,
                   stage_cache=stage_cache)
        vals = np.asarray(field.values, dtype=np.float64)
        if k is None:
            new = np.asarray(res["raster_out"], dtype=np.float64)
        else:
            new = vals.copy()
            new[..., k] = res["raster_out"]
        from dynamix.model.device import keyed_params

        tag = hashlib.sha1(json.dumps(keyed_params(self, params), sort_keys=True,
                                      default=str).encode()).hexdigest()[:8]
        stamp = {"filled_fraction": float(res["_filled_fraction"]), "fill_route": res["_route"],
                 "fill_kind": "estimate", "grid_relations": res["_grid_relations"],
                 "seed": res["_seed"], "iterations": res["_iterations"],
                 "returned": res["_returned"], "stop": res["_stop"],
                 "variance_error": float(res["_variance_error"]),
                 "terms_dropped": res["_terms_dropped"], "veil": res["_veil"],
                 "warnings": list(res["_warnings"])}
        prov = {**(getattr(field, "provenance", None) or {}), "fill": stamp}
        return dataclasses.replace(field, values=new, name=f"{field.name}+fill{tag}",
                                   provenance=prov)

    def reading(self, result, params: dict) -> str:
        from dynamix.model.contracts import fill_of

        stamp = fill_of(result)
        if stamp is None:
            return ""
        return (f"filled {100.0 * float(stamp.get('filled_fraction', 0.0)):.0f} % · "
                f"{stamp.get('fill_route', '')} · analysis inside the region reads a reconstruction")

    def inactive_params(self, params: dict, sends) -> set:
        return inactive_for(params, sends)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))
