# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Derivative datasets: what a result holds -- rasters, extrema, maxima lines -- written once to a
file that opens as an ordinary dataset.

A CHILD layer re-runs its chain on its parent's data whenever the parent changes. A DERIVATIVE
is forked once and from then on is raw data: nothing links it back (its provenance records where
it came from, as history only). The file is a RasterField npz -- exactly the keys
``RasterField.load_npz`` reads, so every opener takes it as a dataset -- with the vectors riding
under ``vec_*`` keys:

- the raster bands picked at the fork, stacked ``(ny, nx, k)`` (``(ny, nx)`` for one band); a
  vector-only derivative carries an all-NaN raster over the same grid;
- ``provenance["bands"]`` names the bands;
- the extrema (per-scale ``x/y/mod/arg/line_id``) and the maxima lines, CSR-packed by the WTMM
  backend's own stage-cache codecs, plus the scale array (``vec_scales``).

The grid is the result's own, never resampled: an ROI result carries its window's axes and frame
(``_roi_axes`` / ``_frame``) and window-local coordinates; any other result lives on the layer's
field grid.
"""
from __future__ import annotations

import json

import numpy as np

__all__ = ["raster_choices", "read_vectors", "result_grid", "vector_counts",
           "write_derivative"]

#: Single rasters a result may carry, in the order a fork offers them.
_SINGLES = (("ssa_recon", "recon"), ("ssa_residual", "residual"),
            ("tucker_recon", "recon"), ("tucker_residual", "residual"),
            ("filtered", "filtered"), ("edge_channel", "edge channel"),
            ("h_map", "h(x)"), ("r2_map", "r²"), ("band_mask", "band mask"))


def raster_choices(result: dict, shown_label: str = "as shown") -> list:
    """Every 2-D raster on the result's grid that a derivative can take, as ``(label, array)``:
    the raster ON SCREEN first (``raster_out``, else ``h_map``), then the decomposition stacks
    (2D-SSA / tucker components in the numbering the Orientation knob picks, PCs), tucker's
    per-band recon / residual on a multiband field, then the single rasters."""
    shape = tuple(result.get("_shape") or ())
    out = []

    def add(label, arr):
        if arr is None:
            return
        a = np.asarray(arr)
        if a.ndim == 2 and (not shape or a.shape == shape):
            out.append((label, a))

    shown = result.get("raster_out")
    add(shown_label, shown if shown is not None else result.get("h_map"))
    combined = (result.get("params") or {}).get("pairs") == "combined"
    tucker = "tucker_combined_components" if combined else "tucker_components"
    for key, prefix in (("ssa_components", "C"), (tucker, "C"), ("pca_images", "PC")):
        stack = result.get(key)
        for k in range(0 if stack is None else len(stack)):
            add(f"{prefix}{k + 1}", stack[k])
    for key, name in (("tucker_recon_bands", "recon band"),
                      ("tucker_residual_bands", "residual band")):
        stack = result.get(key)
        for b in range(0 if stack is None else np.shape(stack)[-1]):
            add(f"{name} {b + 1}", np.asarray(stack)[..., b])
    for key, label in _SINGLES:
        add(label, result.get(key))
    return out


def result_grid(result: dict, field):
    """``(frame, x_axis, y_axis)`` of the grid the result's pixels and coordinates live on."""
    if result.get("_roi_axes") is not None:
        x, y = result["_roi_axes"]
        return (result.get("_frame") or field.frame), np.asarray(x), np.asarray(y)
    return field.frame, np.asarray(field.x_axis), np.asarray(field.y_axis)


def vector_counts(result: dict) -> tuple:
    """``(extrema points, maxima lines)`` the result would give a vector derivative."""
    levels = result.get("extrema") or []
    points = sum(len(e["x"]) for e in levels if isinstance(e, dict) and "x" in e)
    return points, len(result.get("chains") or [])


def _levels(extrema) -> list:
    """The extrema levels with every key the codec needs (``arg`` 0, ``line_id`` -1 when a tool
    did not stamp them)."""
    out = []
    for e in extrema:
        n = len(e["x"])
        level = {"x": e["x"], "y": e["y"], "mod": e.get("mod", np.zeros(n)),
                 "arg": e.get("arg", np.zeros(n)), "line_id": e.get("line_id", np.full(n, -1))}
        if "x_sub" in e and "y_sub" in e:
            level["x_sub"], level["y_sub"] = e["x_sub"], e["y_sub"]
        out.append(level)
    return out


def write_derivative(path, *, bands, frame, x_axis, y_axis, name: str, units: str = "",
                     provenance: "dict | None" = None, vectors: "dict | None" = None):
    """Write ``bands`` (``[(label, 2-D array)]``) and/or ``vectors`` (a result's ``extrema`` /
    ``chains`` / ``scales``) on the grid ``(frame, x_axis, y_axis)`` as one derivative npz.
    Returns ``path``. Refuses an empty fork and a band off the grid."""
    from dynamix.core.frames import frame_to_meta
    from dynamix.core.rasterfield import RASTERFIELD_NPZ_SCHEMA
    from dynamix.core.wtmm_backend import _chains_to_arrays, _extrema_to_arrays

    x = np.asarray(x_axis, dtype=np.float64)
    y = np.asarray(y_axis, dtype=np.float64)
    if not bands and vectors is None:
        raise ValueError("nothing to fork: pick a raster band or the vectors")
    arrays = [np.asarray(a, dtype=np.float64) for _label, a in bands]
    for (label, _a), a in zip(bands, arrays):
        if a.shape != (y.size, x.size):
            raise ValueError(f"band {label!r} is {a.shape}, not on the {y.size} x {x.size} grid")
    if not arrays:
        values = np.full((y.size, x.size), np.nan)
    else:
        values = arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=-1)
    prov = dict(provenance or {})
    prov["bands"] = [label for label, _a in bands]
    payload = {}
    if vectors is not None:
        levels = _levels(vectors.get("extrema") or [])
        scales = vectors.get("scales")
        if scales is None or len(scales) == 0:
            scales = np.arange(1, max(len(levels), 1) + 1, dtype=np.float64)
            prov["vector_scales"] = "level index (none given)"
        payload.update({f"vec_e_{k}": v for k, v in _extrema_to_arrays(levels).items()})
        payload.update({f"vec_c_{k}": v
                        for k, v in _chains_to_arrays(list(vectors.get("chains") or [])).items()})
        payload["vec_scales"] = np.asarray(scales, dtype=np.float64)
    np.savez_compressed(
        path, schema=RASTERFIELD_NPZ_SCHEMA, name=name, values=values, x_axis=x, y_axis=y,
        units=units, frame_meta=json.dumps(frame_to_meta(frame)),
        provenance=json.dumps(prov, default=str), **payload)
    return path


def read_vectors(path) -> "dict | None":
    """The extrema, maxima lines and scales a derivative file carries, or None for a
    raster-only derivative."""
    from dynamix.core.wtmm_backend import _arrays_to_chains, _arrays_to_extrema

    with np.load(path, allow_pickle=False) as d:
        if "vec_scales" not in d.files:
            return None
        ext = {k[len("vec_e_"):]: d[k] for k in d.files if k.startswith("vec_e_")}
        chn = {k[len("vec_c_"):]: d[k] for k in d.files if k.startswith("vec_c_")}
        scales = np.asarray(d["vec_scales"], dtype=np.float64)
    return {"extrema": _arrays_to_extrema(ext), "chains": _arrays_to_chains(chn, scales),
            "scales": scales}
