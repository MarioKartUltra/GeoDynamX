# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The ROI runner: run ANY tool on a saved region.

This is ``halo.py``'s ``wtmm2d_roi`` behaviour,
generalised to every device:

1. the tool declares its MARGIN from its own settings (``device.roi_margin(params)``);
2. ROI + margin is read ONCE -- real data wherever the file has it, the deficit past the file's
   edge reflect-padded (the ``halo._read_halo`` rule);
3. the leading field-stage transforms (``noise``) run over the WHOLE window, then the analyzer;
4. the result is cropped to the ROI and its lines re-labelled inside it -- or the tool does that
   itself partway through (``device.compute_roi``: wtmm chains after the crop; band_recon
   reconstructs from ROI values only).

A function of a REGION, not of a layer: no Qt, no cache, no project -- ``engine.resolve`` calls it
for a layer carrying an ROI, and a future grid of n x n cells calls it once per cell.
Headless: numpy at module level; rasterio and scipy lazily.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["crop_result_to_roi", "read_processing_window", "roi_margin_of",
           "run_on_region", "split_lines"]

_EDGES = ("N", "S", "E", "W")


def _window_offset(field) -> tuple[int, int]:
    window = (getattr(field, "provenance", None) or {}).get("window") or {}
    return int(window.get("row_off", 0)), int(window.get("col_off", 0))


def _reflect_pad_nd(arr: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """``halo._reflect_pad`` for a 2-D array or per band of a (ny, nx, nc) stack."""
    from dynamix.roi.halo import _reflect_pad

    if arr.ndim == 2:
        return _reflect_pad(arr, top, bottom, left, right)
    return np.stack([_reflect_pad(arr[..., k], top, bottom, left, right)
                     for k in range(arr.shape[2])], axis=-1)


def _linear_axis(axis, file_idx, fallback_step: float):
    """``coord = a + b * file_index`` fitted through the field's own axis samples."""
    axis = np.asarray(axis, dtype=np.float64)
    idx = np.asarray(file_idx, dtype=np.float64)
    if axis.size >= 2 and idx[-1] != idx[0]:
        b = (axis[-1] - axis[0]) / (idx[-1] - idx[0])
    else:
        b = float(fallback_step)
    return float(axis[0] - b * idx[0]), float(b)


def _read_hdf4(source, dsname, r0, c0, r1, c1):
    """An HDF4 SDS window via pyhdf (``_FillValue`` -> NaN), with an identity transform: HDF4
    fields carry pixel-index axes (the ``ingest._load_hdf4`` convention)."""
    from pyhdf.SD import SD, SDC
    from rasterio import Affine

    sd = SD(str(source), SDC.READ)
    try:
        ds = sd.select(dsname)
        values = np.asarray(ds.get(start=(r0, c0), count=(r1 - r0, c1 - c0)), dtype=np.float64)
        fill = ds.attributes().get("_FillValue")
    finally:
        sd.end()
    if fill is not None:
        values = np.where(values == fill, np.nan, values)
    return values, Affine.translation(c0, r0)


def _read_file(source, r0, c0, r1, c1):
    """Band 1 of ``source``, rows [r0, r1) x cols [c0, c1); nodata and the BOEM float32
    sentinel (|v| >= 3e38) -> NaN, the whole-field open's convention. Returns (values,
    window transform)."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(source) as src:
        win = Window(c0, r0, c1 - c0, r1 - r0)
        values = np.asarray(src.read(1, window=win), dtype=np.float64)
        transform = src.window_transform(win)
        nodata = src.nodata
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    values = np.where(np.abs(values) >= 3e38, np.nan, values)
    return values, transform


def read_processing_window(field, rect, margin: int):
    """Read ROI ``rect`` = ``(row, col, h, w)`` (FILE pixels) plus ``margin`` on every side.

    Where the pixels come from -- chosen per call, never by format:

    - the IN-MEMORY field when it already holds every native pixel the window needs (a whole
      file loaded at full resolution, a native window covering the margin) -- format-agnostic,
      and no file re-read (a netCDF container path is never re-opened as band 1);
    - the FILE (``provenance["source"]``) for a display picture, or when the margin reaches
      past what is loaded;
    - the in-memory field alone when there is no file at all (npz / derived), with ITS edges
      standing in for the file's.

    Returns ``(window_field, info)``: a ``RasterField`` of shape ``(h + 2m, w + 2m)`` carrying its
    real georeference (extrapolated linearly into any reflected border) and
    ``provenance["window"]`` = its file origin ``(row - m, col - m)``; ``info`` = ``{"offset": m,
    "real": bool mask (False where reflected), "reflected_edges": subset of N/S/E/W,
    "real_frac": fraction of the window that is real AND finite}``.
    """
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.roi.picture import display_stride

    r, c, h, w = (int(v) for v in rect)
    m = int(margin)
    prov = getattr(field, "provenance", None) or {}
    values = np.asarray(field.values, dtype=np.float64)
    ny, nx = values.shape[:2]
    stride = display_stride(field)
    wr, wc = _window_offset(field)
    full = prov.get("full_dims")
    H, W = (int(full[0]), int(full[1])) if full is not None else (wr + ny, wc + nx)
    source = prov.get("source")

    want_r0, want_c0, win_h, win_w = r - m, c - m, h + 2 * m, w + 2 * m
    covered = (stride == 1 and wr <= max(0, want_r0) and wc <= max(0, want_c0)
               and min(H, want_r0 + win_h) <= wr + ny and min(W, want_c0 + win_w) <= wc + nx)
    if covered or not source:
        # The field is (or stands in for) the file; memory-only fields reflect at their edges.
        pr0, pc0, pr1, pc1 = (0, 0, H, W) if covered else (wr, wc, wr + ny, wc + nx)
    else:
        pr0, pc0, pr1, pc1 = 0, 0, H, W
    read_r0, read_c0 = max(pr0, want_r0), max(pc0, want_c0)
    read_r1, read_c1 = min(pr1, want_r0 + win_h), min(pc1, want_c0 + win_w)
    if read_r1 <= read_r0 or read_c1 <= read_c0:
        raise ValueError(f"ROI {rect} lies outside the {H}x{W} px dataset")
    top, left = read_r0 - want_r0, read_c0 - want_c0
    bottom, right = (want_r0 + win_h) - read_r1, (want_c0 + win_w) - read_c1

    if covered or not source:
        core = values[read_r0 - wr:read_r1 - wr, read_c0 - wc:read_c1 - wc].copy()
        fx = getattr(field.frame, "dx", 1.0) or 1.0
        fy = getattr(field.frame, "dy", 1.0) or 1.0
        ax, bx = _linear_axis(field.x_axis, wc + np.arange(nx), fx)
        ay, by = _linear_axis(field.y_axis, wr + np.arange(ny), fy)
    else:
        if prov.get("reader") == "hdf4":
            core, transform = _read_hdf4(source, prov["subdataset"], read_r0, read_c0,
                                         read_r1, read_c1)
        else:
            core, transform = _read_file(source, read_r0, read_c0, read_r1, read_c1)
        # pixel-centre georeference of FILE index i: transform at (i - read_origin + 0.5)
        bx, by = float(transform.a), float(transform.e)
        ax = float(transform.c) + bx * (0.5 - read_c0)
        ay = float(transform.f) + by * (0.5 - read_r0)
    window = _reflect_pad_nd(core, top, bottom, left, right)
    real = np.zeros((win_h, win_w), dtype=bool)
    real[top:top + core.shape[0], left:left + core.shape[1]] = True

    cols = want_c0 + np.arange(win_w, dtype=np.float64)
    rows = want_r0 + np.arange(win_h, dtype=np.float64)
    x_axis, y_axis = ax + bx * cols, ay + by * rows
    frame = field.frame
    if isinstance(frame, LocalFrame):
        frame = dataclasses.replace(frame, x0=float(x_axis[0]), y0=float(y_axis[0]),
                                    dx=abs(bx), dy=abs(by))
    finite = np.isfinite(window if window.ndim == 2 else window[..., 0])
    reflected = {"N": top > 0, "S": bottom > 0, "W": left > 0, "E": right > 0}
    out = RasterField(
        name=f"{field.name}@roi{r},{c}", values=window, frame=frame,
        x_axis=x_axis, y_axis=y_axis, units=getattr(field, "units", ""),
        provenance={"source": source, "full_dims": (H, W), "crs": prov.get("crs"),
                    "window": {"row_off": want_r0, "col_off": want_c0}})
    info = {"offset": m, "real": real,
            "reflected_edges": tuple(e for e in _EDGES if reflected[e]),
            "real_frac": float((real & finite).mean())}
    return out, info


#: Draw orderings and index products computed over the WINDOW: stale after a crop. The chain
#: product is rebuilt by the runner from the cropped extrema (``attach_chain_product``).
_WINDOW_ONLY_KEYS = ("_xs_runs", "_xs_closed", "_ext_base_runs", "_hline_runs",
                     "chain_product", "_selection")


def _crop_points(level: dict, offset: int, h: int, w: int) -> dict:
    """Keep the points whose INTEGER support lies inside the ROI, shifted to ROI-local; every
    other per-point array of the same length is subset in step (``halo._crop_to_roi``
    generalised to any channel a tool emits)."""
    x = np.asarray(level["x"])
    y = np.asarray(level["y"])
    keep = (x >= offset) & (x < offset + w) & (y >= offset) & (y < offset + h)
    out = {}
    for key, val in level.items():
        arr = np.asarray(val) if isinstance(val, (np.ndarray, list)) else None
        if arr is not None and arr.ndim >= 1 and arr.shape[0] == x.size:
            arr = arr[keep]
            if key in ("x", "y", "x_sub", "y_sub"):
                arr = arr - offset
            out[key] = arr
        else:
            out[key] = val
    return out


def _crop_array(arr, offset: int, h: int, w: int, win_shape):
    if not isinstance(arr, np.ndarray) or arr.ndim < 2:
        return arr
    if tuple(arr.shape[:2]) == tuple(win_shape):
        return arr[offset:offset + h, offset:offset + w].copy()
    if arr.ndim == 3 and tuple(arr.shape[1:]) == tuple(win_shape):
        return arr[:, offset:offset + h, offset:offset + w].copy()
    return arr


def crop_result_to_roi(result: dict, offset: int, h: int, w: int, win_shape) -> dict:
    """The generic crop (the design step 5): a result computed over the processing window, cut to
    the ROI ``[offset, offset + h) x [offset, offset + w)`` of it.

    - every ``extrema`` level keeps its in-ROI points, shifted to ROI-local;
    - every array whose leading two dims (or trailing two, for a per-scale stack) equal the
      window's is sliced -- h-maps, masks, reconstructions, filtered snapshots;
    - cross-scale ``chains`` keep their in-ROI members (empty chains dropped);
    - window-computed draw orderings / index products are dropped (``_WINDOW_ONLY_KEYS``).
    Line ids are NOT touched here -- :func:`split_lines` re-labels inside the ROI.
    """
    out = {}
    for key, val in result.items():
        if key in _WINDOW_ONLY_KEYS:
            continue
        if key == "extrema" and isinstance(val, list):
            out[key] = [_crop_points(lvl, offset, h, w) if isinstance(lvl, dict) else lvl
                        for lvl in val]
        elif key == "chains" and isinstance(val, list):
            kept = []
            for ch in val:
                if isinstance(ch, dict) and "x" in ch and "y" in ch:
                    ch = _crop_points(ch, offset, h, w)
                    if np.asarray(ch["x"]).size:
                        kept.append(ch)
                else:
                    kept.append(ch)
            out[key] = kept
        elif isinstance(val, list) and val and all(isinstance(a, np.ndarray) for a in val):
            out[key] = [_crop_array(a, offset, h, w, win_shape) for a in val]
        else:
            out[key] = _crop_array(val, offset, h, w, win_shape)
    return out


def split_lines(level: dict) -> np.ndarray:
    """ROI-local line ids for one extrema level: within each ORIGINAL id >= 0, the 8-connected
    components of its points become separate lines (a line that leaves the ROI and comes back
    is two); split-only -- points of different original ids are never joined; a line reduced
    to one point, and every original ``-1``, is ``-1``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    x = np.asarray(level["x"], dtype=np.int64)
    y = np.asarray(level["y"], dtype=np.int64)
    ids = np.asarray(level["line_id"], dtype=np.int64)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    member = ids >= 0
    idx = np.nonzero(member)[0]
    rows, cols = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if idx.size:
        # pixel -> point index on a dense grid (1-px guard ring); a pixel shared by two
        # points keeps one of them -- which can only ever MISS a join (split-only stays safe)
        y0, x0 = int(y[idx].min()) - 1, int(x[idx].min()) - 1
        grid = np.full((int(y[idx].max()) - y0 + 2, int(x[idx].max()) - x0 + 2), -1,
                       dtype=np.int64)
        grid[y[idx] - y0, x[idx] - x0] = idx
        r_all, c_all = [], []
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            j = grid[y[idx] - y0 + dy, x[idx] - x0 + dx]
            ok = (j >= 0) & (ids[np.maximum(j, 0)] == ids[idx])
            r_all.append(idx[ok])
            c_all.append(j[ok])
        rows, cols = np.concatenate(r_all), np.concatenate(c_all)
    graph = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    _n, comp = connected_components(graph, directed=False)
    counts = np.bincount(comp, minlength=_n)
    out = np.where(member & (counts[comp] > 1), comp, -1).astype(np.int64)
    # dense 0-based ids in first-seen order
    uniq, first = np.unique(out[out >= 0], return_index=True)
    order = uniq[np.argsort(first)]
    remap = {int(c): k for k, c in enumerate(order)}
    return np.array([remap[int(v)] if v >= 0 else -1 for v in out], dtype=np.int64)


def roi_margin_of(device, params) -> tuple[int, bool]:
    """``(margin_px, declared)``: the device's own ``roi_margin(params)``, or ``(0, False)`` for a
    plugin that declares none -- it still runs, on the ROI alone, and the result says so."""
    fn = getattr(device, "roi_margin", None)
    if fn is None:
        return 0, False
    return max(0, int(fn(params))), True


def run_on_region(steps, field, rect, *, progress=None, cancel=None) -> dict:
    """Run ``steps`` -- the leading field-stage transforms and then the analyzer, as
    ``[(device, validated_params), ...]`` -- on ROI ``rect`` = ``(row, col, h, w)`` (FILE
    pixels) of ``field``'s dataset.

    Margin = the sum of every step's declared margin (supports add under composition). The
    processing window is read once (:func:`read_processing_window`); the field stage runs over
    ALL of it; the analyzer then either runs its own ``compute_roi(window_field, core,
    params, info=..., progress=...)`` -- ``core = (m, m, h, w)`` inside the window, the tool
    crops where its method needs to (wtmm before chaining, band_recon before reconstructing)
    -- or its ordinary ``compute`` followed by :func:`crop_result_to_roi` +
    :func:`split_lines` (+ a rebuilt chain product when the tool stamps one).

    The result is ROI-shaped and stamped: ``_shape = (h, w)``; ``_roi`` = the keys
    ``roi_offset``/``display_offset`` read (``source``, ``roi``, ``boundary``) plus ``margin``,
    ``reflected_edges``, ``real_frac``, ``margin_declared``; ``_roi_axes`` = the ROI's own
    ``(x_axis, y_axis)``; ``_frame`` re-anchored at the ROI origin.
    """
    steps = list(steps)
    if not steps:
        raise ValueError("run_on_region needs at least the analyzer")
    r, c, h, w = (int(v) for v in rect)
    margins = [roi_margin_of(d, p) for d, p in steps]
    m = sum(mm for mm, _ok in margins)
    window, info = read_processing_window(field, (r, c, h, w), m)
    # The ROI's own NATIVE values (raw, before any field stage): what a view drapes when it
    # shows this result on the region's own grid (the Vector scene).
    roi_values = np.asarray(window.values)[m:m + h, m:m + w].copy()
    for device, params in steps[:-1]:                        # the field stage (noise, ...)
        window = device.compute(window, params, progress=progress)
    analyzer, aparams = steps[-1]
    compute_roi = getattr(analyzer, "compute_roi", None)
    if compute_roi is not None:
        res = dict(compute_roi(window, (m, m, h, w), aparams, info=info, progress=progress))
    else:
        if getattr(analyzer, "wants_cancel", False):
            full = analyzer.compute(window, aparams, progress=progress, cancel=cancel)
        else:
            full = analyzer.compute(window, aparams, progress=progress)
        had_product = "chain_product" in full
        res = crop_result_to_roi(full, m, h, w, np.asarray(window.values).shape[:2])
        for level in res.get("extrema") or []:
            if isinstance(level, dict) and "line_id" in level:
                level["line_id"] = split_lines(level)
        res["_shape"] = (h, w)
        if had_product:
            from dynamix.core.chain_product import attach_chain_product

            attach_chain_product(res)
    res["_shape"] = (h, w)
    x_axis = np.asarray(window.x_axis)[m:m + w].copy()
    y_axis = np.asarray(window.y_axis)[m:m + h].copy()
    res["_roi_axes"] = (x_axis, y_axis)
    res["_roi_values"] = roi_values
    frame = res.get("_frame", getattr(window, "frame", None))
    from dynamix.core.frames import LocalFrame

    if isinstance(frame, LocalFrame):
        frame = dataclasses.replace(frame, x0=float(x_axis[0]), y0=float(y_axis[0]))
    res["_frame"] = frame
    res["_roi"] = {"source": (getattr(field, "provenance", None) or {}).get("source"),
                   "roi": (r, c, h, w), "boundary": "auto", "margin": m,
                   "reflected_edges": info["reflected_edges"], "real_frac": info["real_frac"],
                   "margin_declared": all(ok for _m, ok in margins)}
    return res
