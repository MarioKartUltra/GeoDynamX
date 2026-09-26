# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Band buses: raw bands routed by REFERENCE into one ``(ny, nx, n)`` stack for multi-input
tools (pca, tucker's band mode, band math).

A **file send** names one raw plane in a file -- ``{"path", "subdataset", "band", "label"}``:
``subdataset`` is a container grid id (``ingest.probe``'s) or ``None``; ``band`` is the
0-based band of a multiband file (GeoTIFF / npz stack) or ``None`` for a 2-D grid. Sends are
never copies: a bus re-reads its planes from their files, so the bus's cache key is its send
list, and a processed plane enters a bus only as a forked derivative (the bounce).

A **layer send** -- ``{"layer": id, "stamp", "label"}`` -- routes what another LAYER shows (a
reconstruction, a noised band) LIVE: ``stamp`` fingerprints that layer's recipe, so it sits in
the bus's own parameters and a change upstream changes the bus's cache key. The shell computes
referenced layers first and files each shown raster here under its stamp
(:func:`register_plane`); a tool never reaches into the project.

**Grid law.** Every send must sit on the target's grid -- same native shape, same
georeference anchor and pixel step, same CRS. Grids of different resolution are refused;
combining them is a deliberate resampling experiment, not a routing side effect.

**Alignment is geometric.** :func:`materialize` locates the target's window inside each
send's file from the target's own pixel-centre axes, so the same code serves a whole field
and an ROI processing window -- including its reflected margin, reproduced exactly as the ROI
runner pads the dataset itself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["grid_mismatch", "materialize", "native_grid", "parse_sends", "register_plane",
           "send_grid", "sends_for_field", "shown_plane"]

#: Shown rasters of referenced layers, by stamp (a recipe fingerprint -> immutable content).
#: Bounded: the engine cache keeps the bus results themselves; this only feeds recomputes.
_PLANES: "dict" = {}
_PLANES_MAX = 32


def register_plane(stamp: str, plane) -> None:
    """File a referenced layer's shown raster (a 2-D field) under its recipe stamp."""
    _PLANES.pop(stamp, None)
    _PLANES[stamp] = plane
    while len(_PLANES) > _PLANES_MAX:
        _PLANES.pop(next(iter(_PLANES)))


def shown_plane(result, field):
    """The raster a layer SHOWS, as a 2-D field on its own grid: a field-producing chain's
    field (noise, a band row); a result's shown raster (``raster_out``, else ``h_map``) on
    the result's grid; the dataset itself for a chain with no transforms. Refuses a stack
    (send its band rows) and a result that shows no raster."""
    import dataclasses

    from dynamix.core.derivative import raster_choices, result_grid

    if not isinstance(result, dict) and hasattr(result, "values"):     # a dict has .values()
        plane = result
    elif not result:
        plane = field
    else:
        rasters = raster_choices(result)
        if not rasters:
            raise ValueError("it shows no raster to send (vectors only)")
        frame, x, y = result_grid(result, field)
        plane = dataclasses.replace(field, values=np.asarray(rasters[0][1], dtype=np.float64),
                                    frame=frame, x_axis=np.asarray(x), y_axis=np.asarray(y))
    if np.asarray(plane.values).ndim != 2:
        raise ValueError("it shows a multi-band stack — send its band rows instead")
    return plane


def parse_sends(text) -> list:
    """The send list from the bus's stored parameter (JSON text); ``[]`` when empty."""
    if not text:
        return []
    sends = json.loads(text) if isinstance(text, str) else list(text)
    if not isinstance(sends, list):
        raise ValueError("a bus's sends must be a list")
    return sends


def native_grid(field) -> tuple:
    """``(ny, nx, x0, dx, y0, dy, crs)`` of the NATIVE pixel grid ``field`` stands for -- the
    file grid of a display picture (never its decimated samples), the field's own grid
    otherwise."""
    from dynamix.roi.picture import file_pixel_grid

    crs = str((getattr(field, "provenance", {}) or {}).get("crs"))
    grid = file_pixel_grid(field)
    if grid is not None:
        x0, dx, y0, dy, nx, ny = grid
        return (int(ny), int(nx), float(x0), float(dx), float(y0), float(dy), crs)
    values = np.asarray(field.values)
    x = np.asarray(field.x_axis, dtype=np.float64)
    y = np.asarray(field.y_axis, dtype=np.float64)
    frame = getattr(field, "frame", None)
    dx = float(x[1] - x[0]) if x.size > 1 else float(getattr(frame, "dx", 1.0) or 1.0)
    dy = float(y[1] - y[0]) if y.size > 1 else float(getattr(frame, "dy", 1.0) or 1.0)
    return (int(values.shape[0]), int(values.shape[1]), float(x[0]), dx, float(y[0]), dy, crs)


def grid_mismatch(a: tuple, b: tuple) -> "str | None":
    """Why grid ``b`` cannot share a bus with grid ``a`` (both :func:`native_grid` tuples), or
    ``None`` when they are the same grid."""
    (ny, nx, x0, dx, y0, dy, crs), (my, mx, u0, du, v0, dv, crs_b) = a, b
    if (ny, nx) != (my, mx):
        return f"a {my} x {mx} grid, not {ny} x {nx}"
    if crs != crs_b:
        return f"CRS {crs_b}, not {crs}"
    tol_x, tol_y = 1e-6 * max(abs(dx), 1e-12), 1e-6 * max(abs(dy), 1e-12)
    if abs(du - dx) > tol_x or abs(dv - dy) > tol_y:
        return f"pixel step ({du:g}, {dv:g}), not ({dx:g}, {dy:g})"
    if abs(u0 - x0) > 1e-3 * abs(dx) or abs(v0 - y0) > 1e-3 * abs(dy):
        return "a different georeference anchor (the same size, somewhere else)"
    return None


def sends_for_field(field, path: str, label: str) -> list:
    """One send per band of ``field`` (a loaded dataset), reading back from ``path``: an
    imported container stack by each band's grid id, a multiband file or npz by band index,
    a 2-D grid as itself. Labels read ``"<label> · <band>"``."""
    from dynamix.core.ingest import GRID_SUFFIXES

    prov = getattr(field, "provenance", {}) or {}
    values = np.asarray(field.values)
    sub = prov.get("subdataset")
    if values.ndim == 2:
        name = str(sub).split("/", 1)[0] if sub else "band 1"
        return [{"path": str(path), "subdataset": sub, "band": None,
                 "label": f"{label} · {name}"}]
    nc = int(values.shape[-1])
    names = list(prov.get("bands") or [])
    by_grid_id = (len(names) == nc and str(path).lower().endswith(GRID_SUFFIXES)
                  and not str(path).lower().endswith(".npz"))
    out = []
    for k in range(nc):
        name = str(names[k]) if k < len(names) else f"band {k + 1}"
        if by_grid_id:
            send = {"path": str(path), "subdataset": names[k], "band": None}
        else:
            send = {"path": str(path), "subdataset": sub, "band": k}
        send["label"] = f"{label} · {name.split('/', 1)[0]}"
        out.append(send)
    return out


def _layer_plane(send: dict):
    plane = _PLANES.get(str(send.get("stamp")))
    if plane is None:
        raise ValueError(f"{send.get('label') or 'a layer send'}: its layer's result is not "
                         f"computed yet — select that layer (or run it), then come back")
    return plane


def _is_hdf4_id(path: str, sub) -> bool:
    return (sub is not None and str(path).lower().endswith(".hdf")
            and not str(sub).startswith(("NETCDF:", "HDF5:", "HDF4")))


def send_grid(send: dict) -> tuple:
    """``(H, W, x0, dx, y0, dy)`` of the send's native plane, read from its file header
    (the npz fallback loads the file -- derivative npz files are in-memory sized)."""
    if "layer" in send:
        return native_grid(_layer_plane(send))[:6]
    path, sub = str(send["path"]), send.get("subdataset")
    if path.lower().endswith(".npz"):
        from dynamix.core.rasterfield import RasterField

        g = native_grid(RasterField.from_file(path))
        return g[:6]
    if _is_hdf4_id(path, sub):
        from pyhdf.SD import SD, SDC

        from dynamix.core.ingest import hdf4_select

        sd = SD(path, SDC.READ)
        try:
            dims = hdf4_select(path, sd, sub).info()[2]
        finally:
            sd.end()
        return (int(dims[0]), int(dims[1]), 0.5, 1.0, 0.5, 1.0)   # ingest's pixel-index axes
    import rasterio

    with rasterio.open(sub if sub is not None else path) as src:
        t = src.transform
        return (int(src.height), int(src.width), float(t.c + 0.5 * t.a), float(t.a),
                float(t.f + 0.5 * t.e), float(t.e))


def _read_window(send: dict, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    """Rows ``[r0, r1)`` x cols ``[c0, c1)`` of the send's plane, float64, with the same
    nodata -> NaN conventions as the whole-plane loaders."""
    if "layer" in send:
        return np.asarray(_layer_plane(send).values, dtype=np.float64)[r0:r1, c0:c1].copy()
    path, sub, band = str(send["path"]), send.get("subdataset"), send.get("band")
    if path.lower().endswith(".npz"):
        from dynamix.core.rasterfield import RasterField

        v = np.asarray(RasterField.from_file(path).values, dtype=np.float64)
        v = v if band is None or v.ndim == 2 else v[..., int(band)]
        return v[r0:r1, c0:c1].copy()
    if _is_hdf4_id(path, sub):
        from dynamix.roi.runner import _read_hdf4

        values, _t = _read_hdf4(path, sub, r0, c0, r1, c1)
        return values
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(sub if sub is not None else path) as src:
        k = 1 if band is None else int(band) + 1
        values = np.asarray(src.read(k, window=Window(c0, r0, c1 - c0, r1 - r0)),
                            dtype=np.float64)
        nodata = src.nodata
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    return np.where(np.abs(values) >= 3e38, np.nan, values)


def materialize(target, sends: list):
    """The bus stack on ``target``'s grid: ``(ny, nx, n)`` in send order (``(ny, nx)`` for one
    send), each plane read over exactly the window ``target``'s axes cover and reflect-padded
    where that window overhangs the file (the ROI runner's margin rule). Frame, axes and CRS
    are ``target``'s; the NAME encodes the send list (``wtmm_backend``'s stage cache keys on
    field names, so two buses must never share one)."""
    import dataclasses

    from dynamix.roi.runner import _reflect_pad_nd

    if not sends:
        raise ValueError("this bus has no sends — right-click it, Edit bus sends…")
    values = np.asarray(target.values)
    ny, nx = int(values.shape[0]), int(values.shape[1])
    x = np.asarray(target.x_axis, dtype=np.float64)
    y = np.asarray(target.y_axis, dtype=np.float64)
    tdx = float(x[1] - x[0]) if x.size > 1 else float(getattr(target.frame, "dx", 1.0) or 1.0)
    tdy = float(y[1] - y[0]) if y.size > 1 else float(getattr(target.frame, "dy", 1.0) or 1.0)
    planes = []
    for send in sends:
        H, W, sx0, sdx, sy0, sdy = send_grid(send)
        if (abs(sdx - tdx) > 1e-6 * max(abs(tdx), 1e-12)
                or abs(sdy - tdy) > 1e-6 * max(abs(tdy), 1e-12)):
            raise ValueError(f"{send.get('label') or send.get('path')}: pixel step "
                             f"({sdx:g}, {sdy:g}) is not the bus grid's ({tdx:g}, {tdy:g})")
        c0 = int(round((x[0] - sx0) / sdx))
        r0 = int(round((y[0] - sy0) / sdy))
        rr0, cc0 = max(0, r0), max(0, c0)
        rr1, cc1 = min(H, r0 + ny), min(W, c0 + nx)
        if rr1 <= rr0 or cc1 <= cc0:
            raise ValueError(f"{send.get('label') or send.get('path')}: this window lies outside "
                             f"its {H} x {W} grid")
        core = _read_window(send, rr0, cc0, rr1, cc1)
        planes.append(_reflect_pad_nd(core, rr0 - r0, (r0 + ny) - rr1,
                                      cc0 - c0, (c0 + nx) - cc1))
    stack = planes[0] if len(planes) == 1 else np.stack(planes, axis=-1)
    tag = hashlib.sha1(json.dumps(sends, sort_keys=True).encode()).hexdigest()[:10]
    prov = dict(getattr(target, "provenance", {}) or {})
    prov["bands"] = [s.get("label") or (Path(s["path"]).stem if "path" in s
                                        else f"layer {s.get('layer')}") for s in sends]
    prov["bus"] = [dict(s) for s in sends]
    return dataclasses.replace(target, values=stack, name=f"{target.name}|bus{tag}",
                               provenance=prov)
