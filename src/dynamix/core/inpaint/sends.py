# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Guides for ``inpaint`` and ``fill``: the bus's send form with a role, each read over the
window of its own grid that covers the target and never stacked.

A send is the bus's dict (a file band or a layer's shown raster, ``dynamix.core.bus``) plus
``role`` (value, structure, weights, mask, veil_seed), ``for`` (a weights send: the label of
the send it weights, or ``target``) and ``shade`` (a DEM to hillshade). This module imports the
bus's lookups and leaves the bus itself alone: the bus stacks sends onto one grid, which a
nested or foreign guide cannot join.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from dynamix.core.bus import _is_hdf4_id, _layer_plane, _read_window, parse_sends, send_grid
from dynamix.core.inpaint.engine import ROLES, Guide
from dynamix.core.inpaint.grid import (_rc, _xy, georef, georef_from, lonlat_of, pixels_of,
                                       relate)

__all__ = ["materialize_guide", "parse_roles", "resolve_guides", "send_georef", "send_label",
           "window_on"]


def send_label(send: dict) -> str:
    if send.get("label"):
        return str(send["label"])
    return f"layer {send['layer']}" if "layer" in send else str(send.get("path"))


def parse_roles(text) -> list:
    sends = parse_sends(text)
    out = []
    for send in sends:
        s = dict(send)
        role = str(s.get("role", "value") or "value")
        if role not in ROLES:
            raise ValueError(f"{send_label(s)}: unknown send role {role!r}; one of {ROLES}")
        if role == "weights" and not s.get("for"):
            raise ValueError(f"{send_label(s)}: a weights send needs 'for' (the label of the "
                             "send it weights, or 'target')")
        s["role"] = role
        out.append(s)
    labels = [send_label(s) for s in out]
    for label in labels:
        if labels.count(label) > 1:
            raise ValueError(f"{label!r}: two sends share this label; give each send its own label")
    return out


def send_crs(send: dict):
    if "layer" in send:
        return (getattr(_layer_plane(send), "provenance", None) or {}).get("crs")
    path, sub = str(send["path"]), send.get("subdataset")
    if path.lower().endswith(".npz"):
        from dynamix.core.rasterfield import RasterField

        return (RasterField.from_file(path).provenance or {}).get("crs")
    if _is_hdf4_id(path, sub):
        return None
    import rasterio

    with rasterio.open(sub if sub is not None else path) as src:
        return str(src.crs) if src.crs is not None else None


def send_lattice(send: dict):
    if "layer" in send:
        return (getattr(_layer_plane(send), "provenance", None) or {}).get("lattice")
    path, sub = str(send["path"]), send.get("subdataset")
    if _is_hdf4_id(path, sub):
        try:
            from dynamix.core.ingest import hdf4_lattice
        except ImportError:
            return None
        return hdf4_lattice(path, sub)
    return None


def send_georef(send: dict) -> dict:
    H, W, x0, dx, y0, dy = send_grid(send)
    lattice = send_lattice(send)
    crs = send_crs(send)
    source = f"layer:{send['layer']}" if "layer" in send else str(send.get("path"))
    return georef_from(crs, x0, dx, y0, dy, (H, W), lattice=lattice, source=source,
                       georeferenced=crs is not None and lattice is None)


def window_on(sg: dict, tg: dict, relation, margin: "int | None" = None) -> tuple:
    """Rows ``[r0, r1)`` and columns ``[c0, c1)`` of the send's grid that cover the target's
    extent, plus ``margin`` send pixels (default: none on the same grid, one otherwise, the
    interpolation's support), clamped to the send. An empty window means no overlap."""
    H, W = sg["shape"]
    ny, nx = tg["shape"]
    if margin is None:
        margin = 0 if relation.kind == "same" else 1
    if relation.kind == "same":
        oy, ox = relation.offset
        r0, r1, c0, c1 = -oy, ny - oy, -ox, nx - ox
    elif relation.kind == "nested" and relation.direction == "coarser":
        f, (oy, ox) = relation.ratio, relation.offset      # guide k covers target rows oy + k f ..
        r0, r1 = (-oy) // f, (ny - 1 - oy) // f + 1
        c0, c1 = (-ox) // f, (nx - 1 - ox) // f + 1
    elif relation.kind == "nested":
        f, (oy, ox) = relation.ratio, relation.offset
        r0, r1, c0, c1 = oy, oy + f * ny, ox, ox + f * nx
    else:
        # only the target's border pixels are mapped (a full-raster transform through
        # rasterio's lists takes seconds on a 2048^2 target)
        bi = np.r_[np.zeros(nx), np.full(nx, ny - 1.0), np.arange(ny, dtype=float), np.arange(ny, dtype=float)]
        bj = np.r_[np.arange(nx, dtype=float), np.arange(nx, dtype=float), np.zeros(ny), np.full(ny, nx - 1.0)]
        if tg.get("lattice") is None and sg.get("lattice") is None and tg["crs"] == sg["crs"]:
            rr, cc = _rc(sg, *_xy(tg, bi, bj))
        else:
            rr, cc = pixels_of(sg, *lonlat_of(tg, bi, bj))
        ok = np.isfinite(rr) & np.isfinite(cc)
        if not ok.any():
            r0 = r1 = c0 = c1 = 0
        else:
            r0, r1 = int(np.floor(rr[ok].min())), int(np.ceil(rr[ok].max())) + 1
            c0, c1 = int(np.floor(cc[ok].min())), int(np.ceil(cc[ok].max())) + 1
    r0, c0 = max(0, int(r0) - margin), max(0, int(c0) - margin)
    r1, c1 = min(H, int(r1) + margin), min(W, int(c1) + margin)
    return r0, c0, r1, c1


def materialize_guide(send: dict, target) -> Guide:
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    label = send_label(send)
    try:
        sg = send_georef(send)
    except ValueError as exc:
        if str(exc).startswith(label):
            raise
        raise ValueError(f"{label}: {exc}") from exc
    tg = georef(target)
    rel = relate(tg, sg)
    if rel.kind == "foreign" and rel.mapping_source == "none":
        raise ValueError(f"{label}: no mapping to the target (a raster without a georeference "
                         "or a lattice can only be sent to a window of its own band)")
    r0, c0, r1, c1 = window_on(sg, tg, rel)
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"{label}: this send lies outside the target's extent "
                         f"({sg['shape'][0]} x {sg['shape'][1]} grid)")
    values = _read_window(send, r0, c0, r1, c1)
    x0, dx, y0, dy = sg["x0"], sg["dx"], sg["y0"], sg["dy"]
    field = RasterField(
        name=label, values=values,
        frame=LocalFrame(x0=x0 + dx * c0, y0=y0 + dy * r0, dx=abs(dx), dy=abs(dy),
                         units=_units(sg)),
        x_axis=x0 + dx * (c0 + np.arange(c1 - c0)), y_axis=y0 + dy * (r0 + np.arange(r1 - r0)),
        provenance={"crs": sg["crs"], "source": sg["source"], "lattice": sg["lattice"],
                    "subdataset": send.get("subdataset"), "window": {"row_off": r0, "col_off": c0},
                    "full_dims": sg["full_shape"],
                    **({} if sg["georeferenced"] else {"georef": "none (pixel-index axes)"})})
    stamp = send.get("stamp")
    if not stamp:
        payload = dict(send)
        if "path" in send and os.path.exists(str(send["path"])):
            st = os.stat(str(send["path"]))
            payload.update(mtime_ns=st.st_mtime_ns, size=st.st_size)
        stamp = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    stamp = str(stamp)
    # the relation of the WINDOW read, whose origin is the window's own
    return Guide(label, str(send.get("role", "value")), field, relate(tg, georef(field)),
                 shade=bool(send.get("shade", False)), stamp=stamp,
                 weights_for=str(send.get("for", "target") or "target"))


def _units(sg: dict) -> str:
    if not sg["georeferenced"]:
        return "px"
    try:
        from rasterio.crs import CRS

        return "deg" if CRS.from_user_input(sg["crs"]).is_geographic else "m"
    except Exception:
        return "m"


def resolve_guides(text, target) -> list:
    """Every send as a :class:`Guide` on its own grid; a weights send bound to another send
    becomes that guide's ``weights`` (it must sit on the same grid with no offset), a weights
    send for the target stays a guide of its own."""
    sends = parse_roles(text)
    guides = [materialize_guide(s, target) for s in sends]
    by_label = {g.label: g for g in guides}
    out = []
    for g in guides:
        if g.role == "weights" and g.weights_for != "target":
            host = by_label.get(g.weights_for)
            if host is None or host.role == "weights":
                raise ValueError(f"{g.label}: weights for {g.weights_for!r}, which is not a send "
                                 "of this step")
            rel = relate(georef(host.field), georef(g.field))
            if (rel.kind != "same" or rel.offset != (0, 0) or any(rel.subpixel)
                    or host.field.values.shape != g.field.values.shape):
                raise ValueError(f"{g.label}: a weights send must sit on its measurement's grid "
                                 f"({host.label}'s), with no offset and the same extent")
            host.weights = g.field
            continue
        out.append(g)
    return out
