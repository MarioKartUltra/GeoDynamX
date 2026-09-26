# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Multi-sensor grid ingestion: multiband GeoTIFF, netCDF, HDF5, HDF4 (2026-09-21).

Driver reality on this machine (measured): rasterio's bundled GDAL reads
**netCDF** and **HDF5** (subdataset enumeration + georeferencing when present) but NOT HDF4 --
ASTER L1T (HDF-EOS2) goes through **pyhdf** instead. All three libraries are already in the
env; nothing new is pinned.

Axis/frame conventions MIRROR the frozen EQSelect ``rasterfield`` (never modified, never
imported for this): pixel-CENTER axes (``arange + 0.5`` through the geotransform), nodata ->
NaN, projected CRS -> ``LocalFrame`` with the true ground pixel size (WTMM scales stay metric
-- never reproject before analysis), geographic CRS -> ``GeographicFrame``, no/identity
transform -> pixel-index axes with a default ``LocalFrame`` and an honest provenance note.

GDAL's own orientation conventions are KEPT (netCDF commonly reads bottom-up relative to
the raw array -- GDAL's north-up reading; the pinned test records it). Multi-band sources
land as ``(ny, nx, nc)`` fields -- the shape the tensor path and the new
``pca``/``tucker_havok`` devices consume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["probe", "load_grid", "load_grid_stack", "multiband_count", "GRID_SUFFIXES"]


def multiband_count(path) -> int:
    """The band count of a directly-openable raster (0 when unreadable) -- the cheap probe
    ``opening.open_field`` uses to route multiband GeoTIFFs through :func:`load_grid`."""
    try:
        import rasterio

        with rasterio.open(str(path)) as src:
            return int(src.count)
    except Exception:
        return 0

#: Suffixes this module ingests (beyond the plain-GeoTIFF path ``opening.open_field`` owns).
GRID_SUFFIXES = (".nc", ".nc4", ".cdf", ".h5", ".hdf5", ".he5", ".hdf")


def probe(path) -> dict:
    """What is in this file: ``{"kind": "gdal"|"hdf4", "subdatasets": [(id, description)],
    "count": n_bands_when_directly_openable}``.

    A netCDF/HDF5 container with multiple variables lists them as GDAL subdataset ids (the
    strings :func:`load_grid` accepts); a file GDAL opens directly reports its band count and
    an empty subdataset list. HDF4 falls back to ``pyhdf`` and lists 2-D/3-D scientific
    datasets by name.
    """
    p = Path(path)
    try:
        import rasterio

        with rasterio.open(p) as src:
            subs = list(src.subdatasets)
            if subs:
                return {"kind": "gdal",
                        "subdatasets": [(s, s.split(":")[-1]) for s in subs],
                        "count": 0}
            return {"kind": "gdal", "subdatasets": [], "count": src.count}
    except Exception:
        pass
    if p.suffix.lower() == ".hdf":
        try:
            from pyhdf.SD import SD, SDC

            sd = SD(str(p), SDC.READ)
            n_ds, _n_attr = sd.info()
            groups = _hdf4_group_names(p)
            rows = []
            for i in range(n_ds):
                ds = sd.select(i)
                dsname, rank, dims, _dt, _na = ds.info()
                if rank in (2, 3):
                    dims = tuple(int(d) for d in (dims if isinstance(dims, (list, tuple))
                                                  else [dims]))
                    rows.append((i, dsname, groups.get(ds.ref()), dims))
                ds.endaccess()
            sd.end()
            # ASTER names EVERY band's array "ImageData" (one per swath Vgroup), so a
            # name-keyed listing collapses 15 bands into one entry. Qualify with the swath
            # ("VNIR_Band1/ImageData"); a collision left inside one group falls back to the
            # SDS index. Largest grids first: the picker should lead with the images, not
            # the ancillary tables.
            from collections import Counter

            counts = Counter((g, nm) for _i, nm, g, _d in rows)
            subs = []
            for i, nm, g, dims in rows:
                sid = f"{g}/{nm}" if g else nm
                if counts[(g, nm)] > 1:
                    sid = f"{sid}#{i}"
                subs.append((sid, f"{sid} {dims}", int(np.prod(dims)), dims))
            subs.sort(key=lambda t: (-t[2], t[0]))
            return {"kind": "hdf4", "subdatasets": [(s, d) for s, d, _a, _dm in subs],
                    "dims": {s: dm for s, _d, _a, dm in subs}, "count": 0}
        except Exception as exc:
            raise ValueError(f"{p}: not readable by GDAL, and pyhdf failed too ({exc})") from exc
    raise ValueError(f"{p}: no reader could open this file (GDAL refused; not .hdf)")


def _hdf4_group_names(path) -> dict:
    """SDS ref -> the name of its enclosing HDF-EOS group: the nearest ancestor Vgroup of
    class SWATH/GRID/POINT (ASTER: a band's ImageData sits under <swath>/Data Fields), else
    the topmost named ancestor. Returns {} when the Vgroup walk fails, and callers degrade
    to index-suffixed ids -- a listing must never die on group bookkeeping."""
    try:
        from pyhdf import V  # noqa: F401 -- vgstart() needs the submodule linked
        from pyhdf.HDF import HC, HDF

        tag_ndg = getattr(HC, "DFTAG_NDG", 720)
        tag_vg = getattr(HC, "DFTAG_VG", 1965)
        h = HDF(str(path))
        v = h.vgstart()
        meta, members = {}, {}
        ref = -1
        while True:
            try:
                ref = v.getid(ref)
            except Exception:
                break
            vg = v.attach(ref)
            meta[ref] = (vg._name, vg._class)
            try:
                members[ref] = list(vg.tagrefs())
            except Exception:
                members[ref] = []
            vg.detach()
        v.end()
        h.close()
    except Exception:
        return {}

    def bookkeeping(vgref) -> bool:
        # libhdf4's own groups (Var0.0, Dim0.0, CDF0.0, RIG0.0, ...) also claim every SDS;
        # they are wiring, not a home, and must not shadow the real Data Fields group.
        return "0." in str(meta.get(vgref, ("", ""))[1])

    parent, holds = {}, {}
    for vgref, trs in members.items():
        if bookkeeping(vgref):
            continue
        for t, r in trs:
            if t == tag_ndg:
                holds.setdefault(r, vgref)
            elif t == tag_vg:
                parent.setdefault(r, vgref)
    out = {}
    for ndg, vgref in holds.items():
        seen = set()
        while (meta.get(vgref, ("", ""))[1] not in ("SWATH", "GRID", "POINT")
               and vgref in parent and vgref not in seen):
            seen.add(vgref)
            vgref = parent[vgref]
        name, cls = meta.get(vgref, ("", ""))
        if name and not bookkeeping(vgref):
            out[ndg] = name
    return out


def hdf4_select(path, sd, subdataset: str):
    """The SDS a probe id names, on an open ``SD`` handle: a bare NAME (first match -- the
    pre-qualification form, kept so saved provenance still resolves), ``Group/Name``, or
    either with ``#<index>`` from :func:`probe`."""
    sid = str(subdataset)
    try:
        return sd.select(sid)          # a real name that merely LOOKS qualified
    except Exception:
        pass
    base, hashmark, idx = sid.rpartition("#")
    if hashmark and idx.isdigit():
        ds = sd.select(int(idx))
        if ds.info()[0] != base.split("/")[-1]:
            raise ValueError(f"{path}: dataset index {idx} is no longer {base!r} "
                             f"(the file changed since this id was recorded)")
        return ds
    if "/" in sid:
        grp, nm = sid.split("/", 1)
        groups = _hdf4_group_names(path)
        n_ds, _ = sd.info()
        for i in range(n_ds):
            ds = sd.select(i)
            if ds.info()[0] == nm and groups.get(ds.ref()) == grp:
                return ds
            ds.endaccess()
        raise ValueError(f"{path}: no dataset {nm!r} in group {grp!r}")
    raise ValueError(f"{path}: no dataset named {sid!r}")


def load_grid(path, *, subdataset: "str | None" = None, name: "str | None" = None):
    """One grid (2-D or ``(ny, nx, nc)``) from ``path`` as a :class:`RasterField`.

    ``subdataset``: a GDAL subdataset id from :func:`probe` (netCDF/HDF5 variables), or an
    HDF4 scientific-dataset NAME; ``None`` opens the file directly (plain rasters).
    """
    p = Path(path)
    if subdataset is not None and not str(subdataset).startswith(
            ("NETCDF:", "HDF5:", "HDF4")) and p.suffix.lower() == ".hdf":
        return _load_hdf4(p, str(subdataset), name)
    target = subdataset if subdataset is not None else str(p)
    return _load_gdal(p, str(target), name)


def load_grid_stack(path, subdatasets, *, name: "str | None" = None):
    """Several SAME-GRID 2-D subdatasets of one container stacked ``(ny, nx, nc)``, band
    order = the list's order, ``provenance["bands"]`` naming them.

    Refuses mixed shapes: grids of different resolution (ASTER's VNIR vs SWIR vs TIR, the
    3B backsight) are different DATASETS -- never resampled onto one another (WTMM is
    scale-sensitive; combining them is band math / registration, done deliberately later).
    One subdataset degrades to a plain :func:`load_grid`."""
    ids = [str(s) for s in subdatasets]
    if not ids:
        raise ValueError(f"{path}: nothing to stack -- pick at least one grid")
    if all(i.startswith("#") for i in ids):
        return _band_subset(path, [int(i[1:]) for i in ids], ids, name)
    if len(ids) == 1:
        return load_grid(path, subdataset=ids[0], name=name)
    fields = [load_grid(path, subdataset=s) for s in ids]
    for sid, f in zip(ids, fields):
        if np.asarray(f.values).ndim != 2:
            raise ValueError(f"{path}: {sid} is not a 2-D grid; only 2-D grids stack")
    shapes = {np.asarray(f.values).shape for f in fields}
    if len(shapes) > 1:
        detail = ", ".join(f"{sid} {np.asarray(f.values).shape}"
                           for sid, f in zip(ids, fields))
        raise ValueError(f"{path}: bands on different grids cannot stack ({detail}) -- "
                         f"import them as separate datasets")
    base = fields[0]
    prov = dict(base.provenance)
    prov["subdataset"] = None
    prov["bands"] = ids
    return type(base)(
        name=name or f"{Path(path).stem}:{len(ids)} bands",
        values=np.stack([np.asarray(f.values, dtype=np.float64) for f in fields], axis=-1),
        frame=base.frame, x_axis=base.x_axis, y_axis=base.y_axis, units=base.units,
        provenance=prov,
    )


def _band_subset(path, ks: list, ids: list, name: "str | None"):
    """Bands ``ks`` (0-based) of ONE multiband file -- a GeoTIFF / npz stack after a band was
    removed from the dataset. ``provenance["bands"]`` keeps the file's own band names where
    it has them, else the ``"#k"`` tokens."""
    if str(path).lower().endswith(".npz"):
        from dynamix.core.rasterfield import RasterField

        base = RasterField.from_file(path)
    else:
        base = load_grid(path)
    values = np.asarray(base.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{path}: a band subset needs a multiband file; this one is 2-D")
    nc = values.shape[-1]
    bad = [k for k in ks if not 0 <= k < nc]
    if bad:
        raise ValueError(f"{path}: band(s) {bad} do not exist in its {nc} bands")
    names = list((base.provenance or {}).get("bands") or [])
    prov = dict(base.provenance or {})
    prov["bands"] = [str(names[k]) if k < len(names) else ids[j] for j, k in enumerate(ks)]
    picked = values[..., ks]
    return type(base)(
        name=name or base.name, values=picked[..., 0] if len(ks) == 1 else picked,
        frame=base.frame, x_axis=base.x_axis, y_axis=base.y_axis, units=base.units,
        provenance=prov)


def _load_gdal(p: Path, target: str, name: "str | None"):
    import rasterio

    from dynamix.core.frames import GeographicFrame, LocalFrame
    from dynamix.core.rasterfield import RasterField

    with rasterio.open(target) as src:
        arr = np.asarray(src.read(), dtype=np.float64)         # (bands, ny, nx)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    values = arr[0] if arr.shape[0] == 1 else np.moveaxis(arr, 0, -1)
    ny, nx = values.shape[:2]
    georef = transform is not None and not transform.is_identity
    if georef:
        cols = np.arange(nx, dtype=np.float64) + 0.5           # pixel CENTRES (rasterfield)
        rows = np.arange(ny, dtype=np.float64) + 0.5
        x_axis, _ = transform * (cols, np.zeros_like(cols))
        _, y_axis = transform * (np.zeros_like(rows), rows)
        x_axis = np.asarray(x_axis, dtype=np.float64)
        y_axis = np.asarray(y_axis, dtype=np.float64)
        if crs is not None and crs.is_projected:
            frame = LocalFrame(x0=float(x_axis[0]), y0=float(y_axis[0]),
                               dx=abs(float(transform.a)), dy=abs(float(transform.e)),
                               units=crs.linear_units)
            units = crs.linear_units
        else:
            frame = GeographicFrame()
            units = ""
    else:
        x_axis = np.arange(nx, dtype=np.float64) + 0.5
        y_axis = np.arange(ny, dtype=np.float64) + 0.5
        frame = LocalFrame()
        units = ""
    sub_tail = target.split(":")[-1] if target != str(p) else None
    return RasterField(
        name=name or (f"{p.stem}:{sub_tail}" if sub_tail else p.stem),
        values=values, frame=frame, x_axis=x_axis, y_axis=y_axis, units=units,
        provenance={"source": str(p), "subdataset": (target if target != str(p) else None),
                    "crs": str(crs) if crs is not None else None,
                    **({} if georef else {"georef": "none (pixel-index axes)"})},
    )


def _load_hdf4(p: Path, dsname: str, name: "str | None"):
    """ASTER-class HDF-EOS2 via pyhdf: values + fill-value masking; pixel-index axes (the
    swath/geolocation solve is a recorded follow-up -- provenance says so honestly)."""
    from pyhdf.SD import SD, SDC

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    sd = SD(str(p), SDC.READ)
    try:
        ds = hdf4_select(p, sd, dsname)
        arr = np.asarray(ds.get(), dtype=np.float64)
        attrs = ds.attributes()
    finally:
        sd.end()
    fill = attrs.get("_FillValue")
    if fill is not None:
        arr = np.where(arr == fill, np.nan, arr)
    if arr.ndim == 3:
        # Heuristic, documented: the smallest axis is the band axis; move it last.
        band_ax = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, band_ax, -1)
    elif arr.ndim != 2:
        raise ValueError(f"{p}:{dsname}: expected a 2-D or 3-D dataset, got shape {arr.shape}")
    ny, nx = arr.shape[:2]
    return RasterField(
        name=name or f"{p.stem}:{dsname}",
        values=arr, frame=LocalFrame(),
        x_axis=np.arange(nx, dtype=np.float64) + 0.5,
        y_axis=np.arange(ny, dtype=np.float64) + 0.5,
        provenance={"source": str(p), "subdataset": dsname,
                    "georef": "none (HDF4 v1: pixel-index axes; geolocation follow-up)"},
    )
