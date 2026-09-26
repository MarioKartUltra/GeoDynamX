# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Raster footprints: where a file's data sits on the globe, read from its HEADER alone.

The data browser's first object ("the catalog is a derived
metadata store; fetches create sources-of-truth"). A footprint is derived, cheap and disposable
-- 37 ASTER GDEM tiles scan in ~50 ms because nothing here reads a pixel -- and it names a file
the user can import as an ordinary source + layer. Corners are carried in WGS84 lon/lat so the
scene can place them in any geo mode; the native bounds and CRS ride along for provenance.

``rasterio`` is imported LAZILY inside the functions that need it, matching
:mod:`dynamix.geo.mapping` and :mod:`dynamix.core.rasterfield` (the ``io`` extra is optional).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Suffixes the scanner treats as candidate rasters. Sidecars (``.tif.aux.xml``, ``.ovr``,
#: ``.tfw``) never end in one of these, so they are never opened.
RASTER_SUFFIXES = (".tif", ".tiff")


@dataclass(frozen=True)
class Footprint:
    path: str                                   # absolute path, what ``open_path`` receives
    name: str                                   # file stem
    width: int
    height: int
    count: int                                  # bands
    dtype: str
    crs: str                                    # rasterio CRS string (e.g. "EPSG:4326")
    bounds: tuple[float, float, float, float]   # native (west, south, east, north)
    corners: tuple[tuple[float, float], ...]    # WGS84 (lon, lat): SW, SE, NE, NW
    swath: bool = False                         # corners are the VALID-DATA quadrilateral, not the box

    @property
    def stack_key(self) -> tuple[float, ...]:
        """Rasters sharing a footprint (ASTER's ``_dem`` + ``_num`` tiles) share this key --
        the "same spot, several layers" case the browser's submenu groups by. Corners are
        rounded to 1e-6 degrees (~0.1 m) so two files written from the same grid agree."""
        return tuple(round(v, 6) for xy in self.corners for v in xy)

    @property
    def folder(self) -> str:
        return os.path.dirname(self.path)


def _to_lonlat(crs, xs, ys) -> tuple[tuple[float, float], ...]:
    from rasterio.warp import transform as warp_transform

    if crs.to_epsg() == 4326:
        lons, lats = xs, ys
    else:
        lons, lats = warp_transform(crs, "EPSG:4326", list(xs), list(ys))
    return tuple((float(lo), float(la)) for lo, la in zip(lons, lats))


def _corners_lonlat(crs, bounds) -> tuple[tuple[float, float], ...]:
    w, s, e, n = bounds
    return _to_lonlat(crs, [w, e, e, w], [s, s, n, n])          # SW, SE, NE, NW


#: Below this valid-data fraction a raster is treated as a swath inside a fill border (an ASTER
#: scene covers ~85% of its north-up box); above it the box IS the footprint (a GDEM tile).
_SWATH_MAX_VALID_FRACTION = 0.98
#: Coarsest overview whose short side still has at least this many pixels -- enough to fit the
#: swath's straight edges to within a couple of full-resolution pixels.
_SWATH_MIN_OVERVIEW_PX = 200


def _swath_corners(ds):
    """The valid-data quadrilateral of an open dataset in ITS OWN CRS as ``[SW, SE, NE, NW]``,
    from the coarsest usable overview -- 2.7 ms per ASTER granule against 80 ms for row strips,
    because LZW blocks make a full-width row as expensive as a full read. ``None`` when the
    raster has no overviews (a decimated read would pull every pixel: not silently) or when it
    is valid to its edges. Edges are fitted as straight lines through the first/last valid
    column of a dozen sampled rows, the shape a pushbroom swath has."""
    from rasterio.enums import Resampling

    overviews = ds.overviews(1)
    if not overviews:
        return None
    h, w = ds.height, ds.width
    usable = [f for f in overviews if min(h, w) / f >= _SWATH_MIN_OVERVIEW_PX]
    factor = max(usable) if usable else min(overviews)
    oh, ow = max(1, round(h / factor)), max(1, round(w / factor))
    a = ds.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
    fill = 0 if ds.nodata is None else ds.nodata
    valid = a != fill
    if valid.mean() > _SWATH_MAX_VALID_FRACTION:
        return None
    rows = np.flatnonzero(valid.any(axis=1))
    if rows.size < 2:
        return None
    sample = rows[:: max(1, rows.size // 12)]
    spans = [(r, np.flatnonzero(valid[r])[0], np.flatnonzero(valid[r])[-1] + 1) for r in sample]
    r = np.array([s[0] + 0.5 for s in spans], float)
    left = np.array([s[1] for s in spans], float)
    right = np.array([s[2] for s in spans], float)
    al, bl = np.polyfit(r, left, 1)
    ar, br = np.polyfit(r, right, 1)
    r0, r1 = float(rows.min()), float(rows.max() + 1)
    sx, sy = w / ow, h / oh
    nw, ne = (al * r0 + bl, r0), (ar * r0 + br, r0)
    sw, se = (al * r1 + bl, r1), (ar * r1 + br, r1)
    return [ds.transform * (c * sx, rw * sy) for c, rw in (sw, se, ne, nw)]


def scan_footprints(folder) -> list[Footprint]:
    """Every raster under ``folder`` (recursive), headers only, sorted by path. A raster with
    no CRS cannot be placed on the globe and is skipped; a missing folder raises."""
    import rasterio

    root = Path(folder).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"{root}: not a folder")
    out: list[Footprint] = []
    swath_memo: dict = {}      # one overview read per ASTER granule, shared by its band files
    for path in sorted(p for p in root.rglob("*") if p.is_file()
                       and p.suffix.lower() in RASTER_SUFFIXES
                       and not p.name.startswith(".")):     # ._ AppleDouble sidecars on exFAT
        try:
            with rasterio.open(path) as ds:
                if ds.crs is None:
                    continue
                b = ds.bounds
                m = _ASTER_RE.match(path.stem)
                key = m.group(0) if m else str(path)
                if key not in swath_memo:
                    swath_memo[key] = _swath_corners(ds)
                quad = swath_memo[key]
                if quad is not None:
                    corners = _to_lonlat(ds.crs, [q[0] for q in quad], [q[1] for q in quad])
                else:
                    corners = _corners_lonlat(ds.crs, (b.left, b.bottom, b.right, b.top))
                out.append(Footprint(
                    path=str(path.resolve()), name=path.stem,
                    width=int(ds.width), height=int(ds.height), count=int(ds.count),
                    dtype=str(ds.dtypes[0]), crs=ds.crs.to_string(),
                    bounds=(float(b.left), float(b.bottom), float(b.right), float(b.top)),
                    corners=corners, swath=quad is not None,
                ))
        except rasterio.errors.RasterioIOError:
            continue                                    # not a raster after all; keep scanning
    return out


def _point_in_polygon(lon: float, lat: float, corners) -> bool:
    """Even-odd test over the footprint's lon/lat quadrilateral (PNPOLY, scalar form -- four
    edges, no need for :func:`dynamix.core.selection.points_in_polygon`'s vectorised path)."""
    inside = False
    n = len(corners)
    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]
        if (y0 > lat) != (y1 > lat):
            x_cross = x0 + (lat - y0) * (x1 - x0) / (y1 - y0)
            if lon < x_cross:
                inside = not inside
    return inside


def _area(corners) -> float:
    xs = np.array([c[0] for c in corners]); ys = np.array([c[1] for c in corners])
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2.0)


def footprints_at(footprints, lon: float, lat: float) -> list[Footprint]:
    """Every footprint containing (lon, lat), smallest first (the most specific dataset on top),
    ties -- stacked variants of one tile -- by name so ``_dem`` precedes ``_num``."""
    hits = [f for f in footprints if _point_in_polygon(lon, lat, f.corners)]
    return sorted(hits, key=lambda f: (_area(f.corners), f.name))


# ----------------------------------------------------------------- names -> scene / band labels

#: LP DAAC ASTER granule ids: ``AST_<product>_<vvv><MMDDYYYY><HHMMSS>_<processing>_...`` --
#: e.g. ``AST_07XT_00411282015022621_20250809164339_SRF_VNIR_B01`` is product 07XT, version 004,
#: acquired 2015-11-28 02:26:21. The date is what a human needs to tell 13 overlapping passes
#: over one spot apart; the filename alone never says it in a readable order.
_ASTER_RE = re.compile(r"^AST_(?P<product>[A-Z0-9]+)_(?P<ver>\d{3})(?P<mm>\d{2})(?P<dd>\d{2})"
                       r"(?P<yyyy>\d{4})(?P<hh>\d{2})(?P<mi>\d{2})(?P<ss>\d{2})_")


def common_prefix(names) -> str:
    """The names' shared prefix cut back to a ``_`` token boundary -- ``B01``/``B02``/``B03N``
    share the characters ``…_B0`` but the TOKEN they share ends at ``…_VNIR_``."""
    names = list(names)
    if not names:
        return ""
    first, last = min(names), max(names)
    i = 0
    while i < min(len(first), len(last)) and first[i] == last[i]:
        i += 1
    prefix = first[:i]
    if len(names) > 1 and i < len(first) and "_" in prefix:
        prefix = prefix[:prefix.rfind("_") + 1]
    return prefix


def scene_label(names) -> tuple[str, str | None]:
    """``(label, iso_time | None)`` for a group of files that share one footprint: an ASTER
    granule reads ``"AST_07XT · 2015-11-28 02:26"`` with its acquisition time in ISO form for
    sorting; anything else is the names' common prefix (trailing ``_`` stripped) and ``None``."""
    names = list(names)
    m = _ASTER_RE.match(names[0]) if names else None
    if m and all(_ASTER_RE.match(n) and _ASTER_RE.match(n).group(0) == m.group(0) for n in names):
        d = m.groupdict()
        iso = f"{d['yyyy']}-{d['mm']}-{d['dd']}T{d['hh']}:{d['mi']}:{d['ss']}"
        return f"AST_{d['product']} · {d['yyyy']}-{d['mm']}-{d['dd']} {d['hh']}:{d['mi']}", iso
    prefix = common_prefix(names).rstrip("_")
    return (prefix or (names[0] if names else "")), None


def band_label(name: str, names) -> str:
    """What distinguishes ``name`` inside its footprint group -- ``B01`` / ``B03N`` /
    ``QA_DataPlane`` for an ASTER granule, ``dem`` / ``num`` for a GDEM tile; the whole name
    when the group has one member (or nothing in common)."""
    names = list(names)
    if len(names) < 2:
        return name
    prefix = common_prefix(names)
    rest = name[len(prefix):].lstrip("_")
    return rest or name


def group_key(fp: Footprint):
    """What the browser groups a footprint under: the ASTER granule id when the name carries one
    (a granule's VNIR files at 15 m and SWIR files at 30 m sit a few metres apart -- their
    ``stack_key``s differ -- yet they are one scene), else the exact footprint
    (:attr:`Footprint.stack_key`), which is what makes a GDEM tile's ``_dem`` + ``_num`` one group."""
    m = _ASTER_RE.match(fp.name)
    return ("aster", m.group(0)) if m else fp.stack_key


_BAND_RE = re.compile(r"_B(\d+)(N?)$")


def band_sort_key(name: str):
    """Bands in numeric order (``B01``, ``B03N``, ``B04`` … ``B09``), then QA planes, then
    everything else by name -- the order a person reads a granule's files in."""
    m = _BAND_RE.search(name)
    if m:
        return (0, int(m.group(1)), m.group(2), name)
    if "QA" in name:
        return (1, 0, "", name)
    return (2, 0, "", name)


# ------------------------------------------------------------------------------ previews

def overview_field(path, max_px: int = 400, name: str | None = None):
    """A small :class:`~dynamix.core.rasterfield.RasterField` of ``path`` for PREVIEW draping
    on the world -- never for analysis (WTMM is scale-sensitive; the preview is resampled by
    construction and says so in its name and provenance).

    Reads the coarsest overview with its long side still ≥ ``max_px`` … at most ``max_px``
    px on the long side; a raster without overviews is read decimated (GDAL pulls the pixels
    it needs). The fill value (``nodata``, else 0 -- the ASTER swath border) becomes NaN so the
    drape is transparent outside the scene. Frame/axes follow ``RasterField.from_geotiff_window``
    exactly: pixel CENTRES, a :class:`LocalFrame` in the CRS's linear units for a projected
    source, :class:`GeographicFrame` otherwise, ``provenance["crs"]`` for the geo mapping."""
    import rasterio
    from rasterio.enums import Resampling

    from dynamix.core.frames import GeographicFrame, LocalFrame
    from dynamix.core.rasterfield import RasterField

    path = Path(path)
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        factor = max(1, -(-max(h, w) // max_px))            # ceil(long side / max_px)
        oh, ow = max(1, round(h / factor)), max(1, round(w / factor))
        band = ds.read(1, out_shape=(oh, ow), resampling=Resampling.nearest).astype(np.float64)
        fill = 0.0 if ds.nodata is None else float(ds.nodata)
        transform, crs = ds.transform, ds.crs
    band[band == fill] = np.nan
    # BOEM west declares nodata 0.0 while its border holds float32-lowest: extreme sentinels
    # are fill regardless of what the header claims -- same rule as mapping's drapes.
    from dynamix.geo.mapping import _mask_sentinels
    band = _mask_sentinels(band)
    sx, sy = w / ow, h / oh
    cols = (np.arange(ow, dtype=np.float64) + 0.5) * sx
    rows = (np.arange(oh, dtype=np.float64) + 0.5) * sy
    x_axis, _ = transform * (cols, np.zeros_like(cols))
    _, y_axis = transform * (np.zeros_like(rows), rows)
    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    if crs is not None and crs.is_projected:
        frame = LocalFrame(x0=float(x_axis[0]), y0=float(y_axis[0]),
                           dx=abs(float(transform.a)) * sx, dy=abs(float(transform.e)) * sy,
                           units=crs.linear_units)
    else:
        frame = GeographicFrame()
    return RasterField(
        name=name or f"{path.stem}@preview", values=band, frame=frame,
        x_axis=x_axis, y_axis=y_axis,
        units=(crs.linear_units if (crs is not None and crs.is_projected) else ""),
        provenance={"source": str(path), "overview": int(factor), "preview": True,
                    "crs": str(crs) if crs is not None else None},
    )
