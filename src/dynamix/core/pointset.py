# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""PointSet: a georeferenced point catalogue (earthquakes, first) read out of a plain CSV.

Headless, like every other module under ``dynamix/core`` (nothing outside
``dynamix/shell/`` imports PySide6) -- stdlib ``csv`` throughout, no pandas. The shell's
``point_import.py`` is the only place a lon/lat mismatch ever becomes a dialog; this module only
ever raises.

**Column detection.** :data:`_DETECT` maps each of the four canonical axes this app knows about
(``lon``, ``lat``, ``depth``, ``mag``) to the header spellings it recognises, case-insensitively.
``lon``/``lat`` are required -- :func:`read_csv_points` raises ``ValueError`` naming the actual
header found when neither can be identified, so the shell can turn that into a mapping dialog
without this module ever knowing dialogs exist. ``depth`` becomes :attr:`PointSet.depth` (the one
other axis with a dedicated dataclass field); ``mag`` has none, so a detected magnitude column
lands in :attr:`PointSet.attrs["mag"]`` instead -- ``attrs`` exists precisely for axes the
dataclass has no named slot for.

**Quote-aware load.** The data rows are read with
stdlib ``csv.reader`` -- the SAME quote-aware, C-accelerated reader the header row already used --
building one Python list per surviving column and converting each to a ``np.ndarray`` at the end,
rather than ``np.genfromtxt(delimiter=",")``, which is not quote-aware: a quoted field containing a
comma (USGS catalogues' own ``place`` column, e.g. ``"10km ENE of Somewhere, CA"``) splits into an
extra column under naive comma-splitting, disagrees with the header's own column count, and used to
be dropped by ``genfromtxt(invalid_raise=False)`` SILENTLY -- never entering ``dropped_rows``, so a
real catalogue could lose a large fraction of its events with a provenance claiming near-zero drops.
Addressed by parsing quote-aware, and by counting EVERY dropped row honestly: a row is dropped (and
counted) for either of two reasons -- (1) its column count does not match the header's, or (2) its
lon or lat value fails to parse as a float. An unparsable value in an OPTIONAL column (depth/mag)
does not drop the row -- it becomes ``nan`` in place for that one axis, exactly as it always did;
only lon/lat unparsability makes a row unusable as a point.
"""
from __future__ import annotations

import csv
import dataclasses

import numpy as np

__all__ = ["PointSet", "read_csv_points", "resolve_columns"]

#: Auto-detection table: canonical axis -> recognised header spellings, lower-cased (detection is
#: case-insensitive; see :func:`_header_lookup`).
_DETECT = {
    "lon": ("lon", "longitude", "long", "x_wgs84"),
    "lat": ("lat", "latitude", "y_wgs84"),
    "depth": ("depth", "depth_km", "z"),
    "mag": ("mag", "magnitude", "ml", "mw"),
}


@dataclasses.dataclass
class PointSet:
    """A loaded point catalogue: parallel arrays, one entry per surviving row."""

    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray | None = None
    attrs: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    provenance: dict[str, str] = dataclasses.field(default_factory=dict)


def _header_lookup(headers) -> dict[str, tuple[str, int]]:
    """``lower-cased header -> (raw header text, column index)``, first occurrence wins."""
    lookup: dict[str, tuple[str, int]] = {}
    for i, h in enumerate(headers):
        key = str(h).strip().lower()
        lookup.setdefault(key, (h, i))
    return lookup


def resolve_columns(headers, *, mapping: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve the four canonical axes to actual header text in ``headers``.

    ``mapping`` (a ``{"lon": "<header>", ...}`` dict, e.g. the shell's mapping dialog's answer)
    overrides auto-detection PER KEY: a key present in ``mapping`` is looked up exactly (never
    auto-detected, even if the named header doesn't exist in this file -- silently falling back
    would override the caller's explicit choice with a possibly different column); a key absent
    from ``mapping`` falls through to :data:`_DETECT`. Returns only the axes actually found --
    ``depth``/``mag`` are simply absent from the result when neither the mapping nor detection
    names one.

    Raises ``ValueError`` (naming the header this ACTUALLY saw) when ``lon`` or ``lat`` cannot be
    resolved either way -- the one place this module ever raises, and the shell's cue to open the
    mapping dialog.
    """
    lookup = _header_lookup(headers)
    mapping = mapping or {}
    resolved: dict[str, str] = {}
    for key in ("lon", "lat", "depth", "mag"):
        if key in mapping:
            named = mapping[key]
            if named:
                found = lookup.get(str(named).strip().lower())
                if found is not None:
                    resolved[key] = found[0]
            continue
        for candidate in _DETECT[key]:
            found = lookup.get(candidate)
            if found is not None:
                resolved[key] = found[0]
                break
    if "lon" not in resolved or "lat" not in resolved:
        raise ValueError(
            f"could not identify lon/lat columns from header {list(headers)!r}"
        )
    return resolved


def _to_float_or_nan(text: str) -> float:
    """``float(text)``, or ``nan`` when it can't parse -- the same tolerance ``np.genfromtxt``'s
    own dtype coercion used to give every column: an unparsable OPTIONAL value (depth/mag) must
    not cost the whole row, only :func:`read_csv_points`'s own explicit lon/lat check does that."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def read_csv_points(path, *, mapping: dict[str, str] | None = None) -> PointSet:
    """Read a point catalogue CSV into a :class:`PointSet`.

    ``mapping`` overrides auto-detection (see :func:`resolve_columns`); a missing/unmappable
    lon or lat raises ``ValueError`` there, unhandled here -- the shell catches it and opens the
    mapping dialog, this module stays headless.

    One quote-aware pass over the data rows (see the module docstring's "Quote-aware load"
    section) -- a row is dropped, and counted into ``provenance["dropped_rows"]``, when its column
    count disagrees with the header's, or when its lon or lat value fails to parse as a float.
    """
    path = str(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        resolved = resolve_columns(header, mapping=mapping)
        column_index = {h: i for i, h in enumerate(header)}
        n_cols = len(header)

        lon_idx = column_index[resolved["lon"]]
        lat_idx = column_index[resolved["lat"]]
        depth_idx = column_index[resolved["depth"]] if "depth" in resolved else None
        mag_idx = column_index[resolved["mag"]] if "mag" in resolved else None

        lon_list: list[float] = []
        lat_list: list[float] = []
        depth_list: list[float] | None = [] if depth_idx is not None else None
        mag_list: list[float] | None = [] if mag_idx is not None else None
        dropped = 0

        for row in reader:
            if len(row) != n_cols:
                dropped += 1                              # wrong column count -- unusable row
                continue
            lon_v = _to_float_or_nan(row[lon_idx])
            lat_v = _to_float_or_nan(row[lat_idx])
            if np.isnan(lon_v) or np.isnan(lat_v):
                dropped += 1                               # lon/lat itself unparsable
                continue
            lon_list.append(lon_v)
            lat_list.append(lat_v)
            if depth_list is not None:
                depth_list.append(_to_float_or_nan(row[depth_idx]))
            if mag_list is not None:
                mag_list.append(_to_float_or_nan(row[mag_idx]))

    attrs: dict[str, np.ndarray] = {}
    if mag_list is not None:
        attrs["mag"] = np.asarray(mag_list, dtype=np.float64)

    return PointSet(
        lon=np.asarray(lon_list, dtype=np.float64),
        lat=np.asarray(lat_list, dtype=np.float64),
        depth=np.asarray(depth_list, dtype=np.float64) if depth_list is not None else None,
        attrs=attrs,
        provenance={"source": path, "dropped_rows": str(dropped)},
    )
