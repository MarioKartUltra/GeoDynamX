# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""CRS-linear-unit-aware physical readings -- the one conversion every reading in the shell goes
through, so they can never disagree with each other.

The bug this module closes ("Readings and display"):
``MainWindow._px_to_unit`` used to read the pixel size straight off ``field.x_axis`` and hand it
to the display unlabeled-but-for ``field.frame.units``. That is correct for a metre CRS and for a
non-georeferenced field (an EBSD map in microns, a plain image in pixels) -- the axis is already
in the unit it claims to be in. It is not correct for a PROJECTED CRS whose linear unit is not the
metre: BOEM's Gulf bathymetry is on a 40 **US-survey-foot** grid, and the old code showed "40 US
survey foot" (honestly labeled, at least) rather than converting to the metres every other reading
in the app assumes. This module does the conversion, once, so the transport's scale reading, the
canvas scale bar, the wavelet bar and a device's derived reading (``dynamix.devices.wtmm.
WTMM2D.derived_reading``) all agree.

Pure module: string parsing over the CRS text a ``RasterField`` already carries in
``provenance["crs"]`` (``RasterField.from_geotiff_window``/``geotiff_info``,
dynamix/core/rasterfield.py, stash ``str(rasterio_crs)`` there). No rasterio import here -- by the
time a field reaches the shell, the WKT is already provenance; re-opening the file to ask a CRS
library for its units would be strictly more work for the same answer, and would break the "no
rasterio import" rule pure shell modules follow so they stay importable without the optional
dependency installed.
"""
from __future__ import annotations

import re

import numpy as np

__all__ = ["linear_unit_to_m", "px_to_metres"]

# Factor-to-metres for every linear unit name this app recognises, matching the exact WKT UNIT[]
# conversion factors GDAL/PROJ emit (EPSG 9001 metre, 9003 US survey foot, 9002 international
# foot) rather than a rounded approximation -- a 40-ft BOEM cell converts to 12.192024384048768 m,
# not 12.192.
_UNIT_TO_METRES = {
    "metre": 1.0,
    "meter": 1.0,
    "us survey foot": 0.30480060960121924,
    "foot": 0.3048,
    "feet": 0.3048,
}

# WKT1 uses a bare UNIT["name", factor] for BOTH angular and linear axes; WKT2 disambiguates with
# LENGTHUNIT[...]/ANGLEUNIT[...]. Matching LENGTHUNIT-or-UNIT (never ANGLEUNIT) keeps one regex
# honest for either dialect: a WKT2 GEOGCRS's ellipsoid carries its own LENGTHUNIT["metre",1] for
# the datum's radius, so the geographic check below runs FIRST and short-circuits before this
# regex ever gets to a bare geographic CRS.
_LENGTH_UNIT_RE = re.compile(r'(?:LENGTHUNIT|UNIT)\s*\[\s*"([^"]+)"', re.IGNORECASE)
_GEOGRAPHIC_RE = re.compile(r'\b(GEOGCS|GEOGCRS|GEODCRS)\b', re.IGNORECASE)
_PROJECTED_RE = re.compile(r'\b(PROJCS|PROJCRS)\b', re.IGNORECASE)

#: Above this, a raw unrecognised string is not worth echoing back as a "unit label" -- it is
#: almost certainly a WKT fragment that failed to parse, not a short unit name.
_MAX_UNKNOWN_LABEL_LEN = 40


def linear_unit_to_m(crs_or_provenance) -> tuple[float | None, str]:
    """``(factor to metres, display unit)`` for a CRS's linear UNIT.

    ``crs_or_provenance`` is either a provenance dict (its ``"crs"`` entry is read) or a raw CRS
    string. ``RasterField.from_geotiff_window``/``geotiff_info`` both stash ``str(rasterio's
    CRS)`` in ``provenance["crs"]``, which is the FULL WKT whenever the source CRS has no exact
    EPSG match (BOEM's custom NAD27 grids -- verified against a real ``rasterio.crs.CRS`` while
    building this module, see ``tests/test_shell_units.py``) and a bare ``"EPSG:<code>"`` when it
    does. A bare code carries no unit text to parse; see the "unknown" case below.

    A recognised linear unit (metre/meter, US survey foot, foot) converts to ``(factor, "m")`` --
    the DISPLAY unit becomes plain metres even when the source was feet, because a number that has
    been converted belongs to the unit it was converted TO, not the one it came from. A bare
    geographic CRS (``GEOGCS``/``GEOGCRS`` with no ``PROJCS``/``PROJCRS`` wrapping it) has no
    linear unit at all: ``(None, "deg")``, meaning stay in native degrees -- never fabricate a
    metre figure for an angular unit. Anything else -- no CRS text, a bare EPSG code, or a linear
    unit name this app does not recognise -- returns ``(None, <best available name>)``.

    ``None`` is the point of this function existing. A caller that falls back to a silent ``1.0``
    here would print a false metre reading for a foot-based grid -- the exact bug this module
    replaces. Every caller must handle ``None`` explicitly (:func:`px_to_metres` does, by staying
    in the native unit rather than converting).
    """
    text = crs_or_provenance.get("crs") if isinstance(crs_or_provenance, dict) else crs_or_provenance
    if not text:
        return None, "unknown"
    text = str(text)

    # A bare unit name needs no WKT structure to resolve -- convenient for a caller that already
    # has `crs.linear_units` (a plain string like "US survey foot") rather than the full WKT.
    bare = text.strip().strip('"').lower()
    if bare in _UNIT_TO_METRES:
        return _UNIT_TO_METRES[bare], "m"

    if _GEOGRAPHIC_RE.search(text) and not _PROJECTED_RE.search(text):
        return None, "deg"

    matches = _LENGTH_UNIT_RE.findall(text)
    if not matches:
        return None, (text if len(text) <= _MAX_UNKNOWN_LABEL_LEN else "unknown")

    # The LAST UNIT[]/LENGTHUNIT[] in a PROJCS WKT is the projected (linear) one; any earlier
    # match belongs to the nested GEOGCS's degrees (WKT1 lists GEOGCS, with its own UNIT["degree",
    # ...], before the outer PROJCS's own UNIT[...]).
    name = matches[-1].strip()
    factor = _UNIT_TO_METRES.get(name.lower())
    if factor is None:
        return None, name
    return factor, "m"


def _native_px_size(field) -> float | None:
    """Pixel size straight off the field's own x-axis, in whatever unit the axis is already
    expressed in -- the exact computation ``MainWindow._px_to_unit`` used before this module
    existed. ``None`` (never a fabricated ``1.0``) when the axis is too short to have a spacing at
    all; the caller decides what a missing pixel size should fall back to."""
    axis = np.asarray(getattr(field, "x_axis", []), dtype=float)
    if axis.size < 2:
        return None
    return float(abs(axis[-1] - axis[0]) / (axis.size - 1))


def px_to_metres(field) -> tuple[float | None, str]:
    """``(metres-per-pixel or None, display unit)`` for ``field`` -- the one conversion every
    physical reading in the shell goes through: the transport's scale reading, the canvas scale
    bar, the wavelet bar, and a device's derived reading (``WTMM2D.derived_reading``).

    Two regimes:

    - ``field.provenance`` carries a ``"crs"`` key (georeferenced: opened from a GeoTIFF) and the
      field's own frame is not already reporting degrees -- the pixel size is in the CRS's OWN
      linear unit (``RasterField.from_geotiff_window`` copies ``rasterio``'s ``transform.a``
      straight into the axis with no conversion), so it is run through :func:`linear_unit_to_m`. A
      recognised linear unit converts to real metres. A linear unit this app cannot name keeps the
      raw axis-derived number under whatever label :func:`linear_unit_to_m` returned -- "stay
      native" rather than lie about the unit.
    - No ``"crs"`` key (a plain image, an EBSD ``.npz``, a bare array) OR the field's frame already
      reports ``"deg"`` (``GeographicFrame`` -- this codebase's own discriminator for "this axis is
      already lon/lat", more reliable than re-detecting geographic-ness from a CRS string that may
      have been abbreviated to a bare EPSG code with no unit text left to parse) -- the axis is
      ALREADY in the frame's own physical unit; there is no CRS conversion to apply, so the factor
      is implicitly 1.0 and the field's own unit is the label.

    ``None`` propagates only when the pixel size itself cannot be determined at all (an axis too
    short to have a spacing) -- there is nothing to convert either way.
    """
    frame_unit = str(getattr(getattr(field, "frame", None), "units", "px") or "px")
    dx = _native_px_size(field)
    if dx is None:
        return None, frame_unit

    provenance = getattr(field, "provenance", None) or {}
    if "crs" not in provenance or frame_unit == "deg":
        return dx, frame_unit

    factor, label = linear_unit_to_m(provenance)
    if factor is None:
        return dx, label
    return dx * factor, label
