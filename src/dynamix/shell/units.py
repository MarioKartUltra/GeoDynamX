# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""CRS-linear-unit-aware physical readings -- the one conversion every reading in the shell goes
through, so they can never disagree with each other.

This module re-exports ``dynamix.core.units``, where the conversion now lives; see that module.

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

from dynamix.core.units import (  # noqa: F401  -- the shell's historical import path
    _GEOGRAPHIC_RE, _LENGTH_UNIT_RE, _M_PER_DEG, _MAX_UNKNOWN_LABEL_LEN, _PROJECTED_RE, _SI, _SI_NAME,
    _UNIT_TO_METRES, UNIT_CHOICES, _native_px_size, deg_to_metres, in_units, linear_unit_to_m,
    physical, px_to_metres)

__all__ = ["deg_to_metres", "in_units", "linear_unit_to_m", "px_to_metres", "UNIT_CHOICES"]
