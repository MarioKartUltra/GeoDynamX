# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The shell's historical import path for the CRS-linear-unit-aware readings: every name is
re-exported from ``dynamix.core.units``, where the conversion lives and whose docstring tells the
bug it closes (BOEM's US-survey-foot bathymetry) and why the module stays free of rasterio."""

from __future__ import annotations

from dynamix.core.units import (  # noqa: F401  -- the shell's historical import path
    _GEOGRAPHIC_RE, _LENGTH_UNIT_RE, _M_PER_DEG, _MAX_UNKNOWN_LABEL_LEN, _PROJECTED_RE, _SI, _SI_NAME,
    _UNIT_TO_METRES, UNIT_CHOICES, _native_px_size, deg_to_metres, in_units, linear_unit_to_m,
    physical, px_to_metres)

__all__ = ["deg_to_metres", "in_units", "linear_unit_to_m", "px_to_metres", "UNIT_CHOICES"]
