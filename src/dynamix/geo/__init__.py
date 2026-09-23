# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Qt-free geography helpers for the arrangement view: pixel -> lon/lat mapping with sentinel
masking. No Qt, no ``dynamix.shell`` import,
ever -- see :mod:`dynamix.geo.mapping`'s module docstring for the boundary this package holds.
"""
from dynamix.geo.mapping import NoGeoreference, field_lonlat_grid, points_lonlat

__all__ = ["NoGeoreference", "field_lonlat_grid", "points_lonlat"]
