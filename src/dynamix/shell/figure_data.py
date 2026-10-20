# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Shared immutable sample views and reference-frame conversion for scientific figures."""
from __future__ import annotations

import numpy as np

from dynamix.core.figure_samples import direction_samples, tangent_arguments
from dynamix.core.orientation import Frame, Orientation, from_arg, to_azimuth, grid_to_true


class FigureData:
    def __init__(self, result, field, offset=(0, 0)):
        self.result = result or {}
        self.field = field
        self.offset = offset
        self.scales = np.asarray(self.result.get("scales", ()), float)
        self._samples = {}
        self._tangents = {}
        from dynamix.geo.mapping import field_north
        self.north = field_north(field) if field is not None else None

    def samples(self, population="wtmmm"):
        if population not in self._samples:
            self._samples[population] = direction_samples(self.result, population)
        return self._samples[population]

    def orientations(self, index, population="wtmmm", frame="pixel", representation="gradient"):
        points = self.samples(population)[index]
        arg = np.asarray(points["arg"])
        if representation == "normal + 90°":
            arg = arg + np.pi/2
        elif representation == "chain tangent":
            if index not in self._tangents:
                original = self.result["extrema"][index]
                shape = self.result.get("_shape") or np.shape(self.field.values)[:2]
                self._tangents[index] = tangent_arguments(original, shape)
            arg = self._tangents[index][points["point_id"]]
        o = from_arg(arg)
        if frame == "pixel":
            return o
        if self.north is None:
            raise ValueError("No geographic reference for these sample axes")
        col_east, row_north, north = self.north
        az = to_azimuth(o, col_east=col_east, row_north=row_north, north=north)
        if frame == "true" and north is Frame.GRID:
            from dynamix.geo.mapping import grid_north_bearing
            az = grid_to_true(az, grid_north_bearing(self.field,
                points["x"]+self.offset[1], points["y"]+self.offset[0]))
        return az

    def components(self, index, frame="pixel"):
        points = self.samples()[index]
        o = self.orientations(index, frame=frame)
        a = np.radians(o.degrees)
        m = np.asarray(points["mod"], float)
        return (m*np.cos(a), m*np.sin(a)) if frame == "pixel" else (m*np.sin(a), m*np.cos(a))

    def context(self):
        return {"source": getattr(self.field, "name", ""), "shape": list(np.shape(getattr(self.field, "values", []))),
                "offset": list(self.offset), "scales_px": self.scales.tolist(),
                "analysis": self.result.get("params", {}),
                "source_provenance": getattr(self.field, "provenance", {})}

    def scale_label(self, index):
        from dynamix.core.units import physical
        a = float(self.scales[index])
        value, unit = physical(a, self.field)
        return f"a={a:g}px" + (f" ({value:g} {unit})" if unit not in ("px", "pixel", "pixels") else "")
