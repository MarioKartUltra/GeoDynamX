# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""core.units is the unit conversion; shell.units re-exports it (the layering gate)."""
from __future__ import annotations

import ast
import pathlib

import numpy as np

from dynamix.core.frames import GeographicFrame, LocalFrame
from dynamix.core.rasterfield import RasterField


def _local(dx, units, provenance=None):
    n = 4
    frame = LocalFrame(x0=0.0, y0=0.0, dx=dx, dy=dx, units=units)
    axis = np.arange(n, dtype=float) * dx
    return RasterField(name="f", values=np.zeros((n, n)), frame=frame, x_axis=axis, y_axis=axis,
                       provenance=provenance or {})


def _geographic():
    n = 4
    axis = np.linspace(-120.0, -119.97, n)
    return RasterField(name="g", values=np.zeros((n, n)), frame=GeographicFrame(),
                       x_axis=axis, y_axis=axis, provenance={"crs": "EPSG:4326"})


def test_core_and_shell_units_agree_on_local_and_geographic_fields():
    from dynamix.core import units as core_units
    from dynamix.shell import units as shell_units

    survey = 'PROJCS["x",UNIT["US survey foot",0.30480060960121924]]'
    for field in (_local(40.0, "m"), _local(40.0, "US survey foot", provenance={"crs": survey}),
                  _geographic()):
        assert core_units.px_to_metres(field) == shell_units.px_to_metres(field)
        assert core_units.physical(3.0, field) == shell_units.physical(3.0, field)


def test_shell_units_is_a_pure_re_export_of_core_units():
    from dynamix.core import units as core_units
    from dynamix.shell import units as shell_units

    for name in ("linear_unit_to_m", "physical", "_native_px_size", "px_to_metres", "in_units",
                 "deg_to_metres", "UNIT_CHOICES", "_SI", "_SI_NAME", "_M_PER_DEG"):
        assert getattr(shell_units, name) is getattr(core_units, name), name


def test_no_device_imports_the_shell():
    devices = pathlib.Path(__file__).resolve().parents[1] / "src" / "dynamix" / "devices"
    for path in sorted(devices.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not (mod or "").startswith("dynamix.shell"), path
            assert not any(n.startswith("dynamix.shell") for n in names), path
