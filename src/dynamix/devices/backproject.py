# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The ``backproject`` transform: registers a point layer's lon/lat onto ANOTHER layer's pixel
grid.

**ENGINE LAW.** This device never touches the target layer's own field object -- it only ever
sees the point layer's own :class:`~dynamix.core.pointset.PointSet` (``field`` in ``compute``).
The shell (``dynamix.shell.main_window``'s ``_stamp_backproject``) resolves ``params["target"]``
against the OTHER layer, once, on chain edit, and stamps seven scalar readings straight from that
layer's field onto this device's own params: the target CRS text, its axis origin/step (x0/dx,
y0/dy) and its pixel-grid size (nx, ny). ``compute`` reconstructs the exact inverse-affine pixel
mapping :func:`dynamix.geo.mapping.lonlat_to_pixels` performs directly against a field object --
same lazy ``rasterio`` import, same pip-hint ``ValueError``, same
``col = (x - x0) / dx`` / ``row = (y - y0) / dy`` math -- from those seven numbers alone. One
batched ``rasterio.warp.transform`` call for the whole point set; never a per-point loop.

This split is why retargeting is cheap and honest: the seven scalars ride ``params`` like any
other knob, so ``cache_key`` (a content hash of ``params``) changes automatically the instant the
shell stamps a different target's axes in -- no separate invalidation path to keep in sync.

**Honest no-op.** An empty ``target`` or an unresolved/non-georeferenced one (the shell's own
refusal path) stamps every scalar to zero -- a "zeroed grid spec". ``compute`` treats that (or any
``target`` text with ``nx <= 0`` or ``ny <= 0``) as UNBOUND: it returns ``points_px: None`` and
``_unbound: True`` rather than dividing by a zero step or fabricating a placement, with every other
key (``inside``, ``attrs``, ``_target``, ``_shape``) still present in the same shape the bound case
returns -- a consumer never has to special-case the schema, only the ``points_px``/``_unbound``
values.
"""
from __future__ import annotations

import numpy as np

from dynamix.model.param import Param, ParamKind

#: Generous, never-realistically-binding bounds on the stamped FLOAT scalars -- present only so
#: ``dynamix.shell.knobs.control_spec`` (called unconditionally over every registered device's
#: every param by ``tests/test_shell_knobs.py``) has a legal ``(hi - lo)`` to compute. No strip
#: ever actually RENDERS a control for these -- ``workflow_zone._grouped_params`` filters every
#: ``_``-prefixed param out of the box's control grid (see that function's own docstring for the
#: mechanism) -- so these bounds are never seen or nudged by a user; they exist purely to keep the
#: param declaration legal. ``_COORD_BOUND`` covers any CRS's projected coordinate (metres or US
#: survey feet alike -- BOEM's own false-easting is already in the millions);
#: ``_STEP_BOUND`` covers any plausible per-pixel step.
_COORD_BOUND = 1e12
_STEP_BOUND = 1e9


def _lonlat_to_pixels_from_scalars(crs_text, lon, lat, x0, dx, y0, dy):
    """WGS84 lon/lat arrays -> fractional (cols, rows), reconstructed from seven scalar grid
    readings instead of a field object -- see the module docstring's "ENGINE LAW" section.

    Mirrors :func:`dynamix.geo.mapping.lonlat_to_pixels` exactly: one batched
    ``rasterio.warp.transform("EPSG:4326", crs, lon, lat)`` into the target's CRS, then the
    axis-derived inverse of the pixel-center mapping, ``col = (x - x0) / dx``,
    ``row = (y - y0) / dy`` -- the same lazy-``rasterio``-import, same pip-hint ``ValueError``
    pattern as ``mapping.py``'s own ``_to_lonlat``.
    """
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform as warp_transform
    except ImportError as exc:
        raise ValueError(f"backproject needs rasterio: pip install rasterio ({exc})") from exc

    crs = CRS.from_user_input(crs_text)
    x, y = warp_transform("EPSG:4326", crs, lon, lat)   # ONE batched call; no per-point loop
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cols = (x - x0) / dx
    rows = (y - y0) / dy
    return cols, rows


class Backproject:
    """Registers a point layer's lon/lat onto another layer's pixel grid, from stamped scalars
    alone -- see the module docstring."""

    name = "backproject"
    params = (
        Param("target", ParamKind.TEXT, default="", label="Target layer", editable=True),
        # The seven scalars below are shell-stamped, never user-edited -- see the module
        # docstring's "ENGINE LAW" section and workflow_zone._grouped_params' filtering mechanism.
        # Defaults are all zero: the all-zero grid spec IS the "unbound" state (see `compute`), so
        # a freshly dropped `backproject` step with no target yet is a well-formed, honest no-op
        # rather than a device waiting on values nobody has written.
        Param("_target_crs", ParamKind.TEXT, default="", label="Target CRS"),
        Param("_target_x0", ParamKind.FLOAT, default=0.0, min=-_COORD_BOUND, max=_COORD_BOUND,
              label="Target x0"),
        Param("_target_dx", ParamKind.FLOAT, default=0.0, min=-_STEP_BOUND, max=_STEP_BOUND,
              label="Target dx"),
        Param("_target_y0", ParamKind.FLOAT, default=0.0, min=-_COORD_BOUND, max=_COORD_BOUND,
              label="Target y0"),
        Param("_target_dy", ParamKind.FLOAT, default=0.0, min=-_STEP_BOUND, max=_STEP_BOUND,
              label="Target dy"),
        Param("_target_nx", ParamKind.INT, default=0, min=0, label="Target nx"),
        Param("_target_ny", ParamKind.INT, default=0, min=0, label="Target ny"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        target = str(params.get("target") or "")
        nx = int(params["_target_nx"])
        ny = int(params["_target_ny"])
        lon = np.asarray(field.lon, dtype=np.float64)
        lat = np.asarray(field.lat, dtype=np.float64)
        attrs = dict(getattr(field, "attrs", None) or {})

        if not target or nx <= 0 or ny <= 0:
            # Honest no-op: nothing to register against yet (empty target, or the shell's own
            # zeroed refusal stamp) -- never a crash, never a fabricated placement. Same
            # result-dict SHAPE as the bound case below (every key still present).
            return {"points_px": None, "inside": np.zeros(lon.shape, dtype=bool),
                    "attrs": attrs, "_target": target, "_shape": (ny, nx), "_unbound": True}

        cols, rows = _lonlat_to_pixels_from_scalars(
            params["_target_crs"], lon, lat,
            float(params["_target_x0"]), float(params["_target_dx"]),
            float(params["_target_y0"]), float(params["_target_dy"]))
        inside = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
        return {"points_px": {"x": cols, "y": rows}, "inside": inside, "attrs": attrs,
                "_target": target, "_shape": (ny, nx)}

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
