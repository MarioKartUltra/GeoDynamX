# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The user's own colour ramps: named lists of ``(position, "#rrggbb")`` stops kept in the settings
file (``Settings.ramps``), resolved to a pyqtgraph ``ColorMap`` wherever a colormap name is looked
up (``canvas._resolve_cmap``). They belong to the user, so a project names a ramp and a machine
without it falls back to the default colormap."""
from __future__ import annotations

import pyqtgraph as pg

from dynamix.shell.settings import load_settings, settings_path, update_settings

_cache: tuple = (None, None, {})          # (settings path, its mtime, the ramps read from it)


def _library() -> dict:
    """The ramps in the settings file, read again only when the file changed."""
    global _cache
    path = settings_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = None
    if _cache[:2] != (path, mtime):
        _cache = (path, mtime, load_settings().ramps)
    return _cache[2]


def ramp_names() -> list:
    return sorted(_library())


def ramp_stops(name) -> "list | None":
    """The ramp ``name``'s ``[position, "#rrggbb"]`` stops, or None for a name not in the library."""
    return _library().get(name)


def ramp_colormap(name) -> "pg.ColorMap | None":
    stops = ramp_stops(name)
    if stops is None:
        return None
    return pg.ColorMap([p for p, _c in stops], [pg.mkColor(c) for _p, c in stops])


def builtin_colormap(name) -> bool:
    """Whether ``name`` is a colormap the app has already, pyqtgraph's or matplotlib's."""
    import matplotlib

    if name in matplotlib.colormaps:
        return True
    try:
        pg.colormap.get(name)
    except Exception:                                        # noqa: BLE001 -- any miss is "no"
        return False
    return True


def save_ramp(name: str, stops) -> None:
    """Keep ``stops`` (two or more ``(position in [0, 1], "#rrggbb")``) as the ramp ``name``."""
    global _cache
    name = str(name).strip()
    if not name:
        raise ValueError("a ramp needs a name")
    if builtin_colormap(name):
        raise ValueError(f"{name!r} is a built-in colormap: choose another name")
    rows = sorted([float(p), str(c).lower()] for p, c in stops)
    if len(rows) < 2:
        raise ValueError("a ramp needs at least two stops")
    update_settings(ramps={**load_settings().ramps, name: rows})
    _cache = (None, None, {})


def delete_ramp(name: str) -> None:
    global _cache
    ramps = dict(load_settings().ramps)
    ramps.pop(name, None)
    update_settings(ramps=ramps)
    _cache = (None, None, {})
