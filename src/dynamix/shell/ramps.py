# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The user's own colour ramps: named lists of ``(position, "#rrggbb")`` stops kept in the settings
file (``Settings.ramps``), resolved to a pyqtgraph ``ColorMap`` wherever a colormap name is looked
up (``canvas._resolve_cmap``). They belong to the user, so a project names a ramp and a machine
without it falls back to the default colormap. A ramp flagged diverging (``Settings.diverging_ramps``)
puts its middle colour on the layer's centre value, as a built-in diverging colormap does."""
from __future__ import annotations

import pyqtgraph as pg

from dynamix.shell.settings import load_settings, settings_path, update_settings

#: (settings path, its mtime, the ramps read from it, the names flagged diverging)
_cache: tuple = (None, None, {}, frozenset())


def _read() -> tuple:
    """The ramps in the settings file and the names flagged diverging, read again only when the
    file changed."""
    global _cache
    path = settings_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = None
    if _cache[:2] != (path, mtime):
        s = load_settings()
        _cache = (path, mtime, s.ramps, frozenset(s.diverging_ramps))
    return _cache[2], _cache[3]


def _library() -> dict:
    return _read()[0]


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


def is_diverging(name) -> bool:
    """Whether the colormap ``name`` is diverging: one of the user's ramps flagged so, or a
    diverging built-in (:func:`dynamix.core.stretch.diverging_name`)."""
    from dynamix.core.stretch import diverging_name

    library, flagged = _read()
    return name in flagged if name in library else diverging_name(name)


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


def save_ramp(name: str, stops, diverging: bool = False) -> None:
    """Keep ``stops`` (two or more ``(position in [0, 1], "#rrggbb")``) as the ramp ``name``,
    flagged diverging or not."""
    global _cache
    name = str(name).strip()
    if not name:
        raise ValueError("a ramp needs a name")
    if builtin_colormap(name):
        raise ValueError(f"{name!r} is a built-in colormap: choose another name")
    rows = sorted([float(p), str(c).lower()] for p, c in stops)
    if len(rows) < 2:
        raise ValueError("a ramp needs at least two stops")
    s = load_settings()
    flagged = set(s.diverging_ramps) - {name} | ({name} if diverging else set())
    update_settings(ramps={**s.ramps, name: rows}, diverging_ramps=sorted(flagged))
    _cache = (None, None, {}, frozenset())


def delete_ramp(name: str) -> None:
    global _cache
    s = load_settings()
    ramps = dict(s.ramps)
    ramps.pop(name, None)
    update_settings(ramps=ramps, diverging_ramps=[n for n in s.diverging_ramps if n != name])
    _cache = (None, None, {}, frozenset())
