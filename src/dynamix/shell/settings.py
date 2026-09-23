# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""App-level settings. One file, one dataclass, no Qt.

The file lives in the platform user-data directory, NOT the repo and NOT the
project payload -- these are machine preferences, not measurements.
"""
from __future__ import annotations
import dataclasses, json, os, sys
from pathlib import Path

#: The three legal values of ``MainWindow._center_view`` --
#: restated here (not imported from ``main_window``, which imports Qt) so this module stays
#: Qt-free per its own module docstring.
_CENTER_VIEWS = ("raster", "vector", "geo")


@dataclasses.dataclass
class Settings:
    auto_run_wtmm: bool = False
    splitter_sizes: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    view_options: dict[str, object] = dataclasses.field(default_factory=dict)
    #: The center-zone view a window should restore on its next
    #: open -- ``MainWindow.load_field``'s own restore-once call reads this back after the
    #: FIRST layer lands (there is nothing for the vector/geo admission logic to run over any
    #: earlier than that). Round-tripped by :func:`update_settings` exactly like every other
    #: field here.
    center_view: str = "raster"
    #: A raster to open at launch when none is given on the command line (handy for testing). ``None`` or a path that no longer exists
    #: falls back to :func:`dynamix.shell.app._default_raster`'s demo/fixture choice. Set it by
    #: hand in settings.json or via ``update_settings(open_on_launch=...)``.
    open_on_launch: str | None = None
    #: Edge, in pixels, of the centred window a too-big GeoTIFF opens as (``opening.open_field``'s
    #: ``window_size``; 4096 was the hard-coded value). Smaller = faster WTMM while testing the
    #: shell; anything that is not a positive int
    #: falls back to 4096. Fresh opens only -- reopening a project keeps the window it was saved with.
    open_window_px: int = 4096
    #: Folders the footprint browser re-scans at every window open (2026-08-28): raster headers
    #: only, milliseconds per folder. A folder that no longer exists is skipped, never an error.
    footprint_folders: list[str] = dataclasses.field(default_factory=list)
    #: The master compute-engine choice: "auto" = mlx when importable, else
    #: the numpy/FFT fallback (the auto-detect ``cwt2d`` always had); "mlx" / "numpy" force one
    #: engine app-wide. Applied at startup (and on menu toggle) via
    #: ``dynamix.core.wtmm_backend.set_default_engine`` -- NEVER part of a cache key or recipe:
    #: the backends-agree-to-float32 law makes engine choice infrastructure, not physics.
    compute_engine: str = "auto"
    #: 2026-09-22: FFT precision for EVERY FFT in the app, 32 (the default,
    #: xsmurf's own single precision) or 64 (runs on FFTW3 -- mlx is single precision only).
    #: Engine choices are now "auto" (mlx on Apple Silicon, else FFTW3), "mlx", "fftw"; a file
    #: saved with the old "numpy" choice loads as the CPU engine, "fftw". Applied at STARTUP
    #: through dynamix.core.fft_policy.configure (restart to apply).
    compute_precision: int = 32
    #: The
    #: ONE size, in pixels, above which a raster opens as the display-only whole-extent PICTURE
    #: (analysis then runs on ROIs, reading native pixels) instead of loading whole. 64M was
    #: ``opening.open_field``'s hard-coded ``max_pixels`` (~8000 x 8000; one float64 copy =
    #: 512 MB). Anything that is not a positive int falls back to 64M.
    open_max_pixels: int = 64_000_000

def settings_path() -> Path:
    env = os.environ.get("DYNAMIX_SETTINGS_PATH")
    if env:
        return Path(env)
    # GeoDynamix's own folder, never DynamiX's: both apps can live on one machine.
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path.home() / "Library" / "Application Support"
    return base / "GeoDynamix" / "settings.json"

def load_settings() -> Settings:
    try:
        d = json.loads(settings_path().read_text())
    except (OSError, ValueError):
        return Settings()
    def _dict_of_int_lists(v):
        if not isinstance(v, dict):
            return {}
        try:
            return {str(k): [int(x) for x in vs] for k, vs in v.items()}
        except (TypeError, ValueError):
            return {}

    center_view = d.get("center_view")
    if center_view not in _CENTER_VIEWS:
        center_view = "raster"
    compute_engine = d.get("compute_engine")
    if compute_engine == "numpy":
        compute_engine = "fftw"            # the CPU engine is FFTW3 now (2026-09-22)
    if compute_engine not in ("auto", "mlx", "fftw"):
        compute_engine = "auto"
    compute_precision = d.get("compute_precision")
    if compute_precision not in (32, 64):
        compute_precision = 32
    return Settings(
        auto_run_wtmm=bool(d.get("auto_run_wtmm", False)),
        splitter_sizes=_dict_of_int_lists(d.get("splitter_sizes")),
        view_options=d.get("view_options") if isinstance(d.get("view_options"), dict) else {},
        center_view=center_view,
        open_on_launch=d["open_on_launch"] if isinstance(d.get("open_on_launch"), str) else None,
        open_window_px=_positive_int(d.get("open_window_px"), 4096),
        footprint_folders=[str(x) for x in d.get("footprint_folders", [])]
        if isinstance(d.get("footprint_folders"), list) else [],
        compute_engine=compute_engine,
        open_max_pixels=_positive_int(d.get("open_max_pixels"), 64_000_000),
        compute_precision=compute_precision,
    )

def _positive_int(value, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


def save_settings(s: Settings) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dataclasses.asdict(s), indent=2))

def update_settings(**changes) -> Settings:
    """Read-modify-write: the only sanctioned single-field write. A bare
    ``save_settings(Settings(x=...))`` resets every OTHER field to its default -- the
    clobber the auto-run toggle shipped with while Settings had one field."""
    s = load_settings()
    s = dataclasses.replace(s, **changes)
    save_settings(s)
    return s
