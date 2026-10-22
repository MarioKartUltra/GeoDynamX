# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""App-level settings. One file, one dataclass, no Qt.

The file lives in the platform user-data directory, NOT the repo and NOT the
project payload -- these are machine preferences, not measurements.
"""
from __future__ import annotations
import dataclasses, json, os, sys
from pathlib import Path

from dynamix.model.atomic_write import write_json_atomically

#: The three legal values of ``MainWindow._center_view`` --
#: restated here (not imported from ``main_window``, which imports Qt) so this module stays
#: Qt-free per its own module docstring.
_CENTER_VIEWS = ("raster", "map", "vector", "geo")


#: Fast Fork's defaults: a temporary dataset of its own holding the raster shown, no vectors.
FAST_FORK_DEFAULTS = {"nest": False, "temporary": True, "rasters": "shown", "vectors": False}


def _fast_fork(value) -> dict:
    """A stored Fast Fork entry when every field is valid, else the defaults."""
    if not isinstance(value, dict) or set(value) != set(FAST_FORK_DEFAULTS):
        return dict(FAST_FORK_DEFAULTS)
    if not all(isinstance(value[k], bool) for k in ("nest", "temporary", "vectors")) \
            or value["rasters"] not in ("shown", "all"):
        return dict(FAST_FORK_DEFAULTS)
    return dict(value)


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
    #: The theme applied at launch: a bundled name (``restrained_dark``, ``mid_grey``, ``light``) or a
    #: file stem under the user's themes folder (``dynamix.shell.theme.load_theme``).
    theme: str = "restrained_dark"
    #: A raster to open at launch when none is given on the command line (handy for testing). ``None`` or a path that no longer exists
    #: falls back to :func:`dynamix.shell.app._default_raster`'s demo/fixture choice. Set it by
    #: hand in settings.json or via ``update_settings(open_on_launch=...)``.
    open_on_launch: str | None = None
    #: Edge, in pixels, of the centred window a too-big GeoTIFF opens as (``opening.open_field``'s
    #: ``window_size``; 4096 was the hard-coded value). Smaller = faster WTMM while testing the
    #: shell; anything that is not a positive int
    #: falls back to 4096. Fresh opens only -- reopening a project keeps the window it was saved with.
    open_window_px: int = 4096
    #: Folders the footprint browser re-scans at every window open: raster headers
    #: only, milliseconds per folder. A folder that no longer exists is skipped, never an error.
    footprint_folders: list[str] = dataclasses.field(default_factory=list)
    #: The master compute-engine choice: "auto" = mlx on Apple Silicon, else FFTW3; "mlx" /
    #: "fftw" force one engine app-wide. Applied at startup (and on menu toggle) via
    #: ``dynamix.core.wtmm_backend.set_default_engine`` -- NEVER part of a cache key or recipe:
    #: the backends-agree-to-float32 law makes engine choice infrastructure, not physics.
    compute_engine: str = "auto"
    #: The startup FFT precision, 32 (the default, xsmurf's own single precision) or 64 (runs on
    #: FFTW3 -- mlx is single precision only): a newly placed tool's "FFT precision" knob starts
    #: at it, and the FFTs outside a tool's compute run at it. A placed tool runs at its own knob.
    #: Engine choices are "auto" (mlx on Apple Silicon, else FFTW3), "mlx", "fftw"; a file
    #: saved with a "numpy" choice loads as the CPU engine, "fftw". Applied at STARTUP
    #: through dynamix.core.fft_policy.configure (restart to apply).
    compute_precision: int = 32
    #: The
    #: ONE size, in pixels, above which a raster opens as the display-only whole-extent PICTURE
    #: (analysis then runs on ROIs, reading native pixels) instead of loading whole. 64M was
    #: ``opening.open_field``'s hard-coded ``max_pixels`` (~8000 x 8000; one float64 copy =
    #: 512 MB). Anything that is not a positive int falls back to 64M.
    open_max_pixels: int = 64_000_000
    #: Zero selects one quarter of physical RAM; a positive value is an explicit byte budget.
    cache_ram_bytes: int = 0
    #: The engine cache's bound on unpinned analysis results (``Cache.max_results``): every
    #: transform knob value mints new ones, and past it the least-recent are evicted; what a layer
    #: shows stays. Anything that is not a positive int falls back to 8.
    result_cache_entries: int = 8
    #: The level the under-sampling warning is judged against: η, the finest wavelet's energy beyond
    #: Nyquist (``dynamix.core.undersampling``). Applied at startup and from Settings > Under-sampling
    #: Warning Level; anything that is not a positive number falls back to 1e-3 (spec 19's level).
    eta_warn: float = 1e-3
    #: What Fast Fork (Y) writes without its dialog (:data:`FAST_FORK_DEFAULTS`): nested under the
    #: dataset or a row of its own, temporary or saved to a file, the raster shown or every raster,
    #: and the result's vectors. Set from Settings > Fast Fork…; an invalid entry reads as the defaults.
    fast_fork: dict = dataclasses.field(default_factory=lambda: dict(FAST_FORK_DEFAULTS))
    #: The folder every open, save and export dialog starts in and a saved fork is offered in
    #: (:func:`start_path`), set from Settings > Working Folder…; ``None``, or a folder that no
    #: longer exists, leaves each dialog where it starts without one.
    working_folder: str | None = None
    #: Where each floating window was left, ``{key: [x, y, w, h]}`` in screen pixels
    #: (``dynamix.shell.window_fit.remember_geometry``); a saved geometry that lands on no
    #: screen is ignored on restore.
    window_geometry: dict[str, list[int]] = dataclasses.field(default_factory=dict)

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


def themes_dir() -> Path:
    """The user's theme files (``dynamix.shell.theme.load_theme``): a ``themes`` folder beside the
    settings file."""
    return settings_path().parent / "themes"


def racks_dir() -> Path:
    """The user's rack files (``dynamix.model.racks``): a ``racks`` folder beside the settings
    file, so ``DYNAMIX_SETTINGS_PATH`` moves both."""
    return settings_path().parent / "racks"


def devices_dir() -> Path:
    """The user's device files (``dynamix.model.plugins.discover``): a ``devices`` folder beside
    the settings file, so ``DYNAMIX_SETTINGS_PATH`` moves it too."""
    return settings_path().parent / "devices"


def scratch_dir() -> Path:
    """Where a result too large for memory lives (a figure's memory-mapped signed CWT stack): a
    ``scratch`` folder beside the settings file, created on demand, emptied by
    :func:`sweep_scratch` at every start."""
    path = settings_path().parent / "scratch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sweep_scratch() -> None:
    """Remove every file in :func:`scratch_dir`: a result's file goes with its window, and this
    takes what a crash or a still-mapped file on Windows left behind."""
    for child in scratch_dir().iterdir():
        try:
            if child.is_dir() and child.name.startswith("cache-"):
                import psutil
                import shutil
                pid = child.name.removeprefix("cache-")
                if pid.isdigit() and not psutil.pid_exists(int(pid)):
                    shutil.rmtree(child)
            elif child.is_file():
                child.unlink()
        except OSError:
            pass


def cache_ram_budget(settings: Settings) -> int:
    """Resolve the automatic RAM tier independently of the stage store's 2 GB bound."""
    import psutil
    return settings.cache_ram_bytes or int(psutil.virtual_memory().total // 4)


def _cache_budget(data) -> int:
    if "cache_ram_bytes" in data:
        return _positive_int(data["cache_ram_bytes"], 0)
    legacy = _positive_int(data.get("output_cache_bytes"), 2_000_000_000)
    return 0 if legacy == 2_000_000_000 else legacy


def load_settings() -> Settings:
    try:
        d = json.loads(settings_path().read_text(encoding="utf-8"))
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
        compute_engine = "fftw"            # the CPU engine is FFTW3
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
        cache_ram_bytes=_cache_budget(d),
        result_cache_entries=_positive_int(d.get("result_cache_entries"), 8),
        eta_warn=_positive_float(d.get("eta_warn"), 1e-3),
        window_geometry=_dict_of_int_lists(d.get("window_geometry")),
        fast_fork=_fast_fork(d.get("fast_fork")),
        working_folder=d["working_folder"] if isinstance(d.get("working_folder"), str) else None,
        theme=d["theme"] if isinstance(d.get("theme"), str) and d["theme"] else "restrained_dark",
    )

def _positive_int(value, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


def _positive_float(value, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        return default
    return float(value)


def start_path(name: str = "") -> str:
    """Where a file dialog starts: ``name`` inside the working folder while one is set and exists,
    else ``name`` alone (the dialog's own starting place)."""
    folder = load_settings().working_folder
    if folder and Path(folder).expanduser().is_dir():
        return str(Path(folder).expanduser() / name) if name else str(Path(folder).expanduser())
    return name


def save_settings(s: Settings) -> None:
    """Replace the settings file whole, so an interrupted save keeps the previous settings."""
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomically(p, dataclasses.asdict(s))

def update_settings(**changes) -> Settings:
    """Read-modify-write: the only sanctioned single-field write. A bare
    ``save_settings(Settings(x=...))`` resets every OTHER field to its default -- the
    clobber the auto-run toggle shipped with while Settings had one field."""
    s = load_settings()
    s = dataclasses.replace(s, **changes)
    save_settings(s)
    return s
