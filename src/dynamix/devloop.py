# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dynamix.devloop: the state-preserving reload harness.

    python -m dynamix.devloop [field.npz] [--ipython]

The cost of restarting the shell while iterating on it is not startup time -- it is throwing away
the loaded raster, the computed WTMM scale stack, and the view you had navigated to. This module
keeps the interpreter alive across an edit instead: a module-scope :data:`SESSION` dict holds the
loaded field, the transform :class:`~dynamix.engine.Cache`, the :class:`~dynamix.model.project.
Project` and the current chain -- everything a rebuilt window needs to look exactly like the one
it replaced. ``SESSION`` lives here, in ``dynamix.devloop``, which is never itself reloaded, so it
survives every reload cycle by construction.

**Reload set: ``dynamix.shell.*`` only, not core/engine/model.** A stdlib ``QTimer`` (wrapped in
:class:`DevLoop`, below) scans ``src/dynamix/shell/**/*.py`` for mtime changes every 0.5 s
(watchfiles was considered and rejected for slice 1 -- simplicity first, and stdlib polling is
plenty fast for a handful of files). On a change, the affected modules are ``importlib.reload``-ed
in leaf-to-window order (``theme, knobs, knob_widgets, worker, chain_strip, transport, canvas,
main_window`` -- :data:`_MODULE_ORDER`, fixed because the import graph among these eight is itself
fixed), the old window is closed, and a new one is built from the just-reloaded ``main_window``
module and handed ``SESSION``. The analysis core (``dynamix.core``, ``dynamix.engine``,
``dynamix.model``) is deliberately NOT in the watch set: its state is exactly what ``SESSION``
already carries forward (the cache, the field, the chain), so reloading it would buy nothing while
adding a second, much larger, dependency graph to keep straight. This is the smallest loop that is
still honest about what it does: edit the shell, see it rebuilt; edit the core, restart.

A change detected while the window is mid-compute is DEFERRED, not dropped: rebuilding then would
leave two live windows sharing one ``Cache`` that is not built for two threads to touch it at
once -- the old window's ``ResolveWorker`` still writing into it while the new window immediately
reads from and writes to the same instance. ``DevLoop.poll()`` remembers the change (``pending``)
and retries on the next tick, so a reload that lands mid-scale-sweep is simply a little late, never
a race.

**Why the cache re-attachment is what makes this fast.** ``MainWindow.load_field`` always runs
its first resolve on :class:`~dynamix.shell.worker.ResolveWorker` (the ``main_window.py``
docstring: "A TRANSFORM param moved -> hand the whole resolve to the worker"). Handing the
rebuilt window the SAME ``Cache`` instance the old one used means that resolve is a guaranteed
cache HIT -- the worker still runs on its own thread, but it returns as fast as a queued-signal
round trip rather than recomputing a WTMM stack. Reusing the same ``Project`` matters for the same
reason: ``Project.add_source`` is idempotent by path, so the rebuilt layer gets the identical
``source_id`` and therefore the identical cache key.

**Why no ``synchronous=True`` mode.** EQSelect's ``_WtmmWorker`` carries an inline-execution mode
that runs the worker's finish/fail slots directly, without a real ``QThread`` or event loop, so the
result path is testable headlessly. DynamiX's dev loop does not need it, for a structural reason:
this harness never hot-swaps code inside a running compute. A shell edit tears the window down
and rebuilds it; it never reaches into ``ResolveWorker`` mid-flight. There is therefore no REPL
call site here that needs a synchronous ``resolve()`` -- ``--ipython`` runs under ``%gui qt``,
which keeps a REAL Qt event loop pumping in the background, so a queued worker signal is delivered
exactly as it would be under ``app.exec()``; there is nothing for an inline mode to unblock. And
headless testability, that mode's actual motivation, is already solved a different way in this
codebase: ``app.py``'s ``--render`` mode pumps ``processEvents()`` until the window reports itself
idle, which exercises the identical queued-signal path a real user's session takes -- the tests in
``tests/test_shell_window.py`` do the same via ``qtbot.waitSignal``. Adding a second, parallel
execution mode to ``ResolveWorker`` to serve a need this harness does not have would be exactly
the kind of speculative flexibility the project's simplicity rule warns against. If a future slice
wants a REPL that calls ``resolve()`` synchronously without going through the worker at all, that
is a new, separable feature request -- not a gap in this one.

**``--ipython``** embeds an ``IPython.terminal.embed.InteractiveShellEmbed`` with ``%gui qt`` (so
the Qt timer above keeps ticking while the REPL is waiting on input) and ``%autoreload 2``
(function-BODY edits across the whole interpreter, including outside the shell, take effect live
without a rebuild). ``SESSION`` and a ``window()`` accessor are dropped
into the REPL's namespace; ``window()`` is a callable rather than a plain binding because a rebuild
replaces the window object out from under any name captured before it happened.
"""
from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

#: Session state that survives every reload: the loaded field, the transform cache, the project
#: (and through it, the current layer/chain), and the chain steps a rebuilt window should open
#: with. Nothing here is a Qt object, so nothing here can ever be left dangling by a reload.
SESSION: dict = {
    "path": None,      # the source identity load_field/open_path was given
    "field": None,      # the loaded RasterField (or bare array)
    "cache": None,      # dynamix.engine.Cache -- re-attaching this is what makes a rebuild instant
    "project": None,    # dynamix.model.project.Project -- carries the layer(s) and their chains
    "layer": None,       # the current dynamix.model.layer.Layer, for inspection (not rebuild input)
    "steps": None,      # (device_name, params) per chain step -- what a rebuilt window opens with
    "view": None,       # [[x0, x1], [y0, y1]] -- the canvas camera, so a rebuild lands where you were
    "bands": None,      # the active dataset's band list (an imported group is file + bands)
    "fields": None,     # layer_id -> field for EVERY row, so a rebuild keeps every dataset
}

_SHELL_DIR = Path(__file__).resolve().parent / "shell"

#: Reload order: leaf modules before their dependents, ending at main_window. Fixed rather than
#: computed, because the import graph among these eight modules is itself fixed (grep-verified
#: against every `from dynamix.shell...` import in the package) -- a runtime topological sort
#: would be one more moving part standing in for a constant.
_MODULE_ORDER = (
    "theme", "knobs", "knob_widgets", "worker", "opening", "point_import", "units", "settings",
    "browser", "chain_strip", "workflow_zone", "transport", "canvas", "inspector", "roi_panel",
    "right_panel", "layer_panel", "arrangement_facade", "view_dialog", "topology_panel",
    "skeleton_dialog", "transect_panel", "profile_dialog", "multifractal_window",
    "spectrum_window", "surface_dialog", "levels_dialog", "main_window",
)

#: Direct dynamix.shell.* dependencies of each reloadable module (self excluded). Used to widen a
#: changed set to its dependents -- reloading chain_strip.py alone after knob_widgets.py changed
#: would leave chain_strip's DeviceStrip closing over the STALE make_control function.
_DEPENDS_ON = {
    "theme": (),
    "knobs": (),
    "knob_widgets": ("knobs",),
    "worker": (),
    "opening": (),
    # CSV point-catalogue import -- self-contained (only
    # ``dynamix.core.pointset`` and PySide6 itself, no other ``dynamix.shell.*`` module).
    "point_import": (),
    "units": (),
    "settings": (),
    "browser": (),
    "chain_strip": ("knobs", "knob_widgets", "units"),
    "workflow_zone": ("knobs", "knob_widgets", "browser"),
    "transport": (),
    "canvas": ("theme", "units"),
    # The per-source floating inspector HOSTS the existing Canvas and the
    # existing Transport (that reuse is the point of it -- see inspector.py's own docstring), so
    # both are real module-level imports and both already sit earlier in _MODULE_ORDER.
    "inspector": ("canvas", "transport"),
    "roi_panel": ("knobs", "knob_widgets"),
    # "canvas": right_panel.py's swatch defaults are derived from
    # canvas.py's own HCHAIN_COLOR/VTRAIL_COLOR/EXTREMA_COLOR constants (see right_panel.py's own
    # _hex docstring) -- a real module-level import, not just a docstring mention. "canvas" is
    # already earlier in _MODULE_ORDER, so this adds no reordering, only a dependency edge.
    "right_panel": ("knobs", "knob_widgets", "canvas"),
    "layer_panel": (),
    # "arrangement" is not itself a _MODULE_ORDER stem (the subpackage this facade stands in for
    # -- see arrangement_facade.py's own docstring); listed here only so the AST import-coverage
    # test (test_shell_boundaries.py) recognises this facade's one real import.
    "arrangement_facade": ("arrangement",),
    # "arrangement" here is the SAME bookkeeping-only entry as
    # arrangement_facade's own comment above describes -- main_window.py now imports MaskRow
    # directly from dynamix.shell.arrangement.mask_row (module-level: the row is built eagerly,
    # unconditionally, not gated behind a Tab press the way the rest of the subpackage is), which
    # is a SEPARATE real import from the arrangement_facade one already listed just below. Neither
    # entry widens what devloop actually reloads (see modules_to_reload's own docstring: "arrangement"
    # is not itself a _MODULE_ORDER stem), so a mask_row.py-only edit during a dev-loop session
    # still needs main_window.py (or arrangement_facade.py) touched to pick it up -- the same known
    # limitation arrangement_facade.py's own module docstring already documents for view.py et al.
    #
    # View_dialog.py imports ModeRow directly from
    # dynamix.shell.arrangement.mode_row (same reasoning as mask_row/group_palette above -- pure
    # Qt, no pyvista of its own), which is what earns "arrangement" a place in ITS OWN tuple too,
    # below.
    "view_dialog": ("theme", "knobs", "knob_widgets", "settings", "arrangement"),
    # The "Topology" right-panel section -- self-contained (only
    # ``dynamix.topology.codes``, no other ``dynamix.shell.*`` module), same shape as "transport"/
    # "layer_panel" above.
    "topology_panel": (),
    # The "Skeleton plot…" dialog -- self-contained (only
    # ``dynamix.core.chain_stats``, no other ``dynamix.shell.*`` module -- see
    # ``skeleton_dialog.py``'s own module docstring, "Highlight color" section, for why it does
    # NOT import ``theme``), same shape as "topology_panel"/"transport" above.
    "skeleton_dialog": (),
    # The transect list -- self-contained (only
    # ``dynamix.model.project`` for the ``TransectRecord`` dataclass, no other ``dynamix.shell.*``
    # module), same shape as "topology_panel"/"skeleton_dialog" above.
    "transect_panel": (),
    # The transect profile dialog -- self-contained (only
    # ``dynamix.core.transect``, no ``dynamix.shell.*`` module -- see the module's own "Accent
    # color" docstring section for why it does NOT import ``theme``, the identical reasoning
    # skeleton_dialog.py already gives for its own highlight color), same shape as
    # "topology_panel"/"skeleton_dialog"/"transect_panel" above.
    "profile_dialog": (),
    # The interactive multifractal-spectrum window -- imports ``theme``
    # for its background/inks (the ``canvas.py`` precedent, unlike skeleton_dialog/profile_dialog
    # whose single-color needs didn't earn the edge) plus ``dynamix.core.spectra``; no other
    # ``dynamix.shell.*`` module.
    "multifractal_window": ("theme",),
    # The singularity-spectrum construction window -- imports theme (same precedent)
    # plus multifractal_window (Q_COLD/Q_WARM/TABLE_CHOICES shared by identity) and
    # core.spectra/core.chain_groups/core.microcanonical.
    "spectrum_window": ("theme", "multifractal_window"),
    # The 3-D surface-source dialog -- pure Qt, self-contained (the
    # topology_panel/transect_panel shape).
    "surface_dialog": (),
    # The density-slice editor -- imports theme (canvas.py precedent) + core.stretch.
    "levels_dialog": ("theme",),
    # "inspector" joins the list because main_window.py now imports
    # InspectorWindow at module level (it opens/closes them and fans ``resolved`` out to them).
    # Purely additive -- nothing dropped, nothing reordered, and "inspector" already sits earlier
    # in _MODULE_ORDER -- and FORCED by test_shell_boundaries.py's AST import-coverage guard,
    # which reads a missing edge as the stale-dependency bug it exists to catch.
    "main_window": ("theme", "canvas", "workflow_zone", "roi_panel", "right_panel", "layer_panel",
                    "transport", "worker", "opening", "point_import", "units", "settings",
                    "browser", "knobs", "knob_widgets", "arrangement_facade", "arrangement",
                    "view_dialog", "topology_panel", "skeleton_dialog", "transect_panel",
                    "profile_dialog", "multifractal_window", "spectrum_window", "surface_dialog",
                    "levels_dialog", "inspector"),
}

USAGE = "python -m dynamix.devloop [field.npz] [--ipython]"


def _scan_mtimes() -> dict[str, float]:
    """module stem -> mtime for every ``.py`` file under ``src/dynamix/shell/``.

    Recursive glob per the design, though the package is flat today -- a future subpackage does
    not silently fall out of the watch. Stems outside :data:`_MODULE_ORDER` (``app.py``,
    ``demo.py``, ``__init__.py``) are tracked too but never trigger a reload; see
    :func:`modules_to_reload`.
    """
    return {p.stem: p.stat().st_mtime for p in _SHELL_DIR.rglob("*.py")}


def modules_to_reload(changed_stems: set[str]) -> list[str]:
    """``changed_stems`` (module stems whose mtime moved) -> dotted module names to
    ``importlib.reload``, widened to dependents and returned in leaf-to-window order.

    Pure and side-effect-free on purpose: it is the one piece of the reload harness that can be
    exercised by a test without a Qt event loop or a running window.
    """
    affected = set(changed_stems) & set(_MODULE_ORDER)
    grown = True
    while grown:
        grown = False
        for name, deps in _DEPENDS_ON.items():
            if name not in affected and affected & set(deps):
                affected.add(name)
                grown = True
    return [f"dynamix.shell.{name}" for name in _MODULE_ORDER if name in affected]


def _reload(names: list[str]) -> None:
    for name in names:
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)
        else:
            importlib.import_module(name)


def _capture(window) -> None:
    """Copy state out of the live window into SESSION before it is closed.

    The cache and project travel by REFERENCE -- SESSION never copies them -- so the rebuilt
    window's first resolve lands on an already-warm cache (see the module docstring).

    ``SESSION["path"]`` is re-derived from the window's OWN project/layer, not left as whatever
    the CLI arg (or the last capture) set it to. It used to be set once, from argv, and never
    refreshed -- so a raster opened later through the GUI's Open button (no CLI path at all) left
    ``path`` stuck at ``None``, and ``_build_window``'s ``field is not None and path is not None``
    guard then skipped ``load_field`` entirely on the next rebuild: the window came back with no
    raster, silently. Worse, opening a SECOND file over a FIRST one (CLI path A, GUI-opened file
    B) would rebuild via ``load_field(field_B, path_A)`` -- and ``Project.add_source``'s path-dedup
    can then surface file A's cached result under file B's display: silent wrong-data, the one
    error class this whole design exists to rule out. The window's ``project.sources`` registry is
    the single source of truth for "what path does the CURRENT layer actually point at"; reading
    it back here keeps SESSION honest instead of trusting a value set once and never revisited.

    The VIEW travels too, and that is not a nicety. This module's own docstring promises a rebuild
    that "looks exactly like the one it replaced", and the spec's acceptance criterion 4 asks for
    the same raster, stack, layer AND view -- but ``load_field`` ends in ``autoRange()``, so every
    rebuild used to snap back out to the whole raster. Zoomed in on one lineament, a one-character
    edit threw away the navigation, which on a big DEM is most of what you had.
    """
    SESSION["view"] = [list(axis) for axis in window.canvas.view.viewRange()]
    SESSION["field"] = window.field
    SESSION["cache"] = window.cache
    SESSION["project"] = window.project
    SESSION["layer"] = window.layer
    SESSION["fields"] = dict(window._fields)
    if window.layer is not None:
        SESSION["steps"] = tuple((ref.device, dict(ref.params))
                                 for ref in window.layer.chain.steps)
        source = window.project.sources.get(window.layer.source_id)
        if source is not None:
            SESSION["path"] = source.path
            SESSION["bands"] = list(source.bands) if source.bands else None
        # else: no matching source (should not happen -- add_layer requires one) -- keep whatever
        # SESSION["path"] already held rather than clobbering it with a guess.


def _restore_view(window) -> None:
    """Put the captured camera back, AFTER ``load_field`` -- which ends in ``autoRange()`` and
    would otherwise overwrite anything set before it.

    A separate function rather than four lines inline for the same reason ``DevLoop`` is a class:
    it is the part of the rebuild a test can drive against a bare ``Canvas``, without a window, a
    field, or an event loop. ``padding=0`` because the captured range is already the padded one --
    padding it again would zoom out a little further on every reload.
    """
    if SESSION["view"] is None:
        return
    (x0, x1), (y0, y1) = SESSION["view"]
    window.canvas.view.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)


def _build_window(app):
    """(Re)build the window from whatever is CURRENTLY in ``sys.modules``.

    Never from a name bound at this module's own import time -- after the first reload that name
    would still point at the pre-reload class object. ``importlib.import_module`` returns the
    live module (the just-reloaded one, if a reload just happened), so re-fetching attributes off
    it here always sees the latest code.
    """
    main_window_mod = importlib.import_module("dynamix.shell.main_window")
    theme_mod = importlib.import_module("dynamix.shell.theme")

    steps = main_window_mod.DEMO_CHAIN if SESSION["steps"] is None else SESSION["steps"]
    window = main_window_mod.MainWindow(steps=steps)
    theme_mod.apply_theme(app, theme_mod.RESTRAINED_DARK)

    if SESSION["cache"] is not None:
        window.cache = SESSION["cache"]
    if SESSION["project"] is not None:
        window.project = SESSION["project"]

    window.resize(1400, 860)
    window.show()

    if SESSION["field"] is not None and SESSION["path"] is not None:
        window.load_field(SESSION["field"], SESSION["path"], bands=SESSION.get("bands"))
        _restore_rows(window, SESSION.get("fields") or {}, SESSION.get("layer"))
        _restore_view(window)
    return window


def _restore_rows(window, fields: dict, layer) -> None:
    """Give every restored row its OWN field back, and rebuild the rows of every OTHER
    dataset too: ``load_field``'s adoption rebuilds only the active dataset's family, all on
    the one session field -- with several datasets loaded, a rebuild used to drop the rest
    from the panel and draw the active field under every row. Re-selects the captured layer."""
    for lay in window.project.layers:
        field = fields.get(lay.layer_id)
        if field is None:
            continue
        if lay.layer_id in window._layer_by_id:
            window._fields[lay.layer_id] = field
        else:
            window.add_layer_row(lay, field)
    if layer is not None and layer.layer_id in window._layer_by_id:
        window.layer_list.select_layer(layer.layer_id)


def _open(window, path: str) -> None:
    SESSION["path"] = path
    window.open_path(path)
    SESSION["field"] = window.field


class DevLoop:
    """Owns the poll loop's mutable state: the last-seen mtimes, the current window, and any
    changed modules a compute in flight has forced ``poll()`` to defer.

    Pulled out of ``main()`` into its own class for two reasons: a ``QTimer`` needs a bound method
    to call every tick, and a test needs something it can drive directly without a Qt event loop
    or a real window (see ``tests/test_shell_boundaries.py``'s ``test_poll_defers_a_reload_while_
    the_window_is_computing``).
    """

    def __init__(self, app, window):
        self.app = app
        self.window = window
        self.mtimes = _scan_mtimes()
        self.pending: set[str] = set()

    def poll(self) -> None:
        """One 0.5 s tick: scan, fold any change into ``pending``, and reload -- UNLESS the
        current window is mid-compute.

        Reloading mid-compute would leave two live windows sharing one ``Cache`` that is not
        built to be touched from two places at once: the old window's ``ResolveWorker`` thread is
        still writing into it while the freshly rebuilt window immediately starts reading from and
        writing to the very same instance. Deferring costs nothing -- ``pending`` remembers the
        change across ticks, so nothing gets lost, and the very next non-computing tick reloads
        exactly what would have reloaded immediately if the window had been idle.
        """
        current = _scan_mtimes()
        changed = {stem for stem, mtime in current.items() if mtime != self.mtimes.get(stem)}
        self.mtimes = current
        self.pending |= changed
        if not self.pending:
            return
        if self.window.is_computing:
            return          # deferred; self.pending carries the change to the next tick
        names = modules_to_reload(self.pending)
        self.pending = set()
        if not names:
            return
        print(f"[devloop] reloading: {', '.join(names)}")
        _capture(self.window)
        _reload(names)
        self.window.close()
        self.window = _build_window(self.app)
        print("[devloop] rebuilt")


def _run_ipython(app, loop: "DevLoop") -> int:
    try:
        import IPython  # noqa: F401
    except ImportError:
        print("devloop: --ipython needs IPython in this environment (it ships with the xsmurf "
              "env; `pip install ipython` elsewhere). Run without --ipython, or install it and "
              "retry.", file=sys.stderr)
        return 2
    from IPython.terminal.embed import InteractiveShellEmbed

    shell = InteractiveShellEmbed(
        banner1="DynamiX devloop -- SESSION dict and window() are in scope. Edits under "
                "src/dynamix/shell/ rebuild the window; %autoreload 2 patches everything else "
                "live. Ctrl-D to quit.")
    shell.enable_gui("qt")
    shell.extension_manager.load_extension("autoreload")
    shell.run_line_magic("autoreload", "2")
    shell.user_ns.update(SESSION=SESSION, app=app, window=lambda: loop.window)
    shell()
    return 0


def main(argv=None) -> int:
    # A VTK/Cocoa segfault otherwise dies with no Python frame in the crash report;
    # faulthandler prints the Python stack to stderr first.
    import faulthandler
    try:
        faulthandler.enable()
    except (ValueError, OSError, AttributeError, io.UnsupportedOperation):
        pass        # a captured stderr (pytest) has no fileno; nothing to do
    argv = list(sys.argv[1:] if argv is None else argv)
    use_ipython = "--ipython" in argv
    if use_ipython:
        argv.remove("--ipython")
    if len(argv) > 1:
        print(f"usage: {USAGE}", file=sys.stderr)
        return 2
    path = argv[0] if argv else None
    if path is None:
        # Same default the app icon uses (Settings.open_on_launch, else demo/fixture) -- a
        # devloop that opens on the raster under test saves a File > Open per rebuild.
        from dynamix.shell.app import _default_raster
        path = _default_raster()

    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _build_window(app)
    if path:
        try:
            _open(window, path)
        except Exception as exc:
            # A bad path must not kill the loop (a rasterio traceback would otherwise take the
            # whole launch down) -- the window is already up; say what failed and let
            # File > Open do its job.
            print(f"devloop: could not open {path!r}: {exc}", file=sys.stderr)

    loop = DevLoop(app, window)
    timer = QtCore.QTimer()
    timer.timeout.connect(loop.poll)
    timer.start(500)
    loop.timer = timer          # keep it alive -- an unreferenced local QTimer can be GC'd

    if use_ipython:
        return _run_ipython(app, loop)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
