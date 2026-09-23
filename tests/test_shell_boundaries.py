# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Shell discipline: The Theme Rule and the no-docks rule, AST/grep-enforced."""
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SHELL = REPO / "src" / "dynamix" / "shell"
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
FAMILY = re.compile(r"\b(Menlo|Helvetica|Monaco|Courier|Arial|Inter)\b")


def test_theme_rule_no_literal_colors_or_fonts_outside_theme():
    """``rglob``, not ``glob``: the package is flat today, and a rule that silently stops applying
    the moment someone adds ``shell/widgets/`` is not a rule. Same reasoning devloop's own file
    scan already used."""
    for path in sorted(SHELL.rglob("*.py")):
        if path.name in ("theme.py", "demo.py"):      # demo.py predates the rule; grandfathered
            continue
        text = path.read_text()
        assert not HEX.search(text), f"{path.name} holds a literal color"
        assert not FAMILY.search(text), f"{path.name} holds a font family"


def test_no_docks_ever():
    for path in sorted(SHELL.rglob("*.py")):
        assert "QDockWidget" not in path.read_text(), f"{path.name} uses a dock"


def test_vtrail_color_is_the_theme_accent():
    """``VTRAIL_COLOR`` is a Theme-Rule carve-out (a chain's color is data identity, not chrome),
    but it was CHOSEN to be the amber accent, and nothing said so. Re-tinting the theme would
    have left the trails on the old accent, silently -- a palette that no longer matches the app
    it belongs to. Pin the relationship so a future accent change has to be a decision."""
    from dynamix.shell.canvas import VTRAIL_COLOR
    from dynamix.shell.theme import RESTRAINED_DARK

    digits = RESTRAINED_DARK.amber.lstrip("#")
    assert len(digits) == 6
    assert VTRAIL_COLOR == tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))


def test_roi_bounds_color_is_the_theme_accent():
    """Sibling of the VTRAIL test above, same convention and same reason. ``ROI_BOUNDS_COLOR`` is
    a Theme-Rule carve-out (which region a result was measured over is data identity, not chrome)
    but it was CHOSEN to be the amber accent -- the spec grants ROI selection the One-Accent
    exception explicitly -- and nothing else would say so. It is deliberately its OWN constant
    rather than a reuse of ``VTRAIL_COLOR``: they happen to share a value, and re-tinting either
    must not silently re-tint the other."""
    from dynamix.shell.canvas import ROI_BOUNDS_COLOR
    from dynamix.shell.theme import RESTRAINED_DARK

    digits = RESTRAINED_DARK.amber.lstrip("#")
    assert len(digits) == 6
    assert ROI_BOUNDS_COLOR == tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))


def test_seam_color_is_the_theme_seam_role():
    """Sibling of the VTRAIL/ROI_BOUNDS tests above. ``SEAM_COLOR`` is a Theme-Rule carve-out (a
    flagged chain's color is data identity, not chrome) but it was CHOSEN to match the theme's own
    ``seam`` role, and nothing else would say so. Own constant, not a reuse: they happen to share a
    value, and re-tinting either must not silently re-tint the other."""
    from dynamix.shell.canvas import SEAM_COLOR
    from dynamix.shell.theme import RESTRAINED_DARK

    digits = RESTRAINED_DARK.seam.lstrip("#")
    assert len(digits) == 6
    assert SEAM_COLOR == tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))


def test_devloop_is_the_only_shell_importer_outside_shell():
    import ast
    src = REPO / "src" / "dynamix"
    for path in sorted(src.rglob("*.py")):
        if src / "shell" in path.parents or path.name == "devloop.py":
            continue
        imported = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add((node.module or "").split(".")[0])
        assert not (imported & {"PySide6", "pyqtgraph", "pyvista"}), path


# ----------------------------------------------------------------------------------------------
# devloop reload machinery (the fallback for the manual, non-automated acceptance criterion:
# "devloop edit -> rebuild with state". The full GUI loop is recorded as manually verified in the
# commit message; these exercise the module-selection logic underneath it headlessly, without a
# Qt event loop or a running window, so a regression here is still caught by the suite.)


def test_modules_to_reload_widens_a_leaf_change_to_its_dependents():
    import dynamix.devloop as devloop

    # theme.py has no dynamix.shell dependencies of its own, but canvas.py and view_dialog.py
    #both import it directly, and right_panel.py (shell-
    # parity plan) imports canvas.py directly (for its swatch defaults) -- so a theme.py change
    # now widens transitively onto right_panel too, and main_window.py imports all three of those,
    # all reloaded in leaf-to-window order.
    #
    # inspector.py joins that widening for the same reason right_panel did:
    # the floating inspector HOSTS the existing Canvas, so a theme edit that changes what a canvas
    # looks like must rebuild the inspector's copy of it too -- an inspector left closed over a
    # stale theme is exactly the "edits that never reach the screen" failure this seam exists to
    # prevent. The expected list is retargeted to that new visible consequence, not shortened.
    #
    # multifractal_window.py joins it too: its plot background/inks
    # resolve through theme directly (the canvas.py precedent), so a theme edit must rebuild it.
    assert devloop.modules_to_reload({"theme"}) == [
        "dynamix.shell.theme", "dynamix.shell.canvas", "dynamix.shell.inspector",
        "dynamix.shell.right_panel", "dynamix.shell.view_dialog",
        "dynamix.shell.multifractal_window", "dynamix.shell.spectrum_window",
        "dynamix.shell.levels_dialog", "dynamix.shell.main_window",
    ]


def test_modules_to_reload_is_empty_for_a_file_outside_the_reload_set():
    import dynamix.devloop as devloop

    # app.py and demo.py are watched (scanned) but are not part of the dependency graph
    # main_window.py is built from, so a touch to either is correctly a no-op.
    assert devloop.modules_to_reload({"app"}) == []
    assert devloop.modules_to_reload(set()) == []


def test_devloop_scan_detects_a_touched_file_and_resolves_its_dependents():
    """The scan/reload seam, exercised against the REAL shell package: touch theme.py's mtime,
    ask devloop what changed, and confirm it decides to reload theme.py plus the modules that
    import it (directly or transitively) -- the exact mechanism the QTimer callback drives.
    """
    import os

    import dynamix.devloop as devloop

    theme_path = devloop._SHELL_DIR / "theme.py"
    original = theme_path.stat().st_mtime
    before = devloop._scan_mtimes()
    try:
        os.utime(theme_path, (original + 5.0, original + 5.0))
        after = devloop._scan_mtimes()
        changed = {stem for stem, mtime in after.items() if mtime != before.get(stem)}
        assert "theme" in changed
        assert devloop.modules_to_reload(changed) == [
            "dynamix.shell.theme", "dynamix.shell.canvas", "dynamix.shell.inspector",
            "dynamix.shell.right_panel", "dynamix.shell.view_dialog",
            "dynamix.shell.multifractal_window", "dynamix.shell.spectrum_window",
            "dynamix.shell.levels_dialog", "dynamix.shell.main_window",
        ]
    finally:
        os.utime(theme_path, (original, original))


def test_devloop_module_order_covers_every_shell_module():
    """_MODULE_ORDER is hand-maintained -- it IS the reload harness's leaf-to-window order, so it
    can't be derived automatically. A new shell module that isn't added to it would be silently
    excluded from the reload set: no error, just a file whose edits never rebuild the window.
    Guard it here instead of finding out by staring at an unchanged screen."""
    import dynamix.devloop as devloop

    # app.py: the pre-devloop launcher; not part of the import graph main_window.py is
    # built from, so devloop has no reason to reload it. demo.py: predates devloop and has its own
    # standalone headless entry point. __init__.py: the package marker, not a reloadable module.
    #
    # TOP-LEVEL only, and deliberately so -- unlike the Theme Rule and no-docks scans above, which
    # went recursive. _MODULE_ORDER is a flat leaf-to-window list of module STEMS; a submodule in
    # a future `shell/widgets/` would collide on stem and has no place in that ordering anyway
    # (`modules_to_reload` builds `dynamix.shell.<stem>`). The filter is written out rather than
    # left as a bare `glob` so that this exclusion reads as a decision, not an oversight.
    excluded = {"app", "demo", "__init__"}
    all_stems = {p.stem for p in SHELL.rglob("*.py") if p.parent == SHELL}
    assert all_stems - excluded == set(devloop._MODULE_ORDER)


def test_depends_on_covers_every_real_shell_import():
    """_DEPENDS_ON is hand-maintained, same as _MODULE_ORDER above -- so it can silently go stale
    the same way: workflow_zone.py grew a ``from dynamix.shell.browser import ...`` (the Task-5
    drag-drop mime constants) and nobody added "browser" to its _DEPENDS_ON tuple. The result is
    not an error, just a devloop session where editing browser.py rebuilds main_window but leaves
    workflow_zone.py's stale DEVICE_MIME/PRESET_MIME closed over -- drag-drop silently stops
    matching. Parse every top-level shell module's OWN imports (not transitive, and not the
    docstrings/comments that regularly reference dynamix.shell.* by name) and assert each one is
    declared -- a superset in _DEPENDS_ON is fine (over-reloading is merely wasteful), a subset is
    the bug this guards.

    Same file-discovery convention as ``test_devloop_module_order_covers_every_shell_module``:
    top-level ``.py`` files under ``shell/`` only, ``app``/``demo``/``__init__`` excluded (they are
    not part of the reload dependency graph at all).
    """
    import ast

    import dynamix.devloop as devloop

    excluded = {"app", "demo", "__init__"}
    for path in sorted(SHELL.rglob("*.py")):
        if path.parent != SHELL or path.stem in excluded:
            continue
        imported = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 \
                    and (node.module or "").startswith("dynamix.shell."):
                imported.add(node.module.split(".")[2])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("dynamix.shell.") and alias.name.count(".") >= 2:
                        imported.add(alias.name.split(".")[2])
        imported.discard(path.stem)      # a module never depends on itself
        declared = set(devloop._DEPENDS_ON.get(path.stem, ()))
        assert imported <= declared, (
            f"{path.name} imports {sorted(imported - declared)} not listed in "
            f"_DEPENDS_ON[{path.stem!r}]"
        )


# ----------------------------------------------------------------------------------------------
# review fixes: SESSION["path"] desync (F1) and reload-during-compute (F2)


def _reset_session(devloop):
    """SESSION is a module-level global shared across the whole test session -- restore it so one
    test's stub state can never leak into another's."""
    original = dict(devloop.SESSION)
    return original


class _StubCanvas:
    """The one thing ``_capture`` asks a canvas for: the camera it is currently showing."""

    def __init__(self, view_range=((0.0, 1.0), (0.0, 1.0))):
        self.view = self
        self._range = view_range

    def viewRange(self):
        return self._range


def test_capture_refreshes_path_from_the_live_window():
    """F1: _capture() used to leave SESSION["path"] exactly as the CLI arg set it, forever. A
    window opened via the GUI (no CLI path) or opened over a SECOND file mid-session would then
    rebuild with a path that no longer names the field actually on screen -- at best the raster
    silently disappears (the field/path guard in _build_window fails), at worst load_field(field_B,
    path_A) hands the wrong path to Project.add_source's dedup and a rebuild displays file A's
    result under file B's name. The path must be read back from the window's own project/layer,
    the same source of truth _build_window's load_field call uses.
    """
    import dynamix.devloop as devloop
    from dynamix.model.chain import Chain
    from dynamix.model.project import Project

    original = _reset_session(devloop)
    try:
        devloop.SESSION["path"] = "old/path/A.npz"      # stale -- e.g. left over from the CLI arg
        project = Project(title="t")
        source = project.add_source("new/path/B.npz")   # the file actually opened via the GUI
        layer = project.add_layer("B", source.source_id, Chain())

        class StubWindow:
            field = object()
            cache = object()
            canvas = _StubCanvas()

        window = StubWindow()
        window.project = project
        window.layer = layer

        devloop._capture(window)

        assert devloop.SESSION["path"] == "new/path/B.npz"
    finally:
        devloop.SESSION.clear()
        devloop.SESSION.update(original)


def test_capture_falls_back_to_the_existing_path_with_no_layer():
    """A window with no layer yet (nothing opened) has no source to read a path from -- SESSION's
    existing path (if any) must survive rather than being clobbered with None."""
    import dynamix.devloop as devloop

    original = _reset_session(devloop)
    try:
        devloop.SESSION["path"] = "kept/path.npz"

        class StubWindow:
            field = None
            cache = object()
            project = object()
            layer = None
            canvas = _StubCanvas()

        devloop._capture(StubWindow())

        assert devloop.SESSION["path"] == "kept/path.npz"
    finally:
        devloop.SESSION.clear()
        devloop.SESSION.update(original)


def test_capture_stores_the_canvas_view_range(qtbot):
    """C-2: the rebuilt window has to come back where you were looking. ``load_field`` ends in
    ``autoRange()``, so without carrying the camera across, every reload snapped back out to the
    whole raster -- and the acceptance criterion asks for the same raster, stack, layer AND
    view."""
    import dynamix.devloop as devloop

    original = _reset_session(devloop)
    try:
        class StubWindow:
            field = None
            cache = None
            project = None
            layer = None
            canvas = _StubCanvas(((10.0, 110.0), (20.0, 70.0)))

        devloop._capture(StubWindow())

        assert devloop.SESSION["view"] == [[10.0, 110.0], [20.0, 70.0]]
    finally:
        devloop.SESSION.clear()
        devloop.SESSION.update(original)


def test_session_view_round_trips_into_a_rebuilt_canvas(qtbot):
    """The other half: what ``_capture`` stored is what ``_restore_view`` puts back, through a
    REAL ``Canvas`` on both ends (an aspect-locked ViewBox adjusts whatever range it is handed,
    so a stub on the receiving side would prove nothing about what the user sees).

    Driven against bare canvases rather than a whole rebuild: ``_build_window`` also dispatches a
    worker, and this seam is the part that can go wrong on its own.
    """
    import dynamix.devloop as devloop
    from dynamix.shell.canvas import Canvas

    original = _reset_session(devloop)
    try:
        source = Canvas()
        qtbot.addWidget(source)
        source.view.setRange(xRange=(10.0, 110.0), yRange=(20.0, 70.0), padding=0)

        class Captured:
            field = None
            cache = None
            project = None
            layer = None

        captured = Captured()
        captured.canvas = source
        devloop._capture(captured)

        rebuilt = Canvas()
        qtbot.addWidget(rebuilt)
        rebuilt.view.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0)

        class Rebuilt:
            pass

        target = Rebuilt()
        target.canvas = rebuilt
        devloop._restore_view(target)

        # flattened: pytest.approx will not walk a nested sequence
        flat = [v for axis in rebuilt.view.viewRange() for v in axis]
        assert flat == pytest.approx([v for axis in source.view.viewRange() for v in axis],
                                     rel=1e-3)
        assert devloop.SESSION["view"] == [list(axis) for axis in source.view.viewRange()]
    finally:
        devloop.SESSION.clear()
        devloop.SESSION.update(original)


def test_restore_view_is_a_no_op_with_nothing_captured():
    """A first launch has no view to restore; it must not reach into the window at all."""
    import dynamix.devloop as devloop

    original = _reset_session(devloop)
    try:
        devloop.SESSION["view"] = None
        devloop._restore_view(object())          # no .canvas -- would raise if it looked
    finally:
        devloop.SESSION.clear()
        devloop.SESSION.update(original)


def test_poll_defers_a_reload_while_the_window_is_computing():
    """F2: reloading mid-compute would leave two live windows sharing one non-thread-safe Cache
    until the old worker finishes -- a race, not a rebuild. DevLoop.poll() must detect the change,
    remember it, and do nothing else while window.is_computing is True; only once compute finishes
    does the NEXT poll tick actually capture/reload/rebuild.
    """
    import dynamix.devloop as devloop

    calls = {"scan": 0, "built": 0}
    # theme.py's mtime moves once, between the first and second scan; every scan after that is
    # unchanged, so any reload beyond the first observed change is proof pending state survived.
    scans = [{"theme": 1.0}, {"theme": 2.0}, {"theme": 2.0}, {"theme": 2.0}]

    def fake_scan():
        i = min(calls["scan"], len(scans) - 1)
        calls["scan"] += 1
        return scans[i]

    class StubWindow:
        def __init__(self):
            self.is_computing = True
            self.closed = False

        def close(self):
            self.closed = True

    rebuilt_window = StubWindow()
    rebuilt_window.is_computing = False

    monkeypatch_targets = {
        "_scan_mtimes": fake_scan,
        "_capture": lambda w: None,
        "_reload": lambda names: None,
        "_build_window": lambda app: (calls.__setitem__("built", calls["built"] + 1),
                                      rebuilt_window)[1],
    }
    originals = {name: getattr(devloop, name) for name in monkeypatch_targets}
    for name, fn in monkeypatch_targets.items():
        setattr(devloop, name, fn)
    try:
        window = StubWindow()
        loop = devloop.DevLoop(app=None, window=window)

        loop.poll()                          # sees theme change; window is computing -> defer
        assert calls["built"] == 0
        assert loop.window is window
        assert not window.closed
        assert "theme" in loop.pending

        window.is_computing = False
        loop.poll()                          # no NEW fs change, but the pending one still fires
        assert calls["built"] == 1
        assert loop.window is rebuilt_window
        assert window.closed
        assert loop.pending == set()
    finally:
        for name, fn in originals.items():
            setattr(devloop, name, fn)
