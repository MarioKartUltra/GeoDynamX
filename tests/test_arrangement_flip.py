# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import pytest
from PySide6 import QtCore, QtWidgets

from dynamix.shell.settings import update_settings
from tests.test_shell_window import STUB_CHAIN, _FIELD, loaded, stub_devices, window  # noqa: F401


def test_app_boots_with_arrangement_never_built(window):
    assert window._center_stack.currentIndex() == 0
    assert window._arrangement is None            # lazy: nothing built yet


def test_tab_flips_and_builds_lazily(qtbot, window):
    # 2026-08-18 three-view supersession: Tab now cycles THREE
    # states -- raster -> vector -> geo -> raster (MainWindow._cycle_center_view) -- not two, so
    # a full round trip takes three presses, not two. The arrangement still builds lazily on the
    # very first press (into "vector"), which is what this test originally pinned down.
    qtbot.keyClick(window, QtCore.Qt.Key_Tab)
    assert window._center_stack.currentIndex() == 1
    assert window._arrangement is not None
    assert window._center_view == "vector"
    qtbot.keyClick(window, QtCore.Qt.Key_Tab)
    assert window._center_stack.currentIndex() == 1
    assert window._center_view == "geo"
    qtbot.keyClick(window, QtCore.Qt.Key_Tab)
    assert window._center_stack.currentIndex() == 0
    assert window._center_view == "raster"


def test_tab_inside_a_line_edit_does_not_flip(qtbot, window):
    edit = QtWidgets.QLineEdit(window)
    edit.show(); edit.setFocus()
    qtbot.keyClick(edit, QtCore.Qt.Key_Tab)
    assert window._center_stack.currentIndex() == 0


def test_missing_pyvista_degrades_to_notice(qtbot, window, monkeypatch):
    import dynamix.shell.arrangement.view as av
    monkeypatch.setattr(av, "_import_pyvista", lambda: None)
    qtbot.keyClick(window, QtCore.Qt.Key_Tab)
    assert window._center_stack.currentIndex() == 1     # flip still works
    assert window._arrangement.available is False       # notice, not crash


# --------------------------------------------------------------------------------------------
# Realistic delivery: the four tests above send Tab straight to ``window`` (or, for
# the line-edit case, to a never-shown ``window``'s child) -- synthetic deliveries that a real
# Qt app never performs. A genuine key press goes to whichever widget currently has KEYBOARD
# FOCUS, and that widget's own inherited ``QWidget.event()`` consumes Tab for focus-chain
# navigation before it would ever reach a handler up at the window. These tests show the flip
# working (and correctly NOT working) through that real path: the window is actually SHOWN and
# EXPOSED, Tab is sent to the focused DESCENDANT, never to ``window`` itself.


def test_tab_flips_when_the_open_button_has_focus(qtbot, window):
    window.show()
    qtbot.waitExposed(window)
    window.open_button.setFocus()
    qtbot.keyClick(window.open_button, QtCore.Qt.Key_Tab)
    assert window._center_stack.currentIndex() == 1


def test_tab_flips_when_a_drag_value_knob_has_focus(qtbot, loaded):
    """A ``DragValue`` (every chain-strip control) is a real focusable widget with its own
    ``keyPressEvent`` for Up/Down nudges -- Tab is not one of the keys it claims, so it falls
    through to ``QLabel``'s (its base class) default handling, which is exactly the
    focus-chain-consuming path the window-level fix has to reach BEFORE."""
    loaded.show()
    qtbot.waitExposed(loaded)
    knob = loaded.strips.strip(3).controls["frac"]      # modulus_threshold's DragValue
    knob.setFocus()
    qtbot.keyClick(knob, QtCore.Qt.Key_Tab)
    assert loaded._center_stack.currentIndex() == 1


def test_tab_inside_a_shown_line_edit_does_not_flip_and_still_navigates(qtbot, window):
    """The QLineEdit guard is exercised through the SAME real-focus path as the two tests
    above (not the synthetic direct-delivery the original suite used) -- and, because the
    filter genuinely declines this event rather than merely no-op'ing internally, Tab still
    reaches the line edit's own focus-chain handling: the assertion that focus has moved OFF
    ``edit`` is the proof the guard doesn't just suppress the flip, it truly steps aside."""
    window.show()
    qtbot.waitExposed(window)
    edit = QtWidgets.QLineEdit(window)
    other = QtWidgets.QLineEdit(window)          # somewhere for normal Tab navigation to land
    edit.show(); other.show()
    edit.setFocus()
    qtbot.wait(10)                                # let the focus-change event actually land
    assert QtWidgets.QApplication.focusWidget() is edit
    qtbot.keyClick(edit, QtCore.Qt.Key_Tab)
    assert window._center_stack.currentIndex() == 0
    assert edit.hasFocus() is False              # real navigation happened, not just a no-op


# --------------------------------------------------------------------------------------------
# Cross-window discrimination: EVERY live MainWindow installs its
# OWN app-level filter, and Qt runs all of them (most-recently-installed first) against every
# event flowing through qApp -- the first one to return True wins. Without checking WHOSE
# hierarchy the event's actual target belongs to, a second window's filter could consume a Tab
# addressed to the first window's focused widget: confirmed directly against the pre-fix code
# (two shown windows, focus A's open_button, press Tab -> B flipped instead of A, and A silently
# lost its own Tab handling too, since B's filter had already returned True and consumed it).


def test_tab_does_not_cross_flip_between_two_live_windows(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    win_a = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win_a)
    win_b = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win_b)
    with qtbot.waitSignal(win_a.resolved, timeout=10000):
        win_a.load_field(_FIELD, "mem:stub-a")
    with qtbot.waitSignal(win_b.resolved, timeout=10000):
        win_b.load_field(_FIELD, "mem:stub-b")
    win_a.show(); qtbot.waitExposed(win_a)
    win_b.show(); qtbot.waitExposed(win_b)

    win_a.open_button.setFocus()
    qtbot.wait(10)
    qtbot.keyClick(win_a.open_button, QtCore.Qt.Key_Tab)
    assert win_a._center_stack.currentIndex() == 1
    assert win_b._center_stack.currentIndex() == 0        # B untouched by A's flip

    knob_b = win_b.strips.strip(3).controls["frac"]        # modulus_threshold's DragValue
    knob_b.setFocus()
    qtbot.wait(10)
    qtbot.keyClick(knob_b, QtCore.Qt.Key_Tab)
    assert win_b._center_stack.currentIndex() == 1
    assert win_a._center_stack.currentIndex() == 1         # A unchanged by B's flip


# --------------------------------------------------------------------------------------------
# Scene manager: activate() now builds a real pyvistaqt.QtInteractor + Scene on first
# flip, when pyvista genuinely imported. CONFIRMED (independently of pytest, via a minimal
# throwaway repro script -- exit code 139/SIGSEGV) that constructing a real QtInteractor under
# Qt's "offscreen" QPA platform crashes the WHOLE PROCESS on this macOS/Apple-Silicon setup:
# VTK's native render-window embedding feeds ``self.winId()`` straight into
# ``vtkRenderWindow.SetWindowInfo()`` inside pyvistaqt's own
# ``QVTKRenderWindowInteractor.__init__``, unconditionally, before any ``off_screen`` kwarg is
# even consulted -- so no caller-side flag avoids it. Since every test in this suite runs under
# ``QT_QPA_PLATFORM=offscreen`` (mandated repo-wide), ``ArrangementView.activate()`` detects this
# specific platform and deliberately leaves the real build unattempted rather than crashing (see
# its own comment) -- this test pins that behaviour down explicitly, honestly, rather than
# letting it be an invisible side effect nobody asserts on. A real user session always runs under
# the "cocoa" platform (this app's only target), where the guard never triggers and the real
# build in test_tab_flips_and_builds_lazily's spirit runs for real.


def test_activate_under_the_offscreen_qpa_platform_does_not_crash_and_leaves_the_scene_unbuilt(
        qtbot, window):
    from PySide6 import QtWidgets as _QtWidgets

    assert _QtWidgets.QApplication.instance().platformName() == "offscreen"

    qtbot.keyClick(window, QtCore.Qt.Key_Tab)          # would SIGSEGV the whole run if unguarded

    assert window._center_stack.currentIndex() == 1     # the flip itself still happens
    assert window._arrangement.available is True        # pyvista genuinely imported in this env
    assert window._arrangement._scene is None            # ...but no real Scene got built
    assert window._arrangement._built is False            # ...so a real platform can still try


# --------------------------------------------------------------------------------------------
# The view's own face slims to a single header strip ("View…" button)
# + the render surface -- ModeRow, the last of the four controls once built directly on this
# view's own face, is retired from here exactly like MaskRow/GroupPalette/Commit were (see view.py's own module docstring, "Correction" note). Until the View dialog (which reaches this header's button) there was NO mode control anywhere in
# the running app -- a freshly built scene simply starts at its own DEFAULT_MODE ("mercator").


def test_arrangement_view_builds_only_a_header_when_available(qtbot):
    pytest.importorskip("pyvista", reason="pyvista not installed")
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    assert view._layout.count() == 1          # header only -- the interactor arrives on activate()
    assert not hasattr(view, "_mode_row")      # ModeRow retired from this view's face


def test_arrangement_view_header_has_one_view_options_button(qtbot):
    pytest.importorskip("pyvista", reason="pyvista not installed")
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    buttons = view._header.findChildren(QtWidgets.QPushButton)
    assert len(buttons) == 1
    assert buttons[0].text() == "View…"
    assert buttons[0] is view._view_options_button


def test_view_options_button_click_emits_view_options_requested(qtbot):
    pytest.importorskip("pyvista", reason="pyvista not installed")
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    with qtbot.waitSignal(view.viewOptionsRequested, timeout=1000):
        view._view_options_button.click()


def test_activate_adds_the_interactor_next_to_the_header_and_passes_the_theme_background(
        qtbot, monkeypatch):
    """Stubs past the offscreen-QPA segfault guard (see this file's own documented reason above,
    ``test_activate_under_the_offscreen_qpa_platform_does_not_crash_and_leaves_the_scene_unbuilt``)
    with a plain ``QWidget`` standing in for ``pyvistaqt.QtInteractor`` -- enough to prove the
    LAYOUT and the ARGUMENT this task adds, without needing a real native render window. The stub
    ``Scene`` raises immediately after recording its constructor args, aborting ``activate()``
    right there -- before anything past ``Scene(...)`` touches attributes this bare-QWidget
    stand-in doesn't have (``track_click_position``, ``iren``, ...)."""
    pytest.importorskip("pyvista", reason="pyvista not installed")
    import types

    import dynamix.shell.arrangement.scene as scene_mod
    import dynamix.shell.arrangement.view as av
    from dynamix.shell.theme import RESTRAINED_DARK

    view = av.ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    monkeypatch.setattr(av, "_real_window_surface_unavailable", lambda: False)

    class _FakeInteractor(QtWidgets.QWidget):
        pass

    fake_pyvistaqt = types.SimpleNamespace(QtInteractor=lambda parent: _FakeInteractor(parent))
    monkeypatch.setattr(av, "_pyvistaqt_module", fake_pyvistaqt)

    captured = {}

    class _StubScene:
        def __init__(self, plotter, background=None):
            captured["plotter"] = plotter
            captured["background"] = background
            raise RuntimeError("stop-after-scene")

    monkeypatch.setattr(scene_mod, "Scene", _StubScene)

    with pytest.raises(RuntimeError, match="stop-after-scene"):
        view.activate()

    assert view._layout.count() == 2                       # header + interactor
    assert isinstance(view._interactor, _FakeInteractor)
    assert captured["plotter"] is view._interactor
    assert captured["background"] == RESTRAINED_DARK.ground


# --------------------------------------------------------------------------------------------
# The View dialog's new ArrangementView passthroughs
# (camera_state/set_camera_state/reset_camera/set_graticule/set_vertical_exaggeration/
# set_background) -- every one shares the exact no-op-before-a-scene/interactor-exists guard
# set_mask/set_mode already use. Every test in THIS suite runs under the offscreen QPA platform
# (this file's own segfault-guard note above), so a freshly built, never-activated
# ArrangementView() is already the real, live "no scene/interactor yet" case for these -- no
# stubbing needed, unlike the interactor-construction test just above.


def test_camera_state_is_none_before_the_scene_is_built(qtbot):
    pytest.importorskip("pyvista", reason="pyvista not installed")
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    assert view.camera_state() is None


def test_set_camera_state_reset_camera_graticule_vexag_background_no_op_before_a_scene(qtbot):
    pytest.importorskip("pyvista", reason="pyvista not installed")
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    # None of these may raise, and none may build a scene/interactor as a side effect.
    view.set_camera_state(45.0, 10.0, 2.0)
    view.reset_camera()
    view.set_graticule(True)
    view.set_vertical_exaggeration(3.0)
    view.set_background("#204060")

    assert view._scene is None
    assert view._interactor is None
    assert view._camera is None


# --------------------------------------------------------------------------------------------
# MainWindow's own two new pieces -- opening the View dialog on
# "View…", and pushing whatever was last persisted into Settings.view_options right after every
# ArrangementView.activate() (the mask-row precedent's own sibling, see _toggle_center_view's own
# docstring). ``window`` (imported above from tests/test_shell_window.py) already carries
# ``clean_registry``/settings isolation through its own fixture chain.


def test_view_options_button_opens_the_view_dialog(qtbot, window):
    pytest.importorskip("pyvista", reason="pyvista not installed")
    from dynamix.shell.view_dialog import ViewDialog

    qtbot.keyClick(window, QtCore.Qt.Key_Tab)      # builds + activates the arrangement
    if not window._arrangement.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")
    assert window._view_dialog is None

    window._arrangement._view_options_button.click()

    assert isinstance(window._view_dialog, ViewDialog)
    assert window._view_dialog.isVisible()


def test_view_options_button_reuses_the_same_dialog_instance(qtbot, window):
    pytest.importorskip("pyvista", reason="pyvista not installed")

    qtbot.keyClick(window, QtCore.Qt.Key_Tab)
    if not window._arrangement.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")
    window._arrangement._view_options_button.click()
    first = window._view_dialog

    window._arrangement._view_options_button.click()

    assert window._view_dialog is first


def test_activation_pushes_the_persisted_view_options_into_the_view(qtbot, window, monkeypatch):
    """Extends the SAME post-``activate()`` push ``_on_mask_row_changed``'s own docstring already
    documents for the mask row -- mode/graticule/vexag/background now travel the identical way.
    ``ArrangementView.set_mode``/etc. are monkeypatched at the CLASS level so this pins the WINDOW
    side of the wiring (which values reach which call) independent of Scene's own guard, which
    ``test_set_camera_state_reset_camera_graticule_vexag_background_no_op_before_a_scene`` above
    (and the Scene-level tests in tests/test_arrangement_scene.py) already cover.

    2026-08-18 three-view supersession: the full
    mode/graticule/vexag/background push now happens only on the "geo" view --
    MainWindow._set_center_view's own vector/geo split pushes background ONLY on "vector"
    (mode-independent chrome; no CRS to project a mode against). Tab lands on "vector" first, so
    this drives a SECOND Tab press to reach "geo" before asserting the full push landed."""
    from dynamix.shell.arrangement.view import ArrangementView

    update_settings(view_options={"mode": "globe", "graticule": True, "vexag": 3.5,
                                   "background": "#204060"})
    calls: list[tuple] = []
    monkeypatch.setattr(ArrangementView, "set_mode",
                         lambda self, mode: calls.append(("set_mode", mode)))
    monkeypatch.setattr(ArrangementView, "set_graticule",
                         lambda self, v: calls.append(("set_graticule", v)))
    monkeypatch.setattr(ArrangementView, "set_vertical_exaggeration",
                         lambda self, v: calls.append(("set_vertical_exaggeration", v)))
    monkeypatch.setattr(ArrangementView, "set_background",
                         lambda self, v: calls.append(("set_background", v)))

    qtbot.keyClick(window, QtCore.Qt.Key_Tab)     # raster -> vector (background-only push)
    qtbot.keyClick(window, QtCore.Qt.Key_Tab)     # vector -> geo (the full push this test covers)
    assert window._center_view == "geo"

    assert ("set_mode", "globe") in calls
    assert ("set_graticule", True) in calls
    assert ("set_vertical_exaggeration", 3.5) in calls
    assert ("set_background", "#204060") in calls


def test_activation_with_no_saved_view_options_pushes_defaults(qtbot, window, monkeypatch):
    # 2026-08-18 three-view supersession: same retargeting as
    # test_activation_pushes_the_persisted_view_options_into_the_view above -- the full push only
    # happens on "geo", which now takes a SECOND Tab press (the first lands on "vector").
    from dynamix.shell.arrangement.view import ArrangementView
    from dynamix.shell.view_dialog import DEFAULT_BACKGROUND

    calls: list[tuple] = []
    monkeypatch.setattr(ArrangementView, "set_mode",
                         lambda self, mode: calls.append(("set_mode", mode)))
    monkeypatch.setattr(ArrangementView, "set_graticule",
                         lambda self, v: calls.append(("set_graticule", v)))
    monkeypatch.setattr(ArrangementView, "set_vertical_exaggeration",
                         lambda self, v: calls.append(("set_vertical_exaggeration", v)))
    monkeypatch.setattr(ArrangementView, "set_background",
                         lambda self, v: calls.append(("set_background", v)))

    qtbot.keyClick(window, QtCore.Qt.Key_Tab)     # raster -> vector (background-only push)
    qtbot.keyClick(window, QtCore.Qt.Key_Tab)     # vector -> geo (the full push this test covers)
    assert window._center_view == "geo"

    assert ("set_mode", "mercator") in calls
    assert ("set_graticule", False) in calls
    assert ("set_vertical_exaggeration", 1.0) in calls
    assert ("set_background", DEFAULT_BACKGROUND) in calls
