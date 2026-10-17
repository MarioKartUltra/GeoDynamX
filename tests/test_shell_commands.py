# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every menu action is a command; the shortcuts kept their keys; Space toggles the transport once."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.commands import by_id
from dynamix.shell.main_window import MainWindow
from tests.test_shell_window import _FIELD, loaded, stub_devices, window  # noqa: F401


def _menu_actions(w):
    out = []
    for menu in w.menuBar().findChildren(QtWidgets.QMenu):
        out += [a for a in menu.actions() if not a.isSeparator() and a.menu() is None]
    return out


def test_every_menu_action_carries_a_command_id_and_the_bar_follows_the_order(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    titles = [a.text() for a in w.menuBar().actions()]
    assert titles == ["File", "Edit", "Layer", "Analysis", "View", "Racks", "Settings", "Help"]
    table = by_id()
    for action in _menu_actions(w):
        cid = action.data()
        assert cid and (cid in table or cid.startswith(("device.", "view.layouts.apply.",
                                                        "settings.theme."))), action.text()


def test_shortcuts_keep_their_keys_and_window_context(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    for cid, key in {"chain.run": "Ctrl+Return", "transport.play": "Space", "view.selection.click": "C",
                     "view.selection.cycle": "V", "edit.cancel_transect": "Esc",
                     "edit.undo_rack_drop": "Ctrl+Shift+Z", "file.save_project": "Ctrl+S"}.items():
        action = w.commands[cid]
        assert key in [s.toString() for s in action.shortcuts()], cid
        assert action.shortcutContext() == QtCore.Qt.WindowShortcut, cid
    assert [s.toString() for s in w._zoom_action.shortcuts()] == ["Ctrl+0", "F"]


def test_settings_groups_are_exclusive_and_reflect_the_settings_file(qtbot, clean_registry):
    from dynamix.shell.settings import load_settings

    w = MainWindow()
    qtbot.addWidget(w)
    s = load_settings()
    engine = {k: w.commands[k] for k in ("settings.engine.auto", "settings.engine.mlx", "settings.engine.fftw")}
    assert [k for k, a in engine.items() if a.isChecked()] == [f"settings.engine.{s.compute_engine}"]
    prec = {k: w.commands[k] for k in ("settings.precision.32", "settings.precision.64")}
    assert [k for k, a in prec.items() if a.isChecked()] == [f"settings.precision.{s.compute_precision}"]


def test_the_analysis_menu_lists_devices_by_family(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    analysis = next(m for m in w.menuBar().findChildren(QtWidgets.QMenu) if m.title() == "Analysis")
    families = [a.menu().title() for a in analysis.actions() if a.menu() is not None]
    assert "Edge detection" in families
    ids = [a.data() for m in analysis.findChildren(QtWidgets.QMenu) for a in m.actions() if a.data()]
    assert "device.wtmm2d" in ids                      # the palette's id for the same action


def test_play_toggles_exactly_once_per_trigger_and_once_per_space_with_the_button_focused(qtbot, loaded):
    from PySide6 import QtTest

    w = loaded
    with qtbot.waitActive(w, timeout=3000):    # a window shortcut fires only in the active window
        w.show()
        w.activateWindow()
    tr = w.transport
    assert not tr.clock.playing
    w.commands["transport.play"].trigger()
    assert tr.clock.playing
    w.commands["transport.play"].trigger()
    assert not tr.clock.playing
    QtTest.QTest.mouseClick(tr.play_button, QtCore.Qt.LeftButton)   # the user's way to focus it
    assert tr.clock.playing                                # the click itself toggles once
    assert w.focusWidget() is not tr.play_button           # NoFocus: the button never takes focus
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_Space)
    assert not tr.clock.playing                            # once, by the window's command
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_Space)
    assert tr.clock.playing


def test_the_palette_opens_on_ctrl_k_filters_and_returns_focus(qtbot, loaded):
    from PySide6 import QtTest

    w = loaded
    with qtbot.waitActive(w, timeout=3000):
        w.show()
        w.activateWindow()
    w.commands["view.palette"].trigger()
    palette = w._palette
    assert palette.isVisible() and palette.line.hasFocus()
    shown = palette.filter("stub")
    assert "device.stub_stack" in shown and all("stub" in s for s in shown)
    assert "file.open_project" not in shown
    palette.filter("open project")
    assert palette.list.count() >= 1
    assert palette.list.item(0).data(QtCore.Qt.UserRole) == "file.open_project"
    QtTest.QTest.keyClick(palette.line, QtCore.Qt.Key_Escape)
    assert not palette.isVisible()
    qtbot.waitUntil(w.isActiveWindow, timeout=3000)


def test_the_settings_menu_keeps_its_submenus_and_file_and_view_their_separators(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    settings = {a.menu().title(): a.menu() for a in w.menus["Settings"].actions() if a.menu() is not None}
    assert list(settings) == ["Compute engine", "FFT precision", "Theme"]
    assert [a.data() for a in settings["Compute engine"].actions()] == [
        "settings.engine.auto", "settings.engine.mlx", "settings.engine.fftw"]
    assert [a.data() for a in settings["FFT precision"].actions()] == [
        "settings.precision.32", "settings.precision.64"]
    file_actions = w.menus["File"].actions()
    assert [a.isSeparator() for a in file_actions].index(True) == 4      # after Save Project As…
    view = w.menus["View"].actions()
    i = [a.data() for a in view].index("view.dock.browser")
    assert view[i - 1].isSeparator()


def test_the_palette_runs_the_current_row_after_the_arrow_keys(qtbot, loaded):
    from PySide6 import QtTest

    w = loaded
    w.show()
    qtbot.waitExposed(w)
    w.commands["view.palette"].trigger()
    palette = w._palette
    shown = palette.filter("dock")
    assert shown[:2] == ["view.dock.browser", "view.dock.layers"]
    QtTest.QTest.keyClick(palette.line, QtCore.Qt.Key_Down)
    assert palette.list.currentRow() == 1
    was_hidden = w.docks["dock.layers"].isHidden()
    QtTest.QTest.keyClick(palette.line, QtCore.Qt.Key_Return)
    assert not palette.isVisible()
    assert w.docks["dock.layers"].isHidden() != was_hidden       # the second entry ran
    assert not w.docks["dock.browser"].isHidden()                 # the first did not


def test_commands_needing_a_raster_are_disabled_and_out_of_the_palette_before_one_is_open(qtbot, window):
    w = window
    w.refresh_commands()
    for cid in ("chain.run", "file.export_chains", "file.export_derived", "layer.roi"):
        assert not w.commands[cid].isEnabled(), cid
    w.show_palette()
    assert "file.export_chains" not in w._palette.filter("")


def test_commands_needing_a_raster_enable_once_one_is_open(qtbot, loaded):
    assert loaded.commands["chain.run"].isEnabled()
    assert loaded.commands["file.export_chains"].isEnabled()


def test_ctrl_return_runs_the_chain_as_soon_as_a_raster_is_open(qtbot, window):
    """The enabled checks follow the window's field, not the next menu that happens to open."""
    from PySide6 import QtTest

    w = window
    with qtbot.waitActive(w, timeout=3000):
        w.show()
        w.activateWindow()
    assert not w.commands["chain.run"].isEnabled()
    with qtbot.waitSignal(w.resolved, timeout=10000):
        w.load_field(_FIELD, "mem:stub")
    assert w.commands["chain.run"].isEnabled()
    with qtbot.waitActive(w, timeout=3000):     # offscreen drops activation on a load; a desktop does not
        w.activateWindow()
    with qtbot.waitSignal(w.commands["chain.run"].triggered, timeout=2000):
        QtTest.QTest.keyClick(w, QtCore.Qt.Key_Return, QtCore.Qt.ControlModifier)


def test_the_dev_loop_knows_the_command_modules():
    import dynamix.devloop as devloop

    order = list(devloop._MODULE_ORDER)
    assert order.index("commands") < order.index("main_window")
    assert order.index("controller") < order.index("main_window")
    assert devloop._DEPENDS_ON["commands"] == () and devloop._DEPENDS_ON["controller"] == ()
    assert {"commands", "controller"} <= set(devloop._DEPENDS_ON["main_window"])


def test_the_palette_shows_every_shortcut_and_save_layout_as_sits_in_the_layouts_submenu(qtbot, loaded):
    w = loaded
    w.show_palette()
    w._palette.filter("zoom")
    label = w._palette.list.item(0).text()
    native = QtGui.QKeySequence.NativeText
    for key in ("Ctrl+0", "F"):
        assert QtGui.QKeySequence(key).toString(native) in label
    view = w.menus["View"]
    plain = [a.text() for a in view.actions() if a.menu() is None]
    assert "Save Layout As…" not in plain
    layouts = next(a.menu() for a in view.actions() if a.menu() is not None and a.menu().title() == "Layouts")
    assert [a.text() for a in layouts.actions()].count("Save Layout As…") == 1


def test_the_palette_filters_wtmm_to_the_wtmm_commands_and_devices(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    w = MainWindow()
    qtbot.addWidget(w)
    w.show_palette()
    shown = w._palette.filter("wtmm")
    assert "device.wtmm2d" in shown and all("wtmm" in s for s in shown)


def test_the_palette_shows_native_shortcut_text_and_not_itself(qtbot, loaded):
    w = loaded
    w.show_palette()
    shown = w._palette.filter("")
    assert "view.palette" not in shown
    labels = [w._palette.list.item(i).text() for i in range(w._palette.list.count())]
    save = next(label for label in labels if label.startswith("Save Project "))
    assert save.endswith(QtGui.QKeySequence("Ctrl+S").toString(QtGui.QKeySequence.NativeText))


def test_a_named_layout_gets_a_slug_id_that_never_collides_or_empties(qtbot, loaded):
    w = loaded
    for name in ("My Desk", "my-desk", "!!!"):
        w.save_layout(name)
    ids = [a.data() for a in w._layouts_menu.actions() if str(a.data()).startswith("view.layouts.apply.")]
    assert ids == ["view.layouts.apply.my_desk", "view.layouts.apply.my_desk_2", "view.layouts.apply.layout"]


def test_the_window_keeps_no_group_attribute(qtbot, loaded):
    assert not hasattr(loaded, "_command_groups")


def test_the_centre_view_switch_keeps_its_docstring():
    assert MainWindow._set_center_view.__doc__.startswith("Switch")


def test_transport_and_key_handling_live_in_the_controller(qtbot, loaded):
    from dynamix.shell.controller import CommandController

    w = loaded
    controller = CommandController(w)
    handlers = controller.handlers()
    assert {"transport.play", "edit.cancel_transect", "edit.undo_transect", "edit.undo_rack_drop",
            "view.selection.click", "view.selection.cycle", "view.zoom_to_dataset"} <= set(handlers)
    assert w.command_handlers()["transport.play"] == w.controller.toggle_transport
    w.controller.toggle_transport()
    assert w.transport.clock.playing
    w.controller.toggle_transport()                 # and back: a running sweep outlives the test
    assert not w.transport.clock.playing
    w.controller.selection_cycle()
    assert w._selection_mode != "click"
    w.controller.selection_click()
    assert w._selection_mode == "click"


def test_the_controller_imports_no_shell_module():
    """The controller is a leaf the window uses; the dev loop lists it with no dependencies."""
    import ast
    import pathlib

    import dynamix.shell.controller as controller

    tree = ast.parse(pathlib.Path(controller.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("dynamix.shell"):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names if a.name.startswith("dynamix.shell"))
    assert imported == set()
    assert controller.SELECTION_MODE_CYCLE == ("box", "lasso", "transect")
