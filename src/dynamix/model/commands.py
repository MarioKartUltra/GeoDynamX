# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The command table: every user action the menus, shortcuts, palette and scripting can name. No Qt
here; ``dynamix.shell.commands`` binds these to QActions. Ids are dotted and stable: the ``.gprv``
writer records them as the action that produced a step."""
from __future__ import annotations

import dataclasses

MENU_ORDER = ("File", "Edit", "Layer", "Analysis", "View", "Racks", "Settings", "Help")


@dataclasses.dataclass(frozen=True)
class Command:
    id: str
    title: str
    category: str
    #: Default shortcuts as Qt key strings ("Ctrl+Return"); Ctrl reads as Cmd on macOS.
    shortcuts: tuple[str, ...] = ()
    checkable: bool = False
    #: Checkable commands sharing a group are exclusive (one checked at a time).
    group: str = ""
    #: Runs without a window (``python -m dynamix.run <id>``): the settings commands.
    headless: bool = False
    #: A submenu title inside the category's menu ("Compute engine"), or none.
    submenu: str = ""
    #: A separator sits before this entry in its menu.
    separator_before: bool = False


COMMANDS: tuple[Command, ...] = (
    Command("file.open_raster", "Open…", "File"),
    Command("file.open_project", "Open Project…", "File"),
    Command("file.save_project", "Save Project", "File", ("Ctrl+S",)),
    Command("file.save_project_as", "Save Project As…", "File"),
    Command("file.scan_footprints", "Scan Folder for Footprints…", "File", separator_before=True),
    Command("file.open_reference", "Open Reference Layer…", "File"),
    Command("file.open_overview", "Open Full Extent (Overview)…", "File"),
    Command("file.export_chains", "Export Chains (.npz)…", "File"),
    Command("file.export_derived", "Export Derived Raster (.npz)…", "File"),
    Command("edit.undo_transect", "Undo Transect", "Edit", ("Ctrl+Z",)),
    Command("edit.undo_rack_drop", "Undo Rack Removal", "Edit", ("Ctrl+Shift+Z",)),
    Command("edit.cancel_transect", "Cancel Transect", "Edit", ("Esc",)),
    Command("layer.roi", "ROI…", "Layer"),
    Command("chain.run", "Run", "Layer", ("Ctrl+Return",)),
    Command("transport.play", "Play / Pause", "Layer", ("Space",)),
    Command("view.zoom_to_dataset", "Zoom to Dataset", "View", ("Ctrl+0", "F")),
    Command("view.map_crs", "Map CRS…", "View"),
    Command("view.center.raster", "Raster", "View", checkable=True, group="view.center"),
    Command("view.center.map", "Map", "View", checkable=True, group="view.center"),
    Command("view.center.vector", "Vector", "View", checkable=True, group="view.center"),
    Command("view.center.geo", "Globe", "View", checkable=True, group="view.center"),
    Command("view.selection.click", "Click Selection", "View", ("C",)),
    Command("view.selection.cycle", "Next Selection Mode", "View", ("V",)),
    Command("view.dock.browser", "Devices", "View", checkable=True, separator_before=True),
    Command("view.dock.layers", "Processing", "View", checkable=True),
    Command("view.dock.inspector", "Inspector", "View", checkable=True),
    Command("view.dock.chain", "Chain", "View", checkable=True),
    Command("view.reset_layout", "Reset Layout", "View"),
    Command("view.layouts.save_as", "Save Layout As…", "View", submenu="Layouts"),
    Command("view.palette", "Command Palette…", "View", ("Ctrl+K",)),
    Command("racks.editor", "Rack Editor…", "Racks"),
    Command("settings.auto_run", "Auto-run transforms", "Settings", checkable=True, headless=True),
    Command("settings.engine.auto", "Auto (MLX on Apple Silicon, else FFTW3)", "Settings",
            checkable=True, group="settings.engine", headless=True, submenu="Compute engine"),
    Command("settings.engine.mlx", "MLX (Apple GPU, 32-bit)", "Settings", checkable=True,
            group="settings.engine", headless=True, submenu="Compute engine"),
    Command("settings.engine.fftw", "FFTW3 (CPU)", "Settings", checkable=True,
            group="settings.engine", headless=True, submenu="Compute engine"),
    Command("settings.precision.32", "32-bit (single)", "Settings", checkable=True,
            group="settings.precision", headless=True, submenu="FFT precision"),
    Command("settings.precision.64", "64-bit (double, FFTW3)", "Settings", checkable=True,
            group="settings.precision", headless=True, submenu="FFT precision"),
    Command("help.about", "About GeoDynamix", "Help"),
)


def by_id() -> dict:
    return {c.id: c for c in COMMANDS}
