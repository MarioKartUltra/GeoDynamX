# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The main window's docks: a small fixed set, built here and nowhere else, with today's arrangement
as the default layout (browser and layers left, inspector right, chain strip along the bottom)."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

DOCK_NAMES = ("dock.browser", "dock.layers", "dock.inspector", "dock.chain")
TITLES = {"dock.browser": "Devices", "dock.layers": "Layers", "dock.inspector": "Inspector",
          "dock.chain": "Chain"}
_SIDES = QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
_FREE = (QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
         | QtWidgets.QDockWidget.DockWidgetClosable)


def _make(name: str, widget: QtWidgets.QWidget, areas, features) -> QtWidgets.QDockWidget:
    dock = QtWidgets.QDockWidget(TITLES[name])
    dock.setObjectName(name)                 # saveState keys on it; fixed for every layout ever saved
    dock.setAllowedAreas(areas)
    dock.setFeatures(features)
    dock.setWidget(widget)
    return dock


def build_docks(window: QtWidgets.QMainWindow, *, browser, layers, inspector, chain,
                widths=(220, 260), chain_height=120) -> dict:
    """Place the four docks around ``window``'s central widget in the default arrangement and
    size them; returns them by name. The chain stays along the bottom, full width, pinned: it can
    be hidden from the View menu and nothing else."""
    docks = {
        "dock.browser": _make("dock.browser", browser, _SIDES, _FREE),
        "dock.layers": _make("dock.layers", layers, _SIDES, _FREE),
        "dock.inspector": _make("dock.inspector", inspector, _SIDES, _FREE),
        "dock.chain": _make("dock.chain", chain, QtCore.Qt.BottomDockWidgetArea,
                            QtWidgets.QDockWidget.DockWidgetClosable),
    }
    window.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.BottomDockWidgetArea)
    window.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.BottomDockWidgetArea)
    window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks["dock.browser"])
    window.splitDockWidget(docks["dock.browser"], docks["dock.layers"], QtCore.Qt.Vertical)
    window.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks["dock.inspector"])
    window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, docks["dock.chain"])
    window.resizeDocks([docks["dock.browser"], docks["dock.inspector"]],
                       [int(widths[0]), int(widths[1])], QtCore.Qt.Horizontal)
    window.resizeDocks([docks["dock.chain"]], [int(chain_height)], QtCore.Qt.Vertical)
    return docks


def seed_sizes(splitter_sizes, left: int, right: int, chain: int) -> tuple:
    """Dock sizes from a settings file written before docks existed: ``splitter_sizes["work"]``
    was ``[left, centre, right]`` and ``["main"]`` ``[work, chain]``; absent or malformed, the
    defaults given."""
    work = splitter_sizes.get("work") if isinstance(splitter_sizes, dict) else None
    main = splitter_sizes.get("main") if isinstance(splitter_sizes, dict) else None
    if isinstance(work, list) and len(work) == 3:
        left, right = int(work[0]), int(work[2])
    if isinstance(main, list) and len(main) == 2:
        chain = int(main[1])
    return left, right, chain
