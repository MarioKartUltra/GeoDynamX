# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""DeviceBrowser: the source of new chain steps.

One responsibility: list every registered device under its scientific families
(``dynamix.model.families``; a device in several families is listed under each), the devices kept
out of the palette (the stubs, the tables below, and any device declaring an empty ``families``)
under a collapsed "Dev", plus whatever rack presets the caller injects under "Racks", and let a
row be dragged out as a mime payload the workflow zone will read. A device row shows its display
label with its id beside it in a muted second column; its payload is always the id. It knows
nothing about ``MainWindow`` -- presets arrive through
:meth:`DeviceBrowser.set_presets` and each row's applicability to the active layer through
:meth:`DeviceBrowser.set_step_states`, not an import, so ``dynamix.shell.main_window`` can depend
on this module without this module depending back.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model import DEVICES, is_transform
from dynamix.model.families import FAMILY_ORDER, display_name, families_of, label_of

#: Drag mime types a row encodes itself as. The drop target reads these back.
DEVICE_MIME = "application/x-dynamix-device"
PRESET_MIME = "application/x-dynamix-preset"

#: The engine-role grouping (Transforms and Filters split by ``is_transform``). The tree is built
#: by family; "Dev" and "Racks" keep these names in :data:`_TAIL`.
_CATEGORIES = ("Transforms", "Filters", "Dev", "Racks")

#: The headers after the families, in display order.
_TAIL = ("Racks", "Dev")

#: Naming convention for stub devices — routed to the "Dev" category, not a registry change.
_DEV_PREFIX = "stub_"

#: Superseded devices: still registered — saved projects name them and the never-delete rule
#: holds — but the palette shows their per-method successors instead, so these join the stubs in
#: the collapsed "Dev" category. tucker_havok is shown as tucker_HOOI_HOSVD, its algorithm a
#: toggle.
#: The band_recon variants' reconstruction is the band output rows of holder_measure and
#: holder_multiaffine.
_SUPERSEDED = ("holder_map", "band_recon", "tucker_havok", "band_recon_measure",
               "band_recon_multiaffine")

#: Devices only the shell places (a derivative dataset's vector loader, a bus, the arrangement
#: view's committed groups): never dropped by hand, so they sit in "Dev" too.
_SHELL_PLACED = ("derived_vectors", "bus", "group_paint")

#: Item-data role holding the mime type a row's ``mimeData()`` should encode under. Unset (``None``)
#: on the category headers -- that absence is what makes a header undraggable.
_MIME_ROLE = QtCore.Qt.UserRole

#: Item-data role holding the payload a row drags out: the device id, or the preset name. A
#: device row's text is its display label, so the payload is read from here.
_ID_ROLE = QtCore.Qt.UserRole + 1


class DeviceBrowser(QtWidgets.QTreeWidget):
    """Registered devices by family, plus rack presets, each row draggable as one mime payload.

    Built once, from whatever is in ``DEVICES`` at construction time -- the registry is populated
    by ``register_builtin_devices()`` before ``MainWindow`` builds its panel, so this always sees
    the real device set. There is no re-population API: nothing in this slice adds a device or a
    preset after the window opens (YAGNI).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setColumnCount(2)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        #: The muted ink of the id column and of a greyed row: the palette's disabled text.
        self._muted = self.palette().brush(QtGui.QPalette.Disabled, QtGui.QPalette.Text)
        #: device id -> every row listing it (one per family it sits in).
        self._rows: dict[str, list[QtWidgets.QTreeWidgetItem]] = {}
        #: device id -> the tooltip its rows carry while it applies.
        self._tips: dict[str, str] = {}

        members: dict[str, list[str]] = {}
        for name, device in DEVICES.items():
            families = families_of(device)
            # An empty families tuple is a device's own mark that it left the palette.
            if (name.startswith(_DEV_PREFIX) or name in _SUPERSEDED or name in _SHELL_PLACED
                    or not families):
                members.setdefault("Dev", []).append(name)
                continue
            for family in families:
                listed = members.setdefault(family if family in FAMILY_ORDER else "Other", [])
                if name not in listed:
                    listed.append(name)
        self._categories = {}
        for title in FAMILY_ORDER + _TAIL:
            if title in members or title in _TAIL:
                header = QtWidgets.QTreeWidgetItem(self, [title])
                header.setFirstColumnSpanned(True)
                self._categories[title] = header
        for title, names in members.items():
            for name in names:
                self._add_device_row(self._categories[title], name)
        self.expandAll()
        self.collapseItem(self._categories["Dev"])

    def set_presets(self, presets: dict[str, tuple]) -> None:
        """Add one row per preset name under "Racks". ``presets`` maps a display name to a chain
        step tuple (``((device_name, params), ...)``) -- only the name is shown or dragged; the
        step tuple itself is the workflow zone's business, not the browser's."""
        racks = self._categories["Racks"]
        for name in presets:
            self._add_row(racks, name, PRESET_MIME)

    def set_step_states(self, states) -> None:
        """Grey the rows of every device that cannot apply as the next step on the active layer,
        its reason in the tooltip; ``states`` maps a device id to its
        :class:`~dynamix.model.contracts.StepState`. A greyed row stays draggable: a dropped step
        that does not apply is drawn inactive in the chain. Rows of a device absent from
        ``states`` keep what they show."""
        for name, state in states.items():
            for item in self._rows.get(name, ()):
                if state.applies:
                    item.setData(0, QtCore.Qt.ForegroundRole, None)
                    item.setToolTip(0, self._tips[name])
                else:
                    item.setForeground(0, self._muted)
                    item.setToolTip(0, state.reason)

    def _add_row(self, parent: QtWidgets.QTreeWidgetItem, name: str, mime_type: str) -> None:
        item = QtWidgets.QTreeWidgetItem(parent, [name])
        item.setData(0, _MIME_ROLE, mime_type)
        item.setData(0, _ID_ROLE, name)

    def _add_device_row(self, parent: QtWidgets.QTreeWidgetItem, name: str) -> None:
        """One device row: its label, its id muted in the second column (empty when the label is
        the id), the id as its payload."""
        device = DEVICES[name]
        label = label_of(device)
        item = QtWidgets.QTreeWidgetItem(parent, [label, name if label != name else ""])
        item.setData(0, _MIME_ROLE, DEVICE_MIME)
        item.setData(0, _ID_ROLE, name)
        item.setForeground(1, self._muted)
        self._tips[name] = display_name(device)
        item.setToolTip(0, self._tips[name])
        self._rows.setdefault(name, []).append(item)

    def mimeData(self, items):
        """Encode the FIRST row's id under the mime type it registered with. A category header
        set no ``_MIME_ROLE`` (no row helper is ever called on one), so it and any empty selection
        yield a payload-free ``QMimeData`` -- ``hasFormat`` false for both known types, never a
        crash from indexing an empty list."""
        data = QtCore.QMimeData()
        if not items:
            return data
        mime_type = items[0].data(0, _MIME_ROLE)
        if mime_type is None:
            return data
        data.setData(mime_type, str(items[0].data(0, _ID_ROLE)).encode("utf-8"))
        return data
