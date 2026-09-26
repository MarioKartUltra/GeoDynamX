# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""DeviceBrowser: the source of new chain steps.

One responsibility: list every registered device (split Transforms/Filters, per
``dynamix.model.is_transform``) plus whatever rack presets the caller injects under "Racks", and
let a row be dragged out as a mime payload the workflow zone will read. It knows nothing about
``MainWindow`` -- presets arrive through :meth:`DeviceBrowser.set_presets`, not an import, so
``dynamix.shell.main_window`` can depend on this module without this module depending back.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.model import DEVICES, is_transform

#: Drag mime types a row encodes itself as. The drop target reads these back.
DEVICE_MIME = "application/x-dynamix-device"
PRESET_MIME = "application/x-dynamix-preset"

#: Top-level category items, in display order.
_CATEGORIES = ("Transforms", "Filters", "Dev", "Racks")

#: Naming convention for stub devices — routed to the "Dev" category, not a registry change.
_DEV_PREFIX = "stub_"

#: Superseded devices (2026-09-19 split): still registered — saved projects name them and the
#: never-delete rule holds — but the palette shows their per-method successors instead, so
#: these join the stubs in the collapsed "Dev" category. tucker_havok (2026-09-23) is shown as
#: tucker_HOOI_HOSVD, its algorithm a toggle.
_SUPERSEDED = ("holder_map", "band_recon", "tucker_havok")

#: Devices only the shell places (a derivative dataset's vector loader): never dropped by hand,
#: so they sit in "Dev" too.
_SHELL_PLACED = ("derived_vectors", "bus")

#: Item-data role holding the mime type a row's ``mimeData()`` should encode under. Unset (``None``)
#: on the category headers -- that absence is what makes a header undraggable.
_MIME_ROLE = QtCore.Qt.UserRole


class DeviceBrowser(QtWidgets.QTreeWidget):
    """Registered devices by category, plus rack presets, each row draggable as one mime payload.

    Built once, from whatever is in ``DEVICES`` at construction time -- the registry is populated
    by ``register_builtin_devices()`` before ``MainWindow`` builds its panel, so this always sees
    the real device set. There is no re-population API: nothing in this slice adds a device or a
    preset after the window opens (YAGNI).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)

        self._categories = {name: QtWidgets.QTreeWidgetItem(self, [name])
                            for name in _CATEGORIES}
        for name, device in DEVICES.items():
            if name.startswith(_DEV_PREFIX) or name in _SUPERSEDED or name in _SHELL_PLACED:
                category = "Dev"
            else:
                category = "Transforms" if is_transform(device) else "Filters"
            self._add_row(self._categories[category], name, DEVICE_MIME)
        self.expandAll()
        self.collapseItem(self._categories["Dev"])

    def set_presets(self, presets: dict[str, tuple]) -> None:
        """Add one row per preset name under "Racks". ``presets`` maps a display name to a chain
        step tuple (``((device_name, params), ...)``) -- only the name is shown or dragged; the
        step tuple itself is the workflow zone's business, not the browser's."""
        racks = self._categories["Racks"]
        for name in presets:
            self._add_row(racks, name, PRESET_MIME)

    def _add_row(self, parent: QtWidgets.QTreeWidgetItem, name: str, mime_type: str) -> None:
        item = QtWidgets.QTreeWidgetItem(parent, [name])
        item.setData(0, _MIME_ROLE, mime_type)

    def mimeData(self, items):
        """Encode the FIRST row's name under the mime type it registered with. A category header
        set no ``_MIME_ROLE`` (``_add_row`` is never called on one), so it and any empty selection
        yield a payload-free ``QMimeData`` -- ``hasFormat`` false for both known types, never a
        crash from indexing an empty list."""
        data = QtCore.QMimeData()
        if not items:
            return data
        mime_type = items[0].data(0, _MIME_ROLE)
        if mime_type is None:
            return data
        data.setData(mime_type, items[0].text(0).encode("utf-8"))
        return data
