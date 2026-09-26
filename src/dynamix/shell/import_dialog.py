# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Importing from a multi-grid container (netCDF / HDF): which rasters, as which datasets.

A sensor product holds many grids -- ASTER: 15 image bands across three telescopes plus
supplements, geolocation lattices and correction tables. The dialog groups the IMAGE grids
by (sensor, resolution) with a checkbox per band, and each checked group imports as ONE
multiband dataset. Grids of different resolution (VNIR vs SWIR vs TIR, the 3B backsight)
can never land in one stack -- combining across resolutions is deliberate band math /
registration later, not an import-time resample. Everything non-image stays available under
"Ancillary grids", each importing alone.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["ImportGridsDialog", "group_container_grids"]

#: A 2-D grid whose short side reaches this many samples counts as an IMAGE; smaller or
#: skinnier grids (geolocation lattices, correction tables, supplements) list as ancillary.
IMAGE_MIN_SIDE = 128


def group_container_grids(subs, dims) -> list:
    """``[(label, [subdataset ids], is_image_group)]`` from probe's listing.

    Image grids group by (sensor, shape): the sensor is the swath-group prefix before
    ``_Band`` (``VNIR_Band3N`` -> ``VNIR``), so same-resolution bands of one telescope share
    a group while the 3B backsight (its own grid) gets its own. A grid without known dims,
    and every non-image grid, stands alone.
    """
    order: list = []
    image: dict = {}
    for sid, _desc in subs:
        d = tuple(int(v) for v in (dims.get(sid) or ()))
        if len(d) == 2 and min(d) >= IMAGE_MIN_SIDE:
            grp = sid.split("/", 1)[0] if "/" in sid else ""
            sensor = grp.split("_Band")[0] if "_Band" in grp else grp
            key = (sensor, d)
            if key not in image:
                image[key] = []
                order.append(("image", key))
            image[key].append(sid)
        else:
            order.append(("single", sid))
    out = []
    for kind, key in order:
        if kind == "image":
            sensor, d = key
            label = f"{sensor} {d}" if sensor else f"grids {d}"
            out.append((label, image[key], True))
        else:
            out.append((key, [key], False))
    return out


class ImportGridsDialog(QtWidgets.QDialog):
    """Checkbox tree over a container's grids; OK returns the checked selection through
    :meth:`groups` as ``[(dataset label, [subdataset ids])]`` -- one entry per dataset to
    import (a multi-band image group, or a lone grid)."""

    def __init__(self, container_name: str, subs, dims, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import from {container_name}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Tick the rasters to import. Each sensor group becomes one multiband dataset; "
            "different resolutions import as separate datasets."))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        layout.addWidget(self.tree)
        ancillary_parent = None
        for label, ids, is_image in group_container_grids(subs, dims):
            desc = dict(subs)
            if is_image:
                parent_item = QtWidgets.QTreeWidgetItem(self.tree, [f"{label} — {len(ids)} band(s)"])
                parent_item.setFlags(parent_item.flags() | QtCore.Qt.ItemIsAutoTristate)
                for sid in ids:
                    child = QtWidgets.QTreeWidgetItem(parent_item, [desc.get(sid, sid)])
                    child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                    child.setCheckState(0, QtCore.Qt.Unchecked)
                    child.setData(0, QtCore.Qt.UserRole, sid)
                parent_item.setExpanded(True)
                parent_item.setData(0, QtCore.Qt.UserRole, ("group", label))
            else:
                if ancillary_parent is None:
                    ancillary_parent = QtWidgets.QTreeWidgetItem(self.tree, ["Ancillary grids"])
                    ancillary_parent.setFlags(
                        ancillary_parent.flags() | QtCore.Qt.ItemIsAutoTristate)
                    ancillary_parent.setExpanded(False)
                sid = ids[0]
                child = QtWidgets.QTreeWidgetItem(ancillary_parent, [desc.get(sid, sid)])
                child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                child.setCheckState(0, QtCore.Qt.Unchecked)
                child.setData(0, QtCore.Qt.UserRole, sid)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def groups(self) -> list:
        """The checked selection: one ``(label, [ids])`` per image group with any checked
        band (only the checked ones, in tree order), plus one per checked ancillary grid."""
        out = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            role = top.data(0, QtCore.Qt.UserRole)
            checked = [top.child(j).data(0, QtCore.Qt.UserRole)
                       for j in range(top.childCount())
                       if top.child(j).checkState(0) == QtCore.Qt.Checked]
            if not checked:
                continue
            if isinstance(role, tuple) and role[0] == "group":
                out.append((role[1], checked))
            else:                                    # the ancillary bucket: each grid alone
                out.extend((sid, [sid]) for sid in checked)
        return out
