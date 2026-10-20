# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The hidden shared scale range: a handshake between the fitting windows.

Every fitting window (structure functions, log-cumulants, large deviations, the generalized
spectrum, the partition functions) fits over a log2-scale range of its own. The main window
holds one hidden shared range beside them. A window that follows publishes its range whenever
the user moves it and takes the range any other window publishes; the last change wins; there
is no owner. Each window detaches with its own switch and keeps its range; switching back on
takes the shared range as it stands. Applying a shared range never re-publishes it, so a
change is said once and echoes nowhere.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["SharedRangeFollower", "SharedScaleRange"]


class SharedScaleRange(QtCore.QObject):
    """The shared range itself: ``value`` is ``(log2_a_min, log2_a_max)`` or None until a
    window publishes; ``changed`` carries the new range and the window that moved it."""

    changed = QtCore.Signal(float, float, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.value: "tuple[float, float] | None" = None

    def publish(self, log2_a_min: float, log2_a_max: float, source=None) -> None:
        lo, hi = sorted((float(log2_a_min), float(log2_a_max)))
        if self.value == (lo, hi):
            return
        self.value = (lo, hi)
        self.changed.emit(lo, hi, source)


class SharedRangeFollower:
    """What a fitting window adds to take part: a follow switch (:meth:`_make_follow_check`),
    :meth:`attach_shared_range`, and two calls in its own code: :meth:`_publish_range` where
    the user's change lands, and :meth:`_apply_scale_window` setting the window's range
    silently. Mixed in before the Qt base class; no ``__init__`` of its own."""

    _shared: "SharedScaleRange | None" = None
    _applying_shared: bool = False
    follow_check: "QtWidgets.QCheckBox | None" = None

    def _make_follow_check(self) -> QtWidgets.QCheckBox:
        self.follow_check = QtWidgets.QCheckBox("follow shared range")
        self.follow_check.setChecked(True)
        self.follow_check.setToolTip("Take the scale range the other fitting windows publish, and publish "
                                     "this one's; off, this window keeps its own range")
        self.follow_check.toggled.connect(self._on_follow_toggled)
        return self.follow_check

    def attach_shared_range(self, shared: SharedScaleRange) -> None:
        """Join the handshake: hear every published range, and take the current one now."""
        self._shared = shared
        shared.changed.connect(self._on_shared_range)
        if shared.value is not None and self._follows():
            self._take(*shared.value)

    def _follows(self) -> bool:
        return self.follow_check is None or self.follow_check.isChecked()

    def _publish_range(self, log2_a_min: float, log2_a_max: float) -> None:
        """A user's change of this window's range, said to the others (never while taking one)."""
        if self._shared is not None and self._follows() and not self._applying_shared:
            self._shared.publish(log2_a_min, log2_a_max, self)

    def _on_shared_range(self, lo: float, hi: float, source) -> None:
        if source is self or not self._follows():
            return
        self._take(lo, hi)

    def _on_follow_toggled(self, on: bool) -> None:
        if on and self._shared is not None and self._shared.value is not None:
            self._take(*self._shared.value)

    def _take(self, lo: float, hi: float) -> None:
        self._applying_shared = True
        try:
            self._apply_scale_window(lo, hi)
        finally:
            self._applying_shared = False

    def _apply_scale_window(self, log2_a_min: float, log2_a_max: float) -> None:
        raise NotImplementedError
