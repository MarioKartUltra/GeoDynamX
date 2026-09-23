# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Widgets earned by a ControlSpec. The number IS the control (The Reading Rule)."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.param import ParamKind
from dynamix.shell.knobs import ControlSpec, format_reading, nudge, rebind_range


def _modifier_name(modifiers: QtCore.Qt.KeyboardModifiers) -> str:
    if modifiers & QtCore.Qt.ShiftModifier:
        return "coarse"
    if modifiers & QtCore.Qt.AltModifier:
        return "fine"
    return "normal"


class DragValue(QtWidgets.QLabel):
    valueChanged = QtCore.Signal(object)

    def __init__(self, spec: ControlSpec, value):
        super().__init__()
        self._spec, self._value = spec, value
        self.setProperty("reading", "true")
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setCursor(QtCore.Qt.SizeVerCursor)
        self._editor: QtWidgets.QLineEdit | None = None
        self._drag_origin = None       # (press_y, value_at_press)
        self._drag_last_emitted = None
        self._refresh()

    # -- display --------------------------------------------------------
    def set_value(self, v) -> None:
        """Update the display without emitting (guard against feedback loops).

        This is the ONLY place ``self._value`` -- the confirmed, displayed value -- changes
        (besides the explicit reset-to-default, which sets and emits together). Keyboard nudges,
        drags and inline-edit commits only ever PROPOSE a candidate via ``valueChanged``; they
        never write it back locally. That keeps repeated gestures (e.g. a coarse nudge followed
        by a fine one) each computed from the same last-confirmed value instead of compounding
        off an unconfirmed candidate -- the owner is expected to call ``set_value`` once it has
        applied (and possibly re-quantized) the proposal.
        """
        self._value = v
        self._refresh()

    def set_range(self, lo: float, hi: float) -> None:
        """Rebinds this control's SOFT bounds -- view state only, never the value itself and
        never ``self._spec.param`` (see ``knobs.rebind_range``'s own docstring for why this can
        never affect ``cache_key``). Used by ``DeviceBox.sync_from_result``
when a device's own ``data_hints`` reports a real data
        range, so a drag/nudge steps in units scaled to what the data actually spans."""
        self._spec = rebind_range(self._spec, lo, hi)

    def _refresh(self) -> None:
        self.setText(format_reading(self._value, self._spec.param))

    def _propose(self, new_value) -> None:
        """Emit a candidate value if it differs from the last confirmed value. Does not mutate
        ``self._value`` -- see ``set_value``."""
        if new_value == self._value:
            return
        self.valueChanged.emit(new_value)

    # -- keyboard nudge ---------------------------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            direction = 1 if event.key() == QtCore.Qt.Key_Up else -1
            modifier = _modifier_name(event.modifiers())
            # Pass the LIVE spec, not `.param` -- see `nudge`'s own docstring.
            # `.param` forced `nudge` to rebuild a fresh ControlSpec from the STATIC declared soft
            # range every call, silently discarding any `set_range` rebind (a real keyboard nudge
            # on a data-hinted control stepped at the device's static size, never the rebound one).
            self._propose(nudge(self._value, self._spec, direction, modifier))
            return
        super().keyPressEvent(event)

    # -- vertical mouse drag ------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.setFocus(QtCore.Qt.MouseFocusReason)
            self._drag_origin = (event.position().y(), self._value)
            self._drag_last_emitted = self._value
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_origin is None:
            super().mouseMoveEvent(event)
            return
        origin_y, origin_value = self._drag_origin
        dy = origin_y - event.position().y()            # up = increase
        steps = int(dy // 2)                             # recomputed from origin every move --
        direction = 1 if steps > 0 else -1                # never accumulated from prior moves
        modifier = _modifier_name(event.modifiers())
        new_value = origin_value
        for _ in range(abs(steps)):
            # Same fix as keyPressEvent -- the live spec, not `.param`.
            new_value = nudge(new_value, self._spec, direction, modifier)
        if new_value != self._drag_last_emitted:
            self._drag_last_emitted = new_value
            self.valueChanged.emit(new_value)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_origin = None
            self._drag_last_emitted = None
            return
        super().mouseReleaseEvent(event)

    # -- inline edit -----------------------------------------------------
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.begin_edit()

    def begin_edit(self) -> None:
        if self._editor is not None:
            return
        editor = QtWidgets.QLineEdit(self)
        # Seed with the EXACT stored value (repr round-trips float64 losslessly), never the
        # rounded display text: seeding "0.00" for a stored 0.001 meant double-click + Enter
        # silently committed 0.0 -- the display's rounding became the real value. The reading
        # is a display; the editor is the machine-precision surface.
        editor.setText(repr(self._value))
        editor.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        editor.setGeometry(self.rect())
        editor.editingFinished.connect(lambda: self.commit_edit(editor.text()))
        editor.installEventFilter(self)
        self._editor = editor
        editor.show()
        editor.setFocus(QtCore.Qt.MouseFocusReason)
        editor.selectAll()

    def commit_edit(self, text: str) -> None:
        if self._editor is None:
            return
        editor, self._editor = self._editor, None
        editor.editingFinished.disconnect()
        editor.hide()
        editor.deleteLater()
        try:
            raw = int(text) if self._spec.param.kind is ParamKind.INT else float(text)
            new_value = self._spec.param.validate(raw)
        except ValueError:
            return                      # revert silently, keeping the old (still-displayed) value
        self._propose(new_value)

    def _cancel_edit(self) -> None:
        if self._editor is None:
            return
        editor, self._editor = self._editor, None
        editor.editingFinished.disconnect()
        editor.hide()
        editor.deleteLater()

    def eventFilter(self, obj, event):
        if obj is self._editor and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Escape:
                self._cancel_edit()
                return True
        return super().eventFilter(obj, event)

    # -- context menu ------------------------------------------------------
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        reset = menu.addAction("Reset to default")
        chosen = menu.exec(event.globalPos())
        if chosen is reset:
            default = self._spec.param.default
            self.set_value(default)
            self.valueChanged.emit(default)


class LabelReading(QtWidgets.QLabel):
    """Non-interactive display for ``ParamKind.TEXT`` -- committed groups are a reading, not an
    editor (the zone renders the commit transaction's groups, it does not let anyone
    type into them). Carries ``valueChanged`` for signature parity with the other controls --
    every caller (``chain_strip.py``, ``main_window.py``, ``workflow_zone.py``) connects a
    device's controls to it unconditionally by param -- but nothing here ever emits it."""

    valueChanged = QtCore.Signal(object)

    #: Elide long spec_json/group text rather than let it blow out a knob column's width -- the
    #: full value is still available via the tooltip.
    _ELIDE_WIDTH = 220

    def __init__(self, spec: ControlSpec, value):
        super().__init__()
        self._spec, self._value = spec, value
        self.setProperty("reading", "true")
        self.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self._refresh()

    def set_value(self, v) -> None:
        self._value = v
        self._refresh()

    def _refresh(self) -> None:
        text = format_reading(self._value, self._spec.param)
        self.setToolTip(text)
        self.setText(self.fontMetrics().elidedText(text, QtCore.Qt.ElideRight, self._ELIDE_WIDTH))


class TextEdit(QtWidgets.QLineEdit):
    """Editable ``ParamKind.TEXT`` control -- earned by ``Param.editable=True`` (``group_filter.group`` has no other way for a user to name a group; ``LabelReading`` above
    stays the default for a TEXT param nobody should type into, e.g. ``group_paint.spec_json``).

    Emits ``valueChanged`` on ``editingFinished`` (Enter, or focus-out) -- the same commit-on-
    confirm idiom every other control in this module uses (``DragValue`` proposes on nudge/drag-
    release/inline-edit-commit, never on every keystroke); a plain ``QLineEdit.textChanged`` would
    fire a param write, and so a resolve, per keystroke."""

    valueChanged = QtCore.Signal(object)

    def __init__(self, spec: ControlSpec, value):
        super().__init__()
        self._spec, self._value = spec, value
        self.setProperty("reading", "true")
        self.setText(str(value))
        self.editingFinished.connect(self._commit)

    def set_value(self, v) -> None:
        """Update the display without emitting -- same guard-against-feedback-loop contract as
        every other control's own ``set_value`` (see ``DragValue``'s docstring). ``setText`` never
        fires ``editingFinished`` on its own, so this cannot re-enter :meth:`_commit`."""
        self._value = v
        self.setText(str(v))

    def _commit(self) -> None:
        new_value = self.text()
        if new_value == self._value:
            return
        self.valueChanged.emit(new_value)


class CycleButton(QtWidgets.QPushButton):
    valueChanged = QtCore.Signal(object)

    def __init__(self, spec: ControlSpec, value):
        super().__init__()
        self._spec, self._value = spec, value
        self._refresh()
        self.clicked.connect(self._advance)

    def set_value(self, v) -> None:
        self._value = v
        self._refresh()

    def _refresh(self) -> None:
        self.setText(str(self._value))

    def _advance(self) -> None:
        choices = self._spec.param.choices
        idx = choices.index(self._value)
        self._select(choices[(idx + 1) % len(choices)])

    def _select(self, new_value) -> None:
        self._value = new_value
        self._refresh()
        self.valueChanged.emit(new_value)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        actions = {menu.addAction(choice): choice for choice in self._spec.param.choices}
        chosen = menu.exec(event.globalPos())
        if chosen in actions:
            self._select(actions[chosen])


class Toggle(QtWidgets.QPushButton):
    valueChanged = QtCore.Signal(object)

    def __init__(self, spec: ControlSpec, value):
        super().__init__()
        self._spec = spec
        self.setCheckable(True)
        self.setText(spec.param.label or spec.param.name)
        self.setChecked(bool(value))
        self.toggled.connect(self._on_toggled)

    def set_value(self, v) -> None:
        self.blockSignals(True)           # guard against feedback loops: no toggled -> no emit
        self.setChecked(bool(v))
        self.blockSignals(False)

    def _on_toggled(self, checked: bool) -> None:
        self.valueChanged.emit(checked)


def make_control(spec: ControlSpec, value) -> QtWidgets.QWidget:
    return {"drag_value": DragValue, "cycle": CycleButton, "toggle": Toggle,
            "label": LabelReading, "text_edit": TextEdit}[spec.widget](spec, value)
