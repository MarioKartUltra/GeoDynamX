# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""DeviceStrip / ChainStripZone: the chain-strip row (DESIGN.md's "Instrument Rack" anatomy).

One compact, fixed-height strip per chain step, one generated control per declared ``Param`` --
no widget is hand-written per device (the same bet ``knobs.py``/``knob_widgets.py`` settled).
Transform strips carry a cached/computing/error state dot; filter strips never do, because a
filter is always instant -- the dot itself is the "which kind is this" affordance, not a label.

**Closing the DragValue propose/confirm loop.** ``DragValue`` is a CONTROLLED component: its keyboard nudge, drag and inline-edit gestures only ever PROPOSE a
candidate value through ``valueChanged`` -- they never write it back to the widget's own display.
Only ``set_value()`` updates the label (``CycleButton``/``Toggle`` self-update on click, but
calling ``set_value`` on them too is harmless, so the strip treats every control uniformly). If
nothing downstream of a gesture ever calls ``set_value()``, the number on screen silently stops
matching the value actually in effect. For this slice, ``DeviceStrip`` IS that downstream: its
``paramChanged`` re-emit path calls ``set_value(value)`` on the originating control with the
applied value -- even when that value is identical to the raw proposal -- before telling anyone
else a param changed. The window only ever updates strips *wholesale*, on recompute; nothing
else closes this loop per-gesture, so if this call were dropped every drag/nudge would show a
stale reading until the next full recompute.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.device import Device, defaults_for, get_device, is_transform
from dynamix.shell.knob_widgets import make_control
from dynamix.shell.knobs import control_spec

#: Uniform strip height (DESIGN.md: "Uniform strip height"). One module constant, not a magic
#: number scattered across the layout code. Raised from 40 to 52 (carried forward): two rows -- the mini-label and the generated control -- need more than the
#: ~32px that 40 left available at the theme's 11pt base size, and were clipping.
STRIP_HEIGHT = 52

#: Content margins inside a strip's own layout. Kept generous enough that a click near the
#: top-left corner is guaranteed to land on the strip's own background rather than a generated
#: child control -- see ChainStripZone's docstring and its selection test.
_MARGINS = (8, 4, 8, 4)

_STATE_DOT_GLYPH = "●"        # ● -- colored entirely by QSS via the `state` property


class DeviceStrip(QtWidgets.QFrame):
    """One device's row: name, state dot (transforms only), one generated control per param each
    with a muted mini-label, and an honesty reading at the far right.

    ``controls`` is keyed by param name -- ``strip.controls["n_oct"]`` -- so a caller (or a test)
    can reach a specific generated widget without knowing the device's param order.
    """

    paramChanged = QtCore.Signal(str, object)
    selected = QtCore.Signal()

    def __init__(self, step_index: int, device: Device, params: dict, field=None, parent=None):
        super().__init__(parent)
        self._step_index = step_index
        self.device = device
        self._field = field
        self._params = params
        self.controls: dict[str, QtWidgets.QWidget] = {}
        self._derived_labels: dict[str, QtWidgets.QLabel] = {}

        self.setProperty("strip", "true")
        self.setProperty("selected", "false")
        self.setFixedHeight(STRIP_HEIGHT)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(*_MARGINS)

        row.addWidget(QtWidgets.QLabel(device.name))

        self.state_dot: QtWidgets.QLabel | None = None
        if is_transform(device):
            self.state_dot = QtWidgets.QLabel(_STATE_DOT_GLYPH)
            self.state_dot.setProperty("state", "idle")
            row.addWidget(self.state_dot)

        for p in device.params:
            column = QtWidgets.QVBoxLayout()
            mini = QtWidgets.QLabel(p.label or p.name)
            mini.setProperty("muted", "true")
            column.addWidget(mini)
            control = make_control(control_spec(p), params[p.name])
            # Same-thread GUI signal, default-arg trick to bind the param name at connect time --
            # the pattern demo.py already uses for this exact shape of fan-out connection. (The
            # "bound methods only" rule in worker.py is about cross-thread QThread signals, where
            # a lambda has no owning QObject for Qt to place the call on; nothing here crosses a
            # thread.)
            control.valueChanged.connect(
                lambda value, name=p.name: self._on_control_changed(name, value))
            column.addWidget(control)
            self.controls[p.name] = control
            row.addLayout(column)

            derived = self._derived_text(p.name, params[p.name])
            if derived is not None:
                derived_label = QtWidgets.QLabel(derived)
                derived_label.setProperty("reading", "true")
                derived_label.setProperty("muted", "true")
                self._derived_labels[p.name] = derived_label
                row.addWidget(derived_label)

        row.addStretch(1)

        self.reading_label = QtWidgets.QLabel("")
        self.reading_label.setProperty("reading", "true")
        self.reading_label.setProperty("muted", "true")
        row.addWidget(self.reading_label)

    @property
    def step_index(self) -> int:
        return self._step_index

    def _derived_text(self, name: str, value) -> str | None:
        """``device.derived_reading(name, value, field, params)`` if the device declares the
        optional method, else ``None`` -- a device with no such method (most of them) gets no
        label, at no cost beyond one ``getattr``. ``params`` is this strip's own dict, so a device
        can read e.g. its wavelet choice into the reading (``WTMM2D.derived_reading``)."""
        derived_reading = getattr(self.device, "derived_reading", None)
        if derived_reading is None:
            return None
        return derived_reading(name, value, self._field, self._params)

    def _on_control_changed(self, name: str, value) -> None:
        """A control proposed ``value``. Nothing in this slice re-quantizes it, so it IS the
        applied value -- but the control must still be told, via ``set_value``, or its own display
        never updates (see the module docstring). ``self._params`` is updated FIRST, before any
        label is recomputed -- a device's ``derived_reading`` may read a SIBLING param out of it
        (``WTMM2D``'s aₘᵢₙ line reads ``wavelet`` for λ), so every derived-reading label, not just
        the one beside this control, is refreshed off the now-current dict: otherwise flipping
        ``wavelet`` would leave the aₘᵢₙ line showing the old wavelet's λ until the whole strip is
        torn down and rebuilt."""
        self._params[name] = value
        self.controls[name].set_value(value)
        for label_name, label in self._derived_labels.items():
            text = self._derived_text(label_name, self._params[label_name])
            if text is not None:
                label.setText(text)
        self.paramChanged.emit(name, value)

    def set_selected(self, value: bool) -> None:
        self.setProperty("selected", "true" if value else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_state(self, state: str) -> None:
        """``"idle"|"computing"|"cached"|"error"``. No-op for a filter strip, which has no dot --
        a filter is always instant, so there is nothing for a state dot to say."""
        if self.state_dot is None:
            return
        self.state_dot.setProperty("state", state)
        self.style().unpolish(self.state_dot)
        self.style().polish(self.state_dot)

    def set_reading(self, text: str) -> None:
        """The honesty counter (e.g. ``"dropped 12"``, from ``_hchains_dropped`` etc.) -- always
        displayed, never hidden, muted only for the boring ``"dropped 0"`` case."""
        self.reading_label.setText(text)
        self.reading_label.setProperty("muted", "true" if text == "dropped 0" else "false")
        self.style().unpolish(self.reading_label)
        self.style().polish(self.reading_label)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.selected.emit()
        super().mousePressEvent(event)


class ChainStripZone(QtWidgets.QFrame):
    """One ``DeviceStrip`` per chain step, laid out left-to-right, with exactly one strip selected
    at a time.

    ``steps`` is any sequence of objects exposing ``.device`` (registry name) and ``.params``
    (values that override that device's declared defaults) -- ``dynamix.model.chain.DeviceRef``
    satisfies this without adaptation. ``field`` is the layer's own field, threaded straight
    through to every strip for its (optional) derived-reading label -- ``None`` is fine; a device
    with no ``derived_reading`` never looks at it, and one that has it handles a missing field the
    same way ``dynamix.shell.units.px_to_metres`` does (falls back to native/px, never crashes).
    """

    paramChanged = QtCore.Signal(int, str, object)

    def __init__(self, steps, field=None, parent=None):
        super().__init__(parent)
        self.setProperty("zone", "strip")
        self._strips: list[DeviceStrip] = []

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        for i, ref in enumerate(steps):
            device = get_device(ref.device)
            params = {**defaults_for(device), **ref.params}
            strip = DeviceStrip(i, device, params, field=field)
            strip.paramChanged.connect(self._on_strip_param_changed)
            strip.selected.connect(self._on_strip_selected)
            row.addWidget(strip)
            self._strips.append(strip)

        row.addStretch(1)

    def strip(self, i: int) -> DeviceStrip:
        return self._strips[i]

    def select(self, i: int) -> None:
        """Enforce single selection: ``set_selected`` on every strip, true only at index ``i``."""
        for s in self._strips:
            s.set_selected(s.step_index == i)

    def _on_strip_selected(self) -> None:
        # `sender()` identifies which strip emitted -- no closures needed, unlike the per-control
        # fan-out above, because there's exactly one connection per strip rather than one per
        # param.
        strip: DeviceStrip = self.sender()
        self.select(strip.step_index)

    def _on_strip_param_changed(self, name: str, value) -> None:
        strip: DeviceStrip = self.sender()
        self.paramChanged.emit(strip.step_index, name, value)
