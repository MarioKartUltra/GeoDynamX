# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""MaskRow: the arrangement view's mask controls (the design
"Exploration is view-state": "a small mask row (modulus percentile, scale range) filtering the
projected actors only, via visibility/mask updates").

Three knobs, generated through the SAME control_spec/make_control machinery every device param
uses -- ``main_window.py``'s per-layer display row (~line 340, "Per-layer display controls") is
the precedent this mirrors, right down to the "muted label above a drag_value" column shape. No
hand-built styling widget. Emits ONE ``maskChanged(dict)`` signal carrying all three current,
CONFIRMED values every time any one of them moves; ``ArrangementView`` wires that straight into
``Scene.set_mask(**payload)`` (lazily -- only once a scene actually exists).

Pure Qt -- no pyvista import here, deliberately. ``dynamix[viz]`` (pyvista/pyvistaqt) and
``dynamix[gui]`` (PySide6/pyqtgraph) are SEPARATE optional-dependency groups (``pyproject.toml``),
and ``dynamix.shell.arrangement.scene.Scene`` is Qt-free in behaviour specifically so it can be
exercised with just the ``viz`` extra installed. This widget is the mirror-image case -- Qt-only,
no pyvista -- for the same reason: keeping the two halves of the arrangement package decoupled
means neither one silently drags the other's dependency group in.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.model.param import Param, ParamKind
from dynamix.shell.knob_widgets import make_control
from dynamix.shell.knobs import control_spec

#: The three MaskRow params, in display order -- ``Scene.set_mask(modulus_pctl, scale_lo,
#: scale_hi)``'s exact argument names, so ``MaskRow.values()``/every ``maskChanged`` payload can
#: be forwarded with a plain ``**payload``.
#:
#: ``scale_lo``/``scale_hi`` bound a chain's own DEPTH (how many scales its ridge reaches) -- a
#: generous static ceiling (``max=255``) rather than a real per-result scale count, which this
#: generic control has no way to know (``Scene.set_mask`` interprets the values against each
#: layer's own chains itself, see its docstring). ``scale_hi``'s default/label follow
#: ``dynamix.devices.filters.HLineLength``'s own ``max_len`` convention exactly: ``0`` means "no
#: cap", not a real depth of zero.
_MASK_PARAMS = (
    Param("modulus_pctl", ParamKind.FLOAT, default=0.0, min=0.0, max=100.0,
          label="Modulus ≥ pctl", units="%"),
    Param("scale_lo", ParamKind.INT, default=0, min=0, max=255,
          soft_min=0, soft_max=15, label="Min scales spanned"),
    Param("scale_hi", ParamKind.INT, default=0, min=0, max=255,
          soft_min=0, soft_max=15, label="Max scales spanned (0 = no cap)"),
)


class MaskRow(QtWidgets.QFrame):
    """Modulus-percentile + scale-range view-state controls. Session results are never touched
-- moving these only ever leads to ``Scene.set_mask``, never a device/resolve
    path."""

    maskChanged = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self._values: dict[str, object] = {p.name: p.default for p in _MASK_PARAMS}
        self._controls: dict[str, QtWidgets.QWidget] = {}
        for p in _MASK_PARAMS:
            column = QtWidgets.QVBoxLayout()
            mini = QtWidgets.QLabel(p.label or p.name)
            mini.setProperty("muted", "true")
            column.addWidget(mini)
            control = make_control(control_spec(p), p.default)
            control.valueChanged.connect(
                lambda value, name=p.name: self._on_control_changed(name, value))
            column.addWidget(control)
            self._controls[p.name] = control
            layout.addLayout(column)

    def values(self) -> dict:
        """The current, confirmed ``{"modulus_pctl", "scale_lo", "scale_hi"}`` -- a copy, so a
        caller (e.g. ``ArrangementView``, applying the current state to a scene that did not exist
        yet the last time ``maskChanged`` fired) can read it without holding a reference in."""
        return dict(self._values)

    def _on_control_changed(self, name: str, value) -> None:
        param = next(p for p in _MASK_PARAMS if p.name == name)
        validated = param.validate(value)
        self._values[name] = validated
        # DragValue never mutates its own displayed value on a proposal (see its docstring) --
        # the owner confirms it back. Nothing here re-quantizes the proposal, so this simply
        # confirms exactly what was proposed.
        self._controls[name].set_value(validated)
        self.maskChanged.emit(dict(self._values))
