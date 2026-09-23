# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ModeRow: the arrangement view's projection-mode segmented control ("Projection mode switch: corner segmented control (pacific / greenwich / mercator / globe),
mercator default. Mode switch is the one legitimate full re-projection/rebuild.").

**Gap, closed.** Nothing had built this control -- ``camera.py``'s own module docstring ("No mode-switcher UI exists yet")
and ``view.py``'s own ("No projection-mode-switcher UI exists in this codebase yet") both recorded
the gap honestly rather than silently leave it. This closes it: four checkable ``QToolButton``s in
one exclusive ``QButtonGroup``, mirroring ``layer_panel.py``'s own H/L/F row (the established shape
for "a small set of mutually-relevant checkable buttons in one row" in this codebase) rather than
introducing a second widget idiom for what is, structurally, the same shape -- a single
``CycleButton`` (the knob idiom ``ParamKind.CHOICE`` earns, ``knobs.py``/``knob_widgets.py``) steps
through one option at a time and hides the other three, which reads further from the spec's own
"segmented control" wording (every option visible, one click reaches any of them) than a plain
button row does.

Emits ONE ``modeChanged(str)`` signal whenever the checked button actually changes (never on the
paired "unchecked" toggle the exclusive group fires on the button that lost the check, and never on
a re-click of the already-active mode -- ``QButtonGroup``'s own exclusivity makes the latter a
no-op click already, but the guard below is explicit rather than incidental). ``ArrangementView``
wires this straight into ``Scene.set_mode`` -- lazily, the same no-op-before-a-scene-exists
contract ``MaskRow.maskChanged``/``GroupPalette.membershipChanged`` already use -- plus
``MomentumCamera.note_mode`` (see that method's own docstring), so switching modes through this
control keeps ``_last_flat_mode`` current the way ``camera.py``'s own module docstring says a
future mode-switcher must.

Pure Qt -- no pyvista import here, same reasoning as ``mask_row.py``'s own module docstring
(``dynamix[viz]``/``dynamix[gui]`` are separate optional-dependency groups; this widget only ever
needs the latter). ``dynamix.core.projection.MODES`` (Qt-free) is the one import from outside
``dynamix.shell`` -- the same "read the mode list from its own source of truth, never re-typed"
discipline ``scene.py``'s own ``set_mode`` validation already follows.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.core.projection import MODES

#: the design's own landing mode -- matches ``scene.py``'s own ``DEFAULT_MODE`` (deliberately NOT
#: ``dynamix.core.projection.DEFAULT_MODE``, which is "pacific" for an unrelated reason -- see that
#: module's own docstring).
DEFAULT_MODE = "mercator"


class ModeRow(QtWidgets.QFrame):
    """Four-way projection-mode segmented control. Session results are never touched
    -- moving this only ever leads to ``Scene.set_mode``/``MomentumCamera.note_mode``, never a
    device/resolve path."""

    modeChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self._mode = DEFAULT_MODE
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QtWidgets.QToolButton] = {}
        for mode in MODES:
            button = QtWidgets.QToolButton()
            button.setCheckable(True)
            button.setText(mode)
            button.setToolTip(f"Projection: {mode}")
            button.setChecked(mode == DEFAULT_MODE)
            self._group.addButton(button)
            button.toggled.connect(lambda checked, m=mode: self._on_toggled(m, checked))
            layout.addWidget(button)
            self._buttons[mode] = button

    @property
    def mode(self) -> str:
        """The current, confirmed mode -- matches whichever button is checked."""
        return self._mode

    def _on_toggled(self, mode: str, checked: bool) -> None:
        """``QButtonGroup``'s exclusivity fires ``toggled`` TWICE per click -- ``False`` on the
        button that just lost the check, ``True`` on the one that gained it. Only the latter is a
        real mode change; the former is filtered here rather than left to the caller to notice
        ``modeChanged`` fired with a value equal to what it already had."""
        if not checked or mode == self._mode:
            return
        self._mode = mode
        self.modeChanged.emit(mode)
