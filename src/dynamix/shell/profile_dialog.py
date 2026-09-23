# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ProfileDialog: the transect value profile -- raw grey + smoothed accent curves, live smoothing
controls, equal-axes, PNG + ``.npz`` save (EQSelect's own
``_TransectDialog``: "Grey raw curve + crimson filtered curve... Live smoothing:
combo over transect.SMOOTHINGS... plus an 'Equal axes (1:1)' checkbox... Saves PNG + .npz").

**Figure construction.** One ``matplotlib.figure.Figure(constrained_layout=True)`` -- the same
colorbar/legend-ratchet-safe choice ``skeleton_dialog.py`` already made -- with a single
axes holding two ``Line2D``s: the raw sample (:data:`_RAW_COLOR`, thin) and the currently smoothed
curve (the theme's ``selection_accent``, thicker -- see "Accent color" below). Both share the same
x (``dist``, whatever unit the caller sampled in -- this dialog never knows or cares whether that
is bare pixels or a physical unit; the caller's axis label states it).

**Accent color -- same choice as ``skeleton_dialog.py``, for the same reason.** EQSelect's own
version of this dialog draws the filtered curve in ``crimson`` against a grey raw curve ("Grey raw curve + crimson filtered curve"). This module keeps EQSelect's own literal color
rather than importing ``dynamix.shell.theme`` for it -- the identical reasoning
``skeleton_dialog.py`` already documents (its own "Highlight color" section): pulling in ``theme``
would make this module's ONLY ``dynamix.shell.*`` dependency a single line color, on a module that
is otherwise exactly as self-contained as ``topology_panel.py``/``skeleton_dialog.py``, and would
widen ``devloop.py``'s theme-change reload set for no reading benefit -- a plain named matplotlib
color ("crimson", not a hex literal, so the shell's Theme Rule scan is unaffected: see
``theme.py``'s own module docstring) already reads unambiguously as "the curve that changed" next
to a muted grey raw curve.

**Smoothing controls** mirror EQSelect's own ranges: a
``dynamix.core.transect.SMOOTHINGS`` combo, sigma 0.5-50 (default 3.0), window 3-201 ODD (default
11, enforced by bumping an even manual entry up by one rather than rejecting it -- the least
surprising response to a spin box's own arrow keys, which only ever step by 2 from an odd seed,
disagreeing with a value typed directly).

**Equal axes (1:1 px).** A pure geometric aspect-ratio toggle (``Axes.set_aspect``) -- it makes NO
claim about what the y-axis unit means (EQSelect's own version is always km-vs-km, a physically
meaningful terrain aspect; DynamiX's transect samples an arbitrary field, so "1:1" here just means
"one x-unit occupies the same drawn width as one y-unit", exactly as the checkbox's own label says).

**Lazy matplotlib**, per ``right_panel.py``'s/``skeleton_dialog.py``'s own precedent: imported only
inside :meth:`_build_ui`, never at module scope.

**Lifecycle.** Independent per open -- ``MainWindow`` builds a NEW ``ProfileDialog`` on every Plot
click / row double-click (unlike ``SkeletonDialog``'s single-instance-reused rule): EQSelect itself
opens one dialog PER transect plot, so several profiles stay open and comparable
side by side. Each instance is parented to the main window (``parent=self`` at the call site),
which is what keeps it alive -- Qt's C++-side parent/child ownership, not a Python list this module
or ``MainWindow`` needs to maintain.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from dynamix.core.transect import SMOOTHINGS, smooth_profile

__all__ = ["ProfileDialog"]

#: Raw-sample curve color -- a plain, muted matplotlib gray (EQSelect's own "grey raw curve"). Not a theme role: it is deliberately NOT chrome (a data curve, the Theme Rule's
#: own carve-out for data identity -- ``canvas.py``'s palette constants document the identical
#: reasoning), and matplotlib's own named-gray string is the plain, honest way to say it.
_RAW_COLOR = "0.6"

#: Smoothed-curve color -- EQSelect's own literal choice for the identical curve ("crimson filtered curve"), kept as-is rather than resolved through ``dynamix.shell.theme`` --
#: see the module docstring's "Accent color" section for why.
_SMOOTH_COLOR = "crimson"

#: EQSelect's own smoothing-control ranges.
_SIGMA_RANGE = (0.5, 50.0)
_SIGMA_DEFAULT = 3.0
_WINDOW_RANGE = (3, 201)
_WINDOW_DEFAULT = 11


class ProfileDialog(QtWidgets.QDialog):
    """The value profile for one transect. ``dist``/``z`` are the RAW sample
    (:func:`dynamix.core.transect.sample_profile`'s own return) -- this dialog re-derives the
    smoothed curve itself, live, on every control change; it never re-samples the raster."""

    def __init__(self, dist, z, *, label: str = "", parent=None):
        # Qt.Window, not the QDialog default -- a real, independently resizable top-level window
        # with min/max/close chrome, same choice ``SkeletonDialog`` made and for the
        # identical reason: several of these may be open side by side (see the module docstring's
        # "Lifecycle" section), so each needs its own full window chrome, not modal-dialog chrome.
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)

        self._dist = np.asarray(dist, dtype=np.float64)
        self._z = np.asarray(z, dtype=np.float64)
        self.setWindowTitle(f"Transect profile — {label}" if label else "Transect profile")

        self._build_ui()
        self._redraw()

    # -- construction ---------------------------------------------------------------------------
    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self._figure = Figure(constrained_layout=True)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Smoothing"))
        self._smoothing_combo = QtWidgets.QComboBox()
        self._smoothing_combo.addItems(list(SMOOTHINGS))
        self._smoothing_combo.currentTextChanged.connect(lambda _text: self._redraw())
        controls.addWidget(self._smoothing_combo)

        controls.addWidget(QtWidgets.QLabel("σ"))
        self._sigma_spin = QtWidgets.QDoubleSpinBox()
        self._sigma_spin.setRange(*_SIGMA_RANGE)
        self._sigma_spin.setValue(_SIGMA_DEFAULT)
        self._sigma_spin.valueChanged.connect(lambda _value: self._redraw())
        controls.addWidget(self._sigma_spin)

        controls.addWidget(QtWidgets.QLabel("window"))
        self._window_spin = QtWidgets.QSpinBox()
        self._window_spin.setRange(*_WINDOW_RANGE)
        self._window_spin.setSingleStep(2)
        self._window_spin.setValue(_WINDOW_DEFAULT)
        self._window_spin.valueChanged.connect(self._on_window_changed)
        controls.addWidget(self._window_spin)

        self._equal_axes_check = QtWidgets.QCheckBox("Equal axes (1:1 px)")
        self._equal_axes_check.toggled.connect(self._on_equal_axes_toggled)
        controls.addWidget(self._equal_axes_check)
        controls.addStretch(1)
        layout.addLayout(controls)

        save_row = QtWidgets.QHBoxLayout()
        self._save_png_button = QtWidgets.QPushButton("Save PNG…")
        self._save_png_button.clicked.connect(self._on_save_png_clicked)
        save_row.addWidget(self._save_png_button)
        self._save_npz_button = QtWidgets.QPushButton("Save .npz…")
        self._save_npz_button.clicked.connect(self._on_save_npz_clicked)
        save_row.addWidget(self._save_npz_button)
        save_row.addStretch(1)
        layout.addLayout(save_row)

        self._raw_line, = self._ax.plot([], [], color=_RAW_COLOR, linewidth=1.0, label="raw")
        self._smooth_line, = self._ax.plot(
            [], [], color=_SMOOTH_COLOR, linewidth=1.6, label="smoothed")
        self._ax.legend(loc="best")
        self._ax.set_xlabel("distance")
        self._ax.set_ylabel("value")

        self.resize(760, 460)

    # -- redraw -----------------------------------------------------------------------------------
    def _current_smoothed(self) -> np.ndarray:
        return smooth_profile(self._z, self._smoothing_combo.currentText(),
                              self._sigma_spin.value(), self._window_spin.value())

    def _redraw(self) -> None:
        self._raw_line.set_data(self._dist, self._z)
        self._smooth_line.set_data(self._dist, self._current_smoothed())
        self._ax.relim()
        self._ax.autoscale_view()
        self._canvas.draw_idle()

    # -- gestures ---------------------------------------------------------------------------------
    def _on_window_changed(self, value: int) -> None:
        """Force the window ODD ("window spin (3–201 odd, default 11)") -- bumping an
        even value UP by one rather than refusing it, the least surprising response to a value
        typed directly (the arrow keys, stepping by 2 from an odd seed, can never produce an even
        one in the first place)."""
        if value % 2 == 0:
            self._window_spin.blockSignals(True)
            self._window_spin.setValue(value + 1)
            self._window_spin.blockSignals(False)
        self._redraw()

    def _on_equal_axes_toggled(self, checked: bool) -> None:
        self._ax.set_aspect(1.0 if checked else "auto", adjustable="box")
        self._canvas.draw_idle()

    def _on_save_png_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save profile PNG", "", "PNG image (*.png)")
        if path:
            self._figure.savefig(path)

    def _on_save_npz_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save profile data", "", "NumPy archive (*.npz)")
        if path:
            np.savez(path, dist=self._dist, z=self._z, z_filt=self._current_smoothed(),
                    smoothing=self._smoothing_combo.currentText(),
                    sigma=self._sigma_spin.value(), window=self._window_spin.value())
