# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""RightPanel: the third work-split slot -- the selected dataset's controls, sectioned.

A ``QScrollArea``, not a fixed column: DESIGN.md's per-selection controls accumulate over the
life of this branch -- Display now, the arrangement's mask and group palette later (Tasks 4/5/9)
-- and a fixed layout would eventually run out of room on a laptop screen. ``add_section`` is the
one way anything lands here; the "Display" section uses it, and every later section is added through the identical call, so this module never needs to know what a mask row or a
group palette IS.

**Dependency injection, not an import.** The three Display knobs are built from
``MainWindow._DISPLAY_PARAMS``, but ``main_window.py`` is what constructs a ``RightPanel`` --
importing it back from here would be circular. The same precedent already governs
``DeviceBrowser.set_presets`` (``main_window.py``'s own comment on that call): the params come in
as data through the constructor, and this module stays ignorant of where they came from.

**Wheel discipline.** A ``DragValue`` (or a future slider/combo a later task's section adds) can
eat a wheel event meant for the panel's own scrollbar -- the exact bug EQSelect's
``_PanelWheelFilter`` (``app_window.py:1085-1103``) exists to prevent. ``_WheelRedirect`` below is
DynamiX's own version of the same idea: one instance per panel, installed on the viewport and on
every widget any section ever adds (current or future), so a wheel over ANY control scrolls the
panel instead of nudging a value -- the only way to adjust a knob from in here stays drag, click,
arrow keys or typing, exactly as EQSelect's own docstring states it.
"""
from __future__ import annotations

import numpy as np
from dynamix.core.stretch import STRETCHES
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.shell.canvas import EXTREMA_COLOR, HCHAIN_COLOR, VTRAIL_COLOR
from dynamix.shell.knob_widgets import make_control
from dynamix.shell.knobs import control_spec

__all__ = ["RightPanel"]

#: Minimum width: narrow enough that the splitter's widening gesture still means
#: something, wide enough that a DragValue's mono reading never clips. No maximum is ever set --
#: the splitter boundary, not this panel, is what makes it wider.
_MIN_WIDTH = 200

#: The colormap combo's 20 entries -- sequential, diverging, cyclic
#: and terrain families, ported from EQSelect's own dropdown (``app_window.py:186-191``'s
#: ``_CMAPS``) rather than imported: this widget sits outside the ``dynamix/core`` verbatim-copy
#: contract, so it is DynamiX's own code, the same LIST of names.
_CMAPS = (
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "coolwarm", "RdBu", "Spectral", "seismic", "bwr",
    "gist_earth", "terrain", "ocean", "cubehelix",
    "viridis_r", "plasma_r", "jet", "twilight", "hsv",
)


def _hex(rgb: tuple[int, int, int]) -> str:
    """``(r, g, b)`` -> a lowercase ``"#rrggbb"`` string -- swatch faces start at the canvas's OWN
    defaults (``canvas.py``'s ``HCHAIN_COLOR``/etc.), the same derivation ``main_window.py``'s own
    ``_hex`` performs off the identical constants, so a re-tint of the canvas module moves both
    for free with no second literal to drift out of sync."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


#: key (matches ``_display_style_of``'s dict and ``ui.<name>`` tag), button label, canvas default.
_SWATCH_SPECS = (
    ("color_hchain", "H chains", HCHAIN_COLOR),
    ("color_vtrail", "V trails", VTRAIL_COLOR),
    ("color_extrema", "Extrema", EXTREMA_COLOR),
)


def _cmap_pixmap(name: str, w: int = 72, h: int = 14) -> QtGui.QPixmap:
    """A ``w``x``h`` horizontal gradient ``QPixmap`` sampled ONCE from matplotlib's colormap
    table for ``name`` -- EQSelect's ``_cmap_pixmap`` (``app_window.py:244-251``), ported. The
    matplotlib import is lazy and lives HERE, the one place this module ever touches it: shell-only,
    and matplotlib is already a dependency of the pyvista-backed viz stack, but no shell module
    imports it at top level."""
    import matplotlib
    rgb = (matplotlib.colormaps[name](np.linspace(0.0, 1.0, w))[:, :3] * 255).astype(np.uint8)
    row = np.ascontiguousarray(np.repeat(rgb[None, :, :], h, axis=0))     # (h, w, 3), contiguous
    qimg = QtGui.QImage(row.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)  # QImage copies
    return QtGui.QPixmap.fromImage(qimg)


def _all_cmap_names() -> tuple:
    """The curated set first (EQSelect's own dropdown, unchanged defaults), a separator,
    then EVERY other matplotlib colormap -- reversed (``_r``) variants excluded from the long
    tail to halve the list; any ``_r`` name still works when typed into a project file,
    and the curated set keeps its own ``viridis_r``/``plasma_r`` entries."""
    import matplotlib
    rest = sorted(n for n in matplotlib.colormaps
                  if n not in _CMAPS and not n.endswith("_r"))
    return _CMAPS + (None,) + tuple(rest)


def _make_cmap_combo(names=None) -> QtWidgets.QComboBox:
    """A ``QComboBox`` whose entries each carry a gradient icon beside the colormap's name --
    built with ``currentTextChanged`` left UNCONNECTED: the caller wires it only after every item
    has been added, since connecting first would fire once per ``addItem`` as the box's first
    entry seeds ``currentIndex``."""
    combo = QtWidgets.QComboBox()
    combo.setIconSize(QtCore.QSize(72, 14))
    for name in (_all_cmap_names() if names is None else names):
        if name is None:
            combo.insertSeparator(combo.count())
            continue
        combo.addItem(QtGui.QIcon(_cmap_pixmap(name)), name)
    combo.setMaxVisibleItems(24)              # the long tail scrolls; typing jumps by name
    return combo


class _WheelRedirect(QtCore.QObject):
    """A wheel event over any watched widget scrolls ``scroll_area`` instead of reaching that
    widget -- EQSelect's ``_PanelWheelFilter`` (see module docstring), ported. One instance per
    panel, installed on the viewport and on every control ``add_section`` ever adds."""

    def __init__(self, scroll_area: QtWidgets.QScrollArea):
        super().__init__(scroll_area)
        self._scroll = scroll_area

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Wheel:
            bar = self._scroll.verticalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y())
            return True                              # consumed: never reaches the control
        return super().eventFilter(obj, event)


class RightPanel(QtWidgets.QScrollArea):
    """The selected dataset's controls, sectioned. ``styleChanged`` carries a Display-knob edit
    out to whoever owns the layer (``MainWindow._on_display_style_changed``, unchanged);
    ``set_style_values`` is the reverse, silent sync a layer switch uses."""

    #: A Display knob moved -- (param name, new value). ``MainWindow`` connects this straight to
    #: its existing ``_on_display_style_changed``, replacing the per-control lambda that used to
    #: do the same job when the three knobs lived in the left panel.
    styleChanged = QtCore.Signal(str, object)
    #: "3-D surface…" clicked -- main_window opens the SurfaceDialog (it owns the layer list;
    #: the resulting choice comes back through the ordinary styleChanged tag-filing path).
    surfaceDialogRequested = QtCore.Signal()
    #: "Slice…" clicked -- main_window opens the LevelsDialog over the DISPLAYED raster's
    #: values (only it knows whether that is the field or a holder_map h-map right now).
    levelsDialogRequested = QtCore.Signal()
    #: "Reconstruct…" clicked -- main_window opens the BandDialog over the result's h-map
    #: (analysis, not display -- the 2026-09-16 split; it needs a holder/band result to exist).
    bandDialogRequested = QtCore.Signal()

    def __init__(self, display_params, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(_MIN_WIDTH)               # no setMaximumWidth -- see module docstring

        content = QtWidgets.QWidget()
        self._column = QtWidgets.QVBoxLayout(content)
        self._column.setContentsMargins(6, 6, 6, 6)
        self._column.addStretch(1)          # sections insert above this -- keeps them top-packed
        self.setWidget(content)

        self._wheel_filter = _WheelRedirect(self)
        self.viewport().installEventFilter(self._wheel_filter)

        #: name -> control widget. Same registry name and shape MainWindow has always exposed as
        #: ``self._display_controls`` -- built here now, aliased back onto the window unchanged
        #: (``main_window.py``'s ``__init__``: ``self._display_controls =
        #: self.right_panel._display_controls``).
        self._display_controls: dict[str, QtWidgets.QWidget] = {}
        self._build_display_section(display_params)

    # -- sections ----------------------------------------------------------------------------
    def add_section(self, title: str, widget: QtWidgets.QWidget, *, view_scoped: bool = False
                    ) -> None:
        """Append a titled section holding ``widget``.

        ``view_scoped`` only ever adds a divider line above the section, visually setting it apart
        from a section like Display that reads as always live -- it does NOT hide, disable, or
        otherwise gate the section's applicability to the active center-stack view; there is no
        visibility-gating consumer of this flag anywhere (confirmed by grep; see
        ``main_window.py``'s own "Groups" section, whose comment records this exact correction).
        Callers choose it purely as a labeling/grouping cue for whichever sections they consider
        conceptually tied to the arrangement view (the mask section, e.g.).
        """
        frame = QtWidgets.QFrame()
        frame.setProperty("view_scoped", view_scoped)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(4, 6, 4, 6)
        if view_scoped:
            divider = QtWidgets.QFrame()
            divider.setFrameShape(QtWidgets.QFrame.HLine)
            layout.addWidget(divider)
        heading = QtWidgets.QLabel(title)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)
        layout.addWidget(widget)
        self._protect_from_wheel(frame)
        self._column.insertWidget(self._column.count() - 1, frame)   # before the trailing stretch

    def _protect_from_wheel(self, widget: QtWidgets.QWidget) -> None:
        """Install the wheel filter on ``widget`` and every current descendant -- a section built
        elsewhere (``main_window.py``, a future task's own module) hands in a widget this panel
        has never seen before, so the filter cannot rely on having been installed at construction
        time the way the viewport's own is."""
        widget.installEventFilter(self._wheel_filter)
        for child in widget.findChildren(QtWidgets.QWidget):
            child.installEventFilter(self._wheel_filter)

    # -- Display section (relocated from the left panel, plumbing unchanged) ---------
    def _build_display_section(self, display_params) -> None:
        """The Display knobs, built exactly as they were in the left panel's old
        ``display_row`` -- same ``make_control(control_spec(p), p.default)`` call, same
        mini-label-over-control column, same registry dict. Only the parent (a section frame
        here, a bare ``QFrame`` there) and the outward wiring (``styleChanged`` instead of a
        direct ``_on_display_style_changed`` lambda) changed.

        Three more rows follow these three knobs, in this
        SAME section (``add_section`` only ever takes one widget, so everything below is packed
        into one ``container`` first): the colormap combo, the three color swatches, and the
        trails checkbox. All four groups ride the identical ``styleChanged`` signal into
        ``MainWindow._on_display_style_changed`` -- no second data path.
        """
        container = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QFrame()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for p in display_params:
            knob_column = QtWidgets.QVBoxLayout()
            mini = QtWidgets.QLabel(p.label or p.name)
            mini.setProperty("muted", "true")
            knob_column.addWidget(mini)
            control = make_control(control_spec(p), p.default)
            control.valueChanged.connect(
                lambda value, name=p.name: self.styleChanged.emit(name, value))
            knob_column.addWidget(control)
            self._display_controls[p.name] = control
            row_layout.addLayout(knob_column)
        column.addWidget(row)

        column.addWidget(self._build_cmap_row())
        column.addWidget(self._build_stretch_row())
        column.addWidget(self._build_swatch_row())
        self.trails_check = QtWidgets.QCheckBox("Show V-trails")
        self.trails_check.toggled.connect(
            lambda checked: self.styleChanged.emit("show_trails", checked))
        column.addWidget(self.trails_check)
        # Hillshade (2026-08-29): the raster shaded by the Sun az / Sun alt / Vert. exag. knobs
        # above, on the canvas and on the drape. Same styleChanged path as everything here.
        self.hillshade_check = QtWidgets.QCheckBox("Hillshade")
        self.hillshade_check.toggled.connect(
            lambda checked: self.styleChanged.emit("hillshade", checked))
        column.addWidget(self.hillshade_check)
        # 3-D surface (2026-08-29; 2026-09-16 user redesign): in the Vector/Globe views the
        # raster stands up with z = a chosen HEIGHT SOURCE -- this layer's own values, or
        # another loaded raster's (the drape case: an h(x) layer standing on a DEM). The whole
        # choice (off / same / other+which) lives in a dialog main_window owns (it knows the
        # layer list; this panel deliberately does not), opened from this button; the button
        # label mirrors the current state via set_style_values.
        self.surface_button = QtWidgets.QPushButton("3-D surface…")
        self.surface_button.clicked.connect(self.surfaceDialogRequested.emit)
        column.addWidget(self.surface_button)
        self.depth_check = QtWidgets.QCheckBox("Depth positive (flip sign)")
        self.depth_check.toggled.connect(
            lambda checked: self.styleChanged.emit("depth_positive", checked))
        column.addWidget(self.depth_check)

        self.add_section("Display", container)

    def _build_cmap_row(self) -> QtWidgets.QWidget:
        """The colormap combo, mini-labelled like the three knobs above it. Built with all 20
        entries in place before ``currentTextChanged`` is connected (see ``_make_cmap_combo``'s
        own docstring), so populating it never itself proposes an edit."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        mini = QtWidgets.QLabel("Colormap")
        mini.setProperty("muted", "true")
        layout.addWidget(mini)
        self.cmap_combo = _make_cmap_combo()
        self.cmap_combo.currentTextChanged.connect(
            lambda name: self.styleChanged.emit("colormap", name))
        layout.addWidget(self.cmap_combo)
        return row

    def _build_stretch_row(self) -> QtWidgets.QWidget:
        """Stretch (2026-08-29): which contrast stretch colours the raster (``core.stretch``);
        the Clip % knob above feeds the ``percent`` mode. Same styleChanged path as the colormap."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        mini = QtWidgets.QLabel("Stretch")
        mini.setProperty("muted", "true")
        layout.addWidget(mini)
        self.stretch_combo = QtWidgets.QComboBox()
        for name in STRETCHES:
            self.stretch_combo.addItem(name)
        self.levels_edit = QtWidgets.QLineEdit()
        self.levels_edit.setPlaceholderText("levels: 5  or  -0.5, 0, 0.8")
        self.levels_edit.setToolTip(
            "Density slice (ENVI): a class count (quantile slices) or comma-separated data-unit "
            "breaks — e.g. h boundaries. Replaces the stretch while set; empty = off.")
        self.levels_edit.editingFinished.connect(
            lambda: self.styleChanged.emit("levels", self.levels_edit.text().strip()))
        self.stretch_combo.currentTextChanged.connect(
            lambda name: self.styleChanged.emit("stretch", name))
        layout.addWidget(self.stretch_combo)
        levels_row = QtWidgets.QHBoxLayout()
        levels_row.addWidget(self.levels_edit, 1)
        self.slice_button = QtWidgets.QPushButton("Slice…")
        self.slice_button.setToolTip("Edit class bounds and colors on a histogram of the "
                                     "displayed values (ENVI density slice — display only)")
        self.slice_button.clicked.connect(self.levelsDialogRequested.emit)
        levels_row.addWidget(self.slice_button)
        self.band_button = QtWidgets.QPushButton("Reconstruct…")
        self.band_button.setToolTip("Pick an h band on the exponent histogram and reconstruct "
                                    "from it (band_recon — analysis, separate from coloring)")
        self.band_button.clicked.connect(self.bandDialogRequested.emit)
        levels_row.addWidget(self.band_button)
        layout.addLayout(levels_row)
        return row

    def _build_swatch_row(self) -> QtWidgets.QWidget:
        """The three color swatches -- a plain ``QPushButton`` per key, its face colored via
        stylesheet, its text carrying the label (no separate mini-label column: the design's "labels H chains / V trails / Extrema" names the BUTTON text, not a widget above it)."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        self._swatch_buttons: dict[str, QtWidgets.QPushButton] = {}
        #: The color a swatch would open ``QColorDialog`` on next -- kept alongside the button's
        #: own stylesheet (rather than parsed back out of it) so ``_on_swatch_clicked`` never has
        #: to reverse-parse CSS.
        self._swatch_colors: dict[str, str] = {}
        for key, label, default_rgb in _SWATCH_SPECS:
            default_hex = _hex(default_rgb)
            button = QtWidgets.QPushButton(label)
            button.setStyleSheet(f"background-color: {default_hex}")
            button.clicked.connect(lambda checked=False, k=key: self._on_swatch_clicked(k))
            self._swatch_buttons[key] = button
            self._swatch_colors[key] = default_hex
            layout.addWidget(button)
        return row

    def _pick_color(self, initial: str) -> str | None:
        """Wraps ``QColorDialog.getColor`` -- the ONE seam through which a swatch click ever
        reaches the real (blocking, platform-native) dialog. Tests monkeypatch this method
        directly rather than drive the dialog itself; nothing else in this class opens it."""
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(initial), self)
        return color.name() if color.isValid() else None

    def _on_swatch_clicked(self, key: str) -> None:
        chosen = self._pick_color(self._swatch_colors[key])
        if chosen is None:                      # dialog cancelled -- nothing changes, nothing emits
            return
        self._swatch_colors[key] = chosen
        self._swatch_buttons[key].setStyleSheet(f"background-color: {chosen}")
        self.styleChanged.emit(key, chosen)

    # -- sync ----------------------------------------------------------------------------------
    def set_style_values(self, style: dict) -> None:
        """Non-emitting: show ``style`` on every Display control without proposing an edit -- the
        ``_sync_display_controls`` hook a layer switch uses. Safe by ``DragValue.set_value``'s
        own guard-against-feedback-loop contract (``knob_widgets.py``): it updates the display
        only, never calling back into ``valueChanged``. The combo and checkbox extend the same
        contract by hand (``blockSignals`` around the programmatic set); the swatches need no
        such guard -- ``setStyleSheet`` emits nothing Qt-side to begin with."""
        for name, control in self._display_controls.items():
            if name in style:               # a partial style (tests, older callers) leaves a knob as is
                control.set_value(style[name])
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.setCurrentText(style["colormap"])
        self.cmap_combo.blockSignals(False)
        for key, button in self._swatch_buttons.items():
            hex_value = style[key]
            button.setStyleSheet(f"background-color: {hex_value}")
            self._swatch_colors[key] = hex_value
        self.trails_check.blockSignals(True)
        self.trails_check.setChecked(style["show_trails"])
        self.trails_check.blockSignals(False)
        self.hillshade_check.blockSignals(True)
        self.hillshade_check.setChecked(bool(style.get("hillshade", False)))
        self.hillshade_check.blockSignals(False)
        self.stretch_combo.blockSignals(True)
        self.stretch_combo.setCurrentText(style.get("stretch", "linear"))
        self.stretch_combo.blockSignals(False)
        self.levels_edit.blockSignals(True)
        self.levels_edit.setText(style.get("levels", ""))
        self.levels_edit.blockSignals(False)
        self.depth_check.blockSignals(True)
        self.depth_check.setChecked(bool(style.get("depth_positive", False)))
        self.depth_check.blockSignals(False)
        # The surface button mirrors state in its label (a button has no checked state to sync;
        # the "other" arm shows the source generically -- only main_window knows layer names).
        if not style.get("surface", False):
            label = "3-D surface: off…"
        elif style.get("surface_source", "same") == "same":
            label = "3-D surface: same dataset…"
        else:
            label = "3-D surface: other dataset…"
        self.surface_button.setText(label)


class CompositePanel(QtWidgets.QWidget):
    """The multiband composite mixer: one row per band with its channel assignment (R/G/B)
    and DAW-style Solo / Mute buttons. Emits ``compositeChanged`` with the canvas's
    ``set_composite`` spec; :meth:`set_bands` rebuilds silently from stored state. A channel
    feeds from at most one band -- assigning it steals it from whichever band held it."""

    compositeChanged = QtCore.Signal(dict)
    _CHANNELS = ("—", "R", "G", "B")
    #: The composite's OWN stretch (not the Display section's): one type for every channel,
    #: each channel computed from its own band. (mode, label, value it uses)
    _STRETCH_MODES = (("linear", "linear (min–max)", None),
                      ("percent", "percent clip", "pct"),
                      ("stddev", "std-dev", "k"),
                      ("log", "log", None),
                      ("histogram", "histogram equalize", None),
                      ("bipolar", "bipolar (about 0)", "pct"))

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Stretch"))
        self.stretch_combo = QtWidgets.QComboBox()
        for mode, label, _v in self._STRETCH_MODES:
            self.stretch_combo.addItem(label, mode)
        self.stretch_value = QtWidgets.QDoubleSpinBox()
        self.stretch_value.setDecimals(1)
        self.stretch_value.setSingleStep(0.5)
        row.addWidget(self.stretch_combo, 1)
        row.addWidget(self.stretch_value)
        outer.addLayout(row)
        self._grid = QtWidgets.QGridLayout()
        outer.addLayout(self._grid)
        self._rows: list = []
        self._loading = False
        self._pct, self._k = 2.0, 2.0
        self.stretch_combo.currentIndexChanged.connect(self._on_stretch_mode)
        self.stretch_value.valueChanged.connect(self._on_stretch_value)
        self._sync_value_box()

    def _value_kind(self) -> "str | None":
        mode = self.stretch_combo.currentData()
        return next(v for m, _l, v in self._STRETCH_MODES if m == mode)

    def _sync_value_box(self) -> None:
        """The one value box shows the percent (percent clip, bipolar) or k (std-dev) the
        current type uses, and hides for the types that use none."""
        kind = self._value_kind()
        self.stretch_value.blockSignals(True)
        if kind == "pct":
            self.stretch_value.setRange(0.0, 49.9)
            self.stretch_value.setSuffix(" %")
            self.stretch_value.setValue(self._pct)
        elif kind == "k":
            self.stretch_value.setRange(0.1, 10.0)
            self.stretch_value.setSuffix(" σ")
            self.stretch_value.setValue(self._k)
        self.stretch_value.blockSignals(False)
        self.stretch_value.setVisible(kind is not None)

    def _on_stretch_mode(self, *_a) -> None:
        self._sync_value_box()
        self._emit()

    def _on_stretch_value(self, value: float) -> None:
        if self._value_kind() == "k":
            self._k = float(value)
        else:
            self._pct = float(value)
        self._emit()

    def set_bands(self, names, spec: dict) -> None:
        """One row per band, seeded from ``spec`` (non-emitting)."""
        self._loading = True
        self._pct = float(spec.get("stretch_pct", 2.0) if spec.get("stretch_pct") is not None
                          else 2.0)
        self._k = float(spec.get("stretch_k", 2.0) if spec.get("stretch_k") is not None
                        else 2.0)
        mode_idx = self.stretch_combo.findData(spec.get("stretch") or "percent")
        self.stretch_combo.blockSignals(True)
        self.stretch_combo.setCurrentIndex(max(mode_idx, 0))
        self.stretch_combo.blockSignals(False)
        self._sync_value_box()
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = []
        chan_of = {}
        for c in ("r", "g", "b"):
            b = spec.get(c)
            if isinstance(b, int):
                chan_of.setdefault(b, c.upper())
        solo = set(spec.get("solo") or ())
        mute = set(spec.get("mute") or ())
        for i, name in enumerate(names):
            label = QtWidgets.QLabel(str(name))
            combo = QtWidgets.QComboBox()
            combo.addItems(self._CHANNELS)
            combo.setCurrentText(chan_of.get(i, "—"))
            combo.currentTextChanged.connect(
                lambda text, row=i: self._on_channel_changed(row, text))
            s_btn = QtWidgets.QToolButton()
            s_btn.setText("S")
            s_btn.setCheckable(True)
            s_btn.setChecked(i in solo)
            s_btn.setToolTip("Solo — show only this band (grayscale when it is the only one)")
            m_btn = QtWidgets.QToolButton()
            m_btn.setText("M")
            m_btn.setCheckable(True)
            m_btn.setChecked(i in mute)
            m_btn.setToolTip("Mute — silence this band's channel in the composite")
            s_btn.toggled.connect(self._emit)
            m_btn.toggled.connect(self._emit)
            self._grid.addWidget(label, i, 0)
            self._grid.addWidget(combo, i, 1)
            self._grid.addWidget(s_btn, i, 2)
            self._grid.addWidget(m_btn, i, 3)
            self._rows.append((combo, s_btn, m_btn))
        self._loading = False

    def _on_channel_changed(self, row: int, text: str) -> None:
        if self._loading:
            return
        if text != "—":                       # a channel feeds from at most one band
            self._loading = True
            for i, (combo, _s, _m) in enumerate(self._rows):
                if i != row and combo.currentText() == text:
                    combo.setCurrentText("—")
            self._loading = False
        self._emit()

    def spec(self) -> dict:
        out = {"r": None, "g": None, "b": None, "solo": [], "mute": [],
               "stretch": self.stretch_combo.currentData(), "stretch_pct": self._pct,
               "stretch_k": self._k}
        for i, (combo, s_btn, m_btn) in enumerate(self._rows):
            text = combo.currentText()
            if text != "—":
                out[text.lower()] = i
            if s_btn.isChecked():
                out["solo"].append(i)
            if m_btn.isChecked():
                out["mute"].append(i)
        return out

    def _emit(self, *_a) -> None:
        if not self._loading:
            self.compositeChanged.emit(self.spec())
