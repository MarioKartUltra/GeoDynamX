# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ViewDialog: the arrangement view's "View…" button opens this. Non-modal, tabbed,
settings-persisted -- the four control groups retired from ``ArrangementView``'s own face
(``ModeRow``) plus three new capabilities (camera readout/reset, graticule, vertical exaggeration +
background) land here, one tab each.

**Offscreen-safe by holding a real reference, not by staying decoupled.** Unlike ``MaskRow``/
``GroupPalette`` (which never reference ``ArrangementView`` at all -- ``MainWindow`` bridges their
signals into it instead, see ``arrangement/mask_row.py``'s own module docstring), this dialog is
constructed WITH a real ``ArrangementView`` and calls its passthrough methods directly:
``view.set_mode``/``set_graticule``/``set_vertical_exaggeration``/``set_background``/
``camera_state``/``set_camera_state``/``reset_camera``. Every one of those already no-ops (or
returns ``None``) when no interactor has been built (``view.py``'s own guard pattern -- this
harness's mandated offscreen QPA platform never builds one, so every test in this suite exercises
that exact path), so holding the reference costs nothing: this dialog constructs and every one of
its controls is fully interactive with no live interactor at all. Tests here pass a lightweight
stub in place of a real ``ArrangementView`` -- only its own wiring is under test, not
``ArrangementView``'s guards (covered separately in ``tests/test_arrangement_flip.py``).

**Frame mode.** :meth:`ViewDialog.set_frame_mode` hides
ONLY the Projection tab (no CRS/projection mode to control in a native frame) -- the Camera and
Display tabs stay fully reachable, since camera reset and vertical exaggeration are exactly as
load-bearing in frame mode as in geo mode. See that method's own docstring for the full
rationale, including why ``ArrangementView``'s own "View…" button is no longer hidden at all (this
dialog is its narrower replacement).

**Persistence: ``Settings.view_options`` keys ``{"mode", "graticule", "vexag", "background"}``,
written by THIS dialog, not ``MainWindow``.** Every edit (mode, graticule, vexag, background --
camera az/el/zoom is deliberately NOT one of the four; it is live view state, not a saved
preference) calls ``update_settings(view_options=...)`` with the dialog's own, always-normalized
``self._opts``. :func:`normalized_view_options` is the single place "missing or garbage" is
resolved to a default (``mercator`` / ``False`` / ``1.0`` / the theme's own ground tone) -- shared
by this dialog's own restore-on-construction and ``main_window.py``'s post-``activate()`` push, so
the two callers can never define "default" differently.

**The mode path must not double-apply.** ``ModeRow`` (reused verbatim as the Projection tab's
widget, unmodified -- see its own module docstring) only emits ``modeChanged`` on a REAL
transition (its own ``_on_toggled`` checked-filter). Restoring this dialog's initial displayed mode
to whatever was last persisted -- when that differs from ``ModeRow``'s own hardcoded construction
default ("mercator") -- still has to poke the correct button's checked state, which WOULD reach
``ModeRow``'s already-live internal wiring and flip ``self._mode``/emit. Rather than lean on that
filter to swallow a spurious echo, construction here simply does the poke BEFORE connecting this
dialog's own ``modeChanged`` handler at all: nothing is listening yet, so the restore reaches
``ModeRow``'s own internal state (silently, correctly) without ever reaching ``view.set_mode`` or
``update_settings``. The first (and only) time this dialog ever calls either for mode is a real
user click on a button, or ``MainWindow``'s own separate post-``activate()`` push of the stored
value straight into ``view.set_mode`` (not through this dialog at all) -- see
``arrangement/view.py``'s own module docstring, "The View dialog" section.

**Camera "zoom": mode-aware, not a bare ``Camera.parallel_scale``.** pyvista's ``Camera.zoom(value)``
is a RELATIVE multiply against whatever the camera already has (a raw ``vtkCamera.Zoom()`` call) --
calling it twice with the same argument does not return to the same state, so it cannot back an
EDITABLE, round-trippable field this dialog's ``zoom`` ``DragValue`` needs. Neither can a bare
``Camera.parallel_scale`` alone, though: VTK documents (and the review confirmed
empirically) that ``parallel_scale`` has NO EFFECT under perspective projection -- globe mode's own
camera. ``ArrangementView.camera_state``/``set_camera_state`` go through
:func:`~dynamix.shell.arrangement.camera._read_zoom`/:func:`~dynamix.shell.arrangement.camera.
_write_zoom` instead (``camera.py``, not here -- that module already owns the orthographic-vs-
perspective split, see its own docstring's "Camera zoom, mode-aware" section): ``parallel_scale``
under the three flat (orthographic) modes, ``Camera.view_angle`` under globe's perspective one --
both true VTK attributes with real absolute get/set pairs, and both share the same "smaller value
= more zoomed in" direction (confirmed empirically, not assumed), so this dialog's single ``zoom``
field needs no per-mode sign flip to feel consistent across a mode switch. See
``ArrangementView.camera_state``'s own docstring for the parallel note on ``azimuth``/``elevation``
(pyvista's own tracked, absolute properties -- but NOT updated by the raw ``vtkCamera.Azimuth``/
``Elevation`` calls ``MomentumCamera`` drives during a spin).

**``QColorDialog`` behind a seam.** :meth:`ViewDialog._pick_color` is the ONLY place this module
touches ``QColorDialog`` -- tests monkeypatch it directly rather than ever opening a real modal
dialog (which would hang an offscreen test run waiting for a user who is never coming).
"""
from __future__ import annotations

import math

from PySide6 import QtGui, QtWidgets

from dynamix.core.projection import MODES
from dynamix.model.param import Param, ParamKind
from dynamix.shell.arrangement.mode_row import DEFAULT_MODE, ModeRow
from dynamix.shell.knob_widgets import make_control
from dynamix.shell.knobs import control_spec
from dynamix.shell.settings import load_settings, update_settings
from dynamix.shell.theme import RESTRAINED_DARK

#: Frame tab's read-only display-frame text per mode -- a SEPARATE dict from ``dynamix.core.
#: projection.LABELS`` (that one names the projection STYLE for the mode-picker itself; this one
#: describes the WGS84-referenced display frame the Frame tab reports, the design's own wording).
FRAME_LABELS = {
    "pacific":   "Equirectangular (WGS84, Pacific-centred)",
    "greenwich": "Equirectangular (WGS84, Greenwich-centred)",
    "mercator":  "Web Mercator (WGS84)",
    "globe":     "WGS84 ECEF (globe)",
}

#: The theme's own ground tone -- ``Settings.view_options["background"]``'s default when missing
#: or garbage. Resolved here (not left as a literal) for the identical reason ``view.py`` resolves
#: it before handing it to ``Scene``: this is the one non-``theme.py`` place allowed to read it.
DEFAULT_BACKGROUND = RESTRAINED_DARK.ground

#: Camera tab params -- editable, absolute, round-trippable (see the module docstring's "zoom"
#: section for why ``zoom`` is declared FLOAT, not something ``Camera.zoom()``-shaped).
_CAMERA_PARAMS = (
    Param("azimuth", ParamKind.ANGLE, default=0.0, wrap=360.0, units="°", label="Azimuth"),
    Param("elevation", ParamKind.FLOAT, default=0.0, soft_min=-89.0, soft_max=89.0,
          units="°", label="Elevation"),
    Param("zoom", ParamKind.FLOAT, default=1.0, min=0.01, soft_min=0.1, soft_max=10.0,
          label="Zoom"),
)

#: Display tab's vertical-exaggeration knob -- "float, default 1.0, soft 0.5..20" per the design;
#: hard-floored at 0 (a negative factor has no physical meaning here) but otherwise uncapped.
_VEXAG_PARAM = Param("vexag", ParamKind.FLOAT, default=1.0, min=0.0, soft_min=0.5, soft_max=20.0,
                      units="×", label="Vertical exaggeration")

#: The Display tab stretch control -- "frame-units per
#: octave" per the design. Hard-floored at 0 (a negative stretch has no physical meaning: it
#: would put coarser scales BELOW the finest one); the soft range starts at a placeholder
#: ``(0, 1.0)`` -- :meth:`ViewDialog._on_scale_space_toggled` rebinds it live via ``set_range``
#: (the same data-aware-slider idiom the ``DeviceBox.sync_from_result`` established,
#: ``knobs.rebind_range``) to ``(0, max_grid_dim)`` once the active geometry is known, per the
#: spec's own "slider range 0..max_grid_dim" wording -- a static soft range here would only ever
#: be a momentary placeholder before construction's first real geometry, so it is deliberately
#: narrow rather than a guess at a "typical" grid size.
_SCALE_SPACE_STRETCH_PARAM = Param("scale_space_stretch", ParamKind.FLOAT, default=0.0, min=0.0,
                                    soft_min=0.0, soft_max=1.0, units="", label="Stretch (per octave)")


def normalized_view_options(raw) -> dict:
    """Validate and default a raw ``Settings.view_options`` payload -- a missing key, or one
    holding a value of the wrong shape/type, silently falls back to its own default rather than
    raising or propagating garbage into a live scene. Shared by :class:`ViewDialog`'s own
    restore-on-construction and ``main_window.py``'s post-``activate()`` push, so the two callers
    can never define "default" differently (see the module docstring)."""
    raw = raw if isinstance(raw, dict) else {}

    mode = raw.get("mode")
    if mode not in MODES:
        mode = DEFAULT_MODE

    graticule = raw.get("graticule")
    if not isinstance(graticule, bool):
        graticule = False

    vexag = raw.get("vexag")
    try:
        vexag = float(vexag)
    except (TypeError, ValueError):
        vexag = 1.0
    else:
        if not math.isfinite(vexag) or vexag <= 0:
            vexag = 1.0

    background = raw.get("background")
    if not isinstance(background, str) or not background:
        background = DEFAULT_BACKGROUND

    # "3-D scale-space (stack by log₂ a)" + its stretch
    # slider. ``scale_space_stretch``'s own default/sentinel is 0.0 -- see
    # ``ViewDialog._on_scale_space_toggled``'s own docstring for why 0.0 doubles as "no stretch
    # chosen yet" AND is itself a perfectly safe value (a 0.0 stretch degenerates to z=0
    # regardless of ``scale_space``'s own state, so there is no unsafe garbage this key could
    # ever hold).
    scale_space = raw.get("scale_space")
    if not isinstance(scale_space, bool):
        scale_space = False

    scale_space_stretch = raw.get("scale_space_stretch")
    try:
        scale_space_stretch = float(scale_space_stretch)
    except (TypeError, ValueError):
        scale_space_stretch = 0.0
    else:
        if not math.isfinite(scale_space_stretch) or scale_space_stretch < 0:
            scale_space_stretch = 0.0

    return {"mode": mode, "graticule": graticule, "vexag": vexag, "background": background,
            "scale_space": scale_space, "scale_space_stretch": scale_space_stretch}


class ViewDialog(QtWidgets.QDialog):
    """Non-modal, 4-tab View dialog: Projection (``ModeRow``), Camera (az/el/zoom + Reset), Frame
    (frame readout + Graticule checkbox), Display (vertical exaggeration + background swatch).
    ``view`` is a real :class:`~dynamix.shell.arrangement.view.ArrangementView` in the live app --
    every call this dialog makes on it is already guarded there (see the module docstring)."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self._view = view
        self.setWindowTitle("View")
        self.setModal(False)

        self._opts = normalized_view_options(load_settings().view_options)
        self._camera_values = {p.name: p.default for p in _CAMERA_PARAMS}
        self._camera_controls: dict[str, QtWidgets.QWidget] = {}

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.addTab(self._build_projection_tab(), "Projection")
        self._tabs.addTab(self._build_camera_tab(), "Camera")
        self._tabs.addTab(self._build_frame_tab(), "Frame")
        self._tabs.addTab(self._build_display_tab(), "Display")
        #: Fixed at construction -- ``QTabWidget.indexOf`` on the SAME widget reference every time,
        #: so :meth:`set_frame_mode` never needs to re-derive which tab is "Projection" (a tab
        #: index can shift if tabs are ever reordered; a widget reference cannot).
        self._projection_tab = self._tabs.widget(0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._tabs)

        self._refresh_camera_readout()      # initial pull -- None/no-op headless, harmless either way

    def set_frame_mode(self, enabled: bool) -> None:
        """Hide ONLY the Projection tab
        (``ModeRow`` -- there is no CRS/projection mode for this dialog to control in a native
        frame) while ``enabled``; the Camera tab (reset + az/el/zoom) and the Display tab
        (vertical exaggeration + background) stay fully reachable -- both are exactly as
        load-bearing in frame mode as in geo mode ("camera reset and vertical
        exaggeration must be reachable there"; sec 6 makes exaggeration load-bearing for real).
        The Frame tab (the read-only projection-label readout + Graticule checkbox) is left alone
        too -- neither the design nor the spec names it, and the checkbox itself already
        degrades safely on its own (``Scene``'s own "Frame mode" docstring section: the graticule
        never draws in frame mode regardless of the checkbox's state).

        ``ArrangementView.set_frame_mode`` no longer hides the "View…" button itself (see that
        method's own "Correction" docstring section) -- this dialog is the button's own, narrower
        replacement for that gating, now scoped to the one control that is actually meaningless.
        ``MainWindow`` is the one place that knows WHEN this is called -- the three-view
        switcher's own state -- mirroring exactly how ``ArrangementView.set_frame_mode`` itself is
        driven; this dialog only reacts."""
        self._tabs.setTabVisible(self._tabs.indexOf(self._projection_tab), not bool(enabled))

    # -- Projection ---------------------------------------------------------------------------

    def _build_projection_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._mode_row = ModeRow()
        # Restore-without-emitting (see the module docstring's "mode path must not double-apply"
        # section): poke the checked button BEFORE this dialog's own modeChanged handler is
        # connected, so a stored mode that differs from ModeRow's own hardcoded default reaches
        # ONLY ModeRow's internal state here, never view.set_mode/update_settings.
        button = self._mode_row._buttons.get(self._opts["mode"])
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._mode_row.modeChanged.connect(self._on_mode_changed)
        layout.addWidget(self._mode_row)
        layout.addStretch(1)
        return tab

    def _on_mode_changed(self, mode: str) -> None:
        self._opts["mode"] = mode
        if hasattr(self, "_frame_value"):
            self._frame_value.setText(FRAME_LABELS.get(mode, mode))
        self._view.set_mode(mode)
        update_settings(view_options=dict(self._opts))

    # -- Camera ---------------------------------------------------------------------------------

    def _build_camera_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        row = QtWidgets.QHBoxLayout()
        for p in _CAMERA_PARAMS:
            column = QtWidgets.QVBoxLayout()
            mini = QtWidgets.QLabel(p.label or p.name)
            mini.setProperty("muted", "true")
            column.addWidget(mini)
            control = make_control(control_spec(p), self._camera_values[p.name])
            control.valueChanged.connect(
                lambda value, name=p.name: self._on_camera_control_changed(name, value))
            column.addWidget(control)
            self._camera_controls[p.name] = control
            row.addLayout(column)
        layout.addLayout(row)
        reset = QtWidgets.QPushButton("Reset")
        reset.clicked.connect(self._on_reset_clicked)
        layout.addWidget(reset)
        layout.addStretch(1)
        return tab

    def _on_camera_control_changed(self, name: str, value) -> None:
        param = next(p for p in _CAMERA_PARAMS if p.name == name)
        validated = param.validate(value)
        self._camera_values[name] = validated
        self._camera_controls[name].set_value(validated)
        self._view.set_camera_state(self._camera_values["azimuth"],
                                     self._camera_values["elevation"],
                                     self._camera_values["zoom"])

    def _on_reset_clicked(self) -> None:
        self._view.reset_camera()
        self._refresh_camera_readout()

    def _refresh_camera_readout(self) -> None:
        """Pull, never live-stream -- called once at construction and again every
        time this dialog is shown (:meth:`showEvent`) or Reset is clicked. A ``None`` (headless, or
        the interactor was never built) leaves whatever is already displayed alone."""
        state = self._view.camera_state()
        if state is None:
            return
        for name in ("azimuth", "elevation", "zoom"):
            if name not in state:
                continue
            value = state[name]
            self._camera_values[name] = value
            self._camera_controls[name].set_value(value)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._refresh_camera_readout()

    # -- Frame ------------------------------------------------------------------------------

    def _build_frame_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        mini = QtWidgets.QLabel("Frame")
        mini.setProperty("muted", "true")
        layout.addWidget(mini)
        self._frame_value = QtWidgets.QLabel(FRAME_LABELS.get(self._opts["mode"], self._opts["mode"]))
        self._frame_value.setProperty("reading", "true")
        layout.addWidget(self._frame_value)

        self._graticule_checkbox = QtWidgets.QCheckBox("Graticule")
        self._graticule_checkbox.setChecked(self._opts["graticule"])
        self._graticule_checkbox.toggled.connect(self._on_graticule_toggled)
        layout.addWidget(self._graticule_checkbox)
        layout.addStretch(1)
        return tab

    def _on_graticule_toggled(self, checked: bool) -> None:
        self._opts["graticule"] = bool(checked)
        self._view.set_graticule(self._opts["graticule"])
        update_settings(view_options=dict(self._opts))

    # -- Display ----------------------------------------------------------------------------

    def _build_display_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        column = QtWidgets.QVBoxLayout()
        mini = QtWidgets.QLabel(_VEXAG_PARAM.label)
        mini.setProperty("muted", "true")
        column.addWidget(mini)
        self._vexag_control = make_control(control_spec(_VEXAG_PARAM), self._opts["vexag"])
        self._vexag_control.valueChanged.connect(self._on_vexag_changed)
        column.addWidget(self._vexag_control)
        layout.addLayout(column)

        # "3-D scale-space (stack by log₂ a)" + its
        # stretch slider -- the design, next to vertical exaggeration per the spec's own wording.
        self._scale_space_checkbox = QtWidgets.QCheckBox("3-D scale-space (stack by log₂ a)")
        self._scale_space_checkbox.setChecked(self._opts["scale_space"])
        self._scale_space_checkbox.toggled.connect(self._on_scale_space_toggled)
        layout.addWidget(self._scale_space_checkbox)

        stretch_column = QtWidgets.QVBoxLayout()
        stretch_mini = QtWidgets.QLabel(_SCALE_SPACE_STRETCH_PARAM.label)
        stretch_mini.setProperty("muted", "true")
        stretch_column.addWidget(stretch_mini)
        self._scale_space_stretch_control = make_control(
            control_spec(_SCALE_SPACE_STRETCH_PARAM), self._opts["scale_space_stretch"])
        self._scale_space_stretch_control.valueChanged.connect(self._on_scale_space_stretch_changed)
        stretch_column.addWidget(self._scale_space_stretch_control)
        layout.addLayout(stretch_column)

        self._background_button = QtWidgets.QPushButton("Background…")
        self._background_button.clicked.connect(self._on_background_clicked)
        self._update_background_swatch()
        layout.addWidget(self._background_button)
        layout.addStretch(1)
        return tab

    def _on_vexag_changed(self, value) -> None:
        validated = _VEXAG_PARAM.validate(value)
        self._opts["vexag"] = validated
        self._vexag_control.set_value(validated)
        self._view.set_vertical_exaggeration(validated)
        update_settings(view_options=dict(self._opts))

    def _on_scale_space_toggled(self, checked: bool) -> None:
        """The "3-D scale-space" checkbox.

        **Default stretch when enabling with none set**: a stored
        ``scale_space_stretch`` of exactly ``0.0`` doubles as BOTH "the user never chose one yet"
        AND a value that is always safe to hold regardless (0.0 stretch degenerates to z=0 either
        way, so there is no distinct "unset" sentinel needed). On a transition INTO enabled with
        that sentinel still in place, ``self._view.default_scale_space_stretch()`` computes
        ``max_grid_dim / (2 * n_scales)`` from whatever is on screen right now
        (``Scene.default_scale_space_stretch``'s own docstring); the stretch control's own SOFT
        range is rebound to ``(0, max_grid_dim)`` via :meth:`DragValue.set_range` at the same
        moment (spec's own "slider range 0..max_grid_dim" wording) -- the identical data-aware-
        slider idiom the ``DeviceBox.sync_from_result`` already established for a live data
        range. Disabling, or enabling with a real stretch already chosen, pushes the current
        values straight through unchanged."""
        self._opts["scale_space"] = bool(checked)
        if checked and self._opts["scale_space_stretch"] == 0.0:
            default_stretch = self._view.default_scale_space_stretch()
            max_grid_dim = self._view.default_scale_space_max_grid_dim()
            self._scale_space_stretch_control.set_range(0.0, max_grid_dim)
            self._opts["scale_space_stretch"] = default_stretch
            self._scale_space_stretch_control.set_value(default_stretch)
        self._view.set_scale_space(self._opts["scale_space"], self._opts["scale_space_stretch"])
        update_settings(view_options=dict(self._opts))

    def _on_scale_space_stretch_changed(self, value) -> None:
        validated = _SCALE_SPACE_STRETCH_PARAM.validate(value)
        self._opts["scale_space_stretch"] = validated
        self._scale_space_stretch_control.set_value(validated)
        self._view.set_scale_space(self._opts["scale_space"], validated)
        update_settings(view_options=dict(self._opts))

    def _on_background_clicked(self) -> None:
        picked = self._pick_color(self._opts["background"])
        if picked is None:
            return
        self._opts["background"] = picked
        self._update_background_swatch()
        self._view.set_background(picked)
        update_settings(view_options=dict(self._opts))

    def _update_background_swatch(self) -> None:
        self._background_button.setStyleSheet(f"background-color: {self._opts['background']};")

    def _pick_color(self, initial: str) -> str | None:
        """The ONLY place this module touches ``QColorDialog`` -- monkeypatched directly in tests
        (never opened for real there; a real modal dialog would hang an offscreen test run).
        Returns the picked color as a lowercase ``#rrggbb`` hex string, or ``None`` if the user
        cancelled."""
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(initial), self, "Background color")
        if not color.isValid():
            return None
        return color.name()
