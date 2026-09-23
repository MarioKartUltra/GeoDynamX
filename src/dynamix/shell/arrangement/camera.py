# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""MomentumCamera: the arrangement view's Google-Earth-style momentum spin + the ``r`` reset
("Ported momentum-spin camera (globe mode only), ``r`` reset,
orthographic camera for flat modes / perspective for globe, per the audit").

**Port, not reinvention.** The spin mechanism (``_on_spin_press``/``_on_spin_move``/
``_on_spin_release``/``_spin_tick``, EQSelect's ``app_window.py:3541-3599``) is carried over
verbatim -- same constants (``_SPIN_GAIN=0.28``, ``_SPIN_DECAY=0.90``), same EMA smoothing
(``alpha=0.5``), same release gate (held-still-for-<0.12s AND a >=2.0px flick), same per-tick
decay-until-<0.2px stop, same 16ms (~60fps) ``QTimer``. Only the plumbing changed:

- ``self.plotter``/``self.projection`` (EQSelect's own ``QMainWindow`` attributes) become
  constructor arguments -- a plain ``plotter`` handle (duck-typed: a ``pyvista.Plotter`` in every
  test here, a ``pyvistaqt.QtInteractor`` in the live app -- see ``scene.py``'s own module
  docstring for the same substitution) and a :class:`~dynamix.shell.arrangement.scene.Scene`
  (queried for ``.mode``, driven via ``.set_mode()``). This is what makes the whole class
  constructible, and every method callable, against a plain OFFSCREEN ``pv.Plotter`` -- no
  ``pyvistaqt.QtInteractor`` (which cannot be built under this harness's mandated offscreen QPA
  platform, see ``view.py``'s ``activate()``) is needed anywhere in this module or its tests.
- EQSelect wires press/move/release through ``self.plotter.iren.add_observer(...)`` (a VTK
  mechanism, no Qt dependency of its own -- the same fact ``view.py``'s own click-picking module
  docstring already establishes for ``track_click_position``); this module's own methods
  (:meth:`on_press`/:meth:`on_move`/:meth:`on_release`) are plain callables with that exact
  ``(caller, event)`` VTK-observer signature (``*_a``, ignored, matching EQSelect's own), so
  ``ArrangementView.activate()`` wires them the identical way.

**The ``r`` reset** (EQSelect's ``_reset_view``, ``app_window.py:3880-3897``, audit doc lines
85-93): cancels any in-flight coast, then -- if currently on the globe, switches to the
last-used FLAT mode first (which re-fires the scene's own rebuild) and re-frames; if already flat,
just re-applies the orthographic top-down default view. There is no separate "reset camera" that
leaves the projection alone -- reset and re-projection are the same code path on the globe,
distinct only when already flat, exactly as the audit describes.

**One deliberate non-port: EQSelect's ``off_screen``-gated view-orientation skip.** EQSelect's own
``_apply_default_view`` (``app_window.py:3370-3388``) skips ``view_xy()``/``view_vector()``/
``enable_parallel_projection()``/``disable_parallel_projection()`` when ``self.off_screen`` is
True (an EQSelect app-level batch-export flag, not "this Qt platform can't render") -- it still
calls ``reset_camera()`` unconditionally either way. That gating has no equivalent concept in
DynamiX and, ported literally, would make :meth:`reset`'s own core view-orientation logic
UNTESTABLE in this repo's own offscreen test harness -- the opposite of the design's explicit
"camera logic must be testable against a plain offscreen pv.Plotter's camera". Every view-
orientation call here runs unconditionally, offscreen or not -- the same posture ``scene.py``'s
``set_colormap()``/``_apply_mask_to_all()`` already take for their own ``plotter.render()`` calls
("``pv.Plotter.render()`` works offscreen, so this is unconditional").

**Mode-switcher UI (closed).** The design names a "corner segmented
control (pacific / greenwich / mercator / globe)" for switching projection mode; nothing built it at first, so for a while
``Scene.set_mode`` had no live caller anywhere in the shipped app, only in tests, and globe mode
(and therefore the momentum spin itself) was unreachable through the running app. ``ArrangementView``
now builds a :class:`~dynamix.shell.arrangement.mode_row.ModeRow` (see that module's own docstring)
and wires its ``modeChanged`` signal to both ``Scene.set_mode`` and :meth:`note_mode` below, so the
mechanism this module always implemented in full is reachable for real. One consequence of the
GAP, not the fix, stays documented here for history: :attr:`_last_flat_mode` (EQSelect's
``_last_flat_projection``) is normally kept current by the projection-change HANDLER itself
(``_on_projection_changed``, ``app_window.py:3870``) every time the mode changes to a flat one --
this class only observes the mode at construction time and at each :meth:`reset` call (its own
non-globe branch updates it) UNLESS something else calls :meth:`note_mode` too, which
``ModeRow``'s own wiring now does on every switch.

**Camera zoom, mode-aware.** The View
dialog's Camera tab (``dynamix.shell.view_dialog.ViewDialog``) needs one abstract "zoom" reading/
writing pair; ``ArrangementView.camera_state``/``set_camera_state`` used to read/write
``cam.parallel_scale`` UNCONDITIONALLY -- which VTK documents as having no effect at all under
perspective projection, the globe mode's own camera (:meth:`MomentumCamera._apply_default_view`,
right above, already calls ``disable_parallel_projection()`` for it). Confirmed empirically (not
just from the docstring): a ``parallel_scale`` edit under perspective produced a byte-identical
render. :func:`_read_zoom`/:func:`_write_zoom` are the fix -- this module, not ``view.py``, is
where they live, because this is already the one place in the codebase that KNOWS which mode gets
which projection type (:meth:`_apply_default_view`'s own ``enable_parallel_projection()``/
``disable_parallel_projection()`` split, just below); ``ArrangementView.camera_state``/
``set_camera_state`` import and call them rather than touching ``cam.parallel_scale``/
``cam.view_angle`` directly.

They branch on the CAMERA's own live ``GetParallelProjection()`` flag, not on
``self._scene.mode`` -- a plain ``pv.Camera()``/``pv.Plotter().camera`` has no ``scene`` to ask,
and the flag is the actual ground truth the renderer reads regardless of which mode set it.
Direction was verified empirically, not assumed: a screen-space two-point-distance probe (project
two off-axis world points through ``GetCompositeProjectionTransformMatrix``, the exact mechanism
``scene.py``'s own ``Scene.pick``/``_camera_mvp`` already use) shows a SMALLER ``parallel_scale``
*and* a SMALLER ``view_angle`` both increase the on-screen distance -- i.e. both are "more zoomed
in" in the SAME direction, not opposite ones. No inversion/normalization is applied: :func:`_read_
zoom`/:func:`_write_zoom` are a straight pass-through of whichever attribute is currently live, and
the dialog's Zoom ``DragValue`` needs no per-mode sign flip to feel consistent -- dragging down
zooms in either way. The two attributes' NUMERIC RANGES do still differ (``parallel_scale`` is
typically O(0.1-10) at this app's own draped-geography scale; ``view_angle`` is a field-of-view in
degrees, VTK's own default 30) -- ``_CAMERA_PARAMS``'s ``zoom`` ``Param`` (``view_dialog.py``) was
tuned against the former, so the Zoom knob's step size/soft range are calibrated for the three flat
modes, not globe's perspective one; a real magnitude-rescaling between the two (so the same DIALOG
number means the same relative field-of-view fraction in both) was judged out of scope for this
fix -- direction consistency was the actual concern, and that is what was broken.
"""
from __future__ import annotations

import time

from dynamix.shell.arrangement.scene import DEFAULT_MODE

_SPIN_GAIN = 0.28           # degrees of camera azimuth/elevation per pixel of flick velocity
_SPIN_DECAY = 0.90          # per-tick velocity decay (higher = longer glide)
_SPIN_INTERVAL_MS = 16      # ~60 fps -- the design's <16ms-per-tick perf bar (audit doc)
_SPIN_EMA_ALPHA = 0.5       # release velocity = the RECENT motion, not the whole drag's average
_SPIN_HOLD_STILL_S = 0.12   # held still this long before release -> stop dead, no coast
_SPIN_MIN_FLICK_PX = 2.0    # |vx| + |vy| below this at release -> too small a flick, ignore
_SPIN_MIN_VELOCITY_PX = 0.2  # |vx| + |vy| below this mid-coast -> glided to a stop


def _read_zoom(camera) -> float:
    """The camera's current zoom, mode-aware -- see the module docstring's "Camera zoom,
    mode-aware" section for the fix this closes and the empirical direction check. ``camera`` is
    any object carrying VTK's own ``Camera`` interface (``GetParallelProjection``,
    ``parallel_scale``, ``view_angle`` -- a plain ``pv.Camera()``, ``pv.Plotter().camera``, or a
    live ``pyvistaqt.QtInteractor.camera`` all qualify). Branches on the camera's OWN live
    projection flag, not on any ``Scene``/mode string -- this function has no ``Scene`` to ask,
    and the flag is the actual ground truth the renderer reads. A straight pass-through of
    whichever attribute is currently live: no unit conversion, no sign flip (both attributes
    already share the same "smaller is more zoomed in" direction -- confirmed, not assumed)."""
    if camera.GetParallelProjection():
        return float(camera.parallel_scale)
    return float(camera.view_angle)


def _write_zoom(camera, value: float) -> None:
    """The mode-aware inverse of :func:`_read_zoom`. Writes ONLY the attribute that is actually
    live under the camera's current projection -- a perspective write never also touches
    ``parallel_scale`` (harmless under VTK's own perspective no-op rule, but leaving it alone
    means a LATER switch back to orthographic does not silently resurrect a stale, unrelated
    edit), and vice versa."""
    if camera.GetParallelProjection():
        camera.parallel_scale = float(value)
    else:
        camera.view_angle = float(value)


class MomentumCamera:
    """Google-Earth-style momentum spin (globe mode only) + the ``r`` reset, over a plain
    ``plotter`` handle and a :class:`~dynamix.shell.arrangement.scene.Scene`. See the module
    docstring for the full port rationale.

    Requires an existing ``QApplication`` (a live Qt event loop to drive the ``QTimer``) but
    needs no ``pyvistaqt.QtInteractor`` -- constructible, and every method callable, against a
    plain offscreen ``pyvista.Plotter`` paired with a real ``Scene`` around it (this module's own
    tests do exactly that).
    """

    def __init__(self, plotter, scene) -> None:
        # Imported here, not at module top: this is the ONE place this module needs PySide6 (a
        # QTimer), and importing lazily keeps the module trivially importable in a plain python
        # shell for inspection without a QApplication already running -- matching no stronger
        # convention than "PySide6 stays under dynamix/shell/" already requires, just deferred to
        # first use like view.py's own _import_pyvista().
        from PySide6 import QtCore

        self._plotter = plotter
        self._scene = scene
        self._timer = QtCore.QTimer()
        self._timer.setInterval(_SPIN_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._dragging = False
        self._prev: tuple[float, float] | None = None
        self._vel: tuple[float, float] | None = None
        self._t = 0.0
        #: EQSelect's ``_last_flat_projection`` -- see the module docstring's "No mode-switcher UI
        #: exists yet" section for why this can only be seeded here and refreshed in
        #: :meth:`reset`, not kept live by a projection-change handler that does not exist.
        self._last_flat_mode = scene.mode if scene.mode != "globe" else DEFAULT_MODE

    @property
    def is_spinning(self) -> bool:
        """Whether the coast timer is currently running -- read-only, for tests/callers that want
        to observe the spin state without reaching into ``_timer`` directly."""
        return self._timer.isActive()

    # -- drag -> release -> coast (EQSelect app_window.py:3548-3599) ------------------------------

    def on_press(self, *_a) -> None:
        """``LeftButtonPressEvent``: a new grab cancels any in-flight coast."""
        self._timer.stop()
        self._dragging = True
        self._prev = self._event_pos()
        self._vel = None
        self._t = time.monotonic()

    def on_move(self, *_a) -> None:
        """``MouseMoveEvent``: EMA-smoothed pixel velocity, so the release velocity reflects the
        RECENT motion, not the whole drag's average.

        ``_event_pos()`` is guarded the same way :meth:`_tick`
        already guards its own live-plotter calls. ``view.py`` registers this one callback through
        ``iren.style``'s own TRACKED ``add_observer`` (see that module's comment for why this
        event specifically needs the style, not the interactor, as its registration target) --
        that path, unlike ``RenderWindowInteractor.add_observer`` (which every OTHER observer here
        goes through), does NOT wrap the callback in pyvista's own ``try_callback`` -- so nothing
        upstream of this method would catch and log a raising callback; without this guard, any
        exception would surface as a raw, unhandled traceback out of VTK's C++ dispatch instead of
        a clean no-op."""
        if not self._dragging or self._prev is None:
            return
        try:
            cur = self._event_pos()
        except Exception:
            return
        dx, dy = cur[0] - self._prev[0], cur[1] - self._prev[1]
        if self._vel is None:
            self._vel = (dx, dy)
        else:
            a = _SPIN_EMA_ALPHA
            self._vel = (a * dx + (1 - a) * self._vel[0], a * dy + (1 - a) * self._vel[1])
        self._prev = cur
        self._t = time.monotonic()

    def on_release(self, *_a) -> None:
        """``LeftButtonReleaseEvent``: starts the coast timer iff there is real, RECENT velocity
        (release within :data:`_SPIN_HOLD_STILL_S` of the last move, and at least
        :data:`_SPIN_MIN_FLICK_PX` fast) AND the scene is currently in globe mode -- flat maps
        never spin (audit doc's "gated to the globe only")."""
        self._dragging = False
        v = self._vel
        self._vel = None
        if v is None or self._scene.mode != "globe":
            return
        if time.monotonic() - self._t > _SPIN_HOLD_STILL_S:
            return
        if abs(v[0]) + abs(v[1]) < _SPIN_MIN_FLICK_PX:
            return
        self._vel = v
        self._timer.start()

    def _tick(self) -> None:
        """``QTimer.timeout``, ~60 fps: spin the camera by the current (decaying) velocity, then
        decay it further -- or stop once it has glided below :data:`_SPIN_MIN_VELOCITY_PX`."""
        v = self._vel
        if v is None:
            self._timer.stop()
            return
        try:
            cam = self._plotter.camera
            cam.Azimuth(-v[0] * _SPIN_GAIN)         # drag-x spins the globe about its axis
            cam.Elevation(v[1] * _SPIN_GAIN)
            cam.OrthogonalizeViewUp()
            self._plotter.render()
        except Exception:
            self._timer.stop()
            return
        self._vel = (v[0] * _SPIN_DECAY, v[1] * _SPIN_DECAY)
        if abs(self._vel[0]) + abs(self._vel[1]) < _SPIN_MIN_VELOCITY_PX:
            self._vel = None
            self._timer.stop()

    def _event_pos(self) -> tuple[float, float]:
        """The live VTK interactor's current event position -- EQSelect's ``_iren_pos``
        (``app_window.py:3545-3546``)."""
        return tuple(self._plotter.iren.interactor.GetEventPosition())

    def stop(self) -> None:
        """Halt any in-flight coast and forget its velocity -- idempotent (a harmless no-op when
        nothing is spinning). Parking the view (``ArrangementView.
        deactivate()``) must not leave a 60fps render loop running against a now-hidden
        interactor -- dormant only because globe mode (the spin's own gate) is unreachable through
        the running app today (see the module docstring's "No mode-switcher UI exists yet"), not
        because the risk isn't real the moment that control lands. :meth:`reset` also calls this
        (it always cancels a coast first, the identical effect) rather than duplicating the two
        lines."""
        self._timer.stop()
        self._vel = None

    # -- 'r' reset (EQSelect app_window.py:3880-3897) ---------------------------------------------

    def reset(self) -> None:
        """Snap to a flat, north-up, top-down 2-D map view (Google-Earth-style ``r``). On the
        globe, switches to the last-used flat projection first (which re-fires
        ``Scene.set_mode``'s own rebuild); on a flat map, just re-frames. Always cancels any
        in-flight momentum coast first (:meth:`stop`) -- see the module docstring for why there is
        no separate "reset camera" that leaves the projection alone."""
        self.stop()
        if self._scene.mode == "globe":
            self._scene.set_mode(self._last_flat_mode)
        else:
            self._last_flat_mode = self._scene.mode
        self._apply_default_view()

    def note_mode(self, mode: str) -> None:
        """Keep :attr:`_last_flat_mode` current for an external mode switch that bypasses
        :meth:`reset` entirely -- :class:`~dynamix.shell.arrangement.mode_row.ModeRow` (the
        module docstring's "Mode-switcher UI" section) calls this on every ``modeChanged``,
        alongside its own ``Scene.set_mode(mode)`` call, so a later globe ``reset()`` still
        returns to whichever flat mode the user actually picked rather than a stale
        construction-time/previous-``reset()`` value. A switch TO globe is a no-op here --
        :attr:`_last_flat_mode` only ever names a FLAT mode, by the same contract :meth:`reset`'s
        own non-globe branch already keeps."""
        if mode != "globe":
            self._last_flat_mode = mode

    def _apply_default_view(self) -> None:
        """Frame the scene right-side-up for the CURRENT projection: flat maps orthographic,
        top-down, north-up; the globe perspective, Pacific-facing, pole up -- EQSelect's
        ``_apply_default_view`` (``app_window.py:3370-3388``), minus the ``off_screen``-gated skip
        (see the module docstring's "One deliberate non-port" section)."""
        if self._scene.mode == "globe":
            self._plotter.disable_parallel_projection()   # perspective: a sphere needs it to read
            self._plotter.view_vector((1.0, 0.15, 0.25), viewup=(0.0, 0.0, 1.0))
        else:
            # ORTHOGRAPHIC top-down: no perspective parallax, so a screen pixel maps to an exact
            # lon/lat at any depth -- load-bearing for click/pick accuracy (audit doc), not
            # cosmetic.
            self._plotter.enable_parallel_projection()
            self._plotter.view_xy()                        # look down -Z, north up
        self._plotter.reset_camera()
