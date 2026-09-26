# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Momentum camera + reset, benchmarks, and the arrangement-view integration sweep.

Four sections, in order:

1. **Camera unit tests** -- ``MomentumCamera`` driven directly against a plain offscreen
   ``pv.Plotter`` + a real ``Scene`` (the design's instruction: "camera logic must be testable
   ... without the QtInteractor").
2. **Wiring tests** -- the exact observer-registration pattern ``ArrangementView.activate()`` uses
   (VTK ``iren.add_observer``/``plotter.add_key_event``), reproduced against a plain offscreen
   ``pv.Plotter`` substituted for ``view._interactor`` -- the SAME substitution
   ``tests/test_arrangement_picking.py``'s own click-leg tests use, for the identical reason: a
   real ``pyvistaqt.QtInteractor`` cannot be BUILT under this harness's mandated offscreen QPA
   platform (``tests/test_arrangement_flip.py``'s documented segfault guard), so ``activate()``'s
   own first-build branch never runs here -- these tests prove the WIRING PATTERN works via real,
   synthetic VTK events, not the literal ``activate()`` call. See this file's own closing comment
   for the one thing that genuinely cannot be exercised this way (real-display smoke).
3. **Benchmarks** -- projection-mode switch at the spec's own load scenario (2 draped fields <=2M
   pts + 50k chain vertices, reusing ``tests/test_arrangement_mask.py``'s own fixture), <5s
   asserted offscreen (generous CI bound; the <1s bar is real-hardware, reported not asserted, per the design) -- plus a bonus, unasserted "flip-in" number.
4. **The end-to-end sweep** -- two synthetic geo fields, computed, flipped in, draped with
   vectors, masked, picked, grouped, committed, flipped back, session repainted, saved, reloaded,
   groups survive -- and the real-app smoke (``scripts/shell.sh --render``, the never-flipped
   path).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista not installed")
pytest.importorskip("rasterio", reason="rasterio not installed (needed by the geo fixtures)")
pv.OFF_SCREEN = True

from dynamix.core.selection import project_points
from dynamix.devices import register_builtin_devices
from dynamix.devices.groups import decode_groups
from dynamix.model.layer import Layer
from dynamix.shell.arrangement.camera import MomentumCamera
from dynamix.shell.arrangement.group_palette import GROUP_COLORS
from dynamix.shell.arrangement.scene import DEFAULT_MODE, Scene

from tests.test_arrangement_mask import _big_field, _synthetic_chains
from tests.test_arrangement_resolve import _geo_field
from tests.test_arrangement_scene import _make_chain, _plotter, _result
from tests.test_shell_window import _stub_stack

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==================================================================================================
# 1. MomentumCamera -- unit tests, against a plain offscreen pv.Plotter + a real Scene
# ==================================================================================================


def _camera(qtbot):
    """``qtbot`` is required even though no widget is ever built here: ``MomentumCamera``'s own
    ``QTimer`` needs a live ``QApplication``/event dispatcher to ``.start()`` at all (confirmed
    empirically -- without it, ``.start()`` is a silent no-op and ``isActive()`` stays ``False``
    forever, logging "current thread's event dispatcher has already been destroyed"); requesting
    ``qtbot`` is this suite's own established way of guaranteeing one exists."""
    scene = Scene(_plotter())
    cam = MomentumCamera(scene._plotter, scene)
    return cam, scene


def _drag(cam, dx, dy):
    """One press -> one move of (dx, dy) pixels -> release, via the camera's own public gesture
    methods (not synthetic VTK events -- that leg is covered separately in section 2)."""
    iren = cam._plotter.iren.interactor
    iren.SetEventPosition(0, 0)
    cam.on_press()
    iren.SetEventPosition(dx, dy)
    cam.on_move()
    cam.on_release()


def test_flat_mode_never_spins_on_release(qtbot):
    cam, scene = _camera(qtbot)
    assert scene.mode == DEFAULT_MODE != "globe"

    _drag(cam, 50, 0)

    assert cam.is_spinning is False
    assert cam._vel is None


def test_globe_release_starts_a_coast_that_decays_to_a_stop(qtbot):
    cam, scene = _camera(qtbot)
    scene.set_mode("globe")

    _drag(cam, 50, 0)

    assert cam.is_spinning is True
    assert cam._vel == (50.0, 0.0)

    for _ in range(200):        # 0.90 ** 200 is far below the 0.2px stop threshold
        cam._tick()
        if not cam.is_spinning:
            break

    assert cam.is_spinning is False
    assert cam._vel is None


def test_press_cancels_an_in_flight_coast(qtbot):
    cam, scene = _camera(qtbot)
    scene.set_mode("globe")
    _drag(cam, 50, 0)
    assert cam.is_spinning is True

    cam.on_press()

    assert cam.is_spinning is False


def test_release_after_holding_still_does_not_coast(qtbot):
    """The audit's own gate: release must be within 0.12s of the last move, or the drag is read
    as 'held still' -- no coast, even with plenty of raw velocity."""
    cam, scene = _camera(qtbot)
    scene.set_mode("globe")
    iren = cam._plotter.iren.interactor
    iren.SetEventPosition(0, 0)
    cam.on_press()
    iren.SetEventPosition(50, 0)
    cam.on_move()
    cam._t -= 1.0        # backdate the last move well past the 0.12s hold-still window

    cam.on_release()

    assert cam.is_spinning is False


def test_release_with_a_slow_flick_does_not_coast(qtbot):
    cam, scene = _camera(qtbot)
    scene.set_mode("globe")

    _drag(cam, 1, 0)      # |vx| + |vy| = 1.0 < the 2.0px minimum flick

    assert cam.is_spinning is False


def test_tick_stops_on_a_plotter_error(qtbot):
    """A dead/torn-down plotter (``cam.Azimuth`` or ``.render()`` raising) must stop the timer
    cleanly rather than spin forever on a broken render surface -- EQSelect's own verbatim
    ``except Exception: self._spin_timer.stop(); return`` -- which stops WITHOUT clearing
    ``_vel`` (an immediate ``return``, same as the source): the timer is what matters (a stopped
    timer never reads ``_vel`` again), not the leftover value."""
    class _Boom:
        @property
        def camera(self):
            raise RuntimeError("no camera")

    cam, scene = _camera(qtbot)
    scene.set_mode("globe")
    _drag(cam, 50, 0)
    assert cam.is_spinning is True
    cam._plotter = _Boom()

    cam._tick()

    assert cam.is_spinning is False


# ------------------------------------------------------------------------------------- 'r' reset


def test_reset_while_flat_reframes_without_changing_mode(qtbot):
    cam, scene = _camera(qtbot)

    cam.reset()

    assert scene.mode == DEFAULT_MODE
    assert scene._plotter.camera.GetParallelProjection() == 1     # orthographic, per the audit


def test_reset_from_globe_returns_to_the_seeded_flat_mode_and_reframes(qtbot):
    cam, scene = _camera(qtbot)
    assert scene.mode == "mercator"          # seeded at construction (DEFAULT_MODE, flat)
    scene.set_mode("globe")

    cam.reset()

    assert scene.mode == "mercator"
    assert scene._plotter.camera.GetParallelProjection() == 1     # flat framing was applied, not
                                                                   # the globe's perspective one


def test_reset_cancels_an_in_flight_coast(qtbot):
    cam, scene = _camera(qtbot)
    scene.set_mode("globe")
    _drag(cam, 50, 0)
    assert cam.is_spinning is True

    cam.reset()

    assert cam.is_spinning is False
    assert cam._vel is None


def test_stop_halts_an_in_flight_coast_and_zeroes_velocity(qtbot):
    """``MomentumCamera.stop()`` -- the primitive ``ArrangementView.
    deactivate()`` needs so parking the view doesn't leave a 60fps coast running against a hidden
    interactor -- halts the timer and forgets the velocity, independent of ``reset()`` (which
    reframes/re-projects too; ``stop()`` must not do either of those, only cancel the coast)."""
    cam, scene = _camera(qtbot)
    scene.set_mode("globe")
    _drag(cam, 50, 0)
    assert cam.is_spinning is True
    mode_before = scene.mode

    cam.stop()

    assert cam.is_spinning is False
    assert cam._vel is None
    assert scene.mode == mode_before            # unlike reset(): no re-projection, no reframing


def test_reset_updates_last_flat_mode_for_a_later_globe_reset(qtbot):
    """``_last_flat_mode`` tracks whichever flat mode reset() last observed -- switching to
    ``pacific`` then resetting (while still flat) remembers ``pacific``, so a LATER globe reset
    returns there, not to the construction-time default."""
    cam, scene = _camera(qtbot)
    scene.set_mode("pacific")
    cam.reset()                    # while flat: just re-frames, but also refreshes _last_flat_mode
    assert scene.mode == "pacific"

    scene.set_mode("globe")
    cam.reset()

    assert scene.mode == "pacific"


def test_globe_view_is_perspective_not_orthographic(qtbot):
    cam, scene = _camera(qtbot)
    scene.set_mode("pacific")
    cam.reset()                    # seed _last_flat_mode = pacific
    scene.set_mode("globe")

    cam._apply_default_view()      # exercise the globe branch directly (reset() always converges
                                    # back to flat -- see the module docstring -- so this is the
                                    # one way to reach it deterministically)

    assert scene._plotter.camera.GetParallelProjection() == 0


# ==================================================================================================
# 1b. _read_zoom / _write_zoom -- the mode-aware zoom mapping
# ==================================================================================================
#
# ArrangementView.camera_state/set_camera_state used to read/write cam.parallel_scale
# unconditionally -- a documented VTK no-op under perspective projection, globe mode's own camera
# (MomentumCamera._apply_default_view's own disable_parallel_projection() call, tested just
# above). _read_zoom/_write_zoom (camera.py -- see its own module docstring's "Camera zoom,
# mode-aware" section for why that file, not view.py) branch on the camera's OWN live
# GetParallelProjection() flag. Driven directly against a bare pv.Camera() here -- no Plotter, no
# Scene, no QtInteractor needed at all -- so the mapping itself is finally exercised end-to-end by
# this suite, not just its None/no-op guard branch (tests/test_arrangement_flip.py) or the
# dialog-level wiring against a stub (tests/test_view_dialog.py).


def test_read_zoom_uses_parallel_scale_under_orthographic():
    from dynamix.shell.arrangement.camera import _read_zoom

    cam = pv.Camera()
    cam.SetParallelProjection(True)
    cam.parallel_scale = 3.5
    cam.view_angle = 45.0        # untouched by the orthographic read -- proves the branch, not luck

    assert _read_zoom(cam) == 3.5


def test_read_zoom_uses_view_angle_under_perspective():
    from dynamix.shell.arrangement.camera import _read_zoom

    cam = pv.Camera()
    cam.SetParallelProjection(False)
    cam.view_angle = 12.5
    cam.parallel_scale = 3.5     # untouched by the perspective read -- proves the branch, not luck

    assert _read_zoom(cam) == 12.5


def test_write_zoom_round_trips_under_orthographic():
    from dynamix.shell.arrangement.camera import _read_zoom, _write_zoom

    cam = pv.Camera()
    cam.SetParallelProjection(True)

    _write_zoom(cam, 2.25)

    assert cam.parallel_scale == 2.25
    assert _read_zoom(cam) == 2.25


def test_write_zoom_round_trips_under_perspective():
    from dynamix.shell.arrangement.camera import _read_zoom, _write_zoom

    cam = pv.Camera()
    cam.SetParallelProjection(False)

    _write_zoom(cam, 40.0)

    assert cam.view_angle == 40.0
    assert _read_zoom(cam) == 40.0


def test_write_zoom_under_perspective_does_not_touch_parallel_scale():
    from dynamix.shell.arrangement.camera import _write_zoom

    cam = pv.Camera()
    cam.SetParallelProjection(False)
    cam.parallel_scale = 9.0

    _write_zoom(cam, 25.0)

    assert cam.parallel_scale == 9.0       # untouched
    assert cam.view_angle == 25.0


def test_write_zoom_under_orthographic_does_not_touch_view_angle():
    from dynamix.shell.arrangement.camera import _write_zoom

    cam = pv.Camera()
    cam.SetParallelProjection(True)
    cam.view_angle = 55.0

    _write_zoom(cam, 4.0)

    assert cam.view_angle == 55.0          # untouched
    assert cam.parallel_scale == 4.0


def _screen_span(plotter):
    """The on-screen distance (pixels) between two OFF-AXIS world points, projected through the
    LIVE camera's own composite projection matrix -- the exact mechanism ``Scene.pick``/
    ``_camera_mvp`` already use (``scene.py``). A direct, render-free way to measure "how zoomed
    in is the camera right now" without a screenshot -- the empirical measurement this file's own
    section-1 precedent (a plain offscreen ``pv.Plotter``) already relies on for camera behavior."""
    viewport = (200, 200)
    mvp = pv.array_from_vtkmatrix(
        plotter.camera.GetCompositeProjectionTransformMatrix(1.0, -1, 1))
    pts = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])   # off the camera's own viewing axis
    pts2d, visible = project_points(pts, mvp, viewport)
    assert visible.all()
    return float(np.linalg.norm(pts2d[0] - pts2d[1]))


def test_smaller_zoom_value_is_more_zoomed_in_under_both_projections():
    """Empirical direction check: confirms -- rather than assumes -- that a
    SMALLER zoom value means MORE zoomed in under BOTH projections, the same direction (not
    opposite ones), so :func:`_read_zoom`/:func:`_write_zoom` need no per-mode sign flip for the
    dialog's Zoom knob to feel consistent across a mode switch (see camera.py's own module
    docstring for the write-up)."""
    from dynamix.shell.arrangement.camera import _write_zoom

    ortho = pv.Plotter(off_screen=True, window_size=(200, 200))
    ortho.add_mesh(pv.Sphere(radius=1.0))
    ortho.camera_position = [(5, 0, 0), (0, 0, 0), (0, 0, 1)]
    ortho.enable_parallel_projection()
    ortho.reset_camera()
    _write_zoom(ortho.camera, 4.0)
    ortho.render()
    span_wide = _screen_span(ortho)
    _write_zoom(ortho.camera, 1.0)
    ortho.render()
    span_narrow = _screen_span(ortho)
    assert span_narrow > span_wide        # smaller parallel_scale -> more zoomed in

    persp = pv.Plotter(off_screen=True, window_size=(200, 200))
    persp.add_mesh(pv.Sphere(radius=1.0))
    persp.camera_position = [(5, 0, 0), (0, 0, 0), (0, 0, 1)]
    persp.disable_parallel_projection()
    persp.reset_camera()
    _write_zoom(persp.camera, 60.0)
    persp.render()
    span_wide2 = _screen_span(persp)
    _write_zoom(persp.camera, 20.0)
    persp.render()
    span_narrow2 = _screen_span(persp)
    assert span_narrow2 > span_wide2      # smaller view_angle -> more zoomed in too, same direction


# ---------------------------------------------------------------------------- MomentumCamera.note_mode
#
# The mode-switcher control (mode_row.py) bypasses reset()
# entirely, so it has to keep _last_flat_mode current itself -- see camera.py's own module
# docstring ("Mode-switcher UI, closed") and note_mode's own docstring.


def test_note_mode_records_a_flat_mode_without_touching_the_scene(qtbot):
    cam, scene = _camera(qtbot)
    mode_before = scene.mode

    cam.note_mode("pacific")

    assert cam._last_flat_mode == "pacific"
    assert scene.mode == mode_before          # note_mode never itself calls Scene.set_mode


def test_note_mode_to_globe_leaves_last_flat_mode_untouched(qtbot):
    cam, scene = _camera(qtbot)
    cam.note_mode("greenwich")
    assert cam._last_flat_mode == "greenwich"

    cam.note_mode("globe")

    assert cam._last_flat_mode == "greenwich"          # globe never overwrites the flat memory


def test_note_mode_then_a_later_globe_reset_returns_to_the_noted_mode(qtbot):
    """The whole point: an external switcher (ModeRow) that changes the scene's mode WITHOUT ever
    calling reset() must still leave a later globe reset() returning to the right place."""
    cam, scene = _camera(qtbot)
    scene.set_mode("pacific")          # the switcher's own Scene.set_mode call
    cam.note_mode("pacific")           # ...and its note_mode call, same modeChanged handler
    scene.set_mode("globe")            # switch to globe -- still bypassing reset()

    cam.reset()

    assert scene.mode == "pacific"


# ==================================================================================================
# 2. Wiring -- the real observer-registration pattern activate() uses, via synthetic VTK events
# ==================================================================================================


def _wired_view(qtbot):
    """An ``ArrangementView`` with a plain offscreen ``pv.Plotter`` substituted for
    ``view._interactor`` and wired EXACTLY the way ``activate()``'s own first-build branch wires
    it (see ``view.py``) -- copied here rather than calling ``activate()`` itself, because
    ``activate()`` needs a real QWidget-compatible interactor to add into its layout (a plain
    ``pv.Plotter`` is not one), the same reason ``tests/test_arrangement_picking.py``'s own click-
    leg tests never call ``activate()`` either."""
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    view._interactor = pv.Plotter(off_screen=True, window_size=(800, 600))
    view._scene = Scene(view._interactor)
    view._camera = MomentumCamera(view._interactor, view._scene)
    iren = view._interactor.iren
    iren.add_observer("LeftButtonPressEvent", view._camera.on_press)
    iren.style.AddObserver("MouseMoveEvent", view._camera.on_move)   # see view.py's own comment:
    # the VTK trackball-camera style GrabFocus()es on LeftButtonPressEvent, which routes every
    # subsequent MouseMoveEvent straight to the style's own dispatcher -- registering directly on
    # the interactor (pyvista's public add_observer) never fires again mid-drag.
    iren.add_observer("LeftButtonReleaseEvent", view._camera.on_release)
    # Activate()'s own wiring now routes 'r'
    # through view.reset_camera (not view._camera.reset directly) so the keypress also updates
    # Scene's camera memory -- see view.py's own module docstring, "The view's own 'r' key
    # binding" section, for why. Mirrored here so this reproduction stays byte-for-byte what
    # activate() actually wires.
    view._interactor.add_key_event("r", view.reset_camera)
    return view


def _drag_view(iren, dx, dy):
    """One press -> one move of ``(dx, dy)`` pixels from the origin -> release, via genuine
    synthetic VTK events on an already-``_wired_view``'d interactor -- the same
    ``SetEventPosition``/``InvokeEvent`` sequence ``test_wired_globe_spin_fires_from_real_
    synthetic_vtk_events`` uses inline, factored out once a THIRD test (the mode-switcher one, below)
    needed the identical sequence."""
    iren.SetEventPosition(0, 0)
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.SetEventPosition(dx, dy)
    iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")


def test_wired_globe_spin_fires_from_real_synthetic_vtk_events(qtbot):
    view = _wired_view(qtbot)
    view._scene.set_mode("globe")
    iren = view._interactor.iren.interactor

    iren.SetEventPosition(0, 0)
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.SetEventPosition(50, 0)
    iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")

    assert view._camera.is_spinning is True


def test_wired_r_key_press_resets_via_add_key_event(qtbot):
    """Fires a genuine ``KeyPressEvent`` (``SetKeyEventInformation`` + ``InvokeEvent``, VTK's own
    key-dispatch mechanism -- confirmed empirically to reach ``add_key_event``'s registered
    callbacks the same way a real keypress on the render surface would) at ``'r'`` and confirms
    the SAME reset ``MomentumCamera.reset()`` performs by direct call."""
    view = _wired_view(qtbot)
    view._scene.set_mode("globe")
    iren = view._interactor.iren.interactor

    iren.SetKeyEventInformation(0, 0, "r", 0, "r")
    iren.InvokeEvent("KeyPressEvent")

    assert view._scene.mode == "mercator"   # switched back to the seeded flat default


def test_wired_r_key_press_also_updates_the_current_views_camera_memory(qtbot):
    """The view's own ``'r'`` binding is now
    ``view.reset_camera`` (see ``_wired_view``'s own comment), not ``view._camera.reset`` directly
    -- so the SAME genuine ``KeyPressEvent`` the sibling test above fires must also leave
    ``Scene``'s own camera-memory dict holding exactly the just-reset live camera state under the
    CURRENT ``camera_key()``. This is the consistency half of the fix (``view.py``'s own module
    docstring, "Per-view camera memory" section) -- NOT the staleness guard itself (that is
    ``Scene.set_frame_mode``/``set_layers``'s own unconditional flip-time resample, covered
    separately in ``tests/test_arrangement_scene.py``)."""
    view = _wired_view(qtbot)
    iren = view._interactor.iren.interactor

    iren.SetKeyEventInformation(0, 0, "r", 0, "r")
    iren.InvokeEvent("KeyPressEvent")

    key = view._scene.camera_key()
    assert key in view._scene._camera_memory
    assert view._scene._camera_memory[key] == view._scene._camera_snapshot()


def test_deactivate_stops_an_in_flight_coast(qtbot, monkeypatch):
    """Momentum stop at the ``ArrangementView`` level (not just ``MomentumCamera``
    directly, above): flip away WHILE a coast is running must not leave it spinning against the
    now-hidden interactor. ``activate()`` cannot be called here (offscreen segfault guard), so
    this drives the SAME wiring ``_wired_view`` reproduces, then calls the real ``deactivate()``.

    A plain offscreen ``pv.Plotter`` (unlike the real ``pyvistaqt.QtInteractor`` ``deactivate()``
    is written for) has no ``.hide()`` -- and ``pv.Plotter`` blocks new instance attributes
    (``_NoNewAttrMixin``), so a bare ``view._interactor.hide = ...`` stub raises. Patched onto the
    CLASS instead, via ``monkeypatch`` (auto-reverted at teardown, so this never leaks into any
    other test's real ``pv.Plotter`` instances) -- the substitution this whole file's "wiring"
    section already leans on throughout, extended by the one extra method this specific test needs
    that the others don't."""
    monkeypatch.setattr(pv.Plotter, "hide", lambda self: None, raising=False)
    view = _wired_view(qtbot)
    view._scene.set_mode("globe")
    iren = view._interactor.iren.interactor
    iren.SetEventPosition(0, 0)
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.SetEventPosition(50, 0)
    iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    assert view._camera.is_spinning is True

    view.deactivate()

    assert view._camera.is_spinning is False


def test_mode_row_reaches_the_wired_scene_and_camera_globe_gate(qtbot):
    """The mode-switcher closing test: the mode-switcher control is not just built, it REACHES the
    real wiring -- setting globe through the view's mode entry point genuinely activates the
    momentum camera's globe-only gate (``on_release``'s own ``self._scene.mode != "globe"`` check,
    the existing camera mechanism ``tests/test_arrangement_integration.py``'s section 1 already
    covers in isolation); switching back to a flat mode genuinely disables it again, AND keeps
    ``_last_flat_mode`` current via ``MomentumCamera.note_mode`` -- the ``camera.py`` module
    docstring's own "a future mode-switcher control should keep this in sync itself" contract,
    now actually exercised end to end rather than just documented.

    **Correction.** ``ModeRow`` has since been retired from this
    view's own face entirely (see ``view.py``'s own module docstring, "Correction" note) -- there
    is no widget left here to click. This test now drives the same entry point the eventual dialog will: ``ArrangementView.set_mode`` (the public passthrough
    ``ModeRow.modeChanged`` used to be wired straight into), proving the wiring THIS test exists
    for is unchanged even though what used to reach it is gone for now."""
    view = _wired_view(qtbot)
    assert view._scene.mode == "mercator"
    iren = view._interactor.iren.interactor

    # -- flat mode: the globe-only spin gate stays closed ------------------------------------
    _drag_view(iren, 50, 0)
    assert view._camera.is_spinning is False

    # -- switch to globe through the view's own set_mode(), not a direct Scene.set_mode call -
    view.set_mode("globe")
    assert view._scene.mode == "globe"

    _drag_view(iren, 50, 0)
    assert view._camera.is_spinning is True          # the gate is open now
    view._camera.stop()               # done observing this coast; note_mode's own contract
                                       # (below) is about _last_flat_mode, not coast lifecycle --
                                       # switching mode is not specified to cancel one in flight

    # -- switch to a flat mode through set_mode() again ---------------------------------------
    view.set_mode("pacific")

    assert view._scene.mode == "pacific"
    assert view._camera._last_flat_mode == "pacific"       # note_mode kept this current

    _drag_view(iren, 50, 0)
    assert view._camera.is_spinning is False          # the gate is closed again


# ==================================================================================================
# 3. Benchmarks
# ==================================================================================================


def _big_two_layer_scene():
    """The design's own benchmark scenario, reusing ``tests/test_arrangement_mask.py``'s fixture
    verbatim (2 draped fields <=2M points each + 50k chain vertices total)."""
    ny = nx = 1414        # 1,999,396 points -- just under field_lonlat_grid's 2M stride cap
    field_a = _big_field(ny, nx, "a")
    field_b = _big_field(ny, nx, "b")
    chains_a = _synthetic_chains(500, 50, ny, nx, seed=1)      # 25,000 vertices
    chains_b = _synthetic_chains(500, 50, ny, nx, seed=2)      # 25,000 vertices -- 50,000 total
    layer_a = Layer(layer_id=1, name="A", source_id="mem:a")
    layer_b = Layer(layer_id=2, name="B", source_id="mem:b")
    entries = [
        {"layer": layer_a, "field": field_a, "result": _result(chains_a), "status": "ok"},
        {"layer": layer_b, "field": field_b, "result": _result(chains_b), "status": "ok"},
    ]
    return entries


def test_flip_in_benchmark_two_fields_at_the_stride_cap_plus_50k_chain_vertices():
    """Bonus number (design: 'print the flip-in (activate+set_layers) timing ... if cheap to
    add'), not asserted. ``ArrangementView.activate()``'s own first-build branch is exactly
    ``Scene(interactor)`` + ``Scene.set_layers(entries)`` (see view.py) -- timed directly here
    since ``activate()`` itself is unreachable offscreen (the segfault guard)."""
    entries = _big_two_layer_scene()

    t0 = time.perf_counter()
    scene = Scene(_plotter())
    scene.set_layers(entries)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nArrangement flip-in (Scene() + set_layers, 2 layers, 50,000 chain vertices, "
          f"1414x1414 draped fields): {elapsed_ms:.1f} ms")
    assert scene.actor_count() == 4      # 2 raster + 2 chains actors -- the build actually landed


def test_projection_mode_switch_benchmark_at_spec_load():
    """The design / plan Global Constraints: 'projection-mode switch <1s at 2x2M-point fields +
    50k chain vertices'.

    **Correction to this test's own original framing.** The first
    version of this benchmark measured ~2.5s/switch and reported that gap against the <1s bar as
    "offscreen software rendering overhead, real hardware is faster" -- the same framing
    ``test_set_mask_benchmark_two_fields_at_the_stride_cap_plus_50k_chain_vertices`` uses in
    ``tests/test_arrangement_mask.py``. That explanation was WRONG for THIS benchmark: profiling
    showed ~86% of the time was ``field_lonlat_grid``'s own ``rasterio.warp.transform`` CRS
    conversion -- CPU/GDAL work, not GPU/render work, and MODE-INVARIANT (the same lon/lat grid,
    recomputed from scratch on every switch even though nothing about the field had changed) --
    so it would have persisted, un-shrunk, on real hardware too. Fixed at the source
    (``Scene._lonlat_cache``/``_lonlat_grid_for``, keyed on each layer's own ``id(field)`` -- see
    ``scene.py``): this benchmark now measures **~100-170 ms/switch** (an ~15-20x improvement),
    comfortably inside the real-hardware <1s bar even on THIS offscreen run, not merely close to
    it. The <5s assert below is kept as the CI safety margin regardless (per the review's own
    instruction) -- offscreen render timing on shared CI hardware can still vary -- but is no
    longer the number doing the real work of proving the spec bar is met; the measured numbers
    printed below are."""
    entries = _big_two_layer_scene()
    scene = Scene(_plotter())
    scene.set_layers(entries)
    assert scene.mode == DEFAULT_MODE

    modes = ("globe", "pacific", "greenwich", "mercator")
    times_ms = []
    for mode in modes:
        t0 = time.perf_counter()
        scene.set_mode(mode)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    for mode, ms in zip(modes, times_ms):
        print(f"\nScene.set_mode({mode!r}): {ms:.1f} ms (2 layers, 50,000 chain vertices, "
              f"1414x1414 draped fields)")
    print(f"Scene.set_mode: mean {np.mean(times_ms):.1f} ms, max {np.max(times_ms):.1f} ms "
          f"over {len(modes)} switches")

    for mode, ms in zip(modes, times_ms):
        assert ms < 5000.0, (
            f"set_mode({mode!r}) took {ms:.1f} ms, exceeds the generous 5000 ms CI bound "
            f"(real-hardware bar is <1000 ms, reported above, not asserted)")


# ==================================================================================================
# 4. End-to-end sweep + the real-app smoke
# ==================================================================================================


class _E2EChainStub:
    """A WTMM-shaped result (via ``_stub_stack``, reused from ``tests/test_shell_window.py``) PLUS
    two chains with real pixel geometry AND ``log2_mod`` -- unlike ``tests/
    test_arrangement_commit.py``'s own ``_ChainStub``/``_chains()`` (which carry no ``log2_mod`` at
    all: fine there, since that file never calls ``Scene.pick``/``Scene.set_mask``, but
    ``Scene.set_mask``'s modulus-percentile visibility needs a FINITE ``max(log2_mod)`` per chain --
    a chain missing it stays permanently invisible, per ``scene.py``'s own ``_point_visibility``
    docstring, which would silently break this sweep's mask/pick steps). Built with
    ``tests/test_arrangement_scene.py``'s own ``_make_chain`` (the established fixture for exactly
    this shape) instead.

    Chain 0 (slope 0.1) has a LOW max(log2_mod); chain 1 (slope 2.0) has a HIGH one -- the mask
    step hides chain 0 and leaves chain 1 pickable, deterministically.
    """

    name = "e2e_chain_stub"
    params = ()

    def compute(self, field, params, *, progress=None):
        out = _stub_stack(field, 1)
        out["chains"] = [_make_chain(1, 1, n=4, slope=0.1), _make_chain(6, 6, n=4, slope=2.0)]
        return out

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


@pytest.fixture
def e2e_devices(clean_registry):
    from dynamix.model.device import register_device

    register_builtin_devices()
    register_device(_E2EChainStub())
    return clean_registry


def test_end_to_end_sweep_two_fields_compute_flip_mask_pick_group_commit_save_load(
        qtbot, e2e_devices, tmp_path):
    """Two synthetic geo fields -> compute both -> Tab -> both draped + vectors -> mask -> pick ->
    group -> commit -> flip back -> session shows group colors -> save/load project -> groups
    survive. Builds on ``tests/test_arrangement_commit.py``'s real-Scene multi-layer pattern
    (``win._arrangement._scene = Scene(pv.Plotter(off_screen=True))``, the same offscreen-
    QtInteractor-segfault substitution used throughout) and ``tests/test_project_menu.py``'s
    save/load window round-trip.
    """
    from dynamix.shell.main_window import MainWindow

    # -- open two synthetic geo fields; compute both -----------------------------------------
    win = MainWindow(steps=(("e2e_chain_stub", {}),))
    qtbot.addWidget(win)

    active_field = _geo_field(tmp_path, "active")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, str(tmp_path / "active.tif"))
    active = win.layer

    bg_field = _geo_field(tmp_path, "bg")
    bg_src = win.project.add_source(str(tmp_path / "bg.tif"))
    bg = win.project.add_layer("bg", bg_src.source_id, active.chain)
    win.add_layer_row(bg, bg_field)
    assert win.layer is active                          # adding a row never re-selects

    # -- Tab: flip in; the offscreen segfault guard means activate() never builds a real Scene
    # (tests/test_arrangement_flip.py) -- substitute one directly, the established pattern
    # tests/test_arrangement_commit.py's own multi-layer test uses, then resync so BOTH layers
    # (the just-cached `active` and the not-yet-computed `bg`) land on it. This also genuinely
    # computes `bg` (a real worker dispatch through _sync_arrangement's queue) -- "compute
    # both" completes by the time this settles.
    win._toggle_center_view()
    win._arrangement._scene = Scene(pv.Plotter(off_screen=True, window_size=(800, 600)))
    win._sync_arrangement()
    qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)
    win._sync_arrangement()                              # pick up bg's now-landed result

    scene = win._arrangement._scene
    assert scene.legend_lines == []                                    # both entries are "ok"
    for layer_id in (active.layer_id, bg.layer_id):
        assert f"layer-{layer_id}-raster" in scene._plotter.actors     # draped
        assert f"layer-{layer_id}-chains" in scene._plotter.actors     # vectors: chains
        assert f"layer-{layer_id}-extrema" in scene._plotter.actors    # vectors: extrema

    # -- mask: hide the low-modulus chain (index 0) on every layer ---------------------------
    # MaskRow lives in MainWindow's right panel -- `win._mask_row`
    # now, not `win._arrangement._mask_row` (the view itself no longer builds one).
    win._mask_row._controls["modulus_pctl"].valueChanged.emit(75.0)
    assert scene._mask == (75.0, 0, 0)
    _active_starts, active_chain_idx = scene._chain_lookup[active.layer_id]
    active_colors = np.asarray(
        scene._plotter.actors[f"layer-{active.layer_id}-chains"].mapper.dataset.point_data["colors"])
    assert (active_colors[active_chain_idx == 0, 3] == 0).all()        # chain 0: masked out
    assert (active_colors[active_chain_idx == 1, 3] == 255).all()      # chain 1: still visible

    # -- pick: aim the real camera at chain 1's own first vertex on the active layer ---------
    pts3d = np.asarray(
        scene._plotter.actors[f"layer-{active.layer_id}-chains"].mapper.dataset.points)
    target_idx = int(np.nonzero(active_chain_idx == 1)[0][0])
    target = pts3d[target_idx]
    cam = scene._plotter.camera
    cam.position = (float(target[0]), float(target[1]), 10.0)
    cam.focal_point = (float(target[0]), float(target[1]), 0.0)
    cam.up = (0.0, 1.0, 0.0)
    viewport = tuple(scene._plotter.window_size)
    hit = scene.pick(viewport[0] / 2.0, viewport[1] / 2.0, viewport)   # mvp=None: the live camera

    assert hit == (active.layer_id, 1)

    # -- group: the pick joins a new group; a second member from `bg` joins by hand (the pick
    # mechanism itself, exercised above, is what this step is actually proving; a second real
    # camera aim isn't needed to prove the cross-layer commit path too) -----------------------
    # GroupPalette + the Commit button live in MainWindow's right panel -- `win._group_palette`/`win._commit_button` now, not `win._arrangement.
    # _group_palette`/`win._arrangement._commit_button` (the view itself no longer builds either).
    palette = win._group_palette
    palette.new_group("e2e_group")
    palette.add_pick(hit, shift=False)
    palette.add_pick((bg.layer_id, 1), shift=True)
    assert palette.groups()["e2e_group"]["chains"] == [(active.layer_id, 1), (bg.layer_id, 1)]

    # -- commit -> flip back -> session shows group colors ------------------------------------
    win._commit_button.click()

    assert decode_groups(active.tags["groups"])["e2e_group"]["chains"] == [1]
    assert decode_groups(bg.tags["groups"])["e2e_group"]["chains"] == [1]

    win._toggle_center_view()
    assert win._center_stack.currentIndex() == 0

    color = tuple(GROUP_COLORS[0])
    item = win.canvas._group_items[color]
    gx, _gy = item.getData()
    assert gx.size > 0                                    # the active layer's session repainted

    # -- save / load project: groups survive ---------------------------------------------------
    proj_path = win._save_project_to(tmp_path / "e2e.dynamix")

    win2 = MainWindow(steps=())
    qtbot.addWidget(win2)
    with qtbot.waitSignal(win2.resolved, timeout=10000):
        win2._open_project_path(proj_path)

    assert len(win2.project.layers) == 2
    active2 = next(l for l in win2.project.layers if l.name == "active")
    bg2 = next(l for l in win2.project.layers if l.name == "bg")
    assert decode_groups(active2.tags["groups"])["e2e_group"]["chains"] == [1]
    assert decode_groups(bg2.tags["groups"])["e2e_group"]["chains"] == [1]
    assert any(s.device == "group_paint" for s in active2.chain.steps)
    assert any(s.device == "group_paint" for s in bg2.chain.steps)

    item2 = win2.canvas._group_items[color]
    gx2, _gy2 = item2.getData()
    assert gx2.size > 0                                   # re-resolve on reload re-painted it too


# --------------------------------------------------------------------------------- real-app smoke


def test_real_app_smoke_never_flipped_path_still_renders(tmp_path):
    """The design's real-app smoke command, run as a genuine subprocess (not an in-process
    ``dynamix.shell.app.main()`` call) -- launches the real launcher script exactly as a user
    would, with a fresh, isolated settings file. Tab is never pressed, so ``self._arrangement``
    stays ``None`` and ``dynamix.shell.arrangement``/``pyvista`` are never even imported ("the
    app must run identically with the arrangement view never instantiated") -- this is the
    regression check that ``view.py`` costs the ordinary session path nothing.

    **Not covered here: the REAL-DISPLAY Tab-flip smoke** -- an actual ``QtInteractor`` on the
    real ``cocoa`` QPA platform, which this offscreen-mandated harness cannot run at all (the
    segfault guard, ``tests/test_arrangement_flip.py``). That check needs a real launch.
    """
    out_png = tmp_path / "arr_smoke.png"
    settings_path = tmp_path / "settings.json"
    env = dict(os.environ)
    env["DYNAMIX_SETTINGS_PATH"] = str(settings_path)
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        ["scripts/shell.sh", "docs/demo/dem_crop.npz", "--render", str(out_png)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert out_png.is_file() and out_png.stat().st_size > 0
    assert "rendered ->" in result.stdout


# ==================================================================================================
# 6. First-flip framing -- the blank-view regression
# ==================================================================================================
#
# The live first-Tab sequence is: activate() builds Scene + MomentumCamera and calls
# camera.reset() while the scene is EMPTY (view.py's own build tail), and only then does
# ``_sync_arrangement`` deliver the first entries via ``set_layers`` -- whose every actor is
# added with ``reset_camera=False`` by design. ``reset()``'s bounds-fit on emptiness parks the
# camera on VTK's default unit cube at the origin; the drape then lands hundreds of Mercator
# degrees away, out of frame: a themed, dark, EMPTY view on real glass, reproducible pixel-for-
# pixel off_screen (0.0000 non-background fraction without a re-frame). The guard lives in
# ``Scene.set_layers``: the empty -> non-empty transition re-frames via ``reset_camera()``,
# which preserves the current projection AND orientation (top-down stays top-down) and only
# re-fits bounds -- so it can never fight the user's navigation on later relayerings.


def _nonbackground_fraction(plotter, background="#131313"):
    """Rendered non-background pixel fraction -- the symptom itself, measured."""
    img = plotter.screenshot(return_img=True)
    bg = np.array(pv.Color(background).int_rgb)
    return float((np.abs(img[..., :3].astype(int) - bg).sum(axis=-1) > 12).mean())


def _first_flip_scene(tmp_path, qtbot):
    """Scene + MomentumCamera composed exactly as activate() composes them, camera reset on
    the still-empty scene -- the live first-flip preamble."""
    plotter = pv.Plotter(off_screen=True, window_size=(400, 300))
    scene = Scene(plotter, background="#131313")
    cam = MomentumCamera(plotter, scene)
    cam.reset()
    layer = Layer(layer_id=1, name="first", source_id="mem:first")
    entry = {"layer": layer, "field": _geo_field(tmp_path, "first"), "result": None,
             "status": "ok"}
    return scene, cam, entry


def test_first_layers_reframe_the_empty_reset_camera(tmp_path, qtbot):
    scene, cam, entry = _first_flip_scene(tmp_path, qtbot)
    assert _nonbackground_fraction(scene._plotter) == 0.0   # the blank precondition is real

    scene.set_layers([entry])

    assert _nonbackground_fraction(scene._plotter) > 0.01   # the drape is actually in frame
    assert scene._plotter.camera.GetParallelProjection() == 1   # top-down orientation kept


def test_relayering_never_reframes_a_navigated_camera(tmp_path, qtbot):
    scene, cam, entry = _first_flip_scene(tmp_path, qtbot)
    scene.set_layers([entry])
    cam_obj = scene._plotter.camera
    cam_obj.position = (cam_obj.position[0] + 0.005, cam_obj.position[1], cam_obj.position[2])
    parked = (cam_obj.position, cam_obj.focal_point, cam_obj.GetParallelScale())

    scene.set_layers([entry])                                # non-empty -> non-empty: no re-frame

    assert (cam_obj.position, cam_obj.focal_point, cam_obj.GetParallelScale()) == parked


def test_empty_to_nonempty_transition_reframes_again(tmp_path, qtbot):
    scene, cam, entry = _first_flip_scene(tmp_path, qtbot)
    scene.set_layers([entry])
    scene.set_layers([])                                     # every layer hidden/removed
    cam_obj = scene._plotter.camera
    cam_obj.position = (200.0, 200.0, 1.0)                   # user (or drift) left it nowhere
    cam_obj.focal_point = (200.0, 200.0, 0.0)

    scene.set_layers([entry])                                # data returns: frame it again

    assert _nonbackground_fraction(scene._plotter) > 0.01
