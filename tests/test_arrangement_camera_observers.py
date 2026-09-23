# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The momentum camera's observers must not disable VTK's trackball ("I can't
click and drag to twist it" in the Vector view).

VTK rule (vtkInteractorStyle::ProcessEvents): when an observer is registered ON THE STYLE for an
event, VTK invokes the observer INSTEAD of the style's own handler (``OnMouseMove`` & co). pyvista's
own ``InteractorStyleCaptureMixin`` honours that by calling ``self.OnLeftButtonDown()`` from its
press observer; ``view.py`` registered ``MouseMoveEvent`` on the style without doing the same, so
every drag reached the velocity tracker and never the camera. Off-screen plotter, no Qt widget:
``pv.Plotter(off_screen=True)`` still owns a real interactor + style that can be driven directly.
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista not installed")
pv.OFF_SCREEN = True

from dynamix.shell.arrangement.view import install_momentum_observers  # noqa: E402


class _Tracker:
    def __init__(self):
        self.presses = self.moves = self.releases = 0

    def on_press(self, *_a):
        self.presses += 1

    def on_move(self, *_a):
        self.moves += 1

    def on_release(self, *_a):
        self.releases += 1


def _plotter():
    p = pv.Plotter(off_screen=True, window_size=(300, 300))
    p.add_mesh(pv.Sphere())
    p.show(auto_close=False)
    return p


def _drag(p):
    vi = p.iren.interactor
    vi.SetEventInformation(150, 150, 0, 0, chr(0), 0, None)
    vi.LeftButtonPressEvent()
    for i in range(1, 15):
        vi.SetEventInformation(150 + 6 * i, 150 + 3 * i, 0, 0, chr(0), 0, None)
        vi.MouseMoveEvent()
    vi.LeftButtonReleaseEvent()


def test_a_bare_style_observer_on_mouse_move_silently_disables_the_trackball():
    """The trap, pinned so nobody re-introduces it: this is what view.py used to do."""
    p = _plotter()
    before = np.array(p.camera.position)
    p.iren.style.add_observer("MouseMoveEvent", lambda *_a: None)
    _drag(p)
    assert np.allclose(p.camera.position, before)
    p.close()


def test_install_momentum_observers_tracks_the_drag_and_keeps_the_trackball_rotating():
    p = _plotter()
    tracker = _Tracker()
    install_momentum_observers(p.iren, tracker)
    before = np.array(p.camera.position)
    _drag(p)
    assert tracker.presses == 1 and tracker.releases == 1 and tracker.moves >= 14
    assert not np.allclose(p.camera.position, before), "the trackball rotate must still run"
    p.close()
