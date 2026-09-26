# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ArrangementView -- the arrangement (index 1) half of MainWindow's center-zone Tab flip.

Lazy build, park/resume, and the pyvista-missing degradation, plus the real
``pyvistaqt.QtInteractor`` render surface and the ``Scene`` it hosts. This class exists so
``MainWindow`` has something real to flip to (a `QStackedWidget` needs two widgets) and so the
degrade path is provable independently of the scene: pyvista's absence must never crash a Tab
press, only replace the whole widget with a one-line notice (spec section 6, "Dependencies").

Every pyvista/pyvistaqt touch goes through :func:`_import_pyvista`, cached after the first call
(success or failure) for the life of the process -- the ONE place this package references the
optional ``viz`` dependency group. ``dynamix.shell.arrangement.scene`` (which imports ``pyvista``
at its own module top -- lawful there, see its docstring) is likewise never imported here until
:meth:`activate` has already confirmed, via ``self.available``, that both packages are present --
so a bare ``dynamix[gui]`` install never imports ``pyvista`` at all until Tab is pressed with the
``viz`` extra actually installed.

A :class:`~dynamix.shell.arrangement.mask_row.MaskRow` is built eagerly, alongside the
notice/interactor branch below -- it is pure Qt (no pyvista import of its own, see its module
docstring), so unlike the interactor it needs no lazy build. Its ``maskChanged`` signal is wired to
:meth:`_on_mask_changed`, which forwards straight to ``Scene.set_mask`` -- lazily, a no-op when no
scene exists yet (the row cannot actually be interacted with before the first Tab press builds
one, since it lives inside this same not-yet-visible widget).

**Correction: MaskRow relocated.** The paragraph above describes this
view's ORIGINAL shape; it is no longer accurate. ``MaskRow`` now lives in ``MainWindow``'s
right panel (a view-scoped section) instead of being built here -- this view never
constructs one, and has no ``_mask_row`` attribute at all. What survives here is only the forwarding half: :meth:`set_mask` (the public passthrough :meth:`_on_mask_changed`'s body became)
still forwards a payload into ``Scene.set_mask``, lazily, a no-op when no scene exists yet --
``MainWindow`` calls it on every ``MaskRow.maskChanged`` edit, and once more with the panel's
current values right after every :meth:`activate` (replacing this view's former self-read of its
own, now-removed row).

A :class:`~dynamix.shell.arrangement.group_palette.GroupPalette` is built eagerly
alongside ``MaskRow``, for the same reason (pure Qt, no pyvista import). Its ``membershipChanged``
signal is wired to :meth:`_on_membership_changed`, which forwards the palette's current selection
+ group-preview state into ``Scene.set_selection``/``Scene.set_group_preview`` -- lazily, same
no-op-before-a-scene-exists contract as :meth:`set_mask`. The click -> pick -> palette leg
is registered in :meth:`activate` against the REAL ``pyvistaqt.QtInteractor`` (which, as a Qt
widget, still cannot be BUILT under this harness's mandated offscreen QPA platform -- see
:meth:`activate`'s own segfault note) via ``pyvista.Plotter.track_click_position``, handled by
:meth:`_on_click` (converts VTK's own bottom-left-origin display coordinates to the top-left-origin
screen pixels ``Scene.pick``/``core.selection`` use, reads Shift state off the VTK interactor).

**Correction:** an earlier version of this docstring claimed the click leg was
"exercised live only", unlike ``MaskRow``'s wiring -- that was wrong. ``track_click_position`` and
the VTK event it installs a callback for are plain-VTK mechanisms with no Qt dependency of their
own (only ``QtInteractor``'s own CONSTRUCTION needs a real native window); a plain, offscreen
``pv.Plotter`` -- the exact substitution ``tests/test_arrangement_scene.py``/
``test_arrangement_mask.py`` already use throughout for ``Scene`` -- registers
``track_click_position`` and dispatches a real, synthetic ``LeftButtonPressEvent`` (via
``SetEventPosition``/``InvokeEvent``) identically to a live click. ``tests/
test_arrangement_picking.py`` exercises :meth:`_on_click` exactly this way.

A Commit button is built eagerly alongside the palette, same reasoning (pure Qt).
:meth:`_on_commit_clicked` snapshots the palette's current cross-layer membership ONCE and emits
:attr:`groupsCommitted` (``layer_id``, that layer's own slice of the snapshot) once per layer with
at least one committed member, then :attr:`commitFinished` once the whole click has been handled --
``MainWindow`` (the one place that knows what a layer's chain and tags actually are) does the
actual write per ``groupsCommitted``, deferring its own re-resolve/resync to ``commitFinished``;
see ``main_window.py``'s ``_on_groups_committed``/``_on_commit_finished``/
``_ensure_group_paint_step`` for the transaction itself. Both signals were widened/added after a live, per-layer palette pull inside the emit loop turned out to
race a mid-loop arrangement resync -- see :meth:`_on_commit_clicked`'s own docstring.

**Correction: GroupPalette + Commit button relocated.** The two
paragraphs above describe this view's ORIGINAL shape; they are no longer accurate.
``GroupPalette`` and its Commit button now live in ``MainWindow``'s right panel (a second
view-scoped section, alongside ``MaskRow``'s) instead of being built here -- this
view never constructs either, and has no ``_commit_button`` attribute at all. ``_group_palette``
survives, unlike ``_mask_row``, because the click -> pick leg still needs somewhere to read it
FROM -- it is now a stored, settable reference rather than a widget this view owns. What survives here is only the reading/forwarding halves: :meth:`set_group_palette` (called once, right
after ``MainWindow`` constructs this view -- see its own docstring) stores the reference and wires
``membershipChanged`` into :meth:`_on_membership_changed` exactly as the eagerly-built palette used
to wire itself; :meth:`_on_click` still forwards every real click's pick result through it (a
no-op when no palette was ever attached); and :meth:`commit` (the public passthrough
:meth:`_on_commit_clicked`'s body became) still snapshots ``palette.groups()`` exactly ONCE per
click and emits :attr:`groupsCommitted`/:attr:`commitFinished` unchanged. ``MainWindow`` reaches
:meth:`commit` through its own ``_commit_button.clicked`` -> ``_on_commit_button_clicked``, a
no-op-before-``self._arrangement``-exists wrapper mirroring ``_on_mask_row_changed``'s identical
contract for the mask (see ``main_window.py``'s own comments on both).

A :class:`~dynamix.shell.arrangement.camera.MomentumCamera` is built alongside the scene
in :meth:`activate`'s first-build branch (never eagerly -- unlike ``MaskRow``/``GroupPalette``, it
wraps the real render surface itself, so it has the same lazy-build gate ``Scene`` does). Wired to
the interactor's own VTK observers exactly like the click leg just above it: press/move/
release drive the globe-only momentum spin, and ``'r'`` (``plotter.add_key_event``) drives
:meth:`~dynamix.shell.arrangement.camera.MomentumCamera.reset`. See ``camera.py``'s own module
docstring for the full EQSelect port (``app_window.py:3541-3599`` + ``3880-3897``) and its one
deliberate non-port.

**Projection-mode-switcher UI (closed).** A
:class:`~dynamix.shell.arrangement.mode_row.ModeRow` is now built eagerly, same reasoning as
``MaskRow``/``GroupPalette`` above (pure Qt, no pyvista import of its own -- see that module's own
docstring). Its ``modeChanged`` signal is wired to :meth:`_on_mode_changed`, which forwards into
``Scene.set_mode`` (lazily, the same no-op-before-a-scene-exists contract every other eager
control's wiring already uses) and ``MomentumCamera.note_mode`` (lazily too -- the camera itself
is built later, in :meth:`activate`'s first-build branch, exactly like the spin/reset it belongs to). Globe mode, and therefore the momentum spin, is reachable through the running app for
the first time via this control -- the camera mechanism was always a complete, correct port per the design; this closes the one gap between "correct" and "reachable".

**Correction: ModeRow retired from this view too.** The paragraph
above describes this view's shape immediately after the merge-blocking fix that added ``ModeRow``;
it is no longer accurate. ``ModeRow`` was the last of the four eagerly-built controls (after
``MaskRow``/``GroupPalette``/Commit already left) still living directly on this view's own face -- it is retired from here too, leaving this view nothing but a slim header
strip (one "View…" button) plus the render surface itself. This view never constructs a
``ModeRow`` any more and has no ``_mode_row`` attribute at all -- unlike ``MaskRow``/
``GroupPalette``, there is no replacement widget stored here either, because (deliberately) there is NO mode control anywhere in the running app until the View dialog, which reaches this header's own ``View…`` button with a real dialog that will eventually
host a relocated ``ModeRow`` (the FILE, ``mode_row.py``, is untouched -- only this view's
construction of it is gone). Until then, a freshly built ``Scene`` simply starts at its own
``DEFAULT_MODE`` ("mercator", ``scene.py``'s own default). What survives is only the forwarding
half: :meth:`set_mode` (the public passthrough :meth:`_on_mode_changed`'s body became) still
forwards a mode string into ``Scene.set_mode`` + ``MomentumCamera.note_mode``, lazily, the same
no-op-before-a-scene/camera-exists contract every other eager control's wiring already used --
this is the hook into this view, waiting for a caller.

**Themed background.** The freshly built ``Scene`` is now handed
``dynamix.shell.theme.RESTRAINED_DARK.ground`` as its ``background`` argument -- this is the one
place in this module that reads ``theme.py``, and the only reason it may: ``Scene`` itself stays
Qt-free and never imports theme (see its own module docstring's "Themed background" section) --
the view is exactly where a color role gets resolved into a literal string and handed down to
pyvista.

**The View dialog.** :meth:`set_graticule`/:meth:`set_vertical_
exaggeration`/:meth:`set_background`/:meth:`camera_state`/:meth:`set_camera_state`/
:meth:`reset_camera` are new public passthroughs, all sharing the exact no-op-before-a-scene/
interactor-exists guard :meth:`set_mask`/:meth:`set_mode` already use. ``dynamix.shell.
view_dialog.ViewDialog`` holds a real reference to THIS class (not a stub) and calls every one of
them directly -- unlike ``MaskRow``/``GroupPalette``, which never reference this view at all and
rely on ``MainWindow`` to bridge their signals into it, the dialog is offscreen-safe by
CONSTRUCTION rather than by staying decoupled: every call it can make is already guarded here, so
holding the reference costs nothing even with no live interactor (this harness's mandated
offscreen QPA platform, every test in this suite). See ``view_dialog.py``'s own module docstring
for the dialog side of this pairing, and ``main_window.py``'s ``_toggle_center_view`` for the
other caller of :meth:`set_mode`/:meth:`set_graticule`/:meth:`set_vertical_exaggeration`/
:meth:`set_background`: the post-``activate()`` push of whatever was last persisted to
``Settings.view_options``, mirroring exactly what it already did for :meth:`set_mask`.

**Scale-space chain stacking.** :meth:`set_scale_space`/:meth:`default_scale_space_stretch` are
two more Task-7-shaped passthroughs (identical no-op-before-a-scene guard) added by a LATER task
of the same plan -- the View dialog's Display tab "3-D scale-space (stack by log₂ a)" checkbox +
stretch slider reach ``Scene.set_scale_space``/``Scene.default_scale_space_stretch`` through
these. **Fix round 1:** ``main_window.py``'s ``_set_center_view`` pushes the
persisted value UNCONDITIONALLY, alongside ``set_background`` -- NOT gated behind ``frame_mode``
like ``set_mode``/``set_graticule``/``set_vertical_exaggeration`` -- see that method's own
docstring table for why (a session restoring straight into "vector" was silently dropping a
persisted scale-space state under the original, vexag-mirroring gated placement).

**Frame mode.** :meth:`set_frame_mode` is this view's own half of
``Scene.set_frame_mode`` (``scene.py``'s own module docstring, "Frame mode" section) -- store,
forward lazily (same no-op-before-a-scene-exists contract, plus the same pending-value-on-first-
build buffering :meth:`set_layers` already uses, applied in :meth:`activate`'s own first-build
branch). ``MainWindow`` is the one place that decides WHEN this is called -- the
three-view switcher's own state -- this view only knows how to react to it. **Correction:
** this method no longer hides the header's "View…" button while
enabled -- see its own docstring's "Correction" section for why (the Camera/Display tabs behind
that button stay load-bearing in frame mode; only the dialog's own Projection tab is hidden now,
by ``ViewDialog.set_frame_mode`` instead).

**Per-view camera memory -- what actually prevents
staleness.** The anti-staleness guard is entirely ``Scene.set_frame_mode``/``Scene.set_layers``
themselves (``scene.py``'s own module docstring, "Per-view camera memory" section): both call
``remember_camera()`` UNCONDITIONALLY, right before anything else changes, which reads the LIVE
camera fresh off the plotter -- so whatever state the camera actually is in at flip time (mid
user-navigation, freshly reset, mid-momentum-coast, or untouched since the last flip) is exactly
what gets remembered, regardless of what put it there. :meth:`reset_camera` also calling
``Scene.remember_camera()`` right after the real reset is NOT what prevents a stale snap-back on a
later flip-away-then-back -- the next flip's own unconditional live resample would already capture
the reset correctly with or without this call. It exists purely so ``Scene``'s memory dict does
not visibly LAG the screen between a manual reset and whatever flip comes next (a would-be caller
that reads the dict directly, without going through a flip, would otherwise see a stale value) --
consistency, not the staleness fix itself; this view stays exactly as thin a passthrough as every
other camera method here.

**The view's own ``'r'`` key binding.** ``activate()`` used to wire
``add_key_event("r", self._camera.reset)`` DIRECTLY -- bypassing this view's own
:meth:`reset_camera` entirely, so pressing ``'r'`` reset the camera but never updated ``Scene``'s
memory for it. Per the paragraph above this was never a staleness bug (the next flip resamples
live regardless), but it did leave the memory dict needlessly behind the screen until that next
flip. Now wired as ``add_key_event("r", self.reset_camera)`` instead: the identical camera effect
(:meth:`reset_camera` calls the SAME ``self._camera.reset()``, which itself cancels any in-flight
momentum coast FIRST -- its own documented responsibility, see ``camera.py``'s own module
docstring -- before re-framing) plus the memory update. pyvista's OWN default ``'r'`` binding (a
bare ``plotter.reset_camera()``, appended alongside ours rather than replaced -- see
:meth:`activate`'s own comment at the ``add_key_event`` call) still fires too, on the identical
keypress, and still updates no memory of its own -- harmless for the same reason: the next real
flip resamples the live camera regardless of which binding last touched it.

**Box/lasso selection in the 3-D views (the design's last bullet).** Two new module-level classes, both pure Qt (no pyvista import of their own -- like
``MaskRow``/``GroupPalette`` before them, they cost nothing to import even when pyvista never
loaded, though in practice they are only ever built inside :meth:`activate`'s own pyvista-confirmed
branch, alongside the interactor):

- :class:`_RegionSelectFilter` -- a ``QObject`` event filter, installed on the live
  ``pyvistaqt.QtInteractor`` widget itself in :meth:`activate` (``EQSelect``'s own
  ``_DragSelectFilter`` shape: there is no Python-side ``mousePressEvent`` to
  override on a native VTK render widget, so intercepting a gesture ahead of VTK's own trackball
  camera means installing a Qt event filter, not subclassing). Reads
  :attr:`ArrangementView._selection_mode` (below) on every event; its own docstring carries the
  full mode x event decision table. In box/lasso mode it swallows the whole left-button drag
  (press/move/release all return ``True`` -- the VTK trackball never sees it) and draws feedback
  -- a plain ``QRubberBand`` rectangle for box, a top-level :class:`_LassoRegionOverlay` for
  lasso (below) -- then, on release, resolves the captured pixel geometry through
  :meth:`ArrangementView._apply_region_pick`. In click mode it returns ``False`` immediately,
  leaving VTK's own trackball rotate and the existing click-pick observer (:meth:`_on_click`,
  unchanged by this task) exactly as they were. In transect mode it also returns ``False`` --
  DynamiX's transect gesture does not exist yet (deferred), so true
  pass-through is the honest placeholder; see the class's own docstring for why this is not yet
  EQSelect's own move-swallowing transect behaviour.
- :class:`_LassoRegionOverlay` -- EQSelect's own ``_LassoOverlay`` shape, ported
  onto the theme's ``selection_accent`` color instead of EQSelect's hard-coded yellow: a
  *top-level* frameless, translucent, click-through (``WindowTransparentForInput``) window
  positioned exactly over the interactor's own global geometry, repainted on every move with the
  true polyline (a ``QRubberBand`` can only draw a rectangle) plus a dashed closing segment back
  to the start. **Its actual on-screen compositing is not offscreen-testable** (this harness's
  mandated ``QT_QPA_PLATFORM=offscreen`` gives a top-level window nothing real to composite
  against, or a way to inspect rendered pixels) -- kept deliberately trivial, points in and a
  polyline out, so this is the ONLY part of the whole feature whose actual VISUAL correctness is
  not covered by a headless test. Its paint CODE, however, is reachable offscreen (a shown
  top-level widget's ``paintEvent`` genuinely fires under "offscreen" once the event queue is
  flushed) and is smoke-tested for exactly that -- ``tests/test_arrangement_picking.py``'s own
  ``test_lasso_region_overlay_paints_without_crashing`` caught a real ``QtGui.QPointF`` typo
  (``QPointF`` lives in ``QtCore``) this way during this task's own implementation; see the
  class's own docstring.

:meth:`ArrangementView.set_selection_mode` is the new public passthrough the filter reads: stores
``self._selection_mode`` (default ``"click"``, mirroring ``MainWindow``'s own default) -- no scene
or interactor is needed to store it, so unlike :meth:`set_mask`/:meth:`set_mode` there is nothing
to buffer-and-apply-on-first-build either; the filter simply reads whatever is currently stored,
whenever an event arrives, same as it would read any other live attribute. ``MainWindow`` calls
this on every ``set_selection_mode`` mode change AND once more right after every :meth:`activate`
(mirroring exactly how :meth:`set_mask` is pushed on flip-in, ``main_window.py``'s own
``_toggle_center_view``) -- so a mode chosen while the Raster tab was showing is still in effect
the moment the Vector/Globe tab is flipped to.

:meth:`ArrangementView._apply_region_pick` is the region-resolution half -- ``Scene.
pick_in_region`` -> ``GroupPalette.apply_picks`` -- called by the filter on release, but written
as its own guarded method (not inlined into the filter) precisely so it stays testable exactly
like :meth:`_on_click` already is: a no-op before a scene exists, and a no-op on the palette
hand-off specifically when no palette was ever attached, while the pick against ``Scene`` still
runs regardless (same two-tier guard, same reasoning, see that method's own docstring).
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.shell.theme import RESTRAINED_DARK

_pyvista_module = None
_pyvistaqt_module = None
_pyvista_checked = False


def _import_pyvista():
    """Import ``pyvista`` and ``pyvistaqt``, once, caching both modules -- including failure -- so
    a missing ``viz`` install is one ``ImportError`` per process, not one per Tab press. Returns
    the ``pyvista`` module, or ``None`` if either package is absent (``pyvistaqt``'s module is
    cached alongside it, for :meth:`ArrangementView.activate`'s ``QtInteractor`` construction, but
    is not part of this function's return contract -- existing callers that only check truthiness
    of the return value are unaffected)."""
    global _pyvista_module, _pyvistaqt_module, _pyvista_checked
    if not _pyvista_checked:
        try:
            import pyvista
            import pyvistaqt
        except ImportError:
            _pyvista_module = None
            _pyvistaqt_module = None
        else:
            _pyvista_module = pyvista
            _pyvistaqt_module = pyvistaqt
        _pyvista_checked = True
    return _pyvista_module


def _real_window_surface_unavailable() -> bool:
    """True when the running ``QApplication`` uses a QPA platform with no genuine native window
    handle -- ``"offscreen"`` (the test harness's mandated platform; also ``"minimal"``, Qt's
    other synthetic platform) -- under which a real ``pyvistaqt.QtInteractor`` cannot be built
    (see :meth:`ArrangementView.activate`'s comment for the confirmed segfault). ``False`` (build
    for real) whenever there is no live ``QApplication`` to ask, so this never suppresses a real
    build outside the one condition it exists to catch."""
    app = QtWidgets.QApplication.instance()
    return app is not None and app.platformName() in ("offscreen", "minimal")


class _LassoRegionOverlay(QtWidgets.QWidget):
    """Top-level, click-through, translucent lasso-path overlay for the 3-D views -- EQSelect's own ``_LassoOverlay`` shape, ported onto the theme's
    ``selection_accent`` color instead of EQSelect's own hard-coded yellow.

    A ``QRubberBand`` (used for box mode instead -- see :class:`_RegionSelectFilter`) can only
    draw a rectangle, so lasso mode needs something else to show the actual freehand shape being
    traced. A CHILD ``QWidget`` painted directly over the interactor would not reliably composite
    either: the render surface underneath is a native VTK window, not a Qt-painted one. So this is
    a *top-level* ``Qt.Tool | WindowStaysOnTopHint | WindowTransparentForInput`` window instead,
    positioned exactly over the interactor's own global geometry (:meth:`begin`) and repainted on
    every captured point (:meth:`set_points`) -- ``WindowTransparentForInput`` (plus the
    ``WA_TransparentForMouseEvents``/``WA_ShowWithoutActivating`` attributes below) lets the drag
    itself pass straight through to whatever is underneath, so this window only ever paints; it
    never intercepts the gesture :class:`_RegionSelectFilter` is already capturing on the
    interactor widget beneath it.

    **Its actual on-screen compositing is not offscreen-testable.** A top-level window's real
    appearance has nothing meaningful to assert under this harness's mandated
    ``QT_QPA_PLATFORM=offscreen`` -- there is no real display for ``begin``'s ``mapToGlobal``/
    ``show``/``raise_`` to land on, and no way to inspect rendered pixels. Kept deliberately
    trivial -- points in, a polyline out, no state :class:`_RegionSelectFilter` ever reads back --
    so this is the ONLY untested-for-VISUAL-correctness surface the whole feature has; see the
    task report for this exact caveat, and the module docstring's own "Box/lasso selection"
    section. Its paint CODE is not left entirely unexercised, though: a shown top-level widget's
    own ``paintEvent`` genuinely fires offscreen once the event queue is flushed (confirmed
    empirically, and now smoke-tested -- ``tests/test_arrangement_picking.py``'s
    ``test_lasso_region_overlay_paints_without_crashing``), which is enough to prove the drawing
    calls themselves do not raise, even though nothing can assert what they actually drew.
    """

    def __init__(self):
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._pts: list[tuple[float, float]] = []

    def begin(self, widget: QtWidgets.QWidget) -> None:
        """Position this overlay exactly over ``widget`` (global screen coordinates) and show it
        -- called once, by :class:`_RegionSelectFilter`, at the start of a lasso drag."""
        top_left = widget.mapToGlobal(widget.rect().topLeft())
        self.setGeometry(top_left.x(), top_left.y(), widget.width(), widget.height())
        self._pts = []
        self.show()
        self.raise_()

    def set_points(self, pts) -> None:
        """Replace the captured polyline and repaint -- called on every mouse-move while the
        lasso drag is in progress."""
        self._pts = list(pts)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override signature)
        """EQSelect's own paint recipe: a faint fill, a solid 2px stroke along the true
        captured path, and a dashed 1px segment closing it back to the start (the polygon is not
        actually closed until release -- this dashed segment is what tells the user where release
        will close it). Fewer than 2 points (a press with no move yet) draws nothing -- a single
        point has no polyline to show."""
        if len(self._pts) < 2:
            return
        poly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in self._pts])
        accent = QtGui.QColor(RESTRAINED_DARK.selection_accent)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        fill = QtGui.QColor(accent)
        fill.setAlpha(40)
        painter.setBrush(fill)
        painter.drawPolygon(poly)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(accent, 2))
        painter.drawPolyline(poly)
        dashed = QtGui.QColor(accent)
        dashed.setAlpha(130)
        painter.setPen(QtGui.QPen(dashed, 1, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(poly.last(), poly.first())


class _RegionSelectFilter(QtCore.QObject):
    """Qt event filter (the design's last bullet) that turns a box/lasso
    drag on the live 3-D interactor into a :meth:`Scene.pick_in_region` region resolution --
    EQSelect's own ``_DragSelectFilter`` shape, installed on the live ``pyvistaqt.
    QtInteractor`` widget only (:meth:`ArrangementView.activate`, never head-less -- there is no
    real interactor to install onto under this harness's mandated offscreen QPA platform).

    A native VTK render widget has no Python-side ``mousePressEvent`` to override the way
    ``dynamix.shell.canvas.Canvas`` (the RASTER canvas's own box/lasso wiring) does --
    its mouse events drive the VTK trackball camera directly, underneath Qt. Intercepting a
    gesture ahead of that means installing a ``QObject.eventFilter`` on the widget instead of
    subclassing it, which is exactly what this class is.

    **``self._dragging`` is gated BEFORE the live-mode check.** The
    very first version of this method re-consulted ``self._view._selection_mode`` -- the LIVE
    mode -- on every single call, including mid-drag, reasoning that a mode change mid-drag was
    an unreachable edge case with no keyboard focus available to trigger it. That reasoning was
    WRONG: ``c``/``v`` are ``WindowShortcut``-context ``QShortcut``s (``main_window.py``) that
    fire independent of whatever widget has the mouse grab, and Tab's own app-level
    ``eventFilter`` (also ``main_window.py``) is exactly the same kind of independent,
    grab-agnostic interception -- both a hotkey mode-switch and a Tab flip-away mid-drag are
    perfectly ordinary two-handed interactions (mouse held down, other hand on the keyboard), not
    a hypothetical. Re-consulting the live mode on every event meant either could un-swallow a
    ``MouseMove`` the instant the mode left box/lasso (an in-flight drag suddenly reaching the
    VTK trackball -> a camera jump) and/or strand the rubber band or the top-level lasso overlay
    on screen forever, since the very release that would hide it never arrives once this method
    starts returning ``False`` for it. Fixed by gating on ``self._dragging`` FIRST: once a
    gesture is in progress, EVERY event routes through using the gesture's own FROZEN
    ``self._mode_at_press`` -- the live mode is consulted ONLY to decide whether a brand new
    gesture is allowed to START. See :meth:`cancel` for the matching fix on the ``deactivate()``
    side (a Tab flip-away is exactly the "no release ever arrives" case just described).

    **Mode x event decision table** (this method's own contract -- deliberately testable by
    calling :meth:`eventFilter` directly with stub events, no real interactor required; see the
    module docstring's "Box/lasso selection" section for why only :class:`_LassoRegionOverlay`'s
    own painting is NOT reachable this way):

    ==================  =================================  =======================================
    state               event                              result
    ==================  =================================  =======================================
    not dragging,       anything                            ``False`` immediately -- VTK's own
    live mode click                                         trackball rotate and the existing
                                                              click-pick observer
                                                              (:meth:`ArrangementView._on_click`)
                                                              both keep working exactly as before
                                                              this task.
    not dragging,       anything                             ``False`` -- DynamiX's transect
    live mode transect                                       gesture does not exist yet; true pass-through is the
                                                              honest placeholder. Revisit this branch when transect's own two-click capture needs to swallow
                                                              moves the way EQSelect's own transect
                                                              mode does without
                                                              blocking the click observers its
                                                              gesture still needs.
    not dragging,       left MouseButtonPress                ``True`` (swallowed) -- STARTS the
    live mode box/lasso                                      gesture: freezes
                                                              ``self._mode_at_press`` from the live
                                                              mode, records the anchor, shows the
                                                              box/lasso feedback.
    not dragging,       anything else                        ``False`` -- e.g. a right-click, or a
    live mode box/lasso                                      stray release with no drag in
                                                              progress.
    dragging            MouseMove                            ``True`` (swallowed), interpreted per
                                                              the FROZEN ``self._mode_at_press`` --
                                                              grows the rubber band / lasso path
                                                              regardless of what the live mode has
                                                              become. The VTK trackball never sees
                                                              this event, so the camera cannot
                                                              rotate mid-drag, no matter what a
                                                              hotkey/Tab did in the meantime.
    dragging            left MouseButtonRelease               ``True`` (swallowed) -- resolves the
                                                              region under the FROZEN mode through
                                                              ``ArrangementView._apply_region_pick``,
                                                              then :meth:`cancel`\\ s (hides
                                                              feedback, resets state).
    dragging            left MouseButtonPress                ``True`` (swallowed) -- stale-state
                                                              self-heal: a fresh press should never
                                                              arrive while ``_dragging`` is already
                                                              ``True`` (the matching release always
                                                              clears it first); :meth:`cancel`\\ s
                                                              rather than starting a nested gesture.
    dragging            anything else                        ``False`` -- not this filter's event
                                                              to swallow (e.g. a right-click
                                                              mid-drag).
    ==================  =================================  =======================================

    A degenerate lasso (fewer than 3 captured points -- a press released with no real drag) and a
    degenerate box (the drag never actually started, e.g. a press this filter never saw) both
    resolve to no region-pick call at all -- there is nothing a 0- or 1-2-point polygon/box could
    honestly mean, mirroring ``dynamix.shell.canvas.Canvas.mouseReleaseEvent``'s identical
    "fewer than 3 points -> empty list" guard for the raster canvas's own lasso.
    """

    #: "Alt/⌥-drag subtracts" -- the box/lasso op modifier, read at RELEASE only (never at
    #: press), so an Alt-held drag still draws the same box/lasso feedback and only changes what
    #: happens to the result. Mirrors ``dynamix.shell.canvas.LASSO_MODIFIER`` exactly (same
    #: physical key, same ``AltModifier`` Qt reports it as on every platform Qt runs on -- see
    #: that module's own comment on the Control<->Meta swap this SIDESTEPS by naming the physical
    #: Option key directly) -- not imported from there since canvas.py is the raster canvas's own
    #: module, unrelated to this one; a bare re-declaration costs nothing and keeps this module
    #: free of a cross-widget-family import for one constant.
    _SUBTRACT_MODIFIER = QtCore.Qt.KeyboardModifier.AltModifier

    def __init__(self, view: "ArrangementView", widget: QtWidgets.QWidget):
        super().__init__(widget)
        self._view = view
        self._widget = widget
        self._dragging = False
        self._mode_at_press: str | None = None   # which gesture is in progress (box vs lasso)
        self._origin = QtCore.QPoint()            # box mode's anchor corner
        self._pts: list[tuple[float, float]] = []  # lasso mode's captured path
        #: A plain rectangle rubber band, CHILD of the interactor widget -- unlike the lasso's own
        #: top-level overlay, a rectangle composites over a native child window as long as it is
        #: painted by the SAME top-level window as its VTK sibling (Qt's own native-child overlay
        #: support), which is why EQSelect (and this port) uses two different mechanisms for what
        #: look like two similar jobs -- see :class:`_LassoRegionOverlay`'s own docstring.
        self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, widget)
        self._lasso = _LassoRegionOverlay()

    def _scale(self) -> tuple[float, float]:
        """Logical-widget-px -> render-window-px scale (EQSelect's own ``_scale`` pattern: HiDPI screens, or a widget whose Qt size and VTK render-window size have drifted
        apart, mean 1 logical pixel is not always 1 render-window pixel). ``Scene.pick_in_region``
        (like ``Scene.pick`` before it) works entirely in RENDER-WINDOW pixels (the same space
        ``project_points`` maps world points into via ``viewport``), but every Qt mouse-event
        coordinate this filter captures is in LOGICAL widget pixels -- this is the one place that
        seam gets closed, once, right before a region is built and handed to ``Scene``."""
        vw, vh = self._view._interactor.window_size
        ww = max(self._widget.width(), 1)
        wh = max(self._widget.height(), 1)
        return float(vw) / ww, float(vh) / wh

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override signature)
        event_type = event.type()

        if self._dragging:
            # Fix round 1: while a gesture is in progress, the LIVE selection
            # mode is never consulted again -- every event routes through on the gesture's own
            # FROZEN `self._mode_at_press` instead. See the class docstring's own "gated BEFORE
            # the live-mode check" section for exactly why the earlier, unconditional live-mode
            # re-check on every call was a real bug, not a defensible simplification.
            if (event_type == QtCore.QEvent.MouseButtonPress
                    and event.button() == QtCore.Qt.MouseButton.LeftButton):
                # Stale-state self-heal: `self._dragging` is always cleared by the matching
                # release (below) before any other press could legitimately arrive -- a SECOND
                # press showing up while it is still True means some earlier event never reached
                # this filter (a missed release). Cancel the stale gesture/visuals rather than
                # start a nested one on top of it; THIS press is swallowed and dropped, not
                # reinterpreted as the start of a new gesture -- the next clean press starts one.
                self.cancel()
                return True

            if event_type == QtCore.QEvent.MouseMove:
                pos = event.position()
                if self._mode_at_press == "lasso":
                    self._pts.append((pos.x(), pos.y()))
                    self._lasso.set_points(self._pts)
                else:
                    self._band.setGeometry(QtCore.QRect(self._origin, pos.toPoint()).normalized())
                return True

            if (event_type == QtCore.QEvent.MouseButtonRelease
                    and event.button() == QtCore.Qt.MouseButton.LeftButton):
                # Read everything :meth:`cancel` is about to reset into locals FIRST -- resolution
                # below still needs the frozen mode, the anchor and the captured path.
                gesture_mode = self._mode_at_press
                origin = self._origin
                pts = list(self._pts)
                pos = event.position()
                subtract = bool(event.modifiers() & self._SUBTRACT_MODIFIER)
                self.cancel()          # hides feedback, resets _dragging/_pts/_origin/_mode_at_press
                sx, sy = self._scale()
                if gesture_mode == "box":
                    region = ("box", origin.x() * sx, origin.y() * sy, pos.x() * sx, pos.y() * sy)
                    self._view._apply_region_pick(region, subtract)
                else:                                                    # "lasso"
                    if len(pts) >= 3:
                        poly = [(x * sx, y * sy) for x, y in pts]
                        self._view._apply_region_pick(("poly", poly), subtract)
                return True

            return False        # not this filter's event to swallow (e.g. a right-click mid-drag)

        # Not currently dragging: a NEW gesture may only START while the LIVE mode is box/lasso
        # -- this is the ONLY place the live mode is still consulted.
        mode = self._view._selection_mode
        if mode not in ("box", "lasso"):
            return False       # click: VTK rotates + click-pick; transect: deferred (see docstring)

        if (event_type == QtCore.QEvent.MouseButtonPress
                and event.button() == QtCore.Qt.MouseButton.LeftButton):
            self._dragging = True
            self._mode_at_press = mode
            pos = event.position()
            self._origin = pos.toPoint()
            self._pts = [(pos.x(), pos.y())]
            if mode == "lasso":
                self._lasso.begin(self._widget)
                self._lasso.set_points(self._pts)
            else:
                self._band.setGeometry(QtCore.QRect(self._origin, self._origin))
                self._band.show()
            return True

        return False

    def cancel(self) -> None:
        """Abort any gesture currently in progress: hide both the
        box's ``QRubberBand`` and the lasso's top-level ``_LassoRegionOverlay`` -- whichever, if
        either, is actually showing; ``QWidget.hide()`` on an already-hidden widget is a harmless
        no-op -- and reset every piece of drag state (``_dragging``, ``_pts``, ``_origin``,
        ``_mode_at_press``) back to its constructor default. Never resolves a region itself -- an
        aborted gesture reports no picks at all, the same as it never having started;
        :meth:`eventFilter`'s own release branch reads whatever state it still needs into locals
        BEFORE calling this, precisely so this method is free to just reset everything with no
        special-casing for "but a caller still needs X".

        Two callers:

        - :meth:`eventFilter` itself, in two places: on release (state reset AFTER resolution,
          the ordinary end of a gesture) and defensively when a fresh press arrives while
          ``self._dragging`` is somehow already ``True`` (state reset with NO resolution -- see
          that branch's own comment).
        - :meth:`ArrangementView.deactivate`, unconditionally, on every Tab-away. The trace: Tab's own app-level ``eventFilter`` (``main_window.py``) fires independent of
          whatever widget currently holds the mouse grab, so a Tab press mid-lasso is an ordinary
          two-handed interaction, not an unreachable edge case (see the class docstring's own
          "gated BEFORE the live-mode check" section for the matching mid-drag fix). Left
          uncancelled, the top-level, click-through ``_LassoRegionOverlay`` would keep floating
          over whatever tab is now showing -- undismissable, since its own
          ``WindowTransparentForInput`` means no click ever reaches it to close it -- while this
          filter kept holding stale ``_dragging``/``_pts`` state for a gesture the user can no
          longer see or finish.
        """
        self._dragging = False
        self._mode_at_press = None
        self._pts = []
        self._origin = QtCore.QPoint()
        self._band.hide()
        self._lasso.hide()


def install_momentum_observers(iren, camera) -> None:
    """Wire a :class:`~dynamix.shell.arrangement.camera.MomentumCamera` (anything with
    ``on_press``/``on_move``/``on_release``) to a pyvista ``RenderWindowInteractor`` WITHOUT
    disabling the trackball.

    **The VTK rule this exists for** (``vtkInteractorStyle::ProcessEvents``): when an observer is
    registered ON THE STYLE for an event, VTK invokes that observer INSTEAD of the style's own
    handler (``OnMouseMove``, ``OnLeftButtonDown``, ...). pyvista's own
    ``InteractorStyleCaptureMixin`` honours it -- its press observer calls
    ``self.OnLeftButtonDown()`` itself. The move observer has to be on the style (the interactor's
    own MouseMoveEvent observers never fire during a button grab -- see :meth:`ArrangementView.
    activate`'s comment), so it must call ``style.OnMouseMove()`` too, or every drag reaches the
    velocity tracker and never the camera: no rotate, no pan, wheel-zoom still fine. Pinned by
    ``tests/test_arrangement_camera_observers.py``.
    """
    iren.add_observer("LeftButtonPressEvent", camera.on_press)
    style = iren.style

    def _on_move(*args):
        camera.on_move(*args)
        style.OnMouseMove()      # the style's own handler -- rotate / pan / zoom per its state

    style.add_observer("MouseMoveEvent", _on_move)
    iren.add_observer("LeftButtonReleaseEvent", camera.on_release)


class ArrangementView(QtWidgets.QWidget):
    """The arrangement view widget. ``available`` is False when pyvista/pyvistaqt failed to
    import, in which case this widget is nothing but a muted notice label -- no pyvista symbol is
    ever touched on that path, so a bare ``dynamix[gui]`` install degrades instead of crashing."""

    #: Declared at the class level per the interface contract; every INSTANCE overrides it in
    #: ``__init__`` with its own actual availability, so this default is never what a caller
    #: reads off a real widget.
    available: bool = True

    #: Emitted once per layer that has at least
    #: one committed member, in response to a Commit-button click -- ``(layer_id, groups)``,
    #: ``groups`` being that ONE layer's own slice of the commit's snapshot (the shape
    #: ``dynamix.devices.groups.encode_groups`` wants: ``{name: {"chains": [idx, ...], "color":
    #: [r, g, b]}}``). Widened from the originally-reserved ``Signal(int)`` (meth:`commit`'s own docstring, the public name ``_on_commit_clicked`` was renamed to): the payload MUST travel with the signal, not be pulled
    #: back by a live re-read of the palette, which a mid-loop resync can have already mutated by
    #: the time a later layer's turn comes.
    groupsCommitted = QtCore.Signal(int, dict)

    #: Fires once, after every :attr:`groupsCommitted` for one
    #: Commit-button click has already been handled -- the single point ``MainWindow`` defers its
    #: own re-resolve/arrangement-resync to, rather than resyncing once per affected layer inside
    #: :meth:`commit`'s own emit loop (``_on_commit_clicked``, before the rename). Carries no
    #: payload; ``MainWindow`` tracks which
    #: layers it actually wrote (a locked one is refused, never written) itself.
    commitFinished = QtCore.Signal()

    #: The header's one "View…" button, fired straight from
    #: ``QPushButton.clicked`` (see ``__init__`` -- the same signal-to-signal-emit pattern
    #: ``workflow_zone.py``'s own ``remove_button`` already uses in this codebase). Carries no payload; it is connected to a dialog that will host the view-level controls this
    #: face used to build directly (``ModeRow`` now, ``MaskRow``/``GroupPalette``/Commit already
    #: relocated to ``MainWindow``'s right panel in Tasks 4/5 instead -- see the module
    #: docstring's "Correction" notes for exactly which control went where).
    viewOptionsRequested = QtCore.Signal()
    #: Right-click landed on one or more footprints: the hit list, smallest first
    #: (``Scene.footprints_at_screen``). MainWindow turns it into the Import menu; a right-click
    #: that hits nothing emits nothing.
    footprintsRightClicked = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pv = _import_pyvista()
        self.available = self._pv is not None
        self._active = False        # True between activate() and the matching deactivate()
        self._built = False         # True once the scene has been constructed
        self._layers: list = []     # set_layers's last payload; forwarded into Scene once built
        self._footprints: list = [] # set_footprints's last payload; same forwarding rule
        self._previews: list = []   # set_previews's last payload; same forwarding rule
        self._reference_layers: list = []   # set_reference_layers's last payload; same rule
        #: The Vector tab's placement switch, buffered exactly like
        #: `self._layers` above -- a pre-build `set_frame_mode` call (this widget's own
        #: not-yet-visible window between construction and the first `activate()`) is simply
        #: remembered and applied once `activate()` builds the real Scene, same pattern.
        self._frame_mode = False
        self._interactor = None     # pyvistaqt.QtInteractor, built once on first activate()
        self._scene = None          # Scene(self._interactor), built alongside it
        self._camera = None         # MomentumCamera(self._interactor, self._scene)
        # GroupPalette, stored by set_group_palette() -- the widget itself lives in MainWindow's right panel, so this view never builds one. There is no
        # `_commit_button` attribute at all any more (see set_group_palette's own docstring, and
        # the module docstring's "Correction" note) -- the button is entirely MainWindow's now.
        self._group_palette = None
        # Which gesture a box/lasso-mode drag on the 3-D interactor
        # performs -- read live by `_RegionSelectFilter.eventFilter` (installed in `activate`,
        # below); no scene/interactor is needed to store this, so unlike `_frame_mode`/`_layers`
        # there is no pending-value buffering here either -- see `set_selection_mode`'s own
        # docstring. Default "click" mirrors `MainWindow`'s own default (`main_window.py`'s
        # `self._selection_mode = "click"`).
        self._selection_mode = "click"
        self._region_filter = None  # _RegionSelectFilter, built once alongside the interactor

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if not self.available:
            notice = QtWidgets.QLabel("3-D view unavailable — pyvista not installed")
            notice.setAlignment(QtCore.Qt.AlignCenter)
            notice.setProperty("muted", "true")     # theme.py's QLabel[muted="true"] rule
            layout.addWidget(notice)
        else:
            # This view's whole face, now that ModeRow -- the last of
            # the four controls once built directly here -- has followed MaskRow/GroupPalette/
            # Commit out (Tasks 4/5) into MainWindow's right panel or, for ModeRow, into limbo
            # until the dialog exists to hold it (see the module docstring's own "Correction"
            # notes). A slim, theme-consistent QWidget strip -- no QToolBar chrome -- holding one
            # "View…" button; the interactor lands below it once activate() first builds one.
            header = QtWidgets.QWidget()
            header_layout = QtWidgets.QHBoxLayout(header)
            header_layout.setContentsMargins(4, 2, 4, 2)
            header_layout.addStretch(1)
            self._view_options_button = QtWidgets.QPushButton("View…")
            # Same signal-to-signal-emit pattern workflow_zone.py's own remove_button already uses
            # (`self.remove_button.clicked.connect(self.removeRequested.emit)`) -- clicked's own
            # `bool` checked-state argument is simply dropped, since viewOptionsRequested carries
            # no payload of its own.
            self._view_options_button.clicked.connect(self.viewOptionsRequested.emit)
            header_layout.addWidget(self._view_options_button)
            layout.addWidget(header)
            self._header = header
        self._layout = layout

    def activate(self) -> None:
        """Resume this view: build the scene on first use, or un-park a previously built one.

        No-op when pyvista never imported (the notice is the whole widget already). First call
        builds the real render surface -- a ``pyvistaqt.QtInteractor`` (which duck-types as a
        ``pyvista.Plotter``, so ``Scene`` never needs to know it's talking to a Qt widget) --
        added into this widget's own layout, plus a ``Scene`` around it; any ``set_layers`` call
        that arrived before this first build is applied immediately so nothing is lost. Guarded by
        ``self._built`` so a second flip resumes (just re-shows the interactor) rather than
        rebuilding either object (spec section 3, "Lazy + parked").

The mask is deliberately NOT applied here anymore -- ``MaskRow``
        no longer lives inside this view (see the module docstring's "Correction" note), so there
        is no row to read a current value off. ``MainWindow`` calls :meth:`set_mask` with the
        panel's current values right after every call to this method instead (both the first,
        scene-building one and every later park/resume) -- see :meth:`set_mask`'s own docstring.
        """
        if not self.available:
            return
        if not self._built:
            if _real_window_surface_unavailable():
                # Never crash a Tab press (spec section 6) -- confirmed (outside pytest, via a
                # minimal repro) that constructing a real ``pyvistaqt.QtInteractor`` under Qt's
                # "offscreen" QPA platform SEGFAULTS on this macOS/Apple-Silicon setup: VTK's
                # native render-window embedding feeds ``self.winId()`` straight into
                # ``vtkRenderWindow.SetWindowInfo()`` inside pyvistaqt's own
                # ``QVTKRenderWindowInteractor.__init__`` -- unconditionally, before any
                # ``off_screen`` kwarg is even consulted, so no caller-side flag avoids it. A real
                # user session always runs under the "cocoa" platform (this app's only target); "offscreen" is set only by the test harness
                # (``QT_QPA_PLATFORM=offscreen``, mandated repo-wide). Left unbuilt rather than
                # crashing the whole process: ``self._scene`` stays ``None``, and a later
                # ``activate()`` under a real platform still attempts the genuine build (this
                # branch does not set ``self._built``).
                self._active = True
                return
            from dynamix.shell.arrangement.camera import MomentumCamera  # lazy, same reason
            from dynamix.shell.arrangement.scene import Scene   # lazy: only once pyvista is real
            self._interactor = _pyvistaqt_module.QtInteractor(self)
            self._layout.addWidget(self._interactor)
            # Themed background -- the one place this view reads
            # theme.py, handed down as a plain string so Scene itself stays Qt-free (see both
            # modules' own "Themed background" docstring sections).
            self._scene = Scene(self._interactor, background=RESTRAINED_DARK.ground)
            if self._footprints:
                self._scene.set_footprints(self._footprints)
            if self._previews:
                self._scene.set_previews(self._previews)
            if self._reference_layers:
                self._scene.set_reference_layers(self._reference_layers)
            # Apply whatever frame-mode value was buffered before
            # this first build -- same pending-value-on-first-build pattern as self._layers just
            # below (Scene's own default already matches self._frame_mode's False default, so this
            # is a genuine no-op in the overwhelmingly common case, not a needless rebuild).
            self._scene.set_frame_mode(self._frame_mode)
            if self._layers:
                self._apply_layers_to_scene()
            #No mode control lives anywhere in the running app right
            # now -- ModeRow left this view's own face without a replacement (see the module
            # docstring's own "Correction" note) -- so there is nothing left to read a starting
            # mode off here any more. The freshly built Scene simply starts at its own
            # ``DEFAULT_MODE`` ("mercator", ``scene.py``'s own default), which ``MomentumCamera``'s
            # ``_last_flat_mode`` seeding (``camera.py``'s own ``__init__``, reads ``scene.mode``
            # once) picks up unchanged, just below.
            # Click -> pick -> palette. viewport=True is REQUIRED, not cosmetic: the default (viewport=False) callback receives a 3-D WORLD-
            # SPACE pick position (pyvista's own pick_click_position(), a real VTK geometry pick)
            # -- `x, y = point` on that 3-tuple raises ValueError on every real click. Confirmed
            # empirically (see the task report) against a plain offscreen pv.Plotter, firing a
            # real LeftButtonPressEvent via SetEventPosition/InvokeEvent: viewport=True's callback
            # receives the raw 2-D pixel tuple straight from vtkRenderWindowInteractor's own
            # GetEventPosition() (bottom-left origin) -- exactly what _on_click's own docstring
            # documents and flips. side="left" per the design ("click selects a chain").
            self._interactor.track_click_position(self._on_click, side="left", viewport=True)
            # Footprint browser: the right button through the same VTK observer path,
            # resolved by _on_right_click against Scene.footprints_at_screen.
            self._interactor.track_click_position(self._on_right_click, side="right",
                                                  viewport=True)
            # Box/lasso gesture capture --
            # installed on the interactor WIDGET itself (a plain Qt event filter, not a VTK
            # observer like every other wiring in this method), so it sees every mouse event
            # BEFORE VTK's own trackball camera does; in box/lasso mode it swallows the whole
            # drag (see _RegionSelectFilter's own docstring for the full mode x event decision
            # table) so the click-picking/momentum-camera observers just above and below this
            # line never see a box/lasso drag as a rotate or a click. Built here, alongside the
            # interactor, exactly like MomentumCamera just below -- it wraps the real render
            # surface itself, so unlike MaskRow/GroupPalette it cannot be built any earlier.
            self._region_filter = _RegionSelectFilter(self, self._interactor)
            self._interactor.installEventFilter(self._region_filter)
            # The momentum-spin camera + 'r' reset (see camera.py's own module docstring
            # for the full EQSelect port rationale). Wired the identical way track_click_position's
            # own VTK event just above is -- plain vtkRenderWindowInteractor observers, no Qt
            # dependency of their own. Multiple observers on the SAME event name is normal VTK
            # usage (EQSelect itself registers LeftButtonPressEvent twice, once for click-picking
            # and once for the spin drag start); order between the two here is immaterial -- they
            # read the same event, neither consumes or blocks the other.
            self._camera = MomentumCamera(self._interactor, self._scene)
            iren = self._interactor.iren
            # The three registrations below live in install_momentum_observers() (module level,
            # testable off-screen); the MouseMoveEvent one must also drive the style's own handler
            # or it DISABLES the trackball -- see that function's docstring. The comments that
            # follow describe WHY each event lands where it does.
            install_momentum_observers(iren, self._camera)
            # MouseMoveEvent is registered on the INTERACTOR STYLE object, not the interactor
            # itself, unlike every other observer here -- confirmed empirically (see the task
            # report), not assumed: pyvista's default style (vtkInteractorStyleTrackballCamera)
            # calls vtkInteractorObserver::GrabFocus() on its own LeftButtonPressEvent handler (the
            # very press event registered just above ALSO triggers the style's own camera-rotate
            # drag start, unavoidably -- there is no separate "rotate" vs. "spin-tracking" press),
            # which routes every subsequent MouseMoveEvent DIRECTLY to the style's own internal
            # dispatcher for the rest of the drag, bypassing the interactor's normal observer list
            # entirely -- an observer registered via iren.add_observer("MouseMoveEvent", ...)
            # never fires again until release. pyvista's OWN add_observer already works around the
            # identical grab for LeftButtonReleaseEvent/RightButtonReleaseEvent (its own comment:
            # "Release events are swallowed by the interactor, but registering on the interactor
            # style seems to work", referencing pyvista/pyvista issue no. 4976) but does not
            # extend that fallback to MouseMoveEvent, so this one registration targets ``iren.
            # style`` directly, THROUGH ITS OWN TRACKED ``add_observer`` (revised: ``iren.style`` is a ``pyvista.plotting.render_window_interactor.
            # InteractorStyleCaptureMixin`` subclass -- confirmed against the installed pyvista,
            # not assumed -- which carries its own lowercase, TRACKED ``add_observer`` (appends to
            # ``style._observers``, exactly mirroring ``RenderWindowInteractor.add_observer``'s own
            # bookkeeping one level up). Using the tracked API here, not a raw ``AddObserver`` call,
            # matters for cleanup: ``RenderWindowInteractor.close()`` already calls ``self.style.
            # remove_observers()`` when the style supports it (confirmed by reading pyvista's own
            # source) -- so this observer is torn down for free whenever the interactor itself is
            # ever closed, with no bespoke bookkeeping needed here. The one thing the tracked API
            # does NOT do that ``RenderWindowInteractor.add_observer`` does is wrap the callback in
            # pyvista's own ``try_callback`` -- ``MomentumCamera.on_move`` guards its own body
            # against that gap directly (see its docstring).
            # (registered by install_momentum_observers above:
            #   iren.style.add_observer("MouseMoveEvent", <camera.on_move + style.OnMouseMove>)
            #   iren.add_observer("LeftButtonReleaseEvent", self._camera.on_release))
            # pyvista's own default 'r' binding resets the camera to fit visible bounds; add_key_
            # event APPENDS rather than replaces (confirmed empirically), so this runs alongside
            # it, not instead of it -- harmless, since camera.reset()'s own effect (flat top-down
            # re-frame, or globe-first-then-re-frame) is a strict superset of a bare bounds-fit
            # (matches EQSelect's own "idempotent with the QShortcut above" comment,
            # app_window.py:3499-3502). Scoped to the render surface's own key handler only (fires
            # when the interactor itself has keyboard focus) -- no window-level QShortcut is added
            # here; see the task report for why that is a real, narrow, documented limitation
            # rather than a silent gap.
            #
            # Routed through THIS view's own
            # :meth:`reset_camera` rather than ``self._camera.reset`` directly -- see that method's
            # own docstring, and the module docstring's "The view's own 'r' key binding" section,
            # for exactly what changes (momentum-cancel-then-reset is unchanged either way;
            # ``reset_camera()`` calls the identical ``self._camera.reset()``) and why this is
            # consistency, not the actual staleness fix.
            self._interactor.add_key_event("r", self.reset_camera)
            # Initial framing: a freshly built plotter's default camera is not guaranteed to be
            # looking top-down at the draped geography (nothing else in this build path ever calls
            # reset_camera() -- every add_mesh above passes reset_camera=False, see scene.py).
            # Mirrors EQSelect's own _build_cloud() -> _apply_default_view() call immediately after
            # building its first actor.
            self._camera.reset()
            self._built = True
        self._interactor.show()
        self._active = True

    def deactivate(self) -> None:
        """Park this view on flip-away: hide the render surface, never destroy it, so only one
        render window is ever live at a time without throwing the scene away between flips.
        No-op when pyvista never imported, or when nothing has been built yet.

        Stops the momentum camera FIRST: parking the view must not
        leave its 60fps coast timer running against a now-HIDDEN interactor -- ``hide()`` does not
        pause a ``QTimer``, and a coast in flight at flip-away would otherwise keep calling
        ``plotter.render()`` against a widget nobody can see until it either glides to a natural
        stop or the next ``activate()`` resumes it. Dormant in practice only because globe mode
        (the spin's own gate) is unreachable through the running app today -- see ``camera.py``'s
        own module docstring -- not because the risk isn't real.

        Cancels any in-progress box/lasso gesture too: ``_RegionSelectFilter.cancel()`` -- see that method's own docstring for
        the full "Tab mid-lasso would otherwise strand an undismissable overlay" trace. Guarded on
        ``self._region_filter is not None`` for the same reason every other lazily-built
        first-``activate()`` attribute here is -- ``deactivate()`` can fire before anything was
        ever built at all (this harness's mandated offscreen QPA platform never builds a real
        interactor, so ``self._region_filter`` stays ``None`` forever in every test that never
        substitutes one directly)."""
        if not self.available:
            return
        if self._camera is not None:
            self._camera.stop()
        if self._region_filter is not None:
            self._region_filter.cancel()
        if self._interactor is not None:
            self._interactor.hide()
        self._active = False

    def set_layers(self, entries) -> None:
        """Store ``entries`` and forward them into the scene once it exists. A call that arrives
        before the first ``activate()`` (no ``Scene`` yet) is just remembered -- ``activate()``
        applies it to the freshly built scene itself."""
        self._layers = list(entries)
        if self._scene is not None:
            self._apply_layers_to_scene()

    def _apply_layers_to_scene(self) -> None:
        """``Scene.set_layers(self._layers)``, plus the staleness-guard follow-through:
``Scene.set_layers`` reports which ``layer_id``s it just
        pruned its OWN ``_selection``/``_group_preview`` entries for (a background recompute
        landed a NEW result for that layer -- see ``Scene.set_layers``'s own docstring); this is
        the one place that report is turned into ``GroupPalette.prune_layer`` calls, so the
        palette's SEPARATELY-held committed-group membership never keeps naming a chain the scene
        itself has already disowned. The single call site both ``activate()``'s first-build path
        and :meth:`set_layers` go through, so neither can forget the follow-through."""
        pruned = self._scene.set_layers(self._layers)
        if self._group_palette is not None:
            for layer_id in pruned:
                self._group_palette.prune_layer(layer_id)

    def set_mask(self, payload: dict) -> None:
        """Forward ``payload`` (``{"modulus_pctl", "scale_lo", "scale_hi"}``) into
        ``Scene.set_mask``, lazily: a no-op when no scene has been built yet.

        Public passthrough: ``MaskRow`` itself now lives in
        ``MainWindow``'s right panel, not inside this view (see the module docstring's own
        "Correction" note) -- this method is what used to be ``_on_mask_changed``'s private body,
        wired directly to a row this view built itself. ``MainWindow`` calls this both on every
        ``MaskRow.maskChanged`` edit (while the arrangement is showing) and once, with the panel's
        current values, right after every :meth:`activate` -- replacing this view's former
        self-read of its own, now-removed row."""
        if self._scene is not None:
            self._scene.set_mask(**payload)

    def set_group_palette(self, palette) -> None:
        """Store ``palette`` -- a :class:`~dynamix.shell.arrangement.group_palette.GroupPalette``
        ``MainWindow`` now owns and hosts in the right panel's Groups section -- as the reference
        this view reads every pick INTO (:meth:`_on_click` -> ``add_pick``) and snapshots FROM at
        commit time (:meth:`commit`'s own ``groups()`` read -- the one-snapshot rule,
        unchanged). Wires ``palette.membershipChanged`` into :meth:`_on_membership_changed` exactly
        as this view's own, formerly eagerly-built palette used to wire itself before the widget was relocated out of this view.

        Called once, by ``MainWindow._toggle_center_view``'s own first-build branch, right after
        this view is constructed -- unlike :meth:`set_mask`, which re-applies the panel's current
        VALUES on every :meth:`activate`, this hands over a REFERENCE, and a reference only ever
        needs handing over once: the palette itself (unlike the scene) is never rebuilt or reset by
        a park/resume cycle."""
        self._group_palette = palette
        palette.membershipChanged.connect(self._on_membership_changed)

    def set_mode(self, mode: str) -> None:
        """Forward ``mode`` into ``Scene.set_mode`` + ``MomentumCamera.note_mode``, both lazily: a
        no-op before a scene/camera exists (same not-yet-visible-widget, no-op-before-a-scene-
        exists contract :meth:`set_mask` already uses -- neither this view's mask nor its mode can
        actually be touched by a real gesture before ``activate()``'s first-build branch has run;
        unlike :meth:`set_mask`, no pending value is remembered for a later apply either -- a
        pre-build call is simply lost, mirroring exactly what :meth:`set_mask` already does).
        ``note_mode`` is called SECOND, after ``Scene.set_mode`` has already applied -- order is
        immaterial here (``note_mode`` only ever reads ``mode`` itself, never
        ``self._scene.mode``), but this mirrors the "the scene is the state, the camera is a view
        onto it" ordering :meth:`activate` already uses elsewhere in this class.

        Public passthrough: renamed from the private
        ``_on_mode_changed`` -- ``ModeRow`` itself no longer lives inside this view, or anywhere
        else in the running app yet (see the module docstring's own "Correction" note) -- this was
        the hook the dialog would call once it existed. **Correction:** it exists
        now -- ``dynamix.shell.view_dialog.ViewDialog``'s Projection tab calls this on every real
        ``ModeRow.modeChanged``, and ``MainWindow``'s own post-``activate()`` push calls
        it once more with whatever mode was last persisted, mirroring exactly what :meth:`set_mask`
        already does for the mask row's values."""
        if self._scene is not None:
            self._scene.set_mode(mode)
        if self._camera is not None:
            self._camera.note_mode(mode)

    def set_frame_mode(self, enabled: bool) -> None:
        """The Vector-tab/Georeference-tab switch: store, forward
        into ``Scene.set_frame_mode`` (lazily -- a no-op before a scene exists, same pending-
        value-on-first-build contract :meth:`set_layers` already uses, applied in
        :meth:`activate`'s own first-build branch).

        **Correction.** An earlier version of this method also hid this
        view's own "View…" button while ``enabled`` -- reasoned, at the time, that the projection/
        mode dialog it opens has nothing meaningful to show in a native frame (no CRS, no
        projection mode). That reasoning still holds for the PROJECTION control specifically, but
        hiding the whole button also hid the Camera tab's reset/az/el/zoom controls and the
        Display tab's vertical-exaggeration knob -- both of which are exactly as load-bearing in
        frame mode as in geo mode ("camera reset and vertical exaggeration must be
        reachable there"). The button is now ALWAYS visible; ``dynamix.shell.view_dialog.
        ViewDialog.set_frame_mode`` is the narrower fix -- it hides only the Projection tab (the
        one genuinely meaningless control) once the dialog exists, pushed by ``main_window.py``
        alongside this method's own call, not by this view."""
        self._frame_mode = bool(enabled)
        if self._scene is not None:
            self._scene.set_frame_mode(self._frame_mode)

    def set_selection_mode(self, mode: str) -> None:
        """Which gesture a left-button drag
        on the 3-D interactor performs -- ``"click"``, ``"box"``, ``"lasso"`` or ``"transect"`` --
        read live by :class:`_RegionSelectFilter` (installed in :meth:`activate`) on every mouse
        event.

        This is the view's own PASSIVE mirror, not the validating setter -- mirrors
        ``ArrangementView.set_mode``/``dynamix.shell.canvas.Canvas.set_selection_mode``'s own
        identical "store whatever string I'm handed" contract: ``MainWindow.set_selection_mode``
        is the one that raises on an unknown mode and syncs the mode-row buttons; this one just
        stores it. No scene or interactor is needed to store a plain string, so unlike
        :meth:`set_frame_mode`/:meth:`set_layers` there is no pending-value-on-first-build
        buffering here either -- a pre-``activate()`` call is simply remembered on
        ``self._selection_mode`` and read fresh whenever the filter next asks, exactly the same
        whether that first ask happens before or after this view's first :meth:`activate`.

        ``MainWindow`` calls this on every real mode change AND once more, with the current mode,
        right after every :meth:`activate` -- mirroring exactly how :meth:`set_mask` is pushed on
        flip-in (``main_window.py``'s own ``_toggle_center_view``) -- so a mode chosen while the
        Raster tab was showing is already in effect the instant the Vector/Globe tab appears,
        rather than silently reverting to "click" on every flip.
        """
        self._selection_mode = str(mode)

    # -- the View dialog's Camera/Frame/Display capabilities --------
    #
    # Every method below shares the same no-op-before-a-scene/interactor-exists guard every other
    # passthrough in this class already uses -- the View dialog holds a real ArrangementView
    # reference and calls these directly (see dynamix/shell/view_dialog.py's own module
    # docstring), so it must stay fully constructible and functional with no live interactor at
    # all (this harness's mandated offscreen QPA platform never builds one -- see activate()'s own
    # comment).

    def set_graticule(self, enabled: bool) -> None:
        """Forward into ``Scene.set_graticule``, lazily: a no-op before a scene exists. The View
        dialog's Frame tab checkbox reaches the scene through here."""
        if self._scene is not None:
            self._scene.set_graticule(enabled)

    def set_vertical_exaggeration(self, factor: float) -> None:
        """Forward into ``Scene.set_vertical_exaggeration``, lazily: a no-op before a scene
        exists. The View dialog's Display tab knob reaches the scene through here."""
        if self._scene is not None:
            self._scene.set_vertical_exaggeration(factor)

    def preview_raster_values(self, layer_id, values, src_stride: int = 1) -> bool:
        """Forward into ``Scene.preview_raster_values``; False before a scene exists -- the
        live band-preview tick's Vector-view half."""
        if self._scene is not None:
            return self._scene.preview_raster_values(layer_id, values, src_stride)
        return False

    def set_background(self, color: str) -> None:
        """Forward into ``Scene.set_background``, lazily: a no-op before a scene exists. The View
        dialog's Display tab color-swatch button reaches the scene through here."""
        if self._scene is not None:
            self._scene.set_background(color)

    def set_scale_space(self, enabled: bool, stretch: float) -> None:
        """Forward into ``Scene.set_scale_space``, lazily: a no-op before a scene exists. The View dialog's Display tab "3-D scale-space (stack by
        log₂ a)" checkbox + stretch slider reach the scene through here -- the identical
        lazy-passthrough shape :meth:`set_graticule`/:meth:`set_vertical_exaggeration` already
        use just above."""
        if self._scene is not None:
            self._scene.set_scale_space(enabled, stretch)

    def default_scale_space_stretch(self) -> float:
        """Forward into ``Scene.default_scale_space_stretch``, or ``1.0`` (a plain, harmless,
        strictly-positive fallback) before a scene exists -- the View dialog's own checkbox calls
        this once, when checked with no stretch value yet chosen, to compute a sensible starting
        point (``max_grid_dim / (2 * n_scales)``)."""
        if self._scene is not None:
            return self._scene.default_scale_space_stretch()
        return 1.0

    def default_scale_space_max_grid_dim(self) -> float:
        """Forward into ``Scene.default_scale_space_max_grid_dim``, or ``1.0`` before a scene
        exists -- the View dialog's own checkbox uses this to rebind its stretch control's soft
        range ("slider range 0..max_grid_dim") at the same moment it computes the
        default stretch value above, from the SAME on-screen entry (``Scene``'s own shared
        ``_default_scale_space_geometry`` helper)."""
        if self._scene is not None:
            return self._scene.default_scale_space_max_grid_dim()
        return 1.0

    def camera_state(self) -> dict | None:
        """``{"azimuth", "elevation", "zoom"}`` off the live camera, or ``None`` before a real
        interactor exists (the same guard every other passthrough here uses) -- the View dialog's
        Camera tab pulls this once, on open, never streams it live.

        ``azimuth``/``elevation`` are pyvista's own ``Camera.azimuth``/``Camera.elevation``
        properties -- absolute and round-trip-safe (Python-side bookkeeping backing a real
        ``vtkCamera.Azimuth``/``Elevation`` call underneath), unlike the raw VTK calls
        :class:`~dynamix.shell.arrangement.camera.MomentumCamera` drives directly during a spin
        (which do not update this bookkeeping -- a readout taken mid-spin can therefore read stale
        against the camera's true orientation; the dialog's own "pull on open, never live-stream"
        contract does not attempt to catch that). ``zoom`` goes through
        :func:`~dynamix.shell.arrangement.camera._read_zoom`:
        the raw ``Camera.parallel_scale`` this used to read unconditionally is a documented VTK
        no-op under perspective projection -- globe mode's own camera
        (``MomentumCamera._apply_default_view``) -- so an unconditional read/write silently did
        nothing there. ``_read_zoom`` branches on the camera's OWN live projection flag instead;
        see that function's own docstring (in ``camera.py``, the file that already knows which
        mode gets which projection type) for the full rationale and the empirical direction
        check.

        Imported LOCALLY, not at module top -- ``camera.py`` imports ``scene.py`` (for
        ``DEFAULT_MODE``), which imports ``pyvista`` at ITS OWN module top (lawful there, unlawful
        here -- see this module's own docstring's opening paragraph). Reached only past the guard
        above, i.e. only once ``activate()``'s own first-build branch has already imported
        ``camera``/``scene``/pyvista successfully, so this import is always a cheap
        ``sys.modules`` hit, never the first."""
        if self._interactor is None:
            return None
        from dynamix.shell.arrangement.camera import _read_zoom
        cam = self._interactor.camera
        return {"azimuth": cam.azimuth, "elevation": cam.elevation, "zoom": _read_zoom(cam)}

    def set_camera_state(self, azimuth: float, elevation: float, zoom: float) -> None:
        """Push an absolute ``(azimuth, elevation, zoom)`` onto the live camera -- a no-op before
        a real interactor exists. The View dialog's Camera tab DragValues push here on every edit.
        See :meth:`camera_state`'s own docstring, and :func:`~dynamix.shell.arrangement.camera.
        _write_zoom`'s, for why ``zoom`` goes through the mode-aware helper rather than a bare
        ``cam.parallel_scale`` assignment. Imported locally -- see :meth:`camera_state`'s own
        docstring for why."""
        if self._interactor is None:
            return
        from dynamix.shell.arrangement.camera import _write_zoom
        cam = self._interactor.camera
        cam.azimuth = float(azimuth)
        cam.elevation = float(elevation)
        _write_zoom(cam, zoom)
        self._interactor.render()

    def reset_camera(self) -> None:
        """Forward into ``MomentumCamera.reset()`` -- the same ``r``-key path (see
        :meth:`activate`'s own ``add_key_event`` wiring, ROUTED THROUGH this method rather than
        ``self._camera.reset`` directly -- see the module docstring's own "The
        view's own 'r' key binding" section), now reachable from the View dialog's Camera tab
        Reset button too. A no-op before the camera exists (same guard). ``self._camera.reset()``
        itself always cancels any in-flight momentum coast FIRST (its own documented
        responsibility, ``camera.py``'s own module docstring) before re-framing -- unchanged by
        routing the key binding through here instead of calling it directly.

        **Per-view camera memory -- consistency, NOT the
        staleness guard itself.** The actual guard against a stale camera on a later flip is
        ``Scene.set_frame_mode``/``Scene.set_layers`` (see ``scene.py``'s own module docstring,
        "Per-view camera memory" section): both call ``remember_camera()`` unconditionally, on
        every flip, reading the LIVE camera fresh -- so whatever this method (or anything else)
        left the camera at is exactly what gets remembered, with or without the call below. The
        ``self._scene.remember_camera()`` call here exists only so ``Scene``'s own memory dict does
        not visibly lag the screen between a manual reset and whatever flip comes next -- a plain
        consistency measure, not what makes "reset, then flip away, then flip back" work."""
        if self._camera is not None:
            self._camera.reset()
        if self._scene is not None:
            self._scene.remember_camera()

    def _on_click(self, point) -> None:
        """``pyvista.Plotter.track_click_position``'s own callback, registered with
        ``viewport=True`` (see :meth:`activate`'s own comment -- REQUIRED, not cosmetic: the
        default delivers a 3-D world-space pick instead). ``point`` is the raw ``(x, y)`` PIXEL
        tuple straight from ``vtkRenderWindowInteractor.GetEventPosition()``
        (``Plotter.store_click_position``'s own source), in VTK's display-coordinate convention --
        origin at the BOTTOM-left of the render window -- confirmed empirically (see the task
        report) by firing a synthetic ``LeftButtonPressEvent`` at a known ``SetEventPosition`` on a
        plain offscreen ``pv.Plotter`` and reading what the callback actually received back, not
        assumed from pyvista's own (in this instance, misleading) docstring. Flipped here, ONCE, to
        the top-left-origin screen pixels ``Scene.pick``/``dynamix.core.selection.project_points``
        use throughout (the OpenGL/Qt screen convention), so the coordinate-system seam between
        "a VTK click" and "our own picking math" lives in exactly one place.

        Shift state is read directly off the live VTK interactor at click time
        (``vtkRenderWindowInteractor.GetShiftKey()``) -- the one modifier the design's gesture
        needs ("click selects a chain, shift-click adds"; the exact semantics live in
        ``group_palette.py``'s own module docstring). A miss (``Scene.pick`` returns ``None``) is
        forwarded to the palette exactly like a hit -- :meth:`GroupPalette.add_pick` already
        defines what a miss means for each gesture. A no-op on the palette hand-off specifically
        when no palette was ever attached (meth:`set_group_palette` never called) -- the pick against ``Scene`` still runs regardless, since that half is
        independent of who, if anyone, is listening for the result.
        """
        if self._scene is None:
            return
        x, y = point
        w, h = self._interactor.window_size
        y_top = float(h) - float(y)
        picked = self._scene.pick(float(x), y_top, (w, h))
        if self._group_palette is None:
            return
        shift = bool(self._interactor.iren.interactor.GetShiftKey())
        self._group_palette.add_pick(picked, shift)

    def _on_right_click(self, point) -> None:
        """Right button (footprint browser): the identical pixel flip :meth:`_on_click`
        documents (VTK's bottom-left origin -> the top-left screen pixels the picking math uses),
        then :meth:`Scene.footprints_at_screen`; hits are emitted for MainWindow's Import menu."""
        if self._scene is None:
            return
        x, y = point
        w, h = self._interactor.window_size
        hits = self._scene.footprints_at_screen(float(x), float(h) - float(y), (w, h))
        if hits:
            self.footprintsRightClicked.emit(list(hits))

    def set_footprints(self, footprints) -> None:
        """Store, and forward into the scene once it exists -- :meth:`set_layers`'s own rule."""
        self._footprints = list(footprints)
        if self._scene is not None:
            self._scene.set_footprints(self._footprints)

    def set_previews(self, previews) -> None:
        """Store, and forward into the scene once it exists -- :meth:`set_layers`'s own rule."""
        self._previews = list(previews)
        if self._scene is not None:
            self._scene.set_previews(self._previews)

    def set_reference_layers(self, entries) -> None:
        """Store, and forward into the scene once it exists -- :meth:`set_layers`'s own rule."""
        self._reference_layers = list(entries)
        if self._scene is not None:
            self._scene.set_reference_layers(self._reference_layers)

    def set_colormap(self, name: str) -> None:
        """Forward the in-place LUT swap (Scene.set_colormap) once the scene exists."""
        if self._scene is not None:
            self._scene.set_colormap(name)

    def zoom_to_reference(self, ref_id: str) -> None:
        if self._scene is not None:
            self._scene.zoom_to_reference(ref_id)

    def set_reference_visible(self, ref_id: str, visible: bool) -> None:
        for e in self._reference_layers:
            if e.get("ref_id") == ref_id:
                e["visible"] = bool(visible)
        if self._scene is not None:
            self._scene.set_reference_visible(ref_id, visible)

    def _apply_region_pick(self, region, subtract: bool) -> None:
        """``_RegionSelectFilter``'s own release handler ->
        ``Scene.pick_in_region`` -> ``GroupPalette.apply_picks`` -- the 3-D views' own sibling of
        ``MainWindow._apply_region_picks`` (``canvas.py``'s identical box/lasso wiring).

        ``region`` is exactly ``Scene.pick_in_region``'s own contract: ``("box", x0, y0, x1, y1)``
        or ``("poly", [(x, y), ...])``, ALREADY converted from logical widget pixels to
        render-window pixels (``_RegionSelectFilter._scale``'s own job -- this method never
        touches HiDPI scaling itself). ``subtract`` is whether Alt was held at release ("box/lasso = ADD by default, ⌥ = subtract") -- resolved into ``GroupPalette.apply_picks``'s
        own ``op`` string here, the one place that string literal is spelled for this view.

        Split out of the filter itself (not inlined into its ``eventFilter``) precisely so this
        half stays testable exactly like :meth:`_on_click` already is, with the SAME two-tier
        guard and the SAME reasoning: a no-op before a scene exists (the gesture cannot fire
        before :meth:`activate` has built one anyway -- there is no live interactor to drag on),
        and a no-op on the palette hand-off specifically when no palette was ever attached
        (:meth:`set_group_palette` never called) -- the pick against ``Scene`` still runs
        regardless, since that half is independent of who, if anyone, is listening for the
        result, letting a test assert the resolution half alone with no palette attached at all.
        """
        if self._scene is None:
            return
        viewport = tuple(self._interactor.window_size)
        picked = self._scene.pick_in_region(region, viewport)
        if self._group_palette is None:
            return
        self._group_palette.apply_picks(picked, "subtract" if subtract else "add")

    def _on_membership_changed(self, groups: dict) -> None:
        """``GroupPalette.membershipChanged`` -> ``Scene.set_group_preview``/``Scene.
        set_selection``, lazily: a no-op when no scene has been built yet (same contract as
        :meth:`set_mask` -- the palette cannot actually be touched by a real click before
        a scene exists either). Connected by :meth:`set_group_palette`
        rather than ``__init__`` -- the palette is no longer built here."""
        if self._scene is not None:
            self._scene.set_group_preview(groups)
            self._scene.set_selection(self._group_palette.selection())

    # -- the commit transaction (public, button relocated) ---

    def commit(self) -> None:
        """The Commit button: the palette's current cross-layer membership
        (``GroupPalette.groups()`` -- ``{name: {"chains": [(layer_id, chain_index), ...],
        "color": [r, g, b]}}``) names every ``layer_id`` any group has at least one member in,
        and emits :attr:`groupsCommitted` once per affected layer -- "for each layer with
        committed members", per the spec.

        **Relocated.** Renamed from the private ``_on_commit_clicked``
        -- the Commit button itself now lives in ``MainWindow``'s right panel, not inside this view
        (see the module docstring's own "Correction" note), wired through ``MainWindow``'s own
        ``_on_commit_button_clicked``, a no-op-before-``self._arrangement``-exists wrapper (mirrors
        ``_on_mask_row_changed``'s identical contract for the mask -- see ``main_window.py``). A
        no-op here too when no palette was ever attached (``self._group_palette is None``) -- the
        same headless/never-attached contract :meth:`_on_click` and :meth:`_on_membership_changed`
        already keep.

        **Payload fix.** The ORIGINAL version of this method emitted only
        ``layer_id`` and left ``MainWindow`` to pull the payload back with a separate
        ``groups_for_layer()`` call, LIVE against ``self._group_palette``, once per emitted id --
        which is unsafe: handling layer 1's commit can trigger ``MainWindow._sync_arrangement()``
        (a background layer's own re-resolve path), whose fresh, content-equal ``Scene.set_layers``
        call used to read as staleness (see ``scene.py``'s own Critical-1c fix) and prune layer 2's
        membership from the SAME palette BEFORE this loop ever reached layer 2 -- silently
        committing an empty group for every layer after the first. Fixed two ways: (a) the WHOLE
        transaction's membership is read ONCE, here, into ``per_layer`` (a plain dict, immune to
        whatever ``GroupPalette.prune_layer`` does to the LIVE palette afterward) and
        ``groupsCommitted`` now carries that layer's own slice directly -- ``MainWindow`` never
        reads the palette at all; (b) :attr:`commitFinished` fires ONCE, after every
        ``groupsCommitted`` for this click has already landed, so ``MainWindow`` defers its own
        re-resolve/resync to there instead of running one resync per background layer mid-loop
        (the repeated-resync itself is what triggered the staleness misread above).
        """
        if self._group_palette is None:
            return
        groups = self._group_palette.groups()
        per_layer: dict[int, dict] = {}
        for name, spec in groups.items():
            for layer_id, idx in spec["chains"]:
                entry = per_layer.setdefault(int(layer_id), {})
                g = entry.setdefault(name, {"chains": [], "color": list(spec["color"])})
                g["chains"].append(idx)
        for layer_id in sorted(per_layer):
            self.groupsCommitted.emit(layer_id, per_layer[layer_id])
        if per_layer:
            self.commitFinished.emit()
