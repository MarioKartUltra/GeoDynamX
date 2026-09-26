# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The arrangement scene manager: per-visible-layer raster draping onto one geography, plus the
4-mode projection switch, vector actors, view-state masking, and LUT
discipline, and screen-space picking + the pre-commit selection/group
preview (Tasks 4-5 and 7).

``Scene`` is Qt-free in behaviour -- it only ever calls methods on the ``plotter`` object handed
to its constructor (a real ``pyvista.Plotter`` in tests, a ``pyvistaqt.QtInteractor`` -- which
duck-types as one -- in the live app via ``ArrangementView.activate()``). It lives under
``dynamix/shell/arrangement/`` (pyvista/Qt territory per the project's), so a MODULE-TOP ``import
pyvista`` is lawful here; the degradation law ("import failure -> Tab shows a one-line notice")
is upheld one level up, by ``view.py``, which never imports this module until its own
``_import_pyvista()`` has already confirmed both ``pyvista`` and ``pyvistaqt`` import cleanly.

Raster draping is the EQSelect-audited idiom:
one ``pv.StructuredGrid`` per layer, points = projected pixel centers, one scalar array = the
pixel values, ``nan_opacity=0`` so masked/sentinel cells render transparent rather than a wrong
colour. The data itself is never reprojected -- only pixel CENTERS are projected for display
(the ``field_lonlat_grid``/``points_lonlat`` already return lon/lat in EPSG:4326;
:func:`project` maps those straight to scene xyz for the chosen mode).

Non-"ok" entries ("computing" / "error:<msg>" / "no-georeference") never get placed geometry --
they are listed, honestly, as rows in one corner text block. Faking a placement for data that
isn't there yet (or errored, or has no CRS) is exactly the kind of thing the design's "never a
crash, never a fake placement" rules out. A layer whose VECTOR build fails (a malformed chain, a
bad ROI tuple, ...) demotes the whole layer -- raster included -- to the same "error:<msg>" legend
line: :meth:`Scene._add_layer_geometry` rolls back every actor it added for that layer before
re-raising, so a half-built layer never leaves a stray mesh on screen next to its own "errored"
legend row.

**Vector actors.** Per "ok" layer, up to three more actors, their names appended into the
SAME ``self._layer_actors[layer_id]`` list the raster actor's name already lives in (so
``actor_count()`` and ``clear()`` cover them for free):

- ``layer-<id>-chains``: every chain in ``result["chains"]`` with >= 2 points, as ONE
  ``pv.PolyData`` of line cells (one polyline per chain). Per-chain color: ``chain["group_color"]``
  if present (a committed group), else :data:`SEAM_COLOR` if ``chain["tags"]`` is truthy (a
  ``chain_classify`` seam flag), else :data:`VTRAIL_COLOR` (the plain V-trail amber) -- see
  :func:`_chain_color`.
- ``layer-<id>-extrema``: the result's FINEST-scale extrema layer only (``result["extrema"][0]``,
  index 0 by this codebase's fixed convention -- see ``dynamix.devices.chain_classify``'s own
  docstring), as a point cloud in :data:`EXTREMA_COLOR`. Arrangement does not track a "displayed
  scale" the way the session canvas's scrubber does; showing every scale's extrema at once would
  be visual noise on a whole-geography view, so v1 shows only the finest one.
- ``layer-<id>-roi``: when the result carries ``result["_roi"]["roi"] = (row, col, h, w)``, its
  bounding rectangle as a closed line loop in :data:`ROI_BOUNDS_COLOR`.

**Vector bookkeeping (the picking seam).** ``self._chain_lookup: dict[layer_id, (starts,
chain_indices)]``:

- ``starts``: ``int64`` array, length ``len(result["chains"]) + 1`` -- CSR-style offsets, keyed by
  a chain's own index in ``result["chains"]`` (chain identity = list position, the convention the
  commit transaction and ``group_paint``/``group_filter`` already use throughout this plan): chain
  ``i``'s points in the chains actor's ``PolyData.points`` are ``[starts[i]:starts[i+1]]``. A
  chain with < 2 points (nothing to draw a line segment between) gets a zero-width slice, not a
  gap, so ``starts`` always has one entry per ORIGINAL chain plus one, regardless of how many were
  actually drawable.
- ``chain_indices``: ``int64`` array, length = the chains actor's own ``n_points`` -- ``starts``
  expanded to one entry per point (``np.repeat``): the direct ``chain_indices[vertex_id]`` lookup
  a picker wants, kept alongside ``starts`` rather than derived from it on every call because
  the :meth:`Scene.set_mask` (below) needs exactly this same expansion on every call too.

A layer with no drawable chains (no chains, or none with >= 2 points) is simply absent from
``self._chain_lookup`` -- there is nothing to look up.

**Masked-out points stay PICKABLE -- picking consults the visibility mask itself.**
:meth:`Scene.set_mask` (below) hides a chain by zeroing its points' ALPHA channel, never by
removing geometry or toggling actor visibility -- and a VTK point/cell picker has no concept of
per-point scalar alpha; it hit-tests geometry, which is still there, fully opaque or not. A picker
built against ``self._chain_lookup`` alone would therefore happily return a chain the user cannot
currently see. :meth:`Scene.pick` closes this by RECOMPUTING the exact same visibility
:meth:`_apply_mask_to_layer` uses -- both now share :meth:`_point_visibility`, so there is exactly
one place this logic lives, not two that could quietly drift apart.

**Picking.** :meth:`Scene.pick` is pure screen-space math over each layer's chains actor
(``dynamix.core.selection.project_points`` + ``nearest_point``, Qt-free, unit-testable with an
injected camera matrix): every VISIBLE point (frustum-visible per ``project_points`` AND
mask-visible per :meth:`_point_visibility`) across every layer with drawable chains is a pick
candidate; the globally nearest one within :data:`_PICK_MAX_DIST_PX` (8 px, the design's pick bar)
wins, returned as ``(layer_id, chain_index)``. ``mvp=None`` derives the matrix from the live
``self._plotter.camera`` (``vtkCamera.GetCompositeProjectionTransformMatrix``); tests inject a
known one directly. A miss (nothing within range, or no drawable chains anywhere) is ``None``.

**Region picking (the design's last bullet).** :meth:`Scene.
pick_in_region` is :meth:`pick`'s box/lasso sibling, reached by the 3-D views' own gesture
capture (``view.py``'s ``_RegionSelectFilter``) instead of a single click. Both methods now share
ONE candidate-assembly implementation, :meth:`_pick_candidates` -- extracted, additively, from
what used to be :meth:`pick`'s own inline loop (byte-identical behaviour; :meth:`pick`'s existing
tests pass unmodified) -- so a box/lasso drag can never disagree with what a plain click could
ever have hit: the exact same mask-visible ∧ frustum-visible candidate pool feeds both. Where
:meth:`pick` narrows that pool to the single globally nearest point, :meth:`pick_in_region` narrows
it to every point ``dynamix.core.selection.points_in_box``/``points_in_polygon`` reports as inside
the region, then reports a chain as selected the moment ANY ONE of its own candidate points is
inside -- not every point, which would make a thin sliver of an otherwise-enclosed chain invisible
to the gesture (EQSelect's own ``chains_in_box``/``chains_in_polygon`` "any point" rule). Returns a sorted, de-duplicated
``list[(layer_id, chain_index)]`` -- deterministic regardless of ``self._chain_lookup``'s own
insertion order or which candidate point within a chain happened to land inside first.

**Selection highlight + group preview.** :meth:`Scene.set_selection` (the white "just
picked" highlight, :data:`SELECTION_COLOR`) and :meth:`Scene.set_group_preview` (each
not-yet-committed group's OWN color, from ``GroupPalette``'s live state) both recolor through the
SAME per-point RGBA path :meth:`set_mask` already uses -- no actor/mapper/``PolyData`` is ever
touched. Both survive a later rebuild (``set_layers``/``set_mode``), exactly like the active mask.
Compositing order in :meth:`_apply_mask_to_layer`: base color -> mask alpha -> group-preview RGB
override -> selection RGB override (selection wins where both apply -- it marks what the cursor
just touched, a more urgent signal than steady-state group membership). Neither override ever
touches alpha -- a masked-out (alpha-zeroed) chain that is also selected or grouped stays
invisible; the override only changes what color it WOULD be if visible, matching that a fully
transparent point's RGB is moot to begin with.

**Masking.** :meth:`Scene.set_mask` is a pure view-state filter over VECTOR actors,
applied WITHOUT ever creating or destroying an actor (the design's <16ms bar: the whole point is
that this is cheap enough to run on every knob nudge). See its own docstring for the exact
modulus/scale-depth semantics -- ``scale_lo``/``scale_hi`` bound a chain's own DEPTH (how many
scales its ridge reaches), ``scale_hi == 0`` meaning "no cap" (mirrors
``dynamix.devices.filters.HLineLength``'s own ``max_len`` convention). Only the chains actor's
per-point alpha channel changes; the extrema and ROI-outline actors are untouched by masking (they
carry no per-chain modulus/scale statistic to filter by). The chosen mask (``self._mask``) is
remembered and silently reapplied after every subsequent rebuild (``set_layers``/``set_mode``), so
switching projection mid-exploration does not reset an active filter -- see the "stays pickable"
note above for the one thing this masking mechanism does NOT do.

**Themed background.** ``Scene.__init__`` takes an optional
``background`` -- a color string (anything ``pyvista.Plotter.set_background`` accepts, in
practice a hex string) -- and, when given, calls ``self._plotter.set_background(background)``
ONCE, at construction. ``Scene`` stays Qt-free even here: it never imports or reaches into
``dynamix.shell.theme`` itself, it just takes a plain string. The live app's own caller
(``ArrangementView.activate()``) is the one place that reads ``dynamix.shell.theme.
RESTRAINED_DARK.ground`` and passes it in -- the same "the view knows about theme, the scene
only knows about pyvista" split ``view.py``'s own module docstring already draws for every other
Qt-vs-pyvista seam in this pair of modules. Default ``None`` leaves pyvista's own default
background untouched -- every existing offscreen ``Scene`` test predates this parameter and
still constructs a plain ``Scene(plotter)``, so this default is load-bearing, not cosmetic.

**Colormap swap (the EQSelect anti-pattern done right).** :meth:`Scene.set_colormap`
reassigns each RASTER actor's ``mapper.lookup_table`` to a freshly built ``pv.LookupTable``, never
touching the actor/mapper/``PolyData`` objects themselves or calling ``add_mesh`` again. Vector
actors are unaffected -- their color is per-point RGBA DATA (group/seam/trail identity), never a
scalar-mapped LUT.

**The View dialog's Camera/Frame/Display capabilities.**
:meth:`Scene.set_vertical_exaggeration` stores a factor and rebuilds (threaded into every
``project(..., vexag=...)`` call site above -- raster, chains, extrema, ROI outline, and the new
graticule); :meth:`Scene.set_graticule` adds/removes exactly one actor, a 10-degree lat/lon grid
(:func:`_graticule_geometry`) projected through the CURRENT mode and rebuilt alongside everything
else on a mode switch (:meth:`_rebuild`'s own tail); :meth:`Scene.set_background` is the RUNTIME
sibling of the constructor's one-time ``background`` argument just above -- same plain hex-string
contract, called on every Display-tab color pick rather than once at construction. None of the
three import ``dynamix.shell.theme`` or any other Qt module -- ``Scene`` stays exactly as Qt-free
as it always has been; ``ArrangementView``'s own passthroughs are what a real Qt dialog reaches
through to get here (``arrangement/view.py``'s own module docstring).

**Frame mode.** :meth:`Scene.set_frame_mode` is the Vector-tab sibling of :meth:`set_mode` -- stores a
bool and rebuilds, mirroring its shape exactly. In frame mode, every placement call site above
(raster, chains, extrema, ROI outline) is rerouted from ``geo.mapping``'s CRS pipeline
(``field_lonlat_grid``/``points_lonlat`` + :func:`~dynamix.core.projection.project`) to the
field's own ``x_axis``/``y_axis`` used DIRECTLY as planar coordinates (bypassing
``field.frame.to_scene``, which is a pass-through for ``LocalFrame`` but routes a
``GeographicFrame`` through the world-map ``mod 360`` longitude canonicalisation, tearing any
seam-crossing raster into edge strips with the bridging cells smeared across the width; native-
frame display is seam-free by doctrine, see :meth:`_scene_points`). No CRS, no projection mode,
and critically no :class:`NoGeoreference` can ever be raised (the exception exists to guard the
CRS pipeline this mode never calls). The
:meth:`_scene_points` helper is the one seam :meth:`_build_chain_geometry`, the extrema block of
:meth:`_build_vector_actors`, and :meth:`_build_roi_actor` all route their point placement through
now, geo or frame; :meth:`_add_raster_actor` has its own parallel frame-mode branch (a whole-field
grid, not an indexed point list, so it does not go through :meth:`_scene_points`) that reuses
``geo.mapping``'s own :func:`~dynamix.geo.mapping._stride_for` decimation so the two draping paths
pick identical stride for the identical shape/budget -- :meth:`_lonlat_grid_for`'s CRS-transform
cache is untouched and unused here (there is no CRS transform to cache: plain ``x_axis``/``y_axis``
indexing is already cheap). A ``kind="points"`` entry has no frame-mode placement at all (points
are WGS84-placed by contract, :meth:`_add_points_geometry`'s own docstring) -- :meth:`_rebuild`
demotes it directly to a dedicated ``"frame-points"`` legend status ("session view only (frame
mode)") without ever calling :meth:`_add_points_geometry` (which also guards itself with a
``ValueError`` for any other caller). The graticule (lon/lat geometry) never draws in frame mode
either -- see :meth:`_apply_graticule`. **Layer admission -- which entries even reach this Scene
in frame mode, e.g. filtering to the active layer's own compatible frames
(:func:`dynamix.core.frames.frames_compatible`) -- is the CALLER's job**; this Scene
trusts whatever ``status`` an entry already carries, exactly as it always has.

**Per-view camera memory ("vector-view raster stretched all weird").**
Before this task, the camera was fit exactly ONCE per ``Scene`` lifetime (:meth:`set_layers`'s own
empty->non-empty transition, below) and never again -- so a geo<->frame flip (or a frame-mode
active-layer swap) left whichever mode's camera happened to be live parked on screen while a
totally different-scale geometry was drawn underneath it (measured: a ~0.005-unit lon/lat window
reused, untouched, against a 126-unit-wide native-frame mesh -- an ~11,500x mismatch).
:meth:`camera_key` names the CURRENT view: ``"geo:<mode>"`` (geo mode draws every admitted layer
in one shared, WGS84-projected coordinate system, so only the projection mode distinguishes one
view from another) or ``"frame:<bounds-sig>"`` (frame mode's own coordinate system is native to
whichever geometry is on screen, so the key is that geometry's own combined extent --
:meth:`_frame_bounds_signature`). :meth:`remember_camera` snapshots the live camera's full
navigable state (position/focal_point/up/parallel_scale/parallel_projection/clipping_range) under
THAT key; :meth:`restore_camera` re-applies a remembered state by key, returning whether one was
found. Both :meth:`set_frame_mode` (a geo<->frame flip) and :meth:`set_layers` (a frame-mode
active-layer swap -- see that method's own "Frame-mode active-layer swap" comment) follow the
same contract: remember the OUTGOING view under its own key before anything changes, then, once
the new geometry is on screen, restore the INCOMING key's own remembered state -- or, the first
time this ``Scene`` has ever shown that exact key, ``reset_camera()`` fits it fresh, exactly like
the pre-existing empty->non-empty framing below.

**This "remember before anything changes" call is what actually prevents staleness -- and it is
UNCONDITIONAL, reading the live ``self._plotter.camera`` fresh every time, regardless of what put
the camera in its current state.** Whether the user navigated the trackball, a momentum coast
glided to a stop, ``reset_camera()``/``r`` re-framed it, or nothing happened at all since the last
flip -- the very next flip's own ``remember_camera()`` call snapshots whatever is ACTUALLY on
screen at that moment, not some earlier cached notion of it. Any OTHER caller that also happens to
call ``remember_camera()`` outside a flip (``ArrangementView.reset_camera()`` does, after a manual
``'r'``/Reset-button reset -- ``view.py``'s own module docstring, "Per-view camera memory" section)
is consistency, keeping this dict from visibly lagging the screen in between flips -- it is never
load-bearing for correctness the way the flip-time call above is, since the next flip resamples
live regardless of whether that extra call ever happened. A SAME-key resync (identical bounds, e.g. a plain
re-filter of the same field -- every ``Filter`` always mints a fresh ``result`` dict,
``dynamix/engine/resolve.py``'s own module docstring) never remembers or restores anything at all
-- the camera stays exactly where the user left it; live exploration never gets yanked out from
under itself. Geo behavior for a user who never switches PROJECTION modes -- let alone ever visits
frame mode -- is untouched: :meth:`set_mode` (switching mercator/pacific/etc. within geo) does not
participate in this scheme at all; only the geo<->frame boundary the design scoped
does.

**Per-layer colormap + trail color (the model half).** ``set_layers``'s
entries may now carry ``"colormap"`` and ``"vtrail_color"`` (read off a layer's own ``ui.*`` tags
by ``main_window._sync_arrangement``): :meth:`_add_raster_actor` drapes THAT layer with its own
colormap instead of the one shared ``self._colormap`` :meth:`set_colormap` sets, and
:func:`_chain_color` falls back to that layer's own trail color instead of the module's
:data:`VTRAIL_COLOR` constant for a plain (untagged, ungrouped) chain -- group and seam colors are
untouched, they already win over any layer-wide default. Both keys are optional: an entry that
omits either (every ``set_layers`` call that predates this task, and every existing offscreen test)
gets exactly the old behavior, ``self._colormap``/``VTRAIL_COLOR``. ``Scene.set_colormap`` itself
is unchanged -- still a scene-wide LUT swap with zero callers in the live app -- and still wins on
a raster whose OWN entry carries no colormap preference.

**Scale-space chain stacking.** :meth:`Scene.set_scale_space` mirrors
:meth:`set_mode`/:meth:`set_frame_mode`'s own shape exactly: store, then rebuild, no-op when
every stored value already matches. When enabled, :meth:`_build_chain_geometry` lifts every
CHAIN point (never extrema/ROI/raster -- those stay at their existing height) to
``z = stretch * (log2 a - log2 a_min)`` using that point's OWN ``log2_scales`` value -- finest
scale at ``z = 0``, coarser scales rising, the literal "extrema converge to the finest point"
picture in-scene. ``a_min`` is the minimum log2-scale value across every point of every
DRAWABLE (>= 2 point) chain in the layer being built -- not a separately-threaded
``result["scales"][0]``, because ``wtmm_backend.chains2d``'s own finest-anchored-prefix
guarantee (this task's own FIRST-STEP verification, recorded in the task-6 report) means every
real chain's ``log2_scales[0]`` already equals that value exactly; computing it from the drawn
points themselves needs no extra plumbing and stays correct even for a hypothetical future
chain source whose chains do not all share one global ladder.

**Composition with vertical exaggeration -- a DELIBERATE divergence from EQSelect's own
choice.** The lift is computed BEFORE :attr:`self._vexag`'s multiply (threaded as the
``height`` argument to :meth:`_scene_points`, which applies it exactly where every other
draped height already goes through ``project(..., vexag=...)``/``frame.to_scene(...)`` then
``*= self._vexag``) -- so the FINAL z is ``vexag * stretch * (log2 a - log2 a_min)``: vertical
exaggeration composes as an ADDITIONAL stretch on top of the scale-space lift. EQSelect's own
``_project_te`` instead calls ``_project_lld(..., vexag=1.0)`` -- it BYPASSES
its own vertical-exaggeration slider entirely in scale-space mode, so ``te_vexag`` is the only
vertical stretch in effect there. The design specifies composition instead
(the design's own wording: "vexag composes as an additional stretch"), so it is implemented
that way here, documented as an intentional choice rather than a port of EQSelect's UI, not an
oversight.

**Missing-scale gaps -- an honest split, never a fake link.**
:meth:`_split_scale_runs` breaks one chain's own point range into separate line cells wherever
its own scale STEP exceeds 1.5x the layer's "one voice-step" reference
(:meth:`_expected_scale_step` -- ``1/n_voice`` from the result's own resolved params when
available, else the MEDIAN point-to-point step pooled across this layer's own chains). Every
real ``wtmm_backend`` chain today is gap-free by construction (the same finest-anchored-prefix
guarantee above), so this path is exercised only by synthetic/hand-built chains in the test
suite -- it exists for the honest-gap contract regardless, since nothing here assumes
``wtmm_backend`` is the only chain source forever.

**Per-scale discrete colormap (``color_by_scale``, default ``True`` in scale-space mode).**
:meth:`_scale_space_colors` replaces :func:`_chain_color`'s one-color-per-CHAIN convention with
one color per POINT, ranked fine->coarse through a discrete ``pv.LookupTable`` sampled to
exactly as many entries as there are distinct scale values actually drawn in this layer right
now (mirrors ``n_colors = n_scales``) -- the same colormap-name -> RGBA
sampling idiom :meth:`set_colormap` already uses elsewhere in this module, so no new
third-party import is introduced. No separate UI toggle exists for this in v1 (the View
dialog's own checkbox/slider are the only two new controls this task adds); the keyword exists
on :meth:`set_scale_space` for internal testability and a documented, harmless escape hatch.

**Per-view camera memory folds in the scale-space flag.** :meth:`camera_key` appends
``"|sspace:on"`` when scale-space is enabled, and appends NOTHING (the exact pre-Task-6 string)
when it is not -- so every existing exact-string camera-key assertion (``"geo:mercator"``, etc.)
stays byte-identical with the checkbox off, while toggling it on earns its own remembered
framing: :meth:`set_scale_space` follows the identical remember-before/restore-or-reset-after
contract :meth:`set_frame_mode` already uses, because a toggled-on cone's z-extent has nothing
to do with the flat view's own camera fit -- reusing that fit would frame the wrong thing, and
a first-ever visit to ``"...|sspace:on"`` re-fits fresh exactly like any other first visit.

**Picking (the machinery) needs no changes at all.** :meth:`pick`/:meth:`pick_in_region`
read live actor points off ``mapper.dataset.points`` -- when those points are lifted, picking
already operates on the lifted geometry automatically; nothing about the pick math above knows
or needs to know that z is no longer always 0.

**Palette constants: redeclared here, not imported from ``dynamix.shell.canvas``.** The values
below match ``canvas.py``'s ``VTRAIL_COLOR``/``SEAM_COLOR``/``EXTREMA_COLOR`` exactly (session
palette parity, the design's "session palette carried"), but ``canvas.py`` imports ``pyqtgraph`` at
module scope, and ``pyvista`` (this module's own import, the ``viz`` optional-dependency group)
and ``pyqtgraph``/``PySide6`` (the SEPARATE ``gui`` group, ``pyproject.toml``) are deliberately
decoupled -- ``Scene`` is Qt-free in behaviour precisely so it can be exercised with just ``viz``
installed. Importing ``canvas`` here would silently make every arrangement/``Scene`` test (and any
future ``viz``-only install) require ``pyqtgraph`` too. ``tests/test_arrangement_scene.py`` pins
the parity between the two modules' constants directly (guarded by its own
``pytest.importorskip("pyqtgraph")``, so it skips honestly rather than forcing the coupling this
module itself avoids).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pyvista as pv

from dynamix.core.frames import GeographicFrame
from dynamix.core.projection import KM_PER_DEG, MODES, project
from dynamix.core.selection import (nearest_point, points_in_box, points_in_polygon,
                                     project_points)
from dynamix.core.chain_product import layer_point_keep, selection_of, stub_capped_chains
from dynamix.core.hlines import hline_runs
from dynamix.model.project import REFERENCE_COLORS
from dynamix.geo.footprints import band_sort_key as _band_sort_key
from dynamix.geo.footprints import group_key as _footprint_group_key
from dynamix.geo.mapping import (NoGeoreference, _mask_sentinels, _stride_for, axis_at,
                                 field_lonlat_grid, points_lonlat)

#: The scene's own default -- deliberately NOT ``dynamix.core.projection.DEFAULT_MODE`` (which is
#: "pacific", chosen there for a different reason: the flat equirect split-arc fix out of the box).
#: The design, "Landing mode": "Web Mercator (regional data); 4-mode switch ported."
DEFAULT_MODE = "mercator"

#: Vertical lift for VECTOR actors (chains/extrema/ROI/point layers) above the raster drape they
#: sit on. Coplanar geometry at the drape's exact z loses the depth test to the opaque surface
#: nearly everywhere under a top-down parallel view -- the Vector tab's default framing -- so
#: chains vanish except where they overhang the drape's half-pixel border (only points on the
#: border show). The loss is angle-dependent; geometric separation fixes it, where VTK's
#: shader-side coincident-topology offsets showed no effect in this pipeline. Geo modes lift via the
#: existing ``height`` plumbing in KILOMETRES (the globe turns it into a radial lift for free;
#: flat modes into a small z) -- 0.5 km is invisible at map scale and vexag-scaling it stays
#: invisible top-down. Frame mode lifts by a fixed fraction of the layer's own extent, so the
#: separation survives any data units (degrees, metres, survey feet, px).
_VECTOR_LIFT_KM = 0.5
_VECTOR_LIFT_FRAC = 1e-3

#: Name of the corner legend's text actor -- excluded from :meth:`Scene.actor_count`, and never
#: collides with a layer's own mesh-actor name (see :meth:`Scene._mesh_actor_name`).
_LEGEND_NAME = "__arrangement_legend__"
_PREVIEW_PREFIX = "__preview__"           # one actor per previewed scene (data browser)
_REFERENCE_PREFIX = "__ref__"             # one actor per reference layer
_FOOTPRINT_NAME = "__footprints__"      # the data browser's footprint loops, ONE actor
FOOTPRINT_COLOR = (255, 230, 120)       # pale amber: outlines of rasters NOT loaded -- distinct from
                                        # ROI_BOUNDS_COLOR's deeper amber (a box on loaded data)

#: Data-layer palette -- what a chain/extremum/ROI-box IS, not chrome (see the module docstring's
#: "redeclared here, not imported" note for why these are a separate copy of
#: ``dynamix.shell.canvas``'s own constants of the same name, rather than an import).
EXTREMA_COLOR = (200, 200, 200)     # isolated extrema: light gray -- canvas.py's EXTREMA_COLOR
HLINE_COLOR = (235, 235, 235)       # H-line polylines: white-ish -- canvas.py's HCHAIN_COLOR
VTRAIL_COLOR = (255, 160, 40)       # plain V-chain drift trails: amber -- canvas.py's VTRAIL_COLOR
SEAM_COLOR = (255, 112, 67)         # seam-flagged V-chain trails -- canvas.py's SEAM_COLOR
ROI_BOUNDS_COLOR = (255, 160, 40)   # ROI outline -- canvas.py's ROI_BOUNDS_COLOR (own constant,
                                     # not a reuse of VTRAIL_COLOR, for the same independent-re-
                                     # tint reasoning canvas.py's own comment gives)

#: The pick highlight -- bright white, deliberately distinct from every OTHER color a chain
#: can wear: not VTRAIL_COLOR/SEAM_COLOR/EXTREMA_COLOR/ROI_BOUNDS_COLOR above, and not any entry
#: in ``group_palette.GROUP_COLORS`` either (the Okabe & Ito colorblind-safe palette has no white
#: member). Wins over a chain's own group-preview color wherever both apply -- see
#: :meth:`Scene._apply_mask_to_layer`'s compositing order, documented in the module docstring.
SELECTION_COLOR = (255, 255, 255)

#: Point-layer drape default -- matches ``dynamix.shell.canvas``'s own
#: ``POINTS_COLOR`` exactly (same "redeclared here, not imported" reasoning the module docstring's
#: "Palette constants" section already gives for EXTREMA_COLOR/VTRAIL_COLOR/SEAM_COLOR/
#: ROI_BOUNDS_COLOR above: this module stays importable with just the ``viz`` optional-dependency
#: group, never pulling in ``pyqtgraph``). An entry without its own ``points_color`` (every
#: pre-Task-12 call, and any point layer whose layer carries no ``ui.color_points`` preference)
#: falls back to this constant, the same "None means the module default" contract
#: ``vtrail_color``/``colormap`` already use.
POINTS_COLOR = (0x4F, 0xC3, 0xF7)

#: the design's pick performance bar ("Pick response: <50 ms") is about LATENCY, not this radius --
#: this is the click-tolerance radius itself (``nearest_point(pts2d, xy, max_dist=8.0)``), in screen pixels.
_PICK_MAX_DIST_PX = 8.0

#: The graticule's own actor name -- excluded from :meth:`Scene.
#: actor_count` (the legend text is excluded the same way), never collides with a layer's own
#: mesh-actor name (see :meth:`Scene._mesh_actor_name`).
_GRATICULE_NAME = "__arrangement_graticule__"

#: A muted neutral, deliberately distinct from every DATA color above (VTRAIL/SEAM/EXTREMA/
#: ROI_BOUNDS/SELECTION) -- the grid is chrome, not something a chain could ever be mistaken for.
#: This module sits outside the Theme Rule's own reach (pyvista territory, not Qt/QSS --
#: ``tests/test_shell_boundaries.py``'s literal-color scan only covers Qt modules), the same carve-
#: out the module docstring's "redeclared here, not imported" section already claims for the data
#: palette above.
_GRATICULE_COLOR = (70, 70, 78)

_GRATICULE_STEP_DEG = 10.0        # the design, Frame tab: "graticule toggle" -- a 10-degree grid
_GRATICULE_LAT_LIMIT = 80.0       # meridians stop short of the true poles -- see _graticule_geometry

#: The discrete fine->coarse colormap
#: :meth:`Scene._scale_space_colors` samples -- an arbitrary but intentional choice (EQSelect
#: names no specific colormap for its own analog), reusing ``viridis`` since it
#: is already this module's own default scene-wide raster colormap (:attr:`Scene._colormap`'s
#: own initial value), so a fresh Scene's chains and rasters start from one consistent palette
#: family rather than two unrelated ones.
_SCALE_SPACE_CMAP = "viridis"

#: A scale STEP strictly greater than this multiple of the layer's own "one voice-step"
#: reference (:meth:`Scene._expected_scale_step`) is a missing-scale GAP, not floating-point
#: noise around a clean, single-step link. Comfortably between 1.0 (an exact, ungapped step)
#: and 2.0 (a single skipped scale) so it never mistakes one for the other.
_SCALE_SPACE_GAP_FACTOR = 1.5

#: The six keys :meth:`Scene._camera_snapshot` writes and
#: :meth:`Scene.restore_camera` reads back by name. An imported entry missing any one of them
#: could only fail later, inside a restore, so :meth:`Scene.import_camera_memory` uses exactly
#: this tuple as its whole validity test and drops anything short of it.
_CAMERA_FIELDS = ("position", "focal_point", "up", "parallel_scale", "parallel_projection",
                  "clipping_range")



def _frame_z_scale(frame) -> float:
    """Frame-mode z per the DATASET's own metadata: a GEOGRAPHIC frame lays x/y out in DEGREES
    while surface values are metres (the geo branch's assumed-metres convention), so raw-z
    meshes come out ~7000x taller than wide -- a mountain DEM window at 0.38 degrees
    footprint standing 2.7 "units" tall. Metres convert to degree-equivalents
    (m -> km -> / KM_PER_DEG), the exact convention project()'s flat geo modes already
    apply, so frame and geo modes finally agree on what a metre of relief looks like.
    Projected/local frames keep 1.0: their axes are in the values' own linear units (a
    UTM DEM in metres over metres is already coherent)."""
    return 1.0 / (1000.0 * KM_PER_DEG) if isinstance(frame, GeographicFrame) else 1.0

def _graticule_geometry(mode: str, vexag: float):
    """``(points, lines)`` for a 10-degree lat/lon grid under ``mode`` -- one polyline cell per
    meridian (a fixed longitude, walking latitude from -80..80) and per parallel (a fixed
    latitude, walking longitude across the full -180..180 range), each projected through
    :func:`~dynamix.core.projection.project` exactly like every other actor this module builds.
    One polyline per line, not one continuous NaN-split path -- a graticule has no shared-vertex
    continuity to preserve the way a single chain's own points do, so this mirrors
    :meth:`Scene._build_chain_geometry`'s own ``lines`` cell-array construction directly (see that
    method's docstring for the exact shape).

    Meridians stop at +/-:data:`_GRATICULE_LAT_LIMIT` rather than the true poles: every flat mode
    maps a pole to a single point already (a meridian ending there is degenerate), and ``globe``
    would draw every meridian converging on the identical two points -- visually redundant, not
    wrong, but an 80-degree cutoff reads as a grid, not a cage.
    """
    segments = []
    for lon_deg in np.arange(-180.0, 180.0, _GRATICULE_STEP_DEG):
        lat = np.linspace(-_GRATICULE_LAT_LIMIT, _GRATICULE_LAT_LIMIT, 17)
        segments.append((np.full_like(lat, lon_deg), lat))
    lat_count = int(round(2 * _GRATICULE_LAT_LIMIT / _GRATICULE_STEP_DEG)) + 1
    for lat_deg in np.linspace(-_GRATICULE_LAT_LIMIT, _GRATICULE_LAT_LIMIT, lat_count):
        lon = np.linspace(-180.0, 180.0, 37)
        segments.append((lon, np.full_like(lon, lat_deg)))

    points_parts = []
    lines_cells: list[int] = []
    offset = 0
    for lon_arr, lat_arr in segments:
        pts = project(lon_arr, lat_arr, np.zeros_like(lon_arr), mode=mode, vexag=vexag)
        k = pts.shape[0]
        points_parts.append(pts)
        lines_cells.append(k)
        lines_cells.extend(range(offset, offset + k))
        offset += k
    points = np.concatenate(points_parts, axis=0)
    return points, np.array(lines_cells, dtype=np.int64)


def _chain_color(chain: dict, default_vtrail=None) -> tuple[int, int, int]:
    """One chain's UNMASKED display color -- priority order per the design ("session palette
    carried: seam flags stay seam-colored; committed groups their group colors") and the contract: a committed group's OWN color wins first (``chain["group_color"]``, an ``[r, g, b]``
    list -- ``dynamix.devices.groups.GroupPaint`` stamps this onto a chain's tags/`` alongside its
    ``"group:<name>"`` tag), else :data:`SEAM_COLOR` for any other truthy ``tags`` (a
    ``chain_classify`` seam flag -- a group-tagged chain also has truthy ``tags``, which is why the
    group-color check must come first), else the plain trail color.

    ``default_vtrail`` is the entry's own ``ui.color_vtrail`` preference
    (an ``(r, g, b)`` tuple, converted from hex at the window boundary -- see
    ``main_window._sync_arrangement``'s own docstring), used for that last "plain" case instead of
    the module's own :data:`VTRAIL_COLOR` constant. ``None`` (every call site that predates this
    task, and every entry whose layer has no such preference recorded) falls back to
    :data:`VTRAIL_COLOR` exactly as before -- group and seam colors are UNCHANGED by this
    parameter; only which color a plain, untagged chain gets moves.
    """
    group_color = chain.get("group_color")
    if group_color:
        try:
            r, g, b = (int(c) for c in group_color[:3])
            return (r, g, b)
        except (TypeError, ValueError):
            pass    # malformed color: fall through to the tag-based choice rather than crash
    if chain.get("tags"):
        return SEAM_COLOR
    if default_vtrail:
        return tuple(int(c) for c in default_vtrail[:3])
    return VTRAIL_COLOR


def _chain_root(chain: dict) -> tuple[int, int, int]:
    """One chain's ``(root x, root y, length)`` -- the per-chain unit :meth:`Scene._fingerprint`
hashes into a positional digest. ``(0, 0, 0)`` for an empty chain
    (nothing to root on) rather than raising on ``x[0]`` -- a length-0 chain is a legitimate,
    already-handled case elsewhere in this module (:func:`_chain_color`'s own callers skip them
    for drawing; a digest still needs SOME value for it, and a fixed sentinel is as good as any
    since a length-0 entry contributes nothing else to the digest either)."""
    x, y = chain.get("x"), chain.get("y")
    n = len(x) if x is not None else 0
    if n == 0:
        return (0, 0, 0)
    return (int(x[0]), int(y[0]), n)


@dataclasses.dataclass(frozen=True)
class _VectorMaskState:
    """One layer's chains-actor bookkeeping that ONLY :meth:`Scene.set_mask`/
    :meth:`Scene._apply_mask_to_layer` need -- deliberately separate from ``self._chain_lookup``
    (the ``(starts, chain_indices)`` pair the picking consumes), so that seam stays exactly
    the shape documented for it without also carrying this task's own recoloring internals.

    ``max_log2_mod``/``hi_scale_idx`` are indexed by a chain's ORIGINAL position in
    ``result["chains"]`` -- the same indexing ``chain_indices`` (in ``self._chain_lookup``) uses,
    so ``chain_visible[chain_indices]`` (built from these two arrays) is a valid per-point mask
    with no remapping step.
    """

    chains_actor_name: str
    max_log2_mod: np.ndarray    # float64 (n_chains,) -- see set_mask's docstring
    hi_scale_idx: np.ndarray    # int64 (n_chains,) -- see set_mask's docstring
    base_rgba: np.ndarray       # uint8 (n_points, 4), alpha always 255 -- the UNMASKED colors



def _subpixel_of(layer: dict, idx) -> "tuple | None":
    """``(x_sub, y_sub)`` of extrema layer ``layer`` at indices ``idx`` -- the subpixel
    refinement a detector stamped (follow always; nms with Interpolate on) -- with any
    non-finite entry falling back to its integer pixel; ``None`` when the layer has none."""
    if not isinstance(layer, dict) or "x_sub" not in layer or "y_sub" not in layer:
        return None
    idx = np.asarray(idx)
    xs = np.asarray(layer["x_sub"], dtype=np.float64)[idx]
    ys = np.asarray(layer["y_sub"], dtype=np.float64)[idx]
    xi = np.asarray(layer["x"], dtype=np.float64)[idx]
    yi = np.asarray(layer["y"], dtype=np.float64)[idx]
    return (np.where(np.isfinite(xs), xs, xi), np.where(np.isfinite(ys), ys, yi))

class Scene:
    """Owns every actor drawn in the arrangement view's ``plotter`` and the current projection
    mode. One raster mesh actor per "ok" layer, plus up to three vector actors; every
    other status is a legend line, never geometry. ``set_mode`` is the one legitimate full rebuild.
"""

    def __init__(self, plotter, background: str | None = None) -> None:
        self._plotter = plotter
        # The one place this scene ever touches the plotter's
        # background -- see the module docstring's own "Themed background" section. A plain
        # color-string argument, not a theme import: this class stays Qt-free.
        if background is not None:
            self._plotter.set_background(background)
        self._mode = DEFAULT_MODE
        #: The Vector tab's placement switch -- see the module
        #: docstring's "Frame mode" section. ``False`` (this default) is byte-for-byte today's
        #: geo behavior; every existing test/caller that predates this task never touches it.
        self._frame_mode = False
        self._colormap = "viridis"
        #: (modulus_pctl, scale_lo, scale_hi) -- see set_mask's docstring. The starting value
        #: excludes nothing: scale_lo=0 never excludes on depth (every chain's deepest index is
        #: >= 0), scale_hi=0 is the "no cap" sentinel (HLineLength's own convention), and
        #: modulus_pctl=0 keeps everything (the 0th percentile is each layer's own minimum, and
        #: every chain is >= its own layer's minimum by construction).
        self._mask: tuple[float, int, int] = (0.0, 0, 0)
        self._entries: list[dict] = []
        #: layer_id -> actor names owned by that layer (raster + the vector actors, in the
        #: order they were added).
        self._layer_actors: dict[int, list[str]] = {}
        #: layer_id -> (starts, chain_indices) -- see the module docstring's "Vector bookkeeping"
        #: section. Present only for layers with at least one drawable (>= 2 point) chain.
        self._chain_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        #: layer_id -> _VectorMaskState -- see that class's docstring. Same presence rule as
        #: self._chain_lookup (built together, in the same method, always in sync).
        self._vector_masks: dict[int, _VectorMaskState] = {}
        #: layer_id -> {chain_index: (r, g, b)} -- the pre-commit group-color preview, from
        #: GroupPalette's own live state (never a committed layer.tags group). Survives a rebuild
        #: exactly like self._mask -- see _rebuild()'s own re-application. Empty until
        #: set_group_preview() is ever called.
        self._group_preview: dict[int, dict[int, tuple[int, int, int]]] = {}
        #: layer_id -> {chain_index, ...} -- the "just picked" highlight (SELECTION_COLOR).
        #: Same survives-a-rebuild treatment as self._group_preview/self._mask.
        self._selection: dict[int, set[int]] = {}
        #: layer_id -> id(result) (or ``None`` for a result-less entry) as of the last
        #: ``set_layers`` call -- the staleness guard, see :meth:`set_layers`'s own
        #: docstring. Deliberately an IDENTITY map (``id()``), not a value/content comparison: two
        #: different result dicts could coincidentally compare equal in content, but they are two
        #: DIFFERENT chain lists, and ``(layer_id, chain_index)`` only means anything against the
        #: exact list it was picked from.
        self._result_identity: dict[int, object] = {}
        #: layer_id -> ``(n_chains, positional_digest, signature)`` as of the last ``set_layers``
        #: call, or ``None`` for a result-less entry (revised round
        #: 2). ``_result_identity`` alone over-fires: ``dynamix/engine/resolve.py``'s own module
        #: docstring -- every Filter always copies -- means EVERY resolve() call on a chain with
        #: at least one filter (which is nearly every real chain in this app) mints a brand-new
        #: ``result`` object even when NOTHING about the actual computation changed, so a plain
        #: repeated ``MainWindow._sync_arrangement()`` pass (two landings in a row, or the multi-layer commit needing one resync per affected layer) used to read as staleness and
        #: silently wipe live selection/group-preview/commit-membership state for no reason.
        #:
        #: Round 1 fingerprinted on ``(chain_count, signature)`` alone -- caught a genuine
        #: TRANSFORM re-run, but is structurally blind to a count-preserving FILTER change: a
        #: filter's own params never enter the transform signature by design, so a re-selection
        #: upstream (e.g. a cutoff filter admitting a DIFFERENT, equal-sized set of chains) left
        #: the old fingerprint unchanged -- a stale ``(layer_id, chain_index)`` selection/preview/
        #: commit-membership entry would then silently point at the WRONG chain, a real bug the
        #: pre-fix bare-identity check happened to catch only by ACCIDENTALLY over-firing (the original bug). Fixed by replacing the count with a POSITIONAL DIGEST
        #: (:func:`_chain_root` per chain, hashed) -- see :meth:`_fingerprint`'s own docstring for
        #: why it is order-sensitive, reselection-sensitive AND copy-immune (the property the commit resync still depends on).
        self._result_fingerprint: dict[int, object] = {}
        #: layer_id -> (id(field), lon2d, lat2d, values2d, stride) -- ``field_lonlat_grid``'s own CRS transform (``rasterio.warp.transform`` over up to
        #: ~2M pixel centers) is MODE-INVARIANT -- ``project()``, the one thing that actually
        #: depends on ``self._mode``, is a cheap ``numpy`` reprojection of the lon/lat this grid
        #: already carries (see :meth:`_add_raster_actor`). Profiling a ``set_mode`` rebuild at the
        #: spec's own benchmark scale (2 fields at the 2M-point stride cap) found ~86% of the total
        #: time inside ``field_lonlat_grid`` -- i.e. RECOMPUTING the identical lon/lat grid on
        #: every single mode switch, for a field that has not changed at all. Cached here, keyed by
        #: each layer's OWN ``id(field)`` -- the same identity-not-content discipline
        #: ``self._result_identity`` already uses, for the same reason (two field objects could
        #: coincidentally compare equal without being the SAME field). Sibling of ``canvas.py``'s
        #: ``_cached_geometry`` (the trail-geometry scrub-cache precedent this plan's own commit
        #: transaction fix report already cites) and this module's OWN mask arrays
        #: (``_VectorMaskState``) -- same idiom, same reason: recomputing a pure function of
        #: something that has not changed is waste, not correctness. Invalidated per-layer in
        #: :meth:`set_layers` -- see its own docstring -- whenever that layer's field object
        #: changes identity, or the layer_id itself disappears from the entry set.
        self._lonlat_cache: dict[int, tuple] = {}
        self._legend_lines: list[str] = []
        self._legend_present = False
        self._footprints: list = []          # dynamix.geo.footprints.Footprint records
        self._previews: list = []            # (name, RasterField) previews draped on the world
        self._surface_fields: dict = {}      # id(field) -> negate, for layers shown as a 3-D surface
        # Live drape preview: layer_id -> {"grid", "stride", "dims", "rgba"} --
        # the handles preview_raster_values needs to swap a drape's scalars IN PLACE (a
        # histogram drag tick) without the full set_layers rebuild. Cleared with the actors.
        self._raster_grids: dict = {}
        self._hline_base_cache: dict = {}    # layer_id -> (key, pts, vert_pos, run_id); _hline_actors_from_base
        self._reference_layers: list = []    # reference-layer entries (see set_reference_layers)
        #: The Display tab's vertical-exaggeration factor, threaded
        #: into every ``project(..., vexag=...)`` call site this module owns (raster, chains,
        #: extrema, ROI outline, graticule). Every draped height is currently always 0 (no
        #: elevation dataset exists yet -- see :meth:`_add_raster_actor`), so this has no visible
        #: effect today; it exists so a future elevation source needs no further plumbing here.
        self._vexag: float = 1.0
        #: The Frame tab's Graticule checkbox state, and whether an actor for it is
        #: CURRENTLY on the plotter -- same "remembered, silently reapplied after a rebuild"
        #: contract as ``self._mask``/``self._selection`` (see :meth:`_rebuild`'s own tail).
        self._graticule: bool = False
        self._graticule_present = False
        #: ``camera_key() -> snapshot`` -- see the
        #: module docstring's "Per-view camera memory" section and :meth:`remember_camera`'s own
        #: docstring for the snapshot's exact shape. Starts empty: a brand-new ``Scene`` has never
        #: shown any view before, so every key's first visit falls through to a fresh
        #: ``reset_camera()`` fit, same as it always has.
        self._camera_memory: dict[str, dict] = {}
        #: The View dialog's "3-D scale-space (stack by
        #: log₂ a)" checkbox + stretch slider -- see the module docstring's "Scale-space chain
        #: stacking" section for the full contract. ``False``/``0.0``/``True`` are byte-identical
        #: to every pre-Task-6 test: :meth:`_build_chain_geometry` only computes a z-lift at all
        #: when ``self._scale_space_enabled`` is True.
        self._scale_space_enabled: bool = False
        self._scale_space_stretch: float = 0.0
        self._color_by_scale: bool = True

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def legend_lines(self) -> list[str]:
        """Read-only snapshot of the current legend rows (one per non-"ok" entry)."""
        return list(self._legend_lines)

    # -- public interface ----------------------------------------------------------------------

    def set_layers(self, entries) -> list[int]:
        """Replace the whole layer set and rebuild every actor + the legend from scratch.

        **The staleness guard.** A background recompute
        landing while the arrangement is parked (or simply a fresh ``_sync_arrangement()`` pass) can hand a layer a BRAND NEW ``result`` -- a new chains list, under which any
        ``(layer_id, chain_index)`` this scene is holding in ``self._selection``/
        ``self._group_preview`` (or ``GroupPalette`` is holding as committed-so-far membership) may
        now name the WRONG chain, or one that no longer exists. Detected here by comparing each
        entry's ``id(result)`` against what the LAST ``set_layers`` call saw for that same
        ``layer_id`` (an identity check, not a content one -- see ``self._result_identity``'s own
        docstring for why): on a change (including the layer disappearing from ``entries``
        entirely), this scene's own ``self._selection``/``self._group_preview`` entries for that
        ``layer_id`` are dropped. A transition from "no result yet" to a first real result is NOT
        flagged -- ``pick()`` can only ever have returned a hit for a layer already carrying a real
        result (:attr:`_chain_lookup` only exists for those), so nothing could have been selected
        or grouped for a layer that never had one yet; flagging that routine "computing -> ok"
        transition would be noise, not a real staleness event.

        **Identity change alone is no longer
        sufficient.** Every real chain has at least one Filter, and a Filter always returns a
        fresh ``dict`` (never the same object back -- ``dynamix/engine/resolve.py``'s own module
        docstring), so TWO ``resolve()`` calls for a layer whose computation has not changed AT
        ALL still produce two DIFFERENT ``id(result)``s. A caller that resyncs more than once for
        the same reason (the multi-layer commit needs one resync per affected layer; two
        landings in a row do too) used to have every OTHER layer's exploration state wiped by its
        own re-sync, for nothing. An identity change now only prunes when the entry's fingerprint
        (:meth:`_fingerprint` -- a chain COUNT plus a per-chain POSITIONAL DIGEST, plus the
        transform signature) ALSO moved -- i.e. the transform genuinely re-ran, or the filtered
        chain SET genuinely changed shape or membership; a content-equal re-filter (same chains,
        merely a fresh dict) leaves selection/group-preview/commit membership untouched. Round 1's
        fingerprint used chain COUNT alone alongside the signature, which is blind to a
        count-preserving FILTER re-selection (a filter's own params are never part of the
        transform signature) -- see :meth:`_fingerprint`'s own docstring for the fix.

        Returns the sorted list of pruned ``layer_id``s (empty when nothing changed) -- this
        Scene's own state is already updated by the time this returns; the caller
        (``ArrangementView``) is responsible for telling ``GroupPalette`` to prune the SAME
        ``layer_id``s from its own, separately-held committed-group membership
        (:meth:`~dynamix.shell.arrangement.group_palette.GroupPalette.prune_layer`) -- this method
        has no reference to any palette and cannot do that itself.

        Exploration state lost this way is honest and acceptable (exploration is
        view-state, not a durable commitment) -- the transform-signature guard covers
        commit-time staleness independently; this is only about not silently poisoning the LIVE
        pre-commit preview between now and a commit.

        **``self._lonlat_cache`` pruning.** A layer_id absent
        from ``entries`` (removed, or this ``Scene`` reused for a different layer set -- the same
        condition :meth:`_detect_and_prune_stale_layers` already checks for the identity/
        fingerprint maps) has its cached lon/lat grid dropped too, so a ``Scene`` that outlives
        many different layer sets over its lifetime does not accumulate one entry per layer_id
        ever seen. A layer whose FIELD OBJECT changed (still present, same layer_id, different
        ``id(field)`` -- e.g. a reopened source) is NOT pruned here: :meth:`_lonlat_grid_for`
        itself detects that on its next call (the cached identity no longer matches) and recomputes
        exactly that one layer's grid -- pruning it here too would just mean recomputing it one
        call sooner, at the cost of a second place this invalidation rule has to be kept correct.
        """
        entries = list(entries)     # materialized once: a one-shot iterable must not be consumed
                                     # by the detection pass below and left empty for self._entries.
        pruned = self._detect_and_prune_stale_layers(entries)
        seen_layer_ids = {entry["layer"].layer_id for entry in entries}
        for layer_id in set(self._lonlat_cache) - seen_layer_ids:
            del self._lonlat_cache[layer_id]
            self._hline_base_cache.pop(layer_id, None)
        was_empty = not self._entries
        # Capture the OUTGOING frame-mode camera key
        # BEFORE ``self._entries`` is replaced below -- see the "Frame-mode active-layer swap"
        # comment a few lines down for what this is for. ``None`` whenever this call cannot
        # possibly be a same-mode active-layer swap: geo mode (there is no single "active layer"
        # -- every admitted layer shares one camera/coordinate system, so nothing here could ever
        # mean "the geometry changed under the camera") or the empty -> non-empty transition the
        # pre-existing ``was_empty`` branch below already re-frames unconditionally.
        old_frame_key = self.camera_key() if (self._frame_mode and not was_empty) else None
        old_ids = {e["layer"].layer_id for e in self._entries}
        if self._filter_only_resync(entries):
            # Filter-only change: same layers, fields, signatures and raster styles -- only
            # filtered results moved. The vector actors of the changed layers are rebuilt in
            # place; the raster drapes, reference layers, previews, legend and CAMERA are all
            # left exactly as they are.
            self._entries = entries
            return pruned
        self._entries = entries
        self._rebuild()
        self._apply_raster_visibility()
        # **First-flip framing.** Every actor above was added with ``reset_camera=False``
        # (deliberate: a relayering must never yank a navigated camera), and the live first-Tab
        # order is camera-reset-on-EMPTY-scene first, entries after
        # (``ArrangementView.activate``'s build tail runs before ``_sync_arrangement`` delivers
        # anything) -- so the camera sits framed on VTK's default unit bounds at the origin
        # while the first drape lands hundreds of projected degrees away: a themed, blank
        # view. The empty -> non-empty TRANSITION is the one moment a re-frame can never fight
        # the user (there was nothing to navigate on), so it is the one moment this method
        # re-frames: ``reset_camera()`` re-fits bounds only, preserving the current projection
        # and orientation (top-down stays top-down; globe stays globe).
        if was_empty and entries:
            self._fit_to_data()
        elif old_frame_key is not None:
            # **Frame-mode active-layer swap.** A
            # ``set_layers`` call while ALREADY parked in frame mode can swap which geometry is on
            # screen with no ``set_frame_mode`` flip at all (e.g. selecting a different layer on
            # the Vector tab). ``camera_key()`` is bounds-scoped in frame mode (module docstring's
            # "Per-view camera memory" section), so comparing it before/after this call swapped
            # ``self._entries`` in is exactly "did the on-screen geometry's own extent just
            # change". A same-layer RESYNC (a fresh ``result`` dict for the SAME field --
            # ``dynamix/engine/resolve.py``'s own module docstring: every ``Filter`` always
            # copies) keeps the identical signature, so this branch does nothing at all -- the
            # camera stays exactly where the user left it (navigation is sacred, the same doctrine
            # the comment above already states for a plain relayering). A genuine swap (a
            # different active layer, a disjoint frame/shape) remembers the outgoing view under
            # its own key, then restores whatever the incoming key last saw, or re-fits fresh on a
            # first visit -- the identical restore-or-fit contract :meth:`set_frame_mode` uses for
            # its own flip.
            new_frame_key = self.camera_key()
            if new_frame_key != old_frame_key:
                self._camera_memory[old_frame_key] = self._camera_snapshot()
                if not self.restore_camera(new_frame_key):
                    self._fit_to_data()
        elif not self._frame_mode and old_ids and entries \
                and old_ids.isdisjoint(e["layer"].layer_id for e in entries):
            # **Geo-mode dataset swap.** The geo camera is mode-scoped (one world, one camera),
            # which is right while layers ADD -- but when every layer on screen is replaced by
            # others, the new data can sit on the far side of the globe from a camera fitted on
            # the old (ASTER in the Pilbara -> BOEM in the Gulf leaves the new raster out of
            # view). Only the disjoint case re-frames; adding or resyncing never does.
            self._fit_to_data()
        return pruned

    @staticmethod
    def _fingerprint(entry: dict, result) -> object:
        """``(n_chains, positional_digest, signature)`` for a real result, ``None`` for a
        result-less entry -- see ``self._result_fingerprint``'s own docstring for the fix this implements.

        ``positional_digest`` is ``hash(tuple(_chain_root(c) for c in chains))`` -- ``_chain_root``
        being each chain's own ``(root x, root y, length)``. Three properties, all load-bearing:

        - **Order-sensitive.** A REORDER of the same chain set changes which chain's root lands at
          which position in the hashed tuple, so the digest changes -- a plain chain-count
          fingerprint (round 1's own) cannot see a reorder at all.
        - **Reselection-sensitive.** A count-PRESERVING filter change upstream (e.g. a cutoff
          admitting a DIFFERENT, equal-sized set of chains) is exactly what round 1's fingerprint
          missed: a filter's own params never enter the transform signature by design, so the old
          ``(count, signature)`` pair stayed identical while every ``(layer_id, chain_index)`` this
          scene/palette held now silently named a DIFFERENT chain. A different surviving chain
          almost certainly has a different root position, so the digest moves.
        - **Copy-immune.** ``dynamix.devices.groups.GroupPaint`` stamps tags onto COPIES of member
          chains but never touches ``x``/``y``/their length -- so a commit's own resync (the commit resync, which this predicate must not re-break) produces the IDENTICAL digest for
          the identical chain set, and still does not prune.

        ``entry.get("signature")`` rides alongside as a second, already-threaded, essentially-free
        signal (kept rather than dropped: the digest alone is sufficient, but a coincidental digest
        collision across a genuine transform re-run is one more thing this catches for free).

        Cost: one ``(int, int, len)`` tuple per chain, per ``set_layers`` call -- microseconds even
        at the ~20k-chain scale this app's own perf bar is benchmarked against,
        nowhere near its 16ms budget.
        """
        if result is None:
            return None
        chains = result.get("chains") or ()
        digest = hash(tuple(_chain_root(c) for c in chains))
        return (len(chains), digest, entry.get("signature"))

    def _detect_and_prune_stale_layers(self, entries) -> list[int]:
        """The comparison :meth:`set_layers` describes -- split out as its own method purely for
        readability; see that method's own docstring for the full contract. Mutates
        ``self._result_identity``/``self._result_fingerprint``/``self._selection``/
        ``self._group_preview`` directly."""
        pruned: set[int] = set()
        seen_layer_ids: set[int] = set()
        for entry in entries:
            layer_id = entry["layer"].layer_id
            seen_layer_ids.add(layer_id)
            result = entry.get("result")
            new_identity = id(result) if result is not None else None
            new_fingerprint = self._fingerprint(entry, result)
            prior = self._result_identity.get(layer_id, "unset")
            if prior not in ("unset", None) and prior != new_identity:
                # Identity moved -- real staleness only if the CONTENT moved too:
                # `new_fingerprint is None` covers a transition INTO a result-less status (error/
                # no-georeference/computing), always a real change; otherwise compare against
                # what the LAST call recorded for this layer_id.
                if new_fingerprint is None or self._result_fingerprint.get(layer_id) != new_fingerprint:
                    pruned.add(layer_id)
            self._result_identity[layer_id] = new_identity
            self._result_fingerprint[layer_id] = new_fingerprint
        # A layer_id that was tracked before but is simply absent from `entries` now (removed, or
        # this Scene reused for a different layer set) -- anything held for it is equally stale.
        for layer_id in set(self._result_identity) - seen_layer_ids:
            del self._result_identity[layer_id]
            self._result_fingerprint.pop(layer_id, None)
            pruned.add(layer_id)
        for layer_id in pruned:
            self._selection.pop(layer_id, None)
            self._group_preview.pop(layer_id, None)
        return sorted(pruned)

    def set_mode(self, mode: str) -> None:
        """Switch the projection mode and rebuild every raster AND vector actor under it -- the
        one legitimate full rebuild. Validated before anything is touched, so an
        unknown mode leaves the scene exactly as it was rather than half-cleared. The active mask
        (:meth:`set_mask`) and colormap (:meth:`set_colormap`) both survive the rebuild -- see
        :meth:`_rebuild`."""
        if mode not in MODES:
            raise ValueError(f"unknown projection mode {mode!r}; choose from {MODES}")
        self._mode = mode
        self._rebuild()

    def set_frame_mode(self, enabled: bool) -> None:
        """The Vector tab's placement switch -- mirrors
        :meth:`set_mode`'s own shape exactly: store, then rebuild every actor under it; a no-op
        when the requested state already matches (no needless rebuild, same discipline
        :meth:`set_graticule` already uses for its own bool). See the module docstring's "Frame
        mode" section for what actually changes under a rebuild once this is set.

        **Per-view camera memory.** A flip is exactly
        as legitimate a "the coordinate system just changed under the camera" event as
        :meth:`set_layers`'s own empty->non-empty transition -- see the module docstring's own
        "Per-view camera memory" section for the full contract this implements:
        :meth:`remember_camera` snapshots the OUTGOING view under its own :meth:`camera_key`
        BEFORE ``self._frame_mode`` flips (so it is keyed by the mode being LEFT, not entered),
        then, once the rebuild has the new geometry on screen, :meth:`restore_camera` re-applies
        whatever the INCOMING key last saw -- or ``reset_camera()`` fits it fresh on this view's
        first-ever visit, the identical fallback :meth:`set_layers` uses for a brand-new ``Scene``.
        """
        enabled = bool(enabled)
        if enabled == self._frame_mode:
            return
        self.remember_camera()
        self._frame_mode = enabled
        self._rebuild()
        if not self.restore_camera(self.camera_key()):
            self._fit_to_data()

    def set_scale_space(self, enabled: bool, stretch: float, color_by_scale: bool = True) -> None:
        """The View dialog's "3-D scale-space (stack by log₂ a)" checkbox + stretch slider
-- mirrors :meth:`set_mode`/
        :meth:`set_frame_mode`'s own shape exactly: store, then rebuild every chain actor under
        it, no-op when ``(enabled, stretch, color_by_scale)`` already matches every currently
        stored value (the same cheap re-toggle/re-nudge protection :meth:`set_frame_mode`/
        :meth:`set_graticule` already use for their own state). See the module docstring's
        "Scale-space chain stacking" section for what a rebuild under this state actually
        changes -- :meth:`_build_chain_geometry`'s own per-point z-lift, the vexag composition,
        gap splitting, and the per-scale colormap.

        ``color_by_scale`` (default ``True``, matching the spec's own "default True in
        scale-space mode" wording) has no dedicated View-dialog control in v1 -- the checkbox
        and slider are the only two new widgets this task adds; this keyword exists for internal
        testability and as a documented, harmless escape hatch for a future control.

        **Per-view camera memory (the module docstring's own "folds in the scale-space flag"
        section).** Exactly :meth:`set_frame_mode`'s own remember-before/restore-or-reset-after
        contract: :meth:`remember_camera` snapshots the OUTGOING view under its own
        :meth:`camera_key` BEFORE ``self._scale_space_enabled`` changes, then, once the rebuild
        has the new (possibly towering) geometry on screen, :meth:`restore_camera` re-applies
        whatever the INCOMING key last saw, or ``reset_camera()`` fits it fresh on this exact
        state's first-ever visit -- a toggled-on cone deserves a fit, not the flat view's stale
        camera.
        """
        enabled = bool(enabled)
        stretch = float(stretch)
        color_by_scale = bool(color_by_scale)
        if (enabled, stretch, color_by_scale) == (
                self._scale_space_enabled, self._scale_space_stretch, self._color_by_scale):
            return
        self.remember_camera()
        self._scale_space_enabled = enabled
        self._scale_space_stretch = stretch
        self._color_by_scale = color_by_scale
        self._rebuild()
        if not self.restore_camera(self.camera_key()):
            self._plotter.reset_camera()

    def _default_scale_space_geometry(self):
        """``(max_grid_dim, n_scales)`` from the first CURRENTLY on-screen ("ok") entry that
        carries both a field (for ``max_grid_dim = max(nx, ny)``, the field's own pixel-grid
        span -- the natural horizontal scale the vertical stretch should read against) and at
        least one chain with a real ``log2_scales`` array (for ``n_scales``, the count of
        distinct scale rungs actually reached -- mirrors :meth:`_scale_space_colors`'s own
        ``n_scales`` derivation, so a slider tick and a colormap band always agree on what "one
        scale" means). ``None`` when nothing currently on screen qualifies -- shared by
        :meth:`default_scale_space_stretch` and :meth:`default_scale_space_max_grid_dim` so the
        two can never disagree about which entry they measured."""
        for entry in self._entries:
            if entry.get("status") != "ok":
                continue
            field = entry.get("field")
            result = entry.get("result")
            if field is None or result is None:
                continue
            nx, ny = getattr(field, "nx", None), getattr(field, "ny", None)
            if not nx or not ny:
                continue
            chains = result.get("chains") or ()
            scale_values: list[float] = []
            for chain in chains:
                log2_scales = chain.get("log2_scales")
                if log2_scales is not None and len(log2_scales):
                    scale_values.extend(float(v) for v in log2_scales)
            if not scale_values:
                continue
            n_scales = len(set(round(v, 9) for v in scale_values))
            if n_scales <= 0:
                continue
            return max(nx, ny), n_scales
        return None

    def default_scale_space_stretch(self) -> float:
        """The View dialog's own suggested slider default: ``max_grid_dim /
        (2 * n_scales)`` (:meth:`_default_scale_space_geometry`). ``1.0`` (a plain, harmless,
        strictly-positive fallback -- never 0 or NaN, since the View dialog's checkbox is never
        gated on chains existing) when nothing currently on screen qualifies."""
        geometry = self._default_scale_space_geometry()
        if geometry is None:
            return 1.0
        max_grid_dim, n_scales = geometry
        return max_grid_dim / (2.0 * n_scales)

    def default_scale_space_max_grid_dim(self) -> float:
        """The View dialog's own stretch-slider soft-range upper bound ("slider
        range 0..max_grid_dim") -- the SAME ``max_grid_dim`` :meth:`default_scale_space_stretch`
        derives its own default from (:meth:`_default_scale_space_geometry`), exposed separately
        so the dialog can rebind its control's soft range independently of, but always
        consistently with, the suggested default value. ``1.0`` (matching
        :meth:`default_scale_space_stretch`'s own fallback) when nothing on screen qualifies."""
        geometry = self._default_scale_space_geometry()
        return float(geometry[0]) if geometry is not None else 1.0

    # -- per-view camera memory ---------------------
    #
    # See the module docstring's own "Per-view camera memory" section for the full contract; both
    # :meth:`set_frame_mode` and :meth:`set_layers` are the only current callers of
    # :meth:`remember_camera`/:meth:`restore_camera`, but neither is special -- a future caller
    # (e.g. a "recompute camera fit" affordance) can use the same public seam.

    def camera_key(self) -> str:
        """The CURRENT view's own camera-memory key -- ``"geo:<mode>"`` in geo mode (one shared
        WGS84-projected coordinate system regardless of which layer is "active"), or
        ``"frame:<bounds-sig>"`` in frame mode, where the bounds signature is the on-screen
        geometry's own combined extent (:meth:`_frame_bounds_signature`) -- frame mode's own
        coordinate system is native to whichever field is drawn, so distinct geometries need
        distinct keys even at the identical projection mode/frame-mode flag.

        **Scale-space addition:** an ``"|sspace:on"`` suffix when
        :attr:`self._scale_space_enabled` -- toggling scale-space changes the z-extent of every
        chain in the scene, which is exactly as much "the coordinate system just changed under
        the camera" as a geo<->frame flip, so it earns its own remembered framing (see
        :meth:`set_scale_space`) rather than reusing whatever the flat view last had. NOTHING is
        appended while disabled -- every pre-Task-6 exact-string key (``"geo:mercator"``, etc.)
        is unchanged."""
        if self._frame_mode:
            key = f"frame:{self._frame_bounds_signature()}"
        else:
            key = f"geo:{self._mode}"
        if self._scale_space_enabled:
            key += "|sspace:on"
        return key

    def _apply_raster_visibility(self) -> None:
        """Each entry's ``show_raster`` (default True) onto its drape actor -- the dataset hidden
        from the source header row while its vectors stay. Re-applied after every
        :meth:`set_layers` and every :meth:`_rebuild`, since both can mint the actor afresh."""
        actors = self._plotter.renderer.actors
        for entry in self._entries:
            actor = actors.get(f"layer-{entry['layer'].layer_id}-raster")
            if actor is not None:
                actor.SetVisibility(bool(entry.get("show_raster", True)))

    def _fit_to_data(self) -> None:
        """Frame the DATA layers' actors (rasters + their vectors, :attr:`_layer_actors`) -- not
        the graticule, footprints, previews or reference layers, which can span the world
        (BOEM's ``website/states``) and would shrink a 12 km window to nothing. On the globe
        the camera is first put OUTSIDE the Earth on the data's own radial, so a fit never looks
        through the planet at a dataset on the far side; flat modes keep their top-down normal.
        Falls back to a plain ``reset_camera()`` when no data actor is on screen."""
        actors = self._plotter.renderer.actors
        names = [n for ns in self._layer_actors.values() for n in ns if n in actors]
        if not names:
            self._plotter.reset_camera()
            return
        b = np.array([actors[n].GetBounds() for n in names], dtype=np.float64)
        bounds = (b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max(), b[:, 4].min(), b[:, 5].max())
        if self._mode == "globe" and not self._frame_mode:
            centre = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
            r = float(np.linalg.norm(centre))
            if r > 0:
                n = centre / r
                up = np.array([0.0, 0.0, 1.0]) - n * n[2]                 # north, made tangent
                cam = self._plotter.camera
                cam.focal_point = tuple(centre)
                cam.position = tuple(centre + n * r)
                cam.up = tuple(up / np.linalg.norm(up)) if np.linalg.norm(up) > 1e-6 else (0.0, 1.0, 0.0)
        self._plotter.reset_camera(bounds=bounds)

    def remember_camera(self) -> None:
        """Snapshot the live camera's full navigable state under :meth:`camera_key`'s CURRENT
        value -- see :meth:`_camera_snapshot` for exactly what is captured. Overwrites whatever
        was remembered for that key before, if anything."""
        self._camera_memory[self.camera_key()] = self._camera_snapshot()

    def restore_camera(self, key: str) -> bool:
        """Re-apply a previously :meth:`remember_camera`'d state, by explicit ``key``. Returns
        whether ``key`` was known -- ``False`` (camera left completely untouched) for a key this
        ``Scene`` has never remembered, so a caller can fall back to ``reset_camera()`` for a
        first-ever visit to that key, exactly like :meth:`set_layers`'s own empty->non-empty
        framing. Renders once, at the end, on an actual restore -- setting ``camera.position``/
        etc. directly (unlike ``add_mesh``) does not trigger pyvista's own auto-render."""
        state = self._camera_memory.get(key)
        if state is None:
            return False
        cam = self._plotter.camera
        cam.position = state["position"]
        cam.focal_point = state["focal_point"]
        cam.up = state["up"]
        cam.parallel_scale = state["parallel_scale"]
        cam.SetParallelProjection(state["parallel_projection"])
        cam.clipping_range = state["clipping_range"]
        self._plotter.render()
        return True

    # -- crossing the process boundary -------------
    #
    # ``Project.cameras`` (model/project.py, the additive "cameras" payload key) is
    # where a remembered viewpoint RESTS between sessions; :attr:`self._camera_memory` is where it
    # LIVES during one. These two methods are the only seam between them -- nothing about the live
    # dict, :meth:`remember_camera` or :meth:`restore_camera` changes -- and neither touches the
    # plotter, so both are exercisable against a plain off-screen ``pv.Plotter``.

    def export_camera_memory(self) -> dict[str, dict]:
        """A JSON-safe COPY of this ``Scene``'s whole camera memory, shaped for
        ``Project.cameras``. Every :meth:`_camera_snapshot` tuple (position/focal_point/up/
        clipping_range) goes out as a list, matching what ``json.dumps`` would write anyway, so
        the exported dict already carries the on-disk types and a save->open->save cycle cannot
        silently re-type a viewpoint from tuples to lists halfway through. A copy, deliberately:
        the caller owns what it gets back and mutating it must never reach the live memory."""
        return {key: {k: (list(v) if isinstance(v, tuple) else v) for k, v in state.items()}
                for key, state in self._camera_memory.items()}

    def import_camera_memory(self, payload: dict) -> None:
        """MERGE a saved ``Project.cameras`` blob into this ``Scene``'s camera memory -- the
        inverse of :meth:`export_camera_memory`, with every JSON list re-tupled back into the
        exact :meth:`_camera_snapshot` shape so an imported entry compares equal to a live one.

        Merges, never replaces: opening a project mid-session must not discard viewpoints this
        ``Scene`` already remembers for keys the file says nothing about. A key the payload DOES
        name adopts the payload's value -- a viewpoint deliberately saved with the project is the
        more considered of the two.

        A malformed entry (a non-string key, a non-dict value, or a dict short of any of
        :data:`_CAMERA_FIELDS`) is SKIPPED, never raised: a remembered viewpoint is a preference,
        not a measurement, and one corrupt entry must not be able to block the
        project that carries it."""
        for key, state in payload.items():
            if not isinstance(key, str) or not isinstance(state, dict):
                continue
            if not all(name in state for name in _CAMERA_FIELDS):
                continue
            self._camera_memory[key] = {k: (tuple(v) if isinstance(v, list) else v)
                                        for k, v in state.items()}

    def _camera_snapshot(self) -> dict:
        """The live camera's full navigable state -- position/focal_point/up/parallel_scale/
        parallel_projection/clipping_range -- as a plain dict, the value type
        :attr:`self._camera_memory` traffics in. Split out from :meth:`remember_camera` so
        :meth:`set_layers`'s active-layer-swap branch can snapshot under an EXPLICIT key (the
        OUTGOING one, computed before ``self._entries`` was reassigned) rather than whatever
        :meth:`camera_key` would return AFTER."""
        cam = self._plotter.camera
        return {
            "position": tuple(cam.position),
            "focal_point": tuple(cam.focal_point),
            "up": tuple(cam.up),
            "parallel_scale": float(cam.parallel_scale),
            "parallel_projection": bool(cam.GetParallelProjection()),
            "clipping_range": tuple(cam.clipping_range),
        }

    def _frame_bounds_signature(self) -> str:
        """A short string identifying the CURRENT frame-mode geometry's own combined extent,
        rounded to a few decimal places -- half of :meth:`camera_key`'s ``"frame:<bounds-sig>"``
        value.

        Computed as the union of every currently-admitted "ok", non-points entry's own four
        corners, run through ITS OWN ``field.frame.to_scene`` (never a whole-grid pass -- four
        points per layer is plenty to bound a rectangle), rather than singling out one "active"
        layer by id -- this ``Scene`` never needs to know which layer ``main_window.py`` currently
        considers active (its own frame-mode admission rule already guarantees every OTHER
        admitted raster layer shares the active one's own frame AND grid shape, so in practice
        this reduces to that one layer's own extent regardless -- see ``_sync_arrangement``'s own
        "Frame mode" docstring section). Skips a field with no ``.frame``/``.x_axis``/``.y_axis``
        of its own (a bare-``ndarray`` sibling -- ``main_window``'s own admission rule already
        excludes these from a REAL frame-mode display; a leftover here can only be a caller
        handing raw entries straight to this ``Scene``, e.g. a test) and any field whose own
        placement raises -- the same per-layer crash-avoidance posture :meth:`_rebuild` already
        takes one level up -- returning an honest ``"empty"`` signature rather than crashing when
        nothing could be measured at all.
        """
        xs: list[float] = []
        ys: list[float] = []
        for entry in self._entries:
            if entry.get("status") != "ok" or entry.get("kind") == "points":
                continue
            field = entry.get("field")
            frame = getattr(field, "frame", None)
            x_axis = getattr(field, "x_axis", None)
            y_axis = getattr(field, "y_axis", None)
            if frame is None or x_axis is None or y_axis is None:
                continue
            if len(x_axis) == 0 or len(y_axis) == 0:
                continue
            try:
                # Planar axis extremes directly -- the signature must describe the geometry
                # frame mode actually DRAWS, and both drawing paths are planar (see
                # _scene_points's "Frame mode" section); running the corners through
                # frame.to_scene here would re-import the world-map wrap and let the camera key
                # disagree with the on-screen bounds.
                x0, x1 = float(x_axis[0]), float(x_axis[-1])
                y0, y1 = float(y_axis[0]), float(y_axis[-1])
            except Exception:
                continue
            xs.extend([min(x0, x1), max(x0, x1)])
            ys.extend([min(y0, y1), max(y0, y1)])
        if not xs:
            return "empty"
        return "{:.3f},{:.3f},{:.3f},{:.3f}".format(min(xs), max(xs), min(ys), max(ys))

    def clear(self) -> None:
        """Remove every actor this scene owns (mesh + legend) and forget the last
        ``set_layers`` payload -- a subsequent ``set_mode()`` has nothing left to rebuild."""
        self._clear_actors()
        self._entries = []
        self._legend_lines = []

    def actor_count(self) -> int:
        """Every mesh actor this scene owns (raster + the vector actors) -- the legend text
        is never counted here."""
        return sum(len(names) for names in self._layer_actors.values())

    def set_mask(self, modulus_pctl: float, scale_lo: int, scale_hi: int) -> None:
        """View-state chain visibility filter -- VECTOR actors only, applied WITHOUT creating or
        destroying any actor (the design "view-state... via visibility/mask updates"; sec 7's
        <16ms bar exists because this must be cheap enough to run on every knob nudge). Only the
        chains actor's per-point ALPHA channel changes on a COPY of that layer's own
        ``base_rgba``; the RGB, and the actor/mapper/``PolyData`` objects themselves, are exactly
        the ones the last rebuild produced.

        ``modulus_pctl`` (0-100, clamped): a chain is visible only if its own ``max(log2_mod)`` is
        at or above the ``modulus_pctl``-th percentile of every DRAWABLE chain's ``max(log2_mod)``
        in that SAME layer (each layer's own distribution -- not a global one across layers).
        ``0`` keeps everything (the 0th percentile is the minimum, and every chain is ``>=`` its
        own layer's minimum by construction).

        ``scale_lo``/``scale_hi`` (a CHAIN-DEPTH range, adjudicated semantics -- see below):
        arrangement's extrema actor is finest-scale-only (module docstring), so there is no
        per-scale extrema layer here to filter the way ``dynamix.devices.filters.ScaleSelect``
        does. v1 redirects the scale range onto each CHAIN's own DEPTH instead --
        ``wtmm_backend.chains2d``'s own comment ("Contiguous finest-anchored chains -> valid
        entries are a row-wise prefix") is the grounding fact: every chain is seeded at the finest
        scale (index 0) and walks toward coarser scales with no gaps, so its DEEPEST scale index is
        always ``len(chain) - 1`` (``hi_scale_idx``). A chain is visible iff ``scale_lo <=
        hi_scale_idx <= scale_hi``, with ``scale_hi == 0`` meaning "no cap" (mirrors
        ``dynamix.devices.filters.HLineLength``'s own ``max_len`` convention: ``0`` is not a real
        depth ceiling, it is the sentinel for "don't apply one"). Both bounds discriminate:
        ``scale_lo`` hides chains too SHALLOW to reach it, ``scale_hi`` (when nonzero) hides chains
        too DEEP -- an earlier revision of this filter used interval-INTERSECTION against a span
        that always starts at 0, under which ``scale_hi`` could never exclude anything except by
        going negative; this depth-RANGE test was adjudicated to replace it precisely because both
        knobs need to matter.

        Extrema and the ROI outline are never touched -- neither carries a per-chain modulus or
        depth statistic to filter by. The chosen mask is remembered (``self._mask``) and silently
        reapplied after the next rebuild (see :meth:`_rebuild`). **Masking never removes geometry
        or toggles actor visibility -- see the module docstring's "stays pickable" note: a masked-
        out chain's points are still there, just alpha-zeroed, and a naive VTK pick would still
        hit them.**
        """
        self._mask = (float(np.clip(modulus_pctl, 0.0, 100.0)), int(scale_lo), int(scale_hi))
        self._apply_mask_to_all()

    def set_colormap(self, name: str) -> None:
        """Swap every RASTER actor's LUT in place -- a fresh ``pv.LookupTable(cmap=name)``
        assigned to ``mapper.lookup_table`` -- WITHOUT rebuilding the actor/mapper/``StructuredGrid``
        (the EQSelect-audited anti-pattern, done right this time: a full ``add_mesh`` per colormap
        change). The new table's ``scalar_range``/``nan_opacity`` are copied from the mapper's
        current values so the swap changes only the HUE mapping, not the range or the "sentinel
        cells stay transparent" contract :meth:`_add_raster_actor` set up (pyvista's own default
        for a brand-new ``LookupTable`` is an opaque NaN color and a ``(0, 1)`` range, neither of
        which matches what is already on screen -- confirmed empirically, not assumed). Vector
        actors are untouched (see the module docstring). The chosen colormap survives a later
        ``set_mode`` rebuild (:meth:`_add_raster_actor` reads ``self._colormap``, not a literal).

        Reassigning ``mapper.lookup_table`` is a VTK-level mutation, not an ``add_mesh`` call --
        every OTHER actor-producing path in this module repaints itself for free via
        ``add_mesh``'s own ``render=True`` default, but this one does not, so a repaint is
        requested explicitly (confirmed by an offscreen screenshot probe: the frame stayed stale
        without it). ``pv.Plotter.render()`` works offscreen, so this is unconditional.
        """
        self._colormap = str(name)
        for names in self._layer_actors.values():
            for actor_name in names:
                if not actor_name.endswith("-raster"):
                    continue
                mapper = self._plotter.actors[actor_name].mapper
                if "rgba" in mapper.dataset.point_data:
                    continue                   # a hillshaded drape carries baked colours
                lut = pv.LookupTable(cmap=self._colormap)
                lut.nan_opacity = 0
                lut.scalar_range = mapper.scalar_range
                mapper.lookup_table = lut
        self._plotter.render()

    # -- the View dialog's Camera/Frame/Display capabilities --------

    def set_vertical_exaggeration(self, factor: float) -> None:
        """The Display tab's vertical-exaggeration knob -- stores ``factor`` and triggers a full
        rebuild (:meth:`_rebuild`, the same one :meth:`set_mode` performs), since exaggeration is
        a property of how every point is PLACED (threaded into every ``project(..., vexag=...)``
        call site in this module), not a view-state overlay :meth:`set_mask` could patch in
        place. See ``self._vexag``'s own docstring for why this has no visible effect yet."""
        self._vexag = float(factor)
        self._rebuild()

    def set_graticule(self, enabled: bool) -> None:
        """The Frame tab's Graticule checkbox -- a 10-degree lat/lon grid (:func:`_graticule_geometry`)
        projected through the CURRENT mode, as one actor. A no-op when already in the requested
        state (never removes-then-re-adds needlessly); survives a later rebuild via
        :meth:`_apply_graticule`, called at :meth:`_rebuild`'s own tail -- the same "remembered,
        silently reapplied" contract ``self._mask`` already keeps across a mode switch."""
        enabled = bool(enabled)
        if enabled == self._graticule:
            return
        self._graticule = enabled
        self._apply_graticule()

    def set_background(self, color: str) -> None:
        """Runtime background setter -- the Display tab's color-swatch button calls this on every
        pick. Sibling of the constructor's own one-time ``background`` argument (see the module
        docstring's "Themed background" section): the identical plain hex-string contract, no
        ``dynamix.shell.theme`` import here either -- ``Scene`` stays Qt-free."""
        self._plotter.set_background(color)

    def _apply_graticule(self) -> None:
        """Remove whatever graticule actor currently exists, then re-add one iff
        ``self._graticule`` is True AND ``self._frame_mode`` is False -- called both from
        :meth:`set_graticule` (a real toggle) and from :meth:`_rebuild`'s own tail (mode/layer/
        frame-mode changes, where the geometry itself may be stale even if the on/off state did
        not change). The graticule is lon/lat geometry (:func:`_graticule_geometry` walks
        latitude/longitude, not pixels) -- there is no frame-mode equivalent, so frame mode simply
        suppresses it rather than drawing something meaningless; the checkbox's own remembered
        state (``self._graticule``) is untouched, so toggling frame mode back off resumes exactly
        whatever the checkbox was already set to."""
        self._remove_graticule_actor()
        if self._graticule and not self._frame_mode:
            self._add_graticule_actor()

    def _remove_graticule_actor(self) -> None:
        if self._graticule_present:
            self._plotter.remove_actor(_GRATICULE_NAME, render=False)
            self._graticule_present = False

    def _add_graticule_actor(self) -> None:
        points, lines = _graticule_geometry(self._mode, self._vexag)
        self._plotter.add_mesh(pv.PolyData(points, lines=lines), color=_GRATICULE_COLOR,
                                line_width=1.0, name=_GRATICULE_NAME, reset_camera=False)
        self._graticule_present = True

    def set_selection(self, selected) -> None:
        """The "just picked" highlight -- ``selected`` is an iterable of ``(layer_id,
        chain_index)`` pairs (``GroupPalette.selection()``'s own return shape). Recolors those
        chains' points :data:`SELECTION_COLOR`, through the SAME per-point RGBA path
        :meth:`set_mask` uses -- no actor/mapper/``PolyData`` is created or destroyed, and alpha
        (visibility) is untouched (see the module docstring's compositing-order note). Replaces
        the ENTIRE previous selection -- this is not an accumulating add; ``GroupPalette`` is the
        one place selection accumulation semantics live (its own ``add_pick``), and it always
        hands this method its full current selection, not a delta.

        A chain index outside the range :meth:`pick` could ever have returned for its layer is
        silently ignored when the mask is next (re)applied (``_apply_mask_to_layer`` masks its
        per-point comparisons against that layer's own ``chain_indices``, which can never contain
        an out-of-range value) -- so a stale selection entry naming a layer/chain that has since
        been rebuilt away is inert, not an error.
        """
        by_layer: dict[int, set[int]] = {}
        for layer_id, chain_index in selected:
            by_layer.setdefault(int(layer_id), set()).add(int(chain_index))
        self._selection = by_layer
        self._apply_mask_to_all()

    def set_group_preview(self, groups: dict) -> None:
        """The PRE-COMMIT group-color preview -- ``groups`` is ``GroupPalette.groups()``'s own
        shape, ``{name: {"chains": [(layer_id, chain_index), ...], "color": [r, g, b]}}``. Every
        chain named by any group shows that group's color, through the same per-point RGBA path
        :meth:`set_mask` uses (no rebuild, alpha untouched -- see the module docstring). This is
        NOT the committed ``chain["group_color"]`` :func:`_chain_color` reads at build time (that
        is baked into ``base_rgba`` from the last-resolved RESULT, and only ever changes via
        The commit + a fresh resolve) -- this is a live overlay on top of it, showing what
        WOULD be committed if the user hit Commit right now. Replaces the entire previous preview
        (same non-accumulating contract as :meth:`set_selection`, for the same reason).

        A chain claimed by two groups at once (should never happen -- ``GroupPalette`` does not
        enforce exclusivity, but nothing in this task's UI offers a way to cause it either) shows
        whichever group's entry is processed last in ``groups`` -- an arbitrary but harmless tie-
        break, not a crash.
        """
        by_layer: dict[int, dict[int, tuple[int, int, int]]] = {}
        for spec in groups.values():
            color = tuple(int(c) for c in spec.get("color", (0, 0, 0)))
            for layer_id, chain_index in spec.get("chains", ()):
                by_layer.setdefault(int(layer_id), {})[int(chain_index)] = color
        self._group_preview = by_layer
        self._apply_mask_to_all()

    def pick(self, x_px: float, y_px: float, viewport, mvp=None):
        """Screen-space click -> ``(layer_id, chain_index)``, or ``None`` on a miss (the design "click selects a chain"). Pure math over ``dynamix.core.selection.project_points``/
        ``nearest_point`` -- Qt-free, unit-testable with an injected ``mvp`` (a 4x4 model-view-
        projection matrix; see ``core/selection.py``'s own docstring for the exact convention).

        ``mvp=None`` (the live app's own call) derives the matrix from the CURRENT
        ``self._plotter.camera`` via ``vtkCamera.GetCompositeProjectionTransformMatrix`` -- see
        :meth:`_camera_mvp`.

        Candidates come from :meth:`_pick_candidates` (this loop was extracted out of what used to be this method's own body, additively, so :meth:`pick_in_region`
        can share it byte-for-byte -- see that method's own docstring and the module docstring's
        "Region picking" section; this method's OWN behaviour is unchanged by the extraction, and
        every test written against it before that task still passes unmodified): every layer with
        drawable chains contributes its chains actor's OWN points (read straight off
        ``mapper.dataset.points`` -- always current, never a stale copy), filtered to those that
        are BOTH in front of the camera and inside the view frustum (``project_points``'s own
        ``visible`` return) AND mask-visible RIGHT NOW (:meth:`_point_visibility` -- the module
        docstring's "stays pickable" note: a chain ``set_mask`` has alpha-zeroed is still real VTK
        geometry, and must be excluded here explicitly, not assumed invisible to a picker).

        The single globally nearest candidate (across every layer at once, not per-layer) within
        :data:`_PICK_MAX_DIST_PX` wins; a tie is broken by whichever layer's chain lookup was
        iterated first (dict insertion order -- the order layers were last built in), which is
        arbitrary but deterministic, never a crash.
        """
        if mvp is None:
            mvp = self._camera_mvp(viewport)
            if mvp is None:
                return None

        pts2d_cat, layer_cat, chain_cat = self._pick_candidates(mvp, viewport)
        if pts2d_cat.shape[0] == 0:
            return None

        idx = nearest_point(pts2d_cat, (x_px, y_px), max_dist=_PICK_MAX_DIST_PX)
        if idx < 0:
            return None
        return int(layer_cat[idx]), int(chain_cat[idx])

    def pick_in_region(self, region, viewport, mvp=None):
        """Box/lasso resolution in the 3-D views (the design's last
        bullet,) -- the region-selection sibling of :meth:`pick`, over the exact same
        candidate pool (:meth:`_pick_candidates`) so a box/lasso drag can never disagree with
        what a plain click could ever have hit.

        ``region`` is ``("box", x0, y0, x1, y1)`` (:func:`dynamix.core.selection.points_in_box`,
        which normalizes the two corners itself -- either drag direction is fine) or ``("poly",
        [(x, y), ...])`` (:func:`dynamix.core.selection.points_in_polygon`, the lasso's own
        captured vertices, auto-closed by that function). Both are pixel coordinates in the SAME
        top-left-origin convention :meth:`pick`/``project_points`` already use.

        A chain is selected the moment ANY ONE of its own candidate points falls inside the
        region -- see the module docstring's "Region picking" section for why "any point", not
        "every point". ``mvp=None`` derives the matrix from the live camera, identically to
        :meth:`pick` (and is likewise treated as a miss -- an empty list here, not a crash -- when
        the viewport has zero height).

        Returns a sorted, de-duplicated ``list[(layer_id, chain_index)]``; an empty list on a
        region that encloses nothing (or when there are no drawable chains at all), never
        ``None`` -- unlike :meth:`pick`'s single-hit contract, a region gesture's natural "found
        nothing" answer is an empty batch, the shape ``GroupPalette.apply_picks`` already expects.
        """
        if mvp is None:
            mvp = self._camera_mvp(viewport)
            if mvp is None:
                return []

        pts2d_cat, layer_cat, chain_cat = self._pick_candidates(mvp, viewport)
        if pts2d_cat.shape[0] == 0:
            return []

        kind = region[0]
        if kind == "box":
            _kind, x0, y0, x1, y1 = region
            inside = points_in_box(pts2d_cat, x0, x1, y0, y1)
        elif kind == "poly":
            _kind, polygon = region
            inside = points_in_polygon(pts2d_cat, polygon)
        else:
            raise ValueError(f"unknown pick_in_region kind {kind!r}")

        if not inside.any():
            return []
        hit_idx = np.nonzero(inside)[0]
        pairs = {(int(layer_cat[i]), int(chain_cat[i])) for i in hit_idx}
        return sorted(pairs)

    def _pick_candidates(self, mvp, viewport):
        """Shared candidate assembly for :meth:`pick` and :meth:`pick_in_region` -- extracted, additively, from what used to be :meth:`pick`'s own inline
        loop (byte-identical behaviour; :meth:`pick`'s existing tests pass unmodified against
        this factoring).

        Every layer with drawable chains (present in ``self._chain_lookup``) contributes its
        chains actor's OWN points (read straight off ``mapper.dataset.points`` -- always current,
        never a stale copy), filtered to those that are BOTH in front of the camera and inside the
        view frustum (``project_points``'s own ``visible`` return) AND mask-visible right now
        (:meth:`_point_visibility` -- the module docstring's "stays pickable" note: a chain
        ``set_mask`` has alpha-zeroed is still real VTK geometry, and must be excluded here
        explicitly, not assumed invisible to a picker).

        Returns ``(pts2d, layer_ids, chain_indices)``, three parallel 1-D arrays (length N,
        possibly 0), concatenated across every contributing layer in ``self._chain_lookup``'s own
        insertion order -- a well-shaped empty triple (not ``None``) when there is nothing to
        report, so both callers can test ``pts2d.shape[0] == 0`` uniformly rather than each
        re-deriving an ``if not candidates`` guard of their own.
        """
        all_pts2d = []
        owner_layer = []
        owner_chain = []
        for layer_id, (_starts, chain_indices) in self._chain_lookup.items():
            actor = self._plotter.actors.get(self._chains_actor_name(layer_id))
            if actor is None:
                continue
            point_visible = self._point_visibility(layer_id)
            if point_visible is None or not point_visible.any():
                continue
            pts3d = np.asarray(actor.mapper.dataset.points)
            pts2d, frustum_visible = project_points(pts3d, mvp, viewport)
            candidate = point_visible & frustum_visible
            if not candidate.any():
                continue
            idxs = np.nonzero(candidate)[0]
            all_pts2d.append(pts2d[idxs])
            owner_layer.append(np.full(idxs.shape, layer_id, dtype=np.int64))
            owner_chain.append(chain_indices[idxs])

        if not all_pts2d:
            return (np.empty((0, 2), dtype=float), np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int64))

        return (np.concatenate(all_pts2d, axis=0), np.concatenate(owner_layer),
                np.concatenate(owner_chain))

    def _camera_mvp(self, viewport):
        """The live camera's model-view-projection matrix as a plain ``(4, 4)`` numpy array --
        ``vtkCamera.GetCompositeProjectionTransformMatrix(aspect, -1, 1)`` (the "concatenation of
        the ViewTransform and the ProjectionTransform", VTK's own docstring), converted via
        ``pv.array_from_vtkmatrix``. ``None`` when ``viewport`` has zero height (nothing sane to
        divide by) -- callers treat that as a pick miss, not a crash."""
        w, h = viewport
        if not h:
            return None
        aspect = float(w) / float(h)
        matrix = self._plotter.camera.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
        return pv.array_from_vtkmatrix(matrix)

    # -- rebuild ---------------------------------------------------------------------------------

    _STYLE_KEYS = ("status", "signature", "colormap", "hillshade", "stretch", "surface",
                   "vtrail_color", "show_raster", "kind",
                   # Surface-source + drape: identity stamps computed by
                   # main_window (arrays are immutable; a new z-source layer or a fresh h_map
                   # mints a new id), so the filter-only fast path correctly falls back to a
                   # full rebuild when either changes.
                   "surface_field_id", "drape_id", "levels")

    def _filter_only_resync(self, entries) -> bool:
        """True when ``entries`` differ from the current ones ONLY in filtered results -- same
        ordered layers, same fields, same signatures and raster styles, every result real -- and
        the changed layers' vector actors were rebuilt in place. Any other difference returns
        False untouched, and :meth:`set_layers` takes the full rebuild."""
        if not self._entries or len(entries) != len(self._entries):
            return False
        changed = []
        for old, new in zip(self._entries, entries):
            if old["layer"].layer_id != new["layer"].layer_id:
                return False
            if id(old.get("field")) != id(new.get("field")):
                return False
            if any(old.get(k) != new.get(k) for k in self._STYLE_KEYS):
                return False
            if id(old.get("result")) == id(new.get("result")):
                continue
            if old.get("result") is None or new.get("result") is None:
                return False
            changed.append(new)
        for entry in changed:
            if not self._resync_vector_actors(entry):
                return False
        if changed:
            if self._mask is not None:
                self._apply_mask_to_all()
            self._plotter.render()
        return True

    def _resync_vector_actors(self, entry) -> bool:
        """Replace ONE layer's chains/extrema/H-line/ROI actors from its new (re-filtered)
        result, leaving its raster drape alone. False on any failure -- the caller then falls
        back to the full rebuild rather than trusting a half-updated layer."""
        layer_id = entry["layer"].layer_id
        old_names = self._layer_actors.get(layer_id, [])
        raster_names = [n for n in old_names if n.endswith("-raster")]
        try:
            for n in old_names:
                if n not in raster_names:
                    self._plotter.remove_actor(n, render=False)
            self._chain_lookup.pop(layer_id, None)
            self._vector_masks.pop(layer_id, None)
            names = list(raster_names)
            chain_lookup_entry, vector_mask_entry = self._build_vector_actors(
                layer_id, entry["field"], entry["result"], names, entry.get("vtrail_color"))
            self._layer_actors[layer_id] = names
            if chain_lookup_entry is not None:
                self._chain_lookup[layer_id] = chain_lookup_entry
            if vector_mask_entry is not None:
                self._vector_masks[layer_id] = vector_mask_entry
            return True
        except Exception:
            return False

    def _rebuild(self) -> None:
        self._clear_actors()
        self._raster_grids.clear()
        legend_lines: list[str] = []
        for entry in self._entries:
            layer = entry["layer"]
            result = entry.get("result")
            status = entry["status"]
            if status == "ok" and entry.get("kind") == "points" and self._frame_mode:
                # Decided HERE, before ever calling
                # _add_points_geometry, rather than caught as a generic "error:<msg>" below -- a
                # point layer has no frame-mode placement at all (that method's own docstring), so
                # this is an honest, expected posture ("session view only"), not a malfunction.
                status = "frame-points"
            elif status == "ok" and entry.get("kind") == "points":
                try:
                    self._add_points_geometry(layer, entry["pointset"],
                                              entry.get("points_color"))
                    continue
                except Exception as exc:
                    # Same crash-avoidance posture as the raster branch below: a malformed
                    # PointSet demotes only this layer to an honest legend line.
                    status = f"error:{exc}"
            elif status == "ok":
                field = entry["field"]
                try:
                    self._add_layer_geometry(layer, field, result,
                                             colormap=entry.get("colormap"),
                                             vtrail_color=entry.get("vtrail_color"),
                                             hillshade=entry.get("hillshade"),
                                             stretch=entry.get("stretch"),
                                             surface=entry.get("surface"),
                                             surface_field=entry.get("surface_field"),
                                             drape=entry.get("drape"),
                                             levels=entry.get("levels"))
                    continue
                except NoGeoreference:
                    # Contract: "ok" entries carry a georeferenced field. Guarded
                    # anyway rather than crashing on a caller/field mismatch -- an honest legend
                    # line beats a traceback for what is, either way, un-placeable data.
                    status = "no-georeference"
                except Exception as exc:
                    # Crash-avoidance posture ("never a crash"): a malformed field OR
                    # a malformed chain/extrema/ROI -- demotes only THIS layer to an honest
                    # "error:<reason>" legend line (reusing the same formatting as an upstream
                    # compute error). It must never take the whole flip down, and every OTHER
                    # layer in the same set_layers() call still drapes normally (each entry is
                    # handled independently, in its own try/except, inside this same loop).
                    status = f"error:{exc}"
            elif status != "no-georeference" and entry.get("kind") != "points" \
                    and entry.get("field") is not None:
                # A loaded raster shows in the Vector view BEFORE any compute -- the drape (and
                # the 3-D surface) need only the field; chains/extrema join when a result lands
                # (_add_layer_geometry already treats result=None as raster-only). The legend
                # line still carries the honest status ("edited — run to compute" / the compute
                # error), so nothing pretends to be resolved.
                try:
                    self._add_layer_geometry(layer, entry["field"], None,
                                             colormap=entry.get("colormap"),
                                             vtrail_color=entry.get("vtrail_color"),
                                             hillshade=entry.get("hillshade"),
                                             stretch=entry.get("stretch"),
                                             surface=entry.get("surface"),
                                             surface_field=entry.get("surface_field"),
                                             drape=entry.get("drape"),
                                             levels=entry.get("levels"))
                except NoGeoreference:
                    status = "no-georeference"
                except Exception as exc:
                    status = f"error:{exc}"
            legend_lines.append(self._legend_line(layer, status))
        self._legend_lines = legend_lines
        self._set_legend(legend_lines)
        self._draw_footprints()
        self._draw_previews()
        self._draw_reference_layers()
        self._apply_raster_visibility()
        # A rebuild (set_layers/set_mode) starts every actor fully visible; re-apply whatever mask
        # was already active so switching projection mid-exploration doesn't silently reset it.
        self._apply_mask_to_all()
        # The graticule is mode-dependent geometry (see _graticule_geometry) -- rebuild it
        # too, on every set_layers/set_mode pass, exactly like the mask above. A no-op render-wise
        # when self._graticule is False (_apply_graticule's own remove-then-maybe-add is cheap
        # either way).
        self._apply_graticule()

    def _add_layer_geometry(self, layer, field, result, *, colormap=None,
                            vtrail_color=None, hillshade=None, stretch=None,
                            surface=None, surface_field=None, drape=None,
                            levels=None) -> None:
        """Everything ONE "ok" layer contributes: the raster drape, plus (when ``result`` carries
        them) its chains/extrema/ROI-outline vector actors -- ATOMIC per layer. Any exception
        raised while building the VECTOR actors rolls back every actor this call already added
        (including the raster one) before re-raising, so ``_rebuild``'s per-layer try/except
        (which demotes the layer to an "error:<msg>" legend line) never leaves a stray actor
        behind for a layer the legend says is un-placed. A ``result`` of ``None`` (no compute has
        landed for this layer yet, or none is expected) simply skips the vector step -- draping
        alone is a complete, honest "ok" entry, exactly as it was before vector actors existed.

        ``colormap``/``vtrail_color`` are the entry's own ``ui.*``
        preferences, threaded straight through to :meth:`_add_raster_actor` and
        :meth:`_build_vector_actors` -- ``None`` (every call site that predates this task) falls
        back to ``self._colormap``/:data:`VTRAIL_COLOR` at each of those, exactly as before.

        **Frame mode.** In frame mode, neither this method nor
        anything it calls (:meth:`_add_raster_actor`, :meth:`_build_vector_actors`) ever reaches
        ``geo.mapping`` at all -- see the module docstring's "Frame mode" section -- so
        :class:`NoGeoreference` simply cannot arise here; a field with no CRS places exactly as
        readily as one with a CRS. Whether a given entry belongs in this Scene's ``self._entries``
        at all (e.g. filtering to frames compatible with the active layer) is decided upstream,
        by the caller building ``set_layers``' payload, not here.

        ``names`` is built up HERE and handed to :meth:`_build_vector_actors` to append into
        DIRECTLY, in place, as each actor lands on the plotter -- fixed bug: an
        earlier version had ``_build_vector_actors`` accumulate its own LOCAL list and only
        RETURN it at the end, which meant a failure partway through (say, after the chains actor
        landed but before the ROI outline) discarded that partial list along with the exception --
        the chains actor stayed on ``self._plotter`` with no name anywhere that could ever remove
        it again, a PERMANENT orphan surviving every future ``clear()``/``set_mode()``. Sharing one
        list by reference closes that: whatever got appended before the failure is still in
        ``names`` when the ``except`` below runs.
        """
        layer_id = layer.layer_id
        if surface and surface[0]:
            self._surface_fields[id(field)] = bool(surface[1]) if len(surface) > 1 else False
        else:
            self._surface_fields.pop(id(field), None)
        raster_name = self._add_raster_actor(layer, field, colormap, hillshade, stretch, surface,
                                             surface_field=surface_field, drape=drape,
                                             levels=levels)
        names = [raster_name]
        chain_lookup_entry = None
        vector_mask_entry = None
        try:
            if result is not None:
                chain_lookup_entry, vector_mask_entry = self._build_vector_actors(
                    layer_id, field, result, names, vtrail_color)
        except Exception:
            for name in names:
                self._plotter.remove_actor(name, render=False)
            raise

        self._layer_actors[layer_id] = names
        if chain_lookup_entry is not None:
            self._chain_lookup[layer_id] = chain_lookup_entry
        if vector_mask_entry is not None:
            self._vector_masks[layer_id] = vector_mask_entry

    def _lonlat_grid_for(self, layer_id, field):
        """``field_lonlat_grid(field)``, cached per-layer on ``field``'s own identity (``self._lonlat_cache``'s own docstring for the profiling
        finding and the caching rationale). A cache hit needs ``id(field)`` to match exactly what
        was cached for THIS ``layer_id`` last time -- a different field (even a content-identical
        reopen of the same source) is treated as a genuine miss and recomputed, never assumed
        equivalent by value."""
        cached = self._lonlat_cache.get(layer_id)
        if cached is not None and cached[0] == id(field):
            return cached[1], cached[2], cached[3], cached[4]
        lon2d, lat2d, values2d, stride = field_lonlat_grid(field)
        self._lonlat_cache[layer_id] = (id(field), lon2d, lat2d, values2d, stride)
        return lon2d, lat2d, values2d, stride

    def _add_raster_actor(self, layer, field, colormap=None, hillshade=None, stretch=None,
                          surface=None, surface_field=None, drape=None,
                          levels=None) -> str:
        """Build and add this layer's raster drape; return its actor name (NOT yet recorded into
        ``self._layer_actors`` -- :meth:`_add_layer_geometry` commits that only once the whole
        layer, vectors included, has built successfully).

        **Surface source + drape.** ``surface_field`` (a same-grid RasterField, or
        ``None`` = this layer's own values) supplies the 3-D surface HEIGHTS; ``drape`` (a
        same-shape 2-D array, or ``None`` = this layer's own values) supplies the COLOR scalars
        -- e.g. a holder_map result's h(x) draped over the DEM's elevation. Both are decimated
        with the exact stride arithmetic of the grid builders; a shape mismatch falls back to
        the layer's own values (main_window's sync already refuses mismatched sources with a
        notice, so the fallback here is belt-and-braces, never the primary refusal). Hillshade
        composes the right way around: SHADING from the height source, COLORMAP over the drape.
        Known gap, deliberate: ``_surface_ride`` still rides chains on the layer's own values --
        holder results carry no chains, so the drape case never hits it; an other-layer z-source
        under a chain-carrying result will ride at the old heights until that is taught.

        ``colormap`` is the entry's own ``ui.colormap`` preference --
        ``None`` (the pre-Task-8 default, and any entry whose layer has no such preference) falls
        back to ``self._colormap``, the scene-wide value :meth:`set_colormap` sets -- unchanged
        behavior for every caller that predates per-layer colormaps.

        **Frame mode** branches to :meth:`_frame_grid_for` instead
        of :meth:`_lonlat_grid_for` -- same decimation budget/shape (:func:`~dynamix.geo.mapping.
        _stride_for`, reused directly rather than reimplemented), but the coordinate SOURCE is
        ``field.x_axis``/``field.y_axis`` used DIRECTLY as planar coordinates (deliberately
        NOT ``frame.to_scene``, whose GeographicFrame branch applies the world-map
        ``mod 360`` longitude canonicalisation and tears seam-crossing rasters; see
        :meth:`_scene_points`'s own "Frame mode" section for the full rationale -- the two
        placement paths must agree or the drape and its own chains land in different frames);
        the data itself is never resampled either way, only which pixel CENTERS get placed and
        how. ``_lonlat_grid_for``'s own cache is untouched and unconsulted on this path -- see
        the module docstring's "Frame mode" section for why no analogous cache exists here."""
        def _slice_like(arr, ref2d, stride):
            """Decimate a companion array with the grid builders' exact stride arithmetic;
            None on any shape disagreement (caller falls back to the layer's own values)."""
            a = np.asarray(arr)
            if a.ndim == 3:
                a = a[..., 0]
            if a.ndim != 2:
                return None
            rows = np.arange(a.shape[0] // stride, dtype=np.intp) * stride
            cols = np.arange(a.shape[1] // stride, dtype=np.intp) * stride
            out = a[np.ix_(rows, cols)]
            return out if out.shape == np.asarray(ref2d).shape[:2] else None

        if self._frame_mode:
            x2d, y2d, values2d, _stride = self._frame_grid_for(field)
            z2d, color2d = values2d, values2d
            if surface_field is not None:
                cand = _slice_like(surface_field.values, values2d, _stride)
                z2d = cand if cand is not None else z2d
            if drape is not None:
                cand = _slice_like(drape, values2d, _stride)
                color2d = cand if cand is not None else color2d
            if surface_field is None and drape is not None:
                # "Same dataset" heights = THIS layer's own values (picking
                # "this layer's values as height" on an h-map/recon layer stood up the
                # MASTER's DEM): a derived layer's values ARE the drape -- h on a holder
                # layer, elevation-like recon on a recon layer. The DEM-under-h pairing is
                # the Other-dataset pick (the master's field).
                z2d = color2d
            # 3-D surface: z = the height source (sign-flipped for depth-positive data) so a
            # DEM or bathymetry stands up in its own frame; else the flat drape. The Display
            # section's "Vert. exag." (the hillshade z_factor) scales the SURFACE HEIGHTS as
            # well as the shading, per layer, composing with the scene-wide View vexag.
            zf = float(hillshade[3]) if hillshade and len(hillshade) > 3 else 1.0
            z = (self._surface_heights_rebased(z2d, surface).ravel() * zf
                 * _frame_z_scale(getattr(field, "frame", None))
                 if surface and surface[0]
                 else np.zeros(x2d.size, dtype=np.float64))
            pts = np.column_stack([x2d.ravel().astype(np.float64),
                                   y2d.ravel().astype(np.float64), z])
            pts[:, 2] *= self._vexag     # vertical exaggeration composes on the surface height
            ny, nx = x2d.shape
        else:
            lon2d, lat2d, values2d, _stride = self._lonlat_grid_for(layer.layer_id, field)
            z2d, color2d = values2d, values2d
            if surface_field is not None:
                cand = _slice_like(surface_field.values, values2d, _stride)
                z2d = cand if cand is not None else z2d
            if drape is not None:
                cand = _slice_like(drape, values2d, _stride)
                color2d = cand if cand is not None else color2d
            if surface_field is None and drape is not None:
                # Same rule as the frame branch above.
                z2d = color2d
            lon_flat, lat_flat = lon2d.ravel(), lat2d.ravel()
            # height 0 for draping -- an ARRAY of zeros, not a bare python float: project()'s flat
            # modes (greenwich/pacific/mercator) column_stack the height term against lon/lat,
            # which does not broadcast a scalar/0-d array against an N-length one (only "globe"
            # broadcasts, via a plain multiply). Every other project() call site in this codebase
            # already passes a same-shape array for this reason (see tests/test_projection.py).
            # geo modes take height in KM: values are assumed metres (a DEM/bathymetry in m)
            # Same per-layer height exaggeration as the frame branch above.
            zf = float(hillshade[3]) if hillshade and len(hillshade) > 3 else 1.0
            h_km = (self._surface_heights_rebased(z2d, surface).ravel() * zf / 1000.0
                    if surface and surface[0] else np.zeros_like(lon_flat))
            pts = project(lon_flat, lat_flat, h_km, mode=self._mode, vexag=self._vexag)
            ny, nx = lon2d.shape

        grid = pv.StructuredGrid()
        grid.points = pts
        grid.dimensions = [nx, ny, 1]          # x fastest, matching the ravel() order above
        classes01, class_rgba = self._classified(color2d, levels)
        if class_rgba is not None and not (hillshade and hillshade[0]):
            # Explicit slice colors: bake RGBA, same as the hillshade path --
            # a LUT maps one scalar and these colors are the user's own table.
            grid.point_data["rgba"] = (class_rgba.reshape(-1, 4) * 255.0).astype(np.uint8)
            name = self._mesh_actor_name(layer.layer_id)
            self._plotter.add_mesh(grid, scalars="rgba", rgba=True, show_scalar_bar=False,
                                    name=name, reset_camera=False)
            return name
        if classes01 is not None:
            # Density slice: discrete classes replace the continuous stretch --
            # the LUT paints piecewise-constant slices, same field the canvas shows.
            grid.point_data["value"] = classes01.ravel()
        elif stretch and stretch[0] != "linear" and color2d.ndim == 2:
            # Stretch: the drape's scalars are the stretched [0, 1] field, so the
            # LUT (and its later swaps) sees exactly what the canvas shows.
            from dynamix.core.stretch import stretch as _stretch
            grid.point_data["value"] = _stretch(color2d, stretch[0], percent=stretch[1]).ravel()
        else:
            grid.point_data["value"] = color2d.ravel()

        name = self._mesh_actor_name(layer.layer_id)
        self._raster_grids[layer.layer_id] = {
            "grid": grid, "stride": _stride, "dims": (ny, nx),
            "rgba": bool(hillshade and hillshade[0] and color2d.ndim == 2)}
        if hillshade and hillshade[0] and color2d.ndim == 2:
            # Hillshade: colours baked as RGBA -- colormap × shaded relief -- because
            # a LUT maps one scalar and this needs two. Same core function and the same
            # "frame spacing × stride" the canvas uses, so the two surfaces shade identically.
            # Surface source + drape: SHADE from the height source, COLOR the drape.
            grid.point_data["rgba"] = self._shaded_rgba(field, z2d, _stride, colormap, hillshade,
                                                        stretch, color2d=color2d, levels=levels)
            self._plotter.add_mesh(grid, scalars="rgba", rgba=True, show_scalar_bar=False,
                                    name=name, reset_camera=False)
            return name
        self._plotter.add_mesh(grid, scalars="value", cmap=colormap or self._colormap,
                                nan_opacity=0.0, show_scalar_bar=False, name=name,
                                reset_camera=False)
        return name

    def preview_raster_values(self, layer_id, values, src_stride: int = 1) -> bool:
        """Swap one drape's COLOR scalars in place -- the live band-reconstruction tick.
        ``values`` is a (possibly ``src_stride``-decimated) full-field raster;
        nearest-neighbor sampled onto the drape's own grid, scalar range refreshed, one
        render. False (caller notifies instead) when the layer has no plain-LUT drape here --
        unbuilt, or RGBA-baked (hillshade), where a per-tick rebake is not v1."""
        entry = self._raster_grids.get(layer_id)
        if entry is None or entry["rgba"]:
            return False
        try:
            v = np.asarray(values, dtype=np.float64)
            if v.ndim != 2:
                return False
            ny, nx = entry["dims"]
            stride = entry["stride"]
            rows = np.minimum(np.arange(ny) * stride // max(int(src_stride), 1),
                              v.shape[0] - 1)
            cols = np.minimum(np.arange(nx) * stride // max(int(src_stride), 1),
                              v.shape[1] - 1)
            sampled = v[np.ix_(rows, cols)]
            entry["grid"].point_data["value"] = sampled.ravel()
            actor = self._plotter.renderer.actors.get(self._mesh_actor_name(layer_id))
            if actor is not None:
                finite = sampled[np.isfinite(sampled)]
                if finite.size:
                    actor.mapper.scalar_range = (float(finite.min()), float(finite.max()))
            self._plotter.render()
            return True
        except Exception:                                       # noqa: BLE001
            return False

    @staticmethod
    def _classified(color2d, levels):
        """``(classes01, rgba)`` for an active density slice, else ``(None, None)``.
        ``levels`` is the entry's ``(spec_text, colors_text)`` pair; classes01 is
        the [0, 1] class field for the LUT path, rgba a float [0, 1] table-painted image when
        EXPLICIT per-class colors are set (count-matched), else None. Unparseable text falls
        back to the continuous stretch silently -- same posture as the canvas."""
        spec_text = levels[0] if levels else ""
        colors_text = levels[1] if levels and len(levels) > 1 else ""
        if not spec_text or np.asarray(color2d).ndim != 2:
            return None, None
        from dynamix.core.stretch import (classify, parse_class_colors, parse_levels,
                                          resolve_breaks, slice_indices)
        try:
            spec = parse_levels(spec_text)
            if spec is None:
                return None, None
            classes01 = classify(color2d, spec)
            sieve_px = int(levels[2]) if len(levels) > 2 and levels[2] else 0
            if sieve_px > 0:
                from dynamix.core.sieve import sieve_classes
                from dynamix.core.stretch import resolve_breaks
                n_cls = len(resolve_breaks(np.asarray(color2d), spec)) + 1
                classes01 = sieve_classes(classes01, n_cls, sieve_px)
            colors = parse_class_colors(colors_text) if colors_text else None
            if colors is not None:
                idx, n_classes = slice_indices(color2d, resolve_breaks(color2d, spec))
                if len(colors) == n_classes:
                    table = np.asarray(colors, dtype=np.float64) / 255.0   # (n, 4)
                    rgba = np.zeros(np.asarray(color2d).shape + (4,), dtype=np.float64)
                    ok = idx >= 0
                    rgba[ok] = table[idx[ok]]
                    rgba[..., 3] = np.where(ok, rgba[..., 3], 0.0)
                    return classes01, rgba
            return classes01, None
        except (ValueError, TypeError):
            return None, None

    def _shaded_rgba(self, field, values2d, stride, colormap, hillshade, stretch=None,
                     color2d=None, levels=None) -> np.ndarray:
        """``(N, 4)`` uint8 colours for a hillshaded drape: the layer's colormap over the COLOR
        array, multiplied by relief shading from ``values2d`` (the HEIGHT source -- the same
        array when no drape/surface-source is in play);
        NaN cells transparent."""
        import matplotlib
        from dynamix.core.hillshade import hillshade as _hillshade
        if color2d is None:
            color2d = values2d
        _on, azimuth, altitude, z_factor = hillshade
        # A display picture's samples are display_stride native pixels
        # apart -- sample_spacing multiplies it in with the scene's own stride.
        from dynamix.roi.picture import sample_spacing
        dx, dy = sample_spacing(field, stride)
        shade = _hillshade(values2d, dx, dy, azimuth=azimuth, altitude=altitude, z_factor=z_factor)
        from dynamix.core.stretch import stretch as _stretch
        finite = np.isfinite(color2d)
        mode, pct = (stretch[0], stretch[1]) if stretch else ("linear", 2.0)
        classes01, class_rgba = self._classified(color2d, levels)
        if class_rgba is not None:
            # Explicit slice colors x shaded relief.
            class_rgba = class_rgba.copy()
            class_rgba[..., :3] *= np.nan_to_num(shade, nan=0.0)[..., None]
            class_rgba[..., 3] *= np.where(np.isfinite(shade), 1.0, 0.0)
            return (class_rgba.reshape(-1, 4) * 255.0).astype(np.uint8)
        norm = np.nan_to_num(classes01 if classes01 is not None
                             else _stretch(color2d, mode, percent=pct), nan=0.0)
        try:
            cmap = matplotlib.colormaps[colormap or self._colormap]
        except KeyError:
            cmap = matplotlib.colormaps["viridis"]
        rgba = cmap(norm)                                    # (ny, nx, 4) floats in [0, 1]
        rgba[..., :3] *= np.nan_to_num(shade, nan=0.0)[..., None]
        rgba[..., 3] = np.where(finite & np.isfinite(shade), 1.0, 0.0)
        return (rgba.reshape(-1, 4) * 255.0).astype(np.uint8)

    @staticmethod
    def _frame_grid_for(field, max_points: int = 2_000_000):
        """Decimated ``(x2d, y2d, values2d, stride)`` for draping ``field`` in frame mode -- the
        frame-mode sibling of ``dynamix.geo.mapping.field_lonlat_grid``, sharing its own
        :func:`~dynamix.geo.mapping._stride_for` decimation (identical shape/budget arithmetic,
        so the two draping paths pick the identical stride for the identical field) but reading
        ``field.x_axis``/``field.y_axis`` directly rather than running a CRS transform -- there is
        no CRS in frame mode at all. NOT cached the way :meth:`_lonlat_grid_for` is: that cache
        exists specifically because ``field_lonlat_grid``'s ``rasterio`` warp over up to ~2M
        points is the expensive part of a mode-switch rebuild (see ``self._lonlat_cache``'s own
        docstring); plain array indexing here is already cheap enough that a cache would only add
        bookkeeping, not save time."""
        stride = _stride_for((field.ny, field.nx), max_points)
        rows = np.arange(field.ny // stride, dtype=np.intp) * stride
        cols = np.arange(field.nx // stride, dtype=np.intp) * stride
        x2d, y2d = np.meshgrid(field.x_axis[cols], field.y_axis[rows])
        values2d = _mask_sentinels(field.values[np.ix_(rows, cols)])
        return x2d, y2d, values2d, stride

    def _add_points_geometry(self, layer, pointset, color=None) -> None:
        """One point layer's ENTIRE contribution: a single ``pv.PolyData`` points actor, no
        raster, no vector bookkeeping.

        **The display law: surface position, depth DROPPED.** A point layer projects at
        ``(lon, lat, 0)`` -- ``pointset.depth`` (when present) is never consulted here, mirroring
        exactly how ``_add_raster_actor`` above always drapes a raster at height 0 too (no
        elevation dataset exists yet -- see ``self._vexag``'s own docstring). Depth is real data
        (it survives into ``dynamix.devices.backproject``'s ``attrs``-adjacent PointSet field for
        whatever DOES want it, e.g. a future depth-colored point style), but the ARRANGEMENT's own
        placement rule is surface-only, same as every raster already drawn here -- this is not a
        3-D hypocenter plot.

        Unlike :meth:`_add_layer_geometry`, there is no CRS transform at all: ``pointset.lon``/
        ``pointset.lat`` are already WGS84 (``dynamix.core.pointset.read_csv_points``'s own
        contract), so this never calls ``field_lonlat_grid``/``points_lonlat`` and never raises
        :class:`NoGeoreference` -- a point layer has no "no-georeference" status to demote to.

        **Frame mode: raises unconditionally.** A point layer is
        WGS84-placed by contract (the paragraph above) -- there is no frame-mode placement for it
        at all, ever. :meth:`_rebuild`'s own points branch decides this BEFORE ever calling here
        (a dedicated ``"frame-points"`` legend status, for cleaner text than this exception's
        message would format as) -- this guard is defense in depth for any OTHER caller, not the
        path :meth:`_rebuild` itself takes.
        """
        if self._frame_mode:
            raise ValueError("points layers are WGS84-placed; not drawable in a native frame")
        lon = np.asarray(pointset.lon, dtype=np.float64)
        lat = np.asarray(pointset.lat, dtype=np.float64)
        # + vector lift (see _VECTOR_LIFT_KM): point layers sit ON any draped raster's surface,
        # same coplanar depth loss as chains under a top-down view -- same remedy.
        pts = project(lon, lat, np.full_like(lon, _VECTOR_LIFT_KM),
                      mode=self._mode, vexag=self._vexag)

        name = self._points_actor_name(layer.layer_id)
        self._plotter.add_mesh(pv.PolyData(pts), color=color or POINTS_COLOR, style="points",
                                point_size=4.0, name=name, reset_camera=False)
        self._layer_actors[layer.layer_id] = [name]

    @staticmethod
    def _surface_base(v: np.ndarray) -> float:
        """The rebasing datum for a (possibly negated) height array: the finite value in
        ``[min, max]`` closest to 0. Data straddling a datum (bathymetry + topography
        sharing sea level) keeps 0 -- the historical rooting, exactly -- while
        single-signed data (a land DEM) roots at its own near edge, so vertical
        exaggeration stretches RELIEF in place instead of multiplying an absolute offset.
"""
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return 0.0
        lo, hi = float(finite.min()), float(finite.max())
        return min(max(0.0, lo), hi)

    @staticmethod
    def _surface_heights_rebased(values2d, surface) -> np.ndarray:
        """Heights for the 3-D surface, REBASED to :meth:`_surface_base` before any
        exaggeration multiplies them; NaN cells land AT the base (transparent anyway, and
        no phantom z-bounds). Negation (depth-positive data) applies before rebasing."""
        v = np.asarray(values2d, dtype=np.float64)
        if v.ndim == 3:
            v = v[..., 0]
        if surface and len(surface) > 1 and surface[1]:
            v = -v
        return np.nan_to_num(v - Scene._surface_base(v), nan=0.0)

    @staticmethod
    def _surface_heights(values2d, surface) -> np.ndarray:
        """Heights for the 3-D surface option: the value (first component of a multi-component
        field), NaN -> 0 (those cells are transparent anyway), negated for depth-positive data
        so land and seafloor share sea level = 0. SUPERSEDED for the mesh and the ride by
        :meth:`_surface_heights_rebased`; kept as the absolute-height reading."""
        v = np.asarray(values2d, dtype=np.float64)
        if v.ndim == 3:
            v = v[..., 0]
        h = np.nan_to_num(v, nan=0.0)
        return -h if (surface and len(surface) > 1 and surface[1]) else h

    def _surface_ride(self, field, cols, rows):
        """Per-point surface height for a layer shown as a 3-D surface (frame-mode units
        follow :func:`_frame_z_scale` -- degree-equivalents over a geographic frame, native
        units otherwise; km in geo modes), so chains, extrema and H-lines sit ON the
        surface instead of under it. ``None`` when the layer is flat."""
        negate = self._surface_fields.get(id(field))
        if negate is None:
            return None
        vals = np.asarray(field.values, dtype=np.float64)
        if vals.ndim == 3:
            vals = vals[..., 0]
        v = -vals if negate else vals
        # Same rebasing datum as the mesh -- computed over the WHOLE field,
        # so chains sit ON the rebased surface instead of floating a base-level above it.
        base = self._surface_base(v)
        h = np.nan_to_num(v[np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)]
                          - base, nan=0.0)
        if self._frame_mode:
            return h * _frame_z_scale(getattr(field, "frame", None))
        return h / 1000.0

    def _scene_points(self, field, cols, rows, height=None, sub=None) -> np.ndarray:
        """One placement seam for pixel-index arrays ``cols``/``rows`` (parallel integer arrays --
        the same ``(field, cols, rows)`` shape ``dynamix.geo.mapping.points_lonlat`` takes) ->
        ``(N, 3)`` float64 scene coordinates, GEO or FRAME mode per ``self._frame_mode``. Every vector-actor call site that used to inline
        ``points_lonlat(...)`` + :func:`~dynamix.core.projection.project` directly now goes
        through here instead: :meth:`_build_chain_geometry`, the extrema block of
        :meth:`_build_vector_actors`, and :meth:`_build_roi_actor`.

        **Frame mode:** ``field.x_axis[cols]``/``field.y_axis[rows]`` -- the identical pixel-center
        lookup ``points_lonlat`` performs before its own CRS transform -- used DIRECTLY as planar
        scene coordinates. Deliberately NOT ``field.frame.to_scene``: for a
        ``LocalFrame`` that call is a documented pass-through (identical to planar), but for a
        ``GeographicFrame`` it routes through ``projection.project``'s WORLD-MAP longitude
        canonicalisation (``mod 360``), which tears any raster whose lon range crosses the active
        mode's seam (the demo DEM, lon -30..+12.5 under "pacific", split into two edge strips with
        the bridging cells smeared across the whole width). Native-frame display is seam-free by
        doctrine -- the field's own axes are already continuous, and the Vector tab shows the
        dataset in ITS coordinates, not on a world map. No :class:`NoGeoreference` can arise on
        this branch -- it never calls anything in ``geo.mapping``. ``z`` defaults to zero;
        multiplied by ``self._vexag`` anyway (a no-op today) purely so a future frame that DOES
        carry a real ``z`` stays plumbed through vexag exactly like every geo-mode placement
        already is (mirrors ``self._vexag``'s own docstring rationale).

        **Geo mode:** unchanged -- ``points_lonlat(field, cols, rows)`` + ``project(...,
        mode=self._mode, vexag=self._vexag)``, exactly what every call site above inlined before
        this helper existed.

        **``height``: the scale-space z-lift.**
        ``None`` (every pre-Task-6 call site, and extrema/ROI today) is EXACTLY the pre-Task-6
        behaviour -- zeros, byte-identical. A real array (only ever passed by
        :meth:`_build_chain_geometry` in scale-space mode) is threaded in as the height BEFORE
        ``self._vexag``'s own multiply -- geo mode via ``project(lon, lat, height, ...)``'s own
        ``height_km * vexag`` composition, frame mode via ``frame.to_scene(x, y, height)`` then
        the identical ``pts[:, 2] *= self._vexag`` line every other frame-mode placement already
        runs -- so vertical exaggeration composes as an ADDITIONAL stretch on top of the lift in
        BOTH modes, via the exact same code path every other height already uses (see the module
        docstring's "Composition with vertical exaggeration" section for why this is a
        deliberate divergence from EQSelect's own choice to bypass vexag in scale-space mode).

        **``sub``:** optional ``(cols_f, rows_f)`` -- a maximum's SUBPIXEL position
        (``x_sub``/``y_sub``), used to PLACE the point (linear along the axes); every
        index-based lookup (the surface ride) stays on the integer ``cols``/``rows``. Without
        it an H-line follows the pixel staircase."""
        if self._frame_mode:
            cols = np.asarray(cols, dtype=np.intp)
            rows = np.asarray(rows, dtype=np.intp)
            if sub is not None:
                x = axis_at(field.x_axis, sub[0])
                y = axis_at(field.y_axis, sub[1])
            else:
                x = np.asarray(field.x_axis, dtype=np.float64)[cols]
                y = np.asarray(field.y_axis, dtype=np.float64)[rows]
            z = (np.asarray(height, dtype=np.float64) if height is not None
                 else np.zeros_like(x))
            ride = self._surface_ride(field, cols, rows)     # 3-D surface: sit on it
            if ride is not None:
                z = z + ride
            pts = np.column_stack([x, y, z])
            pts[:, 2] *= self._vexag
            # Vector lift (see _VECTOR_LIFT_FRAC): ADDED after the vexag multiply so the
            # separation is constant regardless of exaggeration, and after the scale-space
            # z-lift so a lifted cone rides the same epsilon above the drape as flat chains do.
            pts[:, 2] += self._frame_vector_lift(field)
            return pts
        lon, lat = points_lonlat(field, cols, rows, sub=sub)
        h = np.asarray(height, dtype=np.float64) if height is not None else np.zeros_like(lon)
        ride = self._surface_ride(field, cols, rows)         # 3-D surface: sit on it (km)
        if ride is not None:
            h = h + ride
        # Vector lift, geo half (see _VECTOR_LIFT_KM): through the existing height plumbing --
        # radial on the globe, a small z on the flat modes; vexag scales it like every other
        # height, which stays invisible top-down at any slider value.
        return project(lon, lat, h + _VECTOR_LIFT_KM, mode=self._mode, vexag=self._vexag)

    @staticmethod
    def _frame_vector_lift(field) -> float:
        """Frame-mode vector lift in the field's own data units: ``_VECTOR_LIFT_FRAC`` of the
        larger axis span, so the depth separation is proportionally identical for a 42-degree
        DEM and a 160k-foot UTM window. Falls back to the fraction itself for a degenerate
        (single-sample or missing) axis -- any nonzero beats coplanar."""
        try:
            x_axis, y_axis = field.x_axis, field.y_axis
            span = max(abs(float(x_axis[-1]) - float(x_axis[0])),
                       abs(float(y_axis[-1]) - float(y_axis[0])))
        except Exception:
            span = 0.0
        return _VECTOR_LIFT_FRAC * span if span > 0 else _VECTOR_LIFT_FRAC

    def _hline_actors_from_base(self, layer_id, field, result, ext0, k, ex, ey, names) -> bool:
        """H-line + singleton actors from the BASE layer's cached ordering: ``result["_ext_base"]``
        is the id-stable unfiltered layer (ScaleSelect's stamp), so the ``hline_runs`` walk and
        the ``_scene_points`` projection are paid once per computed stack; every filter tweak
        after that keeps exactly the segments whose BOTH endpoints survived -- an O(n)
        membership test, the scene half of the canvas's ``_masked_hline_geometry``. A kept point
        stripped of both neighbours falls back to a dot (the singleton actor), matching what a
        re-walk would produce. Returns False (build nothing) when there is no base or the
        placement inputs changed -- caller runs the original walk."""
        base = result.get("_ext_base")
        if base is None:
            return False
        key = (id(base), id(field), self._frame_mode, self._mode, float(self._vexag),
               self._surface_fields.get(id(field)))
        cached = self._hline_base_cache.get(layer_id)
        if cached is None or cached[0] != key:
            bx = np.asarray(base["x"], dtype=np.int64)
            by = np.asarray(base["y"], dtype=np.int64)
            runs = result.get("_ext_base_runs")
            if runs is None:
                runs = hline_runs(base, (field.ny, field.nx))   # old cached results: walk here
            if not runs:
                return False
            order = np.concatenate(runs)
            run_id = np.repeat(np.arange(len(runs)), [len(r) for r in runs])
            pts = self._scene_points(field, bx[order], by[order],
                                     sub=_subpixel_of(base, order))
            vert_pos = by[order] * int(field.nx) + bx[order]
            cached = (key, pts, vert_pos, run_id, order)
            self._hline_base_cache[layer_id] = cached
        _key, pts, vert_pos, run_id, order = cached
        sel = selection_of(result)
        prod = result.get("chain_product") if isinstance(result, dict) else None
        if sel is not None and prod is not None:
            # Live selection: the SAME per-layer mask the materializer
            # lands, scattered straight onto the base ordering -- no per-tweak position sort or
            # searchsorted over millions of vertices.
            si = int(result.get("_scale_idx", 0))
            keep = layer_point_keep(prod, sel, si, len(np.asarray(base["x"])))[order]
        else:
            # TRUE O(n) membership, mirroring the canvas: an np.sort+searchsorted test would be
            # O(n log n) (~144 ms on a 1M-point DEM finest scale), the vector view's half of the
            # lag while scrubbing. Scatter the filtered points into a boolean raster grid, gather
            # by the base ordering's flat positions.
            grid = np.zeros(int(field.ny) * int(field.nx), dtype=bool)
            grid[np.asarray(ey[:k], np.int64) * int(field.nx)
                 + np.asarray(ex[:k], np.int64)] = True
            keep = grid[vert_pos]
        pair = keep[:-1] & keep[1:] & (run_id[:-1] == run_id[1:])
        m = int(np.count_nonzero(pair))
        if m:
            i = np.flatnonzero(pair)
            cells = np.empty(m * 3, dtype=np.int64)
            cells[0::3] = 2; cells[1::3] = i; cells[2::3] = i + 1
            hname = self._hlines_actor_name(layer_id)
            self._plotter.add_mesh(pv.PolyData(pts, lines=cells), color=HLINE_COLOR,
                                   line_width=1.5, name=hname, reset_camera=False)
            names.append(hname)
        # Dots: true singletons of the FILTERED layer, plus kept points isolated by the filter.
        has_nb = np.zeros(keep.size, dtype=bool)
        if keep.size > 1:
            has_nb[:-1] |= pair
            has_nb[1:] |= pair
        lonely = keep & ~has_nb
        lid0 = np.asarray(ext0.get("line_id", np.full(k, -1)))[:k]
        dot_pts = []
        iso = np.flatnonzero(lid0 == -1)
        if iso.size:
            dot_pts.append(self._scene_points(field, ex[iso], ey[iso],
                                              sub=_subpixel_of(ext0, iso)))
        if lonely.any():
            dot_pts.append(pts[lonely])
        if dot_pts:
            name = self._extrema_actor_name(layer_id)
            self._plotter.add_mesh(pv.PolyData(np.concatenate(dot_pts)), color=EXTREMA_COLOR,
                                   style="points", point_size=4.0, name=name, reset_camera=False)
            names.append(name)
        return True

    def _build_vector_actors(self, layer_id, field, result, names: list[str], vtrail_color=None):
        """One layer's chains/extrema/ROI-outline actors, added to the plotter. ``names`` is the
        CALLER's own list (:meth:`_add_layer_geometry`) -- appended to directly, in place, as each
        actor is genuinely added, so a failure partway through still leaves the caller holding
        every name that landed before the failure (see that method's docstring for why this
        matters: it is what makes the rollback on a mid-build exception complete rather than
        partial). Returns ``(chain_lookup_entry, vector_mask_entry)`` -- either is ``None`` when
        there were no drawable chains; NEITHER is committed into ``self._chain_lookup``/
        ``self._vector_masks`` here -- the caller does that only once this whole call returns
        without raising.

        ``vtrail_color`` is the entry's own ``ui.color_vtrail``
        preference, threaded straight through to :meth:`_build_chain_geometry` -- see
        :func:`_chain_color`'s own docstring for its fallback.

        ``n_voice`` is read off
        ``result["params"]["n_voice"]`` -- the WTMM run's own resolved dyadic voice count
        (``wtmm_backend._run_wtmm2d_scalar``/``_tensor``'s own ``"params": resolved`` key) --
        and passed straight through to :meth:`_build_chain_geometry` as the PREFERRED source for
        :meth:`_expected_scale_step`'s "one voice-step" reference, per the design
        ("n_voice metadata if present, else the median step"). ``None`` (a result with no
        ``params``, or no ``n_voice`` in it -- e.g. a synthetic test result, or a future
        non-wtmm_backend bundle) falls back to that method's own median-step derivation."""
        chain_lookup_entry = None
        vector_mask_entry = None

        chains = result.get("chains") or []
        # Bounded draw (EQSelect's _TE_MAX): stubs, never renumbering
        # -- chain identity here is list position, the picking lookup's own contract. The honest
        # "drawing X of Y" reading rides the canvas/status note, which shares these counts.
        chains, _cap_note = stub_capped_chains(chains)
        n_voice = (result.get("params") or {}).get("n_voice")
        geometry = self._build_chain_geometry(field, chains, vtrail_color, n_voice=n_voice)
        if geometry is not None:
            poly, starts, chain_indices, max_log2_mod, hi_scale_idx, base_rgba = geometry
            name = self._chains_actor_name(layer_id)
            poly.point_data["colors"] = base_rgba
            self._plotter.add_mesh(poly, scalars="colors", rgba=True, name=name,
                                    line_width=1.0, reset_camera=False)
            names.append(name)
            chain_lookup_entry = (starts, chain_indices)
            vector_mask_entry = _VectorMaskState(chains_actor_name=name,
                                                  max_log2_mod=max_log2_mod,
                                                  hi_scale_idx=hi_scale_idx, base_rgba=base_rgba)

        extrema_layers = result.get("extrema") or []
        if extrema_layers:
            ext0 = extrema_layers[0]     # finest scale only -- see the module docstring
            ex = np.asarray(ext0.get("x", ()), dtype=np.intp)
            ey = np.asarray(ext0.get("y", ()), dtype=np.intp)
            k = min(len(ex), len(ey))
            if k > 0:
                # H-lines as polylines: the same points joined along each labelled
                # line, the way the 2-D canvas has always drawn them. One actor, one cell per
                # contiguous run. Points that sit on a line are drawn AS the line; only the
                # singletons keep a dot (4 px of grey per point would otherwise bury 1.5 px lines).
                built = self._hline_actors_from_base(layer_id, field, result, ext0, k, ex, ey, names)
                if built:
                    runs = None
                else:
                    runs = hline_runs(ext0, (field.ny, field.nx))
                on_line = np.zeros(k, dtype=bool)
                for run in (runs or ()):
                    on_line[run[run < k]] = True
                dot_idx = np.flatnonzero(~on_line) if not built else np.array([], dtype=np.int64)
                if dot_idx.size:
                    pts = self._scene_points(field, ex[dot_idx], ey[dot_idx],
                                             sub=_subpixel_of(ext0, dot_idx))
                    name = self._extrema_actor_name(layer_id)
                    self._plotter.add_mesh(pv.PolyData(pts), color=EXTREMA_COLOR, style="points",
                                            point_size=4.0, name=name, reset_camera=False)
                    names.append(name)
                if runs:
                    cells, offset = [], 0  # slow path: no _ext_base to reuse (old cached results)
                    for run in runs:
                        cells.extend([len(run), *range(offset, offset + len(run))])
                        offset += len(run)
                    order = np.concatenate(runs)
                    hpts = self._scene_points(field, ex[order], ey[order],
                                              sub=_subpixel_of(ext0, order))
                    hname = self._hlines_actor_name(layer_id)
                    self._plotter.add_mesh(pv.PolyData(hpts, lines=np.array(cells, dtype=np.int64)),
                                           color=HLINE_COLOR, line_width=1.5, name=hname,
                                           reset_camera=False)
                    names.append(hname)

        roi = (result.get("_roi") or {}).get("roi")
        if roi is not None:
            roi_name = self._build_roi_actor(layer_id, field, roi)
            if roi_name is not None:
                names.append(roi_name)

        return chain_lookup_entry, vector_mask_entry

    def _build_chain_geometry(self, field, chains, vtrail_color=None, *, n_voice=None):
        """Combine every chain in ``chains`` (``result["chains"]``, ORIGINAL list order -- chain
        identity = index into this list) into ONE ``pv.PolyData`` of line cells, one polyline per
        chain with >= 2 points (or, in scale-space mode, one polyline per GAP-FREE RUN within a
        chain -- see below). Returns ``None`` when not one point qualifies -- nothing to draw.

        Returns ``(poly, starts, chain_indices, max_log2_mod, hi_scale_idx, base_rgba)`` -- see
        the module docstring ("Vector bookkeeping") for ``starts``/``chain_indices``, and
        :class:`_VectorMaskState` for ``max_log2_mod``/``hi_scale_idx``/``base_rgba``. All three
        of the latter are computed for EVERY chain in ``chains`` (not just the drawable ones,
        though a <2-point chain's own value is inert: it owns no points, so it can never appear
        in ``chain_indices`` and therefore never affects anything :meth:`Scene._apply_mask_to_layer`
        does with it) -- keeping the arrays at ``len(chains)`` lets both be indexed directly by
        the SAME original chain index ``chain_indices`` uses, with no remapping step.

        ``vtrail_color`` passes straight through to :func:`_chain_color`
        as its plain-chain fallback -- see that function's own docstring. ``n_voice`` is the result's own resolved WTMM voice count, threaded
        through by :meth:`_build_vector_actors` -- see :meth:`_expected_scale_step`'s own
        docstring.

        **Scale-space mode (``self._scale_space_enabled``).** Byte-identical to the
        paragraph above when disabled (every run is simply ``[np.arange(k)]``, the whole chain,
        exactly as before this task). When enabled:

        1. Each chain's own per-point log2 scale (:meth:`_chain_log2_scales`) is split into
           gap-free RUNS (:meth:`_split_scale_runs`) -- a run shorter than 2 points draws
           nothing, the identical "<2 points, no line" rule the disabled path already applies
           per-chain, now applied per-run.
        2. ``a_min`` (the module docstring's own derivation) is the minimum log2-scale value
           across every point that ends up DRAWN (i.e. survives step 1) anywhere in this layer.
        3. Each drawn point's z-lift is ``stretch * (its own log2 scale - a_min)``, passed to
           :meth:`_scene_points` as ``height`` -- composed with ``self._vexag`` there (module
           docstring's "Composition with vertical exaggeration" section).
        4. Per-point color is either ``self._color_by_scale``'s discrete fine->coarse ranking
           (:meth:`_scale_space_colors`) or, when that flag is off, the ordinary per-CHAIN
           :func:`_chain_color` -- tiled across every point of every run belonging to that chain,
           exactly as the disabled path already does.
        """
        n_chains = len(chains)
        starts = np.zeros(n_chains + 1, dtype=np.int64)
        max_log2_mod = np.full(n_chains, -np.inf, dtype=np.float64)
        hi_scale_idx = np.zeros(n_chains, dtype=np.int64)

        scale_space_on = self._scale_space_enabled
        expected_step = self._expected_scale_step(chains, n_voice) if scale_space_on else None

        cols_parts, rows_parts, log2a_parts, lines_cells = [], [], [], []
        run_lengths: list[int] = []      # parallel to cols_parts -- len of each contributed run
        run_chain_idx: list[int] = []    # which original chain index each run belongs to
        offset = 0
        for i, chain in enumerate(chains):
            x = np.asarray(chain.get("x", ()), dtype=np.intp)
            y = np.asarray(chain.get("y", ()), dtype=np.intp)
            k = min(len(x), len(y))
            log2_mod = chain.get("log2_mod", ())
            if len(log2_mod):
                max_log2_mod[i] = float(np.max(log2_mod))
            hi_scale_idx[i] = max(k - 1, 0)
            if k >= 2:
                if scale_space_on:
                    chain_log2a = self._chain_log2_scales(chain, k)
                    runs = self._split_scale_runs(chain_log2a, expected_step)
                else:
                    chain_log2a = None
                    runs = [np.arange(k)]
                for run in runs:
                    rk = len(run)
                    if rk < 2:
                        continue    # an isolated point after a gap split draws no segment --
                                    # same rule as a whole chain with < 2 points, applied per-run
                    cols_parts.append(x[run])
                    rows_parts.append(y[run])
                    log2a_parts.append(chain_log2a[run] if scale_space_on else None)
                    run_lengths.append(rk)
                    run_chain_idx.append(i)
                    lines_cells.append(rk)
                    lines_cells.extend(range(offset, offset + rk))
                    offset += rk
            starts[i + 1] = offset

        if offset == 0:
            return None

        cols = np.concatenate(cols_parts)
        rows = np.concatenate(rows_parts)

        height = None
        all_log2a = None
        if scale_space_on:
            all_log2a = np.concatenate(log2a_parts)
            a_min_log2 = float(np.min(all_log2a))
            height = self._scale_space_stretch * (all_log2a - a_min_log2)

        pts = self._scene_points(field, cols, rows, height=height)

        if scale_space_on and self._color_by_scale:
            rgb = self._scale_space_colors(all_log2a)
        else:
            rgb_chunks = []
            for run_i, chain_i in enumerate(run_chain_idx):
                rk = run_lengths[run_i]
                rgb_chunks.append(np.tile(np.array(_chain_color(chains[chain_i], vtrail_color),
                                                    dtype=np.uint8), (rk, 1)))
            rgb = np.concatenate(rgb_chunks, axis=0)

        poly = pv.PolyData(pts, lines=np.array(lines_cells, dtype=np.int64))
        chain_indices = np.repeat(np.arange(n_chains, dtype=np.int64), np.diff(starts))
        base_rgba = np.concatenate([rgb, np.full((offset, 1), 255, dtype=np.uint8)], axis=1)
        return poly, starts, chain_indices, max_log2_mod, hi_scale_idx, base_rgba

    @staticmethod
    def _chain_log2_scales(chain: dict, k: int) -> np.ndarray:
        """Per-point log2 scale for one chain's first ``k`` points -- the scale-space z-lift's
        own data source.

        **This task's own FIRST-STEP verification** (mandatory before any z-lift code, per the design; recorded in full in the task-6 report): ``dynamix.core.wtmm_backend``'s
        ``chains2d`` (the live compute path) and ``_arrays_to_chains`` (the cache/CSR round-trip
        path -- both feed the SAME ``Scene`` this method lives on) both stamp a REAL
        ``log2_scales`` array onto every chain -- ``np.log2(scales)[:k]``, the ACTUAL physical
        log2 scale values the wavelet transform used, index 0 = finest. This is NOT an implicit
        index range, and it is NOT the xsmurf greedy-chain bug CW§1 warns about (``chaining.py``'s
        ``chain_scales = scales[:n]`` assumes contiguity it does not itself guarantee) --
        ``chains2d``'s own vectorized walk GUARANTEES every kept chain is a finest-anchored,
        gap-free PREFIX of the scale ladder (its own comment: "Contiguous finest-anchored chains
        -> valid entries are a row-wise prefix" -- the same fact ``set_mask``'s docstring already
        cites for ``hi_scale_idx``), so slicing ``log2_scales_full[:k]`` is exactly correct here,
        never an approximation, for every chain this app's own backend ever produces. A populated
        ``mz_edges`` bundle would need this too, but ``devices/mz_edges.py`` stamps
        ``bundle["chains"] = []`` deliberately (cross-scale V-chains are reserved future work) --
        so today, every chain this ``Scene`` ever actually draws is a ``wtmm_backend`` chain.

        Falls back to plain point INDEX (``np.arange(k)``, a documented, degraded-mode caveat --
        CW§1 and the design's instruction) only when a chain carries no ``log2_scales``
        at all, or a shorter one than ``k`` -- e.g. a hand-built chain dict from some future
        non-wtmm_backend source. Giving THAT case real physical scale values is recorded as a
        follow-up in the task-6 report, not solved here.
        """
        log2_scales = chain.get("log2_scales")
        if log2_scales is not None and len(log2_scales) >= k:
            return np.asarray(log2_scales[:k], dtype=np.float64)
        return np.arange(k, dtype=np.float64)

    @staticmethod
    def _expected_scale_step(chains, n_voice):
        """The layer-wide "one voice-step" reference :meth:`_split_scale_runs` gaps against.

        ``1.0 / n_voice`` when ``n_voice`` is a valid positive number -- ``wtmm_backend.
        compute_scales2d``'s own ladder (``scales[o, v] = a_min * 2**(o + v/n_voice) * norm``,
        flattened octave-major/voice-minor) is EVENLY spaced at exactly this step in log2 space
        at every single consecutive pair, including across an octave boundary (the exponent
        sequence is ``0, 1/n_voice, 2/n_voice, ..., 1, 1+1/n_voice, ...`` -- a plain arithmetic
        progression), so this is not an approximation when the real ladder is known.

        Else the MEDIAN point-to-point log2-scale step, pooled across every chain in this layer
        that carries at least 2 real (non-fallback) ``log2_scales`` values -- the design's documented fallback for a chain source with no ``n_voice`` metadata at all (e.g. a
        synthetic test chain list passed directly to this ``Scene``, with no wrapping ``result``
        to read ``params`` off of).

        ``None`` when neither is derivable (no usable ``n_voice`` AND no chain has 2+ real scale
        points) -- :meth:`_split_scale_runs` treats that as "never split": a layer with no way to
        know what "one step" even means cannot honestly call anything else "a gap".
        """
        if n_voice:
            try:
                n_voice_f = float(n_voice)
            except (TypeError, ValueError):
                n_voice_f = 0.0
            if n_voice_f > 0:
                return 1.0 / n_voice_f
        diffs = []
        for chain in chains:
            log2_scales = chain.get("log2_scales")
            if log2_scales is not None and len(log2_scales) >= 2:
                diffs.append(np.diff(np.asarray(log2_scales, dtype=np.float64)))
        if not diffs:
            return None
        return float(np.median(np.concatenate(diffs)))

    @staticmethod
    def _split_scale_runs(log2a: np.ndarray, expected_step) -> list[np.ndarray]:
        """Split one chain's own point-index range ``[0, len(log2a))`` into contiguous runs at
        every MISSING-SCALE gap (EQSelect's own
        ``_split_scale_runs`` precedent: "a dropped scale shows as an honest gap,
        never a fake straight link"). Returns a list of int index arrays (local to ``log2a``,
        i.e. usable directly as ``x[run]``/``y[run]``/``log2a[run]``), covering every index
        exactly once, in order.

        A gap is declared between point ``j`` and ``j + 1`` when their own log2-scale STEP
        exceeds :data:`_SCALE_SPACE_GAP_FACTOR` times ``expected_step`` -- comfortably between
        "exactly one voice-step" (an ungapped link) and "skipped at least one" (two steps), so
        floating-point noise around a clean match never trips it.

        ``expected_step is None`` (no chain in this layer carried >= 2 usable scale points to
        derive one from -- :meth:`_expected_scale_step`) or ``len(log2a) < 2`` never splits
        anything: one run, the whole point range -- the same "can't tell, so don't guess" posture
        the rest of this module already takes for degraded input.

        Every DynamiX ``wtmm_backend`` chain today is gap-free by construction
        (:meth:`_chain_log2_scales`'s own finest-anchored-prefix guarantee), so in practice this
        only ever fires against a hand-built/synthetic chain -- it exists for the honest-gap
        contract regardless, per the design and the EQSelect precedent above.
        """
        n = len(log2a)
        if expected_step is None or n < 2:
            return [np.arange(n)]
        diffs = np.diff(log2a)
        gap_after = np.nonzero(diffs > _SCALE_SPACE_GAP_FACTOR * expected_step)[0]
        if gap_after.size == 0:
            return [np.arange(n)]
        runs = []
        start = 0
        for j in gap_after:
            runs.append(np.arange(start, j + 1))
            start = j + 1
        runs.append(np.arange(start, n))
        return runs

    def _scale_space_colors(self, log2a: np.ndarray) -> np.ndarray:
        """Per-point RGB for the scale-space discrete fine->coarse colormap ("coloured fine->coarse by a discrete colormap with
        n_colors = n_scales"). ``n_scales`` is the count of DISTINCT log2-scale values actually
        present across every point being DRAWN in this layer right now (rounded to 9 decimals to
        absorb float noise) -- not a separately-threaded ``result['scales']`` length, so this
        stays correct for a synthetic/test chain set too, and for a real WTMM layer it is
        numerically the same count (every distinct rung of the ladder a kept, drawable chain
        actually reaches). Rank 0 (finest) maps to the colormap's own first sampled entry; rank
        ``n_scales - 1`` (coarsest) to its last -- ``pv.LookupTable``'s own ``cmap``/``n_values``
        discrete sampling, the same colormap-name -> RGBA idiom :meth:`set_colormap` already uses
        elsewhere in this module (no new third-party import introduced here)."""
        rounded = np.round(log2a, 9)
        uniq = np.unique(rounded)
        n_colors = max(len(uniq), 1)
        lut = pv.LookupTable(cmap=_SCALE_SPACE_CMAP, n_values=n_colors)
        palette = np.asarray(lut.values, dtype=np.uint8)[:, :3]     # (n_colors, 3), rank order
        rank = np.searchsorted(uniq, rounded)
        return palette[rank]

    # -- footprints (data browser) ---------------------------------------------------------
    def set_footprints(self, footprints) -> None:
        """Outline every :class:`~dynamix.geo.footprints.Footprint` on the world -- rasters the
        user has NOT loaded, drawn from header metadata alone so a folder of 37 ASTER tiles
        appears in milliseconds. One actor for all of them (``_FOOTPRINT_NAME``), re-projected
        by :meth:`_rebuild` on every mode switch; hidden in frame mode, which has no CRS to place
        a lon/lat quadrilateral in. An empty list removes the actor. Fits the camera to the
        footprints only when nothing else is on the scene -- a scan onto an empty world must be
        visible, a scan over loaded layers must not move the user's viewpoint."""
        self._footprints = list(footprints)
        self._draw_footprints()
        if self._footprints and not self._entries:
            self._plotter.reset_camera()
        self._plotter.render()

    def _footprint_scene_points(self, fp) -> np.ndarray:
        lon = np.array([c[0] for c in fp.corners], dtype=np.float64)
        lat = np.array([c[1] for c in fp.corners], dtype=np.float64)
        # lifted like the vector actors so an outline never z-fights a draped raster's surface
        return project(lon, lat, np.full_like(lon, _VECTOR_LIFT_KM), mode=self._mode,
                       vexag=self._vexag)

    def _draw_footprints(self) -> None:
        self._plotter.remove_actor(_FOOTPRINT_NAME, render=False)
        if not self._footprints or self._frame_mode:
            return
        pts, lines = [], []
        for members in self._footprint_groups().values():
            # one loop per GROUP -- an ASTER granule's 5-13 band files trace the same
            # quadrilateral to within metres; drawing each was 7 790 loops for 1 500 scenes
            base = 4 * len(pts)                       # four corners per loop so far
            q = self._footprint_scene_points(members[0])
            pts.append(q)
            lines.extend([5, base, base + 1, base + 2, base + 3, base])
        poly = pv.PolyData(np.vstack(pts), lines=np.array(lines, dtype=np.int64))
        self._plotter.add_mesh(poly, color=FOOTPRINT_COLOR, line_width=1.5, name=_FOOTPRINT_NAME,
                               reset_camera=False)

    def footprints_at_screen(self, x_px: float, y_px: float, viewport, mvp=None) -> list:
        """Every footprint whose on-screen quadrilateral contains the pixel -- the ONE picking
        math (``core.selection``: :func:`project_points` + :func:`points_in_polygon`) over the
        same corner points :meth:`_draw_footprints` placed, so what is drawn is what is hit.
        Smallest screen area first (the most specific dataset on top), then name -- ASTER's
        ``_dem`` before its ``_num``. Empty in frame mode, off-screen, or on a miss."""
        if not self._footprints or self._frame_mode:
            return []
        if mvp is None:
            mvp = self._camera_mvp(viewport)
        if mvp is None:
            return []
        hits = []
        pixel = np.array([[float(x_px), float(y_px)]])
        cam_pos = np.array(self._plotter.camera.position, dtype=np.float64)
        for members in self._footprint_groups().values():
            fp = members[0]                   # the group's drawn representative
            pts3d = self._footprint_scene_points(fp)
            if self._mode == "globe" and float(np.dot(pts3d.mean(axis=0), cam_pos)) <= 0:
                continue                      # far side of the globe, same rule as _pick_candidates
            px, visible = project_points(pts3d, mvp, viewport)
            if not np.all(visible):
                continue
            if points_in_polygon(pixel, px)[0]:
                x, y = px[:, 0], px[:, 1]
                area = abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0
                hits.append((area, fp.name, members))
        out = []
        for _area, _name, members in sorted(hits, key=lambda h: (h[0], h[1])):
            out.extend(members)            # already in band order (_footprint_groups)
        return out

    def _footprint_groups(self) -> dict:
        """Footprints by :func:`dynamix.geo.footprints.group_key`, insertion-ordered, members
        by name -- the unit the world draws and a right-click hits."""
        groups: dict = {}
        for fp in self._footprints:
            groups.setdefault(_footprint_group_key(fp), []).append(fp)
        for members in groups.values():
            members.sort(key=lambda m: _band_sort_key(m.name))
        return groups

    # -- reference layers ------------------------------------------------------------------
    def set_reference_layers(self, entries) -> None:
        """Draw GIS reference layers (``dynamix.geo.vectors``) on the world: one actor per layer.
        ``entries`` are dicts ``{ref_id, name, kind, color, visible, lonlat: [[(N,2) lon/lat
        arrays...], ...], native: same shape in the active field's CRS units or None}``. Geo
        modes place ``lonlat`` through :func:`project`; frame mode needs ``native`` (the Vector
        view has no CRS of its own) and skips a layer without it. Lifted like every vector actor
        so an outline never z-fights the drape; re-projected by :meth:`_rebuild`."""
        self._reference_layers = list(entries)
        self._draw_reference_layers()
        self._plotter.render()

    def set_reference_visible(self, ref_id: str, visible: bool) -> None:
        for e in self._reference_layers:
            if e["ref_id"] == ref_id:
                e["visible"] = bool(visible)
        actor = self._plotter.renderer.actors.get(f"{_REFERENCE_PREFIX}{ref_id}")
        if actor is not None:
            actor.SetVisibility(bool(visible))
            self._plotter.render()

    def zoom_to_reference(self, ref_id: str) -> None:
        """Fit the camera to one reference layer's actor (its FULL extent, raster or not)."""
        actor = self._plotter.renderer.actors.get(f"{_REFERENCE_PREFIX}{ref_id}")
        if actor is None:
            return
        self._plotter.reset_camera(bounds=actor.GetBounds())
        self._plotter.render()

    def _reference_ride(self, pixels, parts):
        """Per-vertex surface height (frame units in frame mode, km in geo modes) for a reference
        layer from its ``pixels`` -- the same vertices in the ACTIVE field's pixel frame -- or
        ``None`` when no layer is shown as a surface, the entry carries no pixels, or the two
        vertex lists do not line up. Vertices outside the raster ride nothing."""
        if not pixels:
            return None
        fields = [e["field"] for e in self._entries if id(e.get("field")) in self._surface_fields]
        if not fields:
            return None
        field = fields[0]
        pix = [np.asarray(q, dtype=np.float64) for qs in pixels for q in qs if len(q)]
        if len(pix) != len(parts) or any(len(a) != len(b) for a, b in zip(pix, parts)):
            return None
        allpix = np.concatenate(pix)
        ny, nx = np.asarray(field.values).shape[:2]
        cols = np.rint(allpix[:, 0]).astype(np.intp)
        rows = np.rint(allpix[:, 1]).astype(np.intp)
        inside = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
        ride = np.zeros(len(allpix))
        if inside.any():
            ride[inside] = self._surface_ride(field, cols[inside], rows[inside])
        return ride

    def _remove_reference_layers(self) -> None:
        for name in [n for n in self._plotter.renderer.actors if n.startswith(_REFERENCE_PREFIX)]:
            self._plotter.remove_actor(name, render=False)

    def _draw_reference_layers(self) -> None:
        self._remove_reference_layers()
        for e in self._reference_layers:
            feats = e.get("native") if self._frame_mode else e.get("lonlat")
            if not feats:
                continue
            parts = [np.asarray(p, dtype=np.float64) for parts in feats for p in parts if len(p)]
            if not parts:
                continue
            allpts = np.concatenate(parts)
            # 3-D surface: a raster shown as a surface rises above the fixed lift and would hide
            # the outlines under it, so ride it from the layer's pixel-frame vertices, exactly
            # as chains and H-lines do through _scene_points.
            ride = self._reference_ride(e.get("pixels"), parts)
            z = ride if ride is not None else np.zeros(len(allpts))
            if self._frame_mode:
                pts = np.column_stack([allpts[:, 0], allpts[:, 1], z])
                pts[:, 2] *= self._vexag
                pts[:, 2] += _VECTOR_LIFT_FRAC * max(np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1]), 1.0)
            else:
                pts = project(allpts[:, 0], allpts[:, 1], z + _VECTOR_LIFT_KM,
                              mode=self._mode, vexag=self._vexag)
            name = f"{_REFERENCE_PREFIX}{e['ref_id']}"
            color = e.get("color") or REFERENCE_COLORS[0]
            if e.get("kind") in ("point", "multipoint"):
                actor = self._plotter.add_mesh(pv.PolyData(pts), color=color, style="points",
                                               point_size=6.0, name=name, reset_camera=False)
            else:
                cells, off = [], 0
                for part in parts:
                    cells.extend([len(part), *range(off, off + len(part))]); off += len(part)
                actor = self._plotter.add_mesh(pv.PolyData(pts, lines=np.array(cells, dtype=np.int64)),
                                               color=color, line_width=1.5, name=name, reset_camera=False)
            actor.SetVisibility(bool(e.get("visible", True)))

    # -- previews (data browser) -----------------------------------------------------------
    def set_previews(self, previews) -> None:
        """Drape small preview fields (:func:`dynamix.geo.footprints.overview_field`) on the
        world so a scene can be judged before it is imported. ``previews`` is a list of
        ``(name, RasterField)``; view state only -- never a layer, never picked, never in the
        legend. Re-projected on every mode switch by :meth:`_rebuild`; hidden in frame mode,
        which has no CRS to place a lon/lat drape in. An empty list clears them."""
        self._previews = list(previews)
        self._draw_previews()
        self._plotter.render()

    def _remove_previews(self) -> None:
        for name in [n for n in self._plotter.renderer.actors if n.startswith(_PREVIEW_PREFIX)]:
            self._plotter.remove_actor(name, render=False)

    def _draw_previews(self) -> None:
        self._remove_previews()
        if not self._previews or self._frame_mode:
            return
        for i, (_name, field) in enumerate(self._previews):
            lon2d, lat2d, values2d, _stride = field_lonlat_grid(field)
            lon_flat, lat_flat = lon2d.ravel(), lat2d.ravel()
            pts = project(lon_flat, lat_flat, np.full_like(lon_flat, _VECTOR_LIFT_KM / 2),
                          mode=self._mode, vexag=self._vexag)
            ny, nx = lon2d.shape
            grid = pv.StructuredGrid()
            grid.points = pts
            grid.dimensions = [nx, ny, 1]
            grid.point_data["value"] = values2d.ravel()
            self._plotter.add_mesh(grid, scalars="value", cmap="gray", nan_opacity=0.0,
                                   opacity=0.9, show_scalar_bar=False,
                                   name=f"{_PREVIEW_PREFIX}{i}", reset_camera=False)

    def _build_roi_actor(self, layer_id, field, roi):
        """The bounding rectangle of ``result["_roi"]["roi"] = (row, col, h, w)`` as a closed line
        loop, or ``None`` for a degenerate (``h <= 0`` or ``w <= 0``) box -- nothing to outline.

        The far corner is CLAMPED to the field's last valid pixel index (``nx - 1`` / ``ny - 1``)
        rather than addressed at ``col + w`` / ``row + h`` directly: those are PIXEL-COUNT offsets
        (the ROI covers columns ``[col, col + w)``), not necessarily a valid pixel INDEX --
        ``points_lonlat`` needs an index into ``field.x_axis``/``field.y_axis``, and an ROI that
        touches the raster's own far edge would otherwise ask for one pixel past it. This is a
        cosmetic v1 outline (at most half a pixel off at that one edge), not a scale measurement.
        """
        r0, c0, h, w = (int(v) for v in roi)
        if h <= 0 or w <= 0:
            return None
        c_hi = min(c0 + w, field.nx - 1)
        r_hi = min(r0 + h, field.ny - 1)
        cols = np.array([c0, c_hi, c_hi, c0, c0], dtype=np.intp)
        rows = np.array([r0, r0, r_hi, r_hi, r0], dtype=np.intp)
        pts = self._scene_points(field, cols, rows)
        name = self._roi_actor_name(layer_id)
        loop = pv.PolyData(pts, lines=np.array([5, 0, 1, 2, 3, 4], dtype=np.int64))
        self._plotter.add_mesh(loop, color=ROI_BOUNDS_COLOR, line_width=2.0, name=name,
                                reset_camera=False)
        return name

    # -- masking -----------------------------------------------------------------------------

    def _apply_mask_to_all(self) -> None:
        for layer_id in self._vector_masks:
            self._apply_mask_to_layer(layer_id)
        # add_mesh's own render=True default is what makes every OTHER actor-producing path in
        # this module repaint itself; this method only MUTATES an existing PolyData's "colors"
        # array (never calls add_mesh), and VTK does not request a repaint for that on its own
        # (confirmed by an offscreen screenshot probe: the frame stayed stale without this).
        # Called from both set_mask() (as its last step -- this IS "render at the end of
        # set_mask") and _rebuild() (whose per-actor add_mesh calls already rendered once each,
        # but BEFORE this method's own re-application of the active mask runs) -- one explicit
        # render() here covers both call sites, including the set_mode()-reapplies-the-mask path
        # the same staleness would otherwise affect. pv.Plotter.render() works offscreen, so this
        # is unconditional.
        self._plotter.render()

    def _point_visibility(self, layer_id: int):
        """Per-point boolean mask-visibility for ``layer_id``'s chains actor, under the CURRENT
        ``self._mask`` -- the exact computation :meth:`_apply_mask_to_layer` needs for its own
        alpha channel, and the SAME one :meth:`pick` consults so a masked-out (alpha-zeroed) chain
        is never returned as a hit (module docstring, "stays pickable"). One shared implementation
        so the two can never quietly drift apart. ``None`` when the layer has no drawable chains
        (mirrors the old early-return this replaces)."""
        state = self._vector_masks.get(layer_id)
        if state is None or state.max_log2_mod.size == 0:
            return None
        max_log2_mod = state.max_log2_mod
        modulus_pctl, scale_lo, scale_hi = self._mask

        # The percentile threshold is computed over DRAWABLE chains only (finite max_log2_mod) --
        # a <2-point chain's -inf sentinel would otherwise skew the threshold low for chains that
        # actually own points on screen, even though that sentinel chain itself never appears in
        # chain_indices and so could never be shown regardless.
        finite = np.isfinite(max_log2_mod)
        threshold = np.percentile(max_log2_mod[finite], modulus_pctl) if finite.any() else np.inf
        modulus_visible = max_log2_mod >= threshold

        # Chain-DEPTH range (adjudicated semantics -- see set_mask's docstring): a chain is
        # visible iff its own deepest scale index (hi_scale_idx) falls within [scale_lo,
        # scale_hi], scale_hi == 0 meaning "no cap" (HLineLength's own max_len convention, mirrored
        # exactly: "(n_of <= hi) if hi > 0 else True"). Both bounds discriminate: scale_lo hides
        # chains too shallow to reach it, scale_hi (when nonzero) hides chains too deep.
        scale_visible = (state.hi_scale_idx >= scale_lo) & (
            (state.hi_scale_idx <= scale_hi) if scale_hi > 0 else True)

        chain_visible = modulus_visible & scale_visible
        _starts, chain_indices = self._chain_lookup[layer_id]
        return chain_visible[chain_indices]

    def _apply_mask_to_layer(self, layer_id: int) -> None:
        state = self._vector_masks[layer_id]
        point_visible = self._point_visibility(layer_id)
        if point_visible is None:
            return
        _starts, chain_indices = self._chain_lookup[layer_id]

        rgba = state.base_rgba.copy()
        rgba[:, 3] = np.where(point_visible, 255, 0)

        # The two overlays -- RGB only, alpha (visibility, above) is never touched by either.
        # See the module docstring's compositing-order note: group-preview first, selection last
        # (selection wins where both apply). A handful of dict entries at most (user-picked chains
        # / a few designated groups), so a plain per-key boolean-mask loop is simplicity-first
        # here, not a perf concern -- unlike the alpha channel above, this never runs on a hot
        # per-vertex path.
        group_colors = self._group_preview.get(layer_id)
        if group_colors:
            for chain_idx, color in group_colors.items():
                rgba[chain_indices == chain_idx, :3] = color
        selected_chains = self._selection.get(layer_id)
        if selected_chains:
            rgba[np.isin(chain_indices, list(selected_chains)), :3] = SELECTION_COLOR

        grid = self._plotter.actors[state.chains_actor_name].mapper.dataset
        grid.point_data["colors"] = rgba

    # -- naming ------------------------------------------------------------------------------

    @staticmethod
    def _hlines_actor_name(layer_id) -> str:
        return f"layer-{layer_id}-hlines"

    @staticmethod
    def _mesh_actor_name(layer_id) -> str:
        return f"layer-{layer_id}-raster"

    @staticmethod
    def _chains_actor_name(layer_id) -> str:
        return f"layer-{layer_id}-chains"

    @staticmethod
    def _extrema_actor_name(layer_id) -> str:
        return f"layer-{layer_id}-extrema"

    @staticmethod
    def _roi_actor_name(layer_id) -> str:
        return f"layer-{layer_id}-roi"

    @staticmethod
    def _points_actor_name(layer_id) -> str:
        return f"layer-{layer_id}-points"

    @staticmethod
    def _legend_line(layer, status: str) -> str:
        if status == "computing":
            detail = "computing…"
        elif status == "no-georeference":
            detail = "no georeference — session view only"     # the design, verbatim wording
        elif status == "frame-points":
            # A point layer has no frame-mode placement at all
            # (WGS84-placed by contract) -- see _add_points_geometry's own docstring.
            detail = "session view only (frame mode)"
        elif status.startswith("error:"):
            detail = f"error — {status[len('error:'):]}"
        else:
            detail = status     # unrecognised status string: show it verbatim, never hide it
        return f"{layer.name}: {detail}"

    def _set_legend(self, lines: list[str]) -> None:
        if not lines:
            if self._legend_present:
                self._plotter.remove_actor(_LEGEND_NAME, render=False)
                self._legend_present = False
            return
        self._plotter.add_text("\n".join(lines), position=(0.02, 0.85), font_size=12,
                                name=_LEGEND_NAME, render=False)
        self._legend_present = True

    def _clear_actors(self) -> None:
        for names in self._layer_actors.values():
            for name in names:
                self._plotter.remove_actor(name, render=False)
        self._layer_actors = {}
        self._chain_lookup = {}
        self._vector_masks = {}
        self._surface_fields = {}
        if self._legend_present:
            self._plotter.remove_actor(_LEGEND_NAME, render=False)
            self._legend_present = False
        self._plotter.remove_actor(_FOOTPRINT_NAME, render=False)
        self._remove_previews()
        self._remove_reference_layers()
