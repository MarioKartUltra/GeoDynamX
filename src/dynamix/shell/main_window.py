# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""MainWindow: the fixed zones, wired end to end -- the core gesture made a window.

Layout is four fixed zones and nothing else (DESIGN.md, "The Instrument Rack"): a narrow left
panel (open, device browser, layer list, split vertically per the workflow-manager spec's 3.1),
the canvas in the centre, and the transport plus the chain strips along the bottom. **No dock
widgets anywhere, ever** (the class name is deliberately not
even spelled in this package -- a test greps for it) -- a dock is a user-rearrangeable zone, and a
rearrangeable instrument has no muscle memory.

The wiring is the whole point of this file, and it has exactly two paths:

- **A FILTER param moved** -> re-``resolve`` synchronously, right here on the GUI thread. This is
  the 16 ms law: the transform is already in the cache, so the resolve re-runs ``apply()`` only
  and reports ``cache_misses == 0``. The engine's own counters are the proof, and a test asserts
  them (``test_filter_knob_drag_is_zero_miss``). The transport's scale sweep is the same path --
  it writes ``scale_select``'s param and re-resolves -- because scrub and playback and the knob
  must be ONE code path or they drift apart.
- **A TRANSFORM param moved** -> hand the whole resolve to :class:`~dynamix.shell.worker.
  ResolveWorker` on its own thread, mark the transform strips ``computing``, and keep showing the
  last good render until it lands. Every worker signal is connected to a BOUND METHOD, never a
  lambda: worker.py documents why (a lambda has no thread affinity, so Qt runs it on the worker
  thread, where touching a widget is a crash waiting to happen).

When the worker finishes, the window does NOT render its ``Renderable`` directly -- it re-resolves
synchronously instead. That resolve is a guaranteed cache hit (the worker just populated the cache
with the same key), and it means there is exactly ONE place a result reaches the screen. Any
filter param the user moved WHILE the transform was computing is picked up by that same resolve,
so a mid-compute knob turn can never leave the screen showing a stale filter setting.

Slice 2 adds a third path that is really the first one wearing a different hat: a ⌘-drag on the
canvas emits ``roiDrawn``, the precision panel (``roi_panel.py``) turns that into the final
integers, and **Create** spawns a grouped layer whose chain is the parent's with ``wtmm2d``
re-pointed at the ROI (:func:`roi_chain`). Selecting that layer -- or any layer -- in the list
switches the whole window to it (:meth:`MainWindow._select_layer`) and resolves through the SAME
worker path a transform change takes, which is what makes flipping between an ROI and its parent a
cache hit rather than a recompute.

Errors and closing are honest, per the spec: a transform that raises marks its own strip with the
error state and puts the message in its reading -- no dialogs, no toasts, and the last good render
stays on screen. Closing during a compute states which stage it is waiting out and completes when
that stage does; there is no cancellation in v1, so pretending otherwise would be a lie.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.core.stretch import STRETCHES
from dynamix.core.frames import LocalFrame, frames_compatible
from dynamix.core.pointset import read_csv_points
from dynamix.core.transect import chains_in_buffer, sample_profile
from dynamix.devices import register_builtin_devices
from dynamix.devices.groups import encode_groups
from dynamix.engine import Cache, cache_key, resolve, source_identity
from dynamix.engine.resolve import output_key, selection_recipe, selection_steps
from dynamix.roi.picture import display_stride, file_pixel_grid, native_shape
from dynamix.geo.footprints import (band_label, band_sort_key, group_key, overview_field,
                                    scan_footprints, scene_label)
from dynamix.geo.mapping import axis_at, has_georeference
from dynamix.geo.vectors import read_shapefile, to_crs, to_field_pixels, to_lonlat
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import (declared_outputs, defaults_for, get_device, is_transform,
                                  keyed_params, validate_params)
from dynamix.model.inspector_state import (InspectorState, SubLayerState,
                                           inspectors_from_payload, inspectors_to_payload)
from dynamix.model.param import Param, ParamKind
from dynamix.model.presets import PRESETS
from dynamix.model.project import Project
from dynamix.model.projectfile import (changed_sources, missing_sources, open_project,
                                       resolve_source, save_project, sha256_of)
from dynamix.model.provenance import provenance_tree
from dynamix.roi.halo import _EDGES, MIN_A_MIN
from dynamix.shell.arrangement.group_palette import GroupPalette
from dynamix.shell.arrangement.mask_row import MaskRow
from dynamix.shell.browser import DeviceBrowser
from dynamix.shell.canvas import (Canvas, EXTREMA_COLOR, HCHAIN_COLOR, POINTS_COLOR, VTRAIL_COLOR,
                                  display_offset, nice_round_scalebar, window_offset)
from dynamix.shell.components_window import ComponentsWindow
from dynamix.shell.fork_dialog import ForkDialog
from dynamix.shell.bus_dialog import BusDialog
from dynamix.shell.import_dialog import ImportGridsDialog
# Module-level, like every other pure-Qt shell widget above (the lazy-import discipline in this
# file guards ``pyvista``, not Qt): the floating inspector imports the existing Canvas and the
# existing Transport, both of which this module already imports eagerly anyway.
from dynamix.shell.inspector import NO_RESULT_TEXT, InspectorWindow
from dynamix.shell.layer_panel import LayerPanel, ReferencePanel, layer_outputs, shown_output
from dynamix.shell.levels_dialog import BandDialog, LevelsDialog
from dynamix.shell.multifractal_window import MultifractalWindow
from dynamix.shell.opening import open_field
from dynamix.shell.anisotropy_window import AnisotropyWindow
from dynamix.shell.spectrum_window import SpectrumWindow
from dynamix.shell.point_import import load_points
from dynamix.shell.profile_dialog import ProfileDialog
from dynamix.shell.right_panel import CompositePanel, ReconstructionPanel, RightPanel
from dynamix.shell.roi_panel import RoiPanel
from dynamix.shell.settings import Settings, load_settings, save_settings, update_settings
from dynamix.shell.skeleton_dialog import SkeletonDialog
from dynamix.shell.surface_dialog import SurfaceDialog
from dynamix.shell.topology_panel import TopologyPanel
from dynamix.shell.transect_panel import TransectPanel
from dynamix.shell.transport import Transport
from dynamix.shell.units import px_to_metres
from dynamix.shell.view_dialog import ViewDialog, normalized_view_options
from dynamix.shell.worker import OutputWorker, ResolveWorker
from dynamix.shell.workflow_zone import BOX_HEIGHT, WorkflowZone
from dynamix.topology.links import ObjRef, suggest_code

#: The chain a freshly opened raster gets. ``demo.py``'s DEMO_CHAIN plus the two topology devices.
#:
#: Order is not free. Transforms come first (the engine rejects an interleaved chain), and among
#: the filters ``scale_select`` comes first for two reasons: it reduces the stack to ONE layer, so
#: every filter after it does 1/n of the work inside the 16 ms budget, and ``min_vchains`` must see
#: a layer whose point rows still match the topology's anchors -- true after ``scale_select``
#: (which selects a layer without rewriting it) and false after any masking filter.
#:
#: ``min_vchains`` opens at 1, not at its declared default of 2: the window's first frame should
#: show what was computed, and the user turns the knob UP to prune.
DEMO_CHAIN = (
    ("wtmm2d", {"n_oct": 3, "n_voice": 4}),
    ("chain_topology", {}),
    ("scale_select", {"scale_idx": 0}),
    ("min_vchains", {"min_vchains": 1}),
    ("orientation_wedge", {"centre": 0.0, "half_width": 90.0}),
    ("modulus_threshold", {"frac": 0.0}),
)

TITLE = "DynamiX"

#: The center zone's three views, in Tab-cycle order -- Raster
#: (the session canvas, ``self._center_stack`` index 0 always) -> Vector (the arrangement drawn
#: in each field's own native frame, no CRS/georeference required -- see ``_sync_arrangement``'s
#: ``frame_mode`` branch) -> Globe (the arrangement's original WGS84 placement, unchanged) ->
#: back to Raster. ``_cycle_center_view`` walks this tuple; ``_set_center_view`` accepts any one
#: directly (the switcher's three buttons, and ``Settings.center_view``'s restored value).
_CENTER_VIEWS = ("raster", "vector", "geo")

#: The legal selection-mode values -- EQSelect's own
#: names ("lasso", not "freehand"). ``"transect"`` is legal from day one (it gets its own gesture) -- :meth:`MainWindow.set_selection_mode` raises on anything
#: outside this tuple.
_SELECTION_MODES = ("click", "box", "lasso", "transect")

#: `v`'s own cycle order ("`v` -> `_cycle_selection_mode`... cycles only the non-click
#: modes"), deliberately excluding ``"click"`` -- `c` owns click directly, and EQSelect's own `v`
#: never visits it either.
_SELECTION_MODE_CYCLE = ("box", "lasso", "transect")

_PANEL_WIDTH = 220          # left zone's default splitter width -- a first-run
                            # hint only; the boundary drags freely and Settings.splitter_sizes
                            # overrides this the moment the user has ever moved it
#: Right zone's default splitter width, same "first-run hint only" status as
#: ``_PANEL_WIDTH`` above. It hosts the Display section, so a FRESH settings file opens it wide enough to show those knobs instead of hiding them.
_RIGHT_PANEL_WIDTH = 260
_TRANSPORT_WIDTH = 340      # transport sits at the strip zone's left edge

#: The two inspector status readings this window owns; ``inspector.NO_RESULT_TEXT`` is the third
#: and lives with the widget (see its own comment there). All three are chosen in ONE place,
#: :meth:`MainWindow._inspector_status`, because an inspector cannot tell a live picture from a
#: kept one on its own -- it holds no ``Project`` and cannot map a ``layer_id`` to a source
#: (the design rule 3). Sec 10 A2 / data doctrine #3: a non-active inspector keeps its last
#: result and SAYS SO -- honest, not blank, and it names the fix.
INSPECTOR_LIVE_TEXT = "showing the live result"
INSPECTOR_LAST_RESULT_TEXT = "last result — layer '{name}' is not active; select it to re-run"

#: What the "I" on a CSV point-catalogue source header says instead of opening a window. Sec 5a
#: asked for "a window of the raster" and a point source has none: its field is a ``PointSet``,
#: which ``Canvas.set_field`` cannot take at all (``np.asarray(field, dtype=float64)`` ->
#: TypeError, the trap ``_select_layer``'s own ``_is_point_layer`` guard already names), and
#: drawing points on the canvas is not built yet. Sec 8 rule 6: the refusal is SPOKEN and names
#: the fix rather than the fault.
INSPECTOR_NO_RASTER_TEXT = ("point catalogues have no raster to inspect — open an inspector on a "
                            "raster source")

#: How a grouped ROI layer is drawn in the flat layer list -- the Ableton group-track read, one
#: level deep, which is as deep as ``parent_id`` goes (an ROI of an ROI is not a thing yet).
_CHILD_PREFIX = "  ↳ "

#: The ten WTMM params ``wtmm2d`` and ``wtmm2d_roi`` share. An ROI is meant to be the SAME
#: analysis over a smaller window, so these ride across unchanged; everything else about the two
#: devices differs (the ROI window and boundary knobs are ``wtmm2d_roi``-only).
#:
#: A param missing here is silently lost on every ROI child: a parent tuned away from its
#: default (``smooth``, ``thresh``, ``dist2_max``, ``box_ratio``, ``similitude``) would make the
#: child run a DIFFERENT analysis than the parent while being presented (by name, by chain
#: lineage) as the same one over a smaller window, which is exactly what ``roi_chain``'s own
#: docstring says an ROI must never do. ``fracint_alpha`` rides here too: the ROI engine applies
#: it (``run_wtmm2d_roi`` resolves it to the backend default), so a parent tuned off η = 1.0
#: must carry that tuning into the child.
_SHARED_WTMM_PARAMS = ("n_oct", "n_voice", "a_min", "wavelet", "min_chain_len",
                      "smooth", "thresh", "dist2_max", "box_ratio", "similitude",
                      "fracint_alpha", "interpolate", "detector")

#: Per-layer overlay styling (EQSelect matrix "layer display controls"), declared as Params so the
#: controls generate exactly like a device's. VIEW state: stored in ``layer.tags["ui.<name>"]``,
#: applied by :meth:`MainWindow._sync_display_controls`, never consulted by the analysis path.
#: Styling a LOCKED layer is allowed — lock guards the measurement, not how it is drawn.
_DISPLAY_PARAMS = (
    Param("opacity", ParamKind.FLOAT, default=1.0, min=0.0, max=1.0,
          soft_min=0.2, soft_max=1.0, label="Overlay opacity"),
    Param("point_size", ParamKind.FLOAT, default=3.0, min=0.1, max=100.0,
          soft_min=1.0, soft_max=8.0, label="Point size"),
    Param("line_width", ParamKind.FLOAT, default=1.0, min=0.05, max=50.0,
          soft_min=0.5, soft_max=3.0, label="Line width"),
    # Hillshade: the sun and the vertical exaggeration of the shaded relief the
    # canvas and the drape show under the chains when ``ui.hillshade`` is on. Display only.
    Param("sun_azimuth", ParamKind.FLOAT, default=315.0, min=0.0, max=360.0,
          soft_min=0.0, soft_max=360.0, units="deg", label="Sun az"),
    Param("sun_altitude", ParamKind.FLOAT, default=45.0, min=0.0, max=90.0,
          soft_min=5.0, soft_max=90.0, units="deg", label="Sun alt"),
    Param("z_factor", ParamKind.FLOAT, default=1.0, min=0.001, max=10000.0,
          soft_min=0.1, soft_max=20.0, label="Vert. exag."),
    # Stretch: the percent-clip for ui.stretch == "percent" (core.stretch).
    Param("stretch_pct", ParamKind.FLOAT, default=2.0, min=0.0, max=49.0,
          soft_min=0.5, soft_max=10.0, units="%", label="Clip %"),
)

#: The three knob names, for :meth:`MainWindow._on_display_style_changed`'s dispatch: a name in
#: this set is a float knob (``repr(float(value))``); everything else is one of the five keys
#: just below (a string, or ``show_trails``'s bool).
_DISPLAY_PARAM_NAMES = {p.name for p in _DISPLAY_PARAMS}

#: The ``Param.section`` whose knobs the right panel's Reconstruction section draws, and its title.
_RECON_SECTION = "reconstruction"
_RECON_TITLE = "Reconstruction"

#: An output row drawn from a cheaper, non-row output until its own is Run: row -> preview. The
#: LastWave recon row shows its one-iteration preview; Run computes the reconstruction, which
#: continues from that preview's state.
_PREVIEW_OF = {"recon": "recon_preview"}

#: The reading of a row waiting for Run (:meth:`MainWindow._awaits_run`): Live is off and neither
#: its output nor its preview is cached for the current knobs, so the raw field stays up.
_AWAITS_RUN = "press Run"
#: The reconstruction rows that wait for Run while Live is off: the recon row through its preview
#: (:meth:`MainWindow._awaits_run`), the others directly (:meth:`MainWindow._waits_manual`).
_RUN_ROWS = ("recon", "recon_edges_only", "residual")

#: Colormap + palette + trails preferences (the MODEL half; the right panel builds their combo/swatch/checkbox controls). Deliberately NOT ``Param``s alongside the three
#: above: ``_DISPLAY_PARAMS``' own float-min/max validation makes no sense for a colormap NAME, a
#: hex color STRING, or a bool -- these five ride the identical ``layer.tags["ui.<name>"]`` scheme
#: (same tag namespace, same "view state, never consulted by the analysis path" contract) but are
#: read by :func:`_display_style_of` directly, with their own tolerant parsing per key, rather than
#: through ``_DISPLAY_PARAMS``' shared float-cast loop.
#:
#: The three color defaults are DERIVED from ``canvas.py``'s own constants (``_hex`` below), not
#: restated as separate literals -- a re-tint of the canvas module's defaults moves these for free,
#: with no second place that could quietly drift out of sync.
_DEFAULT_COLORMAP = "viridis"


def _hex(rgb: tuple[int, int, int]) -> str:
    """``(r, g, b)`` -> a lowercase ``"#rrggbb"`` string -- the inverse of :func:`_hex_to_rgb`."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _hex_to_rgb(text: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> ``(r, g, b)`` ints -- the inverse of :func:`_hex`. Callers only ever pass
    a string ``_display_style_of`` has already validated (:data:`_HEX_RE`), so this never needs to
    guard against a malformed one itself."""
    s = text.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


_DEFAULT_COLOR_HCHAIN = _hex(HCHAIN_COLOR)
_DEFAULT_COLOR_VTRAIL = _hex(VTRAIL_COLOR)
_DEFAULT_COLOR_EXTREMA = _hex(EXTREMA_COLOR)
#: The point-layer scatter's own preference key, riding the identical
#: ``ui.color_points`` tag scheme as the three above -- see ``canvas.py``'s ``POINTS_COLOR`` for
#: why the default is a distinct cool hue rather than a reuse of any existing overlay color. A
#: fourth swatch control in the Display section is not built here -- this is only the tag key and its default, wired through the same pipeline.
_DEFAULT_COLOR_POINTS = _hex(POINTS_COLOR)

#: A garbled ``ui.color_*`` tag (hand-edited project file, a future format change) must fall back
#: to its default rather than reach ``canvas.set_overlay_colors`` as nonsense -- the same tolerance
#: ``_display_style_of``'s float loop already extends to ``ui.opacity``/etc.
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _display_style_of(layer) -> dict:
    """The layer's stored overlay styling, defaults filled — tolerant of absent/garbled tags."""
    out = {}
    for p in _DISPLAY_PARAMS:
        raw = layer.tags.get(f"ui.{p.name}") if layer is not None else None
        try:
            out[p.name] = float(raw) if raw is not None else p.default
        except (TypeError, ValueError):
            out[p.name] = p.default
    tags = layer.tags if layer is not None else {}
    out["colormap"] = tags.get("ui.colormap") or _DEFAULT_COLORMAP
    for name, default in (("color_hchain", _DEFAULT_COLOR_HCHAIN),
                          ("color_vtrail", _DEFAULT_COLOR_VTRAIL),
                          ("color_extrema", _DEFAULT_COLOR_EXTREMA),
                          ("color_points", _DEFAULT_COLOR_POINTS)):
        raw = tags.get(f"ui.{name}")
        out[name] = raw if isinstance(raw, str) and _HEX_RE.match(raw) else default
    out["show_trails"] = tags.get("ui.show_trails") == "True"
    raw = tags.get("ui.arrows")
    out["arrows"] = raw if raw in ("off", "wtmmm", "all") else "off"
    out["hillshade"] = tags.get("ui.hillshade") == "True"
    raw = tags.get("ui.stretch")
    out["stretch"] = raw if raw in STRETCHES else "linear"
    out["levels"] = tags.get("ui.levels", "")   # density slice: "" = off
    out["levels_colors"] = tags.get("ui.levels_colors", "")   # "#rrggbb,..." per class; "" = LUT
    try:
        out["levels_sieve"] = int(tags.get("ui.levels_sieve", "0") or 0)
    except ValueError:
        out["levels_sieve"] = 0
    out["surface"] = tags.get("ui.surface") == "True"                 # 3-D surface
    # Surface HEIGHT source: "same" (this layer's own values -- the
    # pre-existing behavior and the default) or another loaded raster layer's layer_id.
    out["surface_source"] = tags.get("ui.surface_source", "same")
    out["depth_positive"] = tags.get("ui.depth_positive") == "True"   # negate: depth grows down
    return out


#: The field-analyzing chain heads: one of these defines what a layer IS showing. Dropping a
#: DIFFERENT one onto an analyzed layer forks a sibling representation instead of mutating
#: the chain. ``wtmm2d_roi`` is absent
#: deliberately -- its drop has its own armed-placement branch.
_PRIMARY_ANALYZERS = ("wtmm2d", "mz_edges", "wavelet_skeleton", "cdf_edges", "pm_edges",
                      "holder_map", "holder_measure", "holder_multiaffine", "band_recon",
                      "band_recon_measure", "band_recon_multiaffine")

#: The band-reconstruction devices, newest first, and the variant each
#: h-map producer's band fork commits -- the variant whose engine params are the producer's
#: VERBATIM (shared by identity), so the h-map the band was picked from is the h-map the
#: reconstruction uses. ``holder_map`` keeps its conflated ``band_recon``, which saved projects
#: rely on.
_BAND_DEVICES = ("band_recon_measure", "band_recon_multiaffine", "band_recon")
_BAND_FOR_ANALYZER = {"holder_measure": "band_recon_measure",
                      "holder_multiaffine": "band_recon_multiaffine",
                      "holder_map": "band_recon"}


def _eta_note_of(result: dict) -> tuple:
    """``(eta_seed, forward_note)`` derived HONESTLY from the result's own resolved params (the
    ``_skeleton_px_size`` pattern -- the spectrum windows stay dumb): both WTMM paths apply the
    ``a^fracint_alpha`` lift forward, so the seed is the resolved
    ``fracint_alpha`` whenever non-zero, and the note names what was lifted."""
    params = result.get("params") or {}
    alpha = float(params.get("fracint_alpha", 0.0))
    if alpha != 0.0:
        where = "WT derivs (tensor)" if params.get("mode") == "tensor" else "|∇W| (scalar)"
        return alpha, f"forward lift: a^{alpha:g} on {where}"
    return 0.0, "forward lift: none (η = 0)"


def _band_step_for(names) -> "str | None":
    """The band-reconstruction device the recipe already holds (for seeding/updating the
    Band dialog), or None."""
    return next((n for n in _BAND_DEVICES if n in names), None)


def _band_device_for(names) -> str:
    """The band-reconstruction variant a Band-dialog commit should create for THIS recipe:
    the variant matching the recipe's h-map producer, conflated ``band_recon`` when the
    producer is the conflated device or absent."""
    producer = next((n for n in _BAND_FOR_ANALYZER if n in names), "holder_map")
    return _BAND_FOR_ANALYZER[producer]

#: Consumer devices -> the producers whose results they read. THE SHIPPING LAW: devices follow
#: their producer. On a fork, a consumer stranded on the parent whose producers include the
#: NEW analyzer but not the parent's own ships to the sibling WITH its tuned params -- on the
#: parent it could only ever no-op. Consumers of the parent's analyzer never move.
_CHAIN_PRODUCERS = ("wtmm2d", "wtmm2d_roi", "mz_edges", "wavelet_skeleton", "cdf_edges",
                    "pm_edges")
_CONSUMER_PRODUCERS = {name: _CHAIN_PRODUCERS for name in (
    "scale_select", "orientation_wedge", "modulus_threshold", "hline_length",
    "hline_modulus", "hline_holder", "chain_holder", "chain_modulus", "chain_length", "chain_classify",
    "chain_topology", "min_vchains", "group_paint", "group_filter")}


def _analyzer_fork(old_names, old_bypassed, descriptors, *,
                   root: bool = False) -> "tuple | None":
    """``(sibling_steps, shipped_parent_indices)`` for a fork, or None for an ordinary edit.

    Fork iff the proposal introduces a primary analyzer the current chain is not already
    headed by, AND either (a) the current chain has its own primary (a different
    representation), or (b) ``root`` -- the layer is a source's MASTER (parent_id None), which
    is the raw dataset and never takes an analyzer directly (every analysis is a child). The sibling takes
    every proposed step beyond what the current recipe already holds -- a MULTISET diff, so a
    dropped preset that re-includes a device the parent also has still ships a copy -- plus
    the parent's stranded consumers of the new analyzer (the shipping law above), returned as
    indices into the old recipe so the caller can MOVE them (params and all). Same-analyzer
    edits, filter edits and field-prep transforms (noise) on the master commit as before --
    noise deliberately STAYS on the master, so a child needing prep should be dropped as a
    preset carrying it."""
    old_primary = next((n for n, b in zip(old_names, old_bypassed)
                        if not b and n in _PRIMARY_ANALYZERS), None)
    if old_primary is None and not root:
        return None
    new_enabled = [d["device"] for d in descriptors if not d.get("bypassed")]
    new_primary = next((n for n in new_enabled
                        if n in _PRIMARY_ANALYZERS and n != old_primary), None)
    if new_primary is None:
        return None
    from collections import Counter
    have = Counter(old_names)
    sibling = []
    for d in descriptors:
        n = d["device"]
        if have[n] > 0:
            have[n] -= 1
        else:
            sibling.append(dict(d))
    if not sibling:
        return None
    shipped = [i for i, n in enumerate(old_names)
               if n in _CONSUMER_PRODUCERS
               and new_primary in _CONSUMER_PRODUCERS[n]
               and old_primary not in _CONSUMER_PRODUCERS[n]]
    return sibling, shipped


def _roi_spawn(old_names, descriptors) -> "tuple | None":
    """``(sibling_steps, [])`` when a drop on a MASTER with an ROI active introduces an
    ANALYZING transform (not a field stage like noise, not a source-reading device) -- the
    ROI runner spec's "any tool on the region" (pca/tucker are not primary
    analyzers, so the analyzer fork never spawned them). Same multiset diff as
    :func:`_analyzer_fork`; ``None`` otherwise."""
    from collections import Counter
    have = Counter(old_names)
    sibling = []
    for d in descriptors:
        if have[d["device"]] > 0:
            have[d["device"]] -= 1
        else:
            sibling.append(dict(d))
    for d in sibling:
        if d.get("bypassed"):
            continue
        dev = get_device(d["device"])
        if (is_transform(dev) and not getattr(dev, "field_stage", False)
                and not getattr(dev, "reads_source", False)):
            return sibling, []
    return None


#: This session's folder for TEMPORARY derivative datasets (created on the first one).
_DERIVATIVE_SCRATCH: "str | None" = None


def _derivative_scratch() -> str:
    """The session's scratch folder for temporary derivatives, removed when the app exits: a
    temporary derivative is left out of a saved project, so nothing saved points into it, and
    "Save derivative as…" copies one out before then."""
    global _DERIVATIVE_SCRATCH
    if _DERIVATIVE_SCRATCH is None:
        import atexit
        import shutil
        import tempfile

        _DERIVATIVE_SCRATCH = tempfile.mkdtemp(prefix="geodynamix-derived-")
        atexit.register(shutil.rmtree, _DERIVATIVE_SCRATCH, True)
    return _DERIVATIVE_SCRATCH


def _inherited_window(parent, **extra) -> dict:
    """Provenance tags a layer spawned FROM ``parent`` must carry (fork on
    an ROI child left "both child layers" in a confused state): a child dataset's
    ``roi.window`` is DATA IDENTITY -- ``engine.resolve.source_identity`` folds it into the
    cache key as ``|win:`` -- never a display preference. A spawn that resolves against the
    parent's CROP field but drops the tag collides its cache lines with a same-chain run on
    the FULL source (first writer wins, the other layer silently shows the wrong result),
    and falls out of the arrangement's ROI family clause. Used by every spawn-from-layer
    path except ``_on_roi_create``, whose window rides in the chain params instead."""
    tags = dict(extra)
    window = parent.tags.get("roi.window")
    if window:
        tags["roi.window"] = window
    bands = parent.tags.get("data.bands")          # a stack with a band removed: same rule
    if bands:
        tags["data.bands"] = bands
    return tags


def _field_choices(field, note: str) -> list:
    """A produced field as fork choices ``[(label, 2-D array)]``: the field itself when it is
    one band, else one choice per band (tick them all for a multiband derivative)."""
    values = np.asarray(field.values, dtype=np.float64)
    if values.ndim == 2:
        return [(f"as shown ({note})", values)]
    names = list((getattr(field, "provenance", None) or {}).get("bands") or [])
    return [((str(names[k]).split("/", 1)[0] if k < len(names) else f"band {k + 1}")
             + f" ({note})", values[..., k]) for k in range(values.shape[-1])]


def _composite_without_band(raw: str, k: int) -> dict:
    """A stored composite spec after band ``k`` left the stack: a channel it fed goes empty,
    solo/mute forget it, and every later band's index shifts down by one."""
    import json

    try:
        spec = json.loads(raw)
    except ValueError:
        return {}

    def shift(b):
        return None if b == k else (b - 1 if isinstance(b, int) and b > k else b)

    out = {c: shift(spec.get(c)) for c in ("r", "g", "b")}
    for key in ("solo", "mute"):
        out[key] = [shift(b) for b in spec.get(key) or [] if b != k]
    for key in ("stretch", "stretch_pct", "stretch_k"):
        if key in spec:
            out[key] = spec[key]
    return out


def _roi_display_field(result, raster, field, name):
    """A derived raster of an ROI RESULT as its own display field: the
    ROI's own axes and frame, pinned at the ROI's FILE position (``provenance["window"]``) --
    never the parent's field with the values swapped, which would stretch an ROI-sized raster
    over the parent's (picture) extent. ``None`` for a result that is not a region's."""
    roi = (result.get("_roi") or {}).get("roi")
    axes = result.get("_roi_axes")
    if not roi or axes is None:
        return None
    from dynamix.core.rasterfield import RasterField

    prov = getattr(field, "provenance", None) or {}
    out = RasterField(name=name, values=np.asarray(raster, dtype=np.float64),
                      frame=result.get("_frame") or field.frame,
                      x_axis=np.asarray(axes[0]), y_axis=np.asarray(axes[1]),
                      units=getattr(field, "units", ""))
    out.provenance.update({"window": {"row_off": int(roi[0]), "col_off": int(roi[1])},
                           "source": prov.get("source"), "full_dims": prov.get("full_dims"),
                           "crs": prov.get("crs")})
    return out


def _display_raster_of(result) -> "np.ndarray | None":
    """The derived raster a result wants shown IN PLACE of the field: ``raster_out`` (the
    general contract -- band_recon's reconstruction) first, else ``h_map`` (holder_map's own).
    None -> the raw field stays up."""
    if not isinstance(result, dict):
        return None
    out = result.get("raster_out")
    return out if out is not None else result.get("h_map")


def _edges_hidden(layer) -> bool:
    """Whether ``layer``'s maxima are off the drawing: its ``ui.edges_hidden`` tag (the edges
    output row's H), honoured only while its last transform declares that vector row. A tag
    left behind by an analyzer since swapped for one without it hides nothing."""
    return (layer.tags.get("ui.edges_hidden") == "1"
            and any(o.kind == "vector" for o in layer_outputs(layer)[1]))


def _output_note(value) -> str:
    """A lazy output row's reading: a reconstruction's iterations, residual status and SNR;
    empty for an output that carries none (the coarse channel, the thumbnail)."""
    diag = (value or {}).get("diag") or {}
    if "n_iter" in diag:
        return f"{diag['n_iter']} it · {diag['status']} · {float(diag['snr_db']):.1f} dB"
    if "iterations" in diag and "stop" in diag:          # the LastWave engine's diagnostics
        return f"{diag['iterations']} it · {diag['stop']} · {float(diag['snr_db']):.1f} dB"
    return ""


def roi_chain(parent_chain: Chain, roi_params: dict) -> Chain:
    """The parent's chain with its WTMM step re-pointed at one ROI.

    ``wtmm2d`` becomes ``wtmm2d_roi`` IN PLACE, carrying the parent's own ``_SHARED_WTMM_PARAMS``
    (wavelet + PRE/CHAINING tuning) plus the ROI window and boundary mode. Everything downstream
    rides along untouched -- that is the whole point of an ROI layer: the same recipe, over a
    window, so the two layers are comparable rather than merely adjacent.

    ``a_min`` is FLOORED at :data:`~dynamix.roi.halo.MIN_A_MIN`. ``wtmm2d`` accepts values down to
    0.25; the ROI path refuses anything under 1.0 (below that the sampled kernel is close to
    all-pass at Nyquist and no affordable halo is honest -- ``dynamix.roi.halo``'s docstring makes
    the measurement), and ``wtmm2d_roi``'s own Param declares that as a hard minimum. Copying a
    parent tuned below the floor would therefore not produce a wrong answer, it would produce a
    chain that cannot be CONSTRUCTED -- a ValueError out of ``materialized()``, inside a button's
    signal handler. Raising it here is the one place that can happen with the user's actual
    intention still in view.

    A parent with no ``wtmm2d`` step at all gets ``wtmm2d_roi`` PREPENDED at the device's own
    defaults. Prepended, not appended: the engine rejects an interleaved chain, so a transform
    cannot follow a filter.
    """
    steps = [DeviceRef(ref.device, dict(ref.params)) for ref in parent_chain.steps]
    index = next((i for i, ref in enumerate(steps) if ref.device == "wtmm2d"), None)
    if index is None:
        carried = defaults_for(get_device("wtmm2d_roi"))
        steps.insert(0, DeviceRef("wtmm2d_roi", {**carried, **roi_params}))
    else:
        parent_params = steps[index].params
        carried = {name: parent_params[name] for name in _SHARED_WTMM_PARAMS}
        carried["a_min"] = max(float(carried["a_min"]), MIN_A_MIN)
        steps[index] = DeviceRef("wtmm2d_roi", {**carried, **roi_params})
    return Chain(tuple(steps)).materialized()


def _roi_margin_reading(margins) -> str:
    """The ROI transform strip's honesty reading, from ``result["_roi_margins"]``.

    ``margins real 12/12`` when every scale's halo was genuine parent data. Otherwise a shortfall
    has up to two DIFFERENT causes -- ``real_frac`` is reduced by a reflected edge (the halo ran
    off the parent) AND by zero-filled nodata (``dynamix/roi/halo.py``'s module docstring,
    "Missing data" -- a hole is "as fabricated as a reflection" for this reading too) -- and the
    two are named separately rather than folded into one "reflected" word, which would be untrue
    for a scale whose shortfall is pure nodata: ``real 9/12 (NE reflected)`` when every short
    scale reflects at least one edge, ``real 9/12 (missing data)`` when none of them do, and
    ``real 9/12 (NE reflected, missing data)`` when the twelve scales have both kinds.

    Edges are the UNION of ``reflected_edges`` across scales (the coarse scales are the ones that
    run out of parent, so a per-scale list would be a column of near-duplicates; the strip is one
    line), ordered by ``dynamix.roi.halo._EDGES``, imported rather than restated: that tuple is
    the engine's own declaration of "reported edge names, in this order", and a second copy of it
    here would be free to drift into naming the same box "EN".

    "Missing data" is stated whenever at least one short scale reflects NO edge at all -- the only
    way ``real_frac`` can then be below 1.0 is nodata, per :func:`dynamix.roi.halo._read_halo`.
    A scale that both reflects and holds nodata is counted only under "reflected": the two causes
    are not separable from ``real_frac`` alone at that granularity, and stating "missing data"
    only for an all-empty-edges scale is honest about what is actually known, not a claim that no
    reflecting scale ever ALSO holds nodata.
    """
    total = len(margins)
    real = sum(1 for m in margins if float(m["real_frac"]) >= 1.0)
    if real == total:
        return f"margins real {real}/{total}"
    reflected = {edge for m in margins for edge in m["reflected_edges"]}
    names = "".join(edge for edge in _EDGES if edge in reflected)
    has_missing = any(float(m["real_frac"]) < 1.0 and not m["reflected_edges"] for m in margins)
    parts = ([f"{names} reflected"] if names else []) + (["missing data"] if has_missing else [])
    return f"real {real}/{total} ({', '.join(parts)})"


def _n_points(result: dict) -> int:
    return sum(len(layer.get("x", ())) for layer in (result.get("extrema") or ()))


#: The flag storage (layer_panel.py's module docstring): the panel owns no flag state of its
#: own, so ``layer.tags`` is the single source of truth every one of these reads.
def _is_locked(layer) -> bool:
    return layer.tags.get("ui.lock") == "1"


def _is_frozen(layer) -> bool:
    return layer.tags.get("ui.freeze") == "1"


def _lock_notice(layer) -> str | None:
    """What ``_on_param_changed``/``_on_chain_edited`` say in the zone's reading/warning label
    instead of applying an edit -- ``None`` when neither flag is set, so the caller's guard is one
    ``if``. Frozen implies locked (a frozen layer is also read-only), so it is checked first: a
    layer can never be reported merely "locked" while its cached results are also pinned."""
    if _is_frozen(layer):
        return "frozen — unlock in the layer panel to edit"
    if _is_locked(layer):
        return "locked — unlock in the layer panel to edit"
    return None


def _filter_reading(device_name: str, result: dict, is_terminal: bool) -> str:
    """The honesty reading for one filter strip: what it actually did to this result.

    A filter may only claim what it can actually account for. Devices that stamp their own
    counter report it (``dropped 0`` included -- displayed, never hidden, per DESIGN.md). The
    point count is a property of the END of the chain, not of any one step, so only the TERMINAL
    strip shows it; putting it on every strip made four devices each claim the same number as its
    own effect, which is the opposite of an honesty reading. An intermediate filter with nothing
    to report says nothing rather than something untrue.
    """
    if device_name == "min_vchains" and "_hchains_dropped" in result:
        return f"dropped {int(result['_hchains_dropped'])}"
    if device_name.startswith("chain_") and "_chains_dropped" in result:
        return f"dropped {int(result['_chains_dropped'])}"
    return f"{_n_points(result)} pts" if is_terminal else ""


def _shift_chains(chains, row_off: int, col_off: int) -> list:
    """A NEW chains list with every ``x``/``y`` shifted by
    ``(col_off, row_off)`` -- the SAME translation :meth:`Canvas.set_result` applies to an ROI
    result's own drawn trail (see :func:`dynamix.shell.canvas.display_offset`'s docstring). Called
    only where that offset is genuinely nonzero (``_apply``'s own guard) -- an ROI's on-screen
    origin is often large, and picking against the result's UN-shifted coordinates while the trail
    itself is drawn shifted would silently miss (or hit the wrong nearby chain) by exactly that
    amount for every ROI/refined-run result, which is the DEFAULT outcome of that whole workflow,
    not an edge case.

    Never mutates ``chains`` in place: those are the engine's own cached, immutable result arrays
    (the zero-cache-miss law) -- each chain becomes a shallow copy with just ``x``/``y``
    overridden, everything else (``mod``, ``arg``, ``tags``, ...) carried through unchanged.
    ``chain["x"]``/``chain["y"]`` are already numpy arrays (every producer in this codebase hands
    them out that way -- the same assumption :func:`dynamix.core.chain_pick.pick_chain` and
    :mod:`dynamix.shell.canvas`'s own trail helpers already make), so plain ``+`` is enough; no
    numpy import is needed in this module for it.
    """
    return [dict(chain, x=chain["x"] + col_off, y=chain["y"] + row_off) for chain in chains]


def _is_offscreen() -> bool:
    """True under the harness's mandated ``QT_QPA_PLATFORM=offscreen``
    (or any other headless QPA plugin) -- the same ``off_screen`` gate EQSelect's own
    ``_on_wtmm_failed`` checks ("the modal is gated on off_screen ... a modal exec
    would block forever" under an offscreen platform), ported here as a live platform-name check
    rather than a constructor flag: unlike EQSelect's ``MainWindow(off_screen=...)``, this window
    has no such flag of its own, and adding one just for this single call site would be a second,
    parallel way to say the same thing ``QGuiApplication.platformName()`` already answers."""
    app = QtWidgets.QApplication.instance()
    return app is not None and app.platformName() == "offscreen"


#: The all-zero ``backproject`` grid spec -- MainWindow._stamp_backproject's own refusal stamp,
#: and the device's own "unbound" state (dynamix.devices.backproject.Backproject.compute).
#: Duplicated as a plain dict, rather than imported from that module, deliberately: this dict is
#: the SHELL's opinion of what "nothing to register against" looks like, and the device's own
#: defaults (``Param.default`` for each of the seven) already say the identical thing independent
#: of this file -- two honest restatements of the same zero, not one importing the other.
_ZERO_BACKPROJECT_SCALARS = {"_target_crs": "", "_target_x0": 0.0, "_target_dx": 0.0,
                             "_target_y0": 0.0, "_target_dy": 0.0, "_target_nx": 0,
                             "_target_ny": 0}


def _preview_band(members):
    """The band a scene is previewed by: ASTER's B03N (near-infrared, the sharpest VNIR band)
    when present, else the first file in band order (``B01``, or a GDEM tile's ``dem``)."""
    for fp in members:
        if fp.name.endswith("_B03N"):
            return fp
    return min(members, key=lambda m: band_sort_key(m.name))


def _common_prefix(names: list[str]) -> str:
    if not names:
        return ""
    first, last = min(names), max(names)
    i = 0
    while i < min(len(first), len(last)) and first[i] == last[i]:
        i += 1
    return first[:i]


def _georef_signature(field) -> tuple:
    """Everything ``to_field_pixels`` and the inside-count read off a field -- its axes' ends
    and lengths, a picture's file grid, the CRS, the native shape -- as a hashable key: two
    fields sharing it map a reference layer to identical pixels, so the warp can be reused."""
    if field is None:
        return ("none",)
    x = getattr(field, "x_axis", None)
    y = getattr(field, "y_axis", None)
    shape = tuple(np.asarray(getattr(field, "values", field)).shape[:2])
    if x is None or y is None:
        return ("bare", shape)
    from dynamix.roi.picture import file_pixel_grid

    grid = file_pixel_grid(field)
    crs = str((getattr(field, "provenance", {}) or {}).get("crs"))
    return (crs, shape, len(x), float(x[0]), float(x[-1]), len(y), float(y[0]), float(y[-1]),
            None if grid is None else tuple(grid))


def _expand_reference_paths(paths) -> list:
    """Vector files to open as reference layers. A ``.shp`` is itself; an Esri ``.lyr`` is the
    binary, unreadable half of a layer PACKAGE whose data is every shapefile under the sibling
    ``commondata`` tree (BOEM's anomaly package: ``v10/<name>.lyr`` + ``commondata/<folders>/*.shp``;
    the ``.lyr`` names those folders as ``..\\commondata\\<folder>`` and nothing else readable),
    so opening it opens them all; a directory is every shapefile under it; a ``.zip`` holding
    shapefile members (the standard GIS-portal download, e.g. BOEM's gcfaultsg fault traces)
    is extracted once beside itself (``<stem>_shp/``, reused if already there --
    the sidecar .dbf/.prj/.shx must be real files for the stdlib reader) and contributes
    every ``.shp`` inside."""
    import zipfile
    out = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".lyr":
            root = p.parent.parent / "commondata"
            out.extend(sorted(q for q in root.rglob("*.shp")) if root.is_dir() else [p])
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.shp")))
        elif p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p) as zf:
                    if not any(n.lower().endswith(".shp") for n in zf.namelist()):
                        out.append(p)          # not a shapefile zip -- pass through untouched
                        continue
                    dest = p.parent / f"{p.stem}_shp"
                    if not dest.is_dir():
                        zf.extractall(dest)
            except (OSError, zipfile.BadZipFile):
                out.append(p)
                continue
            out.extend(sorted(dest.rglob("*.shp")))
        else:
            out.append(p)
    return [str(q) for q in out]


class MainWindow(QtWidgets.QMainWindow):
    """The window. ``resolved``/``errored`` exist so a test can observe the wiring without
    scraping widgets: every result that reaches the screen is emitted from one place."""

    resolved = QtCore.Signal(object)        # the Renderable that just reached the canvas
    errored = QtCore.Signal(str)            # a transform raised; the message is on its strip

    #: The chain a window builds when nobody names one -- ``DEMO_CHAIN`` under a name that
    #: describes what it's FOR rather than the demo it started life as. The auto-run setting
    #: reads this off the class, not the module global, so a subclass could override it.
    DEFAULT_STEPS = DEMO_CHAIN

    def __init__(self, steps=DEFAULT_STEPS, parent=None):
        super().__init__(parent)
        register_builtin_devices()
        self.setWindowTitle(TITLE)

        self.cache = Cache()
        self.project = Project(title=TITLE)
        self.field = None
        self.layer = None
        # The NAME of the layer whose raster is currently on the canvas's own image item
        # -- set only at ``canvas.set_field``'s two call sites (``load_field``, ``_select_layer``),
        # never for a point layer's own selection (which leaves the canvas showing whatever raster
        # was there before -- see ``_select_layer``'s own comment). ``_apply`` compares a point
        # layer's own ``backproject`` result's ``_target`` against this to decide whether its
        # scatter overlay belongs on the raster actually on screen right now.
        self._displayed_layer_name: str | None = None
        # holder_map display: ``(id(h_map), h_map)`` for the exponent raster
        # currently shown IN PLACE of the raw field, or ``None`` when the raw field is up.
        # Identity-keyed like the canvas's own geometry cache (results are immutable; a new
        # compute mints a new array), with the strong ref so a reused id() can never alias a
        # dead array. See ``_sync_holder_raster``.
        self._holder_raster_ref: "tuple | None" = None
        #: (layer_id, field) when the active row's chain ends on a field stage (noise alone,
        #: a band row, a bus) -- what that row shows, and what a fork takes from it.
        self._active_field: "tuple | None" = None
        # Live band-reconstruction session: (key, BandReconstructor) -- the
        # mask-independent half of the inversion, cached on (field values, h_map) identity so
        # a histogram drag pays only ~20 ms/tick. Strong refs via the reconstructor itself.
        self._band_preview: "tuple | None" = None
        # Derived datasets (a reconstruction is a new dataset held in memory until exported, never a replacement for the one it is derived from): layer_id -> RasterField minted from the layer's own derived raster the moment
        # it lands. Session memory (the result cache holds the arrays either way); exportable
        # via File > Export Derived Raster; usable as a 3-D surface height source. Pruned with
        # the layer.
        self._derived_fields: dict = {}
        # Live-slice vector fidelity: when the drape is RGBA-baked (hillshade /
        # custom class colors) the per-tick LUT swap can't apply, so live ticks fall back to
        # a 300 ms-throttled FULL apply (tags + resync -- the same per-change path the
        # hillshade sliders already take). Pending payload + its timer.
        self._levels_vector_pending: "tuple | None" = None
        self._levels_vector_timer = QtCore.QTimer(self)
        self._levels_vector_timer.setSingleShot(True)
        self._levels_vector_timer.setInterval(300)
        self._levels_vector_timer.timeout.connect(self._flush_levels_vector)
        # The ACTIVE layer's own most recent result dict, stamped
        # at the top of ``_apply`` (the one place a result reaches the screen) -- reused by
        # ``_refresh_topology_panel`` to read ``result["topology"]``'s edge count without a
        # second ``resolve()`` call for what ``_apply`` was just handed anyway.
        self._active_result: dict | None = None
        # The currently open "Skeleton plot…" dialog, or ``None``
        # before the first click / after the last one was replaced -- rebuild-on-each-open (see
        # ``_on_skeleton_button_clicked``'s own docstring for the lifecycle decision), kept only
        # so ``_on_group_membership_changed`` has somewhere to push the OTHER half of the
        # bidirectional sync ("Scene selection changes re-draw the dialog's own
        # highlight").
        self._skeleton_dialog: SkeletonDialog | None = None
        # The multifractal-spectrum window, SkeletonDialog's lifecycle
        # exactly (one at a time, fresh instance per click, previous one closed first) -- kept
        # only so the next click can close its predecessor.
        self._multifractal_window: MultifractalWindow | None = None
        # The singularity-spectrum CONSTRUCTION window (SpectrumWindow), the fitter's
        # sibling -- same lifecycle. When both are open the fitter's scale window drives this one
        # (the workbook's cell-49-inherits-cell-47 coupling).
        self._spectrum_construction_window: SpectrumWindow | None = None
        # The decomposition grouping aids (ComponentsWindow), same lifecycle; unlike the windows
        # above it stays live -- every landing result re-syncs it (``_refresh_components_button``).
        self._components_window: ComponentsWindow | None = None
        # The rows list last pushed into ``self._topology_panel`` via
        # ``set_rows`` -- ``_refresh_topology_panel`` skips the call entirely when a freshly
        # computed rows list compares equal to this, so an UNRELATED ``_apply`` (a filter knob
        # moving, a scale scrub -- anything that reaches ``_refresh_topology_panel`` without the
        # link set itself having changed) never rebuilds the ``QListWidget`` out from under an
        # in-progress "select a row, then click Unlink" gesture (``QListWidget.clear()`` drops the
        # current-row selection unconditionally).
        self._topology_rows_shown: list[dict] | None = None
        # Row-for-row with the layer list, and the field each layer resolves against. A project
        # holds several layers over (potentially) several sources, so "the field" is a property of
        # the selected LAYER, not of the window -- ``self.field`` is just whichever one that is.
        self._row_layers: list = []
        self._fields: dict[int, object] = {}
        # id-> Layer, populated alongside _row_layers by add_layer_row -- what every LayerPanel
        # signal (layer_id-keyed, not row-index-keyed) resolves back to an actual Layer through.
        self._layer_by_id: dict[int, object] = {}
        # The OPEN floating inspectors, keyed by source_id (the inspector
        # is per SOURCE, not per layer) -- an entry exists exactly while its window does, which is
        # what makes ``layer_list``'s "I" toggle and the window's own close button one state.
        self._inspectors: dict[str, InspectorWindow] = {}
        # What every inspector this window knows about should come back as --
        # the OPEN ones re-captured from their live windows at each write, the closed ones kept
        # exactly as they were left, so closing a window remembers where it was rather than
        # forgetting it. Seeded from ``Settings.view_options["inspectors"]`` by the one-shot
        # restore (:meth:`_restore_inspectors`), which is the only place it is read from disk.
        self._inspector_states: dict[str, InspectorState] = {}
        # True only while that restore is putting windows back. A restore is not a gesture, and
        # every step of it (opening the window, seeding its rows, adopting its scale) would
        # otherwise write straight back over the very state being restored FROM -- the first
        # write, made before the saved scale had been applied, would persist a scale of 0.
        self._restoring_inspectors = False
        # source_id -> the last Renderable that reached the screen for a layer over that source.
        # The ONE store behind every inspector status reading and behind "push the last known
        # result" when a window opens after its result already landed: an inspector that had to
        # remember this itself would need the layer->source mapping it is deliberately denied.
        self._last_renderables: dict[str, object] = {}
        # Per-layer bypass/rack VIEW state, keyed by layer_id, written by ``_snapshot_recipe``
        # wherever ``_names``/``_params``/``_bypassed``/``_rack`` change for the CURRENT layer and
        # read by ``_select_layer`` in preference to deriving from ``layer.chain``:
        # that chain excludes bypassed steps entirely and has no ``rack`` field, so a fresh derive
        # would silently drop a bypassed step on every switch away and back.
        self._recipes: dict[int, list[dict]] = {}
        self.strips: WorkflowZone | None = None
        # The text ``_notify``'s "status"/"modal" tiers most recently posted
        # to the status bar -- tracked here rather than relied on ``QStatusBar.currentMessage()``
        # alone so a caller (or a test) has one obvious place to read it back from.
        self._status_message: str = ""
        self._set_recipe([name for name, _ in steps], [dict(params) for _, params in steps])
        self._scales: tuple[float, ...] = ()
        self._thread: QtCore.QThread | None = None
        self._worker: ResolveWorker | None = None
        # (layer_id, transform signature) the running/just-finished worker was dispatched for --
        # the shared worker thread serves either the ACTIVE layer or a background
        # arrangement-queue layer, so the landing handlers need to know WHOSE compute this was,
        # not only what it computed.
        self._dispatched = None
        # §5e hybrid: the CHAIN the active layer's worker was dispatched with
        # (``_dispatched_chain``), promoted to ``_committed_chain`` on a successful landing. While
        # a transform edit is pending, a filter edit re-resolves the committed transforms (cached)
        # + the CURRENT filters, so "filter edits stay live against the last run" is real rather
        # than dropped. Both None until the first successful active compute.
        self._dispatched_chain = None
        self._committed_chain = None
        # Arrangement multi-layer resolve. ``_arr_queue`` holds layer_ids still waiting
        # on the shared worker thread (never the active layer -- see ``_dispatch_next``'s own
        # priority for that); ``_arr_errors`` remembers the last error a layer's OWN compute
        # produced, keyed by the cache key it was for, so ``_sync_arrangement`` reports it
        # honestly instead of silently retrying it forever (mirrors ``_errored`` for the active
        # layer, generalised to every layer since the arrangement can show several at once).
        self._arr_queue: list[int] = []
        self._arr_errors: dict[int, tuple[str | None, str]] = {}
        # Reentrancy guard for _sync_arrangement -- see that method's
        # own docstring. _resolve_now (the filter path's own synchronous resolve)
        # resyncs the arrangement itself when flipped; _sync_arrangement's own tail
        # (_dispatch_next) can, in turn, call _resolve_now (the "resolve" pending branch) --
        # without this flag that nests BACK into _sync_arrangement mid-call, rebuilding entries
        # from underneath its own still-running outer call.
        self._arr_syncing = False
        #: Layers whose TRANSFORM edits await an explicit Run.
        self._pending_layers: set[int] = set()
        # The ``frame_mode`` the LAST ``_sync_arrangement`` call
        # actually used. ``_sync_arrangement(frame_mode=None)`` (its own default -- every call
        # site that does not itself know "vector or geo" right now calls it with NO argument at
        # all) resolves ``None`` to THIS attribute rather than to a plain ``False``, so an async
        # completion while the Vector tab is showing does not silently re-admit every layer under
        # the geo (``has_georeference``) gate. Set at the top of every ``_sync_arrangement`` call,
        # resolved or not; the two call sites that actually KNOW which view is being entered
        # (``_set_center_view``'s ``"vector"``/``"geo"`` branches) pass an explicit
        # ``True``/``False``, which is what seeds this for every later argument-less call to
        # reuse. See ``_sync_arrangement``'s own docstring for why a plain ``bool = False``
        # default was rejected: several argument-less call sites are monkeypatched with
        # ZERO-argument fakes by the arrangement-view test suite, which a positional/keyword
        # ``frame_mode`` at those call sites would break.
        self._arr_frame_mode: bool = False
        # Layer_ids ACTUALLY written by _on_groups_committed
        # during the current Commit-button click (never a locked refusal, never an empty groups
        # slice) -- drained by _on_commit_finished into exactly one deferred re-resolve/resync,
        # rather than one per layer mid-loop (see that method's own docstring for why the repeat
        # was the bug).
        self._commit_batch: list[int] = []
        # What the ACTIVE layer still needs once the shared thread frees up,
        # recorded at the moment its own dispatch attempt found the thread busy with someone
        # else's compute (``_start_worker_for``'s busy branch) or a filter change found it busy
        # (``_reresolve``'s busy branch) -- "compute" (dominates) or "resolve". Consumed,
        # unconditionally, at the top of ``_dispatch_next`` -- unconditionally because the cache
        # probe that runs after it cannot tell "already resolved" from "was marked computing and
        # never actually redispatched", which is exactly the failure this guards against.
        self._active_pending: str | None = None
        self._t0 = 0.0
        self._stage = ""                    # last progress stage, for the close-during-compute
        self._closing = False
        self._syncing = False               # re-ranging the transport, not a user scrub
        self._errored = False               # last transform raised => nothing is cached
        # User stop:
        # ``_user_stopped`` marks the NEXT cancelled-landing as a user stop (restore UI, no
        # redispatch) instead of a preempt; ``_stopped_sig`` suppresses _dispatch_next's
        # cache-probe branch for the stopped recipe -- without it the probe would instantly
        # restart the very compute the user killed (its tail is uncached). Any knob turn
        # changes the signature and the gate reopens; any explicit dispatch clears it.
        self._user_stopped = False
        self._stopped_sig = None
        # Lazily computed outputs (the M-Z reconstruction, say) run as jobs on the SAME worker
        # slot (``_thread``/``_worker``), one at a time, after any analysis, and land in their
        # own handlers, which never touch the analysis bookkeeping (``_dispatched``,
        # ``_stopped_sig``). ``_out_job`` is the job in flight as ``(layer, name, key)``;
        # ``_out_request`` the job the active layer's display waits for, dispatched once the
        # slot is free; ``_out_held`` ``(key, "stopped" | "failed")`` keeps a key the user
        # stopped (or that raised) from re-dispatching while that output stays on show;
        # ``_out_notes`` key -> its row reading; ``_out_keys`` layer_id -> the output keys
        # computed for it, which Pin (F) pins with the analysis keys.
        self._out_job: "tuple | None" = None
        self._out_request: "tuple | None" = None
        self._out_held: "tuple | None" = None
        self._out_notes: dict[str, str] = {}
        self._out_keys: dict[int, set] = {}
        self._out_t0 = 0.0
        # ``(layer_id, key)`` of the reconstruction Run asked for: until it lands (or is stopped,
        # superseded or fails) its row requests and draws it in place of the preview
        # (``_draws_preview``).
        self._recon_run: "tuple | None" = None
        self._stop_btn = QtWidgets.QToolButton()
        self._stop_btn.setText("■ Stop")
        self._stop_btn.setToolTip("stop the in-flight compute at its next stage boundary")
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._stop_compute)
        self.statusBar().addPermanentWidget(self._stop_btn)
        # True right after an inert (auto_run=False) load_field wiped _names/_params to empty --
        # see load_field's auto_run branch: the NEXT auto_run=True load reseeds from
        # DEFAULT_STEPS rather than resolving the empty chain that call left behind. Stays False
        # for a window legitimately constructed with ``steps=()`` (a deliberate blank chain,
        # e.g. the ROI-only smoke test), which must NOT get silently reseeded.
        self._inert_open = False
        # The center zone's own three-way state -- see
        # ``_CENTER_VIEWS``'s own comment. ``_center_view_restored`` guards the ONE-SHOT restore
        # of ``Settings.center_view`` inside ``load_field`` (its own comment explains why that
        # restore cannot happen any earlier than a window's first loaded layer) -- a second,
        # third, ... ``load_field`` call on the SAME window must never re-apply a persisted view
        # out from under a user who has since switched away on purpose.
        self._center_view: str = "raster"
        self._center_view_restored: bool = False
        # The SECOND one-shot, at the same point in ``load_field`` and for the
        # same reason (there is no source to open an inspector OVER any earlier than the first
        # layer landing). Deliberately a second flag rather than a share of the first: a user who
        # switched center views on purpose is a different situation from one who closed an
        # inspector, and one flag would make either gesture suppress the other's restore.
        self._inspectors_restored: bool = False
        # True from the moment the APPLICATION is on its
        # way out. An inspector closing is normally the user's own gesture and is remembered as
        # one (``open: False``); an inspector closed BY THE SHUTDOWN is not a gesture at all, and
        # remembering it as one is what made the acceptance sentence's own first clause -- "quit
        # with an inspector open" -- false: cmd-Q reaches a Qt app as ``QEvent::Quit`` on the
        # QApplication, ``QApplication::event`` answers it by ``close()``ing every top-level
        # window, and a floating inspector IS one (a ``Qt.Tool``), so the state went to disk as
        # closed on the way out and the next launch restored nothing.
        #
        # Set at the two points a shutdown is first VISIBLE to this window, because neither one
        # alone is enough: :meth:`eventFilter` sees the Quit event before Qt has closed anything
        # (the only ordering that is deterministic -- ``closeAllWindows`` walks
        # ``topLevelWidgets()`` in an order that genuinely varies run to run, measured), and
        # :meth:`closeEvent`'s closing branch covers the orderings and paths where this window
        # goes first. Closing ONLY the main window never needed either: a parented ``Qt.Tool``
        # window is HIDDEN with its parent, never closed, so that path never wrote anything.
        self._shutting_down: bool = False

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        # Zone tree: two nested QSplitters, not a fixed QVBox/QHBox pair. Every
        # boundary is a drag handle and every zone collapses to zero
        # (``setChildrenCollapsible`` below). ``self._work_split`` is the horizontal work area --
        # left panel | center stack | right-panel placeholder; ``self._main_split`` stacks that
        # work area over the bottom rack row. Sizes persist under the ``"work"``/``"main"`` keys
        # of ``Settings.splitter_sizes`` (``closeEvent`` below writes them) and restore here on
        # open.
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(160)      # drags open/shut
        panel_column = QtWidgets.QVBoxLayout(panel)
        self.open_button = QtWidgets.QPushButton("Open…")
        self.open_button.clicked.connect(self._on_open_clicked)
        # Browser above, layer list below, in a splitter -- the design's "Left panel, sectioned:
        # browser above, layer list below". Dependency injection, not an import: the
        # browser knows nothing of MainWindow, so its "WTMM standard" preset is handed in as data.
        self.browser = DeviceBrowser()
        self.browser.set_presets({"WTMM standard": MainWindow.DEFAULT_STEPS, **PRESETS})
        self.layer_list = LayerPanel(self.project)
        self.layer_list.layerSelected.connect(self._on_layer_selected)
        self.layer_list.hideToggled.connect(self._on_hide_toggled)
        self.layer_list.lockToggled.connect(self._on_lock_toggled)
        self.layer_list.freezeToggled.connect(self._on_freeze_toggled)
        self.layer_list.removeRequested.connect(self._on_remove_requested)
        self.layer_list.removeSourceRequested.connect(self._on_remove_source_requested)
        self.layer_list.renameRequested.connect(self._on_rename_requested)
        self.layer_list.refinedRunRequested.connect(self._on_refined_run)
        # The per-SOURCE "I" toggle opens/closes a floating inspector,
        # and ``resolved`` -- ``_apply``'s own "the ONE place a result reaches the screen" signal,
        # emitted unchanged -- fans out to whichever inspector is open over that result's source.
        # ONE connection, made here once, for N inspectors: no new compute path, no second
        # resolve, no polling.
        self.layer_list.inspectorToggled.connect(self._on_inspector_toggled)
        self.layer_list.sourceHideToggled.connect(self._on_source_hide_toggled)
        # Saved ROIs are rows of their dataset: selecting one makes it the region
        # the next tool runs on; its H hides only its outline.
        self.layer_list.roiSelected.connect(self._on_roi_row_selected)
        self.layer_list.roiHideToggled.connect(self._on_roi_hide_toggled)
        # The two deletes, ROI delete and several rows at once.
        self.layer_list.removeLayerOnlyRequested.connect(self._on_remove_layer_only_requested)
        self.layer_list.removeRoiRequested.connect(self._on_remove_roi_requested)
        self.layer_list.removeManyRequested.connect(self._on_remove_many_requested)
        # Derivative datasets: fork what a result shows; keep a temporary one as a file.
        self.layer_list.forkDerivativeRequested.connect(self._on_fork_derivative)
        self.layer_list.removeBandRequested.connect(self._on_remove_band_layer)
        self.layer_list.busRequested.connect(self._on_bus_requested)
        self.layer_list.stackRequested.connect(self._on_stack_requested)
        self.layer_list.busEditRequested.connect(self._on_bus_edit_requested)
        self.layer_list.saveDerivativeRequested.connect(self._on_save_derivative)
        # A result's output rows: a raster row is a radio over its step's Show, the edges row
        # hides the maxima drawing.
        self.layer_list.outputHideToggled.connect(self._on_output_hide_toggled)
        self.resolved.connect(self._fan_out_to_inspectors)
        self._left_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._left_split.addWidget(self.browser)
        self._left_split.addWidget(self.layer_list)
        # Reference layers: interpretation over the data, listed under the layers.
        self.reference_panel = ReferencePanel()
        self.reference_panel.visibilityToggled.connect(self._on_reference_visibility)
        self.reference_panel.zoomRequested.connect(self._on_reference_zoom)
        self.reference_panel.removeRequested.connect(self._on_reference_remove)
        self.reference_panel.setVisible(False)
        self._left_split.addWidget(self.reference_panel)
        self.roi_panel = RoiPanel()
        self.roi_panel.createRequested.connect(self._on_roi_create)
        # Saved ROIs; the active one is what a tool drop runs on.
        self._active_roi_id = None
        self._roi_scene_memo: dict = {}      # layer_id -> (result, roi field, scene result)
        # Canvas overlay deferred while the Vector view was up (_apply / flip-back). Set here
        # too: a fresh window flipping back before any landing would otherwise raise
        # AttributeError.
        self._canvas_overlay_dirty = False
        self.roi_panel.saveRequested.connect(self._on_roi_save)
        self.roi_panel.roiActivated.connect(self._on_roi_activated)
        self.roi_panel.childRequested.connect(self._on_roi_child_create)
        self.roi_panel.placeRequested.connect(self._on_roi_place_arm)
        self.roi_panel.valuesEdited.connect(self._on_roi_values_edited)
        self.roi_panel.closeRequested.connect(self._on_roi_panel_closed)
        # The per-layer display controls (EQSelect matrix: "layer display controls —
        # cheap table stakes") live in the right panel's Display section -- see
        # ``self.right_panel`` below. The left column holds Open, the browser/layers split and
        # the ROI panel.
        # The ROI… button is the panel's second entry point -- no drag
        # required, the numbers are typed. Same panel, same band-tracking, same Create/Child
        # buttons; the ⌘-drag remains the gestural way in.
        self.roi_tool_button = QtWidgets.QPushButton("ROI…")
        self.roi_tool_button.clicked.connect(self._on_roi_tool_clicked)
        panel_column.addWidget(self.open_button)
        panel_column.addWidget(self._left_split, 1)
        panel_column.addWidget(self.roi_tool_button)
        panel_column.addWidget(self.roi_panel)
        self.canvas = Canvas()
        self.canvas.roiDrawn.connect(self._on_roi_drawn)
        self.canvas.roiPlaced.connect(self._on_roi_placed)
        # The canvas re-derives both corner overlays whenever the camera moves, but the distance
        # LABEL is the one part it cannot: only the window knows the field's physical units. A
        # bound method, per the same rule the worker signals follow.
        self.canvas.viewChanged.connect(self._update_scale_bar)
        # Center zone: Ableton's session/arrangement Tab flip. Index 0 is the canvas; index 1 is
        # the arrangement view, added lazily on first Tab (``_toggle_center_view``) so a window
        # that never flips never even imports ``dynamix.shell.arrangement.view`` (or anything
        # ``view.py`` itself imports -- ``scene``, ``camera``). ``mask_row`` and
        # ``group_palette`` are two deliberate exceptions: both pure Qt, no pyvista of their own
        # (see each module's own docstring) -- and, unlike those others, both are HOSTED by
        # ``MainWindow`` rather than the view, so ``self._mask_row``/``self._group_palette``
        # below are built eagerly and unconditionally, same as every other right-panel control,
        # through their own always-present module-level imports. ``view_dialog`` is a THIRD such
        # exception, for the identical reason -- see ``self._view_dialog``'s own comment below.
        self._center_stack = QtWidgets.QStackedWidget()
        self._center_stack.addWidget(self.canvas)
        self._arrangement = None
        self._footprints: list = []   # dynamix.geo.footprints records (data browser)
        self._previews: dict = {}     # group_key -> (name, preview RasterField) draped on the world
        self._reference_layers: dict = {}   # ref_id -> geo.vectors.VectorLayer (native CRS)
        self._reference_inside: dict = {}   # ref_id -> (features inside this raster, total)
        self._reference_pixels: dict = {}   # ref_id -> [parts...] in the field's pixel frame
        # The warp caches: layer geometry is immutable once read, so each (layer, grid) pixel
        # warp and each layer's lon/lat / per-CRS form are computed once, not on every layer
        # switch (over a basin-wide shapefile that recompute is a visible pause per click).
        self._reference_pixel_cache: dict = {}   # (ref_id, georef sig) -> ([parts...], inside)
        self._reference_world_cache: dict = {}   # ref_id -> {"lonlat": [...], "native": {crs: [...]}}
        # The View dialog -- built lazily, the first time
        # ``ArrangementView.viewOptionsRequested`` fires (``_toggle_center_view``'s own first-build
        # branch wires that signal; it cannot be wired any earlier, since the button it comes from
        # does not exist before that), same lazy-once shape as ``self._arrangement`` itself. Unlike
        # the arrangement view, this dialog is pure Qt (no pyvista of its own -- see
        # ``view_dialog.py``'s own module docstring), so its CLASS is imported at module level,
        # same as ``MaskRow``/``GroupPalette``; only the INSTANCE waits for the first click.
        self._view_dialog = None
        # Third work-split slot: the selected dataset's controls, sectioned. It holds the Display
        # knobs; other sections are added through ``RightPanel.add_section`` without this file
        # needing to know their shape. ``styleChanged`` reaches ``_on_display_style_changed``
        # through one signal for every control. ``self._display_controls`` (name -> control
        # widget) is an ALIAS onto the panel's own registry, not a dict this file builds itself.
        self.right_panel = RightPanel(_DISPLAY_PARAMS)
        self.right_panel.styleChanged.connect(self._on_display_style_changed)
        self.right_panel.surfaceDialogRequested.connect(self._on_surface_dialog_requested)
        self.right_panel.levelsDialogRequested.connect(self._on_levels_dialog_requested)
        self.right_panel.bandDialogRequested.connect(self._on_band_dialog_requested)
        self._display_controls = self.right_panel._display_controls
        self._right_panel_host = self.right_panel      # kept for any code that still names it
        # The Reconstruction section: the output step's section="reconstruction" knobs, Run, Stop
        # and the reading of what the recon row draws (_sync_recon_section). Shown while the
        # active chain's output step declares such knobs.
        self._recon_panel = ReconstructionPanel()
        self._recon_panel.paramChanged.connect(self._on_recon_knob_changed)
        self._recon_panel.levelsChanged.connect(self._on_recon_levels_changed)
        self._recon_panel.runRequested.connect(self._on_recon_run)
        self._recon_panel.stopRequested.connect(self._stop_compute)
        self.right_panel.add_section(_RECON_TITLE, self._recon_panel)
        self.right_panel.apply_relevance({_RECON_TITLE: False})
        # The arrangement's mask row (``arrangement/mask_row.py`` --
        # widget itself unchanged), relocated out of ``ArrangementView`` into a view-scoped section
        # here -- built and hosted unconditionally, at window-construction time, not gated on
        # whether the user ever flips to the arrangement (mirrors the Display section just above:
        # both live in the panel regardless of which center-stack page is showing). Every edit
        # reaches ``ArrangementView.set_mask`` through ``_on_mask_row_changed``; on every flip-in,
        # ``_toggle_center_view`` also PUSHES the row's then-current values into the freshly
        # activated view, replacing the view's former self-read of its own (now-removed) row.
        # The multiband composite mixer (one row per band: R/G/B assignment, Solo, Mute).
        # Hosted like every section but VISIBLE only while the active dataset is a stack --
        # _sync_composite_panel shows/hides the hosting frame and pushes the spec (persisted
        # per layer as the ui.composite tag) onto the canvas.
        self.composite_panel = CompositePanel()
        self.composite_panel.compositeChanged.connect(self._on_composite_changed)
        self.right_panel.add_section("Composite", self.composite_panel)
        self._composite_frame = self.composite_panel.parentWidget()
        self._composite_frame.setVisible(False)
        self._mask_row = MaskRow()
        self._mask_row.maskChanged.connect(self._on_mask_row_changed)
        self.right_panel.add_section(
            "Display mask — hides, never removes", self._mask_row, view_scoped=True)
        # The arrangement's group palette + its Commit button
        # (``arrangement/group_palette.py`` -- widget itself unchanged), relocated out of
        # ``ArrangementView`` into a second view-scoped section here -- built and hosted
        # unconditionally, same reasoning as ``self._mask_row`` just above. Wrapped together in
        # one plain container widget (``add_section`` only ever takes one), palette on top,
        # Commit button under it, per the design.
        #
        # ``self._commit_button.clicked`` is wired to the wrapper ``_on_commit_button_clicked``
        # here, in ``__init__`` -- not connected straight to ``self._arrangement.commit`` inside
        # ``_toggle_center_view`` -- deliberately: the button is built and interactive from
        # construction time, same as the palette and the mask row, so it needs a no-op-before-
        # ``self._arrangement``-exists guard exactly like ``_on_mask_row_changed`` already has; a
        # direct connection deferred until the first flip would leave a pre-flip click silently
        # unwired instead of a documented no-op. ``ArrangementView.set_group_palette`` gets the
        # palette's REFERENCE once, in ``_toggle_center_view``'s own first-build branch (not
        # pushed on every flip like the mask row's VALUES are) -- see that method's own docstring.
        self._group_palette = GroupPalette()
        self._commit_button = QtWidgets.QPushButton("Commit")
        self._commit_button.clicked.connect(self._on_commit_button_clicked)
        groups_body = QtWidgets.QWidget()
        groups_layout = QtWidgets.QVBoxLayout(groups_body)
        groups_layout.setContentsMargins(0, 0, 0, 0)
        groups_layout.addWidget(self._group_palette)
        groups_layout.addWidget(self._commit_button)
        # Not view-scoped: the arrangement view's own pick gesture and the raster canvas's
        # click/box/lasso picking (wired just below) feed this exact same palette regardless of
        # which center-stack page is showing, so a divider line implying "only matters in
        # Vector/Globe" would misdescribe it -- "a selection must never mutate an invisible
        # widget". The section is drawn regardless of `_center_view` either way
        # (`RightPanel.add_section`'s `view_scoped` only ever adds a divider line -- see that
        # method's own docstring; there is no visibility-gating consumer anywhere), so the flag
        # sets the label only.
        self.right_panel.add_section("Groups", groups_body, view_scoped=False)
        # The RASTER canvas's own pick gesture (click/shift-click/
        # ⌥-lasso, and now box) feeds this SAME palette -- one shared
        # selection model regardless of which center view is showing. Wired unconditionally, at
        # construction time, the same eager reasoning as the palette/mask row/commit button just
        # above: both `self.canvas` and `self._group_palette` already exist by this point. No
        # `_center_view` gating is needed here -- the canvas only ever RECEIVES real mouse events
        # while it is the visible page of `_center_stack` (index 0, "raster"), so a pick can
        # never arrive while another view is showing in the first place.
        self.canvas.chainPicked.connect(self._on_canvas_chain_picked)
        self.canvas.chainsLassoed.connect(self._on_canvas_chains_lassoed)
        self.canvas.chainsBoxed.connect(self._on_canvas_chains_boxed)
        # "after every apply_picks, push sorted(selection
        # indices for the active layer) into canvas.set_selection_chains" -- `membershipChanged`
        # is `GroupPalette`'s own "always emits, whatever it changed" signal (its own docstring),
        # already fired by every `add_pick`/`apply_picks` call regardless of which gesture or
        # which center-stack page produced it, and already the identical signal the arrangement's
        # own selection highlight rides (`ArrangementView._on_membership_changed`, wired
        # separately in `_set_center_view`'s first-build branch) -- one more listener on it, not
        # a second, competing selection-plumbing path.
        self._group_palette.membershipChanged.connect(self._on_group_membership_changed)

        # User-asserted topology links between chains
        # (``topology/links.py``'s ``LinkStore``, living on ``self.project.user_links``). NOT
        # view-scoped, unlike Groups/Display mask just above -- a link names two chains by
        # ``(layer_id, transform, kind, obj_id)``, which means the same thing regardless of which
        # center-stack page is currently showing, so this section stays visible in all three.
        self._topology_panel = TopologyPanel()
        self._topology_panel.linkRequested.connect(self._on_topology_link_requested)
        self._topology_panel.unlinkRequested.connect(self._on_topology_unlink_requested)
        # Every ObjRef this slice ever builds is kind "line" (chains, whichever transform drew
        # them) in a 2-D pixel space (_on_topology_link_requested's own hard-coded "line"/2), so
        # the override combo's choices are fixed for the life of the window -- no per-pick
        # re-population needed, unlike a future slice that might link other kinds/dimensions.
        self._topology_panel.set_code_choices("line", "line", 2)
        self.right_panel.add_section("Topology", self._topology_panel)

        # The log-log coefficient-vs-scale plot (spec Mission). A single button, its own small
        # section (beside Topology, per the design) -- everything else lives in the dialog itself
        # (``skeleton_dialog.py``). Starts disabled; ``_refresh_skeleton_button`` (called
        # alongside ``_refresh_topology_panel`` at every one of ITS OWN call sites -- both read
        # the same ``self._active_result``) is what turns it on once a real result with chains
        # has landed.
        self._skeleton_button = QtWidgets.QPushButton("Skeleton plot…")
        self._skeleton_button.clicked.connect(self._on_skeleton_button_clicked)
        self.right_panel.add_section("Skeleton", self._skeleton_button)
        self._refresh_skeleton_button()

        # The interactive multifractal-spectrum fitter -- the Skeleton
        # section's shape exactly (a single button; everything else lives in its own floating
        # window, ``multifractal_window.py``, outside the center view). Gated on the PARTITION
        # TABLES rather than chains: a finest-scale preview stamps ``hd_std=None`` and a
        # points-layer result carries neither table.
        self._spectrum_button = QtWidgets.QPushButton("Multifractal spectrum…")
        self._spectrum_button.clicked.connect(self._on_spectrum_button_clicked)
        # The construction window's own button, same section -- gated more loosely
        # (an h_map-bearing holder result can show the microcanonical histogram with no tables).
        self._dh_button = QtWidgets.QPushButton("D(h) construction…")
        self._dh_button.clicked.connect(self._on_dh_button_clicked)
        spectrum_box = QtWidgets.QWidget()
        spectrum_lay = QtWidgets.QVBoxLayout(spectrum_box)
        spectrum_lay.setContentsMargins(0, 0, 0, 0)
        spectrum_lay.addWidget(self._spectrum_button)
        spectrum_lay.addWidget(self._dh_button)
        # WTMMM angle statistics (Arneodo, Decoster & Roux 2000): P_a(A) across scales, the
        # gradient plane, M per angle sector -- pixel frame or bearings.
        self._anisotropy_button = QtWidgets.QPushButton("Anisotropy (WTMMM angles)…")
        self._anisotropy_button.clicked.connect(self._on_anisotropy_button_clicked)
        self._anisotropy_window: AnisotropyWindow | None = None
        spectrum_lay.addWidget(self._anisotropy_button)
        self.right_panel.add_section("Spectrum", spectrum_box)
        self._refresh_spectrum_button()
        self._refresh_dh_button()
        self._refresh_anisotropy()
        self._refresh_panel_relevance()

        # The grouping aids of a decomposition (ssa2d / tucker): thumbnails, shares and
        # w-correlations in a floating window; picking thumbnails sets the tool's Group knob.
        self._components_button = QtWidgets.QPushButton("Components…")
        self._components_button.clicked.connect(self._on_components_button_clicked)
        self.right_panel.add_section("Decomposition", self._components_button)
        self._refresh_components_button()

        # The transect list -- NOT view-scoped (a drawn
        # transect is project-level state, meaningful to review/delete/plot regardless of which
        # center-stack page happens to be showing; only the DRAWING gesture itself is Raster-
        # canvas-only, the design's own "canvas only, v1" scoping). Wired unconditionally, at
        # construction time, same eager reasoning as every other right-panel section: `self.canvas`
        # already exists by this point.
        self._transect_panel = TransectPanel()
        self._transect_panel.recordsChanged.connect(self._on_transect_records_changed)
        self._transect_panel.selectionChanged.connect(self._on_transect_selection_changed)
        self._transect_panel.plotRequested.connect(self._on_transect_plot_requested)
        self.right_panel.add_section("Transect", self._transect_panel)
        self._refresh_panel_relevance()
        self.canvas.transectDrawn.connect(self._on_canvas_transect_drawn)

        self._work_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._work_split.setObjectName("work")
        self._work_split.addWidget(panel)
        self._work_split.addWidget(self._center_stack)
        self._work_split.addWidget(self._right_panel_host)
        self._work_split.setStretchFactor(1, 1)          # the center stack takes the growth
        self._work_split.setChildrenCollapsible(True)     # Qt default, pinned explicitly

        bottom = QtWidgets.QWidget()
        # No longer fixed height (was ``setFixedHeight(BOX_HEIGHT + 28)``) -- the rack row now
        # drags open/shut like every other zone; the QScrollArea inside still keeps the strips
        # from being squeezed below the height they asked for.
        bottom.setMinimumHeight(0)
        bottom_row = QtWidgets.QHBoxLayout(bottom)
        bottom_row.setContentsMargins(4, 4, 4, 0)
        self.transport = Transport(1, self._scale_reading)
        self.transport.setFixedWidth(_TRANSPORT_WIDTH)
        self.transport.scaleChanged.connect(self._on_scale_changed)
        # This transport IS the master track. A SECOND connection,
        # added beside the first rather than in place of it -- the first one is the ACTIVE layer's
        # own filter path and nothing about it changes. Order matters and is the one it reads:
        # `_on_scale_changed` re-resolves and lands the new picture in the active source's
        # inspector first, then every FOLLOWING inspector adopts the index.
        self.transport.scaleChanged.connect(self._drive_follower_inspectors)
        # Run: transforms never auto-run -- edits mark the block pending and this
        # button (or ⌘↩) dispatches. Enabled only while something is pending.
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setToolTip("Run the transform chain (⌘↩) — transform edits wait here")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_transforms)
        bottom_row.addWidget(self.run_button)
        bottom_row.addWidget(self.transport)
        self._strip_host = QtWidgets.QWidget()
        self._strip_layout = QtWidgets.QHBoxLayout(self._strip_host)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        # The rack scrolls sideways (as Ableton's does) rather than setting the window's minimum
        # width: six strips of generated controls demand ~2500 px, and without this the window
        # cannot be made narrower than that on any screen.
        self._strip_scroll = QtWidgets.QScrollArea()
        self._strip_scroll.setWidgetResizable(True)
        self._strip_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._strip_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._strip_scroll.setWidget(self._strip_host)
        bottom_row.addWidget(self._strip_scroll, 1)

        self._main_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._main_split.setObjectName("main")
        self._main_split.addWidget(self._work_split)
        self._main_split.addWidget(bottom)
        self._main_split.setStretchFactor(0, 1)           # the work area takes the growth
        self._main_split.setChildrenCollapsible(True)      # Qt default, pinned explicitly

        # Initial sizes: a persisted list restores only when its length still matches this
        # splitter's pane count (a settings file from before a pane was added/removed is
        # treated as absent rather than applied partially/wrongly). Otherwise a first-run
        # default -- left panel near its old fixed width, right panel opened to
        # ``_RIGHT_PANEL_WIDTH`` now that it hosts the Display section (a FRESH window must not hide the knobs moved here), rack near
        # its old fixed height. A settings file that already has a stored size always wins --
        # this branch only ever runs for a window that has never persisted one.
        saved_sizes = load_settings().splitter_sizes
        work_sizes = saved_sizes.get("work")
        if isinstance(work_sizes, list) and len(work_sizes) == self._work_split.count():
            self._work_split.setSizes([int(v) for v in work_sizes])
        else:
            self._work_split.setSizes([_PANEL_WIDTH, 10_000, _RIGHT_PANEL_WIDTH])
        main_sizes = saved_sizes.get("main")
        if isinstance(main_sizes, list) and len(main_sizes) == self._main_split.count():
            self._main_split.setSizes([int(v) for v in main_sizes])
        else:
            self._main_split.setSizes([10_000, BOX_HEIGHT + 28])

        # The three-view switcher -- a slim strip ABOVE the zone
        # tree (never inside it: this is chrome for CHOOSING which zone shows, not a zone of its
        # own, so it gets no splitter handle). Three checkable ``QToolButton``s in an exclusive
        # ``QButtonGroup``, right-aligned; a real click calls :meth:`_set_center_view` directly
        # (Tab/:meth:`_cycle_center_view` drive the identical method). No literal color/font here
        # -- the buttons pick up the app-wide ``QWidget`` rule from ``theme.py``'s generated QSS,
        # same as every other bare widget in this window (the Theme Rule's own carve-out is for
        # DATA colors, not chrome, and this is chrome).
        switcher_strip = QtWidgets.QWidget()
        switcher_row = QtWidgets.QHBoxLayout(switcher_strip)
        switcher_row.setContentsMargins(4, 2, 4, 2)

        # The selection-mode row -- Click | Box |
        # Lasso | Transect, EQSelect's own names -- LEFT-aligned in this SAME strip (the design's "a strip near the view switcher"; the stretch just below pushes the view switcher
        # itself to the right, so both live in one row without a splitter handle of their own).
        # Same construction shape as the view switcher just below, one level down: checkable
        # ``QToolButton``s in an exclusive ``QButtonGroup``, a real click calling
        # :meth:`set_selection_mode` directly (hotkeys `c`/`v`, below, drive the identical
        # method), kept in sync by the identical reentrancy-guarded pattern
        # (:meth:`_sync_selection_mode_row` mirrors :meth:`_sync_view_switcher`).
        self._selection_mode = "click"
        self._selection_mode_group = QtWidgets.QButtonGroup(self)
        self._selection_mode_group.setExclusive(True)
        self._selection_mode_buttons: dict[str, QtWidgets.QToolButton] = {}
        for mode, label in (("click", "Click"), ("box", "Box"), ("lasso", "Lasso"),
                             ("transect", "Transect")):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, m=mode: self.set_selection_mode(m))
            self._selection_mode_group.addButton(button)
            self._selection_mode_buttons[mode] = button
            switcher_row.addWidget(button)
        self._selection_mode_buttons[self._selection_mode].setChecked(True)
        self.canvas.set_selection_mode(self._selection_mode)     # sync the canvas's own mirror

        switcher_row.addStretch(1)
        self._view_switcher = switcher_strip
        self._view_switcher_group = QtWidgets.QButtonGroup(self)
        self._view_switcher_group.setExclusive(True)
        self._view_switcher_buttons: dict[str, QtWidgets.QToolButton] = {}
        for view, label in (("raster", "Raster"), ("vector", "Vector"), ("geo", "Globe")):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, v=view: self._set_center_view(v))
            self._view_switcher_group.addButton(button)
            self._view_switcher_buttons[view] = button
            switcher_row.addWidget(button)
        self._view_switcher_buttons[self._center_view].setChecked(True)
        outer.addWidget(switcher_strip)

        outer.addWidget(self._main_split)
        self.setCentralWidget(central)

        # Window-scoped, so it fires wherever focus sits inside the window but never steals Space
        # from another window of the same app.
        self._key_run = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self)
        self._key_run.activated.connect(self._run_transforms)
        self._space = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        self._space.setContext(QtCore.Qt.WindowShortcut)
        self._space.activated.connect(self._toggle_transport)

        # `c`/`v` selection-mode hotkeys -- the SAME
        # mechanism as `self._space` just above, not Tab's bespoke app-level `eventFilter`. Tab
        # needs that heavier machinery only because Qt's OWN focus-chain navigation consumes a
        # real Tab keypress before it ever reaches a `QShortcut`/`keyPressEvent` at all
        # (`eventFilter`'s own docstring); `c`/`v`, like Space, are plain, unclaimed keys with no
        # such built-in consumer, so a `WindowShortcut`-context `QShortcut` is already the
        # simplest thing that works: it fires from anywhere focus sits inside this window, and
        # Qt's own `ShortcutOverride` mechanism defers to a focused `QLineEdit`'s normal text
        # entry first (proven by `self._space` already coexisting with renaming a layer, which
        # needs literal spaces) -- so typing "c" or "v" into a rename field or a numeric box
        # keeps working exactly as typing a space already does, with no extra guard needed here.
        self._key_click_mode = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C), self)
        self._key_click_mode.setContext(QtCore.Qt.WindowShortcut)
        self._key_click_mode.activated.connect(lambda: self.set_selection_mode("click"))

        self._key_cycle_mode = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_V), self)
        self._key_cycle_mode.setContext(QtCore.Qt.WindowShortcut)
        self._key_cycle_mode.activated.connect(self._cycle_selection_mode)

        # Esc cancels an in-progress transect first click
        # -- the SAME `WindowShortcut`-context idiom as `c`/`v`/Space above (fires from anywhere
        # focus sits inside this window; `Canvas.cancel_transect` is a harmless no-op when nothing
        # is in progress, so no mode guard is needed here).
        self._key_cancel_transect = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        self._key_cancel_transect.setContext(QtCore.Qt.WindowShortcut)
        self._key_cancel_transect.activated.connect(self.canvas.cancel_transect)

        # Transect delete's undo STACK -- ⌘Z (``QKeySequence.Undo`` resolves to the
        # platform's own undo chord; Cmd+Z on macOS, the only target per the project's). Bound at the
        # WINDOW level, same idiom as every other hotkey here, rather than on `TransectPanel`
        # itself: a `QShortcut` scoped to that widget alone would only fire while it happens to
        # hold keyboard focus, and undo needs to work regardless of where the user's focus is.
        self._key_undo_transect = QtGui.QShortcut(QtGui.QKeySequence.Undo, self)
        self._key_undo_transect.setContext(QtCore.Qt.WindowShortcut)
        self._key_undo_transect.activated.connect(self._transect_panel.undo)

        # Rack removal's own undo. The design's literal
        # wording ("removed rack X — ⌘Z restores") names the SAME ``QKeySequence.Undo`` chord already bound above -- but a second ``QShortcut`` on that identical key sequence, in
        # the identical ``WindowShortcut`` context, does not stack: Qt considers two enabled
        # shortcuts sharing a key sequence AMBIGUOUS and fires neither ``activated()`` handler (only
        # ``activatedAmbiguously()``, which nothing here connects to) -- confirmed against Qt's own
        # shortcut-dispatch docs, not assumed. Rather than invent a focus/context routing rule to
        # arbitrate between two independent undo stacks sharing one chord (the design's other
        # offered option -- real complexity for a case a second chord dissolves for free), rack
        # removal binds Shift+⌘Z instead: still a standard, discoverable macOS chord (the same one
        # native apps use for Redo -- repurposed here for a second, INDEPENDENT undo track, not
        # actual redo semantics), distinct from the plain ⌘Z, and named explicitly in the
        # removal's own status message ("⇧⌘Z restores", ``workflow_zone.py``'s
        # ``_on_rack_remove_requested``) so the binding is discoverable without this comment. Built
        # from the literal chord string rather than ``QKeySequence.Redo`` -- the STANDARD KEY's own
        # name would misleadingly suggest this shortcut performs a real redo.
        #
        # Bound through a thin wrapper (:meth:`_undo_rack_removal`), NOT
        # ``self.strips.undo_removal`` directly -- unlike ``self._transect_panel`` (built once, in
        # ``__init__``, and never replaced), ``self.strips`` is torn down and REBUILT by
        # :meth:`_build_strips` on every layer switch/reseed, which would leave a direct connection
        # bound to a deleted ``WorkflowZone``. The wrapper re-reads ``self.strips`` at ACTIVATION
        # time instead, so it always reaches whichever zone is live right now.
        self._key_undo_rack = QtGui.QShortcut(QtGui.QKeySequence("Shift+Ctrl+Z"), self)
        self._key_undo_rack.setContext(QtCore.Qt.WindowShortcut)
        self._key_undo_rack.activated.connect(self._undo_rack_removal)

        # Application-level, not window-scoped: a real Tab press goes to whichever DESCENDANT
        # currently has focus, never to this window object (see ``eventFilter``'s docstring) --
        # only a filter installed on ``qApp`` sees every key event regardless of which widget
        # is the actual target. Removed again in ``closeEvent`` so a closed window's filter
        # never lingers on ``qApp`` to catch key events meant for some other, later window.
        QtWidgets.QApplication.instance().installEventFilter(self)

        # File menu: the .dynamix save/load machinery (model/projectfile.py) existed fully built
        # and tested with NO menu action anywhere -- the EQSelect functionality matrix's cheapest
        # finding. Open/Save/Save As are the wiring, nothing more; the dialogs are file pickers,
        # the one dialog class DESIGN.md's "minimal file-open" already admits.
        self._project_path = None
        file_menu = self.menuBar().addMenu("File")
        open_action = QtGui.QAction("Open Project…", self)
        open_action.triggered.connect(self._on_open_project_clicked)
        file_menu.addAction(open_action)
        save_action = QtGui.QAction("Save Project", self)
        save_action.setShortcut(QtGui.QKeySequence.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)
        save_as_action = QtGui.QAction("Save Project As…", self)
        save_as_action.triggered.connect(self._on_save_project_as_clicked)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        # Footprint browser (the first in-app data-browser step):
        # scan a folder's raster HEADERS, outline every dataset on the world, right-click to
        # import one. The scanned folders persist (Settings.footprint_folders) and re-scan on the
        # next window open -- see _restore_footprint_folders.
        scan_action = QtGui.QAction("Scan Folder for Footprints…", self)
        scan_action.triggered.connect(self._on_scan_footprints_clicked)
        file_menu.addAction(scan_action)
        ref_action = QtGui.QAction("Open Reference Layer…", self)
        ref_action.triggered.connect(self._on_open_reference_clicked)
        file_menu.addAction(ref_action)
        overview_action = QtGui.QAction("Open Full Extent (Overview)…", self)
        overview_action.triggered.connect(self._on_open_overview_clicked)
        file_menu.addAction(overview_action)
        export_chains_action = QtGui.QAction("Export Chains (.npz)…", self)
        export_chains_action.triggered.connect(self._on_export_chains_clicked)
        file_menu.addAction(export_chains_action)
        export_derived_action = QtGui.QAction("Export Derived Raster (.npz)…", self)
        export_derived_action.triggered.connect(self._on_export_derived_clicked)
        file_menu.addAction(export_derived_action)

        # Minimal Settings menu: one checkable action, read fresh at every open rather than
        # cached on the window, so a setting changed in one window's menu is honoured the next
        # time ANY window opens a file -- including this one, without a restart.
        view_menu = self.menuBar().addMenu("View")
        # ⌘0 and F ("frame"): WindowShortcut like Space / c / v, so a focused text field keeps
        # its own typing first.
        self._zoom_action = QtGui.QAction("Zoom to Dataset", self)
        self._zoom_action.setShortcuts([QtGui.QKeySequence("Ctrl+0"),
                                        QtGui.QKeySequence(QtCore.Qt.Key_F)])
        self._zoom_action.setShortcutContext(QtCore.Qt.WindowShortcut)
        self._zoom_action.triggered.connect(self._zoom_to_dataset)
        view_menu.addAction(self._zoom_action)
        settings_menu = self.menuBar().addMenu("Settings")
        self._auto_run_action = QtGui.QAction("Auto-run transforms", self, checkable=True)
        self._auto_run_action.setChecked(load_settings().auto_run_wtmm)
        self._auto_run_action.toggled.connect(self._on_auto_run_toggled)
        settings_menu.addAction(self._auto_run_action)
        # The master compute-engine choice: three exclusive radio actions.
        # Applied immediately (set_default_engine) AND persisted -- engine choice never touches
        # cache keys (backends agree to float32 tolerance), so flipping it mid-session keeps
        # every stage cache valid and only changes who does the next FFT.
        # Engine + precision cover EVERY FFT in the app, persisted and applied at STARTUP
        # (restart to apply) through fft_policy; the immediate apply described above is the
        # legacy path (``_on_engine_selected_legacy``), which the menu's own values bypass.
        engine_menu = settings_menu.addMenu("Compute engine")
        engine_group = QtGui.QActionGroup(self)
        engine_group.setExclusive(True)
        saved = load_settings()
        self._engine_actions = {}
        for value, label in (("auto", "Auto (MLX on Apple Silicon, else FFTW3)"),
                             ("mlx", "MLX (Apple GPU, 32-bit)"),
                             ("fftw", "FFTW3 (CPU)")):
            action = QtGui.QAction(label, self, checkable=True)
            action.setChecked(value == saved.compute_engine)
            action.triggered.connect(
                lambda _checked=False, v=value: self._on_engine_selected(v))
            engine_group.addAction(action)
            engine_menu.addAction(action)
            self._engine_actions[value] = action
        precision_menu = settings_menu.addMenu("FFT precision")
        precision_group = QtGui.QActionGroup(self)
        precision_group.setExclusive(True)
        self._precision_actions = {}
        for value, label in ((32, "32-bit (single)"), (64, "64-bit (double, FFTW3)")):
            action = QtGui.QAction(label, self, checkable=True)
            action.setChecked(value == saved.compute_precision)
            action.triggered.connect(
                lambda _checked=False, v=value: self._on_precision_selected(v))
            precision_group.addAction(action)
            precision_menu.addAction(action)
            self._precision_actions[value] = action
        from dynamix.core import fft_policy
        from dynamix.core.wtmm_backend import set_default_engine
        fft_policy.configure(saved.compute_engine, saved.compute_precision)
        set_default_engine(None)          # the wavelet transform follows the FFT policy
        self._restore_footprint_folders()

    # -- reference layers ------------------------------------------------------------------
    def _on_open_overview_clicked(self) -> None:
        """The WHOLE raster, decimated to display resolution -- ``Settings.open_window_px`` windows
        every fresh native-resolution open, and the full BOEM west is 800 Mpx, so full extent
        means the OVERVIEW: :func:`~dynamix.geo.footprints.overview_field` at a 4096-px long
        side, provenance stamped ``overview``/``preview`` and ``@ov`` in the name. Context and
        display only -- analysis wants native pixels (the no-resampling rule), so run WTMM on a
        windowed open or an ROI, never on this."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open full extent (decimated overview)", "", "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        # The display PICTURE, not the overview dataset
        # (overview_field stays for the footprint browser's previews) -- one coordinate
        # system, file pixels; analysis runs on saved ROIs, which read native pixels.
        from dynamix.roi.picture import read_picture

        try:
            field = read_picture(path)
        except Exception as exc:
            self._notify(f"Could not open {Path(path).name}: {exc}", "status")
            return
        self.load_field(field, path)
        f = int(field.provenance.get("display_stride", 1))
        self._notify(f"{Path(path).name}: full extent at 1/{f} resolution — display only; "
                     f"save an ROI and run tools on it (native pixels)", "status")

    def _on_open_reference_clicked(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open reference layers", "",
            "Shapefiles, layer files, shapefile zips (*.shp *.lyr *.zip);;All files (*)")
        if paths:
            self._open_reference_layers(paths)

    def _open_reference_layers(self, paths) -> None:
        """Register + read vector files as reference layers and show them everywhere."""
        opened = 0
        for path in _expand_reference_paths(paths):
            try:
                layer = read_shapefile(path)
            except (OSError, ValueError) as exc:
                self._notify(f"Could not open {Path(path).name}: {exc}", "status")
                continue
            rec = self.project.add_reference_layer(str(Path(path).resolve()), name=layer.name)
            self._drop_reference_caches(rec.ref_id)   # a re-open re-reads the file: fresh warps
            self._reference_layers[rec.ref_id] = layer
            opened += 1
        self.reference_panel.set_records(self.project.reference_layers)
        self._push_reference_layers()
        n = len(self.project.reference_layers)
        counts = [(self._reference_layers[r].name, a, b)
                  for r, (a, b) in self._reference_inside.items() if r in self._reference_layers]
        if len(counts) > 6:                       # a whole package: name only the ones that show
            empty = sum(1 for _, a, _b in counts if a == 0)
            counts = [c for c in counts if c[1] > 0]
            tail = [f"{empty} with none"] if empty else []
        else:
            tail = []
        inside = ", ".join([f"{n} {a}/{b}" for n, a, b in counts] + tail)
        self._notify(f"{opened} opened — {n} reference layer{'s' if n != 1 else ''}"
                     + (f" · features inside this raster: {inside}" if inside else "")
                     + "; tick/untick in the list below the layers", "status")

    def _reload_reference_layers(self) -> None:
        """After a project opens: re-read every record's file; a missing file keeps its record
        (the project still names it) and is reported, never fatal."""
        self._reference_layers = {}
        self._reference_inside = {}
        self._reference_pixels = {}
        self._reference_pixel_cache = {}
        self._reference_world_cache = {}
        missing = []
        for rec in self.project.reference_layers:
            try:
                self._reference_layers[rec.ref_id] = read_shapefile(rec.path)
            except (OSError, ValueError):
                missing.append(Path(rec.path).name)
        self.reference_panel.set_records(self.project.reference_layers)
        self._push_reference_layers()
        if missing:
            self._notify("Reference layer file(s) not found: " + ", ".join(missing)
                         + " — relocate them and reopen the project", "status")

    def _on_reference_zoom(self, ref_id: str) -> None:
        """Double-click on a Reference row: fit the view to the layer's whole extent -- the
        layers are always drawn whole. Canvas from its own pixel-frame item; world from the
        scene's actor."""
        item = self.canvas.reference_items.get(ref_id)
        if item is not None:
            x, y = (item.getData() if hasattr(item, "getData") else (item.data["x"], item.data["y"]))
            x, y = np.asarray(x, float), np.asarray(y, float)
            if np.isfinite(x).any():
                self.canvas.view.setRange(xRange=(float(np.nanmin(x)), float(np.nanmax(x))),
                                          yRange=(float(np.nanmin(y)), float(np.nanmax(y))), padding=0.05)
        if self._arrangement is not None:
            self._arrangement.zoom_to_reference(ref_id)

    def _on_reference_visibility(self, ref_id: str, visible: bool) -> None:
        for rec in self.project.reference_layers:
            if rec.ref_id == ref_id:
                rec.visible = bool(visible)
        self.canvas.set_reference_visible(ref_id, visible)
        if self._arrangement is not None:
            self._arrangement.set_reference_visible(ref_id, visible)

    def _on_reference_remove(self, ref_id: str) -> None:
        """Unload a reference layer everywhere (panel, canvas, world). The FILE is never
        touched -- removing forgets the layer, and re-opening the file brings it back."""
        layer = self._reference_layers.pop(ref_id, None)
        removed = self.project.remove_reference_layer(ref_id)
        if layer is None and not removed:
            return
        self._reference_pixels.pop(ref_id, None)
        self._reference_inside.pop(ref_id, None)
        self._drop_reference_caches(ref_id)
        self.reference_panel.set_records(self.project.reference_layers)
        self._push_reference_layers()
        name = layer.name if layer is not None else ref_id
        self._notify(f"{name}: removed — the file on disk is untouched", "status")

    def _drop_reference_caches(self, ref_id: str) -> None:
        """Forget every cached warp of one reference layer (its geometry was re-read or the
        layer removed). The pixel cache is keyed (ref_id, grid), so this sweeps its keys."""
        self._reference_world_cache.pop(ref_id, None)
        for key in [k for k in self._reference_pixel_cache if k[0] == ref_id]:
            del self._reference_pixel_cache[key]

    def _push_reference_layers(self) -> None:
        """Canvas: layers in the current field's pixel frame; world: lon/lat (+ the field's CRS
        for the Vector view). Called on open and whenever the field changes. Each (layer, grid)
        warp lands in ``_reference_pixel_cache`` and is reused on later switches to any field
        with the same georeference -- the geometry is immutable, so only the grid can change."""
        records = {r.ref_id: r for r in self.project.reference_layers}
        canvas_entries = []
        if self.field is not None:
            sig = _georef_signature(self.field)
            for ref_id, layer in self._reference_layers.items():
                rec = records.get(ref_id)
                if rec is None:
                    continue
                cached = self._reference_pixel_cache.get((ref_id, sig))
                if cached is None:
                    try:
                        px = to_field_pixels(layer, self.field)
                    except Exception as exc:             # a CRS rasterio cannot transform: say so
                        self._notify(f"{layer.name}: cannot place on this raster ({exc})",
                                     "status")
                        continue
                    parts = [f.parts for f in px.features]
                    # How much of the layer this raster can show at all (a 12 km window holds 3
                    # of BOEM's 20 980 seep polygons).
                    ny, nx = native_shape(self.field)  # FILE pixels (a picture's own extent)
                    n_in = sum(1 for f in px.features if any(
                        np.any((pt[:, 0] >= -0.5) & (pt[:, 0] <= nx - 0.5) & (pt[:, 1] >= -0.5) & (pt[:, 1] <= ny - 0.5))
                        for pt in f.parts if len(pt)))
                    cached = (parts, (n_in, len(px.features)))
                    self._reference_pixel_cache[(ref_id, sig)] = cached
                canvas_entries.append({"ref_id": ref_id, "name": layer.name, "kind": layer.kind,
                                       "color": rec.color, "visible": rec.visible,
                                       "features": cached[0]})
                self._reference_inside[ref_id] = cached[1]
                self._reference_pixels[ref_id] = cached[0]
        self.canvas.set_reference_layers(canvas_entries)
        if self._arrangement is not None:
            self._arrangement.set_reference_layers(self._scene_reference_entries())

    def _scene_reference_entries(self) -> list:
        records = {r.ref_id: r for r in self.project.reference_layers}
        field_crs = (getattr(self.field, "provenance", {}) or {}).get("crs") if self.field is not None else None
        entries = []
        for ref_id, layer in self._reference_layers.items():
            rec = records.get(ref_id)
            if rec is None:
                continue
            # Cached like the pixel warp above: lon/lat depends only on the layer, the native
            # form only on (layer, CRS); a failure caches as its empty/None result -- it would
            # fail identically on every push.
            world = self._reference_world_cache.get(ref_id)
            if world is None:
                try:
                    lonlat = [f.parts for f in to_lonlat(layer).features]
                except Exception:
                    lonlat = []
                world = {"lonlat": lonlat, "native": {}}
                self._reference_world_cache[ref_id] = world
            native = None
            if field_crs:
                key = str(field_crs)
                if key not in world["native"]:
                    try:
                        world["native"][key] = [f.parts
                                                for f in to_crs(layer, key).features]
                    except Exception:
                        world["native"][key] = None
                native = world["native"][key]
            entries.append({"ref_id": ref_id, "name": layer.name, "kind": layer.kind, "color": rec.color,
                            "visible": rec.visible, "lonlat": world["lonlat"], "native": native,
                            "pixels": self._reference_pixels.get(ref_id)})
        return entries

    # -- footprint browser -----------------------------------------------------------------
    def _on_scan_footprints_clicked(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Scan folder for raster footprints")
        if not folder:
            return
        self._scan_footprint_folder(folder, persist=True)
        if self._center_view == "raster":
            self._set_center_view("geo")        # footprints live on the world, not the canvas

    def _restore_footprint_folders(self) -> None:
        """Re-scan every remembered folder (headers only, milliseconds); a vanished folder is
        skipped silently -- a preference cannot be a fault."""
        for folder in load_settings().footprint_folders:
            if Path(folder).is_dir():
                self._scan_footprint_folder(folder, persist=False)

    def _scan_footprint_folder(self, folder: str, *, persist: bool) -> None:
        try:
            found = scan_footprints(folder)
        except (OSError, ValueError) as exc:
            self._notify(f"Footprint scan failed: {exc}", "status")
            return
        known = {f.path for f in self._footprints}
        added = [f for f in found if f.path not in known]
        self._footprints.extend(added)
        if persist:
            folders = load_settings().footprint_folders
            if folder not in folders:
                update_settings(footprint_folders=[*folders, folder])
        if self._arrangement is not None:
            self._arrangement.set_footprints(self._footprints)
            self._arrangement.set_previews(list(self._previews.values()))
            self._arrangement.set_reference_layers(self._scene_reference_entries())
        self._notify(f"{len(found)} footprints in {folder} ({len(added)} new, "
                     f"{len(self._footprints)} total) — right-click one on the Globe to import",
                     "status")

    def _footprint_menu(self, hits) -> QtWidgets.QMenu:
        """The Import menu for a right-click: one entry per raster under the cursor; rasters
        that belong together (:func:`group_key` -- an ASTER granule's VNIR + SWIR bands + QA
        planes by granule id, a GDEM tile's ``_dem`` + ``_num`` by identical footprint) fold
        into one submenu, a "little arrow" so overlapping datasets at the same spot stay one gesture away. Submenus are
        titled by :func:`scene_label` (an ASTER granule shows its acquisition date) and ordered
        newest first, undated groups after; entries inside show only what differs
        (:func:`band_label`) plus dtype and size."""
        menu = QtWidgets.QMenu(self)
        groups: dict = {}
        for fp in hits:
            groups.setdefault(group_key(fp), []).append(fp)

        def _import(fp):
            return lambda _checked=False, path=fp.path: self.open_path(path)

        labelled = []
        for members in groups.values():
            label, when = scene_label([m.name for m in members])
            labelled.append((when is None, "" if when is None else when, label, members))
        # dated groups first, newest first; undated groups after, by label
        labelled.sort(key=lambda t: (t[0], "" if t[0] else "\uffff", t[2]))
        dated = sorted((t for t in labelled if not t[0]), key=lambda t: t[1], reverse=True)
        undated = sorted((t for t in labelled if t[0]), key=lambda t: t[2])
        def _preview(members):
            return lambda _checked=False, ms=tuple(members): self._preview_footprints(ms)

        for _undated, _when, label, members in dated + undated:
            if len(members) == 1:
                fp = members[0]
                menu.addAction(f"Import {fp.name}").triggered.connect(_import(fp))
                menu.addAction(f"Preview {fp.name}").triggered.connect(_preview(members))
                continue
            sub = menu.addMenu(f"{label} ▸")
            members = sorted(members, key=lambda m: band_sort_key(m.name))
            names = [m.name for m in members]
            rep = _preview_band(members)
            sub.addAction(f"Preview scene ({band_label(rep.name, names)})"
                          ).triggered.connect(_preview(members))
            sub.addSeparator()
            for fp in members:
                sub.addAction(f"{band_label(fp.name, names)}  ({fp.dtype}, {fp.width}×{fp.height})"
                              ).triggered.connect(_import(fp))
        if self._previews:
            menu.addSeparator()
            menu.addAction("Clear previews").triggered.connect(lambda _c=False: self._clear_previews())
        return menu

    def _preview_footprints(self, members) -> None:
        """Drape one representative band of a footprint group on the world (the data browser's
        preview): the coarsest overview via :func:`overview_field`, view state only.
        Re-previewing a group replaces its entry; the world shows every current preview."""
        members = list(members)
        if not members:
            return
        rep = _preview_band(members)
        try:
            field = overview_field(rep.path, max_px=400)
        except Exception as exc:                        # unreadable file: say so, never crash
            self._notify(f"Preview failed for {rep.name}: {exc}", "status")
            return
        self._previews[group_key(rep)] = (rep.name, field)
        if self._arrangement is not None:
            self._arrangement.set_previews(list(self._previews.values()))
        self._notify(f"Previewing {rep.name} ({len(self._previews)} on the world) — "
                     f"right-click → Import to analyse, or Clear previews", "status")

    def _clear_previews(self) -> None:
        self._previews = {}
        if self._arrangement is not None:
            self._arrangement.set_previews([])

    def _on_footprints_right_clicked(self, hits) -> None:
        menu = self._footprint_menu(hits)
        if menu.actions():
            menu.exec(QtGui.QCursor.pos())

    # -- opening ---------------------------------------------------------------------------
    def open_path(self, path: str) -> None:
        """Load any raster :func:`~dynamix.shell.opening.open_field` can read and make it the
        current layer -- ``.npz``, a GeoTIFF, or a zip holding exactly one GeoTIFF (opened
        straight from the archive; never extracted). A ``.csv`` is a POINT catalogue instead:
:func:`~dynamix.shell.point_import.load_points` handles the
        whole thing -- column detection/mapping, source+layer registration, selection -- and
        never touches ``load_field``'s raster-only machinery."""
        if Path(path).suffix.lower() == ".csv":
            load_points(self, path)
            return
        # A shapefile or an Esri layer file is a REFERENCE layer, whichever dialog it came through
        # (the raster dialog included).
        if Path(path).suffix.lower() in (".shp", ".lyr"):
            self._open_reference_layers([path])
            return
        # A multi-grid container (netCDF/HDF) asks WHICH rasters to import, grouped by
        # sensor/resolution -- each group one multiband dataset. Cancel = no open; None =
        # not a container (or a single grid), the ordinary open below handles it.
        groups = self._pick_container_grids(path)
        if groups is False:
            return
        if groups is not None:
            self._open_container_groups(path, groups)
            return
        subdataset = None
        # A too-big GeoTIFF opens as a centred window; ``Settings.open_window_px`` sets its edge
        # (default 4096). Fresh opens only -- the project reopen path below keeps the window a
        # saved project was made with.
        settings = load_settings()
        self.load_field(open_field(path, window_size=settings.open_window_px,
                                   max_pixels=settings.open_max_pixels,
                                   subdataset=subdataset), path)

    def load_field(self, field, path: str, *, inert: bool = False,
                   bands: "list | None" = None) -> None:
        """Take ``field`` (a ``RasterField`` or a bare array) as a new layer and resolve it.

        ``bands``: the band list when ``field`` is a SUBSET of ``path`` (an imported sensor
        group) -- it joins the source's identity, so each group is its own dataset.

        ``inert`` forces the inert open whatever the auto-run setting says (a derivative
        dataset: raw data the user decides what to run on -- with auto-run the new layer would
        otherwise take the window's CURRENT recipe as its own chain).

        ``path`` is the source IDENTITY, not necessarily a file: it keys the project's source
        registry and, through it, every cached transform result.

        **Open is inert by default.** Unless the "Auto-run WTMM on open" setting is on,
        the new layer gets an EMPTY chain and no worker is dispatched -- the raster still displays
        (``set_field``/``autoRange``/``_update_scale_bar`` all still run), but nothing computes
        until the user builds a chain deliberately. This is what makes open a cheap look rather
        than a commitment to a multi-second WTMM run on a file the user may only be previewing.

        **Settings.center_view restore order.** A persisted
        ``"vector"``/``"geo"`` view is restored at the TAIL of this method, on the FIRST call for
        this window only (``self._center_view_restored``, set in ``__init__``) -- deliberately
        not in ``__init__`` itself: ``_set_center_view`` ends in ``_sync_arrangement``, whose
        admission logic (the active layer, ``frames_compatible`` against it, the
        ``has_georeference`` gate) has nothing to run over before a real ``self.layer``/
        ``self._fields`` entry exists, which ``__init__`` alone never produces. A SECOND (or
        later) ``load_field`` call on the same window does not re-apply it -- the user may have
        switched views on purpose since the first one landed.

        **The floating inspectors restore at the same tail**, behind their own
        one-shot flag (``self._inspectors_restored``), for the identical reason and with the
        identical caveat: there is no source to open an inspector OVER before the first layer
        lands, and a user who closed one since must not have it pushed back at them by the next
        open. Two flags rather than one -- see :meth:`_restore_inspectors`.
        """
        self.field = field
        self._push_reference_layers()          # reference layers re-pixel to the new raster
        # A new raster is a new subject: a box drawn on the previous one names a region of an
        # image that is no longer on screen. (``_select_layer`` does the same for a switch; this
        # path never goes through it -- see the selection comment below.)
        self._reset_roi_selection()
        source = self.project.add_source(path, bands=bands)
        # Devloop-rebuild adoption: a rebuilt window is handed the RESTORED project, and
        # adding ANOTHER master for a source whose family already exists would duplicate the
        # master and strand every existing layer (the rebuilt panel only shows rows added
        # through add_layer_row). The rebuild
        # signature is a window that knows NO rows for an already-populated source; adopt
        # the family instead: rebuild every row (parents precede children in
        # ``project.layers``, so grouping lands right; adopted rows resolve against the
        # session field -- windowed ROI fields are not persisted by the devloop session)
        # and take the first root as the active layer. The constructor already seeded the
        # recipe from SESSION["steps"], so the inert-open branch must not wipe it. Opening
        # the same file again in a LIVE window (windowed open then the @ov overview) still
        # adds masters: that window already shows rows for the source.
        _existing = [l for l in self.project.layers if l.source_id == source.source_id]
        _adopt = bool(_existing) and not any(
            l.layer_id in self._layer_by_id for l in _existing)
        auto_run = load_settings().auto_run_wtmm and not inert
        if _adopt:
            for _l in _existing:
                self.add_layer_row(_l, field)
            self.layer = next(l for l in _existing if l.parent_id is None)
        elif auto_run:
            if self._inert_open:
                # Auto-run just turned on (or was already on) over a window an inert open left
                # with nothing to build -- "run the chain" has to mean the DEFAULT chain here, or
                # this open would silently resolve an empty one and dispatch a worker with
                # nothing to compute. A window that already has a real (possibly knob-tweaked)
                # chain is untouched: only the wiped state reseeds.
                steps = type(self).DEFAULT_STEPS
                self._set_recipe([name for name, _ in steps], [dict(params) for _, params in steps])
                self._inert_open = False
            chain = self._chain()
        else:
            # No chain means no honest ``_names``/``_params`` either -- they are this window's
            # working copy of "the chain being edited", and leaving them at whatever ``steps`` the
            # constructor was given would claim a recipe the layer does not actually have.
            self._set_recipe([], [])
            self._inert_open = True
            chain = Chain()
        if not _adopt:
            self.layer = self.project.add_layer(Path(path).stem or path, source.source_id,
                                                chain)
            self.add_layer_row(self.layer, field)
            self._ensure_band_layers(self.layer, field)
        # ``self.layer`` is already this layer, so the selection handler recognises the row as the
        # one it is on and stands aside -- the rest of this method IS the switch.
        self.layer_list.select_layer(self.layer.layer_id)
        self.setWindowTitle(f"{TITLE} — {self.layer.name}")
        self._build_strips()
        self.canvas.set_field(field)
        self._apply_raster_visibility()
        self._displayed_layer_name = self.layer.name
        self._holder_raster_ref = None       # raw field is up -- holder display re-syncs on land
        # Frame the RASTER, not every item: reference vectors can span far beyond a small
        # raster (a derivative fork against a basin-wide shapefile), and a bare autoRange()
        # fits them all -- opening would zoom to the vectors instead of the dataset.
        self.canvas.view.autoRange(items=[self.canvas.image_item])
        self._update_scale_bar()
        self._sync_composite_panel()
        self._refresh_panel_relevance()
        if auto_run:
            self._start_worker()
        if not self._center_view_restored:
            self._center_view_restored = True
            saved_view = load_settings().center_view
            if saved_view in ("vector", "geo"):
                self._set_center_view(saved_view)
        if not self._inspectors_restored:
            self._inspectors_restored = True
            self._restore_inspectors()

    def _on_open_clicked(self) -> None:
        # The file picker is the one dialog in this app, and it is the spec's "minimal file-open",
        # not an error or progress dialog -- those are the ones DESIGN.md forbids.
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open raster", "",
            "Raster, points or reference layers (*.npz *.npy *.tif *.tiff *.zip *.png *.jpg "
            "*.csv *.shp *.lyr *.nc *.nc4 *.cdf *.h5 *.hdf5 *.he5 *.hdf)")
        if path:
            self.open_path(path)

    def _pick_container_grids(self, path: str) -> "list | None | bool":
        """When a container holds several grids, ask WHICH rasters to import
        (:class:`ImportGridsDialog`: sensor groups with a checkbox per band). Returns
        ``[(dataset label, [subdataset ids])]``, ``None`` (not a multi-grid container --
        the ordinary open path applies), or ``False`` (cancelled / nothing ticked)."""
        from dynamix.core.ingest import GRID_SUFFIXES, probe

        if not str(path).lower().endswith(GRID_SUFFIXES):
            return None
        try:
            info = probe(path)
        except ValueError as exc:
            self._notify(str(exc), "status")
            return False
        subs = info["subdatasets"]
        if len(subs) <= 1:
            return None
        dialog = ImportGridsDialog(Path(path).name, subs, info.get("dims") or {}, parent=self)
        if not dialog.exec():
            return False
        groups = dialog.groups()
        if not groups:
            self._notify("nothing ticked — nothing imported", "status")
            return False
        return groups

    def _open_container_groups(self, path: str, groups) -> None:
        """Import each ``(label, [ids])`` as its own dataset: one band loads plain, several
        stack ``(ny, nx, nc)`` (same grid only -- :func:`~dynamix.core.ingest.load_grid_stack`
        refuses mixed resolutions). Every dataset keys the same source ``path``, exactly like
        re-opening the file, so each lands as its own master row."""
        from dynamix.core.ingest import load_grid_stack
        from dynamix.shell.opening import _stamp_source

        stem = Path(path).stem
        opened = []
        for label, ids in groups:
            try:
                field = load_grid_stack(path, ids, name=f"{stem} · {label}")
            except ValueError as exc:
                self._notify(f"{label}: {exc}", "status")
                continue
            ny, nx = np.asarray(field.values).shape[:2]
            # Its OWN source (file + band list): its own dataset row, cache lines and reopen.
            self.project.add_source(path, label=field.name, bands=list(ids))
            self.load_field(_stamp_source(field, str(path), ny, nx), path, bands=list(ids))
            self.layer.name = field.name            # load_field names a layer by the file stem
            self.layer_list.set_layer_name(self.layer.layer_id, field.name)
            opened.append(label)
        if opened:
            self._notify(f"{Path(path).name}: imported {', '.join(opened)}", "status")

    def _pick_subdataset(self, path: str) -> "str | None | bool":
        """When a container holds several grids (netCDF/HDF variables), ask which one. Returns
        the subdataset id, ``None`` (no choice needed), or ``False`` (user cancelled).
        Superseded by :meth:`_pick_container_grids` on the open path; kept for callers that
        want exactly one grid."""
        from dynamix.core.ingest import GRID_SUFFIXES, probe

        if not str(path).lower().endswith(GRID_SUFFIXES):
            return None
        try:
            info = probe(path)
        except ValueError as exc:
            self._notify(str(exc), "status")
            return False
        subs = info["subdatasets"]
        if len(subs) <= 1:
            return None
        labels = [desc for _sid, desc in subs]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Choose grid", f"{Path(path).name} holds {len(subs)} grids:",
            labels, 0, False)
        if not ok:
            return False
        return subs[labels.index(choice)][0]

    # -- display styling ---------------------------------------------------------------------
    def _apply_display_style(self, style: dict, *, points_only: bool = False) -> None:
        """Push a layer's full style dict — the three knobs plus the colormap/colors/trails
        and the point-overlay color — onto the canvas through every setter it owns. ONE
        call site so
        :meth:`_on_display_style_changed` and :meth:`_sync_display_controls` can never drift onto
        calling a different subset of these.

        ``points_only``: ``True`` exactly when the
        style being pushed belongs to a POINT layer -- one whose own field the canvas never draws
        as its raster/overlay image (``Canvas.set_field`` is skipped for a point layer, see
        ``_select_layer``'s own comment: the canvas keeps showing whatever raster it last had).
        Pushing that layer's colormap/overlay-colors/show_trails onto the canvas would restyle the
        STILL-DISPLAYED raster with an unrelated layer's preferences (tune the raster to magma,
        select an EQ point layer to work its backproject chain, watch magma silently snap to
        viridis) -- exactly the bleed this guard closes. Only the one setter that genuinely belongs
        to a point layer (``set_points_style``) still runs; every raster-display setter is skipped
        entirely, leaving the canvas exactly as the DISPLAYED raster layer's own selection last
        left it.

        Order matters, for the non-point-only setters: ``set_display_style`` runs FIRST because it
        records ``self._overlay_line_width`` on the canvas, and ``set_overlay_colors`` reads that
        same attribute to build its own fresh pens — a group item created after a style edit
        already relies on this same "record width, THEN color" ordering (see
        ``canvas._draw_grouped_trails``'s own docstring)."""
        if not points_only:
            self.canvas.set_display_style(opacity=style["opacity"], point_size=style["point_size"],
                                          line_width=style["line_width"])
            self.canvas.set_colormap(style["colormap"])
            self.canvas.set_overlay_colors(style["color_hchain"], style["color_vtrail"],
                                           style["color_extrema"])
            self.canvas.set_show_trails(style["show_trails"])
            self.canvas.set_arrow_mode(style["arrows"])
            self.canvas.set_stretch(style["stretch"], style["stretch_pct"])
            self.canvas.set_levels(style["levels"], style["levels_colors"],
                                   style["levels_sieve"])
            self.canvas.set_hillshade(style["hillshade"], style["sun_azimuth"],
                                      style["sun_altitude"], style["z_factor"])
        # The point-layer scatter reuses ui.point_size (one size knob, two overlays --
        # the design's instruction) rather than a new one. Always runs, point layer active or
        # not -- the one setter a point layer's own selection is allowed to reach.
        self.canvas.set_points_style(style["color_points"], style["point_size"])

    def _on_display_style_changed(self, name: str, value) -> None:
        """A display control moved: persist on the ACTIVE layer's tags and restyle the canvas.

        Allowed on locked/frozen layers — these controls change how the measurement is DRAWN,
        never the measurement, so the lock's edit guard does not apply (same reasoning as
        scrubbing a locked layer's display being refused only because it rewrites the chain,
        which this never does).

        ``name`` is one of the three ``_DISPLAY_PARAMS`` knobs (a float, RightPanel's own
        ``DragValue`` controls) or one of the five new keys (``colormap``/``color_hchain``/
        ``color_vtrail``/``color_extrema`` — a plain string, no ``repr(float(...))`` cast — or
        ``show_trails`` — a bool, stored as the same ``"True"``/``"False"`` text
        :func:`_display_style_of` reads back). The combo/swatch/checkbox controls emit
        through this identical ``styleChanged`` signal, so this handler already knows how to file
        whatever they send.
        """
        if self.layer is None:
            return
        if name in _DISPLAY_PARAM_NAMES:
            self.layer.tags[f"ui.{name}"] = repr(float(value))
        elif name in ("show_trails", "hillshade", "surface", "depth_positive"):
            self.layer.tags[f"ui.{name}"] = "True" if value else "False"
        else:
            self.layer.tags[f"ui.{name}"] = str(value)
        # The proposal is now the layer's truth, so show it: ``DragValue`` never updates its own
        # label on a drag (knob_widgets.py's feedback-loop contract -- the OWNER calls
        # ``set_value`` once it has applied the value). Without this the thickness applied while
        # the number on the knob stayed put. Non-emitting by contract.
        self.right_panel.set_style_values(_display_style_of(self.layer))
        # While the ACTIVE layer is a point layer, only its point-relevant setter may
        # reach the canvas -- the raster/overlay setters would restyle whatever raster is still
        # displayed with this (unrelated) layer's prefs. See ``_apply_display_style``'s own
        # docstring.
        self._apply_display_style(_display_style_of(self.layer),
                                  points_only=self._is_point_layer(self.layer))
        self._refresh_panel_relevance()
        # The arrangement scene reads THREE of the new keys (colormap for a layer's own
        # drape, color_vtrail as `_chain_color`'s fallback, color_points for a point layer's own
        # drape -- see `_sync_arrangement`'s entries); color_hchain/color_extrema/show_trails are
        # session-canvas-only, so resyncing for THOSE would rebuild the arrangement for nothing it
        # draws differently. Gated on the arrangement actually being the current view, mirroring
        # every other resync-trigger in this file (``_on_layer_selected``, ``_on_hide_toggled``): a
        # flipped-away edit is picked up for free by ``_toggle_center_view``'s own unconditional
        # resync on the next flip-in.
        if name == "colormap" and self._center_view != "raster" \
                and not _display_style_of(self.layer)["hillshade"]:
            # In-place LUT swap: a colormap change on a plain drape never needs the full
            # set_layers rebuild the sync runs -- Scene.set_colormap swaps mapper LUTs and
            # renders. Hillshade bakes RGBA, so that case (and every geometry-shaping key below)
            # still resyncs.
            if self._arrangement is not None:
                self._arrangement.set_colormap(str(value))
        elif (name in ("colormap", "color_vtrail", "color_points",
                       "hillshade", "sun_azimuth", "sun_altitude", "z_factor",
                       "stretch", "stretch_pct", "levels", "levels_colors", "levels_sieve",
                       "surface", "surface_source", "depth_positive")
                and self._center_view != "raster"):
            self._sync_arrangement()

    def _on_surface_dialog_requested(self) -> None:
        """"3-D surface…" clicked (right panel) -- the full surface choice in one modal dialog:
off / z from this layer's own values / z from another loaded
        raster. Eligible "other" sources are loaded, non-point rasters on the IDENTICAL grid
        shape -- never resampled (the same shape law the arrangement's frame mode enforces);
        the dialog itself stays dumb and only shows what this method admits. The choice files
        through the ordinary ``styleChanged`` tag path, so persistence and Vector-view resync
        behave exactly like every other display pref (source first, then the surface flag, so
        the flag's own resync sees the new source already filed)."""
        if self.layer is None:
            return
        style = _display_style_of(self.layer)
        own = self._fields.get(self.layer.layer_id)
        own_shape = getattr(getattr(own, "values", None), "shape", None)
        eligible = []
        if own_shape is not None:
            for layer in self.project.layers:
                if layer.layer_id == self.layer.layer_id or self._is_point_layer(layer):
                    continue
                f = self._fields.get(layer.layer_id)
                shape = getattr(getattr(f, "values", None), "shape", None)
                if shape is not None and shape[:2] == own_shape[:2]:
                    eligible.append((layer.layer_id, layer.name))
            # Derived datasets: an h-map or band reconstruction is a real
            # same-grid raster -- offer it as a height source too ("recon as relief under h
            # colors" is a legitimate pairing). Keyed "derived:<layer_id>".
            for lid, dfield in self._derived_fields.items():
                shape = getattr(getattr(dfield, "values", None), "shape", None)
                owner = self._layer_by_id.get(lid)
                if shape is not None and shape[:2] == own_shape[:2] and owner is not None:
                    eligible.append((f"derived:{lid}", f"{owner.name} (derived)"))
        dialog = SurfaceDialog(enabled=style["surface"], source=style["surface_source"],
                               layers=eligible, parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        enabled, source, apply_all = dialog.selection()
        if apply_all:
            # Sibling tags first, so the active layer's ordinary styleChanged path below
            # triggers ONE resync that already sees the whole family configured. A sibling
            # whose id equals the chosen source resolves to its own field -- "same", which
            # is exactly right for the master standing under its children.
            for sib in self.project.layers:
                if (sib.source_id == self.layer.source_id
                        and sib.layer_id != self.layer.layer_id
                        and not self._is_point_layer(sib)):
                    sib.tags["ui.surface_source"] = str(source)
                    sib.tags["ui.surface"] = "True" if enabled else "False"
        self._on_display_style_changed("surface_source", source)
        self._on_display_style_changed("surface", enabled)

    def _on_levels_dialog_requested(self) -> None:
        """"Slice…" (right panel) -- the color-slice editor over the DISPLAYED raster's values
        (the derived h-map/reconstruction when one is up, else the raw field): DISPLAY only --
        it files the ``ui.levels``/``ui.levels_colors`` tags through the ordinary styleChanged
        path and never touches the chain; the h-band mask
        for reconstruction is :meth:`_on_band_dialog_requested`'s separate dialog."""
        if self.layer is None or self.field is None:
            return
        if self._holder_raster_ref is not None:
            values = self._holder_raster_ref[1]
        else:
            values = getattr(self.field, "values", None)
        if values is None:
            return
        style = _display_style_of(self.layer)
        dialog = LevelsDialog(np.asarray(values, dtype=np.float64),
                              levels_text=style["levels"],
                              colors_text=style["levels_colors"],
                              colormap_name=style["colormap"],
                              min_island=style["levels_sieve"], parent=self)
        dialog.levelsApplied.connect(self._on_levels_applied)
        dialog.levelsPreviewRequested.connect(self._on_levels_preview_requested)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_band_dialog_requested(self) -> None:
        """"Reconstruct…" (right panel) -- the h-band mask dialog over the ACTIVE result's own
        FULL h-map (holder_map's and band_recon's results both carry one, unmasked -- the
        histogram always shows every exponent). Analysis only: live drags render the
        reconstruction preview, Reconstruct commits the band_recon step; no display tag is
        ever written, so the color ramp cannot move. Seeds from an
        active band_recon step's own knobs."""
        if self.layer is None or self.field is None:
            return
        result = self._active_result
        h_map = result.get("h_map") if isinstance(result, dict) else None
        if h_map is None:
            self._notify("no h-map on the active result — run holder_measure or "
                         "holder_multiaffine (or a band reconstruction) first", "status")
            return
        h_lo = h_hi = None
        band_step = _band_step_for(self._names)
        if band_step is not None:
            pp = self._params[self._names.index(band_step)]
            h_lo, h_hi = pp.get("h_lo"), pp.get("h_hi")
        dialog = BandDialog(np.asarray(h_map, dtype=np.float64), h_lo=h_lo, h_hi=h_hi,
                            parent=self)
        dialog.reconstructRequested.connect(self._on_band_reconstruct_requested)
        dialog.bandPreviewRequested.connect(self._on_band_preview_requested)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_band_preview_requested(self, h_lo: float, h_hi: float,
                                   mode: str = "mask", min_island: int = 0) -> None:
        """A live histogram-drag tick (Band dialog, Live on). Mode "mask" (the default, and
        the DECISION view) renders the band's binary singularity
        set: FFT-free, a strided band test per tick. Mode "reconstruction" renders the
        decimated inverse (~20 ms/tick -- setup cached on (field, h_map) identity, two
        forward rFFTs + one inverse per tick). Preview only either way: the canvas shows a
        derived strided RasterField; ``_holder_raster_ref`` is cleared so the next real
        landing repaints honestly; the Reconstruct button commits at full res."""
        if self.layer is None or self.field is None:
            return
        result = self._active_result
        h_map = result.get("h_map") if isinstance(result, dict) else None
        # An ROI result's h-map is the ROI's: reconstruct the ROI's own values and draw on the
        # ROI's own axes (comparing it with the WHOLE field made every tick skip silently).
        roi_vals = result.get("_roi_values") if isinstance(result, dict) else None
        vals = roi_vals if roi_vals is not None else getattr(self.field, "values", None)
        if h_map is None or vals is None:
            return
        vals = np.asarray(vals)
        if vals.ndim != 2 or vals.shape != np.asarray(h_map).shape:
            return
        source = self.project.sources.get(self.layer.source_id)
        if (source is not None and source.hidden) or not self.layer.visible:
            self._notify("live reconstruction hidden — unhide the dataset (H on the source "
                         "row) and the layer (H on its row) to see it", "status")
            return
        if mode == "mask":
            h_arr = np.asarray(h_map)
            s = max(1, -(-max(h_arr.shape) // 512))
            hs = h_arr[::s, ::s]
            m = np.isfinite(hs) & (hs >= h_lo) & (hs < h_hi)
            if min_island > 0:
                # Settle-tier only (drag ticks pass 0 -- the lazy contract). Threshold scaled
                # by the preview stride: an island of N full-res pixels holds ~N/s^2 here.
                from dynamix.core.sieve import sieve_mask
                m = sieve_mask(m, max(1, int(min_island) // (s * s)))
            recon = m.astype(np.float64)
            label = "mask" if min_island <= 0 else "mask·sieved"
        else:
            key = (id(vals), id(h_map))
            if self._band_preview is None or self._band_preview[0] != key:
                from dynamix.core.microcanonical import BandReconstructor
                self._band_preview = (key, BandReconstructor(vals, h_map))
            br = self._band_preview[1]
            recon = br.reconstruct(h_lo, h_hi).astype(np.float64)
            s = br.stride
            label = "band"
        roi_field = (_roi_display_field(result, recon, self.field,
                                        f"{self.layer.name}·{label}[{h_lo:.2f},{h_hi:.2f})")
                     if roi_vals is not None else None)
        if roi_field is not None:
            derived = dataclasses.replace(roi_field, x_axis=roi_field.x_axis[::s],
                                          y_axis=roi_field.y_axis[::s])
        else:
            derived = dataclasses.replace(
                self.field, values=np.asarray(recon, dtype=np.float64),
                x_axis=np.asarray(self.field.x_axis)[::s],
                y_axis=np.asarray(self.field.y_axis)[::s],
                name=f"{self.field.name}·{label}[{h_lo:.2f},{h_hi:.2f})")
        self.canvas.set_field(derived)
        self._apply_raster_visibility(showing_product=True)
        self._holder_raster_ref = None
        # Vector view live too (center_view="vector" is a persisted
        # default, and the raster canvas is HIDDEN behind that tab):
        # swap the drape's scalars in place, or say honestly why not (hillshade bakes RGBA).
        if self._center_view != "raster" and self._arrangement is not None \
                and self.layer is not None:
            if not self._arrangement.preview_raster_values(self.layer.layer_id, recon, s):
                self._notify("live preview drawn on the Raster tab (this Vector drape is "
                             "hillshade-baked; Apply/Reconstruct update it)", "status")

    def _on_band_reconstruct_requested(self, h_lo: float, h_hi: float,
                                       min_island: int = 0) -> None:
        """The Band dialog's "Reconstruct" (revised): the
        reconstruction is a NEW GROUPED LAYER, never an edit of the microcanonical layer --
        "instead of making a new one ... replaces the microcanonical chain" was the wrong
        semantics. Mirrors ``_on_roi_create`` exactly: ``add_layer(..., parent_id=parent)``,
        the parent's field object reused, select -> worker. The child's chain is the single
        ``band_recon`` step carrying the parent's holder_map ENGINE params verbatim (shared
        tuples -- the h-map the band was picked from is the h-map the reconstruction uses)
        plus the band; the parent layer, its chain and its h-map display stay untouched.

        One refinement: when the ACTIVE layer already IS a reconstruction layer (its chain is
        headed by band_recon), the new band updates it in place -- re-banding a recon layer
        should not mint a sibling per drag-commit."""
        parent = self.layer
        if parent is None or self._is_point_layer(parent):
            return
        band = {"h_lo": float(h_lo), "h_hi": float(h_hi),
                "min_island": int(min_island)}
        band_step = _band_step_for(self._names)
        if band_step is not None:
            names = list(self._names)
            params = [dict(pp) for pp in self._params]
            params[names.index(band_step)].update(band)
            descriptors = [{"device": n, "params": dict(pp), "bypassed": b, "rack": r}
                           for n, pp, b, r in zip(names, params, self._bypassed, self._rack)]
            self.strips.set_steps(descriptors, field=self.field)
            self._on_chain_edited(descriptors)
            return
        recon_name = _band_device_for(self._names)
        producer = next((n for n in _BAND_FOR_ANALYZER if n in self._names), None)
        engine = dict(defaults_for(get_device(recon_name)))
        if producer is not None:
            engine.update(self._params[self._names.index(producer)])
        engine.update(band)
        chain = Chain((DeviceRef(recon_name, engine),)).materialized()
        field = self._fields.get(parent.layer_id, self.field)
        # Short, semantic name (the panel truncated the inherited lineage
        # to "...holder_map", hiding what the layer IS): the DATASET's base name + the band --
        # never the full ancestry.
        src = self.project.sources.get(parent.source_id)
        base = (src.label or Path(src.path).stem) if src is not None else parent.name
        layer = self.project.add_layer(
            f"{base} · recon[{h_lo:g},{h_hi:g})", parent.source_id, chain,
            parent_id=parent.layer_id, tags=_inherited_window(parent))
        self.add_layer_row(layer, field)
        self.layer_list.select_layer(layer.layer_id)   # -> _select_layer -> worker

    def _on_levels_preview_requested(self, breaks_text: str, colors_text: str,
                                     min_island: int = 0) -> None:
        """A live Slice tick (bounds dragging / a color picked): recolor the RASTER canvas
        immediately -- pure view state through ``Canvas.set_levels``, no tag filed, so
        scrubbing is free and Apply remains the persistence moment. Vector-view half: the
        classified [0, 1] field swaps in through the drape's plain-LUT path when available
        (class colors show as colormap classes there; the exact custom colors land on
        Apply's full resync -- the RGBA-baked path is not per-tick territory, same v1 line
        as the hillshade preview)."""
        if self.layer is None:
            return
        self.canvas.set_levels(breaks_text, colors_text, int(min_island))
        if self._center_view != "raster" and self._arrangement is not None:
            values = (self._holder_raster_ref[1] if self._holder_raster_ref is not None
                      else getattr(self.field, "values", None))
            swapped = False
            if values is not None and np.asarray(values).ndim == 2:
                from dynamix.core.stretch import classify, parse_levels
                try:
                    spec = parse_levels(breaks_text)
                    if spec is not None:
                        classes01 = classify(np.asarray(values, dtype=np.float64), spec)
                        if min_island > 0:
                            from dynamix.core.sieve import sieve_classes
                            from dynamix.core.stretch import resolve_breaks
                            n_cls = len(resolve_breaks(
                                np.asarray(values, dtype=np.float64), spec)) + 1
                            classes01 = sieve_classes(classes01, n_cls, int(min_island))
                        swapped = self._arrangement.preview_raster_values(
                            self.layer.layer_id, classes01, 1)
                except (ValueError, TypeError):
                    pass
            if not swapped:
                # RGBA-baked drape (hillshade / custom colors): full fidelity needs the
                # apply path -- throttled so a drag lands ~3 updates/s, the same budget the
                # hillshade sliders' own per-change resync already spends.
                self._levels_vector_pending = (breaks_text, colors_text, int(min_island))
                if not self._levels_vector_timer.isActive():
                    self._levels_vector_timer.start()

    def _flush_levels_vector(self) -> None:
        if self._levels_vector_pending is not None:
            breaks_text, colors_text, min_island = self._levels_vector_pending
            self._levels_vector_pending = None
            self._on_levels_applied(breaks_text, colors_text, min_island)

    def _on_levels_applied(self, breaks_text: str, colors_text: str,
                           min_island: int = 0) -> None:
        self._on_display_style_changed("levels_sieve", str(int(min_island)))
        self._on_display_style_changed("levels_colors", colors_text)
        self._on_display_style_changed("levels", breaks_text)

    def _sync_display_controls(self, layer) -> None:
        """On layer switch: show the layer's stored styling and apply it to the canvas."""
        style = _display_style_of(layer)
        self.right_panel.set_style_values(style)     # non-emitting -- RightPanel's own contract
        # A point layer's own styling never reaches the canvas's raster/overlay state
        # -- see ``_apply_display_style``'s own docstring.
        self._apply_display_style(style, points_only=self._is_point_layer(layer))
        self._sync_composite_panel()
        self._refresh_panel_relevance()

    def _default_composite(self, nc: int) -> dict:
        return {"r": 0 if nc >= 1 else None, "g": 1 if nc >= 2 else None,
                "b": 2 if nc >= 3 else None, "solo": [], "mute": [],
                "stretch": "percent", "stretch_pct": 2.0, "stretch_k": 2.0}

    def _sync_composite_panel(self, field=None) -> None:
        """Show the Composite section iff the displayed field (``field``, default the active
        dataset's) is a multiband stack, seeded from the layer's ``ui.composite`` tag (else the
        first-three-bands default), and push the spec onto the canvas. Band rows label from
        ``provenance["bands"]``; a bus's sends keep their dataset so same-named bands from two
        datasets stay apart."""
        import json

        field = self.field if field is None else field
        values = np.asarray(getattr(field, "values", np.empty(0)))
        if values.ndim != 3:
            self._composite_frame.setVisible(False)
            self.canvas.set_composite(None)
            return
        nc = int(values.shape[-1])
        spec = None
        raw = self.layer.tags.get("ui.composite") if self.layer is not None else None
        if raw:
            try:
                spec = json.loads(raw)
            except ValueError:
                spec = None
        if not isinstance(spec, dict):
            spec = self._default_composite(nc)
        prov = getattr(field, "provenance", {}) or {}
        if prov.get("bus"):
            names = [" · ".join(str(n).split(" · ")[-2:]) for n in prov.get("bands") or []]
        else:
            names = [str(n).split("/", 1)[0] for n in prov.get("bands") or []]
        names += [f"band {i + 1}" for i in range(len(names), nc)]
        self.composite_panel.set_bands(names, spec)
        self._composite_frame.setVisible(True)
        self.canvas.set_composite(spec)

    def _on_composite_changed(self, spec: dict) -> None:
        import json

        if self.layer is not None:
            self.layer.tags["ui.composite"] = json.dumps(spec)
        self.canvas.set_composite(spec)

    # -- arrangement mask -----------------------------------------
    def _on_mask_row_changed(self, payload: dict) -> None:
        """``self._mask_row.maskChanged`` -> ``ArrangementView.set_mask``. A no-op whenever the
        arrangement has never been built (``self._arrangement is None``, the ordinary state for
        every session that hasn't pressed Tab yet) -- the row itself lives in the right panel and
        is fully interactive regardless, so an edit made before the first flip is simply not
        forwarded anywhere; :meth:`_set_center_view` pushes the row's then-current values into
        the view itself the moment it is actually built/activated, so nothing already set here is
        lost."""
        if self._arrangement is not None:
            self._arrangement.set_mask(payload)

    # -- arrangement groups ----------------------------------------
    def _on_commit_button_clicked(self) -> None:
        """``self._commit_button.clicked`` -> ``ArrangementView.commit()``. A no-op whenever the
        arrangement has never been built (``self._arrangement is None``) -- mirrors
        :meth:`_on_mask_row_changed`'s own no-op contract exactly: the button and palette live in
        the right panel and stay interactive regardless, so a click made before the first flip
        just has nothing to commit into yet (no scene was ever built, so no pick was ever possible
        without one -- see ``ArrangementView.commit``'s own docstring for its own, second-layer
        no-op guard, ``self._group_palette is None`` on the VIEW side, which this call can never
        actually trigger: :meth:`_set_center_view` hands the view its palette reference in the
        same first-build branch that constructs it, strictly before this method could ever reach
        it)."""
        if self._arrangement is not None:
            self._arrangement.commit()

    # -- canvas picking ----------------------------------------
    def _on_canvas_chain_picked(self, chain_index, shift: bool) -> None:
        """``Canvas.chainPicked`` -> ``GroupPalette.add_pick``, naming the ACTIVE layer.

        The canvas only ever draws the ACTIVE layer's own result (``_apply``'s own ``set_result``
        call site), so there is no other layer a raster-canvas pick could possibly mean -- unlike
        the arrangement's ``Scene.pick``, which resolves its own layer identity from whichever
        mesh the 3-D ray actually hit (several layers can be on screen there at once).

        A miss (``chain_index is None``) still reaches ``add_pick``: its own miss handling (a
        plain-click miss clears the selection, a shift-click miss is a no-op -- ``group_palette.
        py``'s own "Gesture semantics" docstring) is exactly what a raster miss should do too, the
        same semantics the arrangement's own misses already get. Guarded on ``self.layer`` only
        for the HIT case -- a miss needs no layer identity at all to clear/no-op the selection.
        """
        if chain_index is None:
            self._group_palette.add_pick(None, shift)
            return
        if self.layer is None:
            return
        self._group_palette.add_pick((self.layer.layer_id, int(chain_index)), shift)

    def _apply_region_picks(self, chain_indices: list, subtract: bool) -> None:
        """Shared body of :meth:`_on_canvas_chains_lassoed`/:meth:`_on_canvas_chains_boxed`:
both ``Canvas`` signals report the
        identical shape -- every enclosed chain index, plus whether ⌥ was held (subtract) -- so
        both apply through ONE ``GroupPalette.apply_picks`` batch call naming the ACTIVE layer,
        rather than duplicating this body twice. A no-op with no active layer (nothing a picked
        index could name); an empty ``chain_indices`` still reaches ``apply_picks`` (a harmless
        no-op on the selection buffer that still emits, matching every other pick path's
        "always emits" contract)."""
        if self.layer is None:
            return
        lid = self.layer.layer_id
        picks = [(lid, int(i)) for i in chain_indices]
        self._group_palette.apply_picks(picks, "subtract" if subtract else "add")

    def _on_canvas_chains_lassoed(self, chain_indices: list, subtract: bool) -> None:
        """``Canvas.chainsLassoed`` -> one batch ``GroupPalette.apply_picks`` call.

        ``subtract`` is ``True`` only for a first-class lasso-MODE gesture completed with ⌥
        held; the pre-existing click-mode ⌥-shortcut always reports ``False`` (Alt is what
        TRIGGERS that gesture there, so it cannot also mean "subtract" -- see
        ``Canvas.mousePressEvent``'s own comment). Either way this still ADDS/unions every
        reported index in one call -- a lasso is a multi-select gesture by nature, the same
        reading ``ArrangementView``'s own lasso-equivalent wiring already gives it."""
        self._apply_region_picks(chain_indices, subtract)

    def _on_canvas_chains_boxed(self, chain_indices: list, subtract: bool) -> None:
        """``Canvas.chainsBoxed`` -> one batch ``GroupPalette.apply_picks`` call -- the box-mode
        sibling of :meth:`_on_canvas_chains_lassoed`, identical op semantics ("box/lasso =
        ADD by default, ⌥ = subtract")."""
        self._apply_region_picks(chain_indices, subtract)

    def _on_group_membership_changed(self, groups: dict) -> None:
        """``GroupPalette.membershipChanged`` -> ``Canvas.set_selection_chains`` -- the RASTER
        canvas's own half of "visible selection everywhere". Fires on every
        ``add_pick``/``apply_picks`` call (that signal's documented "always emits" contract),
        regardless of which gesture (click/shift-click/box/lasso) or which center-stack page
        produced it -- the same signal the arrangement's own selection highlight already rides
        (``ArrangementView._on_membership_changed``, a SEPARATE listener on the identical
        signal, not touched by this method).

        ``groups`` (committed-group MEMBERSHIP) is ignored on purpose: what this pushes is the
        live SELECTION buffer (``GroupPalette.selection()``), a different thing (see
        ``group_palette.py``'s own "membership accumulates" doctrine) -- the argument is only
        here because it is what the signal carries.

        ``None`` (clears the overlay) with no active layer at all -- there is no "active
        layer's own indices" to speak of then. An active layer with zero selected chains still
        gets a real, sorted (possibly empty) list -- a legitimate empty selection, not "nothing
        to show yet".

        Also pushes the SAME per-layer selection into the open ``SkeletonDialog``, if any -- ``_push_skeleton_selection``'s own no-op guards ("no dialog
        open" / "no active layer") mean this is always safe to call unconditionally here."""
        if self.layer is None:
            self.canvas.set_selection_chains(None)
            self._push_skeleton_selection()
            return
        lid = self.layer.layer_id
        indices = sorted(idx for (layer_id, idx) in self._group_palette.selection()
                          if layer_id == lid)
        self.canvas.set_selection_chains(indices)
        self._push_skeleton_selection()

    # -- topology links ----------------------------------------
    def _on_topology_link_requested(self, code) -> None:
        """``TopologyPanel.linkRequested`` -> ``self.project.user_links.link(...)``, naming the
        two picks in the group palette's own multi-select buffer -- EXACTLY two, refused
        otherwise (not "the last two", since there is no principled way to
        pick a pair out of three-plus).

        ``GroupPalette.selection()`` is an unordered ``set`` (``group_palette.py``'s own
        ``_selection: set[tuple[int, int]]``) -- it carries no record of insertion order at all.
        With exactly two picks that is moot: a two-element set names an unambiguous pair
        regardless of order (sorted here only so the two ends of a link are deterministic across
        calls, not to recover a temporal ordering the set never had). With any other count --
        zero, one, or three-plus -- there is no honest way to guess which two chains the user
        meant, so this refuses outright rather than silently dropping the "extra" picks by an
        arbitrary rule; ``code`` is the panel's own combo override, or ``None`` to auto-suggest.
        """
        picks = sorted(self._group_palette.selection())
        if len(picks) != 2:
            self.statusBar().showMessage(
                "Topology: select exactly two chains to link", 4000)
            return
        (layer_a, chain_a), (layer_b, chain_b) = picks

        ref_a = self._topology_obj_ref(layer_a, chain_a)
        ref_b = self._topology_obj_ref(layer_b, chain_b)
        if ref_a is None or ref_b is None:
            self.statusBar().showMessage(
                "Topology: could not identify one of the picked chains", 4000)
            return
        pts_a = self._topology_chain_points(layer_a, chain_a)
        pts_b = self._topology_chain_points(layer_b, chain_b)
        if pts_a is None or pts_b is None:
            self.statusBar().showMessage(
                "Topology: no cached geometry for one of the picked chains", 4000)
            return

        # ``scale`` is stored on the link EXACTLY as ``_current_scale_px`` reports it -- ``None``
        # stays ``None`` ("not yet measured", the design; ``TopologyPanel.set_rows`` already
        # renders that as ``@a=?``). Only the GEOMETRY call needs an actual number --
        # ``suggest_code``'s ``contact_scale`` is a distance threshold, not a record of what was
        # measured -- so a missing scale falls back to 1.0px there alone, never in what gets stored.
        scale = self._current_scale_px()
        if code is None:
            code = suggest_code(pts_a, pts_b, contact_scale=scale if scale is not None else 1.0)
        try:
            self.project.user_links.link(ref_a, ref_b, code, scale_first_contact=scale)
        except ValueError as exc:
            self.statusBar().showMessage(f"Topology: {exc}", 4000)
            return
        self._refresh_topology_panel()

    def _on_topology_unlink_requested(self, index: int) -> None:
        try:
            self.project.user_links.unlink(index)
        except IndexError:
            return
        self._refresh_topology_panel()

    def _topology_obj_ref(self, layer_id: int, chain_index: int) -> ObjRef | None:
        """``(layer_id, chain_index)`` -> ``ObjRef``, naming the chain by its layer's own last
        TRANSFORM device (the thing that actually produced the chain geometry). ``None`` for a
        layer that no longer exists, or one whose chain has no transform at all (an empty/
        filters-only chain -- nothing could have produced a chain index to pick in the first
        place, but this stays a documented no-op rather than an IndexError)."""
        layer = self._layer_by_id.get(layer_id)
        if layer is None or not layer.chain.transforms:
            return None
        return ObjRef(layer_id, layer.chain.transforms[-1].device, "line", int(chain_index))

    def _topology_chain_points(self, layer_id: int, chain_index: int):
        """The chain's own ``(x, y)`` points, IN THE DISPLAY FRAME -- straight from a resolve of
        ITS layer (the same ``resolve(layer, field, self.cache, source_id=...)`` call
        ``_sync_arrangement`` already makes for every visible layer; ``self.cache`` is
        content-hash keyed, so this is a cache hit whenever that layer's chain has not changed
        since its last redraw, not a fresh compute), then shifted by ``display_offset(result,
        field)`` -- the SAME per-layer shift ``_apply`` already applies
        before handing chains to the canvas for picking (``_shift_chains``'s own docstring: an
        ROI result's raw ``chains`` sit in the file-absolute frame the ROI device read its halos
        from, while the picture on screen -- and every OTHER layer's own chains -- sit in the
        displayed window's frame). Without this, two chains that visibly touch on screen because
        one of them belongs to an ROI/refined-run layer would classify by their raw, unshifted
        coordinates instead -- possibly thousands of pixels apart on a windowed raster (this
        module's own ``display_offset`` docstring gives BOEM as the worked example). A same-layer
        link is unaffected: both ends get the identical shift, and ``suggest_code`` only reads
        relative geometry (distances, bounding boxes), so a common translation changes nothing
        about it. Not ``self.canvas``'s own ``_pick_chains`` -- that only ever holds the ACTIVE
        layer's chains (``_on_canvas_chain_picked``'s own docstring), while a pick named here can
        come from any layer visible in the arrangement view. ``None`` on a missing field, a
        resolve failure, or an index the result's ``chains`` list no longer has."""
        layer = self._layer_by_id.get(layer_id)
        field = self._fields.get(layer_id)
        if layer is None or field is None:
            return None
        try:
            renderable = resolve(layer, field, self.cache, source_id=layer.source_id)
        except Exception:
            return None
        chains = renderable.result.get("chains")
        if not chains or not (0 <= chain_index < len(chains)):
            return None
        c = chains[chain_index]
        x = np.asarray(c["x"], dtype=np.float64)
        y = np.asarray(c["y"], dtype=np.float64)
        row_off, col_off = display_offset(renderable.result, field)
        if row_off or col_off:
            x = x + col_off
            y = y + row_off
        return np.column_stack([x, y])

    def _current_scale_px(self) -> float | None:
        """The active result's currently displayed scale, in px, or ``None`` when no scale is
        cleanly available. ``self._scales`` already holds nominal scale values IN PIXELS
        ("scales are stored in pixels, converted for display, never the reverse") --
        the same array ``_scale_reading`` reads for the transport's own label -- so the current
        index into it is the contact scale with no conversion at all.

        Returns ``None`` when there is no ``scale_select`` step in the chain, or nothing has been
        resolved yet -- a fabricated ``1.0`` px would be a made-up number that looks like a
        measurement; callers that need an actual float for geometry (``suggest_code``'s
        ``contact_scale``) are the ones that decide on a fallback, and ``LinkStore``'s own
        ``scale_first_contact`` stores the ``None`` honestly -- the design's "not yet measured",
        which the panel already renders as ``@a=?`` rather than a fabricated px reading."""
        i = self._index_of("scale_select")
        if i is None or not self._scales:
            return None
        idx = int(self._params[i].get("scale_idx", 0))
        if not (0 <= idx < len(self._scales)):
            return None
        return float(self._scales[idx])

    def _refresh_topology_panel(self) -> None:
        """Push the store's current links, and the active layer's own cached ``chain_topology``
        edge count (``self._active_result``, stamped by ``_apply`` -- reused rather than a second
        ``resolve()`` call, since ``_apply`` already has the exact result this panel wants to
        summarise), into ``self._topology_panel``.

        ``set_rows`` is skipped when the freshly computed ``rows`` compares
        equal to ``self._topology_rows_shown`` -- this method runs on EVERY ``_apply`` (so the
        chain-graph label tracks the active layer's own redraws), which used to rebuild the list
        widget, and therefore drop its current-row selection, on every unrelated redraw too. The
        chain-graph count is re-pushed unconditionally regardless -- it is one label with its own
        cheap, idempotent setter, not a selection-bearing widget."""
        rows = [
            {"code": e.code, "kind_a": e.a.kind, "kind_b": e.b.kind, "space_dim": e.space_dim,
             "a_id": e.a.node_id(), "b_id": e.b.node_id(),
             "scale_first_contact": e.scale_first_contact}
            for e in self.project.user_links.all()
        ]
        if rows != self._topology_rows_shown:
            self._topology_panel.set_rows(rows)
            self._topology_rows_shown = rows
        topo = self._active_result.get("topology") if self._active_result is not None else None
        self._topology_panel.set_chain_graph_count(len(topo.edges) if topo is not None else None)

    # -- skeleton dialog -------------------------------------------
    def _refresh_skeleton_button(self) -> None:
        """Enable/disable "Skeleton plot…" off the ACTIVE layer's own latest result
        (``self._active_result``, the exact field ``_refresh_topology_panel`` already reads) --
        a layer with no ``chains`` (no transform that produces them yet, or one that resolved to
        zero) has nothing for the dialog to show, and a disabled control names the reason in its
        tooltip rather than opening on an empty plot."""
        chains = self._active_result.get("chains") if self._active_result is not None else None
        enabled = bool(chains)
        self._skeleton_button.setEnabled(enabled)
        self._skeleton_button.setToolTip(
            "Open the WTMM skeleton plot -- log₂ scale vs log₂|W|, colored by Hölder slope"
            if enabled else
            "No chains in the active layer's result -- add a transform that produces chains "
            "(e.g. wtmm2d) before opening the skeleton plot")

    def _skeleton_px_size(self) -> float | None:
        """The active field's own physical pixel size, honestly -- ``None`` for a bare-pixel
        frame (``LocalFrame(units="px")``, ``RasterField``'s own default) or a
        :class:`~dynamix.core.frames.GeographicFrame`; a real float only for a ``LocalFrame``
        whose ``units`` name an actual physical unit (an EBSD map in microns, a lab raster in
        mm) -- ``frame.dx``, frame units per pixel, exactly what ``SkeletonDialog``'s own
        ``px_size`` parameter multiplies the scale axis by (``x = log2_scales +
        log2(px_um)``).

        Deliberately narrower than ``units.py``'s own CRS-aware ``px_to_metres`` (which also
        converts a projected GeoTIFF's feet to metres, and reports a ``GeographicFrame``'s own
        degree spacing) -- the skeleton dialog receives a bare float with no unit STRING to label
        a converted number with, so this only ever answers with a value already expressed in the
        frame's own named physical unit, or refuses honestly with ``None``."""
        frame = getattr(self.field, "frame", None)
        if isinstance(frame, LocalFrame) and frame.units != "px":
            return float(frame.dx)
        return None

    def _on_skeleton_button_clicked(self) -> None:
        """"Skeleton plot…" clicked -- build a FRESH ``SkeletonDialog`` from the active layer's
        CURRENT chains, every time (the design's own lifecycle call: "simplest honest lifecycle: rebuild on each open — document; caching across opens is NOT required").

        A cached, reused instance would need its own staleness tracking (did the chains change
        under a filter/transform edit made while the dialog sat open?) for no real benefit --
        ``SkeletonDialog`` construction is cheap (a numpy OLS pass over however many chains the
        active result carries, no I/O), so there is nothing to amortize by keeping one around.
        Any PREVIOUS dialog still on screen is closed first, so "re-shown" never means "now two
        skeleton windows, one plotting stale data" -- one skeleton window at a time, always
        showing what "Skeleton plot…" was just clicked on."""
        chains = self._active_result.get("chains") if self._active_result is not None else None
        if not chains or self.layer is None:
            return          # the button is disabled in this state -- a defensive no-op
        if self._skeleton_dialog is not None:
            self._skeleton_dialog.close()
        dialog = SkeletonDialog(chains, self._skeleton_px_size(), parent=self)
        dialog.selectionRequested.connect(self._on_skeleton_selection_requested)
        self._skeleton_dialog = dialog
        self._push_skeleton_selection()      # seed it with whatever is already selected
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    # -- multifractal spectrum window ---------------------------------
    def _refresh_spectrum_button(self) -> None:
        """Enable/disable "Multifractal spectrum…" off the ACTIVE layer's own latest result --
        gated on BOTH partition tables being present (``hd_std``/``hd_cmax``): a finest-scale
        preview stamps them ``None``, and a points-layer/empty result has neither. The disabled
        tooltip names the reason, same doctrine as ``_refresh_skeleton_button``."""
        result = self._active_result
        enabled = bool(result) and result.get("hd_std") is not None \
            and result.get("hd_cmax") is not None
        self._spectrum_button.setEnabled(enabled)
        self._spectrum_button.setToolTip(
            "Open the interactive τ(q)/D(h) scale-window fitter over the partition tables"
            if enabled else
            "No partition tables in the active layer's result yet -- run a full wtmm2d compute "
            "(a finest-scale preview has none) before opening the spectrum window")

    def _on_spectrum_button_clicked(self) -> None:
        """"Multifractal spectrum…" clicked -- a FRESH ``MultifractalWindow`` on the active
        result's own tables, previous one closed first (``_on_skeleton_button_clicked``'s exact
        lifecycle, for the same reasons: construction is cheap, staleness tracking is not).

        The eta seed is derived HONESTLY from the result's own resolved params (the
        ``_skeleton_px_size`` pattern -- the window stays dumb): BOTH paths now apply the
        ``a^fracint_alpha`` lift forward (the tensor path on its WT
        derivatives, the scalar path on ``|∇W|`` via ``wtmm_backend._apply_fracint2d``), so the
        seed is ``params["fracint_alpha"]`` whenever it is non-zero, regardless of mode. The
        window displays the note verbatim so an edited η stays an informed decision."""
        result = self._active_result
        hd_std = result.get("hd_std") if result is not None else None
        hd_cmax = result.get("hd_cmax") if result is not None else None
        if hd_std is None or hd_cmax is None:
            return          # the button is disabled in this state -- a defensive no-op
        eta_seed, note = _eta_note_of(result)
        if self._multifractal_window is not None:
            self._multifractal_window.close()
        window = MultifractalWindow(hd_std, hd_cmax, eta_seed=eta_seed, forward_note=note,
                                    parent=self)
        self._multifractal_window = window
        # The coupling channel (cell-49-inherits-cell-47): the fitter's window drives an open
        # construction window; connected on whichever of the two opens second.
        if self._spectrum_construction_window is not None:
            window.scaleWindowChanged.connect(
                self._spectrum_construction_window.set_scale_window)
        window.show()
        window.raise_()
        window.activateWindow()

    # -- singularity-spectrum construction window -----------------------------------------------
    def _refresh_panel_relevance(self) -> None:
        """Show only the right-panel controls that apply to what is on screen -- keyed on what
        the active RESULT carries (maxima, chains, an h-map, partition tables, a decomposition),
        never on tool names, so every tool (wtmm2d, M-Z / CDF edges, Perona-Malik diffusion,
        the Hölder tools) gets exactly its controls. Hidden, not greyed: nothing takes space
        it does not use."""
        if not hasattr(self, "_anisotropy_button") or not hasattr(self, "_transect_panel"):
            return                                          # the panel is still being built
        res = self._active_result if isinstance(self._active_result, dict) else {}
        points = self.layer is not None and self._is_point_layer(self.layer)
        shown = getattr(self.canvas, "_field", None)
        ndim = np.ndim(getattr(shown, "values", ())) if shown is not None else 0
        raster = not points and ndim in (2, 3)
        single, stack = raster and ndim == 2, raster and ndim == 3
        ext = [e for e in (res.get("extrema") or []) if isinstance(e, dict)]
        has_ext = any(len(e.get("x", ())) for e in ext)
        has_lines = has_ext and any(np.any(np.asarray(e.get("line_id", ())) >= 0) for e in ext)
        has_chains = bool(res.get("chains"))
        overlays = has_ext or has_chains or points
        style = _display_style_of(self.layer)
        shade = single and style["hillshade"]
        arrangement = getattr(self, "_center_view", "raster") != "raster"
        tables = res.get("hd_std") is not None and res.get("hd_cmax") is not None
        self.right_panel.apply_relevance({
            "knob:opacity": overlays,
            "knob:point_size": has_ext or points,
            "knob:line_width": has_lines or has_chains,
            "knob:sun_azimuth": shade, "knob:sun_altitude": shade,
            "knob:z_factor": shade or (single and style["surface"]),
            "colormap": single, "stretch": single,
            "reconstruct": res.get("h_map") is not None,
            "swatches": has_ext or has_chains,
            "swatch:color_hchain": has_lines, "swatch:color_vtrail": has_chains,
            "swatch:color_extrema": has_ext,
            "trails": has_chains,
            "arrows": has_ext and "arg" in ext[0],
            "hillshade": single, "surface": single,
            "depth": single and style["surface"],
            "Display mask — hides, never removes": arrangement,
            "Groups": arrangement and has_chains,
            "Topology": has_chains or bool(self._topology_rows_shown),
            "Skeleton": has_chains,
            "Spectrum": tables or res.get("h_map") is not None or has_ext,
            "Decomposition": self._decomposition_of(res) is not None,
            "Transect": raster,
            _RECON_TITLE: self._recon_step() is not None,
        })

    def _refresh_anisotropy(self) -> None:
        """Enable the Anisotropy button for a result with per-scale maxima, and feed an OPEN
        window the active result (with its ROI offset, for bearings)."""
        result = self._active_result
        ok = bool(result) and bool(result.get("extrema"))
        self._anisotropy_button.setEnabled(ok)
        self._anisotropy_button.setToolTip(
            "WTMMM angle statistics — P_a(A) across scales, the gradient plane, M per angle "
            "sector (Arnéodo, Decoster & Roux 2000)" if ok else
            "Needs a WTMM result with per-scale maxima on the active layer")
        win = self._anisotropy_window
        if ok and win is not None and win.isVisible():
            self._feed_anisotropy(win)

    def _feed_anisotropy(self, win) -> None:
        from dynamix.shell.canvas import display_offset

        result = self._active_result
        offset = display_offset(result, self.field) if self.field is not None else (0, 0)
        win.set_result(result, self.field, offset=offset,
                       scale_idx=int(self.transport.slider.value()))

    def _on_anisotropy_button_clicked(self) -> None:
        if self._anisotropy_window is None:
            self._anisotropy_window = AnisotropyWindow(self)
        self._feed_anisotropy(self._anisotropy_window)
        self._anisotropy_window.show()
        self._anisotropy_window.raise_()

    def _refresh_dh_button(self) -> None:
        """Enable/disable "D(h) construction…": needs the partition tables (canonical + hull
        constructions) OR an ``h_map`` (microcanonical histogram alone) on the active result."""
        result = self._active_result
        has_tables = bool(result) and result.get("hd_std") is not None \
            and result.get("hd_cmax") is not None
        has_hmap = bool(result) and result.get("h_map") is not None
        self._dh_button.setEnabled(has_tables or has_hmap)
        self._dh_button.setToolTip(
            "Construct D(h): canonical parametric points, optional Legendre hull of τ(q), "
            "microcanonical histogram, and dominant/non-dominant chain groups"
            if has_tables or has_hmap else
            "Needs partition tables (a full wtmm2d compute) or an h-map "
            "(holder_measure / holder_multiaffine) on the active result")

    def _on_dh_button_clicked(self) -> None:
        """"D(h) construction…" clicked -- a fresh :class:`SpectrumWindow` on the active
        result's own tables/chains/h-map (the MultifractalWindow lifecycle exactly)."""
        result = self._active_result
        if not result:
            return
        hd_std, hd_cmax = result.get("hd_std"), result.get("hd_cmax")
        h_map = result.get("h_map")
        if (hd_std is None or hd_cmax is None) and h_map is None:
            return          # button disabled in this state -- defensive no-op
        params = result.get("params") or {}
        eta_seed, note = _eta_note_of(result)
        chains = result.get("chains") or None
        scales = result.get("scales") if chains else None
        if self._spectrum_construction_window is not None:
            self._spectrum_construction_window.close()
        shape = result.get("_shape")
        log2_L = (float(np.log2(min(int(shape[0]), int(shape[1]))))
                  if shape is not None and len(shape) >= 2 else None)
        window = SpectrumWindow(
            hd_std, hd_cmax, chains=chains, scales=scales, h_map=h_map,
            h_map_estimator=str(params.get("estimator", "")),
            eta_seed=eta_seed, forward_note=note,
            min_chain_len=int(params.get("min_chain_len", 2)),
            log2_L=log2_L, parent=self)
        self._spectrum_construction_window = window
        if self._multifractal_window is not None:
            self._multifractal_window.scaleWindowChanged.connect(window.set_scale_window)
            fit_lo, fit_hi = self._multifractal_window._region.getRegion()
            window.set_scale_window(float(fit_lo), float(fit_hi))
        window.show()
        window.raise_()
        window.activateWindow()

    # -- decomposition grouping aids -----------------------------------------------------------
    def _decomposition_of(self, result) -> "dict | None":
        """The result's decomposition as the grouping window takes it -- ssa2d's eigentriples and
        w-correlations, or tucker's components in the numbering its Orientation knob picks -- or
        None when the result is not a decomposition."""
        if not result:
            return None
        pp = result.get("params") or {}
        view = {"group": str(pp.get("group", "all")), "show": str(pp.get("show", "recon"))}
        if result.get("ssa_components") is not None:
            return {"components": result["ssa_components"], "shares": result["ssa_eigen_share"],
                    "eigenarrays": result.get("ssa_eigenarrays"),
                    "w_correlation": result.get("ssa_w_correlation"),
                    "share_label": "eigenvalue share", **view}
        if result.get("tucker_components") is not None:
            combined = pp.get("pairs") == "combined"
            return {"components": result["tucker_combined_components" if combined
                                          else "tucker_components"],
                    "shares": result["tucker_combined_energy" if combined
                                      else "tucker_component_energy"],
                    "eigenarrays": None, "w_correlation": None,
                    "share_label": "core-energy share" + (", combined" if combined else ""),
                    **view}
        return None

    def _refresh_components_button(self) -> None:
        """Enable "Components…" on a decomposition result, and keep an open window on the
        active one: reloaded when its component stack changed (a recompute, another
        decomposition layer, tucker's Orientation), its picks following the Group knob;
        disabled while the active result is not a decomposition."""
        deco = self._decomposition_of(self._active_result)
        self._components_button.setEnabled(deco is not None)
        self._components_button.setToolTip(
            "Thumbnails, shares and w-correlations of the decomposition -- pick the group the "
            "reconstruction sums (the residual is the data minus it)"
            if deco is not None else
            "Needs a decomposition (ssa2d or tucker_HOOI_HOSVD) on the active layer")
        win = self._components_window
        if win is None or not win.isVisible():
            return
        win.setEnabled(deco is not None)
        if deco is None:
            return
        if win.components is not deco["components"]:
            win.set_decomposition(deco["components"], deco["shares"],
                                  eigenarrays=deco["eigenarrays"],
                                  w_correlation=deco["w_correlation"],
                                  share_label=deco["share_label"])
            win.setWindowTitle(self._components_title())
        win.set_group(deco["group"])
        win.set_show(deco["show"])

    def _components_title(self) -> str:
        return f"Components — {self.layer.name}" if self.layer is not None else "Components"

    def _on_components_button_clicked(self) -> None:
        """"Components…" clicked -- a fresh :class:`ComponentsWindow` on the active
        decomposition, previous one closed first (the SkeletonDialog lifecycle)."""
        deco = self._decomposition_of(self._active_result)
        if deco is None:
            return          # the button is disabled in this state -- a defensive no-op
        if self._components_window is not None:
            self._components_window.close()
        window = ComponentsWindow(
            deco["components"], deco["shares"], eigenarrays=deco["eigenarrays"],
            w_correlation=deco["w_correlation"], group=deco["group"], show=deco["show"],
            share_label=deco["share_label"], title=self._components_title(), parent=self)
        window.groupChanged.connect(self._on_components_group_changed)
        window.forkRequested.connect(
            lambda: self._on_fork_derivative(self.layer.layer_id) if self.layer else None)
        self._components_window = window
        window.show()
        window.raise_()
        window.activateWindow()

    def _on_components_group_changed(self, text: str) -> None:
        """The grouping window picked a group: write it into the tool's Group knob through the
        knob's own edit path -- the control shows it, and the ordinary param change redraws
        from the cache (Group is view-only)."""
        for i, name in enumerate(self._names):
            if (name in ("ssa2d", "tucker_HOOI_HOSVD", "tucker_havok")
                    and not self._bypassed[i]):
                self.strips.strip(i)._on_control_changed("group", text)
                return

    # -- derivative datasets -------------------------------------------------------------------
    def _on_fork_derivative(self, layer_id: int) -> None:
        """"Fork derivative dataset…": what a result holds -- the rasters picked as bands and/or
        the extrema and maxima lines it shows -- written ONCE as a dataset of its own and opened
        like any raw one (:mod:`dynamix.core.derivative`). Unlike a child it never re-processes;
        its provenance records where it came from, as history only. The fork takes the result
        ON SCREEN, so the row is selected first; a result that is computing or pending (its
        knobs moved, not yet run) is refused."""
        from datetime import datetime

        from dynamix.core.derivative import (raster_choices, result_grid, vector_counts,
                                             write_derivative)

        if self.layer is None or self.layer.layer_id != layer_id:
            self.layer_list.select_layer(layer_id)
        layer, result = self.layer, self._active_result
        # A chain ending on a field stage (noise, a band row, a bus) shows a produced FIELD,
        # not a result: the fork takes that field's bands, on its own grid.
        produced = (self._active_field[1] if self._active_field is not None and layer is not None
                    and self._active_field[0] == layer.layer_id and not result else None)
        if (layer is None or layer.layer_id != layer_id or (not result and produced is None)
                or self.is_computing or layer_id in self._pending_layers):
            self._notify("fork a derivative from a result that is on screen — run it first",
                         "status")
            return
        src = self.project.sources.get(layer.source_id)
        dataset = (src.label or Path(src.path).stem) if src is not None else layer.name
        row = self.layer_list.layer_text(layer_id) or layer.name
        prefix = f"{layer.name} · "              # the row reads "<layer name> · <view note>"
        note = row[len(prefix):] if row.startswith(prefix) else "shown"
        rasters = (_field_choices(produced, note) if produced is not None
                   else raster_choices(result, shown_label=f"as shown ({note})"))
        dialog = ForkDialog(name=f"{dataset} · {row}", raster_labels=[l for l, _a in rasters],
                            vector_counts=vector_counts(result), parent_name=dataset,
                            parent=self)
        if not dialog.exec():
            return
        choice = dialog.choices()
        path = self._derivative_path(choice["name"], temporary=choice["temporary"])
        if path is None:
            return
        field = self._fields.get(layer_id, self.field)
        if produced is not None:
            frame, x_axis, y_axis = produced.frame, produced.x_axis, produced.y_axis
        else:
            frame, x_axis, y_axis = result_grid(result, field)
        derived = {"from_dataset": src.path if src is not None else None,
                   "from_sha256": src.sha256 if src is not None else None,
                   "from_layer": layer.name,
                   "chain": [{"device": r.device, "params": dict(r.params)}
                             for r in layer.chain.steps],
                   "roi_window": layer.tags.get("roi.window"),
                   "forked": datetime.now().isoformat(timespec="seconds")}
        # The georeference rides over (geo.mapping, the units, backproject read provenance
        # "crs" with the grid's axes). Nothing else of the parent's provenance does: its
        # source / window / full_dims describe the PARENT file, and a native read through
        # them would fetch the parent's pixels instead of the derivative's.
        provenance = {"derived": derived}
        crs = (getattr(field, "provenance", None) or {}).get("crs")
        if crs:
            provenance["crs"] = crs
        try:
            write_derivative(path, bands=[rasters[i] for i in choice["bands"]], frame=frame,
                             x_axis=x_axis, y_axis=y_axis, name=choice["name"],
                             units=str(getattr(field, "units", "") or ""),
                             provenance=provenance,
                             vectors=result if choice["vectors"] and result else None)
        except ValueError as exc:
            self._notify(f"fork refused: {exc}", "status")
            return
        self._open_derivative(path, name=choice["name"], temporary=choice["temporary"],
                              nest_under=layer.source_id if choice["nest"] else None,
                              vectors=choice["vectors"])

    def _derivative_path(self, name: str, *, temporary: bool) -> "str | None":
        """Where a derivative is written: this session's scratch folder (temporary), or wherever
        the Save-As dialog says (None when cancelled)."""
        import uuid

        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "derivative"
        if temporary:
            return str(Path(_derivative_scratch()) / f"{slug}-{uuid.uuid4().hex[:8]}.npz")
        folder = self._project_path.parent if self._project_path is not None else Path.home()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save derivative dataset", str(folder / f"{slug}.npz"),
            "Derivative dataset (*.npz)")
        if not path:
            return None
        return path if path.lower().endswith(".npz") else path + ".npz"

    def _open_derivative(self, path: str, *, name: str, temporary: bool,
                         nest_under: "str | None", vectors: bool) -> None:
        """Register and open a derivative file as a dataset (the ordinary ``load_field`` path).
        The source is registered FIRST with its name and placement, so the dataset row is built
        with them; a derivative carrying vectors gets its loader as its first step."""
        from dynamix.core.rasterfield import RasterField

        source = self.project.add_source(path, label=name, sha256=sha256_of(path))
        source.temporary = temporary
        source.nest_under = nest_under
        self.load_field(RasterField.from_file(path), path, inert=True)
        self.layer.name = name                  # load_field names a layer by its file's stem
        self.layer_list.set_layer_name(self.layer.layer_id, name)
        if vectors:
            self._set_recipe(["derived_vectors"], [{"_path": str(path)}])
            self.layer.chain = self._chain()
            self._snapshot_recipe()
            self._build_strips()
            self._start_worker()

    def _on_save_derivative(self, source_id: str) -> None:
        """"Save derivative as…" on a TEMPORARY derivative: its file is copied where the user
        says, and from then on it is permanent (a saved project keeps it)."""
        import shutil

        source = self.project.sources.get(source_id)
        if source is None or not source.temporary:
            return
        folder = self._project_path.parent if self._project_path is not None else Path.home()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source.label or "derivative").strip("_")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save derivative dataset", str(folder / f"{slug or 'derivative'}.npz"),
            "Derivative dataset (*.npz)")
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path += ".npz"
        shutil.copy2(source.path, path)
        old_path = source.path
        source.path, source.temporary, source.sha256 = path, False, sha256_of(path)
        for layer in self.project.layers:
            if layer.source_id == source_id:
                layer.chain = Chain(tuple(
                    DeviceRef(r.device, {**r.params, "_path": path})
                    if r.device == "derived_vectors" else r
                    for r in layer.chain.steps)).materialized()
        if self.layer is not None and self.layer.source_id == source_id \
                and "derived_vectors" in self._names:
            self._params = [{**p, "_path": path} if n == "derived_vectors" else p
                            for n, p in zip(self._names, self._params)]
            self.layer.chain = self._chain()
            self._start_worker()
        self._repoint_bus_sends(old_path, path)

    def _zoom_to_dataset(self) -> None:
        """Fit the raster view to the selected row: its ROI window when it runs on one, else
        the raster on screen (the same framing an open does)."""
        window = self.layer.tags.get("roi.window") if self.layer is not None else None
        if window:
            r, c, h, w = (int(v) for v in window.split(","))
            self.canvas.view.setRange(xRange=(c - 0.5, c + w - 0.5),
                                      yRange=(r - 0.5, r + h - 0.5), padding=0.02)
            return
        self.canvas.view.autoRange(items=[self.canvas.image_item])

    # -- band rows: removing a band from a stack dataset --------------------------------------
    def _ensure_band_layers(self, master, field) -> None:
        """A STACK dataset's bands as child rows, once: each a layer whose rack starts with
        ``band_select`` naming its band (by name, so removing another band never shifts
        it) -- drop any tool on a band row to run it on that band. A stack whose provenance
        does not name its bands gets ``"#k"`` names (the multiband-file band tokens)."""
        values = np.asarray(getattr(field, "values", None)) if field is not None else None
        if values is None or values.ndim != 3 or master.parent_id is not None:
            return
        if any(l.parent_id == master.layer_id and l.tags.get("band.id")
               for l in self.project.layers):
            return
        nc = int(values.shape[-1])
        names = list((field.provenance or {}).get("bands") or [])
        if len(names) != nc:
            names = [f"#{k}" for k in range(nc)]
            field.provenance["bands"] = names
        for k, band_id in enumerate(names):
            chain = Chain((DeviceRef("band_select", {"band": k + 1,
                                                     "_band_id": str(band_id)}),)).materialized()
            layer = self.project.add_layer(
                str(band_id).split("/", 1)[0], master.source_id, chain,
                parent_id=master.layer_id,
                tags=_inherited_window(master, **{"band.id": str(band_id)}))
            self.add_layer_row(layer, field)

    def _on_remove_band_layer(self, layer_id: int) -> None:
        """A band row's "Remove band from project" / Delete: the row goes (with any results
        under it -- one confirmation, the ordinary delete) and its band leaves the dataset.
        The last band cannot go: remove the dataset instead."""
        layer = self._layer_by_id.get(layer_id)
        master = self._layer_by_id.get(layer.parent_id) if layer is not None else None
        field = self._fields.get(master.layer_id) if master is not None else None
        if layer is None or field is None or not layer.tags.get("band.id"):
            return
        names = list((field.provenance or {}).get("bands") or [])
        band_id = layer.tags["band.id"]
        if np.asarray(field.values).ndim != 3 or band_id not in names:
            self._notify(f"{layer.name}: the dataset's last band — remove the dataset instead",
                         "status")
            return
        self._on_remove_requested(layer_id)
        if layer_id in self._layer_by_id:           # refused (locked) or cancelled
            return
        self._on_remove_band(master.source_id, names.index(band_id))

    def _on_remove_band(self, source_id: str, k: int) -> None:
        """"Remove band from project": band ``k`` leaves the dataset -- never the file. The
        stack is sliced in memory, the source's band list shrinks (a saved project reopens
        exactly the bands left), and every layer of the dataset takes a ``data.bands``
        identity tag so no result computed on the fuller stack answers for this one; those
        layers recompute when next shown. The last band cannot go: remove the dataset."""
        import dataclasses
        import hashlib
        import json

        source = self.project.sources.get(source_id)
        master = next((l for l in self.project.layers
                       if l.source_id == source_id and l.parent_id is None), None)
        field = self._fields.get(master.layer_id) if master is not None else None
        values = np.asarray(getattr(field, "values", None)) if field is not None else None
        if source is None or values is None or values.ndim != 3:
            return
        nc = int(values.shape[-1])
        if not 0 <= k < nc:
            return
        bands = list(source.bands) if source.bands else [f"#{i}" for i in range(nc)]
        names = list((field.provenance or {}).get("bands") or bands)
        gone = str(names[k]).split("/", 1)[0] if k < len(names) else f"band {k + 1}"
        kept = np.delete(values, k, axis=-1)
        prov = {**(field.provenance or {}),
                "bands": [n for i, n in enumerate(names) if i != k]}
        new = dataclasses.replace(field, values=kept[..., 0] if kept.shape[-1] == 1 else kept,
                                  provenance=prov)
        source.bands = [b for i, b in enumerate(bands) if i != k]
        tag = hashlib.sha1(json.dumps(source.bands).encode()).hexdigest()[:10]
        for layer in self.project.layers:
            if layer.source_id != source_id:
                continue
            layer.tags["data.bands"] = tag
            if self._fields.get(layer.layer_id) is field:
                self._fields[layer.layer_id] = new
            raw = layer.tags.get("ui.composite")
            if raw:
                layer.tags["ui.composite"] = json.dumps(_composite_without_band(raw, k))
        if self.field is field:
            self.field = new
        self.layer_list.sync_band_rows(source_id)
        if self.layer is not None and self.layer.source_id == source_id:
            self._select_layer(self.layer)
        self._notify(f"{gone}: removed from the project — the file is untouched", "status")

    # -- band buses (rack-head routing) --------------------------------------------------------
    def _root_layer(self, layer):
        """The dataset (master) row ``layer`` descends from."""
        by_id = {l.layer_id: l for l in self.project.layers}
        while layer is not None and layer.parent_id is not None:
            layer = by_id.get(layer.parent_id)
        return layer

    def _bus_candidates(self, host_field) -> list:
        """Every loaded raster dataset as a routing candidate for a bus on ``host_field``'s
        grid: its bands as sends, or greyed out with the reason it cannot send (another
        grid, or no file to read the raw band back from)."""
        from dynamix.core.bus import grid_mismatch, native_grid, sends_for_field

        host = native_grid(host_field)
        out = []
        for layer in self.project.layers:
            if layer.parent_id is not None or self._is_point_layer(layer):
                continue
            field = self._fields.get(layer.layer_id)
            if field is None or np.asarray(getattr(field, "values", None)).ndim not in (2, 3):
                continue
            src = self.project.sources.get(layer.source_id)
            path = src.path if src is not None else None
            if not path or not Path(path).is_file():
                out.append({"label": layer.name, "sends": [],
                            "reason": "in memory only — no file to read raw bands from"})
                continue
            reason = grid_mismatch(host, native_grid(field))
            out.append({"label": layer.name, "sends": sends_for_field(field, path, layer.name),
                        "reason": f"another grid: {reason}" if reason else None})
        return out

    def _on_bus_requested(self, layer_id: int) -> None:
        """"Route bands to a new bus…": pick raw bands on this dataset's grid; the bus lands as
        a CHILD of the dataset whose rack starts with it (on the active ROI when there is one,
        like any tool child). Drop a tool on it to consume the bus; its results are the
        returns."""
        root = self._root_layer(next((l for l in self.project.layers
                                      if l.layer_id == layer_id), None))
        host = self._fields.get(root.layer_id) if root is not None else None
        if host is None:
            return
        dialog = BusDialog(self._bus_candidates(host), parent=self)
        if not dialog.exec():
            return
        sends = dialog.sends()
        if not sends:
            self._notify("no bands routed — nothing to build", "status")
            return
        label = f"bus ({len(sends)} band{'s' if len(sends) != 1 else ''})"
        self._add_bus_child(root, host, sends, label)

    def _add_bus_child(self, root, host, sends: list, label: str) -> None:
        """A bus-headed child of dataset ``root`` (on its active ROI when there is one, like
        any tool child), selected so it resolves."""
        import json

        chain = Chain((DeviceRef("bus", {"_sends": json.dumps(sends)}),)).materialized()
        roi = self._active_roi_for(root.source_id)
        tags = _inherited_window(root)
        if roi is not None:
            tags["roi.window"] = f"{roi.row},{roi.col},{roi.h},{roi.w}"
        name = f"{label} @{roi.label}" if roi is not None else f"{root.name} · {label}"
        layer = self.project.add_layer(name, root.source_id, chain,
                                       parent_id=root.layer_id, tags=tags)
        if roi is not None:
            layer.roi_id = roi.roi_id
        self.add_layer_row(layer, host)
        self.layer_list.select_layer(layer.layer_id)
        self._notify(f"{label} routed — drop pca, tucker or another multi-band tool on it",
                     "status")

    # -- live layer sends -------------------------------------------------------------------
    def _layer_stamp(self, ref) -> str:
        """The fingerprint of what layer ``ref`` shows: its resolve identity, its predicted
        cache keys and its whole chain (view choices included). Any upstream change changes
        it, and with it the key of every bus that sends ``ref``."""
        import hashlib
        import json

        payload = [source_identity(ref), self._cache_keys_for(ref),
                   [(r.device, r.params) for r in ref.chain.steps]]
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()
                            ).hexdigest()[:16]

    def _refresh_bus_stamps(self, layer, _visiting: tuple = ()) -> bool:
        """Bring ``layer``'s LAYER sends' stamps up to date (a bus it sends from first, so
        stamps nest); ``True`` when anything changed. A locked / frozen bus keeps its stamps
        (its cache is pinned). Raises on a routing loop."""
        import json

        from dynamix.core.bus import parse_sends

        steps = list(layer.chain.steps)
        if not steps or steps[0].device != "bus" or _lock_notice(layer) is not None:
            return False
        if layer.layer_id in _visiting:
            raise ValueError(f"routing loop: {layer.name} feeds itself through a stack")
        sends = parse_sends(steps[0].params.get("_sends"))
        changed = False
        for send in sends:
            ref = self._layer_by_id.get(int(send["layer"])) if "layer" in send else None
            if ref is None:
                continue
            self._refresh_bus_stamps(ref, _visiting + (layer.layer_id,))
            stamp = self._layer_stamp(ref)
            if send.get("stamp") != stamp:
                send["stamp"], changed = stamp, True
        if not changed:
            return False
        text = json.dumps(sends)
        steps[0] = DeviceRef("bus", {**steps[0].params, "_sends": text})
        layer.chain = Chain(tuple(steps)).materialized()
        recipe = self._recipes.get(layer.layer_id)
        if recipe and recipe[0].get("device") == "bus":
            recipe[0] = {**recipe[0], "params": {**recipe[0].get("params", {}), "_sends": text}}
        if layer is self.layer and self._names[:1] == ["bus"]:
            self._params[0] = dict(self._params[0], _sends=text)
        return True

    def _bus_plan(self, layer, _visiting: tuple = ()) -> list:
        """``[(ref_layer, ref_field, stamp)]`` the worker must resolve BEFORE ``layer`` -- its
        layer sends whose shown raster is not filed yet, dependencies first. Raises (with the
        send named) for a deleted referenced layer or a routing loop."""
        from dynamix.core.bus import _PLANES, parse_sends

        steps = layer.chain.steps
        if not steps or steps[0].device != "bus":
            return []
        if layer.layer_id in _visiting:
            raise ValueError(f"routing loop: {layer.name} feeds itself through a stack")
        plan, seen = [], set()
        for send in parse_sends(steps[0].params.get("_sends")):
            if "layer" not in send:
                continue
            ref = self._layer_by_id.get(int(send["layer"]))
            if ref is None:
                raise ValueError(f"{send.get('label') or 'a send'}: its layer was deleted — "
                                 f"right-click the stack, Edit bus sends…")
            for item in self._bus_plan(ref, _visiting + (layer.layer_id,)):
                if item[2] not in seen:
                    seen.add(item[2])
                    plan.append(item)
            stamp = str(send.get("stamp"))
            if stamp not in _PLANES and stamp not in seen:
                seen.add(stamp)
                plan.append((ref, self._fields.get(ref.layer_id), stamp))
        return plan

    def _on_stack_requested(self, layer_ids) -> None:
        """"Build band stack from N selected layers": a stack child of their dataset whose
        rack starts with a bus of LIVE layer sends -- what each layer shows, referenced, never
        copied: change one and the stack follows. Same grid only; an ROI result has its own
        grid and is refused (build on whole-grid layers)."""
        from dynamix.core.bus import grid_mismatch, native_grid

        layers = [self._layer_by_id[i] for i in layer_ids if i in self._layer_by_id]
        if len(layers) < 2:
            self._notify("select two or more layers to build a band stack", "status")
            return
        root = self._root_layer(layers[0])
        host = self._fields.get(root.layer_id) if root is not None else None
        if host is None:
            return
        refused = []
        for lay in layers:
            field = self._fields.get(lay.layer_id)
            if lay.tags.get("roi.window"):
                refused.append(f"{lay.name}: an ROI result (its own grid)")
            elif field is None:
                refused.append(f"{lay.name}: not loaded")
            else:
                why = grid_mismatch(native_grid(host), native_grid(field))
                if why:
                    refused.append(f"{lay.name}: another grid ({why})")
        if refused:
            self._notify("band stack refused — " + "; ".join(refused), "status")
            return
        sends = [{"layer": lay.layer_id, "label": lay.name, "stamp": self._layer_stamp(lay)}
                 for lay in layers]
        self._add_bus_child(root, host, sends, f"stack ({len(sends)} layers)")

    def _on_bus_edit_requested(self, layer_id: int) -> None:
        """"Edit bus sends…": re-open the routing with the bus's current sends pre-ticked, in
        order; OK writes the new list through the ordinary parameter path (locks honoured,
        recompute, cache hit when routed back)."""
        import json

        from dynamix.core.bus import parse_sends

        layer = next((l for l in self.project.layers if l.layer_id == layer_id), None)
        if layer is None or not layer.chain.steps or layer.chain.steps[0].device != "bus":
            return
        root = self._root_layer(layer)
        host = self._fields.get(root.layer_id) if root is not None else None
        if host is None:
            return
        current = parse_sends(layer.chain.steps[0].params.get("_sends"))
        dialog = BusDialog(self._bus_candidates(host), current, title="Edit bus sends",
                           parent=self)
        if not dialog.exec():
            return
        sends = dialog.sends()
        if not sends:
            self._notify("a bus needs at least one send — unchanged", "status")
            return
        if self.layer is None or self.layer.layer_id != layer_id:
            self.layer_list.select_layer(layer_id)
        self._on_param_changed(self._names.index("bus"), "_sends", json.dumps(sends))

    def _land_field_result(self, result) -> None:
        """A chain that ENDS on a field stage produced a field, not a result -- and you see
        what the chain makes, the way an inserted effect is heard at once: the produced field
        goes on the canvas (a stack through the Composite mixer), so noise shows its noise, a
        band row its band, a bus its sends. Bypass the stage to A/B against the dry data. The
        field is what "Fork derivative dataset…" takes from this row."""
        self._active_result = {}               # never a previous layer's result, for the fork
        self._active_field = None
        if not hasattr(result, "values"):
            self._notify("chain ends on a field transform (noise) — add wtmm2d after it to "
                         "analyse", "status")
            return
        self.canvas.set_field(result)
        values = np.asarray(result.values)
        self._holder_raster_ref = (id(values), values)   # the next result lands the dataset back
        self._active_field = (self.layer.layer_id if self.layer is not None else None, result)
        self._sync_composite_panel(result)
        self._refresh_panel_relevance()
        tail = self._names[-1] if self._names else ""
        if self._names == ["bus"]:
            self._notify("bus: drop pca, tucker or another multi-band tool after it to "
                         "analyse its sends", "status")
        elif tail == "noise":
            self._notify("showing the noised field — add wtmm2d (or any tool) after it to "
                         "analyse; bypass noise to compare with the raw data", "status")
        elif tail != "band_select":
            self._notify(f"showing what {tail} produces — add an analysing tool after it",
                         "status")

    def _repoint_bus_sends(self, old_path: str, new_path: str) -> None:
        """A temporary derivative saved elsewhere: every bus sending from its old file follows
        it (the temporary file goes with the session)."""
        import json

        from dynamix.core.bus import parse_sends

        for layer in self.project.layers:
            steps = list(layer.chain.steps)
            if not steps or steps[0].device != "bus":
                continue
            sends = parse_sends(steps[0].params.get("_sends"))
            if not any(s.get("path") == old_path for s in sends):
                continue
            moved = [{**s, "path": new_path} if s.get("path") == old_path else s
                     for s in sends]
            steps[0] = DeviceRef("bus", {**steps[0].params, "_sends": json.dumps(moved)})
            layer.chain = Chain(tuple(steps)).materialized()
            recipe = self._recipes.get(layer.layer_id)
            if recipe and recipe[0].get("device") == "bus":
                recipe[0] = {**recipe[0], "params": {**recipe[0].get("params", {}),
                                                     "_sends": json.dumps(moved)}}
        if self.layer is not None and self._names[:1] == ["bus"]:
            self._params[0] = dict(self.layer.chain.steps[0].params)

    # -- holder_map raster display -------------------------------------------------------------
    def _sync_holder_raster(self, result: dict) -> None:
        """When the ACTIVE layer's result carries ``"h_map"`` (the holder_map transform's own
        contract -- no other device stamps that key, the ``backproject``/``points_px``
        precedent), show the exponent raster on the canvas IN PLACE of the raw field, through
        the ordinary :meth:`Canvas.set_field` path so stretch/hillshade/decimation apply to it
        like any raster; restore the raw field the moment a result WITHOUT one lands (device
        removed or bypassed). Both directions are identity-edge-triggered on
        ``self._holder_raster_ref`` so an ordinary landing never re-pays ``set_field``'s
        decimation/stretch work; a param scrub mints a NEW h_map array (results are immutable),
        which is exactly what re-triggers the display.

        ``_displayed_layer_name`` deliberately stays the layer's own name: the exponent raster
        lives on the SAME pixel grid as the field it replaces, so a point layer's backproject
        scatter registered against this layer still belongs on screen.

        The show branch also clears the WTMM overlays/picking: a holder_map result carries no
        chains, and the landing's own overlay redraw only runs for non-empty ``extrema`` --
        without this, the PREVIOUS result's chains would linger over the exponent raster."""
        h_map = _display_raster_of(result)
        # Row identity: while a derived raster is displayed, the layer's row
        # shows WHAT it is ("name · h(x)" / "name · band[lo,hi)") -- display text only,
        # layer.name untouched; restored with the raw field below.
        if h_map is not None and self.field is not None and self.layer is not None \
                and self.layer.visible and not self._is_point_layer(self.layer):
            if self._holder_raster_ref is None or self._holder_raster_ref[0] != id(h_map):
                derived = self._strided_display_field(result, h_map)
                if derived is None:
                    derived = _roi_display_field(result, h_map, self.field, self.layer.name)
                if derived is None:
                    derived = dataclasses.replace(
                        self.field, values=np.asarray(h_map, dtype=np.float64),
                        name=f"{self.layer.name}")
                offset = result.get("_raster_display_offset")
                if result.get("_display_stride"):
                    # A drawing on its own coarser grid: never a surface source or an export.
                    self._derived_fields.pop(self.layer.layer_id, None)
                else:
                    if offset is not None:
                        # The dataset sits where its samples register: its axes move by the
                        # offset, so an export keeps that registration.
                        dx, dy = (float(d) for d in offset)
                        derived = dataclasses.replace(
                            derived,
                            x_axis=axis_at(derived.x_axis, np.arange(len(derived.x_axis)) + dx),
                            y_axis=axis_at(derived.y_axis, np.arange(len(derived.y_axis)) + dy))
                    self._derived_fields[self.layer.layer_id] = derived
                if offset is not None:
                    # The canvas places a raster by pixel index, so the copy it draws carries
                    # the offset; the dataset above has none in its provenance.
                    derived = dataclasses.replace(derived, provenance={
                        **(derived.provenance or {}), "display_offset": tuple(offset)})
                self.canvas.set_field(derived)
                self._apply_raster_visibility(showing_product=True)
                self.canvas.clear_overlays()
                self.canvas.set_pick_chains(None)
                self._holder_raster_ref = (id(h_map), h_map)
                if result.get("raster_out") is not None:
                    # Name WHAT the derived raster is from the producer's own params -- the
                    # band devices write the band, pca the component, tucker its show mode;
                    # anything else says "derived" rather than faking a band note such as
                    # "recon[0,0)" for a non-band producer.
                    pp = result.get("params", {})
                    if result.get("_view_note"):
                        note = str(result["_view_note"])        # the tool's own label
                    elif "h_lo" in pp:
                        note = f"recon[{pp.get('h_lo', 0):g},{pp.get('h_hi', 0):g})"
                    elif pp.get("show") == "component":
                        # A decomposition component: its index (clipped as the view clips it)
                        # and its share of the core energy.
                        combined = pp.get("pairs") == "combined"
                        shares = result.get("tucker_combined_energy" if combined
                                            else "tucker_component_energy")
                        n = len(shares) if shares is not None else 1
                        k = min(max(int(pp.get("component", 1)), 1), max(n, 1))
                        tail = ", combined" if combined else ""
                        note = (f"C{k} ({100.0 * float(shares[k - 1]):.0f}%{tail})"
                                if shares is not None and n else f"C{k}")
                    elif "show" in pp:
                        note = str(pp.get("show"))
                    elif "component" in pp:
                        note = f"PC{int(pp.get('component', 1))}"
                    else:
                        note = "derived"
                else:
                    note = "h(x)"
                self.layer_list.set_layer_name(self.layer.layer_id,
                                               f"{self.layer.name} · {note}")
                psnr = result.get("psnr_db")
                if psnr is not None:
                    self._notify(f"band reconstruction: {psnr:.1f} dB "
                                 f"({result.get('band_density', 0) * 100:.0f}% of pixels)",
                                 "status")
        elif self._holder_raster_ref is not None:
            if self.field is not None:
                self.canvas.set_field(self.field)
                self._apply_raster_visibility(showing_product=False)
            if self.layer is not None:
                self.layer_list.set_layer_name(self.layer.layer_id, self.layer.name)
            self._holder_raster_ref = None

    def _strided_display_field(self, result: dict, raster):
        """The display field of an output drawn on its own coarser grid (``result[
        "_display_stride"]``, which :meth:`_show_lazy_output` sets for the M-Z thumbnail), or
        ``None`` for any other result. Sample k describes the centre of file pixels [k*s,
        (k+1)*s), pixel k*s + (s-1)/2, so the canvas's own picture registration draws it over
        exactly that block, and the axes are the field's own interpolated at those centres
        (held at the last pixel for a partial last block, as a picture's are). The provenance is
        a COPY (the field's shared dict is never mutated), and the field is a drawing only:
        nothing analyses it.

        An output that states where its samples register (``result["_raster_display_offset"]
        = (dx, dy)``, copied from the output's ``display_offset``) holds point samples instead:
        sample k is file pixel k*s, registered at k*s + dx along x and k*s + dy along y, like
        the LastWave M-Z thumbnail. The copy then carries ``display_anchor`` "sample", so the
        canvas centres block k on file pixel k*s and the offset moves it onto the registered
        position, and the axes are the field's own at those positions (linear along each
        axis)."""
        s = result.get("_display_stride")
        if not s or self.field is None:
            return None
        s = int(s)
        field = self.field
        prov = {**(getattr(field, "provenance", None) or {}), "display_stride": s,
                "full_dims": tuple(result["_full_dims"])}
        values = np.asarray(raster, dtype=np.float64)
        offset = result.get("_raster_display_offset")
        if offset is not None:
            prov["display_anchor"] = "sample"
            dx, dy = (float(d) for d in offset)
            return dataclasses.replace(
                field, values=values,
                x_axis=axis_at(field.x_axis, np.arange(values.shape[1]) * s + dx),
                y_axis=axis_at(field.y_axis, np.arange(values.shape[0]) * s + dy),
                provenance=prov, name=self.layer.name)

        def centres(axis, m):
            axis = np.asarray(axis, dtype=np.float64)
            return np.interp(np.arange(m) * s + (s - 1) / 2, np.arange(axis.size), axis)

        return dataclasses.replace(field, values=values,
                                   x_axis=centres(field.x_axis, values.shape[1]),
                                   y_axis=centres(field.y_axis, values.shape[0]),
                                   provenance=prov, name=self.layer.name)

    def _on_skeleton_selection_requested(self, indices: list) -> None:
        """``SkeletonDialog.selectionRequested`` -> ``GroupPalette.apply_picks(op="replace")``,
        naming the ACTIVE layer -- "Select h-range" REPLACES the shared selection ("applying feeds the SHARED selection (op=replace), so the scene highlights what the plot
        selected"), the same op the transect swath uses for an identical "this IS now the
        selection" gesture. A no-op with no active layer -- the button that
        opens the dialog is already gated on one existing, but the dialog itself outlives a
        layer removal (nothing closes it), so this guard is not purely defensive."""
        if self.layer is None:
            return
        lid = self.layer.layer_id
        self._group_palette.apply_picks([(lid, int(i)) for i in indices], "replace")

    def _push_skeleton_selection(self) -> None:
        """The OTHER half of the bidirectional sync ("Scene selection changes
        re-draw the dialog's own highlight") -- pushes the active layer's own current selection
        (``GroupPalette.selection()``, the exact same buffer ``_on_group_membership_changed``
        already reads for the canvas overlay) into the open dialog, or does nothing when none is
        open. A separate method (not inlined into ``_on_group_membership_changed``) so
        ``_on_skeleton_button_clicked`` can call it once, right after construction, without
        duplicating the "no active layer" guard/index derivation a SECOND time."""
        if self._skeleton_dialog is None:
            return
        if self.layer is None:
            self._skeleton_dialog.set_selection([])
            return
        lid = self.layer.layer_id
        indices = sorted(idx for (layer_id, idx) in self._group_palette.selection()
                          if layer_id == lid)
        self._skeleton_dialog.set_selection(indices)

    # -- transects ------------------------------------
    def _on_canvas_transect_drawn(self, a: tuple, b: tuple) -> None:
        """The canvas's second click landed -- append it to the panel (which selects the new row,
        triggering :meth:`_on_transect_selection_changed`'s own swath-select/highlight)."""
        self._transect_panel.add_record(a, b)

    def _on_transect_records_changed(self, records: list) -> None:
        """Every add/delete/undo/visibility/buffer edit -- keep ``Project.transects`` (the LIVE
        persistence copy, mirroring ``self.project.user_links``'s own always-current-object
        doctrine) and the canvas's drawn lines both in step with the panel's own state."""
        self.project.transects = list(records)
        self.canvas.set_transects(records, self._transect_panel.selected_id())

    def _on_transect_selection_changed(self, record) -> None:
        """A row was selected (or a buffer edit re-fired this for the same row): highlight its
        line on the canvas (thicker, via ``Canvas.set_transects``' own selected-id argument) and
        swath-select the chains within its buffer, REPLACING the shared selection (the same ``op="replace"`` the skeleton dialog's "Select h-range" already uses for an
        identical "this IS now the selection" gesture, see ``_on_skeleton_selection_requested``'s
        own docstring).

        Uses ``Canvas.pick_chains()`` -- the SAME display-shifted chains list an ordinary click
        pick already resolves against -- not a fresh, unshifted read of the active
        result. A no-op (highlight still updates) with no active layer or no chains on screen:
        there is nothing to select against."""
        self.canvas.set_transects(
            self._transect_panel.records(), None if record is None else record.transect_id)
        if record is None or self.layer is None:
            return
        chains = self.canvas.pick_chains() or []
        indices = chains_in_buffer(chains, record.a, record.b, record.buffer_px)
        self._group_palette.apply_picks(
            [(self.layer.layer_id, int(i)) for i in indices], "replace")

    def _on_transect_plot_requested(self, record) -> None:
        """Plot button, or a double-click on a row -- sample the ACTIVE RASTER (``self.field``,
        the same field the canvas itself is currently showing -- the design: "the panel asks
        MainWindow for the active field at plot time") along ``record``'s own segment and open an
        independent :class:`~dynamix.shell.profile_dialog.ProfileDialog` for it. ``spacing`` reuses
        ``_skeleton_px_size`` -- the identical "frame units per pixel, or None for bare pixels"
        reading the skeleton dialog's own x-axis offset already relies on, falling back to ``1.0``
        (bare pixels) exactly as :func:`dynamix.core.transect.sample_profile`'s own default does.
        A no-op with no field loaded -- there is no raster to sample."""
        if self.field is None:
            return
        values = np.asarray(getattr(self.field, "values", self.field))
        spacing = self._skeleton_px_size() or 1.0
        a, b = record.a, record.b
        st = display_stride(self.field)
        if st > 1:
            # A display picture: the endpoints are FILE pixels; sample k
            # of the picture is file pixel k*st + st//2, so map through that and keep the
            # distance axis in file pixels.
            a = ((a[0] - st // 2) / st, (a[1] - st // 2) / st)
            b = ((b[0] - st // 2) / st, (b[1] - st // 2) / st)
            spacing *= st
        dist, z = sample_profile(values, a, b, spacing=spacing)
        dialog = ProfileDialog(dist, z, label=f"T{record.transect_id}", parent=self)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    # -- project files -----------------------------------------------------------------------
    def _save_project_to(self, path) -> Path:
        """Write the project to ``path`` (dialog-free — the testable inner method)."""
        written = save_project(self.project, path)
        self._project_path = written
        self.setWindowTitle(f"{TITLE} — {written.stem}")
        temporary = self.project.temporary_sources()
        if temporary:
            names = ", ".join(s.label or Path(s.path).stem for s in temporary)
            self._notify(f"left out of the saved project (temporary derivatives): {names} — "
                         "“Save derivative as…” on its row keeps one", "status")
        return written

    def _on_save_project(self) -> None:
        if self._project_path is None:
            self._on_save_project_as_clicked()
        else:
            self._save_project_to(self._project_path)

    def _on_save_project_as_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project", "", "DynamiX project (*.dynamix)")
        if path:
            self._save_project_to(path)

    def _on_open_project_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", "", "DynamiX project (*.dynamix)")
        if path:
            self._open_project_path(path)

    def _open_project_path(self, path) -> None:
        """Replace the window's world with a saved project (dialog-free inner method).

        Sources open once each and their field is shared by every layer over them, mirroring how
        ROI and refined-run layers already share the parent's field object -- EXCEPT for a points
        source, where the actual field content is layer-scoped, not source-scoped: two layers can share one CSV source yet each carry a
        DIFFERENT ``points.mapping`` tag (the same ambiguous-header dialog flow reachable twice on
        import). ``fields_by_source`` is therefore keyed on ``(sid, raw_mapping)``, not ``sid``
        alone -- a raster source's key is always ``(sid, None)`` (every raster layer over it
        shares the identical field, unchanged), while a points source's key varies with its
        layer's own mapping, so a second layer's distinct mapping is read and cached under its
        OWN key rather than silently collapsing onto whichever layer this loop met first.

        A missing source skips its layers with a visible count rather than refusing the whole
        project — the user needs the project OPEN to repair it (projectfile.py's own doctrine);
        ``relocate_source`` has no gesture yet, so the honest v1 is skip-and-say. Changed sources
        load (the bytes are different, not absent) with the drift stated. Selection lands on the
        first restored layer, which rebuilds the zone and dispatches its compute exactly like a
        fresh open.
        """
        path = Path(path)
        project = open_project(path)
        missing = set(missing_sources(project, path))
        drifted = set(changed_sources(project, path))
        self.project = project
        self._reload_reference_layers()        # reference layers ride the project
        self._project_path = path
        self._row_layers = []
        self._layer_by_id = {}
        self._fields = {}
        self._recipes = {}
        # The reopened project's layers may reuse the old ids: output bookkeeping keyed by them
        # goes. An output job still in flight lands on a layer that is no longer registered and
        # is ignored (_land_output's identity check).
        self._out_keys = {}
        self._out_request = None
        self._out_held = None
        self._recon_run = None
        self.layer = None
        self._active_result = None
        # Double-update guard: True when a result landed while the raster canvas was hidden
        # (3-D view on screen), so its overlay draw was deferred; _set_center_view redraws it on
        # flip back to raster.
        self._canvas_overlay_dirty = False
        self.layer_list.reset_project(project)
        # Restore the transect list + drawn lines --
        # additive "transects" payload key; an OLD project file has none (`Project.from_payload`'s
        # `payload.get("transects", [])` -> an empty list), so this is a no-op there, exactly like
        # every other additive-key restore in this method.
        self._transect_panel.set_records(project.transects)
        self.canvas.set_transects(self._transect_panel.records(), self._transect_panel.selected_id())
        fields_by_source: dict[tuple[str, str | None], object] = {}
        skipped = 0
        first_id = None
        for layer in project.layers:
            sid = layer.source_id
            if sid in missing:
                skipped += 1
                continue
            is_points = project.sources[sid].kind == "points"
            # Reopen a CSV point source through its OWN stored mapping, never through
            # ``point_import.load_points`` -- that function's auto-detect/dialog path exists only
            # for the FIRST import; a reopen must never re-ask, even for a file whose
            # auto-detection would now be ambiguous (see ``point_import.py``'s own module
            # docstring for why the mapping is persisted rather than re-derived).
            raw_mapping = layer.tags.get("points.mapping") if is_points else None
            key = (sid, raw_mapping)
            if key not in fields_by_source:
                resolved_path = str(resolve_source(project, sid, path))
                if is_points:
                    mapping = json.loads(raw_mapping) if raw_mapping else None
                    fields_by_source[key] = read_csv_points(resolved_path, mapping=mapping)
                elif project.sources[sid].bands:
                    # An imported group / a stack with a band removed: exactly its bands.
                    from dynamix.core.ingest import load_grid_stack
                    from dynamix.shell.opening import _stamp_source

                    src = project.sources[sid]
                    stack = load_grid_stack(resolved_path, src.bands, name=src.label or None)
                    ny, nx = np.asarray(stack.values).shape[:2]
                    fields_by_source[key] = _stamp_source(stack, resolved_path, ny, nx)
                else:
                    fields_by_source[key] = open_field(resolved_path)
            self.add_layer_row(layer, fields_by_source[key])
            if first_id is None:
                first_id = layer.layer_id
        for layer in list(project.layers):
            if layer.parent_id is None and layer.layer_id in self._fields:
                self._ensure_band_layers(layer, self._fields[layer.layer_id])
        # Every restored layer's output rows carry their greying and readings now; landings
        # only sync the active layer's.
        for layer in project.layers:
            self._sync_output_rows(layer)
        self.setWindowTitle(f"{TITLE} — {path.stem}")
        if first_id is not None:
            self.layer_list.select_layer(first_id)
        else:
            # Nothing restorable: build the empty zone anyway so the missing-source notice has a
            # surface to land on -- a silently blank window would hide exactly what went wrong.
            self._set_recipe([], [])
            self._build_strips()
        notices = []
        if skipped:
            notices.append(f"{len(missing)} source(s) missing — {skipped} layer(s) skipped")
        if drifted:
            notices.append(f"{len(drifted)} source(s) changed since save")
        if notices and self.strips is not None:
            self.strips._show_warning("; ".join(notices))
        # The reopened project's own saved links show up immediately, not only after the
        # next Link/Unlink click -- ``first_id is None`` (nothing restorable) still calls this, so
        # a project's stale links from BEFORE this open never linger on screen either.
        self._refresh_topology_panel()
        self._refresh_skeleton_button()
        self._refresh_spectrum_button()
        self._refresh_dh_button()
        self._refresh_anisotropy()
        self._refresh_panel_relevance()
        self._refresh_components_button()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        update_settings(auto_run_wtmm=checked)

    def _on_precision_selected(self, precision: int) -> None:
        """Settings > FFT precision: persisted; applied at the next start."""
        from dynamix.core.fft_policy import precision_note

        saved = update_settings(compute_precision=int(precision))
        note = " (64-bit runs on FFTW3)" if int(precision) == 64 else ""
        warn = precision_note(saved.compute_engine, int(precision))
        self._notify(f"FFT precision set to {int(precision)}-bit{note} — restart DynamiX to "
                     "apply" + (f". {warn}" if warn else ""), "status")

    def _on_engine_selected(self, engine: str) -> None:
        """Settings > Compute engine: persisted; applied at the next start through fft_policy
        (restart to apply). The legacy immediate-apply body below stays
        for the old values (never-delete); the new values return before it."""
        if engine in ("auto", "mlx", "fftw"):
            from dynamix.core.fft_policy import precision_note

            saved = update_settings(compute_engine=engine)
            extra = ""
            if engine == "mlx":
                try:
                    import mlx.core  # noqa: F401
                except ImportError:
                    extra = " (mlx is not installed here — FFTW3 will be used)"
            warn = precision_note(engine, saved.compute_precision)
            if warn:
                extra += f". {warn}"
            self._notify(f"compute engine set to {engine}{extra} — restart DynamiX to apply",
                         "status")
            return
        self._on_engine_selected_legacy(engine)

    def _on_engine_selected_legacy(self, engine: str) -> None:
        """Settings > Compute engine: apply app-wide (``set_default_engine``) and persist.
        Forcing "mlx" on a machine without mlx is honored (the user asked for it) but warned
        about immediately rather than at the next compute's ImportError."""
        from dynamix.core.wtmm_backend import set_default_engine

        set_default_engine(engine)
        update_settings(compute_engine=engine)
        if engine == "mlx":
            try:
                import mlx.core  # noqa: F401
            except ImportError:
                self._notify("mlx is not importable in this environment — the next compute "
                             "will fail; pick Auto or NumPy/FFT", "status")
                return
        self._notify(f"compute engine: {engine}", "status")

    # -- layers ----------------------------------------------------------------------------
    def add_layer_row(self, layer, field) -> None:
        """Put ``layer`` in the panel (grouped under its source, and under its parent layer too if
        it is an ROI) and record the field it resolves against. Does NOT select it -- the caller
        decides, because ``load_field`` and Create want opposite things from the selection
        handler.

        SAME name and signature as before ``layer_panel.LayerPanel`` existed -- ``load_field`` and
        ``_on_roi_create`` call this unchanged; only the row-widget it delegates to changed."""
        self._row_layers.append(layer)
        self._fields[layer.layer_id] = field
        self._layer_by_id[layer.layer_id] = layer
        self.layer_list.add_layer_row(layer, field)

    def _on_layer_selected(self, layer_id: int) -> None:
        layer = self._layer_by_id.get(layer_id)
        if layer is not None and layer.parent_id is None and self._active_roi_id is not None:
            # The dataset row means the WHOLE field: selecting it deactivates the ROI a
            # previous ROI-row selection made active -- it replaces the panel's Deselect.
            self._active_roi_id = None
            if layer is self.layer:
                self._refresh_saved_rois()
        if layer is not None and layer is not self.layer:
            self._select_layer(layer)
            if self._center_stack.currentIndex() == 1:
                # The arrangement's own idea of "the active layer" (which id the shared
                # worker thread should favour over the rest of the queue, see
                # ``_dispatch_next``) just changed -- resync while flipped so its entries and
                # queue reflect the new selection promptly rather than waiting for the next
                # hide-toggle or landing.
                self._sync_arrangement()

    def _select_layer(self, layer) -> None:
        """Make ``layer`` the one the window is showing: its chain, its strips, its field, its
        result.

        The recipe is read back from ``self._recipes`` when this layer has one --
        i.e. it was ever the target of a chain edit -- rather than derived fresh from
        ``layer.chain``, which excludes bypassed steps entirely and has no ``rack`` field: a fresh
        derive would silently drop a bypassed step, and its rack membership, on every switch away
        and back. A layer that was never edited has no recipe yet, and derives from its own
        (fully honest, in that case) chain exactly as before.

        Either way, ``_names``/``_params``/``_bypassed``/``_rack`` become the window's working
        copy of "the chain being edited" for THIS layer -- every knob writes into them and
        ``_chain()`` rebuilds the layer's chain from them, so leaving them pointed at the previous
        layer would make the first knob turn on this one silently overwrite the chain with the
        other layer's recipe.

        The resolve goes through the normal worker path, which means switching back to a layer
        that has already been computed is a cache HIT (the transform's key has not moved), not a
        recompute -- the same property the filter path relies on.
        """
        self.layer = layer
        recipe = self._recipes.get(layer.layer_id)
        if recipe is not None:
            self._names = [d["device"] for d in recipe]
            self._params = [dict(d.get("params", {})) for d in recipe]
            self._bypassed = [bool(d.get("bypassed", False)) for d in recipe]
            self._rack = [d.get("rack") for d in recipe]
        else:
            self._set_recipe([ref.device for ref in layer.chain.steps],
                             [dict(ref.params) for ref in layer.chain.steps])
        # ``_names``/``_params`` are freshly, legitimately populated from THIS layer's own recipe
        # (even if that chain happens to be empty) -- whatever inert ``load_field`` last wiped
        # them for is no longer what they hold, so the next auto-run open must not reseed over a
        # layer switch that happened in between.
        self._inert_open = False
        self.field = self._fields.get(layer.layer_id, self.field)
        self._push_reference_layers()
        self._scales = ()
        self._errored = False
        self._reset_roi_selection()
        # Re-stamp any `backproject` step's target scalars on every switch onto this
        # layer -- the target's field may have changed (or been loaded for the first time) since
        # this point layer was last shown, and the stamps otherwise only refresh on a chain edit
        # (see ``_stamp_backproject``'s own docstring). Skipped while locked/frozen, matching every
        # other write into ``layer.chain`` elsewhere in this file (``_lock_notice``'s guard) --
        # and skipped entirely when there is no `backproject` step to re-stamp, so an ordinary
        # raster-layer switch (the overwhelming majority) pays nothing here at all.
        if (self._is_point_layer(layer) and "backproject" in self._names
                and _lock_notice(layer) is None):
            self._stamp_backproject(self._names, self._params)
            self.layer.chain = self._chain()
            self._snapshot_recipe()
        # A bus with LIVE layer sends: re-fingerprint what it sends, so a referenced layer
        # edited since shows up here (its stamp moves the bus's cache key). Same moment as
        # backproject's re-stamp above, for the same reason.
        if self._names[:1] == ["bus"] and _lock_notice(layer) is None:
            try:
                if self._refresh_bus_stamps(layer):
                    self.layer.chain = self._chain()
                    self._snapshot_recipe()
            except ValueError as exc:
                self._notify(str(exc), "status")
        self.setWindowTitle(f"{TITLE} — {layer.name}")
        self._build_strips()
        # Locked/frozen: the CHAIN still displays (built normally, above) but the whole zone AND
        # the transport go read-only. The layer panel, not either of those, is the unlock surface
        # (its own H/L/F buttons stay live regardless), so disabling every control here traps
        # nothing.
        self._sync_lock_ui(layer)
        self._sync_display_controls(layer)
        # A point layer's field is a PointSet, not a RasterField/bare array -- Canvas.set_field
        # unconditionally does ``np.asarray(field, dtype=float64)`` on whatever it is handed, and
        # a PointSet is not array-like (TypeError). Drawing points on the canvas is a separate step; until then the canvas simply keeps showing whatever it last had (the previous
        # raster, or nothing) -- never a crash for selecting a point layer.
        if not self._is_point_layer(layer):
            self.canvas.set_field(self.field)
            self._apply_raster_visibility()
            self._refresh_saved_rois()
            self._displayed_layer_name = layer.name
            self._holder_raster_ref = None   # raw field is up -- holder display re-syncs on land
        self._update_scale_bar()
        self._start_worker()
        # The ACTIVE LAYER just changed, which is what turns
        # every open inspector's picture from live into kept or back again -- and it changes here,
        # not when a result lands. Waiting for ``resolved`` to re-word them leaves a window
        # claiming to be live across the whole compute, and on ``_start_worker_for``'s
        # ``field is None`` bail-out (:meth:`_start_worker_for`) no result ever lands at all.
        # A no-op while nothing is open.
        self._refresh_inspector_statuses()

    def _is_raw_raster_dataset(self) -> bool:
        """The active layer is a RAW raster dataset: a raster master whose enabled chain holds no
        analyzing transform (field stages such as noise may sit on it). A tool dropped on it
        spawns a child instead of replacing the data. A point catalogue (its own mapping
        transforms, e.g. backproject) and a master that already carries an analyzer (older
        projects) commit edits as before."""
        if self.layer is None or self._is_point_layer(self.layer):
            return False
        # A derivative dataset's vector loader (``vector_source``) is part of the dataset.
        return not any(is_transform(get_device(n))
                       and not getattr(get_device(n), "field_stage", False)
                       and not getattr(get_device(n), "vector_source", False)
                       for n, b in zip(self._names, self._bypassed) if not b)

    def _is_point_layer(self, layer) -> bool:
        """``True`` when ``layer``'s source was imported as a CSV point catalogue (``SourceRef.
        kind == "points"``) rather than a raster -- the discriminator ``_select_layer``
        and (later) the overlay path use to skip raster-only machinery."""
        source = self.project.sources.get(layer.source_id)
        return source is not None and source.kind == "points"

    # -- backproject target stamping ---------------------------
    def _resolve_backproject_target(self, target_name: str):
        """The field of the layer named ``target_name`` among ``self._layer_by_id``'s own values
        -- BY NAME, per the design's contract, never by id. ``None`` for an empty name or one
        that names no CURRENTLY known layer. A name shared by two layers resolves to whichever
        this window happens to have added first (``_layer_by_id``'s insertion order) -- an
        accepted ambiguity, not something this method tries to disambiguate."""
        if not target_name:
            return None
        for candidate in self._layer_by_id.values():
            if candidate.name == target_name:
                return self._fields.get(candidate.layer_id)
        return None

    @staticmethod
    def _backproject_scalars_for(field) -> dict | None:
        """The seven ``backproject`` scalars for a georeferenced target ``field``, or ``None``
        when ``field`` is missing, carries no CRS in its provenance, or has too small an axis to
        derive a per-pixel step from (mirrors ``dynamix.geo.mapping.lonlat_to_pixels``'s own
        ``nx == 1 or ny == 1`` guard -- the same degenerate-axis case that function itself refuses
        rather than dividing by zero)."""
        if field is None or not has_georeference(field):
            return None
        nx, ny = int(field.nx), int(field.ny)
        if nx < 2 or ny < 2:
            return None
        grid = file_pixel_grid(field)
        if grid is not None:
            # A display PICTURE: target its FILE-pixel grid -- the grid
            # the canvas draws in -- never its s x coarser sample grid.
            x0, dx, y0, dy, gnx, gny = grid
            return {"_target_crs": str(field.provenance["crs"]), "_target_x0": x0,
                    "_target_dx": dx, "_target_y0": y0, "_target_dy": dy,
                    "_target_nx": gnx, "_target_ny": gny}
        dx = (float(field.x_axis[-1]) - float(field.x_axis[0])) / (nx - 1)
        dy = (float(field.y_axis[-1]) - float(field.y_axis[0])) / (ny - 1)
        return {"_target_crs": str(field.provenance["crs"]), "_target_x0": float(field.x_axis[0]),
                "_target_dx": dx, "_target_y0": float(field.y_axis[0]), "_target_dy": dy,
                "_target_nx": nx, "_target_ny": ny}

    def _stamp_backproject(self, names: list, params: list) -> None:
        """For every ``backproject`` step in the parallel ``(names, params)`` lists, resolve
        ``params[i]["target"]`` against layer NAMES and stamp the seven ``_target_*`` scalars from
        that target's own field axes + provenance CRS (:meth:`_backproject_scalars_for`).

        An EMPTY target is quietly stamped zero -- nothing to warn about, the device's own honest
        unbound state (:mod:`dynamix.devices.backproject`). A NAMED target that fails to resolve
        (unknown name, or a real layer with no georeference) warns through
        ``self.strips._show_warning`` -- the same refusal-notice surface every other zone warning
        in this file already uses (the ``_show_warning`` precedent, ``main_window.py:637-638`` in the design) -- and ALSO stamps zero: the device no-ops honestly rather than dividing by a
        zero step or fabricating a placement.

        Mutates ``params`` in place; callers decide when the result is worth writing back into
        ``self._params``/``self.layer.chain`` (see the three call sites: ``_on_param_changed``,
        ``_on_chain_edited``, ``_select_layer``'s own re-stamp).

        **Cache-key honesty.** ``cache_key`` (``dynamix.engine.cache.cache_key``) is a content hash
        of ``params`` -- once these seven are written, retargeting a step (or the target's own
        axes changing) invalidates the cached compute automatically. There is no separate
        cache-busting step anywhere in this method, or anywhere else.
        """
        for i, name in enumerate(names):
            if name != "backproject":
                continue
            target_name = str(params[i].get("target", "") or "")
            scalars = self._backproject_scalars_for(
                self._resolve_backproject_target(target_name))
            if scalars is None:
                if target_name and self.strips is not None:
                    self.strips._show_warning(f"backproject: '{target_name}' has no georeference")
                scalars = dict(_ZERO_BACKPROJECT_SCALARS)
            params[i].update(scalars)

    # -- layer panel: hide / lock / freeze / remove / rename --------------------------------
    def _on_hide_toggled(self, layer_id: int, hidden: bool) -> None:
        """``layer.visible`` is the flag; the canvas only reacts when the layer just toggled is
        the ACTIVE one -- v1 scope (``layer_panel.py``'s module docstring), since nothing here
        composites several layers' overlays together. Un-hiding the active layer re-resolves
        (a cache hit, same as any other filter-path redraw) to bring its overlays back."""
        layer = self._layer_by_id.get(layer_id)
        if layer is None:
            return
        layer.visible = not hidden
        if self._center_stack.currentIndex() == 1:
            # ANY layer's visibility is part of the arrangement's own compositing rule
            # ("every layer with visible=True") -- resync regardless of whether the
            # toggled layer is the active one, which the branches below still gate on.
            self._sync_arrangement()
        if layer is not self.layer:
            return
        if hidden:
            self.canvas.clear_overlays()
            self.canvas.clear_points()
            # The hidden overlays include the chain trails a pick
            # would be reading against -- nothing left on screen to pick.
            self.canvas.set_pick_chains(None)
            # A derived raster (h-map / band reconstruction) is a PRODUCT: hiding the layer
            # restores the raw dataset raster and the row's plain name.
            self._sync_holder_raster(self._active_result
                                     if isinstance(self._active_result, dict) else {})
        else:
            self._reresolve()

    def _on_lock_toggled(self, layer_id: int, locked: bool) -> None:
        layer = self._layer_by_id.get(layer_id)
        if layer is None:
            return
        if locked:
            layer.tags["ui.lock"] = "1"
        else:
            layer.tags.pop("ui.lock", None)
        self._sync_lock_ui(layer)

    def _on_output_hide_toggled(self, layer_id: int, name: str, hidden: bool) -> None:
        """An output row's H (``LayerPanel.outputHideToggled``).

        The vector "edges" row toggles the ``ui.edges_hidden`` tag: a drawing choice, so a
        locked layer takes it too. The canvas maxima follow at once on the active layer, the
        Vector/Geo scene on a resync; ``result["extrema"]`` is untouched, so filters, the
        spectrum and anisotropy keep reading them.

        A raster row is a radio over the step's view-only ``show``: un-hiding it shows it,
        hiding the shown one puts the raw field back (``show = "edges"``). The write goes through
        the Show knob's own control on the ACTIVE layer, so a row of another layer selects that
        layer first. The control is the one path that reaches the box, the zone's descriptor
        list and :meth:`_on_param_changed` (a cache-hit re-resolve) together; the next zone
        gesture rebuilds the chain from those descriptors, so a value written past them would
        revert there. A locked layer is refused before the control moves, since the box would
        otherwise hold a value the window never took. The rows are re-synced from the model
        afterwards either way: a refused click snaps back."""
        layer = self._layer_by_id.get(layer_id)
        if layer is None:
            return
        step, outputs = layer_outputs(layer)
        output = next((o for o in outputs if o.name == name), None)
        if output is not None and output.kind == "vector":
            if hidden:
                layer.tags["ui.edges_hidden"] = "1"
            else:
                layer.tags.pop("ui.edges_hidden", None)
            if layer is self.layer:
                self.canvas.set_maxima_visible(not hidden)
            if self._center_stack.currentIndex() == 1:
                self._sync_arrangement()
        elif output is not None and (shown_output(step, outputs) == name) == hidden:
            # Hiding the shown row, or un-hiding a hidden one; anything else is already true.
            if layer is not self.layer:
                self.layer_list.select_layer(layer_id)
            index = self._output_step_index() if layer is self.layer else None
            if index is not None:
                notice = _lock_notice(layer)
                if notice is not None:
                    self.strips._show_warning(notice)
                else:
                    self.strips.strip(index)._on_control_changed(
                        "show", "edges" if hidden else name)
        self._sync_output_rows(layer)

    def _output_step_index(self) -> int | None:
        """The index (into ``_names``) of the active chain's step whose outputs the rows show:
        the last enabled transform, when its device declares outputs; else ``None``."""
        enabled = [i for i, (n, b) in enumerate(zip(self._names, self._bypassed))
                   if not b and is_transform(get_device(n))]
        if not enabled or not declared_outputs(get_device(self._names[enabled[-1]])):
            return None
        return enabled[-1]

    def _sync_output_rows(self, layer) -> None:
        """Push ``layer``'s output state into its rows, non-emitting: the raster output its
        step's ``show`` puts on the canvas, and the ``ui.edges_hidden`` tag. The group is
        rebuilt first when the chain's declared outputs changed (a chain edit). Lazy output
        rows also carry their reading and their greying (:meth:`_output_row_state`). The active
        layer's Reconstruction section follows (:meth:`_sync_recon_section`)."""
        if layer is None:
            return
        self.layer_list.sync_output_rows(layer)
        step, outputs = layer_outputs(layer)
        if step is not None:
            notes, disabled = self._output_row_state(layer, step, outputs)
            self.layer_list.set_output_state(layer.layer_id, shown_output(step, outputs),
                                             layer.tags.get("ui.edges_hidden") == "1",
                                             notes=notes, disabled=disabled)
        if layer is self.layer:
            self._sync_recon_section()

    # -- the Reconstruction section ------------------------------------------------------------
    def _recon_step(self) -> "tuple | None":
        """``(index, device)`` of the active chain's step the Reconstruction section edits: the
        step whose outputs the rows show (:meth:`_output_step_index`), when its device declares
        ``section="reconstruction"`` params; else ``None``."""
        if self.layer is None:
            return None
        index = self._output_step_index()
        if index is None:
            return None
        device = get_device(self._names[index])
        if not any(getattr(p, "section", "") == _RECON_SECTION for p in device.params):
            return None
        return index, device

    def _sync_recon_section(self) -> None:
        """Push the active chain into the Reconstruction section, non-emitting: shown only while
        :meth:`_recon_step` finds a step; its knobs at the step's values (those that do not apply
        hidden); Run enabled when the recon row runs on request (the LastWave engine) and can be
        computed at all; the reading is the recon row's own (:meth:`_output_row_state`), which
        says "manual" while the row waits for Run (:meth:`_awaits_run`)."""
        panel = getattr(self, "_recon_panel", None)
        if panel is None:
            return
        found = self._recon_step()
        self.right_panel.apply_relevance({_RECON_TITLE: found is not None})
        if found is None:
            return
        index, device = found
        values = {**defaults_for(device), **self._params[index]}
        knobs = tuple(p for p in device.params if getattr(p, "section", "") == _RECON_SECTION
                      and p.kind is not ParamKind.TEXT)
        if panel.set_params(knobs, values):
            self.right_panel._protect_from_wheel(panel)
        table = "recon_levels" in values and values.get("algorithm") == "lastwave"
        panel.levels.setVisible(table)
        if table:
            reading = (self._active_result or {}).get("_recon_reading", "")
            panel.levels.set_levels(int(values["n_levels"]), values["recon_levels"],
                                    self._shown_level(), reading)
        step, outputs = layer_outputs(self.layer)
        recon = next((o for o in outputs if o.name == "recon"), None)
        if values.get("algorithm") != "lastwave":
            reason = "the printed algorithm runs on show"
        elif step is None or recon is None:
            reason = "no reconstruction on this chain"
        else:
            reason = self._output_refusal(self.layer, step, recon)
        panel.set_run(reason is None, reason or "compute the reconstruction with these knobs, "
                                                "continuing from the preview")
        notes = self._output_row_state(self.layer, step, outputs)[0] if step is not None else {}
        reading = notes.get("recon", "")
        panel.set_reading(f"manual · {reading}" if reading == _AWAITS_RUN else reading)

    def _on_recon_knob_changed(self, name: str, value) -> None:
        """A Reconstruction-section knob moved: written through the step's own box, the route
        the output rows use, so the chain, the box and the zone's descriptors agree. A locked
        layer refuses it before the box moves, and the section snaps back."""
        found = self._recon_step()
        if found is None or self.strips is None:
            return
        notice = _lock_notice(self.layer)
        if notice is not None:
            self.strips._show_warning(notice)
            self._sync_recon_section()
            return
        if name == "per_level" and value:
            self._snapshot_level_filters(found[0])
        self.strips.strip(found[0])._on_control_changed(name, value)

    def _on_recon_levels_changed(self, text: str) -> None:
        """The level table changed: its whole ``recon_levels`` text, written like a knob."""
        self._on_recon_knob_changed("recon_levels", text)

    # -- per-level filter settings (mz_edges' Per-level filters) --------------------------------
    def _shown_level(self) -> int:
        """The level (1-based) the Scale slider shows."""
        i = self._index_of("scale_select")
        return int(self._params[i].get("scale_idx", 0)) + 1 if i is not None else 1

    def _filter_step_keys(self) -> list:
        """``[(index, key)]`` of the chain's filters a per-level setting is stored under: the
        engine's :func:`~dynamix.engine.resolve.selection_steps` over the enabled steps."""
        index = [i for i, n in enumerate(self._names) if not self._bypassed[i]]
        refs = [DeviceRef(self._names[i], self._params[i]) for i in index]
        keys = {id(ref): key for key, ref in selection_steps(refs)}
        return [(i, keys[id(ref)]) for i, ref in zip(index, refs) if id(ref) in keys]

    def _per_level_step(self) -> "tuple | None":
        """``(index, values)`` of the Reconstruction step while its Per-level filters are on (the
        LastWave engine), else ``None``."""
        found = self._recon_step()
        if found is None:
            return None
        values = {**defaults_for(found[1]), **self._params[found[0]]}
        if values.get("algorithm") != "lastwave" or not values.get("per_level"):
            return None
        return found[0], values

    def _levels_data(self, index: int) -> dict:
        text = {**defaults_for(get_device(self._names[index])),
                **self._params[index]}.get("recon_levels", "")
        try:
            return json.loads(text) if text else {}
        except ValueError:
            return {}

    def _write_levels(self, index: int, data: dict) -> None:
        text = json.dumps(data, sort_keys=True, separators=(",", ":"))
        self.strips.strip(index)._on_control_changed("recon_levels", text)

    def _snapshot_level_filters(self, index: int) -> None:
        """Per-level filters turned on: every level starts from the rack's current settings."""
        J = int({**defaults_for(get_device(self._names[index])),
                 **self._params[index]}.get("n_levels", 0))
        snap = {key: dict(validate_params(get_device(self._names[i]), self._params[i]))
                for i, key in self._filter_step_keys()}
        data = self._levels_data(index)
        data["filters"] = {str(l): {k: dict(v) for k, v in snap.items()} for l in range(1, J + 1)}
        self._write_levels(index, data)

    def _store_level_filter(self, step_index: int) -> None:
        """A filter knob moved with Per-level filters on: the step's settings become the shown
        level's."""
        found = self._per_level_step()
        key = dict(self._filter_step_keys()).get(step_index)
        if found is None or key is None:
            return
        data = self._levels_data(found[0])
        level = data.setdefault("filters", {}).setdefault(str(self._shown_level()), {})
        level[key] = dict(validate_params(get_device(self._names[step_index]),
                                          self._params[step_index]))
        self._write_levels(found[0], data)

    def _load_level_filters(self, level: int) -> None:
        """The Scale slider moved with Per-level filters on: the rack's filter knobs take
        ``level``'s stored settings (a level with none keeps what the knobs show)."""
        found = self._per_level_step()
        if found is None or self.strips is None:
            return
        stored = self._levels_data(found[0]).get("filters", {}).get(str(level), {})
        for i, key in self._filter_step_keys():
            values = stored.get(key)
            if not values:
                continue
            box = self.strips.strip(i)
            names = {p.name for p in get_device(self._names[i]).params}
            for name, value in values.items():
                if name not in names:
                    continue
                self._params[i][name] = value
                box._params[name] = value
                control = box.controls.get(name)
                if control is not None:
                    control.set_value(value)
            box._apply_active_when()

    def _on_recon_run(self) -> None:
        """Run: compute the shown reconstruction row (recon, recon (edges only) or residual; the
        recon row when another row is shown) with the section's knobs and show it. The recon's
        compute continues from the cached preview of the same decay, clipping and coarse (the
        device fetches that preview). A reconstruction already cached for these knobs is shown
        at once; from another row the recon row is put on show first, through the Show knob."""
        layer = self.layer
        if layer is None or self._recon_step() is None:
            return
        step, outputs = layer_outputs(layer)
        shown = shown_output(step, outputs)
        target = shown if shown in _RUN_ROWS else "recon"
        key = self._output_keys(layer).get(target)
        if key is None:
            return
        run = (layer.layer_id, key)
        if key not in self.cache:           # a cached one dispatches nothing, so nothing clears it
            self._recon_run = run
        if shown == target:
            self._reresolve()
            return
        self._on_output_hide_toggled(layer.layer_id, "recon", False)
        step, outputs = layer_outputs(layer)
        if shown_output(step, outputs) != "recon" and self._recon_run == run:  # refused (locked)
            self._recon_run = None

    # -- lazily computed outputs ---------------------------------------------------------------
    def _output_refusal(self, layer, step, output) -> "str | None":
        """Why lazy ``output`` of ``layer`` cannot be computed, or ``None``: an ROI result (the
        region runner does not reconstruct), or an output pinning the 2^J thumbnail (Coarse =
        thumbnail, for the outputs that read Coarse) on a grid 2^J does not divide."""
        if layer.tags.get("roi.window"):
            return "not on ROI results yet"
        if "coarse" in output.params:
            params = validate_params(get_device(step.device), step.params)
            if params.get("coarse") == "thumbnail":
                J = int(params.get("n_levels", 0) or 0)
                values = getattr(self._fields.get(layer.layer_id), "values", None)
                shape = np.shape(values)[:2] if values is not None else ()
                if J and len(shape) == 2 and (shape[0] % 2 ** J or shape[1] % 2 ** J):
                    return f"needs the grid divisible by 2^J (J = {J})"
        return None

    def _output_disabled(self, layer, step, outputs) -> dict:
        """``{name: reason}`` of ``layer``'s greyed-out lazy output rows: those
        :meth:`_output_refusal` refuses, and an output on its own coarser grid while the
        Vector/Geo view is up (it draws in the 2-D view only; the scene drapes the field)."""
        disabled = {}
        for output in outputs:
            if not output.lazy:
                continue
            reason = self._output_refusal(layer, step, output)
            if reason is None and output.grid == "stride" and self._center_view != "raster":
                reason = "shown in the 2-D view"
            if reason is not None:
                disabled[output.name] = reason
        return disabled

    def _output_keys(self, layer) -> dict:
        """``{name: key}`` of ``layer``'s lazy outputs, derived from its chain the way
        :func:`~dynamix.engine.resolve.resolve_output` derives them (the analysis key from
        :meth:`_cache_keys_for`); empty without lazy outputs and for an ROI result. Outputs
        without a row of their own (a preview, :data:`_PREVIEW_OF`) are keyed too."""
        step, _outputs = layer_outputs(layer)
        if step is None or layer.tags.get("roi.window"):
            return {}
        device = get_device(step.device)
        lazy = [o for o in declared_outputs(device) if o.lazy]
        if not lazy:
            return {}
        keys = self._cache_keys_for(layer)
        if not keys:
            return {}
        params = validate_params(device, step.params)
        selection = selection_recipe(layer)
        return {o.name: output_key(device.name, o, params, keys[-1],
                                   selection if o.selects else None) for o in lazy}

    def _output_row_state(self, layer, step, outputs) -> tuple:
        """``(notes, disabled)`` for ``LayerPanel.set_output_state``: "computing…" on a row whose
        job is requested or in flight, a computed output's reading (:func:`_output_note`),
        "failed" after an error, and the greyed-out rows (:meth:`_output_disabled`). Reads key
        membership and the window's own notes only, never a cached value, so it is safe while
        a job holds the worker slot. A row drawing its preview (:meth:`_draws_preview`) reads
        the preview's state, or :data:`_AWAITS_RUN` while it waits for Run
        (:meth:`_awaits_run`)."""
        disabled = self._output_disabled(layer, step, outputs)
        busy = {job[2] for job in (self._out_job, self._out_request) if job is not None}
        notes = {}
        keys = self._output_keys(layer)
        for name, key in keys.items():
            if name in disabled:
                continue
            if self._draws_preview(layer, step, name, key):
                key = keys[_PREVIEW_OF[name]]
                if self._awaits_run(step, key):
                    notes[name] = _AWAITS_RUN
                    continue
            elif self._waits_manual(layer, step, name, key):
                notes[name] = _AWAITS_RUN
                continue
            if key in busy:
                notes[name] = "computing…"
            elif key in self.cache and self._out_notes.get(key):
                notes[name] = self._out_notes[key]
            elif self._out_held == (key, "failed"):
                notes[name] = "failed"
        return notes, disabled

    def _draws_preview(self, layer, step, name, key) -> bool:
        """Whether row ``name`` of ``layer`` draws its preview (:data:`_PREVIEW_OF`) in place of
        its own output ``key``: a step on the LastWave engine that declares the preview, while
        that output is neither cached nor the one Run asked for (``_recon_run``)."""
        preview = _PREVIEW_OF.get(name)
        if preview is None or key is None or step is None:
            return False
        device = get_device(step.device)
        if (validate_params(device, step.params).get("algorithm") != "lastwave"
                or not any(o.name == preview and o.lazy for o in declared_outputs(device))):
            return False
        return key not in self.cache and self._recon_run != (layer.layer_id, key)

    def _awaits_run(self, step, preview_key) -> bool:
        """Whether a row drawing its preview (:meth:`_draws_preview`) waits for Run instead:
        ``step``'s Live (``recon_live``) is off and the preview ``preview_key`` is not cached.
        Such a row draws the raw field, reads :data:`_AWAITS_RUN` and requests nothing; Run
        (``_recon_run``) makes it draw its own output, whose compute runs the preview's pass."""
        params = validate_params(get_device(step.device), step.params)
        return not params.get("recon_live", True) and preview_key not in self.cache

    def _waits_manual(self, layer, step, name, key) -> bool:
        """Whether reconstruction row ``name`` without a preview (recon (edges only), residual;
        :data:`_RUN_ROWS`) waits for Run: Live is off on a LastWave step, the output is not
        cached and Run has not asked for it (``_recon_run``). Such a row draws the raw field,
        reads :data:`_AWAITS_RUN` and requests nothing."""
        if name not in _RUN_ROWS or name in _PREVIEW_OF or key is None or step is None:
            return False
        params = validate_params(get_device(step.device), step.params)
        return (params.get("algorithm") == "lastwave" and not params.get("recon_live", True)
                and key not in self.cache and self._recon_run != (layer.layer_id, key))

    def _shown_lazy(self, layer, renderable) -> "tuple | None":
        """``(output, key)`` of the lazy output ``layer``'s step shows, keyed from
        ``renderable``'s analysis, or ``None``: nothing lazy on show, a refused output
        (:meth:`_output_refusal`), or a renderable whose analysis is another step's. A row
        drawing its preview (:meth:`_draws_preview`) gives the preview's, and nothing while it
        waits for Run (:meth:`_awaits_run`)."""
        if layer is None or renderable.analysis_key is None:
            return None
        step, outputs = layer_outputs(layer)
        name = shown_output(step, outputs)
        output = next((o for o in outputs if o.name == name and o.lazy), None)
        if (output is None or renderable.analysis_device != step.device
                or self._output_refusal(layer, step, output) is not None):
            return None
        selection = selection_recipe(layer)
        key = output_key(renderable.analysis_device, output, renderable.analysis_params,
                         renderable.analysis_key, selection if output.selects else None)
        if self._draws_preview(layer, step, name, key):
            output = next(o for o in declared_outputs(get_device(step.device))
                          if o.name == _PREVIEW_OF[name])
            key = output_key(renderable.analysis_device, output, renderable.analysis_params,
                             renderable.analysis_key, selection if output.selects else None)
            if self._awaits_run(step, key):
                return None
        elif self._waits_manual(layer, step, name, key):
            return None
        return output, key

    def _displayed_output_key(self, layer) -> "str | None":
        """The key of the output ``layer``'s shown row draws (its own, or its preview's; none
        while it waits for Run, :meth:`_awaits_run`), derived from the chain as
        :meth:`_output_keys` derives it."""
        step, outputs = layer_outputs(layer)
        name = shown_output(step, outputs)
        keys = self._output_keys(layer)
        key = keys.get(name)
        if self._draws_preview(layer, step, name, key):
            preview_key = keys[_PREVIEW_OF[name]]
            return None if self._awaits_run(step, preview_key) else preview_key
        if self._waits_manual(layer, step, name, key):
            return None
        return key

    def _show_lazy_output(self, renderable, result: dict) -> dict:
        """The active result as the canvas shows it. When the step's Show names a LAZY output,
        a cache hit becomes ``raster_out`` of a COPY -- the cached result is never mutated; an
        output on its own grid also carries its stride (:meth:`_strided_display_field`) -- and
        the output's label names the row. A miss leaves the result as it is (the raw field
        stays up) and requests the job, dispatched once the worker slot is free
        (:meth:`_dispatch_output`). A key the user stopped, or that failed, is not requested
        again while that output stays on show; a hidden layer, or one whose transforms await
        Run, requests nothing."""
        self._drop_output_request()
        layer = self.layer
        shown = self._shown_lazy(layer, renderable)
        if self._out_held is not None and (shown is None or shown[1] != self._out_held[0]):
            self._out_held = None
        if shown is None:
            return result
        output, key = shown
        # One read: an output job on the worker thread may evict between a probe and a get.
        value = self.cache.get(key)
        if value is not None:
            self._remember_output(layer, key, value)
            out = {**result, "raster_out": value["raster"],
                   "_view_note": output.label or output.name}
            if output.grid == "stride":
                out["_display_stride"] = int(value["display_stride"])
                out["_full_dims"] = tuple(value["full_dims"])
            if value.get("display_offset") is not None:
                # Where the output's samples register; the result's own ``_display_offset``
                # stays the maxima's.
                out["_raster_display_offset"] = tuple(value["display_offset"])
            return out
        if (self._out_held is None and layer.visible
                and layer.layer_id not in self._pending_layers
                and (self._out_job is None or self._out_job[2] != key)):
            self._out_request = (layer, output.name, key)
            QtCore.QTimer.singleShot(0, self._dispatch_output)
        return result

    def _cached_output_raster(self, layer, renderable):
        """The cached raster of the lazy output ``layer`` shows, when it lies on the field's own
        grid (the Vector/Geo drape); ``None`` otherwise. Never computes anything."""
        value = self._cached_output_value(layer, renderable)
        return None if value is None else value["raster"]

    def _cached_output_value(self, layer, renderable):
        """The cached value (``raster``, ``diag``, and ``display_offset`` where its samples
        register off their pixel index) of the lazy output ``layer`` shows, when it lies on the
        field's own grid; ``None`` otherwise. One cache read. Never computes anything."""
        shown = self._shown_lazy(layer, renderable)
        if shown is None or shown[0].grid != "native":
            return None
        return self.cache.get(shown[1])

    def _remember_output(self, layer, key: str, value) -> None:
        """Record a computed output of ``layer``: its row reading, and its key among the layer's
        output keys, pinned at once while the layer is pinned (F)."""
        self._out_notes[key] = _output_note(value)
        self._out_keys.setdefault(layer.layer_id, set()).add(key)
        if _is_frozen(layer):
            self.cache.pin(key)

    def _drop_output_request(self) -> None:
        """Forget the output job waiting for the worker slot. The rows of its layer, while that
        layer still exists, are pushed again so none keeps reading "computing…" for a job that
        will not run."""
        request, self._out_request = self._out_request, None
        if request is not None and self._layer_by_id.get(request[0].layer_id) is request[0]:
            self._sync_output_rows(request[0])

    def _dispatch_output(self) -> None:
        """Start the job :meth:`_show_lazy_output` requested, once the worker slot is free. An
        analysis always goes first: a busy slot leaves the request for
        :meth:`_land_after_worker`. A request for another key than the output job in flight
        supersedes that job: it is cancelled, and its landing dispatches this one. The worker
        gets a snapshot of the layer, so the key cannot move under it; a request whose key no
        longer matches the layer's chain is dropped (the next landing asks again)."""
        request = self._out_request
        if request is None or self._closing or self._shutting_down:
            return
        if self._thread is not None:
            if (self._out_job is not None and self._out_job[2] != request[2]
                    and self._worker is not None):
                self._worker.cancel()
            return
        layer, name, key = request
        if (layer is not self.layer or self.field is None
                or self._output_keys(layer).get(name) != key):
            self._drop_output_request()
            return
        self._out_request = None
        snapshot = dataclasses.replace(layer, tags=dict(layer.tags))
        worker = OutputWorker(snapshot, self.field, self.cache, layer.source_id, name)
        # BOUND METHODS ONLY (worker.py's documented trap), as for the analysis worker.
        worker.progress.connect(self._on_output_progress)
        worker.finished.connect(self._on_output_finished)
        worker.error.connect(self._on_output_error)
        worker.cancelled.connect(self._on_output_cancelled)
        self._worker = worker
        self._out_job = (layer, name, key)
        self._out_t0 = time.perf_counter()
        self._thread = worker.start()
        self._stop_btn.setVisible(True)
        self._sync_output_rows(layer)

    def _on_output_progress(self, stage: str, frac: float) -> None:
        """An output job's progress, on the strip reading the analysis progress uses."""
        self._stage = stage
        if self._out_job is not None and self._out_job[0] is self.layer:
            self._set_compute_reading(f"{stage} {frac:.0%}")
        if self._closing:
            self._show_waiting_title()

    def _end_output_job(self) -> tuple:
        """Take the landed output job off the worker slot; returns its ``(layer, name, key)``.
        A Stop that arrived as the job finished is consumed here, never left for the next
        analysis's cancel to misread. The job Run asked for is done with however it ended:
        landed, its row draws it from the cache; stopped, superseded or failed, the preview."""
        job, self._out_job = self._out_job, None
        self._teardown_thread()
        self._user_stopped = False
        if job is not None and self._recon_run == (job[0].layer_id, job[2]):
            self._recon_run = None
        return job

    def _on_output_finished(self, value) -> None:
        job = self._end_output_job()
        layer = job[0]
        if self._layer_by_id.get(layer.layer_id) is layer:
            self._remember_output(layer, job[2], value)
            if layer is self.layer:
                ms = (time.perf_counter() - self._out_t0) * 1000.0
                self._set_compute_reading(f"{job[1]} {ms:.0f} ms")
        self._land_output(job)

    def _on_output_cancelled(self) -> None:
        """A cancelled output job cached nothing. A USER stop holds its key (no re-dispatch
        while that output stays on show; the row reads nothing); a superseded job's landing
        dispatches whatever superseded it. The analysis is never marked stale by the job
        itself. A user stop also outranks an analysis of the active layer queued behind the job
        (an analysis knob turned while it ran, which is what cancelled it): that analysis
        stands down exactly as :meth:`_on_cancelled` stands one down, and nothing redraws,
        since the live chain's analysis is not cached."""
        stopped = self._user_stopped
        job = self._end_output_job()
        if stopped:
            self._out_held = (job[2], "stopped")
            if job[0] is self.layer:
                if self._active_pending == "compute":
                    self._active_pending = None
                    self._stopped_sig = self._transform_signature()
                    self._sync_lock_ui(self.layer)
                    self._set_transform_states("idle")
                    self._set_compute_reading("stopped")
                    self._mark_stopped_pending()
                    self._sync_output_rows(self.layer)
                    self._land_after_worker()
                    return
                self._set_compute_reading("stopped")
        self._land_output(job)

    def _on_output_error(self, message: str) -> None:
        job = self._end_output_job()
        self._out_held = (job[2], "failed")
        if job[0] is self.layer:
            self._set_compute_reading(f"{job[1]} failed")
            self._notify(f"{job[1]}: {message}", "status")
        self._land_output(job)

    def _land_output(self, job) -> None:
        """After an output job: the display path again for its layer while it is still the
        active one (a cache hit now, or the next request), unless an analysis is pending,
        whose own landing redraws; its rows otherwise. Then whatever the slot does next. A
        layer removed, or replaced by a project reopen, since the dispatch is left alone."""
        layer = job[0]
        if self._layer_by_id.get(layer.layer_id) is layer:
            if layer is self.layer and self._active_pending != "compute":
                if self._active_pending == "resolve":
                    self._active_pending = None          # the redraw below is that resolve
                self._reresolve()
            else:
                self._sync_output_rows(layer)
        self._land_after_worker()

    def _on_freeze_toggled(self, layer_id: int, frozen: bool) -> None:
        """Freeze pins every one of ``layer``'s OWN transform-step cache keys (:meth:`
        _cache_keys_for`, the SAME derivation ``resolve()`` uses) so an eviction policy can never
        drop what freeze promised to hold; unfreeze unpins the same keys. The entries themselves
        are untouched either way -- pin/unpin only change whether they are ALLOWED to be dropped,
        never whether they currently exist. The lazy outputs computed for the layer
        (``_out_keys``) are pinned and unpinned with them."""
        layer = self._layer_by_id.get(layer_id)
        if layer is None:
            return
        keys = self._cache_keys_for(layer) + sorted(self._out_keys.get(layer_id, ()))
        if frozen:
            layer.tags["ui.freeze"] = "1"
            for key in keys:
                self.cache.pin(key)
        else:
            layer.tags.pop("ui.freeze", None)
            for key in keys:
                self.cache.unpin(key)
        self._sync_lock_ui(layer)

    def _sync_lock_ui(self, layer) -> None:
        """Live (re-)application of the lock/freeze gate: the zone AND the transport both go
        read-only for a locked/frozen ACTIVE layer, never for one that merely carries the flag
        while some OTHER layer is on screen.

        Called from three places: ``_select_layer`` (switching onto a layer applies whatever it
        already carries), ``_on_lock_toggled``/``_on_freeze_toggled`` (toggling H/L/F on the layer
        CURRENTLY on screen must not wait for a re-select to take visible effect), and
        ``_on_finished``/``_on_error`` (a compute landing must not blindly re-enable the transport
        out from under a lock that took effect -- or was still in effect the whole time -- while
        that compute was running; see those methods' own comments).

        The zone alone is not enough: ``_on_scale_changed`` is a SECOND, independent write path
        into ``_params``/``layer.chain`` (the transport's scrub/playback), reachable via keyboard
        (Space) and mouse even while the zone is disabled -- it carries its own ``_lock_notice``
        guard for that reason, but the transport must also visibly refuse the gesture rather than
        silently no-op it.
        """
        if layer is not self.layer:
            return
        enabled = not (_is_locked(layer) or _is_frozen(layer))
        if self.strips is not None:
            self.strips.setEnabled(enabled)
        self.transport.setEnabled(enabled)

    def _cache_keys_for(self, layer) -> list[str]:
        """Every transform step's cache key for ``layer``'s OWN chain, threaded through
        ``upstream`` exactly as :func:`dynamix.engine.resolve.resolve` derives them -- reused via
        the imported :func:`~dynamix.engine.cache_key` AND :func:`~dynamix.engine.source_identity`,
        never re-typed, so freeze can never pin a DIFFERENT key than the one a resolve would
        actually look up (a duplicated format string -- or a duplicated ``source_id`` derivation -- is exactly the kind of drift ``cache.py``'s own module
        docstring warns about). Filters are skipped -- they are never cached, and ``resolve`` never
        advances ``upstream`` for one either, so skipping them here reproduces its threading
        precisely for a chain that is (per the engine's own transforms-then-filters invariant) all
        transforms first."""
        sid = source_identity(layer)
        keys: list[str] = []
        upstream: str | None = None
        for ref in layer.chain.steps:
            device = get_device(ref.device)
            if not is_transform(device):
                continue
            params = validate_params(device, ref.params)
            key = cache_key(device.name, sid, keyed_params(device, params), upstream=upstream)
            keys.append(key)
            upstream = key
        return keys

    def _arr_tail_key(self, layer) -> str | None:
        """The LAST entry of :meth:`_cache_keys_for` -- whether ``layer``'s own transform tail
        is (or isn't) sitting in the cache is exactly the "is this layer already resolved"
        question the arrangement's multi-layer resolve (:meth:`_sync_arrangement`) and its error
        bookkeeping (:attr:`_arr_errors`) both ask, repeatedly. ``None`` for a chain with no
        transforms at all -- trivially always resolved, nothing to probe."""
        keys = self._cache_keys_for(layer)
        return keys[-1] if keys else None

    def _on_remove_requested(self, layer_id: int, *, confirm: bool = True) -> None:
        """Locked refuses outright (a notice in the zone's reading label, nothing removed). A
        layer with children needs ONE confirmation before the cascade -- ``Project.remove_layer``
        will happily take out every descendant in one call, and that is exactly the action a
        single click must not be allowed to do silently. Cache entries are never touched: freeze's
        pins (if any) simply become pins on keys nothing will ever resolve again -- inert, not
        wrong, and cheap to leave rather than thread a cascading unpin through this path too.

        Bookkeeping is pruned BEFORE ``layer_list.remove_rows`` runs, not after:
        ``remove_rows`` blocks the panel's own signals for the removal itself, but pruning first
        too means even an UNBLOCKED stray ``layerSelected`` (any future call site that forgets to
        route through here) would resolve a removed id to ``None`` via ``_layer_by_id`` rather than
        to a half-dead ``Layer`` object. The actual post-removal selection is made exactly once,
        explicitly, through :meth:`layer_panel.LayerPanel.select_layer` below -- never left to
        whatever Qt's own removal happened to promote.
        """
        layer = self._layer_by_id.get(layer_id)
        if layer is None:
            return
        if _is_locked(layer):
            if self.strips is not None:
                self.strips._show_warning("locked — unlock in the layer panel to remove")
            return
        has_children = any(l.parent_id == layer_id for l in self.project.layers)
        if has_children and confirm:
            reply = QtWidgets.QMessageBox.question(
                self, "Remove layer",
                f"Remove {layer.name!r} and its grouped layers?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        removed_ids = self.project.remove_layer(layer_id, cascade=True)
        removed = set(removed_ids)
        active_removed = self.layer is not None and self.layer.layer_id in removed
        self._row_layers = [l for l in self._row_layers if l.layer_id not in removed]
        for rid in removed_ids:
            self._layer_by_id.pop(rid, None)
            self._derived_fields.pop(rid, None)
            self._fields.pop(rid, None)
            self._recipes.pop(rid, None)
            self._arr_errors.pop(rid, None)   # No dead entry for a layer_id gone for good
            self._out_keys.pop(rid, None)
        self.layer_list.remove_rows(removed_ids)
        # The 3-D view learns about removals too (the vector view kept
        # framing a REMOVED dataset's extent): resync drops the stale actors. The camera
        # deliberately stays -- removing one layer must not yank the view (the
        # camera-observer design).
        if self._arrangement is not None:
            self._sync_arrangement()
        if active_removed and self.project.layers:
            self.layer_list.select_layer(self.project.layers[0].layer_id)
        elif active_removed:
            self._reset_to_empty()

    def _on_remove_layer_only_requested(self, layer_id: int) -> None:
        """"Delete layer": remove this ONE layer; its children move up to its
        parent (rows re-nested there), then the ordinary removal runs with nothing left to
        cascade. A dataset's own layer has no parent to hand its children to -- that is the
        dataset row's "Remove dataset"."""
        layer = self._layer_by_id.get(layer_id)
        if layer is None:
            return
        if layer.parent_id is None:
            self._notify("a dataset's results need it — remove the dataset instead", "status")
            return
        if _is_locked(layer):
            if self.strips is not None:
                self.strips._show_warning("locked — unlock in the layer panel to remove")
            return
        children = [l for l in self.project.layers if l.parent_id == layer_id]
        subtree, stack = [], list(reversed(children))
        while stack:
            node = stack.pop()
            subtree.append(node)
            stack.extend(reversed([l for l in self.project.layers
                                   if l.parent_id == node.layer_id]))
        for child in children:
            child.parent_id = layer.parent_id
        self.layer_list.remove_rows([l.layer_id for l in subtree])
        for node in subtree:                         # parents before their own children
            self.layer_list.add_layer_row(node, self._fields.get(node.layer_id, self.field))
        self._on_remove_requested(layer_id)

    def _on_remove_roi_requested(self, roi_id: str) -> bool:
        """"Delete ROI": refused while any result was computed on it -- the notice names them; otherwise the record and its row go. Returns
        whether it was deleted."""
        roi = next((r for r in self.project.rois if r.roi_id == roi_id), None)
        if roi is None:
            return False
        users = [l for l in self.project.layers if l.roi_id == roi_id]
        if users:
            self._notify(f"ROI {roi.label} still has results ({', '.join(l.name for l in users)})"
                         " — delete them first", "status")
            return False
        self.project.rois.remove(roi)
        if self._active_roi_id == roi_id:
            self._active_roi_id = None
        self._refresh_saved_rois()
        if self.layer is not None and self.layer.parent_id is None \
                and self.layer_list.currentItem() is None:
            self.layer_list.select_layer(self.layer.layer_id)
        return True

    def _on_remove_many_requested(self, layer_ids, roi_ids, source_ids) -> None:
        """Delete with several rows selected: ONE confirmation for all of
        them, then datasets, then layers (with their children), then ROIs -- so an ROI selected
        together with its results goes too. Refusals (a locked layer, an ROI whose results
        were not selected) are listed once."""
        n = len(layer_ids) + len(roi_ids) + len(source_ids)
        if n == 0:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Delete", f"Delete the {n} selected row(s) and everything under them?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        for source_id in source_ids:
            self._on_remove_source_requested(source_id, confirm=False)
        for layer_id in layer_ids:
            if layer_id in self._layer_by_id:
                self._on_remove_requested(layer_id, confirm=False)
        refused = [r.label for r in list(self.project.rois)       # a snapshot: rois shrinks
                   if r.roi_id in roi_ids and not self._on_remove_roi_requested(r.roi_id)]
        if refused:
            self._notify(f"kept ROI {', '.join(refused)} — results still use it", "status")

    def _on_remove_source_requested(self, source_id: str, *, confirm: bool = True) -> None:
        """Header-row "Remove dataset": the whole family under ONE
        confirmation -- every root layer of the source cascaded through
        ``Project.remove_layer`` with the same bookkeeping prune as
        :meth:`_on_remove_requested`, then the header row itself. The ``Source`` entry
        stays in ``project.sources``: the model offers no remove_source (and the model is
        not hot-reloadable mid-session); an orphaned SourceRef is inert -- ``add_source``
        is idempotent by path, and a layer-less source draws no header on reopen. A locked
        layer anywhere in the family refuses the whole dataset, matching the per-layer
        rule."""
        family = [l for l in self.project.layers if l.source_id == source_id]
        if not family:
            self.layer_list.remove_source_row(source_id)
            return
        if any(_is_locked(l) for l in family):
            if self.strips is not None:
                self.strips._show_warning("locked layer in dataset — unlock to remove")
            return
        src = self.project.sources.get(source_id)
        label = (src.label or Path(src.path).name) if src is not None else source_id
        reply = QtWidgets.QMessageBox.Yes if not confirm else QtWidgets.QMessageBox.question(
            self, "Remove dataset",
            f"Remove {label!r} and its {len(family)} layer(s)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        removed_ids = []
        for root in [l for l in family if l.parent_id is None]:
            removed_ids.extend(self.project.remove_layer(root.layer_id, cascade=True))
        removed = set(removed_ids)
        active_removed = self.layer is not None and self.layer.layer_id in removed
        self._row_layers = [l for l in self._row_layers if l.layer_id not in removed]
        for rid in removed_ids:
            self._layer_by_id.pop(rid, None)
            self._derived_fields.pop(rid, None)
            self._fields.pop(rid, None)
            self._recipes.pop(rid, None)
            self._arr_errors.pop(rid, None)
            self._out_keys.pop(rid, None)
        self.layer_list.remove_rows(removed_ids)
        self.layer_list.remove_source_row(source_id)
        # Removing a whole DATASET changes the subject: resync the 3-D view and refit its
        # camera to what remains (the load_field-autoRange analog for the vector view). Without
        # the refit the camera stays on the removed dataset's extent, far zoomed out and hard to
        # control, since orbit speeds scale with camera distance.
        if self._arrangement is not None:
            self._sync_arrangement()
            self._arrangement.reset_camera()
        if active_removed and self.project.layers:
            self.layer_list.select_layer(self.project.layers[0].layer_id)
        elif active_removed:
            self._reset_to_empty()

    def _reset_to_empty(self) -> None:
        """Removing the LAST layer leaves nothing for ``self.layer``/``self.field`` to point at.
        Without this, ``self.layer`` kept pointing at the ``Layer`` object ``Project.remove_layer``
        had just dropped -- a zombie active layer every knob kept editing (into an object nothing
        will ever read again) and every save would have silently discarded. The window instead
        goes back to exactly the state :meth:`__init__` leaves it in before any ``load_field`` ever
        runs: no layer, no field, an empty disabled workflow zone, a bare canvas, the plain title,
        and a disabled transport (there is no chain left to scrub).
        """
        self.layer = None
        self.field = None
        self._active_result = None
        self._refresh_skeleton_button()      # no active result left -- back to disabled
        self._refresh_spectrum_button()
        self._refresh_dh_button()
        self._refresh_anisotropy()
        self._refresh_panel_relevance()
        self._refresh_components_button()
        self._set_recipe([], [])
        self._build_strips()
        self.strips.setEnabled(False)
        self.canvas.clear_field()
        self._reset_roi_selection()
        self.setWindowTitle(TITLE)
        self.transport.setEnabled(False)

    def _on_rename_requested(self, layer_id: int, name: str) -> None:
        layer = self._layer_by_id.get(layer_id)
        if layer is None:
            return
        if not name:
            # A blanked-out edit (every character deleted, then Enter) names nothing -- revert the
            # row to what ``layer.name`` still is, rather than leaving the panel showing an empty
            # row for a layer that was never actually renamed.
            self.layer_list.set_layer_name(layer_id, layer.name)
            return
        layer.name = name
        self.layer_list.set_layer_name(layer_id, name)
        if layer is self.layer:
            self.setWindowTitle(f"{TITLE} — {layer.name}")

    # -- the floating inspectors -----------------------------------------
    def _field_for_source(self, source_id: str):
        """The field an inspector for ``source_id`` should show: the first layer over that source
        in ``self._fields``, or ``None`` when the project has a source but this window has no
        field for it yet (a project just opened whose rasters are not all loaded).

        First, not "the active one": every layer over one source is a view of the SAME raster
        (that is what a source IS -- ``Project.add_source``'s registry), so any of them names the
        same picture, and the earliest is the one whose row the header was built from.

        RASTER sources only: a point catalogue's field is a ``PointSet``, which the caller's own
        guard refuses before ever reaching here (:data:`INSPECTOR_NO_RASTER_TEXT`). A future
        second caller must carry that guard too, or hand ``Canvas.set_field`` a ``PointSet``.
        """
        for layer_id, field in self._fields.items():
            layer = self._layer_by_id.get(layer_id)
            if layer is not None and layer.source_id == source_id:
                return field
        return None

    def _inspector_is_live(self, source_id: str) -> bool:
        """Whether what ``source_id``'s inspector is showing came from the layer the user is
        driving right now. Derived on every call, never cached on the window: the SAME result
        becomes a KEPT one the moment the user selects a layer over another source, without
        anything new arriving for this one -- a stored flag would go quietly stale into the exact
        lie sec 10 A2 exists to prevent."""
        renderable = self._last_renderables.get(source_id)
        if renderable is None:
            return False
        return self._layer_by_id.get(renderable.layer_id) is self.layer

    def _inspector_status(self, source_id: str) -> str:
        """What ``source_id``'s inspector should be saying right now -- the ONE place all three
        readings are chosen (see :data:`INSPECTOR_LIVE_TEXT`)."""
        renderable = self._last_renderables.get(source_id)
        if renderable is None:
            return NO_RESULT_TEXT
        if self._inspector_is_live(source_id):
            return INSPECTOR_LIVE_TEXT
        layer = self._layer_by_id.get(renderable.layer_id)
        name = layer.name if layer is not None else source_id
        return INSPECTOR_LAST_RESULT_TEXT.format(name=name)

    def _refresh_inspector_statuses(self) -> None:
        """Re-word every open inspector, not just the one a result just reached: what makes a
        picture "last" is a change of ACTIVE LAYER somewhere else entirely."""
        for source_id, inspector in self._inspectors.items():
            inspector.set_status(self._inspector_status(source_id),
                                 live=self._inspector_is_live(source_id))

    # -- what an inspector remembers between sessions --------------------
    def _capture_inspector_state(self, source_id: str,
                                 inspector: InspectorWindow) -> InspectorState:
        """One OPEN window, as it should come back next time.

        Every field is read off the live widgets at WRITE time, never accumulated from the
        signals that announced the changes. That matters most for ``scale_idx``:
        ``InspectorWindow.set_n_scales`` deliberately swallows the transport's own clamp echo (a
        slider at 4 meeting a 3-scale result lands on 2 and announces nothing), so a value
        remembered from the last ``scaleChanged`` would go stale on any shortening recompute --
        while the slider itself cannot.
        """
        rect = inspector.geometry()
        return InspectorState(
            source_id=source_id,
            open=True,
            geometry=(rect.x(), rect.y(), rect.width(), rect.height()),
            scale_idx=int(inspector.transport.slider.value()),
            follow_master=inspector.follow_master,
            sublayers={key: SubLayerState(visible=visible, opacity=opacity)
                       for key, (visible, opacity) in inspector.sublayer_states().items()})

    def _persist_inspectors(self) -> None:
        """Write every inspector's state into ``Settings.view_options["inspectors"]``.

        Through ``update_settings`` (``settings.py``), the read-modify-write that exists to make
        a single-field write safe: ``view_options`` holds the arrangement's own view options too,
        and a bare ``save_settings`` here would reset them. ``settings.py``'s schema is NOT
        touched -- ``view_options`` is already a free-form JSON dict, and
        ``model/inspector_state.py`` owns the shape that goes in it (this is a
        preference, so it is a settings key, not a project-document one).

        A no-op while the restore is running: see ``self._restoring_inspectors``.
        """
        if self._restoring_inspectors:
            return
        for source_id, inspector in self._inspectors.items():
            self._inspector_states[source_id] = self._capture_inspector_state(source_id,
                                                                              inspector)
        update_settings(view_options={**load_settings().view_options,
                                      "inspectors": inspectors_to_payload(
                                          self._inspector_states)})

    def _on_inspector_state_changed(self, *_args) -> None:
        """Any inspector announced a change worth remembering -- geometry, follow-master, a
        sub-layer's visibility or opacity.

        One slot for four signals of different shapes (hence ``*_args``): what each carries is
        already in the window, and :meth:`_capture_inspector_state` reads it from there. The
        signals differ; what to do about them does not.

        NOT connected to a follower's ADOPTED scale index (``_drive_follower_inspectors`` ->
        ``set_scale_index``, which is silent by design): a follower's index is the master's, so
        persisting it on every scrub step would be a settings write per playback tick for a
        number that is re-derived from the master the moment it is restored. An inspector's OWN
        scale gesture is persisted, in :meth:`_on_inspector_scale_changed`.
        """
        self._persist_inspectors()

    def _restore_inspectors(self) -> None:
        """Put back the inspectors this machine last had open (the one-shot, from
        ``load_field``'s tail).

        **Honest drops.** An entry naming a ``source_id`` this project does not have is
        skipped silently -- settings are per machine while sources are per project, so a
        remembered window belonging to some other project is a preference that does not apply
        here, not a fault to report. So is an entry whose window ``_on_inspector_toggled``
        refuses (a point catalogue has no raster to inspect); that path has already said so and
        pushed its own button back down.

        The whole loop runs under ``self._restoring_inspectors`` so that none of it writes back:
        opening the window, adopting its scale and seeding its rows all announce themselves, and
        the first of those writes would persist a freshly-built window's defaults over the state
        still being restored.

        **A remembered geometry is replayed only where a screen can show it.** Settings are per
        MACHINE and a machine's screens are not: a window saved on a desk monitor restores
        off-screen on the laptop alone, and the layer list's "I" button would then read "open"
        over a window nobody can see -- the same two-opinions-about-one-state failure sec 5b
        exists to prevent, arrived at from the other side. When no screen contains the saved
        rectangle's centre the geometry is simply not applied and the platform places the window,
        which is what a window that has never been moved gets anyway.

        **NAMED LIMITATION -- source ids are PROJECT-LOCAL.** ``src0`` is the first source of
        EVERY project and sub-layer keys are ``"0:0"``, ``"0:1"``..., while this state lives in
        the per-machine settings file. So an entry written over raster A is matched entry-for-
        entry when raster B is opened later, and B's inspector comes up at A's geometry, A's scale
        index, with A's product dimmed -- the "unknown source is dropped" rule above never fires,
        because the id is never unknown. That is the plan's own resolved key (keyed by
        ``source_id``), so it is a limitation of this slice rather than a defect in it; the fix is
        a source-IDENTITY key (path or content hash -- ``engine.source_identity`` already exists)
        instead of the per-project counter, and that is a design decision, not this method's.
        """
        saved = inspectors_from_payload(load_settings().view_options.get("inspectors"))
        self._inspector_states = saved
        self._restoring_inspectors = True
        try:
            for source_id, state in saved.items():
                if not state.open or source_id not in self.project.sources:
                    continue
                self._on_inspector_toggled(source_id, True)
                inspector = self._inspectors.get(source_id)
                if inspector is None:
                    continue
                self.layer_list.set_inspector_open(source_id, True)
                rect = (None if state.geometry is None else QtCore.QRect(*state.geometry))
                if rect is not None and \
                        QtGui.QGuiApplication.screenAt(rect.center()) is not None:
                    inspector.setGeometry(rect)
                inspector.follow_button.setChecked(state.follow_master)
                # Held, not clamped away: no result has landed yet, so the transport still knows
                # one scale (``InspectorWindow.restore_scale_index``'s own docstring).
                inspector.restore_scale_index(state.scale_idx)
                for key, sub in state.sublayers.items():
                    inspector.set_sublayer_state(key, visible=sub.visible, opacity=sub.opacity)
        finally:
            self._restoring_inspectors = False

    def _on_source_hide_toggled(self, source_id: str, hidden: bool) -> None:
        """The source header's H: hide the DATASET -- the canvas image when the
        active layer is over this source, and every raster drape of its layers in the world --
        while the layers' products (extrema, chains) stay exactly as they are. ``SourceRef.hidden``
        is the flag; persisted with the project."""
        source = self.project.sources.get(source_id)
        if source is None:
            return
        source.hidden = bool(hidden)
        self._apply_raster_visibility()
        if self._center_stack.currentIndex() == 1:
            self._sync_arrangement()

    def _resolve_surface_source(self, text) -> "object | None":
        """A ``ui.surface_source`` tag value -> the field it names, or None. Tags are strings
        while ``_fields`` keys are int layer_ids -- an unconverted str key silently misses every
        "other dataset" lookup; ``derived:<id>`` names a derived dataset."""
        s = str(text)
        if s.startswith("derived:"):
            try:
                return self._derived_fields.get(int(s[len("derived:"):]))
            except ValueError:
                return None
        try:
            return self._fields.get(int(s))
        except (TypeError, ValueError):
            return self._fields.get(s)

    def _apply_raster_visibility(self, showing_product: "bool | None" = None) -> None:
        """Canvas image follows the ACTIVE layer's source ``hidden`` flag -- except a child
        dataset: a layer carrying ``roi.window`` shows its OWN crop, which is a product of
        the ROI gesture, not "the dataset" the header's H names.

        And except a result's OWN raster (tucker's reconstruction, a filtered field, h(x), a
        band reconstruction -- ``showing_product``): the dataset's H hides the DATASET, never a
        result computed from it; that raster follows the result's own H (hiding a result
        restores the raw raster, :meth:`_sync_holder_raster`). ``None`` = whatever the canvas
        is showing now (``_holder_raster_ref``); callers that just swapped the image say which."""
        if showing_product is None:
            showing_product = self._holder_raster_ref is not None
        source = self.project.sources.get(self.layer.source_id) if self.layer is not None else None
        exempt = self.layer is not None and bool(self.layer.tags.get("roi.window"))
        self.canvas.image_item.setVisible(showing_product or exempt or
                                          not (source is not None and source.hidden))

    def _on_inspector_toggled(self, source_id: str, checked: bool) -> None:
        """The layer list's per-source "I" button (a ``source_id``, not a ``layer_id``).

        Opening pushes what this window already knows -- the source's raster, and its last
        Renderable if one ever landed -- and dispatches NOTHING: an inspector is a second view of
        a result that has already been computed, so opening one must never start a worker or add
        a cache entry (asserted in ``tests/test_shell_window.py``).

        Opening over a source that is not the active layer's also states that once on the status
        bar, on top of the window's own persistent strip: the picture in a brand-new window is a
        kept one, and the user has no way to tell that from a live one by looking at it.

        A POINT-CATALOGUE source is refused here, before any window exists (see
        :data:`INSPECTOR_NO_RASTER_TEXT` for why it has nothing to show). Refusing BEFORE
        constructing is what keeps sec 5b's one-state rule in the OPENING direction: a window
        registered but never shown would leave the layer list's "I" reading "open" over nothing,
        so the button is pushed back down in the same breath as the message.
        """
        if not checked:
            self._close_inspector(source_id)
            return
        if source_id in self._inspectors:
            return
        source = self.project.sources.get(source_id)
        if source is not None and source.kind == "points":
            self._notify(INSPECTOR_NO_RASTER_TEXT, "status")
            # ``set_inspector_open`` blocks the panel's signals for the call, so pushing the
            # button back down cannot echo a second ``inspectorToggled`` at this handler.
            self.layer_list.set_inspector_open(source_id, False)
            return
        node = next((n for n in provenance_tree(self.project) if n.source_id == source_id), None)
        if node is None:
            return
        inspector = InspectorWindow(source_id, node.label, node, self._scale_reading, parent=self)
        inspector.closed.connect(self._on_inspector_closed)
        # The inspector's OWN scale control is routed back into
        # that inspector's display and nowhere else. Routed HERE rather than acted on inside the
        # window because this class is the one that COULD have written ``_params``/``layer.chain``
        # -- putting the route where the temptation lives is what makes the containment
        # guarantee visible to the next reader.
        inspector.scaleChanged.connect(self._on_inspector_scale_changed)
        # Every OTHER way this window's remembered state changes. Connected before it is
        # registered above, so the first geometry events (which arrive with ``show()``, below)
        # already find it in ``self._inspectors`` and are captured rather than dropped.
        inspector.geometryChanged.connect(self._on_inspector_state_changed)
        inspector.followMasterToggled.connect(self._on_inspector_state_changed)
        inspector.subLayerToggled.connect(self._on_inspector_state_changed)
        inspector.subLayerOpacityChanged.connect(self._on_inspector_state_changed)
        self._inspectors[source_id] = inspector
        field = self._field_for_source(source_id)
        if field is not None:
            inspector.canvas.set_field(field)
            # The same pair ``load_field`` uses on the center canvas: ``set_field`` registers the
            # image but does not frame it, and an unframed view opens on pyqtgraph's default
            # range rather than on the raster.
            inspector.canvas.view.autoRange()
        renderable = self._last_renderables.get(source_id)
        if renderable is not None:
            inspector.show_result(
                renderable, active=(self._layer_by_id.get(renderable.layer_id) is self.layer))
        status = self._inspector_status(source_id)
        inspector.set_status(status, live=self._inspector_is_live(source_id))
        inspector.show()
        # An open window is state worth remembering. After ``show()``, so the geometry
        # written is the one the window actually took.
        self._persist_inspectors()
        if source_id != getattr(self.layer, "source_id", None):
            self._notify(status, "status")

    def _on_inspector_closed(self, source_id: str) -> None:
        """The window closed itself (its own close button, or :meth:`_close_inspector`). Retiring
        it here rather than at each call site is what keeps the two ways of closing ONE state --
        the layer-list toggle un-checks under ``blockSignals`` (``LayerPanel.set_inspector_open``),
        so pushing it back can never echo a second toggle at the window that just closed.

        **Shutdown is not a gesture.** ``open: False`` is written only while
        :attr:`_shutting_down` is False. cmd-Q closes every top-level window, this one included,
        and writing "the user closed it" for a window the QUIT closed would make "quit with an
        inspector open ... reopen and the inspector reappears" false. The geometry/scale/row
        capture still runs on the way out, because where the window WAS is worth remembering
        either way; the one field that is suppressed is the one the shutdown has no opinion about.
        """
        inspector = self._inspectors.pop(source_id, None)
        self.layer_list.set_inspector_open(source_id, False)
        if inspector is not None:
            # Capture WHERE it was and how it was set up before it goes -- a window
            # reopened later should come back where the user left it, not at the platform's
            # default corner -- then record the one thing that did change: it is closed.
            state = self._capture_inspector_state(source_id, inspector)
            if not self._shutting_down:
                state.open = False
            self._inspector_states[source_id] = state
            inspector.deleteLater()
        self._persist_inspectors()

    def _close_inspector(self, source_id: str) -> None:
        """Close one from the outside (the "I" toggle going off). ``close()`` routes through the
        window's own ``closeEvent`` -> ``closed`` -> :meth:`_on_inspector_closed`, so a window
        shut this way and one the user closed retire down the identical path."""
        inspector = self._inspectors.get(source_id)
        if inspector is not None:
            inspector.close()

    def _drive_follower_inspectors(self, idx: int) -> None:
        """The master track: push the master's scale INDEX into every FOLLOWING
        inspector.

        The index only. Hölder exponents compare across datasets; raw modulus coefficients do not
        without the sec 11 calibration stage, so a master that drove anything more than the index
        would be inviting a comparison the physics does not support yet -- recorded here because
        the next hand to touch this method is the one that would be tempted.

        Each adopt is silent (``InspectorWindow.set_scale_index`` -> ``Transport.sync_to``): this
        window has already applied the change, and an echo would re-run a filter chain for it.
        An inspector whose "M" is off is skipped, which is the whole of what un-checking it buys
        -- its slider freezes while the master keeps moving, so the two windows visibly stop
        agreeing about which scale index they are on.

        **What this does NOT yet buy.** Sec 5d's use case is the comparison of two extrema SETS
        at two scales, and that is not what un-checking "M" delivers today: the picture in an
        inspector is the chain's own resolved result, which is post-``scale_select`` (ONE extrema
        layer), so a window driven to another index has no other scale in it to draw. It says so
        instead (``InspectorWindow.redraw_at`` -> ``SCALE_NOT_SHOWN_TEXT``). Delivering the
        divergent PICTURE needs a per-source result at a per-source scale -- the pre-filter stack
        kept per source, or a cached re-filter -- and that is an open sec 5b "no new compute
        path" decision.

        Called from every place the master's index actually SETTLES, not from
        ``Transport.scaleChanged`` alone: that signal is emitted by ``_set_index`` only -- a
        scrub or a playback tick -- while the strip knob (``_on_param_changed``) and a layer
        switch (``_sync_transport``) settle the master through the SILENT ``sync_to``, and a
        follower those never reached would sit at a stale index with its "M" still checked. The
        ``_syncing`` guard below is ``_on_scale_changed``'s, for ``_on_scale_changed``'s reason:
        while ``_sync_transport`` is re-ranging, this signal carries the transport's own stale
        position echoed out of ``set_n_scales``' clamp, not a settled index -- and pushing THAT
        into a follower is the divergence, not the fix.
        """
        if self._syncing:
            return
        for inspector in self._inspectors.values():
            if inspector.follow_master:
                inspector.set_scale_index(int(idx))

    def _on_inspector_scale_changed(self, source_id: str, idx: int) -> None:
        """One inspector's OWN scale control moved.

        Into that inspector's display only -- never ``_params``, never ``layer.chain``, never the
        master. An inspector must not be able to move the ACTIVE layer's chain behind the user's
        back, which is why the route is this narrow and why it lives here rather than anywhere
        the chain is written.
        """
        inspector = self._inspectors.get(source_id)
        if inspector is not None:
            inspector.redraw_at(int(idx))
            # This one IS the user's own gesture on this window (unlike a follower's
            # silent adopt), so it is what a restart should bring back.
            self._persist_inspectors()

    def _fan_out_to_inspectors(self, renderable) -> None:
        """``resolved`` reached the window: hand it to the inspector open over ITS source.

        The one subscriber to ``resolved`` (sec 5b's live-update contract; ``_apply`` itself is
        untouched). It lives here rather than in each inspector because resolving
        ``renderable.layer_id`` to a source needs ``_layer_by_id`` -- the mapping an inspector is
        deliberately denied (sec 7 rule 3) -- and because ONE connection for N windows is what
        keeps a filter edit inside the existing frame budget.

        The last-Renderable stamp happens even with nothing open, so an inspector opened AFTER a
        result landed still comes up showing it instead of an empty canvas.
        """
        layer = self._layer_by_id.get(renderable.layer_id)
        source_id = getattr(layer, "source_id", None)
        if source_id is None:
            return
        self._last_renderables[source_id] = renderable
        inspector = self._inspectors.get(source_id)
        if inspector is not None:
            inspector.show_result(renderable, active=(layer is self.layer))
        self._refresh_inspector_statuses()

    # -- the ROI flow ----------------------------------------------------------------------
    def _reset_roi_selection(self) -> None:
        """Drop the drawn selection: the panel goes away and the amber box comes off the canvas.

        Called whenever the window changes what it is showing -- a layer switch or a new file. A
        drawn box belongs to the LAYER it was drawn on, and leaving the panel up across a switch
        is not merely untidy: Create acts on whatever layer is selected NOW, so a stale panel is a
        button pointed at the wrong target.
        """
        self.roi_panel.setVisible(False)
        self.canvas.clear_roi_band()

    def _on_roi_panel_closed(self) -> None:
        """The panel's ×: close it, drop the unsaved drawn box and any pending Place box. Saved
        ROIs and the active one are not the panel's to change."""
        self.canvas.disarm_roi_placement()
        self._reset_roi_selection()

    def _roi_blocked_reason(self) -> str:
        """Why the selected layer cannot take an ROI, or ``""``.

        Nested ROIs are out of scope for this slice, and the chain says so plainly: a layer whose
        chain already holds ``wtmm2d_roi`` would get a SECOND one from ``roi_chain`` (which
        replaces ``wtmm2d``, and there is none), producing a two-transform chain that can only
        ever error. Refused at the panel, with a reason, rather than minted and left to fail.
        """
        if self.layer is not None and any(ref.device == "wtmm2d_roi"
                                          for ref in self.layer.chain.steps):
            return ("this layer is already an ROI — select its parent to draw another "
                    "(ROIs of ROIs are not supported yet)")
        return ""

    def _on_roi_values_edited(self, spec: dict) -> None:
        """The panel's numbers changed and describe a legal box: move the drawn band to match.

        One direction only -- the canvas never writes back into the panel -- so a drag and a
        numeric edit cannot fight. The band and the fields now always agree, which is the whole
        reading: without it, typing a row/col has no visual echo and the user is aiming blind."""
        if self.canvas._roi_place is not None:       # armed placement: edits resize the ghost
            self.canvas.arm_roi_placement(spec["roi_h"], spec["roi_w"])
            return
        self.canvas.show_roi_band(spec["roi_row"], spec["roi_col"], spec["roi_h"], spec["roi_w"])

    def _on_roi_drawn(self, row: int, col: int, h: int, w: int) -> None:
        """A ⌘-drag finished on the canvas: show the precision panel over what it drew.

        The conversion pair goes in whole, so the panel's physical readouts are the same numbers
        the transport, the scale bar and the wavelet bar are showing for this field.

        ``dims`` is the DISPLAYED image's own shape -- ``row``/``col``/``h``/``w`` are local to
        that image, same as the drag itself, so the panel's edit-time bound has to check against
        it rather than against ``provenance["full_dims"]`` (the whole parent file, a different and
        usually much larger extent on a windowed field).
        """
        values = getattr(self.field, "values", self.field)
        # FILE pixels: a display picture reports its file's dims.
        dims = native_shape(self.field) if hasattr(values, "shape") else None
        self.roi_panel.show_roi(row, col, h, w, px_to_metres(self.field),
                                blocked=self._roi_blocked_reason(), dims=dims)

    def _on_roi_tool_clicked(self) -> None:
        """"ROI…" (left column): open the precision panel WITHOUT a drag -- the manual-dimensions
        entry point. Re-opens over the panel's own previous numbers when it holds a legal spec
        (reproducing/adjusting an ROI is the main use); otherwise seeds a centered box of up to
        512 px. Everything downstream -- live band tracking, Create, Child dataset -- is the
        drag path's own machinery (``_on_roi_drawn``)."""
        if self.field is None or (self.layer is not None and self._is_point_layer(self.layer)):
            self._notify("open a raster first — the ROI tool needs an image to window", "status")
            return
        dims = native_shape(self.field)          # FILE pixels
        prev = self.roi_panel.values()
        if prev is not None:
            row, col = int(prev["roi_row"]), int(prev["roi_col"])
            h, w = int(prev["roi_h"]), int(prev["roi_w"])
        else:
            h = w = int(min(512, dims[0], dims[1]))
            row = max(0, (dims[0] - h) // 2)
            col = max(0, (dims[1] - w) // 2)
        self._on_roi_drawn(row, col, h, w)
        # The drag paints its own band; the tool path has no drag, so echo the seed onto the
        # canvas the same way an edit would -- the band and the fields agree from the start.
        spec = self.roi_panel.values()
        if spec is not None:
            self._on_roi_values_edited(spec)

    def _on_roi_place_arm(self, h: int, w: int) -> None:
        """The panel's "Place box": arm the same hover ghost the wtmm2d_roi drop uses, but in
        REPOSITION mode -- the click updates the panel's row/col (and the band), it never
        creates a layer. The armed-drop path keeps its create-on-click behavior."""
        if self.field is None:
            return
        self._roi_place_reposition = True
        self.canvas.arm_roi_placement(int(h), int(w))

    def _on_roi_placed(self, row: int, col: int, h: int, w: int) -> None:
        """The armed footprint was clicked down: stamp the ROI and run -- the exact create path
        the precision panel's button takes (_on_roi_create), so window-offset translation,
        minting and selection stay one implementation. Boundary comes from the panel when it
        holds a legal spec, else the device default.

        REPOSITION mode (the panel's "Place box"): the click only re-aims the panel's numbers
        -- nothing is created, nothing runs."""
        if getattr(self, "_roi_place_reposition", False):
            self._roi_place_reposition = False
            self._on_roi_drawn(row, col, h, w)
            return
        spec = self.roi_panel.values() or {}
        boundary = spec.get("boundary") or defaults_for(get_device("wtmm2d_roi"))["boundary"]
        self.canvas.disarm_roi_placement()       # idempotent for the canvas's own click path
        self.roi_panel.setVisible(False)
        self._on_roi_create({"roi_row": int(row), "roi_col": int(col),
                             "roi_h": int(h), "roi_w": int(w), "boundary": boundary})

    def _on_roi_create(self, spec: dict) -> None:
        """Create was pressed: spawn the grouped ROI layer and select it.

        **The ROI window is translated into the SOURCE's coordinates.** What the user drew is in
        the coordinates of the IMAGE on screen, which for a raster too large to load whole is one
        window of the file (``open_field`` reads a centered native-resolution window; its offsets
        are recorded in ``provenance["window"]``). ``wtmm2d_roi`` re-reads its own per-scale halos
        straight off the source file, so its ``roi_row``/``roi_col`` address the FILE. On the BOEM
        rasters this feature exists for those two frames differ by thousands of pixels, and
        without this the app would analyse -- and cache, and label -- a region nobody selected,
        silently and plausibly. A whole-file layer has no window offset and is unaffected.

        **The parent's FIELD OBJECT is reused for the ROI layer.** It is not a stand-in: the ROI
        device never reads pixels from the field it is handed (see ``WTMM2DROI.compute``, which
        goes to ``provenance["source"]`` for every window), so the field's only jobs here are to
        supply that provenance and the frame the result carries. Building a second field for the
        ROI would mean re-reading the raster to produce an array nothing will look at -- and,
        worse, one whose provenance would then have to be rewritten to keep pointing at the
        parent file.
        """
        parent = self.layer
        if parent is None:
            return
        field = self._fields.get(parent.layer_id, self.field)
        prov = getattr(field, "provenance", None) or {}
        window = prov.get("window") or {}
        # Overview drill-down: a footprint drawn on a decimated OVERVIEW
        # (geo.footprints.overview_field, provenance["overview"] = the factor) is in overview
        # pixels -- scale all four numbers to NATIVE file pixels so wtmm2d_roi reads full-res
        # halos from the source. An overview starts at the file origin, so its window offset
        # is 0; a native windowed field has overview 1 and keeps its own offsets.
        ov = int(prov.get("overview", 1) or 1)
        roi_params = dict(spec)
        roi_params["roi_row"] = int(spec["roi_row"]) * ov + int(window.get("row_off", 0))
        roi_params["roi_col"] = int(spec["roi_col"]) * ov + int(window.get("col_off", 0))
        if ov != 1:
            roi_params["roi_h"] = int(spec["roi_h"]) * ov
            roi_params["roi_w"] = int(spec["roi_w"]) * ov

        # "Save + run WTMM" -- the region it runs on is saved too.
        roi = self.project.add_roi(int(roi_params["roi_row"]), int(roi_params["roi_col"]),
                                   int(roi_params["roi_h"]), int(roi_params["roi_w"]),
                                   source_id=parent.source_id,
                                   label=self._next_roi_label(parent.source_id))
        self._active_roi_id = roi.roi_id
        bands = parent.tags.get("data.bands")
        layer = self.project.add_layer(
            f"{parent.name} ROI {roi_params['roi_row']},{roi_params['roi_col']}",
            parent.source_id, roi_chain(parent.chain, roi_params), parent_id=parent.layer_id,
            tags={"data.bands": bands} if bands else None)
        layer.roi_id = roi.roi_id
        self.add_layer_row(layer, field)
        self.layer_list.select_layer(layer.layer_id)   # -> _select_layer -> worker

    # -- saved ROIs --------------------------------------------------------
    def _next_roi_label(self, source_id) -> str:
        """A, B, ..., Z, AA, AB, ... -- per source, in save order."""
        n = sum(1 for r in self.project.rois if r.source_id == source_id)
        label = ""
        n += 1
        while n:
            n, rem = divmod(n - 1, 26)
            label = chr(ord("A") + rem) + label
        return label

    def _active_roi_for(self, source_id):
        """The active saved ROI when it belongs to ``source_id``, else ``None``."""
        if not self._active_roi_id:
            return None
        return next((r for r in self.project.rois
                     if r.roi_id == self._active_roi_id and r.source_id == source_id), None)

    def _refresh_saved_rois(self) -> None:
        """Layer-list ROI rows, panel list + canvas outlines for the displayed layer's
        dataset. A hidden ROI (its row's H, ``RoiRecord.visible``) draws no outline."""
        self.layer_list.sync_roi_rows()
        source = self.layer.source_id if self.layer is not None else None
        rois = [r for r in self.project.rois if r.source_id == source and source is not None]
        self.roi_panel.set_saved([(r.roi_id, r.label) for r in rois],
                                 active=self._active_roi_id)
        self.canvas.set_saved_rois([{"label": r.label, "row": r.row, "col": r.col, "h": r.h,
                                     "w": r.w, "active": r.roi_id == self._active_roi_id}
                                    for r in rois if r.visible])

    def _on_roi_save(self, spec: dict) -> None:
        """Save ROI: the box becomes a ``RoiRecord`` on
        the displayed layer's SOURCE, in FILE pixels -- the panel's numbers plus the field's
        own window offset (a picture's is (0, 0); a centred native window's is its origin) --
        and becomes the active ROI: the next tool dropped runs on it."""
        if self.layer is None or self._is_point_layer(self.layer):
            return
        field = self._fields.get(self.layer.layer_id, self.field)
        off_r, off_c = window_offset(field)
        roi = self.project.add_roi(int(spec["roi_row"]) + off_r, int(spec["roi_col"]) + off_c,
                                   int(spec["roi_h"]), int(spec["roi_w"]),
                                   source_id=self.layer.source_id,
                                   label=self._next_roi_label(self.layer.source_id))
        self._active_roi_id = roi.roi_id
        self._refresh_saved_rois()
        self.layer_list.select_roi(roi.roi_id)           # its row is the selection now
        self._notify(f"ROI {roi.label} saved — drop a tool to run it on this region", "status")

    def _on_roi_activated(self, roi_id: str) -> None:
        self._active_roi_id = roi_id or None
        self._refresh_saved_rois()
        # Keep the layer list's highlight on what is active: the ROI's row, or the dataset row.
        if roi_id:
            self.layer_list.select_roi(roi_id)
        elif self.layer is not None and self.layer.parent_id is None:
            self.layer_list.select_layer(self.layer.layer_id)

    def _on_roi_row_selected(self, roi_id: str) -> None:
        """A saved-ROI row became current: show its dataset and make the ROI active, so the
        next tool dropped runs on it (the drop path itself is unchanged)."""
        roi = next((r for r in self.project.rois if r.roi_id == roi_id), None)
        if roi is None:
            return
        master = next((l for l in self.project.layers
                       if l.source_id == roi.source_id and l.parent_id is None), None)
        if master is not None and master is not self.layer:
            self._select_layer(master)
            if self._center_stack.currentIndex() == 1:
                self._sync_arrangement()
        self._active_roi_id = roi.roi_id
        self._refresh_saved_rois()

    def _on_roi_hide_toggled(self, roi_id: str, hidden: bool) -> None:
        """An ROI row's H: its outline only -- never the results computed on it."""
        roi = next((r for r in self.project.rois if r.roi_id == roi_id), None)
        if roi is None:
            return
        roi.visible = not hidden
        self._refresh_saved_rois()

    def _on_roi_child_create(self, spec: dict) -> None:
        """Child dataset was pressed: the drawn box becomes a NATIVE-PIXEL CROP of the
        displayed field, spawned as a grouped child layer with an EMPTY chain -- the fresh-open
        precedent (``load_field``: raster displays, no worker, the user builds a chain
        deliberately) -- so ANY tool can head it.

        This is deliberately NOT ``wtmm2d_roi``: no halo, no per-scale honesty margins -- the
        crop's edges carry ordinary edge effects and managing them is the analyst's call (the
        halo-honest path remains the wtmm2d_roi device). The no-resampling law holds: the crop
        is a slice of the native grid, which is also why a decimated OVERVIEW refuses -- a child
        cut from decimated pixels would silently analyze resampled data.

        **Cache identity:** the child layer carries ``tags["roi.window"]`` in FILE-absolute
        coordinates, folded into ``engine.resolve.source_identity`` -- two different windows on
        one source (or a window vs the whole) never share a cache line.
        """
        parent = self.layer
        if parent is None or self._is_point_layer(parent):
            return
        field = self._fields.get(parent.layer_id, self.field)
        if field is None:
            return
        prov = dict(getattr(field, "provenance", None) or {})
        r, c = int(spec["roi_row"]), int(spec["roi_col"])
        h, w = int(spec["roi_h"]), int(spec["roi_w"])
        vals = np.asarray(field.values)
        if r < 0 or c < 0 or r + h > vals.shape[0] or c + w > vals.shape[1]:
            return                              # the panel's own legality gate already said no
        ov = int(prov.get("overview", 1) or 1)
        if ov != 1:
            # A box on a whole-extent OVERVIEW cuts native pixels by reading the window straight
            # off the source at full resolution -- display -> native is r*ov + off, the same
            # mapping _on_roi_create below applies for wtmm2d_roi. Only a field with no source
            # path to read from is refused.
            src_path = prov.get("source")
            if not src_path:
                self._notify("a child dataset cuts NATIVE pixels — this overview carries no "
                             "source path to read them from", "status")
                return
            from dynamix.core.rasterfield import RasterField

            window = prov.get("window") or {}
            abs_r = r * ov + int(window.get("row_off", 0))
            abs_c = c * ov + int(window.get("col_off", 0))
            nat_h, nat_w = h * ov, w * ov
            full = prov.get("full_dims")
            if full is not None:
                nat_h = min(nat_h, int(full[0]) - abs_r)
                nat_w = min(nat_w, int(full[1]) - abs_c)
            src = self.project.sources.get(parent.source_id)
            base = (src.label or Path(src.path).stem) if src is not None else parent.name
            child_field = RasterField.from_geotiff_window(
                src_path, row_off=abs_r, col_off=abs_c, height=nat_h, width=nat_w,
                name=f"{base}@r{abs_r}c{abs_c}")
            child_field.provenance.setdefault("full_dims", prov.get("full_dims"))
            layer = self.project.add_layer(
                f"{base} · child[{abs_r},{abs_c} {nat_h}×{nat_w}]",
                parent.source_id, Chain(()), parent_id=parent.layer_id)
            layer.tags["roi.window"] = f"{abs_r},{abs_c},{nat_h},{nat_w}"
            self.add_layer_row(layer, child_field)
            self.layer_list.select_layer(layer.layer_id)
            return
        # CORE-ONLY crop (the ROI defines the core; any apron a
        # tool needs is that TOOL's decision at compute time -- COI of the max wavelet scale,
        # kernel support, diffusion domain -- never a number chosen at crop time). The child's
        # provenance keeps the parent linkage (source, absolute window, full dims) so a future
        # per-tool apron sampler has everything it needs.
        window = prov.get("window") or {}
        abs_r = r + int(window.get("row_off", 0))
        abs_c = c + int(window.get("col_off", 0))
        prov["window"] = {"row_off": abs_r, "col_off": abs_c}
        child_field = dataclasses.replace(
            field, values=vals[r:r + h, c:c + w].copy(),
            x_axis=np.asarray(field.x_axis)[c:c + w].copy(),
            y_axis=np.asarray(field.y_axis)[r:r + h].copy(),
            name=f"{field.name}@r{abs_r}c{abs_c}", provenance=prov)
        src = self.project.sources.get(parent.source_id)
        base = (src.label or Path(src.path).stem) if src is not None else parent.name
        layer = self.project.add_layer(
            f"{base} · child[{abs_r},{abs_c} {h}×{w}]",
            parent.source_id, Chain(()), parent_id=parent.layer_id)
        layer.tags["roi.window"] = f"{abs_r},{abs_c},{h},{w}"
        self.add_layer_row(layer, child_field)
        self.layer_list.select_layer(layer.layer_id)

    # -- refined run -------------------------------------------------------------------------
    def _on_refined_run(self, layer_id: int) -> None:
        """The layer panel's "New refined run" context action: a grouped layer holding the
        parent's TRANSFORM steps only -- filters are dropped, not carried over, since a refined
        run is meant to be re-tuned from a clean filter tail rather than starting from whatever
        the parent's screen happened to be showing.

        Every carried transform's params are DEEP-copied (``copy.deepcopy``, not the shallow
        ``dict(ref.params)`` ``roi_chain`` uses elsewhere) -- unlike an ROI window's few scalar
        overrides, an arbitrary transform param can itself be a mutable container (a device is
        free to declare one), and this path copies the WHOLE parent chain rather than rewriting
        one named step, so nothing here can rely on knowing which params are safe to alias.

        ``parent.chain.transforms`` is already ``Chain._split()``'s leading half -- a validated
        layer's chain is never interleaved, so this can never raise the way it would over an
        arbitrary (not-yet-validated) list.

        **Lock is not consulted here, deliberately.** ``_on_param_changed``/``_on_chain_edited``/
        ``_on_scale_changed`` all refuse under ``_lock_notice`` because they EDIT the layer they
        are called for. This method never writes to ``parent`` at all -- it reads its chain and
        mints a brand-new layer elsewhere in the project -- so a locked (or frozen) parent is not
        an edit-guard violation and is allowed through unchanged, same as the ROI flow above,
        which carries no lock check either for the identical reason.

        **No data flows from the parent's own outputs yet.** ``fed_by`` is provenance metadata
        only, for the source chip (:meth:`_update_source_box`) -- there is no reprocess loop that
        feeds the parent's RESULT into this layer's computation; that is future work, not this
        slice's job.
        """
        parent = self._layer_by_id.get(layer_id)
        if parent is None:
            return
        field = self._fields.get(parent.layer_id, self.field)
        transforms = tuple(DeviceRef(ref.device, copy.deepcopy(ref.params))
                           for ref in parent.chain.transforms)
        layer = self.project.add_layer(
            f"{parent.name} · refined", parent.source_id, Chain(transforms).materialized(),
            parent_id=parent.layer_id,
            tags=_inherited_window(parent, fed_by=str(parent.layer_id)))
        self.add_layer_row(layer, field)
        self.layer_list.select_layer(layer.layer_id)   # -> _select_layer -> worker

    # -- notifications (the design / EQ§7: the three-tier fault-volume doctrine) --------
    # -- chain-product export ------------------------
    def _on_export_chains_clicked(self) -> None:
        """File > Export Chains: the ACTIVE result's draw-ready chain product, written as the
        EQSelect-compatible schema-v4 npz -- the SAME bundle in memory and on disk, so labels
        attach to the saved dataset and the two demo apps interoperate."""
        err = self._chain_export_blocker()
        if err:
            self._notify(err, "status")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Chains (.npz)", "", "WTMM chains (*.npz);;All files (*)")
        if not path:
            return
        self._notify(self._export_chain_product(path), "status")

    def _on_export_derived_clicked(self) -> None:
        """File > Export Derived Raster: the ACTIVE layer's derived dataset (h-map or band
        reconstruction) to .npz -- the "temporary mem until exported" half of the derived-
        dataset contract. RasterField.save_npz keeps axes/frame, so the
        export reloads as an ordinary dataset."""
        if self.layer is None:
            return
        dfield = self._derived_fields.get(self.layer.layer_id)
        if dfield is None:
            self._notify("no derived raster on the active layer — run holder_measure/"
                         "holder_multiaffine or a band reconstruction first", "status")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export derived raster", f"{dfield.name}.npz", "NumPy archive (*.npz)")
        if path:
            dfield.save_npz(path)
            self._notify(f"derived raster exported — {path}", "status")

    def _chain_export_blocker(self) -> str | None:
        """Why the export cannot run right now, or None -- split from the click handler so the
        export logic is testable without a file dialog."""
        result = self._active_result
        if not isinstance(result, dict) or result.get("chain_product") is None:
            return "no computed chain product to export — run a WTMM transform first"
        if (result.get("_roi") or {}).get("roi"):
            return ("ROI results are not exportable yet — their coordinates are ROI-local and "
                    "the parent field's axes would mislabel them")
        if self.field is None:
            return "no field loaded"
        return None

    def _export_chain_product(self, path) -> str:
        """Write the active product to ``path``; returns the status-line sentence."""
        import json as _json

        from dynamix.core.chain_product import export_chain_product_npz

        result = self._active_result
        params = result.get("params") or {}
        params = _json.loads(_json.dumps(params, default=list))   # q_list arrays -> lists
        info = export_chain_product_npz(path, result["chain_product"],
                                        field=self.field, params=params)
        return (f"exported {info['n_horizontal']} H + {info['n_vertical']} V chains "
                f"→ {path}")

    def _notify(self, msg: str, tier: str = "status") -> None:
        """The single place every surface this slice touches funnels an outward-facing message
        through, so the tiering rule lives in exactly one spot rather than one ad hoc
        ``statusBar().showMessage`` call per call site (pre-existing status-bar calls elsewhere in
        this file predate this method and are left as they were -- see the module's Non-goals: "any
        analysis-path change" is out of scope, and retrofitting unrelated call sites is not this
        task's job).

        ``"log"`` -- ``print(msg, file=sys.stderr)`` only, and returns before touching the status
        bar at all: never interrupts a gesture, dev-visible only (EQ§7's silent tier -- drag-filter
        selection failure, pick failures, and the like).

        ``"status"`` -- ``statusBar().showMessage(msg)`` with NO timeout: persists until the next
        ``_notify`` call replaces it (the design's own departure from EQSelect's fixed-ms fade). :attr:`_status_message` tracks
        the text alongside, so a caller or a test can read back what is currently showing without
        depending on ``QStatusBar.currentMessage()``'s own Qt-internal state.

        ``"modal"`` -- a ``QMessageBox.information`` box, gated on a REAL display
        (:func:`_is_offscreen`, EQSelect's own ``_on_wtmm_failed`` pattern: the modal
        exec is skipped entirely under the offscreen QPA platform this whole suite runs under,
        where it would otherwise block forever). The status bar is ALWAYS ALSO written first,
        unconditionally, for both "status" and "modal" -- the headless report a modal-only message
        would otherwise never produce -- which is what makes ``_notify(msg, "modal")`` safe to call
        from anywhere, test or real window, without the caller having to know which situation it is
        in.
        """
        if tier == "log":
            print(msg, file=sys.stderr)
            return
        self._status_message = msg
        self.statusBar().showMessage(msg)
        if tier == "modal" and not _is_offscreen():
            QtWidgets.QMessageBox.information(self, TITLE, msg)

    def _undo_rack_removal(self) -> None:
        """Wrapper target for the Shift+⌘Z shortcut (see its own construction-site comment for why
        a wrapper, not a direct connection, is needed here) -- re-reads ``self.strips`` at
        activation time so it always reaches whichever ``WorkflowZone`` is live right now, even
        across a rebuild. A no-op with no zone built yet (mirrors every other ``self.strips is not
        None`` guard already in this file, e.g. :meth:`_sync_lock_ui`)."""
        if self.strips is not None:
            self.strips.undo_removal()

    # -- chain / strips --------------------------------------------------------------------
    def _set_recipe(self, names: list, params: list) -> None:
        """The window's working copy of the chain being edited: ``_names``/``_params`` plus
        parallel bypass/rack bookkeeping, always kept the same length as each other.

        A fresh recipe -- a new layer, a reseed, an inert open, or a switch to a layer never
        edited before -- starts with nothing bypassed and no rack grouping: those are VIEW state
        ``WorkflowZone`` tracks, never round-tripped from the persisted ``Chain``/``DeviceRef``
        model, which has no such fields. ``_on_chain_edited`` and ``_select_layer``'s own restore
        from ``self._recipes`` are the two places that write anything ELSE into
        ``_bypassed``/``_rack`` -- from a live ``WorkflowZone`` descriptor list, or a previously
        recorded one, rather than from a fresh chain."""
        self._names = list(names)
        self._params = [dict(p) for p in params]
        self._bypassed = [False] * len(self._names)
        self._rack: list[str | None] = [None] * len(self._names)

    def _chain(self) -> Chain:
        """The ENABLED steps only -- a bypassed step is excluded from what actually computes,
        even though it stays in ``_names``/``_params``/``self.strips`` at its own position."""
        return Chain(tuple(DeviceRef(self._names[i], dict(self._params[i]))
                           for i in range(len(self._names))
                           if not self._bypassed[i])).materialized()

    def _build_strips(self) -> None:
        """Build the workflow zone from ``_names``/``_params``/``_bypassed``/``_rack`` -- the
        window's own working copy, not ``layer.chain.steps`` directly, so a strip always exists
        for EVERY tracked step, including a bypassed one ``layer.chain`` no longer carries.

        Called only right after those four are freshly populated for the layer about to be shown
        -- either ``_set_recipe`` (a fresh layer load, or a switch to one never edited before,
        where bypass/rack reset to ``False``/``None`` for every step) or ``_select_layer``'s own
        restore from ``self._recipes`` (a switch back to a layer that WAS edited before -- bypass/rack there are whatever that layer last had, not reset).
        """
        if self.strips is not None:
            self._strip_layout.removeWidget(self.strips)
            self.strips.setParent(None)
            self.strips.deleteLater()
        self.strips = WorkflowZone()
        self.strips.set_presets({"WTMM standard": MainWindow.DEFAULT_STEPS, **PRESETS})
        self.strips.paramChanged.connect(self._on_param_changed)
        self.strips.chainEdited.connect(self._on_chain_edited)
        # A fresh SourceBox lives inside this fresh WorkflowZone, so the chip-hover ->
        # flash relay has to be rewired every rebuild, same as paramChanged/chainEdited above.
        self.strips.chipHovered.connect(self.layer_list.flash_row)
        # Relay both zone-level report signals to the window's status bar --
        # the zone ALSO shows each message locally (dropPlaced's own _commit_or_revert call site;
        # rackRemoved's own _on_rack_remove_requested), so this is the dual zone-local-plus-window
        # reporting the module docstring describes, not a replacement for it. A
        # genuine REFUSAL (persistent zone warning + flash) is deliberately NOT relayed here --
        # that path stays zone-local, already loud -- reporting it again at the window
        # level would be the double-report the design rules out.
        self.strips.dropPlaced.connect(lambda _device, message: self._notify(message, "status"))
        self.strips.rackRemoved.connect(lambda _title, message: self._notify(message, "status"))
        descriptors = [{"device": n, "params": dict(p), "bypassed": b, "rack": r}
                       for n, p, b, r in zip(self._names, self._params, self._bypassed, self._rack)]
        self.strips.set_steps(descriptors, field=self.field)
        self._strip_layout.addWidget(self.strips)
        self.strips.select(0)
        self._update_source_box()
        self._apply_pending_ui()
        self._sync_recon_section()

    def _update_source_box(self) -> None:
        """Make the zone's ``SourceBox`` real. Called at the end of :meth:`_build_strips`
        -- the only two call sites (``load_field``, ``_select_layer``) both have ``self.layer``/
        ``self.field`` already pointed at the layer about to be shown, so this always names
        exactly the layer that just became active, for EVERY selection, not only a refined-run
        child.

        ``fed_by`` is read from the layer's OWN ``tags["fed_by"]`` (:meth:`_on_refined_run`'s own
        stamp) -- never from ``parent_id``, which an ROI child also carries and which gets no chip
        of its own. A tag that names a parent no longer in ``_layer_by_id`` (defensive only; the
        cascade in ``_on_remove_requested`` removes a refined child along with its parent) simply
        shows no chip rather than a name it cannot honestly produce.
        """
        if self.strips is None or self.layer is None:
            return
        fed_by_id = self.layer.tags.get("fed_by")
        fed_by_name = None
        parsed_id = None
        if fed_by_id is not None:
            parsed_id = int(fed_by_id)
            parent = self._layer_by_id.get(parsed_id)
            fed_by_name = parent.name if parent is not None else None
        self.strips.set_source(self.layer.name, getattr(self.field, "provenance", None),
                               fed_by=fed_by_name, fed_by_layer_id=parsed_id)

    def _transform_indices(self) -> list[int]:
        """Positions (in the FULL ``_names`` index space) of every ENABLED transform -- a
        bypassed one contributes nothing to the worker, so it claims no state dot and no share of
        ``_transform_signature``."""
        return [i for i, name in enumerate(self._names)
                if not self._bypassed[i] and is_transform(get_device(name))]

    def _index_of(self, device_name: str) -> int | None:
        return self._names.index(device_name) if device_name in self._names else None

    def _mark_transforms_pending(self) -> None:
        """A transform edit under manual run: record it, show it, dispatch nothing.
        The display keeps the LAST RUN's result; Run (or ⌘↩) commits everything at once."""
        if self.layer is None:
            return
        self._pending_layers.add(self.layer.layer_id)
        self._apply_pending_ui()
        self._notify("transform edited — press Run (⌘↩) to compute", "status")

    def _run_transforms(self) -> None:
        """Run answers EVERY click: pending edits dispatch; an errored or never-computed chain
        dispatches too; a fully computed chain says so instead of silently ignoring the click."""
        if self.layer is None:
            self._notify("no layer to run — open a dataset first", "status")
            return
        if self.layer.layer_id in self._pending_layers:
            self._pending_layers.discard(self.layer.layer_id)
            self._apply_pending_ui()
            self._start_worker()
            return
        keys = self._cache_keys_for(self.layer)
        if self._errored or (keys and keys[-1] not in self.cache):
            self._start_worker()
        elif not keys:
            self._notify("the chain has no transform — drop wtmm2d in the rack, then Run", "status")
        else:
            self._notify("already computed — edit a transform (or its params) to re-run", "status")

    def _mark_stopped_pending(self) -> None:
        """A Stop stood down the active layer's analysis: its recipe is uncomputed again, so it
        waits for Run like an edit does. The Run button is enabled only while edits wait, and
        the recipe's edit was taken off the pending set when its run was dispatched."""
        self._pending_layers.add(self.layer.layer_id)
        self._apply_pending_ui()

    def _apply_pending_ui(self) -> None:
        pending = self.layer is not None and self.layer.layer_id in self._pending_layers
        self.run_button.setEnabled(pending)
        if pending:
            self._set_transform_states("pending")

    def _transform_signature(self):
        """What the worker would compute: the source plus every ENABLED transform step's params.
        A change here is the ONLY thing that needs the worker; anything else -- a filter, or any
        change to a BYPASSED step -- is not (rack membership never appears here at all: it is
        pure rendering, invisible to the computed chain, which is what keeps the zero-cache-miss
        law true across rack flattening)."""
        return (self.layer.source_id if self.layer is not None else None,
                tuple((self._names[i], tuple(sorted(
                    keyed_params(get_device(self._names[i]), self._params[i]).items())))
                      for i in self._transform_indices()))

    def _snapshot_recipe(self) -> None:
        """Record the CURRENT ``_names``/``_params``/``_bypassed``/``_rack`` for ``self.layer`` in
        ``self._recipes``, so a later switch away and back (``_select_layer``)
        restores bypass/rack state a fresh derive from ``layer.chain`` cannot -- that chain
        excludes bypassed steps entirely and has no ``rack`` field at all. Called at the end of
        every path that mutates that state for the currently selected layer."""
        if self.layer is None:
            return
        self._recipes[self.layer.layer_id] = [
            {"device": n, "params": dict(p), "bypassed": b, "rack": r}
            for n, p, b, r in zip(self._names, self._params, self._bypassed, self._rack)
        ]

    # -- the two paths ---------------------------------------------------------------------
    def _on_param_changed(self, step_index: int, name: str, value) -> None:
        if self.layer is None:          # no active layer (e.g. the last one was just removed)
            return
        notice = _lock_notice(self.layer)
        if notice is not None:
            self.strips._show_warning(notice)
            return
        self._params[step_index][name] = value
        if self._names[step_index] == "scale_select" and name == "scale_idx":
            self._load_level_filters(int(value) + 1)
        elif not is_transform(get_device(self._names[step_index])):
            self._store_level_filter(step_index)
        if self._is_point_layer(self.layer):
            # Re-derive the seven `_target_*` scalars whenever a point layer's own
            # `backproject` step moved -- most directly when this WAS the "target" text edit, but
            # unconditionally on any edit for this layer is simplest and cheap: the loop is a
            # no-op the instant no step is named "backproject".
            self._stamp_backproject(self._names, self._params)
        self.layer.chain = self._chain()
        self._snapshot_recipe()
        if self._names[step_index] == "scale_select" and name == "scale_idx":
            # The scale is addressable from two places; the one that did not originate this
            # change still has to follow it, or the slider and its reading contradict the canvas
            # and the next playback tick snaps back to the stale position.
            self.transport.sync_to(int(value))
            # ... and so does every FOLLOWING inspector:
            # ``sync_to`` is silent by design, so the master's second connection never fires on
            # this path and a follower would be left behind by an ordinary knob turn.
            self._drive_follower_inspectors(int(value))
        device = get_device(self._names[step_index])
        view_only = any(p.name == name and p.view for p in device.params)
        if not self._bypassed[step_index] and is_transform(device) and not view_only:
            if self._auto_run_action.isChecked():
                self._pending_layers.discard(self.layer.layer_id)
                self._start_worker()
            else:
                self._mark_transforms_pending()
        else:
            self._reresolve()
        self._sync_recon_section()

    def _on_chain_edited(self, descriptors) -> None:
        """A drag-drop assembly gesture landed: rebuild ``_names``/``_params``/``_bypassed``/
        ``_rack`` wholesale from what the zone now holds -- the only way the window and the zone
        can never desync, since nothing here trusts an index to have stayed put across a reorder,
        a nest, or a remove.

        The candidate ``Chain`` is built FIRST, from LOCAL variables, before anything on ``self``
        is touched: the zone's own refusal gate
        (``WorkflowZone._commit_or_revert``) should mean this method never actually receives an
        illegal descriptor list, but if a future bug ever slips one through anyway,
        ``Chain(...).materialized()`` raises HERE, before ``_names``/``_params``/``_bypassed``/
        ``_rack``/``layer.chain`` have been reassigned -- leaving the window's bookkeeping exactly
        as it was, not half-mutated the way it used to be (the repro: bypass a filter,
        drop a transform after it, un-bypass -- the third step used to raise here with
        ``_names``/``_params`` already pointing at the illegal recipe, wedging every subsequent
        knob edit on the same line).

        Mirrors ``_on_param_changed``'s branch exactly, generalised from "was the one step that
        changed a transform" to "did the set of enabled transforms (or their params) change at
        all" -- the signature comparison IS that generalisation: editing a filter, or toggling
        bypass on a filter, never moves it; adding/removing/re-enabling a transform, or editing an
        enabled transform's param via a preset, always does.
        """
        if self.layer is None:          # no active layer (e.g. the last one was just removed)
            return
        notice = _lock_notice(self.layer)
        if notice is not None:
            self.strips._show_warning(notice)
            return
        if (display_stride(self.field) > 1 and not self.layer.tags.get("roi.window")
                and self._active_roi_for(self.layer.source_id) is None
                and any(is_transform(get_device(d["device"]))
                        and not getattr(get_device(d["device"]), "reads_source", False)
                        for d in descriptors if not d.get("bypassed"))):
            # No tool runs on the display PICTURE -- it is a drawing of a
            # file too big to analyse whole. Revert the zone (after this signal unwinds, the
            # armed-roi idiom) and say how to get native pixels.
            revert = [{"device": n, "params": dict(pp), "bypassed": b, "rack": r}
                      for n, pp, b, r in zip(self._names, self._params, self._bypassed,
                                             self._rack)]
            QtCore.QTimer.singleShot(0, lambda: self.strips.set_steps(revert, field=self.field))
            self._notify("this dataset is shown as a picture (too big to analyse whole) — "
                         "draw a box and Save ROI, then drop the tool to run it on the ROI",
                         "status")
            return
        if self.layer.tags.get("roi.window"):
            # An ROI-holder child NEVER forks. A drop that introduces a DIFFERENT primary
            # analyzer replaces the one already in the proposal, in place -- the child is a
            # region plus one representation at a time, filters kept. The zone is resynced to
            # the transformed proposal after this signal unwinds (the armed-roi revert idiom).
            enabled = [d["device"] for d in descriptors if not d.get("bypassed")]
            primaries = [n for n in enabled if n in _PRIMARY_ANALYZERS]
            if len(primaries) > 1:
                old_primary, new_primary = primaries[0], primaries[-1]
                new_desc = next(d for d in reversed(descriptors)
                                if d["device"] == new_primary and not d.get("bypassed"))
                kept, replaced = [], False
                for d in descriptors:
                    if d is new_desc:
                        continue                  # lift it out of its dropped position...
                    if not replaced and d["device"] == old_primary and not d.get("bypassed"):
                        kept.append(new_desc)     # ...into the old representation's slot,
                        replaced = True           # keeping transforms-then-filters legal
                        continue
                    kept.append(d)
                descriptors = kept
                QtCore.QTimer.singleShot(
                    0, lambda ds=[dict(d) for d in descriptors]:
                    self.strips.set_steps(ds, field=self.field))
        names = [d["device"] for d in descriptors]
        params = [dict(d.get("params", {})) for d in descriptors]
        bypassed = [bool(d.get("bypassed", False)) for d in descriptors]
        rack = [d.get("rack") for d in descriptors]
        # Armed ROI placement (dropping wtmm2d_roi arms a placement: choose the window, hover to see the footprint, click to set the ROI and run). A drop that ADDS wtmm2d_roi is a
        # placement gesture, not a chain edit: nothing is committed and nothing runs -- the zone
        # is reverted to the layer's real chain (after this signal unwinds), the precision panel
        # opens on the window size, the canvas ghosts the footprint under the cursor, and the
        # click takes _on_roi_create's own path. An edit to a chain that already HAS the device
        # (reorder, param change, an ROI layer's own rack) commits exactly as before.
        if ("wtmm2d_roi" in names and self.field is not None
                and not any(ref.device == "wtmm2d_roi" for ref in self.layer.chain.steps)):
            revert = [{"device": n, "params": dict(pp), "bypassed": b, "rack": r}
                      for n, pp, b, r in zip(self._names, self._params, self._bypassed, self._rack)]
            QtCore.QTimer.singleShot(0, lambda: self.strips.set_steps(revert, field=self.field))
            d = defaults_for(get_device("wtmm2d_roi"))
            values = getattr(self.field, "values", self.field)
            dims = native_shape(self.field) if hasattr(values, "shape") else None
            h = min(int(d["roi_h"]), dims[0]) if dims else int(d["roi_h"])
            w = min(int(d["roi_w"]), dims[1]) if dims else int(d["roi_w"])
            self.canvas.arm_roi_placement(h, w)
            self.roi_panel.show_roi(0, 0, h, w, px_to_metres(self.field),
                                    blocked="hover the raster — click to place and run", dims=dims)
            self._notify("wtmm2d_roi armed — set the window size below, hover the raster, "
                         "click to place and run", "status")
            return
        fork = (None if self.layer.tags.get("roi.window")
                else _analyzer_fork(self._names, self._bypassed, descriptors,
                                    root=self.layer.parent_id is None))
        if (fork is None and self.layer.parent_id is None
                and not self.layer.tags.get("roi.window")
                and (self._active_roi_for(self.layer.source_id) is not None
                     or self._is_raw_raster_dataset())):
            # The master is the raw dataset and never takes an analyzer directly: ANY analyzing
            # transform (pca/tucker included, which the primary-analyzer fork never knew) spawns
            # a child -- on the active ROI when there is one, else on the whole field. This used
            # to run only with an ROI active, so tucker without one replaced the dataset.
            fork = _roi_spawn(self._names, descriptors)
        if (fork is None and display_stride(self.field) > 1
                and not self.layer.tags.get("roi.window")
                and any(is_transform(get_device(d["device"]))
                        and not getattr(get_device(d["device"]), "reads_source", False)
                        for d in descriptors if not d.get("bypassed"))):
            # Nothing would spawn, and a transform would commit onto the PICTURE itself (a
            # field stage alone, e.g. noise): refuse and say how.
            revert = [{"device": n, "params": dict(pp), "bypassed": b, "rack": r}
                      for n, pp, b, r in zip(self._names, self._params, self._bypassed,
                                             self._rack)]
            QtCore.QTimer.singleShot(0, lambda: self.strips.set_steps(revert, field=self.field))
            self._notify("noise is a field stage — drop it together with an analyzer (e.g. as a "
                         "preset) and it runs on the active ROI and its margin", "status")
            return
        if fork is not None and self.field is not None:
            sibling_steps, shipped_idx = fork
            # A DIFFERENT analyzer landed on an analyzed layer: fork the new representation
            # into a grouped sibling (the _on_roi_create/band-commit pattern) and leave this
            # layer -- chain, recipe, display -- untouched. Deferred one tick: this handler
            # runs inside the zone's own chainEdited signal, and the spawn's select rebuilds
            # the zone (the same mid-signal hazard the roi branch's deferral guards).
            parent = self.layer
            field = self._fields.get(parent.layer_id, self.field)
            revert = [{"device": n, "params": dict(pp), "bypassed": b, "rack": r}
                      for n, pp, b, r in zip(self._names, self._params, self._bypassed,
                                             self._rack)]

            shipped = [{"device": self._names[i], "params": dict(self._params[i])}
                       for i in shipped_idx]

            def _spawn():
                # Deferred one tick, so the world may have moved: the layer switched, the
                # parent removed, even (tests) the registry restored -- verify, and put
                # EVERYTHING under the try so a late failure reverts instead of leaking.
                if self.layer is not parent or parent not in self.project.layers:
                    return
                try:
                    # Sibling steps: the drop's new steps + the parent's shipped consumers,
                    # laid out transforms-then-filters (the chain law) with encounter order
                    # kept inside each half.
                    combined = sibling_steps + shipped
                    transforms = [d for d in combined
                                  if is_transform(get_device(d["device"]))]
                    filters = [d for d in combined
                               if not is_transform(get_device(d["device"]))]
                    chain = Chain(tuple(DeviceRef(d["device"], dict(d.get("params", {})))
                                        for d in transforms + filters)).materialized()
                except Exception as exc:                        # noqa: BLE001
                    self.strips.set_steps(revert, field=self.field)
                    self.strips._show_warning(f"could not fork the new analysis: {exc}")
                    return
                if shipped_idx:
                    # The shipping law: MOVE the stranded consumers off the parent (params
                    # went with them). Parent recipe + chain updated; its next selection
                    # re-resolves from cache.
                    keep = [i for i in range(len(self._names)) if i not in set(shipped_idx)]
                    self._names = [self._names[i] for i in keep]
                    self._params = [self._params[i] for i in keep]
                    self._bypassed = [self._bypassed[i] for i in keep]
                    self._rack = [self._rack[i] for i in keep]
                    parent.chain = Chain(tuple(
                        DeviceRef(self._names[i], dict(self._params[i]))
                        for i in range(len(self._names))
                        if not self._bypassed[i])).materialized()
                    self._snapshot_recipe()
                label = next((d["device"] for d in sibling_steps
                              if d["device"] in _PRIMARY_ANALYZERS),
                             sibling_steps[0]["device"])
                roi = self._active_roi_for(parent.source_id)
                tags = _inherited_window(parent)
                if roi is not None:
                    # The result child runs on the active saved ROI --
                    # roi.window is data identity (resolve's cache fold) and what the engine's
                    # runner reads (+ the tool's own margin) off the file.
                    tags["roi.window"] = f"{roi.row},{roi.col},{roi.h},{roi.w}"
                name = (f"{label} @{roi.label}" if roi is not None
                        else f"{parent.name} · {label}")
                layer = self.project.add_layer(name, parent.source_id, chain,
                                               parent_id=parent.layer_id, tags=tags)
                if roi is not None:
                    layer.roi_id = roi.roi_id
                self.add_layer_row(layer, field)
                self.layer_list.select_layer(layer.layer_id)   # -> _select_layer -> worker

            QtCore.QTimer.singleShot(0, _spawn)
            self._notify("new analysis forked into its own layer — the current one is "
                         "untouched", "status")
            return
        if self._is_point_layer(self.layer):
            # Stamped BEFORE the candidate Chain is built, same discipline as the build-first rule this method's own docstring documents: everything here still
            # operates on LOCAL variables, so a `backproject` step with a bad target still warns
            # and stamps zero honestly, and the candidate Chain below is built from those stamped
            # (zeroed, if unbound) params -- never from a half-written mid-mutation state.
            self._stamp_backproject(names, params)
        chain = Chain(tuple(DeviceRef(names[i], dict(params[i]))
                            for i in range(len(names)) if not bypassed[i])).materialized()

        before = self._transform_signature()
        self._names, self._params, self._bypassed, self._rack = names, params, bypassed, rack
        self.layer.chain = chain
        self._snapshot_recipe()
        # Master-row collapse follows the chain: a first step on a lone master un-hides its
        # row; emptying it back re-collapses.
        self.layer_list.refresh_master_rows()
        # The Outputs group follows the chain too: the declared outputs of its last transform.
        self._sync_output_rows(self.layer)
        if self._transform_signature() != before:
            if self._auto_run_action.isChecked():
                self._pending_layers.discard(self.layer.layer_id)
                self._start_worker()
            else:
                self._mark_transforms_pending()
        else:
            self._reresolve()

    def _on_scale_changed(self, idx: int) -> None:
        """The transport writes ``scale_select``'s param and takes the filter path -- the same one
        the knob takes. Scrub, playback and the strip control cannot be allowed to diverge.

        The ``_syncing`` guard comes FIRST, ahead of the params write as well as the resolve.
        While ``_sync_transport`` is re-ranging the slider, this signal does not carry a user's
        intention -- it carries the transport's own stale position, echoed back out of
        ``set_n_scales``' clamp. Writing it into the params would discard whatever scale the chain
        had actually been built with: a window opened at scale 4 -- from a project file, a
        devloop rebuild, or DEMO_CHAIN -- would silently display scale 0 on its first frame.
        """
        if self._syncing:
            return
        i = self._index_of("scale_select")
        if i is None or self.layer is None:
            return
        # The zone being disabled does not stop this signal: the transport is a SECOND, keyboard-
        # and mouse-reachable write path into ``_params``/``layer.chain`` (Space, scrub) that never
        # goes through a strip control at all -- it needs its own lock/freeze refusal, before the
        # first write below, exactly as ``_on_param_changed``/``_on_chain_edited`` already have.
        notice = _lock_notice(self.layer)
        if notice is not None:
            self.strips._show_warning(notice)
            return
        self._params[i]["scale_idx"] = int(idx)
        self._load_level_filters(int(idx) + 1)
        self.layer.chain = self._chain()
        self._snapshot_recipe()
        self._reresolve()

    def _reresolve(self) -> None:
        """A filter moved and the screen needs a redraw.

        Normally that is the synchronous, cache-hit resolve this whole design exists to make
        possible. Two states make it the WRONG call, and both would run an uncached transform
        inline on the GUI thread:

        - a compute is in flight -- the finish handler re-resolves the CURRENT chain, so the
          change lands there instead;
        - the last transform RAISED, so its result was never cached. Recomputing it here would
          freeze the window on the very path that is supposed to be instant (and would do it
          inside a signal handler, where the exception has nowhere to go). It is the worker's
          job, exactly as a transform change is.

        "the finish handler re-resolves the CURRENT chain" above holds only when the busy thread
        is the active layer's own worker. A busy thread may belong to a BACKGROUND arrangement
        layer instead, whose landing never calls ``_resolve_now()`` for the active layer at all
        -- so the filter change this method is about would sit unapplied on screen until some
        UNRELATED later event happened to trigger a resolve, if ever. Recording the intent here
        (unless a stronger "compute" is already pending -- a transform edit already implies a
        fresh resolve once it lands) is what ``_dispatch_next`` consumes once the thread is
        actually free, closing that gap.

        Recording "resolve" is scoped to EXACTLY that gap -- a busy thread
        that belongs to a DIFFERENT (background/previously-active) layer, checked via
        ``self._dispatched``'s own layer id. When the busy thread is the ACTIVE layer's OWN
        worker, nothing is recorded here at all: that landing already re-resolves the CURRENT
        chain unconditionally on success (``_on_finished``'s active-match body, which reads
        ``self._params``/``self.field`` fresh, so this filter change is picked up there for
        free), or reports the error honestly on failure (``_on_error``'s active-match body) --
        recording "resolve" in EITHER of those cases would fire ``_dispatch_next``'s
        unconditional ``_resolve_now()`` a second time on success (a duplicate ``resolved``
        emission for one user action), or, worse, on an error landing where ``_errored`` is now
        ``True`` and nothing is cached for the raising chain -- exactly the synchronous,
        GUI-thread, uncached-transform hazard this method's own docstring forbids.

        An OUTPUT job of the active layer holding the slot redraws at its own landing
        (:meth:`_land_output`); a change of which output is on show, or of a knob that output
        reads, supersedes it here -- the job is cancelled and that landing dispatches the new one.
        Live turned off supersedes a preview job the same way (the row now waits for Run and
        draws no output key), while a Run's job keeps its key and runs on.
        An output job never recomputes an analysis, so while one holds the slot and the
        analysis tail this redraw reads is cached, the redraw runs now (a cache hit, the same
        synchronous resolve as with the slot free) instead of waiting for the job to land.
        """
        job = self._out_job
        if job is not None and job[0] is self.layer and self._worker is not None:
            if self._displayed_output_key(self.layer) != job[2]:
                self._worker.cancel()
        if self.layer is not None and self.layer.layer_id in self._pending_layers:
            # Manual-run pending: the live chain's NEW transform tail is
            # uncached by DESIGN, so resolving IT here would be the GUI-thread catch-up compute
            # this method's docstring forbids. But §5e also says "filter edits stay live against
            # the last run" -- so instead of dropping the edit (which leaves filters dead,
            # 'slaved to Run'), re-resolve the HYBRID: the last-run committed transforms
            # (cached -> no compute) + the CURRENT filters. Only once a run exists to hybridize
            # against, and only when the thread is free (an in-flight landing re-resolves anyway)
            # or holds an output job while the committed tail is cached.
            if self._committed_chain is not None and (
                    self._thread is None or self._tail_cached_during_output_job(
                        dataclasses.replace(self.layer, chain=self._committed_chain))):
                self._resolve_now(self._committed_chain)
            return
        if self._errored:
            self._start_worker()
        elif self._thread is None or self._tail_cached_during_output_job(self.layer):
            self._resolve_now()
        elif self._active_pending != "compute":
            active_id = self.layer.layer_id if self.layer is not None else None
            if self._dispatched is not None and self._dispatched[0] != active_id:
                self._active_pending = "resolve"

    def _tail_cached_during_output_job(self, layer) -> bool:
        """True while the worker slot holds an OUTPUT job and ``layer``'s last transform result
        is cached (or it has no transform), so resolving ``layer`` on the GUI thread computes
        nothing; the redraw path of :meth:`_reresolve` then need not wait for the landing."""
        if self._out_job is None or layer is None:
            return False
        keys = self._cache_keys_for(layer)
        return not keys or keys[-1] in self.cache

    def _resolve_now(self, committed_chain=None) -> None:
        """Synchronous resolve on the GUI thread -- filters only, by construction.

        ``committed_chain`` (§5e hybrid): when given, resolve the last-run transforms
        from it + the current filters instead of ``self.layer.chain`` -- the pending-state path,
        where the live chain's transform tail is deliberately uncached. ``None`` is the ordinary
        path (the whole current chain, whose transforms are cache hits).

        The guard is not decoration: this runs inside a Qt signal handler, where an escaping
        exception is printed and swallowed, leaving a window that has silently stopped redrawing.
        A filter that raises therefore lands in the same visible error path as a transform that
        raises (``devices/chain_filters.py`` documents the same hazard from its own side).

        **Error routing.** ``_on_param_changed``'s filter branch, ``_on_chain_edited`` and
        ``_on_scale_changed`` all route a filter edit through ``_reresolve`` to here, and NONE of
        them touches the arrangement on its own. Without a resync here, a filter tightened while
        flipped would leave the zone interactive against a session result the scene never
        re-rendered: the on-screen chain indices could shift (a filter admitting a different,
        equal-sized set) while the arrangement kept showing the OLD chains, and a pick made against
        that stale geometry could commit against the WRONG chain with no stale flag to catch it (an
        in-range index defeats ``group_paint``'s own out-of-range floor). This is the single choke
        point every synchronous filter-resolve success reaches, active layer or not (also covers
        ``_on_finished``'s own call here, and ``_dispatch_next``'s "resolve"-pending branch) -- one
        ``_sync_arrangement()`` call while flipped re-derives every visible layer's entry with the
        SAME positional-digest fingerprint (``scene.py``'s own) that already prunes stale
        selection/preview/palette membership honestly, so a picked-then-shifted chain is pruned
        before a commit naming it is even possible. See ``_sync_arrangement``'s own
        docstring for the reentrancy guard this can trip (``_dispatch_next``, called at THIS
        method's OTHER call sites, can call back into ``_resolve_now`` from inside an
        already-running ``_sync_arrangement``)."""
        layer = self.layer
        if committed_chain is not None:
            # §5e hybrid path: resolve the last-run transforms + the CURRENT filters against a
            # throwaway layer, so the committed transform tail is a cache HIT (no GUI-thread
            # compute) while the freshly-edited filters run live. Identity is preserved
            # (dataclasses.replace keeps layer_id/source_id/tags) so ``_apply``'s bookkeeping and
            # the arrangement resync below behave exactly as for an ordinary resolve.
            hybrid = self._hybrid_chain(committed_chain)
            layer = dataclasses.replace(self.layer, chain=hybrid)
        try:
            renderable = resolve(layer, self.field, self.cache,
                                 source_id=self.layer.source_id)
        except Exception as exc:
            self._report_error(str(exc))
            return
        self._apply(renderable)
        if self._center_stack.currentIndex() == 1:
            self._sync_arrangement()

    def _hybrid_chain(self, committed_chain) -> "Chain":
        """The §5e "last-run transform tail + live filter steps" chain: every ENABLED transform
        from ``committed_chain`` (the last successfully computed chain -- cached) followed by the
        window's CURRENT enabled filters. Transforms-then-filters is preserved by construction."""
        transforms = [ref for ref in committed_chain.steps
                      if is_transform(get_device(ref.device))]
        filters = [DeviceRef(self._names[i], dict(self._params[i]))
                   for i in range(len(self._names))
                   if not self._bypassed[i] and not is_transform(get_device(self._names[i]))]
        return Chain(tuple(transforms + filters)).materialized()

    def _apply(self, renderable) -> None:
        """The ONE place a result reaches the screen.

        A hidden ACTIVE layer (H checked, ``layer.visible`` False) never draws its overlays here
        -- not only at the instant ``_on_hide_toggled`` cleared them, but on EVERY redraw this
        method ever handles afterward: a filter edit, a scrub, a switch away and back onto the
        hidden layer, a worker that was already in flight when hide was toggled and lands only
        now. Without this gate any one of those repopulated ``canvas.set_result`` unconditionally
        and silently un-hid the layer the user had just hidden -- the checkbox stayed checked,
        the picture disagreed with it. The raster itself is untouched either way; only the WTMM
        overlay is gated (``layer_panel.py``'s v1-scope hide, ``Canvas.clear_overlays``).

        **Committed-group staleness surfaces here, for the ACTIVE layer only.**
        ``group_paint`` (``devices/groups.py``) flags ``result["_stale_groups"]`` when a
        committed group's member index no longer fits the current ``chains`` list, and
        ``result["_spec_error"]`` when ``spec_json`` itself could not be decoded (a hand-edited or
        cross-version project) -- both honest no-ops at the device layer (nothing raises, nothing
        is silently mis-painted), surfaced to the user right here, the one place every ACTIVE-
        layer result lands, the same ``strips._show_warning`` notice surface every other zone
        warning already uses.

        **A point layer's own ``backproject`` result.** Identified by ``"points_px" in
        result`` (the device's own contract -- no other device ever stamps that key), never by
        ``self._is_point_layer(self.layer)``: the CHECK here is really "does this result carry a
        point-registration reading", not "is the active layer a point layer" -- the two agree in
        practice, but the key is the honest thing to branch on. Shown only when the raster
        CURRENTLY on the canvas (``self._displayed_layer_name``, set only where
        ``canvas.set_field`` is -- selecting a point layer never touches the raster image, see
        ``_select_layer``'s own comment) IS the layer this result is registered against
        (``result["_target"]``); every other case -- unbound, or bound to a DIFFERENT raster than
        the one on screen -- clears the scatter rather than drawing it stale over the wrong image.
        """
        result = renderable.result
        if not isinstance(result, dict):
            # Field-producing chain tail (noise alone; see _on_finished's own guard):
            # nothing to overlay -- land quietly with the hint instead of crashing on result.get.
            self._active_result = {}
            self.canvas.clear_overlays()
            self._land_field_result(result)
            self.resolved.emit(renderable)
            return
        # The ACTIVE layer's own latest result, for _refresh_topology_panel's "chain
        # graph: N edges" summary (result.get("topology"), the chain_topology transform's own
        # product) -- stamped unconditionally, even for a hidden layer or a point-layer result
        # neither of which carries a "topology" key, so a stale count from a PREVIOUS layer never
        # lingers once this one has resolved at all.
        # A lazily computed output on show (the M-Z reconstruction) is part of what lands: a
        # cached one rides a copy of the result as its raster_out, so the display, the fork and
        # the row name all read it; a missing one is requested as a job.
        result = self._show_lazy_output(renderable, result)
        self._active_result = result
        self._refresh_topology_panel()
        self._refresh_skeleton_button()
        self._refresh_spectrum_button()
        self._refresh_dh_button()
        self._refresh_anisotropy()
        self._refresh_panel_relevance()
        self._refresh_components_button()
        self._sync_holder_raster(result)
        self._sync_output_rows(self.layer)
        # A landing while the Vector/Globe tab is up must reach the scene (a forked wtmm computed
        # with the Vector tab showing otherwise has its extrema absent until a manual
        # flip-out/in). _sync_arrangement's own diffing keeps repeat landings cheap
        # (the filter-only fast path), same as the visibility-toggle resync just above it.
        if self._center_stack.currentIndex() == 1 and self._arrangement is not None:
            self._sync_arrangement()
        # Honesty notice: extrema drawn while the DATASET raster is hidden
        # (the source row's H) reads as "I can see the extrema but not the DEM" -- say so.
        src = self.project.sources.get(self.layer.source_id) if self.layer is not None else None
        if src is not None and src.hidden and isinstance(result, dict) \
                and result.get("extrema") and self._center_stack.currentIndex() == 0:
            self._notify("dataset raster is hidden — extrema are drawn over empty ground "
                         "(H on the source row un-hides it)", "status")
        stale = result.get("_stale_groups")
        if stale:
            self.strips._show_warning(f"stale group(s): {', '.join(stale)}")
        spec_error = result.get("_spec_error")
        if spec_error:
            self.strips._show_warning(f"group spec error: {spec_error}")
        # The edges output row's gate, set on every landing before any overlay draws: the
        # maxima of a layer whose "edges" row is hidden stay off the canvas across redraws,
        # layer switches and flips back from the Vector tab (the canvas keeps the flag).
        self.canvas.set_maxima_visible(self.layer is None or not _edges_hidden(self.layer))
        if self.layer is not None and not self.layer.visible:
            self.canvas.clear_overlays()
            self.canvas.clear_points()
            self.canvas.set_pick_chains(None)      # Nothing on screen left to pick
        elif "points_px" in result:
            if (result["points_px"] is not None
                    and result.get("_target") == self._displayed_layer_name):
                self.canvas.set_points_result(result)
            else:
                self.canvas.clear_points()
            # Stale-pick corruption vector: a point layer's own
            # result never touches `set_result`, so whatever WTMM chains a PREVIOUS layer left on
            # the canvas are still sitting in `_pick_chains` here -- a click would silently pick
            # against them and hand `GroupPalette.add_pick` a `(NEW layer_id, OLD chain_index)`
            # pair that names nothing real. Clearing makes a click an honest miss instead. The
            # underlying overlay itself staying visually stale (the extrema/trail ITEMS, not
            # picking) is a separate, pre-existing display gap this does not touch.
            self.canvas.set_pick_chains(None)
        else:
            self.canvas.clear_points()
            if "extrema" in result and "_shape" in result and result["extrema"]:
                layers = result["extrema"]
                idx = 0 if len(layers) == 1 else int(result.get("_scale_idx", 0))
                # Double-update guard: when the 3-D view is the one on screen, the raster canvas
                # is hidden -- redrawing its (heavy) overlay every tweak is pure waste on top of
                # the scene rebuild. Skip it and mark the canvas dirty; _set_center_view's
                # flip-back-to-raster branch redraws it then, from the same _active_result. When
                # the canvas IS on screen, draw it now.
                if self._center_stack.currentIndex() == 0:
                    self.canvas.set_result(result, min(idx, len(layers) - 1))
                    self._canvas_overlay_dirty = False
                    # Draw cap honesty: when the canvas withheld chains
                    # to stay responsive, SAY SO -- a silent truncation reads as "that's all there
                    # is", which is precisely the lie the cap must never tell.
                    if self.canvas.cap_note:
                        self._notify(self.canvas.cap_note, "status")
                else:
                    self._canvas_overlay_dirty = True
                # The SAME chains list just drawn is what a click
                # picks against -- pushed at this one call site, the only place a redraw actually
                # lands a fresh (or unchanged) `chains` list on the canvas.
                #
                # `set_result` draws an ROI result's trail SHIFTED by
                # `display_offset` (the ROI's own on-screen origin -- often large, and the
                # DEFAULT outcome for the whole ROI/refined-run workflow, not an edge case), while
                # `result["chains"]` itself stays in the result's own un-shifted frame. Shifting
                # here, once, keeps a click aligned with what is actually drawn. The non-ROI case
                # (offset always (0, 0)) short-circuits to the ORIGINAL list object -- an
                # unchanged redraw is then still an identity match for anything that compares
                # chains by identity (this canvas's own `_cached_geometry`), not a churned copy.
                chains = result.get("chains")
                row_off, col_off = display_offset(result, self.field)
                if chains and (row_off or col_off):
                    chains = _shift_chains(chains, row_off, col_off)
                self.canvas.set_pick_chains(chains)
            elif self._center_stack.currentIndex() == 0:
                # Nothing to draw -- so nothing may stay drawn. The canvas otherwise keeps the
                # PREVIOUS result's overlay: removing a result selects its parent, whose empty
                # chain carries no extrema, and the removed layer's extrema stayed up over it.
                self.canvas.clear_overlays()
                self.canvas.set_pick_chains(None)
            else:
                self._canvas_overlay_dirty = True
        self._sync_controls()
        self._update_readings(renderable)
        self._update_scale_bar()
        self.resolved.emit(renderable)

    def _sync_controls(self) -> None:
        """Refresh every strip control from ``_params``, wholesale. The strips close the propose/
        confirm loop for their OWN gestures; this is what keeps a control honest when something
        else moved its param -- the transport writing ``scale_idx``, most of all.

        Reads ``self._params`` rather than ``self.layer.chain.steps``: the chain excludes bypassed
        steps (``_chain()``'s "enabled steps only"), so its index space no longer lines up with
        ``self.strips.strip(i)`` once anything is bypassed -- ``_params`` stays parallel to every
        strip, bypassed or not, by construction.
        """
        for i, params in enumerate(self._params):
            strip = self.strips.strip(i)
            for name, value in params.items():
                control = strip.controls.get(name)
                if control is not None:
                    control.set_value(value)

    def _update_readings(self, renderable) -> None:
        """Every strip's state and reading, from the result that just landed.

        A bypassed step contributed nothing to this result, so its reading is cleared rather than
        left showing whatever it last (honestly) reported before being bypassed -- a residual shipped deliberately, since bypass was not yet real for it to contradict.

        An ROI transform's reading is its MARGIN honesty (:func:`_roi_margin_reading`) rather than
        the elapsed time, and it is set here -- after ``_on_finished`` has already written the
        elapsed ms -- deliberately: how much of this answer stands on real parent data is a
        property OF THIS RESULT, and it outlives the run that produced it (a cached ROI redisplays
        the same margins; a cached run has no elapsed time to state). Every other transform, and
        an ROI result from before this key existed, keeps the elapsed reading untouched.
        """
        state = "cached" if renderable.from_cache else "idle"
        enabled = [i for i in range(len(self._names)) if not self._bypassed[i]]
        terminal = enabled[-1] if enabled else None
        for i, name in enumerate(self._names):
            strip = self.strips.strip(i)
            if self._bypassed[i]:
                strip.set_reading("")
                continue
            if is_transform(get_device(name)):
                strip.set_state(state)
                margins = renderable.result.get("_roi_margins") if name == "wtmm2d_roi" else None
                if margins:
                    strip.set_reading(_roi_margin_reading(margins))
            else:
                device = get_device(name)
                if hasattr(device, "reading") or hasattr(device, "data_hints"):
                    # A device
                    # that declares either protocol method owns its own reading/hints from here
                    # on -- see DeviceBox.sync_from_result (workflow_zone.py) for the full
                    # contract (soft-bound rebind, the outside-range snap, the NaN-unavailable
                    # sentinel). This REPLACES the generic fallback below for that strip, never
                    # supplements it -- a device opting in is claiming the whole reading line.
                    strip.sync_from_result(renderable.result)
                else:
                    strip.set_reading(_filter_reading(name, renderable.result, i == terminal))

    # -- the worker ------------------------------------------------------------------------
    def _start_worker(self) -> None:
        if self.layer is None:
            # The active layer was removed while a worker for it was still in flight (the
            # zombie-active-layer fix, ``_reset_to_empty``): ``_on_finished`` sees the landing
            # was not for the (now nonexistent) active layer and calls back in here to "run the
            # current one" -- except there IS no current one any more. Nothing to dispatch, and
            # nothing to touch: the transport is already disabled exactly as ``_reset_to_empty``
            # left it; re-entering the dispatch below would dereference ``self.layer.source_id``
            # on ``None``.
            return
        self._start_worker_for(self.layer)

    def _transform_signature_for(self, layer) -> tuple:
        """Same shape as :meth:`_transform_signature`, derived from ``layer.chain.steps``
        directly -- the persisted, already-bypass-filtered chain -- rather than the window's own
        live ``_names``/``_params`` working copy, which only ever reflects the ACTIVE layer's
        current editing session. For the active layer the two agree by construction (``_chain()``
        is what wrote ``layer.chain`` in the first place); this is the form a background
        arrangement-queue layer -- one nobody is live-editing -- needs instead."""
        return (layer.source_id,
                tuple((ref.device, tuple(sorted(
                    keyed_params(get_device(ref.device), ref.params).items())))
                      for ref in layer.chain.steps if is_transform(get_device(ref.device))))

    def _start_worker_for(self, layer) -> None:
        """Dispatch ``layer``'s resolve on the ONE shared worker thread. ``_start_worker()`` is
        exactly ``_start_worker_for(self.layer)`` -- unchanged in every observable respect
        (transport/strip UI, no-cancel semantics, the signature-mismatch redispatch that
        ``_on_finished`` performs). A background arrangement-queue layer (``layer is not
        self.layer``) takes the identical dispatch mechanics but never touches session UI: it has
        no strips of its own, and disabling the transport for a compute the user cannot even see
        would be wrong -- ``_dispatch_next`` is what decides whether the active layer or a queued
        layer gets the thread next.

        The ``field is None`` bail-out is checked BEFORE anything touches the transport --
        checking it after would leave a disabled transport with no compute ever dispatched to
        re-enable it.
        """
        is_active = layer is self.layer
        field = self.field if is_active else self._fields.get(layer.layer_id)
        if field is None:
            return
        if is_active:
            # A new analysis supersedes an output job waiting for the slot: its own landing
            # requests whatever output the layer shows then.
            self._drop_output_request()
        try:
            # A bus's LIVE layer sends: fresh stamps, then the layers to compute first.
            self._refresh_bus_stamps(layer)
            prelude = self._bus_plan(layer)
        except ValueError as exc:
            if is_active:
                self._report_error(str(exc))
            return
        if is_active:
            # The scale stack is being rebuilt, so there is no honest answer to a scrub: the
            # canvas still holds the last good frame, the slider would move to an index the new
            # stack may not even have, and v1 cannot cancel the run to catch up. Three surfaces
            # claiming three different scales is worse than one control that is visibly
            # unavailable for a moment. Re-enabled in _on_finished and _on_error -- both paths,
            # or an error leaves it dead.
            self.transport.setEnabled(False)
        if self._thread is not None:
            # If it's the active layer trying to preempt, mark the intent in ITS OWN UI *and*
            # record it in ``self._active_pending`` -- the signature-mismatch redispatch only
            # fires for the dispatched layer's OWN later landing, but a landing can belong to an
            # unrelated (background, or previously active) layer entirely, whose routing in
            # ``_on_finished`` never even looks at ``self._transform_signature()``. Dropping the
            # intent here would leave a layer SWITCH mid-compute to an already-cached layer never
            # redispatched -- ``_dispatch_next``'s cache probe sees a hit and does nothing,
            # wedging the transport/strips/canvas indefinitely. ``_dispatch_next`` consumes this
            # unconditionally, regardless of what the cache probe would have said on its own.
            if is_active:
                self._set_transform_states("computing")
                self._active_pending = "compute"
                # Progressive compute (progressive-compute design §3): the active layer is
                # preempting with a fresh recipe -- KILL the in-flight run instead of waiting it
                # out. The old worker checks the flag at its next WTMM stage boundary, raises
                # ComputeCancelled, and lands in _on_cancelled, which dispatches this pending
                # "compute", so a preempted run is never computed to the end only to be
                # discarded. Only the ACTIVE layer's own preempt cancels; a background
                # arrangement compute is left to finish (its result is still wanted).
                if self._worker is not None:
                    self._worker.cancel()
            return
        if is_active:
            # About to be serviced by THIS dispatch -- clear whatever was pending (set while the
            # thread was busy, possibly for a stale, earlier intent) so a later landing's
            # ``_dispatch_next`` doesn't redundantly redispatch on a leftover flag.
            self._active_pending = None
        self._dispatched = (layer.layer_id,
                            self._transform_signature() if is_active
                            else self._transform_signature_for(layer))
        if is_active:
            self._dispatched_chain = layer.chain      # §5e hybrid: promoted on a clean landing
        self._t0 = time.perf_counter()
        if is_active:
            self._set_transform_states("computing")
        # Finest-scale preview (progressive-compute design §3) only for the ACTIVE layer's own
        # compute -- the one whose canvas the user is waiting on. A background
        # arrangement-queue layer gets no preview (nobody is staring at a blank canvas for it).
        worker = ResolveWorker(layer, field, self.cache, layer.source_id, preview=is_active,
                               prelude=prelude)
        # BOUND METHODS ONLY (worker.py's documented trap): Qt can only place a call on the GUI
        # thread if the receiver is a QObject it can ask for a thread affinity.
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.cancelled.connect(self._on_cancelled)
        worker.partial.connect(self._on_preview)
        self._worker = worker
        self._thread = worker.start()
        self._stopped_sig = None              # an explicit dispatch reopens the stop gate
        self._stop_btn.setVisible(True)

    def _set_transform_states(self, state: str) -> None:
        if self.strips is None:
            return
        for i in self._transform_indices():
            if i < len(getattr(self.strips, "_boxes", ())):   # zone mid-rebuild: skip, never throw
                self.strips.strip(i).set_state(state)

    def _set_compute_reading(self, text: str) -> None:
        """Progress and elapsed belong to the FIRST transform strip, and only that one.

        There is one worker, one ``resolve``, one wall clock. A chain with two transforms
        (``wtmm2d`` then ``chain_topology``, as DEMO_CHAIN has) put ``cwt 40%`` and then
        ``820 ms`` on BOTH strips -- each device claiming the whole run as its own cost, which
        reads as two computes and misattributes the time. The staged progress the worker reports
        is not per-device either; it is stages of the one run.

        The state DOT stays on every transform strip: that one is genuinely per-device -- it says
        whether this step's result is in the cache -- so it is not a claim about who ran.
        """
        if self.strips is None:
            return
        for n, i in enumerate(self._transform_indices()):
            self.strips.strip(i).set_reading(text if n == 0 else "")

    def _stop_compute(self) -> None:
        """The status-bar Stop: ask the in-flight worker to abandon at its next stage
        boundary (ResolveWorker.cancel -- the same flag the preempt path sets), and mark
        the landing as a USER stop so _on_cancelled restores the UI instead of
        redispatching. Safe no-op with nothing in flight."""
        if self._worker is None:
            return
        self._user_stopped = True
        self._worker.cancel()

    def _teardown_thread(self) -> None:
        thread, self._thread, self._worker = self._thread, None, None
        if hasattr(self, "_stop_btn"):
            self._stop_btn.setVisible(False)
        if thread is not None:
            thread.quit()
            thread.wait()

    def _on_progress(self, stage: str, frac: float) -> None:
        self._stage = stage
        active_id = self.layer.layer_id if self.layer is not None else None
        if self._dispatched is not None and self._dispatched[0] == active_id:
            # The strip readings belong to the ACTIVE layer's own zone -- a background
            # arrangement-queue compute has no strips to report progress on, and touching
            # ``self.strips`` for one would misattribute its progress to whatever is on screen.
            self._set_compute_reading(f"{stage} {frac:.0%}")
        if self._closing:
            self._show_waiting_title()

    def _on_finished(self, renderable) -> None:
        self._teardown_thread()
        # A Stop the run finished through (its transform never reads the cancel flag) is spent
        # here; left set, the next preempt's cancel would land as a user stop and drop its run.
        self._user_stopped = False
        dispatched_layer_id, dispatched_sig = self._dispatched
        active_id = self.layer.layer_id if self.layer is not None else None
        if dispatched_layer_id != active_id:
            # A background arrangement-queue layer landed (or the active layer changed identity
            # since dispatch -- the same signal, since a background dispatch's layer_id can never
            # equal whatever is active NOW unless the user selected it in the meantime, in which
            # case the branch below is the one that runs instead). Never the session canvas's
            # business: ``_land_after_worker`` resyncs the arrangement, which reads this layer's
            # now-populated cache entry as an honest "ok" (or, on the error path, is never
            # reached here at all -- see ``_on_error``).
            self._land_after_worker()
            return
        if self._transform_signature() != dispatched_sig:
            # The params moved mid-compute (v1 cannot cancel), so this result is for a recipe
            # nobody is asking for any more. Run the current one; a pending close waits that out
            # too, which is longer but still honest -- it never refuses and never pretends.
            self._start_worker()
            return
        # A real result is back -- scrubbing means something again -- UNLESS the active layer is
        # locked/frozen, in which case the transport must stay exactly as read-only as the zone.
        # A blind ``setEnabled(True)`` here would silently undo a lock that took effect (or was
        # already in effect) while this very compute was running: nothing else re-asserts it
        # once a worker's own finish handler has blown it away.
        if self.layer is not None:
            self._sync_lock_ui(self.layer)
        else:
            self.transport.setEnabled(True)
        self._errored = False                # the transform is in the cache again
        # §5e hybrid: this chain's transform tail is now cached, so a later pending filter edit
        # can re-resolve against it (committed transforms + live filters) without a recompute.
        self._committed_chain = self._dispatched_chain
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self._set_compute_reading(f"{elapsed_ms:.0f} ms")
        if not isinstance(renderable.result, dict):
            # A chain ending on a field-producing transform (noise) resolves to a FIELD, not a
            # result dict. result.get on a RasterField would crash right here inside a
            # worker-signal slot -- Qt swallows the traceback, `resolved` never fires, and the
            # window stays wedged (stuck at 0). Land quietly: nothing to overlay, a hint says
            # what to add.
            self._scales = ()
            self.canvas.clear_overlays()
            self._land_field_result(renderable.result)
            self.resolved.emit(renderable)
            self._land_after_worker()
            return
        roi_info = renderable.result.get("_roi") or {}
        if roi_info.get("margin_declared") is False:
            # An outside plugin that declares no roi_margin still runs on
            # the region -- on the ROI alone -- and the user is told what that costs.
            self._notify("this tool declares no ROI margin — it ran on the ROI alone, so the "
                         "ROI border carries ordinary edge effects", "status")
        # `scales` is a numpy ARRAY from the real transform, so no truthiness test may touch it --
        # `x or ()` on an ndarray raises, and it raised here inside a worker-signal slot, where Qt
        # prints the traceback and carries on as if nothing happened.
        scales = renderable.result.get("scales")
        self._scales = () if scales is None else tuple(float(s) for s in scales)
        if self._scales:
            self._sync_transport(len(self._scales))
        self._resolve_now()                  # guaranteed cache hit; see the module docstring
        self._land_after_worker()

    def _on_error(self, message: str) -> None:
        self._teardown_thread()
        self._user_stopped = False           # a Stop the run failed through is spent (_on_finished)
        dispatched_layer_id, _ = self._dispatched
        active_id = self.layer.layer_id if self.layer is not None else None
        if dispatched_layer_id != active_id:
            # A background arrangement-queue layer's own compute raised -- lists with the error,
            # unplaced, never touching the active layer's strips (which belong to a
            # DIFFERENT layer's chain entirely).
            layer = self._layer_by_id.get(dispatched_layer_id)
            if layer is not None:
                self._arr_errors[dispatched_layer_id] = (self._arr_tail_key(layer), f"error:{message}")
            self._land_after_worker()
            return
        # A failed run must not leave the window half-dead -- same lock-respecting re-enable as
        # ``_on_finished``, for the same reason: a blind ``setEnabled(True)`` would undo a lock.
        # No ``else: self.transport.setEnabled(True)`` here (mirrors the guard
        # added to ``_start_worker``): the only way this fires with ``self.layer is None`` is the
        # active layer having been removed while this very compute was in flight
        # (``_reset_to_empty``), and that already left the transport disabled on purpose -- with
        # no layer at all, there is nothing left for it to mean. A blind re-enable would silently
        # undo that, the exact class of bug the lock-respecting branch above exists to prevent.
        if self.layer is not None:
            self._sync_lock_ui(self.layer)
        self._report_error(message)
        self._land_after_worker()

    def _on_preview(self, result) -> None:
        """A finest-scale preview landed (design §3): draw it on the canvas NOW, so the user sees
        the finest scale's lines while the full stack is still computing. Transient -- it does not
        become ``self._active_result`` (the full ``_on_finished`` owns that) and is skipped when
        the canvas is not the view on screen (the full result will populate the 3-D scene when it
        lands). A preview for a layer that is no longer active, or is hidden, is dropped."""
        if not isinstance(result, dict) or not result.get("extrema") or "_shape" not in result:
            return
        if self._center_stack.currentIndex() != 0:
            return                                       # canvas not visible; wait for the full
        if self.layer is None or not self.layer.visible:
            return
        layers = result["extrema"]
        idx = 0 if len(layers) == 1 else int(result.get("_scale_idx", 0))
        self.canvas.set_result(result, min(idx, len(layers) - 1))
        self._notify("preview (finest scale) — computing full stack…", "status")

    def _on_cancelled(self) -> None:
        """A worker abandoned its run at our request (progressive compute, design §3).

        Not an error and not a result: tear the thread down and let ``_land_after_worker`` ->
        ``_dispatch_next`` service whatever is pending -- which, for the active-layer preempt that
        triggered the cancel, is the ``_active_pending = "compute"`` recorded in
        ``_start_worker_for``, i.e. the fresh recipe. The transport/strips were already put in the
        "computing" posture by that same preempt branch, so there is nothing to re-enable here;
        the redispatched run owns them through to its own landing. No ``_report_error`` (a
        cancellation is deliberate), no lock UI churn (the preempt did not change lock state)."""
        self._teardown_thread()
        if self._user_stopped:
            # USER stop, not a preempt: restore the UI and stand down --
            # never redispatch, and suppress the cache-probe redispatch for this recipe.
            self._user_stopped = False
            dispatched_layer_id = self._dispatched[0] if self._dispatched else None
            active_id = self.layer.layer_id if self.layer is not None else None
            if dispatched_layer_id == active_id and self.layer is not None:
                self._active_pending = None          # stop outranks a pending preempt
                self._stopped_sig = self._transform_signature()
                self._sync_lock_ui(self.layer)
                self._set_transform_states("idle")   # honest: the tail is NOT cached
                self._set_compute_reading("stopped")
                self._mark_stopped_pending()
            elif dispatched_layer_id is not None:
                layer = self._layer_by_id.get(dispatched_layer_id)
                if layer is not None:
                    # mirror _on_error's honesty stamp so the arrangement reports
                    # "stopped" instead of silently retrying forever
                    self._arr_errors[dispatched_layer_id] = (self._arr_tail_key(layer),
                                                             "stopped")
            self._land_after_worker()
            return
        self._land_after_worker()

    def _land_after_worker(self) -> None:
        """Whatever the shared worker thread should do next, now that it is free.

        Resyncs the arrangement (which itself dispatches the active layer's own pending need
        first, then the next queued layer -- see ``_dispatch_next``) ONLY while the arrangement is
        the current center view. This used to resync unconditionally
        whenever ``self._arrangement is not None`` -- meaning EVERY ordinary session landing, for
        as long as the window has ever pressed Tab once, paid a full ``set_layers`` ->
        ``Scene._rebuild`` on the (possibly long since parked) arrangement, plus a ``resolve()``
        per visible layer including a redundant re-resolve of the active layer itself. Flipped
        away, a landing still services the active layer / drains the queue via ``_dispatch_next``
        below (never skipped -- only the arrangement's OWN entries go stale until the next
        flip-in, which ``_toggle_center_view`` already resyncs unconditionally the moment the
        arrangement becomes visible again -- "flipping away drains nothing" from the spec, now
        also "and repaints nothing nobody can see").

        Either way, finally checks whether a close was waiting on this compute -- unchanged, just factored out so both the active-layer landing and the arrangement
        landing reach it.

        A session that never touches the arrangement pays nothing extra here ("Modularity": the app runs identically with the arrangement view never instantiated) --
        ``self._arrangement`` stays ``None`` forever in that case, so this is exactly
        ``_dispatch_next()`` plus the closing check, same as the code it replaced.
        """
        if self._arrangement is not None and self._center_stack.currentIndex() == 1:
            self._sync_arrangement()
        else:
            self._dispatch_next()
        # An output job the active layer's display waits for runs after any analysis.
        self._dispatch_output()
        if self._closing:
            self.close()

    def _dispatch_next(self) -> None:
        """Whoever needs the ONE shared worker thread most, dispatched if the thread is free.

        Priority order, per call: (0) whatever ``self._active_pending`` records -- consumed
        UNCONDITIONALLY, before the cache probe below ever runs (that
        attribute's own comment in ``__init__`` for why "cache hit -> nothing to do" is not
        always true); (1) the ACTIVE layer, if its own current chain is a genuine cache miss and
        it hasn't already errored -- a knob turn must stay responsive even while the arrangement
        queue is mid-drain, so it always jumps the queue; (2) the next arrangement-queue layer,
        but ONLY while the arrangement is the current center view (spec: "flipping away drains
        nothing" -- an already-dispatched compute still lands harmlessly, but flipping away
        starts no NEW background work). Self-healing against staleness: a popped queue entry that
        has since been removed, hidden, become the active layer, or already resolved by some
        other path is silently skipped rather than redispatched.

        A no-op while the thread is busy (the current compute's own landing will call this again)
        or while the window is closing (no cancellation in v1, but there is no reason to start
        MORE background work nobody will ever see once a close is already waiting).
        """
        if self._thread is not None or self._closing:
            return
        pending, self._active_pending = self._active_pending, None
        if pending == "compute":
            # Unconditional, deliberately: this is exactly what a plain layer SWITCH mid-compute
            # records (``_start_worker_for``'s busy branch runs on every activation, cache hit or
            # not), and pre-Task-6 the equivalent redispatch was ALSO unconditional -- unlike the
            # cache-probe check below, it must run even when the active layer turns out to
            # already be cached, because what actually needs re-doing is the UI (transport
            # re-enable, strip state, canvas apply), not the compute itself.
            self._start_worker()
            return
        if pending == "resolve":
            self._resolve_now()
        if (self.layer is not None and not self._errored
                and self._transform_signature() != self._stopped_sig
                and self.layer.layer_id not in self._pending_layers):
            keys = self._cache_keys_for(self.layer)
            if keys and keys[-1] not in self.cache:
                self._start_worker()
                return
        if self._center_stack.currentIndex() != 1:
            return
        # The output job the active layer's display waits for goes before background layers.
        self._dispatch_output()
        if self._thread is not None:
            return
        active_id = self.layer.layer_id if self.layer is not None else None
        while self._arr_queue:
            layer_id = self._arr_queue.pop(0)
            layer = self._layer_by_id.get(layer_id)
            if layer is None or not layer.visible or layer_id == active_id \
                    or layer_id in self._pending_layers:
                continue
            if self._fields.get(layer_id) is None:
                continue
            tail = self._arr_tail_key(layer)
            if tail is not None and tail in self.cache:
                continue
            self._start_worker_for(layer)
            return

    def _sync_arrangement(self, frame_mode: bool | None = None) -> None:
        """Rebuild the arrangement's entries from every visible layer, across every source,
        cache-first ("Multi-layer resolve"): a layer whose transform tail is already
        cached resolves synchronously right here -- cheap, by the cache law, since a hit never
        runs ``compute()`` -- and reports "ok" with that result; a layer with no CRS in its
        field's provenance never reaches the cache probe at all and reports "no-georeference"; a
        genuine cache miss is queued for the shared worker thread and reports "computing" until
        its own landing updates it; a layer whose last attempt at THIS SAME chain errored reports
        that error instead of being silently retried forever.

        Called on flip-in (``_set_center_view``, the ``_toggle_center_view`` shim's own delegate)
        and on ``hideToggled``/``layerSelected``
        while already flipped -- the events that can change which layers are visible, which one
        is active, or what any of them are cached as -- and from ``_land_after_worker`` once the
        arrangement has been built at least once, so a landing keeps it fresh too.

        Never touches the SESSION canvas: the only side effects here are
        ``self._arrangement.set_layers(...)`` and (via ``_dispatch_next``, called at the end)
        dispatching the shared worker thread's next job.

        **Signature.** Every entry also carries ``"signature":
        self._transform_signature_for(layer)`` -- ``Scene.set_layers``'s own staleness prune reads
        it (alongside the entry's chain count) to tell "this layer's transform actually re-ran"
        from "the SAME computation, freshly re-filtered into a new dict object" (every ``Filter``
        always copies -- ``dynamix/engine/resolve.py``'s own module docstring -- so a repeated
        sync of an UNCHANGED layer mints a new ``result`` identity on every call, which used to
        read as staleness and silently wipe live exploration/commit state for no reason).

        **Reentrancy guard.** ``_resolve_now`` (the filter path's own
        synchronous resolve) now resyncs the arrangement itself, at its own tail, whenever flipped
        -- and this method's OWN tail (``_dispatch_next``) can call ``_resolve_now`` right back (its
        "resolve"-pending branch), which would otherwise nest a SECOND, full ``_sync_arrangement``
        call inside the first one's still-running loop, rebuilding ``self._arr_queue``/the
        arrangement's entries out from under it. ``self._arr_syncing`` makes the inner call a
        no-op: the outer call is already about to (re-)apply the current state, including whatever
        that nested ``_resolve_now`` just applied to the active layer (a plain filter resolve, which
        never itself changes what THIS method reads), so skipping the nested rebuild loses nothing.

        **Display preferences.** Every entry also carries ``"colormap"`` and
        ``"vtrail_color"``, read off the layer's own ``ui.*`` tags through :func:`_display_style_of`
        -- the same style dict :meth:`_apply_display_style` pushes onto the session canvas, so the
        arrangement drape and a layer's own session view can never disagree about which colormap
        or trail color it prefers. ``vtrail_color`` is converted from the tag's hex string to an
        ``(r, g, b)`` tuple HERE, at the window boundary: ``Scene`` stays Qt-free and every color
        it already knows how to draw (``VTRAIL_COLOR``, ``SEAM_COLOR``, ...) is a plain tuple, not
        a hex string. Carried on every status, not only "ok" -- cheap (one dict read, no I/O) and
        one fewer branch to keep in sync with the branches below.

        **Frame mode.** ``frame_mode`` selects which of two
        DISJOINT admission rules governs the loop below -- ``False`` (the default) is the ORIGINAL
        geo rule, untouched: a layer with no CRS in its field's provenance is listed as
        "no-georeference" rather than resolved. ``True`` is the Vector tab's own rule instead:
        the ``has_georeference`` gate is skipped entirely (the whole point of a native-frame
        display is that CRS is irrelevant to it -- ``ArrangementView.set_frame_mode``'s own
        docstring), and in its place every RASTER layer other than the active one is admitted
        only when BOTH :func:`~dynamix.core.frames.frames_compatible` says its ``field.frame``
        matches the active layer's own AND its ``field.values.shape`` equals the active field's
        own grid shape (AC3, the design's "same grid shape for raster layers" clause -- frame
        equality alone is not enough: two bare rasters both carry the identical default
        ``LocalFrame(dx=1, dy=1, units=px)``, so without the shape check two unrelated grids of
        different sizes would overlay each other) -- excluded ENTIRELY otherwise (no entry at
        all, not even a legend line: the legend should read as "the layers that belong here", not
        as a growing list of exclusions). The active layer is always admitted, whatever its own
        frame -- there being nothing else to be "compatible" with when IT is what everything else
        is compared against.
        A point layer never goes through that comparison at all (points carry no
        :class:`~dynamix.core.frames.CoordinateFrame` of their own -- they are WGS84-placed by
        contract, ``Scene``'s own module docstring) -- in frame mode every visible point layer is
        admitted with status ``"frame-points"`` instead of ``"ok"`` (``Scene._rebuild`` reads this
        directly rather than the raster/vector placement math, and renders the legend line
        "session view only (frame mode)" for it).

        Vector-tab placement trusts ``field.frame`` exactly as stored, with one known caveat:
        ``RasterField._from_geotiff`` (``rasterfield.py:398``, the plain-``.tif`` loader) stamps
        ``GeographicFrame()`` even for a source whose CRS is actually projected, so frame-mode
        admission/placement for such a field can misplace coordinates. Not special-cased here --
        recorded for a follow-up fix, per the shell three-view plan.

        ``frame_mode=None`` (the default) means "whichever rule the LAST real call actually
        used" -- resolved here from ``self._arr_frame_mode`` (set at the end of this same block,
        every time). This is deliberately NOT a plain ``bool = False`` default: every resync call
        site that does not itself know "vector or geo" right now -- a hide-toggle, a layer
        switch, a display-style edit, a landed background compute, a group commit -- calls this
        method with NO argument at all (several of those call sites are monkeypatched with
        zero-argument fakes by the arrangement-view test suite -- see
        ``tests/test_arrangement_commit.py``'s and ``tests/test_arrangement_resolve.py``'s own
        ``_sync_arrangement`` spies), so a plain ``False`` default would silently re-admit every
        layer under the geo (``has_georeference``) gate the instant the Vector tab is showing and
        one of those call sites fires. The two call sites that actually KNOW which view is being
        entered (``_set_center_view``'s ``"vector"``/``"geo"`` branches) pass an explicit
        ``True``/``False`` instead, which is what seeds ``self._arr_frame_mode`` for every later
        argument-less call to reuse.
        """
        if self._arr_syncing:
            return
        if frame_mode is None:
            frame_mode = self._arr_frame_mode
        self._arr_syncing = True
        self._arr_frame_mode = frame_mode
        try:
            entries = []
            queue: list[int] = []
            # Only consulted in frame mode (below); a cheap dict lookup + attribute read, so
            # computing it unconditionally costs the geo path nothing and keeps this one variable
            # the single place "the active layer's own frame" is derived.
            active_field = self._fields.get(self.layer.layer_id) if self.layer is not None else None
            active_frame = getattr(active_field, "frame", None)
            # AC3: the shape half of the vector admission gate, computed once alongside
            # ``active_frame`` above. ``getattr(..., "values", None)`` first -- a bare ndarray
            # sibling (same guard as ``layer_frame`` below) has no ``.values`` at all, so this is
            # ``None`` for it, not an ``AttributeError``.
            active_shape = getattr(getattr(active_field, "values", None), "shape", None)
            for layer in self.project.layers:
                src = self.project.sources.get(layer.source_id)
                # roi.window children are exempt from the source hide --
                # their crop is the ROI gesture's own product, not 'the dataset'.
                show_raster = (bool(layer.tags.get("roi.window"))
                               or not (src is not None and src.hidden))
                if not layer.visible:
                    continue
                field = self._fields.get(layer.layer_id)
                if field is None:
                    continue
                signature = self._transform_signature_for(layer)
                style = _display_style_of(layer)
                colormap = style["colormap"]
                hillshade = (style["hillshade"], style["sun_azimuth"],
                             style["sun_altitude"], style["z_factor"])
                stretch = (style["stretch"], style["stretch_pct"])
                levels = (style["levels"], style["levels_colors"], style["levels_sieve"])
                surface = (style["surface"], style["depth_positive"])
                # Surface HEIGHT source: resolve ui.surface_source to that layer's
                # loaded field, admitted only on an identical grid shape -- never resampled.
                # A stale id (source layer since removed or reshaped) falls back silently to
                # this layer's own values; the dialog only ever OFFERS same-shape sources, so
                # the stale case is the rare leftover, not a flow.
                surface_field = None
                if surface[0] and style["surface_source"] != "same":
                    cand = self._resolve_surface_source(style["surface_source"])
                    own_vals = getattr(field, "values", field)
                    cand_shape = getattr(getattr(cand, "values", None), "shape", None)
                    if cand_shape is not None \
                            and cand_shape[:2] == getattr(own_vals, "shape", ())[:2]:
                        surface_field = cand
                vtrail_color = _hex_to_rgb(style["color_vtrail"])
                if self._is_point_layer(layer):
                    # Point layers drape directly at their own (lon, lat, 0) -- the
                    # display law -- never through the has_georeference/resolve gate below, which
                    # is about a RASTER field's CRS-bearing provenance. A PointSet carries no such
                    # provenance (its lon/lat are already WGS84) and needs no compute at all for
                    # this purpose: `backproject` and its stamped grid exist only to register a
                    # point layer onto a RASTER's own pixel grid for the session canvas overlay
                    # (see `_apply`), never for the arrangement's own placement. In frame
                    # mode a point layer has no native-frame placement at all (see this method's
                    # own "Frame mode" docstring section) -- demoted straight to "frame-points".
                    points_color = _hex_to_rgb(style["color_points"])
                    point_status = "frame-points" if frame_mode else "ok"
                    entries.append({"layer": layer, "field": field, "result": None,
                                    "status": point_status, "signature": signature,
                                    "colormap": colormap, "hillshade": hillshade, "stretch": stretch, "surface": surface, "vtrail_color": vtrail_color, "show_raster": show_raster,
                                    "kind": "points", "pointset": field,
                                    "points_color": points_color})
                    self._arr_errors.pop(layer.layer_id, None)
                    continue
                if frame_mode:
                    # The admission rule -- see this method's "Frame mode" docstring
                    # section. No has_georeference gate: that check is exactly what a native-frame
                    # display exists to not need.
                    #
                    # ``field`` here may be a BARE ndarray, not a RasterField -- ``load_field``'s
                    # own docstring allows it ("a RasterField or a bare array"), and a visible
                    # ROI/refined-run sibling (or one round-tripped through the devloop reload) can
                    # leave such a layer next to a real, RasterField active layer; an unguarded
                    # ``field.frame`` would raise ``AttributeError`` for such a sibling the instant
                    # this loop tried to admit it. Mirrors ``active_frame``'s own already-safe
                    # pattern one line above (``getattr(..., "frame", None)``): a frameless field --
                    # the ACTIVE layer's own included -- has nothing for ``Scene``'s frame-mode
                    # placement (``field.frame.to_scene``) to call, so it is excluded entirely
                    # rather than admitted toward a crash one layer deeper. When the ACTIVE layer's
                    # own field is frameless, every OTHER raster layer is excluded too (nothing to
                    # compare a ``frames_compatible`` check against) -- the net result is zero
                    # raster entries this sync, not a crash; the active layer's non-arrangement
                    # session canvas still shows it exactly as before, unaffected by any of this.
                    #
                    # AC3: frame compatibility alone is not enough to admit a non-active raster
                    # sibling -- two BARE rasters both carry the identical default
                    # ``LocalFrame(dx=1, dy=1, units=px)`` (``rasterfield.py``'s own
                    # ``_from_bare_array`` default), so frame equality alone would overlay two
                    # entirely unrelated grids on top of each other. Grid SHAPE is the tiebreaker
                    # the spec's sec 2 "same grid shape for raster layers" clause mandates --
                    # checked here via the same safe-getattr pattern as ``layer_frame`` above (a
                    # field without ``.values``/``.shape`` -- a bare ndarray sibling -- is excluded
                    # exactly like a frameless one, not compared against ``None`` and crashed on).
                    layer_frame = getattr(field, "frame", None)
                    if layer_frame is None:
                        continue        # no frame to place by -- excluded, active layer included
                    if layer is not self.layer:
                        layer_shape = getattr(getattr(field, "values", None), "shape", None)
                        # ROI family clause (the child crop "snaps to the top
                        # left" instead of overlaying its parent): a child dataset is the
                        # legitimate exception AC3's shape gate must not catch -- same source and
                        # an ``roi.window`` tag on EITHER side (creation selects the child, so the
                        # active layer is just as often the tagged one) marks a parent/child (or
                        # sibling-crop) pair whose axes are literal slices of one grid; the
                        # scene's axes-direct frame-mode placement then overlays them at the
                        # correct relative position with no further math. Unrelated same-frame
                        # grids of different sizes still fail the gate exactly as before -- they
                        # share no source and carry no window tag.
                        roi_family = (self.layer is not None
                                      and layer.source_id == self.layer.source_id
                                      and bool(layer.tags.get("roi.window")
                                               or self.layer.tags.get("roi.window")))
                        if not (active_frame is not None
                                and frames_compatible(layer_frame, active_frame)
                                and layer_shape is not None
                                and (layer_shape == active_shape or roi_family)):
                            continue    # excluded entirely -- not even a legend line
                elif not has_georeference(field):
                    entries.append({"layer": layer, "field": field, "result": None,
                                    "status": "no-georeference", "signature": signature,
                                    "colormap": colormap, "hillshade": hillshade, "stretch": stretch, "surface": surface, "vtrail_color": vtrail_color, "show_raster": show_raster,
                                    "surface_field": surface_field,
                                    "surface_field_id": id(surface_field) if surface_field is not None else None,
                                    "levels": levels})
                    self._arr_errors.pop(layer.layer_id, None)
                    continue
                tail = self._arr_tail_key(layer)
                if tail is None or tail in self.cache:
                    try:
                        renderable = resolve(layer, field, self.cache, source_id=layer.source_id)
                    except Exception as exc:
                        entries.append({"layer": layer, "field": field, "result": None,
                                        "status": f"error:{exc}", "signature": signature,
                                        "colormap": colormap, "hillshade": hillshade, "stretch": stretch, "surface": surface, "vtrail_color": vtrail_color, "show_raster": show_raster,
                                    "surface_field": surface_field,
                                    "surface_field_id": id(surface_field) if surface_field is not None else None,
                                    "levels": levels})
                        continue
                    res = renderable.result if isinstance(renderable.result, dict) else None
                    if res is not None and res.get("_roi_values") is not None:
                        # An ROI result's pixels are ROI-LOCAL, and the
                        # scene places pixels as field.x_axis[cols] -- so the scene gets the
                        # region on its OWN grid (native values + axes, pinned at the ROI) and
                        # an outline in that grid, never the parent (picture) indexed by
                        # local pixels. Memoised per cached result: the scene rebuilds a
                        # layer whenever id(result) changes.
                        memo = self._roi_scene_memo.get(layer.layer_id)
                        if memo is None or memo[0] is not res:
                            roi_field = _roi_display_field(res, res["_roi_values"], field,
                                                           layer.name)
                            if roi_field is not None:
                                rh, rw = res["_shape"][:2]
                                memo = (res, roi_field,
                                        {**res, "_roi": {**res["_roi"], "roi": (0, 0, rh, rw)}})
                                self._roi_scene_memo[layer.layer_id] = memo
                        if memo is not None and memo[0] is res:
                            field, res = memo[1], memo[2]
                    # Drape: a holder_map result's h(x) colors the surface; z stays
                    # the height source above -- the "h over elevation" view. Shape-guarded again
                    # in the scene (belt-and-braces); identity stamps feed _STYLE_KEYS diffing.
                    drape = _display_raster_of(res)
                    drape_offset = None
                    if drape is None and res is not None:
                        # A lazily computed output on show drapes once cached (never computed
                        # here); one on its own coarser grid leaves the field draped. One whose
                        # samples register off their pixel index drapes on the moved axes.
                        value = self._cached_output_value(layer, renderable)
                        if value is not None:
                            drape = value["raster"]
                            drape_offset = value.get("display_offset")
                    entries.append({"layer": layer, "field": field, "result": res,
                                    "status": "ok", "signature": signature,
                                    "colormap": colormap, "hillshade": hillshade, "stretch": stretch, "surface": surface, "vtrail_color": vtrail_color, "show_raster": show_raster or drape is not None,
                                    "surface_field": surface_field,
                                    "surface_field_id": id(surface_field) if surface_field is not None else None,
                                    "drape": drape,
                                    "drape_id": id(drape) if drape is not None else None,
                                    "drape_offset": drape_offset,
                                    "levels": levels,
                                    # The edges output row's H: the scene drops this
                                    # layer's H-lines/dots while it is set.
                                    "edges_hidden": _edges_hidden(layer)})
                    self._arr_errors.pop(layer.layer_id, None)
                    continue
                recorded = self._arr_errors.get(layer.layer_id)
                if recorded is not None and recorded[0] == tail:
                    entries.append({"layer": layer, "field": field, "result": None,
                                    "status": recorded[1], "signature": signature,
                                    "colormap": colormap, "hillshade": hillshade, "stretch": stretch, "surface": surface, "vtrail_color": vtrail_color, "show_raster": show_raster,
                                    "surface_field": surface_field,
                                    "surface_field_id": id(surface_field) if surface_field is not None else None,
                                    "levels": levels})
                    continue
                entries.append({"layer": layer, "field": field, "result": None,
                                "status": ("edited — run to compute"
                                           if layer.layer_id in self._pending_layers else "computing"),
                                "signature": signature,
                                "colormap": colormap, "hillshade": hillshade, "stretch": stretch, "surface": surface, "vtrail_color": vtrail_color, "show_raster": show_raster,
                                    "surface_field": surface_field,
                                    "surface_field_id": id(surface_field) if surface_field is not None else None,
                                    "levels": levels})
                if layer.layer_id not in self._pending_layers:
                    queue.append(layer.layer_id)
            self._arr_queue = queue
            if self._arrangement is not None:
                self._arrangement.set_layers(entries)
            self._dispatch_next()
        finally:
            self._arr_syncing = False

    def _report_error(self, message: str) -> None:
        """The one error path, from the worker or from a GUI-thread resolve. No dialog, no toast:
        the strip that failed says so, and the last good render stays on screen.

        ``_errored`` is the load-bearing part -- it records that there is NO cached transform, so
        the next filter change goes to the worker instead of recomputing inline."""
        self._errored = True
        for i in self._transform_indices():
            self.strips.strip(i).set_state("error")
            self.strips.strip(i).set_reading(message)
        if self.layer is not None:
            # The arrangement classifies EVERY visible layer, active or not, so an error
            # on the active layer has to be remembered the same way a background layer's is --
            # keyed by the cache key it was for, so a later edit that changes the chain (and so
            # the key) doesn't keep reporting a now-stale error.
            self._arr_errors[self.layer.layer_id] = (self._arr_tail_key(self.layer), f"error:{message}")
        self.errored.emit(message)

    @property
    def is_computing(self) -> bool:
        return self._thread is not None

    # -- transport -------------------------------------------------------------------------
    def _sync_transport(self, n: int) -> None:
        """Re-range after a recompute, WITHOUT losing the scale the chain is actually on.

        ``set_n_scales`` emits ``scaleChanged`` as it clamps its own position into the new range,
        and the flag keeps that echo from writing a param or firing a resolve of its own --
        ``_on_finished`` resolves once, immediately after this.

        The CHAIN is the authority on which scale is selected, not the widget: the transport
        starts every window's life at 0, so a chain restored at scale 4 would otherwise be
        overwritten by a control that has never been touched. So the index is read from the params
        first, clamped into the new range, written back, and pushed into the transport with
        ``sync_to`` (silent -- ``_on_finished``'s resolve is the one that acts on it). Clamping
        writes back too: if a recompute leaves fewer scales than the chain asked for, params and
        slider have to land on the SAME survivor, or the chain keeps requesting a scale that no
        longer exists while the slider shows one that does.
        """
        i = self._index_of("scale_select")
        if i is None:
            self._syncing = True
            try:
                self.transport.set_n_scales(n)
            finally:
                self._syncing = False
            # Followers adopt where the clamp actually LEFT the master. The clamp's own echo
            # cannot carry them -- it fires while ``_syncing`` is up, holding the transport's
            # stale position, which is why ``_drive_follower_inspectors`` refuses it.
            self._drive_follower_inspectors(self.transport.slider.value())
            return
        target = max(0, min(int(self._params[i].get("scale_idx", 0)), n - 1))
        self._syncing = True
        try:
            self.transport.set_n_scales(n)
        finally:
            self._syncing = False
        self._params[i]["scale_idx"] = target
        self.layer.chain = self._chain()
        self._snapshot_recipe()
        self.transport.sync_to(target)
        # The index that SETTLES, pushed to the followers. This is the second silent settle:
        # without the push, a layer restored at scale 2 moves the master here and leaves every
        # following inspector on whatever it had -- the divergence with the "M" still lit.
        self._drive_follower_inspectors(target)

    def _toggle_transport(self) -> None:
        self.transport.play_button.toggle()

    def _scale_reading(self, idx: int) -> str:
        """The transport's scale reading: nominal a, then the kernel's real-space sigma --
        the physically meaningful width (dynamix.core.scale_units; smoother-sigma is
        wavelet-independent, so no wavelet plumbing here). The old form called the nominal
        scale "px" and multiplied IT by pixel size -- a physics misreading, not a rounding."""
        if not self._scales or not (0 <= idx < len(self._scales)):
            return "a = —"
        from dynamix.core.scale_units import sigma_px

        a = self._scales[idx]
        sigma = sigma_px(a)
        units = self._units()
        if units == "px":
            return f"a = {a:.1f} · σ {sigma:.1f} px"
        return f"a = {a:.1f} · σ {sigma:.1f} px ≈ {sigma * self._px_to_unit():.3g} {units}"

    # -- readings --------------------------------------------------------------------------
    def _update_scale_bar(self) -> None:
        """The label must describe the bar the canvas actually draws: the canvas re-derives the
        bar's LENGTH from the same :func:`nice_round_scalebar` over the same view width, so the
        length is converted here rather than chosen here."""
        (x0, x1), _ = self.canvas.view.viewRange()
        width = x1 - x0
        if not (width > 0):
            self.canvas.set_scale_bar_text("")
            return
        label, bar_px = nice_round_scalebar(width)
        units = self._units()
        if units == "px":
            self.canvas.set_scale_bar_text(f"{label} px")
        else:
            self.canvas.set_scale_bar_text(f"{bar_px * self._px_to_unit():.4g} {units}")

    def _units(self) -> str:
        _, unit = px_to_metres(self.field)
        return unit

    def _px_to_unit(self) -> float:
        """Physical size of one pixel, in the unit ``_units()`` reports -- routed through
        ``dynamix.shell.units.px_to_metres``, which is CRS-linear-unit aware (a projected GeoTIFF's
        pixel size converts feet/US-survey-feet to real metres; a non-georeferenced field's axis
        is already in its own physical unit and passes through unconverted; a geographic field
        stays in degrees). Falls back to ``1.0`` only in the one case ``px_to_metres`` returns
        ``None`` for -- an axis too short to have a pixel size at all -- matching what this method
        returned before ``units.py`` existed."""
        px, _ = px_to_metres(self.field)
        return 1.0 if px is None else px

    def _show_waiting_title(self) -> None:
        stage = self._stage or "the transform"
        self.setWindowTitle(f"{TITLE} — waiting for {stage} to finish…")

    # -- view state ------------------------------------------------------------------------
    def set_soft_span(self, step_index: int, param_name: str, lo: float, hi: float) -> None:
        """Persist an edited knob span to ``layer.tags["ui.spans"]`` as JSON.

        Soft bounds are VIEW state: the sub-range of the legal one a knob spans. They live in the
        layer's tags, never in params, precisely so they can never reach a cache key -- retuning a
        slider must not invalidate the science.
        """
        spans = json.loads(self.layer.tags.get("ui.spans", "{}"))
        spans[f"{step_index}.{param_name}"] = [float(lo), float(hi)]
        self.layer.tags["ui.spans"] = json.dumps(spans, sort_keys=True)

    def soft_spans(self) -> dict:
        return json.loads(self.layer.tags.get("ui.spans", "{}"))

    # -- arrangement flip --------------------------------------------------------------------
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Ableton's session/arrangement flip: Tab (or shift-Tab, ``Key_Backtab``) swaps the
        center zone, UNLESS a ``QLineEdit`` currently has focus -- an inline editor (layer
        rename, a chain-strip value box, a ROI field) needs Tab for its own field-to-field
        navigation, and stealing it here would leave no way to finish typing and move on.

        An APPLICATION-level filter (installed on ``qApp`` in ``__init__``, removed in
        ``closeEvent``), not a ``keyPressEvent``/``event()`` override on this window, because a
        real Tab press is never delivered to the window -- Qt hands a KeyPress straight to
        whichever widget currently has keyboard focus (the open button, a layer-list row, a
        ``DragValue`` knob...), and THAT widget's own inherited ``QWidget.event()`` consumes
        ``Key_Tab``/``Key_Backtab`` for focus-chain navigation (``focusNextPrevChild()``)
        before the key ever reaches a virtual ``keyPressEvent()`` override up at the window.
        An override on ``MainWindow`` only ever sees a Tab synthetically addressed to the
        window object itself, which is not what a real key press does -- an application-wide
        filter runs BEFORE Qt's normal per-widget dispatch, for every event, regardless of
        which specific descendant is the actual target, which is exactly the "focus-independent
        hotkey" this needs.

        **Target-hierarchy check:** EVERY live ``MainWindow``
        installs its OWN app-level filter, and Qt runs them all against every event that flows
        through ``qApp`` -- the first one to return ``True`` consumes it. Without discriminating
        by WHOSE hierarchy the event actually belongs to, window B's filter (if Qt happens to
        run it first) would happily flip window B for a Tab that was addressed to window A's
        focused widget, while A never sees the key at all -- confirmed empirically with two
        shown windows. So the very first thing this checks (after the cheap KeyPress/Tab type
        test) is whether the event's actual target lives under THIS window: ``target`` is
        ``obj`` when ``obj`` is a real ``QWidget`` (the common case -- ``obj`` is literally who
        Qt is delivering this key press to), falling back to ``QApplication.focusWidget()`` only
        when it is not (an app-level filter sees every ``QObject`` in the process -- timers,
        ``QWindow``s, and other non-widget receivers included, so ``obj`` cannot be assumed to be
        a widget at all). ``target is self or self.isAncestorOf(target)`` is the actual
        ownership test; a ``target`` of ``None`` (nothing focused, or a non-widget event whose
        focus widget genuinely doesn't exist) fails it and this filter declines, leaving the
        event for whichever window it actually belongs to (or nobody, harmlessly).

        Only once an event is confirmed to be THIS window's does the ``QLineEdit`` guard run:
        a genuinely focused line edit inside this window keeps Tab for its own field-to-field
        navigation; a Tab this filter does NOT consume falls through to Qt's normal handling
        unchanged (``super().eventFilter(...)``), which is what lets it actually navigate rather
        than merely not-flip.
        """
        if event.type() == QtCore.QEvent.Quit:
            # The application is quitting -- note it and get out
            # of the way. NOT consumed (the return below is the base class's own ``False``):
            # swallowing it here would stop the app quitting. This filter is already installed on
            # ``qApp``, which is where a Quit is delivered, so the note is free -- and it lands
            # BEFORE Qt closes a single window, which is the only ordering that is deterministic.
            # See :attr:`_shutting_down`.
            self._shutting_down = True
            return super().eventFilter(obj, event)
        if event.type() != QtCore.QEvent.KeyPress or \
                event.key() not in (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab):
            return super().eventFilter(obj, event)
        target = obj if isinstance(obj, QtWidgets.QWidget) else QtWidgets.QApplication.focusWidget()
        if target is None or not (target is self or self.isAncestorOf(target)):
            return super().eventFilter(obj, event)          # not this window's event
        if isinstance(target, QtWidgets.QLineEdit):
            return super().eventFilter(obj, event)           # an inline editor keeps its Tab
        self._cycle_center_view()
        event.accept()
        return True

    def _sync_view_switcher(self) -> None:
        """Re-check exactly the button for ``self._center_view``, wrapped in ``blockSignals`` --
        the reentrancy guard the design calls for: without it, a REAL click has already carried
        the target button's own checked state to ``True`` before its ``clicked`` handler ever
        runs (that is how an exclusive ``QButtonGroup`` works), so a bare ``setChecked`` here
        would be a same-state no-op for THAT button but would still fire ``clicked`` a second
        time for whichever button the group just unchecked, re-entering
        :meth:`_set_center_view` for no reason. Called at the tail of every real transition in
        :meth:`_set_center_view`, so the switcher can never show a view other than the one that
        method just settled on -- whether the transition came from a button, from
        :meth:`_cycle_center_view` (Tab), or from ``Settings.center_view``'s own restore."""
        for view, button in self._view_switcher_buttons.items():
            button.blockSignals(True)
            button.setChecked(view == self._center_view)
            button.blockSignals(False)

    def _redraw_canvas_overlay_if_dirty(self) -> None:
        """Draw the ACTIVE result's overlay onto the (now-visible) raster canvas if a landing
        deferred it while the canvas was hidden (double-update guard). A no-op when nothing was
        deferred; a deferred result with no per-scale extrema clears whatever an earlier result
        left drawn."""
        if not self._canvas_overlay_dirty:
            return
        self._canvas_overlay_dirty = False
        result = self._active_result
        if isinstance(result, dict) and result.get("extrema") and "_shape" in result:
            layers = result["extrema"]
            idx = 0 if len(layers) == 1 else int(result.get("_scale_idx", 0))
            self.canvas.set_result(result, min(idx, len(layers) - 1))
            if self.canvas.cap_note:
                self._notify(self.canvas.cap_note, "status")
        else:
            self.canvas.clear_overlays()
            self.canvas.set_pick_chains(None)

    def _set_center_view(self, view: str) -> None:
        """Switch the center zone to ``view`` -- ``"raster"`` (the session canvas, index 0),
        ``"vector"`` (the arrangement in each layer's own NATIVE frame -- ``field.frame`` straight
        through, no CRS, no georeference gate) or ``"geo"`` (the arrangement's original WGS84
        placement, unchanged). A no-op when ``view`` already IS the current one -- an idempotent
        re-click of the switcher's own currently-checked button, or a redundant restore.

        Persists to ``Settings.center_view`` on every REAL transition (``"raster"`` included) so
        the next window open restores it (:meth:`load_field`'s own one-shot restore call reads
        this back).

        ``"vector"``/``"geo"`` share the arrangement's lazy build -- constructed on first use,
        parked (never rebuilt) on every later switch, same "Lazy + parked" contract the two-state
        flip this replaces always had (spec section 3) -- and most of that flip's own sequence:
        build if needed, push :meth:`ArrangementView.set_frame_mode`, ``activate()``, push the
        mask row's current values. They diverge only in which VIEW-OPTIONS get pushed next
        (``"geo"`` pushes mode/graticule/vexag/background, byte-identical to before; ``"vector"``
        pushes only ``background`` -- mode-independent chrome -- since the projection/mode dialog
        the other three feed is meaningless with no CRS,
        :meth:`ArrangementView.set_frame_mode`'s own docstring -- PLUS scale-space/scale-space-stretch, pushed UNCONDITIONALLY in
        BOTH, alongside ``background`` -- placement-independent (chain points lift the same way
        in geo or frame mode, ``Scene``'s own module docstring), so it is never gated behind
        ``frame_mode`` the way mode/graticule/vexag are -- see the push block's own comment for the gap this fixes) and in the ``frame_mode`` bool
        handed to :meth:`_sync_arrangement`, which is what actually changes layer admission (see
        that method's own "Frame mode" docstring section).

        The import is LOCAL, not module-level, and it is the ONLY reference to
        ``dynamix.shell.arrangement.view`` (and everything ``view.py`` itself imports -- ``scene``,
        ``camera``) anywhere in this file -- unchanged from the two-state flip this replaces; see
        ``__init__``'s own comments on ``self._mask_row``/``self._group_palette``/
        ``self._view_dialog`` for the three deliberate exceptions.
        """
        if view == self._center_view:
            return
        self._center_view = view
        self._refresh_panel_relevance()
        for layer in self.project.layers:           # an output on its own grid is 2-D only
            self._sync_output_rows(layer)
        update_settings(center_view=view)
        if view == "raster":
            if self._arrangement is not None:
                self._arrangement.deactivate()
            self._center_stack.setCurrentIndex(0)
            self._redraw_canvas_overlay_if_dirty()   # deferred while the canvas was hidden
            self._sync_view_switcher()
            return
        if self._arrangement is None:
            from dynamix.shell.arrangement_facade import ArrangementView
            self._arrangement = ArrangementView(self)
            self._center_stack.addWidget(self._arrangement)
            # Hand the view its palette REFERENCE, once -- unlike the mask row's values
            # (pushed on every flip, just below), a reference only needs handing over once, since
            # the palette itself is never rebuilt across park/resume cycles (see
            # ArrangementView.set_group_palette's own docstring).
            self._arrangement.set_group_palette(self._group_palette)
            # The commit transaction's own entry point -- see _on_groups_committed;
            # commitFinished's deferred single resync is _on_commit_finished.
            self._arrangement.groupsCommitted.connect(self._on_groups_committed)
            self._arrangement.commitFinished.connect(self._on_commit_finished)
            # The header's "View…" button -> the (lazily built) View dialog.
            self._arrangement.viewOptionsRequested.connect(self._on_view_options_requested)
            self._arrangement.footprintsRightClicked.connect(self._on_footprints_right_clicked)
            self._arrangement.set_footprints(self._footprints)
        frame_mode = view == "vector"
        # Store + forward the Vector-tab placement switch BEFORE
        # activate() -- ArrangementView buffers a pre-build call exactly like set_layers already
        # does, and this ordering matches the design's bullet list for both "vector" and "geo".
        self._arrangement.set_frame_mode(frame_mode)
        # Push the same bool into the View dialog too,
        # if it has ever been built -- it hides only its own Projection tab (see ViewDialog.
        # set_frame_mode's own docstring), unlike ArrangementView's own no-longer-hidden button. A
        # no-op the overwhelmingly common case (the dialog is lazy-built, only on its first "View…"
        # click) -- see _on_view_options_requested for the matching push on open/build.
        if self._view_dialog is not None:
            self._view_dialog.set_frame_mode(frame_mode)
        self._center_stack.setCurrentIndex(1)
        self._arrangement.activate()
        # Push the panel's current mask values into the just-activated view -- the view
        # has no row of its own to read (see ArrangementView.set_mask's own docstring).
        # Pushed in BOTH "vector" and "geo" -- the mask (modulus/scale range) filters which chains
        # are visible independently of placement, so it applies to a native-frame display exactly
        # as much as a georeferenced one.
        self._arrangement.set_mask(self._mask_row.values())
        # Push the current selection mode
        # into the just-activated view too, mirroring the mask push immediately above -- a mode
        # chosen while the Raster tab was showing (or the arrangement's own former default,
        # "click", if it never was) must already be in effect the instant this tab appears, not
        # silently revert until the next explicit mode change. See
        # ArrangementView.set_selection_mode's own docstring for the full contract.
        self._arrangement.set_selection_mode(self._selection_mode)
        # Push whatever was last persisted to Settings.view_options into the just-
        # activated view too. "geo" pushes all four -- mode/graticule/vexag/background, the same
        # unconditional-on-every-activate contract ``set_mask`` just above already keeps (never
        # deduped against the prior value -- see its own docstring); ``set_graticule`` has its own
        # no-op guard when already in the requested state (Scene.set_graticule's own docstring);
        # ``set_mode``/``set_vertical_exaggeration`` do not, and DO trigger a real
        # ``Scene._rebuild()`` here even when the value is unchanged -- a SECOND one, right before
        # ``_sync_arrangement``'s own (pre-existing, likewise unconditional) rebuild a few lines
        # down. Accepted rather than engineered around: the extra pass only RE-PROJECTS already-
        # cached lon/lat (``Scene._lonlat_cache``, the fix) and already-resolved chain
        # data -- no ``resolve()`` call, the genuinely expensive part -- and every activate already
        # pays a full rebuild via ``_sync_arrangement`` regardless of this push, so this is one
        # more geometry-only pass on top of an existing one, not a new class of cost.
        # "vector" pushes ONLY ``background`` -- mode-independent chrome, per this method's own
        # docstring -- since the other three feed a projection/mode dialog that has nothing
        # meaningful to show with no CRS (``ArrangementView.set_frame_mode``'s own docstring).
        # normalized_view_options is the SAME defaulting function ViewDialog itself restores from,
        # so a missing or garbage settings.json can never leave this push and the dialog's own
        # display disagreeing about what "default" means.
        opts = normalized_view_options(load_settings().view_options)
        if not frame_mode:
            self._arrangement.set_mode(opts["mode"])
            self._arrangement.set_graticule(opts["graticule"])
            self._arrangement.set_vertical_exaggeration(opts["vexag"])
        self._arrangement.set_background(opts["background"])
        # CONDITION: pushed UNCONDITIONALLY, alongside ``set_background`` immediately above. A
        # SESSION RESTORE landing directly on a saved ``center_view == "vector"``
        # (``load_field``'s own one-shot restore call, near the top of this file) never runs the
        # geo branch at all, so gated inside ``if not frame_mode:`` a persisted
        # ``scale_space: True`` would silently never reach the freshly-built ``Scene``.
        # Scale-space is placement-independent (chain points lift identically in geo or frame
        # mode, ``Scene``'s own module docstring), unlike mode/graticule/vexag, which genuinely
        # feed a CRS/projection dialog with nothing meaningful to show in a native frame
        # (``ArrangementView.set_frame_mode``'s own docstring) -- so, unlike those three, there
        # is no reason to gate this one.
        self._arrangement.set_scale_space(opts["scale_space"], opts["scale_space_stretch"])
        self._sync_arrangement(frame_mode)
        self._sync_view_switcher()

    def _cycle_center_view(self) -> None:
        """Tab's own action (``eventFilter`` calls this now, not :meth:`_toggle_center_view`) --
        walks :data:`_CENTER_VIEWS` in order, wrapping ``"geo"`` back to ``"raster"``."""
        i = _CENTER_VIEWS.index(self._center_view)
        self._set_center_view(_CENTER_VIEWS[(i + 1) % len(_CENTER_VIEWS)])

    # -- selection mode ---------------------------
    def set_selection_mode(self, mode: str) -> None:
        """Programmatic equivalent of clicking a mode-row button -- the ONE state three surfaces
        (the mode-row buttons, the ``c``/``v`` hotkeys, and any other caller) all funnel through,
        mirroring EQSelect's own ``set_selection_mode`` contract ["raises on an unknown
        mode and syncs the radio"]. Raises ``ValueError`` for anything outside
        :data:`_SELECTION_MODES` -- there is no silent "do nothing" reading of asking for a mode
        that does not exist, the same "no silent no-op on a bad argument" contract
        ``GroupPalette.apply_picks``/``new_group`` already keep.

        Always syncs the row and pushes to the canvas, even when ``mode`` already IS the current
        one -- unlike :meth:`_set_center_view`'s own no-op-if-same guard, EQSelect's own setter
        carries no such guard either, and both operations here are cheap and idempotent (a
        ``blockSignals``-wrapped re-check, a plain attribute store) so there is nothing to gain
        from skipping them."""
        if mode not in _SELECTION_MODES:
            raise ValueError(f"unknown selection mode {mode!r}")
        self._selection_mode = mode
        self._sync_selection_mode_row()
        self.canvas.set_selection_mode(mode)
        # Forward into the arrangement's
        # Vector/Globe view too -- a no-op-before-it-exists guard, same reasoning as every other
        # `self._arrangement is not None` push in this file (mask/mode/graticule/...): the mode
        # row is reachable long before the first Tab press ever builds the arrangement.
        if self._arrangement is not None:
            self._arrangement.set_selection_mode(mode)

    def _sync_selection_mode_row(self) -> None:
        """Re-check exactly the button for ``self._selection_mode``, wrapped in ``blockSignals`` --
        the identical reentrancy guard :meth:`_sync_view_switcher` uses and for the same reason:
        without it, a REAL click has already carried the target button's own checked state to
        ``True`` before its ``clicked`` handler ever runs (that is how an exclusive
        ``QButtonGroup`` works), so a bare ``setChecked`` here would be a same-state no-op for
        THAT button but would still fire ``clicked`` a second time for whichever button the group
        just unchecked, re-entering :meth:`set_selection_mode` for no reason."""
        for mode, button in self._selection_mode_buttons.items():
            button.blockSignals(True)
            button.setChecked(mode == self._selection_mode)
            button.blockSignals(False)

    def _cycle_selection_mode(self) -> None:
        """`v`'s own action: walks :data:`_SELECTION_MODE_CYCLE` (box -> lasso -> transect
        -> box), skipping click entirely. Pressed while already in click mode (there is no
        principled "next" mode to resume a cycle click was never part of), this starts the cycle
        fresh at its first entry, box."""
        if self._selection_mode not in _SELECTION_MODE_CYCLE:
            self.set_selection_mode(_SELECTION_MODE_CYCLE[0])
            return
        i = _SELECTION_MODE_CYCLE.index(self._selection_mode)
        self.set_selection_mode(_SELECTION_MODE_CYCLE[(i + 1) % len(_SELECTION_MODE_CYCLE)])

    def _toggle_center_view(self) -> None:
        """Flip the center zone between the session canvas (index 0) and the arrangement view in
        its ORIGINAL, georeferenced placement (index 1) -- unchanged in EFFECT from before the
        shell three-view plan: building the arrangement lazily on first use, parking it (never
        rebuilding) on every flip after that (spec section 3, "Lazy + parked"), and pushing the
        full mask/mode/graticule/vexag/background sequence every time it flips in.

        **Retained as a two-state shim.** The flip logic this
        docstring describes is now :meth:`_set_center_view`'s own ``"raster"``/``"geo"`` branches
        (this method's old body, MOVED there rather than duplicated -- ``"geo"`` is exactly the
        two-state flip's old "index 1" branch, with ``frame_mode=False`` throughout, byte-for-byte
        the same pushes in the same order). Tab itself now calls :meth:`_cycle_center_view`
        instead (three states, not two -- see its own docstring). This method survives, delegating,
        because it is still called directly by name across the pre-existing arrangement-view test
        suite (``tests/test_arrangement_*.py``, ``tests/test_right_panel.py``) -- outside this
        task's own file list -- so its own observable behaviour (index 0 <-> index 1, full
        mode/graticule/vexag/background pushes, geo-style admission throughout) must stay exactly
        what it always was.
        """
        if self._center_stack.currentIndex() == 0:
            self._set_center_view("geo")
        else:
            self._set_center_view("raster")

    def _on_view_options_requested(self) -> None:
        """``ArrangementView.viewOptionsRequested`` (the header's "View…" button) -> the View
        dialog -- built lazily, on the FIRST click (mirrors ``self._arrangement``'s own lazy-build
        shape), shown/raised on every click after that. Non-modal (``ViewDialog.setModal(False)``
        in its own constructor), so this never blocks -- the dialog and the arrangement/session
        flip both stay usable at once, per the design."""
        if self._view_dialog is None:
            self._view_dialog = ViewDialog(self._arrangement, self)
        # The dialog may be built lazily while showing
        # a DIFFERENT view than whichever is current by the time a later click reopens it (raster
        # has no "View…" button at all, but geo<->vector can still flip while the dialog stays
        # hidden) -- push the current state unconditionally on every open, the same "no dedup
        # against the prior value" posture _set_center_view's own view-options push already takes.
        self._view_dialog.set_frame_mode(self._arr_frame_mode)
        self._view_dialog.show()
        self._view_dialog.raise_()
        self._view_dialog.activateWindow()

    # -- the commit transaction ------------------------------------------------------
    def _on_groups_committed(self, layer_id: int, groups: dict) -> None:
        """``ArrangementView.groupsCommitted`` landed for ``layer_id``: ``groups``
        is that layer's own slice of the SNAPSHOT ``ArrangementView.commit`` (formerly ``_on_commit_clicked``) took once, up front --
        stamp it with this layer's CURRENT transform signature and write it into both
        ``layer.tags["groups"]`` and a ``group_paint`` chain step. The actual re-resolve/resync is
        deferred to :meth:`_on_commit_finished` -- see that method's own
        docstring for why.

        **Locked/frozen refuses wholesale.** Committing WRITES the layer's chain and tags -- an
        edit like any other the zone's own ``_lock_notice`` guard already refuses
        (``_on_param_changed``/``_on_chain_edited``/``_on_scale_changed``) -- so it gets the same
        treatment here, before anything is touched: no tags write, no chain step, no resolve.
        ``strips._show_warning`` is the one notice surface this window has, used the same way
        ``_on_remove_requested``/``_open_project_path`` already use it for a notice that isn't
        necessarily about the layer currently on screen.

        **The transform signature** is ``repr(self._transform_signature_for(layer))`` -- the same
        tuple ``_dispatched``'s own mismatch check compares by equality, stringified because
        ``encode_groups`` stores it as JSON text. Recorded per the design step 1 ("member
        chain ids plus the transform signature they belong to"); what actually GUARDS staleness
        today is ``group_paint``'s own structural out-of-range check on ``result["_stale_groups"]``
        (the documented floor), surfaced in :meth:`_apply`, not a comparison against this
        stored string -- see ``devices/groups.py``'s own docstring for why a same-length
        reordering is a gap the signature alone cannot close.

        ``layer_id`` is recorded in ``self._commit_batch`` only once this layer's write actually
        happened (never for a locked refusal, never for an empty ``groups`` slice) -- exactly the
        "affected layers" :meth:`_on_commit_finished` needs to act on.
        """
        layer = self._layer_by_id.get(layer_id)
        if layer is None:
            return
        notice = _lock_notice(layer)
        if notice is not None:
            if self.strips is not None:
                self.strips._show_warning(notice)
            return
        if not groups:
            return
        signature = repr(self._transform_signature_for(layer))
        payload = {name: {"chains": spec["chains"], "color": spec["color"], "signature": signature}
                  for name, spec in groups.items()}
        self._ensure_group_paint_step(layer, encode_groups(payload), signature)
        self._commit_batch.append(layer_id)

    def _on_commit_finished(self) -> None:
        """``ArrangementView.commitFinished`` -- every ``groupsCommitted`` for one Commit-button
        click has already landed. ``self._commit_batch`` (populated only
        by an ACTUAL write in :meth:`_on_groups_committed`, never a locked refusal or an empty
        slice) is drained here and acted on exactly ONCE: the active layer's own session canvas
        takes the normal filter-path redraw (``group_paint`` is a FILTER -- appending or updating
        it never moves ``_transform_signature()``, so this is exactly ``_on_chain_edited``'s "no
        transform signature change" branch); a background layer has no live strips to reresolve
        through, so ONE ``_sync_arrangement()`` call re-derives every visible layer's entry
        cache-first (always a hit here -- the transform TAIL is untouched by a filter-only edit)
        instead of one resync per background layer mid-loop, which is what let a resync in
        response to layer 1's own commit prune layer 2's still-uncommitted palette membership
        before this method -- or even :meth:`_on_groups_committed` for layer 2 -- ever ran.
        """
        batch, self._commit_batch = self._commit_batch, []
        if not batch:
            return
        active_id = self.layer.layer_id if self.layer is not None else None
        if active_id in batch:
            self._reresolve()
        if any(lid != active_id for lid in batch):
            self._sync_arrangement()

    def _ensure_group_paint_step(self, layer, encoded: str, signature: str) -> None:
        """Append (first commit) or update in place (re-commit -- idempotent, NEVER a second
        step) ``layer``'s own ``group_paint`` filter step with ``encoded`` (spec_json) and
        ``signature`` (the device's own ``signature`` param -- ``_on_groups_committed``'s single
        per-layer transform-signature string, the same one that is ALSO embedded per-group inside
        ``encoded`` itself), then write ``layer.tags["groups"] = encoded`` -- the exact same
        string, so a later resolve and a later save/load both read back exactly what was just
        committed.

        Mirrors ``_on_chain_edited``'s own bookkeeping discipline: the candidate chain is built
        from LOCAL variables and only assigned to ``layer``/``self`` once ``Chain(...).
        materialized()`` has already succeeded, so a hypothetical construction failure (unreachable
        today -- ``group_paint``'s params are both ``ParamKind.TEXT``, which validates any string
        unconditionally) leaves nothing partially written.

        Generalized -- like :meth:`_transform_signature_for` -- to a layer that may not be the
        active one: the active layer's working copy (``_names``/``_params``/``_bypassed``/
        ``_rack``) is the live state to edit, and the zone is rebuilt afterward so it shows the
        new/updated step (the edit originates OUTSIDE the zone this time, unlike every other
        chain-editing path, so nothing already on screen reflects it the way a live
        ``WorkflowZone`` gesture would). Any other layer's recipe is read from ``self._recipes``
        (falling back to ``layer.chain.steps`` for one never edited before) and
        written back the same way, with no zone to rebuild.
        """
        is_active = layer is self.layer
        if is_active:
            names, params = list(self._names), [dict(p) for p in self._params]
            bypassed, rack = list(self._bypassed), list(self._rack)
        else:
            recipe = self._recipes.get(layer.layer_id)
            if recipe is not None:
                names = [d["device"] for d in recipe]
                params = [dict(d.get("params", {})) for d in recipe]
                bypassed = [bool(d.get("bypassed", False)) for d in recipe]
                rack = [d.get("rack") for d in recipe]
            else:
                names = [ref.device for ref in layer.chain.steps]
                params = [dict(ref.params) for ref in layer.chain.steps]
                bypassed = [False] * len(names)
                rack = [None] * len(names)

        step_params = {"spec_json": encoded, "signature": signature}
        if "group_paint" in names:
            i = names.index("group_paint")
            params[i] = step_params
            # A commit un-bypasses its own painter -- a re-commit onto a step the user had
            # bypassed must not leave it bypassed, or the just-committed groups are silently
            # discarded from every future resolve until someone happens to un-bypass it by hand.
            bypassed[i] = False
        else:
            names = names + ["group_paint"]
            params = params + [step_params]
            bypassed = bypassed + [False]
            rack = rack + [None]

        chain = Chain(tuple(DeviceRef(names[i], dict(params[i]))
                            for i in range(len(names)) if not bypassed[i])).materialized()

        if is_active:
            self._names, self._params, self._bypassed, self._rack = names, params, bypassed, rack
            self.layer.chain = chain
            self._snapshot_recipe()
            self._build_strips()
        else:
            layer.chain = chain
            self._recipes[layer.layer_id] = [
                {"device": n, "params": dict(p), "bypassed": b, "rack": r}
                for n, p, b, r in zip(names, params, bypassed, rack)
            ]
        layer.tags["groups"] = encoded

    # -- closing ---------------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Honest close-during-compute (v1): there is no cancellation, so the window says which
        stage it is waiting out and closes when that stage completes. ``_on_finished`` and
        ``_on_error`` are bound methods already connected to the worker; both call ``close()``
        again once the thread is down, and the branch above then accepts.

        The Tab-flip event filter is removed here, on the branch that ACTUALLY closes, not on
        a deferred one: while a close is deferred for an in-flight compute the window is still
        fully alive and Tab must keep working, and this method runs again (accepting that time)
        once the compute lands. Leaving a closed window's filter installed on ``qApp`` would
        have it keep intercepting Tab presses meant for whatever window opens next -- a
        cross-window leak.

        The splitter tree's current sizes are persisted on the SAME branch, for the
        same reason: a close deferred mid-compute must not snapshot sizes that may still change
        (a lock/unlock or a resize while waiting) before the window actually goes away.
        """
        if self._thread is None:
            # From here on nothing this window does to its
            # inspectors is a user gesture -- see :attr:`_shutting_down`. On the branch that
            # ACTUALLY closes, for ``removeEventFilter``'s own reason: a close deferred for an
            # in-flight compute leaves the window fully alive, and an inspector closed by hand
            # while it waits is still the user closing it.
            self._shutting_down = True
            update_settings(splitter_sizes={"work": self._work_split.sizes(),
                                            "main": self._main_split.sizes()})
            QtWidgets.QApplication.instance().removeEventFilter(self)
            super().closeEvent(event)
            return
        if self._out_job is not None and self._worker is not None:
            # An output job is a view, never worth waiting out: it stops at its next check.
            self._worker.cancel()
        self._closing = True
        self._show_waiting_title()
        event.ignore()
