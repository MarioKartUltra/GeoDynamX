# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Canvas: the raster + vectorized WTMM overlay view (DESIGN.md's flat 2-D reading of a layer).

Analysis runs in each dataset's native domain; this module only DISPLAYS a result that
was already computed in pixel space -- it never reprojects, resamples, or otherwise touches the
data a WTMM run is sensitive to. Pure, numpy-only helpers do the actual geometry work and are
testable without a window:

- :func:`hline_polylines` -- every ordered H-line run at one scale as ONE NaN-separated polyline.
- :func:`vchain_trails` -- every V-chain's scale-drift as ONE NaN-separated polyline.
- :func:`lod_stride` -- display decimation so a huge raster still paints at interactive rates.
- :func:`nice_round_scalebar` -- the 1/2/5 x 10^n "nice length" ladder for the distance scale bar.
- :func:`dilate_coi_mask` -- the per-scale growth of a missing-data mask into its COI.
- :func:`coi_outline` -- NaN-joined isocurve segments around a (dilated) missing-data mask.
- :func:`wavelet_bar_scale_factor` -- the wavelet-bar duo's shared amplitude-normalization factor.
- :func:`roi_from_corners` -- two dragged data-space corners as an image-pixel ROI rectangle.
- :func:`roi_offset` -- an ROI result's ``(row, col)`` origin in its SOURCE FILE's coordinates.
- :func:`window_offset` -- where a displayed field starts inside that same file.
- :func:`display_offset` -- the difference: where an ROI result's overlays go on the image shown.
- :func:`classify_chains` -- partitions a result's
  ``chains`` into plain / seam-flagged / committed-group trails, by ``tags``.

:class:`Canvas` (a ``pyqtgraph.GraphicsLayoutWidget``) wires those helpers to on-screen items. Two
device-agnostic rules shape it: the Signal Rule (data gets a perceptually-uniform colormap --
``viridis``, never rainbow) and the Theme Rule (chrome carries no literal color or font; it comes
from :mod:`dynamix.shell.theme` or a Qt style property). The one carve-out, matching how colormap
NAMES are treated, is that chain/extremum colors are DATA identity, not chrome -- see the palette
constants below.

**Exception to the "canvas.py is not edited" isolation guarantee.** That rule predates the group work: the design separately promises "the session canvas colors
groups like seam flags", which cannot be true without touching the one file that already knows how
to draw a seam flag. Resolved by editing this
file minimally -- one classification helper plus a small, lazily-populated dict of per-color items
alongside the existing ``seam_item``/``ghost_item`` machinery -- rather than duplicating
``set_result``'s geometry/caching/offset logic in a second module to keep the letter of sec 6.
Every existing item, cache key and code path here is unchanged; nothing was deleted.
"""
from __future__ import annotations

import numpy as np
from dynamix.model.project import REFERENCE_COLORS
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core.chain_pick import chains_in_box, chains_in_polygon, pick_chain
from dynamix.core.chain_product import DRAW_CAP as _DRAW_CAP
from dynamix.core.transect import orient_endpoints
from dynamix.roi.picture import display_stride, native_shape, sample_spacing
from dynamix.core.wtmm_backend import _order_lines
from dynamix.shell.theme import RESTRAINED_DARK
from dynamix.shell.units import px_to_metres

__all__ = [
    "hline_polylines",
    "vchain_trails",
    "classify_chains",
    "lod_stride",
    "nice_round_scalebar",
    "dilate_coi_mask",
    "coi_outline",
    "wavelet_bar_scale_factor",
    "roi_from_corners",
    "roi_offset",
    "window_offset",
    "display_offset",
    "ROI_MODIFIER",
    "LASSO_MODIFIER",
    "Canvas",
]

#: The modifier that turns a left-drag on the canvas into an ROI selection -- ⌘-drag, per the
#: spec's "Model and interaction".
#:
#: It is spelled ``ControlModifier`` and that is NOT a mistake. Qt SWAPS Control and Meta on macOS
#: unless the application sets ``AA_MacDontSwapCtrlAndMeta``, which this app never does, so the ⌘
#: key arrives in an event's ``modifiers()`` as ``ControlModifier`` and the physical Control key
#: arrives as ``MetaModifier`` -- exactly the opposite of what the two names suggest. Measured
#: rather than assumed (PySide6 / Qt 6.11.1, macOS 14.5): ``QKeySequence(ControlModifier |
#: Key_A).toString(NativeText)`` renders "⌘A", and the same call with ``MetaModifier`` renders
#: "⌃A". Binding ``MetaModifier`` here -- the name that reads like "the Meta/Command key" -- would
#: therefore have bound Control-drag and left ⌘-drag panning the view.
#: ``tests/test_shell_roi_flow.py`` pins that native rendering so a Qt release (or a future
#: ``AA_MacDontSwapCtrlAndMeta``) that flips the mapping fails loudly here instead of silently
#: rebinding the gesture. macOS is the only target for now; the test skips elsewhere.
ROI_MODIFIER = QtCore.Qt.KeyboardModifier.ControlModifier

#: The lasso gesture's own modifier -- ⌥ (Option/Alt). Unlike ⌘ above, Qt does NOT swap Alt on
#: macOS (only Control<->Meta), so ``AltModifier`` names the physical Option key directly.
LASSO_MODIFIER = QtCore.Qt.KeyboardModifier.AltModifier

#: The raster canvas's own click-tolerance radius, in SCREEN
#: pixels -- the same 8 px convention ``dynamix.shell.arrangement.scene``'s own
#: ``_PICK_MAX_DIST_PX`` uses for the arrangement's 3-D pick, converted to DATA units per click
#: through the ViewBox's current zoom (see :meth:`Canvas._pick_radius_data`) since (unlike the
#: arrangement's screen-space VTK pick) this canvas's data space is the image itself.
_PICK_MAX_DIST_PX = 8.0

#: A press/release pair with more travel than this many SCREEN pixels is a drag (pan), not a
#: click -- the gesture-disambiguation threshold :meth:`Canvas.mouseReleaseEvent` applies before
#: ever treating a plain left-button release as a pick.
_CLICK_MAX_TRAVEL_PX = 3.0

#: Data-layer palette -- what a chain or extremum IS, not how the app is dressed, so (like the
#: "viridis" colormap name below) these are exempt from the Theme Rule. RGB TUPLES rather than hex
#: strings: The chrome-literal grep matches `#rrggbb` text and font-family names, not tuples.
EXTREMA_COLOR = (200, 200, 200)     # isolated extrema (line_id == -1): light gray
HCHAIN_COLOR = (235, 235, 235)      # H-chain polylines: white-ish
VTRAIL_COLOR = (255, 160, 40)       # V-chain drift trails: amber family

#: Point-layer scatter overlay -- a distinct, cool hue, deliberately
#: NOT amber (VTRAIL_COLOR/the ROI band's One-Accent Rule exception) and NOT a seam/ghost/extrema
#: gray: a backprojected point catalogue is its own kind of overlay, not a variant of an existing
#: one. A light, desaturated cyan-blue (hex ``4fc3f7`` -- no leading ``#`` in this comment, on
#: purpose: ``tests/test_shell_boundaries.py``'s Theme Rule scan greps this whole file's TEXT, not
#: just its code, for a literal ``#rrggbb``), wired through the SAME ``ui.color_points`` tag pipeline built for the other three overlay colors (``main_window._display_style_of``) --
#: see that module's own comment for why the default is DERIVED from this constant rather than
#: restated as a second literal.
POINTS_COLOR = (0x4F, 0xC3, 0xF7)

#: A V-chain a classification device (``dynamix.devices.chain_classify.ChainClassify``) tagged as
#: a probable seam artifact (``chain.get("tags")`` truthy) -- drawn on its OWN item
#: (:attr:`Canvas.seam_item`) so the flag stays visible independent of the trails toggle (see
#: :meth:`Canvas.set_result`). Its value matches the theme's ``seam`` role exactly, deliberately,
#: the same way ``ROI_BOUNDS_COLOR`` below matches ``amber`` -- own constant, not a reuse of
#: ``VTRAIL_COLOR``: a plain drift trail and a flagged one are different things to look at, and
#: sharing a constant would mean re-tinting either could never happen without re-tinting both.
#: ``tests/test_shell_boundaries.py`` pins the relationship to ``theme.seam``.
SEAM_COLOR = (255, 112, 67)         # seam-flagged V-chain trails

#: A V-chain a classification device EXCLUDED (``result["chains_excluded"]``, the
#: ``action="exclude"`` path) -- drawn dashed (see :attr:`Canvas.ghost_item`'s pen) and
#: desaturated so discarded evidence stays visible without competing with what survived.
#: Visibility is gated on ``result.get("_show_ghosts", True)``, not the trails toggle: a ghost is
#: evidence about what a device removed, not a drift reading a user opted into. Exempt from the
#: Theme Rule for the same "what it IS, not chrome" reason as the palette above.
GHOST_COLOR = (120, 120, 120)       # excluded-chain ghosts: neutral, deliberately desaturated

#: The wavelet-bar DUO: theta, the smoothing function, in white; psi, the analyzing wavelet actually
#: convolved with the field, in the "green wavelet bar" green named in the 2026-08-05 design doc's
#: Map-panels house style. Before this slice there was one curve here, drawn in the theme's muted
#: `ink_muted` role because it was chrome (what the analysis is doing); now there are two curves
#: and WHICH KERNEL each one is is itself the point, so -- like the chain/extremum colors above --
#: they are literal RGB tuples, exempt from the Theme Rule as data identity, not a theme role.
SMOOTHER_COLOR = (235, 235, 235)    # theta (deriv_order=0): white-ish (same value as HCHAIN_COLOR)
ANALYZER_COLOR = (63, 174, 90)      # psi (deriv_order=1): DESIGN.md's wavelet-bar green

#: The bounding outline drawn around an ROI result's own region, once its overlays have been
#: translated into the parent raster's coordinates (see :meth:`Canvas.set_result`). Data, not
#: chrome, by the same reasoning as the chain/extremum colors above -- it says which region of THIS
#: raster the extrema beside it were measured over, which is a property of the result.
#:
#: Its VALUE is the theme's amber accent, and deliberately (the spec grants ROI selection the
#: One-Accent exception: "the box IS a selection"). It is its own constant rather than a reuse of
#: ``VTRAIL_COLOR`` -- an ROI boundary and a V-chain trail are different things, and sharing one
#: constant would mean re-tinting either could never be done without re-tinting the other.
#: ``tests/test_shell_boundaries.py`` pins BOTH to the theme accent, so the relationship is a
#: decision a future re-tint has to make on purpose (the convention that test established).
ROI_BOUNDS_COLOR = (255, 160, 40)

#: The COI (cone of influence) outline the canvas draws around a scale's dilated missing-data
#: region (dynamix/roi/halo.py's `_missing_mask` + `_coi_radii` -- see the module docstring's
#: "Missing data" section). A muted violet, distinct from every other data-layer color on screen
#: so a contamination boundary is never mistaken for a chain or an extremum -- exempt from the
#: Theme Rule for the same "what it IS, not chrome" reason as the palette above.
COI_COLOR = (160, 120, 200)

#: Display decimation target -- see lod_stride.
_DEFAULT_MAX_DIM = 2048

#: Corner-overlay metrics, in WIDGET (display) pixels: inset from the widget edge, and the gap
#: between a label and the item it annotates. Local constants rather than theme roles -- slice 1
#: centralizes colors and fonts only, and metric centralization is recorded next-slice debt (see
#: the spec's file list).
_MARGIN_PX = 12
_LABEL_GAP_PX = 8

#: SUPERSEDED (scale doctrine, spec 2026-08-10 sec 2). Used to be the half-width, in
#: multiples of the scale ``a``, that :meth:`Canvas._update_wavelet_bar` clipped a ``wtmm``-
#: rendered kernel to -- that kernel's numeric support ran to ~20a, which spanned most of a
#: zoomed view as an unlabelled squiggle, so +/-3a (where a first-derivative Gaussian family has
#: effectively decayed) clipped what was visually meaningless. The bar now draws
#: :func:`dynamix.core.scale_units.kernel_section`'s sections, which are already sampled over
#: their OWN support -- no separate clip is applied or needed, and nothing reads this constant
#: any more. Left defined, unused, per the project's no-deletion law rather than removed.
_WAVELET_SUPPORT_FACTOR = 3.0

#: Fixed on-screen data-HEIGHT the wavelet-bar duo's taller curve is normalized to, as a fraction
#: of the current view's y-range (see :func:`wavelet_bar_scale_factor` and
#: :meth:`Canvas._draw_wavelet`). Height carries NO reading -- only the WIDTH (true data px,
#: unclipped, sampled over :func:`dynamix.core.scale_units.kernel_section`'s own support) is a
#: footprint against the raster -- so this number is free to be whatever keeps the duo legible at
#: any zoom level.
WAVELET_BAR_HEIGHT_FRAC = 0.06


def _resolve_cmap(name):
    """A ``pg.ColorMap`` for ``name``, local-registry first then matplotlib, or None.

    One resolver for :meth:`Canvas.set_colormap` AND :meth:`Canvas._refresh_image`'s
    re-apply -- the two must never disagree, or a stretch/hillshade refresh silently
    reverts a matplotlib-named ramp to whatever LUT survived (the 2026-09-22 stuck-on-
    viridis bug, in refresh form)."""
    try:
        return pg.colormap.get(str(name))
    except Exception:
        try:
            return pg.colormap.get(str(name), source="matplotlib")
        except Exception:
            return None


def roi_from_corners(p0, p1, shape) -> tuple[int, int, int, int]:
    """Two dragged corners in DATA space -> ``(row, col, h, w)`` in IMAGE pixels, clamped.

    ``p0``/``p1`` are ``(x, y)`` data coordinates, in either order (a box dragged up-and-left is
    the same box as one dragged down-and-right). The canvas's data space IS image-pixel space by
    construction -- :meth:`Canvas.set_field` pins the ``ImageItem`` to ``QRectF(-0.5, -0.5, nx, ny)``
    whatever display decimation it chose (center registration: integer data coordinate j is the
    CENTER of pixel j), and every overlay is drawn in full-resolution pixel coordinates -- so x
    is a COLUMN and y is a ROW, with no scale factor between them.

    The near corner FLOORS and the far corner CEILS, so the rectangle covers every pixel the
    drawn box visibly touched rather than rounding a partly-covered edge pixel away. The result
    is then clamped into ``shape`` (``(ny, nx)``): a drag that starts or ends off the raster must
    never name pixels the file does not have -- the ROI transform reads those coordinates
    straight off the source, where an out-of-range window is a read error at best and someone
    else's data at worst.
    """
    ny, nx = int(shape[0]), int(shape[1])
    # Center registration (2026-09-21): pixel j's cell spans [j - 0.5, j + 0.5), so shifting the
    # data coordinates by +0.5 makes the original floor/ceil arithmetic compute exactly the
    # touched-cell range under the new convention.
    row0 = max(0, min(int(np.floor(min(p0[1], p1[1]) + 0.5)), ny))
    row1 = max(0, min(int(np.ceil(max(p0[1], p1[1]) + 0.5)), ny))
    col0 = max(0, min(int(np.floor(min(p0[0], p1[0]) + 0.5)), nx))
    col1 = max(0, min(int(np.ceil(max(p0[0], p1[0]) + 0.5)), nx))
    return row0, col0, row1 - row0, col1 - col0


def roi_offset(result: dict) -> tuple[int, int]:
    """``(row, col)`` of an ROI result's origin in its SOURCE FILE, or ``(0, 0)``.

    ``dynamix.roi.halo.run_wtmm2d_roi`` crops every scale's extrema onto the ROI's own grid, so
    everything in an ROI result -- extrema, chains, the missing mask -- is in ROI-LOCAL
    coordinates. This is where that grid sits, read from the result itself (``_roi["roi"]``, the
    window the transform was actually given) rather than from any UI state, so a result restored
    from a project file or pulled from the cache carries its own position with it.

    FILE coordinates, not screen ones: the ROI device addresses the raster on disk. Use
    :func:`display_offset` to draw with it.
    """
    roi = (result.get("_roi") or {}).get("roi")
    if not roi:
        return 0, 0
    return int(roi[0]), int(roi[1])


def window_offset(field) -> tuple[int, int]:
    """``(row, col)`` where ``field`` starts inside the file it was read from, or ``(0, 0)``.

    ``RasterField.from_geotiff_window`` records this in ``provenance["window"]``; a whole-file
    field (or a bare array, or ``None``) has no window and starts at the origin.
    """
    window = (getattr(field, "provenance", None) or {}).get("window") or {}
    return int(window.get("row_off", 0)), int(window.get("col_off", 0))


def display_offset(result: dict, field) -> tuple[int, int]:
    """Where an ROI result's overlays belong in the coordinates of the image actually on screen.

    Two frames meet here and neither is negotiable. An ROI result's ``_roi`` is FILE-absolute --
    the device reads its per-scale halos straight off the raster on disk, so that is the only
    frame it could be in. The image on screen is ``field``, which for a raster too large to load
    whole is itself a WINDOW of that same file. The overlay offset is therefore the DIFFERENCE,
    and using the raw ROI origin instead puts every chain, point and outline exactly one window
    offset too far out -- on BOEM, thousands of pixels, in a picture that otherwise looks
    entirely reasonable.

    A NON-ROI result gets ``(0, 0)`` even on a windowed field, and that asymmetry is the point:
    ``wtmm2d`` ran on the window itself, so its extrema are already in the displayed image's
    coordinates and subtracting anything would push them off in the opposite direction. The only
    thing that ever needs moving is a result computed against a frame other than the one being
    drawn.
    """
    if not (result.get("_roi") or {}).get("roi"):
        return 0, 0
    roi_row, roi_col = roi_offset(result)
    win_row, win_col = window_offset(field)
    return roi_row - win_row, roi_col - win_col


def _hline_base_geometry(base: dict, shape, runs=None):
    """``(hx, hy, vert_pos)`` for :meth:`Canvas._masked_hline_geometry`'s cache: the full
    ordering plus each vertex's flat pixel position (``row * nx + col``; NaN separators -> -1).
    ``runs`` -- the transform's own worker-side ordering (``result["_ext_base_runs"]``) -- makes
    this a pure concatenation; without it the ``_order_lines`` walk runs here, on the caller's
    (GUI) thread, which is exactly the landing freeze the stamp exists to prevent."""
    nx = int(shape[1])
    if runs is not None:
        # Display coordinates prefer the subpixel channels (2026-09-20 interpolate knob) --
        # the polyline vertices move off-grid, killing the integer-center staircase at zoom.
        # ``vert_pos`` stays derived from the INTEGER support below: it is the pixel IDENTITY
        # the filter-membership gather keys on, never a drawing coordinate.
        bx = np.asarray(base.get("x_sub", base["x"]), dtype=np.float64)
        by = np.asarray(base.get("y_sub", base["y"]), dtype=np.float64)
        bxi = np.asarray(base["x"], dtype=np.int64)
        byi = np.asarray(base["y"], dtype=np.int64)
        # Vectorized scatter (2026-09-14): the old ``for run in runs`` list.extend loop over
        # ~58k runs cost ~70 ms per scale (profiled) -- the scale-scrub lag. Lay all runs into
        # one pre-sized NaN-filled buffer, one point after each run and a NaN separator between,
        # with the per-point output offsets computed by vectorized cumsum/repeat (no Python loop
        # over runs beyond reading their lengths).
        lens = np.fromiter((r.size for r in runs), dtype=np.int64, count=len(runs))
        if lens.size == 0 or lens.sum() == 0:
            return (np.empty(0, np.float64), np.empty(0, np.float64),
                    np.empty(0, np.int64))
        order = np.concatenate(list(runs))               # every run's base indices, in order
        run_id = np.repeat(np.arange(lens.size), lens)
        starts_in = np.concatenate([[0], np.cumsum(lens)[:-1]])
        within = np.arange(int(lens.sum())) - starts_in[run_id]
        starts_out = np.concatenate([[0], np.cumsum(lens + 1)[:-1]])   # +1 = a NaN per run
        out_pos = starts_out[run_id] + within
        out_len = int((lens + 1).sum()) - 1              # drop the trailing separator
        hx = np.full(out_len, np.nan); hy = np.full(out_len, np.nan)
        vert_pos = np.full(out_len, -1, dtype=np.int64)
        keep = out_pos < out_len                         # the very last point's slot may be trimmed
        op = out_pos[keep]; od = order[keep]
        hx[op] = bx[od]; hy[op] = by[od]
        vert_pos[op] = byi[od] * nx + bxi[od]
        return hx, hy, vert_pos
    # Fallback (no worker-stamped runs): walk on the INTEGER coordinates -- vert_pos identity
    # must come from the support, and a rint over a half-pixel offset is ambiguous at ties --
    # then swap the drawn vertices to the subpixel channels by pixel-position lookup.
    base_int = {k: v for k, v in base.items() if k not in ("x_sub", "y_sub")}
    hx, hy = hline_polylines(base_int, shape)
    finite = np.isfinite(hx) & np.isfinite(hy)
    vert_pos = np.full(hx.shape, -1, dtype=np.int64)
    vert_pos[finite] = (hy[finite].astype(np.int64) * nx + hx[finite].astype(np.int64))
    if "x_sub" in base:
        ny = int(shape[0])
        pos = (np.asarray(base["y"], np.int64) * nx + np.asarray(base["x"], np.int64))
        fx = np.full(ny * nx, np.nan); fy = np.full(ny * nx, np.nan)
        fx[pos] = np.asarray(base["x_sub"], np.float64)
        fy[pos] = np.asarray(base["y_sub"], np.float64)
        hx = hx.copy(); hy = hy.copy()
        hx[finite] = fx[vert_pos[finite]]
        hy[finite] = fy[vert_pos[finite]]
    return hx, hy, vert_pos


def hline_polylines(ext: dict, shape) -> tuple[np.ndarray, np.ndarray]:
    """NaN-separated x/y arrays over every ordered H-line run at one scale.

    Mirrors the topology call site (``topology/wtmm.py`` ~55-80): a scratch grid maps every point
    to its row index, ``_order_lines`` groups and orders the LABELLED points (``line_id == -1``
    singletons are excluded upstream -- see its own docstring), and a branching line's several
    walk-restarts come back as separate contiguous runs of the same ``line_id``, split on
    ``np.diff(seg)``. Each run (including the boundary between two distinct lines) is followed by
    a NaN row, so one ``PlotDataItem`` with ``connect="finite"`` renders every H-line at this
    scale in a single draw call without stitching unrelated runs together.
    """
    ny, nx = shape
    x = np.asarray(ext["x"], dtype=np.int64)
    y = np.asarray(ext["y"], dtype=np.int64)
    line_id = np.asarray(ext["line_id"], dtype=np.int64)
    # The walk runs on the INTEGER support (grid adjacency is what orders a line); the DRAWN
    # vertices prefer the subpixel channels when the interpolate knob stamped them (2026-09-20).
    xd = np.asarray(ext.get("x_sub", x), dtype=np.float64)
    yd = np.asarray(ext.get("y_sub", y), dtype=np.float64)
    empty = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
    if x.size == 0:
        return empty

    grid = np.full(ny * nx, -1, dtype=np.int64)
    grid[y * nx + x] = np.arange(x.size, dtype=np.int64)
    # Private-symbol coupling: `_order_lines` is a PRIVATE name in core, and core is a VERBATIM
    # copy of EQSelect (never modify EQSelect, copies stay verbatim). A future re-copy
    # that renames this helper breaks canvas here too, and nothing in core's public surface warns
    # of it (topology/wtmm.py carries the same coupling, at its own call site).
    order, seg, starts = _order_lines(x, y, line_id, grid, nx, ny)
    if order.size == 0:
        return empty

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    nan = np.array([np.nan])
    for li in range(len(starts) - 1):
        sl = order[starts[li]:starts[li + 1]]
        if sl.size == 0:
            continue
        run_breaks = np.nonzero(np.diff(seg[starts[li]:starts[li + 1]]))[0] + 1
        for walk in np.split(sl, run_breaks):
            if walk.size == 0:
                continue
            xs.append(xd[walk])
            ys.append(yd[walk])
            xs.append(nan)
            ys.append(nan)
    if not xs:
        return empty
    return np.concatenate(xs[:-1]), np.concatenate(ys[:-1])       # drop the trailing separator


def cap_polylines(x, y, max_lines):
    """Bound a NaN-separated polyline array to the ``max_lines`` LONGEST polylines (EQSelect's
    ``_TE_MAX`` draw cap): the render cost of a dense DEM overlay is dominated by
    the sheer point count handed to pyqtgraph, and dropping the shortest lines keeps the picture
    readable while making a scale/filter scrub responsive.

    Returns ``(x, y, kept, total)`` where ``kept``/``total`` are the drawn/available line counts
    (equal, and no copy made, when already under the cap) -- the caller turns a shortfall into an
    honest "drawing N of M" status note rather than a silent truncation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return x, y, 0, 0
    seps = np.flatnonzero(~np.isfinite(x))
    bounds = np.concatenate([[-1], seps, [x.size]])
    starts, ends = bounds[:-1] + 1, bounds[1:]
    keep = ends > starts                                # drop empty spans between adjacent NaNs
    starts, ends = starts[keep], ends[keep]
    total = int(starts.size)
    if total <= int(max_lines):
        return x, y, total, total
    lengths = ends - starts
    top = np.argsort(lengths, kind="stable")[::-1][: int(max_lines)]
    top.sort()                                          # keep original draw order
    nan = np.array([np.nan])
    xs, ys = [], []
    for i in top:
        a, b = int(starts[i]), int(ends[i])
        xs.append(x[a:b]); xs.append(nan)
        ys.append(y[a:b]); ys.append(nan)
    return np.concatenate(xs[:-1]), np.concatenate(ys[:-1]), int(top.size), total


def _gather_polylines(px, py, off, idx, pts_keep):
    """NaN-separated polylines for the CSR chains in ``idx`` -- one bounded gather (the caller
    caps ``idx``), with a per-point keep mask NaN-ing dropped vertices so ``connect="finite"``
    breaks the line exactly there."""
    if len(idx) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    xs, ys = [], []
    nan = np.array([np.nan])
    off = np.asarray(off)
    for i in idx:
        a, b = int(off[i]), int(off[i + 1])
        gx = np.asarray(px[a:b], dtype=np.float64)
        gy = np.asarray(py[a:b], dtype=np.float64)
        if pts_keep is not None:
            m = pts_keep[a:b]
            gx = np.where(m, gx, np.nan)
            gy = np.where(m, gy, np.nan)
        xs.extend((gx, nan))
        ys.extend((gy, nan))
    return np.concatenate(xs[:-1]), np.concatenate(ys[:-1])


def vchain_trails(chains) -> tuple[np.ndarray, np.ndarray]:
    """NaN-separated x/y arrays, one drift polyline per V-chain (chain index k <-> scale index k)."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    nan = np.array([np.nan])
    for c in chains:
        cx = np.asarray(c["x"], dtype=np.float64)
        cy = np.asarray(c["y"], dtype=np.float64)
        if cx.size == 0:
            continue
        xs.append(cx)
        ys.append(cy)
        xs.append(nan)
        ys.append(nan)
    if not xs:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(xs[:-1]), np.concatenate(ys[:-1])


def classify_chains(chains) -> tuple[list, list, dict[tuple, list]]:
    """Partition ``chains`` into ``(plain, seam, grouped)`` for :meth:`Canvas.set_result` ("the session canvas colors groups like seam flags").

    A chain carrying a ``"group:<name>"`` tag (:class:`dynamix.devices.groups.GroupPaint`'s own
    stamp, always paired with a ``"group_color"`` entry on the same chain) is GROUPED -- bucketed
    by that color into ``grouped`` -- regardless of whatever ELSE is in ``tags``. A committed
    group is a deliberate user assertion; it outranks a classifier's machine seam flag on the same
    chain (documented in ``main_window.py``'s commit-transaction docstring as a binding choice).

    Everything else with any truthy ``tags`` (a probable-seam flag,
    :class:`dynamix.devices.chain_classify.ChainClassify`, never grouped) is ``seam``; everything
    untagged is ``plain`` -- exactly :meth:`Canvas.set_result`'s pre-Task-8 two-way split, now a
    named, independently testable function instead of an inline closure.
    """
    plain: list = []
    seam: list = []
    grouped: dict[tuple, list] = {}
    for c in chains:
        tags = c.get("tags") or ()
        if any(str(t).startswith("group:") for t in tags):
            color = tuple(int(v) for v in (c.get("group_color") or (128, 128, 128)))
            grouped.setdefault(color, []).append(c)
        elif tags:
            seam.append(c)
        else:
            plain.append(c)
    return plain, seam, grouped


def _composite_rgba(stack, spec: dict):
    """A multiband stack ``(my, mx, nc)`` as display RGBA under the composite law.

    ``spec``: ``r``/``g``/``b`` = 0-based band index feeding that channel (or ``None``),
    ``mute`` silences a band's channels, ``solo`` (DAW semantics) shows only the soloed
    bands -- ONE solo draws that band grayscale in all three channels, several keep each in
    its assigned channel. ``stretch`` (a :data:`~dynamix.core.stretch.STRETCHES` mode, default
    ``percent``) with its ``stretch_pct`` / ``stretch_k`` is ONE choice for all channels, each
    computed from its OWN band's statistics (the ENVI RGB convention). Alpha is 0 where no
    used band is finite. Returns pyqtgraph's (x, y) orientation, uint8."""
    from dynamix.core.stretch import STRETCHES, stretch

    my, mx, nc = stack.shape
    mode = spec.get("stretch") if spec.get("stretch") in STRETCHES else "percent"
    pct = float(spec.get("stretch_pct", 2.0) if spec.get("stretch_pct") is not None else 2.0)
    k = float(spec.get("stretch_k", 2.0) if spec.get("stretch_k") is not None else 2.0)
    solo = [b for b in (spec.get("solo") or []) if isinstance(b, int) and 0 <= b < nc]
    mute = {b for b in (spec.get("mute") or []) if isinstance(b, int)}
    assign = {c: spec.get(c) for c in ("r", "g", "b")}
    assign = {c: (b if isinstance(b, int) and 0 <= b < nc else None)
              for c, b in assign.items()}

    def norm(b):
        return np.clip(np.nan_to_num(stretch(stack[..., b], mode, percent=pct, k=k),
                                     nan=0.0), 0.0, 1.0)

    zeros = np.zeros((my, mx))
    if len(solo) == 1:
        plane = norm(solo[0])
        channels = [plane, plane, plane]
        used = set(solo)
    else:
        def keep(b):
            return b is not None and b not in mute and (not solo or b in solo)

        channels = [norm(assign[c]) if keep(assign[c]) else zeros for c in ("r", "g", "b")]
        used = {assign[c] for c in ("r", "g", "b") if keep(assign[c])}
    finite = np.zeros((my, mx), dtype=bool)
    for b in used:
        finite |= np.isfinite(stack[..., b])
    rgba = np.stack(channels + [finite.astype(np.float64)], axis=-1)
    return (rgba * 255.0).astype(np.uint8).transpose(1, 0, 2)


def lod_stride(shape, max_dim: int = _DEFAULT_MAX_DIM) -> int:
    """Display decimation stride so the larger raster dimension is <= ``max_dim`` (1 = full res).

    Ceiling division: a stride of exactly ``largest / max_dim`` truncated down can still leave the
    decimated axis one pixel over budget, so the stride rounds UP.
    """
    largest = max(int(shape[0]), int(shape[1]))
    if largest <= max_dim:
        return 1
    return -(-largest // max_dim)


def nice_round_scalebar(view_width_units: float) -> tuple[str, float]:
    """Largest 1/2/5 x 10^n length that fits within ~25% of ``view_width_units``.

    Ported math (``app_window.py`` ``_nice_round``, 168-174), adapted per the slice-1 decision:
    unit-agnostic (plain float in, plain float out) and returns the bare NUMBER label -- the
    caller appends the unit string, since this function (and Canvas) never sees physical units.
    """
    target = float(view_width_units) * 0.25
    if not np.isfinite(target) or target <= 0:
        return "1", 1.0
    exp = np.floor(np.log10(target))
    p = 10.0 ** exp
    f = target / p
    nice = 1.0 if f < 2.0 else 2.0 if f < 5.0 else 5.0
    value = float(nice * p)
    return f"{value:g}", value


def dilate_coi_mask(mask, radius: int) -> np.ndarray:
    """Grow a bool missing-data mask by ``radius`` px -- the COI (cone of influence) at one scale:
    everywhere within reach of an originally-missing (zero-filled) pixel, that scale's kernel
    support touched fabricated data.

    SQUARE (8-connectivity / Chebyshev) structuring element, ``iterations=radius`` -- the same
    choice ``dynamix.core.wtmm_backend``'s own ``extrema2d`` makes for its invalid-pixel dilation
    (``binary_dilation(invalid, structure=np.ones((3, 3)), iterations=radius)``, see that
    function's docstring): one iteration grows the region by one pixel in EVERY direction
    including diagonals, matching "the kernel's support reaches this pixel" better than the
    smaller diamond (L1 / 4-connectivity) scipy's default cross structuring element gives.
    Matching the copied core's own choice keeps one dilation semantics in the app, not two.

    ``scipy`` is imported LAZILY here, never at module scope, so importing
    ``dynamix.shell.canvas`` stays cheap even though scipy itself is a core dependency -- the same
    treatment this module already gives ``wtmm`` (an optional dependency) and that
    ``dynamix/roi/halo.py`` gives ``rasterio``.
    """
    from scipy.ndimage import binary_dilation

    mask = np.asarray(mask, dtype=bool)
    radius = int(radius)
    if radius <= 0:
        return mask
    return binary_dilation(mask, structure=np.ones((3, 3)), iterations=radius)


def coi_outline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-separated x/y arrays for the 0.5-isocontour of an (already dilated) bool mask.

    ``pyqtgraph.functions.isocurve(..., connected=False)`` runs marching squares and returns, in
    its own docstring's words, "a single long list of point pairs" -- one independent 2-point
    SEGMENT per grid-cell edge crossing, not stitched into polylines. Each segment gets its own
    NaN-separated pair, mirroring :func:`hline_polylines`/:func:`vchain_trails`, so one
    ``PlotDataItem`` with ``connect="finite"`` draws every segment without joining ones that
    happen to land near each other but share no actual path.

    ``isocurve`` returns each point as ``(i, j)`` indexing ``mask`` itself -- ``mask``'s FIRST axis
    (row) then its SECOND (column), pyqtgraph's own convention, not this module's. Every other
    overlay here (:func:`hline_polylines`, :func:`vchain_trails`, and
    :meth:`Canvas.set_result`'s ``extrema_item.setData``) draws ``x=column, y=row`` -- so the two
    components are swapped on the way out (``p[1]`` -> x, ``p[0]`` -> y), or the outline lands
    transposed relative to every point and polyline drawn beside it (invisible on a square mask
    near the array's center, wrong by however far off-square and off-center the real one is).
    """
    segments = pg.functions.isocurve(np.asarray(mask, dtype=np.float64), 0.5, connected=False)
    if not segments:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    nan = np.array([np.nan])
    for p1, p2 in segments:
        xs.append(np.array([p1[1], p2[1]], dtype=np.float64))
        ys.append(np.array([p1[0], p2[0]], dtype=np.float64))
        xs.append(nan)
        ys.append(nan)
    return np.concatenate(xs[:-1]), np.concatenate(ys[:-1])       # drop the trailing separator


#: Shared empty-chains sentinel: `result.get("chains") or []` would mint a NEW list every call,
#: giving the geometry cache a different identity per scrub tick for the SAME "no chains" answer --
#: an unbounded stream of cache misses for the cheapest possible geometry. One module-level empty
#: list (never mutated) keys every chainless result to one cache entry.
_NO_CHAINS: list = []


def wavelet_bar_scale_factor(theta: np.ndarray, psi: np.ndarray, target_height: float) -> float:
    """Common amplitude scale for the wavelet-bar duo.

    Both curves are multiplied by the SAME factor -- their RELATIVE amplitude is the one thing the
    wavelet math actually says, and normalizing each independently would destroy it -- chosen so
    the TALLER curve's peak-to-peak span becomes exactly ``target_height``. ``target_height``
    carries no reading of its own (see :meth:`Canvas._draw_wavelet`'s docstring); it exists purely
    so the duo stays legible at any zoom level.

    ``1.0`` (no-op) when either curve is empty or perfectly flat, or ``target_height`` is not a
    usable positive number -- there is nothing sensible to normalize against.
    """
    theta = np.asarray(theta, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    raw_span = max(float(np.ptp(theta)) if theta.size else 0.0,
                   float(np.ptp(psi)) if psi.size else 0.0)
    if (raw_span <= 0 or not np.isfinite(raw_span)
            or not np.isfinite(target_height) or target_height <= 0):
        return 1.0
    return float(target_height) / raw_span


class Canvas(pg.GraphicsLayoutWidget):
    """Raster + vectorized WTMM overlay, one ``ViewBox``: image, H-chains, isolated extrema,
    optional V-chain trails, and two corner-anchored reading overlays (distance + wavelet)."""

    #: Emitted after the camera moved and both corner overlays have been re-drawn. The window
    #: listens: the distance bar's LABEL is the one part of the reading this widget cannot
    #: recompute, because only the window knows the field's physical units.
    viewChanged = QtCore.Signal()

    #: A ⌘-drag finished: ``(row, col, h, w)`` in IMAGE pixels, clamped to the field (see
    #: :func:`roi_from_corners`). The window turns this into the precision panel; the canvas
    #: itself has no idea what an ROI is for.
    roiDrawn = QtCore.Signal(int, int, int, int)
    #: Armed placement (2026-08-30): a click stamped the hovering footprint --
    #: ``(row, col, h, w)`` in image pixels, same shape as ``roiDrawn``.
    roiPlaced = QtCore.Signal(int, int, int, int)
    _roi_place = None            # (h, w) while placement is armed; class default = off

    #: A plain click or shift-click finished -- the picked chain
    #: INDEX into :meth:`set_pick_chains`'s own list, or ``None`` on a miss, plus whether Shift
    #: was held. The window forwards this straight into ``GroupPalette.add_pick``, naming the
    #: ACTIVE layer (the canvas only ever shows the active layer's own result) -- the canvas
    #: itself has no idea what a group is.
    chainPicked = QtCore.Signal(object, bool)

    #: A lasso capture finished (either the pre-existing ⌥-shortcut in click mode, or a plain
    #: drag while :attr:`_selection_mode` is ``"lasso"``): every chain index
    #: :func:`dynamix.core.chain_pick.chains_in_polygon` found enclosed, possibly empty, plus
    #: whether the op is a subtract (⌥ held) -- EQSelect's own rule ("box/lasso = ADD by default, ⌥ = subtract"). The window forwards the whole batch through
    #: one ``GroupPalette.apply_picks`` call. The click-mode ⌥-SHORTCUT always reports
    #: ``subtract=False``: Alt is what TRIGGERS that gesture at all there, so it can never also
    #: mean "subtract" without silently flipping every pre-existing shortcut-lasso from add to
    #: subtract -- see :meth:`mouseReleaseEvent`'s own comment.
    chainsLassoed = QtCore.Signal(list, bool)

    #: A box-mode drag finished (only fires while :attr:`_selection_mode` is ``"box"``): every
    #: chain index :func:`dynamix.core.chain_pick.chains_in_box` found enclosed, possibly empty,
    #: plus whether the op is a subtract (⌥ held during the drag) -- same shape and same EQ§1
    #: doctrine as :attr:`chainsLassoed`, over a rectangle instead of a polygon.
    chainsBoxed = QtCore.Signal(list, bool)

    #: The SECOND click of a transect-mode gesture landed
    #: -- ``(a, b)``, each an ``(x, y)`` DATA-space (pixel) tuple, already AUTO-ORIENTED
    #: (:func:`dynamix.core.transect.orient_endpoints`). The window forwards this straight into
    #: ``TransectPanel.add_record`` -- the canvas itself has no idea what a transect list is.
    transectDrawn = QtCore.Signal(tuple, tuple)

    def __init__(self, parent=None):
        # pyqtgraph's GraphicsView.__init__ fires a synchronous resizeEvent() before this method
        # gets to build any child widgets -- this guard lets _reposition_overlays no-op for that
        # one call rather than needing scale_bar_label to already exist.
        self.scale_bar_label = None
        self.wavelet_label = None
        self._wavelet_curve = None
        self._field = None                 # set by set_field(); None until a layer is loaded
        self._image_stride = 1             # set_field's decimation, reused by _refresh_image
        self._composite = None             # multiband display law (set_composite); None = band 1
        self._hillshade = (False, 315.0, 45.0, 1.0)   # (on, sun azimuth, sun altitude, z_factor)
        self._stretch = ("linear", 2.0)             # (core.stretch mode, percent clip)
        self._levels = ""                            # density-slice spec text ("" = off)
        self._levels_colors = ""                     # "#rrggbb,..." per-class colors ("" = LUT)
        self._levels_sieve = 0                       # min island px (0 = off; settle/apply only)
        # Derived overlay geometry, memoized on the INPUT OBJECT's identity -- same doctrine as
        # the COI cache below (results are cached and immutable; identity is the cheap honest
        # key), with the same strong-ref discipline: `_geom_refs` anchors every cached input so
        # a reused id() can never alias a dead object's arrays. FIFO-capped: a scrub session
        # touches one result's handful of objects, so 64 is generous, and eviction only ever
        # costs a rebuild, never a wrong picture.
        self._geom_cache: dict[tuple, tuple] = {}
        self._geom_refs: dict[tuple, object] = {}
        # Draw cap (EQSelect _TE_MAX): the selection fast path bounds
        # each family's drawn chains at this many, keeping the longest / most persistent.
        # ``cap_note`` is the honest "drawing X of Y" reading for whoever owns a status line --
        # None whenever nothing was withheld.
        self.draw_cap: int = _DRAW_CAP
        self.cap_note: str | None = None
        # scale_idx -> (dilated mask, outline x, outline y), all for the ONE mask array below.
        self._coi_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # STRONG ref to the `_missing_mask` array the cache belongs to -- see _update_coi_outline's
        # docstring for why this is keyed on the MASK's identity, not the result dict's.
        self._coi_mask_ref = None
        self._roi_press = None             # (x, y) data-space corner of a ⌘-drag in progress
        # The pick gesture's own state -- see mousePressEvent/
        # mouseMoveEvent/mouseReleaseEvent. `_pick_chains` is the result's own `chains` list
        # (the `dynamix.core.chain_pick` fixture shape), fed by `set_pick_chains`; `None`
        # until a result lands, and cleared wherever the window clears the overlays themselves
        # (see that method's own docstring). `_press_pos`/`_press_shift` are a plain left-button
        # press's SCREEN-pixel position and Shift state, consumed (and reset) by the matching
        # release; `_lasso_points` is the data-space polyline an ⌥-drag is building, `None` when
        # no lasso is in progress.
        self._pick_chains = None
        # Pin-in-place: the canvas data space is FILE-ABSOLUTE. ``_base_off`` = the displayed
        # field's own window origin (row, col; (0,0) for a whole-file field) -- the image
        # and every field-anchored gesture live at/through it. ``_draw_off`` = the offset
        # the CURRENT result's overlays were drawn at (display_offset + base), for the
        # result-anchored gestures (chain picks, box/lasso selection).
        self._base_off = (0, 0)
        self._draw_off = (0.0, 0.0)
        self._press_pos = None
        self._press_shift = False
        self._lasso_points = None
        # Which gesture a plain (no-⌘) left-button drag performs --
        # MainWindow.set_selection_mode is the validating, hotkey/button-synced setter (raises
        # on an unknown mode); this widget's own `set_selection_mode` just mirrors whatever
        # string it is handed, with no validation of its own. `_box_press` is the box-mode
        # sibling of `_lasso_points`/`_roi_press`: the data-space press corner of a box-mode
        # drag in progress, `None` when no such drag is live.
        self._selection_mode = "click"
        self._box_press = None
        # The transect gesture's own in-progress state -- A's data-space
        # point once the FIRST click has landed, `None` before it or once the SECOND click closes
        # the gesture (see `_handle_transect_click`). Cleared by `cancel_transect` (Esc, wired at
        # the MainWindow level) and by `set_selection_mode` moving away from "transect" (a stray
        # marker left over from a mode switch mid-gesture would otherwise dangle forever).
        self._transect_a = None
        # One PlotDataItem per distinct committed-group color IN USE, built lazily in
        # set_result (the color set isn't known until a painted result lands) and never removed
        # once created -- a later result with fewer/no grouped colors just clears the ones it no
        # longer uses (see _draw_grouped_trails), so this stays small and bounded by how many
        # distinct colors any committed group has ever used in this session, mirroring
        # `_group_items`'s own "no view-item churn" reasoning below. `_overlay_opacity`/
        # `_overlay_line_width` remember the last `set_display_style` call so a group item created
        # AFTER that call still starts styled correctly, rather than at the item defaults the
        # four items built right here already have applied retroactively.
        self._group_items: dict[tuple, pg.PlotDataItem] = {}
        self._overlay_opacity = 1.0
        self._overlay_line_width = 1.0
        super().__init__(parent)
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)

        self._colormap = None
        self.image_item = pg.ImageItem()
        self.set_colormap("viridis")     # the Signal Rule default: never rainbow -- the canvas makes
                                          # this a per-layer setter; see its own docstring
        self.view.addItem(self.image_item)

        # Raster-view extrema as PIXELS: each
        # surviving extremum lights its own cell of the ORIGINAL grid, xsmurf ext-image
        # style -- H-line members in HCHAIN_COLOR, orphan dots in EXTREMA_COLOR. Built in
        # set_result; the rect carries the ROI/window offset and center registration.
        self.extrema_raster_item = pg.ImageItem()
        self.view.addItem(self.extrema_raster_item)

        # INVISIBLE in this view (2026-09-22): interpolated polylines and the line-width
        # style are the VECTOR view's rendering of extrema. The items stay constructed and
        # fed -- their memoized geometry serves the selection machinery and the tests that
        # pin it -- they just do not draw in the raster view.
        self.hchain_item = pg.PlotDataItem(pen=pg.mkPen(HCHAIN_COLOR), connect="finite")
        self.hchain_item.setVisible(False)
        self.view.addItem(self.hchain_item)

        self.vtrail_item = pg.PlotDataItem(pen=pg.mkPen(VTRAIL_COLOR), connect="finite")
        self.vtrail_item.setVisible(False)
        self.view.addItem(self.vtrail_item)

        # Seam-flagged V-chain trails (see SEAM_COLOR above): no visibility toggle of its own --
        # unlike vtrail_item, it stays on regardless of `_show_trails`, which is the entire point
        # of flagging (set_result never calls setVisible(False) on it).
        self.seam_item = pg.PlotDataItem(pen=pg.mkPen(SEAM_COLOR), connect="finite")
        self.view.addItem(self.seam_item)

        # Excluded-chain ghosts (see GHOST_COLOR above), dashed so a discarded chain reads as
        # discarded rather than as more of the same drift data. Visibility follows
        # `result["_show_ghosts"]`, set in set_result.
        self.ghost_item = pg.PlotDataItem(pen=pg.mkPen(GHOST_COLOR, style=QtCore.Qt.DashLine),
                                           connect="finite")
        self.view.addItem(self.ghost_item)

        # The VISIBLE selection overlay -- the picked
        # chains' own polylines redrawn a SECOND time, on top of vtrail_item/seam_item/
        # whichever item actually drew them, in the `selection_accent` role at 2x the current
        # overlay line width (see :meth:`set_selection_chains`/:meth:`set_display_style`). A
        # separate item rather than re-styling vtrail_item in place: the ordinary trail must
        # stay visible underneath (a de-selected chain should not just vanish), and a single
        # PlotDataItem can only ever carry one pen -- the same "always on top of the routine
        # trail" reasoning seam_item/ghost_item already establish for their own overlays.
        # Constructed at width 2.0 (2x `_overlay_line_width`'s own default of 1.0, set above) so
        # it matches its own contract even before any `set_display_style` call ever runs -- that
        # method (see its own docstring) is what keeps the 2x relationship true afterward.
        self.selection_item = pg.PlotDataItem(
            pen=pg.mkPen(RESTRAINED_DARK.selection_accent, width=2.0), connect="finite")
        self.view.addItem(self.selection_item)

        self.extrema_item = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(*EXTREMA_COLOR))
        self.extrema_item.setVisible(False)      # raster view draws pixels, not dots (2026-09-22)
        self.view.addItem(self.extrema_item)

        # The point-layer scatter overlay -- one array-fed ScatterPlotItem, shown only
        # when the raster currently displayed IS the bound target of a point layer's own
        # backproject result (see MainWindow._apply). Same construction shape as extrema_item
        # (size=3 default, no pen -- a filled dot, no outline), styled by set_points_style and
        # fed by set_points_result, both called from the window, never from set_result itself:
        # a point layer's result carries no "extrema" key at all.
        self.points_item = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(*POINTS_COLOR))
        self.view.addItem(self.points_item)
        # Reference layers (2026-08-29): interpretation drawn OVER the data in this raster's
        # pixel frame -- one item per layer, keyed by the project's ref_id (see set_reference_layers).
        self.reference_items: dict[str, object] = {}

        # COI (cone of influence) outline: the missing-data contamination boundary at the
        # displayed scale (see _update_coi_outline). Empty (setData([], [])) whenever a result
        # carries neither `_missing_mask` nor `_coi_radii` -- it rides the same "empty data draws
        # nothing" visibility every other overlay item on this ViewBox already uses, no separate
        # show/hide toggle of its own.
        self.coi_item = pg.PlotDataItem(pen=pg.mkPen(COI_COLOR), connect="finite")
        self.view.addItem(self.coi_item)

        # The ROI bounds outline: which region of the parent raster the displayed result was
        # measured over (see set_result's docstring on in-context rendering). Empty for every
        # non-ROI result, riding the same "empty data draws nothing" visibility as the COI item.
        self.roi_bounds_item = pg.PlotDataItem(pen=pg.mkPen(ROI_BOUNDS_COLOR), connect="all")
        self.view.addItem(self.roi_bounds_item)

        # The ROI rubber band. AMBER, and by the One-Accent Rule rather than in spite of it: the
        # spec grants this the accent explicitly ("the only amber on screen wins its exception
        # here: the box IS a selection"), and a selection is precisely what the accent is for. A
        # theme ROLE, not a literal -- unlike the chain/extremum colors above, this is chrome.
        self.roi_band_item = pg.PlotDataItem(pen=pg.mkPen(RESTRAINED_DARK.amber))
        self.view.addItem(self.roi_band_item)
        # SAVED ROIs of the displayed dataset -- dashed amber outlines,
        # the active one solid, each labelled; FILE pixels (the canvas's own data space).
        self.saved_roi_item = pg.PlotDataItem(
            pen=pg.mkPen(RESTRAINED_DARK.amber, style=QtCore.Qt.PenStyle.DashLine),
            connect="finite")
        self.view.addItem(self.saved_roi_item)
        self.active_roi_item = pg.PlotDataItem(pen=pg.mkPen(RESTRAINED_DARK.amber, width=2),
                                               connect="finite")
        self.view.addItem(self.active_roi_item)
        self._saved_roi_labels: list = []

        # The live lasso polyline during an ⌥-drag pick gesture
        # (now ALSO the lasso-MODE gesture -- see mousePressEvent). Its
        # own item rather than a reuse of roi_band_item because the two gestures (⌘-drag ROI,
        # lasso pick) are conceptually different selections, even though they can never be live
        # at the same time; keeping them separate means neither's add/remove idiom has to reason
        # about the other's state. Pen migrated from `RESTRAINED_DARK.amber` to the dedicated
        # `selection_accent` role: a chain PICK is a different kind of
        # selection from an ROI region, and the two must be able to re-tint independently -- see
        # `theme.py`'s own comment on `selection_accent` for the full reasoning.
        self.lasso_item = pg.PlotDataItem(pen=pg.mkPen(RESTRAINED_DARK.selection_accent))
        self.view.addItem(self.lasso_item)

        # The box-mode rubber-band -- the SAME rectangle-drawing idiom
        # `roi_band_item`/`_draw_roi_band` already use (a closed 5-point polyline through two
        # corners), on its OWN item in the `selection_accent` role rather than a reuse of either
        # `roi_band_item` (amber, a DIFFERENT selection concept -- see just above) or `lasso_item`
        # (a different SHAPE of the same chain-pick concept; kept separate so neither gesture's
        # add/remove idiom has to reason about the other's state, exactly as `lasso_item`'s own
        # comment already argues for the ROI/lasso split).
        self.box_item = pg.PlotDataItem(pen=pg.mkPen(RESTRAINED_DARK.selection_accent))
        self.view.addItem(self.box_item)

        # Persisted transects. `transect_item` draws every
        # VISIBLE transect's line, NaN-separated (mirrors `vchain_trails`'s own multi-polyline
        # idiom), at the ordinary `selection_accent` width; `transect_highlight_item` redraws the
        # SELECTED one a second time on top at 2x width -- the identical "same accent, thicker, on
        # top" idiom `selection_item` already establishes for picked chains, applied here to
        # persisted lines instead. `transect_marker_item` is the IN-PROGRESS first-click marker (a
        # single point) -- cleared the instant the second click closes the gesture or Esc cancels
        # it; the FINISHED line is drawn by `transect_item`/`transect_highlight_item` once
        # `set_transects` is called with the new record, never by this marker.
        self.transect_item = pg.PlotDataItem(
            pen=pg.mkPen(RESTRAINED_DARK.selection_accent), connect="finite")
        self.view.addItem(self.transect_item)
        self.transect_highlight_item = pg.PlotDataItem(
            pen=pg.mkPen(RESTRAINED_DARK.selection_accent, width=2.0), connect="finite")
        self.view.addItem(self.transect_highlight_item)
        self.transect_marker_item = pg.ScatterPlotItem(
            size=8, pen=None, brush=pg.mkBrush(RESTRAINED_DARK.selection_accent))
        self.view.addItem(self.transect_marker_item)

        # Chrome (not data): the scale-bar line is a UI reading, so its color is a theme ROLE,
        # never a literal -- same rule as the QLabel readings below.
        self.scale_bar_line = pg.PlotDataItem(pen=pg.mkPen(RESTRAINED_DARK.ink, width=2))
        self.view.addItem(self.scale_bar_line)

        # DATA, not chrome (see SMOOTHER_COLOR/ANALYZER_COLOR's comment above): theta (the
        # smoothing curve) and psi (the analyzing wavelet) each get their own data-layer color
        # rather than sharing the single muted `ink_muted` role the lone pre-slice-2 curve used.
        self.smoother_item = pg.PlotDataItem(pen=pg.mkPen(SMOOTHER_COLOR))
        self.view.addItem(self.smoother_item)
        self.wavelet_item = pg.PlotDataItem(pen=pg.mkPen(ANALYZER_COLOR))
        self.view.addItem(self.wavelet_item)

        # Built BEFORE scale_bar_label: the `is None` guard above keys off scale_bar_label, so
        # every label _reposition_overlays touches has to exist by the time that one does.
        self.wavelet_label = QtWidgets.QLabel("", self)
        self.wavelet_label.setProperty("reading", "true")
        self.wavelet_label.setVisible(False)

        self.scale_bar_label = QtWidgets.QLabel("", self)
        self.scale_bar_label.setProperty("reading", "true")
        self.scale_bar_label.setVisible(False)

        self._show_trails = False
        self.view.sigRangeChanged.connect(self._on_view_range_changed)
        self._reposition_overlays()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlays()

    def _on_view_range_changed(self, *args) -> None:
        """Zoom or pan moved the camera, so BOTH corner overlays are stale: the distance bar's
        length is a function of the visible width, and the wavelet footprint is anchored to a
        visible corner. Without this they froze at whatever the view was when the last resolve
        landed -- a bar claiming "500 m" over a view now 40 m wide.

        A bound method, never a lambda (the shell's uniform rule). It cannot recurse: ``view.
        autoRange()`` leaves auto-range DISABLED, so giving an item new data never moves the range
        that just changed.
        """
        self._draw_scale_bar()
        self._draw_wavelet()
        self.viewChanged.emit()

    def _reposition_overlays(self) -> None:
        """Corner-anchored in WIDGET (display) pixels, not data coordinates -- the VTK original
        this replaces addressed its scale bar the same way, precisely so it never drifts with the
        camera/view (app-window-audit.md 2.4).

        Each drawn item is then placed ADJACENT to its own label rather than at a fixed fraction
        of the data range. The distance bar used to be drawn at 5% up from the data-space y
        minimum, which under ``invertY`` is the TOP of the screen, while its label sat at the
        bottom of the widget: the number and the length it described were at opposite corners.
        """
        if self.scale_bar_label is None:
            return
        self.wavelet_label.adjustSize()
        self.wavelet_label.move(_MARGIN_PX, _MARGIN_PX)
        self.scale_bar_label.adjustSize()
        self.scale_bar_label.move(
            _MARGIN_PX, max(0, self.height() - _MARGIN_PX - self.scale_bar_label.height()))
        self._draw_scale_bar()
        self._draw_wavelet()

    def _data_at(self, x_px: float, y_px: float) -> QtCore.QPointF:
        """Widget (display) pixel -> data coordinate, through the scene.

        The overlays are POSITIONED in widget pixels, so they hold their corner under any camera;
        the items that draw them live in the ``ViewBox``, whose coordinates are data units. This
        is the single conversion between the two, and it is what keeps a bar next to its label.
        """
        return self.view.mapSceneToView(self.mapToScene(QtCore.QPoint(int(x_px), int(y_px))))

    # -- the ROI / pick / lasso gestures -----------------------------------------------------
    #
    # Handled HERE, on the widget (a ``QGraphicsView``), rather than by overriding ``ViewBox.
    # mouseDragEvent``. Two reasons, both about what a modified gesture must NOT do. First, the
    # drag may not pan: the pan lives behind the graphics SCENE (``ViewBox.mouseDragEvent``) *and*
    # behind ``pyqtgraph.GraphicsView.mouseMoveEvent``'s own left-button translate, and simply not
    # calling ``super()`` here stops both at once -- the scene never sees the event at all, so
    # there is no second path to remember to suppress. Second, the ViewBox route would mean
    # constructing pyqtgraph's own ``MouseDragEvent`` wrappers to test it, where this route takes
    # a plain ``QMouseEvent`` a test can build in one line.
    #
    # ``GraphicsView.lastMousePos`` bookkeeping is deliberately left alone: it is re-seeded by the
    # next ``mousePressEvent`` (a pan can only start with one), so skipping it during an ROI/lasso
    # drag cannot leak a stale delta into a later pan.
    #
    # Two more gestures share the same three methods: a plain
    # left click/shift-click (no modifier -- a PICK, disambiguated from a pan-drag purely by how
    # little the pointer moved between press and release, :data:`_CLICK_MAX_TRAVEL_PX`) and an
    # ⌥-drag (a LASSO, :data:`LASSO_MODIFIER`). Priority order in every handler below is: ⌘ (ROI)
    # first, then ⌥ (lasso), then plain (pick-or-pan) -- the two modified gestures are checked (and
    # can each ``return`` before ``super()`` ever sees the event) exactly like the ROI branch
    # already did; the plain branch never intercepts at all, so a pan is never suppressed by
    # picking being wired in.

    def _roi_modifier_held(self, event) -> bool:
        return (event.button() == QtCore.Qt.MouseButton.LeftButton
                and bool(event.modifiers() & ROI_MODIFIER)
                and self._field is not None)

    def _data_xy(self, event) -> tuple[float, float]:
        point = self._data_at(event.position().x(), event.position().y())
        return point.x(), point.y()

    def _field_shape(self) -> tuple[int, int]:
        # FILE pixels: a display picture reports its file's dims.
        return native_shape(self._field)

    def _pick_radius_data(self, screen_px: float) -> float:
        """SCREEN pixels -> DATA units, through the ViewBox's OWN current scale.

        ``ViewBox.viewPixelSize()`` returns the ``(dx, dy)`` size of one screen pixel in data
        units at whatever zoom the view is currently at -- exactly what a fixed SCREEN-pixel
        click tolerance (:data:`_PICK_MAX_DIST_PX`) needs converted before it can be handed to
        :func:`dynamix.core.chain_pick.pick_chain`, which works in the chains' own data/pixel
        space. ``self.view`` is already a plain ``pg.ViewBox`` (``self.addViewBox()``, never
        wrapped in a ``PlotItem``), so this is called directly -- no ``.vb`` indirection.

        ``dx``/``dy`` are averaged rather than picked independently: the ``ViewBox`` is aspect-
        locked (``__init__``'s ``setAspectLocked(True)``), so the two agree to floating-point
        noise in practice, and averaging is the honest thing to do on the rare view state where
        they do not (a resize mid-gesture) rather than favoring one axis arbitrarily.
        """
        dx, dy = self.view.viewPixelSize()
        return float(screen_px) * (abs(dx) + abs(dy)) / 2.0

    def _draw_roi_band(self, p0, p1) -> None:
        """The live amber box: one closed rectangle through the two corners, in data space."""
        (x0, y0), (x1, y1) = p0, p1
        self.roi_band_item.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])

    def _draw_selection_box(self, p0, p1) -> None:
        """The live box-mode rubber-band -- the SAME closed-rectangle
        idiom :meth:`_draw_roi_band` uses, targeting :attr:`box_item` (the ``selection_accent``
        role) instead of :attr:`roi_band_item` (the ⌘-drag ROI's own amber)."""
        (x0, y0), (x1, y1) = p0, p1
        self.box_item.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])

    def _draw_lasso(self) -> None:
        """The live lasso polyline: every data-space point captured so far in an ⌥-drag, or an
        empty draw once the gesture ends (mirrors :meth:`_draw_roi_band`'s own add/remove idiom,
        just an open polyline instead of a closed rectangle -- the gesture is still IN PROGRESS,
        so closing it early would show a shape the drag has not actually traced yet)."""
        if not self._lasso_points:
            self.lasso_item.setData([], [])
            return
        xs = [p[0] for p in self._lasso_points]
        ys = [p[1] for p in self._lasso_points]
        self.lasso_item.setData(xs, ys)

    _GEOMETRY_CACHE_MAX = 64

    def _selection_geometry(self, result, scale_idx):
        """``((hx, hy), (vx, vy))`` drawn straight off the chain product's index selection:
a bounded gather of at most :attr:`draw_cap` chains per
        family -- EQSelect's list-index materialization -- with :attr:`cap_note` set to an
        honest "drawing X of Y" whenever the cap withheld anything. ``None`` when the result
        carries no live selection; the caller then takes the legacy geometry paths."""
        from dynamix.core.chain_product import capped_h, capped_v, selection_of

        sel = selection_of(result) if isinstance(result, dict) else None
        prod = result.get("chain_product") if isinstance(result, dict) else None
        if sel is None or prod is None:
            self.cap_note = None
            return None
        si = int(result.get("_scale_idx", scale_idx))
        h_scale = np.asarray(prod["h_scale"])
        keep_h = sel["keep_h"]
        at_scale = (np.flatnonzero(h_scale == si) if keep_h is None
                    else keep_h[h_scale[keep_h] == si])
        h_idx, h_total = capped_h(prod, at_scale, self.draw_cap)
        v_idx, v_total = capped_v(prod, sel["keep_v"], self.draw_cap)
        notes = []
        if len(h_idx) < h_total:
            notes.append(f"drawing {len(h_idx)} of {h_total} H chains")
        if len(v_idx) < v_total:
            notes.append(f"drawing {len(v_idx)} of {v_total} V chains")
        self.cap_note = (" · ".join(notes) + " — tighten filters to see the rest"
                         if notes else None)
        # H display prefers the subpixel columns (2026-09-20 interpolate knob); V trails stay on
        # the integer chain coordinates -- chains carry no float channel yet (recorded follow-up).
        return (_gather_polylines(prod.get("h_xf", prod["h_x"]),
                                  prod.get("h_yf", prod["h_y"]), prod["h_off"], h_idx,
                                  sel["keep_h_pts"]),
                _gather_polylines(prod["v_x"], prod["v_y"], prod["v_off"], v_idx, None))

    def _masked_hline_geometry(self, base, ext, shape, runs=None):
        """H-line polylines as the BASE layer's cached ordering with the filter applied as a NaN
        mask (2026-08-30: slow next to EQSelect): the `_order_lines` walk over
        millions of points is paid once per computed stack (``result["_ext_base"]`` is id-stable
        across filter tweaks -- ``ScaleSelect``'s stamp), and every tweak after that is an O(n)
        membership test -- EQSelect's mask-the-fixed-geometry strategy. A vertex whose point the
        filter dropped goes NaN, so ``connect="finite"`` breaks the line exactly there -- the same
        picture a re-walk draws (two points bridged by a dropped one are not grid-adjacent, so the
        walk would not join them either)."""
        nx = int(shape[1])
        ny = int(shape[0])
        hx_full, hy_full, vert_pos = self._cached_geometry(
            "hline_base", base, lambda: _hline_base_geometry(base, shape, runs))
        if ext is base or len(ext["x"]) == len(base["x"]):
            return hx_full, hy_full                       # nothing filtered away
        # TRUE O(n) membership (2026-09-14): the docstring above always PROMISED O(n), but the
        # implementation was a per-tweak np.sort + searchsorted -- O(n log n), measured at 144 ms
        # on a 1M-point DEM finest scale, THE reason a scale/filter scrub lagged so badly vs
        # EQSelect (which masks by index, not by re-hashing positions). Scatter the surviving
        # points into a boolean raster grid once, then gather by the base ordering's own flat
        # positions: two O(n) passes, no sort. ``vert_pos`` carries -1 at the NaN run separators,
        # guarded so it never gathers a real cell.
        grid = np.zeros(ny * nx, dtype=bool)
        grid[np.asarray(ext["y"], np.int64) * nx + np.asarray(ext["x"], np.int64)] = True
        keep = np.zeros(vert_pos.size, dtype=bool)
        valid = vert_pos >= 0
        keep[valid] = grid[vert_pos[valid]]
        return np.where(keep, hx_full, np.nan), np.where(keep, hy_full, np.nan)

    def _cached_geometry(self, kind: str, obj, build):
        """Arrays from ``build()``, memoized under ``(kind, id(obj))``.

        Valid because every ``obj`` handed in comes out of an engine-cached result, which the
        zero-cache-miss law makes immutable once computed -- identity therefore implies identical
        geometry. The strong reference in ``_geom_refs`` is load-bearing: without it a dict keyed
        on ``id()`` could serve a NEW object the arrays of a garbage-collected old one."""
        key = (kind, id(obj))
        hit = self._geom_cache.get(key)
        if hit is not None:
            return hit
        arrays = build()
        self._geom_cache[key] = arrays
        self._geom_refs[key] = obj
        while len(self._geom_refs) > self._GEOMETRY_CACHE_MAX:
            oldest = next(iter(self._geom_refs))
            self._geom_refs.pop(oldest, None)
            self._geom_cache.pop(oldest, None)
        return arrays

    def set_display_style(self, *, opacity: float = 1.0, point_size: float = 3.0,
                          line_width: float = 1.0) -> None:
        """Apply per-layer VIEW styling to the overlay items — never to the raster.

        ``opacity`` fades every overlay together (the raster stays put: fading the DATA to see
        the data makes no sense; fading the measurement drawn over it does). ``point_size`` is
        the extrema dots' pixel size; ``line_width`` the pen width shared by every polyline
        overlay, each keeping its own color/style (the seam's hue, the ghost's dashes). View
        state only: nothing here touches results, caches, or the analysis path.

        ``self._group_items`` rides the same two loops as every other polyline item, PLUS
        it is remembered in ``_overlay_opacity``/``_overlay_line_width`` so a group item created
        by a LATER ``set_result`` (its color not known until a painted result lands) still starts
        styled to match, instead of at item-construction defaults.

        ``selection_item`` keeps its OWN color (``selection_accent``,
        never one of the loop's own pens) but follows ``line_width`` too, at 2x it -- "2x the
        current overlay width", not a fixed number, so a wider/narrower overlay
        setting keeps the selection legibly thicker than the trail it sits on top of either way.
        """
        self._overlay_opacity = float(opacity)
        self._overlay_line_width = float(line_width)
        for item in (self.extrema_item, self.hchain_item, self.vtrail_item,
                     self.seam_item, self.ghost_item, *self._group_items.values()):
            item.setOpacity(float(opacity))
        self.extrema_item.setSize(float(point_size))
        for item in (self.hchain_item, self.vtrail_item, self.seam_item, self.ghost_item,
                     *self._group_items.values()):
            pen = item.opts["pen"]
            pen = pg.mkPen(pen)
            pen.setWidthF(float(line_width))
            item.setPen(pen)
        selection_pen = pg.mkPen(self.selection_item.opts["pen"])
        selection_pen.setWidthF(2.0 * float(line_width))
        self.selection_item.setPen(selection_pen)

    def set_colormap(self, name: str) -> None:
        """Per-layer image colormap (``ui.colormap``) -- swaps ``image_item``'s LUT in
        place. Replaces the constructor's old one-time hard-set viridis: every layer switch now
        calls this with whatever the layer's own tag says (``_display_style_of``'s default is
        still "viridis", so an untouched layer looks exactly as it did before this existed).

        An unknown/garbled name (one ``pyqtgraph`` has no colormap file for) leaves whatever LUT
        is already applied untouched rather than raising -- the same tolerance
        ``_display_style_of`` already extends to a garbled tag value: a renamed or misspelled
        colormap name must never crash a layer switch, only fail to change anything.
        ``pg.ImageItem.setColorMap`` itself only assigns its internal colormap AFTER resolving
        the name, so a lookup failure there already leaves ``image_item`` exactly as it was --
        this method's own ``try`` just keeps that failure from propagating to the caller.
        """
        # 2026-09-22: the
        # ramp combo lists EVERY matplotlib colormap (right_panel, 2026-09-20 request), but
        # pyqtgraph's own registry holds only a handful of local maps -- every other name
        # raised here and was silently swallowed, leaving viridis. Resolve local-first (the
        # historical behavior for viridis/magma/...), then through matplotlib.
        cmap = _resolve_cmap(name)
        if cmap is None:
            return
        self.image_item.setColorMap(cmap)
        self._colormap = str(name)
        if self._hillshade[0]:
            self._refresh_image()          # the shaded RGBA bakes the colormap in

    def set_hillshade(self, enabled: bool, azimuth: float = 315.0, altitude: float = 45.0,
                      z_factor: float = 1.0) -> None:
        """Shaded relief under the overlays (2026-08-29): the raster drawn as colormap × hillshade
        (``dynamix.core.hillshade``) in RGBA, sun from ``azimuth``/``altitude``, slopes scaled by
        ``z_factor``. Uses the field frame's own pixel spacing times the display stride, so the
        shading is physical on a UTM/BLM grid; on a degrees grid the caller's ``z_factor`` is
        the only handle (1° ≠ 1 m), which is why it is a knob. Off restores the plain LUT path.
        View state only."""
        new = (bool(enabled), float(azimuth), float(altitude), float(z_factor))
        if new == self._hillshade:
            return
        self._hillshade = new
        self._refresh_image()

    def set_stretch(self, mode: str = "linear", percent: float = 2.0) -> None:
        """Contrast stretch of the raster image (``dynamix.core.stretch``, 2026-08-29): linear
        keeps the raw values under the LUT (unchanged behaviour); every other mode puts the
        stretched [0, 1] field under the LUT instead. Composes with hillshade. View state."""
        new = (str(mode), float(percent))
        if new == self._stretch:
            return
        self._stretch = new
        self._refresh_image()

    def set_levels(self, spec: str = "", colors: str = "", min_island: int = 0) -> None:
        """Density slice (ENVI, 2026-09-16): a class-count or data-unit-breaks spec
        (``dynamix.core.stretch.parse_levels``); active it REPLACES the continuous stretch so
        the colormap paints discrete classes -- custom h ramps. ``colors`` ("#rrggbb,...", one
        per class -- the slice editor's product) paints EXPLICIT class colors instead of
        sampling the LUT; count-mismatched or malformed colors fall back to the LUT, and
        unparseable specs fall back to the stretch, both silently (freeform fields).
        "" restores the stretch. View state only."""
        new = (str(spec or ""), str(colors or ""), int(min_island))
        if new == (self._levels, self._levels_colors, self._levels_sieve):
            return
        self._levels, self._levels_colors, self._levels_sieve = new
        self._refresh_image()

    def _class_rgba(self, small):
        """(ny, nx, 4) float RGBA in [0, 1] from explicit per-class slice colors, or ``None``
        when levels/colors are absent, mismatched, or malformed (LUT path then applies)."""
        if not (self._levels and self._levels_colors) or small.ndim != 2:
            return None
        from dynamix.core.stretch import (parse_class_colors, parse_levels, resolve_breaks,
                                          slice_indices)
        try:
            spec = parse_levels(self._levels)
            colors = parse_class_colors(self._levels_colors)
            if spec is None or colors is None:
                return None
            idx, n_classes = slice_indices(small, resolve_breaks(small, spec))
        except (ValueError, TypeError):
            return None
        if len(colors) != n_classes:
            return None
        table = np.asarray(colors, dtype=np.float64) / 255.0     # (n, 4) -- alpha 0 = "none"
        rgba = np.zeros(small.shape + (4,), dtype=np.float64)
        ok = idx >= 0
        if self._levels_sieve > 0:
            from dynamix.core.sieve import sieve_mask
            for k in range(n_classes):
                m = idx == k
                if m.any():
                    ok = ok & (~m | sieve_mask(m, self._levels_sieve))
        rgba[ok] = table[idx[ok]]
        rgba[..., 3] = np.where(ok, rgba[..., 3], 0.0)
        return rgba

    def _display_scalars(self, small):
        """The [0, 1] display field per the current levels/stretch, or ``None`` for the
        raw-linear path (levels win over stretch when both are set)."""
        mode, pct = self._stretch
        if self._levels and small.ndim == 2:
            from dynamix.core.stretch import classify, parse_levels, resolve_breaks
            try:
                spec = parse_levels(self._levels)
                if spec is not None:
                    out = classify(small, spec)
                    if self._levels_sieve > 0:
                        from dynamix.core.sieve import sieve_classes
                        n_cls = len(resolve_breaks(small, spec)) + 1
                        out = sieve_classes(out, n_cls, self._levels_sieve)
                    return out
            except (ValueError, TypeError):
                pass
        if mode != "linear" and small.ndim == 2:
            from dynamix.core.stretch import stretch
            return stretch(small, mode, percent=pct)
        return None

    def set_composite(self, spec: "dict | None") -> None:
        """The multiband display law: ``{"r"/"g"/"b": band index or None, "solo": [...],
        "mute": [...], "stretch_pct": float}`` (:func:`_composite_rgba`), or ``None`` to fall
        back to band 1 through the scalar pipeline. View state only -- analysis never reads
        it."""
        self._composite = dict(spec) if spec else None
        self._refresh_image()

    def _refresh_image(self) -> None:
        """Put the current field on ``image_item`` -- plain values through the LUT, or the
        hillshaded RGBA -- at ``set_field``'s decimation."""
        if self._field is None:
            return
        values = np.asarray(getattr(self._field, "values", self._field), dtype=np.float64)
        if values.ndim == 3:
            # A multi-component stack (2026-09-21 sensor ingestion): with a composite spec
            # (set_composite -- channel assignment, solo, mute) the stack draws as RGBA;
            # without one the canvas displays band 0. The FIELD keeps every band for
            # analysis either way (pca/tucker/tensor consume the stack).
            if self._composite:
                s = self._image_stride
                self.image_item.setImage(
                    _composite_rgba(values[::s, ::s, :], self._composite), levels=None)
                return
            values = values[..., 0]
        s = self._image_stride
        small = values[::s, ::s]
        on, azimuth, altitude, z_factor = self._hillshade
        mode, pct = self._stretch
        from dynamix.core.stretch import stretch
        if not on or small.ndim != 2:
            rgba = self._class_rgba(small)
            if rgba is not None:
                self.image_item.setImage((rgba * 255.0).astype(np.uint8).transpose(1, 0, 2),
                                         levels=None)
                return
            disp = self._display_scalars(small)
            if disp is None:
                self.image_item.setImage(small.T)
            else:
                self.image_item.setImage(disp.T, levels=(0.0, 1.0))
            if self._colormap:
                cmap = _resolve_cmap(self._colormap)
                if cmap is not None:
                    self.image_item.setColorMap(cmap)
            return
        from dynamix.core.hillshade import hillshade
        dx, dy = sample_spacing(self._field, s)          # a picture's frame is NATIVE-sized
        shade = hillshade(small, dx, dy, azimuth=azimuth, altitude=altitude, z_factor=z_factor)
        finite = np.isfinite(small)
        class_rgba = self._class_rgba(small)
        if class_rgba is not None:
            # Custom slice colors x shaded relief: same composition as the LUT path below.
            sh = np.nan_to_num(shade, nan=0.0)
            class_rgba[..., :3] *= sh[..., None]
            class_rgba[..., 3] *= np.where(np.isfinite(shade), 1.0, 0.0)
            self.image_item.setImage((class_rgba * 255.0).astype(np.uint8).transpose(1, 0, 2),
                                     levels=None)
            return
        base = self._display_scalars(small)
        if base is None:
            base = stretch(small, mode, percent=pct)
        norm = np.nan_to_num(base, nan=0.0)
        try:
            cmap = pg.colormap.get(self._colormap or "viridis")
        except Exception:
            cmap = pg.colormap.get("viridis")
        rgba = cmap.map(norm, mode="byte").astype(np.float64)        # (ny, nx, 4)
        sh = np.nan_to_num(shade, nan=0.0)
        rgba[..., :3] *= sh[..., None]
        rgba[..., 3] = np.where(finite & np.isfinite(shade), 255.0, 0.0)
        self.image_item.setImage(rgba.astype(np.uint8).transpose(1, 0, 2), levels=None)

    def set_overlay_colors(self, hchain: str, vtrail: str, extrema: str) -> None:
        """Per-layer overlay palette (``ui.color_hchain``/``ui.color_vtrail``/``ui.color_extrema``), each a hex color string -- rebuilds ``hchain_item``'s and ``vtrail_item``'s
        pens and ``extrema_item``'s brush, preserving whatever line width
        ``set_display_style`` last recorded (``self._overlay_line_width``) exactly the way a
        freshly created group item already does (see ``_draw_grouped_trails``).

        Every OTHER overlay item is left alone. ``seam_item``/``ghost_item`` draw a flag/
        exclusion identity, not a layer preference, and a committed group's own trail item
        (``self._group_items``) keeps ITS OWN color too -- ``classify_chains``'s priority order
        (group color first, then seam, then plain) is a fact about the chain, not something a
        layer-wide swatch should ever override.
        """
        pen = pg.mkPen(hchain)
        pen.setWidthF(self._overlay_line_width)
        self.hchain_item.setPen(pen)
        pen = pg.mkPen(vtrail)
        pen.setWidthF(self._overlay_line_width)
        self.vtrail_item.setPen(pen)
        self.extrema_item.setBrush(pg.mkBrush(extrema))

    def set_points_style(self, color: str, size: float) -> None:
        """Per-layer point-overlay styling (``ui.color_points``/``ui.point_size``) --
        ``color`` a hex string, ``size`` the SAME ``point_size`` knob the extrema dots already
        share (the design's "reuse ui.point_size" instruction: one size knob, two overlays).
        View state only, exactly like :meth:`set_overlay_colors`/:meth:`set_display_style` --
        never touches ``points_item``'s DATA."""
        self.points_item.setBrush(pg.mkBrush(color))
        self.points_item.setSize(float(size))

    def set_reference_layers(self, entries) -> None:
        """Draw reference layers (``dynamix.geo.vectors`` layers already in THIS field's pixel
        frame, via ``to_field_pixels``): one NaN-separated ``PlotDataItem`` per polyline/polygon
        layer (rings drawn as outlines), one ``ScatterPlotItem`` per point layer. ``entries`` are
        dicts ``{ref_id, name, kind, color, visible, features: [[(N,2) arrays...], ...]}``. The
        whole set is replaced; nothing here touches results or the analysis path."""
        for item in self.reference_items.values():
            self.view.removeItem(item)
        self.reference_items = {}
        for e in entries:
            color = e.get("color") or REFERENCE_COLORS[0]
            feats = e.get("features") or []
            if e.get("kind") in ("point", "multipoint"):
                pts = np.concatenate([p for parts in feats for p in parts]) if feats else np.empty((0, 2))
                item = pg.ScatterPlotItem(size=6, pen=pg.mkPen(color), brush=pg.mkBrush(color))
                item.setData(x=pts[:, 0], y=pts[:, 1])
            else:
                xs, ys = [], []
                nan = np.array([np.nan])
                for parts in feats:
                    for part in parts:
                        xs.append(np.asarray(part[:, 0], dtype=np.float64)); xs.append(nan)
                        ys.append(np.asarray(part[:, 1], dtype=np.float64)); ys.append(nan)
                item = pg.PlotDataItem(pen=pg.mkPen(color, width=1.5), connect="finite")
                if xs:
                    item.setData(np.concatenate(xs), np.concatenate(ys))
            item.setZValue(15)                       # above the raster and the WTMM overlays
            item.setVisible(bool(e.get("visible", True)))
            self.view.addItem(item)
            self.reference_items[e["ref_id"]] = item

    def set_reference_visible(self, ref_id: str, visible: bool) -> None:
        item = self.reference_items.get(ref_id)
        if item is not None:
            item.setVisible(bool(visible))

    def set_points_result(self, result: dict) -> None:
        """Feed ``points_item`` from a ``backproject`` result's own ``points_px`` arrays.

        Every point is drawn, ``inside`` or not -- a catalogue entry off the currently-registered
        raster's edge is still real data, and the ``ViewBox`` clips it naturally exactly as every
        other overlay item on this canvas already does; there is no separate filtering step here.
        Caller's job to decide WHETHER to call this at all (``MainWindow._apply``'s own
        ``_target``-match gate) -- this method only ever draws what it is handed.
        """
        px = result["points_px"]
        self.points_item.setData(x=np.asarray(px["x"]), y=np.asarray(px["y"]))

    def clear_points(self) -> None:
        """Take the point-layer scatter off the canvas -- the ``points_item`` sibling of
        :meth:`clear_overlays`, kept separate from it (rather than folded in) because a point
        layer's result carries none of what ``clear_overlays`` documents itself as clearing
        (``set_result``'s own WTMM-overlay items); see ``MainWindow._apply``'s own call sites for
        when each is appropriate."""
        self.points_item.setData([], [])

    def arm_roi_placement(self, h: int, w: int) -> None:
        """Enter (or resize) armed ROI placement (2026-08-30): the ``h``x``w`` footprint follows
        the cursor (``mouseMoveEvent``) and a plain click stamps it (``roiPlaced``). The ⌘-drag
        gesture stays untouched -- this supersedes nothing, it adds the drop-then-place flow."""
        self._roi_place = (int(h), int(w))

    def disarm_roi_placement(self) -> None:
        self._roi_place = None

    def _roi_place_origin(self, xy, h: int, w: int) -> tuple[int, int]:
        """Top-left of an ``h``x``w`` box centred on data-space ``xy``, clamped inside the field."""
        x, y = xy
        x -= self._base_off[1]                 # pin-in-place: the cursor is file-absolute
        y -= self._base_off[0]
        ny, nx = self._field_shape()
        # Center registration (2026-09-21): the box's pixel centers run row .. row+h-1, so its
        # own center is row + (h-1)/2 -- not row + h/2, which was the cell-edge convention.
        row = int(round(y - (h - 1) / 2.0))
        col = int(round(x - (w - 1) / 2.0))
        row = max(0, min(row, max(0, ny - h)))
        col = max(0, min(col, max(0, nx - w)))
        return row, col

    def show_roi_band(self, row: int, col: int, h: int, w: int) -> None:
        """Draw (or move) the amber selection band to a box given in image pixels.

        The panel's numeric edits call through here so the drawn band always shows what the
        NUMBERS currently say -- same corner convention as the ⌘-drag release at the bottom of
        this file (x = col, y = row)."""
        # Center registration (2026-09-21) + pin-in-place: local pixels, absolute cells.
        br, bc = self._base_off
        self._draw_roi_band((col + bc - 0.5, row + br - 0.5),
                            (col + w + bc - 0.5, row + h + br - 0.5))

    def set_saved_rois(self, rois) -> None:
        """Outline the saved ROIs ``[{"label", "row", "col", "h", "w", "active"}, ...]`` around
        the CELLS of their file pixels (``[c-0.5, c+w-0.5] x [r-0.5, r+h-0.5]``, the centre
        registration every overlay uses), labelled at their top-left corner."""
        def _rings(items):
            xs, ys = [], []
            for d in items:
                x0, y0 = d["col"] - 0.5, d["row"] - 0.5
                x1, y1 = d["col"] + d["w"] - 0.5, d["row"] + d["h"] - 0.5
                xs += [x0, x1, x1, x0, x0, np.nan]
                ys += [y0, y0, y1, y1, y0, np.nan]
            return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

        rois = list(rois)
        self.saved_roi_item.setData(*_rings([d for d in rois if not d.get("active")]))
        self.active_roi_item.setData(*_rings([d for d in rois if d.get("active")]))
        for item in self._saved_roi_labels:
            self.view.removeItem(item)
        self._saved_roi_labels = []
        for d in rois:
            text = pg.TextItem(str(d["label"]), color=RESTRAINED_DARK.amber, anchor=(0, 1))
            text.setPos(d["col"] - 0.5, d["row"] - 0.5)
            self.view.addItem(text)
            self._saved_roi_labels.append(text)

    def clear_roi_band(self) -> None:
        """Take the drawn selection off the canvas.

        A selection is about the image it was drawn on: once the window switches layers or opens
        another file, an amber box still sitting there claims a region of a raster nobody is
        looking at any more -- and it is the one overlay on this canvas that reads as a live
        instruction rather than a reading.
        """
        self._roi_press = None
        self.roi_band_item.setData([], [])

    def clear_overlays(self) -> None:
        """Take every WTMM-result overlay off the canvas -- everything :meth:`set_result` draws --
        leaving the raster itself and the corner readings untouched.

        Used by ``MainWindow`` when the ACTIVE layer's row is hidden in the layer panel
        (``layer_panel.py``'s module docstring: hide is v1-scoped to the one layer on screen, not
        general multi-layer compositing). Unhiding re-resolves, which redraws through
        :meth:`set_result` exactly as any other cache-hit redisplay does -- this method only ever
        takes overlays OFF, it never has to know how to put them back.
        """
        self.cap_note = None
        # 2026-09-22 (user: hiding the child layer left the extrema standing): the raster-
        # pixel overlay is a set_result product like every item below -- it leaves with them.
        self.extrema_raster_item.clear()
        self.extrema_item.setData([], [])
        self.hchain_item.setData([], [])
        self.vtrail_item.setData([], [])
        self.seam_item.setData([], [])
        self.ghost_item.setData([], [])
        for item in self._group_items.values():        # Committed-group trails, too
            item.setData([], [])
        self.selection_item.setData([], [])             # Ditto
        self.coi_item.setData([], [])
        self.roi_bounds_item.setData([], [])

    def clear_field(self) -> None:
        """Undo :meth:`set_field` and every reading it feeds -- the raster image, every WTMM
        overlay, the ROI band and both corner readings (the distance bar and the wavelet bar) --
        back to exactly the state ``__init__`` leaves this canvas in before any layer is ever
        loaded.

        Used when the LAST layer in the project is removed (``main_window.py``'s zombie-active-
        layer fix): with no layer left to show, every one of those was still naming a raster that
        no longer exists, and none of them redraws itself away on its own -- ``clear_overlays``
        alone leaves the image and both corner readings stale, and there is otherwise no path back
        to a field of ``None`` at all (:meth:`set_field` cannot be called with one; it converts
        ``field`` straight through ``np.asarray(..., dtype=np.float64)``, which raises on
        ``None``).
        """
        self._field = None
        self.image_item.clear()
        self.clear_overlays()
        self.clear_points()
        self.clear_roi_band()
        self.set_scale_bar_text("")
        self.set_wavelet_bar(None, None, None, None)
        self.set_pick_chains(None)
        self.cancel_transect()          # No raster left for an in-progress vertex to sit on

    def set_pick_chains(self, chains) -> None:
        """The result's own ``chains`` list (the ``dynamix.core.chain_pick`` fixture shape --
        dicts with ``"x"``/``"y"`` arrays) that the click/⌥-lasso gesture below picks against.

        The window calls this wherever it pushes overlay geometry today (the ``set_result`` call
        site in ``MainWindow._apply``) and with ``None`` wherever it clears the overlays instead
        (``clear_overlays`` call sites) -- see ``main_window.py``'s own comments at each site.
        ``None`` (the default, and after a clear) means nothing is pickable yet: a click still
        fires :attr:`chainPicked` in that case, reporting an honest miss, exactly as it would
        over a real, unmatched chain list.

        Coordinates are the RAW ``result["chains"]`` arrays, in the display frame. The caller
        (``MainWindow._apply``) closes the ROI-offset gap by shifting chains to display-frame
        coordinates before pushing them here. This setter applies no offset itself: it receives
        chains that are already display-frame-shifted (or frame-shifted for non-ROI results,
        which amounts to the same thing), and stores them for coordinate matching against the
        pick click location, which is already in display-frame coordinates.
        """
        self._pick_chains = chains

    def set_selection_mode(self, mode: str) -> None:
        """Which gesture a plain (no-⌘) left-button drag performs from
        now on -- ``"click"``, ``"box"``, ``"lasso"`` or ``"transect"``.

        This is the canvas's own PASSIVE mirror, not the validating setter: ``MainWindow.
        set_selection_mode`` is the one that raises on an unknown mode, syncs the mode-row
        buttons, and calls through to here -- see that method's own docstring. This one just
        stores whatever string it is handed; MainWindow never hands it anything else.

        Moving AWAY from ``"transect"`` while a first click is still in
        progress cancels it (:meth:`cancel_transect`) -- otherwise a mode switch mid-gesture (the
        `c`/`v` hotkeys fire independent of mouse capture, exactly the reachable interaction found for box/lasso) leaves a stray marker on screen with no
        way left to finish or cancel the gesture that placed it.
        """
        if str(mode) != "transect" and self._transect_a is not None:
            self.cancel_transect()
        self._selection_mode = str(mode)

    def set_selection_chains(self, indices) -> None:
        """Redraw the SELECTED chains a second time, on top of whatever item already drew them,
        in the theme's ``selection_accent`` at 2x the current overlay line width -- the raster
        canvas's own half of "visible selection everywhere". ``MainWindow`` calls
        this after every ``GroupPalette.apply_picks``/``add_pick`` (wired through
        ``GroupPalette.membershipChanged``), naming the ACTIVE layer's own selected indices.

        ``indices`` index into the SAME ``_pick_chains`` list :meth:`set_pick_chains` populates
        -- already display-frame-shifted (see that method's own docstring), so this draws in the
        EXACT coordinates the ordinary trails/picking already use, no extra offset needed.
        ``None``, an empty list, or no chains currently on record (``_pick_chains is None``) all
        clear the overlay -- there is nothing honest to draw in any of those cases.
        """
        if not indices or self._pick_chains is None:
            self.selection_item.setData([], [])
            return
        selected = [self._pick_chains[i] for i in indices if 0 <= i < len(self._pick_chains)]
        sx, sy = vchain_trails(selected)
        self.selection_item.setData(sx, sy)

    def pick_chains(self):
        """The SAME display-shifted ``chains`` list
        :meth:`set_pick_chains` currently holds (or ``None``) -- the swath-select seam. Reading
        THIS list, rather than the raw ``result["chains"]``, is what makes a transect's buffer
        selection agree pixel-for-pixel with an ordinary click pick over the identical raster
        ("your swath select must use the SAME shifted coordinates for chain distance,
        consistent with picking")."""
        return self._pick_chains

    def set_transects(self, records, selected_id=None) -> None:
        """Redraw every VISIBLE persisted transect's line, plus the SELECTED one a second time on
        top at 2x width (mirrors :meth:`set_selection_chains`'s own "thicker, on top" idiom).

        ``records`` is ``TransectPanel.records()``'s own list -- anything carrying ``.a``, ``.b``,
        ``.visible`` and ``.transect_id`` (a :class:`dynamix.model.project.TransectRecord`). This
        method draws exactly what it is handed; a hidden transect (``visible=False``) is simply
        left out of ``transect_item``'s polyline, and out of the highlight too -- there is nothing
        honest to highlight a line that is not currently drawn.
        """
        xs: list[float] = []
        ys: list[float] = []
        for r in records:
            if not r.visible:
                continue
            xs.extend((r.a[0], r.b[0], float("nan")))
            ys.extend((r.a[1], r.b[1], float("nan")))
        if xs:
            xs, ys = xs[:-1], ys[:-1]                  # drop the trailing separator
        self.transect_item.setData(xs, ys)

        highlight = next(
            (r for r in records if r.visible and r.transect_id == selected_id), None)
        if highlight is None:
            self.transect_highlight_item.setData([], [])
        else:
            self.transect_highlight_item.setData(
                [highlight.a[0], highlight.b[0]], [highlight.a[1], highlight.b[1]])

    def cancel_transect(self) -> None:
        """Esc while a transect's first click is in progress -- drops A and removes
        its marker. A harmless no-op with nothing in progress, so ``MainWindow`` can wire this to
        a plain, unconditional ``QShortcut`` (the same idiom `c`/`v`/Space already use) with no
        mode guard needed at the call site."""
        self._transect_a = None
        self.transect_marker_item.setData([], [])

    def _handle_transect_click(self, xy: tuple[float, float]) -> None:
        xy = (xy[0] - self._base_off[1], xy[1] - self._base_off[0])   # pin-in-place
        """One CLICK in transect mode -- :meth:`mouseReleaseEvent`'s own click-vs-drag
        disambiguation (:data:`_CLICK_MAX_TRAVEL_PX`) has already run by the time this is called,
        so this only ever receives a genuine click, never a pan's release point.

        The FIRST click sets A (drawn as a marker, cleared by the second click or
        :meth:`cancel_transect`); the SECOND sets B, auto-orients the pair
        (:func:`dynamix.core.transect.orient_endpoints`) and emits :attr:`transectDrawn`. The
        FINISHED line is drawn by whatever :meth:`set_transects` call follows (``TransectPanel.
        add_record``'s own listener) -- not by this method, which only ever owns the transient
        in-progress marker.
        """
        if self._transect_a is None:
            self._transect_a = xy
            self.transect_marker_item.setData([xy[0]], [xy[1]])
            return
        a, b = orient_endpoints(self._transect_a, xy)
        self._transect_a = None
        self.transect_marker_item.setData([], [])
        self.transectDrawn.emit(a, b)

    def mousePressEvent(self, event) -> None:
        if self._roi_modifier_held(event):
            self._roi_press = self._data_xy(event)
            self._draw_roi_band(self._roi_press, self._roi_press)
            event.accept()
            return
        if (event.button() == QtCore.Qt.MouseButton.LeftButton
                and self._selection_mode == "click"
                and bool(event.modifiers() & LASSO_MODIFIER)):
            # The pre-existing ⌥-drag lasso SHORTCUT --
            # deliberately gated to click mode only. Lasso is also a first-class MODE (see the branch below) where a plain drag alone is enough; in
            # click mode Alt is still what TRIGGERS this gesture at all, so it can never also
            # mean "subtract" there (mouseReleaseEvent always reports subtract=False for this
            # branch) without silently flipping a shortcut that has worked one way since it was
            # built.
            self._lasso_points = [self._data_xy(event)]
            self._draw_lasso()
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._selection_mode == "box":
            # Box MODE -- a plain drag starts the rubber-band with no
            # modifier required ("box... resolves via points_in_box"). ⌥ is read at
            # RELEASE (see mouseReleaseEvent) to pick the op, not here, so an ⌥-held drag still
            # draws a BOX, never diverted into the lasso branch above (which is gated to click
            # mode only, precisely so it cannot compete with this one).
            self._box_press = self._data_xy(event)
            self._draw_selection_box(self._box_press, self._box_press)
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._selection_mode == "lasso":
            # Lasso MODE -- a plain drag lassos with no modifier
            # required ("Lasso keeps the existing ⌥-drag polygon and becomes a first-class
            # mode (no modifier needed when the mode is active)"). ⌥ is read at RELEASE to pick
            # the op (add vs. subtract), same as box mode just above -- it does not gate whether
            # this branch fires at all.
            self._lasso_points = [self._data_xy(event)]
            self._draw_lasso()
            event.accept()
            return
        if (event.button() == QtCore.Qt.MouseButton.LeftButton
                and self._selection_mode in ("click", "transect")):
            # Neither ROI nor lasso/box: could still turn out to be a pick/transect-vertex (a
            # plain click) or a pan (a plain drag) -- that is only decided at release, by how far
            # the pointer actually moved (see mouseReleaseEvent). Recorded, never accepted/
            # returned: `super()` below still runs unconditionally, so pyqtgraph's own pan
            # machinery sees this press exactly as it always did. A
            # transect-mode press reuses this SAME press/travel/release plumbing a plain click
            # already has -- `_press_shift` is recorded but never read in transect mode (it is a
            # click-mode-only concept), harmlessly.
            pos = event.position()
            self._press_pos = (pos.x(), pos.y())
            self._press_shift = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._roi_press is not None:
            # Whole pixels while dragging, too: the live band covers exactly the cells the
            # release will emit (same arithmetic as mouseReleaseEvent), never the raw cursor.
            br, bc = self._base_off
            cur = self._data_xy(event)
            row, col, h, w = roi_from_corners((self._roi_press[0] - bc, self._roi_press[1] - br),
                                              (cur[0] - bc, cur[1] - br),
                                              self._field_shape())
            self._draw_roi_band((col + bc - 0.5, row + br - 0.5),
                                (col + w + bc - 0.5, row + h + br - 0.5))
            event.accept()
            return
        if self._box_press is not None:
            self._draw_selection_box(self._box_press, self._data_xy(event))
            event.accept()
            return
        if self._lasso_points is not None:
            self._lasso_points.append(self._data_xy(event))
            self._draw_lasso()
            event.accept()
            return
        if self._roi_place is not None and self._field is not None:
            # Armed placement (2026-08-30): the footprint rides the cursor, clamped to the field.
            h, w = self._roi_place
            row, col = self._roi_place_origin(self._data_xy(event), h, w)
            # Center registration (2026-09-21) + pin-in-place: absolute cells.
            br, bc = self._base_off
            self._draw_roi_band((col + bc - 0.5, row + br - 0.5),
                                (col + w + bc - 0.5, row + h + br - 0.5))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """End whichever gesture ``mousePressEvent`` started.

        **ROI** (unchanged): snap the band to the rectangle actually emitted, then emit it.
        Snapping matters -- the band stays on screen (the box IS the selection the panel is about
        to describe), so it has to show the CLAMPED, whole-pixel rectangle the ROI transform will
        be given, not the raw sub-pixel one the mouse traced, which after a clamp at the raster's
        edge can be a visibly different box from the one that gets analysed.

        **Lasso**: close the captured polygon
        (:func:`dynamix.core.chain_pick.chains_in_polygon` auto-closes too -- see its own
        docstring -- so this is belt and braces, not load-bearing) and emit every enclosed chain
        index, then take the temporary polyline off screen. A degenerate capture (fewer than 3
        points -- a press immediately released, or released without ever moving) reports an
        empty list rather than calling into polygon math that could not mean anything with fewer
        than three vertices. ``subtract`` is ``True`` only for a lasso-MODE gesture with ⌥ held
        at release; the click-mode ⌥-shortcut always reports ``False`` -- see
        :meth:`mousePressEvent`'s own comment on why Alt cannot double as "subtract" there.

        **Box**: the box-mode sibling of Lasso just above, over
        :func:`dynamix.core.chain_pick.chains_in_box` instead of a polygon test -- same
        ⌥-at-release subtract rule, no click-mode special case (box has no legacy shortcut to
        stay compatible with).

        **Pick / transect vertex**: a plain
        left-button press/release pair is a CLICK, not a pan, only when the pointer travelled at
        most :data:`_CLICK_MAX_TRAVEL_PX` SCREEN pixels between the two -- past that, it was a
        pan, and firing a pick (or placing a transect vertex) on top of one would act on whatever
        happened to end up under the cursor at the FAR end of a drag the user never intended as a
        gesture at all. The release position (not the press position) is what is acted on -- the
        same "the gesture ends where you let go" convention the ROI band's own corner already
        uses. In transect mode a genuine click goes to :meth:`_handle_transect_click` instead of
        the ordinary chain pick -- the two are mutually exclusive per mode, never both fired for
        the same click. `super()` still runs unconditionally afterward: unlike ROI/lasso/box, a
        plain click never `accept()`s the event, so pyqtgraph's own pan-release bookkeeping always
        completes.
        """
        if self._roi_press is not None:
            press, self._roi_press = self._roi_press, None
            br, bc = self._base_off
            rel = self._data_xy(event)
            row, col, h, w = roi_from_corners((press[0] - bc, press[1] - br),
                                              (rel[0] - bc, rel[1] - br),
                                              self._field_shape())
            # Center registration (2026-09-21) + pin-in-place: local pixels, absolute cells.
            self._draw_roi_band((col + bc - 0.5, row + br - 0.5),
                                (col + w + bc - 0.5, row + h + br - 0.5))
            event.accept()
            self.roiDrawn.emit(row, col, h, w)
            return
        if self._box_press is not None:
            press, self._box_press = self._box_press, None
            release = self._data_xy(event)
            self.box_item.setData([], [])          # take the rubber band off, gesture is over
            xmin, xmax = sorted((press[0], release[0]))
            ymin, ymax = sorted((press[1], release[1]))
            picked = chains_in_box(self._pick_chains or [], xmin, xmax, ymin, ymax)
            subtract = bool(event.modifiers() & LASSO_MODIFIER)
            event.accept()
            self.chainsBoxed.emit(picked, subtract)
            return
        if self._lasso_points is not None:
            points = self._lasso_points
            self._lasso_points = None
            self._draw_lasso()
            if len(points) >= 3:
                picked = chains_in_polygon(self._pick_chains or [], points + [points[0]])
            else:
                picked = []
            subtract = (self._selection_mode == "lasso"
                        and bool(event.modifiers() & LASSO_MODIFIER))
            event.accept()
            self.chainsLassoed.emit(picked, subtract)
            return
        if self._press_pos is not None:
            press, self._press_pos = self._press_pos, None
            shift, self._press_shift = self._press_shift, False
            pos = event.position()
            travel = ((pos.x() - press[0]) ** 2 + (pos.y() - press[1]) ** 2) ** 0.5
            if travel <= _CLICK_MAX_TRAVEL_PX:
                xy = self._data_xy(event)
                if self._roi_place is not None:
                    # Armed placement: the click stamps the hovering footprint and disarms.
                    h, w = self._roi_place
                    row, col = self._roi_place_origin(xy, h, w)
                    self._roi_place = None
                    # Center registration (2026-09-21) + pin-in-place: absolute cells.
                    br, bc = self._base_off
                    self._draw_roi_band((col + bc - 0.5, row + br - 0.5),
                                        (col + w + bc - 0.5, row + h + br - 0.5))
                    self.roiPlaced.emit(row, col, h, w)
                elif self._selection_mode == "transect":
                    self._handle_transect_click(xy)
                else:
                    max_dist = self._pick_radius_data(_PICK_MAX_DIST_PX)
                    dr, dc = self._draw_off
                    self.chainPicked.emit(pick_chain(self._pick_chains or [],
                                                     (xy[0] - dc, xy[1] - dr),
                                                     max_dist), shift)
        super().mouseReleaseEvent(event)

    def set_field(self, field, max_dim: int = _DEFAULT_MAX_DIM) -> None:
        """Raster display: an ``ImageItem`` in the ``viridis`` colormap, decimated for on-screen
        size via :func:`lod_stride`. ``field`` may be a ``RasterField`` or a bare ``(ny, nx)``
        array.

        The explicit ``setRect`` is the registration, and it is not optional. ``setImage`` alone
        gives an ``ImageItem`` one data unit per STORED sample, so a stride-4 display of a
        (4096, 3000) raster would occupy a 750x1024 rectangle in data space while the extrema, the
        chain polylines and the scale bar are all drawn in full-resolution pixel coordinates --
        every overlay four times too far out, silently, and only on the big rasters this app
        exists for. Pinning the rect to the ORIGINAL shape makes the decimation purely a drawing
        decision, invisible to every coordinate anyone else uses.

        ``max_dim`` is exposed so a test can force a stride without allocating a 12-megapixel
        array; nothing in the app passes it.

        ``field`` itself (not just its values) is RETAINED on the canvas: the wavelet bar's
        physical-unit conversion (:meth:`_update_wavelet_bar`) needs the field's CRS/frame, and
        this is the one place the canvas is ever handed it, matching how ``MainWindow`` keeps its
        own ``self.field`` from the same call site.
        """
        self._field = field
        self._base_off = window_offset(field)
        # North-up for GEOGRAPHIC fields only (2026-08-19 fix): a file stored south-first
        # (ascending latitude axis -- the demo DEM) drew upside down here while the Vector tab
        # showed it correctly, because this view is image-convention (row 0 at top,
        # ``invertY(True)`` at construction). Un-inverting the ViewBox flips the image AND every
        # overlay together (all live in the same row/col data coords), so registration is
        # untouched. Everything non-geographic keeps image convention exactly -- EBSD/pixel maps
        # WANT row 0 at top, and north-first GeoTIFFs (descending y_axis, e.g. the BOEM windows)
        # are already north-up under it. Display-only: the stored array and every analysis
        # coordinate are never reordered (flipping data would mirror gradient orientations).
        from dynamix.core.frames import GeographicFrame
        y_axis = getattr(field, "y_axis", None)
        south_first_geographic = (
            isinstance(getattr(field, "frame", None), GeographicFrame)
            and y_axis is not None and len(y_axis) > 1 and float(y_axis[-1]) > float(y_axis[0]))
        self.view.invertY(not south_first_geographic)
        values = np.asarray(getattr(field, "values", field), dtype=np.float64)
        self._image_stride = lod_stride(values.shape[:2], max_dim)
        self._refresh_image()
        # Center registration (edges drawn "between two pixels" at max zoom):
        # every overlay draws at integer pixel INDICES (and subpixel channels are index-based,
        # x_sub = j + t, |t| <= 1/2), so integer data coordinate j must be the CENTER of
        # pixel j -- the same sample convention the vector view's axes-direct draping uses.
        # The old QRectF(0, 0, nx, ny) put cell EDGES at integers: every extrema, h-line and
        # chain sat half a pixel up-left, on the seams between pixels.
        # Native-block registration: the drawn image holds
        # m = ceil(n / s_lod) samples per axis, and each covers EXACTLY S = display_stride *
        # s_lod FILE pixels -- sample k spans [k*S, (k+1)*S). So the rect is m*S, never the
        # stored length: for a display PICTURE that is its native extent (one coordinate
        # system -- the box, the ROI result and the image all speak file pixels), and for the
        # canvas's own LOD it removes the far-edge squeeze (n/m < s per sample). An ordinary
        # field at full resolution has S = 1 and m = n: exactly the old shape rect.
        S = display_stride(field) * self._image_stride
        m_y = -(-values.shape[0] // self._image_stride)
        m_x = -(-values.shape[1] // self._image_stride)
        self.image_item.setRect(QtCore.QRectF(self._base_off[1] - 0.5,
                                              self._base_off[0] - 0.5,
                                              m_x * S, m_y * S))

    def set_result(self, result: dict, scale_idx: int) -> None:
        """Redraw the per-scale overlay: isolated extrema (``line_id == -1``) as points, every
        ordered H-line as one NaN-separated polyline, (if enabled) V-chain drift trails, and (for
        an ROI result carrying missing data) the COI contamination outline at this scale.

        **Flagged, grouped and excluded chains split off the ordinary trail.** A classification
        device (``dynamix.devices.chain_classify``) may tag some chains (``chain["tags"]``
        truthy) as probable seam artifacts, or move them out of ``chains`` entirely into
        ``result["chains_excluded"]``. Tagged chains draw on ``seam_item`` instead of
        ``vtrail_item``, and stay visible even with ``_show_trails`` off -- flagging exists
        precisely so a suspect chain is not just another routine drift reading a user can hide.
        Excluded chains draw dashed on ``ghost_item``, gated on ``result.get("_show_ghosts",
        True)``. Neither key is ever present on a result no such device has touched, so both
        splits are no-ops there: every plain chain still draws on ``vtrail_item`` exactly as
        before this existed. :func:`classify_chains` does the split; see its own docstring.

        **Committed groups render in their own color.** A chain
        :class:`dynamix.devices.groups.GroupPaint` stamped with a ``"group:<name>"`` tag (at
        commit time) draws on a per-color item from ``self._group_items`` instead
        of ``seam_item`` or ``vtrail_item`` -- built lazily and reused, one item per distinct
        ``group_color`` currently in use, and (like ``seam_item``) always visible regardless of
        ``_show_trails``: a committed group is an even more deliberate assertion than a machine
        seam flag. A chain carrying BOTH a seam tag and a group tag draws grouped, not seam -- the
        user's own commit outranks the classifier (see :func:`classify_chains`'s docstring).

        **An ROI result renders IN CONTEXT.** Its extrema are in ROI-local coordinates (the halo
        engine crops every scale onto the ROI's own grid), while the image on screen is the
        PARENT raster -- so every result-space overlay is translated by :func:`display_offset` and
        the ROI's own bounding rectangle is outlined, putting the measurement over the ground it
        was measured on. Drawing them unshifted would pile a 512-px ROI's chains into the corner
        of a 30,000-px raster and label it as data from there, which is precisely the class of
        quiet mis-registration this app exists to rule out. The parent stays the displayed image
        on purpose: seeing an ROI's structure against its surroundings is the point of ROIs.

        The offset is :func:`display_offset`, NOT :func:`roi_offset`: the ROI's coordinates are
        file-absolute and the displayed parent may itself be a window of that file, so the window's
        own origin has to come back off. See that function -- getting this wrong is invisible on
        every whole-file raster and wrong by thousands of pixels on the ones this app exists for.

        For an ordinary whole-raster result the offset is ``(0, 0)`` and the bounds rectangle is
        empty, so this path is bit-for-bit what it was before ROIs existed.
        """
        ext = result["extrema"][scale_idx]
        shape = result["_shape"]
        row_off, col_off = display_offset(result, self._field)
        row_off += self._base_off[0]           # pin-in-place: overlays live file-absolute
        col_off += self._base_off[1]
        self._draw_off = (float(row_off), float(col_off))

        ext_x = np.asarray(ext["x"])
        ext_y = np.asarray(ext["y"])
        iso = np.asarray(ext["line_id"]) == -1
        self.extrema_item.setData(x=ext_x[iso] + col_off, y=ext_y[iso] + row_off)

        # The polyline helpers run in the RESULT's own coordinates (`hline_polylines` indexes a
        # scratch grid of `shape`, which is the ROI's shape), so the translation is applied to
        # what they return, never to what they are given. Both helpers are MEMOIZED on the input
        # object's identity (:meth:`_cached_geometry`): results are cached and immutable by the
        # zero-cache-miss law, and a scale scrub hands this method the SAME ext/chains objects out
        # of the cached stack every time -- rebuilding their polylines per tick was the measured
        # scrub cost (Python loop over every H-line). The offsets stay OUTSIDE the cache: an ROI result re-rendered after a
        # window change keeps its geometry and re-adds the translation.
        sel_geom = self._selection_geometry(result, scale_idx)
        if sel_geom is not None:
            hx, hy = sel_geom[0]
        elif (base := result.get("_ext_base")) is not None:
            hx, hy = self._masked_hline_geometry(base, ext, shape, result.get("_ext_base_runs"))
        else:
            hx, hy = self._cached_geometry("hline", ext, lambda: hline_polylines(ext, shape))
        # Draw cap (2026-09-14): bound the H-line points handed to pyqtgraph -- without it a dense
        # DEM finest scale pushes >1M points per redraw (~80 ms, profiled), the reason a scale/
        # filter scrub felt like molasses vs EQSelect. Longest lines kept; the shortfall becomes a
        # status note below. The selection fast path (sel_geom) already caps itself.
        h_kept = h_total = 0
        hx_pre, hy_pre = hx, hy                  # pre-cap: the raster overlay has no point cost
        if sel_geom is None:
            hx, hy, h_kept, h_total = cap_polylines(hx, hy, self.draw_cap)
        self.hchain_item.setData(hx + col_off, hy + row_off)

        # The raster-pixel extrema overlay (2026-09-22): the same visible truth the polylines
        # carried, in pixel form on the result's own grid. Members from the drawn H-line
        # vertices (pre-cap), orphans from the iso dots; the rect places it under the ROI/
        # window offset with the center registration the image itself uses.
        ny_r, nx_r = int(shape[0]), int(shape[1])
        rgba = np.zeros((ny_r, nx_r, 4), dtype=np.ubyte)
        if len(hx_pre):
            fin = np.isfinite(hx_pre) & np.isfinite(hy_pre)
            mx = np.rint(np.asarray(hx_pre)[fin]).astype(np.int64)
            my = np.rint(np.asarray(hy_pre)[fin]).astype(np.int64)
            ok = (mx >= 0) & (mx < nx_r) & (my >= 0) & (my < ny_r)
            rgba[my[ok], mx[ok]] = (*HCHAIN_COLOR, 255)
        if ext_x[iso].size:
            dx_i = np.asarray(ext_x[iso], dtype=np.int64)
            dy_i = np.asarray(ext_y[iso], dtype=np.int64)
            ok = (dx_i >= 0) & (dx_i < nx_r) & (dy_i >= 0) & (dy_i < ny_r)
            rgba[dy_i[ok], dx_i[ok]] = (*EXTREMA_COLOR, 255)
        self.extrema_raster_item.setImage(rgba.transpose(1, 0, 2), autoLevels=False)
        self.extrema_raster_item.setRect(QtCore.QRectF(col_off - 0.5, row_off - 0.5,
                                                       nx_r, ny_r))

        # Split by `tags` -- see the docstring above for why. `chain.get("tags")` is absent on
        # every pre-Task-4 chain, so `seam_chains` is empty and `plain_chains == chains` there.
        # The split and every trail build cache together under the chains list's identity (one
        # `_cached_geometry` entry, "the chains-list identity" -- the group partition
        # rides inside that SAME entry, as a color -> (x, y) dict, rather than minting a second
        # cache key per color: the classification pass is cheap, but only ever runs on a genuine
        # cache MISS this way, exactly as the plain/seam split already did before groups existed).
        chains = result.get("chains") or _NO_CHAINS

        if sel_geom is not None:
            # Live selection: no device rewrote ``chains`` (that would have invalidated it), so
            # nothing carries tags/groups -- every drawn chain is plain, gathered off the product.
            vx, vy = sel_geom[1]
            sx = sy = np.empty(0, dtype=np.float64)
            grouped_xy: dict = {}
        else:
            def _split_trails():
                plain, seam, grouped = classify_chains(chains)
                return vchain_trails(plain) + vchain_trails(seam) + \
                    ({color: vchain_trails(cs) for color, cs in grouped.items()},)

            vx, vy, sx, sy, grouped_xy = self._cached_geometry("trails", chains, _split_trails)
        v_kept = v_total = 0
        if sel_geom is None:
            vx, vy, v_kept, v_total = cap_polylines(vx, vy, self.draw_cap)
        self.vtrail_item.setData(vx + col_off, vy + row_off)
        self.vtrail_item.setVisible(self._show_trails)
        # Honest "drawing N of M" when the cap withheld lines (never a silent truncation).
        notes = []
        if h_total > h_kept:
            notes.append(f"drawing {h_kept} of {h_total} H-lines")
        if v_total > v_kept:
            notes.append(f"drawing {v_kept} of {v_total} V-chains")
        self.cap_note = (" · ".join(notes) + " — tighten filters to see the rest"
                         if notes else self.cap_note)

        # Flagged trails render on `seam_item` regardless of `_show_trails`: that toggle hides the
        # ROUTINE drift reading, but a seam flag is a warning about the data, not a reading a user
        # opted into (see seam_item's construction comment).
        self.seam_item.setData(sx + col_off, sy + row_off)

        self._draw_grouped_trails(grouped_xy, col_off, row_off)

        excluded = result.get("chains_excluded") or _NO_CHAINS
        gx, gy = self._cached_geometry("ghost", excluded, lambda: vchain_trails(excluded))
        self.ghost_item.setData(gx + col_off, gy + row_off)
        self.ghost_item.setVisible(bool(result.get("_show_ghosts", True)))

        self._update_coi_outline(result, scale_idx, row_off, col_off)
        self._update_roi_bounds(result, row_off, col_off)
        self._update_wavelet_bar(result, scale_idx)

    def _draw_grouped_trails(self, grouped_xy: dict, col_off: int, row_off: int) -> None:
        """One :class:`pyqtgraph.PlotDataItem` per distinct committed-group color THIS result
        uses, created lazily into ``self._group_items`` and reused across calls -- never
        removed, so a later result with fewer/no grouped colors just clears the ones it no longer
        needs (the ``else`` branch below) rather than adding/removing view items every redraw (the
        color set is typically small and stable across a scrub session -- see ``_group_items``'s
        own comment in ``__init__``). A freshly created item picks up whatever
        ``set_display_style`` last recorded (``_overlay_opacity``/``_overlay_line_width``), the
        same styling every other overlay item already carries.
        """
        for color, (gx, gy) in grouped_xy.items():
            item = self._group_items.get(color)
            if item is None:
                item = pg.PlotDataItem(pen=pg.mkPen(color), connect="finite")
                item.setOpacity(self._overlay_opacity)
                pen = item.opts["pen"]
                pen.setWidthF(self._overlay_line_width)
                item.setPen(pen)
                self.view.addItem(item)
                self._group_items[color] = item
            item.setData(gx + col_off, gy + row_off)
        for color, item in self._group_items.items():
            if color not in grouped_xy:
                item.setData([], [])

    def _update_roi_bounds(self, result: dict, row_off: int, col_off: int) -> None:
        """Outline the ROI's own rectangle on the parent raster, or clear it for a non-ROI result.

        Its SIZE comes from ``_roi["roi"]`` -- the window the transform was GIVEN, not one
        re-derived from the result's shape -- so it states what was analysed even if a downstream
        filter has since changed what is drawn inside it. Its POSITION is the same
        :func:`display_offset` every overlay inside it was translated by, passed in rather than
        recomputed: a box drawn in one frame around points drawn in another is worse than no box.
        """
        roi = (result.get("_roi") or {}).get("roi")
        if not roi:
            self.roi_bounds_item.setData([], [])
            return
        h, w = int(roi[2]), int(roi[3])
        self.roi_bounds_item.setData(
            [col_off, col_off + w, col_off + w, col_off, col_off],
            [row_off, row_off, row_off + h, row_off + h, row_off])

    def set_show_trails(self, value: bool) -> None:
        self._show_trails = bool(value)
        self.vtrail_item.setVisible(self._show_trails)

    def _update_coi_outline(self, result: dict, scale_idx: int,
                            row_off: int = 0, col_off: int = 0) -> None:
        """Best-effort COI (cone of influence) outline for ``result`` at ``scale_idx``.

        ``row_off``/``col_off`` translate the finished outline into the parent raster's
        coordinates (see :meth:`set_result`). They are applied at ``setData`` time, NOT baked into
        the cache: the cached dilation and isocurve belong to the mask, which is ROI-local and
        does not move, so a future re-window would otherwise have to invalidate a cache that is
        still perfectly valid.

        Only ``dynamix.roi.run_wtmm2d_roi`` results carry ``_missing_mask`` (bool, ROI-shaped --
        the source file's nodata cells, zero-filled before the transform, never assumed) and
        ``_coi_radii`` (that scale's halo width, built in the SAME per-scale loop as ``extrema`` --
        see ``dynamix/roi/halo.py``). Absent either key -- every non-ROI result, and any ROI result
        whose parent raster had no nodata at all -- draws nothing, at ZERO cost: no scipy import,
        no dilation, no isocurve call.

        The scale axis is ``_scale_idx`` where present, exactly like :meth:`_update_wavelet_bar`
        (see that method's comment): ``scale_idx`` addresses ``extrema``, a different axis once a
        filter has narrowed the result to one layer, and ``_coi_radii`` shares ``extrema``'s axis
        because both were appended in the same per-scale loop.

        The cache is keyed on the ``_missing_mask`` ARRAY'S OWN IDENTITY (``self._coi_mask_ref``,
        a STRONG reference), not on ``id(result)``. It used to be the latter, and that never hit
        in real usage: ``ScaleSelect.apply()`` -- the filter every scale-scrub tick runs through --
        does ``out = dict(result)``, a fresh dict (a new ``id()``) on EVERY tick, while ``out[
        "_missing_mask"]`` is a plain shallow-copied reference to the SAME array each time. Keying
        on the dict's id therefore invalidated the cache every single tick; keying on the mask
        array survives the churn, because the array object really is unchanged. Holding the
        STRONG reference also closes the id-reuse hazard a bare ``id(mask)`` would have had
        (an id can be recycled onto an unrelated object once the original is garbage-collected;
        holding the reference here means that never happens while the cache still points at it).

        Both stages are cached together per scale index -- the dilated mask AND the drawn
        NaN-joined outline (x, y) -- so a cache hit re-runs NEITHER ``scipy.ndimage.
        binary_dilation`` NOR pyqtgraph's marching-squares ``isocurve`` (measured ~46 ms on a
        512x512 mask -- the more expensive of the two steps, and a real redraw-budget hazard at
        BOEM ROI sizes if it re-ran every scrub tick). Revisiting a scale after sweeping through
        others (0 -> 1 -> 0) is therefore a cache hit, not a recompute, as long as the mask array
        itself hasn't changed.
        """
        mask = result.get("_missing_mask")
        radii = result.get("_coi_radii")
        a_idx = int(result.get("_scale_idx", scale_idx))
        if mask is None or radii is None or not (0 <= a_idx < len(radii)):
            self.coi_item.setData([], [])
            self._coi_cache = {}
            self._coi_mask_ref = None
            return

        if mask is not self._coi_mask_ref:
            self._coi_cache = {}
            self._coi_mask_ref = mask

        cached = self._coi_cache.get(a_idx)
        if cached is None:
            dilated = dilate_coi_mask(mask, int(radii[a_idx]))
            x, y = coi_outline(dilated)
            cached = (dilated, x, y)
            self._coi_cache[a_idx] = cached
        _, x, y = cached
        self.coi_item.setData(x + col_off, y + row_off)

    def set_scale_bar_text(self, text: str) -> None:
        """Corner-anchored distance-scale overlay: a short data-space line plus the caller-
        supplied mono label. The caller already ran :func:`nice_round_scalebar` to build ``text``
        (a number plus its unit); the bar's on-screen LENGTH is re-derived from that same function
        against the canvas's own current view width, so the drawn bar always matches the label
        sitting under it. Pass ``""`` to clear."""
        self.scale_bar_label.setText(text)
        self.scale_bar_label.setVisible(bool(text))
        self._reposition_overlays()

    def _draw_scale_bar(self) -> None:
        """The bar itself: one horizontal line, at the length its label claims, sitting directly
        above that label in the bottom-left corner. Re-run on every camera move, so the drawn
        length always answers to the width currently visible."""
        if not self.scale_bar_label.text():
            self.scale_bar_line.setData([], [])
            return
        (x0, x1), _ = self.view.viewRange()
        width = x1 - x0
        if not np.isfinite(width) or width <= 0:
            self.scale_bar_line.setData([], [])
            return
        _, bar_len = nice_round_scalebar(width)
        anchor = self._data_at(_MARGIN_PX, self.scale_bar_label.y() - _LABEL_GAP_PX)
        self.scale_bar_line.setData([anchor.x(), anchor.x() + bar_len],
                                    [anchor.y(), anchor.y()])

    def set_wavelet_bar(self, x_theta, theta, x_psi, psi, label: str = "") -> None:
        """Corner-anchored wavelet-bar DUO from already-rendered ``(x_theta, theta)`` (the
        smoothing curve) and ``(x_psi, psi)`` (the analyzing wavelet), in pixel units, plus the
        mono label naming both kernels (see :meth:`_update_wavelet_bar`). Any of the four being
        ``None`` clears all four retained arrays and both curve items.

        The two curves keep SEPARATE x arrays deliberately: :func:`dynamix.core.scale_units.
        kernel_section` samples each section over ITS OWN support (+/-(outer_extremum + 3) sigma,
        at least +/-4 sigma), which differs by kernel and derivative order -- theta's support is
        not psi's. Sharing one x array between the two would silently mis-sample one of them.

        The RAW curves are retained, at whatever amplitude :func:`dynamix.core.scale_units.
        kernel_section` gave them -- the anchoring AND the shared amplitude normalization (see
        :meth:`_draw_wavelet`) are display translations into the current view, so a zoom or pan
        has to re-apply them, and re-rendering the kernels to do that would be needless work on a
        gesture that must stay interactive.
        """
        if x_theta is None or theta is None or x_psi is None or psi is None:
            self._wavelet_curve = None
        else:
            self._wavelet_curve = (np.asarray(x_theta, dtype=np.float64),
                                   np.asarray(theta, dtype=np.float64),
                                   np.asarray(x_psi, dtype=np.float64),
                                   np.asarray(psi, dtype=np.float64))
        self.wavelet_label.setText(label if self._wavelet_curve is not None else "")
        self.wavelet_label.setVisible(bool(self.wavelet_label.text()))
        self._reposition_overlays()

    def _draw_wavelet(self) -> None:
        """Anchor the retained kernel DUO under its own label, top-left, in data units.

        WIDTH stays true data px -- the footprint is drawn TO SCALE against the raster it sits on,
        same as before this slice. HEIGHT is normalized (:func:`wavelet_bar_scale_factor`): both
        curves are scaled by ONE shared factor so the taller one's peak-to-peak span is exactly
        ``WAVELET_BAR_HEIGHT_FRAC`` of the CURRENT view's y-range. Height therefore carries NO
        reading of its own -- it exists purely so the duo stays legible whether the view is zoomed
        to a 4-pixel-wide crop or the whole raster, which is exactly why it is free to be
        arbitrary. Recomputed on every view-range change (see ``_on_view_range_changed``), same as
        the distance bar's length.
        """
        if self._wavelet_curve is None:
            self.smoother_item.setData([], [])
            self.wavelet_item.setData([], [])
            return
        x_theta, theta, x_psi, psi = self._wavelet_curve
        (x0, x1), (y0, y1) = self.view.viewRange()
        width, height = x1 - x0, y1 - y0
        if ((x_theta.size or x_psi.size) and np.isfinite(width) and width > 0
                and np.isfinite(height) and height > 0):
            anchor = self._data_at(
                _MARGIN_PX,
                self.wavelet_label.y() + self.wavelet_label.height() + _LABEL_GAP_PX)
            factor = wavelet_bar_scale_factor(theta, psi, WAVELET_BAR_HEIGHT_FRAC * height)
            x_theta = x_theta - (x_theta.min() if x_theta.size else 0.0) + anchor.x()
            theta = theta * factor + anchor.y()
            x_psi = x_psi - (x_psi.min() if x_psi.size else 0.0) + anchor.x()
            psi = psi * factor + anchor.y()
        self.smoother_item.setData(x_theta, theta)
        self.wavelet_item.setData(x_psi, psi)

    def _update_wavelet_bar(self, result: dict, scale_idx: int) -> None:
        """Best-effort wavelet-bar DUO for ``result`` at ``scale_idx``: theta, the smoothing
        function (``deriv_order=0``), and psi, the analyzing wavelet actually convolved with the
        field (``deriv_order=1``).

        The kernels come from :mod:`dynamix.core.scale_units` -- theta the isotropic radial
        profile, psi the along-gradient section, the wavelet's TRUE 2D shape rather than a 1D
        g-family stand-in (the psi-formula adjudication, spec 2026-08-10 sec 2). That module is
        pure numpy; nothing in this path imports ``wtmm``, so the bar renders identically whether
        or not the optional, private ``wtmm`` package is even installed. Cleared
        (never raised) when: ``result`` carries no wavelet-transform params at all (the audit's
        semantic guard -- a wavelet bar on a non-wavelet layer is misleading), or the transform's
        ``wavelet`` choice can't be resolved to a kernel.

        WTMM2D's own ``wavelet`` param (``"mexican"``/``"gaussian"``) is passed straight through
        as :func:`dynamix.core.scale_units.kernel_section`'s ``smoothing`` argument. The scale it
        renders is ``result["_scale_idx"]`` where present -- see the comment in the body: the
        argument indexes the extrema layers, which is a different axis.

        :func:`dynamix.core.scale_units.analyzing_names` is the table that names the resolved
        kernels for the label: ``"gaussian" -> "g0"/"g1"``, ``"mexican" -> "g2"/"g3"`` -- BOTH of
        WTMM2D's own choices resolve for BOTH derivative orders, so the wavelet bar always renders
        the CORRECT kernels, never merely representative ones. The catch below is defense against
        a smoothing name the registry doesn't know -- a stale project file, or a raw dict built
        outside the device schema -- for which ``kernel_section``/``analyzing_names``/
        ``lambda_peak_px`` raise ``KeyError``/``ValueError``; caught here and treated as "skip" --
        drawing nothing is more honest than drawing the wrong kernel shape.

        The label also names sigma (the smoother's real-space width,
        :func:`dynamix.core.scale_units.sigma_px`) and lambda (the analyzing wavelet's bandpass
        peak wavelength, :func:`dynamix.core.scale_units.lambda_peak_px`) -- the two per-wavelet
        numbers a reading of this bar is actually FOR, not just which registry names got resolved.

        The label's physical half (``≈ {phys}``) goes through
        :func:`dynamix.shell.units.px_to_metres` on the field :meth:`set_field` last retained --
        the one conversion every physical reading in the shell agrees through (the transport, the
        distance scale bar, this bar, and WTMM2D's own derived reading). Omitted whenever that
        conversion returns ``None`` (no field on record yet, or an axis too short to have a pixel
        size at all) -- there is no honest physical number to show, not a fabricated ``1.0``.
        """
        params = result.get("params") or {}
        scales = result.get("scales")
        # ``scale_idx`` addresses ``extrema``, which is NOT the same axis as ``scales`` once a
        # filter has selected a scale: ``ScaleSelect`` hands on a ONE-layer list (so every caller
        # passes 0) and stamps the absolute index it picked as ``_scale_idx``. Looking the kernel
        # up by the layer index therefore pins the wavelet footprint to the finest scale forever,
        # while the extrema under it change with every scrub -- the overlay would quietly
        # contradict the picture it sits on. The stamp is the scale axis's own index; the argument
        # is the fallback for a result that never went through ScaleSelect.
        a_idx = int(result.get("_scale_idx", scale_idx))
        if not params.get("wavelet") or scales is None or not (0 <= a_idx < len(scales)):
            self.set_wavelet_bar(None, None, None, None)
            return
        a = float(scales[a_idx])
        wavelet = params["wavelet"]
        from dynamix.core.scale_units import (analyzing_names, kernel_section,
                                              lambda_peak_px, sigma_px)
        try:
            x_theta, theta = kernel_section(a, wavelet, 0)
            x_psi, psi = kernel_section(a, wavelet, 1)
            theta_name, psi_name = analyzing_names(wavelet)
            lam = lambda_peak_px(a, wavelet, 1)
        except (KeyError, ValueError):
            # A wavelet name the registry doesn't know (stale project file, raw dict built
            # outside the device schema): drawing nothing stays more honest than drawing
            # the wrong kernel shape.
            self.set_wavelet_bar(None, None, None, None)
            return
        sigma = sigma_px(a)
        label = f"θ {theta_name} · ψ {psi_name} @ a = {a:.1f} · σ {sigma:.1f} px · λ {lam:.1f} px"
        m_per_px, unit = px_to_metres(self._field)
        if m_per_px is not None:
            label = (f"θ {theta_name} · ψ {psi_name} @ a = {a:.1f}"
                     f" · σ {sigma:.1f} px ≈ {sigma * m_per_px:.3g} {unit}"
                     f" · λ {lam:.1f} px ≈ {lam * m_per_px:.3g} {unit}")
        self.set_wavelet_bar(x_theta, theta, x_psi, psi, label=label)
