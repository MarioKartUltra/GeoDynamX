# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for MainWindow._sync_arrangement -- multi-layer resolve orchestration and
set_layers wiring ("Multi-layer resolve").

Offscreen Qt (QT_QPA_PLATFORM=offscreen, mandated repo-wide): ``ArrangementView.activate()``
detects the offscreen QPA platform and leaves the real ``pyvistaqt``/``Scene`` build unattempted
(tests/test_arrangement_flip.py pins this down) -- ``window._arrangement._scene`` stays ``None``
throughout this whole file, so ``ArrangementView.set_layers`` only ever stores ``self._layers``
(never forwards to a real Scene). That is exactly what these tests need: they are about WHICH
entries ``MainWindow`` hands to the arrangement and WHEN, not about the pyvista rendering itself
(the job, covered in tests/test_arrangement_scene.py).

Stub devices (``STUB_CHAIN``, ``_StubStack``, ``_BoomStack``, ``_BlockingStack``) come from
tests/test_shell_window.py -- same convention tests/test_arrangement_flip.py already follows.
Extra layers are added directly via ``win.project.add_layer``/``win.add_layer_row`` (never through
``load_field``, which would switch the active layer and auto-dispatch) -- the same pattern
tests/test_shell_roi_flow.py's ``two_layer_window`` fixture uses, chosen here so each layer's
cache state (hit / genuine miss / errored) is exactly controlled rather than incidentally
resolved.

The georeferenced field fixture follows tests/test_geo_mapping.py's approach (a synthetic GeoTIFF
written by rasterio) and tests/test_shell_roi_flow.py's ``parent_tif`` (a plain UTM CRS is enough
here; the WKT/NAD27 detail in test_geo_mapping.py's own fixture is about the CRS TRANSFORM math,
already covered there and untouched by this task).
"""
from __future__ import annotations

import threading

import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.chain import Chain, DeviceRef

from tests.test_shell_window import (STUB_CHAIN, _BlockingStack, _FIELD, _StubStack, _stub_stack,
                                     stub_devices, window)  # noqa: F401


class _BlockingBoomStack(_StubStack):
    """Like ``_BlockingStack`` (test_shell_window.py), but RAISES once released instead of
    returning a result -- lets a test control exactly when a mid-compute error lands, needed to
    reproduce the review's "resolve-while-errored" repro precisely."""

    name = "blocking_boom_stack"

    def __init__(self):
        self.released = threading.Event()

    def compute(self, field, params, *, progress=None):
        if progress:
            progress("cwt", 0.0)
        self.released.wait(timeout=10.0)
        raise RuntimeError("cwt blew up mid-compute")

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
from rasterio.transform import from_origin      # noqa: E402

_GEO_CRS = "EPSG:32615"


def _write_geo_tif(path, values):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float64", crs=_GEO_CRS,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(values, 1)


def _geo_field(tmp_path, label):
    """A real georeferenced RasterField, same (16, 16) shape as STUB_CHAIN's own ``_FIELD`` --
    small and fast, same synthetic-GeoTIFF approach as test_geo_mapping.py's fixture."""
    path = tmp_path / f"{label}.tif"
    _write_geo_tif(path, _FIELD)
    return RasterField.from_geotiff_window(str(path), row_off=0, col_off=0, height=16, width=16)


def _bare_field(label="bare"):
    """A field with no CRS in its provenance at all -- LocalFrame, matching
    test_geo_mapping.py's own no-georeference fixture (``test_field_with_no_crs_in_provenance_
    raises_no_georeference``)."""
    return RasterField._from_bare_array(_FIELD.copy(), label, frame=LocalFrame(units="px"),
                                        name=label)


def _chain_like_stub(first_device: str, first_params: dict | None = None) -> Chain:
    """STUB_CHAIN with its own transform swapped for ``first_device`` -- the same
    ``(("boom_stack", {}),) + STUB_CHAIN[1:]`` idiom tests/test_shell_window.py uses to build a
    whole window's steps, here as a raw ``Chain`` for a layer added directly."""
    steps = [DeviceRef(first_device, dict(first_params or {}))]
    steps += [DeviceRef(n, dict(p)) for n, p in STUB_CHAIN[1:]]
    return Chain(tuple(steps))


def _entry_for(win, layer_id: int) -> dict:
    return next(e for e in win._arrangement._layers if e["layer"].layer_id == layer_id)


@pytest.fixture
def arranged_window(qtbot, stub_devices, tmp_path):
    """A window with three layers over three DIFFERENT sources (so their cache keys never
    collide) and the active layer left on ``cached`` throughout -- matching the real usage this
    task is about: you look at one layer while the arrangement shows every visible one.

    - ``cached``: georeferenced, already resolved via the ordinary ``load_field`` auto-dispatch --
      a cache HIT by the time any test body runs.
    - ``pending``: georeferenced, added directly (never through ``load_field``, never selected) --
      a genuine cache MISS, untouched until ``_sync_arrangement`` queues and dispatches it.
    - ``nogeoref``: a bare (no-CRS) field -- never reaches a cache probe at all.
    """
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)

    cached_field = _geo_field(tmp_path, "cached")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(cached_field, str(tmp_path / "cached.tif"))
    cached = win.layer

    pending_field = _geo_field(tmp_path, "pending")
    pending_src = win.project.add_source(str(tmp_path / "pending.tif"))
    pending = win.project.add_layer("pending", pending_src.source_id, _chain_like_stub("stub_stack"))
    win.add_layer_row(pending, pending_field)

    nogeoref_src = win.project.add_source("mem:nogeoref")
    nogeoref = win.project.add_layer("nogeoref", nogeoref_src.source_id,
                                     _chain_like_stub("stub_stack"))
    win.add_layer_row(nogeoref, _bare_field("nogeoref"))

    assert win.layer is cached       # the fixture's own invariant: no incidental re-selection
    return win, cached, pending, nogeoref


# --------------------------------------------------------------------------- classification


def test_cache_hit_layer_is_ok_immediately(qtbot, arranged_window):
    win, cached, pending, nogeoref = arranged_window

    win._toggle_center_view()        # flip in -> _sync_arrangement runs synchronously

    entry = _entry_for(win, cached.layer_id)
    assert entry["status"] == "ok"
    assert entry["result"] is not None
    assert entry["result"]["scales"]       # a real WTMM-shaped result, not a placeholder


def test_never_computed_layer_becomes_computing_then_ok(qtbot, arranged_window):
    win, cached, pending, nogeoref = arranged_window

    win._toggle_center_view()

    entry = _entry_for(win, pending.layer_id)
    assert entry["status"] == "computing"
    assert entry["result"] is None
    assert win.is_computing is True          # queued and dispatched, not merely labelled

    qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)

    entry = _entry_for(win, pending.layer_id)
    assert entry["status"] == "ok"
    assert entry["result"] is not None
    assert entry["result"]["scales"]


def test_hidden_layer_is_excluded_from_the_arrangement(qtbot, arranged_window):
    win, cached, pending, nogeoref = arranged_window

    win._on_hide_toggled(pending.layer_id, True)     # hidden BEFORE the first flip
    win._toggle_center_view()

    ids = {e["layer"].layer_id for e in win._arrangement._layers}
    assert pending.layer_id not in ids
    assert cached.layer_id in ids


def test_hiding_while_flipped_resyncs_and_excludes_the_layer(qtbot, arranged_window):
    win, cached, pending, nogeoref = arranged_window

    win._toggle_center_view()
    assert any(e["layer"].layer_id == pending.layer_id for e in win._arrangement._layers)

    win._on_hide_toggled(pending.layer_id, True)     # hidden WHILE already flipped

    ids = {e["layer"].layer_id for e in win._arrangement._layers}
    assert pending.layer_id not in ids


def test_no_crs_layer_is_listed_as_no_georeference(qtbot, arranged_window):
    win, cached, pending, nogeoref = arranged_window

    win._toggle_center_view()

    entry = _entry_for(win, nogeoref.layer_id)
    assert entry["status"] == "no-georeference"
    assert entry["result"] is None


# --------------------------------------------------------------------- Colormap/vtrail_color


def test_entries_default_to_the_canvas_constants(qtbot, arranged_window):
    """A layer with no ``ui.colormap``/``ui.color_vtrail`` tags at all -- every layer before this
    task, and every layer this task's own defaults leave untouched -- carries exactly what the
    scene already drew before these keys existed."""
    from dynamix.shell.canvas import VTRAIL_COLOR

    win, cached, pending, nogeoref = arranged_window

    win._toggle_center_view()

    entry = _entry_for(win, cached.layer_id)
    assert entry["colormap"] == "viridis"
    assert entry["vtrail_color"] == VTRAIL_COLOR


def test_entries_carry_the_layers_own_colormap_and_vtrail_color_tags(qtbot, arranged_window):
    win, cached, pending, nogeoref = arranged_window
    cached.tags["ui.colormap"] = "plasma"
    cached.tags["ui.color_vtrail"] = "#112233"

    win._toggle_center_view()

    entry = _entry_for(win, cached.layer_id)
    assert entry["colormap"] == "plasma"
    assert entry["vtrail_color"] == (0x11, 0x22, 0x33)


def test_non_ok_entries_carry_colormap_and_vtrail_color_too(qtbot, arranged_window):
    """Carried on every status, not only "ok" -- ``pending`` (still "computing" the moment the
    arrangement first flips in) and ``nogeoref`` ("no-georeference") both get it too, so a later
    landing that flips their status to "ok" never needs a second code path to backfill it."""
    win, cached, pending, nogeoref = arranged_window
    pending.tags["ui.colormap"] = "magma"

    win._toggle_center_view()

    computing_entry = _entry_for(win, pending.layer_id)
    assert computing_entry["status"] == "computing"
    assert computing_entry["colormap"] == "magma"

    noref_entry = _entry_for(win, nogeoref.layer_id)
    assert noref_entry["status"] == "no-georeference"
    assert "colormap" in noref_entry and "vtrail_color" in noref_entry


def test_display_style_change_resyncs_arrangement_for_colormap_vtrail_and_points_color_only(
        qtbot, arranged_window):
    """Only the three keys the SCENE actually reads (colormap for a layer's own drape,
    color_vtrail as ``_chain_color``'s fallback, color_points for a point layer's own drape) trigger a resync while flipped in -- the other two new
    keys (color_hchain/color_extrema; show_trails is session-canvas-only too) do not, so resyncing
    for them would rebuild the arrangement for nothing it draws differently. See
    ``MainWindow._on_display_style_changed``'s own comment for the documented choice.

    ``color_points`` was DORMANT at first (no live control emitted it yet), so this gate
    originally covered only the first two -- a point-color edit made while flipped silently never
    reached the scene. Proven directly here via the same handler-call injection every other case
    in this test uses, since no real swatch control exists to drive it through yet."""
    win, cached, pending, nogeoref = arranged_window
    win._toggle_center_view()

    orig = win._sync_arrangement
    calls = []

    def spy():
        calls.append(True)
        orig()

    win._sync_arrangement = spy

    win._on_display_style_changed("opacity", 0.5)
    win._on_display_style_changed("color_hchain", "#112233")
    win._on_display_style_changed("color_extrema", "#445566")
    win._on_display_style_changed("show_trails", True)
    assert calls == []

    # 2026-08-30 ("controls only work in the raster view"): a colormap change on a PLAIN drape
    # is an in-place LUT swap (ArrangementView.set_colormap), never a resync; only a hillshaded
    # drape (baked RGBA) still needs the rebuild.
    win._on_display_style_changed("colormap", "magma")
    assert calls == []

    win.layer.tags["ui.hillshade"] = "True"
    win._on_display_style_changed("colormap", "viridis")
    assert calls == [True]

    win._on_display_style_changed("color_vtrail", "#778899")
    assert calls == [True, True]

    win._on_display_style_changed("color_points", "#aabbcc")
    assert calls == [True, True, True]


def test_display_style_colormap_change_does_not_resync_when_never_flipped(qtbot, arranged_window):
    """No arrangement has ever been built -- ``_on_display_style_changed`` must not build one just
    because a colormap tag changed; ``_toggle_center_view``'s own first flip resyncs from scratch
    regardless (spec: "flipping away/never-flipped drains and rebuilds nothing")."""
    win, cached, pending, nogeoref = arranged_window
    assert win._arrangement is None

    win._on_display_style_changed("colormap", "magma")

    assert win._arrangement is None


# --------------------------------------------------------------------------- session isolation


def test_canvas_untouched_by_a_queue_landing(qtbot, arranged_window):
    """The whole point of the arrangement's own resolve path: a background layer's compute
    landing must never reach ``MainWindow._apply``/``canvas.set_result`` -- ``resolved`` is the
    one signal every result that reaches the screen is emitted from (module docstring), so a
    landing that never re-emits it never touched the canvas."""
    win, cached, pending, nogeoref = arranged_window
    emissions: list = []
    win.resolved.connect(emissions.append)

    win._toggle_center_view()                # flip in; dispatches `pending` (cached is a hit)
    assert win.is_computing is True
    count_after_flip = len(emissions)

    qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)

    assert len(emissions) == count_after_flip          # no NEW result reached the canvas
    assert _entry_for(win, pending.layer_id)["status"] == "ok"


def test_flipping_away_leaves_an_in_flight_compute_to_land_harmlessly(qtbot, arranged_window):
    """The design: 'flipping away drains nothing (computes finish and land in cache)'.

    A landing while flipped away no longer repaints the (parked, invisible)
    arrangement's own entries -- so ``pending``'s entry stays whatever it was AT THE MOMENT of the
    flip-away ("computing") until the NEXT flip-in, which resyncs unconditionally. The compute
    itself still lands (in the cache), and flipping back in shows it correctly -- that is the
    actual "harmless" contract, not "the parked view's entries update live"."""
    win, cached, pending, nogeoref = arranged_window

    win._toggle_center_view()                 # flip in; dispatches `pending`
    assert win.is_computing is True
    win._toggle_center_view()                 # flip back to session BEFORE it lands
    assert win._center_stack.currentIndex() == 0

    set_layers_calls: list = []
    win._arrangement.set_layers = (lambda entries, _orig=win._arrangement.set_layers:
                                   (set_layers_calls.append(1), _orig(entries))[1])

    qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)   # lands anyway (no cancel)
    assert win._center_stack.currentIndex() == 0                   # still on the session view
    assert set_layers_calls == []          # no arrangement resync paid for a parked, invisible view

    win._toggle_center_view()                 # flip back IN -- resyncs unconditionally

    assert win._center_stack.currentIndex() == 1
    assert _entry_for(win, pending.layer_id)["status"] == "ok"     # cached, updated honestly


# --------------------------------------------------------------------------- active-layer priority


def test_active_layer_edit_while_flipped_still_correct_after_flip_back(qtbot, arranged_window):
    """A transform knob turned on the ACTIVE layer while the shared worker thread is busy with a
    background arrangement layer must still get serviced -- taking priority over the rest of the
    queue -- and the session must show the right (edited) result once flipped back."""
    win, cached, pending, nogeoref = arranged_window

    win._toggle_center_view()                          # flip in; dispatches `pending`
    assert win.is_computing is True                    # the background compute owns the thread

    with qtbot.waitSignal(win.resolved, timeout=15000) as sig:
        win.strips.strip(0).controls["n_scales"].valueChanged.emit(5)   # cached's own knob

    assert len(sig.args[0].result["scales"]) == 5
    assert win.layer is cached
    assert _entry_for(win, pending.layer_id)["status"] == "ok"     # the queue still drained

    win._toggle_center_view()                          # flip back to the session view

    assert win._center_stack.currentIndex() == 0
    assert win.layer is cached
    assert win.transport.clock.n_scales == 5
    assert len(win._scales) == 5


def test_layer_switch_mid_compute_to_a_cached_layer_does_not_wedge_the_session(
        qtbot, stub_devices, tmp_path):
    """CRITICAL: switching to an ALREADY-CACHED layer while a DIFFERENT layer's
    compute is in flight -- NO arrangement involved at all, ordinary session use.

    Pre-fix: ``_start_worker_for``'s busy branch marked the newly-active layer's strips
    "computing" and disabled the transport, then dropped the intent on the floor.
    ``_on_finished`` routes a landing by layer id now, so the OLD compute's landing took
    the "not for the active layer" branch and handed off to ``_dispatch_next``, which only
    redispatched on a genuine cache MISS -- and the newly-active layer was a cache HIT. Nothing
    ever redispatched: dead transport, a strip stuck reading "computing" forever, and the canvas
    never applying B's result -- ``resolved`` never fired again for it.

    ``self._active_pending`` fixes this by recording the intent unconditionally and consuming it
    unconditionally in ``_dispatch_next``, regardless of what the cache probe would have said.
    """
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    blocker = _BlockingStack()
    register_device(blocker)

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)

    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:b")
    b = win.layer                               # B: ordinary, already resolved -- a cache HIT

    c = win.project.add_layer(
        "c", win.project.add_source("mem:c").source_id,
        Chain((DeviceRef("blocking_stack", {}),) +
              tuple(DeviceRef(n, dict(p)) for n, p in STUB_CHAIN[1:])))
    win.add_layer_row(c, _FIELD)

    resolved: list = []
    win.resolved.connect(resolved.append)

    try:
        win.layer_list.select_layer(c.layer_id)          # C becomes active; its compute blocks
        qtbot.waitUntil(lambda: win.is_computing, timeout=5000)
        assert win.layer is c

        win.layer_list.select_layer(b.layer_id)           # switch to B -- ALREADY cached
        assert win.layer is b
        assert win.transport.isEnabled() is False         # marked computing; thread still busy
        assert win.strips.strip(0).state_dot.property("state") == "computing"

        blocker.released.set()                             # let C's own compute land
        qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)
    finally:
        blocker.released.set()

    assert win.layer is b
    assert win.transport.isEnabled() is True
    assert win.strips.strip(0).state_dot.property("state") != "computing"
    assert len(resolved) == 1                              # B's own (redispatched) result landed
    assert resolved[0].layer_id == b.layer_id


def test_a_filter_edit_during_a_background_compute_reaches_the_canvas_once_it_lands(
        qtbot, arranged_window):
    """IMPORTANT: a FILTER edit (not a transform) on the active layer, made
    while the shared worker thread is busy with a background arrangement layer, must reach the
    canvas once that background compute lands -- with NO further user event needed. Pre-fix this
    was healed only by the NEXT unrelated user action (or never, if none came): the knob showed
    0.5 while the canvas kept showing the unfiltered (0.0) result indefinitely."""
    win, cached, pending, nogeoref = arranged_window
    unfiltered = len(_stub_stack(_FIELD, 3)["extrema"][0]["x"])

    win._toggle_center_view()                              # flip in; dispatches `pending`
    assert win.is_computing is True

    win.strips.strip(3).controls["frac"].valueChanged.emit(0.5)   # modulus_threshold: a FILTER
    assert win.is_computing is True                        # unchanged -- filters dispatch no worker
    assert win._params[3]["frac"] == 0.5

    resolved: list = []
    win.resolved.connect(resolved.append)

    qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)   # NO further user action here

    assert len(resolved) == 1
    landed = resolved[0]
    assert landed.cache_misses == 0                         # the filter path stays a cache hit
    assert 0 < len(landed.result["extrema"][0]["x"]) < unfiltered


# --------------------------------------------------------------------------- own-compute pending
#
# `_active_pending="resolve"` (round 2's fix) must NEVER be recorded when the
# busy thread is the ACTIVE layer's OWN compute -- that landing already handles the resolve (on
# success, `_on_finished`'s active-match body calls `_resolve_now()` unconditionally, reading
# `_params`/`field` fresh) or the error (`_on_error`'s active-match body, via `_report_error`) by
# itself. The two tests below are its two repros.


def test_own_compute_erroring_after_a_pending_filter_edit_never_resolves_uncached(
        qtbot, stub_devices, tmp_path):
    """NEW CRITICAL: transform edit dispatches the active layer's OWN compute -> a filter edit
    lands while THAT SAME compute is still busy -> the compute RAISES. `_dispatch_next` must NOT
    call `_resolve_now()` here: nothing is cached for the raising chain, and `_resolve_now()` on
    the GUI thread with nothing cached is exactly the synchronous, uncached-transform freeze
    `_reresolve`'s own docstring forbids. The error must land normally instead, and the window
    must still be usable afterward (not wedged)."""
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    blocker = _BlockingBoomStack()
    register_device(blocker)

    win = MainWindow(steps=(("blocking_boom_stack", {}),) + STUB_CHAIN[1:])
    qtbot.addWidget(win)
    win.load_field(_FIELD, "mem:own-error")
    qtbot.waitUntil(lambda: win.is_computing, timeout=5000)

    win.strips.strip(3).controls["frac"].valueChanged.emit(0.5)   # filter edit, OWN compute busy
    assert win._active_pending is None            # the fix: never recorded for the own compute

    resolve_now_calls: list = []
    win._resolve_now = (lambda _orig=win._resolve_now:
                        (resolve_now_calls.append(win._errored), _orig())[1])

    try:
        blocker.released.set()
        with qtbot.waitSignal(win.errored, timeout=10000):
            pass
        qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)

        assert resolve_now_calls == []             # never called at all -- not even while errored
        assert win._errored is True
        assert win.strips.strip(0).state_dot.property("state") == "error"
        assert win.transport.isEnabled() is True    # re-enabled, not left dead

        # Recovery: the window is not wedged by the fix -- a further edit still routes correctly
        # to the worker (never a synchronous, uncached GUI-thread resolve), erroring cleanly again.
        win.strips.strip(3).controls["frac"].valueChanged.emit(0.25)
        assert win.is_computing is True
        with qtbot.waitSignal(win.errored, timeout=10000):
            pass
        assert resolve_now_calls == []
    finally:
        blocker.released.set()      # idempotent (already set on the success path); belt-and-suspenders


def test_own_compute_landing_ok_after_a_pending_filter_edit_resolves_exactly_once(
        qtbot, stub_devices, tmp_path):
    """NEW IMPORTANT: transform edit dispatches the active layer's OWN compute -> a filter edit
    lands while THAT SAME compute is still busy -> the compute lands SUCCESSFULLY.
    `_on_finished`'s active-match body already calls `_resolve_now()` unconditionally once;
    `_dispatch_next` must not call it again for the same landing -- `resolved` fires exactly ONCE,
    carrying the filter edit's effect."""
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    blocker = _BlockingStack()
    register_device(blocker)

    win = MainWindow(steps=(("blocking_stack", {}),) + STUB_CHAIN[1:])
    qtbot.addWidget(win)
    win.load_field(_FIELD, "mem:own-ok")
    qtbot.waitUntil(lambda: win.is_computing, timeout=5000)

    win.strips.strip(3).controls["frac"].valueChanged.emit(0.5)   # filter edit, OWN compute busy
    assert win._active_pending is None
    assert win._params[3]["frac"] == 0.5

    resolved: list = []
    win.resolved.connect(resolved.append)

    try:
        blocker.released.set()
        qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)
    finally:
        blocker.released.set()

    assert len(resolved) == 1                       # not twice
    unfiltered = len(_stub_stack(_FIELD, 3)["extrema"][0]["x"])
    assert 0 < len(resolved[0].result["extrema"][0]["x"]) < unfiltered   # the filter edit landed


# --------------------------------------------------------------------------- errors


def test_a_background_layer_error_lists_with_the_error_and_is_not_retried(qtbot, stub_devices,
                                                                          tmp_path):
    # `stub_devices` already registers `_BoomStack()` -- no need to register it again here.
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    cached_field = _geo_field(tmp_path, "cached")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(cached_field, str(tmp_path / "cached.tif"))
    cached = win.layer

    boom_field = _geo_field(tmp_path, "boom")
    boom_src = win.project.add_source(str(tmp_path / "boom.tif"))
    boom = win.project.add_layer("boom", boom_src.source_id, _chain_like_stub("boom_stack"))
    win.add_layer_row(boom, boom_field)

    win._toggle_center_view()
    qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)

    entry = _entry_for(win, boom.layer_id)
    assert entry["status"].startswith("error:")
    assert "cwt blew up" in entry["status"]
    assert entry["result"] is None
    # The active layer's own strips are a DIFFERENT layer's UI -- never touched by this error.
    assert win.layer is cached
    assert win.strips.strip(0).state_dot.property("state") != "error"

    # A later resync (a second hideToggled, say) must not silently retry the same failing chain
    # forever -- the status is remembered, not rediscovered by another doomed dispatch.
    win._sync_arrangement()
    assert _entry_for(win, boom.layer_id)["status"] == entry["status"]
    assert boom.layer_id not in win._arr_queue
    assert win.is_computing is False


# --------------------------------------------------------------------------- close during compute


def test_close_during_an_arrangement_queue_compute_does_not_crash(qtbot, stub_devices, tmp_path):
    """Mirrors tests/test_shell_window.py::test_close_during_compute_waits_for_the_stage, but the
    blocked compute belongs to a BACKGROUND arrangement layer, never the active one -- the new
    code path (``_land_after_worker``'s background branch, reached through ``_sync_arrangement``)
    still has to defer the close honestly and complete once the stage does."""
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    blocker = _BlockingStack()
    register_device(blocker)

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    win.show()

    cached_field = _geo_field(tmp_path, "cached")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(cached_field, str(tmp_path / "cached.tif"))
    cached = win.layer

    blocking_field = _geo_field(tmp_path, "blocking")
    blocking_src = win.project.add_source(str(tmp_path / "blocking.tif"))
    blocking = win.project.add_layer("blocking", blocking_src.source_id,
                                     _chain_like_stub("blocking_stack"))
    win.add_layer_row(blocking, blocking_field)

    try:
        win._toggle_center_view()                          # flip in; dispatches `blocking`
        qtbot.waitUntil(lambda: win.is_computing, timeout=5000)
        win.close()
        assert win.isVisible()                              # deferred, not refused
        assert win.layer is cached                           # the active layer never changed
    finally:
        blocker.released.set()

    qtbot.waitUntil(lambda: not win.isVisible(), timeout=10000)
    assert win.is_computing is False
