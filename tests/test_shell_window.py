# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.main_window -- the window, wired end to end.

Offscreen Qt (the runner sets ``QT_QPA_PLATFORM=offscreen``). Most of these drive a STUB
transform rather than the real ``wtmm2d``: a stub keeps the widget suite fast and keeps the
wiring under test (worker dispatch, the filter re-resolve, transport routing, the close path)
separate from whether the WTMM backend is fast today. The stub's result is shaped like a real
WTMM stack -- ``scales`` / ``extrema`` / ``_shape`` -- so the REAL filters and the REAL canvas
run over it; only the expensive transform is stood in for. ``test_real_wtmm_end_to_end`` then
runs the actual chain, unguarded, exactly as ``tests/test_real_devices.py`` does: the WTMM
backend under ``dynamix/core`` is a self-contained copy with a numpy fallback, so it needs no
optional package (``wtmm_ebsd`` is only needed by the chain-FILTER devices, which this chain
does not use).
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui

from dynamix.core.frames import GeographicFrame, LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices
from dynamix.model.param import Param, ParamKind

# --------------------------------------------------------------------------- stub devices


def _stub_stack(field, n_scales: int) -> dict:
    """A WTMM-SHAPED result: one extrema layer per scale, each with the per-point arrays the
    filters and the canvas actually read. Point counts fall off with scale, as real ones do."""
    values = np.asarray(getattr(field, "values", field), dtype=np.float64)
    ny, nx = values.shape[:2]
    layers = []
    for k in range(n_scales):
        m = max(4, 12 - 2 * k)
        idx = np.arange(m, dtype=np.int64)
        layers.append({
            "x": idx % nx,
            "y": (idx * 2) % ny,
            "mod": np.linspace(1.0, 0.1, m),
            "arg": np.linspace(0.0, np.pi, m),
            # the first four points form one ordered line; the rest are isolated extrema
            "line_id": np.where(idx < 4, 0, -1).astype(np.int64),
        })
    return {"scales": [2.0 * (k + 1) for k in range(n_scales)], "extrema": layers,
            "_shape": (ny, nx), "params": {}}


class _StubStack:
    """Stand-in for wtmm2d: a Transform (compute + cache_key) with one param to turn."""

    name = "stub_stack"
    params = (Param("n_scales", ParamKind.INT, default=3, min=1, max=8, label="Scales"),)

    def compute(self, field, params, *, progress=None):
        if progress:
            progress("stub", 0.0)
            progress("stub", 1.0)
        return _stub_stack(field, int(params["n_scales"]))

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


class _StubTopology:
    """A SECOND transform, so a test chain has more than one transform strip.

    ``resolve`` threads the previous transform's RESULT into the next one's ``compute`` (not the
    raw field), so this takes a result dict and hands it straight back -- the point is only that
    the window sees two transform strips, as the real ``wtmm2d`` + ``chain_topology`` pair does.
    """

    name = "stub_topology"
    params = ()

    def compute(self, result, params, *, progress=None):
        return dict(result)

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


class _BoomStack(_StubStack):
    name = "boom_stack"

    def compute(self, field, params, *, progress=None):
        raise RuntimeError("cwt blew up")


class _BlockingStack(_StubStack):
    """Waits on an Event so a test can hold a compute open while it closes the window."""

    name = "blocking_stack"

    def __init__(self):
        self.released = threading.Event()

    def compute(self, field, params, *, progress=None):
        if progress:
            progress("cwt", 0.0)
        self.released.wait(timeout=10.0)
        return _stub_stack(field, int(params["n_scales"]))


#: The stub chain: one stub transform, then the REAL filters, in the window's own order.
STUB_CHAIN = (
    ("stub_stack", {}),
    ("scale_select", {"scale_idx": 0}),
    ("orientation_wedge", {"centre": 0.0, "half_width": 90.0}),
    ("modulus_threshold", {"frac": 0.0}),
)

_FIELD = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16)

# Auto-run-on-by-default for this whole file (so ``load_field`` dispatches the worker exactly as
# it always did, which most of these tests are actually about) now lives in ONE suite-wide
# fixture, ``tests/conftest.py::_dynamix_settings_isolated`` -- it also isolates every
# MainWindow-constructing test in the repo from the real per-machine settings file, which a
# per-file fixture here could not do. The two inert-open tests below still see the true (off)
# default via their own env/save_settings calls inside the test body.


@pytest.fixture
def stub_devices(clean_registry):
    """Real registry plus the stubs. ``clean_registry`` restores it afterwards."""
    from dynamix.model.device import register_device

    register_builtin_devices()
    register_device(_StubStack())
    register_device(_BoomStack())
    register_device(_StubTopology())
    return clean_registry


@pytest.fixture
def window(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    return win


@pytest.fixture
def loaded(qtbot, window):
    """A window whose first (worker-dispatched) resolve has landed."""
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.load_field(_FIELD, "mem:stub")
    return window


# --------------------------------------------------------------------------- construction


def test_window_constructs_with_stub_chain(qtbot, window):
    """Zones exist and the chain generated one strip per step -- no hand-written per-device UI."""
    window.load_field(_FIELD, "mem:stub")
    assert window.canvas is not None and window.transport is not None
    assert [window.strips.strip(i).device.name for i in range(len(STUB_CHAIN))] == \
        [name for name, _ in STUB_CHAIN]
    assert window.layer_list.count() == 1
    assert window.centralWidget() is not None


def test_remove_and_bypass_buttons_are_enabled_on_every_shipped_strip(qtbot, window):
    """Both title-bar buttons are real now. ``WorkflowZone.chainEdited`` round-trips into
    ``main_window._on_chain_edited``, which rebuilds ``_names``/``_params``/bypass/rack wholesale
    from the payload every time a chain-editing gesture fires -- there is no index-based
    bookkeeping left for a live button to desync (the Critical, reversed)."""
    window.load_field(_FIELD, "mem:stub")
    for i in range(len(STUB_CHAIN)):
        box = window.strips.strip(i)
        assert box.remove_button.isEnabled() is True, i
        assert box.bypass_button.isEnabled() is True, i


def test_a_real_click_on_a_shipped_strips_remove_button_reindexes_without_desyncing(qtbot, window):
    """The old regression test's own scenario, now proving the OPPOSITE (wired) outcome: a real
    click (``qtbot.mouseClick``) on step 1's remove button (``scale_select``, a filter -- removing
    it does not touch the transform signature) actually removes it, reindexing every later step --
    and a param edit on the box that shifted into index 1 (``orientation_wedge``) must land on ITS
    OWN (new) step, not on whatever used to live there. That is the desync the Critical was about,
    now proven closed rather than proven impossible-to-trigger."""
    window.load_field(_FIELD, "mem:stub")

    qtbot.mouseClick(window.strips.strip(1).remove_button, QtCore.Qt.LeftButton)

    after = [window.strips.strip(i).device.name for i in range(len(STUB_CHAIN) - 1)]
    assert after == ["stub_stack", "orientation_wedge", "modulus_threshold"]
    assert window._names == after                     # main_window's own state kept in step

    with qtbot.waitSignal(window.strips.paramChanged) as sig:
        window.strips.strip(1).controls["centre"].valueChanged.emit(45.0)
    assert sig.args[0] == 1                            # orientation_wedge's NEW index
    assert window._params[1]["centre"] == 45.0         # written into the right (shifted) step


def test_first_load_runs_on_the_worker_and_sizes_the_transport(qtbot, window):
    with qtbot.waitSignal(window.resolved, timeout=10000) as sig:
        window.load_field(_FIELD, "mem:stub")
    renderable = sig.args[0]
    assert renderable.result["_scale_idx"] == 0
    assert window.transport.clock.n_scales == 3          # len(result["scales"])
    assert window.is_computing is False


# --------------------------------------------------------------------------- inert open


def test_inert_open_builds_empty_chain_and_does_not_compute(qtbot, window, tmp_path, monkeypatch):
    """Opening a file is INERT by default. No settings file exists at the path this test
    points at, so ``load_settings()`` falls back to ``auto_run_wtmm=False`` and ``load_field``
    must not build a chain or dispatch the worker -- only display the raster."""
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))   # auto_run off
    window.load_field(_FIELD, str(tmp_path / "f.npz"))
    assert window.layer.chain.steps == ()                # no devices
    assert window._thread is None                        # no worker dispatched
    assert window.canvas._field is not None              # raster IS displayed


def test_auto_run_setting_restores_todays_behavior(qtbot, tmp_path, monkeypatch, clean_registry):
    """Flipping the setting on restores the pre-Task-1 behaviour: a freshly opened file's layer
    gets the window's own default chain (``MainWindow.DEFAULT_STEPS``), exactly as an unguarded
    ``load_field`` always built it."""
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import Settings, save_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    save_settings(Settings(auto_run_wtmm=True))

    w = MainWindow()
    qtbot.addWidget(w)
    w.load_field(_FIELD, str(tmp_path / "f.npz"))
    assert [s.device for s in w.layer.chain.steps] == [n for n, _ in type(w).DEFAULT_STEPS]
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)   # drain the worker before teardown


def test_flipping_auto_run_after_an_inert_open_reseeds_the_default_chain(
        qtbot, tmp_path, monkeypatch, clean_registry):
    """CRITICAL regression: an inert open wipes ``_names``/``_params`` to empty,
    since the layer's chain is honestly empty. If auto-run is then switched on and the SAME
    window opens again -- no fresh ``MainWindow()`` -- ``load_field`` must not resolve that empty
    recipe while still dispatching a worker for it; the second open has to build
    ``DEFAULT_STEPS``, exactly as a brand-new auto-run window would.

    Reviewer's repro, verbatim: MainWindow() -> load_field (settings off) ->
    save_settings(Settings(auto_run_wtmm=True)) -> load_field again on the same window.
    """
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import Settings, save_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))   # auto_run off, no file

    w = MainWindow()
    qtbot.addWidget(w)
    w.load_field(_FIELD, str(tmp_path / "a.npz"))
    assert w.layer.chain.steps == ()
    assert w._thread is None

    save_settings(Settings(auto_run_wtmm=True))
    w.load_field(_FIELD, str(tmp_path / "b.npz"))

    assert [s.device for s in w.layer.chain.steps] == [n for n, _ in type(w).DEFAULT_STEPS]
    assert w.is_computing is True                                 # a worker WAS dispatched
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)     # drain it before teardown


def test_auto_run_toggle_does_not_clobber_splitter_sizes(qtbot, tmp_path, monkeypatch, clean_registry):
    """Regression: _on_auto_run_toggled used to call save_settings(Settings(auto_run_wtmm=checked)),
    which clobbered all other fields back to defaults. With update_settings, the toggle must
    preserve splitter_sizes and other fields."""
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import Settings, save_settings, load_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    # Pre-save settings with splitter_sizes set
    save_settings(Settings(splitter_sizes={"work": [1]}))

    w = MainWindow()
    qtbot.addWidget(w)

    # Toggle the auto-run action
    w._auto_run_action.toggled.emit(True)

    # Verify splitter_sizes survives
    s = load_settings()
    assert s.splitter_sizes == {"work": [1]}
    assert s.auto_run_wtmm is True


# --------------------------------------------------------------------------- THE LAW


def test_filter_knob_drag_is_zero_miss(qtbot, loaded):
    """A filter is instant BY CONSTRUCTION: turning one must re-run apply() over the cached
    transform and touch nothing expensive. The engine's own counters are the proof."""
    before = len(_stub_stack(_FIELD, 3)["extrema"][0]["x"])     # what scale 0 holds unfiltered
    strip = loaded.strips.strip(3)                       # modulus_threshold
    with qtbot.waitSignal(loaded.resolved, timeout=5000) as sig:
        strip.controls["frac"].valueChanged.emit(0.5)    # as a drag/nudge would propose
    renderable = sig.args[0]
    assert renderable.cache_misses == 0
    assert renderable.from_cache is True
    assert renderable.cache_hits >= 1                    # the transform was consulted and HIT
    # ... and the filter genuinely re-ran: zero misses must not mean "nothing happened"
    assert 0 < len(renderable.result["extrema"][0]["x"]) < before
    assert renderable.filters_run == ("scale_select", "orientation_wedge", "modulus_threshold")
    assert loaded.is_computing is False                  # no worker was dispatched
    assert loaded._params[3]["frac"] == 0.5


def test_scale_change_routes_through_scale_select_and_is_zero_miss(qtbot, loaded):
    with qtbot.waitSignal(loaded.resolved, timeout=5000) as sig:
        loaded.transport.slider.setValue(2)
    renderable = sig.args[0]
    assert renderable.result["_scale_idx"] == 2
    assert renderable.cache_misses == 0
    # the strip's own control was re-read from the params, so it does not go stale
    assert loaded.strips.strip(1).controls["scale_idx"].text() == "2"


def test_strip_scale_knob_drags_the_transport_with_it(qtbot, loaded):
    """The scale is addressable from the strip AND the transport. Whichever one did not
    originate the change still has to follow it, or the slider and its reading contradict the
    canvas and the next playback tick snaps back to the stale position."""
    emissions = []
    loaded.transport.scaleChanged.connect(emissions.append)

    with qtbot.waitSignal(loaded.resolved, timeout=5000) as sig:
        loaded.strips.strip(1).controls["scale_idx"].valueChanged.emit(2)

    assert sig.args[0].result["_scale_idx"] == 2
    assert loaded.transport.slider.value() == 2
    assert loaded.transport.clock.index == 2
    assert loaded.transport.reading_label.text() == loaded._scale_reading(2)
    assert emissions == [], "the transport echoed a change it did not originate"


def test_a_restored_scale_idx_survives_the_transport_re_ranging(qtbot, stub_devices):
    """C-2: a window built at scale 4 must open ON scale 4.

    ``_sync_transport`` re-ranges the slider after every recompute, and ``set_n_scales`` emits
    ``scaleChanged`` as it clamps -- from the transport's own position, which is still 0. That
    emission used to WRITE ``scale_idx = 0`` into the params (the ``_syncing`` flag suppressed
    only the resolve, not the write), so the scale a project file, a devloop rebuild, or the
    DEMO_CHAIN itself asked for was silently discarded on the very first frame. All three of
    params, slider and canvas have to land on the requested index.
    """
    from dynamix.shell.main_window import MainWindow

    steps = (STUB_CHAIN[0], ("scale_select", {"scale_idx": 2})) + STUB_CHAIN[2:]
    win = MainWindow(steps=steps)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000) as sig:
        win.load_field(_FIELD, "mem:restored")

    assert win._params[1]["scale_idx"] == 2
    assert win.layer.chain.steps[1].params["scale_idx"] == 2
    assert win.transport.slider.value() == 2
    assert win.transport.clock.index == 2
    assert sig.args[0].result["_scale_idx"] == 2          # what the canvas was actually given


def test_a_shrinking_scale_stack_clamps_params_and_transport_together(qtbot, loaded):
    """The other ordering: the transport is at 2 when a recompute leaves only ONE scale. Both the
    param and the slider must land on the clamped index -- if only the slider clamps, the chain
    keeps asking for a scale that no longer exists and the two displays disagree."""
    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        loaded.transport.slider.setValue(2)
    assert loaded._params[1]["scale_idx"] == 2

    loaded.strips.strip(0).controls["n_scales"].valueChanged.emit(1)      # one scale only
    with qtbot.waitSignal(loaded.resolved, timeout=10000) as sig:
        pass

    assert loaded._params[1]["scale_idx"] == 0
    assert loaded.transport.slider.value() == 0
    assert loaded.transport.clock.index == 0
    assert sig.args[0].result["_scale_idx"] == 0


def test_only_the_first_transform_strip_claims_the_compute(qtbot, stub_devices):
    """I-6: two transforms, ONE compute. Progress and elapsed describe the whole worker run, so
    exactly one strip may claim them -- both strips reading "cwt 40%" and then both reading
    "820 ms" said the run happened twice. The other transform keeps its state dot (which IS
    per-strip and honest) and an empty reading."""
    from dynamix.shell.main_window import MainWindow

    steps = (STUB_CHAIN[0], ("stub_topology", {})) + STUB_CHAIN[1:]
    win = MainWindow(steps=steps)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:two_transforms")

    assert win._transform_indices() == [0, 1]
    assert win.strips.strip(0).reading_label.text().endswith(" ms")
    assert win.strips.strip(1).reading_label.text() == ""
    # the dot is per-strip and stays on both -- it says what THIS device's result is, not who ran
    assert win.strips.strip(0).state_dot.property("state") == "cached"
    assert win.strips.strip(1).state_dot.property("state") == "cached"


def test_only_the_terminal_filter_claims_the_point_count(qtbot, loaded):
    """A point count is a property of the END of the chain, not of any one step. Every filter
    strip showing it made four devices each claim the same number as its own effect."""
    readings = [loaded.strips.strip(i).reading_label.text() for i in range(len(STUB_CHAIN))]
    assert readings[-1].endswith(" pts")                  # modulus_threshold, the terminal filter
    assert readings[1] == "" and readings[2] == ""        # scale_select, orientation_wedge
    assert readings[0].endswith(" ms")                    # the transform reports its own cost


# --------------------------------------------------------------------------- chain editing


class _StubDropEvent:
    """Minimal ``QDropEvent`` stand-in -- see ``tests/test_workflow_zone.py``'s copy (and the module docstring's drag-drop section) for why: pytest-qt cannot synthesize a real native drag
    reliably offscreen, so the zone's ``dropEvent`` is called directly."""

    def __init__(self, mime: QtCore.QMimeData):
        self._mime = mime
        self.accepted = None

    def position(self):
        return QtCore.QPointF(0, 0)

    def mimeData(self):
        return self._mime

    def proposedAction(self):
        return QtCore.Qt.MoveAction

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def _device_drop(name: str) -> _StubDropEvent:
    from dynamix.shell.browser import DEVICE_MIME

    mime = QtCore.QMimeData()
    mime.setData(DEVICE_MIME, name.encode("utf-8"))
    return _StubDropEvent(mime)


def test_chain_edited_from_the_zone_rebuilds_bookkeeping_and_recomputes(qtbot, loaded):
    """The real end-to-end wiring, through an ACTUAL drop on the real zone the window ships (not
    a hand-built descriptor list): ``WorkflowZone.dropEvent`` -> ``chainEdited`` ->
    ``MainWindow._on_chain_edited`` rebuilds ``_names``/``_params`` and recomputes."""
    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        loaded.strips.dropEvent(_device_drop("min_vchains"))

    assert loaded._names[-1] == "min_vchains"
    assert loaded.layer.chain.steps[-1].device == "min_vchains"
    assert loaded._bypassed == [False] * len(loaded._names)
    assert loaded._rack == [None] * len(loaded._names)
    assert loaded.strips.strip(len(loaded._names) - 1).device.name == "min_vchains"


def test_bypassing_a_filter_excludes_it_from_the_chain_and_clears_its_reading(qtbot, loaded):
    """Bypass is real now: the excluded step vanishes from ``layer.chain`` (not just from the
    render), and its reading is cleared rather than left showing a stale, no-longer-honest value
    (the residual the design names). Toggling routes through the zone's own
    ``_commit_or_revert``, which REBUILDS the zone -- so the
    strip has to be re-fetched after the toggle; the original reference is torn down."""
    strip = loaded.strips.strip(3)                        # modulus_threshold
    assert strip.reading_label.text() != ""                # something honest was showing before

    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        strip.bypass_button.toggle()

    fresh = loaded.strips.strip(3)
    assert loaded._bypassed[3] is True
    assert "modulus_threshold" not in [ref.device for ref in loaded.layer.chain.steps]
    assert fresh.reading_label.text() == ""
    assert fresh.controls["frac"].isEnabled() is False


def test_a_real_click_on_the_bypass_button_toggles_it_end_to_end(qtbot, loaded):
    strip = loaded.strips.strip(3)
    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        qtbot.mouseClick(strip.bypass_button, QtCore.Qt.LeftButton)
    assert loaded._bypassed[3] is True


def test_racking_a_filter_does_not_change_the_transform_signature_or_miss_the_cache(qtbot, loaded):
    """Zero-cache-miss law extended across rack flattening: assembling
    [stub_stack][scale_select, orientation_wedge racked][modulus_threshold] must produce the SAME
    ``_transform_signature()`` as the same steps unracked -- rack is pure rendering, invisible to
    what the worker computes -- and a resolve after racking must be a cache hit, reusing slice-1's
    ``cache_misses``/``from_cache`` instrumentation."""
    unracked = [{"device": n, "params": dict(p), "bypassed": False, "rack": None}
                for n, p in zip(loaded._names, loaded._params)]

    loaded.strips.chainEdited.emit(unracked)
    sig_unracked = loaded._transform_signature()

    racked = [dict(d) for d in unracked]
    racked[1] = dict(racked[1], rack="post")               # scale_select
    racked[2] = dict(racked[2], rack="post")                # orientation_wedge

    with qtbot.waitSignal(loaded.resolved, timeout=5000) as sig:
        loaded.strips.chainEdited.emit(racked)
    sig_racked = loaded._transform_signature()

    assert sig_racked == sig_unracked
    assert sig.args[0].cache_misses == 0
    assert sig.args[0].from_cache is True
    assert loaded.is_computing is False                    # no worker was dispatched either time


# --------------------------------------------------------------------------- chain-edit guards


def test_on_chain_edited_leaves_bookkeeping_untouched_when_the_descriptor_list_is_illegal(
        qtbot, loaded):
    """``_on_chain_edited`` builds the candidate ``Chain`` FIRST, from local
    variables, and only assigns ``_names``/``_params``/``_bypassed``/``_rack``/``layer.chain``
    after it succeeds. Fed an illegal descriptor list directly -- bypassing the zone's own gate
    entirely, exactly the "a future bug slips one through" scenario the fix guards against -- the
    window's bookkeeping must stay untouched, not half-mutated with the bad recipe."""
    names_before = list(loaded._names)
    params_before = [dict(p) for p in loaded._params]
    chain_before = loaded.layer.chain

    illegal = [{"device": "scale_select", "params": {"scale_idx": 0}, "bypassed": False,
               "rack": None},
              {"device": "stub_stack", "params": {}, "bypassed": False, "rack": None}]

    with pytest.raises(ValueError):
        loaded._on_chain_edited(illegal)

    assert loaded._names == names_before
    assert loaded._params == params_before
    assert loaded.layer.chain is chain_before


def test_unbypassing_after_a_clamped_drop_never_raises(qtbot, loaded):
    """RETARGETED from the OLD
    ``test_unbypassing_after_a_refused_drop_never_raises``, which pinned a SILENT refusal here --
    the end-to-end shape of bug B1 ("plugins don't even drop in the box"), through the REAL window.
    The underlying guarantee still holds: MainWindow must never see, or build, an illegal
    chain. Under smart placement that no longer means the drop is refused -- a
    transform landing after a bypassed filter clamps to the end of the transform block instead,
    landing BEFORE it, and ``chainEdited`` DOES fire and DOES recompute this time. Un-bypassing the
    (still legal) result afterward still never raises -- the confirmed wedge (every
    subsequent knob edit dying on the same ``layer.chain`` assignment) still cannot reproduce."""
    strip = loaded.strips.strip(1)                          # scale_select, a filter
    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        strip.bypass_button.toggle()
    assert loaded._bypassed[1] is True

    # ``stub_topology`` (not a second ``stub_stack``): the SECOND transform stub is the one built
    # to accept a previous transform's RESULT (its own docstring, above) -- chaining two
    # ``stub_stack``s back to back is not a shape that stub's ``compute`` supports, orthogonal to
    # what this test is actually about.
    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        loaded.strips.dropEvent(_device_drop("stub_topology"))  # a transform CLAMPED to land
                                                                  # before the bypassed filter
    assert loaded._names == \
        ["stub_stack", "stub_topology", "scale_select", "orientation_wedge", "modulus_threshold"]
    assert loaded._bypassed[2] is True                        # scale_select, shifted to index 2

    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        loaded.strips.strip(2).bypass_button.toggle()        # un-bypass -- must not raise
    assert loaded._bypassed[2] is False


def test_bypass_survives_a_layer_switch_and_back(qtbot, loaded):
    """Bypass/rack are VIEW state ``main_window`` keeps per-layer in
    ``self._recipes`` -- a fresh derive from ``layer.chain`` (the old behaviour) would silently
    drop a bypassed step on every switch away and back, since a bypassed step is excluded from
    the persisted chain entirely."""
    from dynamix.model.chain import Chain, DeviceRef

    strip = loaded.strips.strip(3)                          # modulus_threshold
    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        strip.bypass_button.toggle()
    assert loaded._bypassed[3] is True
    first_layer = loaded.layer

    second_chain = Chain(tuple(DeviceRef(n, dict(p)) for n, p in STUB_CHAIN)).materialized()
    second_layer = loaded.project.add_layer("second", first_layer.source_id, second_chain)
    loaded.add_layer_row(second_layer, loaded.field)
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(second_layer.layer_id)

    assert loaded.layer is second_layer
    assert loaded._bypassed == [False] * len(loaded._names)     # a fresh, never-edited layer

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(first_layer.layer_id)

    assert loaded.layer is first_layer
    assert loaded._names[3] == "modulus_threshold"
    assert loaded._bypassed[3] is True
    assert "modulus_threshold" not in [ref.device for ref in loaded.layer.chain.steps]


# --------------------------------------------------------------------------- transform path


def test_transform_param_change_recomputes_via_worker(qtbot, loaded):
    strip = loaded.strips.strip(0)
    strip.controls["n_scales"].valueChanged.emit(5)
    assert strip.state_dot.property("state") == "computing"
    assert loaded.is_computing is True
    with qtbot.waitSignal(loaded.resolved, timeout=10000) as sig:
        pass
    assert len(sig.args[0].result["scales"]) == 5
    assert loaded.transport.clock.n_scales == 5
    assert strip.state_dot.property("state") == "cached"


def test_transport_is_disabled_while_a_transform_computes(qtbot, stub_devices):
    """I-1: during a recompute the raster, the overlay and the transport each showed a different
    scale -- the canvas its last good frame, the slider wherever the user dragged it, the reading
    a third thing. Scrubbing a stack that is being rebuilt cannot be answered honestly (there is
    no cancellation in v1), so the control is disabled for the duration and re-enabled the moment
    a real result is back."""
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    blocker = _BlockingStack()
    register_device(blocker)
    win = MainWindow(steps=(("blocking_stack", {}),) + STUB_CHAIN[1:])
    qtbot.addWidget(win)
    assert win.transport.isEnabled() is True

    try:
        win.load_field(_FIELD, "mem:disabled")
        qtbot.waitUntil(lambda: win.is_computing, timeout=5000)
        assert win.transport.isEnabled() is False
    finally:
        # release even on failure: a still-blocked worker thread outlives this test and takes
        # the whole run down with it (the same reason the close-during-compute test does this).
        with qtbot.waitSignal(win.resolved, timeout=10000):
            blocker.released.set()
    assert win.transport.isEnabled() is True


def test_transport_is_re_enabled_after_a_transform_raises(qtbot, stub_devices):
    """An error leaves the window usable: the strip says what failed, and the transport is not
    left permanently dead because the run that disabled it never reached _on_finished."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=(("boom_stack", {}),) + STUB_CHAIN[1:])
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.errored, timeout=10000):
        win.load_field(_FIELD, "mem:boom_enabled")
    assert win.transport.isEnabled() is True


def test_worker_error_marks_the_strip_and_shows_the_message(qtbot, stub_devices):
    """A transform that raises marks its STRIP and puts the message in its reading. No dialog."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=(("boom_stack", {}),) + STUB_CHAIN[1:])
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.errored, timeout=10000):
        win.load_field(_FIELD, "mem:boom")
    strip = win.strips.strip(0)
    assert strip.state_dot.property("state") == "error"
    assert "cwt blew up" in strip.reading_label.text()


def test_filter_change_after_an_error_goes_to_the_worker(qtbot, stub_devices, monkeypatch):
    """After a transform raises there is NOTHING in the cache, so the synchronous filter path
    would run the transform inline on the GUI thread -- a freeze, inside a signal handler where
    the exception has nowhere to go. Once errored, every change is the worker's job."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=(("boom_stack", {}),) + STUB_CHAIN[1:])
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.errored, timeout=10000):
        win.load_field(_FIELD, "mem:boom2")

    inline = []
    monkeypatch.setattr(win, "_resolve_now", lambda: inline.append(1))

    with qtbot.waitSignal(win.errored, timeout=10000):
        win.strips.strip(3).controls["frac"].valueChanged.emit(0.5)

    assert inline == [], "a filter change resolved on the GUI thread with no cached transform"
    assert win.strips.strip(0).state_dot.property("state") == "error"


# --------------------------------------------------------------------------- close during compute


def test_close_during_compute_waits_for_the_stage(qtbot, stub_devices):
    """Honest v1: no silent refusal and no pretend-cancel. The title says which stage it is
    waiting out, and the close completes when that stage does."""
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    blocker = _BlockingStack()
    register_device(blocker)
    win = MainWindow(steps=(("blocking_stack", {}),) + STUB_CHAIN[1:])
    qtbot.addWidget(win)
    win.show()
    try:
        win.load_field(_FIELD, "mem:block")
        # wait for the stage name to have crossed the thread, or the title assertion below races
        qtbot.waitUntil(lambda: "cwt" in win.strips.strip(0).reading_label.text(), timeout=5000)
        assert win.is_computing
        win.close()
        assert win.isVisible()                       # the close was deferred, not refused
        assert "cwt" in win.windowTitle()            # it says which stage it is waiting out
    finally:
        blocker.released.set()
    qtbot.waitUntil(lambda: not win.isVisible(), timeout=10000)
    assert win.is_computing is False


# --------------------------------------------------------------------------- remove during compute


def test_removing_the_sole_layer_mid_compute_lands_without_a_crash(qtbot, stub_devices):
    """The zombie-active-layer fix (``_reset_to_empty``) clears
    ``self.layer`` synchronously the moment the sole layer is removed, but ``_on_remove_requested``
    never consults ``is_computing`` -- the worker for that very layer keeps running on its own
    thread. When it lands, ``_on_finished`` sees ``_transform_signature()`` (None-guarded to
    ``(None, ())`` with no layer) mismatch ``self._dispatched`` (the real signature captured at
    dispatch, which can never equal that) and calls back into ``_start_worker`` to "run the current
    one" -- which used to dereference ``self.layer.source_id`` on ``None``. ``_start_worker`` now
    returns immediately when ``self.layer is None``, so the landing is a no-op: the window stays in
    exactly the empty state ``_reset_to_empty`` already established, and no second worker is
    dispatched.

    Relies on pytest-qt's default behaviour of capturing an exception raised inside a Qt slot and
    failing the test with it -- ``is_computing`` alone cannot tell a crash from a clean landing
    here, since ``_on_finished`` tears the thread down (``self._thread = None``) BEFORE the
    redispatch that used to crash, so a pre-fix run still needs pytest-qt's capture to be caught by
    this test at all (confirmed against a worktree with the ``_start_worker`` guard reverted: the
    AttributeError surfaces as this test's own failure, not a silent pass).
    """
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    blocker = _BlockingStack()
    register_device(blocker)
    win = MainWindow(steps=(("blocking_stack", {}),) + STUB_CHAIN[1:])
    qtbot.addWidget(win)

    try:
        win.load_field(_FIELD, "mem:remove-mid-compute")
        qtbot.waitUntil(lambda: win.is_computing, timeout=5000)
        layer_id = win.layer.layer_id

        win._on_remove_requested(layer_id)          # a leaf layer -- no confirm dialog to accept

        assert win.layer is None
        assert win.field is None
        assert win.is_computing is True              # the worker is STILL running in the background
    finally:
        blocker.released.set()

    qtbot.waitUntil(lambda: not win.is_computing, timeout=10000)   # the landing itself

    assert win.layer is None
    assert win.field is None
    assert win.transport.isEnabled() is False
    assert win.strips.isEnabled() is False
    assert win.strips._boxes == []
    assert win.canvas.image_item.image is None


# --------------------------------------------------------------------------- transport shortcut


def test_space_toggles_the_transport(qtbot, loaded):
    shortcuts = [s for s in loaded.findChildren(QtGui.QShortcut)
                 if s.key() == QtGui.QKeySequence(QtCore.Qt.Key_Space)]
    assert len(shortcuts) == 1
    assert shortcuts[0].context() == QtCore.Qt.WindowShortcut
    shortcuts[0].activated.emit()
    assert loaded.transport.clock.playing is True
    shortcuts[0].activated.emit()
    assert loaded.transport.clock.playing is False


# --------------------------------------------------------------------------- view state


def test_soft_bounds_edits_persist_as_json_in_layer_tags(qtbot, loaded):
    """Editable soft bounds are VIEW state: they live in ``layer.tags["ui.spans"]`` as JSON and
    never in params, so they can never reach a cache key."""
    loaded.set_soft_span(3, "frac", 0.0, 0.25)
    spans = json.loads(loaded.layer.tags["ui.spans"])
    assert spans["3.frac"] == [0.0, 0.25]
    assert "frac" not in loaded.layer.chain.steps[3].params or \
        loaded.layer.chain.steps[3].params["frac"] == loaded._params[3]["frac"]

    with qtbot.waitSignal(loaded.resolved, timeout=5000) as sig:
        loaded.strips.strip(3).controls["frac"].valueChanged.emit(0.1)
    assert sig.args[0].cache_misses == 0              # view state did not invalidate anything


# --------------------------------------------------------------------------- overlays on zoom


def test_scale_bar_label_follows_a_zoom(qtbot, loaded):
    """I-2: the distance reading was computed once per RESOLVE, so zooming or panning left it
    claiming a length the view no longer had -- the one overlay whose entire job is to be true
    about on-screen distance. The canvas re-derives both corner overlays on every camera move and
    tells the window, which re-states the label from the width now visible."""
    from dynamix.shell.canvas import nice_round_scalebar

    loaded.canvas.view.setXRange(0, 16, padding=0)
    wide = loaded.canvas.scale_bar_label.text()

    loaded.canvas.view.setXRange(0, 1.6, padding=0)
    close = loaded.canvas.scale_bar_label.text()

    assert wide != close
    (x0, x1), _ = loaded.canvas.view.viewRange()
    assert close.startswith(nice_round_scalebar(x1 - x0)[0])


def test_scale_bar_line_is_drawn_beside_its_label(qtbot, loaded):
    """The bar used to be placed at 5% up from the data-space y MINIMUM, which under invertY is
    the top of the screen, while its label sat at the bottom of the widget: a number at one
    corner describing a line at the other. They are anchored together now."""
    loaded.resize(800, 600)
    loaded.canvas.view.setXRange(0, 16, padding=0)
    bx, by = loaded.canvas.scale_bar_line.getData()
    assert bx is not None and bx.size == 2

    label = loaded.canvas.scale_bar_label
    label_data = loaded.canvas._data_at(label.x(), label.y())
    (x0, x1), (y0, y1) = loaded.canvas.view.viewRange()
    # within a tenth of the visible extent of the label, in both axes, and on screen
    assert abs(bx[0] - label_data.x()) < 0.1 * (x1 - x0)
    assert abs(by[0] - label_data.y()) < 0.1 * (y1 - y0)
    assert x0 <= bx[0] <= x1 and y0 <= by[0] <= y1


# --------------------------------------------------------------------------- discipline


def test_no_lambda_is_connected_to_a_worker_signal():
    """worker.py's documented trap: a lambda has no thread affinity, so Qt runs it ON THE WORKER
    THREAD, where touching a widget is a crash waiting to happen. Every worker signal must be
    connected to a bound method."""
    src = Path("src/dynamix/shell/main_window.py").read_text()
    for signal in ("progress", "finished", "error"):
        hits = re.findall(rf"\.{signal}\.connect\(([^)]*)\)", src)
        assert hits, f"nothing connects the worker's {signal} signal"
        for target in hits:
            assert re.fullmatch(r"self\._[A-Za-z_]+", target.strip()), \
                f"{signal} connected to {target!r}, not a bound method of the window"


def test_render_flag_without_a_path_is_a_usage_error(capsys):
    """``--render`` as the final argument used to die on an IndexError out of argv parsing."""
    from dynamix.shell.app import main

    assert main(["field.npz", "--render"]) == 2
    assert "usage:" in capsys.readouterr().err


def test_shell_has_no_dock_widgets():
    """DESIGN.md: fixed zones. No docks. Ever."""
    for path in Path("src/dynamix/shell").glob("*.py"):
        assert "QDockWidget" not in path.read_text(), path


# --------------------------------------------------------------------------- the real thing


def test_real_wtmm_end_to_end(qtbot, fbm64, clean_registry):
    """fbm64 -> the real demo chain -> extrema rendered; sweeping three scale indices is
    zero-miss, which is the whole point of the transform/filter split."""
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    field = RasterField(name="fbm", values=fbm64, frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    win = MainWindow()
    qtbot.addWidget(win)
    # generous: a cold numba compile of the maxima-line kernels dominates the first run
    with qtbot.waitSignal(win.resolved, timeout=60000) as sig:
        win.load_field(field, "mem:fbm")

    result = sig.args[0].result
    assert len(result["extrema"]) == 1                        # scale_select picked one
    assert len(result["extrema"][0]["x"]) > 0                 # something to look at
    assert win.canvas.extrema_item.getData()[0] is not None
    assert win.transport.clock.n_scales == len(result["scales"]) > 1
    assert "topology" in result                               # chain_topology ran

    for idx in (1, 2, 3):
        with qtbot.waitSignal(win.resolved, timeout=10000) as swept:
            win.transport.slider.setValue(idx)
        assert swept.args[0].cache_misses == 0, f"scale {idx} recomputed a transform"
        assert swept.args[0].result["_scale_idx"] == idx


def test_real_wtmm_transport_sweep_step_is_within_the_frame_budget(qtbot, fbm64, clean_registry):
    """The spec's timed guard, over the REAL chain: one transport step must fit in a frame.

    The budget is 16 ms -- one frame at 60 Hz, which is what "scrubbing sweeps scales with no
    dropped frames" means and what the transform/filter split exists to buy. The assertion is
    made at 30 ms, and deliberately: this is a wall clock on a shared CI-or-laptop machine, where
    a GC pause or a scheduler hiccup on one step is not a design regression. The MEDIAN of the
    steps is what is measured, so a single outlier cannot fail the run while a genuine
    regression -- a filter that got expensive, or a scrub that started recomputing a transform --
    moves every step and blows through even this margin.
    """
    import statistics
    import time

    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    field = RasterField(name="fbm", values=fbm64, frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    win = MainWindow()
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(field, "mem:fbm_timed")

    n = win.transport.clock.n_scales
    assert n >= 6, "need enough scales to time a sweep"

    timings = []
    for idx in range(1, min(n, 9)):
        win._params[win._index_of("scale_select")]["scale_idx"] = idx
        win.layer.chain = win._chain()
        t0 = time.perf_counter()
        win._reresolve()
        timings.append((time.perf_counter() - t0) * 1000.0)

    assert len(timings) >= 5
    median_ms = statistics.median(timings)
    assert median_ms < 30.0, f"median sweep step {median_ms:.1f} ms (16 ms budget): {timings}"


# ------------------------------------------------------ Skeleton dialog
#
# A THIRD stub, separate from ``_StubStack``/``STUB_CHAIN`` above -- those two are shared by
# every other test in this file and their result carries no top-level ``"chains"`` key at all
# (only ``"extrema"``), which is exactly what makes them useful for asserting the button's
# DISABLED state below. A dedicated stub that stamps ``"chains"`` (the one key
# ``_refresh_skeleton_button``/``SkeletonDialog`` ever read -- the same key the real ``wtmm2d``
# transform's own result already carries) keeps this section's fixtures self-contained rather
# than reshaping a helper dozens of unrelated tests already depend on.


def _stub_chain(n=6, slope=-1.0, seed=0):
    rng = np.random.default_rng(seed)
    log2_scales = np.arange(n, dtype=np.float64)
    log2_mod = slope * log2_scales + rng.normal(scale=0.02, size=n)
    return {"x": np.zeros(n, dtype=np.int64), "y": np.zeros(n, dtype=np.int64),
            "mod": 2.0 ** log2_mod, "log2_scales": log2_scales, "log2_mod": log2_mod}


class _StubChainsStack(_StubStack):
    """Same shape as ``_StubStack`` (the real downstream filters run over it unchanged) plus a
    top-level ``"chains"`` list."""

    name = "stub_chains_stack"

    def compute(self, field, params, *, progress=None):
        out = dict(super().compute(field, params, progress=progress))
        out["chains"] = [_stub_chain(seed=i) for i in range(5)]
        return out


STUB_CHAIN_WITH_CHAINS = (
    ("stub_chains_stack", {}),
    ("scale_select", {"scale_idx": 0}),
    ("orientation_wedge", {"centre": 0.0, "half_width": 90.0}),
    ("modulus_threshold", {"frac": 0.0}),
)


@pytest.fixture
def stub_chain_devices(clean_registry):
    from dynamix.model.device import register_device

    register_builtin_devices()
    register_device(_StubChainsStack())
    return clean_registry


@pytest.fixture
def chains_window(qtbot, stub_chain_devices):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN_WITH_CHAINS)
    qtbot.addWidget(win)
    return win


@pytest.fixture
def chains_loaded(qtbot, chains_window):
    """A window whose active result carries real (stub) ``"chains"`` -- the skeleton button's
    own enabled state."""
    with qtbot.waitSignal(chains_window.resolved, timeout=10000):
        chains_window.load_field(_FIELD, "mem:stub_chains")
    return chains_window


# -- button enable/disable ------------------------------------------------------------------


def test_skeleton_button_starts_disabled_before_any_load(window):
    assert window._skeleton_button.isEnabled() is False


def test_skeleton_button_stays_disabled_after_a_resolve_with_no_chains(loaded):
    """``loaded`` (the file's own shared fixture) carries ``"extrema"`` but no ``"chains"`` --
    the button must not enable off the wrong key."""
    assert loaded._active_result is not None
    assert "chains" not in loaded._active_result
    assert loaded._skeleton_button.isEnabled() is False


def test_skeleton_button_enables_after_a_resolve_with_chains(chains_loaded):
    assert "chains" in chains_loaded._active_result
    assert chains_loaded._skeleton_button.isEnabled() is True


def test_skeleton_button_disables_again_once_the_last_layer_is_removed(chains_loaded):
    win = chains_loaded
    layer_id = win.layer.layer_id

    win._on_remove_requested(layer_id)

    assert win._skeleton_button.isEnabled() is False


# -- px_size derivation (main_window._skeleton_px_size) -----------------------------------


def _bare_field():
    return RasterField(name="f", values=np.zeros((4, 4)), frame=LocalFrame(units="px"),
                        x_axis=np.arange(4.0), y_axis=np.arange(4.0))


def test_skeleton_px_size_is_none_for_a_bare_px_local_frame(window):
    window.field = _bare_field()
    assert window._skeleton_px_size() is None


def test_skeleton_px_size_is_the_local_frame_dx_for_physical_units(window):
    window.field = RasterField(name="f", values=np.zeros((4, 4)),
                                frame=LocalFrame(dx=2.5, dy=2.5, units="µm"),
                                x_axis=np.arange(4.0) * 2.5, y_axis=np.arange(4.0) * 2.5)
    assert window._skeleton_px_size() == pytest.approx(2.5)


def test_skeleton_px_size_is_none_for_a_geographic_frame(window):
    window.field = RasterField(name="f", values=np.zeros((4, 4)), frame=GeographicFrame(),
                                x_axis=np.linspace(0.0, 1.0, 4), y_axis=np.linspace(0.0, 1.0, 4))
    assert window._skeleton_px_size() is None


# -- opening the dialog ---------------------------------------------------------------------


def test_skeleton_button_click_opens_a_dialog_over_the_active_layers_chains(chains_loaded):
    win = chains_loaded

    win._skeleton_button.click()

    from dynamix.shell.skeleton_dialog import SkeletonDialog

    assert isinstance(win._skeleton_dialog, SkeletonDialog)
    assert win._skeleton_dialog._chains == win._active_result["chains"]
    assert win._skeleton_dialog._px_log2 == 0.0     # _FIELD's own frame is bare px -- see above


def test_reclicking_the_button_closes_the_previous_dialog_and_builds_a_fresh_one(chains_loaded):
    win = chains_loaded
    win._skeleton_button.click()
    first = win._skeleton_dialog
    first.show()

    win._skeleton_button.click()

    assert win._skeleton_dialog is not first
    assert first.isVisible() is False


# -- bidirectional selection sync ------------------------------------------------------------


def test_skeleton_selection_requested_replaces_the_group_palette_selection(chains_loaded):
    win = chains_loaded
    win._skeleton_button.click()
    dialog = win._skeleton_dialog
    lid = win.layer.layer_id
    win._group_palette.apply_picks([(lid, 0)], "add")     # a pre-existing pick

    dialog.selectionRequested.emit([2, 4])

    assert win._group_palette.selection() == {(lid, 2), (lid, 4)}     # replaced, not unioned


def test_group_membership_change_pushes_into_the_open_dialog(chains_loaded):
    win = chains_loaded
    win._skeleton_button.click()
    dialog = win._skeleton_dialog
    lid = win.layer.layer_id

    win._group_palette.apply_picks([(lid, 1), (lid, 3)], "add")

    assert dialog._selected == {1, 3}


def test_a_freshly_opened_dialog_is_seeded_with_the_current_selection(chains_loaded):
    win = chains_loaded
    lid = win.layer.layer_id
    win._group_palette.apply_picks([(lid, 0)], "add")

    win._skeleton_button.click()

    assert win._skeleton_dialog._selected == {0}


def test_full_round_trip_select_h_range_updates_the_group_palette(chains_loaded):
    """End to end: a real click on the dialog's own "Select h-range" button reaches
    ``GroupPalette.selection()`` through ``_on_skeleton_selection_requested``. The expected set is
    derived from the dialog's OWN spin values/cached ``_h`` (not assumed to be "every chain") --
    the stub chains' slopes sit within noise of each other, and the spins' 3-decimal rounding of
    the finite-min/max default (EQ§4's own shape) can legitimately exclude a boundary chain from
    a such a tightly clustered set, exactly as it would for EQSelect's own identical-precision
    filter box."""
    win = chains_loaded
    win._skeleton_button.click()
    dialog = win._skeleton_dialog

    dialog._select_button.click()

    lid = win.layer.layer_id
    lo, hi = dialog._h_lo_spin.value(), dialog._h_hi_spin.value()
    expected = {(lid, i) for i, h in enumerate(dialog._h) if np.isfinite(h) and lo <= h <= hi}
    assert expected                                    # the round trip selected something real
    assert win._group_palette.selection() == expected


# -- transect wiring ---------------------------------


def test_canvas_transect_drawn_adds_a_panel_record(loaded):
    win = loaded

    win.canvas.transectDrawn.emit((1.0, 2.0), (5.0, 2.0))

    (record,) = win._transect_panel.records()
    assert record.a == (1.0, 2.0) and record.b == (5.0, 2.0)


def test_selecting_a_transect_swath_selects_chains_via_replace(chains_loaded):
    """Every stub chain sits at ``(0, 0)`` (``_stub_chain``'s own ``x``/``y``) -- a segment
    through the origin, at the default 20px buffer, is within range of all five. ``op="replace"``
    is what this actually tests: an unrelated pre-existing pick from a layer id no chain here
    belongs to must be GONE afterward -- "add" would have left it in place."""
    win = chains_loaded
    lid = win.layer.layer_id
    win._group_palette.apply_picks([(9999, 0)], "add")

    win._transect_panel.add_record((-1.0, -1.0), (1.0, 1.0))

    assert win._group_palette.selection() == {(lid, i) for i in range(5)}


def test_transect_selection_highlights_the_line_on_the_canvas(chains_loaded):
    win = chains_loaded

    record = win._transect_panel.add_record((0.0, 0.0), (10.0, 0.0))

    hx, hy = win.canvas.transect_highlight_item.getData()
    assert list(hx) == [record.a[0], record.b[0]]
    assert list(hy) == [record.a[1], record.b[1]]


def test_deleting_a_transect_removes_its_line_from_the_canvas(chains_loaded):
    win = chains_loaded
    win._transect_panel.add_record((0.0, 0.0), (10.0, 0.0))

    win._transect_panel.delete_selected()

    xs, _ = win.canvas.transect_item.getData()
    assert xs is None or xs.size == 0


def test_transect_records_changed_keeps_project_transects_current(chains_loaded):
    win = chains_loaded

    win._transect_panel.add_record((0.0, 0.0), (10.0, 0.0))

    assert win.project.transects == win._transect_panel.records()


def test_plot_requested_opens_a_profile_dialog_sampling_the_active_field(loaded, monkeypatch):
    """Real end-to-end signal wiring: ``TransectPanel.plotRequested`` -> ``ProfileDialog``
    construction, sampling ``self.field`` ("the panel asks MainWindow for the active
    field at plot time")."""
    win = loaded
    from dynamix.shell import main_window as mw

    captured = {}

    class _FakeDialog:
        def __init__(self, dist, z, *, label="", parent=None):
            captured["dist"], captured["z"], captured["label"] = dist, z, label

        def show(self):
            captured["shown"] = True

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    monkeypatch.setattr(mw, "ProfileDialog", _FakeDialog)
    record = win._transect_panel.add_record((1.0, 1.0), (5.0, 1.0))

    win._transect_panel.plotRequested.emit(record)

    from dynamix.core.transect import sample_profile

    values = np.asarray(getattr(win.field, "values", win.field))
    expected_dist, expected_z = sample_profile(values, record.a, record.b)
    np.testing.assert_allclose(captured["dist"], expected_dist)
    np.testing.assert_allclose(captured["z"], expected_z, equal_nan=True)
    assert captured["label"] == f"T{record.transect_id}"
    assert captured["shown"] is True


def test_plot_requested_is_a_noop_with_no_field_loaded(qtbot, window, monkeypatch):
    from dynamix.shell import main_window as mw
    from dynamix.model.project import TransectRecord

    calls = []
    monkeypatch.setattr(mw, "ProfileDialog", lambda *a, **k: calls.append((a, k)))
    record = TransectRecord(transect_id=1, a=(0.0, 0.0), b=(1.0, 1.0))

    window._on_transect_plot_requested(record)          # window.field is None -- must not raise

    assert calls == []


# ----------------------------------------- The floating inspector, wired in
# the design's live-update contract: ``_apply`` stays "the ONE place a result reaches the screen"
# and keeps emitting ``resolved`` untouched; ONE connection fans that Renderable out to whichever
# inspector is open over ITS source. The plan's resolved ambiguity #3 is why the fan-out lives
# here and not in the window: an inspector holds no ``Project`` (sec 7 rule 3) and so cannot map a
# ``layer_id`` to a ``source_id`` itself. **No new compute path, no second resolve, no polling.**


def _open_inspector(window, source_id):
    """Open one the way the user does: the layer list's own per-source "I" button."""
    window.layer_list._source_rows[source_id].inspector_button.click()
    return window._inspectors[source_id]


def test_toggling_a_source_opens_an_inspector_showing_that_source_raster(loaded):
    from dynamix.shell.inspector import InspectorWindow

    source_id = loaded.layer.source_id
    inspector = _open_inspector(loaded, source_id)

    assert isinstance(inspector, InspectorWindow)
    assert inspector.isVisible() is True
    # the SOURCE's own field, resolved through ``_field_for_source``
    assert inspector.canvas._field is loaded._fields[loaded.layer.layer_id]
    # ... and the center canvas STAYS (the plan's global constraint): nothing was moved out of it
    assert loaded.canvas._field is not None


def test_closing_the_inspector_unchecks_the_layer_list_toggle(loaded):
    """Closing the window and un-toggling the button are ONE state."""
    source_id = loaded.layer.source_id
    inspector = _open_inspector(loaded, source_id)
    button = loaded.layer_list._source_rows[source_id].inspector_button
    assert button.isChecked() is True

    inspector.close()

    assert button.isChecked() is False
    assert source_id not in loaded._inspectors


def test_a_resolve_for_the_active_layer_reaches_its_open_inspector(qtbot, loaded):
    from dynamix.shell.main_window import INSPECTOR_LIVE_TEXT

    inspector = _open_inspector(loaded, loaded.layer.source_id)

    with qtbot.waitSignal(loaded.resolved, timeout=5000) as sig:
        loaded.strips.strip(3).controls["frac"].valueChanged.emit(0.5)   # a filter edit

    assert loaded._last_renderables[loaded.layer.source_id] is sig.args[0]
    assert inspector.showing_live is True
    assert len(inspector.canvas.extrema_item.getData()[0]) > 0      # it actually redrew
    assert inspector.status_label.text() == INSPECTOR_LIVE_TEXT
    assert loaded.is_computing is False                            # no worker, no second resolve


def test_a_resolve_for_another_source_does_not_touch_this_inspector(qtbot, loaded):
    first_source = loaded.layer.source_id
    inspector = _open_inspector(loaded, first_source)
    with qtbot.waitSignal(loaded.resolved, timeout=10000) as first:
        loaded.strips.strip(3).controls["frac"].valueChanged.emit(0.5)
    drawn = inspector.canvas.extrema_item.getData()[0].copy()

    with qtbot.waitSignal(loaded.resolved, timeout=10000) as second:
        loaded.load_field(_FIELD * 2.0, "mem:stub2")

    assert second.args[0].layer_id != first.args[0].layer_id
    assert loaded._last_renderables[first_source] is first.args[0]   # still the FIRST result
    assert loaded._last_renderables[loaded.layer.source_id] is second.args[0]
    np.testing.assert_array_equal(inspector.canvas.extrema_item.getData()[0], drawn)


def test_a_non_active_inspector_says_it_is_showing_its_last_result(qtbot, loaded):
    """Sec 10 A2 / doctrine #3: honest, not blank -- it keeps its picture AND says what it is."""
    from dynamix.shell.main_window import INSPECTOR_LIVE_TEXT

    first_layer = loaded.layer
    inspector = _open_inspector(loaded, first_layer.source_id)
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.strips.strip(3).controls["frac"].valueChanged.emit(0.5)
    assert inspector.status_label.text() == INSPECTOR_LIVE_TEXT

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.load_field(_FIELD * 2.0, "mem:stub2")

    text = inspector.status_label.text()
    assert text.startswith("last result")
    assert f"'{first_layer.name}'" in text
    assert "not active" in text
    assert "select it to re-run" in text
    assert len(inspector.canvas.extrema_item.getData()[0]) > 0   # the picture stayed put
    assert inspector.showing_live is False


def test_opening_an_inspector_on_a_non_active_source_says_so_on_the_status_bar(qtbot, loaded):
    """The zone-local warning surface AND the status bar, once, at the moment of opening."""
    first_source = loaded.layer.source_id
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.load_field(_FIELD * 2.0, "mem:stub2")

    inspector = _open_inspector(loaded, first_source)

    assert loaded._status_message == inspector.status_label.text()
    assert "not active" in loaded._status_message


def test_an_inspector_with_no_result_yet_says_so(qtbot, window, tmp_path, monkeypatch):
    from dynamix.shell.inspector import NO_RESULT_TEXT

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))   # auto_run off
    window.load_field(_FIELD, "mem:inert")

    inspector = _open_inspector(window, window.layer.source_id)

    assert inspector.status_label.text() == NO_RESULT_TEXT
    assert inspector.canvas._field is not None          # the raster is there; only a RESULT isn't
    assert window.is_computing is False


def test_opening_an_inspector_dispatches_no_worker(qtbot, loaded):
    """"No new compute path", asserted rather than assumed."""
    keys_before = set(loaded.cache._store)
    hits_before, misses_before = loaded.cache.hits, loaded.cache.misses

    _open_inspector(loaded, loaded.layer.source_id)

    assert loaded.is_computing is False
    assert loaded._thread is None
    assert set(loaded.cache._store) == keys_before
    assert (loaded.cache.hits, loaded.cache.misses) == (hits_before, misses_before)


def test_two_open_inspectors_share_the_one_resolved_connection(qtbot, loaded):
    """Ambiguity #3 resolved in favour of ONE connection made in ``__init__``: N inspectors must
    not mean N subscriptions to ``resolved``."""
    signature = QtCore.SIGNAL("resolved(PyObject)")
    first_source = loaded.layer.source_id
    assert loaded.receivers(signature) == 1              # the fan-out, connected once

    _open_inspector(loaded, first_source)
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.load_field(_FIELD * 2.0, "mem:stub2")
    _open_inspector(loaded, loaded.layer.source_id)

    assert len(loaded._inspectors) == 2
    assert loaded.receivers(signature) == 1


# --- A POINT-catalogue source has no raster to inspect ---------------------
# There is an "I" on EVERY source header and ``provenance_tree`` yields a node for every source
# in ``project.sources``, so this gesture is fully reachable: import a CSV point catalogue, click
# its header's "I". A point source's field is a ``PointSet``, which ``Canvas.set_field`` cannot
# take (``np.asarray(field, dtype=float64)`` -> TypeError) -- the same trap ``_select_layer``
# already guards at its own ``if not self._is_point_layer(layer)``.

_POINT_CSV = "lon,lat,depth,mag\n1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n"


def _load_point_catalogue(window, tmp_path):
    """Import a CSV point catalogue the way ``point_import`` does and return its ``source_id``."""
    from dynamix.shell.point_import import load_points

    path = tmp_path / "quakes.csv"
    path.write_text(_POINT_CSV)
    load_points(window, str(path))
    return window.layer.source_id


def test_opening_an_inspector_on_a_point_catalogue_raises_nothing(loaded, tmp_path):
    """The handler is called DIRECTLY, not through the button: PySide6 prints an exception that
    escapes a slot and carries on, so routed through the signal this regression would pass while
    the window was never shown."""
    point_source = _load_point_catalogue(loaded, tmp_path)

    loaded._on_inspector_toggled(point_source, True)   # TypeError before the fix

    assert point_source not in loaded._inspectors


def test_a_refused_point_inspector_leaves_its_button_unchecked_and_names_the_fix(loaded,
                                                                                 tmp_path):
    """Sec 8 rule 6 (refusals name the fix) meeting sec 5b's one-state rule: no window means the
    "I" must not read "open", and the user must be told why in words that say what to do."""
    point_source = _load_point_catalogue(loaded, tmp_path)
    button = loaded.layer_list._source_rows[point_source].inspector_button

    button.click()

    assert point_source not in loaded._inspectors
    assert button.isChecked() is False
    assert "raster" in loaded._status_message


def test_a_raster_source_still_opens_while_a_point_catalogue_is_present(loaded, tmp_path):
    """The refusal is per SOURCE KIND, not a blanket off switch: the raster source imported first
    still opens its inspector after a point catalogue joins the project."""
    raster_source = loaded.layer.source_id
    _load_point_catalogue(loaded, tmp_path)

    inspector = _open_inspector(loaded, raster_source)

    assert inspector.isVisible() is True


def test_selecting_a_layer_re_words_an_open_inspector_before_any_result_lands(qtbot, loaded):
    """``_fan_out_to_inspectors`` re-words on RESOLVED, but what makes a picture "last" (or makes
    it live again) is the ACTIVE LAYER changing -- which happens the moment ``_select_layer``
    returns, before any worker has finished, and on the ``_start_worker_for`` bail-out path
    (``field is None``) never gets a resolve at all. Read synchronously, without pumping the
    event loop, so only the layer-switch re-word can satisfy it."""
    from dynamix.shell.main_window import INSPECTOR_LIVE_TEXT

    first_source, first_layer_id = loaded.layer.source_id, loaded.layer.layer_id
    inspector = _open_inspector(loaded, first_source)
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.load_field(_FIELD * 2.0, "mem:stub2")     # a second source is now the active one
    assert "not active" in inspector.status_label.text()

    loaded.layer_list.select_layer(first_layer_id)       # back to the inspected source's layer
    status_now = inspector.status_label.text()
    qtbot.waitSignal(loaded.resolved, timeout=10000).wait()   # drain the dispatched worker

    assert status_now == INSPECTOR_LIVE_TEXT


# ----------------------------------------- The master track
# The main window's transport IS the master. Its ``scaleChanged`` keeps its FIRST connection
# (``_on_scale_changed`` -- the active layer's own filter path, untouched) and gains a SECOND one
# that drives every FOLLOWING inspector. The master drives the scale INDEX only: Hölder exponents
# compare across datasets, raw modulus coefficients do not without the sec 11 calibration stage.
# The containment guarantee runs the other way -- an inspector's own slider can never move the
# ACTIVE layer's chain.


def test_master_transport_moves_a_following_inspector_slider(qtbot, loaded):
    inspector = _open_inspector(loaded, loaded.layer.source_id)
    assert inspector.follow_master is True
    assert inspector.transport.slider.value() == 0

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.transport.slider.setValue(2)

    assert inspector.transport.slider.value() == 2
    # ... AND the reading text moved with it, in the same gesture (the acceptance criterion)
    assert inspector.transport.reading_label.text() == loaded.transport.reading_label.text()
    assert inspector.transport.reading_label.text().startswith("a = ")


def test_master_transport_does_not_move_an_unfollowing_inspector(qtbot, loaded):
    """Un-checking "M" freezes that inspector's scale CONTROL while the master keeps moving.

    Reworded at the review of ``1b405f2``: what this asserts is the index and the reading, not
    two extrema sets. Sec 5d's comparison-of-two-extrema-sets is NOT what un-checking "M" buys
    today -- the picture is the chain's post-``scale_select`` result, one extrema layer, so a
    window held at another index has no second scale in it to draw and says so instead
    (``test_a_follower_whose_picture_cannot_follow_says_so``). The divergent PICTURE needs a
    per-source result at a per-source scale, which is a sec 5b "no new compute path" call for
    the user."""
    inspector = _open_inspector(loaded, loaded.layer.source_id)
    inspector.follow_button.setChecked(False)
    assert inspector.follow_master is False

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.transport.slider.setValue(2)

    assert loaded.transport.slider.value() == 2
    assert inspector.transport.slider.value() == 0
    assert inspector.transport.reading_label.text() != loaded.transport.reading_label.text()


def test_an_inspector_slider_drag_does_not_change_the_active_layer_chain(qtbot, loaded):
    """The containment guarantee: an inspector's own transport drives that inspector and nothing
    else -- never ``_params``/``layer.chain``, so it can never move the active layer's chain
    behind the user's back, and never the master either."""
    source_id = loaded.layer.source_id
    inspector = _open_inspector(loaded, source_id)
    assert inspector.transport.slider.maximum() == 2      # data-derived by the opening push
    chain_before = loaded.layer.chain
    params_before = [dict(p) for p in loaded._params]
    seen: list[tuple[str, int]] = []
    inspector.scaleChanged.connect(lambda sid, i: seen.append((sid, i)))

    inspector.transport.slider.setValue(2)

    assert seen == [(source_id, 2)]                       # it announced itself ...
    assert loaded.layer.chain is chain_before             # ... and changed nothing else
    assert [dict(p) for p in loaded._params] == params_before
    assert loaded.transport.slider.value() == 0
    assert loaded.is_computing is False


def test_two_inspectors_one_following_one_not_diverge_visibly(qtbot, loaded):
    """What visibly diverges is the scale INDEX and its reading, in two windows at once.

    Reworded and extended at the review of ``1b405f2``: neither picture moves to the other's
    scale, and each window now SAYS which scale it is showing rather than letting a moved
    reading imply a moved picture. ``following`` is over a non-active source, so it adopts the
    master's index over a kept picture; ``frozen`` is over the ACTIVE source, so its picture DID
    move (the chain re-ran at the master's index) while its own control stayed put -- the two
    faces of the same confession."""
    from dynamix.shell.inspector import SCALE_NOT_SHOWN_LIVE_TEXT, SCALE_NOT_SHOWN_TEXT

    first_source = loaded.layer.source_id
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.load_field(_FIELD * 2.0, "mem:stub2")
    following = _open_inspector(loaded, first_source)
    frozen = _open_inspector(loaded, loaded.layer.source_id)
    frozen.follow_button.setChecked(False)

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.transport.slider.setValue(2)

    assert following.transport.slider.value() == 2
    assert frozen.transport.slider.value() == 0
    assert following.transport.reading_label.text() == loaded.transport.reading_label.text()
    assert following.transport.reading_label.text() != frozen.transport.reading_label.text()
    # ... and each window states which scale its own picture is actually on
    assert SCALE_NOT_SHOWN_TEXT.format(control=2, picture=0) in following.status_label.text()
    assert SCALE_NOT_SHOWN_LIVE_TEXT.format(control=0, picture=2) in frozen.status_label.text()


# ------------------------- Drive followers from what SETTLES
# ``Transport.scaleChanged`` is emitted by ``_set_index`` alone -- a slider drag or a playback
# tick. Every OTHER way the master's scale moves settles it through the SILENT
# ``Transport.sync_to``, which is the exact divergence ``sync_to``'s own docstring was written
# for ("the same scale is addressable from two places"), reintroduced one level out: a following
# inspector was left behind with its "M" still checked. Both repros below are the review's.


def test_the_scale_select_knob_also_moves_a_following_inspector(qtbot, loaded):
    """Repro 1 -- the STRIP KNOB, not the transport. ``_on_param_changed``'s ``scale_select``
    branch settles the master with ``sync_to`` (silent), so before the fix nothing at all
    reached the follower and it sat at index 0 with "M" checked."""
    inspector = _open_inspector(loaded, loaded.layer.source_id)
    assert inspector.follow_master is True

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.strips.strip(1).controls["scale_idx"].valueChanged.emit(2)

    assert loaded.transport.slider.value() == 2
    assert inspector.transport.slider.value() == 2
    assert inspector.transport.reading_label.text() == loaded.transport.reading_label.text()


def test_a_layer_switch_leaves_master_and_follower_on_the_same_index(qtbot, loaded):
    """Repro 2 -- ``_sync_transport``. Its ``set_n_scales`` clamp echo used to reach the
    followers carrying the transport's STALE index, and the REAL index was then settled by a
    silent ``sync_to`` that never reached them at all: selecting a layer whose chain is restored
    at scale 2 left the master reading 2 and the follower reading 0."""
    from dynamix.model.chain import Chain, DeviceRef

    source_id = loaded.layer.source_id
    inspector = _open_inspector(loaded, source_id)
    at_two = Chain(tuple(
        DeviceRef(n, {**p, "scale_idx": 2} if n == "scale_select" else dict(p))
        for n, p in STUB_CHAIN)).materialized()
    second = loaded.project.add_layer("at scale 2", source_id, at_two)
    loaded.add_layer_row(second, loaded.field)

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(second.layer_id)

    assert loaded._params[1]["scale_idx"] == 2                  # the chain's own scale
    assert loaded.transport.slider.value() == 2
    assert inspector.transport.slider.value() == 2
    assert inspector.transport.reading_label.text() == loaded.transport.reading_label.text()


# --------------- The inspector's own control is ROUTED, or SAID
# The plan's Task-10 interface: an inspector's own slider "emits ``scaleChanged(source_id,
# idx)``, which ``MainWindow`` routes into that inspector's display only -- never into
# ``_params``/``layer.chain``". The route lands here. Where the HELD result cannot show the
# requested scale (it is post-``scale_select``: one extrema layer), the window says so instead
# of letting its reading contradict its picture -- sec 8 rule 6, naming the state and the fix.


def test_an_inspector_slider_drag_is_routed_into_that_inspectors_display(qtbot, loaded):
    from dynamix.shell.inspector import SCALE_NOT_SHOWN_LIVE_TEXT

    source_id = loaded.layer.source_id
    inspector = _open_inspector(loaded, source_id)
    chain_before = loaded.layer.chain
    params_before = [dict(p) for p in loaded._params]

    inspector.transport.slider.setValue(2)

    assert loaded.layer.chain is chain_before                    # containment, still
    assert [dict(p) for p in loaded._params] == params_before
    assert loaded.transport.slider.value() == 0                  # and never the master either
    assert inspector.picture_scale_idx == 0
    # this source's layer IS the active one, so the sentence names the LIMITATION, not a
    # gesture the user has already made
    assert SCALE_NOT_SHOWN_LIVE_TEXT.format(control=2, picture=0) in inspector.status_label.text()


def test_a_follower_whose_picture_cannot_follow_says_so(qtbot, loaded):
    """The reading must not lie about the picture. A follower over a source that is NOT the
    active layer's adopts the master's INDEX (it is following), but its held result is one
    extrema layer at its own scale -- so the picture stays put and the window says which scale
    it is actually showing."""
    from dynamix.shell.inspector import SCALE_NOT_SHOWN_TEXT

    first_source = loaded.layer.source_id
    inspector = _open_inspector(loaded, first_source)
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.load_field(_FIELD * 2.0, "mem:stub2")
    drawn = inspector.canvas.extrema_item.getData()[0].copy()

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.transport.slider.setValue(2)

    assert inspector.transport.slider.value() == 2                # it followed the INDEX ...
    np.testing.assert_array_equal(                                # ... the picture could not
        inspector.canvas.extrema_item.getData()[0], drawn)
    text = inspector.status_label.text()
    assert text.startswith("last result")                         # MainWindow's sentence, kept
    assert SCALE_NOT_SHOWN_TEXT.format(control=2, picture=0) in text


# ------------------------------------- Persistence, and the one-shot restore
# Sec 4a: an inspector's window state is a PREFERENCE, not a measurement, so it lives in
# ``Settings.view_options["inspectors"]`` (``model/inspector_state.py`` owns the shape) and never
# in the project document -- and ``settings.py``'s own schema is NOT touched to hold it, because
# ``view_options`` is already a free-form JSON dict. Restored at the SAME one-shot point
# ``Settings.center_view`` uses (``load_field``'s tail): there is no source, field or layer to
# open an inspector OVER any earlier than the first layer landing. Two flags, not one -- a user
# who switched center views on purpose is a different situation from one who closed an inspector.


def _saved_inspectors():
    """What is on disk right now, read back through the ONE seam to JSON."""
    from dynamix.model.inspector_state import inspectors_from_payload
    from dynamix.shell.settings import load_settings

    return inspectors_from_payload(load_settings().view_options.get("inspectors"))


def test_opening_an_inspector_writes_it_into_view_options(loaded):
    from dynamix.shell.settings import load_settings, update_settings

    # a neighbour key, so the read-modify-write is asserted rather than assumed
    update_settings(view_options={"background": "dark"})
    source_id = loaded.layer.source_id

    _open_inspector(loaded, source_id)

    state = _saved_inspectors()[source_id]
    assert state.open is True
    assert state.geometry is not None                 # where it is, not merely that it is
    assert state.follow_master is True
    assert load_settings().view_options["background"] == "dark"
    json.dumps(load_settings().view_options)          # ... and the blob is what settings saved


def test_closing_an_inspector_writes_open_false(loaded):
    source_id = loaded.layer.source_id
    inspector = _open_inspector(loaded, source_id)
    assert _saved_inspectors()[source_id].open is True

    inspector.close()

    assert _saved_inspectors()[source_id].open is False


def test_a_second_window_restores_the_open_inspector_at_its_saved_scale(qtbot, stub_devices):
    """The acceptance criterion: quit with an inspector open at a scale of its own, reopen the
    app on the same raster, and it comes back there. Two ``MainWindow``s in sequence over the
    isolated settings file the autouse fixture provides."""
    from dynamix.shell.main_window import MainWindow

    first = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(first)
    with qtbot.waitSignal(first.resolved, timeout=10000):
        first.load_field(_FIELD, "mem:persist")
    source_id = first.layer.source_id
    inspector = _open_inspector(first, source_id)
    inspector.follow_button.setChecked(False)          # hold a scale of its own
    inspector.transport.slider.setValue(2)
    assert _saved_inspectors()[source_id].scale_idx == 2

    second = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(second)
    with qtbot.waitSignal(second.resolved, timeout=10000):
        second.load_field(_FIELD, "mem:persist")

    restored = second._inspectors[source_id]
    assert restored.isVisible() is True
    # the button and the window are ONE state on the way back up too
    assert second.layer_list._source_rows[source_id].inspector_button.isChecked() is True
    assert restored.follow_master is False
    assert restored.transport.slider.value() == 2
    assert restored.transport.reading_label.text() == second._scale_reading(2)


def test_restore_is_one_shot_per_window(qtbot, stub_devices):
    """The FLAG, not the file. Settings are shared by every window on this machine, so a second
    window writing ``open: True`` back while this one is alive must not make this one's next
    ``load_field`` re-open a window its user deliberately closed."""
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import load_settings, update_settings

    update_settings(view_options={"inspectors": {"src0": {"open": True}}})
    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:once-a")
    win._inspectors["src0"].close()
    assert "src0" not in win._inspectors
    update_settings(view_options={**load_settings().view_options,
                                  "inspectors": {"src0": {"open": True}}})

    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD.copy(), "mem:once-b")

    assert "src0" not in win._inspectors
    assert win._inspectors_restored is True


def test_an_entry_for_an_unknown_source_is_dropped_silently(qtbot, stub_devices):
    """Sec 4a's honest drop: an entry naming a source this project does not have opens nothing
    and says nothing -- a preference cannot be a fault.

    TWO entries, not one (review of ``33b870b``): with only the stranger written, ``_inspectors
    == {}`` is trivially true when NOTHING restores at all, and the test passed against sources
    with the whole feature removed. The real source beside it is what makes the assertion
    discriminate a DROP from an absence.
    """
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import update_settings

    update_settings(view_options={"inspectors": {"src0": {"open": True},
                                                 "src9": {"open": True}}})
    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:unknown-source")

    assert set(win._inspectors) == {"src0"}       # the real one opened; the stranger did not
    assert "src9" not in win._status_message


def test_sub_layer_opacity_survives_a_window_restart(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    first = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(first)
    with qtbot.waitSignal(first.resolved, timeout=10000):
        first.load_field(_FIELD, "mem:dimmed")
    source_id = first.layer.source_id
    inspector = _open_inspector(first, source_id)
    key = inspector.row_keys()[1]                      # row 0 is the source raster
    inspector.opacity_spins[key].setValue(0.4)
    inspector.rows[key].setCheckState(0, QtCore.Qt.CheckState.Unchecked)

    second = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(second)
    with qtbot.waitSignal(second.resolved, timeout=10000):
        second.load_field(_FIELD, "mem:dimmed")

    restored = second._inspectors[source_id]
    assert restored.opacity_spins[key].value() == pytest.approx(0.4)
    assert restored.rows[key].checkState(0) == QtCore.Qt.CheckState.Unchecked
    # every OTHER row came back at the defaults it was never moved off
    other = restored.row_keys()[0]
    assert restored.opacity_spins[other].value() == pytest.approx(1.0)
    assert restored.rows[other].checkState(0) == QtCore.Qt.CheckState.Checked


def test_restoring_inspectors_does_not_disturb_center_view_restore(qtbot, stub_devices):
    """Both one-shots run at the same tail, and neither eats the other."""
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import load_settings, update_settings
    # imported inside the test: ``tests.test_center_views`` imports THIS module at import time,
    # so a module-level import here would be circular. The recorder stands in for
    # ``ArrangementView`` exactly as that file's own center-view restore tests use it.
    from tests.test_center_views import _FakeArrangement

    update_settings(center_view="vector",
                    view_options={**load_settings().view_options,
                                  "inspectors": {"src0": {"open": True}}})
    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    win._arrangement = _FakeArrangement()

    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:both-restores")

    assert win._center_view == "vector"
    assert win._center_view_restored is True
    assert "src0" in win._inspectors
    assert win._inspectors_restored is True


# ---------------------------------------- The QUIT path itself
# The acceptance sentence starts "quit with an inspector open", and that clause is the one the
# first round did not actually hold: ⌘Q reaches a Qt app as ``QEvent::Quit`` on the QApplication,
# and ``QApplication::event`` answers it by ``close()``ing every top-level window. The inspector
# IS one (a ``Qt.Tool``), so its own ``closeEvent`` fired, ``closed`` reached
# ``_on_inspector_closed``, and ``open: False`` went to disk on the way out -- the next launch
# restored nothing. Closing only the MAIN window never showed it: a parented ``Qt.Tool`` window is
# HIDDEN with its parent, not closed, so that path wrote nothing and restored correctly.


class _QuitSwallower(QtCore.QObject):
    """Eats ``QEvent::Quit`` so Qt's own handler never reaches its second half.

    ``QApplication::event(QEvent::Quit)`` does two things: ``closeAllWindows()``, then
    ``exit(0)``. The second is the one a test PROCESS cannot survive -- it sets the thread's
    ``quitNow``, after which every ``QEventLoop::exec()`` in this interpreter returns ``-1``
    immediately (measured: 0 ms instead of the loop's own 300 ms), which would break
    ``qtbot.waitSignal`` for every test that runs after it in the session.

    So the quit is driven in its two halves, and this filter is the seam: the Quit event really is
    SENT to the QApplication, so it flows through the app-level event filters exactly as ⌘Q's
    would -- which is where the fix hooks -- and is then swallowed here before Qt's handler sees
    it. ``closeAllWindows()`` is called by the test directly, which is the half that matters.

    Installed BEFORE the window under test, because Qt activates event filters in REVERSE order of
    installation: installed first means activated last, so ``MainWindow``'s own app-level filter
    still gets the event ahead of this one.
    """

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Quit:
            return True
        return super().eventFilter(obj, event)


def test_quitting_with_an_inspector_open_leaves_it_open_for_the_next_launch(qtbot, stub_devices):
    """Quit with an inspector open at a scale of its own; the next launch brings it back."""
    from PySide6 import QtWidgets

    from dynamix.shell.main_window import MainWindow

    app = QtWidgets.QApplication.instance()
    swallow = _QuitSwallower()
    app.installEventFilter(swallow)                    # installed first -> activated LAST
    try:
        first = MainWindow(steps=STUB_CHAIN)           # its own app filter -> activated FIRST
        qtbot.addWidget(first)
        with qtbot.waitSignal(first.resolved, timeout=10000):
            first.load_field(_FIELD, "mem:quit")
        source_id = first.layer.source_id
        inspector = _open_inspector(first, source_id)
        inspector.follow_button.setChecked(False)      # a scale of its own, so the index is real
        inspector.transport.slider.setValue(2)
        assert _saved_inspectors()[source_id].open is True

        QtWidgets.QApplication.sendEvent(app, QtCore.QEvent(QtCore.QEvent.Type.Quit))
        QtWidgets.QApplication.closeAllWindows()       # what Qt's own handler does next

        saved = _saved_inspectors()[source_id]
        assert saved.open is True                      # the window was open when the app went
        assert saved.scale_idx == 2

        second = MainWindow(steps=STUB_CHAIN)
        qtbot.addWidget(second)
        with qtbot.waitSignal(second.resolved, timeout=10000):
            second.load_field(_FIELD, "mem:quit")

        restored = second._inspectors[source_id]
        assert restored.isVisible() is True
        assert restored.transport.slider.value() == 2
    finally:
        app.removeEventFilter(swallow)


def test_closing_the_last_inspector_by_hand_then_quitting_still_writes_open_false(qtbot,
                                                                                  stub_devices):
    """The other half of the same discrimination: a window the USER closed stays closed across a
    quit. Without this, "don't write ``open: False`` while shutting down" would be free to become
    "never write it", and every inspector ever opened would come back forever."""
    from PySide6 import QtWidgets

    from dynamix.shell.main_window import MainWindow

    app = QtWidgets.QApplication.instance()
    swallow = _QuitSwallower()
    app.installEventFilter(swallow)
    try:
        win = MainWindow(steps=STUB_CHAIN)
        qtbot.addWidget(win)
        with qtbot.waitSignal(win.resolved, timeout=10000):
            win.load_field(_FIELD, "mem:quit-after-close")
        source_id = win.layer.source_id
        _open_inspector(win, source_id)
        # the way the user closes one: the same "I" button, pushed back down
        win.layer_list._source_rows[source_id].inspector_button.click()
        assert _saved_inspectors()[source_id].open is False

        QtWidgets.QApplication.sendEvent(app, QtCore.QEvent(QtCore.QEvent.Type.Quit))
        QtWidgets.QApplication.closeAllWindows()

        assert _saved_inspectors()[source_id].open is False
    finally:
        app.removeEventFilter(swallow)


def test_a_remembered_geometry_no_screen_can_show_is_not_replayed(qtbot, stub_devices):
    """A geometry saved on a second monitor must not be replayed on the laptop alone: the window
    would come back where nobody can see it while the layer list's "I" button reads open. The
    guard is a screen lookup, and the fallback is the platform's own placement."""
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import update_settings

    update_settings(view_options={"inspectors": {
        "src0": {"open": True, "geometry": [100000, 100000, 420, 380]}}})
    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:offscreen-geometry")

    inspector = win._inspectors["src0"]
    assert inspector.geometry().x() != 100000
    assert QtGui.QGuiApplication.screenAt(inspector.frameGeometry().center()) is not None


def test_a_remembered_geometry_a_screen_can_show_is_replayed(qtbot, stub_devices):
    """... and the guard does not cost the ordinary case its position."""
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import update_settings

    update_settings(view_options={"inspectors": {
        "src0": {"open": True, "geometry": [40, 60, 420, 380]}}})
    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:onscreen-geometry")

    inspector = win._inspectors["src0"]
    assert (inspector.geometry().x(), inspector.geometry().y()) == (40, 60)


# ------------------------------------------------- open the FULL extent, decimated (2026-08-30)
# Opening the full BOEM West file still clipped it to the launch scene -- open_window_px windows every fresh GeoTIFF open at native resolution. The full
# extent is 800 Mpx, so the whole-file open is the OVERVIEW: decimated, provenance says so,
# context only (WTMM on resampled pixels violates the native-grid rule).

def test_open_full_extent_overview_opens_the_whole_raster_decimated(window, tmp_path, monkeypatch):
    import rasterio
    from PySide6 import QtWidgets
    from rasterio.transform import from_origin
    path = tmp_path / "big.tif"
    with rasterio.open(path, "w", driver="GTiff", height=64, width=48, count=1, dtype="float32",
                       crs="EPSG:32750", transform=from_origin(500000, 9000000, 30, 30)) as dst:
        dst.write(np.random.default_rng(0).random((64, 48)).astype(np.float32), 1)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    window._on_open_overview_clicked()
    assert window.field is not None
    # The menu opens the display PICTURE, not an overview
    # dataset -- no ``overview`` key (the x-ov mappings stay inert), file-pixel dims.
    assert "overview" not in window.field.provenance
    assert window.field.provenance.get("display_stride") == 1      # small file: stride 1
    assert window.field.values.shape == (64, 48)                   # the WHOLE extent, no window
    assert "@pic" in window.field.name
    actions = [a.text() for m in window.menuBar().actions() for a in m.menu().actions()]
    assert "Open Full Extent (Overview)…" in actions


def test_expand_reference_paths_extracts_shapefile_zips(tmp_path):
    """A GIS-portal shapefile .zip (2026-09-18, gcfaultsg) expands to its .shp members via a
    one-time sidecar extraction; a non-shapefile zip passes through untouched."""
    import io
    import zipfile

    from dynamix.shell.main_window import _expand_reference_paths

    z = tmp_path / "faults.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            zf.writestr(f"faults{ext}", b"stub")
    out = _expand_reference_paths([z])
    assert len(out) == 1 and out[0].endswith("faults.shp")
    assert (tmp_path / "faults_shp" / "faults.dbf").is_file()   # sidecars extracted beside
    out2 = _expand_reference_paths([z])                          # reused, not re-extracted
    assert out2 == out

    other = tmp_path / "raster.zip"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("scene.tif", b"stub")
    assert _expand_reference_paths([other]) == [str(other)]


def test_roi_create_scales_a_decimated_overview_to_native_pixels(qtbot, loaded):
    """Overview drill-down (2026-09-18): a footprint drawn on a decimation-16 overview lands
    in NATIVE file coordinates — offsets AND size scaled — so wtmm2d_roi reads full-res halos."""
    import dataclasses

    field = loaded._fields[loaded.layer.layer_id]
    if not hasattr(field, "provenance"):
        import numpy as np

        from dynamix.core.frames import LocalFrame
        from dynamix.core.rasterfield import RasterField
        vals = np.asarray(field, dtype=float)
        field = RasterField(name="ov", values=vals, frame=LocalFrame(),
                            x_axis=np.arange(vals.shape[1], dtype=float),
                            y_axis=np.arange(vals.shape[0], dtype=float))
    field = dataclasses.replace(field, provenance=dict(
        getattr(field, "provenance", {}) or {},
        source="/tmp/fake.tif", overview=16,
        window={"row_off": 0, "col_off": 0}))
    loaded._fields[loaded.layer.layer_id] = field
    real_add = loaded.project.add_layer
    captured = {}

    def spy_add_layer(name, source_id, chain, **kw):
        layer = real_add(name, source_id, chain, **kw)
        captured["chain"] = chain
        return layer

    loaded.project.add_layer = spy_add_layer
    loaded.add_layer_row = lambda *a, **k: None
    loaded.layer_list.select_layer = lambda *a, **k: None
    loaded._on_roi_create({"roi_row": 3, "roi_col": 5, "roi_h": 8, "roi_w": 10,
                           "boundary": "reflective"})
    step = next(ref for ref in captured["chain"].steps if ref.device == "wtmm2d_roi")
    assert step.params["roi_row"] == 48 and step.params["roi_col"] == 80
    assert step.params["roi_h"] == 128 and step.params["roi_w"] == 160
