# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The integration sweep -- the whole workflow-manager slice, wired end to end.

Three tests. The first drives the REAL app path a user actually takes: settings off leaves a
freshly opened raster showing nothing but the source box (the inert open), dragging the
"WTMM standard" preset out of the browser through the real ``WorkflowZone`` drop path
builds the DEFAULT_STEPS chain and dispatches the worker (the same worker-wait pattern
``tests/test_shell_window.py`` uses throughout), and the canvas ends up holding real extrema.
``chain_classify`` then drops onto the tail of the chain as an ordinary filter and must
not crash over a field with no seam chains to find.

The second is a pure model-layer round trip (no Qt at all, same style as ``tests/test_project.py``):
every flag, group-collapse state and chain -- including the five WTMM params exposed later --
that ``Project.to_payload()``/``from_payload()`` are supposed to carry survives.

The third pins the boundary that round trip deliberately does NOT cross: bypass/rack are
``WorkflowZone``/``MainWindow`` VIEW state (``main_window.py``'s own ``_recipes`` docstring says
so), never a field on ``Project``/``Layer``/``Chain``/``DeviceRef`` -- a real bypass+rack gesture,
driven through ``main_window._on_chain_edited`` exactly as a drag-drop would, leaves no trace in
what a save would persist.

Offscreen Qt (the runner sets ``QT_QPA_PLATFORM=offscreen``), same convention every other shell
test in this suite uses.
"""
from __future__ import annotations

import json

import numpy as np
from PySide6 import QtCore

from dynamix.devices import register_builtin_devices
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.param import Param, ParamKind
from dynamix.model.project import Project

# --------------------------------------------------------------------------- stub drop plumbing


class _StubDropEvent:
    """Minimal ``QDropEvent`` stand-in -- ``tests/test_workflow_zone.py`` and
    ``tests/test_shell_window.py`` both carry their own copy of this for the same reason: pytest-qt
    cannot synthesize a real native drag reliably offscreen, so the zone's ``dropEvent`` is called
    directly instead of driving one through Qt's own event queue."""

    def __init__(self, mime: QtCore.QMimeData):
        self._mime = mime
        self.accepted: bool | None = None

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


def _preset_drop(name: str) -> _StubDropEvent:
    from dynamix.shell.browser import PRESET_MIME

    mime = QtCore.QMimeData()
    mime.setData(PRESET_MIME, name.encode("utf-8"))
    return _StubDropEvent(mime)


# --------------------------------------------------------------------------- step 1: E2E sweep


def test_inert_open_preset_drop_and_seam_classifier_end_to_end(
        qtbot, fbm64, clean_registry, tmp_path, monkeypatch):
    """settings off -> load_field (synthetic 64x64) -> zone shows the source box and NO devices ->
    drop "WTMM standard" through the real Task-7 path -> the worker computes -> the canvas holds
    extrema -> chain_classify (flag mode, the device default) drops onto the tail and must not
    crash, tagging zero seam chains on this random field."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.shell.main_window import MainWindow

    # No settings file at this path -> load_settings() falls back to auto_run_wtmm=False, the
    # TRUE default -- overriding the suite-wide autouse fixture that turns it on for every other
    # MainWindow test (tests/conftest.py; test_shell_window.py's own inert-open tests do the same).
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    register_builtin_devices()

    field = RasterField(name="fbm", values=fbm64, frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_field(field, "mem:integration")

    # -- inert open: raster displayed, nothing computed, zone holds only the SourceBox
    assert win.layer.chain.steps == ()
    assert win._thread is None
    assert win.canvas._field is not None
    assert win.strips is not None
    assert len(win.strips._boxes) == 0
    assert win.strips.source_box.name_label.text() != ""

    # -- drop the "WTMM standard" preset. Root protection: an analyzer landing on a RAW MASTER forks a child
    # representation, which -- the ROI-create precedent -- computes immediately (creating an
    # analysis layer IS the run gesture; the §5e run gate governs in-place edits, tested
    # where those live). The master stays raw. The spawn is deferred one tick, so the
    # resolved wait also pumps the singleShot.
    master = win.layer
    preset_event = _preset_drop("WTMM standard")
    with qtbot.waitSignal(win.resolved, timeout=60000) as sig:
        win.strips.dropEvent(preset_event)
    assert preset_event.accepted is True
    assert master.chain.steps == ()                     # the master is still the raw dataset
    assert win.layer is not master
    assert win.layer.parent_id == master.layer_id       # the fork: a grouped child
    assert [ref.device for ref in win.layer.chain.steps] == \
        [name for name, _ in MainWindow.DEFAULT_STEPS]
    assert len(win.strips._boxes) == len(MainWindow.DEFAULT_STEPS)

    result = sig.args[0].result
    assert len(result["extrema"]) == 1
    assert len(result["extrema"][0]["x"]) > 0          # something to look at
    xs, ys = win.canvas.extrema_item.getData()
    assert xs is not None and len(xs) > 0
    assert win.is_computing is False

    # -- add chain_classify (flag mode is the device default) as a filter drop
    classify_event = _device_drop("chain_classify")
    with qtbot.waitSignal(win.resolved, timeout=10000) as sig2:
        win.strips.dropEvent(classify_event)

    assert classify_event.accepted is True
    assert win.layer.chain.steps[-1].device == "chain_classify"
    assert win.is_computing is False               # a filter drop never dispatches the worker
    chains = sig2.args[0].result.get("chains") or []
    assert chains, "chain_topology/wtmm2d produced no chains to classify at all"
    seam_tagged = [c for c in chains if c.get("tags")]
    assert seam_tagged == []                        # a smooth synthetic field has no seam lines

    # -- chain_holder, dropped for real through main_window's own wiring, shows
    # a real data-derived reading -- not the old silent-no-op default -- and the hint-triggered
    # snap (routed through _on_param_changed -> _reresolve -> _resolve_now, RE-ENTRANT from
    # inside this very landing's own _update_readings) fires EXACTLY once. Count actual _apply invocations rather than only checking the wait didn't time out -- a
    # spy that quietly regressed to zero extra passes (dead seam) or an unbounded re-entrant loop
    # (bug) would both still pass a bare "no timeout" assertion.
    import dynamix.shell.main_window as main_window_mod

    apply_calls: list = []
    original_apply = main_window_mod.MainWindow._apply

    def _counting_apply(self, renderable):
        apply_calls.append(renderable)
        return original_apply(self, renderable)

    monkeypatch.setattr(main_window_mod.MainWindow, "_apply", _counting_apply)

    holder_event = _device_drop("chain_holder")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.strips.dropEvent(holder_event)

    assert holder_event.accepted is True
    assert win.layer.chain.steps[-1].device == "chain_holder"
    assert win.is_computing is False
    # exactly ONE landing (the initial pass-all default) plus exactly ONE snap-triggered extra
    # pass -- never 1 (the seam would be dead: no snap ever fired) and never >2 (an unbounded
    # re-entrant loop).
    assert len(apply_calls) == 2
    box = win.strips.strip(win._names.index("chain_holder"))
    assert box.reading_label.text().startswith("kept ")
    assert "h ∈" in box.reading_label.text()
    assert box.controls["cutoff"].isEnabled() is True


# --------------------------------------------------------------------------- step 2: round trip


def test_project_round_trip_carries_every_flag_group_state_and_wtmm_param(clean_registry):
    """Two sources (one collapsed), one locked layer, one frozen layer, one refined-run layer
    with ``fed_by``, and a WTMM chain tuned on all five of those params -- every one of those
    survives ``Project.to_payload()`` -> ``Project.from_payload()``, driven at the model layer
    only (no MainWindow/Qt needed for what this is actually testing: the persisted document)."""
    register_builtin_devices()

    p = Project(title="integration round trip")
    src_a = p.add_source("/data/a.tif")
    src_b = p.add_source("/data/b.tif")
    src_a.collapsed = True                      # group state: a collapsed source header

    tuned_wtmm = {"n_oct": 4, "n_voice": 6, "a_min": 1.5, "wavelet": "mexican",
                  "min_chain_len": 3, "smooth": False, "thresh": 0.01, "dist2_max": 75.0,
                  "box_ratio": 2.0, "similitude": 0.5}
    locked = p.add_layer(
        "locked layer", src_a.source_id,
        Chain((DeviceRef("wtmm2d", dict(tuned_wtmm)), DeviceRef("scale_select", {"scale_idx": 1}))),
        tags={"ui.lock": "1"})
    frozen = p.add_layer(
        "frozen layer", src_b.source_id, Chain((DeviceRef("wtmm2d", {}),)),
        tags={"ui.freeze": "1"})
    refined = p.add_layer(
        "locked layer · refined", src_a.source_id, Chain((DeviceRef("wtmm2d", dict(tuned_wtmm)),)),
        parent_id=locked.layer_id, tags={"fed_by": str(locked.layer_id)})

    payload = p.to_payload()
    back = Project.from_payload(payload)

    # -- sources + group (collapse) state
    assert set(back.sources) == {src_a.source_id, src_b.source_id}
    assert back.sources[src_a.source_id].collapsed is True
    assert back.sources[src_b.source_id].collapsed is False

    # -- flags
    back_locked = next(l for l in back.layers if l.layer_id == locked.layer_id)
    back_frozen = next(l for l in back.layers if l.layer_id == frozen.layer_id)
    back_refined = next(l for l in back.layers if l.layer_id == refined.layer_id)
    assert back_locked.tags.get("ui.lock") == "1"
    assert back_frozen.tags.get("ui.freeze") == "1"
    assert back_refined.tags.get("fed_by") == str(locked.layer_id)
    assert back_refined.parent_id == locked.layer_id

    # -- chain, including the five new WTMM params
    wtmm_step = back_locked.chain.steps[0]
    assert wtmm_step.device == "wtmm2d"
    for name, value in tuned_wtmm.items():
        assert wtmm_step.params[name] == value
    assert back_locked.chain.steps[1] == DeviceRef("scale_select", {"scale_idx": 1})

    # -- the strongest single assertion: the round trip is idempotent (tests/test_project.py's
    # own convention), so nothing else about the document silently drifted either
    assert Project.from_payload(payload).to_payload() == payload


# ------------------------------------------------------- bypass/rack are view state, by design


class _FastStack:
    """A minimal, fast, real Transform -- deterministic, no numba/wtmm backend involved -- used
    only to prove the round-trip persistence BOUNDARY below. ``test_real_wtmm_end_to_end`` (and
    this file's own first test) already cover the actual WTMM backend; this one is not trying to
    be it."""

    name = "fast_stack"
    params = (Param("n_scales", ParamKind.INT, default=2, min=1, max=8, label="Scales"),)

    def compute(self, field, params, *, progress=None):
        values = np.asarray(getattr(field, "values", field), dtype=np.float64)
        ny, nx = values.shape[:2]
        n = int(params["n_scales"])
        layers = [{"x": np.arange(4) % nx, "y": np.arange(4) % ny,
                   "mod": np.linspace(1.0, 0.1, 4), "arg": np.zeros(4),
                   "line_id": np.full(4, -1, dtype=np.int64)}
                  for _ in range(n)]
        return {"scales": [2.0 * (k + 1) for k in range(n)], "extrema": layers, "_shape": (ny, nx)}

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


def test_bypass_and_rack_never_reach_the_saved_project(qtbot, clean_registry):
    """NOTE (the design's callout): bypass/rack are ``WorkflowZone``/``MainWindow`` VIEW state --
    ``main_window._recipes`` is where they actually live, keyed by ``layer_id``, and it is never
    part of ``self.project``. ``Project``/``Layer``/``Chain``/``DeviceRef`` have no field for
    either one. This drives a real bypass+rack gesture through ``main_window._on_chain_edited``
    (the same method a real drag-drop drop lands on, per ``tests/test_shell_window.py``'s own
    chain-editing tests) and asserts the DESIGNED behaviour -- gone from what a save would
    persist, still held in the window's own view-state cache -- not anything stronger."""
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    register_device(_FastStack())
    win = MainWindow(steps=(("fast_stack", {}), ("scale_select", {"scale_idx": 0})))
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(np.zeros((8, 8)), "mem:roundtrip_boundary")

    descriptors = [{"device": n, "params": dict(p), "bypassed": False, "rack": None}
                   for n, p in zip(win._names, win._params)]
    descriptors[1]["bypassed"] = True
    descriptors[1]["rack"] = "my rack"
    win._on_chain_edited(descriptors)

    assert win._bypassed[1] is True
    assert win._rack[1] == "my rack"
    # the bypassed step is excluded from the CHAIN entirely, not merely flagged
    assert [ref.device for ref in win.layer.chain.steps] == ["fast_stack"]

    payload = win.project.to_payload()
    blob = json.dumps(payload)
    assert "bypassed" not in blob
    assert "my rack" not in blob
    assert "scale_select" not in blob          # the excluded step left no trace at all

    # the view state is real -- just not the project's. main_window's own per-layer cache holds
    # exactly what the save does not.
    recipe = win._recipes[win.layer.layer_id]
    assert recipe[1]["bypassed"] is True
    assert recipe[1]["rack"] == "my rack"
    assert not hasattr(win.project, "_recipes")
