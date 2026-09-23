# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the commit-to-raster transaction.

Offscreen Qt + pyvista (mandated repo-wide / installed in this environment, same convention as
tests/test_arrangement_picking.py): the Commit button and GroupPalette are built eagerly in
``MainWindow.__init__`` whenever pyvista/pyvistaqt import (``available=True``) -- both were relocated out of ``ArrangementView`` onto ``MainWindow`` itself
(``win._group_palette``/``win._commit_button``, not ``win._arrangement._group_palette``/
``win._arrangement._commit_button`` any more) -- independent of whether a real ``Scene`` was ever
built (it isn't, under offscreen QPA -- see tests/test_arrangement_flip.py's documented segfault
guard) -- so the whole commit flow is testable exactly as the palette/pick tests are, with
``view._scene`` staying ``None`` throughout MOST of this file. The one exception is the
multi-layer-commit section below, which attaches a REAL ``Scene`` against a plain offscreen
``pv.Plotter`` (the same substitution
``test_arrangement_scene.py``/``test_arrangement_picking.py`` use throughout) -- one bug was only reproducible with a real ``Scene`` actually running its own staleness-prune
logic, which every OTHER test in this file (``_scene`` left ``None``) cannot exercise at all.

A custom stub transform, ``_ChainStub``, stands in for ``wtmm2d`` + ``chain_topology``:
STUB_CHAIN's own ``stub_stack`` (tests/test_shell_window.py) is WTMM-shaped
(``scales``/``extrema``/``_shape``) but carries no ``chains`` list at all -- that key is
``dynamix/core``'s own ``wtmm2d`` output (``dynamix/core/wtmm_backend.py``), never built by
``chain_topology`` (which only builds the incidence GRAPH). ``group_paint`` reads ``chains``
directly, so this task's own stub adds it, with a param (``n_chains``) a test can turn to shrink
the list and make a previously-committed index go stale.

The canvas-level section near the bottom needs neither pyvista nor a window; it extends
tests/test_shell_canvas.py's own Task-4/5 ``_flagged_result``-style assertions to the ``classify_chains``/grouped-trail-item machinery, kept here (rather than in that file) so this
task's whole test surface lives in the one file the design names.
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista not installed")
pv.OFF_SCREEN = True

from dynamix.devices import register_builtin_devices
from dynamix.devices.groups import decode_groups
from dynamix.engine import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.param import Param, ParamKind
from dynamix.shell.arrangement.group_palette import GROUP_COLORS
from dynamix.shell.arrangement.scene import Scene
from dynamix.shell.canvas import Canvas

from tests.test_arrangement_resolve import _geo_field
from tests.test_shell_window import _FIELD, _stub_stack


def _chains(n=3):
    all_chains = [
        {"x": np.array([1, 2], dtype=np.int64), "y": np.array([1, 2], dtype=np.int64),
         "mod": np.array([1.0, 1.0])},
        {"x": np.array([4, 5], dtype=np.int64), "y": np.array([4, 5], dtype=np.int64),
         "mod": np.array([1.0, 1.0])},
        {"x": np.array([7], dtype=np.int64), "y": np.array([7], dtype=np.int64),
         "mod": np.array([1.0])},
    ]
    return all_chains[:n]


class _ChainStub:
    """Like ``test_shell_window.py``'s ``_StubStack``, but the result also carries a ``chains``
    list -- what ``group_paint`` reads. ``n_chains`` lets a test shrink the list on a
    re-resolve, to make a previously-committed member index go out of range (the staleness path,
    ``GroupPaint``'s own structural check)."""

    name = "chain_stub"
    params = (Param("n_chains", ParamKind.INT, default=3, min=1, max=3, label="N chains"),)

    def compute(self, field, params, *, progress=None):
        out = _stub_stack(field, 1)
        out["chains"] = _chains(int(params["n_chains"]))
        return out

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key

        return cache_key(self.name, source_id, params)


@pytest.fixture
def commit_devices(clean_registry):
    """Real registry (``group_paint``/``group_filter`` included) plus ``_ChainStub``."""
    from dynamix.model.device import register_device

    register_builtin_devices()
    register_device(_ChainStub())
    return clean_registry


@pytest.fixture
def raster(tmp_path):
    """A real on-disk raster (test_project_menu.py's own proven ``.npz`` convention) -- needed so
    the round-trip test's ``_open_project_path`` can genuinely re-read it."""
    p = tmp_path / "field.npz"
    np.savez(p, np.asarray(_FIELD, dtype=np.float64))
    return p


@pytest.fixture
def win(qtbot, commit_devices, raster):
    """A window on a single ``chain_stub`` layer, flipped to the arrangement.

    The group palette and Commit button are ``win._group_palette``/
    ``win._commit_button`` now -- built unconditionally in ``MainWindow.__init__``, same as the
    mask row, not by ``ArrangementView`` any more. Flipping is still required here: ``ArrangementView.
    commit()`` (what the button now reaches through ``_on_commit_button_clicked``) is a no-op
    before ``MainWindow._toggle_center_view`` has handed the view its palette reference at least
    once (``set_group_palette``, called from that method's own first-build branch)."""
    from dynamix.shell.main_window import MainWindow

    w = MainWindow(steps=(("chain_stub", {}),))
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.resolved, timeout=10000):
        w.open_path(str(raster))
    w._toggle_center_view()          # wires w._group_palette into w._arrangement
    return w


def _commit(win, layer_id, name, indices):
    """New group ``name`` on ``win``'s group palette (``win._group_palette`` -- relocated onto ``MainWindow`` itself), pick ``indices`` (chain indices on
    ``layer_id``) into it, click Commit (``win._commit_button``, likewise relocated). Membership
    accumulates regardless of shift state (the palette's own documented "union, never a replace"
    semantics), so every pick here is a plain click."""
    palette = win._group_palette
    palette.new_group(name)
    for idx in indices:
        palette.add_pick((layer_id, idx), shift=False)
    win._commit_button.click()


# --------------------------------------------------------------------------------------- commit


def test_commit_button_click_is_a_no_op_before_the_arrangement_is_ever_built(qtbot, clean_registry):
    """The group palette and Commit button are built unconditionally
    in ``MainWindow.__init__``, same as the mask row -- so a click reaches ``_on_commit_button_
    clicked`` even for a window that has never pressed Tab. Must not raise (mirrors ``_on_mask_row_
    changed``'s own no-op-before-``self._arrangement``-exists contract)."""
    from dynamix.shell.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    assert w._arrangement is None

    w._group_palette.new_group("fault_a")
    w._group_palette.add_pick((1, 0), shift=False)
    w._commit_button.click()          # must not raise -- no arrangement to forward the click to

    assert w._arrangement is None     # clicking Commit never builds the arrangement by itself


def test_arrangement_view_commit_is_a_no_op_when_no_group_palette_was_ever_attached(qtbot):
    """The VIEW's own half of the same headless guard: calling ``ArrangementView.commit()``
    directly, before ``set_group_palette`` was ever called, must not raise -- mirrors
    tests/test_arrangement_mask.py's ``test_arrangement_view_set_mask_forwards_to_scene_lazily``
    no-op-before-anything-attached contract for the mask."""
    from dynamix.shell.arrangement.view import ArrangementView

    view = ArrangementView()
    qtbot.addWidget(view)
    if not view.available:
        pytest.skip("pyvista/pyvistaqt not installed in this environment")

    view.commit()   # no palette attached -- must not raise


def test_commit_writes_tags_with_signature_and_color(qtbot, win):
    layer = win.layer
    _commit(win, layer.layer_id, "fault_a", [0, 1])

    groups = decode_groups(layer.tags["groups"])
    assert set(groups) == {"fault_a"}
    assert groups["fault_a"]["chains"] == [0, 1]
    assert groups["fault_a"]["color"] == list(GROUP_COLORS[0])
    assert groups["fault_a"]["signature"]                       # non-empty real signature string


def test_commit_appends_exactly_one_group_paint_step(qtbot, win):
    layer = win.layer
    _commit(win, layer.layer_id, "fault_a", [0])

    steps = [s.device for s in layer.chain.steps]
    assert steps.count("group_paint") == 1
    assert steps[-1] == "group_paint"                           # after the existing chain_stub


def test_recommit_updates_params_in_place_never_a_second_step(qtbot, win):
    layer = win.layer
    _commit(win, layer.layer_id, "fault_a", [0])
    first_ref = next(s for s in layer.chain.steps if s.device == "group_paint")
    first_spec = first_ref.params["spec_json"]

    palette = win._group_palette
    palette.add_pick((layer.layer_id, 1), shift=True)           # grow fault_a's own membership
    win._commit_button.click()                                   # re-commit, no new group created

    group_paint_steps = [s for s in layer.chain.steps if s.device == "group_paint"]
    assert len(group_paint_steps) == 1
    second_spec = group_paint_steps[0].params["spec_json"]
    assert second_spec != first_spec
    assert decode_groups(second_spec)["fault_a"]["chains"] == [0, 1]


def test_recommit_clears_bypass_on_its_own_painter_step(qtbot, win):
    """Fold-in fix (review): a commit un-bypasses its own painter -- pre-fix, re-committing onto
    a ``group_paint`` step the user had bypassed left it bypassed, silently discarding the just-
    committed groups (``layer.chain.steps`` excludes a bypassed step entirely -- ``_chain()``'s
    own docstring) from every future resolve until someone happened to un-bypass it by hand."""
    layer = win.layer
    _commit(win, layer.layer_id, "fault_a", [0])
    i = win._names.index("group_paint")
    assert any(s.device == "group_paint" for s in layer.chain.steps)

    win._bypassed[i] = True                      # user bypasses the painter after the first commit
    layer.chain = win._chain()
    win._snapshot_recipe()
    assert not any(s.device == "group_paint" for s in layer.chain.steps)   # excluded while bypassed

    palette = win._group_palette
    palette.add_pick((layer.layer_id, 1), shift=True)
    win._commit_button.click()                                   # re-commit onto the bypassed step

    assert win._bypassed[win._names.index("group_paint")] is False
    assert any(s.device == "group_paint" for s in layer.chain.steps)       # back in the materialized chain


def test_resolve_paints_copies_leaving_cached_originals_untouched(qtbot, win):
    layer, field = win.layer, win.field
    before = resolve(layer, field, win.cache, source_id=layer.source_id)
    originals = list(before.result["chains"])                   # keep references alive

    _commit(win, layer.layer_id, "fault_a", [0, 1])

    after = resolve(layer, field, win.cache, source_id=layer.source_id)
    painted = after.result["chains"]
    assert painted[0] is not originals[0]
    assert painted[1] is not originals[1]
    assert painted[2] is originals[2]                            # untouched index: same object
    assert "tags" not in originals[0] and "tags" not in originals[1]
    assert painted[0]["tags"] == ["group:fault_a"]
    assert painted[0]["group_color"] == list(GROUP_COLORS[0])


def test_canvas_shows_group_colors_after_commit(qtbot, win):
    layer = win.layer
    _commit(win, layer.layer_id, "fault_a", [0, 1])              # -> _reresolve() -> canvas paint

    color = tuple(GROUP_COLORS[0])
    item = win.canvas._group_items[color]
    gx, gy = item.getData()
    assert gx.size > 0


def test_groups_committed_emitted_once_per_affected_layer(qtbot, win):
    """``groupsCommitted`` carries ``(layer_id, groups)`` -- each
    layer's own slice of the ONE snapshot the click took, not a live palette re-read."""
    active = win.layer
    src = win.project.add_source("mem:b")
    bg = win.project.add_layer("bg", src.source_id, Chain((DeviceRef("chain_stub", {}),)))
    win.add_layer_row(bg, _FIELD)

    palette = win._group_palette
    palette.new_group("g")
    palette.add_pick((active.layer_id, 0), shift=False)
    palette.add_pick((bg.layer_id, 0), shift=True)

    received = []
    win._arrangement.groupsCommitted.connect(lambda lid, groups: received.append((lid, groups)))
    win._commit_button.click()

    received_ids = sorted(lid for lid, _ in received)
    assert received_ids == sorted([active.layer_id, bg.layer_id])
    payload_by_id = dict(received)
    assert payload_by_id[active.layer_id]["g"]["chains"] == [0]
    assert payload_by_id[bg.layer_id]["g"]["chains"] == [0]


def test_background_layer_commit_resyncs_arrangement_not_the_active_layer(qtbot, win, monkeypatch):
    """Contract: active layer takes the normal (filter) resolve path; a background one resyncs
    the arrangement instead -- and never touches the layer that ISN'T the one being committed."""
    active = win.layer
    src = win.project.add_source("mem:b")
    bg = win.project.add_layer("bg", src.source_id, Chain((DeviceRef("chain_stub", {}),)))
    win.add_layer_row(bg, _FIELD)
    assert win.layer is active                                  # adding a row never re-selects

    palette = win._group_palette
    palette.new_group("fault_b")
    palette.add_pick((bg.layer_id, 0), shift=False)

    calls = []
    monkeypatch.setattr(win, "_sync_arrangement", lambda: calls.append("sync"))
    win._commit_button.click()

    assert calls == ["sync"]
    assert any(s.device == "group_paint" for s in bg.chain.steps)
    assert decode_groups(bg.tags["groups"])["fault_b"]["chains"] == [0]
    assert not any(s.device == "group_paint" for s in active.chain.steps)
    assert "groups" not in active.tags


# ------------------------------------------------------- A REAL Scene, multi-layer


def test_multilayer_commit_with_a_real_scene_survives_its_own_resync(qtbot, commit_devices,
                                                                       tmp_path):
    """The cross-layer commit bug, end to end, exactly as reproduced: a group spanning an
    ACTIVE and a BACKGROUND layer, committed together, against a REAL ``Scene`` (not the ``None``
    every other test in this file leaves it at). The ORIGINAL bug: handling layer 1's own
    ``groupsCommitted`` used to call ``_sync_arrangement()`` immediately, which re-derives every
    visible layer's entry through a fresh ``resolve()`` (every Filter always copies -- a new
    ``result`` identity even though nothing changed) and hands it to the real ``Scene``, whose
    staleness-prune (before Critical 1c's fingerprint fix) read that as "layer 2 just went stale"
    and erased its still-uncommitted palette membership -- BEFORE layer 2's own ``groupsCommitted``
    handler ever ran. Fixed by (a) snapshotting the palette once, (b) deferring the resync to
    AFTER the whole click's loop, and (c) narrowing the prune predicate; this test would fail on
    any one of the three being reverted.
    """
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=(("chain_stub", {}),))
    qtbot.addWidget(win)
    active_field = _geo_field(tmp_path, "active")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(active_field, str(tmp_path / "active.tif"))
    active = win.layer

    bg_field = _geo_field(tmp_path, "bg")
    bg_src = win.project.add_source(str(tmp_path / "bg.tif"))
    bg = win.project.add_layer("bg", bg_src.source_id, Chain((DeviceRef("chain_stub", {}),)))
    win.add_layer_row(bg, bg_field)
    assert win.layer is active

    win._toggle_center_view()
    win._arrangement._scene = Scene(pv.Plotter(off_screen=True))    # the REAL path, not None
    win._sync_arrangement()          # both layers land on the real Scene once, before any commit

    palette = win._group_palette
    palette.new_group("fault_a")
    palette.add_pick((active.layer_id, 0), shift=False)
    palette.add_pick((bg.layer_id, 0), shift=True)

    win._commit_button.click()

    # Both layers actually got committed -- the bug silently dropped every layer after the first.
    assert decode_groups(active.tags["groups"])["fault_a"]["chains"] == [0]
    assert decode_groups(bg.tags["groups"])["fault_a"]["chains"] == [0]
    # The palette itself survived its OWN commit's resync -- the bug's own symptom.
    assert palette.groups()["fault_a"]["chains"] == [(active.layer_id, 0), (bg.layer_id, 0)]

    active_groups_before = active.tags["groups"]
    bg_groups_before = bg.tags["groups"]

    win._sync_arrangement()          # an extra, later resync -- must not shrink anything

    assert active.tags["groups"] == active_groups_before
    assert bg.tags["groups"] == bg_groups_before
    assert palette.groups()["fault_a"]["chains"] == [(active.layer_id, 0), (bg.layer_id, 0)]

    # Recommit: membership GREW (one more pick on the active layer) -- must not shrink either
    # layer's own entry, which is what "recommits after ANY later sync write a SHRUNKEN spec_json" described.
    palette.add_pick((active.layer_id, 1), shift=True)
    win._commit_button.click()

    assert decode_groups(active.tags["groups"])["fault_a"]["chains"] == [0, 1]
    assert decode_groups(bg.tags["groups"])["fault_a"]["chains"] == [0]


# ------------------------------------------------- Flipped filter-edit resync (fix)


class _ReorderFilter:
    """Test-only filter (the stale-pick repro): same chain COUNT, different chain identity-at-
    position once ``reorder`` flips True -- a filter's own params are never part of the transform
    signature (the engine's own law), so this reproduces exactly the "count-preserving
    reselection" scenario ``scene.py``'s own fingerprint docstring names (a filter admitting a
    DIFFERENT, equal-sized set of chains). Reorders the SAME chain objects the cached transform
    already produced (never mints new ones) -- proving the digest moves because of WHICH chain
    sits at which index, not because anything was recomputed."""

    name = "reorder_filter"
    params = (Param("reorder", ParamKind.BOOL, default=False, label="Reorder"),)

    def apply(self, result, params):
        chains = result.get("chains") or []
        if not params["reorder"] or len(chains) < 2:
            return result
        out = dict(result)
        out["chains"] = list(reversed(chains))
        return out


@pytest.fixture
def reorder_win(qtbot, commit_devices, tmp_path):
    """Like ``win``, but the chain also carries ``reorder_filter`` after ``chain_stub`` -- a
    genuine FILTER whose own param can change WHICH chain sits at index 0 without moving the
    transform signature (or the cache key) at all -- and a REAL ``Scene`` attached (the ``win``
    fixture's own module docstring: every OTHER test in this file leaves ``_scene`` at ``None``,
    which cannot exercise the staleness-prune logic the stale-pick fix depends on)."""
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    register_device(_ReorderFilter())
    w = MainWindow(steps=(("chain_stub", {}), ("reorder_filter", {})))
    qtbot.addWidget(w)
    field = _geo_field(tmp_path, "active")
    with qtbot.waitSignal(w.resolved, timeout=10000):
        w.load_field(field, str(tmp_path / "active.tif"))
    w._toggle_center_view()
    w._arrangement._scene = Scene(pv.Plotter(off_screen=True))
    w._sync_arrangement()
    return w


def test_filter_edit_while_flipped_resyncs_the_scene_with_a_new_digest(qtbot, reorder_win):
    """A filter edit made while flipped must reach the arrangement, not
    just the session canvas -- pre-fix, NONE of ``_on_param_changed``'s filter branch,
    ``_on_chain_edited`` or ``_on_scale_changed`` touched it, so the scene kept rendering the OLD
    chains while the session result silently changed underneath it."""
    win = reorder_win
    layer = win.layer
    scene = win._arrangement._scene
    fp_before = scene._result_fingerprint[layer.layer_id]

    i = win._names.index("reorder_filter")
    win._on_param_changed(i, "reorder", True)     # a genuine FILTER edit, thread idle -> _resolve_now

    fp_after = scene._result_fingerprint[layer.layer_id]
    assert fp_after != fp_before                                    # a genuinely new digest
    entry = next(e for e in win._arrangement._layers if e["layer"] is layer)
    assert entry["result"]["chains"][0] is not entry["result"]["chains"][2]  # sanity: reordered


def test_filter_edit_while_flipped_prunes_the_pick_before_a_wrong_commit_is_possible(
        qtbot, reorder_win):
    """The repro, full loop: pick chain index 0 into a group, then a filter edit
    (while still flipped) reselects WHICH chain sits at index 0 -- an in-range index, so
    ``group_paint``'s own out-of-range floor cannot catch it. The fix must prune the
    now-wrong pick/membership honestly (the SAME positional-digest fingerprint ``scene.py``'s own
    Critical 1c fix already uses) before Commit is even clicked, so the commit binds NOTHING for
    the stale pick rather than painting the wrong chain silently."""
    win = reorder_win
    layer = win.layer
    palette = win._group_palette
    palette.new_group("fault_a")
    palette.add_pick((layer.layer_id, 0), shift=False)
    assert palette.groups()["fault_a"]["chains"] == [(layer.layer_id, 0)]
    assert win._arrangement._scene._selection.get(layer.layer_id) == {0}

    i = win._names.index("reorder_filter")
    win._on_param_changed(i, "reorder", True)          # filter edit while still flipped

    assert palette.groups()["fault_a"]["chains"] == []                        # membership pruned
    assert win._arrangement._scene._selection.get(layer.layer_id, set()) == set()  # highlight pruned

    win._commit_button.click()

    assert "groups" not in layer.tags       # nothing bound for the stale pick -- no wrong chain painted


def test_active_layer_only_commit_also_syncs_the_arrangement_while_flipped(qtbot, win, monkeypatch):
    """Fold-in: even a commit that touches ONLY the active layer must resync
    the flipped arrangement -- falls out of the ``_resolve_now`` choke point
    (``_on_commit_finished``'s active-layer branch runs ``_reresolve()``, which is now the thing
    that resyncs on a successful synchronous resolve), verified directly rather than assumed."""
    layer = win.layer
    calls = []
    monkeypatch.setattr(win, "_sync_arrangement", lambda: calls.append(1))

    _commit(win, layer.layer_id, "fault_a", [0])

    assert calls, "an active-layer-only commit must still resync the flipped arrangement"


def test_locked_layer_refuses_commit_no_partial_write(qtbot, win):
    layer = win.layer
    palette = win._group_palette
    palette.new_group("fault_a")
    palette.add_pick((layer.layer_id, 0), shift=False)

    win._on_lock_toggled(layer.layer_id, True)
    before_tags = dict(layer.tags)
    before_chain = layer.chain

    win._commit_button.click()

    assert layer.chain is before_chain                          # no chain step added
    assert layer.tags == before_tags                             # no tags write
    assert "locked" in win.strips.reading_label.text()


def test_param_change_causing_stale_groups_surfaces_a_warning(qtbot, win):
    layer = win.layer
    _commit(win, layer.layer_id, "fault_a", [0, 1, 2])           # commit against all 3 chains

    # Shrink the transform's own chain count below the committed max index (2) -> group_paint's
    # structural out-of-range check flags "fault_a" stale on the next resolve.
    i = win._names.index("chain_stub")
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win._on_param_changed(i, "n_chains", 1)

    assert "fault_a" in win.strips.reading_label.text()
    result = resolve(layer, win.field, win.cache, source_id=layer.source_id).result
    assert result.get("_stale_groups") == ["fault_a"]


def test_malformed_spec_json_surfaces_a_warning_not_a_crash(qtbot, win):
    layer = win.layer
    _commit(win, layer.layer_id, "fault_a", [0])

    i = win._names.index("group_paint")
    win._params[i]["spec_json"] = "{not valid json"
    layer.chain = win._chain()

    win._reresolve()                                             # a filter-only edit: no worker

    assert "group spec error" in win.strips.reading_label.text()


def test_save_load_round_trips_tags_and_reresolve_repaints(qtbot, win, tmp_path):
    layer = win.layer
    _commit(win, layer.layer_id, "fault_a", [0, 1])
    saved_groups = layer.tags["groups"]

    proj_path = win._save_project_to(tmp_path / "proj.dynamix")

    from dynamix.shell.main_window import MainWindow

    win2 = MainWindow(steps=())
    qtbot.addWidget(win2)
    with qtbot.waitSignal(win2.resolved, timeout=10000):
        win2._open_project_path(proj_path)

    layer2 = win2.layer
    assert layer2.tags["groups"] == saved_groups
    assert any(s.device == "group_paint" for s in layer2.chain.steps)

    color = tuple(GROUP_COLORS[0])
    item = win2.canvas._group_items[color]
    gx, gy = item.getData()
    assert gx.size > 0                                           # re-resolve re-painted it


# ---------------------------------------------------------------------------------- canvas split
#
# Extends tests/test_shell_canvas.py's own Task-4/5 `_flagged_result` assertions to the `classify_chains`/grouped-trail-item machinery. Self-contained (no window, no pyvista Scene).


def _synthetic_result(shape=(64, 64)):
    ext0 = {
        "x": np.array([10, 11, 12, 13], dtype=np.int64), "y": np.full(4, 5, dtype=np.int64),
        "mod": np.array([1.0, 9.0, 2.0, 8.0]), "arg": np.zeros(4),
        "line_id": np.array([0, 0, 0, 0], dtype=np.int64),
    }
    return {"extrema": [ext0], "chains": [], "scales": np.array([1.0]),
            "_shape": shape, "params": {}}


def _chain_at(x, y, tags=None, group_color=None):
    c = {"x": np.array([x], dtype=np.int64), "y": np.array([y], dtype=np.int64),
         "mod": np.array([1.0])}
    if tags is not None:
        c["tags"] = list(tags)
    if group_color is not None:
        c["group_color"] = list(group_color)
    return c


def test_group_tagged_chain_renders_on_a_group_color_item_not_seam_or_plain(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    plain = _chain_at(11, 5)
    grouped = _chain_at(13, 5, tags=["group:fault_a"], group_color=(230, 159, 0))
    result = _synthetic_result()
    result["chains"] = [plain, grouped]

    canvas.set_result(result, 0)

    item = canvas._group_items[(230, 159, 0)]
    gx, gy = item.getData()
    assert gx.tolist() == [13.0] and gy.tolist() == [5.0]
    vx, _ = canvas.vtrail_item.getData()
    assert vx.tolist() == [11.0]                                 # only the plain chain
    sx, _ = canvas.seam_item.getData()
    assert sx is None or sx.size == 0                            # nothing on seam_item


def test_chain_with_both_seam_and_group_tags_renders_group_color_not_seam(qtbot):
    """User assertion outranks the machine flag (main_window.py's commit docstring; canvas.py's
    `classify_chains` docstring) -- a chain tagged BOTH ways draws grouped, never seam."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    both = _chain_at(20, 7, tags=["seam_step", "group:fault_a"], group_color=(0, 114, 178))
    result = _synthetic_result()
    result["chains"] = [both]

    canvas.set_result(result, 0)

    item = canvas._group_items[(0, 114, 178)]
    gx, _ = item.getData()
    assert gx.tolist() == [20.0]
    sx, _ = canvas.seam_item.getData()
    assert sx is None or sx.size == 0


def test_group_item_persists_and_clears_when_a_later_result_has_no_groups(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    grouped = _chain_at(13, 5, tags=["group:fault_a"], group_color=(230, 159, 0))
    result = _synthetic_result()
    result["chains"] = [grouped]
    canvas.set_result(result, 0)
    assert canvas._group_items[(230, 159, 0)].getData()[0].size == 1

    canvas.set_result(_synthetic_result(), 0)                    # no chains at all this time

    gx, gy = canvas._group_items[(230, 159, 0)].getData()
    assert gx is None or gx.size == 0                            # cleared, item itself kept


def test_clear_overlays_empties_group_items_too(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    grouped = _chain_at(13, 5, tags=["group:fault_a"], group_color=(230, 159, 0))
    result = _synthetic_result()
    result["chains"] = [grouped]
    canvas.set_result(result, 0)

    canvas.clear_overlays()

    gx, gy = canvas._group_items[(230, 159, 0)].getData()
    assert gx is None or gx.size == 0
