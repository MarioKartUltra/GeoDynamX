# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.skeleton_dialog.SkeletonDialog -- the WTMM skeleton log-log plot.

Pure Qt + pyqtgraph, exercised entirely offscreen (``QT_QPA_PLATFORM=offscreen``, the harness's
own mandated gate). Chains are synthetic, generated locally (real ``log2_scales``/``log2_mod``
arrays, varying length and slope -- this dialog only ever reads that chain-dict contract) rather
than loaded from ``tests/fixtures/kam_64.npz``, matching ``tests/test_chain_stats.py``'s own
``_chain`` helper one level up the stack.

The drawn geometry is read back off the plot items: every chain is one run of a NaN-separated
``PlotDataItem`` (one item per colour bin), so ``_drawn_runs`` splits the items' data at the
separators and returns the polylines actually on screen.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import chain_stats
from dynamix.shell.skeleton_dialog import MAX_DRAWN, SkeletonDialog


def _chain(n, slope, seed=0):
    """A synthetic chain dict: ``log2_scales = 0..n-1`` (index 0 = finest, CW§1's own
    convention), ``log2_mod = slope * log2_scales + noise`` -- enough to exercise a real OLS fit
    (R² < 1, not a perfectly straight line) with a fixed seed per chain so a test's expected
    numbers never drift between runs."""
    rng = np.random.default_rng(seed)
    log2_scales = np.arange(n, dtype=np.float64)
    log2_mod = slope * log2_scales + rng.normal(scale=0.02, size=n)
    return {"log2_scales": log2_scales, "log2_mod": log2_mod, "mod": 2.0 ** log2_mod,
            "x": np.zeros(n, dtype=np.int64), "y": np.zeros(n, dtype=np.int64)}


def _chains(n_chains=30, n_pts=6, seed=0):
    rng = np.random.default_rng(seed)
    return [_chain(n_pts, float(rng.uniform(-1.5, 0.5)), seed=1000 + i)
            for i in range(n_chains)]


def _runs_of(item) -> list[np.ndarray]:
    """The polylines a NaN-separated ``PlotDataItem`` draws, each an ``(n, 2)`` array."""
    x, y = item.getData()
    if x is None or len(x) == 0:
        return []
    xy = np.column_stack([x, y])
    breaks = np.flatnonzero(~np.isfinite(xy[:, 0]))
    return [run for run in np.split(xy, breaks) if run.size
            for run in [run[np.isfinite(run[:, 0])]] if run.shape[0]]


def _drawn_runs(dialog) -> list[np.ndarray]:
    return [run for item in dialog._bin_items for run in _runs_of(item)]


# ------------------------------------------------------------------------------- construction


def test_dialog_builds_headless_from_synthetic_chains(qtbot):
    dialog = SkeletonDialog(_chains(), None)
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "WTMM skeleton — log₂ scale vs log₂|W|"
    assert dialog.isModal() is False


def test_title_notes_the_cap_when_over_max_drawn(qtbot):
    chains = _chains(n_chains=MAX_DRAWN + 5, n_pts=4)
    dialog = SkeletonDialog(chains, None)
    qtbot.addWidget(dialog)

    assert f"[capped from {MAX_DRAWN + 5}]" in dialog.windowTitle()


def test_title_has_no_cap_note_under_the_cap(qtbot):
    dialog = SkeletonDialog(_chains(n_chains=10), None)
    qtbot.addWidget(dialog)

    assert "capped" not in dialog.windowTitle()


def test_the_dialog_fits_the_window_height_rule(qtbot):
    dialog = SkeletonDialog(_chains(), None)
    qtbot.addWidget(dialog)

    assert dialog.height() <= 800


# ------------------------------------------------------------------------------ drawn chains


def test_drawn_polyline_count_is_capped():
    chains = _chains(n_chains=MAX_DRAWN + 50, n_pts=3)
    dialog = SkeletonDialog(chains, None)

    assert len(_drawn_runs(dialog)) == MAX_DRAWN


def test_drawn_polyline_count_matches_input_under_the_cap():
    chains = _chains(n_chains=17)
    dialog = SkeletonDialog(chains, None)

    assert len(_drawn_runs(dialog)) == 17


def test_colour_bins_come_from_the_cached_ols_stats_and_the_bar_levels():
    chains = _chains(n_chains=12)
    dialog = SkeletonDialog(chains, None)

    h = chain_stats.stats_for(chains, "ols")
    lo, hi = dialog._colour_bar.levels()
    n_bins = len(dialog._bin_items)
    expected = np.rint(np.clip((h - lo) / (hi - lo), 0.0, 1.0) * (n_bins - 1)).astype(int)
    np.testing.assert_array_equal(dialog._bin_of, expected)
    for b, item in enumerate(dialog._bin_items):           # every chain draws in its own bin
        assert len(_runs_of(item)) == int(np.sum(expected == b))


def test_colour_bar_levels_are_the_2_98_percentile_of_the_drawn_h():
    chains = _chains(n_chains=40)
    dialog = SkeletonDialog(chains, None)

    h = chain_stats.stats_for(chains, "ols")
    finite = h[np.isfinite(h)]
    expected = tuple(np.percentile(finite, [2, 98]))
    assert dialog._colour_bar.levels() == pytest.approx(expected)


def test_colour_bar_is_labelled_with_the_slope():
    dialog = SkeletonDialog(_chains(n_chains=5), None)

    assert dialog._colour_bar.getAxis("left").labelText == "Hölder slope h"


# ------------------------------------------------------------------------------------ histogram


def test_histogram_has_50_bins_over_minus3_to_2():
    dialog = SkeletonDialog(_chains(n_chains=20), None)

    edges, counts = dialog._hist_item.getData()
    assert len(counts) == 50 and len(edges) == 51
    assert edges[0] == pytest.approx(-3.0)
    assert edges[-1] == pytest.approx(2.0)
    h = chain_stats.stats_for(_chains(n_chains=20), "ols")
    assert int(counts.sum()) == int(np.isfinite(h).sum())


# --------------------------------------------------------------------------------- px_size offset


def test_px_size_offsets_the_x_axis_by_log2_px_size():
    chain = _chain(5, slope=-1.0, seed=3)
    dialog_bare = SkeletonDialog([chain], None)
    dialog_physical = SkeletonDialog([chain], 4.0)   # 4 physical units per pixel

    x_bare = _drawn_runs(dialog_bare)[0][:, 0]
    x_phys = _drawn_runs(dialog_physical)[0][:, 0]
    np.testing.assert_allclose(x_phys - x_bare, np.log2(4.0))


def test_no_px_size_leaves_x_in_bare_scale_units():
    chain = _chain(5, slope=-1.0, seed=3)
    dialog = SkeletonDialog([chain], None)

    x = _drawn_runs(dialog)[0][:, 0]
    np.testing.assert_allclose(x, chain["log2_scales"])


# ----------------------------------------------------------------------------------- normalize


def test_normalize_checkbox_reanchors_every_chain_at_the_origin(qtbot):
    dialog = SkeletonDialog(_chains(n_chains=8), None)
    qtbot.addWidget(dialog)

    dialog._normalize_check.setChecked(True)

    runs = _drawn_runs(dialog)
    assert len(runs) == 8
    for run in runs:
        assert run[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert run[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_normalize_updates_axis_labels():
    dialog = SkeletonDialog(_chains(n_chains=5), None)
    assert dialog._main.getAxis("bottom").labelText == "log₂ scale"

    dialog._normalize_check.setChecked(True)

    assert "Δ" in dialog._main.getAxis("bottom").labelText
    assert "finest" in dialog._main.getAxis("left").labelText


def test_normalize_toggle_off_restores_the_original_geometry():
    chains = _chains(n_chains=5)
    dialog = SkeletonDialog(chains, None)
    before = [run.copy() for run in _drawn_runs(dialog)]

    dialog._normalize_check.setChecked(True)
    dialog._normalize_check.setChecked(False)

    after = _drawn_runs(dialog)
    assert len(after) == len(before) == 5
    for b, a in zip(before, after):
        np.testing.assert_allclose(a, b)


def test_normalize_does_not_change_the_cached_h_or_the_colour_mapping():
    """A pure translation, slopes unchanged -- so the bins already mapped must not move when
    the checkbox toggles."""
    dialog = SkeletonDialog(_chains(n_chains=9), None)
    h_before = dialog._h
    bins_before = dialog._bin_of.copy()
    levels_before = dialog._colour_bar.levels()

    dialog._normalize_check.setChecked(True)

    assert dialog._h is h_before
    np.testing.assert_array_equal(dialog._bin_of, bins_before)
    assert dialog._colour_bar.levels() == levels_before


# ------------------------------------------------------------------------------- h-range spins


def test_h_range_spin_defaults_are_the_finite_min_max_of_computed_slopes():
    chains = _chains(n_chains=25)
    dialog = SkeletonDialog(chains, None)

    h = chain_stats.stats_for(chains, "ols")
    finite = h[np.isfinite(h)]
    assert dialog._h_lo_spin.value() == pytest.approx(float(finite.min()), abs=1e-3)
    assert dialog._h_hi_spin.value() == pytest.approx(float(finite.max()), abs=1e-3)


def test_h_range_spin_widget_range_is_padded_by_one():
    chains = _chains(n_chains=25)
    dialog = SkeletonDialog(chains, None)

    h = chain_stats.stats_for(chains, "ols")
    finite = h[np.isfinite(h)]
    assert dialog._h_lo_spin.minimum() == pytest.approx(float(finite.min()) - 1.0, abs=1e-3)
    assert dialog._h_hi_spin.maximum() == pytest.approx(float(finite.max()) + 1.0, abs=1e-3)


def test_h_range_defaults_fall_back_when_no_chain_has_two_points():
    """Every chain has a single point -- ``chain_ols_holder`` (the ``stats_for("ols")`` estimator)
    is NaN for all of them (n < 2), so there is no real finite min/max to default to. The dialog
    must still construct -- a documented, defined (-1.0, 1.0) fallback, never a crash."""
    chains = [_chain(1, slope=0.0, seed=i) for i in range(4)]
    dialog = SkeletonDialog(chains, None)

    assert dialog._h_lo_spin.value() == pytest.approx(-1.0)
    assert dialog._h_hi_spin.value() == pytest.approx(1.0)
    assert dialog._colour_bar.levels() == (-1.0, 1.0)


# ------------------------------------------------------------------------------- select h-range


def test_select_h_range_emits_the_matching_indices(qtbot):
    chains = [
        _chain(6, slope=-2.0, seed=1),
        _chain(6, slope=-1.0, seed=2),
        _chain(6, slope=0.0, seed=3),
        _chain(6, slope=0.3, seed=4),
    ]
    dialog = SkeletonDialog(chains, None)
    qtbot.addWidget(dialog)
    h = chain_stats.stats_for(chains, "ols")
    lo, hi = float(h[1]) - 0.05, float(h[2]) + 0.05     # brackets indices 1 and 2 only

    received = []
    dialog.selectionRequested.connect(received.append)
    dialog._h_lo_spin.setValue(lo)
    dialog._h_hi_spin.setValue(hi)
    dialog._select_button.click()

    assert received == [[1, 2]]


def test_select_h_range_tolerates_an_inverted_spin_pair(qtbot):
    """Both values stay inside the padded widget range (so ``setValue`` cannot silently clamp
    them into agreement) but are assigned to the WRONG spin -- the numerically higher value goes
    into the "h ≥" box, the lower into "h ≤" -- and the emitted selection still brackets only the
    chain actually between them once sorted."""
    chains = [_chain(6, slope=-2.0, seed=1), _chain(6, slope=0.3, seed=4)]
    dialog = SkeletonDialog(chains, None)
    qtbot.addWidget(dialog)
    h = chain_stats.stats_for(chains, "ols")

    received = []
    dialog.selectionRequested.connect(received.append)
    dialog._h_lo_spin.setValue(float(h[0]) + 0.05)   # deliberately the WRONG way round
    dialog._h_hi_spin.setValue(float(h[0]) - 0.05)
    dialog._select_button.click()

    assert received == [[0]]


# ------------------------------------------------------------------------------- set_selection


def test_set_selection_draws_a_highlight_without_recomputing_stats():
    dialog = SkeletonDialog(_chains(n_chains=10), None)
    stats_before = dialog._h

    dialog.set_selection([1, 3, 5])

    assert dialog._h is stats_before                     # identity -- never recomputed
    assert len(_runs_of(dialog._highlight_item)) == 3


def test_set_selection_empty_list_clears_the_highlight():
    dialog = SkeletonDialog(_chains(n_chains=6), None)
    dialog.set_selection([0, 1])

    dialog.set_selection([])

    assert len(_runs_of(dialog._highlight_item)) == 0


def test_set_selection_ignores_indices_beyond_the_drawn_cap():
    chains = _chains(n_chains=MAX_DRAWN + 10, n_pts=3)
    dialog = SkeletonDialog(chains, None)

    dialog.set_selection([MAX_DRAWN + 1, 0])

    assert len(_runs_of(dialog._highlight_item)) == 1


def test_set_selection_geometry_follows_current_normalize_state():
    dialog = SkeletonDialog(_chains(n_chains=6), None)
    dialog._normalize_check.setChecked(True)

    dialog.set_selection([2])

    run = _runs_of(dialog._highlight_item)[0]
    assert run[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert run[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_the_highlight_wears_the_selection_accent():
    from dynamix.shell.theme import current
    dialog = SkeletonDialog(_chains(n_chains=4), None)

    assert dialog._highlight_item.opts["pen"].color().name() == current().selection_accent


def test_a_light_ground_never_takes_a_white_centred_map():
    """A diverging map's centre must read against the plot ground: the coolwarm family's white
    centre on a dark ground, a grey-centred blue–red on a light one."""
    import numpy as np
    import pyqtgraph as pg
    from dynamix.shell.skeleton_dialog import _colour_map
    from dynamix.shell.theme import RESTRAINED_DARK, contrast, load_theme

    for theme in (RESTRAINED_DARK, load_theme("light"), load_theme("mid_grey")):
        centre = pg.colormap.get(_colour_map(theme)).map(np.array([0.5]), mode="byte")[0, :3]
        assert contrast(theme.ground, "#%02x%02x%02x" % tuple(centre)) >= 3.0, theme.name
