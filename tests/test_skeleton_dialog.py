# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.skeleton_dialog.SkeletonDialog -- the WTMM skeleton log-log plot.

Pure Qt + matplotlib, exercised entirely offscreen (``QT_QPA_PLATFORM=offscreen``, the harness's
own mandated gate) -- ``FigureCanvasQTAgg`` renders through the Agg raster backend regardless of
platform; this suite's own construction tests build a real one under exactly that gate. Chains are
synthetic, generated locally (real ``log2_scales``/``log2_mod`` arrays, varying length and slope --
this dialog only ever reads that chain-dict contract) rather than loaded from
``tests/fixtures/kam_64.npz``, matching ``tests/test_chain_stats.py``'s own ``_chain`` helper one
level up the stack.
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


# ------------------------------------------------------------------------------ line collection


def test_line_collection_segment_count_is_capped():
    chains = _chains(n_chains=MAX_DRAWN + 50, n_pts=3)
    dialog = SkeletonDialog(chains, None)

    assert len(dialog._line_collection.get_segments()) == MAX_DRAWN


def test_line_collection_segment_count_matches_input_under_the_cap():
    chains = _chains(n_chains=17)
    dialog = SkeletonDialog(chains, None)

    assert len(dialog._line_collection.get_segments()) == 17


def test_line_collection_color_array_is_the_cached_ols_stats():
    chains = _chains(n_chains=12)
    dialog = SkeletonDialog(chains, None)

    expected = chain_stats.stats_for(chains, "ols")
    np.testing.assert_allclose(dialog._line_collection.get_array(), expected)


def test_line_collection_clim_is_the_2_98_percentile_of_the_drawn_h():
    chains = _chains(n_chains=40)
    dialog = SkeletonDialog(chains, None)

    h = chain_stats.stats_for(chains, "ols")
    finite = h[np.isfinite(h)]
    expected = tuple(np.percentile(finite, [2, 98]))
    assert dialog._line_collection.get_clim() == pytest.approx(expected)


# ------------------------------------------------------------------------------------ histogram


def test_histogram_axis_is_populated_with_50_bins_over_minus3_to_2():
    dialog = SkeletonDialog(_chains(n_chains=20), None)

    patches = dialog._ax_hist.patches
    assert len(patches) == 50
    xs = [p.get_x() for p in patches]
    assert min(xs) == pytest.approx(-3.0)
    assert max(xs) + patches[0].get_width() == pytest.approx(2.0)


# --------------------------------------------------------------------------------- px_size offset


def test_px_size_offsets_the_x_axis_by_log2_px_size():
    chain = _chain(5, slope=-1.0, seed=3)
    dialog_bare = SkeletonDialog([chain], None)
    dialog_physical = SkeletonDialog([chain], 4.0)   # 4 physical units per pixel

    x_bare = dialog_bare._line_collection.get_segments()[0][:, 0]
    x_phys = dialog_physical._line_collection.get_segments()[0][:, 0]
    np.testing.assert_allclose(x_phys - x_bare, np.log2(4.0))


def test_no_px_size_leaves_x_in_bare_scale_units():
    chain = _chain(5, slope=-1.0, seed=3)
    dialog = SkeletonDialog([chain], None)

    x = dialog._line_collection.get_segments()[0][:, 0]
    np.testing.assert_allclose(x, chain["log2_scales"])


# ----------------------------------------------------------------------------------- normalize


def test_normalize_checkbox_reanchors_every_chain_at_the_origin(qtbot):
    dialog = SkeletonDialog(_chains(n_chains=8), None)
    qtbot.addWidget(dialog)

    dialog._normalize_check.setChecked(True)

    for seg in dialog._line_collection.get_segments():
        if seg.shape[0] == 0:
            continue
        assert seg[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert seg[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_normalize_updates_axis_labels():
    dialog = SkeletonDialog(_chains(n_chains=5), None)
    assert dialog._ax_main.get_xlabel() == "log₂ scale"

    dialog._normalize_check.setChecked(True)

    assert "Δ" in dialog._ax_main.get_xlabel()
    assert "finest" in dialog._ax_main.get_ylabel()


def test_normalize_toggle_off_restores_the_original_geometry():
    chains = _chains(n_chains=5)
    dialog = SkeletonDialog(chains, None)
    before = [seg.copy() for seg in dialog._line_collection.get_segments()]

    dialog._normalize_check.setChecked(True)
    dialog._normalize_check.setChecked(False)

    after = dialog._line_collection.get_segments()
    for b, a in zip(before, after):
        np.testing.assert_allclose(a, b)


def test_normalize_does_not_change_the_cached_h_or_the_colorbar_mapping():
    """A pure translation, slopes unchanged -- so the colors already mapped must not move
    when the checkbox toggles."""
    dialog = SkeletonDialog(_chains(n_chains=9), None)
    before_colors = np.array(dialog._line_collection.get_array())
    before_clim = dialog._line_collection.get_clim()

    dialog._normalize_check.setChecked(True)

    np.testing.assert_allclose(dialog._line_collection.get_array(), before_colors, equal_nan=True)
    assert dialog._line_collection.get_clim() == before_clim


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
    assert len(dialog._highlight_collection.get_segments()) == 3


def test_set_selection_empty_list_clears_the_highlight():
    dialog = SkeletonDialog(_chains(n_chains=6), None)
    dialog.set_selection([0, 1])

    dialog.set_selection([])

    assert len(dialog._highlight_collection.get_segments()) == 0


def test_set_selection_ignores_indices_beyond_the_drawn_cap():
    chains = _chains(n_chains=MAX_DRAWN + 10, n_pts=3)
    dialog = SkeletonDialog(chains, None)

    dialog.set_selection([MAX_DRAWN + 1, 0])

    assert len(dialog._highlight_collection.get_segments()) == 1


def test_set_selection_geometry_follows_current_normalize_state():
    dialog = SkeletonDialog(_chains(n_chains=6), None)
    dialog._normalize_check.setChecked(True)

    dialog.set_selection([2])

    seg = dialog._highlight_collection.get_segments()[0]
    assert seg[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert seg[0, 1] == pytest.approx(0.0, abs=1e-9)
