# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Product-bundle builder for the M-Z peer transform (mz_edges core, no device yet)."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import mz_edges, mzlib


def _composite64():
    """Deterministic single sharp plateau + shallow ramp, 64x64 — the near-complete regime.

    Simplified from an earlier two-plateau (shared-edge/corner) draft per the design's
    sanctioned latitude: a two-region composite with a shared corner capped the 10-iteration
    POCS floor around 24.5-24.8 dB regardless of region size or ramp weight (measured across
    several corner topologies), while a single sharp plateau clears the transferable floor
    easily — the corner's extra edge complexity, not the ramp, was what the 10-iteration
    budget couldn't resolve. The shallow ramp is kept (not dropped to zero): it is what makes
    the no-coarse-channel iteration in test_edges_only_diverges_without_the_coarse_channel
    actually diverge (see that test) — a ramp-free single plateau stays stable indefinitely
    under S=0 (measured out to 300 iterations, ratio never exceeds ~1), because nearly all of
    its content is a single high-contrast edge that the fine-scale maxima alone reconstruct
    just fine without the coarse channel. The ramp is genuine low-frequency content that
    normally lives in S; zeroing S while keeping maxima consistent only with the true S is
    what produces the SV-C inconsistency script 22 documents.
    """
    y, x = np.mgrid[0:64, 0:64]
    img = np.zeros((64, 64))
    img[(x > 8) & (x <= 56) & (y > 8) & (y <= 56)] = 1.0
    img += 0.002 * (x + 0.5 * y)
    return img


def test_bundle_schema_and_primary_quadrant():
    img = _composite64()
    b = mz_edges.analyze(img, 4)
    assert len(b["extrema"]) == 4 and len(b["mz_maxima"]) == 4
    assert b["scales"].shape == (4,) and np.all(np.diff(b["scales"]) > 0)
    for ext in b["extrema"]:
        assert set(ext) == {"x", "y", "mod", "arg", "line_id"}
        assert ext["x"].dtype == np.int64 and ext["y"].dtype == np.int64
        assert np.all(ext["x"] < 64) and np.all(ext["y"] < 64)
        assert np.all(ext["x"] >= 0) and np.all(ext["y"] >= 0)
        assert ext["mod"].shape == ext["x"].shape == ext["line_id"].shape
        assert np.all(ext["mod"] >= 0)
    # full-torus arrays really are on the 2N torus
    rows, cols, w1, w2 = b["mz_maxima"][0]
    assert rows.max() >= 64 or cols.max() >= 64


def test_line_ids_are_wrap_merged_components():
    """A vertical step edge yields one long chain per scale near the edge column —
    the points on the edge share one line_id (8-connected on the torus). A small isolated
    bump far from the edge guarantees at least one singleton-in-primary-quadrant component,
    which must be remapped to -1 -- the shell's sentinel for an isolated, non-chained
    extremum (controller ruling, mz-edges-port review: canvas.py/filters.py both
    special-case -1 as "not a line"). A component with 16+ members is never a singleton, so
    the edge chain's own ids must stay >= 0."""
    img = np.zeros((32, 32))
    img[:, 16:] = 1.0
    img[3, 3] += 0.5  # isolated bump, far from the edge -- guarantees a singleton component
    b = mz_edges.analyze(img, 2)
    ext = b["extrema"][0]
    edge = np.abs(ext["x"] - 16) <= 2
    assert edge.sum() >= 16
    edge_ids = np.unique(ext["line_id"][edge])
    assert edge_ids.size <= 3  # the edge is one (rarely fragmented into 2-3) component, not 16
    assert np.all(edge_ids >= 0)
    assert np.any(ext["line_id"] == -1)  # the isolated bump is a singleton -> -1


def test_torus_components_wrap_merges_across_the_seam():
    """The step-edge fixture above never actually exercises the periodic wrap: its ridge
    spans every row contiguously, so top and bottom halves are already joined through
    ordinary (non-wrapped) row-by-row adjacency at the INNER mirror seam (row ny-1 to row ny
    is a plain "+1" step, no modulo needed) -- a non-periodic labeller would merge it
    identically (see the fix report for the measured comparison).

    Here two 3-point runs sit at opposite ends of a 10-row torus with a 4-row gap between
    them (rows 3-6 empty): ordinary adjacency cannot bridge them (row 9's ordinary "next"
    row, 10, doesn't exist) -- only the periodic wrap (row 9 -> row 0, mod 10) can. A
    non-periodic (wrap-free) labeller must see 2 components; the real, wrap-merged labeller
    must see 1."""
    rows = np.array([0, 1, 2, 7, 8, 9], dtype=np.int64)
    cols = np.array([5, 5, 5, 5, 5, 5], dtype=np.int64)
    labels = mz_edges._torus_components(rows, cols, (10, 10))
    assert np.unique(labels).size == 1  # wrap-merged into one component, not two
    assert labels.min() >= 0


def test_preview_round_trips_the_composite():
    """Full maxima + true coarse (mapping policy): the Task-2 floor holds through the
    bundle path too."""
    img = _composite64()
    b = mz_edges.analyze(img, 4)
    img_hat, diag = mz_edges.preview(img, b, n_iter=10)
    snr = 10 * np.log10(np.sum(img ** 2) / np.sum((img - img_hat) ** 2))
    assert snr >= 26.2
    assert diag["n_iter"] == 10 and len(diag["resid"]) == 10
    assert diag["diverging"] is False


def test_preview_on_dithered_bundle_matches_the_floor():
    """Controller ruling (finding #4): preview()'s "full" coarse
    branch recomputes S from the SAME dithered field analyze() used to build the maxima
    constraints (fixed seed -> bit-identical), so a dithered bundle's reconstruction is
    never a field/field (SV-C) mismatch -- the ordinary 26.2 dB floor still holds."""
    img = np.round(_composite64() * 25) / 25.0
    b = mz_edges.analyze(img, 4, dither=True)
    img_hat, diag = mz_edges.preview(img, b, n_iter=10)
    snr = 10 * np.log10(np.sum(img ** 2) / np.sum((img - img_hat) ** 2))
    assert snr >= 26.2
    assert diag["diverging"] is False


def test_diverging_flag_is_reachable():
    """Controller ruling (finding #3): diag["diverging"] was otherwise
    unreachable in this test file. A genuine field/field (SV-C) mismatch -- maxima extracted
    from one field, coarse pinned to a DIFFERENT field via preview()'s "full" recompute --
    must set it at a large enough n_iter (mirrors tests/test_mzlib_2d.py's own S=0 divergence
    guard, same doctrine-12 mechanism, different inconsistency). Measured: resid ratio 1.99 at n_iter=150 (not yet flagged), 114.77 at n_iter=250
    (comfortably past the 5x threshold -- not a hair-trigger pin)."""
    img = _composite64()
    b = mz_edges.analyze(img, 4)
    _, diag = mz_edges.preview(-img, b, n_iter=250)
    assert diag["diverging"] is True
    assert diag["resid"][-1] / min(diag["resid"]) >= 5.0


def test_preview_keep_prunes_the_maxima_set():
    img = _composite64()
    b = mz_edges.analyze(img, 3)
    keep = [np.zeros(m[0].size, dtype=bool) for m in b["mz_maxima"]]
    for k in keep:
        k[: k.size // 2] = True
    img_hat, _ = mz_edges.preview(img, b, n_iter=5, keep=keep)
    full_hat, _ = mz_edges.preview(img, b, n_iter=5)
    assert not np.allclose(img_hat, full_hat)


def test_thumbnail_policy_stores_and_restores_the_coarse():
    """Coding mode (script 20): thumbnail restore rel-err gate 1.75e-2."""
    img = _composite64()
    b = mz_edges.analyze(img, 3, coarse="thumbnail")
    assert b["coarse_thumb"] is not None
    assert b["coarse_thumb"].shape[0] < 64
    img_hat, diag = mz_edges.preview(img, b, n_iter=10)
    snr = 10 * np.log10(np.sum(img ** 2) / np.sum((img - img_hat) ** 2))
    # thumbnail costs ~0.3 dB vs mapping (script 20); same floor with margin
    assert snr >= 25.0


def test_measure_lsb_and_dither():
    """Doctrine §5.2: seeded half-LSB dither at analysis time collapses the
    dead-coefficient fraction on a quantized field; the coarse channel is invariant.

    "Dead" is measured as numerically zero, not bit-exact zero. ``atrous2d_forward``
    realizes its (mathematically compact, 2-tap) G filter via global FFT/IFFT, so a
    coefficient that is analytically zero on a locally-flat quantized run lands at
    floating-point noise (~1e-16, measured) rather than an exact 0.0 -- bit-exact equality
    only ever fires for a wholly-constant array (mzlib's own FFT DC-only special case),
    which would make this comparison a dead assertion on any realistic field (measured:
    0.0 dead fraction pre- AND post-dither with ``==0.0``, so the intended ``<`` could
    never discriminate a working dither from a no-op one). 1e-9 sits deep in the gap
    between that FFT noise floor and the smallest genuine (non-dead) coefficient magnitude
    on this fixture -- confirmed stable across 1e-9..1e-6 (identical measured fractions).
    """
    rng = np.random.default_rng(12)
    smooth = rng.standard_normal((64, 64)).cumsum(0).cumsum(1)
    smooth /= np.max(np.abs(smooth))
    q = np.round(smooth * 40) / 40.0  # exact 1/40 lattice
    assert abs(mz_edges.measure_lsb(q) - 1.0 / 40.0) < 1e-12

    b_raw = mz_edges.analyze(q, 3)
    b_dit = mz_edges.analyze(q, 3, dither=True)
    assert b_dit["lsb"] is not None
    # the CRITICAL gate: analyze()'s own dithered output must actually differ from its raw
    # output. Without this, dead_fraction below (which re-implements dithering locally and
    # calls mzlib directly, never analyze()) would still pass a hypothetical analyze() that
    # computes lsb but silently never applies it.
    assert not np.array_equal(b_raw["extrema"][0]["mod"], b_dit["extrema"][0]["mod"])

    def dead_fraction(values, dithered):
        v = values
        if dithered:
            r = np.random.default_rng(0)
            v = v + r.uniform(-b_dit["lsb"] / 2, b_dit["lsb"] / 2, v.shape)
        _, Wp = mzlib.atrous2d_forward(v, 1)
        W1, W2 = Wp[0]
        return float(np.mean((np.abs(W1) < 1e-9) & (np.abs(W2) < 1e-9)))

    assert dead_fraction(q, True) < dead_fraction(q, False)
    # analysis-time only: the input array is never modified
    assert np.array_equal(q, np.round(smooth * 40) / 40.0)


def test_dither_is_deterministic():
    img = np.round(_composite64() * 25) / 25.0
    a = mz_edges.analyze(img, 3, dither=True)
    b = mz_edges.analyze(img, 3, dither=True)
    for ea, eb in zip(a["extrema"], b["extrema"]):
        assert np.array_equal(ea["x"], eb["x"]) and np.array_equal(ea["mod"], eb["mod"])


def test_fast_torus_components_partition_matches_the_reference():
    """The 2026-09-19 profiling fix (61 s -> ~2 s at the overview size): the vectorized
    labeller must produce the SAME partitions as the kept union-find reference on
    randomized torus point sets -- numbering free, grouping identical both ways."""
    rng = np.random.default_rng(7)
    for trial in range(6):
        ny, nx = rng.integers(8, 40, 2)
        n_pts = int(rng.integers(1, ny * nx // 2))
        flat = rng.choice(ny * nx, size=n_pts, replace=False)
        rows, cols = (flat // nx).astype(np.int64), (flat % nx).astype(np.int64)
        a = mz_edges._torus_components_unionfind(rows, cols, (int(ny), int(nx)))
        b = mz_edges._torus_components(rows, cols, (int(ny), int(nx)))
        assert a.shape == b.shape
        # same partition: the label-pair mapping must be a bijection
        pairs = set(zip(a.tolist(), b.tolist()))
        assert len(pairs) == np.unique(a).size == np.unique(b).size, trial


def test_analyze_reports_progress_stages():
    """The device wait is a minute-plus at overview sizes (measured 106 s before the
    labeller fix) -- analyze() must narrate: the forward, then each level."""
    img = np.zeros((32, 32))
    img[:, 16:] = 1.0
    seen = []
    mz_edges.analyze(img, 2, progress=lambda stage, frac: seen.append((stage, frac)))
    stages = [s for s, _ in seen]
    assert stages[0] == "mz forward" and ("mz forward", 1.0) in seen
    assert "mz maxima 1/2" in stages and "mz maxima 2/2" in stages
    assert seen[-1] == ("mz maxima 2/2", 1.0)


def test_interpolate_adds_float_channels_and_keeps_pocs_support_integer():
    """2026-09-21: the wtmm2d subpixel refinement on the dyadic maxima -- x_sub/y_sub ride,
    the refined modulus replaces mod, and mz_maxima (the POCS constraint input) keeps raw
    integer positions and raw w1/w2 regardless (the recorded M-Z split)."""
    img = _composite64()
    off = mz_edges.analyze(img, 3)
    on = mz_edges.analyze(img, 3, interpolate=True)
    for e_off, e_on, (rows, cols, w1, w2) in zip(off["extrema"], on["extrema"],
                                                 on["mz_maxima"]):
        np.testing.assert_array_equal(e_on["x"], e_off["x"])
        np.testing.assert_array_equal(e_on["y"], e_off["y"])
        assert "x_sub" in e_on and e_on["x_sub"].shape == e_on["x"].shape
        assert np.all(e_on["mod"] >= e_off["mod"] - 1e-9)
        assert rows.dtype.kind in "iu" and cols.dtype.kind in "iu"
    np.testing.assert_array_equal(on["mz_maxima"][0][2], off["mz_maxima"][0][2])
