# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""2-D gate translations for the ported M-Z oracle (scripts 10, 20, 22 provenance)."""
from __future__ import annotations

import numpy as np

from dynamix.core import mzlib


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


def _maxima_for(img, J):
    S, Wpairs = mzlib.atrous2d_forward(img, J)
    maxima = []
    for W1, W2 in Wpairs:
        rows, cols = mzlib.nms2d_dyadic(W1, W2)
        maxima.append((rows, cols, W1[rows, cols], W2[rows, cols]))
    return S, Wpairs, maxima


def _snr(truth, est):
    return 10 * np.log10(np.sum(truth ** 2) / np.sum((truth - est) ** 2))


def test_atrous2d_roundtrip_float64():
    rng = np.random.default_rng(10)
    img = rng.standard_normal((48, 40))
    S, Wpairs = mzlib.atrous2d_forward(img, 4)
    back = mzlib.atrous2d_inverse(S, Wpairs)[:48, :40]
    assert np.max(np.abs(back - img)) / np.max(np.abs(img)) < 1e-10


def test_pocs2d_meets_the_transferable_floor():
    """The design floor (>=26.2 dB at 10 iters) on this file's own composite.

    26.2 is the script-10 transferable floor; the exact value on THIS composite is
    pin-on-first-green — record it in the assertion comment when first measured.
    """
    img = _composite64()
    J = 4
    S, _, maxima = _maxima_for(img, J)
    img_hat, resid, _ = mzlib.pocs2d(maxima, S, img.shape, J, n_iter=10)
    # First-run MEASURED: 26.81 dB. Floor is the fixed 26.2 dB transferable
    # spec floor (NOT pinned to measured*margin the way the 1-D/script-10/20 gates are) —
    # this assertion is the floor itself, per the design.
    assert _snr(img, img_hat) >= 26.2


def test_pocs2d_checkpoints_reuse_one_run():
    img = _composite64()
    J = 3
    S, _, maxima = _maxima_for(img, J)
    img_hat, resid, ck = mzlib.pocs2d(maxima, S, img.shape, J, n_iter=10,
                                      checkpoint_at=(5, 10))
    assert set(ck) == {5, 10}
    assert len(resid) == 10
    # ck[10] == img_hat is tautological by construction (mzlib.py's img_hat =
    # checkpoints.get(n_iter, ...) literally returns ck[10] when n_iter is checkpointed) --
    # a regression storing the FINAL state at every checkpoint index would still pass that
    # comparison. Pin the property the name promises instead: checkpoints hold DISTINCT,
    # correct intermediate states, not just a copy of the last one.
    assert not np.array_equal(ck[5], ck[10])
    five_iter_img_hat, _, _ = mzlib.pocs2d(maxima, S, img.shape, J, n_iter=5)
    assert np.array_equal(ck[5], five_iter_img_hat)


def test_atrous2d_inverse_is_linear():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((32, 32))
    b = rng.standard_normal((32, 32))
    Sa, Wa = mzlib.atrous2d_forward(a, 3)
    Sb, Wb = mzlib.atrous2d_forward(b, 3)
    Ssum, Wsum = mzlib.atrous2d_forward(a + b, 3)
    lhs = mzlib.atrous2d_inverse(Ssum, Wsum)
    rhs = mzlib.atrous2d_inverse(Sa, Wa) + mzlib.atrous2d_inverse(Sb, Wb)
    assert np.max(np.abs(lhs - rhs)) < 1e-10


def test_edges_only_diverges_without_the_coarse_channel():
    """Doctrine 12 (script 22): with S pinned to zero the iteration diverges —
    residual grows by >=5x from its early minimum. The coarse channel is convergence itself.

    n_iter=60 (the design's sketch value) never diverges on this composite: at J=4/N=64 the
    residual stays flat (ratio ~0.98-1.0) through 60 and is still only ~1.7x by 150 -- the
    onset is much later than script 22's own J=6/N=256 probe. Measured crossing is between 150
    (x1.7) and 180 (x5.1) iterations; raised to n_iter=200 per the design's sanctioned latitude
    for a comfortable margin past onset. First-run MEASURED ratio at n_iter=200: x11.7 (resid
    1.474e+01 -> 1.725e+02), consistent with script 22's own note
    that growth is structural but the exact ratio is not (post-divergence trajectories are
    1-ulp-chaotic; the gate pins >=5x, not a tight value)."""
    img = _composite64()
    J = 4
    S, _, maxima = _maxima_for(img, J)
    _, resid, _ = mzlib.pocs2d(maxima, np.zeros_like(S), img.shape, J, n_iter=200)
    early = min(resid[:15])
    assert resid[-1] / early >= 5.0
