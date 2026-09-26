# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""End-to-end parity against the REAL xsmurf C, via XSmurfWrapper.

Complements the semantics-level pins (behavioral tests written from reading the C) with an
end-to-end check of whether the chaining results match xsmurf's: this file runs the SAME
derivative images through both stacks in-process and pins the agreement:

- ``xsmurf_wrapper.wtmm2d`` wraps ``Extract_Gradient_Maxima_2D`` (edge/extrema.c -- the exact
  Malandain NMS ``dynamix``'s ``_nms_extrema_scale`` reimplements);
- ``xsmurf_wrapper.chain``/``extract_vertical_chains`` drive the real ``vchain`` C
  (wt2d/chain2.c -- what ``chains2d`` transcribes).

Measured (fbm 96^2, H=0.6, seed 7, n_oct=2, n_voice=2, thresh=0):

- NMS: every interior position identical except knife-edge ties (<= 5 per scale, both
  directions combined); moduli agree to float32 (~1.2e-7 rel). Our extrema are a strict SUPERSET whose
  extras sit at border distance exactly 0 -- the C never tests the one-pixel frame, our
  vectorized NMS does (clamped probes). A documented convention difference, not drift.
- Chaining: every multi-scale chain identical in anchor, length, positions and moduli;
  the ONLY difference is singleton handling (``extract_vertical_chains`` drops length-1
  chains; our ``min_len=1`` keeps them) -- at the app's default ``min_len=2`` the chain sets
  are IDENTICAL.

Machine-specific by nature (needs the compiled XSmurfWrapper); skips cleanly when absent,
exactly like test_no_silent_drift skips without the EQSelect checkout.
"""
from __future__ import annotations

import numpy as np
import pytest

xw = pytest.importorskip("xsmurf_wrapper")

from conftest import fbm2d                                            # noqa: E402
from dynamix.core.wtmm_backend import (_cwt2d_numpy, _nms_extrema_scale,   # noqa: E402
                                       compute_scales2d, get_backend)

N = 96
#: Knife-edge ties per scale the NMS identity tolerates, BOTH directions combined (the
#: halo-oracle budget class: bilinear probe roundings can flip a >= comparison on a razor's
#: edge; measured 0-5 per scale on the fixture).
TIE_BUDGET = 8


@pytest.fixture(scope="module")
def stacks():
    field = fbm2d(N, 0.6, seed=7)
    scales = compute_scales2d(2, 2, 1.0)
    raw = _cwt2d_numpy(field, scales, derivs="first", verbose=False)
    ours_full, ours_interior, ximgs = [], [], []
    for si in range(len(scales)):
        dx = np.ascontiguousarray(raw["dx"][si], dtype=np.float32)
        dy = np.ascontiguousarray(raw["dy"][si], dtype=np.float32)
        e = _nms_extrema_scale(np.hypot(dx, dy).astype(np.float64),
                               np.arctan2(dy, dx).astype(np.float64), thresh=0.0)
        ours_full.append(e)
        keep = (e["x"] > 0) & (e["x"] < N - 1) & (e["y"] > 0) & (e["y"] < N - 1)
        ours_interior.append({k: np.asarray(v)[keep] for k, v in e.items()})
        ximgs.append(xw.wtmm2d(xw.XImage.from_numpy(dx), xw.XImage.from_numpy(dy),
                               float(scales[si]), thresh=0.0)[0])
    return {"scales": scales, "ours_full": ours_full, "ours_interior": ours_interior,
            "ximgs": ximgs}


def test_nms_matches_the_real_c_on_every_interior_pixel(stacks):
    for si, scale in enumerate(stacks["scales"]):
        theirs = stacks["ximgs"][si].get_extrema_arrays()
        b = set(zip(np.asarray(theirs["x"], int).tolist(),
                    np.asarray(theirs["y"], int).tolist()))
        e = stacks["ours_full"][si]
        a = set(zip(e["x"].tolist(), e["y"].tolist()))
        only_ours = np.array(sorted(a - b)) if a - b else np.zeros((0, 2), int)
        only_theirs = sorted(b - a)
        # Our extras are (i) the border frame the C never tests, plus (ii) a handful of
        # interior knife-edge ties (bilinear probe roundings flipping a >= comparison --
        # measured 2 in ~1000 at one scale, both directions combined under TIE_BUDGET).
        interior_extras = 0
        if only_ours.size:
            d = np.minimum.reduce([only_ours[:, 0], only_ours[:, 1],
                                   N - 1 - only_ours[:, 0], N - 1 - only_ours[:, 1]])
            interior_extras = int((d > 0).sum())
        assert interior_extras + len(only_theirs) <= TIE_BUDGET, \
            f"a={scale}: interior-ours={interior_extras}, only-theirs={only_theirs[:5]}"
        # moduli agree to float32 on the common set
        tm = {(int(x), int(y)): float(m) for x, y, m
              in zip(theirs["x"], theirs["y"], theirs["mod"])}
        om = {(int(x), int(y)): float(m) for x, y, m in zip(e["x"], e["y"], e["mod"])}
        common = set(tm) & set(om)
        assert common, f"a={scale}: no common extrema at all"
        dev = max(abs(tm[k] - om[k]) / max(abs(tm[k]), 1e-12) for k in common)
        assert dev < 5e-6, f"a={scale}: modulus rel deviation {dev}"


def _their_chains(stacks):
    xw.chain(stacks["ximgs"], 1.0, 2, 2, box_ratio=1.0, similitude=0.8,
             thresh=0.0, smooth=False, verbose=False)
    return xw.extract_vertical_chains(stacks["ximgs"],
                                      list(map(float, stacks["scales"])))


def test_chaining_matches_the_real_vchain_exactly_at_the_default_min_len(stacks):
    """At the app default ``min_len=2``, our chain set and the real vchain's are IDENTICAL:
    same anchors, same lengths, same per-point positions, moduli to float32."""
    theirs = [c for c in _their_chains(stacks) if len(c["x"]) >= 2]
    backend = get_backend("python")
    ours = backend.chains2d(stacks["ours_interior"], stacks["scales"], similitude=0.8,
                            box_ratio=1.0, dist2_max=50.0, min_len=2, smooth=False)
    key = lambda c: (int(c["x"][0]), int(c["y"][0]))
    A = {key(c): c for c in ours}
    B = {key(c): c for c in theirs}
    assert set(A) == set(B), (f"anchor sets differ: only-ours={len(set(A)-set(B))} "
                              f"only-theirs={len(set(B)-set(A))}")
    for k in A:
        ca, cb = A[k], B[k]
        np.testing.assert_array_equal(ca["x"], np.asarray(cb["x"], np.int64), err_msg=str(k))
        np.testing.assert_array_equal(ca["y"], np.asarray(cb["y"], np.int64), err_msg=str(k))
        np.testing.assert_allclose(ca["mod"], cb["mod"], rtol=1e-5, err_msg=str(k))


def test_the_only_min_len_1_difference_is_singletons(stacks):
    """``extract_vertical_chains`` drops length-1 chains; ``chains2d(min_len=1)`` keeps
    them -- the ONLY full-pipeline difference, pinned so it stays a convention, not drift."""
    theirs = _their_chains(stacks)
    backend = get_backend("python")
    ours = backend.chains2d(stacks["ours_interior"], stacks["scales"], similitude=0.8,
                            box_ratio=1.0, dist2_max=50.0, min_len=1, smooth=False)
    assert min((len(c["x"]) for c in theirs), default=2) >= 2
    extras = [c for c in ours
              if (int(c["x"][0]), int(c["y"][0])) not in
              {(int(t["x"][0]), int(t["y"][0])) for t in theirs}]
    assert extras and all(len(c["x"]) == 1 for c in extras)
