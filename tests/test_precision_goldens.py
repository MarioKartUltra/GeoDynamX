# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The FFT precision goldens: the WTMM 2-D transform, an M-Z reconstruction and 2-D SSA on one
small field, at 32 bits within 1e-4 and at 64 bits within 1e-10 of stored double references
(FFTW3 at 64 bits). Regenerate with ``python tests/test_precision_goldens.py``; the
references change only by a deliberate decision.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from dynamix.core import fft_policy as fb

GOLDEN = pathlib.Path(__file__).parent / "golden" / "precision" / "references_f64.npz"


def _field(n=48, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)).cumsum(0).cumsum(1)


def _outputs() -> dict:
    """The three pinned results, each on the policy's active backend."""
    from dynamix.core import mz_edges
    from dynamix.core.ssa2d import ssa2d
    from dynamix.core.wtmm_backend import compute_scales2d, get_backend

    v = _field()
    out = {"wtmm2d_mod": get_backend("python").cwt2d(v, compute_scales2d(2, 2))["mod"],
           "mz_recon": mz_edges.preview(v, mz_edges.analyze(v, 3), n_iter=5)[0],
           "ssa_recon": ssa2d(v, rows_window=8, cols_window=8, n_components=4,
                                solver="lanczos")["recon"]}
    return {k: np.asarray(x, dtype=np.float64) for k, x in out.items()}


@pytest.mark.parametrize("precision,rtol", [("f32", 1e-4), ("f64", 1e-10)])
def test_every_pinned_result_matches_its_double_reference(precision, rtol):
    fb.configure("auto", 32)
    ref = np.load(GOLDEN)
    with fb.using(precision):
        got = _outputs()
    assert set(got) == set(ref.files)
    for name in ref.files:
        err = np.abs(got[name] - ref[name]).max() / np.abs(ref[name]).max()
        assert err <= rtol, (name, precision, err)


if __name__ == "__main__":
    fb.configure("fftw", 64)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN, **_outputs())
    print("wrote", GOLDEN)
