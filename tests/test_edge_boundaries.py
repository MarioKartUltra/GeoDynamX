# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""What the FFT torus holds beyond the real data (spec 25 §3.2): every convolution is circular, and
the edge mode pads the field before the transform -- mirrored, the edge values repeated, zeros -- or
not at all, ``periodic``, where the raster itself is the torus."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.wtmm_backend import PythonWTMMBackend

SCALES = [1.0, 2.0, 3.0]
ENGINES = ["fftw"] + (["mlx"] if __import__("importlib").util.find_spec("mlx") else [])
NP_MODE = {"reflect": "reflect", "edge_pad": "edge", "zero_pad": "constant"}


def _field(n=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)).cumsum(0).cumsum(1) / n


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mode", sorted(NP_MODE))
def test_a_padded_mode_is_the_periodic_transform_of_the_padded_field(engine, mode):
    f, pad = _field(), 20
    be = PythonWTMMBackend()
    got = be.cwt2d(f, SCALES, pad=pad, pad_mode=mode, _engine=engine)
    ref = be.cwt2d(np.pad(f, pad, mode=NP_MODE[mode]), SCALES, pad=0, pad_mode="periodic",
                   _engine=engine)
    crop = (slice(None), slice(pad, pad + f.shape[0]), slice(pad, pad + f.shape[1]))
    scale = float(np.abs(ref["mod"]).max())
    np.testing.assert_allclose(got["mod"], ref["mod"][crop], rtol=1e-5, atol=1e-5 * scale)


@pytest.mark.parametrize("engine", ENGINES)
def test_periodic_is_the_centre_tile_of_the_tiled_field(engine):
    n = 48
    y, x = np.mgrid[0:n, 0:n] / n
    f = np.sin(2 * np.pi * 3 * x) + 0.5 * np.cos(2 * np.pi * 2 * y + 1.0) + 0.3 * np.sin(2 * np.pi * (x + y))
    be = PythonWTMMBackend()
    got = be.cwt2d(f, SCALES, pad=0, pad_mode="periodic", _engine=engine)
    tiled = be.cwt2d(np.tile(f, (3, 3)), SCALES, pad=0, pad_mode="periodic", _engine=engine)
    centre = (slice(None), slice(n, 2 * n), slice(n, 2 * n))
    scale = float(np.abs(tiled["mod"]).max())
    np.testing.assert_allclose(got["mod"], tiled["mod"][centre], rtol=1e-5, atol=1e-5 * scale)


def test_an_unknown_pad_mode_is_refused():
    with pytest.raises(ValueError, match="pad_mode"):
        PythonWTMMBackend().cwt2d(_field(16), [1.0], pad=4, pad_mode="toroidal", _engine="fftw")


# --------------------------------------------------------------------- the padding width (§3.2)

def test_the_padding_width_by_each_rule():
    import math

    from dynamix.core.edges import pad_widths
    from dynamix.core.reach import reach

    scales = [8.0, 16.0, 32.0]
    fixed, notes = pad_widths(scales, "fixed_px", wavelet="mexican", pad_px=32)
    assert fixed == [32, 32, 32] and notes == []
    pct, _ = pad_widths(scales, "border_pct", wavelet="mexican", pad_pct=0.72)
    assert pct == [int(s * 0.72) for s in scales]            # xsmurf's rule; s carries 6/0.86
    measured, notes = pad_widths(scales, "measured_reach", wavelet="mexican", eps=1e-3)
    assert measured == [int(math.ceil(reach("mexican", None, s, 1e-3).radius)) for s in scales]
    assert measured == sorted(measured) and notes == []


def test_a_scale_without_a_finite_reach_pads_like_the_widest_converged_one():
    """The finest scales ring at Nyquist (spec 22 §3.1): their "measured" radius is the ringing's,
    hundreds of px, never a width to pad by."""
    import math

    from dynamix.core.edges import pad_widths
    from dynamix.core.reach import reach

    assert not reach("mexican", None, 4.0, 1e-3).converged
    widths, notes = pad_widths([4.0, 16.0], "measured_reach", wavelet="mexican")
    widest = int(math.ceil(reach("mexican", None, 16.0, 1e-3).radius))
    assert widths == [widest, widest]
    assert notes and "did not converge at scale(s) 4 px" in notes[0]
    alone, notes = pad_widths([4.0], "measured_reach", wavelet="mexican", pad_px=32)
    assert alone == [32] and "fixed width" in notes[0]


def test_an_unknown_padding_rule_is_refused():
    from dynamix.core.edges import pad_widths

    with pytest.raises(ValueError, match="pad_width"):
        pad_widths([2.0], "halo", wavelet="mexican")


# --------------------------------------------------------------------- the pad mode through run_wtmm2d

def _rf(values):
    from tests.conftest import make_field

    return make_field(values)


def test_run_wtmm2d_pads_by_its_pad_mode_and_keys_it():
    from dynamix.core.wtmm_backend import run_wtmm2d

    f = _rf(_field(64, seed=3))
    base = {"n_oct": 2, "n_voice": 2, "a_min": 1.0, "pad": 24}
    reflect = run_wtmm2d(f, dict(base))
    again = run_wtmm2d(f, dict(base, pad_mode="reflect"))
    zero = run_wtmm2d(f, dict(base, pad_mode="zero_pad"))
    assert reflect["params"]["pad_mode"] == "reflect" and zero["params"]["pad_mode"] == "zero_pad"
    n_r = sum(len(e["x"]) for e in reflect["extrema"])
    assert n_r == sum(len(e["x"]) for e in again["extrema"])
    assert [len(e["x"]) for e in zero["extrema"]] != [len(e["x"]) for e in reflect["extrema"]]


# --------------------------------------------------------------------- the Border cut (§3.3)

def test_the_border_by_each_sizing_and_scope():
    import math

    from dynamix.core.edges import border_widths
    from dynamix.core.reach import reach

    scales = [8.0, 16.0, 32.0]
    kw = dict(wavelet="mexican", eps=1e-3)
    assert border_widths(scales, "none", **kw) == [0, 0, 0]
    own = [int(math.ceil(reach("mexican", None, s, 1e-3).radius)) for s in scales]
    assert border_widths(scales, "measured_reach", scope="per_scale", **kw) == own
    assert border_widths(scales, "measured_reach", scope="largest", **kw) == [own[-1]] * 3
    assert border_widths(scales, "border_pct", pct=0.72, scope="largest", **kw) == [int(32 * 0.72)] * 3
    assert border_widths(scales, "border_pct", pct=0.72, scope="per_scale", **kw) == [int(s * 0.72) for s in scales]
    assert border_widths(scales, "k_scale", k=2.0, scope="per_scale", **kw) == [2.0 * s for s in scales]


def test_the_border_cut_runs_before_the_v_chains():
    """xsmurf's order (study.tcl: follow, rm_ext, then vchain): the chains are those the chaining
    makes from the cut maxima, so a border maximum never takes part in the linking."""
    from dynamix.core import chaining, edges
    from dynamix.core.wtmm_backend import get_backend, run_wtmm2d

    f = _rf(_field(96, seed=5))
    base = {"n_oct": 2, "n_voice": 3, "a_min": 1.0}
    cut = dict(base, border_cut="k_scale", border_k=2.0, border_scope="per_scale")
    raw = run_wtmm2d(f, dict(base))
    got = run_wtmm2d(f, cut)
    scales = list(got["scales"])
    borders = edges.border_widths(scales, "k_scale", wavelet="mexican", k=2.0, scope="per_scale")
    expected_ext = edges.drop_in_cone(raw["extrema"], borders, (0, 0), (96, 96))
    assert [len(e["x"]) for e in got["extrema"]] == [len(e["x"]) for e in expected_ext]
    assert sum(len(e["x"]) for e in expected_ext) < sum(len(e["x"]) for e in raw["extrema"])
    expected = chaining.link(expected_ext, scales, got["params"], get_backend("python"), shape=(96, 96))
    assert len(got["chains"]) == len(expected)
    assert all(np.array_equal(a["x"], b["x"]) for a, b in zip(got["chains"], expected))
    trimmed_after = chaining.border_trimmed(raw["chains"], (96, 96), "coi_trim", 2.0,
                                            int(got["params"]["min_chain_len"]))
    assert [len(c["x"]) for c in got["chains"]] != [len(c["x"]) for c in trimmed_after]
    assert got["_border_cut"]["borders"] == borders


# --------------------------------------------------------------------- the filled gap (§3.1)

def test_the_threshold_reads_the_real_block_and_the_gap_keeps_no_maxima():
    """A Periodic extension output's gap is synthetic: no maxima there, and the threshold's maximum
    is the real block's, so a strong gap never raises the bar for the data."""
    from dynamix.core.wtmm_backend import PythonWTMMBackend

    rng = np.random.default_rng(7)
    real = np.abs(rng.standard_normal((1, 40, 40))).cumsum(1) / 40
    mod = np.full((1, 56, 56), 1000.0)
    mod[:, :40, :40] = real
    mod[:, 40:, 40:] += rng.standard_normal((1, 16, 16))       # gap maxima candidates
    cwt = {"mod": mod, "arg": np.zeros_like(mod)}
    be = PythonWTMMBackend()
    plain = be.extrema2d(cwt, [2.0], thresh=0.01)
    kept = be.extrema2d(cwt, [2.0], thresh=0.01, real_block=(40, 40))
    in_real = lambda e: int(((e["x"] < 40) & (e["y"] < 40)).sum())      # noqa: E731
    assert in_real(plain[0]) == 0                       # 1 % of the gap's 1000 drowns the data
    assert in_real(kept[0]) > 0 and len(kept[0]["x"]) == in_real(kept[0])


def test_run_wtmm2d_keeps_no_maxima_in_the_gap_on_either_detector():
    from dynamix.core.wtmm_backend import run_wtmm2d

    f = _rf(_field(80, seed=9))
    for detector in ("nms", "follow"):
        res = run_wtmm2d(f, {"n_oct": 2, "n_voice": 2, "a_min": 1.0, "detector": detector,
                             "real_block": (64, 60)})
        for e in res["extrema"]:
            assert (np.asarray(e["x"]) < 60).all() and (np.asarray(e["y"]) < 64).all(), detector
        assert sum(len(e["x"]) for e in res["extrema"]) > 0, detector
