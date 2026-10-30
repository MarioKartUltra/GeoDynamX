# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Morse/Cerf chainer: the merge tree against the level sweep it computes, every kernel against
its numpy twin, the fold of two bumps at the scale their profile merges, the event balance, the
mutual rule, the chain schema through the product and the npz round trip, and the WTMM device on
the demo raster."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy import ndimage

from dynamix.core import morse
from dynamix.core.chaining import MORSE_CHAIN_KEYS, MORSE_POINT_KEYS, TREE_PARENTS, link
from dynamix.core.wtmm_backend import compute_scales2d, get_backend
from tests.conftest import fbm2d, make_field

DEM = pathlib.Path(__file__).resolve().parents[1] / "docs" / "demo" / "dem_crop.npz"
SCHEMA = {"x", "y", "mod", "log2_mod", "log2_scales"}


def _noise_levels(scales, *, quantize=None):
    """|blurred white noise| on 48 x 48 at ``scales`` px, the lowest 20 % off the domain;
    ``quantize`` rounds the field to that many decimals, so it holds ties."""
    noise = np.random.default_rng(7).standard_normal((48, 48))
    out = []
    for a in scales:
        M = np.abs(ndimage.gaussian_filter(noise, a, mode="wrap"))
        if quantize is not None:
            M = np.round(M, quantize)
        out.append((M, M >= np.quantile(M, 0.2)))
    return out


def _components_match(lv, M, dom, ts) -> bool:
    """#components of {M >= t} on the domain = #maxima with M >= t - #merge saddles with M >= t."""
    n_comp = [ndimage.label(dom & (M >= t), np.ones((3, 3)))[1] for t in ts]
    pred = [(lv["val"] >= t).sum() - ((lv["sad"] >= 0) & (lv["sadv"] >= t)).sum() for t in ts]
    return n_comp == pred


@pytest.mark.parametrize("quantize", [None, 2])
def test_the_merge_tree_is_the_level_sweep_and_the_climb_is_its_twin(quantize):
    climb = morse.kernels()["climb"]
    for M, dom in _noise_levels((1.0, 1.68, 2.83, 4.76), quantize=quantize):
        lv = morse.merge_tree(M, dom)
        is_max, saddle, peak = morse.merge_tree_twin(M, dom)
        assert np.array_equal(np.sort(lv["pix"]), np.flatnonzero(is_max))
        assert np.array_equal(lv["sad"], saddle[lv["pix"]])
        assert np.array_equal(lv["par"], peak[lv["pix"]])
        # a component's highest maximum pairs with the component's lowest value
        lab, n = ndimage.label(dom, np.ones((3, 3)))
        low = np.asarray(ndimage.minimum(M, lab, np.arange(1, n + 1)))
        top = lv["sad"] < 0
        assert top.sum() == n
        np.testing.assert_array_equal(lv["sadv"][top], low[lab.ravel()[lv["pix"][top]] - 1])
        assert np.all(lv["pers"] >= 0) and np.array_equal(lv["pers"], lv["val"] - lv["sadv"])
        assert _components_match(lv, M, dom, np.unique(M[dom]))
        starts = np.arange(-1, M.size)
        assert np.array_equal(climb(lv["R"], *M.shape, starts),
                              morse.climb_twin(lv["R"], M.shape, starts))


def test_the_owner_and_competition_kernels_match_their_twins():
    k = morse.kernels()
    rng = np.random.default_rng(3)
    for _ in range(20):
        n_a, n_b, n_pix = rng.integers(0, 40, 2).tolist() + [30]
        basin = rng.integers(-1, n_pix, n_b)
        val_b = rng.integers(0, 4, n_b).astype(float)              # ties on purpose
        assert np.array_equal(k["owners"](basin, val_b, n_pix),
                              morse.owners_twin(basin, val_b, n_pix))
        cand = rng.integers(-1, n_b, n_a) if n_b else np.full(n_a, -1)
        basin_a = rng.integers(-1, 6, n_a)
        back = rng.integers(-1, 6, n_b)
        val_a = rng.integers(0, 3, n_a).astype(float)
        assert np.array_equal(k["compete"](cand, basin_a, back, val_a, n_b),
                              morse.compete_twin(cand, basin_a, back, val_a, n_b))


def _two_bumps():
    """Two Gaussian bumps (heights 1 and 0.6, width 4 px, 40 px apart) blurred analytically by
    Gaussians of sigma = 2 .. 32 px at 1/8 octave; and the merge scale sigma* of the profile on the
    line joining them, by bisection on its count of maxima."""
    ny, nx, s0 = 64, 128, 4.0
    bumps = ((1.0, 32, 44), (0.6, 32, 84))

    def field(sig, y, x):
        v = s0 ** 2 + sig ** 2
        return sum(h * s0 ** 2 / v * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * v))
                   for h, cy, cx in bumps)

    sig = 2.0 ** np.arange(1, 5.01, 0.125)
    yy, xx = np.mgrid[0:ny, 0:nx]
    levels = [morse.merge_tree(field(s, yy, xx), np.ones((ny, nx), bool)) for s in sig]
    xs = np.linspace(40, 88, 48001)

    def n_max(s):
        f = field(s, 32.0, xs)
        return np.sum((f[1:-1] > f[:-2]) & (f[1:-1] > f[2:]))

    lo, hi = np.log2(sig[0]), np.log2(sig[-1])
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if n_max(2 ** mid) == 2 else (lo, mid)
    return levels, sig, 2 ** hi, (ny, nx)


def test_two_bumps_fold_once_at_the_merge_scale_of_their_profile():
    levels, sig, s_star, (ny, nx) = _two_bumps()
    tr = morse.track(levels, [lv["pix"] for lv in levels], [lv["val"] for lv in levels])
    ev = tr["events"]
    assert ev[:, 2:5].sum() == 1 and ev[:, 2].sum() == 1          # one death: an annihilation
    assert ev[:, 5:8].sum() == 0 and ev[-1, 8] == 1
    dying = int(np.flatnonzero(tr["death"] == 1)[0])
    i = int(tr["last"][dying])
    assert sig[i] < s_star <= sig[i + 1]
    survivor = int(tr["line"][-1][0])
    assert tr["succ"][dying] == survivor and tr["peak"][dying] == survivor
    # the dying maximum and its merge saddle drain into one maximum one scale coarser
    x, y, m = tr["saddle"][dying]
    climb = morse.kernels()["climb"]
    pix = levels[i]["pix"][tr["line"][i] == dying]
    nxt = levels[i + 1]["R"]
    assert climb(nxt, ny, nx, np.array([int(y) * nx + int(x)])) == climb(nxt, ny, nx, pix)
    assert m == pytest.approx(levels[i]["sadv"][levels[i]["pix"] == pix[0]][0])
    # its persistence falls monotonically towards the fold
    pers = np.array([tr["pers"][s][tr["line"][s] == dying][0] for s in range(i + 1)])
    assert np.all(np.diff(pers) < 0) and pers[-1] < 0.05 * pers[0]


def test_the_events_balance_at_every_step_and_the_mutual_rule_holds():
    levels = [morse.merge_tree(M, dom)
              for M, dom in _noise_levels(2.0 ** np.arange(0, 3.01, 0.125))]
    # the maxima themselves as the points: the mutual steepest ascent of the maxima
    tr = morse.track(levels, [lv["pix"] for lv in levels], [lv["val"] for lv in levels])
    ev = tr["events"]
    assert np.array_equal(ev[:, 8], ev[:, 0] - ev[:, 2:5].sum(1) + ev[:, 5:8].sum(1))
    assert np.array_equal(ev[1:, 0], ev[:-1, 8])
    assert ev[:, 2].sum() > 0 and ev[:, 5:8].sum() > 0
    for s, (cand, back, lk) in enumerate(zip(tr["cand"], tr["back"], tr["link"])):
        dead = np.flatnonzero((lk < 0) & (cand >= 0))
        # no death for a maximum whose successor climbs back to it
        assert not np.any(back[cand[dead]] == levels[s]["pix"][dead])
        alive = lk >= 0
        assert np.array_equal(back[lk[alive]], levels[s]["pix"][alive])


def test_a_scale_without_points_ends_every_line_and_the_next_scale_starts_new_ones():
    levels, _, _, _ = _two_bumps()
    pts = [lv["pix"] for lv in levels]
    vals = [lv["val"] for lv in levels]
    pts[3], vals[3] = pts[3][:0], vals[3][:0]
    tr = morse.track(levels, pts, vals)
    ev = tr["events"]
    assert np.array_equal(ev[:, 8], ev[:, 0] - ev[:, 2:5].sum(1) + ev[:, 5:8].sum(1))
    assert ev[2, 1] == 0 and ev[2, 3] == ev[2, 0] and ev[3, 5:8].sum() == ev[3, 8]
    assert set(tr["last"][:2]) == {2} and np.all(tr["succ"][:2] == -1)


def test_a_point_that_loses_its_basin_s_successor_loses_to_a_larger_point_of_that_basin():
    backend = get_backend("python")
    scales = compute_scales2d(2, 4)
    cwt = backend.cwt2d(fbm2d(96, 0.5, seed=5), scales)
    sm = backend.single_maxima2d(backend.extrema2d(cwt, scales, thresh=1e-3))
    nx = cwt["mod"].shape[2]
    levels = [morse.merge_tree(M, morse.domain(M, 1e-3)) for M in cwt["mod"]]
    pts = [np.asarray(e["y"]) * nx + np.asarray(e["x"]) for e in sm]
    tr = morse.track(levels, pts, [e["mod"] for e in sm])
    ev = tr["events"]
    assert np.array_equal(ev[:, 8], ev[:, 0] - ev[:, 2:5].sum(1) + ev[:, 5:8].sum(1))
    lost = 0
    for s, (cand, back, lk) in enumerate(zip(tr["cand"], tr["back"], tr["link"])):
        basin, mod = tr["basin"][s], np.asarray(sm[s]["mod"])
        mutual_dead = np.flatnonzero((lk < 0) & (cand >= 0) & (basin >= 0))
        mutual_dead = mutual_dead[back[cand[mutual_dead]] == basin[mutual_dead]]
        for i in mutual_dead:
            w = np.flatnonzero(lk == cand[i])
            assert w.size == 1 and basin[w[0]] == basin[i] and mod[w[0]] >= mod[i]
        lost += mutual_dead.size
    assert lost > 0


def _morse_chains(n=96, seed=5, **kw):
    backend = get_backend("python")
    scales = compute_scales2d(2, 4)
    cwt = backend.cwt2d(fbm2d(n, 0.5, seed=seed), scales)
    extrema = backend.extrema2d(cwt, scales, thresh=1e-3)
    params = {"similitude": 0.8, "box_ratio": 1.0, "dist2_max": 50.0, "min_chain_len": 2,
              "smooth": False, "thresh": 1e-3, "chainer": "morse_cerf", **kw}
    return link(extrema, scales, params, backend, fields=cwt["mod"]), extrema, scales, backend


def test_morse_chains_keep_the_schema_rooted_at_the_finest_single_maxima():
    chains, extrema, scales, backend = _morse_chains()
    assert chains
    finest = backend.single_maxima2d(extrema)[0]
    roots = set(zip(finest["x"].tolist(), finest["y"].tolist()))
    for i, c in enumerate(chains):
        assert set(c) == SCHEMA | set(MORSE_POINT_KEYS) | set(MORSE_CHAIN_KEYS)
        k = len(c["mod"])
        assert 2 <= k <= len(scales) and c["x"].dtype == np.int64 and len(c["pers"]) == k
        assert np.array_equal(c["log2_scales"], np.log2(scales[:k]))
        assert (int(c["x"][0]), int(c["y"][0])) in roots
        assert c["line"] == i and c["birth"] == "finest" and c["death"] in morse.DEATHS
        assert -1 <= c["succ"] < len(chains) and -1 <= c["peak"] < len(chains)
        assert c["succ"] != i and len(c["saddle"]) == 3
        assert (c["death"] == "survivor") == (k == len(scales))
    assert any(c["succ"] >= 0 for c in chains) and any(c["death"] == "annihilation" for c in chains)
    with pytest.raises(ValueError, match="fields"):
        link(extrema, scales, {"chainer": "morse_cerf", "smooth": False, "min_chain_len": 2},
             backend)


def test_morse_chains_pass_the_chain_product_and_the_npz_round_trips(tmp_path):
    from dynamix.core.chain_product import build_chain_product
    from dynamix.core.extrema_io import flatten_extrema
    from dynamix.core.wtmm_backend import (_arrays_to_chains, _chains_to_arrays,
                                           export_chains_npz, load_chains_npz)

    chains, extrema, scales, _ = _morse_chains()
    product = build_chain_product(extrema, chains, scales, (96, 96))
    assert len(product["v_persist"]) == len(chains)
    assert np.array_equal(product["v_x"], np.concatenate([c["x"] for c in chains]))
    assert np.isfinite(product["v_arg"]).all()         # each point is a maximum of its scale
    # the stage store's file keeps the Morse keys
    back = _arrays_to_chains(_chains_to_arrays(chains), scales)
    for a, b in zip(chains, back):
        assert set(a) == set(b)
        for key in a:
            np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]))
    field = make_field(np.zeros((96, 96)))
    export_chains_npz(tmp_path / "m.npz", chains=chains, extrema=extrema, scales=scales,
                      field=field, params={"chainer": "morse_cerf"})
    te = load_chains_npz(tmp_path / "m.npz")
    flat = flatten_extrema(te)
    assert flat["n_v_chains"] == len(chains) and te["params"]["chainer"] == "morse_cerf"
    assert flat["v_pts"].shape[0] == sum(len(c["x"]) for c in chains)


@pytest.fixture(scope="module")
def dem():
    from dynamix.core.rasterfield import RasterField

    if not DEM.is_file():
        pytest.skip("the demo raster is not in this checkout")
    return RasterField.load_npz(str(DEM))


def test_the_wtmm_device_rechains_the_demo_raster_by_morse_cerf(clean_registry, dem):
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import StageStore, cache_key
    from dynamix.model.device import get_device, keyed_params, validate_params

    register_builtin_devices()
    device = get_device("wtmm2d")
    base = validate_params(device, {})
    morse_p = validate_params(device, {"chainer": "morse_cerf"})
    elder = validate_params(device, {"chainer": "morse_cerf", "tree_parent": "elder_rule"})
    assert base["tree_parent"] == TREE_PARENTS[0]
    assert "tree_parent" not in keyed_params(device, morse_p)
    keys = {cache_key("wtmm2d", "s", keyed_params(device, p)) for p in (base, morse_p, elder)}
    assert len(keys) == 3
    store = StageStore(maxsize=32)
    bench = device.compute(dem, base, stage_cache=store)
    res = device.compute(dem, morse_p, stage_cache=store)
    assert {"cwt", "extrema"} <= res["cache_hits"] and "chains" not in res["cache_hits"]
    assert res["params"]["chainer"] == "morse_cerf"
    assert res["params"]["tree_parent"] == "basin_successor"
    assert len(res["chains"]) > 100 and all("pers" in c for c in res["chains"])
    assert all("pers" not in c for c in bench["chains"])
    again = device.compute(dem, elder, stage_cache=store)
    assert {"cwt", "extrema", "chains"} <= again["cache_hits"]
    assert again["params"]["tree_parent"] == "elder_rule"
    assert len(res["chain_product"]["v_persist"]) == len(res["chains"])


def test_the_halo_runs_hand_the_modulus_to_morse_cerf():
    from dynamix.devices.wtmm import WTMM2D
    from tests.test_edge_taper import _field, _labyrinth, _params

    res = WTMM2D().compute(_field(_labyrinth(128)), _params(chainer="morse_cerf"))
    assert res["_edge"]["mode"] == "gaussian_taper" and res["_edge"].get("fallback") is None
    assert res["chains"] and all(len(c["pers"]) == len(c["mod"]) for c in res["chains"])
