# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.pca and dynamix.core.tucker_havok -- the ASTER-notebook ports
(the author's aster_him.ipynb), rebuilt under the 2026-09-21 efficiency mandate: covariance-trick PCA,
stride-view Hankel embedding (never materialized), blockwise-Gram HOSVD."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import pca, tucker_havok as th


# --------------------------------------------------------------------------------------- PCA

def _stack(ny=24, nx=20, nc=5, seed=2):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(ny, nx, 2))
    mix = rng.normal(size=(2, nc))
    return latent @ mix + 0.01 * rng.normal(size=(ny, nx, nc))


def test_pca_matches_full_svd_reference():
    """The covariance trick must equal the tall-SVD PCA (sklearn's route) exactly: same
    explained-variance ratios, same score images up to the fixed sign convention."""
    stack = _stack()
    out = pca.fit_pca(stack, 3)
    X = stack.reshape(-1, stack.shape[2])
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ratio_ref = (s ** 2) / (s ** 2).sum()
    np.testing.assert_allclose(out["explained_var_ratio"], ratio_ref[:3], rtol=1e-9)
    scores_ref = Xc @ Vt.T[:, :3]
    for i in range(3):
        got = out["images"][i].ravel()
        # sign fixed by the largest-|loading|-positive convention; compare up to that sign
        sign = np.sign(np.dot(got, scores_ref[:, i]))
        np.testing.assert_allclose(got, sign * scores_ref[:, i], rtol=2e-5, atol=1e-6)


def test_pca_components_are_orthonormal_and_variance_ordered():
    out = pca.fit_pca(_stack(nc=6), 6)
    C = out["components"]
    np.testing.assert_allclose(C.T @ C, np.eye(6), atol=1e-10)
    r = out["explained_var_ratio"]
    assert np.all(np.diff(r) <= 1e-12) and abs(r.sum() - 1.0) < 1e-9


def test_pca_masks_nodata_pixels_everywhere():
    stack = _stack()
    stack[3, 4, 1] = np.nan                     # ONE bad band poisons the pixel, nothing else
    out = pca.fit_pca(stack, 2)
    assert np.isnan(out["images"][:, 3, 4]).all()
    assert np.isfinite(np.delete(out["images"].reshape(2, -1),
                                 3 * stack.shape[1] + 4, axis=1)).all()


def test_pca_standardize_is_correlation_pca():
    stack = _stack()
    stack[..., 0] *= 1000.0                     # a loud band dominates covariance-PCA
    plain = pca.fit_pca(stack, 1)
    std = pca.fit_pca(stack, 1, standardize=True)
    assert abs(plain["components"][0, 0]) > 0.99            # covariance: the loud band wins
    assert abs(std["components"][0, 0]) < 0.9               # correlation: it no longer does


def test_pca_refuses_single_band():
    with pytest.raises(ValueError):
        pca.fit_pca(np.zeros((8, 8, 1)), 1)


# ------------------------------------------------------------------------------ delay embed

def test_delay_embed_is_a_zero_copy_view_with_the_havok_layout():
    a = np.arange(40.0).reshape(10, 4)
    X = th.delay_embed(a, 3)
    assert X.shape == (3, 8, 4)
    assert np.shares_memory(X, a)                # the stride trick, never a materialization
    np.testing.assert_array_equal(X[0, :, 0], a[0:8, 0])
    np.testing.assert_array_equal(X[2, :, 0], a[2:10, 0])


def test_delay_embed_cols_and_band_axis():
    a = np.arange(120.0).reshape(5, 8, 3)
    X = th.delay_embed(a, 4, axis="cols")
    assert X.shape == (4, 5, 5, 3)               # (delay, T'=8-4+1, space=5, band)
    with pytest.raises(ValueError):
        th.delay_embed(a, 6, axis="rows")        # only 5 samples along rows


# ------------------------------------------------------------------------------------ HOSVD

def _low_rank_tensor(shape=(6, 30, 10), ranks=(2, 3, 2), seed=4):
    rng = np.random.default_rng(seed)
    core = rng.normal(size=ranks)
    T = core
    for n, (d, r) in enumerate(zip(shape, ranks)):
        U = np.linalg.qr(rng.normal(size=(d, r)))[0]
        T = np.moveaxis(np.tensordot(U, T, axes=([1], [n])), 0, n)
    return T


def test_hosvd_recovers_an_exactly_low_multilinear_rank_tensor():
    T = _low_rank_tensor()
    dec = th.hosvd_tucker(T, (2, 3, 2))
    for U in dec["factors"]:
        np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-10)
    back = dec["core"]
    for n, U in enumerate(dec["factors"]):
        back = np.moveaxis(np.tensordot(U, back, axes=([1], [n])), 0, n)
    np.testing.assert_allclose(back, T, atol=1e-9)
    assert all(e > 0.999999 for e in dec["energy"])


def test_hosvd_truncation_error_decreases_with_rank():
    rng = np.random.default_rng(9)
    T = _low_rank_tensor() + 0.05 * rng.normal(size=(6, 30, 10))

    def err(ranks):
        dec = th.hosvd_tucker(T, ranks)
        back = dec["core"]
        for n, U in enumerate(dec["factors"]):
            back = np.moveaxis(np.tensordot(U, back, axes=([1], [n])), 0, n)
        return float(np.linalg.norm(back - T))

    assert err((1, 1, 1)) > err((2, 3, 2)) > err((6, 30, 10)) - 1e-9
    assert err((6, 30, 10)) < 1e-9               # full rank = exact


def test_hooi_sweeps_never_increase_the_error():
    rng = np.random.default_rng(11)
    T = _low_rank_tensor() + 0.2 * rng.normal(size=(6, 30, 10))

    def recon(dec):
        back = dec["core"]
        for n, U in enumerate(dec["factors"]):
            back = np.moveaxis(np.tensordot(U, back, axes=([1], [n])), 0, n)
        return back

    e0 = np.linalg.norm(recon(th.hosvd_tucker(T, (2, 2, 2), sweeps=0)) - T)
    e2 = np.linalg.norm(recon(th.hosvd_tucker(T, (2, 2, 2), sweeps=2)) - T)
    assert e2 <= e0 + 1e-9


def test_blockwise_gram_equals_direct():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 40, 7))
    for mode in range(3):
        direct = np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)
        np.testing.assert_allclose(th._mode_gram_blockwise(X, mode, block=8),
                                   direct @ direct.T, rtol=1e-10)


# ----------------------------------------------------------------------------- full pipeline

def test_tucker_havok_full_rank_roundtrips_the_raster():
    rng = np.random.default_rng(6)
    img = rng.normal(size=(40, 12))
    out = th.tucker_havok(img, n_delays=8, ranks=None)     # no truncation anywhere
    np.testing.assert_allclose(out["recon"], img, atol=1e-8)
    np.testing.assert_allclose(out["residual"], 0.0, atol=1e-8)


def test_tucker_havok_truncated_recon_and_nan_remask():
    x = np.linspace(0, 6 * np.pi, 60)
    img = np.sin(x)[:, None] * np.ones((1, 10)) + 0.01 * np.random.default_rng(1).normal(size=(60, 10))
    img[5, 5] = np.nan
    out = th.tucker_havok(img, n_delays=10, ranks=(2, 2, 1))
    assert out["recon"].shape == img.shape
    assert np.isnan(out["recon"][5, 5]) and np.isnan(out["residual"][5, 5])
    finite = np.isfinite(img)
    corr = np.corrcoef(out["recon"][finite], img[finite])[0, 1]
    assert corr > 0.95                                     # a sine is rank-2 in delay space
    assert out["shape_embedded"] == (10, 51, 10)


def test_tucker_havok_multiband_and_cols_axis():
    rng = np.random.default_rng(8)
    stack = rng.normal(size=(12, 50, 3))
    out = th.tucker_havok(stack, n_delays=6, ranks=(4, 10, 6, 2), axis="cols")
    assert out["recon"].shape == stack.shape
    assert out["shape_embedded"] == (6, 45, 12, 3)
    assert len(out["factors"]) == 4


# --------------------------------------------------------------------------------- devices

def _field3(ny=24, nx=20, nc=4, seed=3):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    rng = np.random.default_rng(seed)
    return RasterField(name="stack", values=rng.normal(size=(ny, nx, nc)), frame=LocalFrame(),
                       x_axis=np.arange(nx, dtype=np.float64),
                       y_axis=np.arange(ny, dtype=np.float64))


def test_pca_device_shows_the_chosen_component(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    dev = get_device("pca")
    field = _field3()
    params = dict(defaults_for(dev), n_components=3, component=2)
    res = dev.view(dev.compute(field, params), params)      # ``component`` is view-only
    np.testing.assert_array_equal(res["raster_out"], res["pca_images"][1])
    assert res["raster_out"].shape == (24, 20)
    assert res["_shape"] == (24, 20) and res["chains"] == []
    assert res["explained_var_ratio"].shape == (3,)


def test_pca_device_refuses_single_band_and_result_dicts(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    register_builtin_devices()
    dev = get_device("pca")
    flat = RasterField(name="flat", values=np.zeros((8, 8)), frame=LocalFrame(),
                       x_axis=np.arange(8.0), y_axis=np.arange(8.0))
    # 2026-09-21: the refusal must not reuse the word "components" -- that is the KEEP knob's
    # name -- it must say BANDS and point at what to do instead.
    with pytest.raises(ValueError, match="across BANDS"):
        dev.compute(flat, dict(defaults_for(dev)))
    with pytest.raises(ValueError, match="FIRST"):
        dev.compute({"raster_out": 1}, dict(defaults_for(dev)))


def test_tucker_device_recon_and_residual_share_one_cache_entry(clean_registry):
    """``show`` is view-only: recon and residual come from ONE cached
    decomposition, so they share a key -- switching never re-decomposes. (This test used to pin
    the opposite: the two keyed the cache apart.)"""
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    dev = get_device("tucker_havok")
    field = _field3(nc=3)
    base = dict(defaults_for(dev), embed="delay", n_delays=6, rank_delay=3)   # the 1-D tape
    res = dev.compute(field, base)
    recon = dev.view(res, dict(base, show="recon"))
    resid = dev.view(res, dict(base, show="residual"))
    assert recon["raster_out"].shape == (24, 20)
    np.testing.assert_allclose(recon["raster_out"] + resid["raster_out"],
                               field.values[..., 0], atol=1e-8)
    assert dev.cache_key("s", dict(base, show="recon")) == \
        dev.cache_key("s", dict(base, show="residual"))
    assert res_energy_ok(recon)


def res_energy_ok(res):
    return len(res["tucker_energy"]) == 4 and all(0 < e <= 1 + 1e-12
                                                  for e in res["tucker_energy"])


def test_tucker_device_works_on_scalar_fields_too(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    register_builtin_devices()
    dev = get_device("tucker_havok")
    rng = np.random.default_rng(4)
    flat = RasterField(name="flat", values=rng.normal(size=(40, 12)), frame=LocalFrame(),
                       x_axis=np.arange(12.0), y_axis=np.arange(40.0))
    # The 1-D tape (no longer the default since the symmetric 2-D delay, 2026-09-23).
    res = dev.compute(flat, dict(defaults_for(dev), embed="delay", n_delays=8,
                                 rank_delay=0))
    np.testing.assert_allclose(res["raster_out"], flat.values, atol=1e-8)  # full rank = identity
    assert len(res["tucker_energy"]) == 3


# --------------------------------------------------------------- plain (no-embed) Tucker

def test_tucker_plain_is_truncated_svd_on_a_2d_field():
    """embed='none' on a 2-D field is exactly a truncated SVD -- pinned against numpy."""
    rng = np.random.default_rng(12)
    A = rng.normal(size=(30, 20))
    out = th.tucker_plain(A, ranks=(5, 5))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    svd5 = (U[:, :5] * s[:5]) @ Vt[:5]
    np.testing.assert_allclose(out["recon"], svd5, atol=1e-8)
    assert out["shape_embedded"] == (30, 20)


def test_tucker_plain_full_rank_identity_and_nan():
    rng = np.random.default_rng(13)
    A = rng.normal(size=(12, 9, 3))
    A[2, 2, 1] = np.nan
    out = th.tucker_plain(A, ranks=None)
    finite = np.isfinite(A)
    np.testing.assert_allclose(out["recon"][finite], A[finite], atol=1e-8)
    assert np.isnan(out["recon"][2, 2, 1])


def test_device_embed_none_ignores_delay_knobs(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    dev = get_device("tucker_havok")
    field = _field3(nc=3)
    base = dict(defaults_for(dev), embed="none", rank_time=5, rank_space=5, rank_band=2)
    a = dev.compute(field, dict(base, n_delays=8, rank_delay=2))
    b = dev.compute(field, dict(base, n_delays=32, rank_delay=7))
    np.testing.assert_allclose(a["raster_out"], b["raster_out"], atol=1e-10)
    assert a["shape_embedded"] == (24, 20, 3)            # the OG dataset's own modes
    assert len(a["tucker_energy"]) == 3


# ------------------------------------------- the symmetric 2-D delay embedding (2026-09-23)
# Both axes delayed: Ly x Lx patches, a (delay_y, delay_x, rows', cols'[, band]) tensor. On a 2-D
# field neither axis is "time", so the 1-D tape's rows/cols choice biased the result.

def _brute_2d(a, L, ranks):
    """Reference: MATERIALIZE the 2-D Hankel tensor, run the existing HOSVD, rebuild the
    tensor, average every pixel over the patches that cover it."""
    from dynamix.core.tucker_havok import hosvd_tucker

    A = a if a.ndim == 3 else a[..., None]
    X = np.lib.stride_tricks.sliding_window_view(A, (L, L), axis=(0, 1))
    X = np.moveaxis(X, (-2, -1), (0, 1)).copy()           # (L, L, ny', nx', nc)
    dec = hosvd_tucker(X, [ranks[0], ranks[0], ranks[1], ranks[2], ranks[3]])
    H = dec["core"]
    for n, U in enumerate(dec["factors"]):
        H = np.moveaxis(np.tensordot(U, H, axes=([1], [n])), 0, n)
    ny, nx = A.shape[:2]
    out = np.zeros(A.shape)
    cnt = np.zeros(A.shape[:2] + (1,))
    for dy in range(L):
        for dx in range(L):
            out[dy:dy + ny - L + 1, dx:dx + nx - L + 1] += H[dy, dx]
            cnt[dy:dy + ny - L + 1, dx:dx + nx - L + 1] += 1.0
    out /= cnt
    return out if a.ndim == 3 else out[..., 0]


@pytest.mark.parametrize("ranks", [(2, 0, 0, 0), (3, 5, 4, 0), (2, 0, 6, 0)])
def test_the_2d_delay_equals_the_materialized_reference(ranks):
    from dynamix.core.tucker_havok import tucker_havok_2d

    rng = np.random.default_rng(11)
    a = rng.standard_normal((19, 23)).cumsum(0).cumsum(1)
    L = 4
    full = [L, 19 - L + 1, 23 - L + 1, 1]
    ref = _brute_2d(a, L, [r or f for r, f in zip(ranks, full)])
    out = tucker_havok_2d(a, n_delays=L, ranks=ranks)
    np.testing.assert_allclose(out["recon"], ref, atol=1e-9)
    np.testing.assert_allclose(out["recon"] + out["residual"], a, atol=1e-12)


def test_the_2d_delay_has_no_preferred_axis():
    from dynamix.core.tucker_havok import tucker_havok_2d

    rng = np.random.default_rng(3)
    a = rng.standard_normal((21, 27)).cumsum(1)
    out = tucker_havok_2d(a, n_delays=5, ranks=(2, 6, 6, 0))
    flipped = tucker_havok_2d(a.T, n_delays=5, ranks=(2, 6, 6, 0))
    np.testing.assert_allclose(flipped["recon"], out["recon"].T, atol=1e-9)


def test_the_2d_delay_at_full_rank_gives_the_field_back_and_remasks_nans():
    from dynamix.core.tucker_havok import tucker_havok_2d

    rng = np.random.default_rng(5)
    a = rng.standard_normal((12, 14))
    a[3, 4] = np.nan
    out = tucker_havok_2d(a, n_delays=3, ranks=(0, 0, 0, 0))
    finite = np.isfinite(a)
    np.testing.assert_allclose(out["recon"][finite], a[finite], atol=1e-10)
    assert np.isnan(out["recon"][3, 4]) and np.isnan(out["residual"][3, 4])


def test_the_2d_delay_takes_a_band_mode():
    from dynamix.core.tucker_havok import tucker_havok_2d

    rng = np.random.default_rng(8)
    a = rng.standard_normal((16, 18, 3)).cumsum(0)
    ref = _brute_2d(a, 4, [2, 13, 15, 2])
    out = tucker_havok_2d(a, n_delays=4, ranks=(2, 0, 0, 2))
    np.testing.assert_allclose(out["recon"], ref, atol=1e-9)


def test_the_tucker_device_defaults_to_the_symmetric_2d_delay(clean_registry):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    dev = get_device("tucker_havok")
    params = {p.name: p for p in dev.params}
    assert params["embed"].choices == ("delay_2d", "delay", "none")
    assert params["embed"].default == "delay_2d"
    assert params["axis"].active_when == ("embed", ("delay",))
    rng = np.random.default_rng(2)
    field = RasterField(name="f", values=rng.standard_normal((24, 20)).cumsum(0),
                        frame=LocalFrame(), x_axis=np.arange(20.0), y_axis=np.arange(24.0))
    res = dev.view(dev.compute(field, dict(defaults_for(dev), n_delays=5, rank_delay=2)),
                   dict(defaults_for(dev)))
    assert res["shape_embedded"] == (5, 5, 20, 16)
    np.testing.assert_allclose(res["tucker_recon"] + res["tucker_residual"], field.values,
                               atol=1e-10)


# ------------------------------------------- components you can click through (2026-09-23)
# The rank-truncated reconstruction is a SUM of components (SSA's elementary reconstructions):
# per kept (delay_y, delay_x) pattern pair in 2-D, per delay pattern on the 1-D tape, per rows
# mode with no embedding. Kept once with the result, top-32 by core energy, largest first.

@pytest.mark.parametrize("embed", ["delay_2d", "delay", "none"])
def test_the_components_sum_to_the_reconstruction(embed):
    from dynamix.core.tucker_havok import tucker_havok, tucker_havok_2d, tucker_plain

    rng = np.random.default_rng(21)
    a = rng.standard_normal((22, 26)).cumsum(0).cumsum(1)
    if embed == "delay_2d":
        out = tucker_havok_2d(a, n_delays=5, ranks=(3, 0, 0))
    elif embed == "delay":
        out = tucker_havok(a, n_delays=6, ranks=(4, 0, 0))
    else:
        out = tucker_plain(a, ranks=(5, 0))
    comps, energy = out["components"], out["component_energy"]
    assert comps.shape[1:] == a.shape and len(energy) == len(comps)
    np.testing.assert_allclose(comps.sum(axis=0), out["recon"], atol=1e-8)
    assert np.all(np.diff(energy) <= 1e-12)                      # largest first
    assert 0.0 < energy.sum() <= 1.0 + 1e-9


def test_components_are_capped_at_32_by_energy():
    from dynamix.core.tucker_havok import tucker_plain

    a = np.random.default_rng(1).standard_normal((40, 36))
    out = tucker_plain(a, ranks=(0, 0))                          # 40 rows modes kept
    assert len(out["components"]) == 32


def test_the_tucker_device_steps_through_its_components_without_recomputing(clean_registry):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    register_builtin_devices()
    rng = np.random.default_rng(6)
    field = RasterField(name="f", values=rng.standard_normal((30, 34)).cumsum(0),
                        frame=LocalFrame(), x_axis=np.arange(34.0), y_axis=np.arange(30.0))
    cache = Cache()

    def show(**p):
        params = {"n_delays": 5, "rank_delay": 2, **p}
        layer = Layer(layer_id=1, name="L", source_id="s",
                      chain=Chain((DeviceRef("tucker_havok", params),)).materialized())
        return resolve(layer, field, cache)

    first = show(show="recon")
    c1 = show(show="component", component=1)
    c3 = show(show="component", component=3)
    c99 = show(show="component", component=64)                  # clipped to the last one
    assert first.cache_misses == 1
    assert c1.cache_misses == c3.cache_misses == c99.cache_misses == 0
    comps = c1.result["tucker_components"]
    np.testing.assert_array_equal(c1.result["raster_out"], comps[0])
    np.testing.assert_array_equal(c3.result["raster_out"], comps[2])
    np.testing.assert_array_equal(c99.result["raster_out"], comps[-1])


# ------------------------------------------- HOOI sweeps + combined orientation pairs (2026-09-23)

def _brute_2d_hooi(a, L, ranks, sweeps):
    from dynamix.core.tucker_havok import hosvd_tucker

    A = a if a.ndim == 3 else a[..., None]
    X = np.moveaxis(np.lib.stride_tricks.sliding_window_view(A, (L, L), axis=(0, 1)),
                    (-2, -1), (0, 1)).copy()
    dec = hosvd_tucker(X, [ranks[0], ranks[0], ranks[1], ranks[2], ranks[3]], sweeps=sweeps)
    H = dec["core"]
    for n, U in enumerate(dec["factors"]):
        H = np.moveaxis(np.tensordot(U, H, axes=([1], [n])), 0, n)
    ny, nx = A.shape[:2]
    out, cnt = np.zeros(A.shape), np.zeros(A.shape[:2] + (1,))
    for dy in range(L):
        for dx in range(L):
            out[dy:dy + ny - L + 1, dx:dx + nx - L + 1] += H[dy, dx]
            cnt[dy:dy + ny - L + 1, dx:dx + nx - L + 1] += 1.0
    out /= cnt
    return out if a.ndim == 3 else out[..., 0]


@pytest.mark.parametrize("ranks,sweeps", [((2, 0, 0), 2), ((3, 5, 4), 1), ((2, 0, 6), 3)])
def test_2d_hooi_sweeps_equal_the_materialized_reference(ranks, sweeps):
    from dynamix.core.tucker_havok import tucker_havok_2d

    rng = np.random.default_rng(31)
    a = rng.standard_normal((19, 23)).cumsum(0).cumsum(1) + rng.standard_normal((19, 23))
    L = 4
    full = [L, 19 - L + 1, 23 - L + 1, 1]
    ref = _brute_2d_hooi(a, L, [r or f for r, f in zip(ranks + (0,), full)], sweeps)
    out = tucker_havok_2d(a, n_delays=L, ranks=ranks, sweeps=sweeps)
    np.testing.assert_allclose(out["recon"], ref, atol=1e-8)


def test_2d_hooi_never_keeps_less_core_energy_than_hosvd():
    """HOOI maximizes the energy the kept bases capture in the DELAY TENSOR (||core||^2) --
    monotone over sweeps. The IMAGE residual is not guaranteed monotone (it goes through the
    patch averaging, which is not that projection), so it is not what this pins."""
    from dynamix.core.tucker_havok import tucker_havok_2d

    rng = np.random.default_rng(9)
    a = rng.standard_normal((30, 34)).cumsum(1) + 0.5 * rng.standard_normal((30, 34))
    kept = [np.linalg.norm(tucker_havok_2d(a, n_delays=6, ranks=(2, 0, 0),
                                           sweeps=s)["core"]) for s in (0, 1, 3)]
    assert kept[1] >= kept[0] - 1e-9 and kept[2] >= kept[1] - 1e-9


def test_combined_components_pair_each_vertical_with_its_horizontal_twin():
    """(a, b) and (b, a) combined: r(r+1)/2 groups, each the SUM of its two separate
    components; together they still sum to the reconstruction."""
    from dynamix.core.tucker_havok import tucker_havok_2d

    rng = np.random.default_rng(13)
    a = rng.standard_normal((24, 28)).cumsum(0).cumsum(1)
    out = tucker_havok_2d(a, n_delays=5, ranks=(3, 0, 0))
    comb, pairs = out["combined_components"], out["combined_pairs"]
    assert len(comb) == 6 and len(out["combined_energy"]) == 6
    np.testing.assert_allclose(comb.sum(axis=0), out["recon"], atol=1e-8)
    assert np.all(np.diff(out["combined_energy"]) <= 1e-12)
    sep = {tuple(p): c for p, c in zip(out["component_pairs"], out["components"])}
    for group, c in zip(pairs, comb):
        members = [sep[(i, j)] for i, j in {tuple(group), tuple(group[::-1])}]
        np.testing.assert_allclose(c, np.sum(members, axis=0), atol=1e-10)


def test_the_tucker_device_toggles_separate_and_combined_without_recomputing(clean_registry):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    register_builtin_devices()
    rng = np.random.default_rng(6)
    field = RasterField(name="f", values=rng.standard_normal((30, 34)).cumsum(0),
                        frame=LocalFrame(), x_axis=np.arange(34.0), y_axis=np.arange(30.0))
    cache = Cache()

    def show(**p):
        params = {"n_delays": 5, "rank_delay": 3, "show": "component", **p}
        layer = Layer(layer_id=1, name="L", source_id="s",
                      chain=Chain((DeviceRef("tucker_havok", params),)).materialized())
        return resolve(layer, field, cache)

    sep = show(pairs="separate", component=2)
    comb = show(pairs="combined", component=2)
    assert comb.cache_misses == 0
    np.testing.assert_array_equal(comb.result["raster_out"],
                                  comb.result["tucker_combined_components"][1])
    assert not np.array_equal(sep.result["raster_out"], comb.result["raster_out"])


def test_the_tucker_residual_is_the_data_minus_the_chosen_group(clean_registry):
    """Group = the components summed into the reconstruction; the residual is the data minus
    that sum ("all" keeps the whole truncated reconstruction). Separate or combined numbering
    follows the Orientation knob. Every flip is a cache hit."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    register_builtin_devices()
    rng = np.random.default_rng(7)
    field = RasterField(name="f", values=rng.standard_normal((30, 34)).cumsum(0),
                        frame=LocalFrame(), x_axis=np.arange(34.0), y_axis=np.arange(30.0))
    cache = Cache()

    def show(**p):
        params = {"n_delays": 5, "rank_delay": 3, **p}
        layer = Layer(layer_id=1, name="L", source_id="s",
                      chain=Chain((DeviceRef("tucker_havok", params),)).materialized())
        return resolve(layer, field, cache)

    whole = show(show="residual")
    np.testing.assert_array_equal(whole.result["raster_out"], whole.result["tucker_residual"])
    minus_c1 = show(show="residual", group="1")
    minus_comb = show(show="residual", group="1", pairs="combined")
    recon_12 = show(show="recon", group="1-2")
    assert minus_c1.cache_misses == minus_comb.cache_misses == recon_12.cache_misses == 0
    sep_c = whole.result["tucker_components"]
    comb_c = whole.result["tucker_combined_components"]
    np.testing.assert_allclose(minus_c1.result["raster_out"], field.values - sep_c[0],
                               atol=1e-9)
    np.testing.assert_allclose(minus_comb.result["raster_out"], field.values - comb_c[0],
                               atol=1e-9)
    np.testing.assert_allclose(recon_12.result["raster_out"], sep_c[0] + sep_c[1])
    energy = whole.result["tucker_component_energy"]
    assert minus_c1.result["_view_note"] == f"data − [1] ({100 * energy[0]:.0f}%)"
    assert "_view_note" not in whole.result          # the ungrouped labels are unchanged


def test_tucker_on_a_multiband_field_keeps_every_band_of_its_recon_and_residual(clean_registry):
    """A derivative can take any input band: the display shows band 1, the full stacks ride."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    register_builtin_devices()
    rng = np.random.default_rng(8)
    vals = rng.standard_normal((20, 22, 3)).cumsum(0)
    field = RasterField(name="f", values=vals, frame=LocalFrame(), x_axis=np.arange(22.0),
                        y_axis=np.arange(20.0))
    layer = Layer(layer_id=1, name="L", source_id="s", chain=Chain((DeviceRef(
        "tucker_havok", {"n_delays": 4, "rank_delay": 2}),)).materialized())
    out = resolve(layer, field, Cache()).result
    assert out["tucker_recon_bands"].shape == (20, 22, 3)
    np.testing.assert_array_equal(out["tucker_recon_bands"][..., 0], out["tucker_recon"])
    np.testing.assert_allclose(out["tucker_recon_bands"] + out["tucker_residual_bands"], vals,
                               atol=1e-9)


def test_tucker_hooi_hosvd_toggles_the_algorithm_and_matches_the_old_tool(clean_registry):
    """``tucker_havok`` renamed ``tucker_HOOI_HOSVD`` with the algorithm as a two-state toggle:
    HOSVD (one pass) or HOOI (the HOSVD refined by ``sweeps`` alternating passes). The old name
    stays registered with its old behaviour (its sweeps knob alone chose: 0 = HOSVD), so a saved
    project resolves exactly as before."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.device import defaults_for, get_device
    from dynamix.model.layer import Layer

    register_builtin_devices()
    new = get_device("tucker_HOOI_HOSVD")
    params = {p.name: p for p in new.params}
    assert defaults_for(new)["method"] == "HOSVD"
    assert params["method"].choices == ("HOSVD", "HOOI")
    assert params["sweeps"].active_when == ("method", ("HOOI",))
    assert defaults_for(new)["sweeps"] >= 1
    rng = np.random.default_rng(12)
    field = RasterField(name="f", values=rng.standard_normal((26, 30)).cumsum(0),
                        frame=LocalFrame(), x_axis=np.arange(30.0), y_axis=np.arange(26.0))

    def run(device, **p):
        layer = Layer(layer_id=1, name="L", source_id="s", chain=Chain((DeviceRef(
            device, {"n_delays": 5, "rank_delay": 2, **p}),)).materialized())
        return resolve(layer, field, Cache()).result["raster_out"]

    np.testing.assert_array_equal(run("tucker_HOOI_HOSVD", method="HOSVD", sweeps=3),
                                  run("tucker_havok", sweeps=0))
    np.testing.assert_array_equal(run("tucker_HOOI_HOSVD", method="HOOI", sweeps=2),
                                  run("tucker_havok", sweeps=2))
    assert not np.array_equal(run("tucker_HOOI_HOSVD", method="HOOI", sweeps=2),
                              run("tucker_HOOI_HOSVD", method="HOSVD"))
