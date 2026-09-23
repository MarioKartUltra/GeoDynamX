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
    res = dev.compute(field, params)
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


def test_tucker_device_recon_and_residual_key_the_cache_apart(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    dev = get_device("tucker_havok")
    field = _field3(nc=3)
    base = dict(defaults_for(dev), n_delays=6, rank_delay=3)
    recon = dev.compute(field, dict(base, show="recon"))
    resid = dev.compute(field, dict(base, show="residual"))
    assert recon["raster_out"].shape == (24, 20)
    np.testing.assert_allclose(recon["raster_out"] + resid["raster_out"],
                               field.values[..., 0], atol=1e-8)
    assert dev.cache_key("s", dict(base, show="recon")) != \
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
    res = dev.compute(flat, dict(defaults_for(dev), n_delays=8, rank_delay=0))
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
