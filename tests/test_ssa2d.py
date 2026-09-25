# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""2D-SSA (Golyandina & Usevich 2010): the SVD of the Hankel-block-Hankel matrix of every
L_r x L_c window of the image, its elementary reconstructed components, eigenvalue shares and
w-correlations -- pinned against an explicit SVD of that matrix and the paper's rank results."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


def _brute(F, Lr, Lc):
    """The paper's algorithm, literally: build W (L_r L_c x K_r K_c), SVD it, hankelize every
    rank-1 term back onto the image grid (average over the windows covering each pixel)."""
    ny, nx = F.shape
    Kr, Kc = ny - Lr + 1, nx - Lc + 1
    W = np.stack([F[k:k + Lr, l:l + Lc].ravel() for k in range(Kr) for l in range(Kc)], axis=1)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    comps = []
    counts = np.zeros(F.shape)
    for k in range(Kr):
        for l in range(Kc):
            counts[k:k + Lr, l:l + Lc] += 1.0
    for i in range(len(s)):
        Xi = s[i] * np.outer(U[:, i], Vt[i])
        out = np.zeros(F.shape)
        for col, (k, l) in enumerate((k, l) for k in range(Kr) for l in range(Kc)):
            out[k:k + Lr, l:l + Lc] += Xi[:, col].reshape(Lr, Lc)
        comps.append(out / counts)
    return np.array(comps), s ** 2


def _img(ny=18, nx=21, seed=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((ny, nx)).cumsum(0).cumsum(1) + rng.standard_normal((ny, nx))


def test_components_equal_the_explicit_hankel_block_hankel_svd():
    from dynamix.core.ssa2d import ssa2d

    F = _img()
    ref, lam = _brute(F, 4, 5)
    out = ssa2d(F, rows_window=4, cols_window=5, n_components=8)
    np.testing.assert_allclose(out["components"], ref[:8], atol=1e-9)
    np.testing.assert_allclose(out["eigen_share"], lam[:8] / lam.sum(), atol=1e-12)
    assert out["eigenarrays"].shape == (8, 4, 5)
    assert out["factor_arrays"].shape == (8, 18 - 4 + 1, 21 - 5 + 1)


def test_all_components_sum_back_to_the_image():
    from dynamix.core.ssa2d import ssa2d

    F = _img()
    out = ssa2d(F, rows_window=3, cols_window=4, n_components=12)       # 3 x 4 = full rank
    np.testing.assert_allclose(out["components"].sum(axis=0), F, atol=1e-9)
    np.testing.assert_allclose(out["recon"], F, atol=1e-9)
    assert out["eigen_share"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("field,rank", [
    ("oblique", 2),       # cos(2 pi (wx k + wy l)): rank 2 -- rotation-invariant (paper sec 4.3.3)
    ("product", 4),       # cos(2 pi wx k) cos(2 pi wy l): rank 4
    ("exponents", 2),     # sum of two 2-D exponentials: rank 2 (Prop 4.5)
])
def test_the_papers_2d_ssa_ranks(field, rank):
    from dynamix.core.ssa2d import ssa2d

    k, l = np.mgrid[:40, :44].astype(float)
    F = {"oblique": np.cos(2 * np.pi * (0.07 * k + 0.11 * l)),
         "product": np.cos(2 * np.pi * 0.07 * k) * np.cos(2 * np.pi * 0.11 * l),
         "exponents": 0.98 ** k * 1.01 ** l + 2.0 * 1.02 ** k * 0.97 ** l}[field]
    out = ssa2d(F, rows_window=10, cols_window=12, n_components=8)
    assert out["eigen_share"][:rank].sum() > 1 - 1e-9
    assert out["eigen_share"][rank] < 1e-9
    np.testing.assert_allclose(out["components"][:rank].sum(axis=0), F, atol=1e-8)


def test_w_correlations_are_a_correlation_matrix_and_separate_a_wave_from_a_trend():
    from dynamix.core.ssa2d import ssa2d

    k, l = np.mgrid[:48, :50].astype(float)
    F = 0.02 * k + 0.01 * l + np.cos(2 * np.pi * (0.2 * k + 0.25 * l))
    out = ssa2d(F, rows_window=12, cols_window=12, n_components=6)
    R = out["w_correlation"]
    assert R.shape == (6, 6)
    np.testing.assert_allclose(R, R.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-12)
    assert np.all((R >= -1e-12) & (R <= 1 + 1e-12))
    # the wave pair and the trend components separate (a block structure)
    wave = [i for i in range(6) if np.abs(np.diff(out["components"][i], axis=0)).mean() > 0.1]
    trend = [i for i in range(6) if i not in wave and out["eigen_share"][i] > 1e-6]
    assert len(wave) == 2 and trend
    assert max(R[i, j] for i in wave for j in trend) < 0.05


def test_nans_are_filled_for_the_decomposition_and_remasked():
    from dynamix.core.ssa2d import ssa2d

    F = _img()
    F[5, 7] = np.nan
    out = ssa2d(F, rows_window=4, cols_window=4, n_components=5)
    assert np.isnan(out["recon"][5, 7]) and np.isnan(out["residual"][5, 7])
    assert np.isnan(out["components"][:, 5, 7]).all()
    assert np.isfinite(out["recon"][np.isfinite(F)]).all()


def test_progress_is_live():
    from dynamix.core.ssa2d import ssa2d

    calls = []
    ssa2d(_img(40, 44), rows_window=8, cols_window=8, n_components=6,
          progress=lambda s, f: calls.append(f))
    assert len(calls) >= 10 and calls[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-12 for a, b in zip(calls, calls[1:]))


# ------------------------------------------- the device

@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(ny=30, nx=34):
    return RasterField(name="f", values=_img(ny, nx), frame=LocalFrame(),
                       x_axis=np.arange(float(nx)), y_axis=np.arange(float(ny)))


def _resolve(cache, field, **params):
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    layer = Layer(layer_id=1, name="L", source_id="s",
                  chain=Chain((DeviceRef("ssa2d", params),)).materialized())
    return resolve(layer, field, cache)


def test_the_ssa2d_device_flips_every_view_without_recomputing(builtins):
    from dynamix.engine.cache import Cache

    field, cache = _field(), Cache()
    base = {"rows_window": 6, "cols_window": 6, "n_components": 8}
    first = _resolve(cache, field, **base)
    c3 = _resolve(cache, field, **base, show="component", component=3)
    grp = _resolve(cache, field, **base, show="recon", group="1-2, 4")
    res = _resolve(cache, field, **base, show="residual")
    assert first.cache_misses == 1
    assert c3.cache_misses == grp.cache_misses == res.cache_misses == 0
    comps = c3.result["ssa_components"]
    np.testing.assert_array_equal(c3.result["raster_out"], comps[2])
    # stored at the app's precision (float32 at 32 bit): identities hold to that precision
    tol = 1e-5 * np.abs(field.values).max()
    np.testing.assert_allclose(grp.result["raster_out"], comps[0] + comps[1] + comps[3],
                               rtol=0, atol=tol)
    np.testing.assert_allclose(res.result["raster_out"] + first.result["raster_out"],
                               field.values, rtol=0, atol=tol)


def test_the_residual_is_the_data_minus_the_chosen_group(builtins):
    """Removing just the first component: the residual is the data minus C1, not minus the
    whole kept reconstruction."""
    from dynamix.engine.cache import Cache

    field, cache = _field(), Cache()
    base = {"rows_window": 6, "cols_window": 6, "n_components": 8}
    whole = _resolve(cache, field, **base, show="residual", group="all")
    minus_c1 = _resolve(cache, field, **base, show="residual", group="1")
    minus_some = _resolve(cache, field, **base, show="residual", group="2-3, 5")
    assert minus_c1.cache_misses == minus_some.cache_misses == 0
    comps = whole.result["ssa_components"]
    tol = 1e-5 * np.abs(field.values).max()                  # stored at the app's precision
    np.testing.assert_allclose(minus_c1.result["raster_out"], field.values - comps[0],
                               atol=tol)
    np.testing.assert_allclose(minus_some.result["raster_out"],
                               field.values - comps[1] - comps[2] - comps[4], atol=tol)
    np.testing.assert_allclose(whole.result["raster_out"],
                               field.values - comps.sum(axis=0), atol=tol)
    share = whole.result["ssa_eigen_share"]
    assert minus_c1.result["_view_note"] == f"data − [1] ({100 * share[0]:.0f}%)"
    assert whole.result["_view_note"] == "residual"


def test_a_group_reconstruction_is_labelled_with_its_members(builtins):
    from dynamix.engine.cache import Cache

    field, cache = _field(), Cache()
    base = {"rows_window": 6, "cols_window": 6, "n_components": 8}
    out = _resolve(cache, field, **base, show="recon", group="3-1, 5")
    share = out.result["ssa_eigen_share"]
    pct = 100 * (share[0] + share[1] + share[2] + share[4])
    assert out.result["_view_note"] == f"recon [1-3, 5] ({pct:.0f}%)"
    assert _resolve(cache, field, **base, show="recon", group="all").result["_view_note"] \
        == "recon (8 components)"
    bad = _resolve(cache, field, **base, show="recon", group="zz")
    np.testing.assert_array_equal(bad.result["raster_out"], out.result["ssa_recon"])
    assert bad.result["_view_note"] == "recon (8 components; no valid group)"


@pytest.mark.parametrize("text,n,expected", [
    ("all", 4, [0, 1, 2, 3]), ("", 4, [0, 1, 2, 3]), (" ALL ", 3, [0, 1, 2]),
    ("1-3, 5", 8, [0, 1, 2, 4]), ("3-1", 8, [0, 1, 2]), ("2; 2, 9", 4, [1]),
    ("x, 2", 4, [1]), ("zz", 4, []),
])
def test_parse_group(text, n, expected):
    from dynamix.devices.decompose import parse_group

    assert parse_group(text, n) == expected


def test_format_group_writes_compact_ranges():
    from dynamix.devices.decompose import format_group

    assert format_group([0, 1, 2, 4]) == "1-3, 5"
    assert format_group([3]) == "4"
    assert format_group([]) == ""


def test_the_ssa2d_device_refuses_an_oversized_window(builtins):
    from dynamix.model.device import get_device, validate_params

    with pytest.raises(ValueError, match="window"):
        validate_params(get_device("ssa2d"), {"rows_window": 64, "cols_window": 64,
                                              "solver": "dense"})
    validate_params(get_device("ssa2d"), {"rows_window": 64, "cols_window": 64})   # FFT: allowed
