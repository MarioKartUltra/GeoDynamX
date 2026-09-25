# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""2D-SSA's FFT solvers against the dense (lag-covariance) path.

The FFT solvers never form the Hankel-block-Hankel matrix W: products W v and W^T u are 2-D
correlations of the image done by FFT at its own size (Korobeynikov 2010; Golyandina et al.
2015), and only the k leading eigentriples are found -- by Lanczos on the implicit W W^T, by
PROPACK's bidiagonalization of W, or by a randomized range finder whose power iterations are
re-orthonormalized every pass (Halko et al. 2011; the pragmatic SSA of Lopes et al. 2024). The
FFTs run at the app's precision, so agreement is to float32 tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.ssa2d import ssa2d

_FFT_SOLVERS = ("lanczos", "propack", "randomized")


def _field(ny=64, nx=72, seed=5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((ny, nx)).cumsum(0).cumsum(1) + 3.0 * rng.standard_normal((ny, nx))


@pytest.mark.parametrize("solver", ("lanczos", "propack"))
def test_the_exact_fft_solvers_agree_with_dense(solver):
    F = _field()
    ref = ssa2d(F, rows_window=8, cols_window=9, n_components=6)
    out = ssa2d(F, rows_window=8, cols_window=9, n_components=6, solver=solver)
    assert out["solver_used"] == solver and ref["solver_used"] == "dense"
    np.testing.assert_allclose(out["eigen_share"], ref["eigen_share"], rtol=1e-5)
    scale = np.abs(ref["components"]).max()
    np.testing.assert_allclose(out["components"], ref["components"], rtol=0, atol=1e-4 * scale)
    np.testing.assert_allclose(out["eigenarrays"], ref["eigenarrays"], rtol=0, atol=1e-4)
    np.testing.assert_allclose(out["w_correlation"], ref["w_correlation"], rtol=0, atol=1e-4)


def test_randomized_with_power_iterations_agrees_on_a_decaying_spectrum():
    F = _field()
    ref = ssa2d(F, rows_window=8, cols_window=9, n_components=6)
    out = ssa2d(F, rows_window=8, cols_window=9, n_components=6, solver="randomized",
                power_iterations=2)
    np.testing.assert_allclose(out["eigen_share"], ref["eigen_share"], rtol=1e-3)
    scale = np.abs(ref["components"]).max()
    np.testing.assert_allclose(out["components"][:3], ref["components"][:3], rtol=0,
                               atol=1e-3 * scale)


def test_randomized_is_reproducible():
    F = _field()
    a = ssa2d(F, rows_window=8, cols_window=9, n_components=4, solver="randomized")
    b = ssa2d(F, rows_window=8, cols_window=9, n_components=4, solver="randomized")
    np.testing.assert_array_equal(a["components"], b["components"])


@pytest.mark.parametrize("solver", _FFT_SOLVERS)
@pytest.mark.parametrize("field,rank", [("oblique", 2), ("product", 4), ("exponents", 2)])
def test_the_papers_ranks_hold_for_every_solver(solver, field, rank):
    """Golyandina & Usevich section 4: a degenerate pair has no unique vectors, so the check is
    the rank and the pair's summed reconstruction, not the individual components."""
    k, l = np.mgrid[:40, :44].astype(float)
    F = {"oblique": np.cos(2 * np.pi * (0.07 * k + 0.11 * l)),
         "product": np.cos(2 * np.pi * 0.07 * k) * np.cos(2 * np.pi * 0.11 * l),
         "exponents": 0.98 ** k * 1.01 ** l + 2.0 * 1.02 ** k * 0.97 ** l}[field]
    out = ssa2d(F, rows_window=10, cols_window=12, n_components=8, solver=solver)
    assert out["eigen_share"][:rank].sum() > 1 - 1e-5
    assert out["eigen_share"][rank] < 1e-5
    np.testing.assert_allclose(out["components"][:rank].sum(axis=0), F, rtol=0,
                               atol=1e-4 * np.abs(F).max())


@pytest.mark.parametrize("solver", ("dense",) + _FFT_SOLVERS)
def test_every_eigenarray_has_its_largest_entry_positive(solver):
    out = ssa2d(_field(), rows_window=8, cols_window=9, n_components=5, solver=solver)
    for psi in out["eigenarrays"]:
        assert psi.flat[np.argmax(np.abs(psi))] > 0


def test_a_window_too_large_for_dense_runs_with_the_fft_solvers():
    F = _field(160, 150)
    out = ssa2d(F, rows_window=48, cols_window=48, n_components=4, solver="lanczos")
    assert out["solver_used"] == "lanczos"
    assert out["components"].shape == (4, 160, 150)
    assert 0.0 < out["eigen_share"].sum() <= 1.0 + 1e-6


@pytest.mark.parametrize("solver", _FFT_SOLVERS)
def test_a_tiny_window_hands_over_to_dense(solver):
    out = ssa2d(_field(), rows_window=3, cols_window=3, n_components=16, solver=solver)
    assert out["solver_used"] == "dense"
    assert len(out["eigen_share"]) == 9


def test_propack_failure_is_a_clear_error(monkeypatch):
    import scipy.sparse.linalg._svdp as svdp

    def boom(*a, **k):
        raise np.linalg.LinAlgError("did not converge")

    monkeypatch.setattr(svdp, "_svdp", boom)
    with pytest.raises(ValueError, match="Lanczos"):
        ssa2d(_field(), rows_window=8, cols_window=9, n_components=4, solver="propack")


def test_an_unknown_solver_is_refused():
    with pytest.raises(ValueError, match="solver"):
        ssa2d(_field(), rows_window=8, cols_window=9, solver="toeplitz")


# ------------------------------------------- the energy threshold (Lopes et al. 2024, 2.3.3)

@pytest.mark.parametrize("solver", ("dense", "lanczos"))
def test_the_energy_threshold_keeps_the_fewest_components_that_reach_it(solver):
    F = _field() - _field().mean()               # centred, so the energy is spread over several
    full = ssa2d(F, rows_window=8, cols_window=9, n_components=40, solver="dense")
    cum = np.cumsum(full["eigen_share"])
    m = int(np.searchsorted(cum, 0.9) + 1)
    out = ssa2d(F, rows_window=8, cols_window=9, n_components=40, solver=solver,
                keep="energy", energy_threshold=0.9)
    assert len(out["eigen_share"]) == m and out["energy_reached"]
    assert out["components"].shape[0] == m


def test_the_energy_threshold_says_when_k_was_not_enough():
    out = ssa2d(_field(), rows_window=8, cols_window=9, n_components=2, solver="lanczos",
                keep="energy", energy_threshold=0.9999)
    assert len(out["eigen_share"]) == 2 and not out["energy_reached"]


# ------------------------------------------- automatic grouping (Lopes et al. 2024, 2.3.5)

def _two_waves(ny=96, nx=100, seed=2):
    k, l = np.mgrid[:ny, :nx].astype(float)
    rng = np.random.default_rng(seed)
    clean = (2.0 * np.cos(2 * np.pi * (0.05 * k + 0.08 * l))
             + np.cos(2 * np.pi * (0.21 * k - 0.13 * l)))
    return clean, clean + 0.2 * rng.standard_normal(k.shape)


def test_two_waves_cluster_into_their_pairs():
    from dynamix.core.ssa2d import cluster_components

    _clean, F = _two_waves()
    out = ssa2d(F, rows_window=12, cols_window=12, n_components=6, solver="lanczos")
    clusters, Z = cluster_components(out["w_correlation"], out["eigen_share"], 0.5)
    assert clusters[:2] == [[0, 1], [2, 3]]                 # the strong wave, then the weak one
    assert Z.shape == (5, 4)
    single, _ = cluster_components(np.ones((1, 1)), [1.0])
    assert single == [[0]]


# ------------------------------------------- the ssa2d tool

@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _raster(values):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    ny, nx = values.shape
    return RasterField(name="f", values=values, frame=LocalFrame(),
                       x_axis=np.arange(float(nx)), y_axis=np.arange(float(ny)))


def test_the_tool_defaults_to_lanczos_with_a_one_window_margin(builtins):
    from dynamix.model.device import defaults_for, get_device

    dev = get_device("ssa2d")
    d = defaults_for(dev)
    assert d["solver"] == "lanczos" and d["keep"] == "count" and d["margin_windows"] == 1.0
    p = {q.name: q for q in dev.params}
    assert p["solver"].choices == ("lanczos", "propack", "randomized", "dense")
    assert p["power_iterations"].active_when == ("solver", ("randomized",))
    assert p["energy_threshold"].active_when == ("keep", ("energy",))
    assert p["cluster"].view and p["cluster_distance"].view and "cluster" in p["show"].choices
    assert dev.roi_margin(d) == 15                              # one 16 x 16 window: L - 1
    assert dev.roi_margin({**d, "margin_windows": 0.0}) == 0
    assert dev.roi_margin({**d, "rows_window": 8, "cols_window": 20, "margin_windows": 2.0}) == 38


def test_only_the_dense_solver_is_capped(builtins):
    from dynamix.model.device import defaults_for, get_device, validate_params

    dev = get_device("ssa2d")
    p = {**defaults_for(dev), "rows_window": 64, "cols_window": 64}
    validate_params(dev, p)
    with pytest.raises(ValueError, match="dense"):
        validate_params(dev, {**p, "solver": "dense"})


def test_a_whole_field_run_is_the_plain_decomposition_unless_mirrored(builtins):
    """Mirror extension (Lopes et al.'s signal extension) is an option, off by default: in 2-D a
    reflected wave is a different wave (an oblique one flips a frequency component), so the
    extension adds rank and raised the edge error of two test waves 3-18x."""
    from dynamix.model.device import defaults_for, get_device

    dev = get_device("ssa2d")
    _clean, F = _two_waves()
    p = {**defaults_for(dev), "rows_window": 12, "cols_window": 12, "n_components": 4}
    assert p["edge_extension"] == "none"
    plain = ssa2d(F, rows_window=12, cols_window=12, n_components=4, solver="lanczos")
    res = dev.compute(_raster(F), p)
    np.testing.assert_allclose(res["ssa_recon"], plain["recon"], rtol=0, atol=1e-4)
    mirrored = dev.compute(_raster(F), {**p, "edge_extension": "mirror"})
    assert mirrored["ssa_components"].shape == (4,) + F.shape
    assert not np.allclose(mirrored["ssa_recon"], res["ssa_recon"], atol=1e-3)


def test_the_real_data_margin_lowers_an_rois_edge_error(builtins):
    from dynamix.model.device import defaults_for, get_device
    from dynamix.roi.runner import run_on_region

    dev = get_device("ssa2d")
    clean, F = _two_waves(160, 170)
    p = {**defaults_for(dev), "rows_window": 12, "cols_window": 12, "n_components": 4}
    rect = (50, 60, 60, 50)

    def edge_rms(margin):
        res = run_on_region([(dev, {**p, "margin_windows": margin})], _raster(F), rect)
        err = np.asarray(res["raster_out"], dtype=np.float64) - clean[50:110, 60:110]
        band = np.ones(err.shape, bool)
        band[8:-8, 8:-8] = False
        return np.sqrt(np.mean(err[band] ** 2))

    assert edge_rms(1.0) < 0.95 * edge_rms(0.0)


def test_an_roi_run_reads_the_margin_and_returns_the_roi(builtins):
    from dynamix.model.device import defaults_for, get_device
    from dynamix.roi.runner import run_on_region

    dev = get_device("ssa2d")
    _clean, F = _two_waves()
    p = {**defaults_for(dev), "rows_window": 10, "cols_window": 10, "n_components": 4}
    res = run_on_region([(dev, p)], _raster(F), (30, 30, 40, 36))
    assert res["_roi"]["margin"] == 9
    assert res["raster_out"].shape == (40, 36)
    assert res["ssa_components"].shape == (4, 40, 36)


def test_stored_arrays_follow_the_fft_precision(builtins):
    from dynamix.core.fft_policy import active
    from dynamix.model.device import defaults_for, get_device

    dev = get_device("ssa2d")
    res = dev.compute(_raster(_field(40, 44)), {**defaults_for(dev), "rows_window": 6,
                                                "cols_window": 6, "n_components": 4})
    want = np.float32 if getattr(active(), "precision", 64) == 32 else np.float64
    for key in ("ssa_recon", "ssa_residual", "ssa_components", "ssa_eigenarrays"):
        assert res[key].dtype == want, key
    assert res["ssa_solver_used"] == "lanczos"


def test_the_cluster_view_shows_a_cluster_without_recomputing(builtins):
    from dynamix.model.device import defaults_for, get_device

    dev = get_device("ssa2d")
    _clean, F = _two_waves()
    p = {**defaults_for(dev), "rows_window": 12, "cols_window": 12, "n_components": 6,
         "margin_windows": 0.0}
    res = dev.compute(_raster(F), p)
    v = dev.view(res, {**p, "show": "cluster", "cluster": 2, "cluster_distance": 0.5})
    comps = np.asarray(res["ssa_components"], dtype=np.float64)
    np.testing.assert_allclose(v["raster_out"], comps[2] + comps[3], rtol=1e-6, atol=1e-6)
    assert v["_view_note"].startswith("cluster 2/3: 3-4")
    assert dev.cache_key("s", p) == dev.cache_key("s", {**p, "show": "cluster", "cluster": 2})


def test_the_row_note_says_when_the_energy_threshold_was_not_reached(builtins):
    from dynamix.model.device import defaults_for, get_device

    dev = get_device("ssa2d")
    p = {**defaults_for(dev), "rows_window": 8, "cols_window": 9, "n_components": 2,
         "keep": "energy", "energy_threshold": 0.9999}
    res = dev.compute(_raster(_field()), p)
    assert res["ssa_energy_reached"] is False
    assert dev.view(res, p)["_view_note"].endswith("energy threshold not reached")
    reached = dev.compute(_raster(_field()), {**p, "n_components": 40, "energy_threshold": 0.5})
    assert "not reached" not in dev.view(reached, p)["_view_note"]
