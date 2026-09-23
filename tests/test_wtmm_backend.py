# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""tests/test_wtmm_backend.py (part 1)"""
import sys
import numpy as np
import pytest
from dynamix.core.wtmm_backend import DEFAULT_Q, PythonWTMMBackend, compute_scales2d, get_backend


def test_scales_match_wtmm_ebsd():
    from dynamix._vendor.wtmm_ebsd import cwt2d as ref
    np.testing.assert_allclose(compute_scales2d(3, 4, 1.0), np.asarray(ref.compute_scales(3, 4, 1.0)))

def test_import_is_lazy():
    # dynamix.core.wtmm_backend must not drag in heavy deps at import time (layering test
    # covers GUI; this covers mlx/wtmm/wtmm_ebsd/matplotlib) — fresh-subprocess probe.
    import subprocess
    from pathlib import Path
    code = ("import sys, dynamix.core.wtmm_backend; "
            "bad = [m for m in ('mlx', 'wtmm', 'wtmm_ebsd', 'dynamix._vendor.wtmm', "
            "'dynamix._vendor.wtmm_ebsd', 'matplotlib', 'scipy') "
            "if m in sys.modules]; sys.exit(1 if bad else 0)")
    assert subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
    ).returncode == 0

def test_cwt2d_shapes(fbm64):
    b = PythonWTMMBackend()
    scales = compute_scales2d(2, 3)
    out = b.cwt2d(fbm64, scales)
    assert out["mod"].shape == (6, 64, 64) and out["arg"].shape == (6, 64, 64)
    assert np.all(np.isfinite(out["mod"]))

def test_cwt2d_numpy_engine_parity(fbm64):
    pytest.importorskip("mlx.core")
    b = PythonWTMMBackend()
    scales = compute_scales2d(2, 3)
    ref = b.cwt2d(fbm64, scales)                       # mlx engine (auto)
    alt = b.cwt2d(fbm64, scales, _engine="numpy")      # forced fallback (test hook)
    np.testing.assert_allclose(alt["mod"], ref["mod"], rtol=1e-3, atol=1e-5)

def test_cwt2d_nan_filled(fbm64):
    f = fbm64.copy(); f[10:12, 10:12] = np.nan
    out = PythonWTMMBackend().cwt2d(f, compute_scales2d(1, 2))
    assert np.all(np.isfinite(out["mod"]))

def test_cwt2d_unknown_wavelet_raises(fbm64):
    # Real default path: with mlx installed (the target env), cwt2d auto-selects
    # the mlx engine, which calls wtmm_ebsd's cwt_2d_f32 directly -- bypassing
    # _build_wavelet_filters_numpy entirely. The guard must live at cwt2d's own
    # entry, before engine dispatch, or this default path silently computes
    # gaussian for any unknown wavelet name. No _engine override here on purpose.
    b = PythonWTMMBackend()
    scales = compute_scales2d(1, 1)
    with pytest.raises(ValueError) as excinfo:
        b.cwt2d(fbm64, scales, wavelet="morlet")
    msg = str(excinfo.value)
    assert "morlet" in msg
    assert "gaussian" in msg and "mexican" in msg

def test_cwt2d_unknown_wavelet_raises_numpy_engine(fbm64):
    # Secondary: the numpy fallback engine's own builder-level guard
    # (_build_wavelet_filters_numpy) is defense for any direct caller, kept even
    # though cwt2d's entry guard now covers this engine too.
    b = PythonWTMMBackend()
    scales = compute_scales2d(1, 1)
    with pytest.raises(ValueError, match="morlet"):
        b.cwt2d(fbm64, scales, wavelet="morlet", _engine="numpy")

def test_cwt2d_known_wavelets_still_work(fbm64):
    b = PythonWTMMBackend()
    scales = compute_scales2d(1, 1)
    for wavelet in ("gaussian", "mexican"):
        out = b.cwt2d(fbm64, scales, wavelet=wavelet)
        assert np.all(np.isfinite(out["mod"]))

def test_tensor2d_unknown_wavelet_raises(fbm3_64):
    # tensor2d forwards wavelet into wtmm_ebsd.twtmm.alpha_jacobian_twtmm along
    # with the injected _resolve_cwt2d_engine() (mlx-first, same unguarded shape
    # as cwt2d) -- validate at tensor2d's own entry, before any heavy compute.
    b = PythonWTMMBackend()
    scales = compute_scales2d(1, 1)
    with pytest.raises(ValueError) as excinfo:
        b.tensor2d(fbm3_64, scales, wavelet="morlet")
    msg = str(excinfo.value)
    assert "morlet" in msg
    assert "gaussian" in msg and "mexican" in msg

def test_tensor2d_unknown_wavelet_hessian_raises(fbm3_64):
    # wavelet_hessian is forwarded the same way (derivs="all" Hessian branch) and
    # is exposed to the identical silent-fallthrough risk.
    b = PythonWTMMBackend()
    scales = compute_scales2d(1, 1)
    with pytest.raises(ValueError, match="morlet"):
        b.tensor2d(fbm3_64, scales, wavelet_hessian="morlet")

def test_get_backend():
    assert get_backend().name == "python"
    with pytest.warns(UserWarning):
        assert get_backend("xsmurf").name == "python"   # Phase C not present -> fallback
    with pytest.raises(ValueError):
        get_backend("banana")

def test_cwt2d_numpy_all_derivs_parity(fbm64):
    pytest.importorskip("mlx.core")
    from dynamix._vendor.wtmm_ebsd.cwt2d import cwt_2d_f32
    from dynamix.core.wtmm_backend import _cwt2d_numpy
    scales = compute_scales2d(1, 2)
    ref = cwt_2d_f32(fbm64, scales, derivs="all", verbose=False)
    alt = _cwt2d_numpy(fbm64, scales, derivs="all")
    for k in ("dx", "dy", "dxx", "dxy", "dyy"):
        np.testing.assert_allclose(alt[k], ref[k], rtol=1e-3, atol=1e-5)

def test_tensor_svd_3x2_parity(fbm64):
    pytest.importorskip("mlx.core")
    from dynamix._vendor.wtmm_ebsd.cwt2d import tensor_svd_3x2 as ref_svd
    from dynamix.core.wtmm_backend import _tensor_svd_3x2_numpy
    rng = np.random.default_rng(0)
    g = [rng.normal(size=(16, 16)) for _ in range(6)]
    smax, smin, arg = _tensor_svd_3x2_numpy(*g)
    rmax, rmin, rarg = ref_svd(*g)
    np.testing.assert_allclose(smax, rmax, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(smin, rmin, rtol=1e-4, atol=1e-6)

def test_extrema2d_are_directional_maxima(fbm64):
    b = PythonWTMMBackend(); scales = compute_scales2d(2, 3)
    cwt = b.cwt2d(fbm64, scales)
    ext = b.extrema2d(cwt, scales)
    assert len(ext) == len(scales)
    e0 = ext[0]
    assert e0["x"].dtype == np.int64 and e0["mod"].ndim == 1
    assert len(e0["x"]) > 10                       # fBm has plenty of gradient ridges
    # every extremum beats its two gradient-line neighbours (re-check independently)
    from scipy.ndimage import map_coordinates
    m = cwt["mod"][0].astype(float)
    cx, sx = np.cos(e0["arg"]), np.sin(e0["arg"])
    fore = map_coordinates(m, [e0["y"] + sx, e0["x"] + cx], order=1, mode="nearest")
    back = map_coordinates(m, [e0["y"] - sx, e0["x"] - cx], order=1, mode="nearest")
    ok = (e0["mod"] >= fore - 1e-6) & (e0["mod"] >= back - 1e-6)
    assert ok.mean() > 0.99

def test_extrema2d_threshold_and_lines(fbm64):
    b = PythonWTMMBackend(); scales = compute_scales2d(1, 2)
    cwt = b.cwt2d(fbm64, scales)
    lo = b.extrema2d(cwt, scales, thresh=1e-3)
    hi = b.extrema2d(cwt, scales, thresh=0.3)
    assert len(hi[0]["x"]) < len(lo[0]["x"])
    assert set(np.unique(lo[0]["line_id"])) - {-1}          # some connected lines exist

def test_extrema2d_nan_exclusion(fbm64):
    f = fbm64.copy(); f[20:24, 20:24] = np.nan
    b = PythonWTMMBackend(); scales = compute_scales2d(1, 2)
    ext = b.extrema2d(b.cwt2d(f, scales), scales, field=f)
    for s, e in enumerate(ext):
        r = int(np.ceil(scales[s]))
        inside = ((e["x"] >= 20 - r) & (e["x"] < 24 + r)
                  & (e["y"] >= 20 - r) & (e["y"] < 24 + r))
        assert not inside.any()


"""tests/test_wtmm_backend.py (part 3): chains2d, partition2d, 1D delegation"""


def _pipeline(fbm64, n_oct=2, n_voice=3):
    b = PythonWTMMBackend(); scales = compute_scales2d(n_oct, n_voice)
    cwt = b.cwt2d(fbm64, scales)
    ext = b.extrema2d(cwt, scales)
    return b, scales, b.chains2d(ext, scales)

def test_chains2d_schema(fbm64):
    b, scales, chains = _pipeline(fbm64)
    assert len(chains) > 5
    ch = max(chains, key=lambda c: len(c["mod"]))
    for k in ("x", "y", "mod", "log2_mod", "log2_scales"):
        assert k in ch and len(ch[k]) == len(ch["mod"])
    assert len(ch["mod"]) >= 2                       # min_len honored
    np.testing.assert_allclose(ch["log2_scales"], np.log2(scales[: len(ch["mod"])]))
    np.testing.assert_allclose(ch["log2_mod"], np.log2(np.abs(ch["mod"])))

def test_chains2d_consumable_by_wtmm_ebsd(fbm64):
    from dynamix._vendor.wtmm_ebsd.chain_filters import filter_by_length
    from dynamix._vendor.wtmm_ebsd.partition import build_hd_from_chains
    b, scales, chains = _pipeline(fbm64)
    kept, dropped = filter_by_length(chains, 3)
    assert len(kept) + len(dropped) == len(chains)
    hd_std, hd_cmax = b.partition2d(chains, scales)
    n_q = len(hd_std["q_list"]); n_sc = len(scales)
    assert hd_std["tau_qa"].shape == (n_q, n_sc) == hd_cmax["tau_qa"].shape
    # partition2d must equal a direct build_hd_from_chains call
    ref_std, _ = build_hd_from_chains(chains, scales, hd_std["q_list"])
    np.testing.assert_allclose(hd_std["tau_qa"], ref_std["tau_qa"], equal_nan=True)

def test_fbm_holder_recovered(fbm64):
    # Slope of <log2 max|T|> vs log2(a) over the mid-scale range ≈ H (loose tolerance:
    # 64px, few octaves). Guards against gross normalization errors in the pipeline.
    b, scales, chains = _pipeline(fbm64, n_oct=3, n_voice=4)
    m = np.full((len(chains), len(scales)), np.nan)
    for i, ch in enumerate(chains):                  # test-only loop, fine
        m[i, : len(ch["mod"])] = np.maximum.accumulate(np.abs(ch["mod"]))
    mean_log = np.nanmean(np.log2(m), axis=0)
    x = np.log2(scales); sel = np.isfinite(mean_log)
    slope = np.polyfit(x[sel][2:-2], mean_log[sel][2:-2], 1)[0]
    assert 0.3 < slope < 1.1                         # H=0.7 target, wide but real guard

def _extrema_scale(xs, ys, mods, line_ids=0):
    """One synthetic scale from explicit pixel positions + moduli."""
    n = len(xs)
    return [{
        "x": np.asarray(xs, dtype=np.int64),
        "y": np.asarray(ys, dtype=np.int64),
        "mod": np.asarray(mods, dtype=np.float64),
        "arg": np.zeros(n),
        "line_id": (np.full(n, line_ids, dtype=np.int64)
                    if np.isscalar(line_ids) else
                    np.asarray(line_ids, dtype=np.int64)),
    }]

def _straight_line_extrema(mods, line_id=0):
    """One synthetic scale: a straight horizontal maxima line with given moduli."""
    n = len(mods)
    return _extrema_scale(np.arange(n), np.zeros(n), mods, line_id)

def test_single_maxima2d_subset_and_local_max(fbm64):
    b = PythonWTMMBackend(); scales = compute_scales2d(2, 3)
    ext = b.extrema2d(b.cwt2d(fbm64, scales), scales)
    sm = b.single_maxima2d(ext)
    assert len(sm) == len(ext)
    for e, s in zip(ext, sm):
        for k in ("x", "y", "mod", "arg", "line_id"):
            assert k in s and len(s[k]) == len(s["x"])
        assert 0 < len(s["x"]) < len(e["x"])            # a strict, non-empty subset
        pos_e = set(zip(e["x"].tolist(), e["y"].tolist()))
        assert set(zip(s["x"].tolist(), s["y"].tolist())) <= pos_e
        # `_is_single_max_` (a): a singleton has 0 neighbour runs -> never a WTMMM
        assert not np.any(s["line_id"] < 0)

def test_single_maxima2d_along_line_exact():
    # `_is_single_max_` on a straight line: only INTERIOR points have the required
    # 2 neighbour runs, and each must dominate its 8-neighbour extrema.
    b = PythonWTMMBackend()
    sm = b.single_maxima2d(_straight_line_extrema([1.0, 5.0, 2.0, 3.0, 9.0]),
                           smooth=False)[0]
    np.testing.assert_array_equal(sm["x"], [1])         # x=4 is the max but an END
    np.testing.assert_allclose(sm["mod"], [5.0])
    # the line's maximum sitting at an end: NOTHING qualifies (no interior point
    # beats its neighbours, and the end itself has only 1 run)
    ramp = b.single_maxima2d(_straight_line_extrema([1.0, 2.0, 3.0, 4.0, 9.0]),
                             smooth=False)[0]
    assert len(ramp["x"]) == 0
    # a plateau: interior points qualify on `>=` (as xsmurf), the two ends do not
    flat = b.single_maxima2d(_straight_line_extrema([2.0, 2.0, 2.0]), smooth=False)[0]
    np.testing.assert_array_equal(flat["x"], [1])

def test_single_maxima2d_rejects_ends_singletons_junctions():
    b = PythonWTMMBackend()
    # (a) singletons (line_id == -1) have 0 neighbour runs -> rejected
    solo = _straight_line_extrema([3.0, 1.0, 4.0])
    solo[0]["x"] = np.array([0, 5, 10], dtype=np.int64)          # far apart
    solo[0]["line_id"] = np.array([-1, -1, -1], dtype=np.int64)
    assert len(b.single_maxima2d(solo)[0]["x"]) == 0
    # (b) a 2-point line is two ends (1 run each) -> nothing qualifies
    pair = b.single_maxima2d(_straight_line_extrema([9.0, 1.0]), smooth=False)[0]
    assert len(pair["x"]) == 0
    # (c) a T-junction: (2,1) has 3 neighbour runs -> rejected despite being the max
    tee = _extrema_scale([0, 1, 2, 3, 4, 2], [1, 1, 1, 1, 1, 2],
                         [1.0, 2.0, 9.0, 2.0, 1.0, 1.0])
    assert len(b.single_maxima2d(tee, smooth=False)[0]["x"]) == 0
    # ... and the SAME point qualifies once the junction stub is removed (2 runs)
    plain = _extrema_scale([0, 1, 2, 3, 4], [1, 1, 1, 1, 1],
                           [1.0, 2.0, 9.0, 2.0, 1.0])
    np.testing.assert_array_equal(b.single_maxima2d(plain, smooth=False)[0]["x"], [2])

def test_single_maxima2d_smoothing_fills_dips():
    b = PythonWTMMBackend()
    dip = _straight_line_extrema([5.0, 1.0, 5.0])
    raw = b.single_maxima2d(dip, smooth=False)[0]
    smoothed = b.single_maxima2d(dip, smooth=True)[0]
    assert len(raw["x"]) == 0                           # 1 < 5: not a single max
    np.testing.assert_array_equal(smoothed["x"], [1])   # (5+5)/2 = 5 >= 5 -> kept
    # the SMOOTHED modulus is what gets reported (xsmurf overwrites e->mod in place)
    np.testing.assert_allclose(smoothed["mod"], [5.0])
    # the smoothed mod also feeds the neighbour comparison of OTHER points
    deep = b.single_maxima2d(_straight_line_extrema([3.0, 1.0, 4.0]), smooth=True)[0]
    assert len(deep["x"]) == 0                          # (3+4)/2 = 3.5 < 4

def test_chains2d_terminate_and_never_merge(fbm64):
    # The xsmurf vchain port makes losing chains TERMINATE, so maxima lines merge:
    # chain lengths must spread, N(a) must decay, and no coarse position may be
    # claimed twice.
    b, scales, chains = _pipeline(fbm64)
    n_sc = len(scales)
    lengths = np.array([len(c["mod"]) for c in chains])
    # (a) not every chain spans every scale
    assert len(np.unique(lengths)) > 1 or lengths.max() < n_sc
    # (b) N(a) = #chains alive at scale s is non-increasing and really decays
    n_alive = np.array([(lengths > s).sum() for s in range(n_sc)])
    assert np.all(np.diff(n_alive) <= 0)
    assert n_alive[-1] < n_alive[0]
    # (c) a coarser single maximum is claimed by at most one chain
    keys = [(s, int(x), int(y)) for c in chains
            for s, (x, y) in enumerate(zip(c["x"], c["y"]))]
    assert len(keys) == len(set(keys))
    # (d) min_len honored (small distance cap -> plenty of short chains)
    short = b.chains2d(b.extrema2d(b.cwt2d(fbm64, scales), scales), scales,
                       dist2_max=2.0)
    long_only = b.chains2d(b.extrema2d(b.cwt2d(fbm64, scales), scales), scales,
                           dist2_max=2.0, min_len=3)
    assert min(len(c["mod"]) for c in short) == 2       # some chains are minimal
    assert min(len(c["mod"]) for c in long_only) >= 3
    assert len(long_only) < len(short)

def test_chains2d_tau0_is_negative(fbm64):
    # tau(0) = -D_F: with merging chains, log2 Z(0, a) = log2 N(a) must decay with
    # scale. The pre-fix tree-style linking gave a dead-flat 0.0 here.
    b, scales, chains = _pipeline(fbm64)
    hd_std, _ = b.partition2d(chains, scales)
    qi = int(np.argmin(np.abs(hd_std["q_list"])))
    tau0 = hd_std["tau_qa"][qi]
    log2_a = hd_std["log2_scales"]
    mid = slice(1, len(scales) - 1)
    ok = np.isfinite(tau0[mid])
    slope = np.polyfit(log2_a[mid][ok], tau0[mid][ok], 1)[0]
    assert slope <= -0.5, f"tau(0) slope {slope} is not clearly negative"

def test_1d_delegation():
    from dynamix._vendor.wtmm.signals import ucantor
    sig, _ = ucantor(512, np.array([0.5, 0.5]), np.array([0.4, 0.6]))
    b = PythonWTMMBackend()
    coeffs, scales, vr = b.cwt1d(np.cumsum(sig), 1.0, 3, 4)
    assert coeffs.shape == (12, 512) and len(vr) == 12
    ea, eo, ei = b.extrema1d(coeffs, scales, valid_ranges=vr)
    cl, fl = b.chains1d(ea, eo)
    pf = b.partition1d(eo, 3, 4, 1.0, [-1.0, 0.0, 2.0], coarser_links=cl)
    assert pf["sTq"].shape == (3, 12)

def test_cwt1d_numpy_matches_fftw():
    # Parity of the pure-numpy overlap-save fallback against the pyfftw
    # reference on a real (non-trivial-spectrum) test signal.
    pytest.importorskip("pyfftw")
    from dynamix._vendor.wtmm.cwt import cwtd_fftw
    from dynamix._vendor.wtmm.signals import ucantor
    from dynamix.core.wtmm_backend import _cwtd_numpy

    sig, _ = ucantor(256, np.array([0.5, 0.5]), np.array([0.4, 0.6]))
    sig = np.cumsum(sig)
    ref_coeffs, ref_scales, ref_vr = cwtd_fftw(sig, 1.0, 3, 4)
    alt_coeffs, alt_scales, alt_vr = _cwtd_numpy(sig, 1.0, 3, 4)
    np.testing.assert_allclose(alt_coeffs, ref_coeffs, rtol=1e-8)
    np.testing.assert_allclose(alt_scales, ref_scales)
    assert alt_vr == ref_vr


"""tests/test_wtmm_backend.py (part 4): tensor2d (alpha-Jacobian WTMM via the xsm shim)"""


@pytest.mark.parametrize("block_mlx", [False, True], ids=["mlx-if-present", "no-mlx"])
def test_tensor2d_runs_pure_python(fbm3_64, block_mlx):
    # end-to-end without xsmurf: forbid the import outright inside a fresh subprocess.
    # block_mlx=True additionally forbids `mlx`, pinning the injected CWT engine to
    # _cwt2d_numpy and forcing wtmm_ebsd.twtmm through the no-mlx loader fallback --
    # i.e. the universal-binary path this backend exists to guarantee.
    import subprocess, sys, textwrap
    from pathlib import Path
    code = f"BLOCK_MLX = {block_mlx!r}\n" + textwrap.dedent("""
        import sys
        sys.modules["xsmurf_wrapper"] = None            # any import attempt -> ImportError
        if BLOCK_MLX:
            sys.modules["mlx"] = sys.modules["mlx.core"] = None
        import numpy as np
        from dynamix.core.wtmm_backend import (PythonWTMMBackend, compute_scales2d,
                                           _cwt2d_numpy, _resolve_cwt2d_engine)
        if BLOCK_MLX:
            # 2026-09-22 FFT policy: without mlx the transcription runs on FFTW3 (a partial
            # carrying the policy backend), never on numpy's FFT
            eng = _resolve_cwt2d_engine()
            assert getattr(eng, "func", None) is _cwt2d_numpy, eng
            assert eng.keywords["fft"].name == "pyfftw", eng.keywords
        def fbm2d(n, H, seed):
            r = np.random.default_rng(seed)
            ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.fftfreq(n)[None, :]
            k = np.hypot(kx, ky); k[0, 0] = 1.0
            amp = k ** (-(H + 1.0)); amp[0, 0] = 0.0
            f = np.fft.ifft2(amp * np.exp(2j*np.pi*r.random((n, n)))).real
            return (f - f.mean()) / f.std()
        f3 = 0.05 * np.stack([fbm2d(64, 0.8, s) for s in (1, 2, 3)], axis=-1)
        b = PythonWTMMBackend()
        out = b.tensor2d(f3, compute_scales2d(2, 3))
        assert set(out) == {"sigma_max", "sigma_min", "alpha_sigma_max",
                            "alpha_sigma_min", "modL", "modT"}, set(out)
        for mode, r in out.items():
            assert len(r["chains"]) > 0, mode
            assert r["mod"].shape == (6, 64, 64)
            assert len(r["holders"]) == len(r["chains"])
        assert not any("xsmurf" in m for m in sys.modules if sys.modules[m] is not None)
        print("TENSOR_OK")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert "TENSOR_OK" in res.stdout, res.stderr

def test_tensor2d_chains_feed_partition(fbm3_64):
    from dynamix.core.wtmm_backend import PythonWTMMBackend, compute_scales2d
    b = PythonWTMMBackend(); scales = compute_scales2d(2, 3)
    out = b.tensor2d(fbm3_64, scales, svd_modes=("sigma_max",))
    hd_std, hd_cmax = b.partition2d(out["sigma_max"]["chains"], scales)
    assert np.isfinite(hd_std["tau_qa"]).any()

def test_tensor2d_fills_nan_per_component(fbm3_64):
    # RasterField.values explicitly permits NaN. Unfilled, the FFT smears one NaN
    # across the whole padded image -> all-NaN mod -> all-False NMS mask -> zero
    # chains, silently. The fill must be PER COMPONENT (each carries its own
    # offset), matching the module NaN policy.
    b = PythonWTMMBackend(); scales = compute_scales2d(2, 3)
    holed = fbm3_64.copy()
    holed[..., 1] += 10.0                      # component 1 far off the global mean
    holed[20:26, 30:38, 1] = np.nan            # a masked patch in that component only

    r = b.tensor2d(holed, scales, svd_modes=("sigma_max",))["sigma_max"]
    assert np.isfinite(r["mod"]).all()
    assert len(r["chains"]) > 0
    assert all(np.isfinite(e["mod"]).all() for e in r["extrema"])

    # identical to filling each component with its own finite mean by hand
    per_comp = np.where(np.isfinite(holed), holed, np.nanmean(holed, axis=(0, 1)))
    ref = b.tensor2d(per_comp, scales, svd_modes=("sigma_max",))["sigma_max"]
    np.testing.assert_array_equal(r["mod"], ref["mod"])

    # ...and NOT the same as a single global mean, which would punch a ~6.7-unit
    # step into the hole and manufacture spurious maxima around it
    glob = np.where(np.isfinite(holed), holed, np.nanmean(holed))
    bad = b.tensor2d(glob, scales, svd_modes=("sigma_max",))["sigma_max"]
    assert not np.allclose(r["mod"], bad["mod"])

def test_tensor2d_all_nan_component_raises(fbm3_64):
    b = PythonWTMMBackend(); scales = compute_scales2d(1, 2)
    dead = fbm3_64.copy(); dead[..., 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        b.tensor2d(dead, scales, svd_modes=("sigma_max",))
    with pytest.raises(ValueError, match="non-finite"):
        b.tensor2d(np.full((32, 32, 3), np.nan), scales, svd_modes=("sigma_max",))

def test_tensor2d_requires_3_components(fbm64):
    from dynamix.core.wtmm_backend import PythonWTMMBackend, compute_scales2d
    import pytest
    with pytest.raises(ValueError):
        PythonWTMMBackend().tensor2d(fbm64, compute_scales2d(1, 2))

def test_tensor2d_forwards_chain_params(fbm3_64):
    # similitude / dist2_max must reach chains2d THROUGH the injected chain adapter
    # (twtmm forwards `similitude`; box_ratio/dist2_max/min_len ride on the adapter).
    b = PythonWTMMBackend(); scales = compute_scales2d(2, 3)

    def run(**kw):
        return b.tensor2d(fbm3_64, scales, svd_modes=("sigma_max",), **kw)["sigma_max"]

    def sig(chains):                     # order-independent identity of a chain set
        return sorted((int(c["x"][0]), int(c["y"][0]), len(c["mod"])) for c in chains)

    base = run()["chains"]
    tight = run(similitude=0.99)["chains"]           # a very restrictive modulus band
    assert sig(base) != sig(tight)

    near = run(dist2_max=2.0)["chains"]              # ~1.41 px linking radius
    assert len(near) > 0
    assert sum(len(c["mod"]) for c in near) < sum(len(c["mod"]) for c in base)
    assert len(near) <= len(base)

def test_tensor2d_extrema_and_scales(fbm3_64):
    b = PythonWTMMBackend(); scales = compute_scales2d(2, 3)
    r = b.tensor2d(fbm3_64, scales, svd_modes=("modT",))["modT"]
    assert set(r) == {"chains", "holders", "extrema", "mod", "scales"}
    np.testing.assert_allclose(r["scales"], scales)
    assert len(r["extrema"]) == len(scales)
    for e in r["extrema"]:
        for k in ("x", "y", "mod", "arg", "line_id"):
            assert k in e and len(e[k]) == len(e["x"])
    # holders are the OLS log2|W| vs log2(a) slopes of the very same chains
    from dynamix._vendor.wtmm_ebsd.chain_filters import chain_ols_holder
    assert [h["h"] for h in r["holders"]] == [chain_ols_holder(c) for c in r["chains"]]

def test_xsm_shim_surface(fbm64):
    # the shim must expose EXACTLY what alpha_jacobian_twtmm touches, and its
    # ext-image must round-trip to the extrema2d dict schema.
    from dynamix.core.wtmm_backend import _XsmShim
    b = PythonWTMMBackend(); scales = compute_scales2d(1, 2)
    cwt = b.cwt2d(fbm64, scales)
    mod, arg = cwt["mod"][0].astype(float), cwt["arg"][0].astype(float)
    dx = (mod * np.cos(arg)).astype(np.float32)
    dy = (mod * np.sin(arg)).astype(np.float32)
    xim = _XsmShim.XImage.from_numpy(dx)
    assert xim.data.dtype == np.float32 and (xim.lx, xim.ly) == (64, 64)
    ext, a, c = _XsmShim.wtmm2d(xim, _XsmShim.XImage.from_numpy(dy),
                                float(scales[0]), thresh=1e-3)
    assert a is None and c is None
    arrs = ext.get_extrema_arrays()
    assert set(arrs) == {"x", "y", "pos", "mod", "arg"}
    np.testing.assert_array_equal(arrs["pos"], arrs["y"] * ext.lx + arrs["x"])
    assert ext.extr_nb == len(arrs["x"]) > 0
    # identical to the public extrema2d single-scale output
    ref = b.extrema2d({"mod": mod[None], "arg": arg[None]}, scales[:1])[0]
    np.testing.assert_array_equal(arrs["x"], ref["x"])
    np.testing.assert_allclose(arrs["mod"], ref["mod"])
    lines = ext.get_lines()
    assert lines and all(set(l) == {"extrema_pos"} for l in lines)
    # lines are ordered ALONG themselves: consecutive positions are 8-neighbours
    for l in lines:
        p = np.asarray(l["extrema_pos"], dtype=np.int64)
        dxp = np.abs(np.diff(p % ext.lx)); dyp = np.abs(np.diff(p // ext.lx))
        assert np.all(np.maximum(dxp, dyp) <= 1)
    # exactly the labelled (non-singleton) extrema, each used once
    covered = np.concatenate([l["extrema_pos"] for l in lines])
    lab = ref["line_id"] >= 0
    assert sorted(covered.tolist()) == sorted((ref["y"][lab] * ext.lx + ref["x"][lab]).tolist())


# ---------------------------------------------------------------------------
# Chain npz export/load (schema v3, v2-byte-compatible)
# ---------------------------------------------------------------------------

def _v2_golden(tmp_path):
    """Byte-exact miniature of topo_wtmm.export_extrema's v2 key set."""
    p = tmp_path / "v2.npz"
    np.savez_compressed(
        p,
        h_xyz=np.array([[10.0, 20.0, 1000.0], [10.1, 20.0, 1100.0]], np.float32),
        h_off=np.array([0, 2], np.int64), h_scale=np.array([0], np.int32),
        h_len=np.array([2], np.int32),
        v_xyz=np.array([[10.0, 20.0, 900.0], [10.0, 20.1, 950.0]], np.float32),
        v_off=np.array([0, 2], np.int64), v_persist=np.array([2], np.int32),
        v_scale=np.array([0, 1], np.int32),
        scales=np.array([1.0, 2.0]), n_scales=2, region="golden")
    return p

def test_v2_loads_identically_via_both_loaders(tmp_path):
    from dynamix.core.extrema_io import load_topo_extrema
    from dynamix.core.wtmm_backend import load_chains_npz
    p = _v2_golden(tmp_path)
    a, b = load_topo_extrema(p), load_chains_npz(p)
    assert a["region"] == b["region"] == "golden" and a["n_scales"] == 2
    for k in ("h_segments", "v_segments"):
        assert len(a[k]) == len(b[k])
        np.testing.assert_array_equal(a[k][0], b[k][0])   # incl. the -elev/1000 flip

def test_export_load_roundtrip_local_frame(tmp_path, fbm64):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import (PythonWTMMBackend, compute_scales2d,
                                       export_chains_npz, load_chains_npz)
    rf = RasterField(name="fbm", values=fbm64, frame=LocalFrame(units="px"),
                     x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    b = PythonWTMMBackend(); scales = compute_scales2d(2, 3)
    cwt = b.cwt2d(fbm64, scales); ext = b.extrema2d(cwt, scales)
    chains = b.chains2d(ext, scales)
    p = tmp_path / "fbm_chains.npz"
    info = export_chains_npz(p, chains=chains, extrema=ext, scales=scales, field=rf,
                             params={"wavelet": "gaussian"})
    assert info["n_vertical"] == len(chains) and info["n_horizontal"] > 0
    te = load_chains_npz(p)
    assert te["frame"] == rf.frame and te["params"]["wavelet"] == "gaussian"
    assert len(te["v_segments"]) == info["n_vertical"]
    assert all(np.all(s[:, 2] == 0.0) for s in te["v_segments"])   # local: no depth flip
    assert len(te["v_mod_segments"]) == len(te["v_segments"])
    # geometry in frame units: all points inside the 64px box
    allpts = np.vstack(te["v_segments"] + te["h_segments"])
    assert allpts[:, 0].min() >= 0 and allpts[:, 0].max() < 64

def test_export_load_roundtrip_geographic_frame(tmp_path, fbm64):
    # A GeographicFrame field with a lon/lat box (not the default local pixel frame): v3
    # coordinates are frame-unit x/y (field.x_axis[x] / field.y_axis[y]) regardless of
    # frame kind, and z stays 0.0 in Phase A (no elev_m convention for fresh v3 exports).
    from dynamix.core.frames import GeographicFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import (PythonWTMMBackend, compute_scales2d,
                                       export_chains_npz, load_chains_npz)
    lon_axis = np.linspace(-175.0, -165.0, 64)
    lat_axis = np.linspace(50.0, 58.0, 64)
    rf = RasterField(name="fbm_geo", values=fbm64, frame=GeographicFrame(),
                     x_axis=lon_axis, y_axis=lat_axis)
    b = PythonWTMMBackend(); scales = compute_scales2d(2, 3)
    cwt = b.cwt2d(fbm64, scales); ext = b.extrema2d(cwt, scales)
    chains = b.chains2d(ext, scales)
    p = tmp_path / "fbm_geo_chains.npz"
    info = export_chains_npz(p, chains=chains, extrema=ext, scales=scales, field=rf,
                             params={"wavelet": "gaussian"})
    assert info["n_vertical"] == len(chains) and info["n_horizontal"] > 0
    te = load_chains_npz(p)
    assert isinstance(te["frame"], GeographicFrame) and te["frame"] == rf.frame
    assert len(te["v_segments"]) == info["n_vertical"]
    allpts = np.vstack(te["v_segments"] + te["h_segments"])
    assert np.all(allpts[:, 2] == 0.0)
    assert allpts[:, 0].min() >= -175.0 and allpts[:, 0].max() <= -165.0
    assert allpts[:, 1].min() >= 50.0 and allpts[:, 1].max() <= 58.0


# ---------------------------------------------------------------------------
# run_wtmm2d: staged pipeline orchestrator (scalar per-stage cache, tensor
# run-level cache)
# ---------------------------------------------------------------------------

def test_run_wtmm2d_stage_cache(tmp_path, fbm64):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import run_wtmm2d
    rf = RasterField(name="fbm", values=fbm64, frame=LocalFrame(units="px"),
                     x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    stages = []
    r1 = run_wtmm2d(rf, {"n_oct": 2, "n_voice": 3}, out_dir=tmp_path,
                    progress=lambda st, fr: stages.append(st))
    assert r1["cache_hits"] == set() and r1["npz_path"].exists()
    assert {"cwt", "extrema", "chains", "partition"} <= set(stages)
    r2 = run_wtmm2d(rf, {"n_oct": 2, "n_voice": 3}, out_dir=tmp_path)
    assert r2["cache_hits"] == {"cwt", "extrema", "chains", "partition"}
    # param edit downstream of cwt -> cwt cached, rest recomputed
    r3 = run_wtmm2d(rf, {"n_oct": 2, "n_voice": 3, "thresh": 0.05}, out_dir=tmp_path)
    assert "cwt" in r3["cache_hits"] and "extrema" not in r3["cache_hits"]
    np.testing.assert_allclose(r2["hd_std"]["tau_qa"], r1["hd_std"]["tau_qa"], equal_nan=True)

def test_run_wtmm2d_no_outdir(fbm64):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import run_wtmm2d
    rf = RasterField(name="f", values=fbm64, frame=LocalFrame(units="px"),
                     x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    r = run_wtmm2d(rf, {"n_oct": 1, "n_voice": 2})
    assert r["npz_path"] is None and len(r["chains"]) > 0

def test_run_wtmm2d_tensor_mode(tmp_path, fbm3_64):
    import pytest
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import load_chains_npz, run_wtmm2d
    rf = RasterField(name="ori", values=fbm3_64, frame=LocalFrame(units="µm"),
                     x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    r = run_wtmm2d(rf, {"mode": "tensor", "svd_mode": "sigma_max",
                        "n_oct": 2, "n_voice": 3}, out_dir=tmp_path)
    assert r["npz_path"].exists() and len(r["chains"]) > 0
    te = load_chains_npz(r["npz_path"])
    assert te["params"]["mode"] == "tensor" and te["params"]["svd_mode"] == "sigma_max"
    r2 = run_wtmm2d(rf, {"mode": "tensor", "svd_mode": "sigma_max",
                         "n_oct": 2, "n_voice": 3}, out_dir=tmp_path)
    assert r2["cache_hits"] == {"tensor"}                       # run-level cache
    with pytest.raises(ValueError):
        run_wtmm2d(RasterField(name="s", values=fbm3_64[:, :, 0],
                               frame=LocalFrame(), x_axis=np.arange(64.0),
                               y_axis=np.arange(64.0)), {"mode": "tensor"})

def test_run_wtmm2d_unknown_param_raises(fbm64):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import run_wtmm2d
    rf = RasterField(name="f2", values=fbm64, frame=LocalFrame(units="px"),
                     x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    with pytest.raises(ValueError, match="n_octaves"):
        run_wtmm2d(rf, {"n_octaves": 2})

def test_run_wtmm2d_progress_stages(tmp_path, fbm64):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import run_wtmm2d
    rf = RasterField(name="fprog", values=fbm64, frame=LocalFrame(units="px"),
                     x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    stages = {"cwt", "extrema", "chains", "partition"}

    events = []
    run_wtmm2d(rf, {"n_oct": 1, "n_voice": 2}, out_dir=tmp_path,
              progress=lambda st, fr: events.append((st, fr)))
    zeros = [st for st, fr in events if fr == 0.0]
    ones = [st for st, fr in events if fr == 1.0]
    assert set(zeros) == stages and len(zeros) == 4
    assert set(ones) == stages and len(ones) == 4

    events2 = []
    run_wtmm2d(rf, {"n_oct": 1, "n_voice": 2}, out_dir=tmp_path,
              progress=lambda st, fr: events2.append((st, fr)))
    assert not any(fr == 0.0 for _, fr in events2)
    ones2 = [st for st, fr in events2 if fr == 1.0]
    assert set(ones2) == stages and len(ones2) == 4

def test_run_wtmm2d_qlist_hash_is_exact_not_repr(tmp_path, fbm64):
    # F1 regression: repr() of a >1000-element ndarray is a SUMMARIZED display
    # string (numpy elides the middle and caps float precision), so two
    # genuinely different large q_list arrays could repr() -- and therefore
    # hash -- identically. The partition stage must MISS on this edit, and the
    # cached hd_std must carry the q_list actually requested, not a stale one.
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import run_wtmm2d
    rf = RasterField(name="fqhash", values=fbm64, frame=LocalFrame(units="px"),
                     x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    q1 = np.linspace(-3.0, 6.0, 1200)
    run_wtmm2d(rf, {"n_oct": 1, "n_voice": 2, "q_list": q1}, out_dir=tmp_path)
    q2 = q1.copy()
    q2[500] = 42.0
    r2 = run_wtmm2d(rf, {"n_oct": 1, "n_voice": 2, "q_list": q2}, out_dir=tmp_path)
    assert r2["cache_hits"] == {"cwt", "extrema", "chains"}
    np.testing.assert_allclose(r2["hd_std"]["q_list"], q2)

def test_run_wtmm2d_corrupt_cache_self_heals(tmp_path, fbm64):
    # F2 regression: an interrupted write used to leave a truncated npz AT the
    # final cache path forever (path.exists() is True, np.load() raises
    # BadZipFile on every subsequent run). A corrupt stage file must instead
    # be treated as a miss -- recomputed and atomically rewritten -- so the
    # cache self-heals rather than permanently wedging the pipeline.
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import run_wtmm2d
    rf = RasterField(name="fcorrupt", values=fbm64, frame=LocalFrame(units="px"),
                     x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    run_wtmm2d(rf, {"n_oct": 1, "n_voice": 2}, out_dir=tmp_path)
    cache_dir = tmp_path / "wtmm_cache" / "fcorrupt"
    chains_files = list(cache_dir.glob("chains-*.npz"))
    assert len(chains_files) == 1
    chains_files[0].write_bytes(b"garbage")

    r2 = run_wtmm2d(rf, {"n_oct": 1, "n_voice": 2}, out_dir=tmp_path)     # must not raise
    assert "chains" not in r2["cache_hits"]

    r3 = run_wtmm2d(rf, {"n_oct": 1, "n_voice": 2}, out_dir=tmp_path)     # repaired
    assert "chains" in r3["cache_hits"]


@pytest.mark.parametrize("wavelet", ["gaussian", "mexican"])
@pytest.mark.parametrize("h", [0.3, 0.6])
def test_wtmm2d_is_l1_normalized_at_a_cusp(wavelet, h):
    """The WTMM convention is L1 (pure dilation, Fourier-domain: every filter is psi_hat(a k)
    with theta_hat(0) = 1), so at a cusp |x - x0|^h the maxima-line modulus scales as a^h --
    NOT a^(h + d/2) as an L2 (a^(-d/2)) transform would give. The smallest scales are biased
    by discretization, so the slope is read over the upper half of the ladder (sigma >~ 5 px)."""
    from dynamix.core import wtmm_backend as wb

    n = 256
    y, x = np.indices((n, n)) - n // 2
    rho = np.hypot(x, y)
    scales = wb.compute_scales2d(4, 4, a_min=1.0)
    mod = np.asarray(wb._cwt2d_numpy((rho ** h).astype(np.float32), scales, pad=128,
                                     wavelet=wavelet, verbose=False)["mod"])
    ring = rho <= 0.9 * scales[:, None, None] + 2          # the maxima ring around x0
    peak = np.array([mod[i][ring[i]].max() for i in range(len(scales))])
    upper = slice(len(scales) // 2, None)
    slope = np.polyfit(np.log(scales[upper]), np.log(peak[upper]), 1)[0]
    assert slope == pytest.approx(h, abs=0.03)
