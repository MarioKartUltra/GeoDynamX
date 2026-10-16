# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every q-wavelet has both widths (spec 20): fixed, one β for every q, so q changes the shape and
the width together; or q-paired, β(q) = 1/(2(2 − q)) in 2-D, which holds the escort variance at
one per component so q changes the shape alone (the q-Meyer taper, which has no β, holds its band
peak instead). The WTMM q smoothers take the width as a stretch of their closed form,
θ̂_β(s) = θ̂_½(s/√(2β)); fixed at β = ½ is today's kernel, so no existing key moves."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from dynamix.core import smoothers as sm
from dynamix.devices.wtmm import WTMM2D, smoother_params
from dynamix.devices.wtmm_roi import WTMM2DROI
from dynamix.model.device import defaults_for, keyed_params

Q_SMOOTHERS = ("q_gaussian", "q_mexican", "q_meyer")
BETA_SMOOTHERS = ("q_gaussian", "q_mexican")


def _params(**kw) -> dict:
    return dict(defaults_for(WTMM2D()), **kw)


def _param(dev, name):
    return next(p for p in dev.params if p.name == name)


def _active(param, params: dict) -> bool:
    """``Param.active_when``: one ``(name, allowed)`` pair, or a tuple of pairs that must all hold."""
    cond = param.active_when
    if not cond:
        return True
    pairs = (cond,) if isinstance(cond[0], str) else cond
    return all(params.get(name) in allowed for name, allowed in pairs)


# 1. Fixed at β = ½ is today's smoother, and the defaults leave every key where it was.

@pytest.mark.parametrize("name,qs", [("q_gaussian", (0.5, 1.0, 1.5)),
                                     ("q_mexican", (0.8, 1.0, 1.5)),     # q-Mexican: q > 2/3
                                     ("q_meyer", (0.5, 1.0, 1.5))])
def test_the_default_width_is_todays_smoother_and_no_key_moves(name, qs):
    a = 8.0
    for q in qs:
        params = _params(wavelet=name, q_tsallis=q)
        assert params["q_pairing"] == "fixed" and params["q_beta"] == 0.5
        assert smoother_params(params) == {"q": q}
        today = sm.smoother(name, {"q": q})
        explicit = sm.smoother(name, {"q": q, "beta": 0.5}) if name in BETA_SMOOTHERS else today
        s = np.linspace(0.0, 6.0, 4097)
        np.testing.assert_array_equal(explicit.theta_hat(s, 0.3 * s), today.theta_hat(s, 0.3 * s))
        for m in (0, 1):
            for got, want in zip(explicit.section(a, m), today.section(a, m)):
                np.testing.assert_array_equal(got, want)
        assert explicit.lambda_px(a) == today.lambda_px(a)
    for dev in (WTMM2D(), WTMM2DROI()):
        keyed = keyed_params(dev, dict(defaults_for(dev), wavelet=name))
        assert "q_pairing" not in keyed and "q_beta" not in keyed


# 2. Paired equals fixed ½ at q = 1.

@pytest.mark.parametrize("name", Q_SMOOTHERS)
def test_paired_is_fixed_at_q_one(name):
    fixed = smoother_params(_params(wavelet=name, q_tsallis=1.0))
    paired = smoother_params(_params(wavelet=name, q_tsallis=1.0, q_pairing="q-paired"))
    assert paired == fixed == {"q": 1.0}
    assert smoother_params(_params(wavelet=name, q_tsallis=1.5, q_pairing="q-paired")) != {"q": 1.5}


# 3. The stretched closed form is the q-Gaussian's own transform at that β.

@pytest.mark.parametrize("q", (0.5, 1.5))
@pytest.mark.parametrize("beta", (1.0 / 3.0, 1.0, 2.0))
def test_the_stretched_table_is_the_transform_at_that_width(q, beta):
    """At the table's own nodes mapped through the stretch (between nodes the table is linearly
    interpolated, the same at every width), θ̂_β equals q_gaussian_ft(k, q, β) normalised to one
    at the origin; the q-Mexican hat is |s|² times it, and both sections are today's at the
    scale a/√(2β)."""
    from dynamix.core.q_fourier import q_gaussian_ft

    s_tab, _f = sm._q_table(q)
    s = np.sqrt(2.0 * beta) * s_tab[: 1 << 14 : 7]
    g = sm.smoother("q_gaussian", {"q": q, "beta": beta})
    want = q_gaussian_ft(np.sqrt(2.0) * s, q, beta) / q_gaussian_ft(0.0, q, beta)
    np.testing.assert_allclose(g.theta_hat(s, np.zeros_like(s)), want, rtol=0, atol=1e-10)
    a = 8.0
    names = ("q_gaussian", "q_mexican") if q > 2.0 / 3.0 else ("q_gaussian",)
    for name in names:
        stretched = sm.smoother(name, {"q": q, "beta": beta})
        if name == "q_mexican":
            np.testing.assert_allclose(stretched.theta_hat(s, np.zeros_like(s)), s * s * want,
                                       rtol=0, atol=1e-10)
        today = sm.smoother(name, {"q": q})
        for m in (0, 1):
            x, y = stretched.section(a, m)
            x0, y0 = today.section(a / np.sqrt(2.0 * beta), m)
            np.testing.assert_allclose(x, x0, rtol=1e-12)
            np.testing.assert_allclose(y, y0, rtol=1e-12, atol=1e-12 * np.abs(y0).max())


# 4. Paired holds the 2-D escort variance at one per component.

@pytest.mark.parametrize("q", (0.25, 0.5, 1.5, 1.9))
def test_paired_holds_the_escort_variance_at_one(q):
    """The β the device resolves for a paired q-Gaussian makes the escort density of
    e_q(−β ρ²) (ρ in σ) have variance one per component, by quadrature; the smoother's own
    θ-section is that e_q at that β."""
    from scipy.integrate import quad

    beta = smoother_params(_params(wavelet="q_gaussian", q_tsallis=q, q_pairing="q-paired"))["beta"]
    escort = lambda rho: sm._e_q(-beta * rho * rho, q) ** q          # noqa: E731
    top = 1.0 / np.sqrt((1.0 - q) * beta) if q < 1.0 else np.inf
    m2 = quad(lambda r: r ** 3 * escort(r), 0.0, top, limit=200)[0]
    m0 = quad(lambda r: r * escort(r), 0.0, top, limit=200)[0]
    assert 0.5 * m2 / m0 == pytest.approx(1.0, rel=1e-8)
    a = 8.0
    sigma = a / (np.pi * np.sqrt(2.0))
    x, theta = sm.smoother("q_gaussian", {"q": q, "beta": beta}).section(a, 0)
    np.testing.assert_allclose(theta, sm._e_q(-beta * (x / sigma) ** 2, q), rtol=1e-12, atol=1e-15)


# 5. λ* of the β smoothers scales as 1/√(2β).

@pytest.mark.parametrize("name", BETA_SMOOTHERS)
@pytest.mark.parametrize("q", (0.8, 1.5))
@pytest.mark.parametrize("beta", (1.0 / 3.0, 1.0, 2.0))
def test_lambda_star_scales_with_the_width(name, q, beta):
    """To the λ* finder's own precision: it reads the peak off the tabulated θ̂, linearly
    interpolated, which puts it within 1.8e-4 of the closed form's peak at every width,
    today's ½ included (8e-5 there); a wrong stretch is 18 % off or more."""
    a = 8.0
    half = sm.smoother(name, {"q": q}).lambda_px(a)
    assert sm.smoother(name, {"q": q, "beta": beta}).lambda_px(a) == pytest.approx(
        half / np.sqrt(2.0 * beta), rel=5e-4)


# 6. The q-Meyer taper paired holds its band peak.

@pytest.mark.parametrize("q", (0.5, 1.0, 1.5))
def test_q_meyer_paired_holds_the_band_peak(q):
    """To the λ* finder's precision on the taper, 1.1e-6 measured (8e-7 at q = 1 today)."""
    a = 8.0
    paired = sm.smoother("q_meyer", smoother_params(_params(wavelet="q_meyer", q_tsallis=q,
                                                            q_pairing="q-paired")))
    assert paired.lambda_px(a) == pytest.approx(np.sqrt(2.0) * a, rel=2e-6)
    if q != 1.0:                                                      # fixed: q moves λ*
        fixed = sm.smoother("q_meyer", smoother_params(_params(wavelet="q_meyer", q_tsallis=q)))
        assert fixed.lambda_px(a) != pytest.approx(np.sqrt(2.0) * a, rel=1e-3)


# 7. The knobs: where they apply, out of the key at their defaults, the paired bound.

def test_the_width_knobs_apply_only_to_the_q_smoothers_and_q_beta_only_while_fixed():
    for dev in (WTMM2D(), WTMM2DROI()):
        pairing, beta = _param(dev, "q_pairing"), _param(dev, "q_beta")
        assert (pairing.default, pairing.choices, pairing.label) == ("fixed", ("fixed", "q-paired"),
                                                                     "q width")
        assert (beta.default, beta.label) == (0.5, "q β")
        assert not pairing.keyed_at_default and not beta.keyed_at_default
        assert not pairing.view and not beta.view
        pre = dev.param_groups["PRE"]
        i = pre.index("q_tsallis")
        assert pre[i + 1: i + 3] == ("q_pairing", "q_beta")           # beside q (Tsallis)
        for wavelet in ("mexican", "gaussian", "frac_bspline"):
            p = dict(defaults_for(dev), wavelet=wavelet)
            assert not _active(pairing, p) and not _active(beta, p)
        for wavelet in BETA_SMOOTHERS:
            p = dict(defaults_for(dev), wavelet=wavelet)
            assert _active(pairing, p) and _active(beta, p)
            assert not _active(beta, dict(p, q_pairing="q-paired"))
        p = dict(defaults_for(dev), wavelet="q_meyer")
        assert _active(pairing, p) and not _active(beta, p)           # q-Meyer has no β
    with pytest.raises(ValueError, match="a q-paired width needs q < 2"):
        smoother_params(_params(wavelet="q_gaussian", q_tsallis=2.0, q_pairing="q-paired"))
    assert smoother_params(_params(wavelet="q_gaussian", q_tsallis=1.5, q_beta=1.0)) == {
        "q": 1.5, "beta": 1.0}
    assert smoother_params(_params(wavelet="q_gaussian", q_tsallis=1.5,
                                   q_pairing="q-paired")) == {"q": 1.5, "beta": 1.0}
    assert smoother_params(_params(wavelet="gaussian", q_pairing="q-paired", q_beta=2.0)) is None


def test_the_caption_and_the_reading_name_the_width():
    dev = WTMM2D()
    fixed = _params(wavelet="q_gaussian", q_tsallis=1.5)
    paired = dict(fixed, q_pairing="q-paired")
    assert "q-Gaussian q = 1.5, β 0.5" in dev.wavelet_profile(fixed, 8.0, 0).family
    assert "q-Gaussian q = 1.5, β paired" in dev.wavelet_profile(paired, 8.0, 0).family
    assert "β 2" in dev.wavelet_profile(dict(fixed, q_beta=2.0), 8.0, 0).family
    assert "β paired" in dev.wavelet_profile(dict(paired, wavelet="q_mexican"), 8.0, 0).family
    meyer = dev.wavelet_profile(dict(paired, wavelet="q_meyer"), 8.0, 0).family
    assert "band peak held" in meyer
    assert "band peak held" not in dev.wavelet_profile(dict(fixed, wavelet="q_meyer"), 8.0, 0).family
    for d in (dev, WTMM2DROI()):
        assert "escort variance held" in d.derived_reading("q_pairing", "q-paired", None, paired)
        assert "β 1" in d.derived_reading("q_pairing", "q-paired", None, paired)
        assert "q moves width" in d.derived_reading("q_pairing", "fixed", None, fixed)
        assert "band peak held" in d.derived_reading(
            "q_pairing", "q-paired", None, dict(paired, wavelet="q_meyer"))
        assert d.derived_reading("q_pairing", "fixed", None, _params()) == ""


# 8. End to end on a DEM, and the .gprv record.

def _dem(path, n=64, seed=3):
    """A synthetic projected DEM (the gprv writer tests' own; the demo DEM is not shipped)."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    v = np.random.default_rng(seed).standard_normal((n, n)).cumsum(0).cumsum(1).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:32611", transform=from_origin(500000, 4000000, 30, 30)) as ds:
        ds.write(v, 1)
    return path


def test_a_paired_q_gaussian_runs_on_a_dem_and_its_record_names_the_pairing(clean_registry,
                                                                              tmp_path):
    gprv = pytest.importorskip("gprv")
    from dynamix.core import fft_policy
    from dynamix.core.chain_product import export_chain_product_npz
    from dynamix.core.ingest import load_grid
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.gprv_export import write_record
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.project import Project

    register_builtin_devices()
    fft_policy.configure("fftw", 64)
    dem = _dem(tmp_path / "dem.tif")
    field = load_grid(dem)
    project = Project()
    src = project.add_source(str(dem), sha256=hashlib.sha256(dem.read_bytes()).hexdigest())
    base = {"n_oct": 2, "n_voice": 4, "precision": "f64", "wavelet": "q_gaussian",
            "q_tsallis": 1.5}
    layer = project.add_layer("dem", src.source_id, Chain((DeviceRef(
        "wtmm2d", dict(base, q_pairing="q-paired")),)))
    fixed = project.add_layer("fixed", src.source_id, Chain((DeviceRef("wtmm2d", dict(base)),)))
    rendered = resolve(layer, field, Cache())
    assert rendered.result["extrema"] and rendered.analysis_key != resolve(
        fixed, field, Cache()).analysis_key
    out = tmp_path / "chains.npz"
    export_chain_product_npz(out, rendered.result["chain_product"], field=field,
                             params=json.loads(json.dumps(rendered.result.get("params") or {},
                                                          default=list)))
    data = write_record(project, layer, out, export="chains_npz",
                        result=rendered.result).read_bytes()
    assert gprv.check(data) == []
    run = json.loads(data)["processStep"][0]["processingInformation"]["runTimeParameters"]
    assert run["q_pairing"] == "q-paired" and run["q_tsallis"] == 1.5 and "q_beta" not in run
