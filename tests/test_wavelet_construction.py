# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Real or Fourier construction of the microcanonical kernels.

"real" (the default) samples each kernel on the pixel grid and normalises it there, as Turiel's
discretisation argument does. "fourier" evaluates the same continuous kernel's closed-form 2-D
transform at r k (a pure L1 dilation):
- the measure route's kernels have unit mass exactly;
- the multiaffine route's have zero mean exactly and the mother's L1 norm, a constant with no
  effect on any slope.
Where both constructions are exact (a scale well above the pixel, tails that die inside the
grid) they must give the same projections; where they differ (small scales, compact support,
heavy tails) the difference is the point of the toggle.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import microcanonical as mc


def _field(n=128, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)).cumsum(0).cumsum(1)


def _l1_laplacian(q_env: float, beta_env: float) -> float:
    """int |psi| d^2x of psi = -lap(f) / (4 beta'), f = e_q'^(-beta' rho^2), at unit scale: the
    positive disc inside the one zero crossing holds half, and by the divergence theorem that
    half is (pi / beta') q'^(-q'/(q'-1)) (e^-1 at q' = 1)."""
    tail = np.exp(-1.0) if q_env == 1.0 else q_env ** (-q_env / (q_env - 1.0))
    return 2.0 * np.pi * tail / beta_env


def test_real_is_the_default():
    s = _field(48)
    mu = mc.gradient_measure(s)
    np.testing.assert_array_equal(mc.measure_projections(mu, [2.0, 4.0]),
                                  mc.measure_projections(mu, [2.0, 4.0], construction="real"))
    np.testing.assert_array_equal(mc.ricker_projections(s, [2.0, 4.0]),
                                  mc.ricker_projections(s, [2.0, 4.0], construction="real"))


#: A heavy tail is cut at the padded grid's edge in real space but kept whole in Fourier: at the
#: default pad that alone differs by up to ~1e-4 (tail rho^-6.7 at q = 1.3), so those cases
#: widen the pad until only FFT round-off (~4e-7 at 32 bit) is left.
_WIDE = 400


@pytest.mark.parametrize("kw", [
    dict(wavelet="gaussian"),
    dict(wavelet="q_gaussian", q_tsallis=1.2, pad=_WIDE),
    dict(wavelet="q_gaussian", q_tsallis=1.3, q_beta=mc.paired_q_beta(1.3), pad=_WIDE),
    dict(wavelet="lorentzian", beta=4.0),
])
def test_measure_kernels_built_in_fourier_are_the_same_unit_mass_kernels(kw):
    mu = mc.gradient_measure(_field())
    real = mc.measure_projections(mu, [6.0], **kw)
    four = mc.measure_projections(mu, [6.0], construction="fourier", **kw)
    np.testing.assert_allclose(four, real, rtol=0, atol=2e-6 * np.abs(real).max())


@pytest.mark.parametrize("kw, l1", [
    (dict(wavelet="g2"), _l1_laplacian(1.0, 1.0)),
    (dict(wavelet="q_mexican", q_tsallis=1.2), _l1_laplacian(1.25, 1.0)),
    (dict(wavelet="q_mexican", q_tsallis=1.2, q_beta=0.5), _l1_laplacian(1.25, 0.8)),
    (dict(wavelet="lorentzian_marr", beta=3.0), _l1_laplacian(4.0 / 3.0, 1.0)),
    (dict(wavelet="g3", pad=_WIDE), None),                       # tail rho^-5
    (dict(wavelet="frac_gaussian", frac_n=2.5, pad=_WIDE), None),  # tail rho^-4.5
])
def test_multiaffine_wavelets_built_in_fourier_are_the_same_wavelets(kw, l1):
    s = _field()
    real = mc.ricker_projections(s, [6.0], **kw)
    four = mc.ricker_projections(s, [6.0], construction="fourier", **kw)
    c = np.median(four / real)
    np.testing.assert_allclose(four, c * real, rtol=0, atol=1e-4 * np.abs(four).max())
    if l1 is not None:                           # the constant IS the mother's L1 norm
        assert c == pytest.approx(l1, rel=1e-3)


def test_a_compact_kernel_differs_by_band_limiting_only():
    """Below q = 1 the q-Gaussian is compactly supported; built in Fourier it is band-limited
    at the Nyquist frequency instead, so it rings slightly. At a well-sampled scale the two
    stay close."""
    mu = mc.gradient_measure(_field())
    real = mc.measure_projections(mu, [6.0], wavelet="q_gaussian", q_tsallis=0.5)
    four = mc.measure_projections(mu, [6.0], wavelet="q_gaussian", q_tsallis=0.5,
                                  construction="fourier")
    np.testing.assert_allclose(four, real, rtol=0, atol=1e-2 * np.abs(real).max())


@pytest.mark.parametrize("kw", [
    dict(wavelet="frac_gaussian", frac_n=1.5),        # exp(-rho^n / 2): no closed form
    dict(wavelet="q_gaussian", q_tsallis=2.5),        # infinite mass from q = 2
    dict(wavelet="lorentzian", beta=1.0),             # infinite mass for beta <= 1
])
def test_fourier_refuses_measure_kernels_without_a_finite_closed_form(kw):
    with pytest.raises(ValueError, match="Fourier"):
        mc.measure_projections(np.ones((16, 16)), [2.0], construction="fourier", **kw)


def test_an_unknown_construction_is_refused():
    with pytest.raises(ValueError, match="construction"):
        mc.ricker_projections(np.ones((16, 16)), [2.0], construction="wavelet")


@pytest.mark.parametrize("wavelet", ["g1", "g2", "g3", "q_mexican", "lorentzian_marr",
                                     "frac_gaussian"])
def test_every_multiaffine_wavelet_builds_in_fourier(wavelet):
    T = mc.ricker_projections(_field(48), [1.0, 2.0, 4.0], wavelet=wavelet, q_tsallis=0.5,
                              construction="fourier")
    assert np.all(np.isfinite(T)) and T.max() > 0


# ------------------------------------------- the holder tools (and the band tools reusing them)

@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def test_the_tools_carry_a_real_or_fourier_toggle_defaulting_to_real(builtins):
    from dynamix.model.device import defaults_for, get_device

    for name in ("holder_measure", "holder_multiaffine", "band_recon_measure",
                 "band_recon_multiaffine"):
        assert defaults_for(get_device(name))["construction"] == "real"
        p = {q.name: q for q in get_device(name).params}
        assert p["construction"].choices == ("real", "fourier")


def test_the_tools_refuse_a_fourier_kernel_without_a_closed_form(builtins):
    from dynamix.model.device import defaults_for, get_device, validate_params

    dev = get_device("holder_measure")
    p = {**defaults_for(dev), "wavelet": "frac_gaussian", "construction": "fourier"}
    with pytest.raises(ValueError, match="Fourier"):
        validate_params(dev, p)
    validate_params(dev, {**p, "construction": "real"})


def test_the_toggle_reaches_the_engine_and_keys_the_cache(builtins):
    from dynamix.devices.holder_methods import method_arrays
    from dynamix.model.device import defaults_for, get_device

    s = _field(48)
    for method, name in (("measure", "holder_measure"), ("multiaffine", "holder_multiaffine")):
        dev = get_device(name)
        p = defaults_for(dev)
        real = method_arrays(s, method, p)[0]
        four = method_arrays(s, method, {**p, "construction": "fourier"})[0]
        assert np.isfinite(four).mean() > 0.9
        assert not np.array_equal(real, four, equal_nan=True)
        assert dev.cache_key("s", p) != dev.cache_key("s", {**p, "construction": "fourier"})
