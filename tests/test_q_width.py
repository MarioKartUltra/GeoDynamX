# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The q-family width beta (Borges et al. 2004's e_q^(-beta x^2)): fixed at 1/2, or q-paired so
the escort (q-)variance stays one per component -- beta(q) = 1/(d + 2 - d q), 1/(2(2 - q)) in
2-D, which also pins the q-Mexican hat's zero crossing and exists exactly where the 2-D wavelet
is admissible (q < 2). The scale r keeps its meaning at q = 1 for both tools."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import microcanonical as mc


def _field(n=48, seed=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)).cumsum(0).cumsum(1)


def test_the_default_width_is_todays_kernels():
    s = _field()
    scales = [1.5, 3.0]
    old = mc.measure_projections(mc.gradient_measure(s), scales, wavelet="q_gaussian", q_tsallis=1.5)
    new = mc.measure_projections(mc.gradient_measure(s), scales, wavelet="q_gaussian", q_tsallis=1.5,
                                 q_beta=0.5)
    np.testing.assert_array_equal(new, old)                   # fixed beta = 1/2 IS today's
    for q in (0.5, 1.0, 1.6):
        old = mc.ricker_projections(s, scales, wavelet="q_mexican", q_tsallis=q)
        paired = mc.ricker_projections(s, scales, wavelet="q_mexican", q_tsallis=q,
                                       q_beta=mc.paired_q_beta(q))
        np.testing.assert_allclose(paired, old, rtol=1e-12, atol=1e-15)   # paired IS today's


def test_paired_beta_is_one_over_two_times_two_minus_q():
    assert mc.paired_q_beta(1.0) == pytest.approx(0.5)
    assert mc.paired_q_beta(0.0) == pytest.approx(0.25)
    assert mc.paired_q_beta(1.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="q < 2"):
        mc.paired_q_beta(2.0)


@pytest.mark.parametrize("q", [-0.5, 0.5, 1.0, 1.5, 1.9])
def test_the_q_mexican_zero_crossing_is_pinned_at_r_when_paired_and_drifts_when_fixed(q):
    r = 5.0
    rho = np.linspace(0.01, 4 * r, 40001)             # fixed width at q = 1.9: 3.2 r

    def crossing(q_beta):
        k = mc._marr_kernel(rho ** 2, r, "q_gaussian", 1.0, mc._borges_q(q), 2.0, None,
                            stretch=mc._q_mexican_stretch(q, q_beta))
        return rho[np.argmax(np.sign(k) != np.sign(k[0]))]

    assert crossing(mc.paired_q_beta(q)) == pytest.approx(r, abs=2e-3)
    assert crossing(0.5) == pytest.approx(r / np.sqrt(2 - q), abs=2e-3)


@pytest.mark.parametrize("q", [0.5, 1.0, 1.5])
def test_the_paired_q_gaussian_has_unit_escort_variance_per_component(q):
    """<x^2>_q = r^2 on each axis when beta = 1/(2(2-q)): the escort second moment
    int rho^2 k^q / int k^q is 2 r^2 in 2-D."""
    r, n = 6.0, 601                  # the weight's tail is u^-2: keep the cut far out
    k = np.fft.fftshift(mc._radial_kernel((n, n), r, "q_gaussian", 1.0, q, 2.0,
                                          q_beta=mc.paired_q_beta(q)))
    y = np.arange(n) - n // 2
    rho2 = y[:, None] ** 2 + y[None, :] ** 2
    w = k ** q
    assert np.sum(rho2 * w) / np.sum(w) == pytest.approx(2 * r ** 2, rel=5e-3)


def test_the_support_reach_follows_the_width():
    """Compact below q = 1, so the reach is the cut-off and moves with the width (heavy tails
    above 1 hit the reach cap whatever the width)."""
    wide = mc._support_reach("marr", "q_mexican", 4.0, 1.0, 0.5, 2.0, 0.5)
    narrow = mc._support_reach("marr", "q_mexican", 4.0, 1.0, 0.5, 2.0, 1.0)
    assert wide > narrow
    assert mc._support_reach("measure", "q_gaussian", 4.0, 1.0, 0.5, 2.0, 0.25) > \
        mc._support_reach("measure", "q_gaussian", 4.0, 1.0, 0.5, 2.0, 1.0)


# ------------------------------------------- the two holder tools (and the band tools that
# reuse their engine)

@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def test_both_tools_carry_a_fixed_width_of_one_half_with_a_q_paired_toggle(builtins):
    from dynamix.model.device import defaults_for, get_device

    for name in ("holder_measure", "holder_multiaffine", "band_recon_measure",
                 "band_recon_multiaffine"):
        d = defaults_for(get_device(name))
        assert d["q_pairing"] == "fixed" and d["q_beta"] == 0.5
        p = {q.name: q for q in get_device(name).params}
        assert p["q_pairing"].choices == ("fixed", "q-paired")
    assert {q.name: q for q in get_device("holder_measure").params}["q_beta"].active_when == \
        (("wavelet", ("q_gaussian",)), ("q_pairing", ("fixed",)))
    assert {q.name: q for q in get_device("holder_multiaffine").params}["q_pairing"].active_when \
        == ("wavelet", ("q_mexican",))


def test_the_tools_pass_the_width_the_toggle_names(builtins):
    from dynamix.devices.holder_methods import method_arrays

    s = _field(40)
    base = {"estimator": "regression", "r_min": 1.5, "kappa": 4.0, "n_scales": 4, "beta": 1.0,
            "frac_n": 2.0, "q_tsallis": 1.4}
    for method, wavelet, core in (("measure", "q_gaussian", "measure"),
                                  ("multiaffine", "q_mexican", "marr")):
        p = {**base, "wavelet": wavelet}
        fixed = method_arrays(s, method, {**p, "q_pairing": "fixed", "q_beta": 0.8})[0]
        paired = method_arrays(s, method, {**p, "q_pairing": "q-paired", "q_beta": 0.8})[0]
        paired_other = method_arrays(s, method, {**p, "q_pairing": "q-paired", "q_beta": 3.0})[0]
        np.testing.assert_array_equal(paired, paired_other)          # the knob is inert when paired
        assert not np.allclose(fixed, paired, equal_nan=True)
    # q-paired q_mexican is today's kernel: the same h-map as the core's default width
    p = {**base, "wavelet": "q_mexican", "q_pairing": "q-paired", "q_beta": 0.5}
    h = method_arrays(s, "multiaffine", p)[0]
    T = mc.ricker_projections(s, np.geomspace(1.5, 6.0, 4), wavelet="q_mexican", q_tsallis=1.4)
    np.testing.assert_allclose(h, mc.singularity_map_regression(T, np.geomspace(1.5, 6.0, 4),
                                                                r2_min=0.0)[0], equal_nan=True)


def test_a_q_paired_q_gaussian_refuses_q_of_two_and_above(builtins):
    from dynamix.model.device import defaults_for, get_device, validate_params

    dev = get_device("holder_measure")
    p = {**defaults_for(dev), "wavelet": "q_gaussian", "q_pairing": "q-paired", "q_tsallis": 2.2}
    with pytest.raises(ValueError, match="q < 2"):
        validate_params(dev, p)
    validate_params(dev, {**p, "q_pairing": "fixed"})                # fixed: allowed (it warns)


def test_the_width_keys_the_cache(builtins):
    from dynamix.model.device import defaults_for, get_device

    dev = get_device("holder_multiaffine")
    p = {**defaults_for(dev), "wavelet": "q_mexican"}
    keys = {dev.cache_key("s", {**p, "q_pairing": m, "q_beta": b})
            for m, b in (("fixed", 0.5), ("fixed", 0.7), ("q-paired", 0.5))}
    assert len(keys) == 3
