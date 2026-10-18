# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Disorder function tool (spec 19): the device over ``dynamix.core.disorder`` — the
field-form or envelope-form disorder δ̄(β), its total and rooted mean, of one field, with k₀ found
or given, the derivative scheme recorded, and the densities as rasters."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dynamix.core import disorder
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.contracts import CURVE, FIELD
from dynamix.model.device import defaults_for, get_device, keyed_params, validate_params


@pytest.fixture
def dev(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return get_device("disorder_function")


def _field(values, provenance=None):
    values = np.asarray(values, dtype=np.float64)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(values.shape[1])),
                       y_axis=np.arange(float(values.shape[0])), provenance=provenance or {})


def _stripes(n: int, cycles: int, angle: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    k = 2.0 * math.pi * cycles / n
    return np.cos(k * (x * math.cos(angle) + y * math.sin(angle)))


def test_the_device_is_registered_under_measures_and_produces_a_curve(dev):
    assert dev.label == "Disorder function (Gunaratne)" and dev.families == ("Measures",)
    assert dev.consumes == ((FIELD,),) and CURVE in dev.produces


def test_the_defaults_are_the_papers_and_the_record_names_the_scheme(dev):
    p = validate_params(dev, {})
    assert (p["deriv"], p["k0"], p["harmonic_filter"], p["detrend"]) == (
        "daf", "min_disorder", "daf", "mean")
    assert p["sigma_d"] == 1.0 and p["periodic"] is False
    res = dev.compute(_field(_stripes(128, 6)), p)
    assert res["record"]["deriv"] == {"scheme": "daf", "M": 12, "sigma_over_dx": 4.0,
                                      "reach_px": disorder.reach("daf"), "periodized_by": "daf extension"}
    assert res["record"]["k0"]["method"] == "min_disorder"
    assert {"deriv", "k0", "harmonic_filter", "detrend"} <= set(keyed_params(dev, p))


def test_the_dafs_order_and_width_are_knobs_that_default_to_the_papers(dev):
    """M and σ/Δ sit on the box under the DAF scheme (PRE 57's 12 and 4 by default), reach the
    kernel, the reach and the record, and are keyed; the labels carry no citations."""
    p = validate_params(dev, {"daf_M": 8, "daf_sigma": 3.0, "periodic": True, "k0": "manual",
                              "lambda0": 128 / 6, "harmonic_filter": "none"})
    assert {"daf_M", "daf_sigma"} <= set(keyed_params(dev, p))
    assert dev.roi_margin(p) == disorder.reach("daf", M=8, sigma_over_dx=3.0) < disorder.reach("daf")
    res = dev.compute(_field(_stripes(128, 6)), p)
    assert res["record"]["deriv"]["M"] == 8 and res["record"]["deriv"]["sigma_over_dx"] == 3.0
    assert np.all(res["mean_field_disorder"] == 0.0)          # perfect stripes still read zero
    by_name = {q.name: q for q in dev.params}
    assert by_name["daf_M"].active_when == ("deriv", ("daf",)) and by_name["daf_M"].default == 12
    assert by_name["daf_sigma"].default == 4.0
    for q in dev.params:
        for text in (q.label, *(getattr(q, "choice_labels", None) or ())):
            assert "(" not in text or text in ("D(x)", "D^env(x)", "|A(x)|", "|K(x)|", "|Λ(x)|",
                                               "σ_d (px)", "λ₀ (px)", "min δ̄(1)"), text


def test_perfect_periodic_stripes_read_zero_and_find_their_wavenumber(dev):
    n, cycles = 128, 6
    found = dev.compute(_field(_stripes(n, cycles)), validate_params(dev, {"periodic": True}))
    assert found["k0"] == pytest.approx(2.0 * math.pi * cycles / n, rel=1e-6)
    assert found["mean_field_disorder"][3] < 1e-6          # β = 1 at the found k₀: a residual
    # the exact zero needs the exact k₀: a found one is right to ~1e-12, and (1e-12)^0.1 is 0.06
    res = dev.compute(_field(_stripes(n, cycles)),
                      validate_params(dev, {"periodic": True, "k0": "manual", "lambda0": n / cycles}))
    assert np.array_equal(res["beta_delta"], disorder.DELTA_BETAS)
    assert np.all(res["mean_field_disorder"] == 0.0) and np.all(res["total_field_disorder"] == 0.0)
    assert np.all(res["rooted_mean_field_disorder"] == 0.0)
    assert res["raster_out"] is res["field_disorder_density"] and res["raster_out"].shape == (n, n)
    assert res["_shape"] == (n, n)
    from dynamix.core.derivative import raster_choices

    labels = [label for label, _arr in raster_choices(res)]
    assert labels[0] == "as shown" and "D(x)" in labels


def test_a_given_wavelength_and_show_falls_back_to_the_forms_own_density(dev):
    n = 128
    p = validate_params(dev, {"k0": "manual", "lambda0": 20.0, "show": "envelope_amplitude"})
    res = dev.compute(_field(_stripes(n, 5)), p)
    assert res["k0"] == pytest.approx(2.0 * math.pi / 20.0)
    assert res["record"]["k0"] == {"method": "manual", "value": res["k0"], "lambda_px": 20.0}
    assert res["raster_out"] is res["field_disorder_density"]   # the field form made no envelope
    assert res["mean_field_disorder"][3] > 0.1                     # β = 1: detuned stripes
    # show is view-only: a flip is a cache hit through view(), never a recompute
    assert "show" not in keyed_params(dev, p)
    assert dev.view(res, {"show": "field_disorder_density"})["raster_out"] is res["field_disorder_density"]


def test_spectral_derivatives_need_a_periodized_input(dev):
    field = _field(_stripes(96, 4)[:80, :70])
    bare = {"harmonic_filter": "none"}                 # the DAF filter would periodize it itself
    with pytest.raises(ValueError, match="Periodic extension"):
        dev.compute(field, validate_params(dev, {"deriv": "spectral", **bare}))
    declared = dev.compute(field, validate_params(dev, {"deriv": "spectral", "periodic": True, **bare}))
    assert declared["record"]["deriv"]["periodized_by"] == "declared periodic"
    stamped = _field(_stripes(96, 4), provenance={"edges.filled": {"rows": 80, "cols": 70}})
    extended = dev.compute(stamped, validate_params(dev, {"deriv": "spectral", **bare}))
    assert extended["record"]["deriv"]["periodized_by"] == "edges.filled"
    assert "periodized_by" not in dev.compute(field, validate_params(dev, bare))["record"]["deriv"]


def test_the_reading_names_the_wavelength_and_the_disorder_at_beta_1(dev):
    p = validate_params(dev, {})
    res = dev.compute(_field(_stripes(128, 5, angle=0.4)), p)
    reading = dev.reading(res, p)
    assert "λ₀" in reading and "δ̄^fie(1)" in reading and "px" in reading


def test_a_saved_chain_naming_the_old_invariant_measures_tool_runs_as_the_disorder_function(dev):
    """The split's shim: a step saved as ``invariant_measures`` loads as ``disorder_function``,
    its Hessian knobs dropped and the rest kept."""
    from dynamix.model.chain import Chain

    chain = Chain.from_payload({"steps": [{"device": "invariant_measures",
                                           "params": {"k0": "manual", "lambda0": 24.0,
                                                      "hessian_norm": "std"}}]})
    (step,) = chain.steps
    assert step.device == "disorder_function" and step.params["lambda0"] == 24.0
    assert "hessian_norm" not in step.params and step.params["show"] == "field_disorder_density"


def test_the_roi_margin_is_the_schemes_reach(dev):
    assert dev.roi_margin(validate_params(dev, {})) == disorder.reach("daf")
    assert dev.roi_margin(validate_params(dev, {"deriv": "gaussian", "sigma_d": 2.0})) == disorder.reach("gaussian", 2.0)
    assert dev.roi_margin(validate_params(dev, {"deriv": "spectral", "periodic": True})) == 0


def test_the_daf_harmonic_filter_is_the_default_and_records_its_extension_and_band(dev):
    """The Chaos pipeline on a non-periodic crop: the extension's tuning and the band go in the
    record, the rasters keep the field's shape, and the statistics ignore the filled gap."""
    n, cycles = 160, 8
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    phase = 2.0 * math.pi * cycles / n * (x * math.cos(0.25) + y * math.sin(0.25))
    u = np.cos(phase) + 0.3 * np.cos(2 * phase) + 0.4 * x / n
    res = dev.compute(_field(u), validate_params(dev, {"k0": "manual", "lambda0": n / cycles}))
    rec = res["record"]["harmonic_filter"]
    assert rec["method"] == "daf" and rec["band"] == [0.7, 1.3] and rec["k_t"] > 0
    assert rec["extension"]["fallback"] is None and rec["extension"]["gap"] >= 8
    assert res["record"]["deriv"]["periodized_by"] == "daf extension"
    assert res["raster_out"].shape == (n, n) and res["record"]["valid_pixels"] == n * n
    assert res["mean_field_disorder"][3] < 0.1          # the periodization floor on a tilted crop
    noise = np.random.default_rng(2).standard_normal((96, 96))
    coarse = dev.compute(_field(noise), validate_params(dev, {"k0": "manual", "lambda0": 3.0}))
    assert isinstance(coarse["record"]["harmonic_filter"]["extension"]["fallback"], str)
    assert "periodized_by" not in coarse["record"]["deriv"]


def test_the_bandpass_filter_removes_a_second_harmonic(dev):
    """Stripes plus their second harmonic: the Gaussian band-pass at k₀ (relative width 0.5)
    leaves nearly perfect stripes, so δ̄(1) drops well below the unfiltered value."""
    n, cycles = 192, 6
    u = _stripes(n, cycles) + 0.3 * _stripes(n, 2 * cycles)
    raw = dev.compute(_field(u), validate_params(dev, {"periodic": True, "k0": "manual",
                                                      "lambda0": n / cycles,
                                                      "harmonic_filter": "none"}))
    filtered = dev.compute(_field(u), validate_params(dev, {"periodic": True, "k0": "manual",
                                                           "lambda0": n / cycles,
                                                           "harmonic_filter": "bandpass"}))
    assert filtered["mean_field_disorder"][3] < 0.1 * raw["mean_field_disorder"][3]
    assert filtered["record"]["harmonic_filter"] == {"method": "bandpass", "relative_width": 0.5}


def test_the_envelope_form_publishes_the_prl_measures_and_its_amplitude(dev):
    """Under disorder_form = "envelope" the device publishes δ̄^env, δ^env and δ_root^env with the
    PRL's names, the envelope's density and |A| as rasters."""
    n = 240
    p = validate_params(dev, {"disorder_form": "envelope", "periodic": True, "k0": "manual",
                              "lambda0": n / 3, "harmonic_filter": "none"})
    res = dev.compute(_field(_stripes(n, 5)), p)
    assert np.allclose(res["mean_envelope_disorder"], (2 - res["beta_delta"]) * (8 / 9) ** res["beta_delta"], rtol=1e-6)
    assert "mean_field_disorder" not in res
    assert res["raster_out"] is res["envelope_disorder_density"]
    assert np.allclose(res["envelope_amplitude"], 1.0, atol=1e-6)
    assert res["record"]["disorder_form"] == "envelope"
    assert res["record"]["envelope"]["method"] == "fft_halfplane"
    assert res["record"]["envelope"]["direction_deg"] == pytest.approx(0.0, abs=1e-9)
    assert res["record"]["disorder_normalization"] == {"mean_envelope_disorder": "mean",
                                                        "total_envelope_disorder": "total",
                                                        "rooted_mean_envelope_disorder": "rooted_mean"}
    assert dev.view(res, {"show": "envelope_amplitude"})["raster_out"] is res["envelope_amplitude"]


def test_the_wavelet_ridge_envelope_reads_the_prl_closed_form_and_records_the_ridge(dev):
    """envelope = "wavelet_ridge": the 2-D q-Morlet ridge (Passot–Newell 1994 App. A) gives the
    amplitude and the local wavevector; the phase-field density on detuned stripes is the constant
    |k₁² − k₀²|/(2k₀²), so δ̄^env(β) = (2 − β)(8/9)^β again, the operator route agrees on a plane
    wave, |A| and |K| join the rasters, and the record carries the ridge's settings."""
    n = 240
    base = {"disorder_form": "envelope", "envelope": "wavelet_ridge", "periodic": True,
            "k0": "manual", "lambda0": n / 3, "harmonic_filter": "none"}
    p = validate_params(dev, base)
    assert (p["density"], p["omega"], p["anisotropy"], p["ridge_q"], p["q_pairing"]) == (
        "phase_field", 3.0, 1.0, 1.0, "fixed")
    assert {"envelope", "density", "omega", "anisotropy", "ridge_q", "q_pairing", "q_beta"} <= set(keyed_params(dev, p))
    res = dev.compute(_field(_stripes(n, 5)), p)
    betas = res["beta_delta"]
    assert np.allclose(res["mean_envelope_disorder"], (2 - betas) * (8 / 9) ** betas, rtol=1e-5)
    assert np.allclose(res["envelope_amplitude"], 1.0, atol=1e-6)
    assert np.allclose(res["local_wavenumber"], 2.0 * math.pi * 5 / n, rtol=1e-6)
    rec = res["record"]["envelope"]
    assert (rec["method"], rec["wavelet"], rec["density"], rec["omega"], rec["q"], rec["q_beta"]) == (
        "wavelet_ridge", "q_morlet", "phase_field", 3.0, 1.0, 0.5)
    assert rec["n_unconverged"] == 0 and rec["bank"]["nodes"] == 108 and res["warnings"] == []
    assert rec["direction_deg"] == pytest.approx(0.0, abs=1e-9)
    assert dev.view(res, {"show": "local_wavenumber"})["raster_out"] is res["local_wavenumber"]
    operator = dev.compute(_field(_stripes(n, 5)), validate_params(dev, {**base, "density": "envelope_operator"}))
    assert np.allclose(operator["mean_envelope_disorder"], (2 - betas) * (8 / 9) ** betas, rtol=1e-5)
    assert operator["record"]["envelope"]["density"] == "envelope_operator"
