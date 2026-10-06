# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Band reconstruction as output rows of the Microcanonical MFA tools (holder_measure,
holder_multiaffine): the band knobs are view-only params in the Reconstruction section, Show picks
the h-map or a band output, and the lazy ``band`` / ``band_mask`` outputs equal band_recon_*'s
reconstruction and mask bit for bit while a band scrub leaves the h-map cached."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import fbm2d
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache, cache_key
from dynamix.engine.resolve import resolve, resolve_output
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import declared_outputs, get_device, keyed_params, validate_params
from dynamix.model.layer import Layer

MFA = ("holder_measure", "holder_multiaffine")
BAND_KNOBS = ("h_lo", "h_hi", "min_island", "max_island", "connectivity")
#: A band inside each method's h range on the probe field, with a sieve that removes islands.
BANDS = {"holder_measure": {"h_lo": -0.1, "h_hi": 0.1, "min_island": 3, "connectivity": "4"},
         "holder_multiaffine": {"h_lo": 0.0, "h_hi": 1.0, "min_island": 3,
                                "connectivity": "4"}}
RECON_OF = {"holder_measure": "band_recon_measure",
            "holder_multiaffine": "band_recon_multiaffine"}
#: Engine params as a saved project holds them.
SAVED = {"holder_measure": {"wavelet": "q_gaussian", "q_tsallis": 1.2, "r_min": 1.5,
                            "kappa": 6.0},
         "holder_multiaffine": {"wavelet": "q_mexican", "q_pairing": "q-paired",
                                "q_tsallis": 0.5, "estimator": "punctual"}}
#: The analysis keys those params resolve to over the engine knobs alone (computed at 5f28f50).
KEY_DEFAULT = {"holder_measure": "f0a1974ffaac1ccc2d7af2f9617d72ce",
               "holder_multiaffine": "a716ee6fa300a428a9a015d3c0f63353"}
KEY_SAVED = {"holder_measure": "ac9c2ebd6c0319c77d6477aae49c15c8",
             "holder_multiaffine": "5578978adeb0327c831ced77cc048db2"}
KEY_RECON_DEFAULT = {"band_recon_measure": "e3e9c280716170d3ef3c375ada05959d",
                     "band_recon_multiaffine": "f36edeba6d115af65b18d01dd73b1031"}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(n=64, H=0.8, seed=2):
    return RasterField(name="f", values=fbm2d(n, H, seed=seed), frame=LocalFrame(),
                       x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))


def _layer(*steps) -> Layer:
    refs = tuple(DeviceRef(name, dict(params)) for name, params in steps)
    return Layer(layer_id=1, name="mfa", source_id="s", chain=Chain(refs).materialized())


# --------------------------------------------------------------- the params

@pytest.mark.parametrize("mfa", MFA)
def test_the_band_knobs_are_band_recons_own_as_view_only_params(builtins, mfa):
    from dynamix.devices.band_recon import _BAND_KNOBS

    params = {p.name: p for p in get_device(mfa).params}
    knobs = [params[n] for n in BAND_KNOBS]
    assert all(p.view and p.section == "reconstruction" for p in knobs)
    assert [dataclasses.replace(p, view=False, section="") for p in knobs] == list(_BAND_KNOBS)
    show = params["show"]
    assert show.view and show.section == ""
    assert show.choices == ("h_map", "band", "band_mask") and show.default == "h_map"


@pytest.mark.parametrize("mfa", MFA)
def test_saved_params_without_the_band_knobs_validate_to_their_defaults(builtins, mfa):
    p = validate_params(get_device(mfa), SAVED[mfa])
    assert p["show"] == "h_map"
    assert [p[n] for n in BAND_KNOBS] == [-2.0, 0.0, 0, 0, "8"]


@pytest.mark.parametrize("mfa", MFA)
def test_the_analysis_key_is_the_one_saved_projects_had(builtins, mfa):
    dev = get_device(mfa)
    assert dev.cache_key("src", validate_params(dev, {})) == KEY_DEFAULT[mfa]
    banded = validate_params(dev, {**SAVED[mfa], **BANDS[mfa], "show": "band"})
    assert dev.cache_key("s", banded) == KEY_SAVED[mfa]
    assert cache_key(mfa, "s", keyed_params(dev, banded)) == KEY_SAVED[mfa]
    r = resolve(_layer((mfa, {**SAVED[mfa], **BANDS[mfa], "show": "band"})), _field(), Cache())
    assert r.analysis_key == KEY_SAVED[mfa]


@pytest.mark.parametrize("recon", sorted(KEY_RECON_DEFAULT))
def test_the_band_recon_variants_keep_their_params_and_keys(builtins, recon):
    dev = get_device(recon)
    names = [p.name for p in dev.params]
    assert "show" not in names and len(names) == len(set(names))
    assert names[-5:] == list(BAND_KNOBS)
    assert not any(p.view for p in dev.params)
    assert dev.cache_key("src", validate_params(dev, {})) == KEY_RECON_DEFAULT[recon]


@pytest.mark.parametrize("mfa", MFA)
def test_the_view_hands_over_the_cached_result_which_records_the_engine_knobs(builtins, mfa):
    field, cache = _field(), Cache()
    a = resolve(_layer((mfa, {"show": "h_map"})), field, cache)
    b = resolve(_layer((mfa, {"show": "band", "h_lo": -0.5})), field, cache)
    assert b.cache_misses == 0
    assert b.result is a.result and b.result is cache.get(b.analysis_key)
    dev = get_device(mfa)
    assert b.result["params"] == keyed_params(dev, validate_params(dev, {}))
