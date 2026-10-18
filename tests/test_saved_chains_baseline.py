# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Saved chains keep loading and keep their results: a chain naming each registered device,
loaded through ``Chain.from_payload`` (the project-file path) and resolved without a shown level,
gives the results recorded in ``golden/saved_chains_baseline.json``. Counts must match exactly
and sums to a relative 1e-4. The recording is per machine, because backends agree to float32
tolerance only; on another platform the comparison is skipped.

Record (only when a result is meant to change):
    PYTHONNOUSERSITE=1 python tests/test_saved_chains_baseline.py --record
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "golden" / "saved_chains_baseline.json"

#: Devices that read a file or a point set this test does not build; their own tests cover them.
NEEDS_FILES = ("wtmm2d_roi", "bus", "derived_vectors", "backproject")
#: The analysis the filters read keeps the 32 px reflect pad and every maximum (spec 25's
#: ``FIXED_EDGES``), the tie rule before xsmurf's (``keep_both``) and the dip filling before the
#: single-max test (``smooth``, on when the baseline was recorded), so the filters' fingerprints
#: stay the ones recorded with them. The ``wtmm2d``
#: entry itself runs at the defaults.
_ANALYSIS = {"device": "wtmm2d", "params": {"n_oct": 2, "edge": "reflect", "pad_width": "fixed_px",
                                            "pad_px": 32, "border_cut": "none",
                                            "tie_rule": "keep_both", "smooth": True}}


def _machine() -> str:
    return f"{sys.platform}-{platform.machine()}"


def _field(shape):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    values = np.random.default_rng(0).standard_normal(shape).cumsum(0).cumsum(1)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(shape[1])), y_axis=np.arange(float(shape[0])))


def _chains(devices) -> dict:
    """``{label: (steps payload, field shape)}``: each Transform alone (on the input it reads),
    and each Filter after a small wtmm2d."""
    from dynamix.model.device import is_transform

    out = {}
    for name, device in devices.items():
        if name in NEEDS_FILES:
            continue
        if not is_transform(device):
            out[name] = ([_ANALYSIS, {"device": name, "params": {}}], (64, 64))
        elif name == "chain_topology":
            out[name] = ([_ANALYSIS, {"device": name, "params": {}}], (64, 64))
        elif name in ("pca", "band_select"):
            out[name] = ([{"device": name, "params": {}}], (64, 64, 3))
        else:
            out[name] = ([{"device": name, "params": {}}], (64, 64))
    return out


def _fingerprint(result) -> dict:
    if not isinstance(result, dict):              # a field-stage tail lands as a field
        return {"field": float(np.nansum(np.asarray(result.values)))}
    out = {"levels": [int(len(e.get("x", ()))) for e in result.get("extrema") or []],
           "mod": [float(np.nansum(e["mod"])) for e in result.get("extrema") or [] if "mod" in e],
           "chains": len(result.get("chains") or [])}
    for key in ("h_map", "raster_out", "filtered", "edge_channel"):
        value = result.get(key)
        if isinstance(value, np.ndarray):
            out[key] = float(np.nansum(value))
    return out


def _run(devices) -> dict:
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain
    from dynamix.model.layer import Layer

    cache, out = Cache(), {}
    for label, (steps, shape) in _chains(devices).items():
        chain = Chain.from_payload({"steps": steps})
        layer = Layer(layer_id=1, name=label, source_id="baseline", chain=chain)
        out[label] = _fingerprint(resolve(layer, _field(shape), cache).result)
    return out


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def test_every_saved_chain_loads_and_keeps_its_results(builtins):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if golden["machine"] != _machine():
        pytest.skip(f"recorded on {golden['machine']}")
    now = _run(builtins)
    assert sorted(now) == sorted(golden["chains"])
    for label, want in golden["chains"].items():
        got = now[label]
        assert got.keys() == want.keys(), label
        for key, value in want.items():
            if key in ("levels", "chains"):
                assert got[key] == value, (label, key)
            else:
                assert np.allclose(got[key], value, rtol=1e-4, atol=1e-6), (label, key)


if __name__ == "__main__" and "--record" in sys.argv:
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES

    register_builtin_devices()
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({"machine": _machine(), "chains": _run(DEVICES)}, indent=1,
                                 sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {GOLDEN}")
