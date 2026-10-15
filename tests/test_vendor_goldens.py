# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The five vendored pieces the app still runs, pinned through the app's own seams on fixed inputs:
the 2-D Gaussian-pair CWT (``backend.cwt2d``) and the 1-D CWT (``backend.cwt1d``), the tensor WTMM
(``run_wtmm2d`` in tensor mode), the chain filters (the three chain devices), D(h) from chains
(``chain_groups.subset_hd``), the Dupont–Venugopal cumulants (``spectra.cascade_cumulants``) and the
1-D pipeline (``extrema1d``, ``chains1d``, ``partition1d``). The goldens under ``golden/vendor/``
were recorded from the vendored originals (tag ``vendored-snapshot-2026-10``) before spec 09 trimmed
them; a port of a piece (spec 09, Phase B) must reproduce its golden to rtol 1e-6 for float32 and
1e-10 for float64 on the recording machine (backends agree to float32 tolerance elsewhere).

    PYTHONNOUSERSITE=1 python tests/test_vendor_goldens.py --record
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "golden" / "vendor"
PIECES = ("cwt2d", "cwtd", "tensor", "chain_filters", "dh_from_chains", "cumulants", "pipeline_1d")


def _machine() -> str:
    return f"{sys.platform}-{platform.machine()}"


def _field(values):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    return RasterField(name="golden", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(values.shape[1])),
                       y_axis=np.arange(float(values.shape[0])))


def _inputs():
    rng = np.random.default_rng(20261015)
    v = rng.standard_normal((64, 64)).cumsum(0).cumsum(1)
    stack = np.stack([v, v[::-1], v[:, ::-1]], axis=-1)
    signal = rng.standard_normal(1024).cumsum()
    return v, stack, signal


def _flatten(obj, prefix="", out=None) -> dict:
    """Every array, number or string inside nested dicts, lists and tuples, as a flat
    ``path -> array`` dict (strings become 0-d unicode arrays; a list of arrays of one shape is
    stacked, otherwise indexed)."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _flatten(obj[k], f"{prefix}{k}/", out)
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, (int, float, np.integer, np.floating)) for x in obj):
            out[prefix.rstrip("/")] = np.asarray(obj, dtype=np.float64)
        else:
            for i, x in enumerate(obj):
                _flatten(x, f"{prefix}{i}/", out)
    elif isinstance(obj, (np.ndarray, int, float, np.integer, np.floating, bool, np.bool_)):
        out[prefix.rstrip("/")] = np.asarray(obj)
    elif isinstance(obj, str):
        out[prefix.rstrip("/")] = np.asarray(obj)
    elif obj is None:
        out[prefix.rstrip("/")] = np.asarray("None")
    else:
        out[prefix.rstrip("/")] = np.asarray(repr(obj))
    return out


def _run(piece: str) -> dict:
    from dynamix.core.wtmm_backend import compute_scales2d, get_backend, run_wtmm2d

    v, stack, signal = _inputs()
    backend = get_backend("python")
    scales = compute_scales2d(2, 3)
    if piece == "cwt2d":
        cwt = backend.cwt2d(v, scales)
        return {"mod": cwt["mod"], "arg": cwt["arg"]}
    if piece == "cwtd":
        coeffs, scales1, valid = backend.cwt1d(signal, 1.0, 3, 4)
        return {"coeffs": coeffs, "scales": scales1, "valid": valid}
    if piece == "tensor":
        res = run_wtmm2d(_field(stack), {"n_oct": 2, "n_voice": 3, "mode": "tensor"})
        return {"n_chains": len(res["chains"]),
                "chain_mod_sums": [float(np.sum(c["mod"])) for c in res["chains"]],
                "hd_std": res["hd_std"], "hd_cmax": res["hd_cmax"]}
    res = run_wtmm2d(_field(v), {"n_oct": 2, "n_voice": 3})
    if piece == "chain_filters":
        from dynamix.devices.chain_filters import (ChainHolderFilter, ChainLengthFilter,
                                                   ChainModulusFilter)
        from dynamix.model.device import validate_params

        out = {}
        for device in (ChainHolderFilter(), ChainModulusFilter(), ChainLengthFilter()):
            kept = device.apply(dict(res), validate_params(device, {}))["chains"]
            out[device.name] = {"n": len(kept),
                                "mod_sum": float(sum(np.sum(c["mod"]) for c in kept))}
        return out
    if piece == "dh_from_chains":
        from dynamix.core.chain_groups import subset_hd

        mask = np.arange(len(res["chains"])) % 2 == 0
        hd_std, hd_cmax = subset_hd(res["chains"], res["scales"], [-2.0, -1.0, 0.0, 1.0, 2.0], mask)
        return {"hd_std": hd_std, "hd_cmax": hd_cmax}
    if piece == "cumulants":
        from dynamix.core.spectra import cascade_cumulants

        mods = [np.abs(np.asarray(e["mod"], dtype=np.float64)) for e in res["extrema"]]
        log2 = np.log2(np.asarray(res["scales"]))
        return cascade_cumulants(mods, res["scales"], float(log2[0]), float(log2[-1]))
    if piece == "pipeline_1d":
        coeffs, scales1, valid = backend.cwt1d(signal, 1.0, 3, 4)
        ext = backend.extrema1d(coeffs, scales1, valid_ranges=valid)
        extrep_abs, extrep_ord = ext[0], ext[1]
        chains = backend.chains1d(extrep_abs, extrep_ord)
        part = backend.partition1d(extrep_ord, 3, 4, 1.0, [-2.0, 0.0, 2.0])
        return {"extrema": _count(ext), "chains": _count(chains), "partition": part}
    raise KeyError(piece)


def _count(obj):
    """A nested structure reduced to its lengths and sums: enough to pin a port."""
    if isinstance(obj, dict):
        return {k: _count(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return {"len": len(obj), "items": [_count(x) for x in obj[:8]]}
    arr = np.asarray(obj)
    if arr.dtype.kind in "iufb":
        return {"shape": list(arr.shape), "sum": float(np.nansum(arr.astype(np.float64)))}
    return repr(obj)[:60]


def _tolerance(arr: np.ndarray) -> float:
    return 1e-6 if arr.dtype == np.float32 else 1e-10


@pytest.mark.parametrize("piece", PIECES)
def test_the_vendored_piece_reproduces_its_golden(piece):
    meta = json.loads((GOLDEN / "machine.json").read_text())
    if meta["machine"] != _machine():
        pytest.skip(f"recorded on {meta['machine']}")
    want = dict(np.load(GOLDEN / f"{piece}.npz", allow_pickle=False))
    got = _flatten(_run(piece))
    assert sorted(got) == sorted(want), (sorted(set(got) ^ set(want)))
    for key, w in want.items():
        g = got[key]
        if w.dtype.kind in "US":
            assert str(g) == str(w), key
        else:
            np.testing.assert_allclose(g, w, rtol=_tolerance(np.asarray(g)), atol=0.0,
                                       err_msg=f"{piece}/{key}")


if __name__ == "__main__" and "--record" in sys.argv:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for piece in PIECES:
        flat = _flatten(_run(piece))
        np.savez(GOLDEN / f"{piece}.npz", **flat)
        print(f"recorded {piece}: {len(flat)} arrays")
    (GOLDEN / "machine.json").write_text(json.dumps({"machine": _machine(),
                                                      "snapshot": "vendored-snapshot-2026-10"}) + "\n")
