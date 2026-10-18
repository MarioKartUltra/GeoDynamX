# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The five vendored pieces the app still runs, pinned through the app's own seams on fixed inputs:
the 2-D Gaussian-pair CWT (``backend.cwt2d``) and the 1-D CWT (``backend.cwt1d``), the tensor WTMM
(``run_wtmm2d`` in tensor mode), the chain filters (the three chain devices), D(h) from chains
(``chain_groups.subset_hd``), the Dupont–Venugopal cumulants (``spectra.cascade_cumulants``) and the
1-D pipeline (``extrema1d``, ``chains1d``, ``partition1d``). The goldens under ``golden/vendor/``
were recorded from the vendored originals (tag ``vendored-snapshot-2026-10``) before spec 09 trimmed
them; a port of a piece (spec 09, Phase B) must reproduce its golden on the recording machine. The
1-D pieces are float64 end to end and must agree to 1e-10. The 2-D pieces run through the float32
transform on mlx, where the app's own filter bank and the vendored one differ by rounding, so they
agree to float32 tolerance (1e-5 relative, with the same share of each array's largest value as the
absolute floor), angles modulo 2π, and the per-(q, a) standard errors of h, D and τ, which amplify a
rounding difference through a fit over a handful of chains, to 1e-2. The chain counts are exact.

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


def _environment() -> dict:
    """What the goldens' numbers depend on: the machine, its chip (the mlx GPU), and the versions of
    the libraries the pieces compute through. A runner with another chip or newer libraries is not
    the recording machine: its float32 transforms round differently."""
    import importlib.metadata
    import subprocess

    env = {"machine": _machine()}
    if sys.platform == "darwin":
        try:
            env["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                         capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            pass
    for name in ("numpy", "mlx", "PyWavelets"):
        try:
            env[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return env


def _skip_off_the_recording_environment(meta: dict) -> None:
    recorded = meta.get("environment", {"machine": meta["machine"]})
    here = _environment()
    if {k: here.get(k) for k in recorded} != recorded:
        pytest.skip(f"recorded on {recorded}; this run is on {here}")


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
        res = run_wtmm2d(_field(stack), {"n_oct": 2, "n_voice": 3, "mode": "tensor",
                                         "fracint_alpha": 1.0, "smooth": True})
        return {"n_chains": len(res["chains"]),
                "chain_mod_sums": [float(np.sum(c["mod"])) for c in res["chains"]],
                "hd_std": res["hd_std"], "hd_cmax": res["hd_cmax"]}
    # eta = 1, the keep-both tie rule and the dip filling before the single-max test (smooth) as
    # recorded (the defaults at the snapshot), pinned so a change of default leaves the goldens
    # valid.
    res = run_wtmm2d(_field(v), {"n_oct": 2, "n_voice": 3, "fracint_alpha": 1.0,
                                 "tie_rule": "keep_both", "smooth": True})
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


#: Relative tolerance per piece: the 1-D pieces are float64 end to end; the 2-D pieces are what the
#: float32 transform makes of them.
TOLERANCE = {"cwt2d": 1e-5, "tensor": 1e-5, "chain_filters": 1e-5, "dh_from_chains": 1e-5,
             "cumulants": 1e-5, "cwtd": 1e-10, "pipeline_1d": 1e-10}


def _compare(piece: str, key: str, got: dict, want: dict) -> None:
    """``got[key]`` against ``want[key]`` at the piece's tolerance (see the module docstring). An
    argument is compared modulo 2π and only where the modulus beside it carries a signal (above
    1e-3 of its largest value): where the gradient vanishes the angle is rounding noise."""
    rtol = TOLERANCE[piece]
    if "sigma" in key and rtol > 1e-9:
        rtol = 1e-2
    g, w = np.asarray(got[key], dtype=np.float64), np.asarray(want[key], dtype=np.float64)
    assert g.shape == w.shape, (piece, key, g.shape, w.shape)
    if key.endswith("arg"):
        mod = np.asarray(want[key[:-3] + "mod"], dtype=np.float64)
        lit = mod > 1e-3 * mod.max()
        d = np.abs(np.angle(np.exp(1j * (g[lit] - w[lit]))))
        assert d.max() <= rtol * np.pi, (piece, key, float(d.max()))
        return
    finite = np.isfinite(w)
    assert np.array_equal(finite, np.isfinite(g)), (piece, key, "NaN pattern")
    scale = float(np.abs(w[finite]).max()) if finite.any() else 0.0
    np.testing.assert_allclose(g[finite], w[finite], rtol=rtol, atol=rtol * scale,
                               err_msg=f"{piece}/{key}")


@pytest.mark.parametrize("piece", PIECES)
def test_the_vendored_piece_reproduces_its_golden(piece):
    meta = json.loads((GOLDEN / "machine.json").read_text(encoding="utf-8"))
    _skip_off_the_recording_environment(meta)
    want = dict(np.load(GOLDEN / f"{piece}.npz", allow_pickle=False))
    got = _flatten(_run(piece))
    assert sorted(got) == sorted(want), (sorted(set(got) ^ set(want)))
    for key, w in want.items():
        g = got[key]
        if w.dtype.kind in "US":
            assert str(g) == str(w), key
        else:
            _compare(piece, key, got, want)


@pytest.mark.parametrize("piece", ["cwt2d", "tensor", "chain_filters", "dh_from_chains", "cumulants"])
def test_the_vendored_originals_still_reproduce_the_goldens_behind_the_flag(piece):
    """Ports 1–4 (spec 09, Phase B): the app's own 2-D mlx transform, tensor SVD, per-chain
    Hölder list, partition tables and cumulant fit are the live path; the vendored originals,
    reachable only with ``USE_VENDORED``, must agree with the same goldens until the last port
    removes them."""
    from dynamix.core import wtmm_backend

    pytest.importorskip("mlx.core")
    meta = json.loads((GOLDEN / "machine.json").read_text(encoding="utf-8"))
    _skip_off_the_recording_environment(meta)
    want = dict(np.load(GOLDEN / f"{piece}.npz", allow_pickle=False))
    wtmm_backend.USE_VENDORED = True
    try:
        got = _flatten(_run(piece))
    finally:
        wtmm_backend.USE_VENDORED = False
    for key, w in want.items():
        if w.dtype.kind not in "US":
            _compare(piece, key, got, want)


if __name__ == "__main__" and "--record" in sys.argv:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for piece in PIECES:
        flat = _flatten(_run(piece))
        np.savez(GOLDEN / f"{piece}.npz", **flat)
        print(f"recorded {piece}: {len(flat)} arrays")
    (GOLDEN / "machine.json").write_text(json.dumps({"machine": _machine(), "environment": _environment(),
                                                      "snapshot": "vendored-snapshot-2026-10"}) + "\n",
                                         encoding="utf-8")
