# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""GeoDynamix_Beta's vendored wtmm / wtmm_ebsd: Windows-safe, and verbatim.

- With mlx absent (a Windows / Intel-Mac install), the vendored packages import, the policy runs
  every FFT on FFTW3, and a 2-D WTMM transform + partition function run end to end.
- Every vendored file is byte-identical to its source at the vendoring commit, except the six
  recorded edits (import rewrite, trimmed wtmm_ebsd/__init__.py and wtmm/__init__.py, guarded mlx
  in cwt2d.py, each mlx product evaluated before numpy reads it in cwt2d.py, and the Borges A_q
  Gamma ratios in wtmm/wavelets.py as Pochhammer symbols).
- The packages carry only the modules the app loads (spec 09, Phase A): the survivors are pinned
  below, and a trace of every seam the app runs through (``tests/test_vendor_goldens.py``'s pieces)
  loads exactly them. The whole copy is at tag ``vendored-snapshot-2026-10``.
"""
from __future__ import annotations

import difflib
import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
VENDOR = REPO / "src" / "dynamix" / "_vendor"
SOURCE = pathlib.Path("~/projects/Creep/wavelet").expanduser()


def test_without_mlx_everything_runs_on_fftw3():
    code = textwrap.dedent("""
        import sys
        sys.modules["mlx"] = sys.modules["mlx.core"] = None     # a Windows-like install
        import numpy as np
        import dynamix._vendor.wtmm_ebsd as we                 # the guarded cwt2d import
        from dynamix._vendor.wtmm_ebsd.partition import build_hd_from_chains  # noqa: F401
        from dynamix.core import fft_policy
        from dynamix.core.frames import LocalFrame
        from dynamix.core.rasterfield import RasterField
        from dynamix.core.wtmm_backend import run_wtmm2d
        assert fft_policy.active().name == "pyfftw", fft_policy.active().name
        rng = np.random.default_rng(0)
        v = rng.standard_normal((64, 64)).cumsum(0).cumsum(1)
        f = RasterField(name="nomlx", values=v, frame=LocalFrame(),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
        out = run_wtmm2d(f, {"n_oct": 2, "n_voice": 2, "a_min": 1.0})
        assert len(out["chains"]) > 0 and out["hd_std"]
        assert not any(m == "mlx" or m.startswith("mlx.") for m in sys.modules
                       if sys.modules[m] is not None)
        print("FFTW_OK")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO))
    assert "FFTW_OK" in res.stdout, res.stderr[-2000:]


def _normalised(text: str) -> str:
    """The only permitted content edit: absolute imports into dynamix._vendor."""
    return (text.replace("dynamix._vendor.wtmm_ebsd", "wtmm_ebsd")
                .replace("dynamix._vendor.wtmm", "wtmm"))


@pytest.mark.skipif(not SOURCE.is_dir(), reason=f"source checkout not present at {SOURCE}")
@pytest.mark.parametrize("pkg", ["wtmm", "wtmm_ebsd"])
def test_vendored_files_are_verbatim_except_the_recorded_edits(pkg):
    edited = {("wtmm_ebsd", "__init__.py"), ("wtmm_ebsd", "cwt2d.py"), ("wtmm", "wavelets.py"),
              ("wtmm", "__init__.py")}
    for path in sorted((VENDOR / pkg).glob("*.py")):
        if (pkg, path.name) in edited:
            continue
        original = (SOURCE / pkg / path.name).read_text()
        assert _normalised(path.read_text()) == original, f"{pkg}/{path.name} drifted"


@pytest.mark.skipif(not SOURCE.is_dir(), reason=f"source checkout not present at {SOURCE}")
def test_the_two_edited_files_differ_only_by_their_recorded_edits():
    ours = (VENDOR / "wtmm_ebsd" / "cwt2d.py").read_text().splitlines()
    theirs = (SOURCE / "wtmm_ebsd" / "cwt2d.py").read_text().splitlines()
    removed = [ln for ln in theirs if ln not in ours]
    assert removed == ["import mlx.core as mx"]
    added = [ln.split("#")[0].strip() for ln in ours if ln not in theirs]
    assert added == ["try:", "import mlx.core as mx", "except ImportError:", "mx = None",
                     "mx.eval(conv)"]
    init = (VENDOR / "wtmm_ebsd" / "__init__.py").read_text()
    for gone in ("orientation", "symmetry", "grains", "demean", "cpo", "ipf", "kam",
                 "quality", "shuffles", "spatial_stats", "viewer"):
        assert f"from .{gone} import" not in init and f"    {gone},\n" not in init
    init = (VENDOR / "wtmm" / "__init__.py").read_text()
    for gone in ("colormaps", "hurst", "io", "datasets", "catalog", "spectra", "perline"):
        assert f"from .{gone} import" not in init and f"from . import {gone}" not in init


#: The modules the app loads (``dynamix.core.wtmm_backend``, ``core.spectra``, ``core.chain_groups``,
#: ``devices.chain_filters``) plus ``wtmm.signals`` (a test fixture source) and ``wtmm_ebsd.nye``
#: (named live by the author with the tensor WTMM; no app caller today, kept until the author rules).
SURVIVORS = {
    "wtmm": {"__init__", "wavelets", "cwt", "signals", "extrema", "chains", "partition",
             "dupont_venugopal"},
    "wtmm_ebsd": {"__init__", "chain_filters", "cwt2d", "nye", "partition", "twtmm"},
}


#: Survivors a Phase B port made unreachable from the app's seams (the app runs its own code
#: there now; the vendored module stays on disk, behind ``wtmm_backend.USE_VENDORED``, until the
#: last port removes it).
PORTED = {"wtmm.dupont_venugopal"}


@pytest.mark.parametrize("pkg", sorted(SURVIVORS))
def test_the_vendored_packages_carry_exactly_the_survivors(pkg):
    assert {p.stem for p in (VENDOR / pkg).glob("*.py")} == SURVIVORS[pkg]


def test_every_seam_the_app_runs_loads_only_survivors():
    """A fresh interpreter runs every golden piece (the app's seams into the vendored code) and
    loads exactly the survivors that no port has replaced yet (``PORTED``), so nothing the app
    reaches was trimmed away and each port's module is indeed no longer reached."""
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, "tests")
        import test_vendor_goldens as g
        for piece in g.PIECES:
            g._run(piece)
        print("LOADED", sorted(m for m in sys.modules if m.startswith("dynamix._vendor.")))
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO))
    assert "LOADED" in res.stdout, res.stderr[-2000:]
    loaded = {m.replace("dynamix._vendor.", "") for m in eval(res.stdout.split("LOADED", 1)[1].strip())}
    expected = {f"{pkg}.{m}" for pkg, mods in SURVIVORS.items() for m in mods if m != "__init__"}
    expected |= {"wtmm", "wtmm_ebsd"}
    assert loaded == expected - PORTED, {"loaded only": sorted(loaded - expected),
                                         "still loaded though ported": sorted(loaded & PORTED),
                                         "survivor not loaded": sorted(expected - PORTED - loaded)}


@pytest.mark.skipif(not SOURCE.is_dir(), reason=f"source checkout not present at {SOURCE}")
def test_wavelets_differs_only_by_the_borges_constant_fix():
    """Eqs. 17-18 of Borges et al. 2004 take Gamma(a)/Gamma(a - 5/2) and
    Gamma(a + 3/2)/Gamma(a + 1) with a ~ 2/|q - 1|; the literal ratio overflows to NaN for
    0.988 < q < 1.012. The same ratios as Pochhammer symbols are finite through q = 1."""
    ours = _normalised((VENDOR / "wtmm" / "wavelets.py").read_text()).splitlines()
    theirs = (SOURCE / "wtmm" / "wavelets.py").read_text().splitlines()
    diff = [ln for ln in difflib.unified_diff(theirs, ours, lineterm="", n=0)
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    assert [ln[1:].strip() for ln in diff if ln[0] == "-"] == [
        "np.sqrt((q - 1.0)**2.5 * gamma_func(arg1) / gamma_func(arg1 - 2.5))",
        "np.sqrt((1.0 - q)**0.5 * gamma_func(arg2 + 1.5) / gamma_func(arg2 + 1.0))"]
    assert [ln[1:].strip() for ln in diff if ln[0] == "+"] == [
        "from scipy.special import poch", "",
        "np.sqrt((q - 1.0)**2.5 * poch(arg1 - 2.5, 2.5))",
        "np.sqrt((1.0 - q)**0.5 * poch(arg2 + 1.0, 0.5))"]


def test_an_mlx_error_in_the_vendored_cwt_is_a_python_exception_never_an_abort():
    """cwt_2d_f32 evaluates each product before reading it into numpy: an error raised while
    np.array pulls an unevaluated mlx array cannot cross that C boundary and aborts the
    process (a GPU out-of-memory on a huge raster did). Run in a subprocess: the unfixed
    failure kills the interpreter."""
    pytest.importorskip("mlx.core")
    script = textwrap.dedent("""
        import numpy as np, mlx.core as mx
        import dynamix._vendor.wtmm_ebsd.cwt2d as cwt
        cap = mx.device_info()["max_buffer_length"]
        rows = cap // (8 * 8 * 12000) + 1                  # complex64 just over the Metal cap
        mx.fft.ifft2 = lambda a, *k, **kw: mx.zeros((8, rows, 12000), dtype=mx.complex64)
        try:
            cwt.cwt_2d_f32(np.zeros((16, 16)), [2.0], verbose=False)
        except RuntimeError as exc:
            print("raised:", exc)
    """)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr[-500:]
    assert "raised:" in run.stdout and "metal::malloc" in run.stdout
