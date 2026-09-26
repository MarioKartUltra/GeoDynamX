# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""GeoDynamix_Beta's vendored wtmm / wtmm_ebsd: Windows-safe, and verbatim.

- With mlx absent (a Windows / Intel-Mac install), the vendored packages import, the policy runs
  every FFT on FFTW3, and a 2-D WTMM transform + partition function run end to end.
- Every vendored file is byte-identical to its source at the vendoring commit, except the four
  recorded edits (import rewrite, trimmed wtmm_ebsd/__init__.py, guarded mlx in cwt2d.py, and
  the Borges A_q Gamma ratios in wtmm/wavelets.py as Pochhammer symbols).
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
    edited = {("wtmm_ebsd", "__init__.py"), ("wtmm_ebsd", "cwt2d.py"), ("wtmm", "wavelets.py")}
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
    init = (VENDOR / "wtmm_ebsd" / "__init__.py").read_text()
    for gone in ("orientation", "symmetry", "grains", "demean", "cpo", "ipf", "kam"):
        assert f"from .{gone} import" not in init


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
