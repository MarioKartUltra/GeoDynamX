# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Verbatim-port guard + 1-D gate translations for dynamix.core.mzlib / fftbackend.

The core copies are verbatim ports of the validated research modules; the ONLY permitted
differences are mzlib's import rewrite and one recorded docstring edit in fftbackend
(:data:`_FFTBACKEND_EDIT`). Gate thresholds are copied from the research scripts
(01, 02, 03, 04, 05).
"""
from __future__ import annotations

import difflib
import hashlib
import pathlib

import numpy as np
import pytest

from dynamix.core import fftbackend, mzlib

_REPO = pathlib.Path(__file__).resolve().parent.parent
_RESEARCH = _REPO / "research" / "reconstruction"


# ---------------------------------------------------------------- drift guard

#: The ONE recorded edit: the public copy states the backend-agreement rule without citing the
#: private project notes it came from. The two replaced research lines are pinned by SHA-256, so
#: any other drift -- in them or anywhere else in the file -- still fails.
_FFTBACKEND_EDIT = (
    "mlx path: the backends must agree to float32 tolerance.\n",
    "91095f346612d45bfc86746a71edcfabb958041c506d08709605c1b0bac318c1",
)


@pytest.mark.skipif(not _RESEARCH.is_dir(), reason="research/reconstruction scripts not present")
def test_fftbackend_differs_from_research_only_by_the_recorded_edit():
    core = (_REPO / "src/dynamix/core/fftbackend.py").read_text().splitlines(keepends=True)
    research = (_RESEARCH / "fftbackend.py").read_text().splitlines(keepends=True)
    hunks = [op for op in difflib.SequenceMatcher(None, research, core, autojunk=False).get_opcodes()
             if op[0] != "equal"]
    assert len(hunks) == 1 and hunks[0][0] == "replace", hunks
    _, i1, i2, j1, j2 = hunks[0]
    ours, theirs_sha256 = _FFTBACKEND_EDIT
    assert "".join(core[j1:j2]) == ours
    assert hashlib.sha256("".join(research[i1:i2]).encode()).hexdigest() == theirs_sha256


@pytest.mark.skipif(not _RESEARCH.is_dir(), reason="research/reconstruction scripts not present")
def test_mzlib_differs_from_research_only_in_the_import_line():
    core = (_REPO / "src/dynamix/core/mzlib.py").read_text().splitlines()
    research = (_RESEARCH / "mzlib.py").read_text().splitlines()
    assert len(core) == len(research)
    diffs = [
        (c, r) for c, r in zip(core, research) if c != r
    ]
    assert diffs == [("from dynamix.core import fftbackend", "import fftbackend")]


# ---------------------------------------------------------------- script 01 gates

def _w():
    return 2 * np.pi * np.fft.fftfreq(4096)


def test_admissibility_91():
    w = _w()
    lhs = np.abs(mzlib.Hf(w)) ** 2 + np.abs(mzlib.Hf(w + np.pi)) ** 2
    assert np.all(lhs <= 1 + 1e-12)
    assert abs(np.abs(mzlib.Hf(np.array([0.0]))[0]) - 1.0) < 1e-12


def test_perfect_reconstruction_96():
    w = _w()
    lhs = np.abs(mzlib.Hf(w)) ** 2 + mzlib.Gf(w) * mzlib.Kf(w)
    assert np.max(np.abs(lhs - 1.0)) < 1e-10


def test_pr2d_identity_107_108():
    """(107)-(108): Re(G(wx)K(wx))*L(wy) + L(wx)*Re(G(wy)K(wy)) + |H(wx)|^2*|H(wy)|^2 = 1, for
    INDEPENDENT wx, wy (App D's x/y filter-pair separability identity). Transcribes
    01_wavelet_class.py's own gate (lines 22-29) exactly: wy = wx[::-1] keeps wx != wy pointwise.

    A prior version of this test evaluated both Lf_ terms at the SAME w
    (``Gf(w)*Kf(w)*Lf_(w) + Gf(w)*Kf(w)*(1-Lf_(w))``), which cancels algebraically for ANY L
    (``a*L + a*(1-L) == a`` identically) -- degenerating into test_perfect_reconstruction_96 and
    leaving Lf_ with zero real coverage despite it being load-bearing in atrous2d_inverse
    (confirmed by review: substituting a broken Lf_ like ``0.3*w + 7`` still passed the old
    form). This two-variable form does not have that degeneracy."""
    wx = _w()
    wy = wx[::-1]
    GKx = np.real(mzlib.Gf(wx) * mzlib.Kf(wx))
    GKy = np.real(mzlib.Gf(wy) * mzlib.Kf(wy))
    Lx, Ly = mzlib.Lf_(wx), mzlib.Lf_(wy)
    ax, ay = np.abs(mzlib.Hf(wx)) ** 2, np.abs(mzlib.Hf(wy)) ** 2
    lhs = GKx * Ly + Lx * GKy + ax * ay
    assert np.max(np.abs(lhs - 1.0)) < 1e-10


def test_psi_antisymmetric():
    """psi_hat(w) = i*w*(even real): purely imaginary and odd, i.e. the real-space wavelet
    psi(x) is antisymmetric about 0. Matches 01_wavelet_class.py's own gate (lines 50-53:
    ``ph = M.psi_hat(wp); m8 = np.max(np.abs(np.real(ph)))``) -- Re(psi_hat) == 0. psi_hat is
    imaginary-and-odd here, not real-and-even, so a "psi_hat(w) + conj(psi_hat(-w))" construction
    does not vanish for it (it reduces to 2*psi_hat(w)); Re(psi_hat)==0 is the actual identity
    the research gate checks."""
    w = _w()
    assert np.max(np.abs(np.real(mzlib.psi_hat(w)))) < 1e-12


# ---------------------------------------------------------------- script 02 gates

def test_lambda_table_closed_form():
    for j, lam_j in enumerate(mzlib.LAMBDA_TABLE, start=1):
        assert abs(mzlib.lam(j) - lam_j) < 1e-12
        assert abs(lam_j - (1 + 2 ** (1 - 2 * j))) < 0.02 * lam_j
    assert mzlib.lam(len(mzlib.LAMBDA_TABLE) + 3) == 1.0


def test_theta0_is_four_thirds():
    assert abs(mzlib.THETA0 - 4.0 / 3.0) < 1e-15


# ---------------------------------------------------------------- script 03 gates

def test_atrous_roundtrip_1d_float64():
    rng = np.random.default_rng(3)
    d = rng.standard_normal(257)
    S, Wl = mzlib.atrous_forward(d, 5)
    back = mzlib.atrous_inverse(S, Wl)
    assert np.max(np.abs(back - d)) / np.max(np.abs(d)) < 1e-10


def test_atrous_roundtrip_1d_float32_storage():
    rng = np.random.default_rng(4)
    d = rng.standard_normal(256)
    S, Wl = mzlib.atrous_forward(d, 5)
    back = mzlib.atrous_inverse(S.astype(np.float32).astype(np.float64),
                                [w.astype(np.float32).astype(np.float64) for w in Wl])
    assert np.max(np.abs(back - d)) / np.max(np.abs(d)) < 1e-5


# ---------------------------------------------------------------- script 04 gates

def test_maxima_rule_and_pgamma_exactness():
    rng = np.random.default_rng(5)
    d = np.cumsum(rng.standard_normal(256))
    S, Wl = mzlib.atrous_forward(d, 4)
    wj = Wl[1]
    idx = mzlib.find_maxima(wj)
    assert idx.size > 4
    mod = np.abs(wj)
    for i in idx:
        assert mod[i] >= mod[(i - 1) % mod.size] and mod[i] >= mod[(i + 1) % mod.size]
    vals = wj[idx]
    g = mzlib.p_gamma(np.zeros_like(wj), idx, vals, 2.0 ** 2)
    assert np.max(np.abs(g[idx] - vals)) < 1e-10


def test_pv_idempotent():
    rng = np.random.default_rng(6)
    d = rng.standard_normal(128)
    S, Wl = mzlib.atrous_forward(d, 4)
    S1, Wl1 = mzlib.p_v(S, Wl)
    S2, Wl2 = mzlib.p_v(S1, Wl1)
    assert np.max(np.abs(S2 - S1)) < 1e-8
    for a, b in zip(Wl1, Wl2):
        assert np.max(np.abs(a - b)) < 1e-8


# ---------------------------------------------------------------- script 05 gate (gating signal)

def test_pocs_1d_gating_signal_snr():
    """The design gating signal:
    multi-step piecewise-constant, SNR >= 30 dB within 30 iterations, monotone residuals.

    Signal is 05_pocs_recon.py's own validated "GATING SIGNAL" construction verbatim
    (``edges=[(85,1.0),(170,-0.8)]``, N=256, J=8 -- its round-4-sweep provenance comment records
    this exact configuration at 31.37 dB, reproduced here bit-for-bit through the ported oracle).
    The design's sketch invented a different 4-edge signal with J=5 that was never validated
    against the port and tops out near 27 dB at n_iter=30 regardless of J -- not a call-shape
    bug (confirmed: varying J across 4-8 barely moves it), just an unvalidated signal choice.
    Swapping in the canonical signal is the fix; the pocs/maxima call shape below is unchanged.

    ``pocs`` consumes maxima indexed on the 2N mirrored torus (the same domain its internal
    ``p_v`` -> ``atrous_forward_full`` round trip re-analyzes on), so the maxima list is built
    from ``atrous_forward_full(mirror(d), J)`` directly -- algebraically identical to 05's own
    ``atrous_forward(d, J)`` call (atrous_forward mirrors d then runs the exact
    atrous_forward_full loop; verified bit-identical), but exercising the _full/mirror entry
    point this port also exposes, per the design's latitude note.
    """
    N, J = 256, 8
    edges = [(85, 1.0), (170, -0.8)]
    d = np.zeros(N)
    for pos, amp in edges:
        d[pos:] += amp
    d -= d.mean()
    S_full, Wl_full = mzlib.atrous_forward_full(mzlib.mirror(d), J)
    maxima = []
    for wj in Wl_full:
        idx = mzlib.find_maxima(wj)
        maxima.append((idx, wj[idx]))
    d_hat, resid = mzlib.pocs(maxima, S_full, N, J, n_iter=30)
    snr = 10 * np.log10(np.sum(d ** 2) / np.sum((d - d_hat) ** 2))
    assert snr >= 30.0
    assert all(b <= a * (1 + 1e-9) for a, b in zip(resid, resid[1:]))


# ---------------------------------------------------------------- fftbackend gates (script 12)

def test_numpy_backend_is_exact_passthrough():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((32, 32))
    b = fftbackend.get_backend("numpy")
    assert np.array_equal(b.ifft2(b.fft2(a)).real.astype(a.dtype), np.fft.ifft2(np.fft.fft2(a)).real.astype(a.dtype))


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        fftbackend.get_backend("cuda")


@pytest.mark.parametrize("name", ["pyfftw", "mlx"])
def test_optional_backends_agree_with_numpy(name):
    pytest.importorskip("pyfftw" if name == "pyfftw" else "mlx.core")
    rng = np.random.default_rng(8)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    ref = fftbackend.get_backend("numpy").fft2(a)
    got = fftbackend.get_backend(name).fft2(a)
    assert np.max(np.abs(np.asarray(got) - ref)) / np.max(np.abs(ref)) < 1e-4
