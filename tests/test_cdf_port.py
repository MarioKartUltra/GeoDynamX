# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Port guard + functional gates for dynamix.core.cdf against its research origins.

The bodies are lifted from research/reconstruction scripts 24/25/27 (rescued probes
17-20, 2026-08-17). The ONLY permitted differences are the recorded edits below (the
extrema_io pattern: the scripts' module constants THETA/DT become theta/dt parameters,
and 25's ``lcdf`` is named ``lcdf_per`` here) -- reversing them must reproduce the
origin byte-for-byte, so a recorded change never becomes a blanket exemption.

Functional gates are script 28's own corrected checks (periodic BC + discrete symbol):
the trap 28 documents -- mixing boundary conditions or using the continuous symbol --
is exactly what these pin against."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from dynamix.core import cdf as lcdf

_REPO = pathlib.Path(__file__).resolve().parent.parent
_RES = _REPO / "research" / "reconstruction"

#: ours -> (origin script, origin def name, [(our_line, origin_line), ...])
_LIFTS = {
    "lap": ("24_lcdf_scalespace.py", "lap", []),
    "lcdf_evolve": ("24_lcdf_scalespace.py", "lcdf_evolve", [
        ("def lcdf_evolve(f, n_iter, snapshots=(), *, theta=np.pi / 30, dt=0.2):",
         "def lcdf_evolve(f, n_iter, snapshots=()):"),
        ("    c = np.exp(1j * theta)", "    c = np.exp(1j * THETA)"),
        ("        I = I + dt * c * lap(I)", "        I = I + DT * c * lap(I)"),
    ]),
    "ncdf": ("24_lcdf_scalespace.py", "ncdf", [
        ("def ncdf(f, k, n_iter, *, theta=np.pi / 30, dt=0.2):",
         "def ncdf(f, k, n_iter):"),
        ("        d = np.exp(1j * theta) / (1.0 + (I.imag / (k * theta)) ** 2)",
         "        d = np.exp(1j * THETA) / (1.0 + (I.imag / (k * THETA)) ** 2)"),
        ("        I = I + dt * flux", "        I = I + DT * flux"),
    ]),
    "lap_per": ("25_cdf_edges.py", "lap_per", []),
    "lcdf_per": ("25_cdf_edges.py", "lcdf", [
        ("def lcdf_per(f, n_iter, *, theta=np.pi / 30, dt=0.2):",
         "def lcdf(f, n_iter):"),
        ("    c = np.exp(1j * theta)", "    c = np.exp(1j * THETA)"),
        ("        I = I + dt * c * lap_per(I)", "        I = I + DT * c * lap_per(I)"),
    ]),
    "grad_nms": ("27_lcdf_colored.py", "grad_nms", []),
    "zero_crossings": ("27_lcdf_colored.py", "zero_crossings", []),
}


def _def_block(text: str, name: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"def {name}("))
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t")) or not lines[end].strip()):
        end += 1
    while not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


@pytest.mark.skipif(not _RES.is_dir(), reason="research/reconstruction scripts not present")
@pytest.mark.parametrize("ours_name", sorted(_LIFTS))
def test_lifted_bodies_reverse_to_their_origins(ours_name):
    script, origin_name, edits = _LIFTS[ours_name]
    ours = _def_block((_REPO / "src/dynamix/core/cdf.py").read_text(), ours_name)
    theirs = _def_block((_RES / script).read_text(), origin_name)
    reverted = list(ours)
    for mine, orig in edits:
        assert mine in reverted, (ours_name, mine)
        reverted[reverted.index(mine)] = orig
    assert reverted == theirs, ours_name


# ---------------------------------------------------------- script 28's gates, functional

def _lam_symbol(shape):
    wy = 2 * np.pi * np.fft.fftfreq(shape[0])[:, None]
    wx = 2 * np.pi * np.fft.fftfreq(shape[1])[None, :]
    return 2 * np.cos(wy) + 2 * np.cos(wx) - 4          # 5-point periodic Laplacian symbol


def test_periodic_evolution_equals_the_exact_discrete_symbol():
    """Gate (1): explicit-Euler LCDF on the torus == (1 + dt c lam(w))^n in Fourier, to
    machine precision -- the corrected check of script 28 (its predecessor mixed BCs and
    used the CONTINUOUS symbol, rel err 0.51-0.92; that trap is what this pins)."""
    rng = np.random.default_rng(3)
    f = rng.normal(size=(64, 64))
    theta, dt, n = np.pi / 30, 0.2, 40
    I = lcdf.lcdf_per(f, n, theta=theta, dt=dt)
    lam = _lam_symbol(f.shape)
    sym = (1.0 + dt * np.exp(1j * theta) * lam) ** n
    ref = np.fft.ifft2(sym * np.fft.fft2(f))
    assert np.max(np.abs(I - ref)) / np.max(np.abs(ref)) < 1e-10


def test_im_channel_is_the_ricker_cwt_identity():
    """Gate (2): Im(I) matches its exact first-order-in-theta expansion
    theta_eff * lam * (1 + dt lam cos(theta))^(n-1) applied to f (theta_eff =
    n dt sin(theta)) -- "theta * t * LoG(Gaussian * f)" in discrete-consistent form.
    Single-digit percent, per script 28."""
    rng = np.random.default_rng(4)
    f = rng.normal(size=(64, 64))
    theta, dt, n = np.pi / 30, 0.2, 40
    I = lcdf.lcdf_per(f, n, theta=theta, dt=dt)
    lam = _lam_symbol(f.shape)
    theta_eff = n * dt * np.sin(theta)
    ref = np.real(np.fft.ifft2(
        theta_eff * lam * (1.0 + dt * lam * np.cos(theta)) ** (n - 1) * np.fft.fft2(f)))
    rel = np.max(np.abs(I.imag - ref)) / np.max(np.abs(ref))
    assert rel < 0.1, rel


def test_ncdf_at_huge_k_is_lcdf_exactly():
    """The diffusivity d -> e^{i theta} as k -> inf, and the flux form with constant d IS
    d * lap -- so NCDF degenerates to the edge-BC LCDF step for step, machine-exact."""
    rng = np.random.default_rng(5)
    f = rng.normal(size=(32, 32))
    a = lcdf.ncdf(f, k=1e12, n_iter=15)
    b, _ = lcdf.lcdf_evolve(f, 15)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_edge_extractors_return_masks_with_coefficients():
    yy, xx = np.mgrid[:64, :64].astype(float)
    step = (xx > 32).astype(float)
    I, _ = lcdf.lcdf_evolve(step, 20)
    keep, mag = lcdf.grad_nms(I.real)
    assert keep.any() and mag.shape == step.shape
    cols = np.where(keep[32])[0]
    assert np.all(np.abs(cols - 32) <= 2)              # M-Z edges on the step
    zc, slope = lcdf.zero_crossings(I.imag)
    assert zc.any() and slope.shape == step.shape


def test_sigma_iters_is_the_script26_schedule():
    it = lcdf.sigma_iters((1.0, 2.0, 4.0, 8.0))
    assert it[1.0] >= 1 and it[8.0] > it[4.0] > it[2.0] > it[1.0]
    assert it[2.0] == max(1, int(round(4.0 / (2 * np.cos(np.pi / 30)) / 0.2)))
