# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pytest


def fbm2d(n=64, H=0.7, seed=0):
    """Synthetic 2D fBm with known Hurst H (spectral synthesis, fixed seed)."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.fftfreq(n)[None, :]
    k = np.hypot(kx, ky); k[0, 0] = 1.0
    amp = k ** (-(H + 1.0)); amp[0, 0] = 0.0
    f = np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))).real
    return (f - f.mean()) / f.std()


@pytest.fixture
def fbm64():
    return fbm2d(64, 0.7, seed=0)


@pytest.fixture
def fbm3_64():
    """Synthetic (64, 64, 3) smooth 'log-orientation' field for the tensor-WTMM path."""
    return 0.05 * np.stack([fbm2d(64, 0.8, seed=s) for s in (1, 2, 3)], axis=-1)


# --- Phase 1 device-model stubs ---
# Shared by test_device, test_chain, test_project, test_projectfile

from dynamix.model.param import Param, ParamKind


class StubTransform:
    """Minimal Transform: expensive kind, so it carries compute() and cache_key()."""

    name = "t"
    params = (Param("scale", ParamKind.INT, default=4, min=1, max=8),)

    def compute(self, field, params, *, progress=None):
        return {"scale": params["scale"]}

    def cache_key(self, source_id, params):
        return f"{source_id}:{params['scale']}"


class StubFilter:
    """Minimal Filter: cheap kind, so it carries apply() and no cache_key()."""

    name = "f"
    params = (Param("cut", ParamKind.FLOAT, default=0.5, min=0.0, max=1.0),)

    def apply(self, result, params):
        return dict(result, cut=params["cut"])


@pytest.fixture
def stub_transform():
    return StubTransform()


@pytest.fixture
def stub_filter():
    return StubFilter()


@pytest.fixture(autouse=True)
def _registry_leak_guard():
    """Detect tests that leave DEVICES dirty without requesting clean_registry."""
    from dynamix.model.device import DEVICES

    before = set(DEVICES)
    yield
    after = set(DEVICES)
    if before != after:
        raise AssertionError(
            f"test leaked DEVICES state: before {sorted(before)}, after {sorted(after)}. "
            "Request clean_registry fixture if intentional."
        )


@pytest.fixture
def clean_registry():
    """Empty device registry, restored afterwards. Registration is global state."""
    from dynamix.model.device import DEVICES

    saved = dict(DEVICES)
    DEVICES.clear()
    yield DEVICES
    DEVICES.clear()
    DEVICES.update(saved)


@pytest.fixture(autouse=True)
def _dynamix_settings_isolated(tmp_path, monkeypatch):
    """Every test gets its OWN settings.json, never the real per-machine one under ``~/Library/
    Application Support/DynamiX`` -- ``MainWindow.__init__`` unconditionally calls
    ``load_settings()`` (to seed its Settings-menu checkbox), so without this every
    MainWindow-constructing test that doesn't set its own ``DYNAMIX_SETTINGS_PATH`` would read
    whatever real file happens to exist on the machine running the suite (``test_shell_units.py``
    alone builds four bare ``MainWindow()``s with no override of their own).

    Auto-run defaults ON here too: most of the suite predates the inert-open default
    (``main_window.py``'s ``load_field``) and expects it to dispatch the worker immediately, same
    as it always did. The handful of tests that specifically cover the true (off) default set
    their own ``DYNAMIX_SETTINGS_PATH``/``save_settings`` inside the TEST BODY -- which runs
    after every fixture's setup, including this one -- so their override wins.
    """
    from dynamix.shell.settings import Settings, save_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "dynamix_settings.json"))
    save_settings(Settings(auto_run_wtmm=True))


@pytest.fixture(autouse=True)
def _fft_policy_reset():
    """The FFT engine policy is process-global (fft_policy.configure also re-points mzlib's
    FFT); a MainWindow applies Settings at startup -- put the unconfigured default back after
    every test so no test leaks an engine/precision into the next."""
    yield
    from dynamix.core import fft_policy

    fft_policy.reset()
