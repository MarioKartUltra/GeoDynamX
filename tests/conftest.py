# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import os
import sys

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


def make_field(values=None, shape=(64, 56)):
    """A pixel-framed RasterField: ``values`` or a fixed-seed random walk of ``shape``."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    if values is None:
        values = np.random.default_rng(0).standard_normal(shape).cumsum(0).cumsum(1)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(values.shape[1])),
                       y_axis=np.arange(float(values.shape[0])))


def make_layer(*steps, window=None):
    """A layer whose chain is ``steps`` (``(device id, params)`` pairs), materialized; ``window``
    is an ROI window tag."""
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    chain = Chain(tuple(DeviceRef(n, dict(p)) for n, p in steps)).materialized()
    layer = Layer(layer_id=1, name="L", source_id="s", chain=chain)
    if window is not None:
        layer.tags["roi.window"] = window
    return layer


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
    """Empty device registry, restored afterwards. Registration is global state, and so are the
    plugin origins and the once-per-process discovery memo beside it."""
    from dynamix.model import plugins
    from dynamix.model.device import DEVICES

    saved, origins, found = dict(DEVICES), dict(plugins.ORIGINS), dict(plugins._DISCOVERED)
    DEVICES.clear()
    plugins.ORIGINS.clear()
    plugins._DISCOVERED.clear()
    yield DEVICES
    DEVICES.clear()
    DEVICES.update(saved)
    plugins.ORIGINS.clear()
    plugins.ORIGINS.update(origins)
    plugins._DISCOVERED.clear()
    plugins._DISCOVERED.update(found)


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
    """The FFT engine policy is process-global (mzlib's FFT follows it); a MainWindow applies
    Settings at startup -- put the unconfigured default back after every test so no test
    leaks an engine/precision into the next."""
    yield
    from dynamix.core import fft_policy

    fft_policy.reset()


@pytest.fixture(scope="session", autouse=True)
def _warm_numba_kernels():
    """Compile the M–Z (LastWave) numba kernels once, on the main thread, before the first test.

    They compile lazily on first use (``cache=True``). A Qt test whose worker thread was the first
    to reach them sat inside numba's compiler while the main thread polled the event loop, and the
    offscreen gate stalled there; one headless resolve of the recon preview here puts the compile
    ahead of every test, and the session pays it once instead of inside a worker.
    """
    import numpy as np

    from dynamix.core import fft_policy
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve, resolve_output
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.device import DEVICES
    from dynamix.model.layer import Layer

    saved = dict(DEVICES)
    DEVICES.clear()
    try:
        register_builtin_devices()
        values = np.random.default_rng(0).standard_normal((64, 56)).cumsum(0).cumsum(1)
        field = RasterField(name="warm", values=values, frame=LocalFrame(units="px"),
                            x_axis=np.arange(float(values.shape[1])),
                            y_axis=np.arange(float(values.shape[0])))
        chain = Chain((DeviceRef("mz_edges", {"n_levels": 3, "iterations": 2,
                                               "run_mode": "fixed"}),)).materialized()
        layer = Layer(layer_id=1, name="warm", source_id="warm", chain=chain)
        cache = Cache()
        resolve(layer, field, cache)
        resolve_output(layer, field, cache, "recon_preview")
    finally:
        DEVICES.clear()
        DEVICES.update(saved)
        fft_policy.reset()


@pytest.fixture
def double_fft():
    """FFTW3 at 64-bit for a test that asserts float64 identities (round trips, exact
    reconstruction): the unconfigured policy runs at 32-bit, as the app does by default."""
    from dynamix.core import fft_policy

    fft_policy.configure("fftw", 64)
    yield


@pytest.fixture(scope="session")
def qapp_args(pytestconfig):
    """pytest-qt's application arguments, plus the app's base style (``theme.BASE_STYLE``) where it
    has one: the launch sets it, so a widget test sees the control metrics a user's window has."""
    from dynamix.shell import theme

    args = [pytestconfig.getini("qt_qapp_name")]
    return args if theme.BASE_STYLE is None else [*args, "-style", theme.BASE_STYLE]


# --------------------------------------------------------------------- leaving a Qt or VTK session

def pytest_sessionfinish(session, exitstatus):
    session.config._session_exitstatus = int(exitstatus)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_unconfigure(config):
    """Once pytest is done, a session that loaded Qt or VTK leaves with pytest's own status rather
    than through the interpreter's teardown: Qt and VTK objects still alive there are destroyed out
    of order ("pure virtual method called", an abort after every test had passed on ubuntu)."""
    result = yield
    status = getattr(config, "_session_exitstatus", None)
    if status is not None and ({"PySide6", "vtkmodules"} & set(sys.modules)):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    return result


@pytest.fixture(autouse=True)
def _collect_on_the_main_thread():
    """In a session that loaded Qt, no collection starts during a test, where a compute worker could
    trip one and free a Qt object off its thread (dynamix.shell.gc_guard does the same for the
    app). Once a Qt application exists, each test's garbage is collected whole at its own end, on
    the main thread: a generational collection left some of a test's Qt and VTK objects to a later
    test's teardown, and freeing them there aborted on ubuntu. Without an application (a headless
    session that only imported Qt) the collector's own thresholds decide, since a full collection
    after every headless test slowed that gate by half."""
    if "PySide6" not in sys.modules:
        yield
        return
    import gc

    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        from PySide6 import QtCore

        if QtCore.QCoreApplication.instance() is not None:
            gc.collect()
        else:
            _c0, c1, c2 = gc.get_count()
            _t0, t1, t2 = gc.get_threshold()
            gc.collect(2 if c2 > t2 else 1 if c1 > t1 else 0)
        if was:
            gc.enable()
