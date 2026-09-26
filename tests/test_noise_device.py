# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The noise Transform: seeded dither as a CHAIN STEP before the transform.

Distinct from
wtmm2d's own dither checkbox (auto half-LSB only): this one composes in the chain, has a manual
amplitude for stitched mosaics with no single lattice, a distribution, a seed and a sign."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynamix.core.rasterfield import RasterField
from dynamix.devices.noise import Noise
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import defaults_for, is_transform


def _field(vals, name="n.npy"):
    return RasterField._from_bare_array(np.asarray(vals, dtype=np.float64), Path(name))


def _stitched():
    rng = np.random.default_rng(3)
    a = np.round(rng.random((16, 32)) * 10) * 0.5          # one lattice...
    a[:, 16:] = np.round(a[:, 16:] * 7) / 7.0              # ...stitched to another
    return _field(a)


def _params(**over):
    d = defaults_for(Noise())
    d.update(over)
    return d


def test_is_a_transform_with_honest_defaults():
    assert is_transform(Noise())
    d = defaults_for(Noise())
    assert d["dist"] == "uniform" and d["amplitude"] == 0.0 and d["seed"] == 0 and d["sign"] == "add"


def test_manual_amplitude_bounds_uniform_noise_and_is_deterministic():
    f = _stitched()
    out1 = Noise().compute(f, _params(amplitude=0.05))
    out2 = Noise().compute(f, _params(amplitude=0.05))
    delta = out1.values - f.values
    assert np.abs(delta).max() <= 0.05 and np.abs(delta).max() > 0.0
    np.testing.assert_array_equal(out1.values, out2.values)          # fixed seed
    out3 = Noise().compute(f, _params(amplitude=0.05, seed=7))
    assert not np.array_equal(out1.values, out3.values)              # a different seed differs


def test_gaussian_amplitude_is_the_sigma():
    f = _field(np.zeros((64, 64)))
    out = Noise().compute(f, _params(dist="gaussian", amplitude=0.2))
    assert 0.15 < np.std(out.values) < 0.25


def test_subtract_mirrors_add():
    f = _stitched()
    add = Noise().compute(f, _params(amplitude=0.05))
    sub = Noise().compute(f, _params(amplitude=0.05, sign="subtract"))
    np.testing.assert_allclose(add.values + sub.values, 2.0 * f.values, atol=1e-12)


def test_auto_amplitude_uses_the_measured_lsb_and_passes_lattice_free_data_through():
    q = _field(np.round(np.linspace(0, 9, 64).reshape(8, 8)))        # integer lattice, lsb = 1
    out = Noise().compute(q, _params())
    assert np.abs(out.values - q.values).max() <= 0.5                # half LSB
    # Continuous data has no real lattice: measure_lsb collapses to the smallest unique-value
    # gap (the phase-4 followups caveat), so auto dither is NEGLIGIBLE -- fails toward no-op.
    from dynamix.core.mz_edges import measure_lsb
    smooth = _field(np.random.default_rng(0).random((8, 8)))
    out2 = Noise().compute(smooth, _params())
    assert np.abs(out2.values - smooth.values).max() <= measure_lsb(smooth.values) / 2.0
    const = _field(np.full((8, 8), 3.0))                             # NO gap at all: literal no-op
    assert Noise().compute(const, _params()) is const


def test_output_name_encodes_the_params_for_the_backend_stage_cache():
    f = _stitched()
    a = Noise().compute(f, _params(amplitude=0.05))
    b = Noise().compute(f, _params(amplitude=0.05, seed=1))
    assert a.name != f.name and a.name != b.name


def test_chains_before_wtmm_and_refuses_to_run_on_a_result(clean_registry):
    from dynamix.devices import register_builtin_devices
    register_builtin_devices()
    Chain((DeviceRef("noise", defaults_for(Noise())),
           DeviceRef("wtmm2d", defaults_for(__import__("dynamix.devices.wtmm", fromlist=["WTMM2D"]).WTMM2D())),
           )).materialized()                                          # two transforms: legal
    with pytest.raises(ValueError):
        Noise().compute({"extrema": []}, _params(amplitude=0.1))      # dropped AFTER wtmm


# ------------------------- a chain ending on a field transform lands gracefully
# [noise] alone resolves to a FIELD. Landing it must not crash on result.get(...) in the Qt
# loop: a crash there means resolved never fires, the window stays wedged for every later edit,
# and Noise followed by the WTMM standard preset sits at 0.

from tests.test_shell_roi_flow import parent_tif, _parent_field  # noqa: E402,F401


def test_noise_alone_lands_gracefully_and_wtmm_added_after_still_works(qtbot, clean_registry, parent_tif):
    from dynamix.shell.main_window import MainWindow
    win = MainWindow(steps=(("noise", {"amplitude": 0.05}),))
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=30000):            # must FIRE, not crash
        win.load_field(_parent_field(parent_tif), parent_tif)
    assert "wtmm" in win.statusBar().currentMessage().lower()      # the hint says what to add
    descriptors = [{"device": "noise", "params": {"amplitude": 0.05}},
                   {"device": "wtmm2d", "params": {"n_oct": 2, "n_voice": 2}},
                   {"device": "scale_select", "params": {"scale_idx": 0}}]
    win.strips.set_steps(descriptors, field=win.field)   # the zone originates a real edit
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_chain_edited(descriptors)
    hx, _ = win.canvas.hchain_item.getData()
    assert hx is not None and len(hx) > 0                          # analysis landed after all
