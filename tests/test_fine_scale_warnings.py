# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Fine scales run with a warning (spec 21): η, the knob-warning hook, the ⚠ glyph, the ROI floor as
a recorded warning, the Turiel tools' own conditions on the glyph, and the warning in the .gprv."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dynamix.core import undersampling
from dynamix.core.wtmm_backend import compute_scales2d
from dynamix.devices.wtmm import WTMM2D
from dynamix.devices.wtmm_roi import WTMM2DROI
from dynamix.model.device import defaults_for, validate_params


@pytest.fixture(autouse=True)
def _default_threshold():
    undersampling.configure(1e-3)
    yield
    undersampling.configure(1e-3)


def _finest(a_min: float) -> float:
    return compute_scales2d(1, 1, a_min=a_min)[0]


def _wtmm(**kw) -> dict:
    dev = WTMM2D()
    return dict(defaults_for(dev), **kw)


# 1. η reproduces the spec's table (fixed width, two significant figures).

_TABLE = {
    ("mexican", None): {0.25: 0.86, 0.5: 7.1e-2, 0.75: 1.8e-4, 1.0: 1.4e-8},
    ("gaussian", None): {0.25: 0.43, 0.5: 6.9e-3, 0.75: 4.4e-6, 1.0: 1.2e-10},
    ("q_gaussian", 1.5): {0.25: 0.32, 0.5: 1.4e-2, 0.75: 3.4e-4, 1.0: 5.8e-6},
    ("q_gaussian", 0.5): {0.25: 0.67, 0.5: 5.8e-2, 0.75: 2.1e-2, 1.0: 8.4e-3},
}


@pytest.mark.parametrize("name,q", list(_TABLE))
def test_eta_reproduces_the_spec_table(name, q):
    params = None if q is None else {"q": q}
    for a_min, want in _TABLE[(name, q)].items():
        assert undersampling.eta(name, params, _finest(a_min)) == pytest.approx(want, rel=0.05), \
            (name, q, a_min)


# 2. wtmm2d warns on a_min by η.

def test_wtmm2d_warns_by_eta_on_a_min():
    dev = WTMM2D()
    assert dev.knob_warnings(None, _wtmm(a_min=1.0, wavelet="mexican")) == {}
    msg = dev.knob_warnings(None, _wtmm(a_min=0.5, wavelet="mexican"))["a_min"]
    assert "η = 7.1e-02" in msg and "λ* 2.8 px" in msg and "1e-03" in msg
    q05 = dev.knob_warnings(None, _wtmm(a_min=1.0, wavelet="q_gaussian", q_tsallis=0.5))
    assert "η = 8.4e-03" in q05["a_min"]


# 10. The threshold is the setting's, and the message names it.

def test_the_threshold_moves_the_warning_and_is_named():
    dev = WTMM2D()
    assert "a_min" in dev.knob_warnings(None, _wtmm(a_min=0.65))
    assert dev.knob_warnings(None, _wtmm(a_min=0.75)) == {}
    undersampling.configure(1e-2)
    assert "1e-02" in dev.knob_warnings(None, _wtmm(a_min=0.55))["a_min"]
    assert dev.knob_warnings(None, _wtmm(a_min=0.65)) == {}
    with pytest.raises(ValueError):
        undersampling.configure(0.0)


# 3 and 4. wtmm2d_roi runs below a_min 1 and records both warnings.

def test_the_roi_knob_reaches_the_whole_raster_floor():
    a_min = next(p for p in WTMM2DROI().params if p.name == "a_min")
    assert a_min.min == next(p for p in WTMM2D().params if p.name == "a_min").min


def test_wtmm2d_roi_warns_by_the_honest_halo_and_eta():
    """Without a field the parent is unknown, so a scale whose reach does not converge warns
    (spec 22: the warning follows the smoother's reach, not a flat a_min < 1)."""
    dev = WTMM2DROI()
    base = dict(defaults_for(dev), roi_row=0, roi_col=0, roi_h=32, roi_w=32)
    warned = dev.knob_warnings(None, dict(base, a_min=0.5))["a_min"]
    assert "does not converge" in warned and "η = 7.1e-02" in warned
    assert dev.knob_warnings(None, dict(base, a_min=1.0)) == {}


def test_wtmm2d_roi_runs_below_a_min_one_and_carries_its_warnings(tmp_path):
    from test_wtmm_roi_device import _SMALL_WTMM, _synthetic_parent, _write_tif
    from dynamix.core.rasterfield import RasterField

    tif = _write_tif(tmp_path / "parent.tif", _synthetic_parent())
    field = RasterField.from_geotiff_window(tif, row_off=0, col_off=0, height=80, width=80)
    dev = WTMM2DROI()
    run = lambda a_min: dev.compute(field, validate_params(dev, dict(  # noqa: E731
        _SMALL_WTMM, a_min=a_min, roi_row=16, roi_col=16, roi_h=32, roi_w=32, boundary="auto")))
    fine = run(0.5)
    # The 80 px parent fits the budget: the non-converging scales read it whole (spec 22), so
    # only the lattice warning stays.
    assert fine["extrema"] and len(fine["_warnings"]) == 1
    assert "η = " in fine["_warnings"][0]
    assert fine["_edge"]["scales"][0]["whole"] is True
    assert run(1.0).get("_warnings", []) == []


# 5. The .gprv step carries the warning.

def test_the_record_carries_a_warning_report(clean_registry, tmp_path):
    gprv = pytest.importorskip("gprv")
    import hashlib

    from dynamix.core import fft_policy
    from dynamix.core.chain_product import export_chain_product_npz
    from dynamix.core.ingest import load_grid
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.gprv_export import write_record
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.project import Project
    from test_q_wavelet_widths import _dem

    register_builtin_devices()
    fft_policy.configure("fftw", 64)
    dem = _dem(tmp_path / "dem.tif")
    field = load_grid(dem)
    project = Project()
    src = project.add_source(str(dem), sha256=hashlib.sha256(dem.read_bytes()).hexdigest())

    def record(a_min, name):
        layer = project.add_layer(name, src.source_id, Chain((DeviceRef(
            "wtmm2d", {"n_oct": 2, "n_voice": 4, "precision": "f64", "a_min": a_min}),)))
        rendered = resolve(layer, field, Cache())
        out = tmp_path / f"{name}.npz"
        export_chain_product_npz(out, rendered.result["chain_product"], field=field,
                                 params=json.loads(json.dumps(rendered.result.get("params") or {},
                                                              default=list)))
        data = write_record(project, layer, out, export="chains_npz",
                            result=rendered.result).read_bytes()
        assert gprv.check(data) == []
        return json.loads(data)["processStep"][0].get("report", [])

    fine = record(0.5, "fine")
    warning = next(r for r in fine if r["name"] == "warning")
    assert any("η = 7.1e-02" in m for m in warning["description"]["a_min"])
    assert not any(r["name"] == "warning" for r in record(1.0, "clean"))


# 6. The glyph, in the strip and the box.

@pytest.mark.parametrize("where", ["strip", "box"])
def test_the_glyph_shows_the_warning_and_follows_a_sibling(qtbot, where):
    from dynamix.shell.chain_strip import DeviceStrip
    from dynamix.shell.workflow_zone import DeviceBox

    dev = WTMM2DROI()
    params = dict(defaults_for(dev), a_min=0.5)
    w = (DeviceStrip if where == "strip" else DeviceBox)(0, dev, params)
    qtbot.addWidget(w)
    w.show()
    icon = w._warning_icons["a_min"]
    assert icon.isVisible()
    assert icon.toolTip() == "\n\n".join(dev.knob_warnings(None, params).values())
    w.controls["a_min"].valueChanged.emit(1.0)
    assert not icon.isVisible()
    assert not w._warning_icons["wavelet"].isVisible()

    dev2 = WTMM2D()
    w2 = (DeviceStrip if where == "strip" else DeviceBox)(0, dev2, dict(defaults_for(dev2),
                                                                      a_min=1.0))
    qtbot.addWidget(w2)
    w2.show()
    assert not w2._warning_icons["a_min"].isVisible()
    w2.controls["q_tsallis"].valueChanged.emit(0.5)
    w2.controls["wavelet"].valueChanged.emit("q_gaussian")         # a sibling, no run
    assert w2._warning_icons["a_min"].isVisible()
    assert "η = 8.4e-03" in w2._warning_icons["a_min"].toolTip()


# 7 and 8. The Turiel tools: their own conditions on the glyph, off the reading line.

def _holder(name, **kw):
    from dynamix.model.device import get_device
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    dev = get_device(name)
    return dev, dict(defaults_for(dev), **kw)


def test_the_q_warning_moves_to_the_glyph(clean_registry):
    dev, p = _holder("holder_measure", wavelet="q_gaussian", q_tsallis=2.2)
    assert dev.knob_warnings(None, p)["q_tsallis"]
    assert not dev.derived_reading("q_tsallis", 2.2, None, p)


def test_the_lobe_floor_is_turiels_and_sits_on_the_glyph(clean_registry):
    dev, p = _holder("holder_multiaffine", wavelet="g2", r_min=0.8)
    msg = dev.knob_warnings(None, p)["r_min"]
    assert "1.00 px" in msg and "2 px" in msg and "⚠" not in msg
    assert "r_min" not in dev.knob_warnings(None, dict(p, r_min=1.0))
    for wavelet in ("gaussian", "lorentzian"):
        mdev, mp = _holder("holder_measure", wavelet=wavelet, r_min=0.1)
        assert "r_min" not in mdev.knob_warnings(None, mp)
    mdev, mp = _holder("holder_measure", wavelet="q_gaussian", q_tsallis=0.5, r_min=0.4)
    assert "r_min" in mdev.knob_warnings(None, mp)
    assert "r_min" not in mdev.knob_warnings(None, dict(mp, r_min=0.6))


# 9. η on the Turiel Fourier construction: a reading, never the glyph.

def test_eta_is_a_reading_on_the_turiel_fourier_build(clean_registry):
    dev, p = _holder("holder_multiaffine", wavelet="g2", r_min=1.0, construction="fourier")
    line = dev.derived_reading("r_min", 1.0, None, p)
    assert "beyond Nyquist (Fourier build)" in line and "η = " in line
    assert dev.knob_warnings(None, p) == {}
    real = dev.derived_reading("r_min", 1.0, None, dict(p, construction="real"))
    assert "Nyquist" not in (real or "")


# The level is a setting, applied at once and at start.

def test_the_level_is_a_setting_applied_at_once(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import load_settings

    win = MainWindow(steps=())
    qtbot.addWidget(win)
    assert undersampling.eta_warn() == load_settings().eta_warn == 1e-3
    win._apply_eta_warn(1e-2)
    assert load_settings().eta_warn == 1e-2 and undersampling.eta_warn() == 1e-2
    assert "settings.eta_warn" in win.commands
