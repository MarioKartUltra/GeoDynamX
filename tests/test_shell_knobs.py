# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Param -> control mapping: the generated-knobs bet, as pure logic."""
import pytest

from dynamix.model.param import Param, ParamKind
from dynamix.shell.knobs import control_spec, format_reading, nudge, rebind_range


def _p_float(**kw):
    kw.setdefault("name", "cutoff"); kw.setdefault("kind", ParamKind.FLOAT)
    kw.setdefault("default", 0.5); kw.setdefault("min", 0.0); kw.setdefault("max", 10.0)
    kw.setdefault("soft_min", 0.0); kw.setdefault("soft_max", 2.0)
    return Param(**kw)


def test_float_maps_to_drag_value_with_soft_span():
    s = control_spec(_p_float(units="px"))
    assert s.widget == "drag_value" and (s.lo, s.hi) == (0.0, 2.0)
    assert s.decimals == 2 and s.step == pytest.approx((2.0 - 0.0) / 100)


def test_int_step_is_one_and_zero_decimals():
    p = Param("n_oct", ParamKind.INT, default=3, min=1, max=8)
    s = control_spec(p)
    assert s.widget == "drag_value" and s.step == 1 and s.decimals == 0
    assert (s.lo, s.hi) == (1, 8)            # no soft bounds -> hard bounds span


def test_choice_and_bool_map_to_cycle_and_toggle():
    c = Param("wavelet", ParamKind.CHOICE, default="mexican", choices=("mexican", "morlet"))
    b = Param("smooth", ParamKind.BOOL, default=True)
    assert control_spec(c).widget == "cycle"
    assert control_spec(b).widget == "toggle"


def test_angle_wraps():
    a = Param("centre", ParamKind.ANGLE, default=0.0, wrap=180.0)
    s = control_spec(a)
    assert s.wrap == 180.0
    assert nudge(179.0, a, +1, "coarse") < 180.0    # wrapped, never clamped


def test_nudge_modifiers_scale():
    p = _p_float()
    base = nudge(0.5, p, +1, "normal") - 0.5
    assert nudge(0.5, p, +1, "coarse") - 0.5 == pytest.approx(base * 10)
    assert nudge(0.5, p, +1, "fine") - 0.5 == pytest.approx(base / 10)
    assert nudge(9.99, p, +1, "coarse") == 10.0     # clamps at HARD max


def test_format_reading_is_mono_text_with_unit():
    assert format_reading(8.0, _p_float(units="px")) == "8.00 px"
    assert format_reading(3, Param("n", ParamKind.INT, default=3, min=1, max=8)) == "3"


def test_format_reading_never_rounds_a_nonzero_value_to_zero():
    """wtmm2d's thresh default of 1e-3 displayed as "0.00" -- a false reading that also
    fed the inline editor, so Enter silently committed a real 0.0. Sub-resolution values
    fall back to significant figures; zero itself still reads as the fixed format."""
    thresh_like = Param("thresh", ParamKind.FLOAT, default=1e-3, min=0.0, max=1.0)
    assert format_reading(1e-3, thresh_like) == "0.001"
    assert format_reading(1e-6, thresh_like) == "1e-06"
    assert format_reading(0.0, thresh_like) == "0.00"
    assert format_reading(0.85, thresh_like) == "0.85"


def test_every_builtin_float_default_reads_nonzero_when_it_is_nonzero(clean_registry):
    """Systemic guard: no registered device may ship a FLOAT default whose reading
    displays as zero while the value is not."""
    from dynamix.devices import register_builtin_devices
    from dynamix.model import DEVICES

    register_builtin_devices()
    for device in DEVICES.values():
        for p in device.params:
            if p.kind is ParamKind.FLOAT and p.default:
                reading = format_reading(p.default, p).split(" ")[0]
                assert float(reading) != 0.0, f"{device.name}.{p.name} reads as zero"


def test_int_fine_modifier_has_minimum_step_of_one():
    """For INT params, fine modifier collapses to normal (step=1); a 0.1 step
    rounded to the nearest integer is always a no-op."""
    int_param = Param("n_oct", ParamKind.INT, default=3, min=1, max=8)
    assert nudge(3, int_param, +1, "fine") == 4
    assert nudge(3, int_param, -1, "fine") == 2


def test_int_coarse_vs_normal_scaling():
    """INT params: normal=step 1, coarse=step 10, clamped at hard bounds."""
    int_param = Param("n_oct", ParamKind.INT, default=3, min=1, max=8)
    assert nudge(3, int_param, +1, "normal") == 4
    assert nudge(3, int_param, +1, "coarse") == 8  # 10-step clamped at hard max


def test_nudge_accepts_a_live_control_spec_and_uses_its_rebound_step():
    """Nudge() must not silently discard a live, data-rebound
    ControlSpec by reconstructing a fresh (static) one -- that was exactly the bug (a data-hinted
    control's real keyboard/drag step never changed). Passing a bare Param is still supported,
    unchanged, for every existing call site/test that relies on the STATIC declared soft range."""
    p = _p_float()
    static_step = control_spec(p).step
    rebound = rebind_range(control_spec(p), 0.0, 100.0)
    assert rebound.step == pytest.approx(1.0)
    assert nudge(0.5, rebound, +1, "normal") - 0.5 == pytest.approx(1.0)
    assert nudge(0.5, rebound, +1, "normal") != pytest.approx(0.5 + static_step)
    # a bare Param is UNCHANGED behavior: reconstructs the static spec every time
    assert nudge(0.5, p, +1, "normal") - 0.5 == pytest.approx(static_step)


def test_every_builtin_device_param_builds_a_control_spec(clean_registry):
    """Registry-wide regression: ``chain_classify``'s four Hölder-band FLOAT params once declared
    no min/max, so ``control_spec``'s ``(hi - lo) / _NUDGE_DIVISIONS`` computed on None and raised
    a TypeError the moment the device was dropped into a box -- not caught by any per-device test,
    because no per-device test exercises the GUI's knob construction. Every FLOAT/INT param of
    every registered device must survive ``control_spec`` unconditionally, so a future device with
    the same omission fails here instead of in a live drop."""
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES

    register_builtin_devices()
    for dev in DEVICES.values():
        for p in dev.params:
            control_spec(p)   # must not raise for any registered device's param
