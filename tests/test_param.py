# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.model.param -- the declaration Phase 4 generates knobs from."""
from __future__ import annotations

import pytest

from dynamix.model.param import Param, ParamKind


def test_float_param_validates_and_coerces():
    p = Param("holder", ParamKind.FLOAT, default=0.3, min=-1.0, max=2.0, units="")
    assert p.validate(0.5) == 0.5
    assert p.validate(1) == 1.0                      # int coerces to float
    with pytest.raises(ValueError, match="above max"):
        p.validate(2.5)
    with pytest.raises(ValueError, match="below min"):
        p.validate(-3.0)


def test_int_param_rejects_non_integral():
    p = Param("n_octaves", ParamKind.INT, default=4, min=1, max=10)
    assert p.validate(7) == 7
    with pytest.raises(ValueError, match="not an integer"):
        p.validate(2.5)


def test_choice_param_rejects_unknown_value():
    p = Param("wavelet", ParamKind.CHOICE, default="mexican", choices=("mexican", "morlet"))
    assert p.validate("morlet") == "morlet"
    with pytest.raises(ValueError, match="not one of"):
        p.validate("haar")


def test_choice_param_requires_default_in_choices():
    with pytest.raises(ValueError, match="default"):
        Param("wavelet", ParamKind.CHOICE, default="haar", choices=("mexican", "morlet"))


def test_angle_param_wraps_instead_of_clamping():
    """An orientation is periodic. 190 deg on a mod-180 dial is 10 deg, not an error and not 180."""
    p = Param("strike", ParamKind.ANGLE, default=0.0, units="deg", wrap=180.0)
    assert p.validate(190.0) == pytest.approx(10.0)
    assert p.validate(-10.0) == pytest.approx(170.0)
    assert p.validate(180.0) == pytest.approx(0.0)


def test_angle_param_requires_a_wrap_period():
    with pytest.raises(ValueError, match="wrap"):
        Param("strike", ParamKind.ANGLE, default=0.0)


def test_bounds_on_an_angle_param_are_rejected_not_ignored():
    """validate() returns as soon as it has wrapped, so a min/max declared on an ANGLE was never
    enforced. Silently ignoring a declared bound is how a wedge ends up admitting orientations the
    author of the device believed they had excluded -- and orientation is axial, so nothing about
    the result looks wrong. Refuse the declaration instead."""
    with pytest.raises(ValueError, match="no range"):
        Param("strike", ParamKind.ANGLE, default=0.0, wrap=180.0, min=0.0, max=90.0)
    with pytest.raises(ValueError, match="no range"):
        Param("strike", ParamKind.ANGLE, default=0.0, wrap=180.0, soft_min=0.0)


def test_bounds_on_choice_and_bool_params_are_rejected():
    """Same guard, same reason: neither kind is ordered, so a bound could only ever be ignored."""
    with pytest.raises(ValueError, match="no range"):
        Param("wavelet", ParamKind.CHOICE, default="a", choices=("a", "b"), min=0.0)
    with pytest.raises(ValueError, match="no range"):
        Param("normalise", ParamKind.BOOL, default=True, max=1.0)


def test_a_legal_angle_declaration_still_constructs():
    p = Param("strike", ParamKind.ANGLE, default=0.0, units="deg", wrap=180.0, label="Strike")
    assert p.validate(190.0) == pytest.approx(10.0)


def test_param_payload_round_trips():
    p = Param("holder", ParamKind.FLOAT, default=0.3, min=-1.0, max=2.0, units="", label="Hölder")
    assert Param.from_payload(p.to_payload()) == p


def test_soft_range_payload_round_trips():
    """soft_min/soft_max are the default slider span, distinct from the hard min/max, and must
    survive the same to_payload/from_payload trip as everything else."""
    p = Param("holder", ParamKind.FLOAT, default=0.3, min=-2.0, max=2.0,
              soft_min=-0.5, soft_max=1.5)
    assert Param.from_payload(p.to_payload()) == p


def test_soft_range_outside_hard_bounds_is_rejected():
    with pytest.raises(ValueError, match="soft_min"):
        Param("holder", ParamKind.FLOAT, default=0.3, min=-1.0, max=2.0, soft_min=-1.5)
    with pytest.raises(ValueError, match="soft_max"):
        Param("holder", ParamKind.FLOAT, default=0.3, min=-1.0, max=2.0, soft_max=2.5)


def test_soft_min_above_soft_max_is_rejected():
    with pytest.raises(ValueError, match="soft_min .* soft_max"):
        Param("holder", ParamKind.FLOAT, default=0.3, min=-2.0, max=2.0,
              soft_min=1.5, soft_max=-0.5)


def test_validate_ignores_the_soft_range():
    """A value inside the hard bounds but outside the soft span is not an error -- typing outside
    the soft range only widens the slider's view, it never raises."""
    p = Param("holder", ParamKind.FLOAT, default=0.3, min=-2.0, max=2.0,
              soft_min=-0.5, soft_max=1.5)
    assert p.validate(1.9) == pytest.approx(1.9)
    assert p.validate(-1.9) == pytest.approx(-1.9)


def test_half_specified_soft_bound_out_of_range_is_rejected():
    """A same-side-only check misses this: soft_min alone can still land above max, and soft_max
    alone can still land below min -- the whole default slider span would then be illegal."""
    with pytest.raises(ValueError, match="soft_min"):
        Param("holder", ParamKind.FLOAT, default=0.3, min=-1.0, max=2.0, soft_min=3.0)
    with pytest.raises(ValueError, match="soft_max"):
        Param("holder", ParamKind.FLOAT, default=0.3, min=-1.0, max=2.0, soft_max=-5.0)


def test_half_specified_soft_bound_in_range_is_accepted():
    """Declaring only one end of the soft span is legitimate -- it must not be rejected merely for
    being half-specified."""
    p_lo = Param("holder", ParamKind.FLOAT, default=0.3, min=-1.0, max=2.0, soft_min=0.0)
    assert p_lo.soft_min == 0.0 and p_lo.soft_max is None
    p_hi = Param("holder", ParamKind.FLOAT, default=0.3, min=-1.0, max=2.0, soft_max=1.0)
    assert p_hi.soft_max == 1.0 and p_hi.soft_min is None


def test_a_default_outside_its_own_bounds_is_rejected():
    """A declaration must fully describe a device, and that includes its default. An out-of-range
    default is never caught later: validate_params fills an absent param from ``default`` without
    validating it, so the illegal value flows straight into cache_key and compute."""
    with pytest.raises(ValueError, match="default"):
        Param("n_octaves", ParamKind.INT, default=99, min=1, max=8)
    with pytest.raises(ValueError, match="default"):
        Param("holder", ParamKind.FLOAT, default=-9.0, min=-2.0, max=2.0)


def test_a_non_integral_default_on_an_int_param_is_rejected():
    with pytest.raises(ValueError, match="default"):
        Param("n_octaves", ParamKind.INT, default=2.5, min=1, max=8)


def test_the_default_is_coerced_the_same_way_a_value_would_be():
    """Not range-checked raw but put through validate(), so the default and a user typing the same
    number cannot disagree -- two spellings of one dial position must not make two cache keys."""
    assert Param("strike", ParamKind.ANGLE, default=190.0, wrap=180.0).default == pytest.approx(10.)
    assert isinstance(Param("holder", ParamKind.FLOAT, default=1).default, float)


def test_soft_bounds_without_hard_bounds_do_not_crash():
    """Hard min/max default to None, so every soft-bound guard must tolerate a None hard bound on
    either side without raising TypeError."""
    p = Param("holder", ParamKind.FLOAT, default=0.3, soft_min=-0.5, soft_max=1.5)
    assert p.soft_min == -0.5 and p.soft_max == 1.5


def test_section_defaults_to_the_strip_and_is_presentation_only():
    """``section`` names where the shell draws a knob: "" (the default) on the device strip,
    a name such as "reconstruction" in that right-panel section. Validation ignores it."""
    strip = Param("kappa", ParamKind.FLOAT, default=1.0, min=0.1, max=10.0)
    panel = Param("kappa", ParamKind.FLOAT, default=1.0, min=0.1, max=10.0,
                  section="reconstruction")
    assert strip.section == "" and panel.section == "reconstruction"
    assert panel.validate(2.5) == strip.validate(2.5) == 2.5
    with pytest.raises(ValueError, match="above max"):
        panel.validate(11.0)
    assert strip != panel


def test_section_payload_round_trips_and_an_older_payload_reads_as_the_strip():
    p = Param("clip", ParamKind.BOOL, default=False, section="reconstruction")
    assert Param.from_payload(p.to_payload()) == p
    older = p.to_payload()
    del older["section"]
    assert Param.from_payload(older).section == ""
