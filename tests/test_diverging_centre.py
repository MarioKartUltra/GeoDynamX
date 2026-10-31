# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The centre of a diverging ramp: two slopes meeting at a data value, so the ramp's middle colour
marks it (zero for h and any signed field), the colormaps that diverge, and the display settings'
place outside the recipe and its provenance record."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from dynamix.core.stretch import (CENTER_MODES, DIVERGING, LIMIT_STRETCHES, STRETCHES,
                                  center_mode, centered, centered_limits, diverging_name,
                                  stretch, stretch_at, stretch_limits)


def test_the_centre_maps_to_mid_scale_and_each_side_is_linear_and_monotone():
    v = np.array([-0.3, -0.15, 0.0, 0.6, 1.2, np.nan])
    t = centered(v, -0.3, 1.2, center=0.0)
    assert t[:5] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.isnan(t[5])
    dense = np.linspace(-1.0, 2.0, 301)
    assert np.all(np.diff(centered(dense, -0.3, 1.2)) >= 0.0)
    assert centered(dense, -0.3, 1.2).min() == 0.0 and centered(dense, -0.3, 1.2).max() == 1.0
    assert centered(0.4, -1.0, 2.0, center=0.4) == pytest.approx(0.5)     # any centre value


def test_symmetric_limits_are_one_slope_on_both_sides():
    assert centered_limits(-0.3, 1.2, 0.0, symmetric=True) == (-1.2, 1.2)
    assert centered_limits(-2.0, 0.5, 0.0, symmetric=True) == (-2.0, 2.0)
    t = centered(np.array([-1.2, -0.6, 0.0, 0.6, 1.2]), -0.3, 1.2, symmetric=True)
    assert t == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert t[1] + t[3] == pytest.approx(1.0)          # ±d sit symmetrically about mid-scale


def test_a_field_on_one_side_of_the_centre_uses_that_half_of_the_ramp():
    t = centered(np.array([0.0, 2.5, 5.0]), 0.0, 5.0)                 # KAM-like: all >= 0
    assert t == pytest.approx([0.5, 0.75, 1.0])
    t = centered(np.array([-4.0, -2.0, -1.0]), -4.0, -1.0)            # all below zero
    assert t == pytest.approx([0.0, 0.25, 0.375])
    # A centre outside the limits widens them to reach it: the centre still reads mid-scale.
    assert centered_limits(2.0, 5.0, 0.0) == (0.0, 5.0)
    assert centered(np.array([0.0, 2.0, 5.0]), 2.0, 5.0) == pytest.approx([0.5, 0.7, 1.0])


def test_the_centre_is_on_by_default_only_for_a_field_on_both_sides_of_it():
    assert center_mode(np.array([-0.3, 0.0, 1.2])) == "value"
    assert center_mode(np.array([0.0, 2.5, 5.0])) == "none"         # KAM: zero is its floor
    assert center_mode(np.array([-4.0, -1.0, np.nan])) == "none"
    assert center_mode(np.array([0.2, 0.6]), center=0.4) == "value"
    assert center_mode(np.array([0.2, 0.6]), center=0.6) == "none"
    assert center_mode(np.full(3, np.nan)) == "none"
    for mode in CENTER_MODES[1:]:                                    # a chosen mode stands
        assert center_mode(np.array([0.0, 5.0]), mode=mode) == mode
        assert center_mode(np.array([-1.0, 5.0]), mode=mode) == mode


def test_the_centre_mode_is_a_display_tag_read_as_auto_when_absent_or_unknown():
    from dynamix.model.layer import Layer
    from dynamix.shell.main_window import _display_style_of

    assert _display_style_of(Layer(0, "h", "s0"))["center_mode"] == "auto"
    for mode in CENTER_MODES:
        assert _display_style_of(Layer(0, "h", "s0", tags={"ui.center_mode": mode}))[
            "center_mode"] == mode
    assert _display_style_of(Layer(0, "h", "s0", tags={"ui.center_mode": "middle"}))[
        "center_mode"] == "auto"


def test_degenerate_limits_stay_honest():
    assert centered(np.zeros(4), 0.0, 0.0) == pytest.approx([0.5] * 4)    # flat on the centre
    # A side of zero width: what lies beyond the centre goes to that side's end.
    assert centered(np.array([-1.0, 0.0, 1.0]), 0.0, 1.0) == pytest.approx([0.0, 0.5, 1.0])
    assert centered(np.array([-1.0, 0.0, 1.0]), -1.0, 0.0) == pytest.approx([0.0, 0.5, 1.0])
    # Limits that are not finite (a field with no finite cell) read as the centre.
    assert centered_limits(np.nan, np.nan, 0.25) == (0.25, 0.25)
    assert np.isnan(centered(np.array([np.nan]), np.nan, np.nan)).all()


def test_the_limit_stretches_scale_between_their_limits_and_stretch_at_places_a_value_as_stretch_does():
    v = np.random.default_rng(1).standard_normal((40, 30)) * 3.0 + 1.0
    v[0, 0] = np.nan
    for mode in LIMIT_STRETCHES:
        lo, hi = stretch_limits(v, mode)
        assert lo < hi
        expected = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        assert np.allclose(stretch(v, mode), expected, equal_nan=True)
    assert stretch_limits(v, "linear", limits=(-1.0, 4.0)) == (-1.0, 4.0)
    with pytest.raises(ValueError):
        stretch_limits(v, "histogram")
    for mode in STRETCHES:
        assert np.allclose(stretch_at(v, v, mode), stretch(v, mode), equal_nan=True), mode
    ticks = np.array([-5.0, 0.0, 1.0, 50.0])
    assert np.all(np.diff(stretch_at(v, ticks, "histogram")) >= 0.0)
    assert stretch_at(v, [-1e9, 1e9], "log") == pytest.approx([0.0, 1.0])


def test_the_diverging_colormaps():
    for name in DIVERGING:
        assert name in matplotlib.colormaps, name
        assert diverging_name(name) and diverging_name(f"{name}_r")
    assert diverging_name("CET-D1")
    for name in ("viridis", "binary_r", "twilight", "terrain", "CET-L1"):
        assert not diverging_name(name)


def test_a_ramp_of_the_users_own_is_diverging_when_flagged():
    from dynamix.shell import ramps
    from dynamix.shell.settings import load_settings

    stops = [(0.0, "#0000ff"), (0.5, "#ffffff"), (1.0, "#ff0000")]
    ramps.save_ramp("h about zero", stops, diverging=True)
    assert ramps.is_diverging("h about zero")
    assert load_settings().diverging_ramps == ["h about zero"]
    ramps.save_ramp("h about zero", stops)                  # saved again without the flag
    assert not ramps.is_diverging("h about zero")
    ramps.save_ramp("h about zero", stops, diverging=True)
    ramps.delete_ramp("h about zero")
    assert load_settings().diverging_ramps == [] and not ramps.is_diverging("h about zero")
    assert ramps.is_diverging("RdBu") and not ramps.is_diverging("magma")


def test_a_garbled_diverging_list_in_the_settings_file_reads_as_none(tmp_path):
    import json
    import os

    from dynamix.shell.settings import load_settings

    with open(os.environ["DYNAMIX_SETTINGS_PATH"], "w", encoding="utf-8") as f:
        json.dump({"diverging_ramps": "RdBu"}, f)
    assert load_settings().diverging_ramps == []
    with open(os.environ["DYNAMIX_SETTINGS_PATH"], "w", encoding="utf-8") as f:
        json.dump({"diverging_ramps": ["mine", 3]}, f)
    assert load_settings().diverging_ramps == ["mine"]


#: The display settings the colour bar and the centre add, as a layer's tags hold them.
_VIEW_TAGS = {"ui.colorbar": "True", "ui.center": repr(0.35), "ui.center_mode": "symmetric",
              "ui.colorbar_label": "h", "ui.colormap": "RdBu"}


def test_the_colour_bar_and_the_centre_leave_the_recipe_and_its_record_unchanged(
        clean_registry, tmp_path):
    pytest.importorskip("gprv")
    pytest.importorskip("rasterio")
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.execution import recipe_identity
    from dynamix.engine.gprv_export import record_for
    from dynamix.model.contracts import FIELD

    from tests.test_gprv_writer import _WTMM, _no_time, _project

    register_builtin_devices()
    project, layer, _field = _project(tmp_path, [_WTMM, ("scale_select", {"scale_idx": 0})])
    out = tmp_path / "chains.npz"
    out.write_bytes(b"stand-in export")
    key = recipe_identity(layer, FIELD)
    record = _no_time(record_for(project, layer, out, export="chains_npz"))
    layer.tags.update(_VIEW_TAGS)
    assert recipe_identity(layer, FIELD) == key
    assert _no_time(record_for(project, layer, out, export="chains_npz")) == record
    layer.tags["ui.center"] = repr(-2.0)
    layer.tags["ui.center_mode"] = "none"
    layer.tags["ui.colorbar"] = "False"
    assert recipe_identity(layer, FIELD) == key
    assert _no_time(record_for(project, layer, out, export="chains_npz")) == record
