# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A dyadic scale knob holds powers of two: a drag or nudge doubles or halves it, a typed or saved
value snaps to the nearest, so a scale that must be dyadic is never sent one that is not."""
from __future__ import annotations

import pytest


def test_a_dyadic_knob_doubles_halves_and_snaps_to_powers_of_two():
    from dynamix.model.param import Param, ParamKind
    from dynamix.shell.knobs import nudge

    p = Param("scale_max", ParamKind.INT, default=64, min=8, max=1024, dyadic=True)
    assert nudge(64, p, 1) == 128 and nudge(64, p, -1) == 32
    assert nudge(1024, p, 1) == 1024 and nudge(8, p, -1) == 8
    assert p.validate(33) == 32 and p.validate(48) == 64 and p.validate(1000) == 1024
    with pytest.raises(ValueError):
        p.validate(4)


@pytest.mark.parametrize("name", ["directional_scaling", "wavelet_leaders"])
def test_the_dyadic_tools_take_any_scale_as_its_nearest_power_of_two(clean_registry, name):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import get_device, validate_params

    register_builtin_devices()
    dev = get_device(name)
    params = validate_params(dev, {"scale_min": 5, "scale_max": 33})
    assert (params["scale_min"], params["scale_max"]) == (4, 32)
