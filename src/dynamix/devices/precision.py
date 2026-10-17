# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The FFT precision knob, on every tool whose compute, outputs or preview run FFTs.

The engine runs such a tool inside :func:`dynamix.core.fft_policy.scope_for` its params, so every
FFT in it runs at this precision on the configured engine; mlx has no double FFT, so 64 bits
there run on FFTW3 and the knob's reading says so.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind

#: f32 is the default (xsmurf's single precision) and stays out of the key, so every recipe keyed
#: before the knob existed keeps its key; f64 enters it.
PRECISION_PARAM = Param("precision", ParamKind.CHOICE, default="f32", choices=("f32", "f64"),
                        choice_labels=("32-bit", "64-bit"), label="FFT precision",
                        keyed_at_default=False)


def precision_reading(params: dict | None) -> str | None:
    """The line under the knob when the precision cannot run as asked on the configured engine
    (64 bits where the engine is mlx); ``None`` when it runs as asked."""
    from dynamix.core import fft_policy

    note = fft_policy.note((params or {}).get("precision", "f32"))
    return note or None


def placement_params(device) -> dict:
    """A newly placed tool's params: its defaults, with ``precision`` at the configured one
    (Settings › FFT precision, applied at startup) when the tool carries the knob. A recipe
    placed at 64 bits says so, so its key and its record say so too."""
    from dynamix.core import fft_policy
    from dynamix.model.device import defaults_for

    params = defaults_for(device)
    if "precision" in params and fft_policy.configured_precision() == 64:
        params["precision"] = "f64"
    return params
