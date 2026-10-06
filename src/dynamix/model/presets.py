# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Named starter chains for the browser's Racks section. Data only — no Qt."""
from __future__ import annotations

#: name -> ((device, params-overrides), ...). Params must validate against each
#: device's own schema; a test enforces it. The shown level of a multi-scale analyzer is the
#: Transport's (the layer's ``ui.scale_idx``): ``resolve`` reduces the stack to it before the
#: filters run, so no rack carries a level step.
PRESETS: dict[str, tuple] = {
    "WTMM + Hölder": (
        ("wtmm2d", {"n_oct": 3, "n_voice": 4}),
        ("chain_topology", {}),
        ("chain_holder", {}),
        ("chain_length", {}),
        ("modulus_threshold", {"frac": 0.0}),
    ),
    "M–Z edges": (
        ("mz_edges", {"n_levels": 4}),
        ("hline_length", {}),
    ),
    # A multi-scale analyzer scrubbed by the Transport; hline_length kills the orphan confetti.
    "CDF edges": (
        ("cdf_edges", {"n_levels": 4}),
        ("hline_length", {}),
    ),
    # Perona-Malik proper: the same multi-scale rack shape as CDF edges -- the analyzer plus
    # orphan-confetti control.
    "PM edges": (
        ("pm_edges", {"n_levels": 4}),
        ("hline_length", {}),
    ),
    # Single-scale output; hline_length kills the orphan confetti.
    "Wavelet skeleton": (
        ("wavelet_skeleton", {}),
        ("hline_length", {"min_len": 10}),
    ),
}
